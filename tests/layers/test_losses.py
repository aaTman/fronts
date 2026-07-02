"""Tests for the latitude-aware neighborhood Brier loss."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
losses = pytest.importorskip("fronts.layers.losses")

N_BATCH = 2
N_H = 8
N_W = 8
N_CLASSES = 6
RESOLUTION_DEG = 0.25

EQUATOR_LATITUDES = np.arange(N_H * RESOLUTION_DEG, 0.0, -RESOLUTION_DEG)
POLAR_LATITUDES = np.arange(80.0, 80.0 - N_H * RESOLUTION_DEG, -RESOLUTION_DEG)


def _one_hot_background(n_batch: int = N_BATCH, n_h: int = N_H, n_w: int = N_W) -> np.ndarray:
    y = np.zeros((n_batch, n_h, n_w, N_CLASSES), dtype=np.float32)
    y[..., 0] = 1.0
    return y


def _with_front_row(y: np.ndarray, row: int, cls: int = 1) -> np.ndarray:
    y = y.copy()
    y[:, row, :, 0] = 0.0
    y[:, row, :, cls] = 1.0
    return y


class TestPlanWindows:
    def test_meridional_half_width_constant_with_latitude(self):
        plans_eq = losses._plan_windows(EQUATOR_LATITUDES, RESOLUTION_DEG, (25.0,), max_half_x=128)
        plans_polar = losses._plan_windows(POLAR_LATITUDES, RESOLUTION_DEG, (25.0,), max_half_x=128)
        assert plans_eq[0]["half_y"] == plans_polar[0]["half_y"] == 1

    def test_zonal_half_width_widens_with_latitude(self):
        plans_eq = losses._plan_windows(EQUATOR_LATITUDES, RESOLUTION_DEG, (25.0,), max_half_x=128)
        plans_polar = losses._plan_windows(POLAR_LATITUDES, RESOLUTION_DEG, (25.0,), max_half_x=128)
        assert plans_eq[0]["half_x"].max() == 1
        assert plans_polar[0]["half_x"][0] == 5, "1/cos(80 deg) is ~5.8, so the zonal window must be ~5x wider"
        assert plans_polar[0]["half_x"].min() > plans_eq[0]["half_x"].max()

    def test_zonal_half_width_clipped_at_max(self):
        plans = losses._plan_windows(POLAR_LATITUDES, RESOLUTION_DEG, (250.0,), max_half_x=8)
        assert plans[0]["half_x"].max() == 8

    def test_one_plan_per_tolerance(self):
        plans = losses._plan_windows(EQUATOR_LATITUDES, RESOLUTION_DEG, (25.0, 100.0, 250.0), max_half_x=128)
        assert len(plans) == 3
        assert plans[0]["half_y"] < plans[1]["half_y"] < plans[2]["half_y"]


class TestLatDependentPool:
    def test_pool_of_ones_is_one_everywhere(self):
        field = tf.ones((1, N_H, N_W, 1), tf.float32)
        pooled = losses._lat_dependent_pool(field, half_y=2, half_x_per_row=np.full(N_H, 2), periodic_lon=False)
        np.testing.assert_allclose(pooled.numpy(), 1.0, atol=1e-6)

    def test_edges_use_valid_cell_normalization_not_reflection(self):
        """A positive column next to the boundary must be averaged over in-domain cells only.

        With half_x=1 the window at column 0 covers valid columns {0, 1}: mean 1/2.
        Reflection would give 2/3; plain zero-padding without count normalization, 1/3.
        """
        field = np.zeros((1, N_H, N_W, 1), dtype=np.float32)
        field[:, :, 1, :] = 1.0
        pooled = losses._lat_dependent_pool(
            tf.constant(field), half_y=0, half_x_per_row=np.full(N_H, 1), periodic_lon=False
        )
        np.testing.assert_allclose(pooled.numpy()[:, :, 0, :], 0.5, atol=1e-6)

    def test_periodic_longitude_wraps(self):
        field = np.zeros((1, N_H, N_W, 1), dtype=np.float32)
        field[:, :, -1, :] = 1.0
        pooled = losses._lat_dependent_pool(
            tf.constant(field), half_y=0, half_x_per_row=np.full(N_H, 1), periodic_lon=True
        )
        np.testing.assert_allclose(pooled.numpy()[:, :, 0, :], 1.0 / 3.0, atol=1e-6)


class TestNeighborhoodBrierScore:
    def test_perfect_prediction_is_zero(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerances_km=(25.0, 100.0))
        result = loss_fn(y_true, y_true.copy()).numpy()
        assert result.shape == (N_BATCH,)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_wrong_prediction_is_positive(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        y_pred = _one_hot_background()
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerances_km=(25.0,))
        assert loss_fn(y_true, y_pred).numpy().min() > 0.0

    def test_batch_elements_scored_independently(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        y_pred = y_true.copy()
        y_pred[1] = _one_hot_background(n_batch=1)[0]
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerances_km=(25.0,))
        result = loss_fn(y_true, y_pred).numpy()
        assert result[0] == pytest.approx(0.0, abs=1e-7)
        assert result[1] > 0.0

    def test_displacement_tolerance(self):
        """A 1-pixel near miss must cost far less than a distant miss, and an exact hit costs nothing."""
        n_h = 16
        latitudes = np.arange(n_h * RESOLUTION_DEG, 0.0, -RESOLUTION_DEG)
        y_true = _with_front_row(_one_hot_background(n_h=n_h), row=4)
        loss_fn = losses.neighborhood_brier_score(latitudes=latitudes, tolerances_km=(25.0,))
        exact = loss_fn(y_true, _with_front_row(_one_hot_background(n_h=n_h), row=4)).numpy().mean()
        near = loss_fn(y_true, _with_front_row(_one_hot_background(n_h=n_h), row=5)).numpy().mean()
        far = loss_fn(y_true, _with_front_row(_one_hot_background(n_h=n_h), row=12)).numpy().mean()
        assert exact == pytest.approx(0.0, abs=1e-7)
        assert 0.0 < near < far
        assert near < 0.75 * far, f"Near miss ({near:.5f}) should be well below a far miss ({far:.5f})"

    def test_zero_class_weight_excludes_class(self):
        y_true = _one_hot_background()
        y_pred = np.zeros_like(y_true)
        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        weighted = losses.neighborhood_brier_score(
            latitudes=EQUATOR_LATITUDES, tolerances_km=(25.0,), class_weights=class_weights
        )
        unweighted = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerances_km=(25.0,))
        assert weighted(y_true, y_pred).numpy().max() == pytest.approx(0.0, abs=1e-7)
        assert unweighted(y_true, y_pred).numpy().min() > 0.0

    def test_resolution_inferred_from_descending_latitudes(self):
        loss_fn = losses.neighborhood_brier_score(latitudes=POLAR_LATITUDES, tolerances_km=(25.0,))
        y_true = _with_front_row(_one_hot_background(), row=4)
        result = loss_fn(y_true, _one_hot_background()).numpy()
        assert np.all(np.isfinite(result)) and result.min() > 0.0

    def test_include_pixel_sharpens_near_miss_penalty(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        y_pred = _with_front_row(_one_hot_background(), row=5)
        base = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerances_km=(25.0,))
        with_pixel = losses.neighborhood_brier_score(
            latitudes=EQUATOR_LATITUDES, tolerances_km=(25.0,), include_pixel=True, pixel_weight=1.0
        )
        assert with_pixel(y_true, y_pred).numpy().mean() > base(y_true, y_pred).numpy().mean()

    def test_gradient_flows_to_predictions(self):
        y_true = tf.constant(_with_front_row(_one_hot_background(), row=4))
        y_pred = tf.Variable(np.full((N_BATCH, N_H, N_W, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32))
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerances_km=(25.0, 100.0))
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(loss_fn(y_true, y_pred))
        grad = tape.gradient(loss, y_pred)
        assert grad is not None
        assert float(tf.reduce_max(tf.abs(grad))) > 0.0

    def test_traces_under_tf_function(self):
        y_true = _with_front_row(_one_hot_background(), row=4)
        loss_fn = losses.neighborhood_brier_score(latitudes=EQUATOR_LATITUDES, tolerances_km=(25.0, 100.0, 250.0))
        traced = tf.function(loss_fn)
        result = traced(tf.constant(y_true), tf.constant(_one_hot_background())).numpy()
        assert result.shape == (N_BATCH,)
        assert np.all(np.isfinite(result))
