"""Tests for custom metrics: HSS, FSS, and supporting helpers."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
_metrics = pytest.importorskip("fronts.layers.metrics")
heidke_skill_score = _metrics.heidke_skill_score
fractions_skill_score = _metrics.fractions_skill_score

N_BATCH = 2
N_H = 8
N_W = 8
N_CLASSES = 6


@pytest.fixture
def perfect_pred():
    """y_true and y_pred are identical one-hot arrays."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, N_CLASSES, size=(N_BATCH, N_H, N_W))
    one_hot = (labels[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)
    return one_hot, one_hot.copy()


@pytest.fixture
def front_pred():
    """Realistic scenario: background dominates, front is a small band.

    y_true: class 1 (CF) in rows 3-4 of batch 0; rest is background (class 0).
    y_pred: softmax-like probabilities — CF pixels have p(CF)=0.35, which is below
            the 0.5 hard threshold but clearly above zero. Tests that soft HSS
            (no threshold) responds to this signal while hard threshold=0.5 does not.
    """
    y_true = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
    y_pred = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)

    y_true[..., 0] = 1.0
    y_pred[..., 0] = 0.90
    y_pred[..., 1] = 0.02
    y_pred[..., 2] = 0.02
    y_pred[..., 3] = 0.02
    y_pred[..., 4] = 0.02
    y_pred[..., 5] = 0.02

    y_true[0, 3:5, :, 0] = 0.0
    y_true[0, 3:5, :, 1] = 1.0

    y_pred[0, 3:5, :, 0] = 0.50
    y_pred[0, 3:5, :, 1] = 0.35
    y_pred[0, 3:5, :, 2] = 0.05
    y_pred[0, 3:5, :, 3] = 0.05
    y_pred[0, 3:5, :, 4] = 0.03
    y_pred[0, 3:5, :, 5] = 0.02

    return y_true, y_pred


class TestHeidkeSkillScorePerfect:
    def test_perfect_no_threshold(self, perfect_pred):
        y_true, y_pred = perfect_pred
        result = heidke_skill_score()(y_true, y_pred).numpy()
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_perfect_with_threshold(self, perfect_pred):
        y_true, y_pred = perfect_pred
        result = heidke_skill_score(threshold=0.5)(y_true, y_pred).numpy()
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_perfect_with_window(self, perfect_pred):
        y_true, y_pred = perfect_pred
        result = heidke_skill_score(threshold=0.5, window_size=(3, 3))(y_true, y_pred).numpy()
        assert result == pytest.approx(1.0, abs=1e-5)


class TestHeidkeSkillScoreSoftMode:
    def test_soft_detects_front_where_hard_threshold_does_not(self, front_pred):
        """Soft HSS (no threshold) should be positive when p(CF)=0.35; hard threshold=0.5 should not fire."""
        y_true, y_pred = front_pred
        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        score_soft = heidke_skill_score(class_weights=class_weights)(y_true, y_pred).numpy()
        score_hard = heidke_skill_score(threshold=0.5, class_weights=class_weights)(y_true, y_pred).numpy()
        assert score_soft > 0.0, f"Soft HSS should be positive for a detected front; got {score_soft:.4f}"
        assert score_soft > score_hard, (
            f"Soft HSS ({score_soft:.4f}) should exceed hard-threshold HSS ({score_hard:.4f}) "
            "when softmax front probs sit below 0.5"
        )

    def test_soft_improves_as_predictions_sharpen(self):
        """Soft HSS should increase as the model assigns more probability mass to the correct class."""
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 0] = 1.0
        y_true[0, 4, 4, 0] = 0.0
        y_true[0, 4, 4, 1] = 1.0

        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        hss = heidke_skill_score(class_weights=class_weights)

        def make_pred(cf_prob):
            y_pred = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
            y_pred[..., 0] = 1.0
            y_pred[0, 4, 4, 0] = 1.0 - cf_prob
            y_pred[0, 4, 4, 1] = cf_prob
            return y_pred

        score_low = hss(y_true, make_pred(0.1)).numpy()
        score_mid = hss(y_true, make_pred(0.3)).numpy()
        score_high = hss(y_true, make_pred(0.7)).numpy()

        assert score_low < score_mid < score_high, (
            f"Soft HSS should increase monotonically as CF probability increases: "
            f"{score_low:.4f} < {score_mid:.4f} < {score_high:.4f}"
        )


class TestHeidkeSkillScoreWindow:
    def test_window_helps_near_miss(self):
        """A prediction shifted by 1 pixel should score at least as well with a window."""
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 0] = 1.0
        y_pred[..., 0] = 1.0

        y_true[0, 4, :, 0] = 0.0
        y_true[0, 4, :, 1] = 1.0
        y_pred[0, 5, :, 0] = 0.0
        y_pred[0, 5, :, 1] = 1.0

        class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        score_exact = heidke_skill_score(threshold=0.5, class_weights=class_weights)(y_true, y_pred).numpy()
        score_window = heidke_skill_score(threshold=0.5, window_size=(3, 3), class_weights=class_weights)(
            y_true, y_pred
        ).numpy()

        assert score_window >= score_exact, (
            f"Window score ({score_window:.4f}) should be >= exact score ({score_exact:.4f}) for a 1-pixel offset"
        )


class TestHeidkeSkillScoreClassWeights:
    def test_background_suppressed(self):
        """With background weight=0, missing a front pixel should reduce HSS below 1."""
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 0] = 1.0
        y_pred[..., 0] = 1.0
        y_true[0, 4, 4, 0] = 0.0
        y_true[0, 4, 4, 2] = 1.0

        result = heidke_skill_score(threshold=0.5, class_weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])(y_true, y_pred).numpy()
        assert result < 1.0, "HSS should be less than 1 when a front pixel is missed"

    def test_equal_weights_matches_unweighted(self, perfect_pred):
        """All-equal non-zero weights should give the same result as no weights for perfect predictions."""
        y_true, y_pred = perfect_pred
        score_no_weights = heidke_skill_score(threshold=0.5)(y_true, y_pred).numpy()
        score_equal_weights = heidke_skill_score(threshold=0.5, class_weights=[1.0] * N_CLASSES)(y_true, y_pred).numpy()
        assert score_no_weights == pytest.approx(score_equal_weights, abs=1e-5)


class TestHeidkeSkillScorePredBufferPx:
    def test_asymmetric_buffer_lon_only(self):
        """pred_buffer_lat_px=0, pred_buffer_lon_px=2: y_pred wider than y_true in longitude only.

        This is the case that fails today under the single-scalar ``pred_buffer_px`` API: a
        longitude-only buffer with zero latitude buffer needs the crop to be a genuine no-op on
        the latitude axis (not ``field[:, 0:-0, ...]``, which yields an empty tensor).
        """
        buffer_lon_px = 2
        rng = np.random.default_rng(1)
        labels = rng.integers(0, N_CLASSES, size=(N_BATCH, N_H, N_W))
        core = (labels[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)

        # Pad only the longitude axis; latitude height matches y_true exactly.
        y_pred = np.pad(core, ((0, 0), (0, 0), (buffer_lon_px, buffer_lon_px), (0, 0)), mode="edge")
        # Buffer region disagrees with what the crop should discard, so a wrong crop changes the score.
        y_pred[:, :, :buffer_lon_px, :] = 0.0
        y_pred[:, :, -buffer_lon_px:, :] = 0.0

        result = heidke_skill_score(threshold=0.5, pred_buffer_lat_px=0, pred_buffer_lon_px=buffer_lon_px)(
            core, y_pred
        ).numpy()
        assert np.isfinite(result)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_zero_buffer_on_both_axes_is_true_noop(self, perfect_pred):
        """pred_buffer_lat_px=0, pred_buffer_lon_px=0 must reproduce the un-buffered HSS value exactly.

        A naive ``field[:, 0:-0, 0:-0, :]`` slice would yield an empty tensor rather than a no-op;
        this guards against that regression.
        """
        y_true, y_pred = perfect_pred
        unbuffered = heidke_skill_score(threshold=0.5)(y_true, y_pred).numpy()
        result = heidke_skill_score(threshold=0.5, pred_buffer_lat_px=0, pred_buffer_lon_px=0)(y_true, y_pred).numpy()
        assert result == pytest.approx(unbuffered, abs=1e-7)

    def test_symmetric_buffer_matches_hand_cropped_expected(self):
        """pred_buffer_lat_px=2, pred_buffer_lon_px=2 matches an independently hand-cropped expected value."""
        buffer_px = 2
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, N_CLASSES, size=(N_BATCH, N_H, N_W))
        y_true = (y_true[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)

        rng_pred = np.random.default_rng(3)
        padded_labels = rng_pred.integers(0, N_CLASSES, size=(N_BATCH, N_H + 2 * buffer_px, N_W + 2 * buffer_px))
        y_pred_padded = (padded_labels[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)

        # Hand-crop the buffer off both axes ourselves, independent of the code under test.
        y_pred_cropped = y_pred_padded[:, buffer_px:-buffer_px, buffer_px:-buffer_px, :]
        expected = heidke_skill_score(threshold=0.5)(y_true, y_pred_cropped).numpy()

        result = heidke_skill_score(threshold=0.5, pred_buffer_lat_px=buffer_px, pred_buffer_lon_px=buffer_px)(
            y_true, y_pred_padded
        ).numpy()
        assert result == pytest.approx(expected, abs=1e-6)

    def test_buffer_composes_with_window_size(self):
        """The crop must apply AFTER window_size pooling: composing them equals HSS of hand-cropped pooled fields."""
        buffer_lat_px, buffer_lon_px = 0, 2
        window_size = (3, 3)
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, N_CLASSES, size=(N_BATCH, N_H, N_W))
        y_true = (y_true[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)

        rng_pred = np.random.default_rng(5)
        padded_labels = rng_pred.integers(0, N_CLASSES, size=(N_BATCH, N_H, N_W + 2 * buffer_lon_px))
        y_pred_padded = (padded_labels[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)

        # Independently pool both fields with window_size, then hand-crop the pooled y_pred.
        y_true_pooled = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")
        y_pred_pooled = tf.nn.max_pool(y_pred_padded, ksize=window_size, strides=1, padding="VALID")
        y_pred_pooled_cropped = y_pred_pooled[:, :, buffer_lon_px:-buffer_lon_px, :]
        expected = heidke_skill_score(threshold=0.5)(y_true_pooled, y_pred_pooled_cropped).numpy()

        result = heidke_skill_score(
            threshold=0.5,
            window_size=window_size,
            pred_buffer_lat_px=buffer_lat_px,
            pred_buffer_lon_px=buffer_lon_px,
        )(y_true, y_pred_padded).numpy()
        assert result == pytest.approx(expected, abs=1e-6)

    def test_buffer_composes_with_threshold(self):
        """The crop must apply BEFORE thresholding: composing them equals HSS of hand-cropped, then thresholded."""
        buffer_lat_px, buffer_lon_px = 1, 3
        rng = np.random.default_rng(6)
        y_true = rng.integers(0, N_CLASSES, size=(N_BATCH, N_H, N_W))
        y_true = (y_true[..., np.newaxis] == np.arange(N_CLASSES)).astype(np.float32)

        # Soft (non-one-hot) predictions so thresholding actually does something.
        rng_pred = np.random.default_rng(7)
        y_pred_padded = rng_pred.random((N_BATCH, N_H + 2 * buffer_lat_px, N_W + 2 * buffer_lon_px, N_CLASSES)).astype(
            np.float32
        )

        y_pred_cropped = y_pred_padded[:, buffer_lat_px:-buffer_lat_px, buffer_lon_px:-buffer_lon_px, :]
        expected = heidke_skill_score(threshold=0.5)(y_true, y_pred_cropped).numpy()

        result = heidke_skill_score(threshold=0.5, pred_buffer_lat_px=buffer_lat_px, pred_buffer_lon_px=buffer_lon_px)(
            y_true, y_pred_padded
        ).numpy()
        assert result == pytest.approx(expected, abs=1e-6)


class TestFSSMetricPerfectPrediction:
    def test_perfect_prediction_is_one(self):
        y = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y[..., 1] = 1.0
        score = fractions_skill_score(mask_size=(3, 3))(y, y).numpy()
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_perfect_prediction_with_weights_is_one(self):
        y = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y[..., 1] = 1.0
        score = fractions_skill_score(mask_size=(3, 3), class_weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])(y, y).numpy()
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_all_background_perfect_prediction_is_one_with_weights(self):
        y = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y[..., 0] = 1.0
        score = fractions_skill_score(mask_size=(3, 3), class_weights=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])(y, y).numpy()
        assert score == pytest.approx(1.0, abs=1e-5)


class TestFSSMetricWrongPrediction:
    def test_completely_wrong_cf_prediction_penalised(self):
        """CF truth but background prediction should produce FSS < 1."""
        y_true = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 1] = 1.0
        y_pred = np.zeros((N_BATCH, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_pred[..., 0] = 1.0

        class_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        score = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)(y_true, y_pred).numpy()
        assert score < 0.95, f"Expected low FSS for completely wrong prediction, got {score:.4f}"

    def test_fss_increases_as_predictions_improve(self):
        y_true = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
        y_true[..., 1] = 1.0

        class_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        fss = fractions_skill_score(mask_size=(3, 3), class_weights=class_weights)

        scores = []
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y_pred = np.zeros((1, N_H, N_W, N_CLASSES), dtype=np.float32)
            y_pred[..., 0] = 1.0 - p
            y_pred[..., 1] = p
            scores.append(float(fss(y_true, y_pred).numpy()))

        assert scores == sorted(scores), f"FSS should increase monotonically as CF probability increases: {scores}"
