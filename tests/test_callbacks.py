"""Tests for fronts.callbacks: W&B metric consolidation and test-set visualization helpers."""

import os
from typing import ClassVar

import numpy as np
import pytest
import xarray as xr

from fronts import constants

fc = pytest.importorskip("fronts.callbacks")


class TestMetricsConsolidationCallback:
    def test_aggregates_hss_and_strips_per_output_keys(self):
        logs = {
            "loss": 1.0,
            "sup1_softmax_hss": 0.1,
            "sup1_softmax_loss": 0.4,
            "sup2_softmax_hss": 0.3,
            "sup2_softmax_loss": 0.2,
            "val_loss": 1.5,
            "val_sup1_softmax_hss": 0.2,
            "val_sup1_softmax_loss": 0.5,
            "val_sup2_softmax_hss": 0.4,
            "val_sup2_softmax_loss": 0.3,
        }
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)

        assert logs == {
            "loss": 1.0,
            "val_loss": 1.5,
            "hss": pytest.approx(0.2),
            "val_hss": pytest.approx(0.3),
        }

    def test_aggregates_multiple_custom_metrics_independently(self):
        """A second custom metric (e.g. hss_hard) must aggregate to its own key, not hss's."""
        logs = {
            "loss": 1.0,
            "sup1_softmax_hss": 0.1,
            "sup1_softmax_hss_hard": 0.6,
            "sup1_softmax_loss": 0.4,
            "sup2_softmax_hss": 0.3,
            "sup2_softmax_hss_hard": 0.8,
            "sup2_softmax_loss": 0.2,
        }
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)

        assert logs == {
            "loss": 1.0,
            "hss": pytest.approx(0.2),
            "hss_hard": pytest.approx(0.7),
        }

    def test_noop_on_empty_logs(self):
        logs = {}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {}

    def test_noop_on_none_logs(self):
        # Should not raise even though Keras can call on_epoch_end with logs=None.
        fc.MetricsConsolidationCallback().on_epoch_end(0, None)


class TestMetricsConsolidationCallbackFrontTypeRenaming:
    """Covers the post-consolidation rename step that groups per-front-type W&B keys."""

    @pytest.mark.parametrize(
        ("raw_key", "renamed_key"),
        [
            ("sup1_softmax_hss_CF", "front/CF/hss"),
            ("sup1_softmax_hss_hard_CF", "front/CF/hss_hard"),
            ("sup1_softmax_csi_DL", "front/DL/csi"),
            ("sup1_softmax_pod_OF", "front/OF/pod"),
            ("sup1_softmax_loss_CF", "front/CF/loss"),
            ("sup1_softmax_loss_none", "front/none/loss"),
            ("val_sup1_softmax_hss_CF", "front/CF/val_hss"),
            ("val_sup1_softmax_loss_none", "front/none/val_loss"),
        ],
    )
    def test_renames_per_front_type_keys(self, raw_key, renamed_key):
        logs = {raw_key: 0.42}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {renamed_key: pytest.approx(0.42)}

    @pytest.mark.parametrize("key", ["hss", "hss_hard", "loss", "val_loss", "val_hss"])
    def test_aggregate_keys_are_left_alone(self, key):
        logs = {key: 0.5}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {key: pytest.approx(0.5)}

    def test_key_containing_but_not_ending_with_front_type_token_is_untouched(self):
        """A key containing a front-type token without ending in one must not be renamed."""
        logs = {"CF_hss": 0.5}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {"CF_hss": pytest.approx(0.5)}

    def test_per_front_type_key_on_only_sup1_survives_with_value_intact(self):
        logs = {"sup1_softmax_hss_CF": 0.75}
        fc.MetricsConsolidationCallback().on_epoch_end(0, logs)
        assert logs == {"front/CF/hss": pytest.approx(0.75)}


class TestCompactProgressCallback:
    """Covers the terminal-width-bounded stdout progress display added to replace verbose=1."""

    _FRONT_TYPES: ClassVar[list[str]] = list(constants.FRONT_TYPE_CLASS_INDEX)

    def _make(self, monkeypatch, is_tty, every_n_batches=10, terminal_width=120, steps=450, epochs=5000):
        monkeypatch.setattr(fc.sys.stdout, "isatty", lambda: is_tty)
        monkeypatch.setattr(
            fc.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((terminal_width, 24))
        )
        callback = fc.CompactProgressCallback(every_n_batches=every_n_batches)
        callback.set_params({"steps": steps, "epochs": epochs})
        return callback

    def _logs(self, value_fn, with_validation=False):
        logs = {"loss": value_fn(0)}
        for i, front_type in enumerate(self._FRONT_TYPES, start=1):
            logs[f"front/{front_type}/hss"] = value_fn(i)
            logs[f"front/{front_type}/csi"] = value_fn(i + len(self._FRONT_TYPES))
        if with_validation:
            logs["val_loss"] = value_fn(100)
            for i, front_type in enumerate(self._FRONT_TYPES, start=1):
                logs[f"front/{front_type}/val_hss"] = value_fn(i + 200)
                logs[f"front/{front_type}/val_csi"] = value_fn(i + 300)
        return logs

    def test_default_row_fits_in_80_columns_without_truncation(self, monkeypatch, capsys):
        """The real regression guard: the row must fit by design, not merely avoid wrapping.

        Uses representative (non-edge-case) values at the terminal width the bug report called
        out (80 columns) and asserts every one of the five HSS and five CSI values is present in
        full — i.e. the truncation safety net in _truncate_to_terminal_width never engages here.
        """
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=1, terminal_width=80, steps=450)
        hss_values = [0.412, 0.342, 0.272, 0.202, 0.132]
        csi_values = [0.310, 0.250, 0.190, 0.130, 0.062]
        logs = {"loss": 0.0123}
        for front_type, hss, csi in zip(self._FRONT_TYPES, hss_values, csi_values, strict=True):
            logs[f"front/{front_type}/hss"] = hss
            logs[f"front/{front_type}/csi"] = csi
        callback.on_train_batch_end(311, logs)  # batch_number 312
        row = capsys.readouterr().out.lstrip("\r")
        untruncated = fc._render_metrics_row(
            fc._batch_label(312, 450), fc._row_label_width(450), 0.0123, hss_values, csi_values
        )
        assert row == untruncated, "the 80-column safety net truncated a row that should fit by design"
        assert len(row) < 79
        for value in hss_values + csi_values:
            assert f"{value:.3f}".lstrip("0") in row

    def test_narrow_terminal_safety_net_still_truncates_when_needed(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=40)
        logs = self._logs(lambda i: 12345.6789 + i)
        callback.on_epoch_begin(2, None)
        callback.on_train_batch_end(449, logs)  # final batch (steps=450) -> always updates
        out = capsys.readouterr().out
        for line in out.splitlines():
            assert len(line) <= 39

    def test_tty_batch_update_emits_carriage_return_and_no_newline(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True)
        logs = self._logs(lambda i: 0.1 * i)
        callback.on_train_batch_end(9, logs)  # batch_number 10 -> throttle boundary, fires
        out = capsys.readouterr().out
        assert out.startswith("\r")
        assert "\n" not in out

    def test_batch_update_shows_current_batch_number_not_final(self, monkeypatch, capsys):
        """Regression test for the coordinator's "showed 450/450 instead of 312/450" concern."""
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=1, steps=450, terminal_width=200)
        logs = self._logs(lambda i: 0.1 * i)
        callback.on_train_batch_end(311, logs)  # batch index 311 -> displayed batch number 312
        out = capsys.readouterr().out
        assert "312/450" in out
        assert "450/450" not in out

    def test_non_tty_emits_no_per_batch_output(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=False)
        logs = self._logs(lambda i: 0.1 * i)
        for batch in range(25):
            callback.on_train_batch_end(batch, logs)
        assert capsys.readouterr().out == ""

    def test_throttling_fires_expected_number_of_updates_plus_final(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=10, steps=25)
        logs = self._logs(lambda i: 0.1 * i)
        for batch in range(25):
            callback.on_train_batch_end(batch, logs)
        out = capsys.readouterr().out
        # Batches 10 and 20 hit the throttle boundary; batch 25 is the epoch's final batch.
        assert out.count("\r") == 3

    def test_header_names_front_types_in_constants_order(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        callback.on_epoch_begin(2, None)
        out = capsys.readouterr().out
        assert f"fronts: {' '.join(self._FRONT_TYPES)}" in out
        assert out == "Epoch 3/5000  fronts: CF WF SF OF DL\n"

    def test_header_is_printed_on_non_tty_too(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=False, terminal_width=200)
        callback.on_epoch_begin(2, None)
        out = capsys.readouterr().out
        assert "fronts: CF WF SF OF DL" in out

    def test_front_type_values_appear_in_constants_order_within_row(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200, steps=25)
        values = [0.1 * (i + 1) for i in range(len(self._FRONT_TYPES))]
        logs = {"loss": 0.5}
        for front_type, value in zip(self._FRONT_TYPES, values, strict=True):
            logs[f"front/{front_type}/hss"] = value
        callback.on_train_batch_end(24, logs)  # final batch -> always updates
        out = capsys.readouterr().out
        expected_hss = " ".join(f"{value:.3f}".lstrip("0") for value in values)
        assert expected_hss in out

    def test_missing_keys_degrade_gracefully_instead_of_raising(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200, steps=25)
        callback.on_train_batch_end(24, {})  # no metrics present at all; must not raise
        out = capsys.readouterr().out
        assert "--" in out

    def test_missing_keys_on_epoch_end_do_not_raise(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        callback.on_epoch_end(0, {})
        out = capsys.readouterr().out
        assert "--" in out

    def test_epoch_end_prints_two_rows_train_then_val(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        logs = self._logs(lambda i: 0.1 * i, with_validation=True)
        callback.on_epoch_end(2, logs)
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line]
        assert len(lines) == 2
        assert lines[0].lstrip("\r").strip().startswith("450/450")
        assert lines[1].strip().startswith("val")

    def test_epoch_end_val_row_is_not_truncated_away(self, monkeypatch, capsys):
        """The core bug fix: at epoch end, every validation value must survive in full."""
        callback = self._make(monkeypatch, is_tty=True, terminal_width=80)
        val_hss_values = [0.400, 0.330, 0.260, 0.190, 0.120]
        val_csi_values = [0.300, 0.240, 0.180, 0.120, 0.060]
        logs = {"loss": 0.0123, "val_loss": 0.0141}
        for front_type, hss, csi, val_hss, val_csi in zip(
            self._FRONT_TYPES,
            [0.412, 0.342, 0.272, 0.202, 0.132],
            [0.310, 0.250, 0.190, 0.130, 0.062],
            val_hss_values,
            val_csi_values,
            strict=True,
        ):
            logs[f"front/{front_type}/hss"] = hss
            logs[f"front/{front_type}/csi"] = csi
            logs[f"front/{front_type}/val_hss"] = val_hss
            logs[f"front/{front_type}/val_csi"] = val_csi
        callback.on_epoch_end(0, logs)
        out = capsys.readouterr().out
        val_line = next(line for line in out.splitlines() if line.strip().startswith("val"))
        assert "loss .0141" in val_line
        assert "HSS" in val_line
        assert "CSI" in val_line
        for value in val_hss_values + val_csi_values:
            expected = f"{value:.3f}".lstrip("0")
            assert expected in val_line, f"val value {expected} missing from: {val_line!r}"

    def test_non_tty_epoch_end_also_prints_two_rows_with_val_values(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=False, terminal_width=80)
        logs = self._logs(lambda i: 0.1 * i, with_validation=True)
        callback.on_epoch_end(2, logs)
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert len(lines) == 2
        assert "\r" not in out
        assert lines[1].strip().startswith("val")

    def test_train_and_val_rows_share_label_width_for_column_alignment(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, terminal_width=200)
        logs = self._logs(lambda i: 0.1 * i, with_validation=True)
        callback.on_epoch_end(0, logs)
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line]
        train_loss_col = lines[0].index("loss")
        val_loss_col = lines[1].index("loss")
        assert train_loss_col == val_loss_col

    def test_epoch_end_writes_real_newlines_so_next_epoch_starts_fresh(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True)
        logs = self._logs(lambda i: 0.1 * i, with_validation=True)
        callback.on_epoch_end(0, logs)
        out = capsys.readouterr().out
        assert out.endswith("\n")
        assert out.count("\n") == 2

    def test_single_space_separators_and_no_leading_zero_in_representative_row(self, monkeypatch, capsys):
        callback = self._make(monkeypatch, is_tty=True, every_n_batches=1, terminal_width=200, steps=450)
        logs = {"loss": 0.0123}
        hss_values = [0.412, 0.342, 0.272, 0.202, 0.132]
        csi_values = [0.310, 0.250, 0.190, 0.130, 0.062]
        for front_type, hss, csi in zip(self._FRONT_TYPES, hss_values, csi_values, strict=True):
            logs[f"front/{front_type}/hss"] = hss
            logs[f"front/{front_type}/csi"] = csi
        callback.on_train_batch_end(311, logs)
        out = capsys.readouterr().out
        assert "  312/450 loss .0123 HSS .412 .342 .272 .202 .132 CSI .310 .250 .190 .130 .062" in out


class TestBuildDatasetShapeSummary:
    def test_builds_shape_and_date_range(self):
        times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
        summary = fc.build_dataset_shape_summary(
            split="train", input_shape=(3, 4, 5, 2), target_shape=(3, 4, 5), times=times
        )
        assert summary.split == "train"
        assert summary.input_shape == (3, 4, 5, 2)
        assert summary.target_shape == (3, 4, 5)
        assert summary.date_min == "2020-01-01"
        assert summary.date_max == "2020-01-03"

    def test_out_of_order_times_still_yield_true_min_max(self):
        times = np.array(["2020-03-01", "2020-01-05", "2020-02-10"], dtype="datetime64[D]")
        summary = fc.build_dataset_shape_summary(
            split="val", input_shape=(3, 2, 2), target_shape=(3, 2, 2), times=times
        )
        assert summary.date_min == "2020-01-05"
        assert summary.date_max == "2020-03-01"

    def test_empty_times_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fc.build_dataset_shape_summary(
                split="test", input_shape=(0, 2, 2), target_shape=(0, 2, 2), times=np.array([], dtype="datetime64[D]")
            )


class TestDatasetSummaryCallback:
    def _make_summaries(self) -> list["fc.DatasetShapeSummary"]:
        return [
            fc.DatasetShapeSummary(
                split="train",
                input_shape=(10, 4, 5, 2),
                target_shape=(10, 4, 5),
                date_min="2020-01-01",
                date_max="2020-06-01",
            ),
            fc.DatasetShapeSummary(
                split="val",
                input_shape=(2, 4, 5, 2),
                target_shape=(2, 4, 5),
                date_min="2020-06-02",
                date_max="2020-07-01",
            ),
        ]

    def test_logs_every_split(self, caplog):
        cb = fc.DatasetSummaryCallback(self._make_summaries())
        with caplog.at_level("INFO", logger="fronts.callbacks"):
            cb.on_train_begin()
        assert "train split" in caplog.text
        assert "val split" in caplog.text
        assert "2020-01-01" in caplog.text
        assert "2020-07-01" in caplog.text

    def test_updates_wandb_summary_when_run_active(self, monkeypatch):
        summary_updates = {}
        fake_run = type("FakeRun", (), {"summary": type("FakeSummary", (), {"update": summary_updates.update})()})()
        monkeypatch.setattr(fc.wandb, "run", fake_run)

        cb = fc.DatasetSummaryCallback(self._make_summaries())
        cb.on_train_begin()

        assert summary_updates["data/train"]["input_shape"] == [10, 4, 5, 2]
        assert summary_updates["data/train"]["date_min"] == "2020-01-01"
        assert summary_updates["data/val"]["date_max"] == "2020-07-01"

    def test_no_wandb_call_when_no_run_active(self, monkeypatch):
        monkeypatch.setattr(fc.wandb, "run", None)
        cb = fc.DatasetSummaryCallback(self._make_summaries())
        cb.on_train_begin()  # Must not raise even with no active run.


class TestSelectActiveTestTimestep:
    def test_returns_first_timestep_with_a_front(self):
        data = np.zeros((4, 3, 3), dtype=np.int32)
        data[2, 1, 1] = 1  # CF code at time index 2
        target_da = xr.DataArray(data, dims=["time", "latitude", "longitude"])
        assert fc.select_active_test_timestep(target_da) == 2

    def test_raises_when_no_front_present(self):
        data = np.zeros((4, 3, 3), dtype=np.int32)
        target_da = xr.DataArray(data, dims=["time", "latitude", "longitude"])
        with pytest.raises(ValueError):
            fc.select_active_test_timestep(target_da)


class TestSelectTestSubsample:
    def test_bounded_and_sorted(self):
        idxs = fc.select_test_subsample(n_total=100, sample_size=10, seed=0)
        assert len(idxs) == 10
        assert (np.diff(idxs) > 0).all()
        assert idxs.min() >= 0
        assert idxs.max() < 100

    def test_clamps_to_n_total(self):
        idxs = fc.select_test_subsample(n_total=5, sample_size=200, seed=0)
        assert len(idxs) == 5

    def test_deterministic_for_fixed_seed(self):
        a = fc.select_test_subsample(n_total=50, sample_size=10, seed=42)
        b = fc.select_test_subsample(n_total=50, sample_size=10, seed=42)
        np.testing.assert_array_equal(a, b)


class TestRegionMask:
    def test_whole_domain_is_all_true(self):
        lats = np.array([10.0, 20.0, 30.0])
        lons = np.array([100.0, 200.0])
        mask = fc.region_mask(lats, lons, None)
        assert mask.shape == (3, 2)
        assert mask.all()

    def test_box_restricts_lat_and_lon(self):
        lats = np.array([10.0, 20.0, 30.0, 40.0])
        lons = np.array([100.0, 150.0, 200.0, 250.0])
        region = fc.utils.BoundingBox(lat_min=20.0, lat_max=40.0, lon_min=150.0, lon_max=250.0)
        mask = fc.region_mask(lats, lons, region)
        expected = np.array(
            [
                [False, False, False, False],
                [False, True, True, True],
                [False, True, True, True],
                [False, True, True, True],
            ]
        )
        np.testing.assert_array_equal(mask, expected)


class TestVisualizationCallbackPredict:
    """Tests the viz for predictions in callbacks.

    CallbackPredict must chunk by predict_batch_size rather than calling the model on the full array at once:
    a single unbatched call on e.g. 200 full-resolution test timesteps allocates one huge activation buffer on top
    of training's already resident GPU memory and reliably OOMs (see callbacks.py:on_epoch_end).
    """

    def _make_callback(self, n_samples: int, predict_batch_size: int) -> "fc.TestVisualizationCallback":
        inputs = fc.tf.keras.Input(shape=(2, 2, 1))
        model = fc.tf.keras.Model(inputs, inputs)  # identity: output == input
        cb = fc.TestVisualizationCallback(
            active_day_x=np.zeros((2, 2, 1), dtype=np.float32),
            active_day_y=np.zeros((2, 2, 1), dtype=np.float32),
            active_day_label="active day",
            subsample_x=np.arange(n_samples * 4, dtype=np.float32).reshape(n_samples, 2, 2, 1),
            subsample_y=np.zeros((n_samples, 2, 2, 1), dtype=np.float32),
            lats=np.array([0.0, 1.0]),
            lons=np.array([0.0, 1.0]),
            front_types=["CF"],
            predict_batch_size=predict_batch_size,
        )
        cb.set_model(model)
        return cb

    def test_chunked_prediction_matches_unbatched_input(self):
        # 5 samples with batch_size=2 forces a ragged last chunk (2, 2, 1 samples).
        cb = self._make_callback(n_samples=5, predict_batch_size=2)
        result = cb._predict(cb.subsample_x)
        np.testing.assert_allclose(result, cb.subsample_x)

    def test_never_calls_predict_on_the_full_unchunked_array(self, monkeypatch):
        # Calling model.predict() on the whole subsample at once accumulates every batch's
        # output into one GPU-resident tensor before returning, which is exactly what OOMs on
        # large full-domain subsamples. _predict must call predict() once per
        # predict_batch_size-sized chunk instead (see callbacks.py:_predict) — not
        # predict_on_batch(), which under MirroredStrategy hands its input to
        # distribute_strategy.run() undistributed, so every replica runs the forward pass on
        # the whole chunk and the (duplicate) per-replica outputs get concatenated together,
        # inflating the result to num_replicas x chunk_size rows.
        cb = self._make_callback(n_samples=5, predict_batch_size=2)
        real_predict = cb.model.predict
        call_sizes = []

        def tracking_predict(x, *a, **k):
            call_sizes.append(len(x))
            return real_predict(x, *a, **k)

        monkeypatch.setattr(cb.model, "predict", tracking_predict)
        result = cb._predict(cb.subsample_x)
        np.testing.assert_allclose(result, cb.subsample_x)
        assert call_sizes == [2, 2, 1]

    def test_predict_batch_size_field_is_required(self):
        with pytest.raises(TypeError):
            fc.TestVisualizationCallback(
                active_day_x=np.zeros((2, 2, 1), dtype=np.float32),
                active_day_y=np.zeros((2, 2, 1), dtype=np.float32),
                active_day_label="active day",
                subsample_x=np.zeros((1, 2, 2, 1), dtype=np.float32),
                subsample_y=np.zeros((1, 2, 2, 1), dtype=np.float32),
                lats=np.array([0.0, 1.0]),
                lons=np.array([0.0, 1.0]),
                front_types=["CF"],
            )


class TestVisualizationCallbackOnEpochEnd:
    """On_epoch_end must not pass an explicit `step` to wandb.log: WandbMetricsLogger's.

    Step is the cumulative training batch count, not the epoch number, so a `step=epoch`
    call is always behind the run's current step and gets silently dropped by wandb
    (see callbacks.py:on_epoch_end).
    """

    def _make_callback(self, monkeypatch, every_n_epochs: int) -> "fc.TestVisualizationCallback":
        # CF is class index 1, so 2 channels is the minimum needed to exercise the
        # class-index slicing in on_epoch_end.
        inputs = fc.tf.keras.Input(shape=(2, 2, 2))
        model = fc.tf.keras.Model(inputs, inputs)  # identity
        cb = fc.TestVisualizationCallback(
            active_day_x=np.zeros((2, 2, 2), dtype=np.float32),
            active_day_y=np.zeros((2, 2, 2), dtype=np.float32),
            active_day_label="active day",
            subsample_x=np.zeros((3, 2, 2, 2), dtype=np.float32),
            subsample_y=np.zeros((3, 2, 2, 2), dtype=np.float32),
            lats=np.array([0.0, 1.0]),
            lons=np.array([0.0, 1.0]),
            front_types=["CF"],
            predict_batch_size=2,
            every_n_epochs=every_n_epochs,
        )
        cb.set_model(model)
        # Plotting (cartopy map + table figure) is unrelated to the wandb step bug and
        # would otherwise drag in real map rendering; substitute cheap bare figures.
        monkeypatch.setattr(fc.plot_module, "plot_test_prediction", lambda **_: fc.plot_module.plt.figure())
        monkeypatch.setattr(fc.plot_module, "plot_performance_diagram_lite", lambda **_: fc.plot_module.plt.figure())
        return cb

    def test_logs_one_payload_with_no_explicit_step(self, monkeypatch):
        cb = self._make_callback(monkeypatch, every_n_epochs=1)
        calls = []
        monkeypatch.setattr(fc.wandb, "log", lambda payload, **kwargs: calls.append((payload, kwargs)))

        cb.on_epoch_end(epoch=0)

        assert len(calls) == 1
        payload, kwargs = calls[0]
        assert "step" not in kwargs
        assert "test/prediction" in payload
        assert any(k.startswith("test/performance_diagram/") for k in payload)

    def test_skips_logging_outside_cadence(self, monkeypatch):
        cb = self._make_callback(monkeypatch, every_n_epochs=10)
        calls = []
        monkeypatch.setattr(fc.wandb, "log", lambda payload, **kwargs: calls.append((payload, kwargs)))

        cb.on_epoch_end(epoch=0)

        assert calls == []


class TestAccumulateLiteStats:
    def test_matches_hand_computed_counts(self):
        # (time=1, lat=2, lon=2, n_fronts=1)
        pred = np.array([[[0.9], [0.1]], [[0.4], [0.6]]], dtype=np.float32).reshape(1, 2, 2, 1)
        truth = np.array([[1, 0], [0, 1]], dtype=np.float32).reshape(1, 2, 2, 1)
        weights = np.ones((2, 2), dtype=np.float32)
        thresholds = np.array([0.5], dtype=np.float32)

        tp, fp, tn, fn = fc.accumulate_lite_stats(pred, truth, weights, thresholds)

        assert tp[0, 0] == pytest.approx(2.0)
        assert fp[0, 0] == pytest.approx(0.0)
        assert tn[0, 0] == pytest.approx(2.0)
        assert fn[0, 0] == pytest.approx(0.0)

    def test_zero_weight_excludes_pixel(self):
        pred = np.array([[[0.9], [0.9]], [[0.9], [0.9]]], dtype=np.float32).reshape(1, 2, 2, 1)
        truth = np.ones((1, 2, 2, 1), dtype=np.float32)
        weights = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        thresholds = np.array([0.5], dtype=np.float32)

        tp, fp, tn, fn = fc.accumulate_lite_stats(pred, truth, weights, thresholds)

        # Only the two weight=1 pixels (both true positives) should count.
        assert tp[0, 0] == pytest.approx(2.0)
        assert fp[0, 0] == pytest.approx(0.0)
        assert tn[0, 0] == pytest.approx(0.0)
        assert fn[0, 0] == pytest.approx(0.0)
