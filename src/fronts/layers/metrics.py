"""Custom metrics for U-Net models: BSS, CSI, FSS, HSS, and POD."""

from collections.abc import Callable

import tensorflow as tf

_VALID_STATISTICS = ("hss", "csi", "pod")


def brier_skill_score(
    class_weights: list[int | float] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Brier skill score (BSS).

    Args:
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """

    @tf.function
    def bss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute BSS for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        squared_errors = tf.math.square(tf.subtract(y_true, y_pred))

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            squared_errors *= relative_class_weights

        bss = 1 - tf.math.reduce_sum(squared_errors) / tf.size(squared_errors)

        return bss

    return bss


def critical_success_index(
    threshold: float | None = None,
    window_size: tuple[int, ...] | list[int] | None = None,
    class_weights: list[int | float] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Critical success index (CSI).

    Args:
        threshold: Optional probability threshold that binarizes y_pred. Values >= threshold are set to 1, others to 0.
            Must be in (0, 1) if provided.
        window_size: Pool/kernel size of the max-pooling window for neighborhood statistics. Experimental; may return
            unexpected results.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """

    @tf.function
    def csi(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute CSI for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if window_size is not None:
            y_pred = tf.nn.max_pool(y_pred, ksize=window_size, strides=1, padding="VALID")
            y_true = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")

        if threshold is not None:
            y_pred = tf.where(y_pred >= threshold, 1.0, 0.0)

        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        sum_over_axes = tf.range(
            tf.rank(y_pred) - 1
        )  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)
        false_positives = tf.math.reduce_sum(y_pred * y_true_neg, axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            csi = tf.math.reduce_sum(
                tf.math.divide_no_nan(true_positives, true_positives + false_positives + false_negatives)
                * relative_class_weights
            )
        else:
            csi = tf.math.divide(
                tf.math.reduce_sum(true_positives),
                tf.math.reduce_sum(true_positives)
                + tf.math.reduce_sum(false_negatives)
                + tf.math.reduce_sum(false_positives),
            )

        return csi

    return csi


def fractions_skill_score(
    mask_size: int | tuple[int, ...] | list[int] = (3, 3),
    alpha: int | float = 1.0,
    beta: int | float = 0.5,
    class_weights: list[int | float] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Fractions skill score (FSS) metric.

    Args:
        mask_size: Size of the mask/pool in the AveragePooling layers.
        alpha: Controls how steep the sigmoid function is for discretization. Higher values make it steeper.
        beta: Controls some behaviors of the sigmoid discretization function. Default and recommended value is 0.5.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.

    References:
        Roberts & Lean (2008): https://doi.org/10.1175/2007MWR2123.1
    """
    # keyword arguments for the AveragePooling layer
    pool_args = {"pool_size": mask_size, "strides": 1, "padding": "same"}

    if isinstance(mask_size, int):
        mask_size = (mask_size,)
    elif isinstance(mask_size, list):
        mask_size = tuple(mask_size)

    assert 1 <= len(mask_size) <= 3, f"mask_size must have length between 1 and 3, received length {len(mask_size)}"

    pool = getattr(tf.keras.layers, f"AveragePooling{len(mask_size)}D")(**pool_args)

    cw: tf.Tensor | None = tf.cast(class_weights, tf.float32) if class_weights is not None else None
    relative_cw: tf.Tensor | None = (cw / tf.reduce_sum(cw)) if cw is not None else None

    @tf.function
    def fss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute FSS for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # discretize model predictions and labels
        y_true = tf.math.sigmoid(alpha * (y_true - beta))
        y_pred = tf.math.sigmoid(alpha * (y_pred - beta))

        O_n = pool(y_true)  # observed fractions (Eq. 2 in RL2008)
        M_n = pool(y_pred)  # model forecast fractions (Eq. 3 in RL2008)

        # Class weights applied after pooling so sigmoid targets remain correct for zero-weight classes.
        if relative_cw is not None:
            MSE_n = tf.reduce_mean(tf.square(O_n - M_n) * relative_cw)  # (Eq. 5 in RL2008)
            MSE_ref = tf.reduce_mean(tf.square(O_n) * relative_cw) + tf.reduce_mean(
                tf.square(M_n) * relative_cw
            )  # (Eq. 7 in RL2008)
        else:
            MSE_n = tf.reduce_mean(tf.square(O_n - M_n))  # (Eq. 5 in RL2008)
            MSE_ref = tf.reduce_mean(tf.square(O_n)) + tf.reduce_mean(tf.square(M_n))  # (Eq. 7 in RL2008)

        FSS = 1 - tf.math.divide_no_nan(MSE_n, MSE_ref)  # fractions skill score (Eq. 6 in RL2008)

        return FSS

    return fss


def _crop_pred_buffer(field: tf.Tensor, buffer_lat_px: int, buffer_lon_px: int) -> tf.Tensor:
    """Crops a per-axis buffer off the latitude and longitude axes of ``field``.

    Latitude and longitude are cropped independently, since patch-buffer training (see
    fronts.data.datasets.PatchConfig) only ever pads longitude — every patch already spans the
    full latitude height of the domain, so there is no artificial tile cut to compensate for along
    that axis (Ronneberger et al. 2015's overlap-tile buffering only matters where a tile is
    actually cut). A naive ``field[:, buffer:-buffer, ...]`` slice is wrong when ``buffer == 0``,
    since ``-0 == 0`` makes the slice ``field[:, 0:0, ...]`` — an empty tensor rather than a
    no-op — so each axis is only sliced when its buffer is strictly positive.

    Args:
        field: Tensor shaped (batch, latitude, longitude, classes) to crop.
        buffer_lat_px: Pixels to remove from each side of the latitude (axis 1). 0 is a no-op.
        buffer_lon_px: Pixels to remove from each side of the longitude (axis 2). 0 is a no-op.

    Returns:
        ``field`` with the requested per-axis buffer removed from both sides.
    """
    if buffer_lat_px > 0:
        field = field[:, buffer_lat_px:-buffer_lat_px, :, :]
    if buffer_lon_px > 0:
        field = field[:, :, buffer_lon_px:-buffer_lon_px, :]
    return field


def heidke_skill_score(
    threshold: float | None = None,
    window_size: tuple[int, ...] | list[int] | None = None,
    class_weights: list[int | float] | None = None,
    pred_buffer_lat_px: int = 0,
    pred_buffer_lon_px: int = 0,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Heidke Skill Score (HSS).

    Args:
        threshold: Optional probability threshold that binarizes y_pred. Values >= threshold are set to 1, others to 0.
            Must be in (0, 1) if provided.
        window_size: Pool/kernel size of the max-pooling window for neighborhood statistics. Experimental; may return
            unexpected results.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
        pred_buffer_lat_px: If > 0, y_pred is expected to carry this many extra pixels of context on
            each side of the latitude axis beyond y_true's shape (e.g. from a patch trained with an
            input-only buffer — see fronts.data.datasets.PatchConfig). y_pred is cropped by this
            amount on the latitude axis (after ``window_size`` pooling, if any) before scoring
            against y_true. 0 (default) requires y_pred and y_true to share the same latitude
            extent, matching prior behavior.
        pred_buffer_lon_px: Same as pred_buffer_lat_px, but for the longitude axis. Buffering is
            per-axis because patch-buffer training only ever pads longitude — every patch already
            spans the full latitude height of the domain.
    """

    @tf.function
    def hss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute HSS for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions. When pred_buffer_lat_px/pred_buffer_lon_px
                > 0, this is that many pixels wider than y_true on the corresponding spatial axis.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if window_size is not None:
            y_pred = tf.nn.max_pool(y_pred, ksize=window_size, strides=1, padding="VALID")
            y_true = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")

        y_pred = _crop_pred_buffer(y_pred, pred_buffer_lat_px, pred_buffer_lon_px)

        if threshold is not None:
            y_pred = tf.where(y_pred >= threshold, 1.0, 0.0)

        sum_over_axes = tf.range(
            tf.rank(y_pred) - 1
        )  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_true * y_pred, axis=sum_over_axes)
        false_positives = tf.math.reduce_sum((1 - y_true) * y_pred, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_true * (1 - y_pred), axis=sum_over_axes)
        true_negatives = tf.math.reduce_sum((1 - y_true) * (1 - y_pred), axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            true_positives *= relative_class_weights
            true_negatives *= relative_class_weights
            false_positives *= relative_class_weights
            false_negatives *= relative_class_weights

        a = tf.math.reduce_sum(true_positives)
        b = tf.math.reduce_sum(false_positives)
        c = tf.math.reduce_sum(false_negatives)
        d = tf.math.reduce_sum(true_negatives)

        hss = 2 * tf.math.divide_no_nan((a * d) - (b * c), ((a + c) * (c + d)) + ((a + b) * (b + d)))

        return hss

    return hss


def probability_of_detection(
    threshold: float | None = None,
    window_size: tuple[int, ...] | list[int] | None = None,
    class_weights: list[int | float] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Probability of Detection (POD).

    Args:
        threshold: Optional probability threshold that binarizes y_pred. Values >= threshold are set to 1, others to 0.
            Must be in (0, 1) if provided.
        window_size: Pool/kernel size of the max-pooling window for neighborhood statistics. Experimental; may return
            unexpected results.
        class_weights: Weights to apply to each class. Length must equal the number of classes in y_pred and y_true.
    """

    @tf.function
    def pod(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute POD for a batch.

        Args:
            y_true: One-hot encoded tensor containing labels.
            y_pred: Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if window_size is not None:
            y_pred = tf.nn.max_pool(y_pred, ksize=window_size, strides=1, padding="VALID")
            y_true = tf.nn.max_pool(y_true, ksize=window_size, strides=1, padding="VALID")

        y_pred = tf.where(y_pred >= threshold, 1.0, 0.0) if threshold is not None else y_pred
        y_pred_neg = 1 - y_pred

        sum_over_axes = tf.range(
            tf.rank(y_pred) - 1
        )  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(class_weights / tf.math.reduce_sum(class_weights), tf.float32)
            pod = tf.math.reduce_sum(
                tf.math.divide_no_nan(true_positives, true_positives + false_negatives) * relative_class_weights
            )
        else:
            pod = tf.math.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_negatives))

        return pod

    return pod


def _crop_pred_buffer_trailing_spatial(field: tf.Tensor, buffer_lat_px: int, buffer_lon_px: int) -> tf.Tensor:
    """Crops a per-axis prediction buffer off the two spatial axes preceding the class axis.

    Unlike the module-level ``_crop_pred_buffer`` above (which hardcodes axes 1 and 2, correct
    only for a 4D ``(batch, latitude, longitude, class)`` field), this helper locates latitude
    and longitude by counting from the END of the shape. That makes it correct for BOTH the 4D
    shape and the 5D ``(batch, level, latitude, longitude, class)`` shape
    ``PerClassContingencyMetric.update_state`` also accepts: in 5D, axis 1 is level, not
    latitude, so hardcoding axis 1 as latitude would silently crop the wrong axis (or, given
    mismatched axis sizes, raise a shape error instead of scoring anything). The class axis is
    always last, so latitude is always ``rank - 3`` and longitude is always ``rank - 2``,
    regardless of how many leading batch/level axes precede them.

    As with ``_crop_pred_buffer``, a naive ``field[..., buffer:-buffer, :]`` slice is wrong when
    ``buffer == 0`` (``-0 == 0`` makes it an empty-tensor slice, not a no-op), so each axis is
    only sliced when its buffer is strictly positive.

    Args:
        field: Tensor shaped ``(batch, ..., latitude, longitude, class)`` to crop, where ``...``
            is zero or more leading axes (e.g. pressure level).
        buffer_lat_px: Pixels to remove from each side of the latitude axis (``rank - 3``). 0 is
            a no-op.
        buffer_lon_px: Pixels to remove from each side of the longitude axis (``rank - 2``). 0 is
            a no-op.

    Returns:
        ``field`` with the requested per-axis buffer removed from both sides.

    Raises:
        ValueError: If ``field`` does not have a statically known rank, or that rank is too
            small to contain a latitude axis.
    """
    if buffer_lat_px == 0 and buffer_lon_px == 0:
        return field
    rank = field.shape.rank
    if rank is None:
        raise ValueError("field must have a statically known rank to crop its patch buffer.")
    lat_axis = rank - 3
    lon_axis = rank - 2
    if lat_axis < 0:
        raise ValueError(f"field rank {rank} is too small to have latitude/longitude axes to crop.")
    slices: list[slice] = [slice(None)] * rank
    if buffer_lat_px > 0:
        slices[lat_axis] = slice(buffer_lat_px, -buffer_lat_px)
    if buffer_lon_px > 0:
        slices[lon_axis] = slice(buffer_lon_px, -buffer_lon_px)
    return field[tuple(slices)]


class PerClassContingencyMetric(tf.keras.metrics.Metric):
    """Stateful contingency-table metric for a single front-type class.

    Unlike the module-level metric closures above (``heidke_skill_score``, etc.), which
    compute a ``divide_no_nan``-guarded ratio per batch, this metric accumulates the
    true/false positive/negative counts across every ``update_state`` call and derives
    the requested statistic once, at ``result()`` time, from the running totals.

    This distinction matters for rare front types. Wrapping the existing closures in
    ``tf.keras.metrics.MeanMetricWrapper`` would average a per-batch ratio across an
    epoch; a batch with zero observed pixels of a rare class (e.g. dryline, occluded
    front) makes that ratio ``0.0`` via ``divide_no_nan``, and averaging silently drags
    the epoch-level metric toward zero for exactly the rare classes this metric exists to
    expose. Accumulating counts and dividing once avoids that bias, and makes the reported
    numbers directly comparable to the ``hss_{ft}`` / ``csi_{ft}`` / ``pod_{ft}`` values
    ``evaluate.py`` computes offline from accumulated contingency tables.

    Args:
        class_index: Index into the class axis identifying the front type to score.
        statistic: One of ``"hss"``, ``"csi"``, ``"pod"``.
        threshold: When set, binarize ``y_pred`` at this probability threshold
            (``y_pred >= threshold``) before accumulating. When ``None``, the raw
            (soft) probabilities are accumulated.
        name: Name of the metric instance.
        dtype: Data type of the metric's state variables.
        pred_buffer_lat_px: If > 0, ``y_pred`` is expected to carry this many extra pixels
            of context on each side of the latitude axis beyond ``y_true``'s shape (e.g.
            from a patch trained with an input-only buffer — see
            ``fronts.data.datasets.PatchConfig``). ``y_pred`` is cropped by this amount on
            the latitude axis before accumulating against ``y_true``. 0 (default) requires
            ``y_pred`` and ``y_true`` to share the same latitude extent, matching prior
            behavior.
        pred_buffer_lon_px: Same as ``pred_buffer_lat_px``, but for the longitude axis.
            Buffering is per-axis because patch-buffer training only ever pads longitude —
            every patch already spans the full latitude height of the domain.
    """

    def __init__(
        self,
        class_index: int,
        statistic: str,
        threshold: float | None = None,
        name: str | None = None,
        dtype: tf.DType | str | None = None,
        pred_buffer_lat_px: int = 0,
        pred_buffer_lon_px: int = 0,
    ) -> None:
        if statistic not in _VALID_STATISTICS:
            raise ValueError(f"statistic must be one of {_VALID_STATISTICS}, got {statistic!r}")
        super().__init__(name=name, dtype=dtype)
        self.class_index = class_index
        self.statistic = statistic
        self.threshold = threshold
        self.pred_buffer_lat_px = pred_buffer_lat_px
        self.pred_buffer_lon_px = pred_buffer_lon_px
        self.true_positives = self.add_weight(name="true_positives", initializer="zeros")
        self.false_positives = self.add_weight(name="false_positives", initializer="zeros")
        self.false_negatives = self.add_weight(name="false_negatives", initializer="zeros")
        self.true_negatives = self.add_weight(name="true_negatives", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None) -> None:
        """Accumulate contingency counts for this class from one batch.

        Args:
            y_true: One-hot encoded tensor containing labels, with the class axis last.
                Works for both 4D ``(batch, lat, lon, class)`` and 5D
                ``(batch, level, lat, lon, class)`` inputs.
            y_pred: Tensor containing model predictions. Same shape as ``y_true`` when
                ``pred_buffer_lat_px``/``pred_buffer_lon_px`` are both 0 (the default).
                Otherwise ``y_pred`` is that many pixels wider than ``y_true`` on the
                corresponding spatial axis (latitude is always the second-to-last axis
                before longitude, and longitude the one before the class axis, in both
                the 4D and 5D input shapes) — see ``pred_buffer_lat_px``.
            sample_weight: Unused. Accepted for compatibility with the
                ``tf.keras.metrics.Metric`` interface; this metric is already restricted
                to one class and does not apply additional weighting.
        """
        del sample_weight
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_pred = _crop_pred_buffer_trailing_spatial(y_pred, self.pred_buffer_lat_px, self.pred_buffer_lon_px)

        y_true_class = tf.gather(y_true, self.class_index, axis=-1)
        y_pred_class = tf.gather(y_pred, self.class_index, axis=-1)

        if self.threshold is not None:
            y_pred_class = tf.where(y_pred_class >= self.threshold, 1.0, 0.0)

        y_true_neg = 1 - y_true_class
        y_pred_neg = 1 - y_pred_class

        self.true_positives.assign_add(tf.math.reduce_sum(y_true_class * y_pred_class))
        self.false_positives.assign_add(tf.math.reduce_sum(y_true_neg * y_pred_class))
        self.false_negatives.assign_add(tf.math.reduce_sum(y_true_class * y_pred_neg))
        self.true_negatives.assign_add(tf.math.reduce_sum(y_true_neg * y_pred_neg))

    def result(self) -> tf.Tensor:
        """Compute the configured statistic from the accumulated contingency counts."""
        a = self.true_positives
        b = self.false_positives
        c = self.false_negatives
        d = self.true_negatives

        if self.statistic == "hss":
            return 2 * tf.math.divide_no_nan((a * d) - (b * c), ((a + c) * (c + d)) + ((a + b) * (b + d)))
        if self.statistic == "csi":
            return tf.math.divide_no_nan(a, a + b + c)
        return tf.math.divide_no_nan(a, a + c)

    def reset_state(self) -> None:
        """Zero all accumulated contingency counts."""
        for variable in (self.true_positives, self.false_positives, self.false_negatives, self.true_negatives):
            variable.assign(0.0)

    def get_config(self) -> dict:
        """Return this metric's constructor arguments, for model serialization round-trips."""
        config = super().get_config()
        config.update(
            {
                "class_index": self.class_index,
                "statistic": self.statistic,
                "threshold": self.threshold,
                "pred_buffer_lat_px": self.pred_buffer_lat_px,
                "pred_buffer_lon_px": self.pred_buffer_lon_px,
            }
        )
        return config


def per_front_type_metrics(
    front_type_class_index: dict[str, int],
    hard_threshold: float = 0.5,
    pred_buffer_lat_px: int = 0,
    pred_buffer_lon_px: int = 0,
) -> list[tf.keras.metrics.Metric]:
    """Build the per-front-type metric set attached to the finest model output.

    Args:
        front_type_class_index: Mapping from front-type code (e.g. ``"CF"``) to its index
            in the one-hot encoded class axis. Typically
            ``fronts.constants.FRONT_TYPE_CLASS_INDEX``.
        hard_threshold: Probability threshold used for the hard-thresholded HSS, CSI, and
            POD metrics. CSI and POD only ever use the hard threshold, since their
            probabilistic forms are hard to interpret and the thresholded values are what
            the offline evaluation in ``evaluate.py`` reports.
        pred_buffer_lat_px: Forwarded to every ``PerClassContingencyMetric`` built — see
            ``PerClassContingencyMetric``'s ``pred_buffer_lat_px``. 0 (default) is correct
            for whole-domain training, where ``y_pred`` and ``y_true`` share the same
            latitude/longitude extent.
        pred_buffer_lon_px: Same as ``pred_buffer_lat_px``, but for the longitude axis.

    Returns:
        A list of four ``PerClassContingencyMetric`` instances per front type: soft HSS
        (``hss_{ft}``), hard-thresholded HSS (``hss_hard_{ft}``), hard-thresholded CSI
        (``csi_{ft}``), and hard-thresholded POD (``pod_{ft}``).
    """
    metrics_list: list[tf.keras.metrics.Metric] = []
    for front_type, class_index in front_type_class_index.items():
        metrics_list.append(
            PerClassContingencyMetric(
                class_index=class_index,
                statistic="hss",
                threshold=None,
                name=f"hss_{front_type}",
                pred_buffer_lat_px=pred_buffer_lat_px,
                pred_buffer_lon_px=pred_buffer_lon_px,
            )
        )
        metrics_list.append(
            PerClassContingencyMetric(
                class_index=class_index,
                statistic="hss",
                threshold=hard_threshold,
                name=f"hss_hard_{front_type}",
                pred_buffer_lat_px=pred_buffer_lat_px,
                pred_buffer_lon_px=pred_buffer_lon_px,
            )
        )
        metrics_list.append(
            PerClassContingencyMetric(
                class_index=class_index,
                statistic="csi",
                threshold=hard_threshold,
                name=f"csi_{front_type}",
                pred_buffer_lat_px=pred_buffer_lat_px,
                pred_buffer_lon_px=pred_buffer_lon_px,
            )
        )
        metrics_list.append(
            PerClassContingencyMetric(
                class_index=class_index,
                statistic="pod",
                threshold=hard_threshold,
                name=f"pod_{front_type}",
                pred_buffer_lat_px=pred_buffer_lat_px,
                pred_buffer_lon_px=pred_buffer_lon_px,
            )
        )
    return metrics_list
