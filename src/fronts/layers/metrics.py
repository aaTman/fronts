"""
Custom metrics for U-Net models.
    - Brier Skill Score (BSS)
    - Critical Success Index (CSI)
    - Fractions Skill Score (FSS)
    - Heidke Skill Score (HSS)
    - Probability of Detection (POD)

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.12.19
"""

import tensorflow as tf


def brier_skill_score(class_weights: list[int | float, ...] = None):
    """
    Brier skill score (BSS).

    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def bss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        squared_errors = tf.math.square(tf.subtract(y_true, y_pred))

        if class_weights is not None:
            relative_class_weights = tf.cast(
                class_weights / tf.math.reduce_sum(class_weights), tf.float32
            )
            squared_errors *= relative_class_weights

        bss = 1 - tf.math.reduce_sum(squared_errors) / tf.size(squared_errors)

        return bss

    return bss


def critical_success_index(
    threshold: float = None,
    window_size: tuple[int, ...] | list[int, ...] = None,
    class_weights: list[int | float, ...] = None,
):
    """
    Critical success index (CSI).

    threshold: float or None
        Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        If the threshold is set, it must be greater than 0 and less than 1.
    window_size: tuple or list of ints or None
        Pool/kernel size of the max-pooling window for neighborhood statistics. (e.g. if calculating the CSI with a 3-pixel
            window, this should be set to 3).
        Note that this parameter is experimental and may return unexpected results.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def csi(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if window_size is not None:
            y_pred = tf.nn.max_pool(
                y_pred, ksize=window_size, strides=1, padding="VALID"
            )
            y_true = tf.nn.max_pool(
                y_true, ksize=window_size, strides=1, padding="VALID"
            )

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
            relative_class_weights = tf.cast(
                class_weights / tf.math.reduce_sum(class_weights), tf.float32
            )
            csi = tf.math.reduce_sum(
                tf.math.divide_no_nan(
                    true_positives, true_positives + false_positives + false_negatives
                )
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
    mask_size: int | tuple[int, ...] | list[int, ...] = (3, 3),
    c: float = 1.0,
    binary: bool = False,
    threshold: float = 0.5,
    class_weights: list[int | float, ...] = None,
):
    """
    Fractions skill score metric.

    Parameters
    ----------
    mask_size: int or tuple
        Size of the mask/pool in the AveragePooling layers.
    c: int or float
        C parameter in the sigmoid function. This will only be used if 'binary' is False.
    binary: bool
        Convert y_pred to binary values (0/1).
    threshold: float
        If binary is False, this threshold is used in the sigmoid function.
        If binary is True, this is the threshold used to convert y_pred to binary values (0/1).
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.

    Returns
    -------
    fss: float
        Fractions skill score.

    References
    ----------
    (RL2008) Roberts, N. M., and H. W. Lean, 2008: Scale-Selective Verification of Rainfall Accumulations from High-Resolution
        Forecasts of Convective Events. Mon. Wea. Rev., 136, 78-97, https://doi.org/10.1175/2007MWR2123.1.
    """

    # if mask_size is an int, convert to a tuple
    if isinstance(mask_size, int):
        mask_size = (mask_size,)
    elif isinstance(mask_size, list):
        mask_size = tuple(mask_size)

    num_dims = len(mask_size)

    assert 1 <= num_dims <= 3, (
        "mask_size must have length between 1 and 3, received length %d" % num_dims
    )

    pool_kwargs = {
        "pool_size": mask_size,
        "strides": (1,) * num_dims,
        "padding": "valid",
    }

    pool_class = getattr(tf.keras.layers, "AveragePooling%dD" % num_dims)
    pool1 = pool_class(**pool_kwargs)
    pool2 = pool_class(**pool_kwargs)

    @tf.function
    def fss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if binary:
            y_true_disc = tf.where(y_true > threshold, 1.0, 0.0)
            y_pred_disc = tf.where(y_pred > threshold, 1.0, 0.0)
        else:
            y_true_disc = tf.math.sigmoid(c * (y_true - threshold))
            y_pred_disc = tf.math.sigmoid(c * (y_pred - threshold))

        y_true_density = pool1(y_true_disc)
        n_density_pixels = tf.cast(
            (tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]), tf.float32
        )

        y_pred_density = pool2(y_pred_disc)

        if class_weights is None:
            MSE_n = tf.keras.metrics.mean_squared_error(y_true_density, y_pred_density)
        else:
            relative_class_weights = tf.cast(
                class_weights / tf.math.reduce_sum(class_weights), tf.float32
            )
            MSE_n = tf.reduce_mean(
                tf.math.square(y_true_density - y_pred_density)
                * relative_class_weights,
                axis=-1,
            )

        O_n_squared_image = tf.keras.layers.Multiply()(
            [y_true_density, y_true_density]
        )
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        M_n_squared_image = tf.keras.layers.Multiply()(
            [y_pred_density, y_pred_density]
        )
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)

        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels

        epsilon = tf.keras.backend.epsilon()  # 1e-7, constant for numeric stability

        if binary:
            if MSE_n_ref == 0:
                return 1 - MSE_n
            else:
                return 1 - (MSE_n / MSE_n_ref)
        else:
            return 1 - (MSE_n / (MSE_n_ref + epsilon))

    return fss


def heidke_skill_score(
    threshold: float = None,
    window_size: tuple[int, ...] | list[int, ...] = None,
    class_weights: list[int | float, ...] = None,
):
    """
    Heidke Skill Score (HSS).

    threshold: float or None
        Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        If the threshold is set, it must be greater than 0 and less than 1.
    window_size: tuple or list of ints or None
        Pool/kernel size of the max-pooling window for neighborhood statistics. (e.g. if calculating the HSS with a 3-pixel
            window, this should be set to 3).
        Note that this parameter is experimental and may return unexpected results.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def hss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if window_size is not None:
            y_pred = tf.nn.max_pool(
                y_pred, ksize=window_size, strides=1, padding="VALID"
            )
            y_true = tf.nn.max_pool(
                y_true, ksize=window_size, strides=1, padding="VALID"
            )

        if threshold is not None:
            y_pred = tf.where(y_pred >= threshold, 1.0, 0.0)

        sum_over_axes = tf.range(
            tf.rank(y_pred) - 1
        )  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_true * y_pred, axis=sum_over_axes)
        false_positives = tf.math.reduce_sum((1 - y_true) * y_pred, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_true * (1 - y_pred), axis=sum_over_axes)
        true_negatives = tf.math.reduce_sum(
            (1 - y_true) * (1 - y_pred), axis=sum_over_axes
        )

        if class_weights is not None:
            relative_class_weights = tf.cast(
                class_weights / tf.math.reduce_sum(class_weights), tf.float32
            )
            true_positives *= relative_class_weights
            true_negatives *= relative_class_weights
            false_positives *= relative_class_weights
            false_negatives *= relative_class_weights

        a = tf.math.reduce_sum(true_positives)
        b = tf.math.reduce_sum(false_positives)
        c = tf.math.reduce_sum(false_negatives)
        d = tf.math.reduce_sum(true_negatives)

        hss = 2 * tf.math.divide(
            (a * d) - (b * c), ((a + c) * (c + d)) + ((a + b) * (b + d))
        )

        return hss

    return hss


def probability_of_detection(
    threshold: float = None,
    window_size: tuple[int, ...] | list[int, ...] = None,
    class_weights: list[int | float, ...] = None,
):
    """
    Probability of Detection (POD).

    threshold: float or None
        Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        If the threshold is set, it must be greater than 0 and less than 1.
    window_size: tuple or list of ints or None
        Pool/kernel size of the max-pooling window for neighborhood statistics. (e.g. if calculating the POD with a 5-pixel
            window, this should be set to 5).
        Note that this parameter is experimental and may return unexpected results.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def pod(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if window_size is not None:
            y_pred = tf.nn.max_pool(
                y_pred, ksize=window_size, strides=1, padding="VALID"
            )
            y_true = tf.nn.max_pool(
                y_true, ksize=window_size, strides=1, padding="VALID"
            )

        y_pred = (
            tf.where(y_pred >= threshold, 1.0, 0.0)
            if threshold is not None
            else y_pred
        )
        y_pred_neg = 1 - y_pred

        sum_over_axes = tf.range(
            tf.rank(y_pred) - 1
        )  # Indices for axes to sum over. Excludes the final (class) dimension.

        true_positives = tf.math.reduce_sum(y_pred * y_true, axis=sum_over_axes)
        false_negatives = tf.math.reduce_sum(y_pred_neg * y_true, axis=sum_over_axes)

        if class_weights is not None:
            relative_class_weights = tf.cast(
                class_weights / tf.math.reduce_sum(class_weights), tf.float32
            )
            pod = tf.math.reduce_sum(
                tf.math.divide_no_nan(
                    true_positives, true_positives + false_negatives
                )
                * relative_class_weights
            )
        else:
            pod = tf.math.reduce_sum(
                tf.math.divide_no_nan(
                    true_positives, true_positives + false_negatives
                )
            )

        return pod

    return pod
