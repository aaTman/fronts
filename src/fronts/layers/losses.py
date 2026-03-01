"""
Custom loss functions for U-Net models.
    - Brier Skill Score (BSS)
    - Critical Success Index (CSI)
    - Fractions Skill Score (FSS)
    - Probability of Detection (POD)

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.12.19
"""

import tensorflow as tf


def brier_skill_score(class_weights: list[int | float, ...] = None):
    """
    Brier skill score (BSS) loss function.

    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    """

    @tf.function
    def bss_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        losses = tf.math.square(tf.subtract(y_true, y_pred))

        if class_weights is not None:
            relative_class_weights = tf.cast(
                class_weights / tf.math.reduce_sum(class_weights), tf.float32
            )
            losses *= relative_class_weights

        brier_score_loss = tf.math.reduce_sum(losses) / tf.size(losses)
        return brier_score_loss

    return bss_loss


def critical_success_index(
    threshold: float = None,
    class_weights: list[int | float, ...] = None,
    window_size: int = None,
):
    """
    Critical Success Index (CSI) loss function.

    threshold: float or None
        Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        If the threshold is set, it must be greater than 0 and less than 1.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    window_size: int or None
        Pool/kernel size of the max-pooling window for neighborhood statistics. (e.g. if calculating the loss with a 4-pixel
            window, this should be set to 4).
        Note that this parameter is experimental and may return unexpected results.
    """

    @tf.function
    def csi_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

        if window_size is not None:
            y_pred = tf.nn.max_pool(
                y_pred, ksize=window_size, strides=1, padding="VALID"
            )
            y_true = tf.nn.max_pool(
                y_true, ksize=window_size, strides=1, padding="VALID"
            )

        if threshold is not None:
            y_pred = tf.where(y_pred >= threshold, 1, 0)

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
                    true_positives,
                    true_positives + false_positives + false_negatives,
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

        return 1 - csi

    return csi_loss


def fractions_skill_score(
    mask_size: int | tuple[int, ...] | list[int, ...] = (3, 3),
    c: float = 1.0,
    binary: bool = False,
    threshold: float = 0.5,
    class_weights: list[int | float, ...] = None,
):
    """
    Fractions skill score loss function.

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
    fss_loss: float
        fss_loss = 1 - fractions skill score

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
    def fss_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

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
            MSE_n = tf.reduce_mean(
                tf.math.square(y_true_density - y_pred_density), axis=-1
            )
        else:
            relative_class_weights = tf.cast(
                class_weights / tf.math.reduce_sum(class_weights), tf.float32
            )
            MSE_n = tf.reduce_mean(
                tf.math.square(y_true_density - y_pred_density)
                * relative_class_weights,
                axis=-1,
            )

        O_n_squared_sum = tf.reduce_sum(tf.math.square(y_true_density))
        M_n_squared_sum = tf.reduce_sum(tf.math.square(y_pred_density))

        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels

        epsilon = 1e-7  # constant for numeric stability

        if binary:
            if MSE_n_ref == 0:
                return MSE_n
            else:
                return MSE_n / MSE_n_ref
        else:
            return MSE_n / (MSE_n_ref + epsilon)

    return fss_loss


def probability_of_detection(
    threshold: float = None,
    class_weights: list[int | float, ...] = None,
    window_size: int = None,
):
    """
    Probability of Detection (POD) as a loss function. This turns the function into the miss rate.

    threshold: float or None
        Optional probability threshold that binarizes y_pred. Values in y_pred greater than or equal to the threshold are
            set to 1, and 0 otherwise.
        If the threshold is set, it must be greater than 0 and less than 1.
    class_weights: list of values or None
        List of weights to apply to each class. The length must be equal to the number of classes in y_pred and y_true.
    window_size: int or None
        Pool/kernel size of the max-pooling window for neighborhood statistics. (e.g. if calculating the loss with a 4-pixel
            window, this should be set to 4).
        Note that this parameter is experimental and may return unexpected results.
    """

    @tf.function
    def pod_loss(y_true, y_pred):
        """
        y_true: tf.Tensor
            One-hot encoded tensor containing labels.
        y_pred: tf.Tensor
            Tensor containing model predictions.
        """

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

        return 1 - pod

    return pod_loss
