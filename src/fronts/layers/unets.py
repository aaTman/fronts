"""
Deep learning models and functions for building:
    * U-Net
    * U-Net ensemble
    * U-Net+
    * U-Net++
    * U-Net 3+
    * Attention U-Net

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2024.10.11
"""

from tensorflow.keras.models import Model
import numpy as np

from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Conv3D,
    BatchNormalization,
    MaxPooling2D,
    MaxPooling3D,
    UpSampling2D,
    UpSampling3D,
    Concatenate,
    Input,
)
from fronts.utils import keras_builders
from fronts import model
import tensorflow as tf
import dataclasses
from typing import Literal


def attention_gate(
    x: tf.Tensor,
    g: tf.Tensor,
    kernel_size: int | tuple[int],
    pool_size: tuple[int],
    name: str or None = None,
):
    """
    Attention gate for the Attention U-Net.

    Parameters
    ----------
    x: tf.Tensor
        Signal that originates from the encoder node on the same level as the attention gate.
    g: tf.Tensor
        Signal that originates from the level below the attention gate, which has higher resolution features.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer(s). Only applies to layers that are not forced to a kernel size of 1.
    pool_size: tuple or list
        Pool size for the UpSampling layers, as well as the number of strides in the first
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    References
    ----------
    https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831
    """

    conv_layer = getattr(
        tf.keras.layers, f"Conv{len(x.shape) - 2}D"
    )  # Select the convolution layer for the x and g tensors
    upsample_layer = getattr(
        tf.keras.layers, f"UpSampling{len(x.shape) - 2}D"
    )  # Select the upsampling layer

    shape_x = x.shape  # Shapes of the ORIGINAL inputs
    filters_x = shape_x[-1]

    """
    x: Get the x tensor to the same shape as the gating signal (g tensor)
    g: Perform a 1x1-style convolution on the gating signal so it has the same number of filters as the x signal  
    """
    x_conv = conv_layer(
        filters=filters_x,
        kernel_size=kernel_size,
        strides=pool_size,
        padding="same",
        name=f"{name}_Conv{len(x.shape) - 2}D_x",
    )(x)
    g_conv = conv_layer(
        filters=filters_x,
        kernel_size=1,
        padding="same",
        name=f"{name}_Conv{len(x.shape) - 2}D_g",
    )(g)

    xg = tf.add(
        x_conv, g_conv, name=f"{name}_sum"
    )  # Sum the x and g signals element-wise
    xg = Activation(activation="relu", name=f"{name}_relu")(
        xg
    )  # Pass the summed signals through a ReLU activation layer

    xg_collapse = conv_layer(
        filters=1, kernel_size=1, padding="same", name=f"{name}_collapse"
    )(xg)  # Collapse the number of filters to just 1
    xg_collapse = Activation(activation="sigmoid", name=f"{name}_sigmoid")(
        xg_collapse
    )  # Pass collapsed tensor through a sigmoid activation layer

    # Upsample the collapsed tensor so its dimensions match the original shape of the x signal, then expand the filters to match the g signal filters
    upsample_xg = upsample_layer(
        size=pool_size, name=f"{name}_UpSampling{len(x.shape) - 2}D"
    )(xg_collapse)
    upsample_xg = tf.repeat(upsample_xg, filters_x, axis=-1, name=f"{name}_repeat")

    coeffs = tf.multiply(
        upsample_xg, x, name=f"{name}_multiply"
    )  # Element-wise multiplication onto the original x signal

    attention_tensor = conv_layer(
        filters=filters_x,
        kernel_size=1,
        strides=1,
        padding="same",
        name=f"{name}_Conv{len(x.shape) - 2}D_coeffs",
    )(coeffs)
    attention_tensor = BatchNormalization(name=f"{name}_BatchNorm")(attention_tensor)

    return attention_tensor


def convolution_module(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    num_modules: int = 1,
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str = None,
):
    """
    Insert modules into an encoder or decoder node.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module(s).
    filters: int
        Number of filters in the Conv2D/Conv3D layer(s).
    kernel_size: int or tuple of ints
        Size of the kernel in the Conv2D/Conv3D layer(s).
    num_modules: int
        Number of convolution modules to insert. Must be greater than 0, otherwise a ValueError exception is raised.
    padding: str
        Padding in the Conv2D/Conv3D layer(s). 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layers.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow every Conv2D/Conv3D layer.
    activation: str
        Activation function to use after every Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if num_modules < 1:
        raise ValueError(
            "num_modules must be greater than 0, at least one module must be added"
        )

    if (
        tensor_dims == 4
    ):  # A 2D image tensor has 4 dimensions: (None [for batch size], image_size_x, image_size_y, n_channels)
        conv_layer = Conv2D
    elif (
        tensor_dims == 5
    ):  # A 3D image tensor has 5 dimensions: (None [for batch size], image_size_x, image_size_y, image_size_z, n_channels)
        conv_layer = Conv3D
    else:
        raise TypeError(
            f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions"
        )

    # Arguments for the Conv2D/Conv3D layer.
    conv_kwargs = dict({})
    for arg in [
        "filters",
        "use_bias",
        "kernel_size",
        "padding",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
    ]:
        conv_kwargs[arg] = locals()[arg]

    activation_kwargs = {}
    if activation in [
        "prelu",
        "smelu",
        "snake",
    ]:  # these activation functions have learnable parameters
        activation_kwargs["shared_axes"] = shared_axes

    # Insert convolution modules
    for module in range(num_modules):
        # Create names for the Conv2D/Conv3D layers and the activation layer.
        conv_kwargs["name"] = f"{name}_Conv{tensor_dims - 2}D_{module + 1}"
        activation_kwargs["name"] = f"{name}_{activation}_{module + 1}"

        conv_tensor = conv_layer(**conv_kwargs)(
            tensor
        )  # Perform convolution on the input tensor

        activation_config = keras_builders.ActivationConfig(
            name=activation, config=activation_kwargs
        )
        activation_layer = activation_config.build()

        if batch_normalization:
            batch_norm_tensor = BatchNormalization(
                name=f"{name}_BatchNorm_{module + 1}"
            )(conv_tensor)  # Insert layer for batch normalization
            activation_tensor = activation_layer(
                batch_norm_tensor
            )  # Pass output tensor from BatchNormalization into the activation layer
        else:
            activation_tensor = activation_layer(
                conv_tensor
            )  # Pass output tensor from the convolution layer into the activation layer.

        tensor = activation_tensor

    return tensor


def aggregated_feature_map(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    level1: int,
    level2: int,
    upsample_size: tuple[int],
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str = None,
):
    """
    Connect two nodes in the U-Net 3+ with an aggregated feature map (AFM).

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    filters: int
        Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    level1: int
        Level of the first node that is connected to the AFM. This node will provide the input tensor to the AFM. Must be
        greater than level2 (i.e. the first node must be on a lower level in the U-Net 3+ since we are up-sampling), otherwise
        a ValueError exception is raised.
    level2: int
        Level of the second node that is connected to the AFM. This node will receive the output of the AFM. Must be smaller
        than level1 (i.e. the second node must be on a higher level in the U-Net 3+ since we are up-sampling), otherwise
        a ValueError exception is raised.
    upsample_size: tuple or list of ints
        Upsampling size for rows and columns in the UpSampling2D/UpSampling3D layer.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 <= level2:
        raise ValueError(
            "level2 must be smaller than level1 in aggregated feature maps"
        )

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs["num_modules"] = 1
    for arg in [
        "filters",
        "kernel_size",
        "padding",
        "use_bias",
        "batch_normalization",
        "activation",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
        "name",
    ]:
        module_kwargs[arg] = locals()[arg]

    # Keyword arguments for the UpSampling2D/UpSampling3D layers
    upsample_kwargs = dict({})
    upsample_kwargs["name"] = f"{name}_UpSampling{tensor_dims - 2}D"
    upsample_kwargs["size"] = np.power(upsample_size, abs(level1 - level2))

    if tensor_dims == 4:  # If the image is 2D
        upsample_layer = UpSampling2D
        if len(upsample_size) != 2:
            raise TypeError(
                f"For 2D up-sampling, the pool size must be a tuple or list with 2 integers. Received shape: {np.shape(upsample_size)}"
            )
    elif tensor_dims == 5:  # If the image is 3D
        upsample_layer = UpSampling3D
        if len(upsample_size) != 3:
            raise TypeError(
                f"For 3D up-sampling, the pool size must be a tuple or list with 3 integers. Received shape: {np.shape(upsample_size)}"
            )
    else:
        raise TypeError(
            f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions"
        )

    tensor = upsample_layer(**upsample_kwargs)(
        tensor
    )  # Pass the tensor through the UpSample2D/UpSample3D layer

    tensor = convolution_module(
        tensor, **module_kwargs
    )  # Pass input tensor through convolution module

    return tensor


def full_scale_skip_connection(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    level1: int,
    level2: int,
    pool_size: tuple[int],
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str = None,
):
    """
    Connect two nodes in the U-Net 3+ with a full-scale skip connection (FSC).

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    filters: int
        Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    level1: int
        Level of the first node that is connected to the FSC. This node will provide the input tensor to the FSC. Must be
        smaller than level2 (i.e. the first node must be on a higher level in the U-Net 3+ since we are max-pooling), otherwise
        a ValueError exception is raised.
    level2: int
        Level of the second node that is connected to the FSC. This node will receive the output of the FSC. Must be greater
        than level1 (i.e. the second node must be on a lower level in the U-Net 3+ since we are max-pooling), otherwise
        a ValueError exception is raised.
    pool_size: tuple or list
        Pool size for the MaxPooling2D/MaxPooling3D layer.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if level1 >= level2:
        raise ValueError(
            "level2 must be greater than level1 in full-scale skip connections"
        )

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs["num_modules"] = 1
    for arg in [
        "filters",
        "kernel_size",
        "padding",
        "use_bias",
        "batch_normalization",
        "activation",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
        "name",
    ]:
        module_kwargs[arg] = locals()[arg]

    # Keyword arguments for the MaxPooling2D/MaxPooling3D layer
    pool_kwargs = dict({})
    pool_kwargs["name"] = f"{name}_MaxPool{tensor_dims - 2}D"
    pool_kwargs["pool_size"] = np.power(pool_size, abs(level1 - level2))

    if tensor_dims == 4:  # If the image is 2D
        pool_layer = MaxPooling2D
    elif tensor_dims == 5:  # If the image is 3D
        pool_layer = MaxPooling3D
    else:
        raise TypeError(
            f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions"
        )

    tensor = pool_layer(**pool_kwargs)(
        tensor
    )  # Pass the tensor through the MaxPooling2D/MaxPooling3D layer

    tensor = convolution_module(
        tensor, **module_kwargs
    )  # Pass the tensor through the convolution module

    return tensor


def conventional_skip_connection(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str = None,
):
    """
    Connect two nodes in the U-Net 3+ with a conventional skip connection.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    filters: int
        Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs["num_modules"] = 1
    for arg in [
        "filters",
        "kernel_size",
        "padding",
        "use_bias",
        "batch_normalization",
        "activation",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
        "name",
    ]:
        module_kwargs[arg] = locals()[arg]

    tensor = convolution_module(
        tensor, **module_kwargs
    )  # Pass the tensor through the convolution module

    return tensor


def max_pool(tensor: tf.Tensor, pool_size: tuple[int], name: str = None):
    """
    Connect two encoder nodes with a max-pooling operation.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    pool_size: tuple or list
        Pool size for the MaxPooling2D/MaxPooling3D layer.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    if not isinstance(pool_size, tuple) and not isinstance(pool_size, list):
        raise TypeError(
            f"pool_size can only be a tuple or list. Received type: {type(pool_size)}"
        )

    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    pool_kwargs = dict({})  # Keyword arguments in the MaxPooling layer
    pool_kwargs["name"] = f"{name}_MaxPool{tensor_dims - 2}D"
    pool_kwargs["pool_size"] = pool_size

    if tensor_dims == 4:  # If the image is 2D
        pool_layer = MaxPooling2D
        if len(pool_size) != 2:
            raise TypeError(
                f"For 2D max-pooling, the pool size must be a tuple or list with 2 integers. Received shape: {np.shape(pool_size)}"
            )
    elif tensor_dims == 5:  # If the image is 3D
        pool_layer = MaxPooling3D
        if len(pool_size) != 3:
            raise TypeError(
                f"For 3D max-pooling, the pool size must be a tuple or list with 3 integers. Received shape: {np.shape(pool_size)}"
            )
    else:
        raise TypeError(
            f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions"
        )

    pool_tensor = pool_layer(**pool_kwargs)(
        tensor
    )  # Pass the tensor through the MaxPooling2D/MaxPooling3D layer

    return pool_tensor


def upsample(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: int | tuple[int],
    upsample_size: tuple[int],
    padding: str = "same",
    use_bias: bool = False,
    batch_normalization: bool = True,
    activation: str = "relu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    shared_axes=None,
    name: str = None,
):
    """
    Connect decoder nodes with an up-sampling operation.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    filters: int
        Number of filters in the Conv2D/Conv3D layer.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    upsample_size: tuple or list
        Upsampling size in the UpSampling2D/UpSampling3D layer.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    batch_normalization: bool
        If True, a BatchNormalization layer will follow the Conv2D/Conv3D layer.
    activation: str
        Activation function to use after the Conv2D/Conv3D layer (BatchNormalization layer, if batch_normalization is True).
        See choose_activation_layer for all available activation functions.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    shared_axes: tuple or list of ints
        Axes along which to share the learnable parameters for the activation function.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if not isinstance(upsample_size, tuple) and not isinstance(upsample_size, list):
        raise TypeError(
            f"upsample_size can only be a tuple or list. Received type: {type(upsample_size)}"
        )

    # Arguments for the convolution module.
    module_kwargs = dict({})
    module_kwargs["num_modules"] = 1
    for arg in [
        "filters",
        "kernel_size",
        "padding",
        "use_bias",
        "batch_normalization",
        "activation",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
        "name",
    ]:
        module_kwargs[arg] = locals()[arg]

    # Keyword arguments in the UpSampling layer
    upsample_kwargs = dict({})
    upsample_kwargs["name"] = f"{name}_UpSampling{tensor_dims - 2}D"
    upsample_kwargs["size"] = upsample_size

    if tensor_dims == 4:  # If the image is 2D
        upsample_layer = UpSampling2D
        if len(upsample_size) != 2:
            raise TypeError(
                f"For 2D up-sampling, the pool size must be a tuple or list with 2 integers. Received shape: {np.shape(upsample_size)}"
            )
    elif tensor_dims == 5:  # If the image is 3D
        upsample_layer = UpSampling3D
        if len(upsample_size) != 3:
            raise TypeError(
                f"For 3D up-sampling, the pool size must be a tuple or list with 3 integers. Received shape: {np.shape(upsample_size)}"
            )
    else:
        raise TypeError(
            f"Incompatible tensor shape: {tensor.shape}. The tensor must only have 4 or 5 dimensions"
        )

    upsample_tensor = upsample_layer(**upsample_kwargs)(
        tensor
    )  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    tensor = convolution_module(
        upsample_tensor, **module_kwargs
    )  # Pass the up-sampled tensor through a convolution module

    return tensor


def deep_supervision_side_output(
    tensor: tf.Tensor,
    num_classes: int,
    kernel_size: int | tuple[int],
    output_level: int,
    upsample_size: tuple[int],
    activation: str = "softmax",
    padding: str = "same",
    use_bias: bool = False,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    squeeze_axes: int | tuple[int] = None,
    name: str = None,
):
    """
    Deep supervision output. This is usually used on a decoder node in the U-Net 3+ or the final decoder node of a standard
    U-Net.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor for the convolution module.
    num_classes: int
        Number of classes that the model is trying to predict.
    kernel_size: int or tuple
        Size of the kernel in the Conv2D/Conv3D layer.
    output_level: int
        Level of the decoder node from which the deep supervision output is based.
    upsample_size: tuple or list
        Upsampling size for rows and columns in the UpSampling2D/UpSampling3D layer. Tuples are currently not supported
        but will be supported in a future update.
    activation: str
        Output activation function.
    padding: str
        Padding in the Conv2D/Conv3D layer. 'valid' will apply no padding, while 'same' will apply padding such that the
        output shape matches the input shape. 'valid' and 'same' are case-insensitive.
    use_bias: bool
        If True, a bias vector will be used in the Conv2D/Conv3D layer.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.
    squeeze_axes: int, tuple, or None
        Axis or axes of the input tensor to squeeze.
    name: str or None
        Prefix of the layer names. If left as None, the layer names are set automatically.

    Returns
    -------
    tensor: tf.Tensor
        Output tensor.
    """

    tensor_dims = len(
        tensor.shape
    )  # Number of dims in the tensor (including the first 'None' dimension for batch size)

    if tensor_dims == 4:  # If the image is 2D
        conv_layer = Conv2D
        upsample_layer = UpSampling2D
        if output_level > 1:
            upsample_size_1 = upsample_size
        else:
            upsample_size_1 = None

        if output_level > 2:
            upsample_size_2 = np.power(upsample_size, output_level - 2)
        else:
            upsample_size_2 = None

    elif tensor_dims == 5:  # If the image is 3D
        conv_layer = Conv3D
        upsample_layer = UpSampling3D
        if output_level > 1:
            upsample_size_1 = upsample_size
        else:
            upsample_size_1 = None

        if output_level > 2:
            upsample_size_2 = np.power(upsample_size, output_level - 2)
        else:
            upsample_size_2 = None

    else:
        raise TypeError(
            f"Incompatible tensor shape: {tensor.shape}. The tensor can only have 4 or 5 dimensions"
        )

    # Arguments for the Conv2D/Conv3D layer.
    conv_kwargs = dict({})
    conv_kwargs["name"] = f"{name}_Conv{tensor_dims - 2}D"
    for arg in [
        "use_bias",
        "kernel_size",
        "padding",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
    ]:
        conv_kwargs[arg] = locals()[arg]

    if upsample_size_1 is not None:
        tensor = upsample_layer(
            size=upsample_size_1, name=f"{name}_UpSampling{tensor_dims - 2}D_1"
        )(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    tensor = conv_layer(filters=num_classes, **conv_kwargs)(
        tensor
    )  # This convolution layer contains num_classes filters, one for each class

    if upsample_size_2 is not None:
        tensor = upsample_layer(
            size=upsample_size_2, name=f"{name}_UpSampling{tensor_dims - 2}D_2"
        )(tensor)  # Pass the tensor through the UpSampling2D/UpSampling3D layer

    ### Squeeze the given dimensions/axes ###
    if squeeze_axes is not None:
        conv_kwargs["kernel_size"] = [1 for _ in range(tensor_dims - 2)]

        if isinstance(squeeze_axes, int):
            squeeze_axes = [
                squeeze_axes,
            ]  # Turn integer into a list of length 1 to make indexing easier

        squeeze_axes_sizes = [
            tensor.shape[ax_to_squeeze] for ax_to_squeeze in squeeze_axes
        ]

        for ax, size in enumerate(squeeze_axes_sizes):
            conv_kwargs["kernel_size"][squeeze_axes[ax] - 1] = (
                size  # Kernel size of dimension to squeeze is equal to the size of the dimension because we want the final size to be 1 so it can be squeezed
            )

        conv_kwargs["padding"] = (
            "valid"  # Padding cannot be 'same' since we want to modify the size of the dimension to be squeezed
        )
        conv_kwargs["name"] = f"{name}_Conv{tensor_dims - 2}D_collapse"

        tensor = conv_layer(filters=num_classes, **conv_kwargs)(
            tensor
        )  # This convolution layer contains num_classes filters, one for each class
        tensor = tf.squeeze(
            tensor, axis=squeeze_axes
        )  # Squeeze the tensor and remove the dimension

    activation_kwargs = {"name": f"{name}_{activation}"}
    activation_config = keras_builders.ActivationConfig(
        name=activation, config=activation_kwargs
    )
    activation_layer = activation_config.build()
    sup_output = activation_layer(tensor)  # Final softmax output

    return sup_output


def unet(
    input_shape: tuple[int],
    num_classes: int,
    pool_size: int | tuple[int] | list[int],
    upsample_size: int | tuple[int] | list[int],
    levels: int,
    filter_num: tuple[int] | list[int],
    kernel_size: int = 3,
    squeeze_axes: int | tuple[int] | list[int] = None,
    shared_axes: int | tuple[int] | list[int] = None,
    modules_per_node: int = 2,
    batch_normalization: bool = True,
    activation: str = "relu",
    output_activation: str = "softmax",
    padding: str = "same",
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer: str = None,
    bias_regularizer: str = None,
    activity_regularizer: str = None,
    kernel_constraint: str = None,
    bias_constraint: str = None,
):
    """
    Builds a U-Net model.

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    squeeze_axes: int, tuple, list, or None
        Axis or axes of the input tensor to squeeze.
    shared_axes: int, tuple, list, or None
        Axes along which to share the learnable parameters for the activation function. When left as None, parameters will
            be shared along all arbitrary dimensions (i.e. all dimensions without a defined size).
    modules_per_node: int
        Number of modules in each node of the U-Net.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    activation: str
        Activation function to use in the modules.
        See utils.choose_activation_layer for all supported activation functions.
    output_activation: str
        Output activation function.
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net model.

    Raises
    ------
    ValueError
        If levels < 2
        If input_shape does not have 3 nor 4 dimensions
        If the length of filter_num does not match the number of levels

    References
    ----------
    https://arxiv.org/pdf/1505.04597.pdf
    """

    ndims = (
        len(input_shape) - 1
    )  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")
    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(
            f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}"
        )
    if len(filter_num) != levels:
        raise ValueError(
            f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})"
        )

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    module_kwargs["num_modules"] = modules_per_node
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
    ]:
        module_kwargs[arg] = locals()[arg]

    # MaxPooling keyword arguments
    pool_kwargs = {"pool_size": pool_size}

    # Keyword arguments for upsampling
    upsample_kwargs = dict({})
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "upsample_size",
        "shared_axes",
    ]:
        upsample_kwargs[arg] = locals()[arg]

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    for arg in [
        "padding",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "upsample_size",
        "squeeze_axes",
    ]:
        supervision_kwargs[arg] = locals()[arg]
    supervision_kwargs["activation"] = output_activation

    tensors = dict({})  # Tensors associated with each node and skip connections

    """ Setup the first encoder node with an input layer and a convolution module """
    tensors["input"] = Input(shape=input_shape, name="Input")
    tensors["En1"] = convolution_module(
        tensors["input"], filters=filter_num[0], name="En1", **module_kwargs
    )

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(
        2, levels + 1
    ):  # Iterate through the rest of the encoder nodes
        current_node, previous_node = f"En{encoder}", f"En{encoder - 1}"
        pool_tensor = max_pool(
            tensors[previous_node],
            name=f"{previous_node}-{current_node}",
            **pool_kwargs,
        )  # Connect the next encoder node with a MaxPooling layer
        tensors[current_node] = convolution_module(
            pool_tensor,
            filters=filter_num[encoder - 1],
            name=current_node,
            **module_kwargs,
        )  # Convolution modules

    # Connect the bottom encoder node to a decoder node
    upsample_tensor = upsample(
        tensors[f"En{levels}"],
        filters=filter_num[levels - 2],
        name=f"En{levels}-De{levels}",
        **upsample_kwargs,
    )

    """ Bottom decoder node """
    current_node, next_node = f"De{levels - 1}", f"De{levels - 2}"
    skip_node = f"En{levels - 1}"  # node with an incoming skip connection that connects to 'current_node'
    tensors[current_node] = Concatenate(name=f"{current_node}_Concatenate")(
        [tensors[skip_node], upsample_tensor]
    )  # Concatenate the upsampled tensor and skip connection
    tensors[current_node] = convolution_module(
        tensors[current_node],
        filters=filter_num[levels - 2],
        name=current_node,
        **module_kwargs,
    )  # Convolution module
    upsample_tensor = upsample(
        tensors[current_node],
        filters=filter_num[levels - 3],
        name=f"{current_node}-{next_node}",
        **upsample_kwargs,
    )  # Connect the bottom decoder node to the next decoder node

    """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
    for decoder in np.arange(2, levels - 1)[::-1]:
        current_node, next_node = f"De{decoder}", f"De{decoder - 1}"
        skip_node = f"En{decoder}"  # node with an incoming skip connection that connects to 'current_node'
        tensors[current_node] = Concatenate(name=f"{current_node}_Concatenate")(
            [tensors[skip_node], upsample_tensor]
        )  # Concatenate the upsampled tensor and skip connection
        tensors[current_node] = convolution_module(
            tensors[current_node],
            filters=filter_num[decoder - 1],
            name=current_node,
            **module_kwargs,
        )  # Convolution module
        upsample_tensor = upsample(
            tensors[current_node],
            filters=filter_num[decoder - 2],
            name=f"{current_node}-{next_node}",
            **upsample_kwargs,
        )  # Connect the bottom decoder node to the next decoder node

    """ Final decoder node begins with a concatenation and convolution module, followed by deep supervision """
    tensor_De1 = Concatenate(name="De1_Concatenate")(
        [tensors["En1"], upsample_tensor]
    )  # Concatenate the upsampled tensor and skip connection
    tensor_De1 = convolution_module(
        tensor_De1, filters=filter_num[0], name="De1", **module_kwargs
    )  # Convolution module
    tensors["output"] = deep_supervision_side_output(
        tensor_De1,
        num_classes=num_classes,
        kernel_size=1,
        output_level=1,
        use_bias=True,
        name="final",
        **supervision_kwargs,
    )  # Deep supervision - this layer will output the model's prediction

    model = Model(
        inputs=tensors["input"], outputs=tensors["output"], name=f"unet_{ndims}D"
    )

    return model


def unet_ensemble(
    input_shape: tuple[int] | list[int],
    num_classes: int,
    pool_size: int | tuple[int] | list[int],
    upsample_size: int | tuple[int] | list[int],
    levels: int,
    filter_num: tuple[int] | list[int],
    kernel_size: int = 3,
    squeeze_axes: int | tuple[int] | list[int] = None,
    shared_axes: int | tuple[int] | list[int] = None,
    modules_per_node: int = 2,
    batch_normalization: bool = True,
    activation: str = "relu",
    output_activation: str = "softmax",
    padding: str = "same",
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer: str = None,
    bias_regularizer: str = None,
    activity_regularizer: str = None,
    kernel_constraint: str = None,
    bias_constraint: str = None,
):
    """
    Builds a U-Net ensemble model.
    https://arxiv.org/pdf/1912.05074.pdf

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    squeeze_axes: int, tuple, list, or None
        Axis or axes of the input tensor to squeeze.
    shared_axes: int, tuple, list, or None
        Axes along which to share the learnable parameters for the activation function. When left as None, parameters will
            be shared along all arbitrary dimensions (i.e. all dimensions without a defined size).
    modules_per_node: int
        Number of modules in each node of the U-Net.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    activation: str
        Activation function to use in the modules.
        See utils.choose_activation_layer for all supported activation functions.
    output_activation: str
        Output activation function.
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net model.

    Raises
    ------
    ValueError
        If levels < 2
        If input_shape does not have 3 nor 4 dimensions
        If the length of filter_num does not match the number of levels
    """

    ndims = (
        len(input_shape) - 1
    )  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")
    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(
            f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}"
        )
    if len(filter_num) != levels:
        raise ValueError(
            f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})"
        )

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    module_kwargs["num_modules"] = modules_per_node
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
    ]:
        module_kwargs[arg] = locals()[arg]

    # MaxPooling keyword arguments
    pool_kwargs = {"pool_size": pool_size}

    # Keyword arguments for upsampling
    upsample_kwargs = dict({})
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "upsample_size",
        "shared_axes",
    ]:
        upsample_kwargs[arg] = locals()[arg]

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    for arg in [
        "padding",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "upsample_size",
        "squeeze_axes",
        "num_classes",
    ]:
        supervision_kwargs[arg] = locals()[arg]
    supervision_kwargs["activation"] = output_activation
    supervision_kwargs["use_bias"] = True
    supervision_kwargs["output_level"] = 1
    supervision_kwargs["kernel_size"] = 1

    tensors = dict({})  # Tensors associated with each node and skip connections
    tensors_with_supervision = []  # list of output tensors. If deep supervision is used, more than one output will be produced

    """ Setup the first encoder node with an input layer and a convolution module """
    tensors["input"] = Input(shape=input_shape, name="Input")
    tensors["En1"] = convolution_module(
        tensors["input"], filters=filter_num[0], name="En1", **module_kwargs
    )

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(
        2, levels + 1
    ):  # Iterate through the rest of the encoder nodes
        current_node, previous_node = f"En{encoder}", f"En{encoder - 1}"
        pool_tensor = max_pool(
            tensors[previous_node],
            name=f"{previous_node}-{current_node}",
            **pool_kwargs,
        )  # Connect the next encoder node with a MaxPooling layer
        tensors[current_node] = convolution_module(
            pool_tensor,
            filters=filter_num[encoder - 1],
            name=current_node,
            **module_kwargs,
        )  # Convolution modules

    # Connect the bottom encoder node to a decoder node
    upsample_tensor = upsample(
        tensors[f"En{levels}"],
        filters=filter_num[levels - 2],
        name=f"En{levels}-De{levels}",
        **upsample_kwargs,
    )

    """ Bottom decoder node """
    current_node, next_node = f"De{levels - 1}", f"De{levels - 2}"
    skip_node = f"En{levels - 1}"
    tensors[current_node] = Concatenate(name=f"{current_node}_Concatenate")(
        [upsample_tensor, tensors[skip_node]]
    )  # Concatenate the upsampled tensor and skip connection
    tensors[current_node] = convolution_module(
        tensors[current_node],
        filters=filter_num[levels - 2],
        name=current_node,
        **module_kwargs,
    )  # Convolution module
    upsample_tensor = upsample(
        tensors[current_node],
        filters=filter_num[levels - 3],
        name=f"{current_node}-{next_node}",
        **upsample_kwargs,
    )  # Connect the bottom decoder node to the next decoder node

    for decoder in np.arange(1, levels - 1)[::-1]:
        num_middle_nodes = levels - decoder - 1
        for node in range(1, num_middle_nodes + 1):
            if node == 1:  # if on the first middle node at the given level
                upsample_tensor_for_middle_node = upsample(
                    tensors[f"En{decoder + 1}"],
                    filters=filter_num[decoder - 2],
                    name=f"En{decoder + 1}-Me{decoder}-1",
                    **upsample_kwargs,
                )
            else:
                upsample_tensor_for_middle_node = upsample(
                    tensors[f"Me{decoder + 1}-{node - 1}"],
                    filters=filter_num[decoder - 2],
                    name=f"Me{decoder + 1}-{node - 1}-Me{decoder}-{node}",
                    **upsample_kwargs,
                )
            tensors[f"Me{decoder}-{node}"] = Concatenate(
                name=f"Me{decoder}-{node}_Concatenate"
            )([tensors[f"En{decoder}"], upsample_tensor_for_middle_node])
            tensors[f"Me{decoder}-{node}"] = convolution_module(
                tensors[f"Me{decoder}-{node}"],
                filters=filter_num[decoder - 1],
                name=f"Me{decoder}-{node}",
                **module_kwargs,
            )  # Convolution module
            if decoder == 1:
                tensors[f"sup{decoder}-{node}"] = deep_supervision_side_output(
                    tensors[f"Me{decoder}-{node}"],
                    name=f"sup{decoder}-{node}",
                    **supervision_kwargs,
                )  # deep supervision on middle node located on top level
                tensors_with_supervision.append(tensors[f"sup{decoder}-{node}"])
        tensors[f"De{decoder}"] = Concatenate(name=f"De{decoder}_Concatenate")(
            [tensors[f"En{decoder}"], upsample_tensor]
        )  # Concatenate the upsampled tensor and skip connection
        tensors[f"De{decoder}"] = convolution_module(
            tensors[f"De{decoder}"],
            filters=filter_num[decoder - 1],
            name=f"De{decoder}",
            **module_kwargs,
        )  # Convolution module

        if decoder != 1:  # if not currently on the final decoder node (De1)
            upsample_tensor = upsample(
                tensors[f"De{decoder}"],
                filters=filter_num[decoder - 2],
                name=f"De{decoder}-De{decoder - 1}",
                **upsample_kwargs,
            )  # Connect the bottom decoder node to the next decoder node
        else:
            tensors["output"] = deep_supervision_side_output(
                tensors["De1"], name="final", **supervision_kwargs
            )  # Deep supervision - this layer will output the model's prediction
            tensors_with_supervision.append(tensors["output"])

    model = Model(
        inputs=tensors["input"],
        outputs=tensors_with_supervision,
        name=f"unet_ensemble_{ndims}D",
    )

    return model


def unet_plus(
    input_shape: tuple[int] | list[int],
    num_classes: int,
    pool_size: int | tuple[int] | list[int],
    upsample_size: int | tuple[int] | list[int],
    levels: int,
    filter_num: tuple[int] | list[int],
    kernel_size: int = 3,
    squeeze_axes: int | tuple[int] | list[int] = None,
    shared_axes: int | tuple[int] | list[int] = None,
    modules_per_node: int = 2,
    batch_normalization: bool = True,
    deep_supervision: bool = True,
    activation: str = "relu",
    output_activation: str = "softmax",
    padding: str = "same",
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer: str = None,
    bias_regularizer: str = None,
    activity_regularizer: str = None,
    kernel_constraint: str = None,
    bias_constraint: str = None,
):
    """
    Builds a U-Net+ model.
    https://arxiv.org/pdf/1912.05074.pdf

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    squeeze_axes: int, tuple, list, or None
        Axis or axes of the input tensor to squeeze.
    shared_axes: int, tuple, list, or None
        Axes along which to share the learnable parameters for the activation function. When left as None, parameters will
            be shared along all arbitrary dimensions (i.e. all dimensions without a defined size).
    modules_per_node: int
        Number of modules in each node of the U-Net.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    deep_supervision: bool
        Add deep supervision side outputs to each top node.
        NOTE: The final decoder node requires deep supervision and is not affected if this parameter is False.
    activation: str
        Activation function to use in the modules.
        See utils.choose_activation_layer for all supported activation functions.
    output_activation: str
        Output activation function.
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net model.

    Raises
    ------
    ValueError
        If levels < 2
        If input_shape does not have 3 nor 4 dimensions
        If the length of filter_num does not match the number of levels
    """

    ndims = (
        len(input_shape) - 1
    )  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")
    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(
            f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}"
        )
    if len(filter_num) != levels:
        raise ValueError(
            f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})"
        )

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    module_kwargs["num_modules"] = modules_per_node
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
    ]:
        module_kwargs[arg] = locals()[arg]

    # MaxPooling keyword arguments
    pool_kwargs = {"pool_size": pool_size}

    # Keyword arguments for upsampling
    upsample_kwargs = dict({})
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "upsample_size",
        "shared_axes",
    ]:
        upsample_kwargs[arg] = locals()[arg]

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    for arg in [
        "padding",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "upsample_size",
        "squeeze_axes",
        "num_classes",
    ]:
        supervision_kwargs[arg] = locals()[arg]
    supervision_kwargs["activation"] = output_activation
    supervision_kwargs["use_bias"] = True
    supervision_kwargs["output_level"] = 1
    supervision_kwargs["kernel_size"] = 1

    tensors = dict({})  # Tensors associated with each node and skip connections
    tensors_with_supervision = []  # list of output tensors. If deep supervision is used, more than one output will be produced

    """ Setup the first encoder node with an input layer and a convolution module """
    tensors["input"] = Input(shape=input_shape, name="Input")
    tensors["En1"] = convolution_module(
        tensors["input"], filters=filter_num[0], name="En1", **module_kwargs
    )

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(
        2, levels + 1
    ):  # Iterate through the rest of the encoder nodes
        pool_tensor = max_pool(
            tensors[f"En{encoder - 1}"],
            name=f"En{encoder - 1}-En{encoder}",
            **pool_kwargs,
        )  # Connect the next encoder node with a MaxPooling layer
        tensors[f"En{encoder}"] = convolution_module(
            pool_tensor,
            filters=filter_num[encoder - 1],
            name=f"En{encoder}",
            **module_kwargs,
        )  # Convolution modules

    # Connect the bottom encoder node to a decoder node
    upsample_tensor = upsample(
        tensors[f"En{levels}"],
        filters=filter_num[levels - 2],
        name=f"En{levels}-De{levels}",
        **upsample_kwargs,
    )

    """ Bottom decoder node """
    tensors[f"De{levels - 1}"] = Concatenate(name=f"De{levels - 1}_Concatenate")(
        [upsample_tensor, tensors[f"En{levels - 1}"]]
    )  # Concatenate the upsampled tensor and skip connection
    tensors[f"De{levels - 1}"] = convolution_module(
        tensors[f"De{levels - 1}"],
        filters=filter_num[levels - 2],
        name=f"De{levels - 1}",
        **module_kwargs,
    )  # Convolution module
    upsample_tensor = upsample(
        tensors[f"De{levels - 1}"],
        filters=filter_num[levels - 3],
        name=f"De{levels - 1}-De{levels - 2}",
        **upsample_kwargs,
    )  # Connect the bottom decoder node to the next decoder node

    """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
    for decoder in np.arange(1, levels - 1)[::-1]:
        num_middle_nodes = levels - decoder - 1
        for node in range(1, num_middle_nodes + 1):
            if node == 1:  # if on the first middle node at the given level
                upsample_tensor_for_middle_node = upsample(
                    tensors[f"En{decoder + 1}"],
                    filters=filter_num[decoder - 2],
                    name=f"En{decoder + 1}-Me{decoder}-1",
                    **upsample_kwargs,
                )
                tensors[f"Me{decoder}-1"] = Concatenate(
                    name=f"Me{decoder}-1_Concatenate"
                )([tensors[f"En{decoder}"], upsample_tensor_for_middle_node])
            else:
                upsample_tensor_for_middle_node = upsample(
                    tensors[f"Me{decoder + 1}-{node - 1}"],
                    filters=filter_num[decoder - 2],
                    name=f"Me{decoder + 1}-{node - 1}-Me{decoder}-{node}",
                    **upsample_kwargs,
                )
                tensors[f"Me{decoder}-{node}"] = Concatenate(
                    name=f"Me{decoder}-{node}_Concatenate"
                )([tensors[f"Me{decoder}-{node - 1}"], upsample_tensor_for_middle_node])
            tensors[f"Me{decoder}-{node}"] = convolution_module(
                tensors[f"Me{decoder}-{node}"],
                filters=filter_num[decoder - 1],
                name=f"Me{decoder}-{node}",
                **module_kwargs,
            )  # Convolution module
            if decoder == 1 and deep_supervision:
                tensors[f"sup{decoder}-{node}"] = deep_supervision_side_output(
                    tensors[f"Me{decoder}-{node}"],
                    name=f"sup{decoder}-{node}",
                    **supervision_kwargs,
                )  # deep supervision on middle node located on top level
                tensors_with_supervision.append(tensors[f"sup{decoder}-{node}"])
        tensors[f"De{decoder}"] = Concatenate(name=f"De{decoder}_Concatenate")(
            [tensors[f"Me{decoder}-{num_middle_nodes}"], upsample_tensor]
        )  # Concatenate the upsampled tensor and skip connection
        tensors[f"De{decoder}"] = convolution_module(
            tensors[f"De{decoder}"],
            filters=filter_num[decoder - 1],
            name=f"De{decoder}",
            **module_kwargs,
        )  # Convolution module

        if decoder != 1:  # if not currently on the final decoder node (De1)
            upsample_tensor = upsample(
                tensors[f"De{decoder}"],
                filters=filter_num[decoder - 2],
                name=f"De{decoder}-De{decoder - 1}",
                **upsample_kwargs,
            )  # Connect the bottom decoder node to the next decoder node
        else:
            tensors["output"] = deep_supervision_side_output(
                tensors["De1"], **supervision_kwargs
            )  # Deep supervision - this layer will output the model's prediction
            tensors_with_supervision.append(tensors["output"])

    model = Model(
        inputs=tensors["input"],
        outputs=tensors_with_supervision,
        name=f"unet_plus_{ndims}D",
    )

    return model


def unet_2plus(
    input_shape: tuple[int] | list[int],
    num_classes: int,
    pool_size: int | tuple[int] | list[int],
    upsample_size: int | tuple[int] | list[int],
    levels: int,
    filter_num: tuple[int] | list[int],
    kernel_size: int = 3,
    squeeze_axes: int | tuple[int] | list[int] = None,
    shared_axes: int | tuple[int] | list[int] = None,
    modules_per_node: int = 2,
    batch_normalization: bool = True,
    deep_supervision: bool = True,
    activation: str = "relu",
    output_activation: str = "softmax",
    padding: str = "same",
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer: str = None,
    bias_regularizer: str = None,
    activity_regularizer: str = None,
    kernel_constraint: str = None,
    bias_constraint: str = None,
):
    """
    Builds a U-Net++ model.
    https://arxiv.org/pdf/1912.05074.pdf

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    squeeze_axes: int, tuple, list, or None
        Axis or axes of the input tensor to squeeze.
    shared_axes: int, tuple, list, or None
        Axes along which to share the learnable parameters for the activation function. When left as None, parameters will
            be shared along all arbitrary dimensions (i.e. all dimensions without a defined size).
    modules_per_node: int
        Number of modules in each node of the U-Net.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    deep_supervision: bool
        Add deep supervision side outputs to each top node.
        NOTE: The final decoder node requires deep supervision and is not affected if this parameter is False.
    activation: str
        Activation function to use in the modules.
        See utils.choose_activation_layer for all supported activation functions.
    output_activation: str
        Output activation function.
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net model.

    Raises
    ------
    ValueError
        If levels < 2
        If input_shape does not have 3 nor 4 dimensions
        If the length of filter_num does not match the number of levels
    """

    ndims = (
        len(input_shape) - 1
    )  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")
    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(
            f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}"
        )
    if len(filter_num) != levels:
        raise ValueError(
            f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})"
        )

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    module_kwargs["num_modules"] = modules_per_node
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
    ]:
        module_kwargs[arg] = locals()[arg]

    # MaxPooling keyword arguments
    pool_kwargs = {"pool_size": pool_size}

    # Keyword arguments for upsampling
    upsample_kwargs = dict({})
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "upsample_size",
        "shared_axes",
    ]:
        upsample_kwargs[arg] = locals()[arg]

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    for arg in [
        "padding",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "upsample_size",
        "squeeze_axes",
        "num_classes",
    ]:
        supervision_kwargs[arg] = locals()[arg]
    supervision_kwargs["activation"] = output_activation
    supervision_kwargs["use_bias"] = True
    supervision_kwargs["output_level"] = 1
    supervision_kwargs["kernel_size"] = 1

    tensors = dict({})  # Tensors associated with each node and skip connections
    tensors_with_supervision = []  # list of output tensors. If deep supervision is used, more than one output will be produced

    """ Setup the first encoder node with an input layer and a convolution module """
    tensors["input"] = Input(shape=input_shape, name="Input")
    tensors["En1"] = convolution_module(
        tensors["input"], filters=filter_num[0], name="En1", **module_kwargs
    )

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(
        2, levels + 1
    ):  # Iterate through the rest of the encoder nodes
        pool_tensor = max_pool(
            tensors[f"En{encoder - 1}"],
            name=f"En{encoder - 1}-En{encoder}",
            **pool_kwargs,
        )  # Connect the next encoder node with a MaxPooling layer
        tensors[f"En{encoder}"] = convolution_module(
            pool_tensor,
            filters=filter_num[encoder - 1],
            name=f"En{encoder}",
            **module_kwargs,
        )  # Convolution modules

    # Connect the bottom encoder node to a decoder node
    upsample_tensor = upsample(
        tensors[f"En{levels}"],
        filters=filter_num[levels - 2],
        name=f"En{levels}-De{levels}",
        **upsample_kwargs,
    )

    """ Bottom decoder node """
    tensors[f"De{levels - 1}"] = Concatenate(name=f"De{levels - 1}_Concatenate")(
        [upsample_tensor, tensors[f"En{levels - 1}"]]
    )  # Concatenate the upsampled tensor and skip connection
    tensors[f"De{levels - 1}"] = convolution_module(
        tensors[f"De{levels - 1}"],
        filters=filter_num[levels - 2],
        name=f"De{levels - 1}",
        **module_kwargs,
    )  # Convolution module
    upsample_tensor = upsample(
        tensors[f"De{levels - 1}"],
        filters=filter_num[levels - 3],
        name=f"De{levels - 1}-De{levels - 2}",
        **upsample_kwargs,
    )  # Connect the bottom decoder node to the next decoder node

    """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
    for decoder in np.arange(1, levels - 1)[::-1]:
        num_middle_nodes = levels - decoder - 1
        for node in range(1, num_middle_nodes + 1):
            if node == 1:  # if on the first middle node at the given level
                upsample_tensor_for_middle_node = upsample(
                    tensors[f"En{decoder + 1}"],
                    filters=filter_num[decoder - 2],
                    name=f"En{decoder + 1}-Me{decoder}-1",
                    **upsample_kwargs,
                )
                tensors[f"Me{decoder}-1"] = Concatenate(
                    name=f"Me{decoder}-1_Concatenate"
                )([tensors[f"En{decoder}"], upsample_tensor_for_middle_node])
            else:
                upsample_tensor_for_middle_node = upsample(
                    tensors[f"Me{decoder + 1}-{node - 1}"],
                    filters=filter_num[decoder - 2],
                    name=f"Me{decoder + 1}-{node - 1}-Me{decoder}-{node}",
                    **upsample_kwargs,
                )
                tensors_to_concatenate = []  # Tensors to concatenate in the middle node
                connections_to_add = sorted(
                    [tensor for tensor in tensors if f"Me{decoder}" in tensor]
                )[::-1]  # skip connections to add to the list of tensors to concatenate
                for connection in connections_to_add:
                    tensors_to_concatenate.append(tensors[connection])
                tensors_to_concatenate.append(tensors[f"En{decoder}"])
                tensors_to_concatenate.append(upsample_tensor_for_middle_node)
                tensors[f"Me{decoder}-{node}"] = Concatenate(
                    name=f"Me{decoder}-{node}_Concatenate"
                )(tensors_to_concatenate)
            tensors[f"Me{decoder}-{node}"] = convolution_module(
                tensors[f"Me{decoder}-{node}"],
                filters=filter_num[decoder - 1],
                name=f"Me{decoder}-{node}",
                **module_kwargs,
            )  # Convolution module

            if decoder == 1 and deep_supervision:
                tensors[f"sup{decoder}-{node}"] = deep_supervision_side_output(
                    tensors[f"Me{decoder}-{node}"],
                    name=f"sup{decoder}-{node}",
                    **supervision_kwargs,
                )  # deep supervision on middle node located on top level
                tensors_with_supervision.append(tensors[f"sup{decoder}-{node}"])

        tensors_to_concatenate = []  # tensors to concatenate in the decoder node
        connections_to_add = sorted(
            [tensor for tensor in tensors if f"Me{decoder}" in tensor]
        )[::-1]  # skip connections to add to the list of tensors to concatenate
        for connection in connections_to_add:
            tensors_to_concatenate.append(tensors[connection])
        tensors_to_concatenate.append(tensors[f"En{decoder}"])
        tensors_to_concatenate.append(upsample_tensor)
        tensors[f"De{decoder}"] = Concatenate(name=f"De{decoder}_Concatenate")(
            tensors_to_concatenate
        )  # Concatenate the upsampled tensor and skip connection
        tensors[f"De{decoder}"] = convolution_module(
            tensors[f"De{decoder}"],
            filters=filter_num[decoder - 1],
            name=f"De{decoder}",
            **module_kwargs,
        )  # Convolution module

        if decoder != 1:  # if not currently on the final decoder node (De1)
            upsample_tensor = upsample(
                tensors[f"De{decoder}"],
                filters=filter_num[decoder - 2],
                name=f"De{decoder}-De{decoder - 1}",
                **upsample_kwargs,
            )  # Connect the bottom decoder node to the next decoder node
        else:
            tensors["output"] = deep_supervision_side_output(
                tensors["De1"], name="final", **supervision_kwargs
            )  # Deep supervision - this layer will output the model's prediction
            tensors_with_supervision.append(tensors["output"])

    model = Model(
        inputs=tensors["input"],
        outputs=tensors_with_supervision,
        name=f"unet_2plus_{ndims}D",
    )

    return model


def unet_3plus(
    input_shape: tuple[int] | list[int],
    num_classes: int,
    pool_size: int | tuple[int] | list[int],
    upsample_size: int | tuple[int] | list[int],
    levels: int,
    filter_num: tuple[int] | list[int],
    filter_num_skip: int = None,
    filter_num_aggregate: tuple[int] | list[int] = None,
    kernel_size: int = 3,
    first_encoder_connections: bool = True,
    squeeze_axes: int | tuple[int] | list[int] = None,
    shared_axes: int | tuple[int] | list[int] = None,
    modules_per_node: int = 2,
    batch_normalization: bool = True,
    deep_supervision: bool = True,
    activation: str = "relu",
    output_activation: str = "softmax",
    padding: str = "same",
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer: str = None,
    bias_regularizer: str = None,
    activity_regularizer: str = None,
    kernel_constraint: str = None,
    bias_constraint: str = None,
):
    """
    Creates a U-Net 3+.
    https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net 3+ will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling layers.
    upsample_size: tuple or list
        Size of the mask in the UpSampling layers.
    levels: int
        Number of levels in the U-Net 3+. Must be greater than 2.
    filter_num: iterable of ints
        Number of convolution filters in each encoder of the U-Net 3+. The length must be equal to 'levels'.
    filter_num_skip: int or None
        Number of convolution filters in the conventional skip connections, full-scale skip connections, and aggregated feature maps.
        NOTE: When left as None, this will default to the first value in the 'filter_num' iterable.
    filter_num_aggregate: int or None
        Number of convolution filters in the decoder nodes after images are concatenated.
        When left as None, this will be equal to the product of filter_num_skip and the number of levels.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    first_encoder_connections: bool
        Setting this to True will create full-scale skip connections attached to the first encoder node.
    squeeze_axes: int, tuple, list, or None
        Axis or axes of the input tensor to squeeze.
    shared_axes: int, tuple, list, or None
        Axes along which to share the learnable parameters for the activation function. When left as None, parameters will
            be shared along all arbitrary dimensions (i.e. all dimensions without a defined size).
    modules_per_node: int
        Number of modules in each node of the U-Net 3+.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    deep_supervision: bool
        Add deep supervision side outputs to each decoder node.
        NOTE: The final decoder node requires deep supervision and is not affected if this parameter is False.
    activation: str
        Activation function to use in the modules.
        See utils.choose_activation_layer for all supported activation functions.
    output_activation: str
        Output activation function.
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net 3+ model.
    """

    ndims = (
        len(input_shape) - 1
    )  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 3:
        raise ValueError(f"levels must be greater than 2. Received value: {levels}")
    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(
            f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}"
        )
    if len(filter_num) != levels:
        raise ValueError(
            f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})"
        )

    if filter_num_skip is None:
        filter_num_skip = filter_num[0]

    if filter_num_aggregate is None:
        filter_num_aggregate = levels * filter_num_skip

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
    ]:
        module_kwargs[arg] = locals()[arg]
    module_kwargs["num_modules"] = modules_per_node

    pool_kwargs = {"pool_size": pool_size}

    upsample_kwargs = dict({})
    conventional_kwargs = dict({})
    full_scale_kwargs = dict({})
    aggregated_kwargs = dict({})
    for arg in [
        "activation",
        "batch_normalization",
        "kernel_size",
        "padding",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
    ]:
        upsample_kwargs[arg] = locals()[arg]
        conventional_kwargs[arg] = locals()[arg]
        full_scale_kwargs[arg] = locals()[arg]
        aggregated_kwargs[arg] = locals()[arg]

    conventional_kwargs["filters"] = filter_num_skip
    upsample_kwargs["filters"] = filter_num_skip
    upsample_kwargs["upsample_size"] = upsample_size
    full_scale_kwargs["filters"] = filter_num_skip
    full_scale_kwargs["pool_size"] = pool_size
    aggregated_kwargs["filters"] = filter_num_skip
    aggregated_kwargs["upsample_size"] = upsample_size

    supervision_kwargs = dict({})
    for arg in [
        "kernel_size",
        "padding",
        "squeeze_axes",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "upsample_size",
    ]:
        supervision_kwargs[arg] = locals()[arg]
    supervision_kwargs["activation"] = output_activation
    supervision_kwargs["use_bias"] = True

    tensors = dict({})  # Tensors associated with each node and skip connections
    tensors_with_supervision = []  # Outputs of deep supervision

    """ Setup the first encoder node with an input layer and a convolution module (we are not using skip connections here) """
    tensors["input"] = Input(shape=input_shape, name="Input")
    tensors["En1"] = convolution_module(
        tensors["input"], filters=filter_num[0], name="En1", **module_kwargs
    )

    if first_encoder_connections is True:
        for full_connection in range(2, levels):
            tensors[f"1---{full_connection}_full-scale"] = full_scale_skip_connection(
                tensors["En1"],
                level1=1,
                level2=full_connection,
                name=f"1---{full_connection}_full-scale",
                **full_scale_kwargs,
            )

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(
        2, levels
    ):  # Iterate through the rest of the encoder nodes
        pool_tensor = max_pool(
            tensors[f"En{encoder - 1}"],
            name=f"En{encoder - 1}-En{encoder}",
            **pool_kwargs,
        )  # Connect the next encoder node with a MaxPooling layer
        tensors[f"En{encoder}"] = convolution_module(
            pool_tensor,
            filters=filter_num[encoder - 1],
            name=f"En{encoder}",
            **module_kwargs,
        )  # Convolution modules
        tensors[f"{encoder}---{encoder}_skip"] = conventional_skip_connection(
            tensors[f"En{encoder}"],
            name=f"{encoder}---{encoder}_skip",
            **conventional_kwargs,
        )

        # Create full-scale skip connections
        for full_connection in range(encoder + 1, levels):
            tensors[f"{encoder}---{full_connection}_full-scale"] = (
                full_scale_skip_connection(
                    tensors[f"En{encoder}"],
                    level1=encoder,
                    level2=full_connection,
                    name=f"{encoder}---{full_connection}_full-scale",
                    **full_scale_kwargs,
                )
            )

    # Bottom encoder node
    tensors[f"En{levels}"] = max_pool(
        tensors[f"En{levels - 1}"], name=f"En{levels - 1}-En{levels}", **pool_kwargs
    )
    tensors[f"En{levels}"] = convolution_module(
        tensors[f"En{levels}"],
        filters=filter_num[levels - 1],
        name=f"En{levels}",
        **module_kwargs,
    )
    if deep_supervision:
        tensors[f"sup{levels}_output"] = deep_supervision_side_output(
            tensors[f"En{levels}"],
            num_classes=num_classes,
            output_level=levels,
            name=f"sup{levels}",
            **supervision_kwargs,
        )
        tensors_with_supervision.append(tensors[f"sup{levels}_output"])

    # Add aggregated feature maps using the bottom encoder node
    for feature_map in range(1, levels - 1):
        tensors[f"{levels}---{feature_map}_feature"] = aggregated_feature_map(
            tensors[f"En{levels}"],
            level1=levels,
            level2=feature_map,
            name=f"{levels}---{feature_map}_feature",
            **aggregated_kwargs,
        )

    """ Build the rest of the decoder nodes """
    for decoder in np.arange(1, levels)[::-1]:
        """ The lowest decoder node (levels - 1) is attached to the bottom encoder node via upsampling, so concatenation is slightly different """
        if decoder == levels - 1:
            tensors[f"De{decoder}"] = upsample(
                tensors[f"En{levels}"],
                name=f"En{levels}-De{decoder}",
                **upsample_kwargs,
            )

            # Tensors to concatenate in the Concatenate layer
            tensors_to_concatenate = [
                tensors[f"De{decoder}"],
            ]
            connections_to_add = sorted(
                [tensor for tensor in tensors if f"---{decoder}" in tensor]
            )[::-1]
            for connection in connections_to_add:
                tensors_to_concatenate.append(tensors[connection])
        else:
            tensors[f"De{decoder}"] = upsample(
                tensors[f"De{decoder + 1}"],
                name=f"De{decoder + 1}-De{decoder}",
                **upsample_kwargs,
            )

            # Tensors to concatenate in the Concatenate layer
            tensors_to_concatenate = sorted(
                [tensor for tensor in tensors if f"---{decoder}" in tensor]
            )[::-1]
            for index in range(len(tensors_to_concatenate)):
                tensors_to_concatenate[index] = tensors[tensors_to_concatenate[index]]
            tensors_to_concatenate.insert(levels - 1 - decoder, tensors[f"De{decoder}"])

        # Concatenate tensors, pass through convolution modules, then use deep supervision to create a side output
        tensors[f"De{decoder}"] = Concatenate(name=f"De{decoder}_Concatenate")(
            tensors_to_concatenate
        )
        tensors[f"De{decoder}"] = convolution_module(
            tensors[f"De{decoder}"],
            filters=filter_num_aggregate,
            name=f"De{decoder}",
            **module_kwargs,
        )
        if (
            deep_supervision or decoder == 1
        ):  # Decoder node 1 must always have deep supervision
            tensors[f"sup{decoder}_output"] = deep_supervision_side_output(
                tensors[f"De{decoder}"],
                num_classes=num_classes,
                output_level=decoder,
                name=f"sup{decoder}",
                **supervision_kwargs,
            )
            tensors_with_supervision.append(tensors[f"sup{decoder}_output"])

        """ Add aggregated feature maps """
        for feature_map in range(1, decoder - 1):
            tensors[f"{decoder}---{feature_map}_feature"] = aggregated_feature_map(
                tensors[f"De{decoder}"],
                level1=decoder,
                level2=feature_map,
                name=f"{decoder}---{feature_map}_feature",
                **aggregated_kwargs,
            )

    model = Model(
        inputs=tensors["input"],
        outputs=tensors_with_supervision[::-1],
        name=f"unet_3plus_{ndims}D",
    )

    return model


def attention_unet(
    input_shape: tuple[None | int, ...],
    num_classes: int,
    pool_size: int | tuple[int, ...] | list[int],
    levels: int,
    filter_num: tuple[int] | list[int],
    kernel_size: int = 3,
    squeeze_axes: int | tuple[int] | list[int] = None,
    shared_axes: int | tuple[int] | list[int] = None,
    modules_per_node: int = 2,
    batch_normalization: bool = True,
    activation: str = "relu",
    output_activation: str = "softmax",
    padding: str = "same",
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer: str = None,
    bias_regularizer: str = None,
    activity_regularizer: str = None,
    kernel_constraint: str = None,
    bias_constraint: str = None,
):
    """
    Builds a U-Net model.

    Parameters
    ----------
    input_shape: tuple
        Shape of the inputs. The last number in the tuple represents the number of channels/predictors.
    num_classes: int
        Number of classes/labels that the U-Net will try to predict.
    pool_size: tuple or list
        Size of the mask in the MaxPooling and UpSampling layers.
    levels: int
        Number of levels in the U-Net. Must be greater than 1.
    filter_num: iterable of ints
        Number of convolution filters on each level of the U-Net.
    kernel_size: int or tuple
        Size of the kernel in the convolution layers.
    squeeze_axes: int, tuple, list, or None
        Axis or axes of the input tensor to squeeze.
    shared_axes: int, tuple, list, or None
        Axes along which to share the learnable parameters for the activation function. When left as None, parameters will
            be shared along all arbitrary dimensions (i.e. all dimensions without a defined size).
    modules_per_node: int
        Number of modules in each node of the U-Net.
    batch_normalization: bool
        Setting this to True will add a batch normalization layer after every convolution in the modules.
    activation: str
        Activation function to use in the modules.
        See utils.choose_activation_layer for all supported activation functions.
    output_activation: str
        Output activation function.
    padding: str
        Padding to use in the convolution layers.
    use_bias: bool
        Setting this to True will implement a bias vector in the convolution layers used in the modules.
    kernel_initializer: str or tf.keras.initializers object
        Initializer for the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_initializer: str or tf.keras.initializers object
        Initializer for the bias vector in the Conv2D/Conv3D layers.
    kernel_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the kernel weights matrix in the Conv2D/Conv3D layers.
    bias_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the bias vector in the Conv2D/Conv3D layers.
    activity_regularizer: str or tf.keras.regularizers object
        Regularizer function applied to the output of the Conv2D/Conv3D layers.
    kernel_constraint: str or tf.keras.constraints object
        Constraint function applied to the kernel matrix of the Conv2D/Conv3D layers.
    bias_constraint: str or tf.keras.constrains object
        Constraint function applied to the bias vector in the Conv2D/Conv3D layers.

    Returns
    -------
    model: tf.keras.models.Model object
        U-Net model.

    Raises
    ------
    ValueError
        If levels < 2
        If input_shape does not have 3 nor 4 dimensions
        If the length of filter_num does not match the number of levels

    References
    ----------
    https://arxiv.org/pdf/1804.03999.pdf
    """

    ndims = (
        len(input_shape) - 1
    )  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

    if levels < 2:
        raise ValueError(f"levels must be greater than 1. Received value: {levels}")

    if len(input_shape) > 4 or len(input_shape) < 3:
        raise ValueError(
            f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(input_shape)}"
        )

    if len(filter_num) != levels:
        raise ValueError(
            f"length of filter_num ({len(filter_num)}) does not match the number of levels ({levels})"
        )

    # Keyword arguments for the convolution modules
    module_kwargs = dict({})
    module_kwargs["num_modules"] = modules_per_node
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
    ]:
        module_kwargs[arg] = locals()[arg]

    # MaxPooling keyword arguments
    pool_kwargs = {"pool_size": pool_size}

    # Keyword arguments for upsampling
    upsample_kwargs = dict({})
    for arg in [
        "activation",
        "batch_normalization",
        "padding",
        "kernel_size",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "shared_axes",
    ]:
        upsample_kwargs[arg] = locals()[arg]
    upsample_kwargs["upsample_size"] = pool_size

    # Keyword arguments for the deep supervision output in the final decoder node
    supervision_kwargs = dict({})
    for arg in [
        "padding",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "squeeze_axes",
        "num_classes",
    ]:
        supervision_kwargs[arg] = locals()[arg]
    supervision_kwargs["activation"] = output_activation
    supervision_kwargs["upsample_size"] = pool_size
    supervision_kwargs["use_bias"] = True
    supervision_kwargs["output_level"] = 1
    supervision_kwargs["kernel_size"] = 1

    tensors = dict({})  # Tensors associated with each node and skip connections

    """ Setup the first encoder node with an input layer and a convolution module """
    tensors["input"] = Input(shape=input_shape, name="Input")
    tensors["En1"] = convolution_module(
        tensors["input"], filters=filter_num[0], name="En1", **module_kwargs
    )

    """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
    for encoder in np.arange(
        2, levels + 1
    ):  # Iterate through the rest of the encoder nodes
        pool_tensor = max_pool(
            tensors[f"En{encoder - 1}"],
            name=f"En{encoder - 1}-En{encoder}",
            **pool_kwargs,
        )  # Connect the next encoder node with a MaxPooling layer
        tensors[f"En{encoder}"] = convolution_module(
            pool_tensor,
            filters=filter_num[encoder - 1],
            name=f"En{encoder}",
            **module_kwargs,
        )  # Convolution modules

    tensors[f"AG{levels - 1}"] = attention_gate(
        tensors[f"En{levels - 1}"],
        tensors[f"En{levels}"],
        kernel_size,
        pool_size,
        name=f"AG{levels - 1}",
    )
    upsample_tensor = upsample(
        tensors[f"En{levels}"],
        filters=filter_num[levels - 2],
        name=f"En{levels}-De{levels - 1}",
        **upsample_kwargs,
    )  # Connect the bottom encoder node to a decoder node

    """ Bottom decoder node """
    tensors[f"De{levels - 1}"] = Concatenate(name=f"De{levels - 1}_Concatenate")(
        [tensors[f"AG{levels - 1}"], upsample_tensor]
    )  # Concatenate the upsampled tensor and skip connection
    tensors[f"De{levels - 1}"] = convolution_module(
        tensors[f"De{levels - 1}"],
        filters=filter_num[levels - 2],
        name=f"De{levels - 1}",
        **module_kwargs,
    )  # Convolution module
    tensors[f"AG{levels - 2}"] = attention_gate(
        tensors[f"En{levels - 2}"],
        tensors[f"De{levels - 1}"],
        kernel_size,
        pool_size,
        name=f"AG{levels - 2}",
    )
    upsample_tensor = upsample(
        tensors[f"De{levels - 1}"],
        filters=filter_num[levels - 3],
        name=f"De{levels - 1}-De{levels - 2}",
        **upsample_kwargs,
    )  # Connect the bottom decoder node to the next decoder node

    """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
    for decoder in np.arange(2, levels - 1)[::-1]:
        tensors[f"De{decoder}"] = Concatenate(name=f"De{decoder}_Concatenate")(
            [tensors[f"AG{decoder}"], upsample_tensor]
        )  # Concatenate the upsampled tensor and skip connection
        tensors[f"De{decoder}"] = convolution_module(
            tensors[f"De{decoder}"],
            filters=filter_num[decoder - 1],
            name=f"De{decoder}",
            **module_kwargs,
        )  # Convolution module
        tensors[f"AG{decoder - 1}"] = attention_gate(
            tensors[f"En{decoder - 1}"],
            tensors[f"De{decoder}"],
            kernel_size,
            pool_size,
            name=f"AG{decoder - 1}",
        )
        upsample_tensor = upsample(
            tensors[f"De{decoder}"],
            filters=filter_num[decoder - 2],
            name=f"De{decoder}-De{decoder - 1}",
            **upsample_kwargs,
        )  # Connect the bottom decoder node to the next decoder node

    """ Final decoder node begins with a concatenation and convolution module, followed by deep supervision """
    tensor_De1 = Concatenate(name="De1_Concatenate")(
        [tensors["AG1"], upsample_tensor]
    )  # Concatenate the upsampled tensor and skip connection
    tensor_De1 = convolution_module(
        tensor_De1, filters=filter_num[0], name="De1", **module_kwargs
    )  # Convolution module
    tensors["output"] = deep_supervision_side_output(
        tensor_De1, name="final", **supervision_kwargs
    )  # Deep supervision - this layer will output the model's prediction

    model = Model(
        inputs=tensors["input"],
        outputs=tensors["output"],
        name=f"attention_unet_{ndims}D",
    )

    return model


@dataclasses.dataclass
class UNet(model.ModelConfig):
    pass


@dataclasses.dataclass
class UNetEnsemble:
    num_models: int


@dataclasses.dataclass
class UNetPlus:
    deep_supervision: bool


@dataclasses.dataclass
class UNet2Plus:
    deep_supervision: bool


@dataclasses.dataclass
class UNet3Plus:
    deep_supervision: bool
    num_aggregate_filters: int
    full_scale_skip_connection_filters: int
    first_encoder_connections: bool


@dataclasses.dataclass
class AttentionUNet:
    def __post_init__(self):
        if len(self.upsample_size) > 0:
            raise ValueError(
                "AttentionUNet does not support upsample_size, use empty tuple."
            )


@dataclasses.dataclass
class UNetRegistry(keras_builders.BaseConfig):
    """Registry class for UNet models.

    Attributes:
        name: the string name of the UNet model to build. Must be one of "unet",
        "unet_ensemble", "unet_plus", "unet_2plus", "unet_3plus", or "attention_unet".
        config: a dictionary of keyword arguments to pass to the UNet.
        registry: a dictionary mapping string names to UNet functions.
    """

    name: Literal[
        "unet",
        "unet_ensemble",
        "unet_plus",
        "unet_2plus",
        "unet_3plus",
        "attention_unet",
    ]

    @property
    def registry(self) -> dict[str, type]:
        return {
            "unet": UNet,
            "unet_ensemble": UNetEnsemble,
            "unet_plus": UNetPlus,
            "unet_2plus": UNet2Plus,
            "unet_3plus": UNet3Plus,
            "attention_unet": AttentionUNet,
        }
