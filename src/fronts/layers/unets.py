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
    Concatenate,
    Input,
)
from fronts.utils import keras_builders
from fronts.layers import (
    convolution_module,
    max_pool,
    upsample,
    deep_supervision_side_output,
    full_scale_skip_connection,
    conventional_skip_connection,
    attention_gate,
    aggregated_feature_map,
)
import dataclasses
from typing import Literal


@dataclasses.dataclass
class UNet:
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

    input_shape: tuple[int]
    num_classes: int
    pool_size: int | tuple[int] | list[int]
    upsample_size: int | tuple[int] | list[int]
    levels: int
    filter_num: tuple[int] | list[int]
    kernel_size: int = 3
    squeeze_axes: int | tuple[int] | list[int] = None
    shared_axes: int | tuple[int] | list[int] = None
    modules_per_node: int = 2
    batch_normalization: bool = True
    activation: str = "relu"
    output_activation: str = "softmax"
    padding: str = "same"
    use_bias: bool = True
    kernel_initializer: str = "glorot_uniform"
    bias_initializer: str = "zeros"
    kernel_regularizer: str = None
    bias_regularizer: str = None
    activity_regularizer: str = None
    kernel_constraint: str = None
    bias_constraint: str = None

    def __post_init__(self):
        # Manage exceptions
        if self.levels not in [3, 4]:
            raise ValueError(
                "Input_shape can only have 3 or 4 dimensions (2D image + 1 dimension "
                "for channels OR a 3D image + 1 dimension for channels). Received "
                f"shape: {np.shape(self.input_shape)}"
            )
        if len(self.filter_num) != self.levels:
            raise ValueError(
                f"Length of filter_num ({len(self.filter_num)}) does not match the "
                f"number of levels ({self.levels})"
            )
        self.ndims = (
            len(self.input_shape) - 1
        )  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

        # Keyword arguments for the convolution modules
        self.module_kwargs = dict({})
        self.module_kwargs["num_modules"] = self.modules_per_node
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
            self.module_kwargs[arg] = getattr(self, arg)

        # MaxPooling keyword arguments
        self.pool_kwargs = {"pool_size": self.pool_size}

        # Keyword arguments for upsampling
        self.upsample_kwargs = dict({})
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
            self.upsample_kwargs[arg] = getattr(self, arg)

        # Keyword arguments for the deep supervision output in the final decoder node
        self.supervision_kwargs = dict({})
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
            self.supervision_kwargs[arg] = getattr(self, arg)

    def build(self):
        self.supervision_kwargs["activation"] = self.output_activation

        tensors = dict({})  # Tensors associated with each node and skip connections

        """ Setup the first encoder node with an input layer and a convolution module """
        tensors["input"] = Input(shape=self.input_shape, name="Input")
        tensors["En1"] = convolution_module(
            tensors["input"],
            filters=self.filter_num[0],
            name="En1",
            **self.module_kwargs,
        )

        """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
        for encoder in np.arange(
            2, self.levels + 1
        ):  # Iterate through the rest of the encoder nodes
            current_node, previous_node = f"En{encoder}", f"En{encoder - 1}"
            pool_tensor = max_pool(
                tensors[previous_node],
                name=f"{previous_node}-{current_node}",
                **self.pool_kwargs,
            )  # Connect the next encoder node with a MaxPooling layer
            tensors[current_node] = convolution_module(
                pool_tensor,
                filters=self.filter_num[encoder - 1],
                name=current_node,
                **self.module_kwargs,
            )  # Convolution modules

        # Connect the bottom encoder node to a decoder node
        upsample_tensor = upsample(
            tensors[f"En{self.levels}"],
            filters=self.filter_num[self.levels - 2],
            name=f"En{self.levels}-De{self.levels}",
            **self.upsample_kwargs,
        )

        """ Bottom decoder node """
        current_node, next_node = f"De{self.levels - 1}", f"De{self.levels - 2}"
        skip_node = f"En{self.levels - 1}"  # node with an incoming skip connection that connects to 'current_node'
        tensors[current_node] = Concatenate(name=f"{current_node}_Concatenate")(
            [tensors[skip_node], upsample_tensor]
        )  # Concatenate the upsampled tensor and skip connection
        tensors[current_node] = convolution_module(
            tensors[current_node],
            filters=self.filter_num[self.levels - 2],
            name=current_node,
            **self.module_kwargs,
        )  # Convolution module
        upsample_tensor = upsample(
            tensors[current_node],
            filters=self.filter_num[self.levels - 3],
            name=f"{current_node}-{next_node}",
            **self.upsample_kwargs,
        )  # Connect the bottom decoder node to the next decoder node

        """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
        for decoder in np.arange(2, self.levels - 1)[::-1]:
            current_node, next_node = f"De{decoder}", f"De{decoder - 1}"
            skip_node = f"En{decoder}"  # node with an incoming skip connection that connects to 'current_node'
            tensors[current_node] = Concatenate(name=f"{current_node}_Concatenate")(
                [tensors[skip_node], upsample_tensor]
            )  # Concatenate the upsampled tensor and skip connection
            tensors[current_node] = convolution_module(
                tensors[current_node],
                filters=self.filter_num[decoder - 1],
                name=current_node,
                **self.module_kwargs,
            )  # Convolution module
            upsample_tensor = upsample(
                tensors[current_node],
                filters=self.filter_num[decoder - 2],
                name=f"{current_node}-{next_node}",
                **self.upsample_kwargs,
            )  # Connect the bottom decoder node to the next decoder node

        """ Final decoder node begins with a concatenation and convolution module, followed by deep supervision """
        tensor_De1 = Concatenate(name="De1_Concatenate")(
            [tensors["En1"], upsample_tensor]
        )  # Concatenate the upsampled tensor and skip connection
        tensor_De1 = convolution_module(
            tensor_De1, filters=self.filter_num[0], name="De1", **self.module_kwargs
        )  # Convolution module
        tensors["output"] = deep_supervision_side_output(
            tensor_De1,
            num_classes=self.num_classes,
            kernel_size=1,
            output_level=1,
            use_bias=True,
            name="final",
            **self.supervision_kwargs,
        )  # Deep supervision - this layer will output the model's prediction

        output_model = Model(
            inputs=tensors["input"],
            outputs=tensors["output"],
            name=f"unet_{self.ndims}D",
        )

        return output_model


@dataclasses.dataclass
class UNetEnsemble(UNet):
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

    def build(self):
        self.supervision_kwargs["activation"] = self.output_activation
        self.supervision_kwargs["use_bias"] = True
        self.supervision_kwargs["output_level"] = 1
        self.supervision_kwargs["kernel_size"] = 1

        tensors = dict({})  # Tensors associated with each node and skip connections
        tensors_with_supervision = []  # list of output tensors. If deep supervision is used, more than one output will be produced

        """ Setup the first encoder node with an input layer and a convolution module """
        tensors["input"] = Input(shape=self.input_shape, name="Input")
        tensors["En1"] = convolution_module(
            tensors["input"],
            filters=self.filter_num[0],
            name="En1",
            **self.module_kwargs,
        )

        """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
        for encoder in np.arange(
            2, self.levels + 1
        ):  # Iterate through the rest of the encoder nodes
            current_node, previous_node = f"En{encoder}", f"En{encoder - 1}"
            pool_tensor = max_pool(
                tensors[previous_node],
                name=f"{previous_node}-{current_node}",
                **self.pool_kwargs,
            )  # Connect the next encoder node with a MaxPooling layer
            tensors[current_node] = convolution_module(
                pool_tensor,
                filters=self.filter_num[encoder - 1],
                name=current_node,
                **self.module_kwargs,
            )  # Convolution modules

        # Connect the bottom encoder node to a decoder node
        upsample_tensor = upsample(
            tensors[f"En{self.levels}"],
            filters=self.filter_num[self.levels - 2],
            name=f"En{self.levels}-De{self.levels}",
            **self.upsample_kwargs,
        )

        """ Bottom decoder node """
        current_node, next_node = f"De{self.levels - 1}", f"De{self.levels - 2}"
        skip_node = f"En{self.levels - 1}"
        tensors[current_node] = Concatenate(name=f"{current_node}_Concatenate")(
            [upsample_tensor, tensors[skip_node]]
        )  # Concatenate the upsampled tensor and skip connection
        tensors[current_node] = convolution_module(
            tensors[current_node],
            filters=self.filter_num[self.levels - 2],
            name=current_node,
            **self.module_kwargs,
        )  # Convolution module
        upsample_tensor = upsample(
            tensors[current_node],
            filters=self.filter_num[self.levels - 3],
            name=f"{current_node}-{next_node}",
            **self.upsample_kwargs,
        )  # Connect the bottom decoder node to the next decoder node

        for decoder in np.arange(1, self.levels - 1)[::-1]:
            num_middle_nodes = self.levels - decoder - 1
            for node in range(1, num_middle_nodes + 1):
                if node == 1:  # if on the first middle node at the given level
                    upsample_tensor_for_middle_node = upsample(
                        tensors[f"En{decoder + 1}"],
                        filters=self.filter_num[decoder - 2],
                        name=f"En{decoder + 1}-Me{decoder}-1",
                        **self.upsample_kwargs,
                    )
                else:
                    upsample_tensor_for_middle_node = upsample(
                        tensors[f"Me{decoder + 1}-{node - 1}"],
                        filters=self.filter_num[decoder - 2],
                        name=f"Me{decoder + 1}-{node - 1}-Me{decoder}-{node}",
                        **self.upsample_kwargs,
                    )
                tensors[f"Me{decoder}-{node}"] = Concatenate(
                    name=f"Me{decoder}-{node}_Concatenate"
                )([tensors[f"En{decoder}"], upsample_tensor_for_middle_node])
                tensors[f"Me{decoder}-{node}"] = convolution_module(
                    tensors[f"Me{decoder}-{node}"],
                    filters=self.filter_num[decoder - 1],
                    name=f"Me{decoder}-{node}",
                    **self.module_kwargs,
                )  # Convolution module
                if decoder == 1:
                    tensors[f"sup{decoder}-{node}"] = deep_supervision_side_output(
                        tensors[f"Me{decoder}-{node}"],
                        name=f"sup{decoder}-{node}",
                        **self.supervision_kwargs,
                    )  # deep supervision on middle node located on top level
                    tensors_with_supervision.append(tensors[f"sup{decoder}-{node}"])
            tensors[f"De{decoder}"] = Concatenate(name=f"De{decoder}_Concatenate")(
                [tensors[f"En{decoder}"], upsample_tensor]
            )  # Concatenate the upsampled tensor and skip connection
            tensors[f"De{decoder}"] = convolution_module(
                tensors[f"De{decoder}"],
                filters=self.filter_num[decoder - 1],
                name=f"De{decoder}",
                **self.module_kwargs,
            )  # Convolution module

            if decoder != 1:  # if not currently on the final decoder node (De1)
                upsample_tensor = upsample(
                    tensors[f"De{decoder}"],
                    filters=self.filter_num[decoder - 2],
                    name=f"De{decoder}-De{decoder - 1}",
                    **self.upsample_kwargs,
                )  # Connect the bottom decoder node to the next decoder node
            else:
                tensors["output"] = deep_supervision_side_output(
                    tensors["De1"], name="final", **self.supervision_kwargs
                )  # Deep supervision - this layer will output the model's prediction
                tensors_with_supervision.append(tensors["output"])

        output_model = Model(
            inputs=tensors["input"],
            outputs=tensors_with_supervision,
            name=f"unet_ensemble_{self.ndims}D",
        )

        return output_model


class UNetPlus(UNet):
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

    def build(self):
        self.supervision_kwargs["activation"] = self.output_activation
        self.supervision_kwargs["use_bias"] = True
        self.supervision_kwargs["output_level"] = 1
        self.supervision_kwargs["kernel_size"] = 1

        tensors = dict({})  # Tensors associated with each node and skip connections
        tensors_with_supervision = []  # list of output tensors. If deep supervision is used, more than one output will be produced

        """ Setup the first encoder node with an input layer and a convolution module """
        tensors["input"] = Input(shape=self.input_shape, name="Input")
        tensors["En1"] = convolution_module(
            tensors["input"],
            filters=self.filter_num[0],
            name="En1",
            **self.module_kwargs,
        )

        """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
        for encoder in np.arange(
            2, self.levels + 1
        ):  # Iterate through the rest of the encoder nodes
            pool_tensor = max_pool(
                tensors[f"En{encoder - 1}"],
                name=f"En{encoder - 1}-En{encoder}",
                **self.pool_kwargs,
            )  # Connect the next encoder node with a MaxPooling layer
            tensors[f"En{encoder}"] = convolution_module(
                pool_tensor,
                filters=self.filter_num[encoder - 1],
                name=f"En{encoder}",
                **self.module_kwargs,
            )  # Convolution modules

        # Connect the bottom encoder node to a decoder node
        upsample_tensor = upsample(
            tensors[f"En{self.levels}"],
            filters=self.filter_num[self.levels - 2],
            name=f"En{self.levels}-De{self.levels}",
            **self.upsample_kwargs,
        )

        """ Bottom decoder node """
        tensors[f"De{self.levels - 1}"] = Concatenate(
            name=f"De{self.levels - 1}_Concatenate"
        )(
            [upsample_tensor, tensors[f"En{self.levels - 1}"]]
        )  # Concatenate the upsampled tensor and skip connection
        tensors[f"De{self.levels - 1}"] = convolution_module(
            tensors[f"De{self.levels - 1}"],
            filters=self.filter_num[self.levels - 2],
            name=f"De{self.levels - 1}",
            **self.module_kwargs,
        )  # Convolution module
        upsample_tensor = upsample(
            tensors[f"De{self.levels - 1}"],
            filters=self.filter_num[self.levels - 3],
            name=f"De{self.levels - 1}-De{self.levels - 2}",
            **self.upsample_kwargs,
        )  # Connect the bottom decoder node to the next decoder node

        """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
        for decoder in np.arange(1, self.levels - 1)[::-1]:
            num_middle_nodes = self.levels - decoder - 1
            for node in range(1, num_middle_nodes + 1):
                if node == 1:  # if on the first middle node at the given level
                    upsample_tensor_for_middle_node = upsample(
                        tensors[f"En{decoder + 1}"],
                        filters=self.filter_num[decoder - 2],
                        name=f"En{decoder + 1}-Me{decoder}-1",
                        **self.upsample_kwargs,
                    )
                    tensors[f"Me{decoder}-1"] = Concatenate(
                        name=f"Me{decoder}-1_Concatenate"
                    )([tensors[f"En{decoder}"], upsample_tensor_for_middle_node])
                else:
                    upsample_tensor_for_middle_node = upsample(
                        tensors[f"Me{decoder + 1}-{node - 1}"],
                        filters=self.filter_num[decoder - 2],
                        name=f"Me{decoder + 1}-{node - 1}-Me{decoder}-{node}",
                        **self.upsample_kwargs,
                    )
                    tensors[f"Me{decoder}-{node}"] = Concatenate(
                        name=f"Me{decoder}-{node}_Concatenate"
                    )(
                        [
                            tensors[f"Me{decoder}-{node - 1}"],
                            upsample_tensor_for_middle_node,
                        ]
                    )
                tensors[f"Me{decoder}-{node}"] = convolution_module(
                    tensors[f"Me{decoder}-{node}"],
                    filters=self.filter_num[decoder - 1],
                    name=f"Me{decoder}-{node}",
                    **self.module_kwargs,
                )  # Convolution module
                if decoder == 1 and self.deep_supervision:
                    tensors[f"sup{decoder}-{node}"] = deep_supervision_side_output(
                        tensors[f"Me{decoder}-{node}"],
                        name=f"sup{decoder}-{node}",
                        **self.supervision_kwargs,
                    )  # deep supervision on middle node located on top level
                    tensors_with_supervision.append(tensors[f"sup{decoder}-{node}"])
            tensors[f"De{decoder}"] = Concatenate(name=f"De{decoder}_Concatenate")(
                [tensors[f"Me{decoder}-{num_middle_nodes}"], upsample_tensor]
            )  # Concatenate the upsampled tensor and skip connection
            tensors[f"De{decoder}"] = convolution_module(
                tensors[f"De{decoder}"],
                filters=self.filter_num[decoder - 1],
                name=f"De{decoder}",
                **self.module_kwargs,
            )  # Convolution module

            if decoder != 1:  # if not currently on the final decoder node (De1)
                upsample_tensor = upsample(
                    tensors[f"De{decoder}"],
                    filters=self.filter_num[decoder - 2],
                    name=f"De{decoder}-De{decoder - 1}",
                    **self.upsample_kwargs,
                )  # Connect the bottom decoder node to the next decoder node
            else:
                tensors["output"] = deep_supervision_side_output(
                    tensors["De1"], **self.supervision_kwargs
                )  # Deep supervision - this layer will output the model's prediction
                tensors_with_supervision.append(tensors["output"])

        output_model = Model(
            inputs=tensors["input"],
            outputs=tensors_with_supervision,
            name=f"unet_plus_{self.ndims}D",
        )

        return output_model


class UNet2Plus(UNet):
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

    def build(self):
        self.supervision_kwargs["activation"] = self.output_activation
        self.supervision_kwargs["use_bias"] = True
        self.supervision_kwargs["output_level"] = 1
        self.supervision_kwargs["kernel_size"] = 1

        tensors = dict({})  # Tensors associated with each node and skip connections
        tensors_with_supervision = []  # list of output tensors. If deep supervision is used, more than one output will be produced

        """ Setup the first encoder node with an input layer and a convolution module """
        tensors["input"] = Input(shape=self.input_shape, name="Input")
        tensors["En1"] = convolution_module(
            tensors["input"],
            filters=self.lfilter_num[0],
            name="En1",
            **self.lmodule_kwargs,
        )

        """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
        for encoder in np.arange(
            2, self.levels + 1
        ):  # Iterate through the rest of the encoder nodes
            pool_tensor = max_pool(
                tensors[f"En{encoder - 1}"],
                name=f"En{encoder - 1}-En{encoder}",
                **self.lpool_kwargs,
            )  # Connect the next encoder node with a MaxPooling layer
            tensors[f"En{encoder}"] = convolution_module(
                pool_tensor,
                filters=self.lfilter_num[encoder - 1],
                name=f"En{encoder}",
                **self.lmodule_kwargs,
            )  # Convolution modules

        # Connect the bottom encoder node to a decoder node
        upsample_tensor = upsample(
            tensors[f"En{self.levels}"],
            filters=self.lfilter_num[self.levels - 2],
            name=f"En{self.levels}-De{self.levels}",
            **self.lupsample_kwargs,
        )

        """ Bottom decoder node """
        tensors[f"De{self.levels - 1}"] = Concatenate(
            name=f"De{self.levels - 1}_Concatenate"
        )(
            [upsample_tensor, tensors[f"En{self.levels - 1}"]]
        )  # Concatenate the upsampled tensor and skip connection
        tensors[f"De{self.levels - 1}"] = convolution_module(
            tensors[f"De{self.levels - 1}"],
            filters=self.lfilter_num[self.levels - 2],
            name=f"De{self.levels - 1}",
            **self.lmodule_kwargs,
        )  # Convolution module
        upsample_tensor = upsample(
            tensors[f"De{self.levels - 1}"],
            filters=self.lfilter_num[self.levels - 3],
            name=f"De{self.levels - 1}-De{self.levels - 2}",
            **self.lupsample_kwargs,
        )  # Connect the bottom decoder node to the next decoder node

        """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
        for decoder in np.arange(1, self.levels - 1)[::-1]:
            num_middle_nodes = self.levels - decoder - 1
            for node in range(1, num_middle_nodes + 1):
                if node == 1:  # if on the first middle node at the given level
                    upsample_tensor_for_middle_node = upsample(
                        tensors[f"En{decoder + 1}"],
                        filters=self.lfilter_num[decoder - 2],
                        name=f"En{decoder + 1}-Me{decoder}-1",
                        **self.lupsample_kwargs,
                    )
                    tensors[f"Me{decoder}-1"] = Concatenate(
                        name=f"Me{decoder}-1_Concatenate"
                    )([tensors[f"En{decoder}"], upsample_tensor_for_middle_node])
                else:
                    upsample_tensor_for_middle_node = upsample(
                        tensors[f"Me{decoder + 1}-{node - 1}"],
                        filters=self.lfilter_num[decoder - 2],
                        name=f"Me{decoder + 1}-{node - 1}-Me{decoder}-{node}",
                        **self.lupsample_kwargs,
                    )
                    tensors_to_concatenate = []  # Tensors to concatenate in the middle node
                    connections_to_add = sorted(
                        [tensor for tensor in tensors if f"Me{decoder}" in tensor]
                    )[
                        ::-1
                    ]  # skip connections to add to the list of tensors to concatenate
                    for connection in connections_to_add:
                        tensors_to_concatenate.append(tensors[connection])
                    tensors_to_concatenate.append(tensors[f"En{decoder}"])
                    tensors_to_concatenate.append(upsample_tensor_for_middle_node)
                    tensors[f"Me{decoder}-{node}"] = Concatenate(
                        name=f"Me{decoder}-{node}_Concatenate"
                    )(tensors_to_concatenate)
                tensors[f"Me{decoder}-{node}"] = convolution_module(
                    tensors[f"Me{decoder}-{node}"],
                    filters=self.lfilter_num[decoder - 1],
                    name=f"Me{decoder}-{node}",
                    **self.lmodule_kwargs,
                )  # Convolution module

                if decoder == 1 and self.ldeep_supervision:
                    tensors[f"sup{decoder}-{node}"] = deep_supervision_side_output(
                        tensors[f"Me{decoder}-{node}"],
                        name=f"sup{decoder}-{node}",
                        **self.supervision_kwargs,
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
                filters=self.lfilter_num[decoder - 1],
                name=f"De{decoder}",
                **self.lmodule_kwargs,
            )  # Convolution module

            if decoder != 1:  # if not currently on the final decoder node (De1)
                upsample_tensor = upsample(
                    tensors[f"De{decoder}"],
                    filters=self.lfilter_num[decoder - 2],
                    name=f"De{decoder}-De{decoder - 1}",
                    **self.lupsample_kwargs,
                )  # Connect the bottom decoder node to the next decoder node
            else:
                tensors["output"] = deep_supervision_side_output(
                    tensors["De1"], name="final", **self.supervision_kwargs
                )  # Deep supervision - this layer will output the model's prediction
                tensors_with_supervision.append(tensors["output"])

        output_model = Model(
            inputs=tensors["input"],
            outputs=tensors_with_supervision,
            name=f"unet_2plus_{self.ndims}D",
        )

        return output_model


@dataclasses.dataclass
class UNet3Plus(UNet):
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

    def build(self):
        ndims = (
            len(self.input_shape) - 1
        )  # Number of dimensions in the input image (excluding the last dimension reserved for channels)

        if self.levels < 3:
            raise ValueError(
                f"levels must be greater than 2. Received value: {self.levels}"
            )
        if len(self.input_shape) > 4 or len(self.input_shape) < 3:
            raise ValueError(
                f"input_shape can only have 3 or 4 dimensions (2D image + 1 dimension for channels OR a 3D image + 1 dimension for channels). Received shape: {np.shape(self.input_shape)}"
            )
        if len(self.filter_num) != self.levels:
            raise ValueError(
                f"length of filter_num ({len(self.filter_num)}) does not match the number of levels ({self.levels})"
            )

        if self.filter_num_skip is None:
            filter_num_skip = self.filter_num[0]

        if self.filter_num_aggregate is None:
            filter_num_aggregate = self.levels * filter_num_skip

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
            module_kwargs[arg] = getattr(self, arg)
        module_kwargs["num_modules"] = self.modules_per_node

        pool_kwargs = {"pool_size": self.pool_size}

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
            upsample_kwargs[arg] = getattr(self, arg)
            conventional_kwargs[arg] = getattr(self, arg)
            full_scale_kwargs[arg] = getattr(self, arg)
            aggregated_kwargs[arg] = getattr(self, arg)

        conventional_kwargs["filters"] = filter_num_skip
        upsample_kwargs["filters"] = filter_num_skip
        upsample_kwargs["upsample_size"] = self.upsample_size
        full_scale_kwargs["filters"] = filter_num_skip
        full_scale_kwargs["pool_size"] = self.pool_size
        aggregated_kwargs["filters"] = filter_num_skip
        aggregated_kwargs["upsample_size"] = self.upsample_size

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
            supervision_kwargs[arg] = getattr(self, arg)
        supervision_kwargs["activation"] = self.output_activation
        supervision_kwargs["use_bias"] = True

        tensors = dict({})  # Tensors associated with each node and skip connections
        tensors_with_supervision = []  # Outputs of deep supervision

        """ Setup the first encoder node with an input layer and a convolution module (we are not using skip connections here) """
        tensors["input"] = Input(shape=self.input_shape, name="Input")
        tensors["En1"] = convolution_module(
            tensors["input"], filters=self.filter_num[0], name="En1", **module_kwargs
        )

        if self.first_encoder_connections is True:
            for full_connection in range(2, self.levels):
                tensors[f"1---{full_connection}_full-scale"] = (
                    full_scale_skip_connection(
                        tensors["En1"],
                        level1=1,
                        level2=full_connection,
                        name=f"1---{full_connection}_full-scale",
                        **full_scale_kwargs,
                    )
                )

        """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
        for encoder in np.arange(
            2, self.levels
        ):  # Iterate through the rest of the encoder nodes
            pool_tensor = max_pool(
                tensors[f"En{encoder - 1}"],
                name=f"En{encoder - 1}-En{encoder}",
                **pool_kwargs,
            )  # Connect the next encoder node with a MaxPooling layer
            tensors[f"En{encoder}"] = convolution_module(
                pool_tensor,
                filters=self.filter_num[encoder - 1],
                name=f"En{encoder}",
                **module_kwargs,
            )  # Convolution modules
            tensors[f"{encoder}---{encoder}_skip"] = conventional_skip_connection(
                tensors[f"En{encoder}"],
                name=f"{encoder}---{encoder}_skip",
                **conventional_kwargs,
            )

            # Create full-scale skip connections
            for full_connection in range(encoder + 1, self.levels):
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
        tensors[f"En{self.levels}"] = max_pool(
            tensors[f"En{self.levels - 1}"],
            name=f"En{self.levels - 1}-En{self.levels}",
            **pool_kwargs,
        )
        tensors[f"En{self.levels}"] = convolution_module(
            tensors[f"En{self.levels}"],
            filters=self.filter_num[self.levels - 1],
            name=f"En{self.levels}",
            **module_kwargs,
        )
        if self.deep_supervision:
            tensors[f"sup{self.levels}_output"] = deep_supervision_side_output(
                tensors[f"En{self.levels}"],
                num_classes=self.num_classes,
                output_level=self.levels,
                name=f"sup{self.levels}",
                **supervision_kwargs,
            )
            tensors_with_supervision.append(tensors[f"sup{self.levels}_output"])

        # Add aggregated feature maps using the bottom encoder node
        for feature_map in range(1, self.levels - 1):
            tensors[f"{self.levels}---{feature_map}_feature"] = aggregated_feature_map(
                tensors[f"En{self.levels}"],
                level1=self.levels,
                level2=feature_map,
                name=f"{self.levels}---{feature_map}_feature",
                **aggregated_kwargs,
            )

        """ Build the rest of the decoder nodes """
        for decoder in np.arange(1, self.levels)[::-1]:
            """ The lowest decoder node (levels - 1) is attached to the bottom encoder node via upsampling, so concatenation is slightly different """
            if decoder == self.levels - 1:
                tensors[f"De{decoder}"] = upsample(
                    tensors[f"En{self.levels}"],
                    name=f"En{self.levels}-De{decoder}",
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
                    tensors_to_concatenate[index] = tensors[
                        tensors_to_concatenate[index]
                    ]
                tensors_to_concatenate.insert(
                    self.levels - 1 - decoder, tensors[f"De{decoder}"]
                )

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
                self.deep_supervision or decoder == 1
            ):  # Decoder node 1 must always have deep supervision
                tensors[f"sup{decoder}_output"] = deep_supervision_side_output(
                    tensors[f"De{decoder}"],
                    num_classes=self.num_classes,
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

        output_model = Model(
            inputs=tensors["input"],
            outputs=tensors_with_supervision[::-1],
            name=f"unet_3plus_{ndims}D",
        )

        return output_model


@dataclasses.dataclass
class AttentionUNet(UNet):
    """
    Builds an Attention U-Net model.

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

    def build(self):
        if len(self.upsample_size) > 0:
            raise ValueError(
                "AttentionUNet does not support upsample_size, use empty tuple."
            )
        self.supervision_kwargs["activation"] = self.output_activation
        self.supervision_kwargs["upsample_size"] = self.pool_size
        self.supervision_kwargs["use_bias"] = True
        self.supervision_kwargs["output_level"] = 1
        self.supervision_kwargs["kernel_size"] = 1

        tensors = dict({})  # Tensors associated with each node and skip connections

        """ Setup the first encoder node with an input layer and a convolution module """
        tensors["input"] = Input(shape=self.input_shape, name="Input")
        tensors["En1"] = convolution_module(
            tensors["input"],
            filters=self.filter_num[0],
            name="En1",
            **self.module_kwargs,
        )

        """ The rest of the encoder nodes are handled here. Each encoder node is connected with a MaxPooling layer and contains convolution modules """
        for encoder in np.arange(
            2, self.levels + 1
        ):  # Iterate through the rest of the encoder nodes
            pool_tensor = max_pool(
                tensors[f"En{encoder - 1}"],
                name=f"En{encoder - 1}-En{encoder}",
                **self.pool_kwargs,
            )  # Connect the next encoder node with a MaxPooling layer
            tensors[f"En{encoder}"] = convolution_module(
                pool_tensor,
                filters=self.filter_num[encoder - 1],
                name=f"En{encoder}",
                **self.module_kwargs,
            )  # Convolution modules

        tensors[f"AG{self.levels - 1}"] = attention_gate(
            tensors[f"En{self.levels - 1}"],
            tensors[f"En{self.levels}"],
            self.kernel_size,
            self.pool_size,
            name=f"AG{self.levels - 1}",
        )
        upsample_tensor = upsample(
            tensors[f"En{self.levels}"],
            filters=self.filter_num[self.levels - 2],
            name=f"En{self.levels}-De{self.levels - 1}",
            **self.upsample_kwargs,
        )  # Connect the bottom encoder node to a decoder node

        """ Bottom decoder node """
        tensors[f"De{self.levels - 1}"] = Concatenate(
            name=f"De{self.levels - 1}_Concatenate"
        )(
            [tensors[f"AG{self.levels - 1}"], upsample_tensor]
        )  # Concatenate the upsampled tensor and skip connection
        tensors[f"De{self.levels - 1}"] = convolution_module(
            tensors[f"De{self.levels - 1}"],
            filters=self.filter_num[self.levels - 2],
            name=f"De{self.levels - 1}",
            **self.module_kwargs,
        )  # Convolution module
        tensors[f"AG{self.levels - 2}"] = attention_gate(
            tensors[f"En{self.levels - 2}"],
            tensors[f"De{self.levels - 1}"],
            self.kernel_size,
            self.pool_size,
            name=f"AG{self.levels - 2}",
        )
        upsample_tensor = upsample(
            tensors[f"De{self.levels - 1}"],
            filters=self.filter_num[self.levels - 3],
            name=f"De{self.levels - 1}-De{self.levels - 2}",
            **self.upsample_kwargs,
        )  # Connect the bottom decoder node to the next decoder node

        """ The rest of the decoder nodes (except the final node) are handled in this loop. Each node contains one concatenation of an upsampled tensor and a skip connection """
        for decoder in np.arange(2, self.levels - 1)[::-1]:
            tensors[f"De{decoder}"] = Concatenate(name=f"De{decoder}_Concatenate")(
                [tensors[f"AG{decoder}"], upsample_tensor]
            )  # Concatenate the upsampled tensor and skip connection
            tensors[f"De{decoder}"] = convolution_module(
                tensors[f"De{decoder}"],
                filters=self.filter_num[decoder - 1],
                name=f"De{decoder}",
                **self.module_kwargs,
            )  # Convolution module
            tensors[f"AG{decoder - 1}"] = attention_gate(
                tensors[f"En{decoder - 1}"],
                tensors[f"De{decoder}"],
                self.kernel_size,
                self.pool_size,
                name=f"AG{decoder - 1}",
            )
            upsample_tensor = upsample(
                tensors[f"De{decoder}"],
                filters=self.filter_num[decoder - 2],
                name=f"De{decoder}-De{decoder - 1}",
                **self.upsample_kwargs,
            )  # Connect the bottom decoder node to the next decoder node

        """ Final decoder node begins with a concatenation and convolution module, followed by deep supervision """
        tensor_De1 = Concatenate(name="De1_Concatenate")(
            [tensors["AG1"], upsample_tensor]
        )  # Concatenate the upsampled tensor and skip connection
        tensor_De1 = convolution_module(
            tensor_De1, filters=self.filter_num[0], name="De1", **self.module_kwargs
        )  # Convolution module
        tensors["output"] = deep_supervision_side_output(
            tensor_De1, name="final", **self.supervision_kwargs
        )  # Deep supervision - this layer will output the model's prediction

        output_model = Model(
            inputs=tensors["input"],
            outputs=tensors["output"],
            name=f"attention_unet_{self.ndims}D",
        )

        return output_model


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
