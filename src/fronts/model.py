import dataclasses
from typing import Literal
from fronts.layers.unets import UNetRegistry
from fronts.utils.keras_builders import (
    ConvOutputConfig,
    BiasVectorConfig,
    KernelMatrixConfig,
    OptimizerConfig,
    ActivationConfig,
    LossConfig,
    MetricConfig,
)


@dataclasses.dataclass
class ModelConfig:
    """Configuration for the model architecture and training parameters."""

    name: Literal[
        "unet",
        "unet_ensemble",
        "unet_plus",
        "unet_2plus",
        "unet_3plus",
        "attention_unet",
    ]
    loss: LossConfig
    metric: MetricConfig
    optimizer: OptimizerConfig
    convolution_activity_regularizer: ConvOutputConfig
    bias_vector: BiasVectorConfig
    kernel_matrix: KernelMatrixConfig
    activation: ActivationConfig
    batch_normalization: bool
    num_filters: list[int]
    kernel_size: list[int]
    depth: int
    modules_per_node: int
    padding: Literal["same", "valid"]
    pool_size: tuple[int]
    upsample_size: tuple[int]
    bias: bool
    deep_supervision: bool = False
    first_encoder_connections: bool = False

    def build(self, input_shape: tuple, num_classes: int):
        """Builds and compiles the UNet model based on the configuration.

        Args:
            input_shape: Shape of the model inputs (excluding batch dimension).
                Spatial dims should be None to allow variable-size inference,
                e.g. (None, None, 7, 9) for 3D inputs.
            num_classes: Number of output classes (e.g. 6 for 5 front types + background).

        Returns a compiled tf.keras.Model.
        """
        model = Model(
            name=self.name,
            loss_config=self.loss,
            metric_config=self.metric,
            optimizer_config=self.optimizer,
            convolution_activity_regularizer_config=self.convolution_activity_regularizer,
            bias_vector_config=self.bias_vector,
            kernel_matrix_config=self.kernel_matrix,
            activation_config=self.activation,
            batch_normalization=self.batch_normalization,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            depth=self.depth,
            modules_per_node=self.modules_per_node,
            padding=self.padding,
            pool_size=self.pool_size,
            upsample_size=self.upsample_size,
            bias=self.bias,
            input_shape=input_shape,
            num_classes=num_classes,
            deep_supervision=self.deep_supervision,
            first_encoder_connections=self.first_encoder_connections,
        )
        return model.build()


class Model:
    def __init__(
        self,
        name: Literal[
            "unet",
            "unet_ensemble",
            "unet_plus",
            "unet_2plus",
            "unet_3plus",
            "attention_unet",
        ],
        loss_config: LossConfig,
        metric_config: MetricConfig,
        optimizer_config: OptimizerConfig,
        convolution_activity_regularizer_config: ConvOutputConfig,
        bias_vector_config: BiasVectorConfig,
        kernel_matrix_config: KernelMatrixConfig,
        activation_config: ActivationConfig,
        batch_normalization: bool,
        num_filters: list[int],
        kernel_size: list[int],
        depth: int,
        modules_per_node: int,
        padding: Literal["same", "valid"],
        pool_size: tuple[int],
        upsample_size: tuple[int],
        bias: bool,
        input_shape: tuple,
        num_classes: int,
        deep_supervision: bool = False,
        first_encoder_connections: bool = False,
    ):
        self.name = name
        self.loss_config = loss_config
        self.metric_config = metric_config
        self.optimizer_config = optimizer_config
        self.convolution_activity_regularizer_config = (
            convolution_activity_regularizer_config
        )
        self.bias_vector_config = bias_vector_config
        self.kernel_matrix_config = kernel_matrix_config
        self.activation_config = activation_config
        self.batch_normalization = batch_normalization
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.modules_per_node = modules_per_node
        self.padding = padding
        self.pool_size = pool_size
        self.upsample_size = upsample_size
        self.bias = bias
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.first_encoder_connections = first_encoder_connections

        if len(self.num_filters) != self.depth:
            raise ValueError(
                f"Length of num_filters ({len(self.num_filters)}) must match depth "
                f"({self.depth})"
            )
        # Build keras objects
        self.loss = self.loss_config.build()
        self.metric = self.metric_config.build()
        self.optimizer = self.optimizer_config.build()
        self.activity_regularizer = self.convolution_activity_regularizer_config.build()
        self.bias_vector = self.bias_vector_config.build()
        self.kernel_matrix = self.kernel_matrix_config.build()
        self.activation = self.activation_config.build()

    def build(self):
        """Builds and compiles the Keras model."""
        # For 3D inputs (lat, lon, level, channels), squeeze the level axis (index 2)
        # so the UNet output matches the 2D target shape (lat, lon, classes).
        # For 2D inputs (lat, lon, channels), no squeezing is needed.
        squeeze_axes = determine_squeeze_axes(self.input_shape)
        shared_axes = determine_shared_axes(self.input_shape)

        self.config = {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "pool_size": self.pool_size,
            "upsample_size": self.upsample_size,
            "levels": self.depth,
            "filter_num": self.num_filters,
            "kernel_size": self.kernel_size,
            "squeeze_axes": squeeze_axes,
            "shared_axes": shared_axes,
            "modules_per_node": self.modules_per_node,
            "batch_normalization": self.batch_normalization,
            "activation": self.activation_config.name,
            "padding": self.padding,
            "use_bias": self.bias,
            "kernel_initializer": self.kernel_matrix.kernel_initializer,
            "bias_initializer": self.bias_vector.bias_initializer,
            "kernel_regularizer": self.kernel_matrix.kernel_regularizer,
            "bias_regularizer": self.bias_vector.bias_regularizer,
            "activity_regularizer": self.activity_regularizer.activity_regularizer,
            "kernel_constraint": self.kernel_matrix.kernel_constraint,
            "bias_constraint": self.bias_vector.bias_constraint,
        }
        # UNet3Plus has extra fields not present in other UNet variants.
        # Pass defaults so the dataclass instantiates correctly.
        if self.name == "unet_3plus":
            self.config.update({
                "filter_num_skip": None,       # defaults to filter_num[0]
                "filter_num_aggregate": None,  # defaults to levels * filter_num_skip
                "first_encoder_connections": self.first_encoder_connections,
                "deep_supervision": self.deep_supervision,
            })
        # UNetRegistry.build() instantiates the UNet dataclass; .build() on that
        # dataclass constructs and returns the tf.keras.Model.
        output_model = UNetRegistry(name=self.name, config=self.config).build().build()
        output_model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=[self.metric],
        )
        return output_model


def determine_squeeze_axes(input_shape: tuple) -> int | None:
    """Returns the axis to squeeze for 3D→2D output, or None for 2D inputs.

    The UNet processes 3D inputs (lat, lon, level, channels) but produces
    2D targets (lat, lon, classes). The level axis (index 3, 1-based with
    batch) is squeezed in the final output layer.

    For 2D inputs (lat, lon, channels) no squeeze is needed.
    """
    # input_shape excludes batch dim, e.g. (None, None, 7, 9) = 3D, (None, None, 9) = 2D
    ndims = len(input_shape) - 1  # spatial dims (exclude channel dim)
    return 3 if ndims == 3 else None


def determine_shared_axes(input_shape: tuple) -> int | None:
    """Returns shared axes for learnable activation parameters, or None.

    Following the legacy convention: None (share across all arbitrary dims).
    """
    return None
