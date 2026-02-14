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

    def build(self):
        """Builds the UNet model based on the configuration."""
        model = Model(
            name=self.name,
            convolution_activity_regularizer=self.convolution_activity_regularizer,
            bias_vector=self.bias_vector,
            kernel_matrix=self.kernel_matrix,
            activation=self.activation,
            batch_normalization=self.batch_normalization,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            depth=self.depth,
            modules_per_node=self.modules_per_node,
            padding=self.padding,
            pool_size=self.pool_size,
            upsample_size=self.upsample_size,
            bias=self.bias,
        )
        return model


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
        output_activation_config: ActivationConfig,
        batch_normalization: bool,
        num_filters: list[int],
        kernel_size: list[int],
        depth: int,
        modules_per_node: int,
        padding: Literal["same", "valid"],
        pool_size: tuple[int],
        upsample_size: tuple[int],
        bias: bool,
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
        self.output_activation_config = output_activation_config
        self.batch_normalization = batch_normalization
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.modules_per_node = modules_per_node
        self.padding = padding
        self.pool_size = pool_size
        self.upsample_size = upsample_size
        self.bias = bias

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
        self.output_activation = self.output_activation_config.build()
        # TODO: input_shape and num_classes
        # TODO: modify pool size if needed to match len of dims
        # TODO: match kernel size to depth as well
        # TODO: match upsample_size as well

        # Build config to match arg names in the unet functions
        def build(self):
            self.squeeze_axes = determine_squeeze_axes()
            self.shared_axes = determine_shared_axes()
            """Builds the model based on the configuration."""
            self.config = {
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "pool_size": self.pool_size,
                "upsample_size": self.upsample_size,
                "levels": self.depth,
                "filter_num": self.filter_num,
                "kernel_size": self.kernel_size,
                "squeeze_axes": self.squeeze_axes,
                "shared_axes": self.shared_axes,
                "modules_per_node": self.modules_per_node,
                "batch_normalization": self.batch_normalization,
                "activation": self.activation,
                "output_activation": self.output_activation,
                "padding": self.padding,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_matrix.kernel_initializer,
                "bias_initializer": self.bias_vector.bias_initializer,
                "kernel_regularizer": self.kernel_matrix.kernel_regularizer,
                "bias_regularizer": self.bias_vector.bias_regularizer,
                "activity_regularizer": self.activity_regularizer.activity_regularizer,
                "kernel_constraint": self.kernel_matrix.kernel_constraint,
                "bias_constraint": self.bias_vector.bias_constraint,
            }
            output_model = UNetRegistry(name=self.name, config=self.config).build()
            return output_model


def determine_squeeze_axes() -> int | None:
    pass


def determine_shared_axes() -> int | None:
    pass
