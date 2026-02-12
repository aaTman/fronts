import dataclasses
from typing import Literal, Any
from fronts.models import unets
from fronts.models.keras_builders import (
    ConvRegularizerConfig,
    BiasVectorConfig,
    KernelConfig,
    OptimizerConfig,
)


@dataclasses.dataclass
class CoreUNetConfig:
    loss: LossConfig
    metric: MetricConfig
    optimizer: OptimizerConfig
    convolution_activity_regularizer: ConvRegularizerConfig
    bias_vector: BiasVectorConfig
    kernel_matrix: KernelConfig
    activation: list[str]
    batch_normalization: bool
    num_filters: list[int]
    kernel_size: list[int]
    depth: int
    modules_per_node: int
    padding: Literal["same", "valid"]
    pool_size: tuple[int]
    upsample_size: tuple[int]
    bias: bool

    def __post_init__(self):
        if len(self.num_filters) != self.depth:
            raise ValueError(
                f"Length of num_filters ({len(self.num_filters)}) must match depth "
                f"({self.depth})"
            )
        # TODO: modify pool size if needed to match len of dims
        # TODO: match kernel size to depth as well
        # TODO: match upsample_size as well


@dataclasses.dataclass
class UNetEnsembleConfig(CoreUNetConfig):
    num_models: int


@dataclasses.dataclass
class UNetPlusConfig(CoreUNetConfig):
    deep_supervision: bool


@dataclasses.dataclass
class UNet2PlusConfig:
    deep_supervision: bool


@dataclasses.dataclass
class UNet3PlusConfig(CoreUNetConfig):
    deep_supervision: bool
    num_aggregate_filters: int
    full_scale_skip_connection_filters: int
    first_encoder_connections: bool


@dataclasses.dataclass
class AttentionUNetConfig:
    deep_supervision: bool

    def __post_init__(self):
        if len(self.upsample_size) > 0:
            raise ValueError(
                "AttentionUNet does not support upsample_size, use empty tuple."
            )


@dataclasses.dataclass
class UNetRegistry:
    model: Literal[
        "unet",
        "unet_ensemble",
        "unet_plus",
        "unet_2plus",
        "unet_3plus",
        "attention_unet",
    ]
    config: dict[str, Any]

    def build(self):
        match self.model:
            case "unet":
                from .unet import UNet

                return UNet(self.config)
            case "unet_ensemble":
                from .unet_ensemble import UNetEnsemble

                return UNetEnsemble(self.config)
            case "unet_plus":
                from .unet_plus import UNetPlus

                return UNetPlus(self.config)
            case "unet_2plus":
                from .unet_2plus import UNet2Plus

                return UNet2Plus(self.config)
            case "unet_3plus":
                from .unet_3plus import UNet3Plus

                return UNet3Plus(self.config)
            case "attention_unet":
                from .attention_unet import AttentionUNet

                return AttentionUNet(self.config)
