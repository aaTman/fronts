import dataclasses
from typing import Literal, Any
from fronts.models import unets
from fronts.models.losses import LossConfig
from fronts.models.metrics import MetricsConfig


@dataclasses.dataclass
class CoreUNetConfig:
    activation: list[str]
    batch_normalization: bool
    num_filters: int
    kernel_size: list[int]
    depth: int
    loss: LossConfig
    metric: MetricConfig


@dataclasses.dataclass
class UNetEnsembleConfig(CoreUNetConfig):
    num_models: int


@dataclasses.dataclass
class UNetPlusConfig(CoreUNetConfig):
    activation: list[str]
    batch_normalization: bool
    deep_supervision: bool
    num_filters: int


@dataclasses.dataclass
class UNet2PlusConfig:
    activation: list[str]
    batch_normalization: bool
    deep_supervision: bool
    num_filters: int


@dataclasses.dataclass
class UNet3PlusConfig(CoreUNetConfig):
    num_aggregate_filters: int
    full_scale_skip_connection_filters: int


@dataclasses.dataclass
class AttentionUNetConfig:
    activation: list[str]
    batch_normalization: bool
    deep_supervision: bool
    num_filters: int


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
