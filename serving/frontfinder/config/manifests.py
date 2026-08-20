"""Model manifests: declares exactly which variables/pressure levels each
frontfinder Keras model expects as input, in channel order.

Sourced from Taylor's `fronts` repo configs:
  - best_loss:   configs/sooner_ablations.yaml (n_channels: 30, levels confirmed
                 by Taylor as [1000, 925, 850, 700, 500, 300])
  - model_1702:  configs/model_1702/generate_conus.yaml (pressure_levels:
                 [1000, 950, 900, 850])

Both source configs list a CONUS bounding box (`coordinates:
[25.0, 56.75, 228.0, 299.75]`) as the training/eval domain. Taylor has
confirmed frontfinder should nonetheless run true global inference on the
IFS grid -- these models have not been validated outside CONUS, so the
serving pipeline should be treated as producing a global *extrapolation*,
not a validated global product, until accuracy outside CONUS is checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


# Front classes as predicted by the model output layer (softmax over 6
# classes, matching `model_config.n_classes: 6` in sooner_ablations.yaml:
# background + 5 front types). Class order follows the AIES FrontFinder
# convention used across both configs.
ALL_CLASSES: tuple[str, ...] = (
    "background",
    "cold",
    "warm",
    "stationary",
    "occluded",
    "dryline",
)

# Per Taylor: the model predicts drylines but frontfinder will not serve
# them. "background" is never served either.
SERVED_CLASSES: tuple[str, ...] = ("cold", "warm", "stationary", "occluded")


@dataclass(frozen=True)
class VariableSpec:
    """One input variable. `levels=None` means a single-level/surface field."""

    name: str
    levels: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.levels is not None and len(self.levels) == 0:
            raise ValueError(f"{self.name}: levels must be non-empty or None, not ()")
        if self.levels is not None and len(set(self.levels)) != len(self.levels):
            raise ValueError(f"{self.name}: duplicate pressure levels in {self.levels}")

    @property
    def n_channels(self) -> int:
        return len(self.levels) if self.levels is not None else 1

    def channel_names(self) -> list[str]:
        if self.levels is None:
            return [self.name]
        return [f"{self.name}_{lvl}" for lvl in self.levels]


@dataclass(frozen=True)
class ModelManifest:
    name: str
    weights_filename: str
    variables: tuple[VariableSpec, ...]
    patch_multiple: int = 16
    all_classes: tuple[str, ...] = field(default=ALL_CLASSES)
    served_classes: tuple[str, ...] = field(default=SERVED_CLASSES)

    def __post_init__(self) -> None:
        if len(self.variables) == 0:
            raise ValueError(f"{self.name}: manifest has no variables")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate variable names in manifest: {names}")
        missing = set(self.served_classes) - set(self.all_classes)
        if missing:
            raise ValueError(f"{self.name}: served_classes not in all_classes: {missing}")

    @property
    def n_channels(self) -> int:
        return sum(v.n_channels for v in self.variables)

    def channel_names(self) -> list[str]:
        names: list[str] = []
        for v in self.variables:
            names.extend(v.channel_names())
        return names

    def served_class_indices(self) -> list[int]:
        return [self.all_classes.index(c) for c in self.served_classes]


BEST_LOSS_MANIFEST = ModelManifest(
    name="best_loss",
    weights_filename="_best_loss.keras",
    variables=(
        VariableSpec("equivalent_potential_temperature", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("u_component_of_wind", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("v_component_of_wind", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("specific_humidity", levels=(1000, 925, 850, 700, 500, 300)),
        VariableSpec("potential_vorticity", levels=(1000, 925, 850, 700, 500, 300)),
    ),
)

MODEL_1702_MANIFEST = ModelManifest(
    name="model_1702",
    weights_filename="model_1702.h5",
    variables=(
        VariableSpec("geopotential", levels=(1000, 950, 900, 850)),
        VariableSpec("temperature", levels=(1000, 950, 900, 850)),
        VariableSpec("u_component_of_wind", levels=(1000, 950, 900, 850)),
        VariableSpec("v_component_of_wind", levels=(1000, 950, 900, 850)),
        VariableSpec("specific_humidity", levels=(1000, 950, 900, 850)),
        VariableSpec("surface_pressure", levels=None),
        VariableSpec("2m_temperature", levels=None),
        VariableSpec("2m_dewpoint_temperature", levels=None),
        VariableSpec("10m_u_component_of_wind", levels=None),
        VariableSpec("10m_v_component_of_wind", levels=None),
    ),
)

MANIFESTS: dict[str, ModelManifest] = {
    BEST_LOSS_MANIFEST.name: BEST_LOSS_MANIFEST,
    MODEL_1702_MANIFEST.name: MODEL_1702_MANIFEST,
}


def get_manifest(model_name: str) -> ModelManifest:
    try:
        return MANIFESTS[model_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {model_name!r}; known models: {sorted(MANIFESTS)}"
        ) from exc
