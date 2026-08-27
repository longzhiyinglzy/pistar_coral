"""Composable parameter filters for CORAL expert training and inference."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import enum
from typing import Any

import flax.nnx as nnx

from openpi.shared import nnx_utils


class CoralParamModule(enum.Enum):
    """Atomic parameter groups that can be combined into one expert."""

    LORA = "lora"
    ACTION_HEAD = "action_head"
    IMAGE_TOWER = "image_tower"
    ACTION_EXPERT = "action_expert"


class CoralExpertParamMode(enum.Enum):
    """Legacy predefined combinations kept for existing configs and exports."""

    LORA_ONLY = "lora_only"
    LORA_PLUS_ACTION_HEAD = "lora_plus_action_head"
    OPENPI_DEFAULT_LORA = "openpi_default_lora"


_LEGACY_MODE_MODULES = {
    CoralExpertParamMode.LORA_ONLY: frozenset({CoralParamModule.LORA}),
    CoralExpertParamMode.LORA_PLUS_ACTION_HEAD: frozenset(
        {CoralParamModule.LORA, CoralParamModule.ACTION_HEAD}
    ),
    CoralExpertParamMode.OPENPI_DEFAULT_LORA: frozenset(
        {CoralParamModule.LORA, CoralParamModule.ACTION_HEAD, CoralParamModule.IMAGE_TOWER}
    ),
}


@dataclasses.dataclass(frozen=True)
class CoralParamSpec:
    """Validated, order-independent set of hot-switchable parameter modules."""

    modules: frozenset[CoralParamModule]

    def __post_init__(self) -> None:
        modules = frozenset(_parse_module(module) for module in self.modules)
        if not modules:
            raise ValueError("A CORAL parameter spec must contain at least one module.")
        object.__setattr__(self, "modules", modules)

    @classmethod
    def from_modules(cls, modules: Iterable[CoralParamModule | str]) -> "CoralParamSpec":
        return cls(frozenset(_parse_module(module) for module in modules))

    @property
    def values(self) -> tuple[str, ...]:
        """Stable representation for metadata, logs, and equality diagnostics."""
        return tuple(sorted(module.value for module in self.modules))

    @property
    def label(self) -> str:
        return "+".join(self.values)


CoralParamSpecInput = CoralParamSpec | CoralExpertParamMode | CoralParamModule | str | Iterable[CoralParamModule | str]


_LORA_FILTER = nnx_utils.PathRegex(".*lora.*")
_ACTION_HEAD_FILTERS = (
    nnx_utils.PathRegex(".*action_in_proj.*"),
    nnx_utils.PathRegex(".*state_proj.*"),
    nnx_utils.PathRegex(".*action_time_mlp_in.*"),
    nnx_utils.PathRegex(".*action_time_mlp_out.*"),
    nnx_utils.PathRegex(".*action_out_proj.*"),
    # pi0.5 uses these timestep projection heads instead of action_time_mlp_*.
    nnx_utils.PathRegex(".*time_mlp_in.*"),
    nnx_utils.PathRegex(".*time_mlp_out.*"),
)
_IMAGE_TOWER_FILTER = nnx_utils.PathRegex(".*PaliGemma/img.*")
# Pi0 stores the second Gemma expert (the 300M action expert) with an `_1` suffix.
_ACTION_EXPERT_FILTER = nnx_utils.PathRegex(".*llm.*_1.*")


def parse_param_spec(value: CoralParamSpecInput) -> CoralParamSpec:
    """Normalize a module combination or legacy mode into a parameter spec."""
    if isinstance(value, CoralParamSpec):
        return value
    if isinstance(value, CoralExpertParamMode):
        return CoralParamSpec(_LEGACY_MODE_MODULES[value])
    if isinstance(value, CoralParamModule):
        return CoralParamSpec(frozenset({value}))
    if isinstance(value, str):
        stripped = value.strip()
        try:
            legacy_mode = CoralExpertParamMode(stripped)
        except ValueError:
            modules = [part.strip() for part in stripped.split(",") if part.strip()]
            return CoralParamSpec.from_modules(modules)
        return CoralParamSpec(_LEGACY_MODE_MODULES[legacy_mode])
    return CoralParamSpec.from_modules(value)


def parse_metadata_param_spec(
    *,
    param_modules: Iterable[str] | None,
    param_mode: str | None,
) -> CoralParamSpec:
    """Read new metadata while retaining strict compatibility with old exports."""
    if param_modules is None:
        return parse_param_spec(param_mode or CoralExpertParamMode.LORA_ONLY)

    spec = CoralParamSpec.from_modules(param_modules)
    if param_mode is not None and parse_param_spec(param_mode) != spec:
        raise ValueError(
            f"Conflicting CORAL metadata: param_modules={spec.values}, param_mode='{param_mode}'."
        )
    return spec


def parse_param_mode(mode: CoralExpertParamMode | str) -> CoralExpertParamMode:
    """Parse a legacy mode. New code should use :func:`parse_param_spec`."""
    if isinstance(mode, CoralExpertParamMode):
        return mode
    try:
        return CoralExpertParamMode(str(mode))
    except ValueError as exc:
        valid = ", ".join(item.value for item in CoralExpertParamMode)
        raise ValueError(f"Unknown CORAL expert param mode '{mode}'. Valid modes: {valid}") from exc


def get_module_filter(module: CoralParamModule | str) -> nnx.filterlib.Filter:
    """Return the path filter for one atomic module."""
    module = _parse_module(module)
    if module is CoralParamModule.LORA:
        return _LORA_FILTER
    if module is CoralParamModule.ACTION_HEAD:
        return nnx.Any(*_ACTION_HEAD_FILTERS)
    if module is CoralParamModule.IMAGE_TOWER:
        return _IMAGE_TOWER_FILTER
    if module is CoralParamModule.ACTION_EXPERT:
        return _ACTION_EXPERT_FILTER
    raise AssertionError(f"Unhandled CORAL parameter module: {module}")


def get_coral_expert_filter(spec: CoralParamSpecInput) -> nnx.filterlib.Filter:
    """Return the union of parameters trained and exported for one expert."""
    parsed = parse_param_spec(spec)
    filters = tuple(get_module_filter(module) for module in parsed.modules)
    return filters[0] if len(filters) == 1 else nnx.Any(*filters)


def get_coral_freeze_filter(spec: CoralParamSpecInput) -> nnx.filterlib.Filter:
    """Return a freeze filter whose inverse is exactly the expert filter."""
    return nnx.All(nnx.Param, nnx.Not(get_coral_expert_filter(spec)))


def filter_param_dict(params: dict[str, Any], spec: CoralParamSpecInput) -> dict[str, Any]:
    """Extract a pure parameter pytree using the shared CORAL module spec."""
    import flax.traverse_util

    expert_filter = get_coral_expert_filter(spec)
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_expert = {path: value for path, value in flat_params.items() if expert_filter(path, value)}
    return flax.traverse_util.unflatten_dict(flat_expert)


def count_module_leaves(params: dict[str, Any], spec: CoralParamSpecInput) -> dict[CoralParamModule, int]:
    """Count matching leaves per requested module for export validation."""
    import flax.traverse_util

    parsed = parse_param_spec(spec)
    flat_params = flax.traverse_util.flatten_dict(params)
    return {
        module: sum(1 for path, value in flat_params.items() if get_module_filter(module)(path, value))
        for module in parsed.modules
    }


def _parse_module(module: CoralParamModule | str) -> CoralParamModule:
    if isinstance(module, CoralParamModule):
        return module
    try:
        return CoralParamModule(str(module).strip())
    except ValueError as exc:
        valid = ", ".join(item.value for item in CoralParamModule)
        raise ValueError(f"Unknown CORAL parameter module '{module}'. Valid modules: {valid}") from exc
