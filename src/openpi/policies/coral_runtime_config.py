"""Runtime config and validation for CORAL expert deployment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any, Literal

import flax.traverse_util

from openpi.policies import coral_param_filters
from openpi.policies import lora_expert_manager


SwitchMode = Literal["slow", "hot"]
RUNTIME_CONFIG_FILE = "coral_runtime.json"
_MODULE_ORDER = {
    coral_param_filters.CoralParamModule.LORA: 0,
    coral_param_filters.CoralParamModule.ACTION_HEAD: 1,
    coral_param_filters.CoralParamModule.IMAGE_TOWER: 2,
    coral_param_filters.CoralParamModule.ACTION_EXPERT: 3,
}


@dataclasses.dataclass(frozen=True)
class CoralRuntimeConfig:
    """Explicit deployment contract for one CORAL server process."""

    base_config: str
    experts_dir: str
    root: str | None = None
    base_checkpoint_dir: str | None = None
    default_expert: str | None = None
    expert_names: tuple[str, ...] | None = None
    model_type: str | None = None
    param_modules: tuple[str, ...] | None = None
    switch_mode: SwitchMode = "hot"

    @classmethod
    def from_json_file(cls, path: str | pathlib.Path) -> "CoralRuntimeConfig":
        config_path = pathlib.Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"CORAL runtime config must be a JSON object: {config_path}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CoralRuntimeConfig":
        base_config = _required_str(data, "base_config")
        root = _optional_str(data, "root")
        switch_mode = str(data.get("switch_mode", "hot"))
        if switch_mode not in {"slow", "hot"}:
            raise ValueError("CORAL runtime config field 'switch_mode' must be 'slow' or 'hot'.")
        model_type = _optional_model_type(data.get("model_type"))

        expert_names = data.get("expert_names")
        if expert_names is not None:
            if isinstance(expert_names, str) or not isinstance(expert_names, list):
                raise ValueError("CORAL runtime config field 'expert_names' must be a list of expert names.")
            expert_names = tuple(_non_empty_str(name, "expert_names item") for name in expert_names)
            if len(set(expert_names)) != len(expert_names):
                raise ValueError("CORAL runtime config field 'expert_names' contains duplicates.")

        param_modules = data.get("param_modules")
        if param_modules is not None:
            if isinstance(param_modules, str) or not isinstance(param_modules, list):
                raise ValueError("CORAL runtime config field 'param_modules' must be a list of module names.")
            # Parse once here so config errors point at the runtime config, not later inference setup.
            param_modules = canonical_module_values(str(module) for module in param_modules)

        experts_dir = _optional_str(data, "experts_dir")
        if experts_dir is None:
            if root is None:
                raise ValueError("CORAL runtime config requires either 'experts_dir' or 'root'.")
            if model_type is None:
                raise ValueError("CORAL runtime config with 'root' requires 'model_type'.")
            if param_modules is None:
                raise ValueError("CORAL runtime config with 'root' requires 'param_modules'.")
            experts_dir = str(structured_experts_dir(root, model_type, base_config, param_modules))

        return cls(
            base_config=base_config,
            experts_dir=experts_dir,
            root=root,
            base_checkpoint_dir=_optional_str(data, "base_checkpoint_dir"),
            default_expert=_optional_str(data, "default_expert"),
            expert_names=expert_names,
            model_type=model_type,
            param_modules=param_modules,
            switch_mode=switch_mode,  # type: ignore[arg-type]
        )

    @classmethod
    def for_structured_root(
        cls,
        *,
        root: str | pathlib.Path,
        base_config: str,
        model_type: str,
        param_modules: coral_param_filters.CoralParamSpecInput,
        base_checkpoint_dir: str | None = None,
        default_expert: str | None = None,
        expert_names: Sequence[str] = (),
        switch_mode: SwitchMode = "hot",
    ) -> "CoralRuntimeConfig":
        modules = canonical_module_values(param_modules)
        experts_dir = structured_experts_dir(root, model_type, base_config, modules)
        return cls(
            base_config=base_config,
            experts_dir=str(experts_dir),
            root=str(root),
            base_checkpoint_dir=base_checkpoint_dir,
            default_expert=default_expert,
            expert_names=tuple(expert_names) or None,
            model_type=_required_model_type(model_type),
            param_modules=modules,
            switch_mode=switch_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_config": self.base_config,
            "base_checkpoint_dir": self.base_checkpoint_dir,
            "root": self.root,
            "experts_dir": self.experts_dir,
            "model_type": self.model_type,
            "param_modules": list(self.param_modules) if self.param_modules is not None else None,
            "default_expert": self.default_expert,
            "expert_names": list(self.expert_names) if self.expert_names is not None else None,
            "switch_mode": self.switch_mode,
        }


def with_model_type(config: CoralRuntimeConfig, model_type: str) -> CoralRuntimeConfig:
    """Bind a runtime config to the model type resolved from its base config."""
    resolved = _optional_model_type(model_type)
    assert resolved is not None
    if config.model_type is not None and config.model_type != resolved:
        raise ValueError(
            f"Runtime config model_type='{config.model_type}' does not match base config model_type='{resolved}'."
        )
    return dataclasses.replace(config, model_type=resolved)


def with_expert(config: CoralRuntimeConfig, expert_name: str) -> CoralRuntimeConfig:
    """Return a config whose expert_names/default_expert include a new export."""
    name = _non_empty_str(expert_name, "expert_name")
    expert_names = tuple(config.expert_names or ())
    if name not in expert_names:
        expert_names = (*expert_names, name)
    return dataclasses.replace(
        config,
        default_expert=config.default_expert or name,
        expert_names=expert_names,
    )


def write_json_file(config: CoralRuntimeConfig, path: str | pathlib.Path) -> pathlib.Path:
    """Write a runtime config JSON next to the structured expert directory."""
    config_path = pathlib.Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return config_path


def structured_deployment_dir(
    root: str | pathlib.Path,
    model_type: str,
    base_config: str,
    param_modules: coral_param_filters.CoralParamSpecInput,
) -> pathlib.Path:
    """Directory that groups one model/base/module combination."""
    modules = canonical_module_values(param_modules)
    return pathlib.Path(root) / _required_model_type(model_type) / base_config / module_dir_name(modules)


def structured_experts_dir(
    root: str | pathlib.Path,
    model_type: str,
    base_config: str,
    param_modules: coral_param_filters.CoralParamSpecInput,
) -> pathlib.Path:
    return structured_deployment_dir(root, model_type, base_config, param_modules) / "experts"


def structured_runtime_config_path(
    root: str | pathlib.Path,
    model_type: str,
    base_config: str,
    param_modules: coral_param_filters.CoralParamSpecInput,
) -> pathlib.Path:
    return structured_deployment_dir(root, model_type, base_config, param_modules) / RUNTIME_CONFIG_FILE


def module_dir_name(param_modules: coral_param_filters.CoralParamSpecInput) -> str:
    return "_".join(canonical_module_values(param_modules))


def canonical_module_values(param_modules: coral_param_filters.CoralParamSpecInput) -> tuple[str, ...]:
    spec = coral_param_filters.parse_param_spec(param_modules)
    modules = sorted(spec.modules, key=lambda module: _MODULE_ORDER[module])
    return tuple(module.value for module in modules)


def load_validated_expert_manager(
    config: CoralRuntimeConfig,
    *,
    load_params: bool,
) -> lora_expert_manager.LoraExpertManager:
    """Load configured experts and validate the metadata/weight layout."""
    _validate_base_checkpoint_dir(config)
    _validate_configured_expert_dirs(config)

    manager = lora_expert_manager.LoraExpertManager()
    manager.load_experts(config.experts_dir, load_params=load_params)
    _validate_loaded_experts(manager, config)
    return manager


def validate_hot_switchable_experts(
    manager: lora_expert_manager.LoraExpertManager,
) -> coral_param_filters.CoralParamSpec:
    """Validate that all loaded experts expose one compatible hot-switch shape."""
    expert_names = manager.list_experts()
    if not expert_names:
        raise ValueError("No CORAL experts are loaded.")

    specs = {manager.get_expert(name).metadata.param_spec() for name in expert_names}
    if len(specs) != 1:
        combinations = sorted(spec.values for spec in specs)
        raise ValueError(f"Hot switch requires all experts to use the same parameter modules, got: {combinations}")
    spec = specs.pop()

    expected_paths: set[tuple[Any, ...]] | None = None
    for expert_name in expert_names:
        expert = manager.get_expert(expert_name)
        if expert.lora_params is None:
            raise ValueError(f"Expert '{expert_name}' has no loaded params.")

        _validate_module_coverage(expert_name, expert.lora_params, spec)
        flat_paths = set(flax.traverse_util.flatten_dict(dict(expert.lora_params)))
        if expected_paths is None:
            expected_paths = flat_paths
            continue
        if flat_paths != expected_paths:
            missing = sorted(_format_path(path) for path in expected_paths - flat_paths)
            extra = sorted(_format_path(path) for path in flat_paths - expected_paths)
            raise ValueError(
                f"Expert '{expert_name}' has a different hot-switch parameter set. "
                f"Missing={missing}, extra={extra}"
            )
    return spec


def _validate_base_checkpoint_dir(config: CoralRuntimeConfig) -> None:
    if config.base_checkpoint_dir is None:
        return
    path = pathlib.Path(config.base_checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Configured CORAL base checkpoint does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Configured CORAL base checkpoint path is not a directory: {path}")


def _validate_configured_expert_dirs(config: CoralRuntimeConfig) -> None:
    root = pathlib.Path(config.experts_dir)
    if not root.exists():
        raise FileNotFoundError(f"Configured CORAL experts directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Configured CORAL experts path is not a directory: {root}")

    for expert_name in config.expert_names or ():
        expert_dir = root / expert_name
        if not expert_dir.is_dir():
            raise FileNotFoundError(f"Configured CORAL expert directory does not exist: {expert_dir}")
        metadata_path = expert_dir / lora_expert_manager.EXPERT_METADATA_FILE
        params_dir = expert_dir / lora_expert_manager.LORA_PARAMS_DIR
        if not metadata_path.exists():
            raise FileNotFoundError(f"Configured CORAL expert is missing metadata: {metadata_path}")
        if not params_dir.is_dir():
            raise FileNotFoundError(f"Configured CORAL expert is missing params directory: {params_dir}")


def _validate_loaded_experts(
    manager: lora_expert_manager.LoraExpertManager,
    config: CoralRuntimeConfig,
) -> None:
    loaded = set(manager.list_experts())
    configured = set(config.expert_names or loaded)
    missing = sorted(configured - loaded)
    if missing:
        raise ValueError(f"Configured CORAL experts were not loaded from {config.experts_dir}: {missing}")

    extra = sorted(loaded - configured) if config.expert_names is not None else []
    if extra:
        raise ValueError(f"Unexpected CORAL experts in configured directory: {extra}")

    default_expert = config.default_expert
    if default_expert is not None and default_expert not in loaded:
        raise ValueError(f"Configured default_expert '{default_expert}' is not loaded. Available: {sorted(loaded)}")

    for expert_name in sorted(loaded):
        metadata = manager.get_expert(expert_name).metadata
        if metadata.base_config != config.base_config:
            raise ValueError(
                f"Expert '{expert_name}' was exported for base_config='{metadata.base_config}', "
                f"but runtime base_config='{config.base_config}'."
            )
        spec = metadata.param_spec()
        if config.model_type is not None:
            if metadata.model_type is None:
                raise ValueError(
                    f"Expert '{expert_name}' is missing model_type metadata; re-export it or add model_type."
                )
            if metadata.model_type != config.model_type:
                raise ValueError(
                    f"Expert '{expert_name}' has model_type='{metadata.model_type}', "
                    f"but runtime config expects '{config.model_type}'."
                )
        if config.param_modules is not None:
            expected = coral_param_filters.CoralParamSpec.from_modules(config.param_modules)
            if spec != expected:
                raise ValueError(
                    f"Expert '{expert_name}' has param_modules={spec.values}, "
                    f"but runtime config expects {expected.values}."
                )


def _validate_module_coverage(
    expert_name: str,
    params: Mapping[str, Any],
    spec: coral_param_filters.CoralParamSpec,
) -> None:
    counts = coral_param_filters.count_module_leaves(dict(params), spec)
    missing = sorted(module.value for module, count in counts.items() if count == 0)
    if missing:
        raise ValueError(f"Expert '{expert_name}' is missing hot-switch params for modules: {missing}")


def _required_str(data: Mapping[str, Any], key: str) -> str:
    if key not in data:
        raise ValueError(f"CORAL runtime config is missing required field '{key}'.")
    return _non_empty_str(data[key], key)


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    return None if value is None else _non_empty_str(value, key)


def _optional_model_type(value: Any) -> str | None:
    if value is None:
        return None
    return _required_model_type(value)


def _required_model_type(value: Any) -> str:
    model_type = _non_empty_str(value, "model_type")
    valid = {"pi0", "pi05", "pi0_fast"}
    if model_type not in valid:
        raise ValueError(f"CORAL model_type must be one of {sorted(valid)}, got '{model_type}'.")
    return model_type


def _non_empty_str(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"CORAL runtime config field '{label}' must be non-empty.")
    return text


def _format_path(path: Sequence[Any]) -> str:
    return "/".join(str(part) for part in path)
