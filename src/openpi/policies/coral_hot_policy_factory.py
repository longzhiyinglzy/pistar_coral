"""Factory for no-recompile CORAL hot-switch policies."""

from __future__ import annotations

import gc
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import flax.traverse_util
import jax

from openpi.policies import coral_param_filters
from openpi.policies import coral_policy_factory
from openpi.policies import coral_runtime_config
from openpi.policies import coral_state_overlay
from openpi.policies import lora_expert_manager
from openpi.shared import normalize as _normalize
import openpi.transforms as transforms

logger = logging.getLogger(__name__)


def create_coral_hot_policy(
    train_config: Any,
    *,
    base_params_dir: str | pathlib.Path | None,
    expert_manager: lora_expert_manager.LoraExpertManager,
    default_expert: str,
    sample_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Create a CORAL policy whose JIT function accepts explicit active state."""
    from openpi.policies import coral_hot_policy

    base_params = coral_policy_factory._load_base_params(train_config, base_params_dir)  # noqa: SLF001
    base_params = jax.tree.map(coral_policy_factory._to_bfloat16, base_params)  # noqa: SLF001
    model = train_config.model.load(base_params)
    _, base_state = nnx.split(model)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    param_spec = _single_param_spec(expert_manager)
    overlay = coral_state_overlay.CoralStateOverlay(base_state, param_spec=param_spec)

    logger.info("Preparing CORAL expert patches for %d experts...", len(expert_manager.list_experts()))
    expert_patches = _build_expert_patches(expert_manager, overlay, param_spec)
    expert_metadata = {name: expert_manager.get_expert(name).metadata for name in expert_manager.list_experts()}
    expert_norm_stats = _expert_norm_stats(expert_manager, data_config.norm_stats)
    expert_input_transforms = {}
    expert_output_transforms = {}
    for name, norm_stats in expert_norm_stats.items():
        # Norm is the only per-expert transform; the rest stays shared with the base config.
        expert_input_transforms[name] = [
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
        expert_output_transforms[name] = [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]

    return coral_hot_policy.CoralHotPolicy(
        model,
        active_state=base_state,
        expert_patches=expert_patches,
        expert_metadata=expert_metadata,
        default_expert=default_expert,
        expert_transforms=expert_input_transforms,
        expert_output_transforms=expert_output_transforms,
        sample_kwargs=sample_kwargs,
        metadata={
            **(train_config.policy_metadata or {}),
            "coral_base_config": train_config.name,
            "coral_switch_mode": "hot",
            "coral_param_modules": param_spec.values,
            "coral_norm_stats": "per_expert",
        },
    )


def _single_param_spec(
    expert_manager: lora_expert_manager.LoraExpertManager,
) -> coral_param_filters.CoralParamSpec:
    specs = {expert_manager.get_expert(name).metadata.param_spec() for name in expert_manager.list_experts()}
    if len(specs) != 1:
        combinations = sorted(spec.values for spec in specs)
        raise ValueError(
            f"Hot switch mode requires all experts to use the same parameter modules, got: {combinations}"
        )
    return specs.pop()


def _build_expert_patches(
    expert_manager: lora_expert_manager.LoraExpertManager,
    overlay: coral_state_overlay.CoralStateOverlay,
    param_spec: coral_param_filters.CoralParamSpec,
) -> dict[str, Any]:
    patches = {}
    expected_paths: set[tuple[Any, ...]] | None = None
    for name in expert_manager.list_experts():
        expert = expert_manager.get_expert(name)
        params = expert.lora_params if expert.lora_params is not None else lora_expert_manager.load_expert_params(expert)
        _validate_expert_params(name, params, param_spec, expected_paths)
        paths = set(flax.traverse_util.flatten_dict(dict(params)))
        if expected_paths is None:
            expected_paths = paths
        patches[name] = overlay.build_patch(params)
        if expert.lora_params is None:
            del params
            gc.collect()
    return patches


def _validate_expert_params(
    expert_name: str,
    params: Any,
    param_spec: coral_param_filters.CoralParamSpec,
    expected_paths: set[tuple[Any, ...]] | None,
) -> None:
    counts = coral_param_filters.count_module_leaves(dict(params), param_spec)
    missing = sorted(module.value for module, count in counts.items() if count == 0)
    if missing:
        raise ValueError(f"Expert '{expert_name}' is missing hot-switch params for modules: {missing}")

    paths = set(flax.traverse_util.flatten_dict(dict(params)))
    if expected_paths is not None and paths != expected_paths:
        missing_paths = sorted("/".join(str(part) for part in path) for path in expected_paths - paths)
        extra_paths = sorted("/".join(str(part) for part in path) for path in paths - expected_paths)
        raise ValueError(
            f"Expert '{expert_name}' has a different hot-switch parameter set. "
            f"Missing={missing_paths}, extra={extra_paths}"
        )


def _expert_norm_stats(
    expert_manager: lora_expert_manager.LoraExpertManager,
    fallback_norm_stats: Any,
) -> dict[str, Any]:
    result = {}
    for name in expert_manager.list_experts():
        expert = expert_manager.get_expert(name)
        norm_stats_path = expert.assets_dir / "norm_stats.json"
        if norm_stats_path.exists():
            logger.info("Loading CORAL norm stats for expert '%s' from %s", name, norm_stats_path)
            result[name] = _normalize.load(expert.assets_dir)
        elif fallback_norm_stats is not None:
            logger.info("Using base norm stats for expert '%s'; no %s found", name, norm_stats_path)
            result[name] = fallback_norm_stats
        else:
            raise FileNotFoundError(
                f"Expert '{name}' has no norm stats at {norm_stats_path}, and the base config has no fallback stats."
            )
    return result
