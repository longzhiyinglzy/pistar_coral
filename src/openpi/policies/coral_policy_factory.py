"""Build slow-switch CORAL policies by merging one expert into a base model."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import logging
import pathlib
from typing import Any

import flax.traverse_util
import jax
import jax.numpy as jnp

import openpi.shared.download as download
import openpi.policies.policy as _policy
import openpi.transforms as transforms

logger = logging.getLogger(__name__)


def create_coral_expert_policy(
    train_config: Any,
    *,
    base_params_dir: str | pathlib.Path | None,
    expert_params: Mapping[str, Any],
    default_prompt: str | None = None,
    sample_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Create a regular OpenPI Policy with one exported CORAL expert merged in.

    This is the correctness-first path. Switching experts rebuilds the wrapped
    Policy and may recompile JAX, but it reuses OpenPI's standard transforms.
    """
    base_params = _load_base_params(train_config, base_params_dir)
    merged_params = merge_expert_params(base_params, expert_params)
    merged_params = jax.tree.map(_to_bfloat16, merged_params)

    logger.info("Loading CORAL expert model...")
    model = train_config.model.load(merged_params)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.norm_stats is None:
        raise ValueError(
            "Norm stats are required for CORAL inference. Run compute_norm_stats for the CORAL config first, "
            f"or set an assets path in config '{train_config.name}'."
        )

    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata={
            **(train_config.policy_metadata or {}),
            "coral_base_config": train_config.name,
        },
    )


def merge_expert_params(base_params: Mapping[str, Any], expert_params: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay exported expert params onto a full base LoRA model parameter tree."""
    flat_base = flax.traverse_util.flatten_dict(dict(base_params))
    flat_expert = flax.traverse_util.flatten_dict(dict(expert_params))
    if not flat_expert:
        raise ValueError("Expert parameter tree is empty.")

    merged = dict(flat_base)
    for path, expert_value in flat_expert.items():
        if path not in flat_base:
            joined = "/".join(str(part) for part in path)
            raise KeyError(f"Expert parameter path is not present in the base LoRA model: {joined}")
        base_value = flat_base[path]
        if getattr(base_value, "shape", None) != getattr(expert_value, "shape", None):
            joined = "/".join(str(part) for part in path)
            raise ValueError(
                f"Expert parameter shape mismatch at {joined}: "
                f"base={getattr(base_value, 'shape', None)}, expert={getattr(expert_value, 'shape', None)}"
            )
        merged[path] = expert_value

    return flax.traverse_util.unflatten_dict(merged)


def _load_base_params(train_config: Any, base_params_dir: str | pathlib.Path | None) -> dict[str, Any]:
    train_config = _with_base_params_override(train_config, base_params_dir)
    import flax.nnx as nnx

    abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    if hasattr(abstract_model, "to_pure_dict"):
        base_shape = abstract_model.to_pure_dict()
    else:
        _, state = nnx.split(abstract_model)
        base_shape = state.to_pure_dict()
    return train_config.weight_loader.load(base_shape)


def _with_base_params_override(
    train_config: Any, base_params_dir: str | pathlib.Path | None
) -> Any:
    if base_params_dir is None:
        return train_config
    from openpi.training import weight_loaders

    params_path = _resolve_params_dir(base_params_dir)
    return dataclasses.replace(train_config, weight_loader=weight_loaders.CheckpointWeightLoader(str(params_path)))


def _resolve_params_dir(path: str | pathlib.Path) -> pathlib.Path:
    local_path = pathlib.Path(download.maybe_download(str(path))).resolve()
    nested_params = local_path / "params"
    return nested_params if nested_params.exists() else local_path


def _to_bfloat16(value: Any) -> Any:
    if isinstance(value, jax.ShapeDtypeStruct):
        return value
    if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
        return jnp.asarray(value, dtype=jnp.bfloat16)
    return value
