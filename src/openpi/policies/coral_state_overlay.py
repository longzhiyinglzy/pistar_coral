"""NNX state overlay utilities for CORAL hot switching."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import flax.nnx as nnx
import flax.traverse_util
import jax.numpy as jnp

from openpi.policies import coral_param_filters


class CoralStateOverlay:
    """Build full active NNX states by overlaying expert params on a base state."""

    def __init__(
        self,
        base_state: nnx.State,
        *,
        param_spec: coral_param_filters.CoralParamSpecInput | None = None,
        param_mode: str | None = None,
    ) -> None:
        if param_spec is not None and param_mode is not None:
            raise ValueError("Use either param_spec or legacy param_mode, not both.")
        if param_spec is None and param_mode is None:
            raise ValueError("A CORAL parameter spec is required.")

        selected_spec = param_spec if param_spec is not None else param_mode
        assert selected_spec is not None
        self._base_state = base_state
        self._param_spec = coral_param_filters.parse_param_spec(selected_spec)
        self._base_pure = base_state.to_pure_dict()
        self._flat_base = flax.traverse_util.flatten_dict(self._base_pure)

        expert_filter = coral_param_filters.get_coral_expert_filter(self._param_spec)
        self._allowed_paths = set(base_state.filter(nnx.All(nnx.Param, expert_filter)).flat_state())

    def apply(self, expert_params: Mapping[str, Any]) -> nnx.State:
        """Return a full model state with expert leaves overlaid."""
        flat_active = dict(self._flat_base)
        flat_active.update(flax.traverse_util.flatten_dict(self.build_patch(expert_params)))
        active_state = copy.deepcopy(self._base_state)
        active_state.replace_by_pure_dict(flax.traverse_util.unflatten_dict(flat_active))
        return active_state

    def build_patch(self, expert_params: Mapping[str, Any]) -> dict[str, Any]:
        """Return only the validated expert leaves, converted to match the base state."""
        flat_expert = flax.traverse_util.flatten_dict(dict(expert_params))
        if not flat_expert:
            raise ValueError("Expert parameter tree is empty.")

        flat_patch = {}
        for path, expert_value in flat_expert.items():
            self._validate_path(path, expert_value)
            flat_patch[path] = self._convert_like(expert_value, self._flat_base[path])
        return flax.traverse_util.unflatten_dict(flat_patch)

    def apply_in_place(self, state: nnx.State, expert_params: Mapping[str, Any]) -> nnx.State:
        """Overlay expert leaves onto an existing state without copying the full model."""
        state.replace_by_pure_dict(self.build_patch(expert_params))
        return state

    def validate_same_paths(self, experts: Mapping[str, Mapping[str, Any]]) -> None:
        """Require all experts to expose the same JIT input structure."""
        expected_paths: set[tuple[Any, ...]] | None = None
        for expert_name, expert_params in experts.items():
            paths = set(flax.traverse_util.flatten_dict(dict(expert_params)))
            if expected_paths is None:
                expected_paths = paths
                continue
            if paths != expected_paths:
                missing = sorted("/".join(str(part) for part in path) for path in expected_paths - paths)
                extra = sorted("/".join(str(part) for part in path) for path in paths - expected_paths)
                raise ValueError(
                    f"Expert '{expert_name}' has a different parameter set. Missing={missing}, extra={extra}"
                )

    def _validate_path(self, path: tuple[Any, ...], expert_value: Any) -> None:
        if path not in self._flat_base:
            joined = "/".join(str(part) for part in path)
            raise KeyError(f"Expert parameter path is not present in the base model: {joined}")
        if path not in self._allowed_paths:
            joined = "/".join(str(part) for part in path)
            raise ValueError(f"Expert parameter path is not allowed by modules '{self._param_spec.label}': {joined}")

        base_value = self._flat_base[path]
        if getattr(base_value, "shape", None) != getattr(expert_value, "shape", None):
            joined = "/".join(str(part) for part in path)
            raise ValueError(
                f"Expert parameter shape mismatch at {joined}: "
                f"base={getattr(base_value, 'shape', None)}, expert={getattr(expert_value, 'shape', None)}"
            )

    @staticmethod
    def _convert_like(value: Any, reference: Any) -> Any:
        if hasattr(reference, "dtype") and hasattr(value, "dtype"):
            return jnp.asarray(value, dtype=reference.dtype)
        return value
