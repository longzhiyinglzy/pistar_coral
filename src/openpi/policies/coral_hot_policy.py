"""No-recompile CORAL policy using explicit NNX state inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import threading
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import lora_expert_manager
from openpi.shared import array_typing as at


class CoralHotPolicy(_base_policy.BasePolicy):
    """CORAL policy that switches experts without rebuilding the JIT function."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        expert_states: Mapping[str, nnx.State] | None = None,
        expert_metadata: Mapping[str, lora_expert_manager.LoraExpertMetadata],
        active_state: nnx.State | None = None,
        expert_patches: Mapping[str, Mapping[str, Any]] | None = None,
        default_expert: str,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        expert_transforms: Mapping[str, Sequence[_transforms.DataTransformFn]] | None = None,
        expert_output_transforms: Mapping[str, Sequence[_transforms.DataTransformFn]] | None = None,
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._graphdef, _ = nnx.split(model)
        self._expert_states = dict(expert_states or {})
        self._expert_patches = dict(expert_patches or {})
        self._expert_names = set(self._expert_states) or set(self._expert_patches)
        if default_expert not in self._expert_names:
            raise KeyError(f"Unknown default expert '{default_expert}'.")
        if self._expert_patches and active_state is None:
            raise ValueError("active_state is required when expert_patches are provided.")
        if self._expert_states and active_state is not None:
            raise ValueError("Use either expert_states or active_state/expert_patches, not both.")

        self._expert_metadata = dict(expert_metadata)
        self._active_expert = default_expert
        self._active_state = active_state if active_state is not None else self._expert_states[default_expert]
        if self._expert_patches:
            self._active_state.replace_by_pure_dict(self._expert_patches[default_expert])
        self._expert_input_transforms = self._compose_expert_transforms(expert_transforms, transforms)
        self._expert_output_transforms = self._compose_expert_transforms(expert_output_transforms, output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._rng = rng or jax.random.key(0)
        self._lock = threading.RLock()

        def sample_with_state(state: nnx.State, rng_key: at.KeyArrayLike, observation: _model.Observation):
            module = nnx.merge(self._graphdef, state)
            return module.sample_actions(rng_key, observation, **self._sample_kwargs)

        self._sample_actions = jax.jit(sample_with_state)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        if noise is not None:
            raise NotImplementedError("CoralHotPolicy does not support per-call noise yet.")

        with self._lock:
            requested_expert = self._requested_expert(obs)
            switch_ms = 0.0
            if requested_expert is not None:
                start = time.monotonic()
                self.switch_expert(requested_expert)
                switch_ms = (time.monotonic() - start) * 1000

            inputs = self._prepare_obs(obs)
            inputs = self._expert_input_transforms[self._active_expert](inputs)
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng = jax.random.split(self._rng)
            observation = _model.Observation.from_dict(inputs)

            start = time.monotonic()
            actions = self._sample_actions(self._active_state, sample_rng, observation)
            model_ms = (time.monotonic() - start) * 1000

            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            outputs = self._expert_output_transforms[self._active_expert](outputs)
            outputs["policy_timing"] = {"infer_ms": model_ms}
            outputs["coral"] = {
                "active_expert": self._active_expert,
                "active_prompt": self._active_prompt(),
                "switch_requested": requested_expert,
                "switch_ms": switch_ms,
                "switch_mode": "hot",
            }
            return outputs

    def switch_expert(self, expert_name: str) -> None:
        with self._lock:
            if expert_name == self._active_expert:
                return
            if expert_name not in self._expert_names:
                available = ", ".join(self.list_experts()) or "<none>"
                raise KeyError(f"Unknown LoRA expert '{expert_name}'. Available experts: {available}")
            if self._expert_patches:
                self._active_state.replace_by_pure_dict(self._expert_patches[expert_name])
            else:
                self._active_state = self._expert_states[expert_name]
            self._active_expert = expert_name

    def list_experts(self) -> list[str]:
        return sorted(self._expert_names)

    def active_expert(self) -> str | None:
        return self._active_expert

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _prepare_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        inputs = jax.tree.map(lambda x: x, obs)
        # Server-side expert prompt wins over a stale client prompt.
        inputs["prompt"] = self._active_prompt()
        inputs.pop("tokenized_prompt", None)
        inputs.pop("tokenized_prompt_mask", None)
        return inputs

    def _active_prompt(self) -> str:
        return self._expert_metadata[self._active_expert].task_prompt

    def _compose_expert_transforms(
        self,
        expert_transforms: Mapping[str, Sequence[_transforms.DataTransformFn]] | None,
        fallback: Sequence[_transforms.DataTransformFn],
    ) -> dict[str, _transforms.DataTransformFn]:
        if expert_transforms is None:
            return {name: _transforms.compose(fallback) for name in self._expert_names}

        missing = self._expert_names - set(expert_transforms)
        if missing:
            raise KeyError(f"Missing CORAL transforms for experts: {sorted(missing)}")
        return {name: _transforms.compose(expert_transforms[name]) for name in self._expert_names}

    @staticmethod
    def _requested_expert(obs: dict[str, Any]) -> str | None:
        candidates = [value for value in (obs.get("expert"), obs.get("switch_expert")) if value is not None]
        unique = {str(value) for value in candidates}
        if len(unique) > 1:
            raise ValueError(f"Conflicting expert requests: {sorted(unique)}")
        return None if not unique else unique.pop()
