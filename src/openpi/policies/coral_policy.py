"""CORAL-style policy wrapper with explicit LoRA expert selection hooks."""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.policies import lora_expert_manager


class CoralPolicy(_base_policy.BasePolicy):
    """Policy wrapper that reserves the LoRA expert switching interface.

    The first implementation phase only tracks expert selection and delegates
    inference to an optional base policy. Actual JAX/NNX state replacement is
    added in the later hot-switching phase.
    """

    def __init__(
        self,
        *,
        expert_manager: lora_expert_manager.LoraExpertManager,
        base_policy: _base_policy.BasePolicy | None = None,
        policy_factory: Callable[[lora_expert_manager.LoraExpert], _base_policy.BasePolicy] | None = None,
    ) -> None:
        self._expert_manager = expert_manager
        self._base_policy = base_policy
        self._policy_factory = policy_factory
        self._lock = threading.RLock()

    @override
    def infer(self, obs: dict, *, expert: str | None = None) -> dict:  # type: ignore[override]
        with self._lock:
            requested_expert = self._requested_expert(obs, expert)
            if requested_expert is not None:
                self.switch_expert(requested_expert)

            # If the client didn't provide a prompt, use the active expert's task prompt.
            if "prompt" not in obs:
                metadata = self.active_metadata()
                if metadata is not None:
                    obs["prompt"] = metadata.task_prompt

            if self._base_policy is None:
                raise NotImplementedError("CoralPolicy inference requires a base policy in this implementation phase.")

            result = dict(self._base_policy.infer(obs))
            result["coral"] = {
                "active_expert": self.active_expert(),
                "switch_requested": requested_expert,
            }
            return result

    def switch_expert(self, expert_name: str) -> None:
        with self._lock:
            if expert_name == self._expert_manager.active_expert() and self._base_policy is not None:
                return
            self._expert_manager.switch(expert_name)
            if self._policy_factory is not None:
                self._base_policy = self._policy_factory(self._expert_manager.get_expert(expert_name))

    def list_experts(self) -> list[str]:
        return self._expert_manager.list_experts()

    def active_expert(self) -> str | None:
        return self._expert_manager.active_expert()

    def active_metadata(self) -> lora_expert_manager.LoraExpertMetadata | None:
        return self._expert_manager.get_active_metadata()

    def _requested_expert(self, obs: dict[str, Any], explicit_expert: str | None) -> str | None:
        obs_expert = obs.get("expert")
        switch_expert = obs.get("switch_expert")
        candidates = [value for value in (explicit_expert, obs_expert, switch_expert) if value is not None]
        unique = {str(value) for value in candidates}
        if len(unique) > 1:
            raise ValueError(f"Conflicting expert requests: {sorted(unique)}")
        return None if not unique else unique.pop()
