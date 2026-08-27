from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().rstrip(".\u3002!\uff01?\uff1f").split())


@dataclasses.dataclass(frozen=True)
class ExpertEndpoint:
    name: str
    host: str
    port: int
    prompts: tuple[str, ...]
    keywords: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExpertEndpoint:
        allowed = {"name", "host", "port", "prompts", "keywords"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown expert fields for {value.get('name', '<unnamed>')}: {sorted(unknown)}")

        name = str(value.get("name", "")).strip()
        host = str(value.get("host", "")).strip()
        port = int(value.get("port", 0))
        prompts = tuple(str(item).strip() for item in value.get("prompts", ()) if str(item).strip())
        keywords = tuple(str(item).strip() for item in value.get("keywords", ()) if str(item).strip())
        if not name:
            raise ValueError("Each expert needs a non-empty name.")
        if not host:
            raise ValueError(f"Expert '{name}' needs a host.")
        if not 1 <= port <= 65535:
            raise ValueError(f"Expert '{name}' has invalid port {port}.")
        if not prompts and not keywords:
            raise ValueError(f"Expert '{name}' needs at least one prompt or keyword.")
        return cls(name=name, host=host, port=port, prompts=prompts, keywords=keywords)


@dataclasses.dataclass(frozen=True)
class RouterConfig:
    experts: tuple[ExpertEndpoint, ...]
    default_expert: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> RouterConfig:
        config_path = Path(path).expanduser()
        with config_path.open(encoding="utf-8") as config_file:
            raw = json.load(config_file)
        if not isinstance(raw, dict):
            raise ValueError("Router config must be a JSON object.")
        unknown = set(raw) - {"experts", "default_expert"}
        if unknown:
            raise ValueError(f"Unknown router config fields: {sorted(unknown)}")
        raw_experts = raw.get("experts")
        if not isinstance(raw_experts, list) or not raw_experts:
            raise ValueError("Router config needs a non-empty 'experts' list.")
        experts = tuple(ExpertEndpoint.from_dict(item) for item in raw_experts)
        names = [expert.name for expert in experts]
        if len(names) != len(set(names)):
            raise ValueError(f"Expert names must be unique: {names}")

        default_expert = raw.get("default_expert")
        if default_expert is not None:
            default_expert = str(default_expert).strip()
            if default_expert not in names:
                raise ValueError(f"default_expert '{default_expert}' is not present in experts.")
        return cls(experts=experts, default_expert=default_expert)


class TaskRouter:
    """Resolve an explicit task id or language prompt to one expert."""

    def __init__(self, config: RouterConfig):
        self._config = config
        self._by_name = {expert.name: expert for expert in config.experts}
        self._exact_prompts: dict[str, ExpertEndpoint] = {}
        for expert in config.experts:
            for prompt in expert.prompts:
                normalized = _normalize_text(prompt)
                previous = self._exact_prompts.get(normalized)
                if previous is not None and previous.name != expert.name:
                    raise ValueError(f"Prompt {prompt!r} is assigned to both '{previous.name}' and '{expert.name}'.")
                self._exact_prompts[normalized] = expert

    def select(self, observation: Mapping[str, Any]) -> tuple[ExpertEndpoint, str]:
        explicit_task = observation.get("coral_task")
        if explicit_task is not None:
            task_name = str(explicit_task).strip()
            if task_name not in self._by_name:
                raise ValueError(f"Unknown coral_task {task_name!r}; expected one of {sorted(self._by_name)}.")
            return self._by_name[task_name], "explicit"

        prompt = observation.get("prompt")
        if prompt is None:
            return self._select_default("request has no prompt or coral_task")
        normalized_prompt = _normalize_text(str(prompt))
        if not normalized_prompt:
            return self._select_default("request has an empty prompt")
        if exact := self._exact_prompts.get(normalized_prompt):
            return exact, "exact_prompt"

        matches: list[tuple[int, ExpertEndpoint, str]] = []
        for expert in self._config.experts:
            for keyword in expert.keywords:
                normalized_keyword = _normalize_text(keyword)
                if normalized_keyword and normalized_keyword in normalized_prompt:
                    matches.append((len(normalized_keyword), expert, keyword))
        if matches:
            best_length = max(length for length, _, _ in matches)
            best = {(expert.name, keyword) for length, expert, keyword in matches if length == best_length}
            best_names = {name for name, _ in best}
            if len(best_names) > 1:
                raise ValueError(
                    f"Ambiguous prompt {prompt!r}; longest keyword matches select experts {sorted(best_names)}. "
                    "Send coral_task explicitly."
                )
            expert_name, keyword = next(iter(best))
            return self._by_name[expert_name], f"keyword:{keyword}"
        return self._select_default(f"prompt {prompt!r} did not match any route")

    def _select_default(self, reason: str) -> tuple[ExpertEndpoint, str]:
        if self._config.default_expert is None:
            raise ValueError(
                f"Cannot route request because {reason}. Send coral_task explicitly or add a prompt/keyword route."
            )
        return self._by_name[self._config.default_expert], "default"
