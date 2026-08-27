"""Registry and metadata handling for CORAL-style LoRA experts.

This module intentionally does not load or mutate model parameters yet. It
defines the stable file format and runtime interface used by the later hot
switching implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any

import orbax.checkpoint as ocp

from openpi.policies import coral_param_filters


EXPERT_METADATA_FILE = "expert.json"
LORA_PARAMS_DIR = "lora_params"
ASSETS_DIR = "assets"


@dataclasses.dataclass(frozen=True)
class LoraExpertMetadata:
    """Metadata that identifies a single task LoRA expert."""

    name: str
    base_config: str
    task_prompt: str
    robot: str
    lora_rank: int
    created_from_checkpoint: str | None = None
    model_type: str | None = None
    param_mode: str | None = None
    param_modules: tuple[str, ...] | None = None
    base_checkpoint: str | None = None
    exported_param_count: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoraExpertMetadata":
        required = ("name", "base_config", "task_prompt", "robot", "lora_rank")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Expert metadata is missing required fields: {missing}")

        name = str(data["name"]).strip()
        if not name:
            raise ValueError("Expert metadata field 'name' must be non-empty.")
        base_config = str(data["base_config"]).strip()
        task_prompt = str(data["task_prompt"]).strip()
        robot = str(data["robot"]).strip()
        lora_rank = int(data["lora_rank"])
        if not base_config:
            raise ValueError("Expert metadata field 'base_config' must be non-empty.")
        if not task_prompt:
            raise ValueError("Expert metadata field 'task_prompt' must be non-empty.")
        if not robot:
            raise ValueError("Expert metadata field 'robot' must be non-empty.")
        if lora_rank <= 0:
            raise ValueError("Expert metadata field 'lora_rank' must be positive.")

        raw_modules = data.get("param_modules")
        if raw_modules is not None and (isinstance(raw_modules, str) or not isinstance(raw_modules, list)):
            raise ValueError("Expert metadata field 'param_modules' must be a list of module names.")
        model_type = None if data.get("model_type") is None else str(data["model_type"]).strip()
        if model_type == "":
            raise ValueError("Expert metadata field 'model_type' must be non-empty when provided.")

        metadata = cls(
            name=name,
            base_config=base_config,
            task_prompt=task_prompt,
            robot=robot,
            lora_rank=lora_rank,
            created_from_checkpoint=(
                None if data.get("created_from_checkpoint") is None else str(data["created_from_checkpoint"])
            ),
            model_type=model_type,
            param_mode=(
                str(data["param_mode"])
                if data.get("param_mode") is not None
                else ("lora_only" if raw_modules is None else None)
            ),
            param_modules=None if raw_modules is None else tuple(str(module) for module in raw_modules),
            base_checkpoint=None if data.get("base_checkpoint") is None else str(data["base_checkpoint"]),
            exported_param_count=(
                None if data.get("exported_param_count") is None else int(data["exported_param_count"])
            ),
        )
        metadata.param_spec()
        return metadata

    def param_spec(self) -> coral_param_filters.CoralParamSpec:
        """Return the validated module spec for new or legacy metadata."""
        return coral_param_filters.parse_metadata_param_spec(
            param_modules=self.param_modules,
            param_mode=self.param_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LoraExpert:
    """A registered expert and its on-disk layout."""

    metadata: LoraExpertMetadata
    path: pathlib.Path
    lora_params: Any | None = None
    norm_stats: Any | None = None

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def lora_params_dir(self) -> pathlib.Path:
        return self.path / LORA_PARAMS_DIR

    @property
    def assets_dir(self) -> pathlib.Path:
        return self.path / ASSETS_DIR


class LoraExpertManager:
    """Keeps track of available LoRA experts and the currently selected one."""

    def __init__(self) -> None:
        self._experts: dict[str, LoraExpert] = {}
        self._active_expert: str | None = None

    def load_experts(self, experts_dir: str | pathlib.Path, *, load_params: bool = False) -> None:
        """Load expert metadata from subdirectories of ``experts_dir``."""
        root = pathlib.Path(experts_dir)
        if not root.exists():
            raise FileNotFoundError(f"Experts directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Experts path is not a directory: {root}")

        experts: dict[str, LoraExpert] = {}
        for expert_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            metadata_path = expert_dir / EXPERT_METADATA_FILE
            if not metadata_path.exists():
                continue
            expert = self._load_expert(expert_dir, metadata_path, load_params=load_params)
            if expert.name in experts:
                raise ValueError(f"Duplicate expert name '{expert.name}' in {root}")
            experts[expert.name] = expert

        if not experts:
            raise ValueError(f"No experts with {EXPERT_METADATA_FILE} found in {root}")

        self._experts = experts
        if self._active_expert not in self._experts:
            self._active_expert = None

    def list_experts(self) -> list[str]:
        return sorted(self._experts)

    def active_expert(self) -> str | None:
        return self._active_expert

    def switch(self, expert_name: str) -> None:
        if expert_name not in self._experts:
            available = ", ".join(self.list_experts()) or "<none>"
            raise KeyError(f"Unknown LoRA expert '{expert_name}'. Available experts: {available}")
        self._active_expert = expert_name

    def get_active_lora_params(self) -> Any | None:
        expert = self.get_active_expert()
        return None if expert is None else expert.lora_params

    def get_active_metadata(self) -> LoraExpertMetadata | None:
        expert = self.get_active_expert()
        return None if expert is None else expert.metadata

    def get_active_expert(self) -> LoraExpert | None:
        if self._active_expert is None:
            return None
        return self._experts[self._active_expert]

    def get_expert(self, expert_name: str) -> LoraExpert:
        if expert_name not in self._experts:
            available = ", ".join(self.list_experts()) or "<none>"
            raise KeyError(f"Unknown LoRA expert '{expert_name}'. Available experts: {available}")
        return self._experts[expert_name]

    def _load_expert(self, expert_dir: pathlib.Path, metadata_path: pathlib.Path, *, load_params: bool) -> LoraExpert:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = LoraExpertMetadata.from_dict(json.load(f))
        lora_params = _load_lora_params(expert_dir / LORA_PARAMS_DIR) if load_params else None
        return LoraExpert(metadata=metadata, path=expert_dir, lora_params=lora_params)


def load_expert_params(expert: LoraExpert) -> Any:
    """Load one expert's hot-switch params without retaining them in the manager."""
    return _load_lora_params(expert.lora_params_dir)


def write_expert_metadata(output_dir: str | pathlib.Path, metadata: LoraExpertMetadata) -> pathlib.Path:
    """Create an expert directory skeleton and write ``expert.json``."""
    metadata.param_spec()
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / LORA_PARAMS_DIR).mkdir(exist_ok=True)
    (output_path / ASSETS_DIR).mkdir(exist_ok=True)

    metadata_path = output_path / EXPERT_METADATA_FILE
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return metadata_path


def _load_lora_params(params_dir: pathlib.Path) -> Any:
    if not params_dir.exists():
        raise FileNotFoundError(f"LoRA expert params directory does not exist: {params_dir}")
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(params_dir)
    if "params" not in restored:
        raise ValueError(f"LoRA expert checkpoint is missing top-level 'params': {params_dir}")
    return restored["params"]
