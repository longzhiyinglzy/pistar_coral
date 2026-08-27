import pathlib

import numpy as np
import pytest

from openpi.policies import coral_hot_policy_factory
from openpi.policies import lora_expert_manager
from openpi.shared import normalize as _normalize


def _write_expert(
    root: pathlib.Path,
    name: str,
    *,
    param_mode: str | None = None,
    param_modules: tuple[str, ...] | None = None,
) -> pathlib.Path:
    expert_dir = root / name
    if param_mode is None and param_modules is None:
        param_modules = ("lora",)
    metadata = lora_expert_manager.LoraExpertMetadata(
        name=name,
        base_config="pi0_so101_coral_base",
        task_prompt=f"pick {name}",
        robot="so101",
        lora_rank=16,
        param_mode=param_mode,
        param_modules=param_modules,
    )
    lora_expert_manager.write_expert_metadata(expert_dir, metadata)
    return expert_dir


def _stats(value: float) -> dict[str, _normalize.NormStats]:
    return {
        "state": _normalize.NormStats(
            mean=np.array([value], dtype=np.float32),
            std=np.array([1.0], dtype=np.float32),
            q01=np.array([value - 1.0], dtype=np.float32),
            q99=np.array([value + 1.0], dtype=np.float32),
        )
    }


def test_expert_norm_stats_loads_expert_assets_and_falls_back(tmp_path: pathlib.Path):
    expert_a = _write_expert(tmp_path, "part_a")
    _write_expert(tmp_path, "part_b")
    _normalize.save(expert_a / "assets", _stats(10.0))

    manager = lora_expert_manager.LoraExpertManager()
    manager.load_experts(tmp_path)
    fallback = _stats(20.0)

    norm_stats = coral_hot_policy_factory._expert_norm_stats(manager, fallback)

    assert np.allclose(norm_stats["part_a"]["state"].mean, [10.0])
    assert np.allclose(norm_stats["part_b"]["state"].mean, [20.0])


def test_single_param_spec_accepts_legacy_and_new_equivalent_metadata(tmp_path: pathlib.Path):
    _write_expert(tmp_path, "part_a", param_mode="lora_plus_action_head")
    _write_expert(
        tmp_path,
        "part_b",
        param_mode=None,
        param_modules=("action_head", "lora"),
    )
    manager = lora_expert_manager.LoraExpertManager()
    manager.load_experts(tmp_path)

    spec = coral_hot_policy_factory._single_param_spec(manager)

    assert spec.values == ("action_head", "lora")


def test_single_param_spec_rejects_different_module_sets(tmp_path: pathlib.Path):
    _write_expert(tmp_path, "part_a", param_mode="lora_only")
    _write_expert(
        tmp_path,
        "part_b",
        param_mode=None,
        param_modules=("action_expert", "lora"),
    )
    manager = lora_expert_manager.LoraExpertManager()
    manager.load_experts(tmp_path)

    with pytest.raises(ValueError, match="same parameter modules"):
        coral_hot_policy_factory._single_param_spec(manager)
