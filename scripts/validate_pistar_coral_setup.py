"""Validate the PiStar-CORAL model contract and a prepared LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util
import jax
import pyarrow.parquet as pq

from openpi.policies import coral_param_filters
from openpi.training import config as training_config


TRAIN_CONFIG = "pi05_star_coral_piper_block1_r0"
INFER_CONFIG = "pi05_star_coral_piper_infer"
EXPERT_MODULES = (
    coral_param_filters.CoralParamModule.LORA,
    coral_param_filters.CoralParamModule.ACTION_HEAD,
    coral_param_filters.CoralParamModule.IMAGE_TOWER,
    coral_param_filters.CoralParamModule.ACTION_EXPERT,
)


def _flat_model_spec(config_name: str) -> dict[tuple[str, ...], object]:
    config = training_config.get_config(config_name)
    model = nnx.eval_shape(config.model.create, jax.random.key(0))
    _, state = nnx.split(model)
    return flax.traverse_util.flatten_dict(state.to_pure_dict())


def validate_model_contract() -> None:
    train = training_config.get_config(TRAIN_CONFIG)
    infer = training_config.get_config(INFER_CONFIG)

    for name, config in ((TRAIN_CONFIG, train), (INFER_CONFIG, infer)):
        model = config.model
        assert model.pi05, f"{name}: pi05 must be enabled"
        assert model.pistar, f"{name}: pistar must be enabled"
        assert model.paligemma_variant == "gemma_2b_lora"
        assert model.action_expert_variant == "gemma_300m_lora"
        assert model.action_horizon == 50
        assert model.discrete_state_input is True
        assert model.max_token_len == 203
        assert config.data.extra_delta_transform is True
        assert config.data.side_image_key == "side_image"

    assert train.ema_decay is None
    assert infer.data.adv_ind_dropout is False
    assert infer.model.adv_guidance_beta == 2.0

    train_spec = _flat_model_spec(TRAIN_CONFIG)
    infer_spec = _flat_model_spec(INFER_CONFIG)
    assert train_spec.keys() == infer_spec.keys(), "train/inference parameter paths differ"
    mismatched = [
        "/".join(path)
        for path in train_spec
        if getattr(train_spec[path], "shape", None) != getattr(infer_spec[path], "shape", None)
    ]
    assert not mismatched, f"train/inference parameter shapes differ: {mismatched[:10]}"

    pure_tree = flax.traverse_util.unflatten_dict(train_spec)
    counts = coral_param_filters.count_module_leaves(pure_tree, EXPERT_MODULES)
    missing = [module.value for module, count in counts.items() if count == 0]
    assert not missing, f"empty CORAL expert modules: {missing}"

    print("[ok] Hybrid model contract")
    print(f"  parameter leaves: {len(train_spec)}")
    for module in sorted(counts, key=lambda item: item.value):
        print(f"  {module.value}: {counts[module]} leaves")


def validate_dataset(dataset: Path, expected_episodes: int | None, scan_labels: bool) -> None:
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    info = json.loads(info_path.read_text())
    episodes = int(info["total_episodes"])
    if expected_episodes is not None and episodes != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, found {episodes}")
    if int(info["fps"]) != 30:
        raise ValueError(f"Expected 30 fps, found {info['fps']}")

    features = info["features"]
    required = {"image", "wrist_image", "side_image", "state", "actions", "adv_ind"}
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(f"Dataset is missing features: {missing}")
    for key in ("image", "wrist_image", "side_image"):
        if list(features[key]["shape"]) != [3, 480, 640]:
            raise ValueError(f"{key} must be [3, 480, 640], got {features[key]['shape']}")
    if list(features["state"]["shape"]) != [7] or list(features["actions"]["shape"]) != [7]:
        raise ValueError("Piper state/actions must both be 7D")

    parquet_files = sorted((dataset / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquet_files) != episodes:
        raise ValueError(f"Metadata says {episodes} episodes but found {len(parquet_files)} parquet files")

    print("[ok] Dataset contract")
    print(f"  episodes: {episodes}, frames: {info['total_frames']}, fps: {info['fps']}")
    print("  cameras: image/wrist_image/side_image = 640x480")

    if scan_labels:
        counts = {"positive": 0, "negative": 0}
        unexpected: set[str] = set()
        for path in parquet_files:
            for value in pq.read_table(path, columns=["adv_ind"])["adv_ind"].to_pylist():
                label = str(value)
                if label in counts:
                    counts[label] += 1
                else:
                    unexpected.add(label)
        if unexpected:
            raise ValueError(f"Unexpected adv_ind labels: {sorted(unexpected)}")
        if not all(counts.values()):
            raise ValueError(f"Both advantage classes are required, got {counts}")
        print(f"  advantage frames: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--scan-labels", action="store_true")
    args = parser.parse_args()

    validate_model_contract()
    if args.dataset is not None:
        validate_dataset(args.dataset.expanduser().resolve(), args.expected_episodes, args.scan_labels)


if __name__ == "__main__":
    main()
