import flax.nnx as nnx
import numpy as np
import pytest

from openpi.policies import coral_state_overlay


def test_overlay_replaces_allowed_expert_leaves():
    overlay = coral_state_overlay.CoralStateOverlay(_base_state(), param_mode="lora_only")

    active_state = overlay.apply({"layer": {"lora_a": np.array([3.0], dtype=np.float32)}})
    active = active_state.to_pure_dict()

    np.testing.assert_array_equal(active["layer"]["lora_a"], np.array([3.0], dtype=np.float32))
    np.testing.assert_array_equal(active["layer"]["w"], np.array([1.0], dtype=np.float32))


def test_overlay_rejects_disallowed_path():
    overlay = coral_state_overlay.CoralStateOverlay(_base_state(), param_mode="lora_only")

    with pytest.raises(ValueError, match="not allowed"):
        overlay.apply({"layer": {"w": np.array([3.0], dtype=np.float32)}})


def test_overlay_rejects_shape_mismatch():
    overlay = coral_state_overlay.CoralStateOverlay(_base_state(), param_mode="lora_only")

    with pytest.raises(ValueError, match="shape mismatch"):
        overlay.apply({"layer": {"lora_a": np.array([3.0, 4.0], dtype=np.float32)}})


def test_overlay_accepts_composed_action_expert_spec():
    base_state = nnx.State.from_flat_path(
        {
            ("PaliGemma", "llm", "attn", "kernel"): nnx.Param(np.array([1.0], dtype=np.float32)),
            ("PaliGemma", "llm", "attn_1", "kernel"): nnx.Param(np.array([0.0], dtype=np.float32)),
            ("action_out_proj", "kernel"): nnx.Param(np.array([0.0], dtype=np.float32)),
        }
    )
    overlay = coral_state_overlay.CoralStateOverlay(
        base_state,
        param_spec="action_expert,action_head",
    )

    active = overlay.apply(
        {
            "PaliGemma": {"llm": {"attn_1": {"kernel": np.array([2.0], dtype=np.float32)}}},
            "action_out_proj": {"kernel": np.array([3.0], dtype=np.float32)},
        }
    ).to_pure_dict()

    np.testing.assert_array_equal(active["PaliGemma"]["llm"]["attn_1"]["kernel"], [2.0])
    np.testing.assert_array_equal(active["action_out_proj"]["kernel"], [3.0])
    np.testing.assert_array_equal(active["PaliGemma"]["llm"]["attn"]["kernel"], [1.0])


def _base_state() -> nnx.State:
    return nnx.State.from_flat_path(
        {
            ("layer", "lora_a"): nnx.Param(np.array([0.0], dtype=np.float32)),
            ("layer", "w"): nnx.Param(np.array([1.0], dtype=np.float32)),
        }
    )
