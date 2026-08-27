import flax.nnx as nnx
import numpy as np
import pytest

from openpi.policies import coral_param_filters


def test_lora_only_filter_selects_only_lora_params():
    params = {
        "PaliGemma": {
            "llm": {
                "layer": {
                    "lora_a": np.ones((1,)),
                    "w": np.ones((1,)),
                }
            }
        },
        "action_out_proj": {"kernel": np.ones((1,))},
    }

    expert_params = coral_param_filters.filter_param_dict(
        params,
        coral_param_filters.CoralExpertParamMode.LORA_ONLY,
    )

    assert ("PaliGemma", "llm", "layer", "lora_a") in _flat_keys(expert_params)
    assert ("PaliGemma", "llm", "layer", "w") not in _flat_keys(expert_params)
    assert ("action_out_proj", "kernel") not in _flat_keys(expert_params)


def test_lora_plus_action_head_filter_selects_heads():
    params = {
        "PaliGemma": {"llm": {"layer": {"lora_b": np.ones((1,)), "w": np.ones((1,))}}},
        "state_proj": {"kernel": np.ones((1,))},
        "action_time_mlp_in": {"kernel": np.ones((1,))},
        "action_out_proj": {"kernel": np.ones((1,))},
    }

    expert_params = coral_param_filters.filter_param_dict(
        params,
        coral_param_filters.CoralExpertParamMode.LORA_PLUS_ACTION_HEAD,
    )
    keys = _flat_keys(expert_params)

    assert ("PaliGemma", "llm", "layer", "lora_b") in keys
    assert ("state_proj", "kernel") in keys
    assert ("action_time_mlp_in", "kernel") in keys
    assert ("action_out_proj", "kernel") in keys
    assert ("PaliGemma", "llm", "layer", "w") not in keys


def test_openpi_default_lora_filter_selects_image_tower():
    params = {
        "PaliGemma": {
            "img": {"embedding": {"kernel": np.ones((1,))}},
            "llm": {"layer": {"lora_a": np.ones((1,)), "w": np.ones((1,))}},
        },
        "action_out_proj": {"kernel": np.ones((1,))},
    }

    expert_params = coral_param_filters.filter_param_dict(
        params,
        coral_param_filters.CoralExpertParamMode.OPENPI_DEFAULT_LORA,
    )
    keys = _flat_keys(expert_params)

    assert ("PaliGemma", "img", "embedding", "kernel") in keys
    assert ("PaliGemma", "llm", "layer", "lora_a") in keys
    assert ("action_out_proj", "kernel") in keys
    assert ("PaliGemma", "llm", "layer", "w") not in keys


def test_freeze_filter_is_inverse_of_expert_filter_for_params():
    expert_filter = coral_param_filters.get_coral_expert_filter("lora_only")
    freeze_filter = coral_param_filters.get_coral_freeze_filter("lora_only")
    param = nnx.Param(np.ones((1,)))

    assert expert_filter(("layer", "lora_a"), param)
    assert not freeze_filter(("layer", "lora_a"), param)
    assert freeze_filter(("layer", "w"), param)


def test_composable_filter_selects_full_action_expert():
    params = {
        "PaliGemma": {
            "llm": {
                "attn": {"kernel": np.ones((1,))},
                "attn_1": {"kernel": np.ones((1,))},
                "mlp_1": {"kernel": np.ones((1,))},
            }
        },
        "action_out_proj": {"kernel": np.ones((1,))},
    }
    spec = coral_param_filters.CoralParamSpec.from_modules(
        [
            coral_param_filters.CoralParamModule.ACTION_EXPERT,
            coral_param_filters.CoralParamModule.ACTION_HEAD,
        ]
    )

    keys = _flat_keys(coral_param_filters.filter_param_dict(params, spec))

    assert ("PaliGemma", "llm", "attn_1", "kernel") in keys
    assert ("PaliGemma", "llm", "mlp_1", "kernel") in keys
    assert ("action_out_proj", "kernel") in keys
    assert ("PaliGemma", "llm", "attn", "kernel") not in keys


def test_param_spec_is_order_independent():
    first = coral_param_filters.parse_param_spec("lora,action_head")
    second = coral_param_filters.CoralParamSpec.from_modules(["action_head", "lora"])

    assert first == second
    assert first.values == ("action_head", "lora")


def test_legacy_mode_maps_to_module_spec():
    legacy = coral_param_filters.parse_param_spec("openpi_default_lora")
    modules = coral_param_filters.parse_param_spec("lora,action_head,image_tower")

    assert legacy == modules


def test_param_spec_rejects_empty_and_unknown_modules():
    with pytest.raises(ValueError, match="at least one"):
        coral_param_filters.parse_param_spec("")
    with pytest.raises(ValueError, match="Unknown CORAL parameter module"):
        coral_param_filters.parse_param_spec("unknown")


def test_metadata_spec_rejects_conflicting_legacy_mode():
    with pytest.raises(ValueError, match="Conflicting CORAL metadata"):
        coral_param_filters.parse_metadata_param_spec(
            param_modules=["lora", "action_head"],
            param_mode="lora_only",
        )


def _flat_keys(params):
    import flax.traverse_util

    return set(flax.traverse_util.flatten_dict(params))
