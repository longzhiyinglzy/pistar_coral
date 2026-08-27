from openpi.training import config as training_config


def test_hybrid_train_and_inference_configs_match() -> None:
    train = training_config.get_config("pi05_star_coral_piper_block1_r0")
    infer = training_config.get_config("pi05_star_coral_piper_infer")

    assert train.model.pi05 and train.model.pistar
    assert infer.model.pi05 and infer.model.pistar
    assert train.model.paligemma_variant == infer.model.paligemma_variant == "gemma_2b_lora"
    assert train.model.action_expert_variant == infer.model.action_expert_variant == "gemma_300m_lora"
    assert train.model.action_horizon == infer.model.action_horizon == 50
    assert train.model.discrete_state_input is infer.model.discrete_state_input is True
    assert train.data.extra_delta_transform is infer.data.extra_delta_transform is True
    assert train.data.side_image_key == infer.data.side_image_key == "side_image"
    assert train.ema_decay is None
    assert infer.data.adv_ind_dropout is False
