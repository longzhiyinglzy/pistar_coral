import numpy as np
import pytest

from openpi.training import weight_loaders


def test_checkpoint_overlay_replaces_strict_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = {
        "backbone": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        "lora": {"a": np.zeros((2, 1), dtype=np.float32)},
    }
    base = {
        "backbone": {"kernel": np.ones((2, 2), dtype=np.float32)},
        "lora": {"a": np.zeros((2, 1), dtype=np.float32)},
    }
    overlay = {"lora": {"a": np.full((2, 1), 3, dtype=np.float32)}}

    monkeypatch.setattr(weight_loaders.CheckpointWeightLoader, "load", lambda self, params: base)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: overlay)

    result = weight_loaders.CheckpointOverlayWeightLoader("base", "overlay").load(reference)
    np.testing.assert_array_equal(result["backbone"]["kernel"], base["backbone"]["kernel"])
    np.testing.assert_array_equal(result["lora"]["a"], overlay["lora"]["a"])


def test_checkpoint_overlay_rejects_unknown_path(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = {"known": np.zeros((1,), dtype=np.float32)}
    monkeypatch.setattr(weight_loaders.CheckpointWeightLoader, "load", lambda self, params: reference)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(
        weight_loaders._model,
        "restore_params",
        lambda *args, **kwargs: {"unknown": np.zeros((1,), dtype=np.float32)},
    )

    with pytest.raises(KeyError, match="unknown parameter paths"):
        weight_loaders.CheckpointOverlayWeightLoader("base", "overlay").load(reference)
