from __future__ import annotations

import pytest

from pistar_coral.manager import CoralPiStarManager
from pistar_coral.router import ExpertEndpoint, RouterConfig, TaskRouter


def _config(default_expert: str | None = None) -> RouterConfig:
    return RouterConfig(
        experts=(
            ExpertEndpoint(
                name="assemble_block1",
                host="localhost",
                port=8001,
                prompts=("Pick up the block1 and assemble it.",),
                keywords=("block1", "物块1"),
            ),
            ExpertEndpoint(
                name="assemble_block2",
                host="localhost",
                port=8002,
                prompts=("Pick up the block2 and assemble it.",),
                keywords=("block2", "物块2"),
            ),
        ),
        default_expert=default_expert,
    )


def test_router_prefers_explicit_task() -> None:
    expert, reason = TaskRouter(_config()).select({"coral_task": "assemble_block2", "prompt": "Pick up block1"})
    assert expert.name == "assemble_block2"
    assert reason == "explicit"


def test_router_matches_normalized_exact_prompt() -> None:
    expert, reason = TaskRouter(_config()).select({"prompt": "  PICK up the block1 and assemble it! "})
    assert expert.name == "assemble_block1"
    assert reason == "exact_prompt"


def test_router_matches_keyword() -> None:
    expert, reason = TaskRouter(_config()).select({"prompt": "请拿起物块2并完成装配"})
    assert expert.name == "assemble_block2"
    assert reason == "keyword:物块2"


def test_router_rejects_unknown_prompt_without_default() -> None:
    with pytest.raises(ValueError, match="did not match any route"):
        TaskRouter(_config()).select({"prompt": "move home"})


def test_router_uses_configured_default() -> None:
    expert, reason = TaskRouter(_config(default_expert="assemble_block1")).select({"prompt": "move home"})
    assert expert.name == "assemble_block1"
    assert reason == "default"


class _FakeClient:
    def __init__(self, *, host: str, port: int):
        self.host = host
        self.port = port

    def get_server_metadata(self) -> dict:
        return {"requires_adv_ind": True}

    def infer(self, observation: dict) -> dict:
        return {"actions": [[self.port]], "received": observation}

    def reset(self) -> None:
        pass


def test_manager_routes_and_removes_manager_only_field() -> None:
    manager = CoralPiStarManager(_config(), client_factory=_FakeClient)
    result = manager.infer({"coral_task": "assemble_block2", "prompt": "anything", "adv_ind": "positive"})
    assert result["actions"] == [[8002]]
    assert "coral_task" not in result["received"]
    assert result["received"]["adv_ind"] == "positive"
    assert result["coral"]["expert"] == "assemble_block2"
    assert manager.metadata["requires_adv_ind"] is True
