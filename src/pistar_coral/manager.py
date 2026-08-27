from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from openpi_client import base_policy, websocket_client_policy
from typing_extensions import override

from pistar_coral.router import ExpertEndpoint, RouterConfig, TaskRouter

logger = logging.getLogger(__name__)


class _LazyExpertClient:
    def __init__(
        self,
        endpoint: ExpertEndpoint,
        client_factory: Callable[..., base_policy.BasePolicy],
    ):
        self._endpoint = endpoint
        self._client_factory = client_factory
        self._client: base_policy.BasePolicy | None = None
        self._lock = threading.Lock()

    def connect(self) -> base_policy.BasePolicy:
        with self._lock:
            if self._client is None:
                logger.info(
                    "Connecting CORAL expert '%s' at ws://%s:%d",
                    self._endpoint.name,
                    self._endpoint.host,
                    self._endpoint.port,
                )
                self._client = self._client_factory(host=self._endpoint.host, port=self._endpoint.port)
            return self._client

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._client is None:
                logger.info(
                    "Connecting CORAL expert '%s' at ws://%s:%d",
                    self._endpoint.name,
                    self._endpoint.host,
                    self._endpoint.port,
                )
                self._client = self._client_factory(host=self._endpoint.host, port=self._endpoint.port)
            return self._client.infer(observation)

    def metadata(self) -> dict[str, Any]:
        client = self.connect()
        get_metadata = getattr(client, "get_server_metadata", None)
        return dict(get_metadata() if get_metadata is not None else {})

    def reset(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.reset()


class CoralPiStarManager(base_policy.BasePolicy):
    """OpenPI-compatible manager that delegates inference to a routed PiStar expert."""

    def __init__(
        self,
        config: RouterConfig,
        *,
        client_factory: Callable[..., base_policy.BasePolicy] = websocket_client_policy.WebsocketClientPolicy,
        eager_connect: bool = True,
    ):
        self._router = TaskRouter(config)
        self._clients = {expert.name: _LazyExpertClient(expert, client_factory) for expert in config.experts}
        self._expert_metadata: dict[str, dict[str, Any]] = {}
        if eager_connect:
            self._expert_metadata = {expert.name: self._clients[expert.name].metadata() for expert in config.experts}
        requires_adv_ind = {
            expert.name: bool(self._expert_metadata.get(expert.name, {}).get("requires_adv_ind", True))
            for expert in config.experts
        }
        self.metadata = {
            "deploy_mode": "coral_pistar_router",
            "experts": [expert.name for expert in config.experts],
            "requires_adv_ind": any(requires_adv_ind.values()),
            "expert_requires_adv_ind": requires_adv_ind,
            "routing_fields": ["coral_task", "prompt"],
        }

    @override
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        endpoint, route_reason = self._router.select(observation)
        request = dict(observation)
        request.pop("coral_task", None)
        started = time.monotonic()
        result = dict(self._clients[endpoint.name].infer(request))
        result["coral"] = {
            "expert": endpoint.name,
            "route_reason": route_reason,
            "route_and_backend_ms": (time.monotonic() - started) * 1000.0,
        }
        return result

    @override
    def reset(self) -> None:
        for client in self._clients.values():
            client.reset()
