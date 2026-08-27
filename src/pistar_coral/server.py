from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Any

import websockets.asyncio.server
import websockets.frames
from openpi_client import base_policy, msgpack_numpy

logger = logging.getLogger(__name__)


class ManagerServer:
    """OpenPI websocket-compatible server without depending on the OpenPI package."""

    def __init__(
        self,
        manager: base_policy.BasePolicy,
        *,
        host: str,
        port: int,
        metadata: dict[str, Any],
    ):
        self._manager = manager
        self._host = host
        self._port = port
        self._metadata = metadata

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logger.info("Client %s connected", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        previous_total_time: float | None = None
        while True:
            try:
                started = time.monotonic()
                observation = msgpack_numpy.unpackb(await websocket.recv())
                infer_started = time.monotonic()
                result = await asyncio.to_thread(self._manager.infer, observation)
                infer_ms = (time.monotonic() - infer_started) * 1000.0
                result["server_timing"] = {"infer_ms": infer_ms}
                if previous_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = previous_total_time * 1000.0
                await websocket.send(packer.pack(result))
                previous_total_time = time.monotonic() - started
            except websockets.ConnectionClosed:
                logger.info("Client %s disconnected", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: websockets.asyncio.server.ServerConnection,
    request: websockets.asyncio.server.Request,
) -> websockets.asyncio.server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
