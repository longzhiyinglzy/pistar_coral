from __future__ import annotations

import argparse
import logging

from pistar_coral.manager import CoralPiStarManager
from pistar_coral.router import RouterConfig
from pistar_coral.server import ManagerServer

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone PiStar + CORAL task manager.")
    parser.add_argument("--config", required=True, help="Path to a CORAL router JSON config.")
    parser.add_argument("--host", default="0.0.0.0", help="Manager websocket listen host.")
    parser.add_argument("--port", type=int, default=8000, help="Manager websocket listen port.")
    parser.add_argument(
        "--lazy-connect",
        action="store_true",
        help="Start before expert servers are available; the first request waits for its selected expert.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    config = RouterConfig.load(args.config)
    manager = CoralPiStarManager(config, eager_connect=not args.lazy_connect)
    logger.info("CORAL PiStar routes: %s", ", ".join(expert.name for expert in config.experts))
    logger.info("Listening at ws://%s:%d", args.host, args.port)
    ManagerServer(manager, host=args.host, port=args.port, metadata=manager.metadata).serve_forever()


if __name__ == "__main__":
    main()
