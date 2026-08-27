"""Serve a CORAL policy with slow or no-recompile hot expert switching."""

import atexit
import contextlib
import dataclasses
import logging
import select
import signal
import sys
import termios
import threading
import tty
from typing import Any, Literal

import tyro

from openpi.policies import coral_policy_factory
from openpi.policies import coral_policy
from openpi.policies import coral_runtime_config
from openpi.policies import lora_expert_manager
from openpi.serving import websocket_policy_server


_KB_TTY_STATE: dict[str, object | None] = {"fd": None, "old_settings": None}
_KB_STOP_EVENT = threading.Event()


def _restore_terminal_if_needed() -> None:
    fd = _KB_TTY_STATE.get("fd")
    old_settings = _KB_TTY_STATE.get("old_settings")
    if fd is None or old_settings is None:
        return
    with contextlib.suppress(Exception):
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    _KB_TTY_STATE["fd"] = None
    _KB_TTY_STATE["old_settings"] = None


def _request_keyboard_stop(*_args) -> None:
    _KB_STOP_EVENT.set()
    _restore_terminal_if_needed()


def _handle_termination(signum, _frame) -> None:
    logging.info("Received signal %s, shutting down CORAL server.", signum)
    _request_keyboard_stop()
    raise SystemExit(0)


atexit.register(_restore_terminal_if_needed)


@dataclasses.dataclass
class Args:
    base_config: str | None = None
    experts_dir: str | None = None
    base_checkpoint_dir: str | None = None
    runtime_config: str | None = None
    port: int = 8000
    default_expert: str | None = None
    keyboard_switch: bool = False
    switch_mode: Literal["slow", "hot"] | None = None
    adv_guidance_beta: float | None = None


def _get_train_config(deployment: coral_runtime_config.CoralRuntimeConfig, args: Args):
    from openpi.training import config as _config

    train_config = _config.get_config(deployment.base_config)
    if args.adv_guidance_beta is None:
        return train_config
    if args.adv_guidance_beta <= 0:
        raise ValueError("--adv-guidance-beta must be greater than zero.")
    if not getattr(train_config.model, "pistar", False):
        raise ValueError("--adv-guidance-beta is only valid for a PiStar model config.")
    return dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, adv_guidance_beta=args.adv_guidance_beta),
    )


def create_policy(args: Args) -> Any:
    deployment = _resolve_runtime_config(args)
    train_config = _get_train_config(deployment, args)
    deployment = coral_runtime_config.with_model_type(deployment, train_config.model.model_type.value)
    load_params = deployment.switch_mode != "hot"
    manager = coral_runtime_config.load_validated_expert_manager(deployment, load_params=load_params)
    default_expert = deployment.default_expert or manager.list_experts()[0]

    if deployment.switch_mode == "hot":
        from openpi.policies import coral_hot_policy_factory

        return coral_hot_policy_factory.create_coral_hot_policy(
            train_config,
            base_params_dir=deployment.base_checkpoint_dir,
            expert_manager=manager,
            default_expert=default_expert,
        )

    def policy_factory(expert: lora_expert_manager.LoraExpert):
        if expert.lora_params is None:
            raise ValueError(f"Expert '{expert.name}' has no loaded params.")
        logging.info("Building CORAL policy for expert: %s", expert.name)
        return coral_policy_factory.create_coral_expert_policy(
            train_config,
            base_params_dir=deployment.base_checkpoint_dir,
            expert_params=expert.lora_params,
            default_prompt=expert.metadata.task_prompt,
        )

    policy = coral_policy.CoralPolicy(expert_manager=manager, policy_factory=policy_factory)
    policy.switch_expert(default_expert)
    return policy


def main(args: Args) -> None:
    signal.signal(signal.SIGINT, _handle_termination)
    signal.signal(signal.SIGTERM, _handle_termination)

    policy = create_policy(args)
    if args.keyboard_switch:
        _start_keyboard_switcher(policy)

    logging.info("Serving CORAL policy")
    deployment = _resolve_runtime_config(args)
    train_config = _get_train_config(deployment, args)
    deployment = coral_runtime_config.with_model_type(deployment, train_config.model.model_type.value)
    logging.info("Runtime config: %s", args.runtime_config)
    logging.info("Base config: %s", deployment.base_config)
    logging.info("Model type: %s", deployment.model_type)
    logging.info("Experts dir: %s", deployment.experts_dir)
    logging.info("Base checkpoint override: %s", deployment.base_checkpoint_dir)
    logging.info("Switch mode: %s", deployment.switch_mode)
    logging.info("PiStar advantage guidance beta: %s", getattr(train_config.model, "adv_guidance_beta", None))
    logging.info("Registered experts:")
    for index, expert_name in enumerate(policy.list_experts()):
        marker = "*" if expert_name == policy.active_expert() else " "
        logging.info("  %s [%s] %s", marker, index, expert_name)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        exit_on_handler_exception=True,
        metadata={
            "coral": True,
            "pistar": bool(getattr(train_config.model, "pistar", False)),
            "requires_adv_ind": bool(getattr(train_config.model, "pistar", False)),
            "adv_guidance_beta": getattr(train_config.model, "adv_guidance_beta", None),
            "base_config": deployment.base_config,
            "active_expert": policy.active_expert(),
            "experts": policy.list_experts(),
            "switch_field": "expert",
            "switch_mode": deployment.switch_mode,
        },
    )
    server.serve_forever()


def _resolve_runtime_config(args: Args) -> coral_runtime_config.CoralRuntimeConfig:
    if args.runtime_config is not None:
        config = coral_runtime_config.CoralRuntimeConfig.from_json_file(args.runtime_config)
        overrides = {
            "base_config": args.base_config,
            "experts_dir": args.experts_dir,
            "base_checkpoint_dir": args.base_checkpoint_dir,
            "default_expert": args.default_expert,
            "switch_mode": args.switch_mode,
        }
        provided = sorted(name for name, value in overrides.items() if value is not None)
        if provided:
            raise ValueError(
                "When --runtime-config is provided, keep base/expert fields in that file only. "
                f"Conflicting CLI fields: {provided}"
            )
        return config

    if args.base_config is None or args.experts_dir is None:
        raise ValueError("Either --runtime-config or both --base-config and --experts-dir are required.")
    return coral_runtime_config.CoralRuntimeConfig(
        base_config=args.base_config,
        experts_dir=args.experts_dir,
        base_checkpoint_dir=args.base_checkpoint_dir,
        default_expert=args.default_expert,
        switch_mode=args.switch_mode or "slow",
    )


def _start_keyboard_switcher(policy: Any) -> None:
    experts = policy.list_experts()
    logging.info("Keyboard expert switch enabled. Press 0-%d to switch experts.", len(experts) - 1)

    def run() -> None:
        if not sys.stdin.isatty():
            logging.warning("Keyboard switch requested, but stdin is not a TTY.")
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        _KB_TTY_STATE["fd"] = fd
        _KB_TTY_STATE["old_settings"] = old_settings
        try:
            tty.setcbreak(fd)
            while not _KB_STOP_EVENT.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not readable:
                    continue
                key = sys.stdin.read(1)
                if not key.isdigit():
                    continue
                index = int(key)
                if index >= len(experts):
                    logging.warning("No expert at index %s. Available range: 0-%d", index, len(experts) - 1)
                    continue
                expert_name = experts[index]
                logging.info("Keyboard switch requested: [%s] %s", index, expert_name)
                policy.switch_expert(expert_name)
        except Exception:
            logging.exception("Keyboard switcher stopped unexpectedly.")
        finally:
            _restore_terminal_if_needed()

    threading.Thread(target=run, name="coral-keyboard-switcher", daemon=True).start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
