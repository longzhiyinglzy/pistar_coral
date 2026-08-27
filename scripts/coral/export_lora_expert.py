"""Export the hot-switchable parameter subset for one CORAL LoRA expert."""

import dataclasses
import pathlib
import shutil

import flax.traverse_util
import jax
import numpy as np
from openpi.policies import coral_param_filters
from openpi.policies import coral_runtime_config
from openpi.policies import lora_expert_manager
import orbax.checkpoint as ocp
import tyro


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    expert_name: str
    base_config: str
    task_prompt: str
    output_dir: str | None = None
    # Optional structured export root. Used when --output-dir is not provided.
    output_root: str | None = None
    # Optional runtime config path. If missing, it is created under the structured export directory.
    runtime_config: str | None = None
    robot: str = "so101"
    lora_rank: int = 16
    # Comma-separated atomic modules, for example: lora,action_head,action_expert.
    param_modules: str | None = None
    # Deprecated compatibility flag for existing export commands.
    param_mode: str | None = None
    # Optional base checkpoint path recorded in expert metadata.
    base_checkpoint: str | None = None
    # Optional override for expert metadata. Defaults to the model type from --base-config.
    model_type: str | None = None
    # Optional assets directory containing norm_stats.json.
    norm_stats_dir: str | None = None
    # Replace an existing exported parameter directory.
    overwrite: bool = False


def _save_params(params_dir: pathlib.Path, params: dict) -> None:
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(params_dir, {"params": params})


def _restore_params(params_dir: str) -> dict:
    """Restore an OpenPI params checkpoint as a pure dict without importing models."""
    params_path = _resolve_params_dir(params_dir)
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(params_path)
        item = {"params": metadata["params"]}
        restored = checkpointer.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item),
            ),
        )["params"]

    flat_params = flax.traverse_util.flatten_dict(restored)
    if flat_params and all(path[-1] == "value" for path in flat_params):
        flat_params = {path[:-1]: value for path, value in flat_params.items()}
    return flax.traverse_util.unflatten_dict(flat_params)


def _resolve_params_dir(checkpoint_dir: str) -> pathlib.Path:
    path = pathlib.Path(checkpoint_dir).resolve()
    nested_params = path / "params"
    return nested_params if nested_params.exists() else path


def _copy_norm_stats(norm_stats_dir: str, output_dir: pathlib.Path) -> pathlib.Path:
    source = pathlib.Path(norm_stats_dir).resolve()
    source_path = source if source.name == "norm_stats.json" else source / "norm_stats.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Norm stats file not found: {source_path}")

    target_path = output_dir / lora_expert_manager.ASSETS_DIR / "norm_stats.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return target_path


def _resolve_param_spec(args: Args) -> coral_param_filters.CoralParamSpec:
    if args.param_modules is not None and args.param_mode is not None:
        raise ValueError("Use either --param-modules or legacy --param-mode, not both.")
    if args.param_modules is None and args.param_mode is None:
        raise ValueError("--param-modules is required for new exports; legacy commands may use --param-mode.")
    value = args.param_modules if args.param_modules is not None else args.param_mode
    assert value is not None
    return coral_param_filters.parse_param_spec(value)


def _resolve_model_type(args: Args) -> str:
    if args.model_type is not None:
        model_type = args.model_type.strip()
        if not model_type:
            raise ValueError("--model-type must be non-empty when provided.")
        valid = {"pi0", "pi05", "pi0_fast"}
        if model_type not in valid:
            raise ValueError(f"--model-type must be one of {sorted(valid)}, got '{model_type}'.")
        return model_type

    from openpi.training import config as _config

    return _config.get_config(args.base_config).model.model_type.value


def _resolve_output_dir(
    args: Args,
    *,
    param_spec: coral_param_filters.CoralParamSpec,
    model_type: str,
) -> tuple[pathlib.Path, pathlib.Path | None, coral_runtime_config.CoralRuntimeConfig | None]:
    if args.output_dir is not None:
        if args.output_root is not None or args.runtime_config is not None:
            raise ValueError("Use either --output-dir or structured --output-root/--runtime-config, not both.")
        return pathlib.Path(args.output_dir), None, None

    if args.runtime_config is not None:
        config_path = pathlib.Path(args.runtime_config)
        if config_path.exists():
            deployment = coral_runtime_config.CoralRuntimeConfig.from_json_file(config_path)
            _validate_export_runtime_config(args, deployment, param_spec=param_spec, model_type=model_type)
            deployment = coral_runtime_config.with_model_type(deployment, model_type)
            if deployment.param_modules is None:
                deployment = dataclasses.replace(
                    deployment,
                    param_modules=coral_runtime_config.canonical_module_values(param_spec),
                )
            if deployment.base_checkpoint_dir is None and args.base_checkpoint is not None:
                deployment = dataclasses.replace(deployment, base_checkpoint_dir=args.base_checkpoint)
        else:
            if args.output_root is None:
                raise ValueError("Creating --runtime-config requires --output-root when --output-dir is omitted.")
            deployment = _new_runtime_config(args, param_spec=param_spec, model_type=model_type)
        deployment = coral_runtime_config.with_expert(deployment, args.expert_name)
        return pathlib.Path(deployment.experts_dir) / args.expert_name, config_path, deployment

    if args.output_root is not None:
        deployment = _new_runtime_config(args, param_spec=param_spec, model_type=model_type)
        config_path = coral_runtime_config.structured_runtime_config_path(
            args.output_root,
            model_type,
            args.base_config,
            param_spec,
        )
        deployment = coral_runtime_config.with_expert(deployment, args.expert_name)
        return pathlib.Path(deployment.experts_dir) / args.expert_name, config_path, deployment

    raise ValueError("Either --output-dir or --output-root/--runtime-config is required.")


def _new_runtime_config(
    args: Args,
    *,
    param_spec: coral_param_filters.CoralParamSpec,
    model_type: str,
) -> coral_runtime_config.CoralRuntimeConfig:
    assert args.output_root is not None
    return coral_runtime_config.CoralRuntimeConfig.for_structured_root(
        root=args.output_root,
        base_config=args.base_config,
        base_checkpoint_dir=args.base_checkpoint,
        model_type=model_type,
        param_modules=param_spec,
        default_expert=args.expert_name,
        expert_names=(args.expert_name,),
    )


def _validate_export_runtime_config(
    args: Args,
    deployment: coral_runtime_config.CoralRuntimeConfig,
    *,
    param_spec: coral_param_filters.CoralParamSpec,
    model_type: str,
) -> None:
    if deployment.base_config != args.base_config:
        raise ValueError(
            f"Runtime config base_config='{deployment.base_config}' does not match --base-config='{args.base_config}'."
        )
    if args.output_root is not None and deployment.root is not None:
        if pathlib.Path(args.output_root) != pathlib.Path(deployment.root):
            raise ValueError(
                f"Runtime config root='{deployment.root}' does not match --output-root='{args.output_root}'."
            )
    if deployment.model_type is not None and deployment.model_type != model_type:
        raise ValueError(
            f"Runtime config model_type='{deployment.model_type}' does not match export model_type='{model_type}'."
        )
    if deployment.param_modules is not None:
        expected = coral_param_filters.CoralParamSpec.from_modules(deployment.param_modules)
        if expected != param_spec:
            raise ValueError(
                f"Runtime config param_modules={expected.values} does not match export modules={param_spec.values}."
            )


def main(args: Args) -> None:
    param_spec = _resolve_param_spec(args)
    model_type = _resolve_model_type(args)
    output_dir, runtime_config_path, deployment = _resolve_output_dir(
        args,
        param_spec=param_spec,
        model_type=model_type,
    )
    params_dir = output_dir / lora_expert_manager.LORA_PARAMS_DIR
    if params_dir.exists() and any(params_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Expert params already exist: {params_dir}. Pass --overwrite to replace them.")
        shutil.rmtree(params_dir)

    params = _restore_params(args.checkpoint_dir)
    module_counts = coral_param_filters.count_module_leaves(params, param_spec)
    missing_modules = sorted(module.value for module, count in module_counts.items() if count == 0)
    if missing_modules:
        raise ValueError(f"No checkpoint parameters matched requested CORAL modules: {missing_modules}")

    expert_params = coral_param_filters.filter_param_dict(params, param_spec)
    exported_param_count = len(flax.traverse_util.flatten_dict(expert_params))
    if exported_param_count == 0:
        raise ValueError(f"No parameters matched CORAL module spec '{param_spec.label}'.")

    metadata = lora_expert_manager.LoraExpertMetadata(
        name=args.expert_name,
        base_config=args.base_config,
        task_prompt=args.task_prompt,
        robot=args.robot,
        lora_rank=args.lora_rank,
        created_from_checkpoint=args.checkpoint_dir,
        model_type=model_type,
        param_mode=None,
        param_modules=param_spec.values,
        base_checkpoint=args.base_checkpoint,
        exported_param_count=exported_param_count,
    )
    metadata_path = lora_expert_manager.write_expert_metadata(output_dir, metadata)
    # Orbax requires the checkpoint target directory to not exist.
    if params_dir.exists():
        params_dir.rmdir()
    _save_params(params_dir, expert_params)
    norm_stats_path = _copy_norm_stats(args.norm_stats_dir, output_dir) if args.norm_stats_dir is not None else None
    print(f"Wrote LoRA expert metadata: {metadata_path}")
    if runtime_config_path is not None and deployment is not None:
        coral_runtime_config.write_json_file(deployment, runtime_config_path)
        print(f"Wrote CORAL runtime config: {runtime_config_path}")
    print(f"Model type: {model_type}")
    print(f"CORAL parameter modules: {', '.join(param_spec.values)}")
    for module in sorted(module_counts, key=lambda item: item.value):
        print(f"  {module.value}: {module_counts[module]} leaves")
    print(f"Exported {exported_param_count} expert parameter leaves to: {params_dir}")
    if norm_stats_path is not None:
        print(f"Copied expert norm stats to: {norm_stats_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
