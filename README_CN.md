# PiStar-CORAL

这个项目实现的目标是：

```text
一个共享 pi0.5 VLA backbone
├── 物块1 PiStar-CORAL expert
├── 物块2 PiStar-CORAL expert
└── 物块3 PiStar-CORAL expert
          ↑
      推理时热切换
```

它不是把多个完整 PiStar 服务放在不同端口再做路由。训练时使用 PiStar 的
`adv_ind=positive/negative` 条件目标和 CFG；只更新 CORAL 定义的
`lora + action_head + image_tower + action_expert`。推理时只加载一次
`pi05_base`，然后覆盖当前 expert 的参数。

## block1 r0 的确定流程

已有输入：

- 原始示教：`assemble_block1_v21_demo250_uniform`，250 条。
- 已打 advantage 的 rollout：
  `assemble_block1_coral_dagger250_r0`，250 条、200 成功、50 失败。
- block1 CORAL 初始化 expert（从原 25000 checkpoint 导出）。

policy 数据严格沿用第一轮 recap 的比例：

```text
demo250（全部 positive）
+ rollout 成功200 × 3（保留逐帧 positive/negative）
= 850 episodes
```

失败 50 条只参与 value function 训练和 advantage 计算，不进入本轮 policy
训练。所有正式名称都不再带 `center4x3`。

## 内网数据制作

以下路径按现有 `pistar_train` 容器：

```text
代码：/home/user/code/pistar-coral
宿主机 LeRobot：/root/data_local/lerobot
容器 LeRobot：/home/user/.cache/huggingface/lerobot
宿主机工作目录：/root/data_local（容器内 /workspaces）
```

进入容器：

```bash
docker start pistar_train
docker exec -it pistar_train bash
cd /home/user/code/pistar-coral
```

把原始 demo250 转成 30 Hz 三视角 flat 数据：

```bash
PYTHONPATH=src python scripts/convert_lerobot_v21_to_pistar_flat.py \
  --source /workspaces/assemble_block1_v21_demo250_uniform \
  --output /home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_positive_r0 \
  --target-fps 30 \
  --adv-ind positive \
  --intervention 1
```

过滤 rollout 的 200 条成功 episode：

```bash
PYTHONPATH=src python scripts/filter_success_episodes.py \
  --input-root /home/user/.cache/huggingface/lerobot/piper/assemble_block1_coral_dagger250_r0 \
  --output-root /home/user/.cache/huggingface/lerobot/piper/assemble_block1_coral_dagger200_success_r0 \
  --criterion reward \
  --num-workers 32
```

合并 850 条。重复写三次同一个 source 是有意的：

```bash
PYTHONPATH=src python scripts/merge_datasets.py \
  --sources \
    /home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_positive_r0 \
    /home/user/.cache/huggingface/lerobot/piper/assemble_block1_coral_dagger200_success_r0 \
    /home/user/.cache/huggingface/lerobot/piper/assemble_block1_coral_dagger200_success_r0 \
    /home/user/.cache/huggingface/lerobot/piper/assemble_block1_coral_dagger200_success_r0 \
  --output /home/user/.cache/huggingface/lerobot/piper/assemble_block1_coral_pistar_policy850_r0 \
  --fps 30 \
  --num-workers 32
```

确认结果：

```bash
PYTHONPATH=src python scripts/validate_pistar_coral_setup.py \
  --dataset /home/user/.cache/huggingface/lerobot/piper/assemble_block1_coral_pistar_policy850_r0 \
  --expected-episodes 850
```

## norm stats 与训练

服务器已有 `/mnt/pi05_base/params`，因此不用传 6.8 GB 的完整 CORAL
checkpoint。把本机已导出的 block1 expert 目录打包并放到宿主机：

```text
/root/data_local/coral_block1_init
```

容器中会对应：

```text
/workspaces/coral_block1_init/lora_params
```

这个 expert 含原 CORAL checkpoint 训练过的全部 62 个参数叶：
`lora + action_head + image_tower + action_expert`。训练加载器先读
`pi05_base`，再严格覆盖这 62 个参数，结果等价于从原完整 CORAL checkpoint
初始化，同时少传约 3.7 GB。

计算新 850 条训练集的 norm stats：

```bash
cd /home/user/code/pistar-coral

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/compute_norm_stats.py \
  --config-name pi05_star_coral_piper_block1_r0 \
  --repo-id piper/assemble_block1_coral_pistar_policy850_r0 \
  --local-data-dir /home/user/.cache/huggingface/lerobot/piper/assemble_block1_coral_pistar_policy850_r0
```

使用 6 张 H20 训练。这里的初始化是
`pi05_base + 已有 block1 CORAL expert`，不是单独从 `pi05_base` 重新开始：

```bash
cd /home/user/code/pistar-coral

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src python scripts/train.py pi05_star_coral_piper_block1_r0 \
  --exp-name assemble_block1_coral_pistar_policy850_h20_6gpu_b192_r0 \
  --data.repo-id piper/assemble_block1_coral_pistar_policy850_r0 \
  --assets-base-dir /home/user/code/pistar-coral/assets \
  --checkpoint-base-dir /workspaces/pistar_coral_runs \
  --weight-loader.base-params-path /mnt/pi05_base/params \
  --weight-loader.overlay-params-path /workspaces/coral_block1_init/lora_params \
  --batch-size 192 \
  --fsdp-devices 6 \
  --num-workers 32 \
  --num-train-steps 10000 \
  --wandb-enabled false \
  --overwrite
```

## 导出 block1 expert

以 step 10000 为例：

```bash
cd /home/user/code/pistar-coral

CKPT=/workspaces/pistar_coral_runs/pi05_star_coral_piper_block1_r0/assemble_block1_coral_pistar_policy850_h20_6gpu_b192_r0/10000
EXPORT_ROOT=/workspaces/pistar_coral_weights

PYTHONPATH=src python scripts/coral/export_lora_expert.py \
  --checkpoint-dir "$CKPT" \
  --expert-name assemble_block1 \
  --base-config pi05_star_coral_piper_infer \
  --task-prompt "Pick up the block1 and assemble it." \
  --output-root "$EXPORT_ROOT" \
  --base-checkpoint /mnt/pi05_base/params \
  --param-modules lora,action_head,image_tower,action_expert \
  --norm-stats-dir "$CKPT/assets/piper/assemble_block1_coral_pistar_policy850_r0"
```

block2、block3 必须也经过同一套 PiStar-CORAL 训练后再注册。旧的普通 CORAL
expert 没见过 advantage token，不应该直接和新 block1 expert 混在正式评估中。

## 热切换推理

```bash
cd /home/user/code/pistar-coral

RUNTIME=/workspaces/pistar_coral_weights/pi05/pi05_star_coral_piper_infer/lora_action_head_image_tower_action_expert/coral_runtime.json

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src python scripts/coral/serve_coral_policy.py \
  --runtime-config "$RUNTIME" \
  --port 8000
```

真机请求必须带正优势条件：

```text
adv_ind=positive
expert=assemble_block1
```

server metadata 会声明 `requires_adv_ind=true`。现有 rollout 客户端因此需要加：

```bash
--adv-ind positive
```

expert 切换只替换 CORAL 参数，不重新加载共享 backbone；切换时还要清空 RTC
action queue，不能继续执行上一个任务剩余的 action chunk。

## 旧 router

`src/pistar_coral/` 中原来的多 websocket 服务 router 仍保留，命令改为
`pistar-coral-router`。它只用于兼容或诊断，不是本项目的正式共享-backbone
方案。
