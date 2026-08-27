# PiStar-CORAL 三物块真机连续推理

本文记录当前三物块连续装配的真机推理命令。流程使用 3 个终端：

1. PiStar-CORAL policy server，监听 `8000`。
2. 视觉状态机代理，监听 `8001`，转发到 `8000` 并自动切换 expert。
3. Piper 真机评估客户端，连接 `8001`。

当前 expert 对应关系：

| 状态机任务 | CORAL expert | 导出来源 |
| --- | --- | --- |
| task1 | assemble_block1_recap1 | `/home/user/code/pistar-coral/checkpoints/block1_recap1/9999` |
| task2 | assemble_block2 | `/home/user/code/pistar-coral/checkpoints/block2/29999` |
| task3 | assemble_block3 | `/home/user/code/pistar-coral/checkpoints/block3/29999` |

状态机只自动执行 `task1 -> task2 -> task3`。到达 task3 后会一直使用 `assemble_block3`，不会根据第三个槽位自动进入 `done/holding`。第三个物块装配完成后，由操作者在客户端终端按 `s` 或 `f` 判断整条三物块任务成功或失败。

## 运行前检查

确认机械臂 CAN 和相机没有被其他进程占用：

```bash
ip -brief link show can0
cat /sys/class/net/can0/operstate

ps -ef | grep -E \
  'serve_coral_policy|coral_switch_proxy|piper_pi05_rtc_success_eval' \
  | grep -v grep
```

确认三个导出的 expert 都存在：

```bash
EXPERT_ROOT=/home/user/code/pistar-coral/deployments/three_blocks_recap1/experts

for expert in assemble_block1_recap1 assemble_block2 assemble_block3; do
  test -f "$EXPERT_ROOT/$expert/expert.json" && \
  test -d "$EXPERT_ROOT/$expert/lora_params" && \
  test -f "$EXPERT_ROOT/$expert/assets/norm_stats.json" && \
  echo "[OK] $expert"
done
```

## 终端1：启动 PiStar-CORAL policy server

`BETA` 是 PiStar advantage guidance 强度。当前可直接修改，例如 `1.0`、`1.1` 或 `1.2`。每次修改 beta 后必须重启终端1。

```bash
cd /home/user/code/pistar-coral

BETA=1.0
RUNTIME_CONFIG=/home/user/code/pistar-coral/deployments/three_blocks_recap1/coral_runtime.json

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=/home/user/code/pistar-coral/src \
/home/user/code/pistar/venv/bin/python \
  scripts/coral/serve_coral_policy.py \
  --runtime-config "$RUNTIME_CONFIG" \
  --adv-guidance-beta "$BETA" \
  --port 8000
```

启动成功后应看到类似信息：

```text
PiStar advantage guidance beta: 1.0
assemble_block1_recap1
assemble_block2
assemble_block3
```

不要在这个命令中增加 `--default-expert`，因为 `default_expert` 已经写在 runtime JSON 中。初始 expert 应为 `assemble_block1_recap1`。

## 终端2：启动自动切换状态机代理

状态机使用侧视相机完整的原生 `640x480` 图像判断槽位占用。这里将 `CUDA_VISIBLE_DEVICES` 置空，让小型状态分类器使用 CPU，避免占用 policy server 的 GPU。

```bash
cd /home/user/code/openpi_coral

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export CUDA_VISIBLE_DEVICES=""

PYTHONPATH=/home/user/code/openpi_coral \
/home/user/code/openpi_coral/.venv/bin/python \
  -m state_machine.coral_switch_proxy \
  --listen-host 0.0.0.0 \
  --listen-port 8001 \
  --coral-host 127.0.0.1 \
  --coral-port 8000 \
  --mailbox /tmp/openpi_coral_expert_switch.json \
  --stage-config /home/user/code/openpi_coral/state_machine/config_block1_recap1.yaml \
  --silent-action-horizon 50
```

启动成功后应看到：

```text
CORAL switch proxy listening on ws://0.0.0.0:8001 -> ws://127.0.0.1:8000
Stage reset keyboard enabled: press R in this proxy terminal to reset to task1.
```

状态机切换逻辑：

- 连续 3 帧检测到槽位1已有物块：切换到 `assemble_block2`。
- 连续 3 帧检测到槽位2已有物块：切换到 `assemble_block3`。
- 到达 task3 后一直保持 `assemble_block3`，第三个物块不会触发自动结束。
- 如需人工复位状态机，在终端2按大写或小写 `R`。

## 终端3：运行带 EMA 的三物块连续真机评估

下面默认连续测试 20 条完整三物块任务。每条最多 `4500` 控制步，约等于给每个物块预留原来单任务的 `1500` 步。EMA 位于 RTC 输出之后、Piper 最终关节命令之前，只平滑 6 个机械臂关节；夹爪命令不平滑，回 home 的 reset 命令也会绕过 EMA。

```bash
cd /home/user/code/pistar

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

mkdir -p /home/user/code/pistar/outputs/ema_logs
EMA_LOG=/home/user/code/pistar/outputs/ema_logs/three_blocks_beta1_tau002_$(date +%Y%m%d_%H%M%S).csv

scripts/run_pi05_rtc_success_eval_ema.sh \
  --ema-enabled true \
  --ema-time-constant 0.02 \
  --ema-log-path "$EMA_LOG" \
  --policy-speed-percent 10 \
  --control-repo-path /home/user/code/pistar/control_your_robot \
  --server-host 127.0.0.1 \
  --server-port 8001 \
  --task-name "Assemble block1, block2, and block3 in order." \
  --adv-ind positive \
  --send-full-state-camera \
  --arm-can can0 \
  --arm-name left_arm \
  --state-source joint \
  --cam-head-serial 323522063521 \
  --cam-side-serial 349222061138 \
  --cam-wrist-serial 409122272461 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-warmup-s 1.0 \
  --camera-read-attempts 3 \
  --camera-retry-sleep 0.25 \
  --fps 30 \
  --control-dt 0.033333 \
  --action-horizon 50 \
  --resize-size 224 \
  --num-id 20 \
  --num-position-ood 0 \
  --num-angle-ood 0 \
  --max-step 4500 \
  --timeout-label failure \
  --reset-speed-percent 10 \
  --post-reset-sleep 2.0 \
  --rtc-enabled true \
  --rtc-execution-horizon 10 \
  --rtc-max-guidance-weight 10.0 \
  --rtc-prefix-attention-schedule exp \
  --rtc-measure-inference-delay false \
  --rtc-inference-delay-steps 4 \
  --rtc-prefetch-threshold 20 \
  --rtc-worker-sleep 0.005 \
  --rtc-debug false
```

`--ema-time-constant 0.02` 与参考实现一致。在 `30 Hz` 下对应 `alpha=0.625`；数值越大越平滑，但动作滞后也越明显。启动时终端应打印：

```text
[EMA] enabled=True time_constant=0.020000s dt=0.033333s alpha=0.624998 policy_speed=10% joints=6 gripper=unchanged reset=bypass
```

客户端操作：

- `Enter`：开始当前完整三物块 trial。
- 第三个物块装配完成后按 `s`：整条 trial 成功。
- 任何阶段确定失败后按 `f`：整条 trial 失败。
- 按 `r`：丢弃当前 trial，并重试相同序号。
- 按 `q` 或 `Esc`：退出。

每条新 trial 第一次推理时，客户端会请求状态机回到 task1，因此正常情况下不需要手动到终端2按 `R`。如果终端2显示的 stage 没有回到 `0.0/task1`，再手动按 `R`。

## RTC 延迟调整

默认配置：

```text
rtc-execution-horizon=10
rtc-inference-delay-steps=4
```

如果 policy server 日志或客户端显示推理延迟稳定在约 `160-170 ms`，可把终端3中的：

```bash
--rtc-inference-delay-steps 4
```

改为：

```bash
--rtc-inference-delay-steps 5
```

先保持其他 RTC 参数不变，再比较成功率。beta 与推理耗时没有必然的线性关系；每个 beta 都应重新观察实际 `latency` 后再决定 delay。

## 正确的启动和停止顺序

启动顺序：

1. 终端1：policy server，端口8000。
2. 终端2：状态机代理，端口8001。
3. 终端3：真机客户端，连接8001。

停止顺序：

1. 先在终端3按 `q`，停止机械臂客户端。
2. 再在终端2按 `Ctrl+C`，停止代理。
3. 最后在终端1按 `Ctrl+C`，停止 policy server。

客户端正常退出后，终端2偶尔出现 WebSocket `ConnectionClosedError` 只表示客户端连接已经断开，不代表本次推理或权重出错。
