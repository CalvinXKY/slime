# P2P Shard Weight Update

Slime 非 colocate 模式下，训练侧 Megatron 与 rollout 侧 SGLang 之间的权重同步，默认走 **all_gather + NCCL broadcast**。本分支提供 **shard 级 P2P** 路径：每个训练 TP rank 将自身 shard 经 `dist.send/recv` 直接发给对应推理 TP rank，跳过全量 gather/broadcast。

远端备份分支：`feat/p2p-shard-weight-update-v1`（fork: `CalvinXKY/slime`）。

## 前置条件

| 项 | 说明 |
|---|---|
| 运行模式 | **非 colocate**（训练与 rollout 分进程） |
| 权重模式 | `--update-weight-mode=full`（P2P 不支持 delta） |
| TP 对齐 | **Megatron TP == SGLang TP** 才走 P2P；不等时自动回退 NCCL broadcast |
| SGLang | 必须打 patch，见下文 |
| 后端 | 仅 SGLang；不支持 vLLM |

TP 对齐关系：

```
Megatron TP  = tensor_model_parallel_size
SGLang TP    = rollout_num_gpus_per_engine / sglang_pp_size
```

示例（Qwen3-4B 冒烟脚本）：训练 TP=4，rollout 单 engine 占 4 GPU → SGLang TP=4，满足 P2P。

## 快速开始（容器内 5 step 冒烟）

### 1. 获取代码

```bash
git fetch fork feat/p2p-shard-weight-update-v1
git checkout feat/p2p-shard-weight-update-v1
# 或 rsync 到容器 /root/slime
```

### 2. 应用 SGLang patch（必做）

Slime 侧改动 alone 不够，SGLang 需支持 presharded recv、`tp_tensor_counts`、NCCL barrier 等：

```bash
export SLIME_ROOT=/root/slime          # Slime 根目录
export SGLANG_ROOT=/sgl-workspace/sglang  # 容器内 SGLang 安装路径

bash "${SLIME_ROOT}/scripts/p2p/apply_sglang_p2p_patch.sh"
```

Patch 源文件：`docker/patch/latest/sglang_p2p.patch`。脚本会检测是否已应用（`io_struct.py` 含 `tp_tensor_counts` 则跳过）。

### 3. 准备模型与数据

脚本通过环境变量覆盖默认路径（以下为验证环境示例，请按实际修改）：

```bash
export HF_CKPT=/path/to/Qwen3-4B
export REF_LOAD=/path/to/Qwen3-4B_torch_dist
export PROMPT_DATA=/path/to/gsm8k/train.parquet
export SLIME_ROOT=/root/slime
export PYTHONPATH="${SLIME_ROOT}:/root/Megatron-LM/"
```

### 4. 启动 P2P 训练

```bash
bash scripts/p2p/run_p2p_tp4_qwen3_4b_smoke5step.sh
```

核心 CLI 开关（其余参数见脚本内 `train.py` 调用）：

```bash
python3 train.py \
  --tensor-model-parallel-size 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  --update-weight-mode full \
  --use-p2p-weight-update \
  ...
```

日志默认写到 `${LOG_ROOT:-/data/nfs_87/xky/new_rl/logs/p2p_experiments}/`。

## 脚本一览

| 脚本 | 用途 |
|---|---|
| `apply_sglang_p2p_patch.sh` | 标准 SGLang patch 入口 |
| `run_p2p_tp4_qwen3_4b_smoke5step.sh` | **推荐**：Qwen3-4B TP4，5 rollout step 冒烟 |
| `run_p2p_tp4_qwen3_4b_50step.sh` | P2P 50 step 长跑 |
| `run_nc_baseline_tp4_qwen3_4b_50step.sh` | 对照：默认 NCCL broadcast，50 step |
| `run_sequential_50step_p2p_then_baseline.sh` | 宿主机编排：先 P2P 50 step 再 baseline |
| `run_p2p_tp4_qwen3_4b_timing_multi.sh` | 多次重复测 update 耗时 |
| `deploy_to_verify_container.sh` | 同步代码到验证容器并打 patch |
| `deploy_and_run_on_container.sh` | 容器内打 patch 并后台启动 timing 任务 |

`patch_sglang_model_runner.py`、`patch_sglang_recv_barrier.py` 为早期 ad-hoc 工具；**请优先用 `apply_sglang_p2p_patch.sh`**。

## 宿主机 → 容器部署示例

```bash
# 同步本地/ NFS 代码到容器
rsync -a /path/to/slime/ /root/slime/
find /root/slime/scripts/p2p -name '*.sh' -exec sed -i 's/\r$//' {} +

# 打 patch 并验证
bash /root/slime/scripts/p2p/apply_sglang_p2p_patch.sh
grep tp_tensor_counts /sgl-workspace/sglang/python/sglang/srt/managers/io_struct.py

# 跑冒烟
bash /root/slime/scripts/p2p/run_p2p_tp4_qwen3_4b_smoke5step.sh
```

或使用 `deploy_to_verify_container.sh`（路径需按环境改 `rsync` 源目录）。

## 工作原理（简述）

1. **词表参数**（embed / lm_head）：TP 组内小范围 all_gather，去 Megatron padding 后按 SGLang 分片边界切分（两侧 vocab 划分不同）。
2. **其余参数**：shard 级 Megatron→HF 转换，不做 all_gather。
3. **每个 bucket**：`all_gather_object` 元数据 → rank-0 HTTP 通知 SGLang → NCCL barrier → 并行 `dist.send` 到各 engine → `ray.get` 等待 load 完成。

实现见 `slime/backends/megatron_utils/update_weight/update_weight_from_distributed_p2p.py`。

## 常见问题

**Q: 开了 `--use-p2p-weight-update` 但日志显示 NCCL broadcast？**  
A: Megatron TP 与 SGLang TP 不一致。检查 `tensor_model_parallel_size` 与 `rollout_num_gpus_per_engine`（及 `sglang_pp_size`）。

**Q: NCCL hang / 超时？**  
A: 确认 SGLang patch 已应用且两侧 barrier 一致；调试时可设 `TORCH_NCCL_BLOCKING_WAIT=1`（冒烟脚本默认 0）。

**Q: 与 baseline 对比？**  
A: 同配置下去掉 `--use-p2p-weight-update` 即走原 broadcast；或跑 `run_nc_baseline_tp4_qwen3_4b_50step.sh`。

**Q: colocate 能用吗？**  
A: 不能。colocate 走 `UpdateWeightFromTensor`，与 P2P 无关。

## 相关文件

- Slime：`update_weight_from_distributed_p2p.py`、`actor.py`、`common.py`、`megatron_to_hf/`、`sglang_engine.py`
- SGLang patch：`docker/patch/latest/sglang_p2p.patch`
