#!/bin/bash
# Slime P2P smoke test using the in-container Slime tree (no rsync overwrite).
set -ex

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f "train.py.*use-p2p-weight-update" 2>/dev/null || true
sleep 3

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

# Use the Slime tree already present in the container.
SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
export PYTHONPATH="${SLIME_ROOT}:/root/Megatron-LM/"

LOG_ROOT="${LOG_ROOT:-/data/nfs_87/xky/new_rl/logs/p2p_experiments}"
mkdir -p "${LOG_ROOT}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_ROOT}/slime_p2p_smoke5step_dev5_${TS}.log"
ERR_FILE="${LOG_ROOT}/slime_p2p_smoke5step_dev5_${TS}.err"

HF_CKPT="${HF_CKPT:-/data/nfs_87/xky/models/Qwen3-4B}"
REF_LOAD="${REF_LOAD:-/data/nfs_87/xky/models/Qwen3-4B_torch_dist}"
SLIME_SAVE="${SLIME_SAVE:-/data/nfs_87/xky/models/Qwen3-4B_slime_p2p_smoke5_dev5}"
PROMPT_DATA="${PROMPT_DATA:-/data/nfs_87/wx/data/slime/gsm8k/train.parquet}"

bash "${SLIME_ROOT}/scripts/p2p/apply_sglang_p2p_patch.sh"

echo "=== P2P smoke 5-step | SLIME_ROOT=${SLIME_ROOT} | LOG=${LOG_FILE} ==="

for p in "${HF_CKPT}" "${REF_LOAD}" "${PROMPT_DATA}"; do
  [ -e "${p}" ] || { echo "ERROR: path not found: ${p}"; exit 1; }
done

mkdir -p "${SLIME_SAVE}"
rm -rf "${SLIME_SAVE}/iter_"* "${SLIME_SAVE}/latest_checkpointed_iteration.txt" 2>/dev/null || true

source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"
cd "${SLIME_ROOT}"

python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --tensor-model-parallel-size 4 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --hf-checkpoint "${HF_CKPT}" \
  --ref-load "${REF_LOAD}" \
  --load "${REF_LOAD}" \
  --save "${SLIME_SAVE}" \
  --save-interval 999 \
  --prompt-data "${PROMPT_DATA}" \
  --input-key question \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  --num-rollout 5 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 1 \
  --global-batch-size 256 \
  --balance-data \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef 0.00 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.00 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  --sglang-mem-fraction-static 0.85 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-p2p-weight-update \
  --no-load-optim \
  ${MODEL_ARGS[@]} \
  > >(tee "${LOG_FILE}") 2> >(tee "${ERR_FILE}" >&2)

EXIT_CODE=$?
echo "P2P smoke 5-step finished at $(date -Is) exit_code=${EXIT_CODE}"
echo "LOG_FILE=${LOG_FILE}"
echo "ERR_FILE=${ERR_FILE}"
exit "${EXIT_CODE}"
