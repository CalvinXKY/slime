#!/bin/bash
# Run P2P 50-step validation, then NCCL-broadcast baseline 50-step, inside one container.
set -euo pipefail

CONTAINER="${CONTAINER:-slime-dev6}"
P2P_SRC="${P2P_SRC:-/data/nfs_87/xky/slime_p2p/feat-p2p-shard-weight-update}"
BASELINE_SRC="${BASELINE_SRC:-/data/nfs_87/xky/slime_baseline/main}"
LOG_ROOT="${LOG_ROOT:-/data/nfs_87/xky/new_rl/logs/p2p_experiments}"
MASTER_LOG="${LOG_ROOT}/sequential_50step_p2p_then_baseline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_ROOT}"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=== sequential 50-step P2P then baseline | $(date -Is) ==="
echo "CONTAINER=${CONTAINER}"
echo "P2P_SRC=${P2P_SRC}"
echo "BASELINE_SRC=${BASELINE_SRC}"
echo "MASTER_LOG=${MASTER_LOG}"

run_in_container() {
  docker exec "${CONTAINER}" bash -lc "$1"
}

echo "=== [1/4] sync P2P branch to /root/slime ==="
run_in_container "rsync -a --delete '${P2P_SRC}/' /root/slime/ && sed -i 's/\r$//' /root/slime/scripts/p2p/*.sh"

echo "=== [2/4] P2P 50-step training ==="
run_in_container "bash /root/slime/scripts/p2p/run_p2p_tp4_qwen3_4b_50step.sh"
P2P_RC=$?
echo "P2P exit_code=${P2P_RC}"
if [ "${P2P_RC}" -ne 0 ]; then
  echo "ERROR: P2P 50-step failed, skip baseline"
  exit "${P2P_RC}"
fi

echo "=== [3/4] sync baseline main (unmodified slime) to /root/slime ==="
run_in_container "rsync -a --delete '${BASELINE_SRC}/' /root/slime/ && sed -i 's/\r$//' /root/slime/scripts/p2p/*.sh 2>/dev/null || true"

echo "=== [4/4] NC baseline 50-step training ==="
run_in_container "bash /root/slime/scripts/p2p/run_nc_baseline_tp4_qwen3_4b_50step.sh"
BASELINE_RC=$?
echo "Baseline exit_code=${BASELINE_RC}"

echo "=== all done at $(date -Is) P2P=${P2P_RC} Baseline=${BASELINE_RC} ==="
exit "${BASELINE_RC}"
