#!/bin/bash
# Deploy Slime P2P code into a container and apply the SGLang patch.
set -euo pipefail

CONTAINER="${CONTAINER:-slime-dev6}"
SLIME_SRC="${SLIME_SRC:-/data/nfs_87/xky/slime_p2p/feat-p2p-shard-weight-update}"
SLIME_DST="/root/slime"

echo "=== patch sglang model_runner in ${CONTAINER} ==="
docker exec "${CONTAINER}" bash "${SLIME_DST}/scripts/p2p/apply_sglang_p2p_patch.sh"

echo "=== launch P2P timing multi (nohup) ==="
docker exec "${CONTAINER}" bash -lc "nohup bash ${SLIME_DST}/scripts/p2p/run_p2p_tp4_qwen3_4b_timing_multi.sh > /tmp/p2p_timing_multi_launcher.log 2>&1 & echo PID=\$!"
