#!/bin/bash
# Apply SGLang P2P support from docker/patch/latest/sglang_p2p.patch.
set -e

SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"
PATCH_FILE="${SLIME_ROOT}/docker/patch/latest/sglang_p2p.patch"

if [ ! -f "${PATCH_FILE}" ]; then
  echo "ERROR: patch not found: ${PATCH_FILE}"
  exit 1
fi

if grep -q 'tp_tensor_counts' "${SGLANG_ROOT}/python/sglang/srt/managers/io_struct.py" 2>/dev/null; then
  echo "SGLang P2P patch already applied"
  exit 0
fi

cd "${SGLANG_ROOT}"
if git apply --check "${PATCH_FILE}" 2>/dev/null; then
  git apply "${PATCH_FILE}"
elif git apply --3way "${PATCH_FILE}"; then
  echo "Applied with 3-way merge"
else
  echo "ERROR: failed to apply ${PATCH_FILE}"
  exit 1
fi
find "${SGLANG_ROOT}/python/sglang" -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "Applied SGLang P2P patch from ${PATCH_FILE}"
