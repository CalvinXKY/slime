#!/bin/bash
# Deploy Slime P2P code to the verification container and apply the SGLang patch.
set -ex
export http_proxy=http://10.155.96.7:3128
export https_proxy=http://10.155.96.7:3128
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
git config --global http.sslVerify false

rsync -a /data/nfs_87/xky/slime_p2p/slime-p2p-verify/ /root/slime/
find /root/slime/scripts/p2p -name '*.sh' -exec sed -i 's/\r$//' {} +
chmod +x /root/slime/scripts/p2p/*.sh

echo "=== file checksums ==="
md5sum /root/slime/slime/backends/megatron_utils/update_weight/update_weight_from_distributed_p2p.py
md5sum /root/slime/docker/patch/latest/sglang_p2p.patch

echo "=== apply SGLang P2P patch ==="
bash /root/slime/scripts/p2p/apply_sglang_p2p_patch.sh

echo "=== verify patch applied ==="
grep -c tp_tensor_counts /sgl-workspace/sglang/python/sglang/srt/managers/io_struct.py
grep -c 'load_format == "presharded"' /sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py

echo "=== deploy OK ==="
