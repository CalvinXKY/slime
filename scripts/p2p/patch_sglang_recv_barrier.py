#!/usr/bin/env python3
"""
Deprecated: NCCL recv barrier is included in ``docker/patch/latest/sglang_p2p.patch``.

Use ``scripts/p2p/apply_sglang_p2p_patch.sh`` instead. Running this script alone
can duplicate barriers and deadlock weight updates.
"""

import sys

if __name__ == "__main__":
    print(__doc__, file=sys.stderr)
    sys.exit(1)
