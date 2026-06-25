#!/usr/bin/env python3
"""
Runtime patch for SGLang ``model_runner.py`` (presharded P2P recv path).

Prefer ``docker/patch/latest/sglang_p2p.patch`` applied via
``scripts/p2p/apply_sglang_p2p_patch.sh``. This script remains for ad-hoc
containers where git-apply is unavailable; it patches only ``model_runner.py``
and does not update ``io_struct.py`` / ``tp_worker.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_PATH = Path("/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py")


def main() -> None:
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    content = filepath.read_text(encoding="utf-8")

    already_patched = (
        "tp_tensor_counts" in content
        and 'load_format == "presharded"' in content
        and "torch.distributed.recv" in content
        and "torch.distributed.barrier(group=pg)" in content
        and 'device_id=torch.device("cuda", self.gpu_id)' in content.split("def update_weights_from_distributed")[0].split("init_weights_update_group")[-1]
    )
    if already_patched:
        print(f"SGLang model_runner already patched for P2P: {filepath}")
        return

    has_partial_patch = (
        "_set_presharded_flag" in content
        and 'load_format == "presharded"' in content
    )
    if has_partial_patch:
        print(f"Resetting {filepath} before applying P2P patch")
        import subprocess

        sglang_root = filepath.parents[4]
        subprocess.run(
            ["git", "checkout", "--", str(filepath.relative_to(sglang_root))],
            cwd=sglang_root,
            check=True,
        )
        content = filepath.read_text(encoding="utf-8")

    old_sig = """    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
        delta: Optional[str] = None,
    ):"""

    new_sig = """    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
        delta: Optional[str] = None,
        src_tp_rank: Optional[int] = None,
        tp_tensor_counts: Optional[list] = None,
    ):"""

    if old_sig in content:
        content = content.replace(old_sig, new_sig, 1)
        print("Patched update_weights_from_distributed signature")

    old_block = """        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online.\""""

    new_block = """        try:
            pg = self._model_update_group[group_name]

            if load_format == "presharded":
                if tp_tensor_counts is not None:
                    my_idx = self.tp_rank
                    offset = sum(tp_tensor_counts[:my_idx])
                    my_count = tp_tensor_counts[my_idx]
                    my_names = names[offset:offset + my_count]
                    my_dtypes = dtypes[offset:offset + my_count]
                    my_shapes = shapes[offset:offset + my_count]
                else:
                    my_names = names
                    my_dtypes = dtypes
                    my_shapes = shapes
                torch.distributed.barrier(group=pg)
                weights = []
                src_rank = self.tp_rank
                for name, dtype, shape in zip(my_names, my_dtypes, my_shapes):
                    target_dtype = (
                        dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                    )
                    weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                    torch.distributed.recv(weight, src=src_rank, group=pg)
                    weights.append((name, weight))
                _set_presharded_flag(self.model, True)
                try:
                    self.model.load_weights(weights)
                finally:
                    _set_presharded_flag(self.model, False)
            else:
                weights = []
                handles = []
                for name, dtype, shape in zip(names, dtypes, shapes):
                    target_dtype = (
                        dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                    )
                    weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                    handles.append(
                        torch.distributed.broadcast(
                            weight,
                            src=0,
                            group=pg,
                            async_op=True,
                        )
                    )
                    weights.append((name, weight))
                for handle in handles:
                    handle.wait()

                self.model.load_weights(weights)

            return True, "Succeeded to update parameter online.\""""

    if old_block not in content:
        print(f"ERROR: broadcast block not found in {filepath}")
        sys.exit(1)

    content = content.replace(old_block, new_block, 1)

    old_init = """            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )"""

    new_init = """            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )"""

    if old_init in content and "device_id" not in content.split("init_weights_update_group")[1].split("def ")[0]:
        content = content.replace(old_init, new_init, 1)
        print("Patched init_weights_update_group to pass device_id")

    helper_func = '''

def _set_presharded_flag(model, value: bool):
    for module in model.modules():
        if hasattr(module, "use_presharded_weights"):
            module.use_presharded_weights = value
    for _name, param in model.named_parameters():
        if hasattr(param, "use_presharded_weights"):
            param.use_presharded_weights = value

'''

    class_marker = "\nclass ModelRunner(ModelRunnerKVCacheMixin):"
    if class_marker in content:
        content = content.replace(class_marker, helper_func + class_marker, 1)
    else:
        content += helper_func

    filepath.write_text(content, encoding="utf-8")
    print(f"Applied P2P patch to {filepath}")


if __name__ == "__main__":
    main()
