"""Hard GPU gates for model-scale Gemma and OLMo producers."""
from __future__ import annotations

import subprocess

import torch


def require_cuda(*, minimum_total_gib: float = 70.0) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: model-scale execution is forbidden")
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 2**30
    if total_gib < minimum_total_gib:
        raise RuntimeError(
            f"GPU has {total_gib:.1f} GiB; requires at least {minimum_total_gib:.1f} GiB"
        )
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    _ = x @ x
    torch.cuda.synchronize()
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0].strip()
    return {
        "name": props.name,
        "capability": [props.major, props.minor],
        "total_memory_gib": total_gib,
        "driver": driver,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }
