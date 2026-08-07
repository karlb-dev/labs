"""Hard CUDA gate for model-scale Phase 3 work."""
from __future__ import annotations

import subprocess

import torch


def require_cuda_gpu() -> dict:
    """Refuse CPU fallback and exercise CUDA in the current process."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible in this process. Do not use CPU fallback; "
            "relaunch this model job with host/unsandboxed GPU access.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    a = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    b = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    checksum = float((a @ b)[0, 0].float().cpu())
    del a, b
    torch.cuda.synchronize(device)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).splitlines()[0].strip()
    except Exception:
        driver = "unavailable"
    return {
        "device_index": 0,
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "capability": [int(props.major), int(props.minor)],
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "driver_version": driver,
        "smoke_checksum": checksum,
    }

