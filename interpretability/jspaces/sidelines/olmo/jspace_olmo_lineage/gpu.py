"""Same-process CUDA hard gates for every model-backed action."""
from __future__ import annotations

import subprocess

import torch


def require_cuda_gpu() -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is required; CPU model fallback is forbidden")
    probe = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    product = probe @ probe
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0].strip()
    result = {
        "name": properties.name,
        "capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "driver_version": driver,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "probe_device": str(product.device),
        "probe_shape": list(product.shape),
    }
    del probe, product
    return result


def assert_model_on_cuda(model) -> None:
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise RuntimeError("loaded model has no parameters") from error
    if parameter.device.type != "cuda":
        raise RuntimeError("model parameter is not on CUDA")
