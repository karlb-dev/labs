# VM4 restore sanity: torch on the new GPU + version drift vs VM3's freeze.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import log

import torch
import transformers


def main() -> None:
    log(f"torch {torch.__version__} | transformers {transformers.__version__}")
    log(f"gpu {torch.cuda.get_device_name(0)} "
        f"cap {torch.cuda.get_device_capability(0)}")
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().abs().mean().item()
    log(f"bf16 matmul ok, mean |x@x| = {y:.2f}")


if __name__ == "__main__":
    main()
