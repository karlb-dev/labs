# Phase 0: environment audit for VM2 (VM1's audit is config/environment.json).
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import RUN_DIR, atomic_write_json, ensure_dirs, log

import torch
import transformers


def main() -> None:
    ensure_dirs()
    disk = subprocess.run(["df", "-BG", "/", str(RUN_DIR)], capture_output=True,
                          text=True).stdout
    props = torch.cuda.get_device_properties(0)
    env = {
        "vm": "vm2 (session resumed after VM1 died)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": {
            "name": props.name,
            "total_vram_gb": round(props.total_memory / 2**30, 1),
            "capability": f"{props.major}.{props.minor}",
        },
        "disk_df": disk,
        "hf_hub_cache": str(Path("/content/drive/MyDrive/hf_cache/hub")),
        "jlens_revision": subprocess.run(
            ["git", "-C", "/content/jacobian-lens", "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "seed": 0,
    }
    # Feasibility self-checks: abort loudly if the tier assumptions are wrong.
    assert env["gpu"]["total_vram_gb"] > 90, "expected ~96GB GPU for full-bf16 tier"
    pip = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True).stdout
    (RUN_DIR / "config" / "pip_freeze_vm2.txt").write_text(pip)
    atomic_write_json(env, RUN_DIR / "config" / "environment_vm2.json")
    log(f"wrote {RUN_DIR}/config/environment_vm2.json")


if __name__ == "__main__":
    main()
