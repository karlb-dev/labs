# Environment audit + lock: what exactly is this VM running?
# Usage: jspace-part2 audit-env [--out <dir>]
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from .provenance import git_info


def main(argv: list[str]) -> None:
    out_dir = Path(argv[argv.index("--out") + 1] if "--out" in argv else
                   "/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    lock_sha = hashlib.sha256(freeze.encode()).hexdigest()
    (out_dir / f"pip_freeze_{lock_sha[:12]}.txt").write_text(freeze)
    audit = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "python": sys.version, "platform": platform.platform(),
             "git": git_info(), "pip_freeze_sha256": lock_sha}
    try:
        import torch
        audit["torch"] = torch.__version__
        audit["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            audit["gpu"] = torch.cuda.get_device_name(0)
            audit["cuda"] = torch.version.cuda
    except Exception as e:
        audit["torch_error"] = repr(e)
    try:
        import transformers
        audit["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        jl = subprocess.run(["git", "-C", "/content/jacobian-lens",
                             "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
        audit["jlens_commit"] = jl or None
    except Exception:
        audit["jlens_commit"] = None
    path = out_dir / "environment_audit.json"
    path.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main(sys.argv)
