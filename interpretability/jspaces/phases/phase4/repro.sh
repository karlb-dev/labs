#!/usr/bin/env bash
set -euo pipefail

PHASE4_ROOT="$(cd "$(dirname "$0")" && pwd)"

python -m pip install -q -e "$PHASE4_ROOT/../phase2"
python -m pip install -q -e "$PHASE4_ROOT/../phase3"
python -m pip install -q -e "$PHASE4_ROOT"

python -m pytest "$PHASE4_ROOT/tests" -q

python - "$PHASE4_ROOT/constraints.txt" <<'PY'
import json
import sys

from jspace_phase4.manifests import environment_payload, verify_constraints

required = {
    "huggingface_hub", "matplotlib", "numpy", "pandas", "pyarrow",
    "pyyaml", "scipy", "tokenizers", "torch", "transformers",
}
verification = verify_constraints(sys.argv[1], package_names=required)
print(json.dumps(verification, indent=1))
if not verification["ok"]:
    raise SystemExit("Phase 4 dependency lock mismatch")
environment = environment_payload()
print(json.dumps({
    "torch": environment["torch"],
    "torch_cuda_build": environment["torch_cuda_build"],
    "torch_cuda_available": environment["torch_cuda_available"],
    "gpu": environment["gpu"],
}, indent=1))
PY

python -m jspace_phase4 verify
echo "jspace_phase4 installed; conformance and live-evidence gates passed."
