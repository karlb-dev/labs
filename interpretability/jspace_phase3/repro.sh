#!/usr/bin/env bash
# One command from clean clone to working Phase 3 package:
#   bash interpretability/jspace_phase3/repro.sh
# Installs both study packages (Phase 3 imports Phase 2's stable
# utilities), runs the conformance tests, and audits the environment.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

pip install -q -e "$HERE/../jspace_part2"
pip install -q -e "$HERE"

python -m pytest "$HERE/tests" -q

python - <<'EOF'
import torch, transformers, importlib.metadata as md
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
for p in ("jspace_part2", "jspace_phase3"):
    print(p, md.version(p))
from jspace_phase3.paths3 import run_root
try:
    print("run root:", run_root(create=False))
except RuntimeError as e:
    print("run root: UNSET (", e, ")")
EOF
echo "jspace_phase3 installed; conformance tests green; environment audited."
