#!/usr/bin/env bash
set -euo pipefail

OL_ROOT="$(cd "$(dirname "$0")" && pwd)"

python -m pip install -q -e "$OL_ROOT/../../phases/phase2"
python -m pip install -q -e "$OL_ROOT/../../phases/phase3"
python -m pip install -q -e "$OL_ROOT/../../phases/phase4"
python -m pip install -q -e "$OL_ROOT"

python -m pytest "$OL_ROOT/tests" -q
python -m jspace_olmo_lineage verify

FINAL_BUNDLE="${JSPACE_OLMO_RUN_ROOT:-/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801}/release/IMPORT_BUNDLE_PHASE4.json"
if [[ -f "$FINAL_BUNDLE" ]]; then
  python -m jspace_olmo_lineage.experiments.final_release \
    --config "$OL_ROOT/configs/ol_final_release_v1.yaml" --verify
fi

python - "$OL_ROOT/constraints.txt" <<'PY'
import json
import sys
from jspace_olmo_lineage.manifests import verify_constraints

required = {
    "huggingface_hub", "matplotlib", "numpy", "pandas", "pyarrow",
    "pyyaml", "scipy", "tokenizers", "torch", "transformers",
}
result = verify_constraints(sys.argv[1], package_names=required)
print(json.dumps(result, indent=1))
if not result["ok"]:
    raise SystemExit("OLMo-lineage dependency lock mismatch")
PY

echo "jspace_olmo_lineage installed; tests and registry verification passed."
