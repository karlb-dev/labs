# Phase 4 reproduction guide

**FROZEN RELEASE — CPU VERIFICATION ONLY — NO MODEL-SCALE REPRODUCTION IS
AUTHORIZED FROM THIS GUIDE**

Frozen 2026-08-04 with tag `jspace-phase4-frozen-v1`.

## What reproduces without a GPU

Everything decisional: registry resolution, output hashing, branch-router
re-derivation, durability, inventory, and the full test suites. Model-scale
producers (lens fits, functional gates, influence, interventions) are
frozen history; rerunning them is out of scope for this release and would
require a new prospective protocol.

## Environment

Python 3.12; `pip install -e` each package. Pinned versions:
`interpretability/jspaces/phases/phase4/constraints.txt` (torch 2.11.0+cu128 CPU
build acceptable for verification; CUDA is not required). The campaign
Drive must be mounted at `/content/drive/MyDrive` (or set
`JSPACE4_RUN_ROOT`, `JSPACE_DRIVE_ROOT` accordingly); Drive-resident
registered outputs verify only where that archive is available.

## Verification sequence

```bash
cd /content/labs
git fetch origin --tags
git switch --detach jspace-phase4-frozen-v1

# 1. Package suites (also installs editable packages)
bash interpretability/jspaces/sidelines/gemma/repro.sh
bash interpretability/jspaces/sidelines/olmo/repro.sh
bash interpretability/jspaces/phases/phase4/repro.sh   # exits 1: one known deficit

# 2. Whole-registry durability with the known-deficit policy
python -m jspace_phase4.durability \
  --known-deficits interpretability/jspaces/phases/phase4/protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json \
  --pass-label reproduction \
  --output /tmp/phase4_durability_reproduction.json

# 3. Pre-freeze inventory gates
python -m jspace_phase4.pre_freeze_inventory --pass-label reproduction \
  --json-output /tmp/phase4_inventory.json \
  --markdown-output /tmp/phase4_inventory.md
```

Expected results at the tag:

- Gemma: 71 tests pass; verify ok (23 live events / 71 outputs).
- OLMo: 90 tests pass; verify ok (37 live events / 163 outputs); release
  boundary verify ok.
- Phase 4: 302 tests pass; `python -m jspace_phase4 verify` exits 1 with
  exactly one failure — the permanent A120–A250 `state.json` deficit.
- Durability: 521 references, 520 verified, 1 known deficit, 0 unexpected,
  0 pin conflicts; resolution modes {literal: 515,
  repository-materialization: 3, append-only-registry-prefix: 2}.
- Inventory: `NOT_REVIEW_READY` mechanically, solely on
  `all_live_outputs_verified=false` from the known deficit; every other
  gate passes. This is the expected, policy-aware-clean terminal state.

## Re-deriving the terminal decision

The canonical route recomputes from registered gates alone:

```bash
python - <<'PY'
import json, pathlib
root = pathlib.Path('interpretability/jspaces/phases/phase4')
# The registered canonical decision output (hash-pinned in the registry):
# .../canonical_lens_decision/p4-qwen-canonical-lens-decision-a1000-dev-v1/
# canonical_lens_decision.json  -> payload.canonical_branch == "Q-L4"
PY
```

Follow `reviews/QWEN_A1000_CANONICAL_REVIEW_PACKET.md` and the frozen
producer `jspace_phase4/experiments/p4_qwen_canonical_lens_decision.py` with
config `configs/p4_qwen_canonical_lens_decision_a1000_dev.yaml`; the
independent review (`reviews/PHASE4_INDEPENDENT_REVIEW_20260804.md`)
records a from-scratch reconstruction.

## Import verification

```bash
python -m jspace_phase4.experiments.p4_import_sidelines_study2 \
  --config interpretability/jspaces/phases/phase4/configs/p4_import_gemma_transport_study2.yaml \
  --validate --output /tmp/gemma_study2_revalidation.json
```

After the freeze this exits with "already exists" at the duplicate gate —
the registered event carries the same pins; compare against
`reports/gemma_transport_study2_import_validation_v1.json`. The live tests
in `tests/test_sidelines2_imports.py` perform the registered-event pin
checks automatically.

## Read-only rules

The tag is immutable. Post-freeze analysis lives on a new branch under the
paper-analysis namespace and never writes into Phase 4 registered output
paths. Erratum events require separate authorization.
