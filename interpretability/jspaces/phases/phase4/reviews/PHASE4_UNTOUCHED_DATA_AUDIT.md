# Phase 4 untouched-data audit — 2026-08-04

**INDEPENDENT REVIEWER OUTPUT — NARRATIVE-BLIND SESSION**

Auditor: Claude general-purpose review subagent, session 2026-08-04
(same session as `reviews/PHASE4_INDEPENDENT_REVIEW_20260804.md`).
Commit audited: `8762a9a57cd67044b6520ee9b7a17c57aec50792`
(branch `interp_jspace_phase4_5`). Drive mount audited:
`/content/drive/MyDrive/interpret/special-lab-1/` (read-only), phase4 run
root `phase4_20260731/`. Blindness: no paper/, plan, addendum, INPROGRESS,
RESUME, EXECUTION_RECORD, or development-report document was opened.

Question audited: does any Phase 4 confirmatory or replication intervention
outcome exist anywhere — in the three registries, or on the phase4 run root —
and is every on-disk artifact accounted for by registered development/methods
events or written incident records?

## 1. Registry tier scan (all three registries)

Method: parsed every JSONL row of
`interpretability/jspace_phase4/reports/evidence_events.jsonl` (82 rows),
`interpretability/jspace_gemma/reports/evidence_events.jsonl` (24 rows),
`interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl`
(41 rows); counted `tier` values and searched every tier string for
"confirmatory"/"replication" (case-insensitive); additionally counted raw
substring occurrences in the phase4 file.

Results:

- phase4 tiers: `phase3-confirmatory-import` 1, `methods` 22,
  `phase4-development` 44, `side-development-import` 5, (no tier field on
  the 10 supersede/withdraw/correct bookkeeping rows).
- Raw string counts in the phase4 registry: `phase4-confirmatory` **0**,
  `phase4-replication` **0**.
- The only "confirmatory" hit anywhere is the tier
  `phase3-confirmatory-import` on `p4-import-phase3-release-v1` — the
  immutable Phase 3 completion-boundary import (manifest files only), not a
  Phase 4 outcome.
- gemma tiers: `methods` 14, `historical-development-import` 9 — zero
  confirmatory/replication.
- olmo tiers: `methods` 25, `development` 13 — zero
  confirmatory/replication.

Also verified: no event in the phase4 registry has a native side-study
evidence_id (`gm-*`, `gm2-*`, `ol-*`, `ol2-*`): regex over all 82
`evidence_id` fields → **0 matches** (such IDs appear only inside
`source_evidence_id(s)`/`inputs` metadata of the six import events, as the
import contract requires).

## 2. Full file accounting of the run-root metrics tree

Command (exact logic): collected every `outputs[].path` and
`source_outputs[].path` from all phase4 registry rows; walked
`phase4_20260731/metrics/` with `os.walk`; set-differenced.

Results: **296 files on disk**, 274 of them registered output paths.
The 23 non-registered files are each individually accounted for:

- **17 operational `state.json` files + 1 `input_manifest.json`** inside
  registered evidence directories (OLMo lineage grids/g5 banks, bank-w
  capability). These are producer resume/bookkeeping files of registered
  development events; none contains a new outcome; later Qwen events
  register their state files explicitly.
- **2 preserved blocked prompt-323 null states**:
  `…/p4-qwen-lens-influence-prompt323-dev-v1__blocked_20260803T044939Z_runtime-identity/analysis_state.json`
  and
  `…__blocked_20260803T045815Z_prompt112-control-repeat/analysis_state.json`.
  I hashed both: each is exactly **1,146 bytes**, SHA-256
  `4eef3124ae3259bf…` (identical), with `contribution: null` and zero
  completed layers — byte-matching the registered runtime block record
  (`reports/PHASE4_PART4_PROMPT323_RUNTIME_BLOCK.md`). No influence value
  exists in them; they were never promoted.
- **4 files in one withdrawn functional-gate directory**
  `…functional_gate/p4-…-a500-a1000-published-dev-v1__withdrawn_20260803T030707Z_tied_topk_observer/`.
  All four SHA-256 values match the written incident record
  (`reports/PHASE4_PART4_FUNCTIONAL_INSTRUMENTATION_INCIDENT.md`):
  state.json `8d730e36…`, fixed_endpoint_activations.pt `ad37f972…`,
  fixed_capacity_activations.pt `e441f16b…`,
  capacity_reconstructions_published.pt `30bf51e9…`. The run hard-stopped on
  the FIRST A500 primary item (top-k tie observer invariant) before any
  primary row; the three cache files are byte-identical to the caches of the
  registered re-run (same hashes in the registered event), i.e. the
  recomputation reproduced them exactly. No unregistered outcome exists in
  this directory.

Verdict for this section: **no unaccounted intervention-outcome directory or
file exists under the run-root metrics tree.**

## 3. P4-P2 producer output directory

Command: `find <run root>/metrics/qwen36-27b-mode-variance-pilot -type f | wc -l`

Result: directory exists as an **empty scaffold — 0 files, no
subdirectories**. The producer's output path would be
`metrics/qwen36-27b-mode-variance-pilot/mode_variance_pilot/<evidence_id>/`
(from `p4_qwen_mode_variance_gpu.py::_output_dir`); no such subdirectory was
ever created. The GPU pilot config still carries unbound placeholders
(`BIND_REGISTERED_CANONICAL_DECISION_SHA256`,
`BIND_REGISTERED_A1000_SHA256`) and `permitted_branches: [Q-L1, Q-L2]`,
which Q-L4 makes unsatisfiable. **The P4-P2 producer never ran.**

## 4. A2000 / n2000 artifacts

Commands:
`grep -ci "a2000\|n2000\|n=2000" reports/evidence_events.jsonl` → **0**;
`grep -rli "a2000" configs/ jspace_phase4/` → no files;
`find <run root> -maxdepth 2 -iname "*2000*"` → nothing;
`find <run root>/lens -iname "*2000*"` → nothing.
`ls <run root>/lens/qwen36-27b/nested_fit/draw_a/` shows exactly
`qwen36-27b_jlens_drawA_n0120.pt`, `n0250.pt`, `n0500.pt`, `n1000.pt` plus a
`recovery/` directory of fit checkpoints (engineering state; includes the
n195/n198 adjacent-checkpoint contract artifacts used by the registered
equal-weight assertion). **No A2000 branch or artifact exists.**

## 5. Canonical decision output directory

Command: `find <run root>/metrics/qwen36-27b-canonical-lens-decision-a1000 -type f`

Result: exactly three files —
`canonical_lens_decision.json`, `input_manifest.json`,
`CANONICAL_LENS_DECISION.md` — and all three SHA-256 values match the
registered event (`549a3f42…`, `7afe8d93…`, `bb3c31b4…`). **Nothing else is
present.**

## 6. Figures and other run-root directories

Set-difference of `<run root>/figures/` against registered paths: 52 files,
2 non-registered — `p4f14_bank_w_power.{png,pdf}` — both **byte-identical**
(SHA-256 equal) to the registered repo outputs of `p4-bank-w-power-dev-v1`
at `interpretability/jspace_phase4/reports/figures/`; they are convenience
duplicates of registered methods artifacts, not outcomes.
`diagnostics/` contains only the named incident archives
(`prompt323_runtime_identity_20260803`, `qwen_a1000_fit`); `closeout/`
contains the packet/import snapshots. Neither contains an intervention
outcome directory.

## 7. Untouched partitions

The Bank-B (10 confirmatory / 10 replication families) and Bank-W v3
(28 confirmatory / 20 replication families) untouched sides exist only as
hash-pinned partition payloads inside registered outcome-blind candidate
events (partition payload SHA-256 `361acad0…` for Bank W;
`b88f3ae8…` bank payload for Bank B). No producer output, row store, or
event reads them: the registries contain zero intervention events over any
confirmatory/replication family, and the metrics sweep (section 2) found no
directory for one.

## 8. Execution checks

- `python -m pytest interpretability/jspace_phase4/tests -q` →
  **302 passed, 0 failed** (9.4 s).
- `python -m jspace_phase4 verify` → **exit code 1**; JSON reports 65 live
  events, 522 checked output references, resolutions 516 literal-path + 3
  repository-materialization + 2 append-only-registry-prefix, and exactly
  one failure: the known A120–A250 `state.json`
  (expected `361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8`,
  status missing). This equals the declared permanent deficit and nothing
  else fails.

## Verdict

**PASS.** No Phase 4 confirmatory or replication intervention outcome exists
in any registry or anywhere on the phase4 run root; every non-registered
byte on the run root is accounted for by a registered event's operational
state, a written and hash-matching incident/block record, or a byte-identical
duplicate of a registered artifact; the P4-P2 producer directory is empty;
no A2000 artifact exists; the canonical decision directory is exactly its
three registered files; the untouched partitions remain sealed. The single
verification failure is the declared known permanent deficit, which is an
absence (a missing historical operational file), not an outcome.
