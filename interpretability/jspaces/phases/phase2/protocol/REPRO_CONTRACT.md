# REPRO CONTRACT — how every piece of J-space Part-2 evidence is produced

Adopted 2026-07-27 (user directive + addendum §5.1/§13). Binding for all
work after this commit. The exploratory era (`special_lab2/`, run-dir cells
tagged `exploratory`) is grandfathered with best-effort provenance and is
superseded, not retrofitted.

## The one-sentence rule

**No result exists unless a clean VM + `git clone` + basic deps can
recreate it with one command, and the artifact itself says exactly which
code, config, inputs, and model revisions produced it.**

## Evidence unit

Every claim-bearing artifact (metrics JSON, per-item parquet, figure,
table) is an **evidence item** with an `evidence_id` registered in
`reports/evidence_registry.jsonl` (append-only, in git). A registry row
holds: evidence_id, tier (exploratory | pilot | confirmatory), one-line
what-it-shows, producing command, config path+hash, code commit, input
hashes, output path(s)+hash(es), GPU-hours class, and repro notes
(deterministic vs tolerance-bounded).

## Provenance block (embedded in every result file)

```json
"provenance": {
  "study_id": "jspace-part2",
  "evidence_id": "r1-protected-dynamic-pilot-01",
  "tier": "pilot",
  "code_commit": "<git sha>", "dirty_tree": false,
  "package_version": "jspace_part2 x.y.z",
  "command": "jspace-part2 run-experiment --config configs/....yaml",
  "config_sha256": "...",
  "inputs": {"lens": "sha256:...", "prompts": "sha256:...", "...": "..."},
  "model": {"id": "...", "revision": "<resolved sha>", "config_sha256": "..."},
  "jlens_commit": "581d3986...",
  "seed": 0, "created_utc": "...", "producer": "jspace_part2.<module>"
}
```

Producers refuse to run if the git tree is dirty (override flag
`--allow-dirty` exists for development and stamps `dirty_tree: true`,
which disqualifies the output from confirmatory tier).

## Single-command reproduction

From a fresh VM:

```bash
git clone git@github.com:karlb-dev/labs.git && cd labs
bash interpretability/jspaces/phases/phase2/repro.sh <evidence-id>
```

`repro.sh` = pip-install the package (pinned deps) → environment audit →
resolve the registry row → fetch-or-refit declared heavy inputs → run the
producing command → verify output hashes (exact for deterministic CPU
paths; preregistered numeric tolerances for GPU nondeterminism, recorded
per evidence item).

## Artifact classes

| class | lives where | repro path |
|---|---|---|
| code, configs, tests, docs, registry | git | clone |
| small results (<5 MB JSON/parquet/figures) | git (mirror) + Drive run dir | regenerate or verify hash |
| lenses, traces, layer states, raw generations | Drive run dir (hash in registry) | `--fetch` from Drive/release when available, else `--refit` (documented cost, resumable) |
| model weights | HF hub, pinned revision SHA | auto-download |

Nothing scientific may exist ONLY on an undocumented Drive path.

## Resumability (24 h GPU blocks)

Every GPU phase checkpoints at ≤10-minute granularity and resumes exactly
(same command re-run). The live queue in `inprogress.md` is ordered so any
prefix fits a 24 h block; a VM death mid-phase costs at most the last
checkpoint interval. Interrupted-vs-uninterrupted equivalence is a golden
test (R6) for every accumulator that feeds a result.

## Supersession

When an instrument is repaired, prior results are not edited — they keep
their tier and a `superseded_by: <evidence_id>` field is appended to their
registry row. Reports cite the superseding item; errata stay visible.

## Directory contract

```
interpretability/jspaces/phases/phase2/        # in git — the package
  jspace_part2/                       # code (lib, provenance, adapters, experiments, analysis)
  configs/                            # typed YAML configs (paths, pins, phases)
  tests/                              # unit + golden tests (CPU)
  protocol/                           # this contract, specs, crosswalk
  reports/                            # evidence_registry.jsonl + rendered reports
  repro.sh                            # the single-command entry
Drive: special-lab-1/part2_20260727/
  manifests/  raw/  derived/  confirmatory/   # added to the existing layout
  (metrics/ figures/ lens/ logs/ code/ = exploratory era, frozen semantics)
```
