# J-spaces

Open-weights Jacobian-lens / verbalizable-channel campaign (Lab 37 through
Phase 4, sidelines, and paper analysis).

**Latest product package:** `phases/phase4/`  
Install: `pip install -e interpretability/jspaces/phases/phase4`  
CLI: `jspace-phase4`

Run artifacts are **not** in git. Point env vars at your local or Drive run
root (see Env vars below). Local unpacks historically live at
`interpretability/jspace_runs/` (gitignored).

## Layout

```text
interpretability/jspaces/
  phases/
    phase1/            # Lab 37 exploratory (was interpretability/jspace)
      part2_exploratory/  # historical pre-package Part-2 mirror (see phase2)
    phase2/            # confirmatory package (was jspace_part2)
    phase3/            # mechanism / generalization
    phase4/            # frozen campaign + latest installable package
    paper_analysis/    # offline paper analysis + manuscript drafts (post-freeze)
  sidelines/
    gemma/             # Gemma transport studies
    olmo/              # OLMo lineage studies
  plans/               # complete campaign plan library (see plans/README.md)
```

Each phase/sideline is self-contained: Python package + configs + tests +
registry/release materials + `repro.sh` where applicable.

Python **import names are unchanged** (`jspace_phase4`, `jspace_part2`, …)
so scientific code and tests stay stable. Only monorepo paths moved.

## Old path → new path

Frozen PDFs, handouts, and registered reports may still print the old
paths. Use this table (also applied by path resolvers for `repo://` URIs):

| Old (pre-reorg) | New |
| --- | --- |
| `interpretability/jspace/` | `interpretability/jspaces/phases/phase1/` |
| `interpretability/jspace/part2/` | `interpretability/jspaces/phases/phase1/part2_exploratory/` |
| `interpretability/jspace_part2/` | `interpretability/jspaces/phases/phase2/` |
| `interpretability/jspace_phase3/` | `interpretability/jspaces/phases/phase3/` |
| `interpretability/jspace_phase4/` | `interpretability/jspaces/phases/phase4/` |
| `interpretability/jspace_gemma/` | `interpretability/jspaces/sidelines/gemma/` |
| `interpretability/jspace_olmo_lineage/` | `interpretability/jspaces/sidelines/olmo/` |
| `interpretability/jspace_paper/` | `interpretability/jspaces/phases/paper_analysis/` |

## Freeze tags (provenance)

These tags point at **pre-move** commits. The moved tree is the living
mirror; hashes of registered artifacts are the scientific anchor.

| Tag | Meaning |
| --- | --- |
| `jspace-part2-confirmatory-freeze-v1` / `complete-v1` | Phase 2 closed |
| `jspace-phase3-freeze-v1` / `complete-v1` | Phase 3 closed |
| `jspace-phase4-frozen-v1` | Phase 4 frozen; paper-analysis parent |

See [`ARCHIVE.md`](ARCHIVE.md).

## Install (CPU conformance)

From monorepo root:

```bash
pip install -e interpretability/jspaces/phases/phase2
pip install -e interpretability/jspaces/phases/phase3
pip install -e interpretability/jspaces/phases/phase4
# optional sidelines
pip install -e interpretability/jspaces/sidelines/gemma
pip install -e interpretability/jspaces/sidelines/olmo
```

Or per-package: `bash interpretability/jspaces/phases/phase4/repro.sh` (installs phase2+3+4).

## Env vars (names unchanged)

| Variable | Role |
| --- | --- |
| `JSPACE4_RUN_ROOT` | Phase 4 durable run root |
| `JSPACE4_LOCAL_WORK` | Phase 4 local scratch (not DriveFS) |
| `JSPACE3_RUN_ROOT` / `JSPACE3_LOCAL_WORK` | Phase 3 |
| `JSPACE_PART2_RUN_ROOT` / `JSPACE_PART2_OUT_ROOT` | Phase 2 |
| `JSPACE_GEMMA_RUN_ROOT` / `JSPACE_GEMMA_LOCAL_ROOT` | Gemma sideline |
| `JSPACE_OLMO_RUN_ROOT` / `JSPACE_OLMO_LOCAL_WORK` | OLMo sideline |
| `JSPACE_DRIVE_ROOT` | Campaign Drive / mirror of special-lab-1 |
| `JSPACE_MODEL_ROOT` / `HF_HUB_CACHE` | Model weights / hub cache |

Colab Drive paths remain optional defaults in some modules; non-Colab hosts
should set the env vars. Windows: use pathlib-friendly absolute paths; GPU
queue scripts are bash (Git Bash / WSL / Colab).

## Phase map (science)

| Phase | What it is |
| --- | --- |
| 1 | Exploratory OLMo-3-32B-Think workspace replication (Lab 37) |
| 2 | Assay repair + confirmatory matrix |
| 3 | Mechanism, bridge, generalization |
| 4 | Channel content, Qwen instrument ladder, lineage imports (frozen) |
| Sidelines | Gemma transport applicability; OLMo training-stage trajectory |
| Analysis | Offline claim synthesis and paper drafts after Phase 4 freeze |
