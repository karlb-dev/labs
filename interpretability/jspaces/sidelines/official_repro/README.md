# official_repro — Anthropic released-materials reproduction, Study 1

Runs the released Jacobian-lens evaluations (6 sets) and experiment prompt
sets (11 sets) from `anthropics/jacobian-lens` @ `581d3986` on
**Qwen 3.6 27B** (campaign-validated published lens) and
**OLMo 3.1 32B Instruct** (one prospectively fitted official-estimator
lens), then separates prompt, lens, intervention, capability, and model
effects with a bounded instrument cross-over.

- Governing plan: `MyDrive/interpret/jspace_lab_official_repro_1.md` +
  `jspaces_lab_official_repro_1_addendum.md` (addendum §2 wins on conflict).
- Branch: `interp_jspace_official_repro_1` · evidence prefix `or1-` ·
  tier: development/methods throughout (no confirmatory language).
- Drive root: `MyDrive/interpret/special-lab-1/official_repro_1_20260808/`.
- Local scratch: `/content/or1_work/` (`JSPACE_OR1_LOCAL_WORK`).

## This study is not

Phase 5, a reopening of any frozen campaign result, a Claude test, or a
promise that every paper cell is reproducible from public materials.
Frozen Phase 1–4 / sideline / paper-analysis registries are read-only.

## Quickstart

```bash
pip install -e interpretability/jspaces/sidelines/official_repro
# immutable engine clone (never edited):
git clone https://github.com/anthropics/jacobian-lens /content/or1_work/jacobian-lens
git -C /content/or1_work/jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
pip install -e /content/or1_work/jacobian-lens
export JLENS_ROOT=/content/or1_work/jacobian-lens

jspace-or1 selftest        # CPU conformance
jspace-or1 registry-list   # what or1- evidence exists
jspace-or1 verify          # rehash every registered output
```

## Layout

`external/` — byte-identical vendored copy of the upstream reproducibility
surface (prompts, data READMEs, LICENSE), hash-pinned in
`external_record_manifest.json`. `protocol/` — frozen pre-data contracts.
`preregistration/` — Study-1 preregistration, reference targets,
deviations. `jspace_official_repro/` — the package. `reports/` — registry
+ state of record. `release/` — terminal bundles.

## Fidelity vocabulary

Every result row carries one class: `R0` exact-released ·
`R1` reconstructed-from-paper · `R2` deterministic-adaptation ·
`R3` not-identified-from-release. Gated cells are **missing, never zero**.
