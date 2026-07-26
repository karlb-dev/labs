# Lab 37 (draft): J-space Global Workspace Replication — OLMo-3-32B-Think

> **Status: draft proposed lab, branch `interp_jspace`.**
> v1 pipeline complete (2026-07-26). **v2 delta run in progress** — this
> directory is refreshed at every phase boundary, so partial v2 results are
> normal here; `code/PLAN_v2.md` and the Drive `inprogress.md` say exactly
> what is running. Spec: [`labs/lab37_jspace_workspace.md`](../labs/lab37_jspace_workspace.md).

An open-weights replication of Anthropic's July 2026 paper *"Verbalizable
Representations Form a Global Workspace in Language Models"*
(transformer-circuits.pub/2026/workspace) on `allenai/Olmo-3-32B-Think`,
using the reference [`jacobian-lens`](https://github.com/anthropics/jacobian-lens)
package (pinned `581d3986`), plus a thinking-model extension the paper could
not run and an instrument-audit round (v2) that either flips or hardens the
causal verdict.

**Read first:** [`handout/olmo32b_jspace_handout.pdf`](handout/olmo32b_jspace_handout.pdf)
— the living writeup (prose + figures, updated per phase). Full v1 prose:
[`report/REPORT.md`](report/REPORT.md).

## Findings so far (v1 + v2 partial)

| Claim | Result | Evidence |
|---|---|---|
| Workspace geometry exists | **Replicates, ~10× thinner**: variance share ≤0.67% (paper 6–10%), ~6 active concepts (paper ~25), clean mid-band inverted-U (persistence 0.19–0.20 vs 0.03–0.08 at ends) | `results/v1_*`, figs f1/f2 |
| Broadcast fan-out | 73–94 downstream readers vs 0 random (p≈1e-24) — but non-J high-variance PCs read equally wide → **not J-specific** (variance-matched re-test = v2) | f8 |
| Causal dissociation (paper headline) | **Did not replicate in v1** at rank-matched doses; v1's controls were then shown **energy-mismatched ~10×** (J-span k=10 ≡ 25–33 random dims ≡ 1–2 deep PCs) — energy-matched grid running now | f4/f5, `v2_energy_match.json`, f10 |
| Live per-token top-10 ablation (paper's own protocol) | Lobotomy on OLMo (NLL 2.71→24.34, asterisk degeneration) — the live-computation confound; frozen prompt-selected variant = v2 P2 | f4 |
| Workspace anticipates CoT | **Survives foil calibration**: leads text by median 46 steps; answer detection 0.92 vs 0.06 frequency-matched noise floor; family foils fire concurrently (lead 3) | `v2_cot_foils.json`, f9 |
| Pre-CoT anticipation | **Null** — answer median rank 5613 before any thinking token; loads on demand at answer time (rank 211 suppressed; monotone collapse L30→L60) | f6, `v1_cot_lead.json` |
| Eval-awareness direction | Emerges p<0.001 all layers, semantically legible — but behaviorally inert and confounded by lexical echo | `v1_evalaware.json` |

The open question driving v2: **is the causal null OLMo or the harness?**
Decision tree in `code/PLAN_v2.md` — if clean instruments (energy matching,
frozen selection, late-band lens) still null, the same instruments go to
Qwen 3.6 27B via Neuronpedia's published lens to separate model from method.

## Layout

```
code/            pipeline (s0–s19) + sl1_common.py + PLAN.md / PLAN_v2.md / LOG.md
  scripts/       s0 env … s9 v1 report; s11 energy-match, s12 frozen ablation,
                 s13 foil calibration, s14/s15 late band, s19 v2 figures
figures/         f1–f8 (v1) + f9+ (v2), regenerable from metrics via s9/s19
handout/         living LaTeX writeup + compiled PDF (the primary read)
report/          v1 REPORT.md + summary.json; v2 summary_v2.json (grows)
results/         small metrics JSONs, prefixed v1_/v2_
```

**Heavy artifacts live on Google Drive, not in git** (fitted lenses ~1.1 GB
each, 24 MB descriptive aggregates, generation traces, per-layer PCA state):
`MyDrive/interpret/special-lab-1/{2026-07-25_1726, 2026-07-26_v2}/`. Every
figure regenerates from metrics alone; only lens-refit needs the Drive blobs.

## Reproducing

One 80 GB+ GPU (run history: RTX PRO 6000 Blackwell 96 GB), torch 2.11,
transformers 5.13.1. Full recipe including HF cache layout:
`code/PLAN.md` + the resume block in the Drive `interpret/inprogress.md`.
Quick path:

```bash
git clone https://github.com/anthropics/jacobian-lens && cd jacobian-lens
git checkout 581d3986 && pip install -e . && cd ..
# point SL1_RUN_DIR / SL1_RUN_DIR_V2 at fresh dirs, then run code/scripts in order;
# every script is a no-op re-run when its outputs exist and resumes mid-phase.
```

Tier A smoke (no 32B needed): `s1_smoke_verify.py` reproduces the 7B lens
battery exactly (9/11 probes @rank≤20, deterministic across VMs).

## Working model

The live working dir is `special_lab1/` (untracked); Drive is the durable
checkpoint store; this directory is the published mirror, updated by
`code/../push_lab.sh` at phase boundaries alongside `refresh_handout.sh`.
Three VM reclaims to date, zero data loss.

## Evidence discipline (course terms)

Rungs exercised: OBS (geometry), DECODE (lens readouts vs logit-lens
baseline), CAUSAL (ablation grids with matched controls), AUDIT (foil
floor for the cot-lead claim; energy-match verification; cross-VM integrity
cells). Operationalization audits — the deflationary twins — are built in:
energy-vs-content for the causal story, detector-permissiveness for the
lead claim, lexical echo for eval-awareness. Forbidden claims and the
allowed one-sentence claim are in the lab spec header.
