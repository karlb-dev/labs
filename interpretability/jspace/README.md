# Lab 37 (draft): J-space Global Workspace Replication — OLMo-3-32B-Think

> **Status: COMPLETE — proposed for promotion.** v1 (2026-07-25/26) +
> v2 instrument audit + Qwen cross-model leg + rescue/robustness
> (2026-07-26/27) are all landed; the run queue is empty. Spec + claim
> ledger + promotion checklist:
> [`labs/lab37_jspace_workspace.md`](../labs/lab37_jspace_workspace.md).
> Suggested PR description: [`PR_BODY.md`](PR_BODY.md).

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

## Findings (final)

| Claim | Result | Evidence |
|---|---|---|
| Workspace geometry exists | **Replicates, ~10× thinner**: variance share ≤0.67% (paper 6–10%), ~6 active concepts (paper ~25), clean mid-band inverted-U (persistence 0.19–0.20 vs 0.03–0.08 at ends) | `results/v1_*`, figs f1/f2 |
| Broadcast fan-out | 73–94 downstream readers vs 0 random (p≈1e-24) — but non-J high-variance PCs read equally wide → **not J-specific** (variance-matched re-test = v2) | f8 |
| Causal dissociation (paper headline) | **Null, now clean**: at energy-matched doses (ratios 0.97–1.01/layer) J-span ≈ matched-random ≈ matched-non-J ≈ baseline everywhere; v1's non-J "damage" was pure energy artifact (SQL back to 0.667 from 0.000; twohop_lp deltas −0.01…−0.17, CIs⊃0) | f10/f11, `v2_energy_match.json`, `v2_ablation_v2.json` |
| **Frozen per-item J-ablation (v2 P2)** | **First control-clean content-specific causal effect**: freezing out the item's top-10 J-dirs deletes the retrieved fact (answer logprob −2.9 nats; recall 0.58→0.23, 1-hop and 2-hop alike) while the random-dictionary twin moves nothing and generation stays coherent. **A recall content channel — not the paper's multi-step dissociation** (shallow collapses too) | f12, `v2_frozen_ablation.json` |
| Live per-token top-10 ablation (paper's own protocol) | Lobotomy on OLMo (NLL 2.71→24.34) — and now shown J-specific: matched-live random selection does nothing (0.60 two-hop, NLL 2.80; pool-size caveat) | f4/f12 |
| Workspace anticipates CoT | **Survives foil calibration**: leads text by median 46 steps; answer detection 0.92 vs 0.06 frequency-matched noise floor; family foils fire concurrently (lead 3) | `v2_cot_foils.json`, f9 |
| Pre-CoT anticipation | **Null** — answer median rank 5613 before any thinking token; loads on demand at answer time (rank 211 suppressed; monotone collapse L30→L60) | f6, `v1_cot_lead.json` |
| Eval-awareness direction | Emerges p<0.001 all layers, semantically legible — but behaviorally inert and confounded by lexical echo | `v1_evalaware.json` |
| **Qwen3.6-27B leg (v2 Q)** — same instruments, their published lens | **Both causal verdicts transfer**: frozen-J deletes the fact (−2.4 nats, 0.87→0.37 two-hop; controls clean) and the energy-matched static trio is null (±0.08 nats). **Capacity does NOT transfer**: variance share 4.3–6.8%, 32–42 active concepts — paper-range, vs OLMo's 0.67%/6 under the identical harness. Qwen-only asymmetry: 1-hop barely moves (0.90→0.83) while 2-hop halves | f15, `v2_qwen_causal_grid.json`, `v2_qwen_sanity.json` |
| CoT-rescue of the frozen deletion (v2 P5) | **Externalization largely bypasses the deletion**: under the same frozen-J projectors (silent recall 0.23), think-mode recovers the two-hop answer in 0.80 of traces (control 0.93) and fully rescues one-hop (1.00); frozen-J halves the `</think>`-closure rate — the paper's rescue prediction holds in content-channel form | f16, `v2_cot_rescue.json` |
| Seed-1 + fresh-items robustness (v2 P6) | **Replicates exactly**: on fresh probe-swap items with redrawn pools, frozen-J accuracy is 0.233 under both seeds (disjoint item sets; Δlp −2.82 vs −2.89), controls at baseline, static null intact | f17, `v2_robustness_seed1.json` |

**The question that drove v2 — is the causal null OLMo or the harness? —
is resolved: instruments.** With energy matching and matched-live/frozen
controls, both models agree (static = nothing; frozen per-item = content
deletion), while the 10× capacity difference between OLMo and Qwen/Claude
is a real model property measured under one harness. Full decision tree
and verdict: `report/REPORT_v2.md`.

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
