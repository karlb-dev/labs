# Lab 37: J-space Global Workspace Replication — OLMo-3-32B-Think

> **Location:** `interpretability/jspaces/phases/phase1/` (Lab 37 / Part 1). Nested Part 2 mirror moved to `interpretability/jspaces/phases/phase1/part2_exploratory/`.

> **Status: COMPLETE — proposed for promotion.** v1 (2026-07-25/26) +
> v2 instrument audit + Qwen cross-model leg + rescue/robustness +
> final falsifier pass (2026-07-26/27) are all landed. Spec + claim
> ledger + promotion checklist:
> [`labs/lab37_jspace_workspace.md`](../../../labs/lab37_jspace_workspace.md).

An open-weights replication of Anthropic's July 2026 paper *"Verbalizable
Representations Form a Global Workspace in Language Models"*
(transformer-circuits.pub/2026/workspace) on `allenai/Olmo-3-32B-Think`,
using the reference [`jacobian-lens`](https://github.com/anthropics/jacobian-lens)
package (pinned `581d3986`), plus a thinking-model extension the paper could
not run and an instrument-audit round (v2) that either flips or hardens the
causal verdict.

**Read first:** [`handout/olmo32b_jspace_handout.pdf`](handout/olmo32b_jspace_handout.pdf)
— the writeup (prose + figures). Final verdict + claims table:
[`report/REPORT_v2.md`](report/REPORT_v2.md). Full v1 prose:
[`report/REPORT.md`](report/REPORT.md).

## Start here: the three ideas that make this lab readable

New to J-space? `REPORT_v2.md` §"How to read this lab" is the tutorial;
this is the digest.

1. **The lens is a fitted, averaged Jacobian — a "what would the rest of
   the network make of this state" readout.** `J_ℓ = ∂logits/∂h(ℓ)` is
   estimated once over a fitting corpus and frozen; reading is a matrix
   multiply against push-directions (one per vocab token), not a
   per-prompt backprop. That's why a concept can be J-lens-visible but
   logit-lens-invisible (stored in coordinates that only become
   output-aligned many layers later), why the J-advantage is a mid-band
   phenomenon, and why one-linearization-for-all-inputs is the method's
   built-in noise floor.
2. **Finding hidden thoughts ≠ finding a workspace.** The lens really
   does surface the unstated bridge — probe-swap item 0: *"the language
   spoken in the country where the Amazon River ends is"* → **Brazil**
   (never in the prompt) → **Portuguese** — but that's the *readout*
   claim (a better logit lens). The *workspace* claim is a conjunction:
   small + localized + broadcast + **causally privileged**. This lab
   tests the conjunction. It breaks at the causal joint, on both models.
3. **Read the causal grid with your prior inverted.** Each ablation
   removes <1% of dimensions (≈0.5–1.1% of measured activation energy) —
   the default expectation is *nothing happens*, and the energy-matched
   random control shows exactly that. The three instruments ask
   different questions: **static span** = drop a shared temp-table (the
   paper's workspace signature would be "multi-join queries die,
   single-table lookups live" — never observed here); **frozen per-item**
   = delete the row (expected headline, non-trivial conjunction: linear
   projection sufficed, random-dictionary twin inert, *amnesia not
   aphasia* — see the verbatim coherent-under-ablation generations in
   `results/v2_frozen_ablation.json` and `v2_qwen_causal_grid.json`
   `samples`, and the 1-hop/2-hop damage shape classifies
   content-vs-scratchpad); **live per-token** = delete each word as it's
   born (dramatic, uninterpretable — the replication confound, measured).
   "Deleting key ideas breaks output" being unsurprising is not a
   problem with the lab — formalized with controls, it *is* the lab, and
   it's why the earned label is *content channel*, kept interesting by
   the CoT-rescue (0.23→0.80) and Qwen's spared one-hop.

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
| Final falsifier pass (s24) | Three recorded caveats measured: **pool-size closed** (100k-row random dictionary ≈ baseline: 0.433 vs 0.583, Δlp −0.14 — the deletion needs J-content, not selection depth); **rescue decomposed** (400 filler tokens recover 0.23→0.467, real CoT →0.80: a compute component plus a comparable externalization component); **broadcast non-dissociation measured** (energy-matched non-J PCs read by as many downstream components as J-dirs, 132/107/65 vs 92/73/68; matched-energy random ≈ 1) | `v2_final_falsifiers.json` |

**The question that drove v2 — is the causal null OLMo or the harness? —
is resolved: instruments.** With energy matching and matched-live/frozen
controls, both models agree (static = nothing; frozen per-item = content
deletion), while the 10× capacity difference between OLMo and Qwen/Claude
is a real model property measured under one harness. Full decision tree
and verdict: `report/REPORT_v2.md`.

## Layout

```
code/            analysis pipeline + sl1_common.py (shared harness lib)
  scripts/       s0 env … s9 v1 report; s11 energy-match, s12 frozen ablation,
                 s13 foils, s14/s15 late band, s16 CoT-rescue, s17 seed-1,
                 s18 Qwen leg, s19 v2 figures, s23 P5/P6 extractor,
                 s24 final falsifier pass
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
transformers 5.13.1. Every script no-ops when its outputs exist and
resumes mid-phase, so the pipeline can be run top to bottom or piecemeal.
Quick path:

```bash
git clone https://github.com/anthropics/jacobian-lens && cd jacobian-lens
git checkout 581d3986 && pip install -e . && cd ..
# point SL1_RUN_DIR / SL1_RUN_DIR_V2 at fresh dirs, then run code/scripts in order;
# every script is a no-op re-run when its outputs exist and resumes mid-phase.
```

Tier A smoke (no 32B needed): `s1_smoke_verify.py` reproduces the 7B lens
battery exactly (9/11 probes @rank≤20, deterministic across VMs).

## Provenance

The lab was produced across five preemptible VM sessions (four reclaims,
zero data loss) with Google Drive as the durable checkpoint store; this
directory is the published mirror of the analysis code and results. Run
provenance — plans, the dated decision log, resume recipes, and the
VM/infra tooling — is archived in the Drive run dirs, and the run story
(including a DriveFS failure mode worth knowing about) is summarized in
`report/REPORT_v2.md` §"Run notes".

## Evidence discipline (course terms)

Rungs exercised: OBS (geometry), DECODE (lens readouts vs logit-lens
baseline), CAUSAL (ablation grids with matched controls), AUDIT (foil
floor for the cot-lead claim; energy-match verification; cross-VM integrity
cells). Operationalization audits — the deflationary twins — are built in:
energy-vs-content for the causal story, detector-permissiveness for the
lead claim, lexical echo for eval-awareness. Forbidden claims and the
allowed one-sentence claim are in the lab spec header.
