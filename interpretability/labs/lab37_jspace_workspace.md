# Lab 37: J-space Global Workspace Replication (OLMo-3-32B-Think)

```text
Time estimate: 30-60 minutes Tier A smoke (7B); 8-12 hours full Tier C pipeline (5.2 h lens fit + phases 2-5); v2 instrument-audit delta adds 4-8 h
Compute tier: Tier A = Olmo-3-7B-Instruct smoke path; Tier C = Olmo-3-32B-Think, bf16 on one >=80 GB GPU, no quantization
Dependencies: anthropics/jacobian-lens @ 581d3986 (editable install); WikiText-103 cache; concepts from Lab 6 (circuit readouts), Lab 8 (dictionary decompositions), Lab 22 (eval awareness), Lab 36 (report-channel discipline, claim-template style)
Minimum passing artifacts: jspace/report/REPORT.md, jspace/report/summary.json, jspace/figures/f1-f8, jspace/results/v1_ablation_results.json, v1_cot_lead.json, v1_lens_sanity_32b.json; v2 adds v2_energy_match.json, v2_cot_foils.json, v2_ablation_v2.json, frozen-ablation and late-band artifacts
Main plot: jspace/figures/f15_qwen_grid.png (cross-model causal headline); f11/f12 for the OLMo-only grids
Main table: jspace/report/summary.json (+ summary_v2.json)
Evidence rung: OBS + DECODE + CAUSAL + AUDIT
Forbidden claim: global-workspace consciousness in the cognitive-science sense; any unconditioned "OLMo has no workspace" — only "instrument X at doses Y on band Z found / failed to find effect E"
One-sentence allowed claim: On Olmo-3-32B-Think, J-lens workspace geometry replicates at roughly 10x thinner capacity than the paper's Claude numbers — and the identical harness reads Qwen3.6-27B at paper-range capacity (4.3-6.8% variance share), making the thinness a model property; the paper's multi-step-vs-shallow causal dissociation does not appear under energy-matched or frozen-selection instruments on either model; per-item frozen J-ablation is a control-clean, seed- and item-robust causal handle on retrieved factual content (-2.9 nats OLMo / -2.4 nats Qwen) whose deletion externalized reasoning largely bypasses (silent recall 0.23 -> think-mode 0.80); and the workspace anticipates the chain of thought by a foil-calibrated median 46 steps (49.5 on an independent late-band lens).
Human-label requirement: required before strong claims from CoT-divergence examples or eval-awareness generations
```

## Status

**COMPLETE (2026-07-27) — proposed for promotion.** v1, the v2 instrument
audit (P1–P6), and the Qwen3.6-27B cross-model leg (phase Q) are all
landed; the run queue is empty. Final prose: `jspace/report/REPORT_v2.md`;
primary read: `jspace/handout/olmo32b_jspace_handout.pdf`; suggested PR
description: `jspace/PR_BODY.md`. Promotion checklist state at the
bottom — one human-label box remains open.

## Thesis

The workspace paper makes three separable claims — a sparse verbalizable
subspace exists (OBS/DECODE), it is broadcast (ATTR-adjacent), and ablating
it dissociates multi-step reasoning from fluency (CAUSAL) — plus an identity
claim connecting workspace content to reasoning substrate. Each claim needs
its own instrument, and each instrument has a cheap deflationary twin. This
lab replicates all three on a fully open model and builds the deflationary
twin *into* the protocol: energy-matched controls for the causal claim,
a frequency-matched foil floor for the anticipation claim, matched-live
controls for the paper's own per-token intervention, and a lexical-echo
check for the eval-awareness stretch. A thinking-model extension asks what
the paper could not: does the workspace hold the answer before, during, or
only at the end of explicit reasoning?

## Protocol (phases -> scripts in jspace/code/scripts/)

| Phase | Scripts | Question |
|---|---|---|
| 0 env + smoke | s0, s1 | pipeline proven on 7B before touching 32B; cross-VM exact reproduction gate |
| 1 lens fit + sanity | s2, s3, s4 | 120 WikiText prompts, 21 layers, target L63; multihop bridge-entity readout beats logit lens? |
| 2 descriptive + broadcast | s5, s6 | active concepts, variance share, persistence, fan-out vs matched controls |
| 3 causal battery | s7 | rank-matched static/dynamic ablations, dose-response, bootstrap CIs |
| 4 thinking-model | s8, s8b | pre-CoT anticipation, workspace-vs-text lead, suppressed-CoT loading, divergence audit |
| 5 stretch | s10 | eval-awareness differencing (lexical-echo caveat) |
| v2 P1 | s11 | energy-matched controls (the audit that decides if v1's causal null was an energy artifact) |
| v2 P2 | s12 | frozen prompt-selected top-10 (confound-free variant of the paper's live intervention) + matched-live controls |
| v2 P3 | s14, s15 | late-band lens {46..62}: is OLMo's workspace late-shifted past the fitted band? |
| v2 P4 | s13 | foil calibration of the lead claim (false-positive floor) |
| v2 P5/P6 | s16, s17 | CoT-rescue; second seed; n doubling |
| v2 Q | s18 | same instruments on Qwen3.6-27B via Neuronpedia's published lens — RAN 2026-07-26: sanity + descriptive + energy-match + 6-condition causal grid |
| reporting | s9, s19 | figures + summaries, all regenerable from metrics; living LaTeX handout |
| ops | s20–s23 | VM sanity, one-load driver (s16+s17), disk/DriveFS management, P5/P6 extractor |

## Operationalization audit (the deflationary twins, built in)

- **Causal null vs energy artifact**: v1's rank-matched controls were
  measured energy-mismatched ~10x in opposite directions; v2 matches removed
  raw-h energy to within 1% per layer/dose before comparing.
- **Lead claim vs detector permissiveness**: identical detector on
  frequency-matched foils (floor 0.06 vs answers 0.92) and same-family wrong
  answers (fire concurrently with text, lead 3 vs 46).
- **Paper-protocol ablation vs live-computation deletion**: live top-10
  removal lobotomizes OLMo under matched-live controls; frozen selection is
  the clean instrument.
- **Eval-awareness vs lexical echo**: framing states its cue words, so the
  (p<0.001) direction is a context-verbalization demo unless implied-cue
  variants reproduce it.
- **Model vs harness**: every OLMo null re-measured on Qwen3.6-27B with the
  community's published lens (not ours) — capacity difference survives
  (model), causal static null survives (instruments), frozen deletion
  transfers (real effect).
- **Numerics vs signal**: Qwen's outlier activations diverged the pursuit
  refit (inf/nan variance shares at L20-38); fixed with a provably stable
  step (1/(k+2)); stable layers moved <=0.2pp, and the static-null grid was
  shown insensitive even to the broken selection pass.

## Claim ledger templates

```text
[SL1-C1] DESCRIPTIVE_GEOMETRY | On Olmo-3-32B-Think with a 120-prompt WikiText J-lens (21 layers), workspace geometry replicates qualitatively — mid-band inverted-U, top-1 persistence 0.19-0.20 at L32-40 vs 0.03-0.08 at the ends — at ~10x thinner capacity than the paper's Claude numbers (variance share <=0.67% vs 6-10%; ~6 active concepts at theta=0.01 vs ~25). v2 CLOSURES: (a) not late-shifted — dedicated L46-62 lens reads 0.61-0.72% flat with declining persistence (L62's 1.6% is unembed-adjacent, lowest persistence); (b) not harness — the identical code path reads Qwen3.6-27B at 4.3-6.8% variance share with 32-42/14-21 active concepts at theta=0.01/0.02. The thin OLMo workspace is a MODEL PROPERTY, paired with the higher top-1 persistence (0.19 vs Qwen's 0.03-0.09). OBS+DECODE.
Artifact: jspace/results/v1_lens_sanity_32b.json + v2_descriptive numbers in results/v2_* and summary_v2.json + figures f1/f2/f13/f15 | Falsifier: Dolma-corpus lens recovers paper-scale capacity on OLMo, or the active-concept convention is shown non-commensurable in our disfavor (paper pins re-checked 2026-07-26: no absolute threshold exists; pursuit-k sweep reported).
```

```text
[SL1-C2] CAUSAL_VERDICT | On Olmo-3-32B-Think: (a) at energy-matched removal (ratios 0.97-1.01 per layer/dose) static J-span, random, and non-J subspaces are all indistinguishable from baseline at k<=40 dims/layer — v1's apparent non-J selectivity was an energy artifact; (b) per-item frozen top-10 J-ablation is a control-clean causal handle on retrieved factual content (answer logprob -2.9 nats, recall 0.58->0.23, single- AND multi-hop; random-dictionary twin at baseline; generations coherent); (c) the paper's multi-step-vs-shallow dissociation does NOT appear under any clean instrument on OLMo — shallow recall collapses identically, so the J-space is a content channel, not a demonstrated reasoning scratchpad; (d) the deletion is RESCUABLE by externalization (see C6) and replicates on a second seed with fresh items (P6), and both the frozen deletion and the static null TRANSFER to Qwen3.6-27B (see C7). CAUSAL+AUDIT.
Artifact: jspace/results/v2_ablation_v2.json + v2_frozen_ablation.json + v2_robustness_seed1.json + figures f11/f12/f17 | Falsifier: frozen-rand with a vocab-sized random dictionary reproduces the frozen-J effect (pool-size confound); or harder shallow tasks reveal a dissociation the capital-city probes missed.
```

```text
[SL1-C3] BROADCAST_SPECIFICITY | Top J-directions are read by 73-94 downstream components vs 0 for matched random (p~1e-24), but non-J high-variance PCs are read equally widely (77-125): broadcast is a property of high-variance directions generally until a variance-matched fan-out comparison says otherwise. OBS.
Artifact: jspace/results/v1_broadcast.json + figure f8 | Falsifier: variance-matched fan-out shows J-directions read by significantly more components than energy-matched non-J controls.
```

```text
[SL1-C4] COT_LEAD | On 34 traced thinking-mode items the workspace holds the final answer a median 46 steps before the CoT first states it (91% of items), with a calibrated false-positive floor: identical detector fires at 0.06 on frequency-matched words and 0.32 on family foils (which lead by only 3 steps); the answer beats all its foils in 66% of items. v2-P3c REPLICATE: the independent late-band lens (disjoint layers L46-62) reads the same lead — median 49.5 steps, 86.4% workspace-first, detection 98.9%, n=88. OBS+DECODE+AUDIT.
Artifact: jspace/results/v2_cot_foils.json + v2_cot_lead_late.json + v1_cot_lead.json + figure f9 | Falsifier: a stricter detector (rank-1, single layer) collapses the answer-foil separation, or lead vanishes on non-probe-swap task families.
```

```text
[SL1-C5] ANSWER_TIME_LOADING | The workspace does not pre-compute answers: median answer rank 5613 at the last prompt token (10% <=20), but forcing an immediate answer (closed </think>) pulls it to 211 (23% <=20) with monotone collapse L30->L60 (2048->6), and suppressing CoT costs 0-13pp accuracy. Consistent with on-demand loading, not anticipation. v2-P3b: the falsifier RAN AND FAILED — the dedicated late lens still shows no pre-CoT anticipation (think-mode median rank 3916-7291, 0% <=20) while the suppressed collapse deepens to best-late-layer median 3 (76% <=20); the logit lens reads the late band equally well, so J-over-logit is a mid-band phenomenon. OBS+DECODE.
Artifact: jspace/results/v1_cot_lead.json + v2_late_answer_profile.json + figures f6/f14 | Falsifier (remaining): paper-scale anticipation on task families outside probe-swap, or under a Dolma-fit lens.
```

```text
[SL1-C6] COT_RESCUE | Externalized reasoning largely bypasses the frozen content deletion on Olmo-3-32B-Think: with the SAME per-item frozen-J projectors that cut silent two-hop recall to 0.23, open-<think> generation (<=400 tokens) recovers the answer somewhere in the trace in 0.80 of items (n=30; frozen-random twin 0.93) and fully rescues one-hop (1.00 vs 0.23 silent); arithmetic is untouched in both conditions (0.93 = token-cap scorer floor). Secondary: frozen-J halves the </think>-closure rate (0.20 vs 0.40) — damaged recall lengthens deliberation. Consistent with the paper's workspace-ablation-rescuable-by-CoT prediction, in content-channel form; does not distinguish re-retrieval from re-derivation via the other hop. CAUSAL.
Artifact: jspace/results/v2_cot_rescue.json + figure f16 | Falsifier: matched-length non-reasoning padding (filler tokens, no semantic CoT) reproduces the recovery, which would make it a length/compute effect rather than externalization.
```

```text
[SL1-C7] CROSS_MODEL_VERDICT | Same harness, two models (Qwen3.6-27B run with Neuronpedia's published 1000-prompt lens): (a) J-lens readout advantage replicates on Qwen (multihop pass@1 0.350 vs logit 0.317, n=60); (b) descriptive capacity is a genuine model difference — Qwen variance share 4.3-6.8% with 32-42 active concepts at theta=0.01 (paper-range) vs OLMo 0.61-0.72% and ~6; (c) the energy-matched static null TRANSFERS to Qwen (J-span/matched-rand/matched-nonJ all within +-0.08 nats of baseline at k=20) — the published causal story's static form does not survive clean instruments on the model where the community replication reported it; (d) frozen per-item deletion TRANSFERS (-2.42 nats, two-hop 0.87->0.37, controls clean, fluency intact) with a Qwen-specific asymmetry: one-hop nearly intact (0.90->0.83) where OLMo's collapsed equally — the closest either model came to the paper's shallow-survives signature, NOT upgraded to a dissociation claim (near-ceiling one-hop items; relative-dose caveat). CAUSAL+AUDIT, single seed, n=30/15/20 per cell.
Artifact: jspace/results/v2_qwen_causal_grid.json + v2_qwen_sanity.json + summary_v2.json qwen block + figure f15 | Falsifier: harder single-hop items collapse Qwen's asymmetry to OLMo's pattern (content channel) or reproduce it at matched relative dose (dissociation); a second Qwen seed or their exact eval harness disagreeing with our battery.
```

## Safety / claims wall

This lab measures functional structure in activations. It licenses no claims
about experience, awareness, or their absence (Lab 36's wall applies
verbatim), and no unconditioned negative about the model family: every null
is "instrument X at dose Y on band Z", never "OLMo doesn't do it".

## Promotion to canon

Copy-paste decision once: (1) v2 P1-P4 land with the handout updated,
(2) claim templates above are instantiated with final numbers and signed
off, (3) the Qwen phase-Q comparison runs or is explicitly descoped with
reasoning, (4) a human label pass over the top-20 divergence examples and
eval-awareness generations.

**Checklist state (2026-07-27):**
- [x] (1) P1–P4 landed + handout §§P1–P4 (and P5/P6/Q beyond the ask)
- [x] (2) SL1-C1..C7 instantiated with final numbers above
- [x] (3) phase Q ran in full (sanity + descriptive + energy-match +
      6-condition causal grid); Q3 cot-lead spot check alone was gated
      out and is listed under "what 3 more GPU-hours would have bought"
- [ ] (4) human label pass — **open**; requires the maintainer to label
      the top-20 divergence events (`v1 cot_traces/`) and the s10
      eval-awareness generations. The lab ships as
      complete-pending-label; no claim above depends on the labels.

## Teaching entry points

Students should start with `jspace/README.md` §"Start here" (three-idea
digest), then `jspace/report/REPORT_v2.md` §"How to read this lab" — the
tutorial section: what the fitted-Jacobian lens computes (push-directions,
not resemblance; readout claim ≠ workspace claim), the task battery with
verbatim items (the Amazon→Brazil→Portuguese probe), and the
three-instrument taxonomy for causal cells (shared temp-table vs delete-
the-row vs delete-each-word-at-birth, with the inverted-prior rule for
sub-1% ablations and the amnesia-not-aphasia samples). The handout carries
the compact version as §"How to read the instruments".

## Readings

- Anthropic, Verbalizable Representations Form a Global Workspace in
  Language Models, Transformer Circuits, July 2026.
- Anthropic research post: anthropic.com/research/global-workspace.
- Nanda, replication review on Qwen 3.6 27B (LessWrong, July 2026).
- anthropics/jacobian-lens README + walkthrough.
