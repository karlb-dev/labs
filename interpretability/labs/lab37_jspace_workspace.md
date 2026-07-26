# Lab 37: J-space Global Workspace Replication (OLMo-3-32B-Think)

```text
Time estimate: 30-60 minutes Tier A smoke (7B); 8-12 hours full Tier C pipeline (5.2 h lens fit + phases 2-5); v2 instrument-audit delta adds 4-8 h
Compute tier: Tier A = Olmo-3-7B-Instruct smoke path; Tier C = Olmo-3-32B-Think, bf16 on one >=80 GB GPU, no quantization
Dependencies: anthropics/jacobian-lens @ 581d3986 (editable install); WikiText-103 cache; concepts from Lab 6 (circuit readouts), Lab 8 (dictionary decompositions), Lab 22 (eval awareness), Lab 36 (report-channel discipline, claim-template style)
Minimum passing artifacts: jspace/report/REPORT.md, jspace/report/summary.json, jspace/figures/f1-f8, jspace/results/v1_ablation_results.json, v1_cot_lead.json, v1_lens_sanity_32b.json; v2 adds v2_energy_match.json, v2_cot_foils.json, v2_ablation_v2.json, frozen-ablation and late-band artifacts
Main plot: jspace/figures/f4_ablation_dissociation.png (v1); f11_vmatch_grid.png becomes the causal headline once the v2 grid lands
Main table: jspace/report/summary.json (+ summary_v2.json)
Evidence rung: OBS + DECODE + CAUSAL + AUDIT
Forbidden claim: global-workspace consciousness in the cognitive-science sense; any unconditioned "OLMo has no workspace" — only "instrument X at doses Y on band Z found / failed to find effect E"
One-sentence allowed claim: On Olmo-3-32B-Think, J-lens workspace geometry replicates at roughly 10x thinner capacity than the paper's Claude numbers; the paper's causal dissociation does not replicate under rank-matched static instruments (energy-matched and frozen-selection verdicts tracked in v2); and the workspace anticipates the chain of thought by a foil-calibrated median 46 steps.
Human-label requirement: required before strong claims from CoT-divergence examples or eval-awareness generations
```

## Status

Draft proposed lab on branch `interp_jspace`. v1 complete; v2 delta run
(instrument audit) in progress — live status in `jspace/README.md`,
`jspace/code/PLAN_v2.md`, and the Drive-side `interpret/inprogress.md`.
Not yet canon; promotion criteria at the bottom.

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
| v2 Q | s18 (design) | same instruments on Qwen 3.6 27B via Neuronpedia's published lens — model vs harness |
| reporting | s9, s19 | figures + summaries, all regenerable from metrics; living LaTeX handout |

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

## Claim ledger templates

```text
[SL1-C1] DESCRIPTIVE_GEOMETRY | On Olmo-3-32B-Think with a 120-prompt WikiText J-lens (21 layers), workspace geometry replicates qualitatively — mid-band inverted-U, top-1 persistence 0.19-0.20 at L32-40 vs 0.03-0.08 at the ends — at ~10x thinner capacity than the paper's Claude numbers (variance share <=0.67% vs 6-10%; ~6 active concepts at theta=0.01 vs ~25). OBS+DECODE.
Artifact: jspace/results/v1_lens_sanity_32b.json + figures f1/f2 | Falsifier: late-band or Dolma-corpus lens recovers paper-scale capacity, or the active-concept convention is shown non-commensurable in our disfavor.
```

```text
[SL1-C2] CAUSAL_VERDICT | (v2-gated) Under rank-matched static ablation the J-span is indistinguishable from random despite removing ~3x more energy; the paper-faithful live top-10 destroys fluency (NLL 2.71->24.34) and fails its matched-live control on OLMo. Energy-matched and frozen-selection grids [pending] decide the final wording. CAUSAL+AUDIT.
Artifact: jspace/results/v2_ablation_v2.json + v2 frozen_ablation.json + figures f11/f12 | Falsifier: energy-matched non-J damage vanishes AND frozen-J produces the paper's dissociation — then the v1 null was instruments, and the verdict flips.
```

```text
[SL1-C3] BROADCAST_SPECIFICITY | Top J-directions are read by 73-94 downstream components vs 0 for matched random (p~1e-24), but non-J high-variance PCs are read equally widely (77-125): broadcast is a property of high-variance directions generally until a variance-matched fan-out comparison says otherwise. OBS.
Artifact: jspace/results/v1_broadcast.json + figure f8 | Falsifier: variance-matched fan-out shows J-directions read by significantly more components than energy-matched non-J controls.
```

```text
[SL1-C4] COT_LEAD | On 34 traced thinking-mode items the workspace holds the final answer a median 46 steps before the CoT first states it (91% of items), with a calibrated false-positive floor: identical detector fires at 0.06 on frequency-matched words and 0.32 on family foils (which lead by only 3 steps); the answer beats all its foils in 66% of items. OBS+DECODE+AUDIT.
Artifact: jspace/results/v2_cot_foils.json + v1_cot_lead.json + figure f9 | Falsifier: a stricter detector (rank-1, single layer) collapses the answer-foil separation, or lead vanishes on non-probe-swap task families.
```

```text
[SL1-C5] ANSWER_TIME_LOADING | The workspace does not pre-compute answers: median answer rank 5613 at the last prompt token (10% <=20), but forcing an immediate answer (closed </think>) pulls it to 211 (23% <=20) with monotone collapse L30->L60 (2048->6), and suppressing CoT costs 0-13pp accuracy. Consistent with on-demand loading, not anticipation. OBS+DECODE.
Artifact: jspace/results/v1_cot_lead.json (anticipation profiles) + figure f6/f14 | Falsifier: late-band lens shows paper-scale pre-CoT anticipation the mid-band lens missed.
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

## Readings

- Anthropic, Verbalizable Representations Form a Global Workspace in
  Language Models, Transformer Circuits, July 2026.
- Anthropic research post: anthropic.com/research/global-workspace.
- Nanda, replication review on Qwen 3.6 27B (LessWrong, July 2026).
- anthropics/jacobian-lens README + walkthrough.
