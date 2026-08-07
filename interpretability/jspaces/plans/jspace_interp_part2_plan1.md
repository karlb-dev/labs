# jspace_interp_part2_plan1.md — From Lab Result to Publishable Finding

Paste-line for Claude Code:
> The J-space lab (Lab 37) is merged to main: read `interpretability/labs/lab37_jspace_workspace.md`, `interpretability/jspace/report/REPORT_v2.md`, and the final handout before anything else. All runs, lenses, metrics, and traces live under `/content/drive/MyDrive/interpret/special-lab-1/` (v1: `2026-07-25_1726/`, v2: `2026-07-26_v2/`). This file is the Part 2 campaign plan. Work in a new run dir `part2_<date>/` per workstream, same conventions as before: incremental Drive checkpointing, resumable scripts, `refresh_handout.sh` + push at phase boundaries, claim-ledger discipline, honest nulls welcome. Execute workstreams in the priority order of §8 within whatever GPU budget the user states at launch.

## 0 · Where we are and what "publishable" means here

Parts 1 (v1 replication), 2 (v2 instrument audit), and 3 (final stretch: Qwen leg, rescue, falsifiers) established: descriptive geometry replicates with a 10× cross-model capacity difference (OLMo thin, Qwen paper-range); the paper's static-span causal dissociation is null on both open models under energy-matched, confound-resolved instruments; frozen per-item J-ablation is a control-clean, seed-stable, cross-model causal handle on retrieved content; the CoT battery (foil-calibrated 46-step lead, pre-CoT null, answer-time loading, externalization rescue) is novel. Open questions now outnumber answers, and the current result set has known "why didn't you try X" holes.

The publication bar this plan targets: every hypothesis for the static-null gets a discriminating experiment or an explicit unfalsifiability statement; every instrument degree of freedom (lens corpus, fit size, dictionary type, dose, selection rule, decoding) gets a measured robustness check; the task battery includes probes that *demand* working memory plus the paper's own task sets; and the model matrix separates post-training regime from pretraining from architecture. Anything less invites the obvious reviewer objections; everything here maps to one (§9).

**The hypothesis ladder this campaign adjudicates** (pre-registered; each workstream states its directional predictions):
- **H1 Externalization**: think-trained models put the workspace in tokens; the static dissociation lives in non-externalizing models. Discriminator: Workstream A.
- **H2 Occupancy**: the live verbalizable subspace in our models is occupied by the output stream itself, not deliberative content; dissociation requires workspace/output separation. Discriminator: Workstream D.
- **H3 Training-lab specifics**: unfalsifiable from outside; only shrinkable by exhausting H1/H2/H4/H5.
- **H4 Scale**: weakened already (Qwen has paper-range capacity, still null); state as untestable at our tier.
- **H5 Instruments**: largely closed by v2; residuals are task battery (Workstream C) and dose/selection (B4).

## 1 · Workstream A — The model matrix (the core of Part 2)

Define once, reuse everywhere — **the Core Battery** per model: (i) lens acquisition (fit or transfer, see A0), (ii) sanity gate (21 probes + multihop bridge pass@k, n=60), (iii) descriptive geometry (variance share, active concepts at θ∈{0.01,0.02}, persistence profile — same pursuit code path, post-fix), (iv) energy-matched static grid (J-span / matched-random / matched-non-J, k∈{10,20,40}, twohop_lp + accuracy, n≥60), (v) frozen per-item grid (frozen-J / frozen-rand / **frozen-logit**, one-hop + two-hop + arithmetic + NLL + grammar, n≥60 two-hop with a hard-item subset — see C3), (vi) where a think mode exists: pre-CoT anticipation, cot-lead with foil floor, suppression, rescue. Bootstrap CIs, 2 seeds on the decisive cells, greedy + one temperature-0.7 replicate of the frozen grid (closes the greedy-only caveat).

**A0 — Lens-transfer gate (run first; may save ~10 GPU-hours).** Before fitting any new lens, test whether an existing lens transfers across post-training siblings: run the sanity gate on OLMo-3-32B-**Instruct** using the *Think* lens, and on the Qwen non-think sibling using the Neuronpedia lens. Pass = probe hits and multihop pass@1 within ~15% relative of the donor model's gate. Either outcome is a reportable mini-result ("does the J-dictionary survive post-training?" — nobody has measured this); a pass eliminates two 5-hour fits.

**A1 — OLMo-3-32B-Instruct (the matched-pretraining control; highest priority in the whole plan).** Same pretraining corpus as Think, differing only in post-training — the cleanest possible test of H1. Pre-registered predictions: if H1, the static dissociation (or at minimum a one-hop/two-hop frozen asymmetry and a nonzero static dose-response) *appears or strengthens* on Instruct relative to Think; descriptive capacity may also shift. If Instruct is as null as Think, H1 is substantially weakened and H2/H3 rise.

**A2 — Qwen non-think sibling** (the instruct/base non-reasoning variant of the same generation; pin the exact HF id in PLAN at launch after checking what the Qwen 3.6 line ships). Same logic as A1 with the fat-workspace model: H1 predicts the already-visible one-hop/two-hop asymmetry sharpens toward the paper's full dissociation. Also rerun the asymmetry with the C3 hard one-hop set to kill the ceiling confound.

**A3 — Gemma 4 ~30B (third architecture family; the user's preferred counterpoint).** Adds an architecture axis: sliding-window/local-global attention mix and any multimodal tower change what "downstream" means for the Jacobian. **Instrument-adaptation gate (≤45 min)** before committing: verify (a) final-norm/gain linearization matches Gemma's norm placement, (b) logit softcapping — if present, the lens target must be pre-cap logits or the Jacobian is distorted at exactly the readout stage, (c) jlens fit memory on the architecture. If the gate fails cheaply, record `GEMMA_BLOCKED_<reason>` and proceed — an honest "the reference lens does not support architecture X without modification Y" is itself a useful methods sentence. If it passes: full Core Battery; if a think/reasoning variant of Gemma 4 exists, run both siblings and A0-style transfer between them. Prediction under H1: Gemma non-think patterns with A1/A2 non-think results, independent of architecture; a Gemma-specific deviation instead implicates architecture and is its own finding.

## 2 · Workstream B — Instrument robustness (closing the degrees of freedom)

**B1 — Fitting-set scaling (the user's Q3).** Fit OLMo-Think lenses at n=250 and n=500 (same corpus, seeds, recipe; reuse the 120 slices, fit only the increments, merge). Report: (i) dictionary stability — per-token direction cosine vs the 120-lens, distribution over vocab; (ii) readout deltas — sanity gate + multihop pass@k per lens; (iii) **causal selection stability** — Jaccard overlap of frozen top-10 selections per item across lenses, and one frozen-J grid rerun (n=30) under the 500-lens. Decision rule: if directions are stable (median cos > ~0.9), selections overlap ≥0.7, and the frozen effect size moves <0.5 nats, the 120-prompt recipe is validated and every prior result inherits the robustness; if not, the 500-lens becomes canonical and the affected v1/v2 cells get rerun and flagged.

**B2 — Dolma-corpus lens (pretraining-matched fitting distribution).** Fit a 120-prompt Dolma lens for OLMo; diff dictionaries against WikiText (per-token cosine, top-token overlap per direction), rerun sanity + one frozen-J grid. This is the scientific-legibility payoff of the OLMo choice and closes "your lens corpus is off-distribution for the model."

**B3 — Frozen-logit control (cheap, high leverage; run early).** Identical frozen mechanism, dictionary = plain W_U rows (RMSNorm-gain applied, no Jacobian). If frozen-logit ≈ frozen-J on fact deletion, the Jacobian pullback is not doing causal work on these models and the method claim narrows honestly; if markedly weaker, the J-space framing earns its name causally, not just as readout. Either way it's a required column in the paper's causal table. Extend to each model in Workstream A (it rides along in the Core Battery grid at near-zero marginal cost).

**B4 — Dose and selection escape hatches.** (i) Extend static doses to k∈{80,160} at matched energy — if a workspace needs coherent removal of a wider span, flat-null-to-160 closes it. (ii) **Persistence-selected span**: ablate the ~25 *most persistent* directions (the workspace-like ones by the paper's own signature) rather than top-activity — the last principled static selection rule not yet tried. (iii) Per-layer dose mapping on the frozen effect (which band layers carry the deletion). 
**B5 — Decoding robustness** is folded into the Core Battery (temperature replicate) — listed here so the caveat's closure is explicit.

## 3 · Workstream C — Task battery upgrade (the user's Q4)

**C1 — Working-set battery with parametric load.** Design probes that *require* holding k items across steps, k ∈ {2,3,4,6}: multi-entity binding ("Alice has the red key, Bob the blue... who opens the green door"), n-back-style deferred recall inside a prompt, chained variable updates ("x=3; y=x+2; x=y*2; ... final x?") at controlled depth, 3–4 hop bridge chains, and order-scrambled retrieval ("answer the second question first"). Two uses: (i) behavioral capacity curve per model (accuracy vs k) against workspace capacity from descriptive geometry — H-prediction: Qwen's fat workspace buys a shallower decline than OLMo's thin one if the J-space is functionally a working set at all; (ii) **load × ablation interaction** — rerun the static and persistence-selected ablations on high-load items only; a workspace that only matters under load is the last hiding place for the paper's dissociation, and this is the experiment that flushes it out or closes it.

**C2 — The paper's own task sets, verbatim.** Run the capacity and directed-modulation prompt sets shipped in `jacobian-lens/data/experiments/` on OLMo and Qwen. Closes the objection "your battery isn't theirs" with their exact materials, and directed modulation (steering *into* J-directions, not just out) is a causal mode this lab has never exercised — the positive-control complement to all the ablations.

**C3 — Hard one-hop set.** 60 one-hop items difficulty-matched to the two-hop set by baseline logprob (obscure entities, non-capital facts), replacing the near-ceiling capitals. Required to make the Qwen asymmetry (and any A1/A2/A3 asymmetry) interpretable; without it, "shallow tasks survive" is confounded with "shallow tasks were easy."

## 4 · Workstream D — The occupancy index (new instrument; adjudicates H2)

Define, per model per generation step: overlap between the live top-k J-directions and the imminent-output direction (dictionary row of the actually-emitted next token) — e.g., rank of the emitted token in the live readout, and cosine of its direction to the top-k span. Aggregate into an **output-occupancy index** per model. Compute on saved OLMo/Qwen traces (CPU-mostly) and on each Workstream A model (rides along free during Core Battery generation). Pre-registered prediction: occupancy is high on think models (workspace ≈ output stream → live ablation is gibberish, static is null) and *lower* on any model where a dissociation appears; occupancy should correlate inversely with static-ablation sensitivity across the matrix. This turns the live-computation confound from a complaint into a measurement, and it is the paper's most novel methodological contribution if the correlation holds.

## 5 · Workstream E — Secondary follow-ups (only after A–D are banked)

E1 rescue disambiguation: frozen-J with the *bridge* hop's directions also frozen (re-derivation blocked) vs answer-only freezing — separates re-retrieval from re-derivation; also fix the 32-token answer-window metric caveat. E2 `famously`-attractor forensics across the model matrix. E3 implied-cue eval-awareness with a behavioral endpoint. E4 upstream the pursuit numeric fixes as a `jacobian-lens` PR (CPU; do whenever).

## 6 · Statistics and reporting for publication

n≥60 on every headline cell (two-hop), 2 seeds on decisive grids, BH-FDR across the model×instrument matrix, effect sizes with CIs everywhere, and a `preregistration.md` committed *before* Workstream A launches containing the H1–H5 directional predictions above (verbatim), the decision rules, and the C1 load-interaction prediction — this file is what makes "we predicted it" claimable. Ledger: new claims SL2-C1.. per workstream in the Lab-36 template. Paper skeleton to draft at campaign end: (1) instruments and their confounds, measured; (2) the cross-model matrix — capacity, causal verdicts, occupancy; (3) the content-channel + externalization story; (4) what would falsify it. Venue path: LessWrong/Alignment Forum post first (the conversation lives there), then arXiv; workshop target of BlackboxNLP-class if the matrix lands clean.

## 7 · Compute budget (rough, at Part-1 rates on a 96 GB-class GPU)

A0 gate ~0.5h · A1 ~6–10h (4–5h saved if transfer passes) · A2 ~5–8h (lens likely transfers or exists) · A3 ~8–12h incl. adaptation risk · B1 ~7h (two incremental fits + reruns) · B2 ~5h · B3 ~1h · B4 ~2.5h · C1 ~4h (design is codegen; GPU is eval) · C2 ~2h · C3 ~1.5h · D ~1.5h (mostly rides along) · E ~3h. **Full campaign ≈ 45–55 GPU-hours.**

**Minimal publishable set ≈ 18–24 h**: A0 + A1 + B3 + C1(load×ablation on OLMo pair) + C3 + D + the A2 frozen grid. This closes the reviewer table's rows 1, 3, 4, 6, 7 and adjudicates H1/H2 on the matched pair; Gemma, corpus/fit-size robustness, and the paper's task sets become "additional models and robustness" rather than blockers.

## 8 · Priority order and drop rules

A0 → B3 → A1 → C3 → C1 → D → A2 → B1 → C2 → B4 → A3 → B2 → E. Hard rule from Part 1 carries over: never thin every workstream to fit a budget — execute in order, bank complete cells, and state exactly where the money ran out. If A1 flips the causal result, immediately pull B4 and C1-load forward on the Instruct model before spending anywhere else: characterizing a positive beats adding models.

## 9 · Reviewer-objection table (what closes what)

1. "Your lens is undertrained / corpus-mismatched" → B1, B2 (+ Qwen used the published 1000-prompt lens already).
2. "The Jacobian is branding; logit lens would do" → B3.
3. "Your tasks don't load working memory" → C1, C2.
4. "One-hop/two-hop comparison is ceiling-confounded" → C3.
5. "You only tested think models — of course the workspace is externalized" → A1, A2, A3 (and this objection is H1, i.e., the thesis).
6. "Greedy decoding only" → Core Battery temperature replicate.
7. "Live-ablation critique is just a methods complaint" → D (the confound, quantified as a phenomenon).
8. "Doses too small / wrong selection for a coherent workspace" → B4.
9. "n's are modest, single seed" → §6 standards on every new cell.
10. "Anthropic's models are just different" → acknowledged as H3, shrunk to whatever A–D leave standing.
