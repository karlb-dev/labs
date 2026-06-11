# Lab 3: Attention — Routing, Induction, and What Heads Actually Do

**Evidence levels targeted:** observation (motifs) → attribution (head scores)
→ causality (scoped head ablation). **Prerequisites:** Labs 1–2; this lab
resolves Lab 2's per-layer attention bars into individual heads.

## The question

Which positions are routed where — and when does the routing actually matter
for the output? Those are two different questions, and conflating them is the
most common failure mode in attention analysis. This lab measures both for
every head and keeps them separate all the way into the artifacts.

## Two facts about a head

1. **Its pattern** (routing): the attention weights say which positions the
   head reads from. This is what heatmaps show. It is observational.
2. **Its write** (contribution): the head's slice of the out-projection input,
   mapped through its `W_O` columns, is what the head adds to the residual
   stream. Scored against the answer direction, this is attribution — the same
   frozen-norm convention as Lab 2, now applied per-head. On post-norm models
   (Olmo-3) **two frozen norms compose**: the block's post-attention norm and
   the final norm. Read `head_attribution_scores` — the composition is spelled
   out in the code.

The bench verifies the per-head decomposition before any science
(`diagnostics/head_decomposition_check.json`): head pieces + the shared
projection bias must rebuild each block's attention output.

A third check is new and easy to miss: attention patterns require the *eager*
attention implementation. Under sdpa/flash, transformers 5 returns an **empty
attentions tuple with no warning**. The bench forces eager for this lab and
hard-fails if patterns are missing — remember this the day you wonder why your
own research code's heatmaps are blank.

## The motifs

Three named patterns you should find (Olsson et al.):

- **previous-token head**: attends to position q−1. A circuit *component*,
  not a curiosity — see below.
- **induction head**: from the current token, attends to the token *after the
  previous occurrence* of the current token ("[A][B] … [A] → attend to [B]").
- **first-token sink**: dumps attention on position 0 (BOS when one exists).
  This is a known resting pattern, not a discovery. Its large attention mass
  typically comes with near-zero attribution (gpt2) — the canonical
  heatmap-astrology trap — but sink is a fallback label: on Olmo-3 the
  top-|attribution| heads are sink-labeled, so check the head table before
  equating the label with "contributes nothing".

The labeling rule is deliberately simple and printed in the code: content
motifs (induction, previous-token) above 0.35 win, larger first; **sink is a
fallback, not a competitor** (bar 0.5). The reason is itself a lesson: sink
mass is the model's *resting* pattern, so a naive argmax over raw scores
mislabels half the induction circuitry as sinks — we measured exactly that on
gpt2 while building this lab (L5H1/L5H5 carry ~0.7 sink mass *and* put ~0.4
of their attention on induction targets whenever targets exist). The head
table keeps every raw score so you can re-derive the mistake — and argue with
the fix; finding a head the rule mislabels is worth more than agreeing.

Synthetic prompts use non-alphabetic vocabularies (`B F Q B F Q…`) because
"A B C A B" is a broken microscope: the alphabetic continuation coincides with
the induction answer. Control prompts have no repeats; an induction score on
them is *undefined* (reported as blank), and a head that "inducts" on
`ctrl_paris` is a label failure you should report.

## Composition: the lab's payload

Induction is a **two-head circuit**: a previous-token head writes "I was
preceded by X" into earlier positions; the induction head's keys read that
annotation. So the lab ablates every candidate head in **two scopes**:

- `final_pos` — zero the head's write at the final position only. This is the
  direct path, commensurable with the attribution score (Lab 2's convention).
- `all_pos` — zero the head everywhere, including its writes at earlier
  positions that *later heads read*.

`plots/direct_vs_indirect_effect.png` plots one against the other. Heads on
the diagonal act directly. A previous-token head far **above** the diagonal —
near-zero direct effect, large total effect — is composition caught in the
act, with no patching machinery needed yet. Lab 5 will dissect the indirect
path properly.

## Running it

```bash
python interp_bench.py --lab lab3 --tier a            # smoke first, always
python interp_bench.py --lab lab3 --tier b --prompt-set full --topk 10
```

The bench auto-sets `--attn-implementation eager` for this lab. Ablation
breadth is controlled by `--ablate-top` (top induction heads; previous-token,
sink, attribution, random, and low-attribution heads are added automatically).

## First artifact-reading path

1. `plots/motif_maps.png` — three layer×head grids; where the named patterns
   live in this model.
2. `plots/attention_heads_<showcase>.png` — the motif heads' actual patterns,
   token-labeled. (Rows = query, columns = key, rows sum to 1.)
3. `plots/head_attribution_by_layer.png` — Lab 2's attention bars, resolved
   into heads. Are the high-attribution heads the induction heads?
4. `plots/direct_vs_indirect_effect.png` — the composition scatter.
5. `plots/head_attribution_vs_ablation.png` — does attribution predict the
   direct-path effect? The lab's headline Spearman rho lives here.
6. `tables/head_table.csv` — every head, one row: all motif scores, entropy,
   label, attribution. This is the lab's "key artifact" for grading.
7. `tables/example_head_scores.csv` — the same measurements per prompt. Use
   this to check whether a control prompt accidentally triggers an induction
   label.
8. `tables/natural_confirmation.csv` — do the synthetic/cycle induction
   heads still induct on natural repeated phrases?

## Writeup questions

1. Pick the strongest induction head. Trace its evidence across all three
   levels: pattern (heatmap), attribution (head table), causal (ablation,
   both scopes). Which level is strongest? Which would you bet on?
2. Find the strongest sink head. Reconcile its attention mass with its
   attribution and ablation effect in two sentences.
3. The head table contains three kinds of evidence about the same objects
   (pattern scores, attribution, ablation effects). Rank them by evidentiary
   weight for the claim "head H does induction", and defend the ranking.
4. Report one previous-token head's two ablation numbers. What exactly does
   the gap measure? What would Lab 5 need to show to confirm the composition
   story?
5. Did any head "induct" on a control prompt? If so, is it the rule or the
   head that is wrong? Cite `tables/example_head_scores.csv`.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| "model returned no attention patterns" | You overrode `--attn-implementation`; rerun with `eager` |
| Head decomposition abort | `diagnostics/head_decomposition_check.json`; new architecture needs `O_PROJ_PATH_CANDIDATES` extended |
| All induction scores blank | Your custom prompts have no repeated tokens (check `state/<id>/tokens.csv`) |
| Induction head found at layer 0 | Almost certainly a label artifact — layer-0 heads cannot read composed annotations; check the prompt for trivial copies |
| Sink scores near 1.0 everywhere | Check whether your tokenizer prepends BOS; the sink position is position 0 either way |

## What goes in the ledger

2–3 claims. The drafts in `ledger_suggestions.md` separate the observation
("head X scores Y on the stated rule") from the causal claim ("zeroing it
changes the gap by Z, in scope S"). Keep that separation when you edit — and
tag the scope on every causal claim. An ablation number without its scope is
not evidence, it is an anecdote.
