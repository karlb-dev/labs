# Lab 6 verification report — circuit discovery, the manual way

Date: 2026-06-11 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`

## Context: user-commit re-validation first

Before Lab 6, the user's commit `13fb018` (plot footers, context-stamped CSVs,
tightened metrics across labs 1–5) was reviewed and all five labs re-validated: Tier A
smoke ×5 green, fresh Tier B runs ×5 green, Drive archives refreshed with the new
self-identifying artifacts. Footer renders on every plot; CSVs carry 12 context columns.

## What was built

- **Bench**: `run_with_node_set_ablation` — simultaneous mean/zero ablation of
  arbitrary head sets (hundreds at once, for faithfulness complements) and MLP layers;
  full-sequence options for attention and component captures.
- **Lab**: induction-completion circuit on fixed-length 8-token patterns (6 discovery
  + 4 held-out vocabulary families, dual-tokenizer-verified). Workflow: cheap screen
  (Lab 2/3 attribution + motif scores, reused via import) → causal ranking
  (single-node mean-ablation) → greedy pruning to a faithfulness floor →
  faithfulness / completeness / minimality on discovery AND held-out →
  one edge claim by ablation interaction → circuit card with an MDC filler-terms
  section. Screen-vs-causal scatter is the built-in Syed et al. comparison.

## Scale finding during validation

The screen breadth must scale with model width: 16 candidates out of Olmo-3's 1,024
heads started at faithfulness 0.43 (too thin to prune — the pruner correctly refused);
widening to 34 candidates started at 0.824 and pruned cleanly. The constant now
documents the measurement. gpt2's 144 heads were fine at either breadth.

## Validation evidence

| Measurement | Tier A (gpt2) | Tier B (Olmo-3-7B) |
|---|---|---|
| Baseline gate | 6/6 prompts | 6/6 prompts |
| Final circuit | 6 heads (incl. literature heads L5H6-adjacent band, L7H10) | 9 heads (2 induction-labeled) |
| Faithfulness (discovery / held-out) | 0.79 / 1.25 | 0.70 / **0.74** |
| Completeness ratio (discovery) | 0.16 (strong collapse) | 0.57 (redundant paths — honest 7B finding) |
| Edge claim (prev→induction routed) | 1% | 2% |

Two findings worth the course's attention, both reported rather than smoothed:

1. **The prune trajectory rises before it falls** (0.82 → 0.96 over the first six
   removals on the 7B): several screened candidates were actively hurting the
   behavior. Pruning is not just compression; it is quality control.
2. **Redundancy at 7B scale**: completeness ratio 0.57 and a ~2% routed edge both say
   the same thing — the model has multiple induction pathways, so ablating any one
   (or even the 9-head circuit) leaves substantial behavior. The handout's writeup
   question 4 ("redundancy is not absence") is built on exactly this number.

## Runs included

- `lab06_circuit_discovery-*/` — Tier A (gpt2)
- `lab06_tierb_full/` — Tier B (Olmo-3-7B; ~2,000 multi-node ablation forwards)

Deliverable per run: `circuit_card.md` — task, dataset, metric, candidates, validated
nodes, F/C/M scores, the edge, failure cases, and the scope/filler-terms section.
