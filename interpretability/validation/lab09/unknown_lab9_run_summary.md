# Lab 9 run summary: attribution graphs and circuit tracing

## Run identity

- model: `gpt2` + `jacobdunefsky/gpt2small-transcoders` (full 12-layer MLP transcoder stack)
- primary fact: `The capital of France is` → ` Paris` (vs ` Berlin`)
- evidence level: ATTR for the graph, CAUSAL only for the intervention rows

## 1. What behavior was studied?

One-hop factual recall (Lab 5's domain) with logit-diff metric +3.013; 0 prompts dropped by the baseline gate (see tables/baseline_gate.csv).

## 2. What internal object was measured?

- A LOCAL REPLACEMENT MODEL: frozen attention patterns + frozen LN denominators,
  MLPs replaced by transcoders (mean FVU 0.149) plus error nodes.
  It reproduces the real logits exactly (diagnostics/replacement_exactness.json),
  and is linear - so direct edges are well-defined and must sum to the metric
  (diagnostics/edge_reconstruction_check.json).
- Backward-flow selection kept 16 feature nodes covering 31% of feature edge mass.

## 3. What intervention or control was used?

- suppress the subject supernode on the REAL model: +3.01 → -1.64
- substitute the counterfactual country's features: → -5.19 (p of counterfactual capital 0.0680)
- random suppression of matched size: → +3.02 (the control that makes it causal)
- intervention verdict: validated (specificity gap +4.65 logits)

## 4. What metric changed?

- target-vs-distractor logit diff and p(target) (0.0322 → 0.0010 under suppression); see tables/intervention_results.csv.

## 5. What claim is supported?

- `L09-C1` ATTR: An attribution graph over a transcoder replacement model of gpt2 (exact to the real logits; error nodes absorb FVU) attributes the 'The capital of France is'→' Paris' logit diff mostly to features at the subject token: direct edge shares are features 70%, errors 19%, embeddings 10%.
  - falsifier: The edge-reconstruction check fails on reruns, or the top edges vanish under a different node budget.
- `L09-C2` CAUSAL: Suppressing the graph's subject supernode (25 features at the ' France' position) on the REAL model drops the target logit diff from +3.01 to -1.64; the random matched suppression moves it to +3.02, leaving a graph-specific gap of +4.65 logits. Substituting counterfactual-country features drives the diff to -5.19 and raises p(counterfactual capital) by +0.066.
  - falsifier: The random matched control produces a comparable drop - the effect was generic perturbation.
- `L09-C3` OBS: In the signed edge ledger, features pay +2.34 of the fact's +3.01 logit diff but only +0.37 of the induction prompt's +0.62, where copied token embeddings (+0.93) and error nodes (+0.73) dominate: the replacement model freezes attention, so a routing behavior shows up as embedding mass moved by invisible wiring, not as feature structure - Lab 6's head circuit and this graph see complementary slices of the mechanism.
  - falsifier: On reruns the induction metric's embedding/error contributions do not exceed its feature contribution, or QK-attribution variants show comparable feature ledgers for both behaviors.
- `L09-C4` ATTR: 5 subject-site features recur as kept graph nodes in at least 2 of 3 paraphrases of the fact; the rest are single-template artifacts - recurrence under paraphrase is the cheap robustness screen the graph card requires before any feature is named in the mechanism.
  - falsifier: Recurring features fail to recur on fresh paraphrases, or recur equally for unrelated facts.

## 6. What claim is NOT supported?

- 19% of direct logit-edge mass routes through error nodes -
  computation the transcoders did not re-describe. The graph explains the part of
  the mechanism its dictionary can see, and the share is measured, not hidden.
- Edges are linearized attributions on ONE replacement model at a handful of
  prompts; no claim of invariance over a prompt population is made.
- The induction vignette shows the instrument's blind spot, not a fact about
  induction: frozen attention cannot appear as graph structure.

## 7. What would falsify the interpretation?

- Suppression effects matched by the random control on reruns/other facts.
- Recurring paraphrase features failing on fresh surface forms.
- A replacement model with lower-FVU transcoders attributing the behavior to
  different features entirely (dictionary-dependence).

## Reading order

Instrument health (the receipts) first, then the deliverable and the test.

1. `diagnostics/replacement_exactness.json`, `edge_reconstruction_check.json`, `feature_edit_noop_check.json` - the iron gates. No receipts, no graph.
2. `graph_card.md` - the deliverable (hypothesis written before interventions, real-model verdict, error-node share, Lab 6 confrontation, explicit non-claims).
3. `plots/graph_evidence_dashboard.png` + `tables/graph_evidence_matrix.csv` - the dashboard and ledger that bind receipts, visibility, causality, and robustness.
4. `tables/logit_edge_sources.csv` + `plots/source_token_ledger.png` before `plots/attribution_graph.png` - raw sources and token-level bills before the pruned display graph.
5. `tables/intervention_results.csv` + `plots/intervention_effects.png` + `tables/supernode_features.csv` + `plots/supernode_audit.png` - the causal test on the real model. Look for specificity gap vs random matched control.
6. `plots/influence_composition.png`, `plots/edge_mass_shares.png`, and `plots/budget_and_reconstruction.png` - the confrontation and completeness audit. Features pay for the fact; embeddings + errors dominate for induction. Absolute shares can mislead; signed tells the story.
7. `tables/paraphrase_robustness.csv`, `tables/paraphrase_feature_matrix.csv`, `plots/paraphrase_recurrence.png`, and `plots/paraphrase_feature_matrix.png` - recurring subject-site features vs template artifacts.
8. `graphs/pruned_graph.json`, `supernode_map.json`, `transcoder_stack_report.json`, and the rest of `diagnostics/` - the editable hypothesis and where the dictionary was weakest on this prompt.
