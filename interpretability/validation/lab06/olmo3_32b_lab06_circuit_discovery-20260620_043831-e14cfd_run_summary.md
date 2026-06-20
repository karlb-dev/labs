# Lab 6 run summary: circuit discovery, the manual way

## Run identity

- model: `allenai/Olmo-3-1125-32B` (64 blocks x 40 heads)
- task: induction completion, 3 discovery + 2 held-out prompts (2 baseline-positive for F/C/M)
- evidence level: `CAUSAL` at heads-only circuit scope
- intervention: dataset-mean ablation at all positions
- self-checks: hook parity, lens, component decomposition, head decomposition

## 1-4. Behavior, measurement, intervention, headline

- base metric +7.917; circuit of 1 heads (L27H13)
- pruning stop: one head remains
- faithfulness floor: 0.70; verdict: pass
- discovery: faithfulness 0.8472, completeness ratio 0.97451, completeness effect 0.02549
- heldout: faithfulness 0.247, completeness ratio 0.94041, completeness effect 0.05959
- minimality: worst marginal value 0.01924
- edge: none claimed

## 5. Claims

- `L06-C1` CAUSAL: A 1-head routing circuit (L27H13) in allenai/Olmo-3-1125-32B is faithful at 0.85 of base behavior on 3 8-token induction prompts when every non-circuit head is dataset-mean ablated. Ablating the circuit leaves 0.97 of base. MLPs are left intact, so this is a heads-only routing claim.
  - falsifier: Zero ablation, resample ablation, longer prompts, or natural-text induction collapses the result. That would show the circuit was specific to this off distribution or prompt family.
- `L06-C2` CAUSAL: The heads-only routing circuit transfers from discovery to 2 held-out vocabulary prompts with faithfulness 0.25 versus 0.85 on discovery. The claim is induction-pattern transfer, not natural-language generality.
  - falsifier: Held-out families, longer cycles, or a paraphrased natural-text induction set lose the faithfulness effect.

## 6. The reading order

1. `circuit_card.md` - the deliverable; everything else is evidence for it.
2. `plots/circuit_discovery_dashboard.png` - F/C/M, pruning, screening, and prompt failures on one page.
3. `tables/circuit_evidence_matrix.csv` and `plots/candidate_evidence_matrix.png` - the joined evidence ladder for every screened head.
4. `plots/circuit_graph.png` - validated heads, support MLPs, and any claimed edge.
5. `plots/prune_trajectory.png` and `plots/minimality_ledger.png` - what pruning costs and whether each final node earns its rent.
6. `plots/screen_vs_causal.png` and `plots/causal_motif_atlas.png` - where cheap screening, motif labels, and causal effects agree or disagree.
7. `tables/prompt_failure_modes.csv`, `plots/per_prompt_faithfulness.png`, and `plots/prompt_failure_scatter.png` - the specific prompts the circuit least explains or over-recovers.
8. `plots/edge_interactions.png`, `plots/edge_interaction_map.png`, and `tables/edge_interactions.csv` - ordered interaction checks, weak versus strong, layer-order respected, not path patching.

## 7. Caveats students must carry forward

- The circuit is a heads-only routing graph; MLPs are support, not nodes in the claim. This is the manual baseline Lab 9 will confront with an automated feature graph.
- Mean-ablation defines the off state (dataset mean, fixed length). Changing the off state (zero, different mean, longer prompts) defines a different circuit. The ablation_manifest.json records the exact choice.
- The edge test is an ablation interaction (previous-token effect shrinks when the induction head is already ablated), not path patching. Claims about keys/values or exact subpaths are filler terms.
- Keep this card. Lab 9 will compare this manual graph with an attribution graph so you can see what each method buys and what each quietly assumes.
