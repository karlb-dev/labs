# OFFICIAL_REPRO_CLAIM_LEDGER — Study 1

Each claim binds to live evidence ids (registry:
`reports/evidence_events.jsonl`) and regenerable artifacts. Tier:
development/methods. Nothing here regrades any frozen campaign claim.

| # | Claim (maximum licensed sentence) | Evidence ids | Key artifacts |
|---|---|---|---|
| OR-C1 | The published Qwen lens + campaign-pinned checkpoint form a mechanically valid instrument (exact readout parity, exact no-ops, paper-style slice phenomenology). | or1-conformance-v1, or1-qwen-lens-admission-v1 | qwen_admission.json, slice_examples_qwen.json |
| OR-C2 | The J-lens readout advantage over the logit lens is model-dependent: present on Qwen's latent-content sets, largely absent on OLMo (near-identity in-band Jacobians). | or1-qwen-lens-evals-v1, or1-olmo-lens-evals-v1, or1-olmo-fit-splithalf-audit-v1 | eval_*.json both lanes, splithalf_operator_audit.json |
| OR-C3 | The prospectively fitted OLMo n=250 official-estimator lens is fit-stable (halves ≤0.015 eval gap) and concordant with the frozen campaign lens on the fair intersection. | or1-olmo-fit-{route,half-a,half-b,splithalf-audit,merged}-v1, or1-olmo-lens-evals-v1 | fit_route.json, olmo_or1_*.pt hashes, eval grids |
| OR-C4 | Verbal report shows a concordant partial effect (~25–30% top-5) on both lanes, far below the paper's 88% (Claude); the initial Qwen deficit was three-quarters scoring artifact (or1-001). | or1-qwen-verbal-report-v2 (supersedes v1), or1-olmo-verbal-report-v1 | verbal_report_qwen_v2.json, verbal_report_olmo.json |
| OR-C5 | Paper-literal coordinate swaps under-complete at α=1 on both lanes (5–15% capable across FG/probe-swap; emphatic successes, under-strength failures) and the paper's α=2 gain inverts to 0%. | or1-{qwen,olmo}-flexible-generalization-v1, or1-{qwen,olmo}-probe-swap-v1 | FG/PS jsons + raw records |
| OR-C6 | The released selectivity, modulation, dual-task, and ignition contrasts reproduce directionally on Qwen (selectivity contrast 1.00; focus>suppress>0; asymmetric interference; depth-sharpened transitions). | or1-qwen-battery-group-{a,b,c}-v1 | battery jsons |
| OR-C7 | Verbal introspection and top-down summoning do not reproduce under the reconstructed protocols (no dose-response; Q2−Q1=0). | or1-qwen-battery-group-{b,c}-v1 (+ olmo group b) | verbal_introspection_*.json, top_down_qwen.json |
| OR-C8 | Broad protected J-ablation is selectively destructive on official prompts on both lanes (14/30 and 5–6/19 vs ≤1 matched), while coordinate swaps stay weak, and new-vs-frozen OLMo lenses agree: the campaign-vs-paper discrepancy is intervention-semantics plus population — the two interventions estimate different causal questions (Route E), with task population a major moderator (Route F). | or1-instrument-crossover-{qwen,olmo}-v1 | crossover_*.json, crossover_subset_manifest.json |
| OR-C9 | The g-folding basis choice is material on Qwen (min cos 0.382) and immaterial on OLMo (0.9886) — a concrete instrument-basis difference between campaign and paper machinery, live exactly where interventions are weakest. | or1-conformance-v1, or1-olmo-lens-admission-v1 | admission jsons, D9 |

Forbidden-wording audit: no claim above says exact official
replication, workspace present/absent, paper confirmed/refuted, or
null-for-gated. Gated/not-identified cells: official probe arm (R3),
line-break family (R2 corpus-empty), both carried as states.
