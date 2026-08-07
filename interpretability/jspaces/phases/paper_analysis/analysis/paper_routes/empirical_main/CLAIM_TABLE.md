# Paper A claim table (P8) — every result sentence maps to a ledger row

| # | Sentence (maximum licensed form) | Ledger | Tier | Evidence ids |
|---|---|---|---|---|
| A1 | On Qwen, span-safe output-protected J-ablation produces a content-specific heavy tail beyond an exact per-site rank/energy-matched control; the effect replicates on disjoint held-out families. | C1 | confirmatory + held_out_replication | `n6-confirmatory-analysis-v2`; `n6-replication-analysis-v2`; `p3-inference-audit-v1` |
| A2 | Because the matched control equates rank and removed energy by construction, the difference isolates direction content — dose is ruled out. | C1 | confirmatory | `p3-inference-audit-v1`; MC gates |
| A3 | The Phase-2 Think-vs-Instruct interaction was Holm-rejected but did not replicate; it licenses no replicated cross-model contrast (fallback population disclosed). | C1 (bound) | confirmatory-unreplicated | `n6-replication-analysis-v2` |
| A4 | Protecting the true bridge rescues composed answers more than protecting the frozen chosen distractor; measured geometry does not explain the contrast away. Unreplicated by protocol; distractor chosen, not randomized. | C3 | confirmatory (unreplicated) | `p3-bridge-geometry-qwen36-27b-v2`; `p3-n8-p3-level3-qwen36-27b-v1` |
| A5 | Counterfactual substitution moves preference and generation as intended but is not separable from a direct answer-direction route. | C3 | development | `p3-bridge-swap-endpoint-qwen36-27b-v1` |
| A6 | OLMo damage is organized by answer accessibility; Qwen's by composition with opposite sign; Think's rescue contrast is mildly negative. | C2/C3 context | development | `p3-overlap-mining-v1`; `p3-bridge-mediation-olmo31-think-v1` |
| A7 | Across the OLMo lineage, capacity is broadly conserved while the J-mapped dictionary, selected spans, and Bank-S causal use reorganize at the first released Think transition. | C2 | development | `p4-lineage-trajectory-analysis-olmo-dev-v1`; `ol-capacity-joint-dev-v1`; `ol-geometry-joint-dev-v1` |
| A8 | The official SFT/DPO wedge does not localize the transition: prospective capability cohorts are empty; effects are missing, not zero. | C2 | capability_gated | `ol2-stage-wedge-joint-analysis-v1` |
| A9 | Externalization remains open: no powered intervention exists (16/20 cross-model; pair 0.7788@16, first pass 18). | C4 | gated | `ol2-bank-w-olmo-pair-power-v1`; `p4-bank-w-capability-joint-imported-dev-v1` |
| A10 | Prose damage exceeds task damage in standardized units everywhere; no model earns "selective"; the licensed noun is knowledge-access channel. | C1 boundary | development | `p3-prose-grid-figure-v1` |
| A11 | H6 bounds the lens as a finite-dose predictor, not the removal contrast; the causal license is the control behavior (reconciliation cited). | C5 via A5-recon | methods | `ol2-transport-validation-joint-v1` + reconciliation |
| A12 | Newly fitted sparse Qwen lenses are not fit-invariant (Q-L4); the frozen instruments used here are, where measured (lens-independence r≈0.99). | C6 summary | methods | `p4-qwen-canonical-lens-decision-a1000-dev-v1`; `n6-repl-lens-independence-v2` |

Forbidden anywhere in Paper A: global-workspace assertion; P3-P1
near-miss wording; SFT/DPO attribution; "Bank W negative"; zeros in
gated cells; canonical A1000 lens; unqualified "transport fails".
