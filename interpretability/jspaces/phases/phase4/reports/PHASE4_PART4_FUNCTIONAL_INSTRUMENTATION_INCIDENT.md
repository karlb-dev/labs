# Phase 4.4 functional instrumentation incident

**Development-only recovery record — no evidence event was registered**

## Hard stop

The sealed A1000 postfit queue opened the functional stage from clean commit `fe257056f6243563f5b0cfc74aec80ff037259ed`. It completed the lens-independent caches and the external published comparator cell, then stopped on the first A500 primary item at layer 20, forward 0, position 1. The hard invariant was `selection-margin observer changed top-k intervention`. No A500 primary row was checkpointed, no functional artifact set was finalized, and no functional registry event was created.

The failure was in the outcome-blind observer, not the inherited intervention. The observer requested the eligible top 32 and incorrectly treated its first 10 IDs as an exact replay of the parent's separate top-10 operation. At the real failure boundary, token IDs 47,191 and 2,192 both scored exactly 5.703125. `torch.topk(32)[:10]` selected 47,191 while the inherited `torch.topk(10)` selected 2,192. An isolated diagnostic replay showed that a separate observer-side `topk(10)` matched all parent IDs and scores exactly. This is permitted top-k tie behavior for different requested values of k; it is not a change in intervention, bank, model, endpoint, SESOI, or decision threshold.

## Outcome-blind repair

The capture now performs the diagnostic top-32 and the exact intervention-sized top-10 as separate operations. The parent remains the untouched Phase 3 span-safe ablator. Capture still hard-stops unless the separate top-10 IDs and rounded scores exactly match the parent's log. The successor audit now validates score-defined top-k admissibility and permits an ID difference from the top-32 prefix only at an exact k/k+1 score tie. It still rejects length drift, duplicates, protected IDs, nonpositive scores, ID/score misalignment, and any untied boundary substitution. The functional gate retains every frozen position and all prospective strata.

The focused functional/margin/contract suite passed 27/27 tests. The full Phase 4 suite passed 282/282 tests. New regressions cover both the different-k tied ordering and exact parent-observer intervention identity at a tied boundary.

## Preserved failed state

The incompatible unregistered partial output was not edited or reused across code commits. It was moved intact on Drive to:

`metrics/qwen36-27b-multilens-functional-gate-a500-a1000/functional_gate/p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1__withdrawn_20260803T030707Z_tied_topk_observer`

An exact local copy exists under `/content/sl4_work/postfit_unregistered_backups/` with the same directory name. Its four files are:

- `state.json`: `8d730e368204121c0e28d3fa15128c7e22fee71311ced2865516153eb6b8bf03`
- `fixed_endpoint_activations.pt`: `ad37f972c734f2bb8927ce3d6f4d2ba2ccacafc754a87404bdadd616fad4d85c`
- `fixed_capacity_activations.pt`: `e441f16b3dbd2654f45961fbcd2ddef4bcabef92999286da6910f0fa52474fa3`
- `capacity_reconstructions_published.pt`: `30bf51e999bf20abb0f4060349d89be9eea08359da6c4c2a30b01ffacc039b43`

The exact diagnostic transcript is preserved at Drive root `selection_margin_mismatch_diagnostic_20260803.log`. The functional producer must restart into a new canonical output directory under the repair commit; the prior published comparator cell will therefore be recomputed rather than migrated.
