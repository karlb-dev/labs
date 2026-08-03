# OLMo lineage study 2 — live handoff

Status: all mandatory scientific stages and the required CPU pair-power
closure are complete and registered. Release assembly, verification, and
parent-branch integration remain.

- Parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb`.
- Branch: `interp_jspace_olmo_lineage_2`.
- Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_20260803`.
- Registry before the release event: 37 origins, 36 live events, 145 live
  outputs; all Study-2 measurements are backed by Drive and pushed to GitHub.
- Frozen order completed: registered Gemma G2.1 import, SFT/DPO wedge, Base H6,
  OLMo-3.1 Think H6, then joint/dose audit.

## Stage wedge

- `ol2-checkpoint-ancestry-v1` records exact official SFT/DPO revisions and the
  repository-level, not byte-proven, ancestry qualification.
- Think-SFT and Think-DPO each produced 972 finite, unique exact-generation
  capability rows under the frozen battery.
- Capability rates are 0.00617 (SFT) and 0.00309 (DPO). Neither checkpoint has
  a Bank-S fact capable on both direct and composed variants, so both
  prospective cohorts are empty and no intervention effect was opened.
- Across the paired battery, 965/972 items are incapable at both checkpoints,
  2 are capable at both, 4 lose capability at DPO, and 1 Bank-F bridge item
  appears at DPO. No Bank-S direct+composed fact shows onset.
- `ol2-stage-wedge-joint-analysis-v1` selects `null_or_unresolved`. Effects are
  missing, not zero. The Tier-2 lens trigger is false.

## H6 transport

- The only backend ceiling used is the exact registered OLMo-specific Gemma
  G2.1 import: `0.07870368901355948`. The pooled ceiling is forbidden.
- Base and OLMo-3.1 Think each produced 336 rows (4 prompts × 4 layers × 3
  directions × 7 frozen doses), with the two exact-JVP backends evaluated on
  identical batches. All 672 backend comparisons pass the imported ceiling.
- Base has no layer/dose cell meeting the 0.90 passage floor. Its L56,
  epsilon-0.10 cell reaches 9/12, or 0.75.
- OLMo-3.1 Think passes only L56 at epsilon 0.10, with 12/12 rows. No
  L24/L32/L40 cell passes at either checkpoint.
- `ol2-transport-validation-joint-v1` therefore selects
  `h6_fail_in_band_with_checkpoint_specific_late_anchor`. This narrows
  transport wording and does not invalidate paired ablation evidence.
- The registered dose-source audit finds zero usable tables among six live,
  relevant Phase-3/4 tables. Per-item total-energy summaries and per-position
  protected-subspace energy are not the exact total-dose plus residual-norm
  records required by the frozen mapping. Intervention-dose coverage remains
  null/unresolved, never zero.
- `ol2-transport-validation-figure-v1` is the visually checked paper-facing
  PNG/PDF derivative; it changes no scientific result.

## Bank-W pair planning

- `ol2-bank-w-olmo-pair-power-v1` is outcome-blind methods evidence. It uses
  the registered 0.23-nat Bank-S variance ruler and the unchanged
  0.10-nat-per-doubling SESOI.
- Think and Instruct each have 17 capable families; their exact shared set has
  16. All four max-T type-I scenarios pass.
- Conservative power is 0.7788 at the available support versus the frozen
  0.80 target; 18 common families are required. The route is
  `not-powered-at-current-support`.
- No Bank-W intervention outcome was read. This does not authorize an
  intervention or provide an externalization result.

## Recovery

No model cache is retained. Exact snapshots were downloaded directly from
Hugging Face, verified, used, registered, and deleted only after Drive/GitHub
banking. Resume with CPU-only release work: emit and verify
`IMPORT_BUNDLE_SIDELINES2`, commit/push, then reconcile the updated parent and
merge Gemma before OLMo with ancestry preserved.
