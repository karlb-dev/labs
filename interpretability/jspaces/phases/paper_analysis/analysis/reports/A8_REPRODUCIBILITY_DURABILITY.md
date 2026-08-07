# A8 — reproducibility and durability synthesis

Sources: frozen release records (hash-verified here: the fresh-
materialization JSON matches the release-manifest pin `7cf71215…`) plus
this phase's own reconstruction audit (`tables/recon_*.csv`). This is
the methods-appendix/reproducibility-statement backbone.

## By evidence family

| Family | Registry-backed | Raw rows released | Analysis reconstructable | Independent rerun | Known deficit |
|---|---|---|---|---|---|
| Phase 2 confirmatory/replication (HP1/HP3) | yes (`n6-*-analysis-v2`) | item-level parquets on Drive | **byte-identical**: full frozen-pipeline rerun reproduced both payload SHA-256s (`5df4ace5…`, `f5367e5a…`), every bootstrap decimal (seed 4242) | this phase, fresh process | none |
| Phase 2 occupancy v2 / G4 controls | yes | per-position hist + per-item rows | occ_median re-derived from `occ_hist`; RAW/CENTERED identities verified 6 dp; flip rates re-derived from per-item rows | this phase | per-position raw vectors for the 400-draw CI not released (registered value verified as recorded) |
| Phase 3 primaries + bridge + audits | yes (`p3-*`) | item parquets + release-audit tables | P3-P2/P3-P3/mediation/span/prose re-derived from item rows this phase (`run_analysis.py` + agent reconstruction); N8 ladder at release: L1 61/61 exact, L2 deterministic arms ×3 models, L3 all 188 Qwen rows exact | N8 ladder (registered) + this phase | historical matched-control salt unrecoverable (state-of-record uses explicit `sha256-v1` realization; P3-P1 descriptive) |
| Phase 4 Qwen ladder / Q-L4 | yes (`p4-qwen-*`) | structural/functional gate tables on Drive | reconstruction audit this phase (see `recon_qwen_ladder.csv`) | fresh-materialization pass | **A120–A250 `state.json` permanently absent** (accepted; 16/17 outputs of that event verify; decision role superseded by later live gates) |
| Runtime identity | yes (`p4-runtime-identity-synthesis-v1`) | block records + identity locks | n/a (incident record; no number licensed) | two independent clean processes agreed to <1e-4 relative in-era | historical fit-era backward runtime unreconstructable (distribution-content + kernel-cache identities not preserved) — the finding itself |
| Gemma Study 1+2 | yes (`gm-*`, `gm2-*`) | 1120-row Stage-1 parquet, 1008-row G2.1 grid, selected-slot tensors | **43/45 targets byte-identical** incl. bit-exact 5000-draw bootstrap replays (frozen seeds; numpy 2.0.2 env match) and CPU recompute of GPU parity metrics; 2 render-diffs (17- vs 16-digit float prose) | this phase, independent numpy re-implementations | all-slot batch tensor not released → the 0.00246 backend error rests on hash-verified cross-artifact consistency (selected slot recomputed byte-identical) |
| OLMo Study 1+2 | yes (`ol-*`, `ol2-*`) | batteries, H6 row tables, power-sim configs | reconstruction audit this phase (see `recon_olmo.csv`) | `ol-independent-reconstruction-v1` (registered) + this phase | registered-dose site records never collected (archive schema gap; a finding, not a loss) |

## Whole-release durability (frozen, verified)

- Fresh materialization (fresh VM/mount/clone, 2026-08-04): **521 live
  output references → 520 verified** (515 literal-path, 3
  repository-materialization, 2 append-only-registry-prefix), **1
  failure = the accepted permanent deficit**, 0 unexpected failures, 0
  pin conflicts (`PHASE4_PART5_DURABILITY_FRESH.json`, SHA-256 matches
  the release-manifest pin; `only_known_deficits = true`).
- `raw_all_live_outputs_verified = false` (the one deficit);
  `policy_aware_release_set_verified = true` (reviewer + PI accepted).
- Suites at freeze: phase4 302, gemma 71, olmo 90 — pre- and post-merge.
- Registries at the tag hash-match `FREEZE_HANDOFF.md` exactly
  (recomputed in `ANALYSIS_FOUNDATION.json`).

## The portable lessons (Paper B §reproducibility)

1. **Version-level pinning does not pin backward semantics** (C7): pin
   distribution contents and compiled-kernel caches by hash; carry a
   prospective runtime control with a frozen tolerance in every consumer.
2. **Non-literal resolution modes must be named, never silent**: the
   durability verifier records repository-materialization and
   append-only-prefix modes explicitly; hashes are never relaxed.
3. **A deficit register beats a clean-looking claim**: "520/521 with one
   named, accepted, permanently-missing operational file" is the honest
   sentence; "all artifacts verify" is forbidden wording.
4. **Floats in prose need a canonical repr rule**: the only Gemma
   "discrepancies" found by reconstruction are 17-digit prose renderings
   of 16-digit stored float64s — same value, different print. Papers
   should quote the stored repr.
