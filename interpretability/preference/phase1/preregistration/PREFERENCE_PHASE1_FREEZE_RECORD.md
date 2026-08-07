# PREFERENCE_PHASE1_FREEZE_RECORD.md

**The Phase 1 design is FROZEN as of this commit.** This record contains
nothing but the freeze facts; the commit that adds it is tagged
`preference-phase1-freeze-v1` and contains no other change.

| frozen object | value |
|---|---|
| approval | `reviews/PHASE1_FREEZE_APPROVAL.md` (PI, 2026-08-07, session) |
| preregistration | `preregistration/PREFERENCE_PHASE1_PREREGISTRATION.md` (constants unchanged from candidate) |
| approval-base commit | `ecfc9de4e245b0300137bfb02e0ee3153a624453` |
| bank | `lab38_v2_phase1`, content hash `8d5039af581204a5…` (full in `data/lab38_preference_bank.meta.json`), jsonl sha `634aded4b467f0e3…` |
| codebook | `cb_final_41eec2d774` (AR KP4/PK7; RO VM2/GS2; leading-space none) |
| equality license | `agent_dual_code_provisional` ×2 (registry `pref1-equality-provisional-v1`); limitation carried in every report; PI/panel pass required before publication-grade claims |
| primary model | `allenai/Olmo-3-7B-Instruct` @ `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, bf16 |
| replication model | `allenai/Olmo-3.1-32B-Instruct` @ `ac0587e4a7744a551c059d8cd17ba220bc940dae`, bf16 (retained) |
| generation | greedy; choice max_new_tokens 8; explicit GenerationConfig; margins single-row full-sequence |
| seed policy | content-derived stable seeds with base 1238, committed pre-freeze (no run-time seed choice exists anywhere in the analysis path); the freeze-base commit above is the audit anchor |
| analysis constants | addendum G2 verbatim: SESOI 0.10; 90% hierarchical bootstrap ×10,000; NC p95 floor; LOIO ≥ 0.05; nuisance < min(0.10, effect); invalid-diff < 0.05; margin variance ≥ 0.10 nats over ≥ 24 train cells |
| mechanism entry | plan §9.3 stop rules; estimator/doses/controls per frozen preregistration §4 |
| dropped (documented) | proprietary DG PC; full DG battery (smoke stays); optional lineage; temperature robustness |

Unsealing rule: the frozen partition (full bank, all label sets, all
incidental splits) may now be run exactly once per model under
`--stage behavioral_frozen`; resume of an interrupted frozen run is the
same command (same-command resume), never a config change.
