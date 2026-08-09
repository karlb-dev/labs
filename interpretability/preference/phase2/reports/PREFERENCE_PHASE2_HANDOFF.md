# PREFERENCE_PHASE2_HANDOFF

Branch `interp_preference_phase2`; freeze `preference-phase2-freeze-v1`
(+E2 amendment d3); registry `phase2/reports/evidence_events.jsonl`.
Frozen run dirs (gitignored, Drive-mirrored under
`MyDrive/preference/phase2/runs/`): see `frozen_*/frozen_run_dir.txt`.
Captures: 202 sha-manifested shards in the 32B frozen run dir.

## Repro anchors

- Bank: `pref2 bank-audit` (rebuild + hash match, no model).
- Tests: `python -m pytest interpretability/preference/phase2/tests -q`.
- Any battery: `pref2 run --model <key> --stage <stage> --banks ...`
  (same-command resume; config-hash refusal).
- Reports/figures regenerate from tables: `preference_phase2.closeout`
  + `figures` (source CSVs beside every figure).

## Awaiting the PI (nothing blocks; when convenient)

1. Ratings to replace `agent_dual_code_provisional` (H1/H2/H4 sheets in
   `data/pref2_*review*.csv`; two passes, disagreements preserved).
2. H-CANON: ratify or re-code `data/pref2_semantic_axis_code.csv`
   (frozen pre-outcome; exploratory per D4), then run the 10k-permutation
   composite test against the 15 banked sign targets.
3. The 7B cell's L4 disposition (NC-paraphrase alarm +0.160 p .031 and
   the PC-under-F-SYM failure): confirm the STOP_P reading.
4. Optional next phases: a thought-channel-aware Gemma port; a
   distributed (multi-token span) intervention design for the mechanism
   arm — both need new preregistrations.

---
*Claim ceiling (plan §8): every statement above is about functional choice, semantic decision margins, contextual relative advantage, enacted branches, report-only selection, scenario-local causal handles, and functional choice/report coupling under this battery. No statement licenses mental-state language; the forbidden upgrade list is enforced by the raising language wall. License: agent_dual_code_provisional pending PI ratings.*
