# Gemma transport study 2 — live handoff

Status: G2.1 backend-only grid complete and target-blind; threshold not yet frozen. Both exact snapshots passed full hashing, all 216 full pairs plus 16 nested op pairs ran, and the required fresh-process replay passed.

- Parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb`
- Branch: `interp_jspace_gemma_transport_2`
- Worktree: `/content/labs_gemma2`
- Drive root: `/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_2_20260803`
- Registered foundation: `gm2-foundation-v1` at code commit `325baec7a1662e936e62f400cc794ec5aa9e7667`; registry append pushed at `fbb7142bd9886939978d55d7fd5a28d63ac5f674`.
- Next event: `gm2-backend-parity-calibration-v1`, followed by exactly one G2.2 terminal event.
- G2.1 implementation: `jspace_gemma/experiments/gm2_backend_parity_calibration.py`; phases are strictly `stage`, `run`, `fresh-replay`, `freeze`, and `register`.
- Recovery state: Drive `raw/gm2-backend-parity-calibration-v1/raw_rows_state.json`; atomic checkpoint every four backend pairs plus a per-pair heartbeat under `checkpoints/`.
- The two exact model snapshots are downloaded directly from Hugging Face at their frozen revisions and fully rehashed before each load. Keep only one model GPU-resident.
- Tests before model staging: 56 passed (including direction, dtype-quantum, 216-pair reconstruction, singleton-batch, and target-firewall goldens).
- Pre-data correction: the exact checkpoint and study-1 contract both say `gemma4_text`; the study-2 YAML typo `gemma3_text` was corrected without changing any scientific cell or threshold. See `protocol/G2_PRE_DATA_ARCHITECTURE_CORRECTION.md`. No raw-row state existed at correction time.
- Complete raw state: 232 total pair summaries, 1,008 rows, all finite, exact primal parity throughout, and zero deterministic-replay failures. Target-blind route reconstruction is `benign_scheduling_floor`; pooled full-row q99 is `0.026234563004519824` and the frozen-formula value is `0.07870368901355948`.
- First freeze attempt wrote no threshold or final table. It exposed only a too-tight comparison between the stored float32 all-slot cosine reduction and a float64 reconstruction. The target-blind audit correction is frozen in `protocol/G2_POSTDATA_RECONSTRUCTION_AUDIT_CORRECTION.md`; the maximum residual `9.226683939211888e-06` is within the dimension-derived `1.52587890625e-05` bound. All other reconstruction checks passed.

Recovery rule: require a clean branch, set `JSPACE_GEMMA_RUN_ROOT` to the root above, verify the study-1 registry prefix and imported artifact hashes, and never open `gm2_stage1_relicense.yaml` from the G2.1 run/freeze process.
