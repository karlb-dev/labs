# Gemma transport study 2 — live handoff

Status: foundation registered; G2.1 producer implemented and tested, before model staging. No study-2 model output exists.

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

Recovery rule: require a clean branch, set `JSPACE_GEMMA_RUN_ROOT` to the root above, verify the study-1 registry prefix and imported artifact hashes, and never open `gm2_stage1_relicense.yaml` from the G2.1 run/freeze process.
