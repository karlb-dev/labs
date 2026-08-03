# Gemma transport study 2 — live handoff

Status: foundation implementation, before G2.1. No study-2 model output exists.

- Parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb`
- Branch: `interp_jspace_gemma_transport_2`
- Worktree: `/content/labs_gemma2`
- Drive root: `/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_2_20260803`
- Next event: `gm2-foundation-v1`
- Then: `gm2-backend-parity-calibration-v1`, followed by exactly one G2.2 terminal event.

Recovery rule: require a clean branch, set `JSPACE_GEMMA_RUN_ROOT` to the root above, verify the study-1 registry prefix and imported artifact hashes, and never open `gm2_stage1_relicense.yaml` from the G2.1 run/freeze process.
