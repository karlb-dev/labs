# OLMo lineage study 2 — live handoff

Status: foundation implementation before ancestry registration and before any SFT/DPO weight is downloaded or opened.

- Parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb`
- Branch: `interp_jspace_olmo_lineage_2`
- Worktree: `/content/labs_olmo2`
- Drive root: `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_20260803`
- First study-2 event: `ol2-checkpoint-ancestry-v1`
- Foundation event: `ol2-foundation-v1`
- Model order after Gemma G2.2: Think-SFT, then Think-DPO; each runs G5 then the paired two-frame seven-condition Tier-1 grid.

Recovery: set `JSPACE_OLMO_RUN_ROOT` to the root above, require the clean side branch, verify the frozen 58,286-byte study-1 registry prefix and ancestry source hash, and do not download/load an intermediate until both study-2 foundation events are committed and pushed.
