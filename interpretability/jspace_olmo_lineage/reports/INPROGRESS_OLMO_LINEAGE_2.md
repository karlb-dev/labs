# OLMo lineage study 2 — live handoff

Status: ancestry and foundation are registered; the frozen Study-2 producer is implemented and tested; the exact Think-SFT snapshot is staged on local NVMe but has not yet been model-loaded.

- Parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb`
- Branch: `interp_jspace_olmo_lineage_2`
- Worktree: `/content/labs_olmo2`
- Drive root: `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_20260803`
- First study-2 event: `ol2-checkpoint-ancestry-v1` (registered and pushed)
- Foundation event: `ol2-foundation-v1` (registered and pushed)
- Model order after Gemma G2.2: Think-SFT, then Think-DPO; each runs G5 then the paired two-frame seven-condition Tier-1 grid.
- Producer: `jspace_olmo_lineage.experiments.stage_wedge`; exact capability equality, snapshot/tokenizer/BOS hard gates, frozen capable-cohort manifest, per-item Drive checkpoints, full selected/protected-direction logs, matched-control conformance, and isolated registration.
- Watchdog entrypoint: `interpretability/jspace_olmo_lineage/run_stage_wedge_model.sh think_sft all`; logs and atomic heartbeats are written below the Study-2 Drive root.
- Validation before first load: 77 OLMo/Phase-3 intervention tests pass; Ruff and bytecode checks pass.
- Local SFT snapshot: `/content/hf_local/models--allenai--Olmo-3-32B-Think-SFT/snapshots/9770a0bacc8536a6b5870da62b75e5ba3681930d` (61 GiB cache; downloaded directly from Hugging Face at the frozen revision).
- Pre-model correction: the first CPU-only SFT preflight passed exact shard streaming through the tokenizer-contract step, then stopped because the producer expected `audit_encoding_sha256` in YAML rather than the frozen key `frozen_audit_encoding_sha256`. No tokenizer/model was constructed and no outcome was opened. The explicit field mapping and a regression test were added before retry.

Recovery: use the watchdog entrypoint from a clean `interp_jspace_olmo_lineage_2` branch. The runner verifies every local shard against `ol2-checkpoint-ancestry-v1` before loading weights, then resumes capability or Tier-1 from Drive without reopening completed items. Do not import the Gemma ceiling until both wedge cells and the joint router are banked.
