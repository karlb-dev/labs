# OLMo lineage study 2 — live handoff

Status: both Think-SFT and Think-DPO are complete and registered as capability-gated; mechanical joint routing is next.

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
- Think-SFT result: `ol2-stage-wedge-think-sft-tier1-v1` registered at commit `7f12bd2`. All 972 G5 rows are finite and unique, but exact normalized-generation capability is 0.00617 overall and 0.00833 on Bank S. No Bank-S fact is capable on both direct and composed variants (frozen floors: 72 facts / 20 families), so the cohort is empty and no intervention outcome was opened. Registry verification passes with 27 live events / 110 output hashes.
- DPO staging: all 14 required safetensor links are present at revision `a38eddf84054b578970f53f31217df1556e69571`. An unrestricted Hub download began fetching redundant `.bin` weights; it was stopped, and only those seven completed `.bin` cache copies plus five incomplete `.bin` fragments were deleted (about 55 GB, recoverable from HF). The safetensor snapshot is retained for exact preflight.
- Think-DPO result: `ol2-stage-wedge-think-dpo-tier1-v1` registered at commit `ebaadb0`. All 972 G5 rows are finite and unique; exact normalized capability is 0.00309 overall and 0.00278 on Bank S. Again zero Bank-S facts are capable on both direct and composed variants, so the cohort is empty and no intervention outcome was opened. With both stages gated, no capability onset is observed inside SFT-to-DPO and no effect contrast is estimable. Registry verification passes with 28 live events / 116 output hashes.

Recovery: commit/push the DPO registry row and this note, then implement/run the CPU-only `ol2-stage-wedge-joint-analysis-v1` router from the two registered stage events. The frozen route is not capability onset because DPO did not become capable; intervention effects are missing, not zero. Do not alter capability rules or prompts. After the joint event is banked, import the registered Gemma ceiling and proceed to mandatory H6.
