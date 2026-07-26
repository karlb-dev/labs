# special-lab-1 — J-space Replication on OLMo 3 32B (Thinking)

Replicate Anthropic's "Verbalizable Representations Form a Global Workspace in
Language Models" (transformer-circuits.pub/2026/workspace) on an open-weights
reasoning model, using the reference `jlens` package (github.com/anthropics/jacobian-lens,
cloned at /content/jacobian-lens, installed editable). Comparison points: the
paper's Claude numbers and Nanda's Qwen 3.6 27B replication (LessWrong review;
neuronpedia/jacobian-lens lenses).

**This lab is intentionally outside version control.** Never `git add` anything
here. The single source of durability is the Drive run dir.

## Run dir (resume of prior VM's run)

`/content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726/`
(`$SL1_RUN_DIR` overrides). Layout: `config/ lens/ metrics/ figures/ logs/
report/ code/`. VM1 (same GPU class) completed the 7B smoke test 17:26–17:41
then died with `code/` empty — every script is therefore mirrored to
`code/` via `sync_code.sh` the moment it changes, and every phase checkpoints
to Drive incrementally.

## Decisions (Phase 0)

- **Model: `allenai/Olmo-3-32B-Think`** (64 layers, d_model 5120, bf16 ≈64GB).
  This is the 32B Think model the repo's lab configs have actually run
  (validation/lab{07,10,12,23,26} tier-c). `Olmo-3.1-32B-Think` exists (64.5GB
  on HF) but no lab config has ever run it, its weights are not in the Drive
  cache (62GB free — does not fit), and local disk leaves ~1.5GB margin — too
  tight to run beside. Recorded as feasibility note, not a silent downgrade.
- **Feasibility tier: full bf16 on one 96GB GPU (RTX PRO 6000 Blackwell).**
  No quantization, no gradient checkpointing needed at fit shapes
  (dim_batch 8 × seq 128; measured VRAM recorded in metrics).
- **Fitting corpus: WikiText-103** (`Salesforce/wikitext`, cached on Drive) —
  same corpus family as the neuronpedia reference lenses
  (`.../Salesforce-wikitext/..._n1000.pt`), so lens recipes are comparable.
  Dolma would be thematically ideal for OLMo but adds a large fragile
  download for no methodological gain; noted in report limitations.
- **Smoke model: `allenai/Olmo-3-7B-Instruct`** (matches VM1's smoke lens).
  `Olmo-3-7B-Think` reserved for cheap Phase-4 prototyping if needed.
- Seeds: 0 everywhere (`seed_all` in `sl1_common.py`). HF cache stays on
  Drive (`HF_HUB_CACHE=/content/drive/MyDrive/hf_cache/hub`); weights are
  page-cached in RAM (176GB) after first read, so reloads are cheap.

## Phases → scripts (all in `special_lab1/`, run via `python scripts/<s>.py`)

| Phase | Script | Output (under run dir) |
|---|---|---|
| 0 env audit | `scripts/s0_env_audit.py` | `config/environment_vm2.json` |
| 0 smoke verify | `scripts/s1_smoke_verify.py` | `metrics/smoke_verify_vm2.json` — loads VM1's `lens/smoke7b_lens.pt`, re-runs its exact 11-probe battery, compares |
| 1 corpus | `scripts/s2_corpus.py` | `config/prompts/fitting_corpus.jsonl` (deterministic WikiText slice, seed 0) |
| 1 fit | `scripts/s3_fit_32b.py` | `lens/olmo32bthink_slice{k}.pt` per slice + `lens/olmo32bthink_lens.pt` (merged, fp16) + `metrics/fit_32b.json`; jlens per-prompt ckpt local, copied to Drive every 10 prompts |
| 1 sanity | `scripts/s4_lens_sanity.py` | `metrics/lens_sanity_32b.json` — boot-probe battery + `data/evaluations/lens-eval-multihop.json` pass@k, J-lens vs logit-lens |
| 2 descriptive | `scripts/s5_descriptive.py` | `metrics/descriptive_*.json` — active-concept counts (threshold sweep), variance share by layer, readout archives |
| 2 broadcast | `scripts/s6_broadcast.py` | `metrics/broadcast.json` — downstream fan-out of top J-directions vs matched random + non-J high-variance controls |
| 3 causal | `scripts/s7_ablation.py` | `metrics/ablation_*.json` — multi-step battery vs fluency controls; J-space projection vs random/non-J controls; 25/50/100% dose-response; bootstrap CIs |
| 4 CoT | `scripts/s8_cot.py` | `metrics/cot_*.json` — pre-CoT anticipation ranks, CoT-divergence events (top-20 verbatim), suppressed-think comparison |
| 5 stretch | only if time | eval-awareness differencing (flagged flakiest by Nanda) |
| report | `scripts/s9_report.py` | `figures/*` + `report/REPORT.md` + `report/summary.json`, all regenerable from metrics only |

## Fit sizing (measured, then tuned)

- Source layers (21): early controls [4,8,12,16], middle band [20..44 step 2]
  (paper: workspace ≈33%–92% depth; middle third of 64 = 21–42), late
  controls [48,52,56,60]. Target layer 63 (final block, paper default).
- 120 WikiText prompts × 128 tokens in 4 slices of 30, merged via
  `JacobianLens.merge`. README: quality saturates fast, ~100 usable
  (paper 1000, Nanda 25). Early-stop rule: if multihop pass@k on the
  slice-1..k merged lens plateaus (<1% absolute gain) after slice 3, stop
  and record.
- Cost estimate: ceil(5120/8)=640 activation-grad backwards per prompt over
  the L4→L63 graph ≈ 3–4 min/prompt ≈ 6–8h total, checkpointed. Revisit
  dim_batch (→12) and per-block torch.compile after timing 2 prompts.

## Paper targets to compare against

- ~10–25 active concepts per (position, layer); J-space ≈6–10% of activation
  variance; workspace band ≈33%→92% depth; ablating top-10 J-directions
  across the band kills multi-hop reasoning while classification/extractive
  QA/fluency survive; CoT partially rescues (externalization).

## Checkpoint policy

Every script: (a) is a no-op re-run if its output already exists (`--force`
overrides); (b) writes partial results as it goes (per fitting chunk, per
ablation batch); (c) writes JSON atomically (tmp+rename). Code synced to
Drive by `sync_code.sh` after every edit session; LOG.md records decisions.
