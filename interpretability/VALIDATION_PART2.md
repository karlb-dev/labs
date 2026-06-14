# Part 2 Validation Report — Advanced Labs 12–25

- **Date:** 2026-06-14
- **Branch:** `interpret_part2`
- **Tier A (smoke):** CPU/GPU, `HuggingFaceTB/SmolLM2-135M-Instruct` (fp32) or lab default
- **Tier B (main):** A100-80GB, `allenai/Olmo-3-7B-Instruct` (bf16), `--prompt-set full`
- **Run artifacts:** Google Drive `interpret/validate_part2/labNN/tierb/`

## Headline

All advanced labs **12–25 run green on tier B** (Olmo-3-7B-Instruct, bf16) and pass
their built-in self-checks (hook parity, logit-lens self-check, turn-boundary,
KV-cache parity, leak scans). Labs 21–25 also pass tier-A smoke. Two correctness
bugs were found and fixed (lab16, lab17); both are committed and pushed.

## Fixes landed this pass (committed + pushed)

| Commit | Lab | Fix |
|---|---|---|
| `cf6c911`/`023f0f9` | 16 | **Probe site/depth selected on out-of-sample grouped-CV AUC** instead of in-sample train AUC. In-sample selection overfit to `user_belief_span/depth5` (train 0.976 / eval 0.447); CV selection picks `assistant_boundary/depth24`. |
| `cf6c911`/`023f0f9` | 17 | **Turn-boundary gate** rewritten to mirror lab15: content spans derived from the full chat render with a forward cursor, gated on span validity (decode==content, ws-normalized) instead of `incremental_token_prefix_stable`, which fails benignly on Olmo's count-preserving EOS swap (`im_end` vs `endoftext`). |
| `8aa1889` | 15 | (prior) same multi-turn instrumentation fix + dtype-aware KV-cache parity (bf16 path-rounding tolerance). |

The lab16 fix is scientifically material: it flips the user-belief verdict from
`not_validated_or_control_matched` (AUC 0.447, **below chance**) to
`validated_selective` (AUC **0.689** / control 0.527; local-truth 0.929). The
overfit selection had been hiding a real, selective user-belief direction.

## Per-lab tier-B results

| Lab | Topic | Status | Key result |
|---|---|---|---|
| 12 | Relation directions / patching | green | probe `validated_selective`; patching honestly `not_validated_by_controls` |
| 13 | (prior validated) | green | — |
| 14 | (prior validated) | green | — |
| 15 | Multi-turn harness | green | turn-boundary + KV-cache parity pass on bf16 Olmo |
| 16 | Sycophancy / user belief | green ✅ | user-belief AUC 0.689 (`validated_selective`); local-truth 0.929; sycophancy rate 0.021 |
| 17 | Persona / voice / register | green ✅ | held-out persona AUC 1.0 vs shuffled 0.49; `decodable_but_not_steerable_by_this_run` |
| 18 | Humor as incongruity | green | depth-12 probe AUC ≈1.0; steering specificity gap −0.875 (`not steerable`) |
| 19 | Crosscoder model-diffing | green | 572 prompts; taxonomy {asymmetric 55, shared 73}; 128 features |
| 20 | Model-organism construction | green | 5 sealed blind packages + private construction emitted |
| 21 | LoRA localization / safety-depth | green | safety divergence rows 0; access `public_or_adapter_only` (safety wall holds) |
| 22 | Eval awareness | green | honest null: depth-3 AUC 1.0 == format control → `not_validated_or_format_detector` |
| 23 | Blind audit | green ✅ | full round-trip: discover 5 → leak-scan → commit-freeze → unseal → `scored` |
| 24 | Knowledge conflict / belief revision | green | override depth traces + quadrant audit emitted |
| 25 | Find the wire / self-report | green | self-report grounding + source attribution; `not_validated_by_controls` |

## MD/PY consistency audit (fan-out of 5 read-only agents over labs 16-25)

Labs 18, 19, 20, 22, 23, 25 had **no factual MD/PY discrepancies**. Three real
issues were found and **fixed** (commit `f7274fa`):

- **Lab 16 MD** still documented the old in-sample train-AUC selection rule;
  updated to describe the grouped 5-fold CV selection the PY now uses. *(resolved)*
- **Lab 21** `training_depth_card.md` unconditionally told students to open
  `safety_depth`-only files even in a `lora`-only run; the "Read next" list is now
  mode-aware and the MD artifact tree marks conditional outputs. *(resolved)*
- **Lab 24** dataset expanded 6 → 16 dialogues; verified `--mode both` produces a
  full 96-row quadrant table. *(resolved)*
- Lab 17 "leave-one-out" vs "leave-one-topic-out" wording: minor, left as-is.

## Items flagged for further experiment improvement (not blockers)

1. **Lab 22 (eval awareness):** the probe trivially separates formats at depth 3 with
   AUC 1.0, and the format-control matches it — the lab correctly self-diagnoses a
   "format detector," but the *dataset* makes the interesting eval-awareness signal
   unreachable. Candidate: matched surface forms across eval/natural so the contrast
   isn't decodable by formatting tokens. (Open.)
2. **Lab 18 (humor):** decodable (AUC ≈1.0) but steering is non-specific
   (gap −0.875). Honest result; worth a teaching note that incongruity is *readable*
   but not cleanly *writable* by this direction. (Open.)
3. **Small vs full prompt sets:** several labs default to `--prompt-set small`, which
   produces degenerate early-layer selections (e.g. lab18 depth-1, AUC 1.0 on n=8).
   Tier-B validation must use `--prompt-set full`; consider making `full` the tier-B
   default so a naive run doesn't pick a noise layer. (Open.)

## Self-checks confirmed passing on Olmo (tier B)

- Exact chat-hook parity (max |diff| = 0) on all probe labs
- Logit-lens self-check (top-1 lens==model, top-5 overlap 5/5)
- Turn-boundary content-span validity (labs 15, 17)
- KV-cache parity within bf16 path-rounding tolerance (lab 15)
- Public-leak scan + commitment freeze + blinding verification (lab 23)
- Safety wall: no unsafe sampling / no refusal ablation (lab 21)

## Reproduce

```
export HF_HOME=/content/drive/MyDrive/hf_cache HF_HUB_DISABLE_TELEMETRY=1
cd interpretability
python interp_bench.py --lab labNN --tier b --prompt-set full --run-name labNN_tierb
# lab23 needs an organism source:
LAB23_ORGANISM_DIR=runs/lab20_..._tierb_full LAB23_UNSEAL=1 \
  python interp_bench.py --lab lab23 --tier b --run-name lab23_unseal
```
