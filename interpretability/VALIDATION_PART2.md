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

## Items flagged for experiment/dataset improvement (not blockers)

1. **Lab 22 (eval awareness):** the probe trivially separates formats at depth 3 with
   AUC 1.0, and the format-control matches it — the lab correctly self-diagnoses a
   "format detector," but the *dataset* makes the interesting eval-awareness signal
   unreachable. Candidate: matched surface forms across eval/natural so the contrast
   isn't decodable by formatting tokens.
2. **Lab 24 (belief revision):** only **6 dialogues** selected from
   `belief_revision_dialogues.csv` on tier B — thin for a quadrant audit. Candidate:
   expand the dialogue set so each revision quadrant has enough support.
3. **Lab 18 (humor):** decodable (AUC ≈1.0) but steering is non-specific
   (gap −0.875). Honest result; worth a teaching note that incongruity is *readable*
   but not cleanly *writable* by this direction.
4. **Small vs full prompt sets:** several labs default to `--prompt-set small`, which
   produces degenerate early-layer selections (e.g. lab18 depth-1, AUC 1.0 on n=8).
   Tier-B validation must use `--prompt-set full`; consider making `full` the tier-B
   default so a naive run doesn't pick a noise layer.

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
