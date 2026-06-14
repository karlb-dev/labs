# Part 3 Validation Report - Labs 26-27

- **Date:** 2026-06-14
- **Branch:** `interp_cleanup`
- **Machine:** NVIDIA RTX PRO 6000 Blackwell Server Edition, 97.9 GB VRAM
- **Scope:** Full validation of Lab 26 and Lab 27 after the latest pushed edits.
- **Drive artifacts:** `/content/drive/MyDrive/interpret/verify_part3/`

## Headline

Labs 26 and 27 now validate across Tier A, Tier B, Tier C, Gemma 4 E4B, and
Olmo 3 32B Think, with all trusted runs copied to Drive. The earlier
"GPU unavailable" note was a sandbox artifact: CUDA is visible for escalated
commands and the bench runs correctly on the GPU.

One real Lab 26 bug surfaced on GPU-scale bf16 runs and was fixed in this pass:
Lab 26 had been scoring batched residual patches against clean margins cached
from single-prompt forwards. On large bf16 models this can create
batch-shape-dependent no-op drift. Lab 26 now:

- records raw no-op logit drift and normalized no-op score error separately;
- gates the no-op check on `abs(scrub_score - 1)` because that is the lab's
  plotted and aggregated metric;
- scores batched interventions against a same-batch clean reference;
- falls back to exact single-prompt patch forwards for Gemma 4, whose wrapper
  showed large batch-shape drift;
- exposes `LAB26_RESIDUAL_PATCH_BATCH_SIZE` for model-specific validation
  retries, used for Olmo 3 32B Think at batch size 24.

No Lab 27 code change was needed.

## Run Matrix

| Lab | Target | Status | Run |
|---|---|---|---|
| 26 | Tier A `gpt2` | green | `lab26_causal_abstraction-20260614_203629-de40c2` |
| 26 | Tier B `allenai/Olmo-3-1025-7B` | green | `lab26_tierb_full_verify_20260614_2116` |
| 26 | Tier C `allenai/Olmo-3-1125-32B` | green | `lab26_tierc_full_verify_20260614_2119` |
| 26 | `google/gemma-4-E4B-it` | green | `lab26_gemma4e4b_full_verify_20260614_2136` |
| 26 | `allenai/Olmo-3-32B-Think` | green | `lab26_olmo32bthink_full_verify_20260614_2150_b24` |
| 27 | Tier A `gpt2` | green | `lab27_path_mediation-20260614_203645-fa3f8f` |
| 27 | Tier B `allenai/Olmo-3-1025-7B` | green | `lab27_tierb_full_verify_20260614_2202` |
| 27 | Tier C `allenai/Olmo-3-1125-32B` | green | `lab27_tierc_full_verify_20260614_2205` |
| 27 | `google/gemma-4-E4B-it` | green | `lab27_gemma4e4b_full_verify_20260614_2214` |
| 27 | `allenai/Olmo-3-32B-Think` | green | `lab27_olmo32bthink_full_verify_20260614_2217` |

Every successful run has:

- 57 registered artifacts;
- complete plot manifests with no missing figure or source-table links;
- hook parity passing;
- logit-lens self-check passing;
- patch no-op check passing;
- tokenization gates passing.

## Drive Copies

Each run above was copied under:

```text
/content/drive/MyDrive/interpret/verify_part3/<run_name>/<run_name>/
```

The validation report itself is also copied as:

```text
/content/drive/MyDrive/interpret/verify_part3/VALIDATION_PART3_lab26_27_20260614.md
```

## Lab 26 Results

| Target | `induction_copy_v1` | `relation_identity_v1` | Notes |
|---|---|---|---|
| Tier A `gpt2` | train pass, eval fail | train fail, eval fail | Smoke result is intentionally cautious. |
| Tier B OLMo 7B | train pass, eval fail | train pass, eval fail | Both are train-only claims. |
| Tier C OLMo 32B | train pass, eval pass | train pass, eval fail | Induction survives held-out eval. |
| Gemma 4 E4B | train fail, eval fail | train pass, eval fail | Gemma requires exact single-prompt patching. |
| OLMo 32B Think | train pass, eval pass | train pass, eval fail | Induction survives held-out eval with batch size 24. |

Interpretation: Lab 26 is doing useful falsification. The induction abstraction
becomes stable on the 32B OLMo variants, but not on GPT-2, OLMo 7B, or Gemma.
The relation abstraction is consistently weaker: when it finds a handle, it is
train-only and should not be written as an eval-supported causal abstraction.

## Lab 27 Results

| Target | Factual recall | Induction | Relation swap | Notes |
|---|---|---|---|---|
| Tier A `gpt2` | path-proxy supported | path-proxy supported | path-proxy supported | Tiny smoke path, useful but low-scale. |
| Tier B OLMo 7B | path-proxy supported | path-proxy supported | path-proxy supported | Science-ready. |
| Tier C OLMo 32B | path-proxy supported | path-proxy supported | path-proxy supported | Science-ready. |
| Gemma 4 E4B | path-proxy supported | node-effect only | behavior gate failed | Strong cross-model negative control. |
| OLMo 32B Think | path-proxy supported | path-proxy supported | path-proxy supported | Science-ready. |

Interpretation: Lab 27 is robust on OLMo base and Think models. Gemma is more
selective: factual recall supports residual path-proxy language, induction does
not beat node-effect caveats strongly enough, and relation-swap fails behavior
gates. The lab correctly narrows claims rather than forcing a positive result.

## Reproduce

```bash
cd interpretability

# Lab 26
python interp_bench.py --lab lab26 --tier a
python interp_bench.py --lab lab26 --tier b --low-cpu-mem-usage
python interp_bench.py --lab lab26 --tier c --low-cpu-mem-usage
python interp_bench.py --lab lab26 --tier b --model google/gemma-4-E4B-it --low-cpu-mem-usage
LAB26_RESIDUAL_PATCH_BATCH_SIZE=24 \
  python interp_bench.py --lab lab26 --tier c --model allenai/Olmo-3-32B-Think --low-cpu-mem-usage

# Lab 27
python interp_bench.py --lab lab27 --tier a
python interp_bench.py --lab lab27 --tier b --low-cpu-mem-usage
python interp_bench.py --lab lab27 --tier c --low-cpu-mem-usage
python interp_bench.py --lab lab27 --tier b --model google/gemma-4-E4B-it --low-cpu-mem-usage
python interp_bench.py --lab lab27 --tier c --model allenai/Olmo-3-32B-Think --low-cpu-mem-usage
```

## Disk Note

After pulling the 7B, 32B, Think, and Gemma weights, local disk was at about
198 GB used of 236 GB, with the Hugging Face cache at about 150 GB. The run
artifacts themselves are small relative to the model cache. Further validation
should avoid pulling new large models unless cache space is reclaimed first.
