# Part 3 Validation Report - Labs 26-35

- **Date:** 2026-06-15
- **Branch:** `interp_cleanup`
- **Machine:** NVIDIA RTX PRO 6000 Blackwell Server Edition, 97.9 GB VRAM
- **Scope:** Full validation of Labs 26-35 after the latest pushed edits.
- **Drive artifacts:** `/content/drive/MyDrive/interpret/verify_part3/`

## Headline

Labs 26, 27, 28, 30, 32, and 34 now validate across Tier A, Tier B, Tier C,
Gemma 4 E4B, and Olmo 3 32B Think, with all trusted runs copied to Drive. Labs
29, 31, 33, and 35 validate across Tier A/B/C; they intentionally use a
`gpt2` outer runner or offline/synthetic/scaffolded paths where Gemma/Think
overrides would not change the lab's scientific result. The earlier "GPU
unavailable" note was a sandbox artifact: CUDA is visible for escalated
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

One real Lab 28 bug surfaced on GPU-scale bf16 runs and was fixed in this pass:
the localization stage used batched replacement patching while comparing
self-patch no-op rows against clean margins from single-prompt forwards. Lab 28
now runs localization replacement patches as individual prompt forwards. This
keeps the self-patch identity check exact and matches the already-unbatched
additive edit path.

One Lab 32 bug surfaced in Tier A smoke mode and was fixed in this pass: the
cap sampler could select only one eval preference label, making held-out AUC
undefined in the smallest run. Lab 32 now balances Tier A caps by split,
preference label, and domain where possible. Its run summary also now reports
the original/swap preference shifts, random-direction shift, swap-control gap,
and the final causal support gate so positive-over-random movement cannot be
mistaken for a supported causal result.

No Lab 27, 29, 30, 31, 33, 34, or 35 code change was needed during this
validation pass.

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
| 28 | Tier A `gpt2` | green | `lab28_tiera_full_verify_20260614_2243` |
| 28 | Tier B `allenai/Olmo-3-1025-7B` | green | `lab28_tierb_full_verify_20260614_2252` |
| 28 | Tier C `allenai/Olmo-3-1125-32B` | green | `lab28_tierc_full_verify_20260614_2256` |
| 28 | `google/gemma-4-E4B-it` | green | `lab28_gemma4e4b_full_verify_20260614_2303` |
| 28 | `allenai/Olmo-3-32B-Think` | green | `lab28_olmo32bthink_full_verify_20260614_2307` |
| 29 | Tier A `gpt2` outer runner | green | `lab29_tiera_full_verify_20260614_2317` |
| 29 | Tier B `gpt2` outer runner | green | `lab29_tierb_full_verify_20260614_2320` |
| 29 | Tier C `gpt2` outer runner | green | `lab29_tierc_full_verify_20260614_2322` |
| 30 | Tier A `gpt2` | green | `lab30_tiera_full_verify_20260615_0000` |
| 30 | Tier B `allenai/Olmo-3-1025-7B` | green | `lab30_tierb_full_verify_20260614_2223` |
| 30 | Tier C `allenai/Olmo-3-1125-32B` | green | `lab30_tierc_full_verify_20260614_2226` |
| 30 | `google/gemma-4-E4B-it` | green | `lab30_gemma4e4b_full_verify_20260614_2235` |
| 30 | `allenai/Olmo-3-32B-Think` | green | `lab30_olmo32bthink_full_verify_20260614_2241` |
| 31 | Tier A `gpt2` outer runner | green | `lab31_tiera_full_verify_20260614_2250` |
| 31 | Tier B `gpt2` outer runner | green | `lab31_tierb_full_verify_20260614_2252` |
| 31 | Tier C `gpt2` outer runner | green | `lab31_tierc_full_verify_20260614_2255` |
| 32 | Tier A `gpt2` | green | `lab32_tiera_final_verify_20260614_2340` |
| 32 | Tier B `allenai/Olmo-3-1025-7B` | green | `lab32_tierb_final_verify_20260614_2343` |
| 32 | Tier C `allenai/Olmo-3-1125-32B` | green | `lab32_tierc_final_verify_20260614_2347` |
| 32 | `google/gemma-4-E4B-it` | green | `lab32_gemma4e4b_final_verify_20260614_2352` |
| 32 | `allenai/Olmo-3-32B-Think` | green | `lab32_olmo32bthink_final_verify_20260614_2357` |
| 33 | Tier A `gpt2` outer runner | green | `lab33_tiera_full_verify_20260615_0005` |
| 33 | Tier B `gpt2` outer runner | green | `lab33_tierb_full_verify_20260615_0009` |
| 33 | Tier C `gpt2` outer runner | green | `lab33_tierc_full_verify_20260615_0012` |
| 34 | Tier A `gpt2` | green | `lab34_tiera_full_verify_20260615_0020` |
| 34 | Tier B `allenai/Olmo-3-1025-7B` | green | `lab34_tierb_full_verify_20260615_0026` |
| 34 | Tier C `allenai/Olmo-3-1125-32B` | green | `lab34_tierc_full_verify_20260615_0034` |
| 34 | `google/gemma-4-E4B-it` | green | `lab34_gemma4e4b_full_verify_20260615_0044` |
| 34 | `allenai/Olmo-3-32B-Think` | green | `lab34_olmo32bthink_full_verify_20260615_0050` |
| 35 | Tier A `gpt2` outer runner | green | `lab35_tiera_full_verify_20260615_0100` |
| 35 | Tier B `gpt2` outer runner | green | `lab35_tierb_full_verify_20260615_0104` |
| 35 | Tier C `gpt2` outer runner | green | `lab35_tierc_full_verify_20260615_0107` |

Every successful run has:

- complete registered artifacts and plot manifests, including 85 artifacts for
  Lab 30, 66 for Lab 31, 77 for Lab 32, 73 for Lab 33, 94 for Lab 34, and 84
  for Lab 35;
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
/content/drive/MyDrive/interpret/verify_part3/VALIDATION_PART3_20260614.md
/content/drive/MyDrive/interpret/verify_part3/VALIDATION_PART3_20260615.md
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

## Lab 28 Results

| Target | Supported rows | Dominant caveat | Notes |
|---|---:|---|---|
| Tier A `gpt2` | 4 / 5 | one control-limited row | Smoke path is healthy. |
| Tier B OLMo 7B | 1 / 8 | retain damage | Strong edits often damage retain prompts. |
| Tier C OLMo 32B | 1 / 8 | retain damage | Same pattern as Tier B. |
| Gemma 4 E4B | 3 / 8 | side effects and baseline eligibility | More target-specific than OLMo on this set. |
| OLMo 32B Think | 2 / 8 | retain damage | Strong target movement, limited by side effects. |

Interpretation: Lab 28 is doing the right audit work. It can often move the
target margin, but the retain/paraphrase/neighbor audits prevent most rows from
becoming broad edit claims. The strongest lesson is that localization and target
movement are not enough; a reversible activation edit must survive retain and
control checks before it earns positive language.

## Lab 29 Results

| Target | Final induction accuracy | Final heldout/test accuracy | Final probe selectivity | Event step | Notes |
|---|---:|---:|---:|---:|---|
| Tier A tiny run | 1.0 | 1.0 | 1.0 | 20 | 80 training steps, 11 selected rows. |
| Tier B tiny run | 1.0 | 1.0 | 1.0 | 17 | 420 training steps, all 14 rows. |
| Tier C tiny run | 1.0 | 1.0 | 1.0 | 17 | Same tiny setup as Tier B. |

Interpretation: Lab 29 validates the controlled tiny-training time-lapse. It
supports threshold-ordering language, not exact birth-step language. Tier B/C
warnings correctly flag the rotated-label probe caveat: do not overclaim
readable target identity when a control is close.

## Lab 30 Results

| Target | Supported domains | Mean eval AUC | Mean lineage lift | Notes |
|---|---:|---:|---:|---|
| Tier A `gpt2` | 7 / 8 | 1.0 | 0.4235 | `code` needed refinement; useful smoke-only caveat. |
| Tier B OLMo 7B | 8 / 8 | 0.9366 | 0.4743 | Science-ready supervised lineage directions. |
| Tier C OLMo 32B | 8 / 8 | 0.9413 | 0.4966 | Strongest base-model lineage result. |
| Gemma 4 E4B | 1 / 8 | 0.5958 | 0.3654 | Mostly negative cross-model control; only finance supported. |
| OLMo 32B Think | 7 / 8 | 0.8907 | 0.4892 | Sports was decodability-limited; other domains supported. |

Interpretation: Lab 30 supports scoped "recurring supervised direction" language
on OLMo base and Think models. It does not support monosemantic feature
identity, SAE/crosscoder claims, or external cross-model feature equivalence.
Gemma is a useful negative/heterogeneity result rather than a failure of the
lab.

## Lab 31 Results

| Target | Best supported method | Science-ready | Notes |
|---|---|---|---|
| Tier A tiny run | `structured_local` | smoke-only | 10 selected rows; structured labels beat controls. |
| Tier B full run | `structured_local` | true | 12 rows, 22 human-review rows, 66 artifacts. |
| Tier C full run | `structured_local` | true | Same offline benchmark as Tier B. |

Interpretation: Lab 31 validates the automated-interpretability audit scaffold.
`structured_local` earns the bounded support claim; `test_aware` and
`gold_calibration` remain upper-bound or over-abstaining references, and
shuffled-context controls correctly block cheap surface labels.

## Lab 32 Results

| Target | DPO proxy AUC | Preference direction AUC | Causal gate | Notes |
|---|---:|---:|---|---|
| Tier A `gpt2` | 0.25 | 0.5 | not supported | Smoke run now has defined eval classes after the sampler fix. |
| Tier B OLMo 7B | 0.7037 | 1.0 | not supported | Direction supported; DPO proxy shortcut-limited. |
| Tier C OLMo 32B | 0.7778 | 1.0 | not supported | Direction supported; causal intervention did not clear controls. |
| Gemma 4 E4B | 0.5 | 0.8889 | not supported | Cross-model negative result despite some over-random motion. |
| OLMo 32B Think | 0.8333 | 1.0 | not supported | Strong decode result; no causal preference-shift claim. |

Interpretation: Lab 32 supports bounded preference-direction readout language
for the OLMo models, but no run supports a causal reward/preference-circuit
claim. The updated summary makes the swap-control and random-direction caveats
visible in the first reading pass.

## Lab 33 Results

| Target | Mean connector AUC | Mean visual patch recovery | Science-ready for real VLM | Notes |
|---|---:|---:|---|---|
| Tier A synthetic run | 1.0 | smoke run | false | Synthetic connector and shortcut traps only. |
| Tier B synthetic run | 1.0 | 0.5873 | false | Full 28-row synthetic audit. |
| Tier C synthetic run | 1.0 | 0.5873 | false | Same offline benchmark as Tier B. |

Interpretation: Lab 33 is intentionally a synthetic multimodal mechanism lab.
It validates connector plumbing, patch accounting, OCR/background shortcut
audits, and real-VLM readiness artifacts, but it blocks real-VLM science claims
until a genuine VLM binding is added.

## Lab 34 Results

| Target | Tool-needed decode | Which-tool decode | Causal letter-prompt shift | Notes |
|---|---|---|---|---|
| Tier A `gpt2` | supported | weak/surface-confounded | random/letter limited | Smoke run below science row count. |
| Tier B OLMo 7B | supported | supported | supported narrow | Best full positive run. |
| Tier C OLMo 32B | supported | surface-tied | limited | Tool-needed readout survives; action shift does not. |
| Gemma 4 E4B | supported | weak | limited | Strong need/no-need decode, weak selection. |
| OLMo 32B Think | supported | surface-tied | limited | Decode survives; causal shift fails controls. |

Interpretation: Lab 34 supports narrow tool-needed state readouts on all tested
models and a full toy-tool decode/letter-prompt causal result on OLMo 7B. It
does not support real-world agent reliability, persistent goals, or faithful
tool-use self-report claims.

## Lab 35 Results

| Target | Package checks | Package ready | Science-ready | Notes |
|---|---:|---|---|---|
| Tier A scaffold | 13 / 13 | true | false | 4 seed tracks; source run binding pending. |
| Tier B scaffold | 13 / 13 | true | false | 6 seed tracks; source run binding pending. |
| Tier C scaffold | 13 / 13 | true | false | 6 seed tracks; source run binding pending. |

Interpretation: Lab 35 validates the capstone packaging contract, not a new
source-lab science claim. The expected warnings are `science_ready_false`,
`source_run_unbound`, `review_pending`, and `negative_results_pending`; these
are correct until a frozen source run is bound into the capstone package.

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

# Lab 28
python interp_bench.py --lab lab28 --tier a
python interp_bench.py --lab lab28 --tier b --low-cpu-mem-usage
python interp_bench.py --lab lab28 --tier c --low-cpu-mem-usage
python interp_bench.py --lab lab28 --tier b --model google/gemma-4-E4B-it --low-cpu-mem-usage
python interp_bench.py --lab lab28 --tier c --model allenai/Olmo-3-32B-Think --low-cpu-mem-usage

# Lab 29
python interp_bench.py --lab lab29 --tier a
python interp_bench.py --lab lab29 --tier b
python interp_bench.py --lab lab29 --tier c

# Lab 30
python interp_bench.py --lab lab30 --tier a
python interp_bench.py --lab lab30 --tier b --low-cpu-mem-usage
python interp_bench.py --lab lab30 --tier c --low-cpu-mem-usage
python interp_bench.py --lab lab30 --tier b --model google/gemma-4-E4B-it --low-cpu-mem-usage
python interp_bench.py --lab lab30 --tier c --model allenai/Olmo-3-32B-Think --low-cpu-mem-usage

# Lab 31
python interp_bench.py --lab lab31 --tier a
python interp_bench.py --lab lab31 --tier b
python interp_bench.py --lab lab31 --tier c

# Lab 32
python interp_bench.py --lab lab32 --tier a
python interp_bench.py --lab lab32 --tier b --low-cpu-mem-usage
python interp_bench.py --lab lab32 --tier c --low-cpu-mem-usage
python interp_bench.py --lab lab32 --tier b --model google/gemma-4-E4B-it --low-cpu-mem-usage
python interp_bench.py --lab lab32 --tier c --model allenai/Olmo-3-32B-Think --low-cpu-mem-usage

# Lab 33
python interp_bench.py --lab lab33 --tier a
python interp_bench.py --lab lab33 --tier b
python interp_bench.py --lab lab33 --tier c

# Lab 34
python interp_bench.py --lab lab34 --tier a
python interp_bench.py --lab lab34 --tier b --low-cpu-mem-usage
python interp_bench.py --lab lab34 --tier c --low-cpu-mem-usage
python interp_bench.py --lab lab34 --tier b --model google/gemma-4-E4B-it --low-cpu-mem-usage
python interp_bench.py --lab lab34 --tier c --model allenai/Olmo-3-32B-Think --low-cpu-mem-usage

# Lab 35
python interp_bench.py --lab lab35 --tier a
python interp_bench.py --lab lab35 --tier b
python interp_bench.py --lab lab35 --tier c
```

## Disk Note

After validating Labs 26-35, local disk was at about 199 GB used of 236 GB
with 38 GB free. Drive showed about 36 GB free. The Hugging Face cache was
about 150 GB, local run artifacts were about 551 MB, and the Drive validation
folder was about 523 MB. Further validation should avoid pulling uncached large
models unless cache space is reclaimed first.
