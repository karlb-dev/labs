# Part 3 Validation Report - Labs 26-27

- **Date:** 2026-06-14
- **Branch:** `interp_cleanup`
- **Inspected commit:** `5342cb0`
- **Scope:** Lab 26 causal abstraction and Lab 27 path mediation after the latest pushed edits.
- **Drive artifacts:** `/content/drive/MyDrive/interpret/verify_part3/`

## Headline

Labs 26 and 27 both run cleanly on Tier A (`gpt2`, fp32, small prompt set), write
complete artifact manifests, and pass their instrumentation self-checks. I did
not find a runtime bug, table inconsistency, missing plot source, or MD/PY
contract mismatch that required a code patch in this pass.

The requested larger-model matrix is not complete because this Colab VM cannot
currently see a CUDA device:

```text
torch 2.11.0+cu128
torch.version.cuda 12.8
torch.cuda.is_available() False
torch.cuda.device_count() 0
nvidia-smi: couldn't communicate with the NVIDIA driver
```

I did not attempt 7B, 32B, Gemma, or Think model loads on CPU, because those
would test host failure mode rather than the labs.

## Copied Runs

| Lab | Local run | Drive copy |
|---|---|---|
| 26 | `interpretability/runs/lab26_causal_abstraction-20260614_203629-de40c2` | `interpret/verify_part3/lab26_full_verify_20260614_203629/lab26_causal_abstraction-20260614_203629-de40c2` |
| 27 | `interpretability/runs/lab27_path_mediation-20260614_203645-fa3f8f` | `interpret/verify_part3/lab27_full_verify_20260614_203645/lab27_path_mediation-20260614_203645-fa3f8f` |

## Run Matrix

| Lab | Requested target | Status | Notes |
|---|---|---|---|
| 26 | Tier A `gpt2` | complete | 600 intervention rows, 57 artifacts, 7 plot entries, no missing artifact or plot links. |
| 27 | Tier A `gpt2` | complete | 52 path rows, 156 control rows, 57 artifacts, 8 plot entries, no missing artifact or plot links. |
| 26-27 | Tier B `allenai/Olmo-3-1025-7B` | host-blocked | Needs GPU; current VM reports no CUDA device. |
| 26-27 | Tier C `allenai/Olmo-3-1125-32B` | host-blocked | Needs 40-80GB GPU; current VM reports no CUDA device. |
| 26-27 | `google/gemma-4-E4B-it` | host-blocked | Ordinary `--model` override once GPU is available. |
| 26-27 | `allenai/Olmo-3-32B-Think` | host-blocked | Ordinary `--model` override once GPU is available. |

## Lab 26 Findings

Tier A command:

```bash
cd interpretability
python interp_bench.py --lab lab26 --tier a
```

Key checks:

- Frozen data used: `causal_abstraction_tasks.csv`, sha256 `1037918625043ba90b3b2d6daa2b17ced1bd3f5477161c666a99fae9d8ff6545`.
- Selected rows: 12 of 30, with `induction=6`, `relation=6`, `train=5`, `eval=7`.
- Hook parity passed with max absolute diff `0.0`.
- Logit-lens self-check passed.
- Patch no-op and named-site identity checks passed; max no-op delta `0.0001` against tolerance `0.05`.
- Tokenization gate kept 12 of 12 rows.
- Donor coverage had no missing preserve or break-variable donors.
- Artifact audit found 57 registered artifacts, 7 plot-manifest entries, and no broken links.

Evidence posture:

| Hypothesis | Train | Eval | Posture |
|---|---:|---:|---|
| `induction_copy_v1` | pass | fail | `train_supported_but_eval_failed_or_controls_leaked` |
| `relation_identity_v1` | fail | fail | `needs_refinement_or_negative_result` |

Interpretation: Lab 26 is teaching the right lesson. The induction hypothesis has
a real train-split handle, but it does not survive the held-out split cleanly
because the specificity gap falls below zero on eval. The relation hypothesis is
an honest negative: preservation and damage are visible, but wrong-site/control
behavior prevents a formal causal-abstraction claim. The counterexample gallery
and refinement log are therefore essential artifacts, not decorative extras.

## Lab 27 Findings

Tier A command:

```bash
cd interpretability
python interp_bench.py --lab lab27 --tier a
```

Key checks:

- Frozen data used: `path_mediation_tasks.csv`, sha256 `7a726c8e7f3fb894c2449c3e2c0b7f259aba5ebdccc65bb72b0ea4859bbe8b19`.
- Manifest hash matched exactly.
- Selected rows: 9 of 9, with 3 tasks each for `factual_recall`, `induction`, and `relation_swap`.
- Hook parity, logit-lens self-check, and patch no-op passed.
- Tokenization gate kept 9 of 9 rows.
- Artifact audit found 57 registered artifacts, 8 plot-manifest entries, and no broken links.
- Warning rows are expected and useful: all three domains emit `controls_match_path`, and the run writes 4 counterexample rows.

Evidence posture:

| Domain | Best mediated path recovery | Control floor | Specificity gap | Posture |
|---|---:|---:|---:|---|
| `factual_recall` | 1.0190 | 0.0117 | 1.0073 | `path_proxy_supported` |
| `induction` | 0.8458 | 0.0070 | 0.8388 | `path_proxy_supported` |
| `relation_swap` | 1.1394 | 0.0000 | 1.1394 | `path_proxy_supported` |

Interpretation: Lab 27 supports narrow residual path-proxy language across all
three Tier A domains. It should not be described as exact edge isolation or as
proof that a two-site clean patch is superadditive. The generated counterexamples
make that boundary explicit: several best node or joint-patch controls match or
beat broader interaction language.

## Reproduce When GPU Is Restored

```bash
cd interpretability

# Tier B and Tier C base-model checks
python interp_bench.py --lab lab26 --tier b
python interp_bench.py --lab lab27 --tier b
python interp_bench.py --lab lab26 --tier c
python interp_bench.py --lab lab27 --tier c

# Requested extra model checks
python interp_bench.py --lab lab26 --tier b --model google/gemma-4-E4B-it
python interp_bench.py --lab lab27 --tier b --model google/gemma-4-E4B-it
python interp_bench.py --lab lab26 --tier c --model allenai/Olmo-3-32B-Think
python interp_bench.py --lab lab27 --tier c --model allenai/Olmo-3-32B-Think
```

Use `--run-name` or a fresh validation folder for each run before copying the
artifacts to Drive.
