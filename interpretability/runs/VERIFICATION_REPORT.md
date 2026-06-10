# Lab 1 verification report — patch merge and real-model validation

Date: 2026-06-10 · Machine: Colab A100-SXM4-80GB · Git: `lab1_colab` @ `ada42b3`

## 1. What was merged

The revised Lab 1 bundle (`lab1_patch/revised_interpretability_lab1/`) was merged into the
working tree. Provenance check: applying `lab1_upgrade.patch` to the pre-merge files
reproduced the bundled revised files **byte-for-byte** (all three files `diff`-identical),
so the drop-in replacement and the patch are the same change.

## 2. Code review findings fixed before running

Two independent full-file reviews (bench + lab module, including an interface-contract
audit of every symbol the lab imports from the bench) found the patch clean on security
and transformers-5.x compatibility, and surfaced these issues, all fixed in `ada42b3`:

- **Credential leak**: `run_metadata.json` recorded `HF_*` env values verbatim, including
  tokens; values matching TOKEN/SECRET/KEY/PASSWORD are now redacted.
- **bf16 lens self-check flake**: exact top-1 agreement required with no tolerance; a
  near-tie prompt could abort a valid run. Now accepts a flip only when the model's own
  top-1 logit gap is within the observed numeric noise floor and top-5 overlap ≥ 4,
  recorded as `near_tie_accepted` in the diagnostic.
- **Latent infinite loop** in `interleave_by_category` on unknown categories.
- **Unfriendly crash** (`FileNotFoundError`) on a mistyped `--prompt-set`.
- **Matplotlib figure leaks** on empty-data early returns in two plot functions.
- **Survivorship bias** in category event medians: medians were silently conditional on
  the event occurring; paired `n_*` occurrence columns added to `category_summary.csv`,
  the headline table, and the drafted claims (e.g. "median 23.5 (over 8/12 examples)").
- `artifact_index.json` now written even if a lab run fails partway.
- Silent no-op `--showcase` now warns; help-text corrections.

## 3. Self-verification of the instrument (both runs)

| Check | Tier A (gpt2, fp32) | Tier B (Olmo-3-7B, bf16, A100) |
|---|---|---|
| Hook parity (hooks vs `output_hidden_states`) | OK — max abs diff = 0, 12/12 layers | OK — max abs diff = 0, 32/32 layers |
| Lens self-check (lens(L) vs model logits) | OK — top-1 exact, max diff 1e-4 | OK — top-1 exact, max diff 0.0000 |
| Examples kept / dropped at tokenization | 4 / 0 | 32 / 0 |
| Run completed, all plots + summary written | yes (4.0 s) | yes (8.6 s after 40 s model load) |

Both runs verified the instrument **before** taking any measurement, per the course's
"instrument verifies itself" contract. Every artifact in both runs is indexed in
`artifact_index.json` with size and SHA256.

## 4. Runs included

- `lab01_residual_logit_lens-20260610_231101-fe8a0f/` — Tier A CPU-class smoke
  (gpt2, small set, 4 examples). Plumbing correctness, not science.
- `lab01_tierb_full_controls/` — Tier B science run: `allenai/Olmo-3-1025-7B`,
  bf16, full 32-prompt set **including the optional control family**, top-10 readouts.
  225 artifact files, 3.6 MB.

## 5. Headline result (Tier B)

| category | n | median decision depth | final entropy (bits) | target final top-1 rate |
|---|---:|---:|---:|---:|
| fact | 12 | 29.5/32 | 3.892 | 0.25 |
| ambiguous | 8 | 30.0/32 | 6.488 | — |
| counterfactual | 8 | 23.5/32 | 4.908 | 0.875 |
| control | 4 | 31.0/32 | 8.395 | — |

The four prompt families separate exactly as the lab intends: facts converge to
low-entropy answers, ambiguous prompts stay uncertain, weak controls stay near-uniform
and stabilize only at the last layers, and counterfactual contexts override stored facts
(7/8 final top-1 = in-context target, median target-over-distractor crossing at depth 1.5).
Four drafted OBS-level claims with falsifiers are in `ledger_suggestions.md` per run.

## 6. Environment pins (validation-time freeze)

```
torch==2.11.0+cu128
transformers==5.10.1
tokenizers==0.22.2
accelerate==1.13.0
safetensors==0.7.0
numpy==2.0.2
matplotlib==3.10.0
```

Python 3.12.13 · CUDA 13.0 driver 580.82.07 · NVIDIA A100-SXM4-80GB

## 7. How to audit

Per run: read `run_summary.md`, then one `state/<id>/state_card.md`, then `plots/`,
then `results.csv`; `diagnostics/` holds the parity/lens self-check evidence and the
full tokenization report. `logs/console.log` is the complete run transcript.
