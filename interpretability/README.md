# Mechanistic Interpretability Labs

Hands-on labs that teach interpretability as an experimental craft on open
models. The course design lives in [COURSE.md](COURSE.md) and the lab-authoring
guide in [how_to_design_labs.md](how_to_design_labs.md). This README covers the
code: what runs, where, and how to read what it produces.

The project follows the same pattern as the collective-communication course:
**one shared bench script** (`interp_bench.py`) owns the experiment machinery —
CLI, run directories, console logging, diagnostics, model loading, hooks,
readouts, state dumps, plots — and **thin lab modules** under `labs/` own the
experiments. The bench is the microscope; the labs are what you point it at.

## Primary target

The primary target is **one NVIDIA A100/H100 (Colab)** running Hugging Face
`transformers` in bf16. Every lab also has a CPU smoke path (`--tier a`,
gpt2) that must work on a laptop — debug there, spend GPU minutes on science.

| Tier | Hardware | What runs |
|---|---|---|
| A — smoke | laptop CPU (or MPS) | `gpt2`, 4 examples; correctness of plumbing, not science |
| B — standard | Colab A100/H100, or any 24 GB+ GPU | full labs on `allenai/Olmo-3-1025-7B` in bf16 |
| C — comfortable | 40–80 GB GPU | fp32, larger prompt sets |

## Quick start

```bash
cd interpretability
pip install -r requirements.txt

# CPU smoke test (always do this first):
python interp_bench.py --lab lab1 --tier a

# Full Lab 1 on a Colab A100/H100:
python interp_bench.py --lab lab1 --tier b --prompt-set full

# Lab 2 (direct logit attribution), same pattern:
python interp_bench.py --lab lab2 --tier a
python interp_bench.py --lab lab2 --tier b --prompt-set full --topk 10

# Lab 3 (attention routing; the bench auto-sets eager attention):
python interp_bench.py --lab lab3 --tier a
python interp_bench.py --lab lab3 --tier b --prompt-set full --topk 10
```

On Colab: `Runtime > Change runtime type > A100`, then in a cell:

```python
!git clone https://github.com/<you>/labs.git
%cd labs/interpretability
!pip install -q -r requirements.txt
!python interp_bench.py --lab lab1 --tier b --prompt-set full
```

## Current status

- `interp_bench.py` — shared bench: run dirs, console tee, diagnostics
  (packages/git/GPU/env), model anatomy resolution, residual-stream capture
  with verified semantics, logit lens, verified per-block component capture
  (attn/MLP contributions, post-norm aware), direct-path component ablation,
  human-readable state dumps, plots, claim-ledger plumbing. Implemented and
  validated on gpt2 (fp32) and Olmo-3-7B (bf16, A100).
- Lab 1: residual stream and logit lens — implemented and validated (Tier A+B).
- Lab 2: direct logit attribution — implemented and validated (Tier A+B).
  Adds two instrument self-checks: the component-anatomy probe (hook points
  are verified against per-block residual deltas, not assumed) and the
  decomposition check (components must sum to the final pre-norm stream).
- Lab 3: attention routing, head motifs, induction — implemented and
  validated (Tier A+B). Adds head-level capture (attention patterns require
  eager — the bench forces it, since sdpa returns an empty attentions tuple
  silently in transformers 5), a verified per-head decomposition check, and
  scoped head ablation (final-position vs all-position — the gap measures
  composition).
- Labs 4–11 — designed in COURSE.md, not yet implemented. Lab 4 (probing
  with controls) is next.

## Design decisions (deviations from COURSE.md, on purpose)

1. **Raw HF `transformers` + explicit hooks instead of TransformerLens.**
   COURSE.md proposes TransformerLens 3 / TransformerBridge. Labs 1–2 need
   only residual caching and the unembedding, which raw HF does in ~50
   transparent lines — and a course rule is that nobody runs code they can't
   explain. Revisit when the patching labs (5+) need heavier intervention
   machinery; the bench's hook layer is the abstraction point.
2. **GPU instead of TPU.** The collective-comms course ran on Cloud TPU; this
   one targets Colab A100/H100 with plain PyTorch.
3. **No binary-blob artifacts by default.** Model state is dumped as token
   tables, per-layer statistics, and *decoded* top-k readouts (CSV + markdown
   state cards). Raw tensors are opt-in (`--save-tensors`) and always carry a
   manifest saying what every tensor is.

## The instrument verifies itself

Every run performs self-checks before any science, and aborts on failure:

- **Hook parity** (`diagnostics/hook_parity.json`): forward hooks on every
  decoder block must reproduce `output_hidden_states` bit-for-bit.
- **Lens self-check** (`diagnostics/logit_lens_self_check.json`): the logit
  lens applied at the final depth must reproduce the model's actual output
  logits (top-1 must match, or be a measured near-tie within numeric noise).
- **Component anatomy probe** (Lab 2+, `diagnostics/component_anatomy.json`):
  contribution hook points are selected by verifying which candidate pair
  reconstructs every block's residual delta — never by module-name heuristics
  (post-norm architectures like Olmo-3 add *normed* submodule outputs).
- **Decomposition check** (Lab 2+, `diagnostics/dla_decomposition_check.json`):
  embeddings + all captured attn/MLP contributions must sum to the final
  pre-norm residual stream.

If a transformers upgrade ever changes hidden-state semantics, these fail
loudly and every downstream number is declared suspect — that is their job.

## Run directories and artifacts

Every invocation creates `runs/<lab>-<timestamp>-<id>/`:

```text
run_config.json            # parsed CLI, after tier defaults
run_metadata.json          # packages, git state, GPU report, env vars
logs/console.log           # everything printed during the run
diagnostics/
  model_anatomy.{json,md}  # where blocks/final-norm/unembedding live — read once
  hook_parity.json         # self-check 1
  logit_lens_self_check.json  # self-check 2
  tokenization_report.csv  # every kept/dropped target with token counts
  gpu_memory_*.json
state/<example_id>/        # per-example human-readable model state:
  state_card.md            #   the narrative dump — start here
  tokens.csv               #   exactly what the model saw, with visible whitespace
  lens_trajectory.csv      #   one row per depth: top-1, p(target), entropy, ...
  logit_lens_topk.csv      #   top-k decoded readout per depth
  residual_stats_final_pos.csv
  residual_norms_by_position.csv
results.csv                # lab-specific long-form measurements
metrics.json               # aggregates
tables/*.csv               # per-example and per-category summaries
plots/*.png
run_summary.md             # the seven standard questions, answered with numbers
ledger_suggestions.md      # drafted claim-ledger entries (you edit, then commit)
artifact_index.json        # map of every artifact with a one-line purpose
```

Reading order for any run: `run_summary.md` → one `state_card.md` → plots →
`results.csv` → `diagnostics/` when anything looks wrong.

## The claim ledger

`claim_ledger.md` at this directory's root is the student's running dossier:
every claim carries an evidence tag (`OBS | ATTR | DECODE | CAUSAL |
SELF-REPORT`), the artifact backing it, and the observation that would kill
it. Labs draft claims with measured numbers into `ledger_suggestions.md`;
nothing touches the real ledger unless you pass `--append-ledger`, because
writing the claim is the coursework. The Lab 11 capstone audits this file.

## Common CLI

```bash
python interp_bench.py \
  --lab lab1 \
  --model allenai/Olmo-3-1025-7B \
  --model-revision <pin>   \
  --device cuda --dtype bfloat16 \
  --tier b \
  --prompt-set small|full|path.json \
  --max-examples 0 \
  --topk 5 \
  --seed 0 \
  --save-tensors \
  --no-plots \
  --append-ledger \
  --run-name my_experiment
```

`--tier a` always maps to a CPU-feasible configuration so the smoke path is
one flag, not a recipe.

## Troubleshooting

**Anatomy resolution fails for a new model.** Add its block/norm paths to
`BLOCKS_PATH_CANDIDATES` / `FINAL_NORM_PATH_CANDIDATES` in `interp_bench.py`
(one place, on purpose). Multimodal and encoder-decoder models are out of
scope.

**Lens self-check fails.** Don't continue. Check `model_anatomy.md` (wrong
final-norm path?), quantization (looser numerics are recorded but top-1 must
still match), and whether the model post-processes logits (softcapping is
handled; anything else needs a look).

**OOM on a small GPU.** `--quantization 8bit` or `--model google/gemma-3-1b-pt`.
Lab 1's capture is tiny; the model weights are the cost.

**Slow on CPU with the 7B model.** That's what `--tier a` is for.

## Adding a new lab

When adding `labN`, update together:

```text
labs/labNN_name.py        # the experiment: prompts, loop, plots, summary
labs/labNN_name.md        # the student handout
interp_bench.py           # LAB_PROFILES registry entry
README.md                 # status section above
```

Each lab module exposes `run(ctx, bundle)` and should push any reusable
measurement machinery down into the bench (capture, trajectories, dump
formats) rather than growing private copies. Before release, verify the
checklist in [how_to_design_labs.md](how_to_design_labs.md) §7 — most
importantly: `--tier a` completes on CPU, the self-checks pass, every claim
in `ledger_suggestions.md` cites an artifact that exists, and the handout's
questions are answerable from the artifacts alone.
