# Gemma transport side track

This package is the isolated, development/methods-only Gemma 4 31B transport
autopsy governed by `jspace_lab_gemma_1.md` and its binding addendum. It was
forked from `3b041735d8b842de46a9c0a474fccd0c44e0841a` and writes only to:

```text
repo:  interpretability/jspace_gemma/
Drive: /content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802/
```

G1 is unconditional: analytic/tiny-model goldens and the identical OLMo
positive-control harness must pass before any Gemma transport number is
interpreted. Finite differences are secants, never exact JVPs. Numeric target
thresholds are calibrated on OLMo and committed before the first Gemma target
cell.

The finite baseline, exact JVP primal, and perturbed source use identical
batch shapes and slots. Central, homogeneity, odd-symmetry, and additivity
diagnostics use the separately realized post-cast vectors, and response SNR
includes a deterministic target-dtype quantization floor.

Restart from `/content/drive/MyDrive/interpret/gemma_transport_resume.md` and
the live companion `gemma_transport_inprogress.md`.

Basic conformance:

```bash
bash interpretability/jspace_gemma/repro.sh
```

After the registered positive-control gate, stage and fully hash the exact
Gemma target snapshot on local NVMe without loading the model:

```bash
python -m jspace_gemma.experiments.gm_stage_gemma
```

Run the predeclared late-layer infrastructure smoke and then resume the full
40-cell grid from the same clean pushed commit:

```bash
python -m jspace_gemma.experiments.gm_run_gemma_stage1 --smoke
python -m jspace_gemma.experiments.gm_run_gemma_stage1
```

## Terminal branch

The registered Stage-1 result is now stopped at the separately frozen
actual-Gemma cross-backend gate. Both backends succeed and agree exactly on
the selected replay, but the full matched batch exceeds the precommitted
relative-error ceiling. Under the binding addendum, G2/G3 and mechanism
interpretation do not run on this branch.

After verifying the blocker boundary, publish the model-free terminal release:

```bash
python -m jspace_gemma.experiments.gm_blocked_release
```

The release exports a methods-only state of record, claim ledger, transport
gate protocol, verified inventory/environment lock, and Phase-4 import bundle.
