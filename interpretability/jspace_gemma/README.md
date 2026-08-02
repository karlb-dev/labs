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

Restart from `/content/drive/MyDrive/interpret/gemma_transport_resume.md` and
the live companion `gemma_transport_inprogress.md`.

Basic conformance:

```bash
bash interpretability/jspace_gemma/repro.sh
```
