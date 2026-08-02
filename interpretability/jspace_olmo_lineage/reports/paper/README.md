# OLMo lineage run-specific paper

This directory is intentionally isolated from `interpretability/jspace_paper`.
It records only the OLMo parallel phase and can be merged into the shared paper
in Phase 5 after the main Phase 4 and Gemma workstreams finish.

Contents:

- `olmo_lineage_parallel_phase.tex`: complete source;
- `olmo_lineage_parallel_phase.pdf`: compiled paper;
- `figures/`: exact registered O3 PDF figures copied from the isolated Drive
  run root;
- `compile.sh`: deterministic two-pass build using a fixed metadata epoch.

Compile from the repository root:

```bash
bash interpretability/jspace_olmo_lineage/reports/paper/compile.sh
```

The build requires `pdflatex` and `pdfinfo`. On a fresh Ubuntu/Colab VM, the
toolchain used for this release is provided by `texlive-latex-base`,
`texlive-latex-recommended`, `texlive-fonts-recommended`,
`texlive-latex-extra`, and `poppler-utils`. Installing those system packages
does not alter the experiment dependency lock.

The five figure PDFs must retain the hashes registered by
`ol-geometry-figures-dev-v1`. The final release bundle records the paper and
figure hashes and is authoritative for import.
