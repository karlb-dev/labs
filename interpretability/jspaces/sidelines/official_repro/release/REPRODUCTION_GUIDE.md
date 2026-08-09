# REPRODUCTION_GUIDE — official-repro Study 1

## CPU artifacts (no model weights)

```bash
pip install -e interpretability/jspaces/sidelines/official_repro
python -m pytest interpretability/jspaces/sidelines/official_repro/tests -q
jspace-or1 verify        # rehash every live registered output
python -m jspace_official_repro.headline      # regenerate §16.1 grid
python -m jspace_official_repro.report_data   # regenerate numbers.tex
python -m jspace_official_repro.report_examples
python -m jspace_official_repro.figures qwen  # figures from JSON
python -m jspace_official_repro.figures olmo
cd interpretability/jspaces/sidelines/official_repro/reports/tex && latexmk -pdf official_repro_report.tex
```

Every aggregate, figure, table, and prose number regenerates from
registered JSON without loading a model (plan §12).

## Model-backed reproduction

Follow `README.md` quickstart (pinned engine clone, HF snapshots
at the §2 revisions), then the stage drivers in
`jspace_official_repro/experiments/` in plan §14 order. All
stages are idempotent by output existence; the OLMo fit resumes
from checkpoints under a runtime-sentinel gate.
