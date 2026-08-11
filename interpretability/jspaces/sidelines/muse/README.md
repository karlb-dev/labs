# jspace_muse — Muse Glimmer 30B instrument probe

Development/methods sideline: does the Phase-2 Jacobian-lens assay work on
Meta's **Muse Glimmer 30B**, and is there any workspace-like signal worth a
larger study?

- Branch: `jspace_muse`
- Drive root: `/content/drive/MyDrive/interpret/special-lab-1/jspace-muse/`
- Live state: `/content/drive/MyDrive/interpret/inprogress-muse.md`
- Model: `meta-models/Muse-Glimmer-30B` @ `97c77dff…`
- jlens: `anthropics/jacobian-lens` @ `581d3986`
- Evidence prefix: `muse-`
- Tier: development/methods only (no confirmatory language)

Frozen Phase 1–4 / gemma / olmo / official_repro registries are **read-only**.

## Quickstart

```bash
export HF_HUB_CACHE=/content/hf_local
export JLENS_ROOT=/content/jacobian-lens
pip install -e interpretability/jspaces/sidelines/muse
pip install -e /content/jacobian-lens   # pin 581d3986
# Muse needs recent transformers (muse_glimmer arch):
pip install -U "git+https://github.com/huggingface/transformers.git"

jspace-muse selftest
jspace-muse stage          # download + hash snapshot
jspace-muse admit          # pre-fit geometry gates
jspace-muse fit            # 120 WikiText J-lens (resumable)
jspace-muse battery        # compact promising battery
jspace-muse figures
jspace-muse report         # SoR + TeX/PDF
jspace-muse status
```

## Layout

```
jspace_muse/
  adapters.py       load + jlens wrap
  paths.py          pins, layers, Drive root
  readout.py        parity / g-fold / ranks
  experiments/
    stage.py        HF snapshot
    admission.py    pre/post-fit geometry
    fit.py          120-prompt fit
    battery.py      8-cell battery
  figures.py        PNG from metrics
  report.py         SoR + handout
```

## Resume on a new VM

1. Read `inprogress-muse.md` (this folder's Drive sibling).
2. Clone repo → `jspace_muse`, install package + jlens + transformers.
3. `jspace-muse stage` (no-ops if snapshot present).
4. Re-run the same command that was interrupted (`fit` / `battery`); both
   are idempotent by output existence and checkpoint resume.
