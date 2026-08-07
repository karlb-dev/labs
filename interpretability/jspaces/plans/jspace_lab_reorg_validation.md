# jspace_lab_reorg_validation.md

## Full validation of the reorganized `interpretability/jspaces/` layout on a fresh Colab VM

> **Paste-line for the validation agent**
> Clone `karlb-dev/labs`, check out branch `interp_jspace_part2_reorg`, and
> read this file in full, then `interpretability/jspaces/README.md` and
> `ARCHIVE.md`. Your job is to prove the post-reorg tree **works end to end
> on a machine that has never seen the old layout**: installs, tests,
> registry verification, figure/report regeneration, paper builds, and a
> bounded GPU smoke if a GPU is available. You are validating an
> organization change, not doing science: you may fix mechanical
> path/import/build breakage with clearly-labeled commits, but you must not
> touch any registry, any frozen artifact byte, any preregistration, or any
> scientific number. Produce the verification report described in §8 and
> push it with any fixes to the same branch.

---

# 0. Context: what changed and what "working" means

The campaign trees moved from sibling `interpretability/jspace*` packages
into one tree, `interpretability/jspaces/` (phases/phase1..phase4,
phases/paper_analysis, sidelines/{gemma,olmo}, plans/), in two commits on
this branch. Python import names did **not** change (`jspace_part2`,
`jspace_phase3`, `jspace_phase4`, `jspace_gemma`, `jspace_olmo_lineage`).
Path portability was reworked: repo-root discovery replaced `parents[N]`,
and every `paths*.py` carries an alias table so `repo://` URIs written under
**any** historical prefix still resolve. All 8 `evidence_events.jsonl`
registries moved byte-identical. The tag `pre-jspaces-reorg-v1` marks the
last old-layout commit.

"Working" means: a fresh clone installs, every CPU conformance suite
passes, every package's registry verifier passes (modulo the two known
permanent deficits in §6), figures and reports regenerate from registered
data and match what is committed, the papers compile, and (GPU tier) one
deterministic model-backed sentinel reproduces. Nothing requires the old
paths, `PYTHONPATH` hacks, or a specific working directory.

Already validated on a laptop (do not re-litigate, do re-run): alias
round-trips across all prefix generations, import smoke of all five paths
modules, the phase4 portability tests, zero stale-path greps.

---

# 1. Ground rules

1. **No registry writes.** No new evidence events in any package. Every
   producer you invoke must run in verify/no-register mode or write to the
   scratch root in §2. If a command would append to a registry, do not run
   it; record it as OUT-OF-SCOPE in the report.
2. **No frozen-byte edits.** `evidence_events.jsonl`, `release/` bundles,
   registered outputs, preregistrations, freeze records, and anything
   hash-pinned are read-only. `git diff` on these files at any point is a
   hard stop: revert and record the incident.
3. **Fix policy.** Mechanical breakage (a missed path string, a broken
   import, a wrong relative link, a build flag) may be fixed directly —
   one logical fix per commit, message prefixed `reorg-fix:`. Anything
   deeper (a test that fails for a non-path reason, a number that
   regenerates differently, a design problem) is **reported, not fixed**.
4. **Committed artifacts are not overwritten by regeneration.** Regenerate
   into the scratch root and `diff`/compare against the committed copy.
   Only if a committed figure/report is *provably stale because of the
   reorg itself* (e.g., a figure containing a baked-in old path label) may
   you regenerate it in place, as its own `reorg-fix:` commit.
5. **Expected-red is not failure.** §6 lists known deficits that must stay
   red. Reporting them green would itself be a validation failure.
6. Model weights: download only what a tier needs; verify shard hashes
   against pinned manifests before load; never fall back to CPU for a
   model-scale step.

---

# 2. Bootstrap (T0)

```bash
cd /content
git clone https://github.com/karlb-dev/labs.git
cd labs
git switch interp_jspace_part2_reorg
git log -1 --oneline          # record the validated SHA in the report

export JSPACES_VALIDATION_ROOT=/content/jspaces_validation
mkdir -p "$JSPACES_VALIDATION_ROOT"/{regen,logs,scratch}
```

Record in the report: Python, CUDA driver/runtime, torch, transformers,
GPU name/VRAM (or "none"), disk free, and whether Drive is mounted.

Install order (this is itself a validation target — the chain must work
from a fresh clone with **no** `PYTHONPATH`):

```bash
pip install -e interpretability/jspaces/phases/phase2
pip install -e interpretability/jspaces/phases/phase3
pip install -e interpretability/jspaces/phases/phase4
pip install -e interpretability/jspaces/sidelines/gemma
pip install -e interpretability/jspaces/sidelines/olmo
python -c "import jspace_part2, jspace_phase3, jspace_phase4, jspace_gemma, jspace_olmo_lineage; print('imports ok')"
```

Then verify each CLI entrypoint exists: `jspace-part2 --help` (or
equivalent `python -m` form), same for phase3/phase4 and the sidelines.

---

# 3. Static layout validation (T1 — CPU, minutes)

| Check | Command / rule | Pass criteria |
|---|---|---|
| No symlinks | `find interpretability/jspaces -type l` | empty |
| No stale package paths in code | grep `interpretability/jspace_` and `interpretability/jspace/` in `*.py,*.sh,*.toml,*.yaml,*.tex` under `jspaces/` | hits only inside `paths*.py` alias tables and the portability test |
| Old-prefix URIs resolve | for each of the five packages' resolvers, round-trip one `repo://interpretability/jspace_*` URI, one interim `repo://jspaces/analysis_phase/...`, one current | all resolve to existing files |
| Portability tests | `python -m pytest interpretability/jspaces/phases/phase4/tests/test_repo_root_portability.py -q` | pass |
| Repo-root discovery | run a resolver from `/tmp` as cwd and from a `git worktree` checkout | both find the repo root |
| gitignore | `git check-ignore interpretability/jspace_runs interpretability/jspaces/runs` | both ignored |
| Env-var error path | unset run-root env vars on a host without `/content/drive`; call a run-root helper | clear "set JSPACE4_RUN_ROOT"-style error, not a traceback into a missing Colab path (on Colab, simulate with a bogus `JSPACE4_RUN_ROOT` pointing at a writable scratch dir) |
| Markdown links | check the relative links in `jspaces/README.md`, `plans/README.md`, `phases/README.md`, `sidelines/README.md`, `phases/phase1/README.md` | all targets exist |

---

# 4. CPU conformance suites (T2 — CPU, ~30–60 min)

Run each suite from the repo root **and** capture counts. Expected totals
from the freeze-era record: 279 phase4 + 59 olmo + 48 gemma = 386, plus the
phase2 selftest (27 checks) and the phase3 suite.

```bash
python -m pytest interpretability/jspaces/phases/phase2/tests -q
jspace-part2 selftest
python -m pytest interpretability/jspaces/phases/phase3/tests -q
python -m pytest interpretability/jspaces/phases/phase4/tests -q
python -m pytest interpretability/jspaces/sidelines/gemma/tests -q
python -m pytest interpretability/jspaces/sidelines/olmo/tests -q
```

Also run each package's `repro.sh` once (they chain install + tests + env
audit); confirm they succeed from a fresh shell with only the repo checkout.

Any failure here gets triaged: if the root cause is a path/layout issue →
`reorg-fix:` commit and re-run; anything else → report only.

---

# 5. Registry and artifact verification (T3 — CPU + Drive)

For each package, run its registry/output verifier:

```bash
jspace-part2 registry-list
jspace-phase3 registry-list
jspace-phase4 registry-list
jspace-phase4 verify
python -m jspace_gemma verify
# olmo: use the package's documented verifier (check its README / __main__);
# at minimum, registry-list + rehash of git-resident outputs
```

(These invocations come from the package READMEs; if one differs on the
current tree, use the package's actual documented command and record the
correction in the report.)

Two sub-cases:

- **Git-resident outputs** (registry rows whose outputs live in the repo):
  hash verification must pass completely. This is the core proof that the
  reorg did not disturb a single registered byte.
- **Drive-resident outputs**: if Drive is mounted with the campaign
  account, run the phase4 durability pass against
  `.../special-lab-1/phase4_20260731` with the known-deficits file, and
  spot-verify at least five registered Drive outputs per side track. If
  Drive is not mounted, mark these SKIP (not PASS).

Additionally, re-verify the two study-2 import bundles and the three
mainline `p4-import-*` envelopes (source registry hash → admitted output
hashes) — these exercise the cross-package path resolution harder than
anything else.

---

# 6. Known-expected red (do not "fix", do not hide)

| Item | Expected state |
|---|---|
| `capacity_reconstructions_a120.pt` (historical A120–A250 functional event) | missing on the primary path unless the exact-hash recovery has been run; durability row stays red |
| That event's `state.json` | permanently missing; archival disposition memo governs; row stays red |
| Colab Drive defaults in module constants | fine to exist; only an *error path* when env unset is required |
| `interpretability/jspace_runs/` | absent on a fresh VM; scripts depending on it must SKIP with a clear message once `JSPACE_RUNS_ROOT` is unset, not crash |
| Frozen PDFs/handouts printing old paths | expected; the README migration table is the remedy; never rebuild frozen handouts |

---

# 7. Regeneration and papers (T4/T5 — CPU, ~1–2 h)

## 7.1 Figures and reports from registered data (into `$JSPACES_VALIDATION_ROOT/regen`)

1. **paper_analysis synthesis scripts** (`phases/paper_analysis/analysis/scripts/`):
   rerun the reconstruction/synthesis scripts that read only committed
   parquets + registries (the reconstruction audit assembly, claim/evidence
   graph builders, cross-model matrix). Diff regenerated tables/reports
   against the committed versions: numeric content must match; the report
   records any diff beyond whitespace/timestamps.
2. **Plot scripts** (`phases/paper_analysis/scripts/`): these need the runs
   mirror. If you can unpack the Drive run archives to a local dir, set
   `JSPACE_RUNS_ROOT` and regenerate `figA–figE`, `olmoL*`, `gmT1`;
   compare visually + by size against committed `figures/`. Otherwise SKIP
   with the reason.
3. **Phase-tree figure producers** that regenerate from small committed
   metrics (e.g., phase1's `s9/s19`-style regeneration, phase3
   `figures3.py` paths): run at least one per package where a documented
   regeneration path exists without Drive.

## 7.2 Papers (must pass)

```bash
cd interpretability/jspaces/phases/paper_analysis
latexmk -pdf -interaction=nonstopmode kburtram_jspace.tex
latexmk -pdf -interaction=nonstopmode olmo_lineage.tex
latexmk -pdf -interaction=nonstopmode gemma4_nonlinear_jacobian_handout.tex
```

Zero LaTeX errors; every `\includegraphics` target found. Also build the
two sideline release TeX papers if their sources are in-repo. Build
products are not committed.

---

# 8. GPU smoke tier (T6 — optional but recommended, gated on hardware)

Skip cleanly if no GPU. Otherwise, smallest-first:

1. **Phase-1 Tier-A smoke (7B, ~minutes):** `phases/phase1/code/scripts/`
   `s1_smoke_verify.py` — the deterministic 7B lens battery (9/11 probes at
   rank ≤ 20). This proves model-loading + hook plumbing under the new
   layout at small scale.
2. **One Phase-3 N8 Level-2 sentinel replay (32B/27B, ≤1 h, ≥80 GB GPU
   only):** re-measure the 20-item sentinel subset for ONE model under the
   frozen command in `phases/phase2/protocol/N8_REPRO_PROTOCOL.md` /
   phase3's `n8_level2_sentinels.py`, writing to the scratch root with
   registration disabled. Pass = lp columns bit-exact against the
   registered sentinel values. This is the single strongest "everything
   still works" statement available without a fit.
3. Do **not** run lens fits, grids, wedges, or anything from the Phase 4.4
   / sidelines-2 queues. Those are science blocks, not validation.

If GPU memory or time forces a choice, item 1 alone is acceptable; say so.

---

# 9. The verification report

Write `interpretability/jspaces/VALIDATION_REPORT_<UTC date>.md`:

1. **Header:** validated commit SHA, VM/environment table, Drive
   mounted?, GPU?, wall-clock per tier.
2. **Scorecard:** one row per check in §3–§8 with
   `PASS / FAIL / SKIP(reason) / EXPECTED-RED / OUT-OF-SCOPE`, the exact
   command, and the count (tests passed, hashes verified, URIs resolved).
3. **Registry integrity statement:** explicit counts — N events across the
   8 registries, M git-resident outputs rehashed, K Drive outputs
   spot-checked, zero mismatches (or the list).
4. **Regeneration diffs:** every regenerated artifact vs committed —
   identical / equivalent (timestamps only) / DIFFERENT (with detail).
5. **Fixes made:** each `reorg-fix:` commit, one line each, with why it
   was mechanical.
6. **Issues found, not fixed:** anything non-mechanical, with enough
   detail to act on later.
7. **Verdict:** one of `LAYOUT VALIDATED`, `VALIDATED WITH NOTES`, or
   `BLOCKED (list)` — judged against §0's definition of working. State
   explicitly whether the branch is safe to merge to `main` from a
   layout-integrity standpoint.

Commit the report (and any fix commits) to `interp_jspace_part2_reorg` and
push. Do not merge to `main` yourself.

---

# 10. Stop rules

- Any `git diff` appearing on a registry, release bundle, or registered
  output file → hard stop, revert, record.
- A verifier reporting a hash mismatch on a git-resident registered output
  → hard stop; report with the exact file and both hashes; do not
  "re-register" or touch the file.
- More than ~3 non-mechanical test failures in one package → stop fixing,
  finish the remaining read-only tiers, report the cluster.
- GPU OOM or missing shards → skip the GPU tier; never substitute a
  smaller model or CPU execution for a model-scale check.
- Running low on time → priority order is §4 (suites) > §5 (registries) >
  §7.2 (papers) > §7.1 (regeneration) > §8 (GPU). The report ships
  regardless, with honest SKIPs.
