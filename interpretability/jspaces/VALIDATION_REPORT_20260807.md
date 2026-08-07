# Reorg validation report — `interpretability/jspaces/` on a fresh Colab VM

Per `plans/jspace_lab_reorg_validation.md`. Validation of the layout
consolidation only; no science was run and no registry was written.

## 1. Header

| | |
|---|---|
| Branch | `interp_jspace_part2_reorg` |
| Entry commit (validated tree) | `d032ad7` |
| Fix commits produced by this validation | `84e61d9`, `2b4ce4b`, `931e438`, `0b7f08b`, `16ae358`, `258f72f` (all `reorg-fix:`), plus this report |
| Reorg boundary tag | `pre-jspaces-reorg-v1` (must be pushed with the branch — the study-2 protocol guard and future audits reference it) |
| VM | Colab, Linux 6.6.122+, Python 3.12.13, 176 GB RAM, 236 GB disk (189 GB free at start) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97.9 GiB VRAM, driver 580.82.07 / CUDA 13.0; torch 2.11.0+cu128 (CUDA available: yes) |
| transformers | 5.13.1; jlens engine cloned at pinned `581d3986`, byte-verified against `special-lab-1/2026-07-25_1726/code/jacobian-lens-pkg/jlens` |
| Drive | mounted (campaign account), `MyDrive/interpret` reachable |
| Wall clock | ≈ 3.5 h total: T0 ≈ 10 min · T1 ≈ 15 min · T2 ≈ 45 min (incl. fix/re-run cycles) · T3 ≈ 60 min (two full Drive hash passes) · T4/T5 ≈ 45 min · T6 ≈ 15 min |

## 2. Scorecard

Status vocabulary: PASS / FAIL / SKIP(reason) / EXPECTED-RED / OUT-OF-SCOPE.

### T0 — bootstrap

| Check | Command | Result |
|---|---|---|
| Editable installs, fresh clone, no PYTHONPATH | `pip install -e` phase2, phase3, phase4, gemma, olmo | PASS (5/5) |
| Imports | `python -c "import jspace_part2, jspace_phase3, jspace_phase4, jspace_gemma, jspace_olmo_lineage"` | PASS |
| CLI entrypoints | `jspace-part2` / `jspace-phase3` / `jspace-phase4` / `jspace-gemma` / `jspace-olmo-lineage` `--help`, plus `python -m` forms | PASS (all respond) |

### T1 — static layout

| Check | Result |
|---|---|
| No symlinks under `jspaces/` | PASS (0) |
| Stale package paths in `*.py,*.sh,*.toml,*.yaml,*.tex` | PASS — literal-string hits only in the four `paths*.py` alias tables, the portability test, and references to `interpretability/jspace_runs` (the gitignored runs-mirror name, deliberately unmoved). **Caveat found and fixed:** component-wise joins (`Path("interpretability")/"jspace_part2"`) evade this grep; five living files carried them (fixed in `16ae358`). |
| Old-prefix URI round-trips | PASS — 4 resolvers × 8 URIs (oldest `interpretability/jspace/`, old `interpretability/jspace_*`, interim `jspaces/analysis_phase/`, interim `jspaces/phases/…`, current) = 32/32 resolve to existing files. Correction recorded: **phase3 has no `repo://` resolver and its registry contains zero `repo://` URIs** — resolver check N/A there by design. |
| Portability tests | PASS (`test_repo_root_portability.py`: 3 passed) |
| Repo-root discovery | PASS from `/tmp` cwd and from a detached `git worktree` (worktree `.git` file handled) |
| gitignore | PASS — `.gitignore:274–275` cover `interpretability/jspace_runs/` and `interpretability/jspaces/runs/`; bare `git check-ignore` on the nonexistent names exits 1 (dir-only rules), with trailing slash both match |
| Env-var error path | PASS — bogus `JSPACE4_RUN_ROOT` honored; `run_root(create=False)` on a missing root raises the clear "set JSPACE4_RUN_ROOT or mount the campaign Drive" error, no traceback into a Colab path |
| Markdown links (5 READMEs) | PASS (19 relative links, 0 broken) |

### T2 — CPU conformance suites (after fixes; each suite run separately)

| Suite | Result |
|---|---|
| phase2 `jspace-part2 selftest` | PASS — exit 0, **156 ok-checks, 0 failures, 0 skips** (jlens engine present). Correction recorded: the plan's `python -m pytest phase2/tests` line does not apply — phase2's tests are self-executing scripts driven by `selftest` (README documents this; the README's "27 CPU self-tests" count is stale — actual count today is 156 checks across 12 scripts). |
| phase3 pytest | PASS — 104 passed |
| phase4 pytest | PASS — 305 passed |
| gemma pytest | PASS — 71 passed |
| olmo pytest | PASS — 90 passed |
| `repro.sh` × 5 from fresh shells | phase2 PASS · phase3 PASS · gemma PASS (incl. `verify`) · olmo PASS (incl. `verify` + Drive final-release `--verify`) · phase4: install+tests+constraints+environment PASS, terminal `jspace-phase4 verify` EXPECTED-RED (see T3) |

Counts differ from the plan's freeze-era expectations (279/59/48) because the
suites have grown since; totals above are the current-tree truth.

**Before the fixes** the fresh-VM state was: phase4 6 failed, gemma 5 failed,
olmo 4 failed — all pin/immutability tests, root-caused to the reorg commit
having rewritten path strings inside frozen artifacts (§5 of this report).

### T3 — registries and artifacts

| Check | Result |
|---|---|
| Registry byte-identity across the move | PASS — all 9 registry files (phase2 events + phase2 registry, phase3, phase4, gemma reports + release prefix, olmo reports + release prefix, paper-analysis events) byte-identical to `pre-jspaces-reorg-v1` |
| `registry-list` (part2 / phase3 / phase4 / gemma) | PASS (314 / 263 / 72 / 23 lines, exit 0) |
| Whole-corpus rehash (own sweep) | PASS — 591 events, 1,511 hash references; 119 git-resident references checked: 115 exact, 2 legitimate append-only prefix pins (verified via the phase4 prefix rule), 2 on a superseded non-live event (`p3-bank-f-tranche1-v1`, correct registry semantics); 1,392 Drive-resident; **zero unexplained mismatches** |
| `jspace-phase4 verify` (all live outputs incl. Drive) | 521/522 verified (447 literal, 72 repository-materialization, 2 append-only-prefix). EXPECTED-RED: 1 — `state.json` of `p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1`, the §6 permanent deficit (archival disposition memo governs). Reporting it green would itself be a failure. |
| `python -m jspace_gemma verify` | PASS — ok true, 23 live events, all outputs (incl. Drive-resident) hashed clean |
| `python -m jspace_olmo_lineage verify` (+ final-release `--verify` vs Drive bundle) | PASS — ok true, 37 live events, 163 outputs hashed clean; native release bundle `a2486ec5…` re-verified |
| Drive durability pass with known-deficits file, compared to frozen `PHASE4_PART5_DURABILITY_FRESH.json` | PASS (same shape as the freeze-era record) — see §3 |
| Study-2 import bundles (2) + mainline `p4-import-*` envelopes (3) | PASS — strict replays byte-equal to the frozen validation snapshots (`test_saved_*_matches_fresh_strict_replay`, `verify_bundle_source` both sidelines), after the §5 fixes |

### T4/T5 — regeneration and papers

| Check | Result |
|---|---|
| paper-analysis synthesis scripts (`extract_claims`, `build_evidence_graph`, `build_cross_model_matrix`, `assemble_reconstruction_audit`) | PASS — all tables, reports and parquet outputs regenerate **byte-identical**; the two analysis PDFs differ only in embedded `/CreationDate` (equivalent-timestamps). Committed copies untouched (regen compared, then restored). |
| Figure regeneration from the runs mirror (`JSPACE_RUNS_ROOT` = local unpack of `interpret/jspace_runs.tar.gz` + Drive metrics for the sidelines) | PASS with notes — `run_analysis.py` (figA–figE), `olmo_lineage_plots.py`/`_2` (olmoL*), `gemma_transport_plots.py` (gmT1) all run clean; content matches the committed figures visually (figA and olmoL1 inspected point-for-point). Byte sizes differ ≈10 % from the committed copies (matplotlib/font rendering environment). Pre-existing overlap noted: `olmo_lineage_plots.py` and `olmo_lineage_plots_2.py` write the same olmoL* filenames with different title wording — not reorg-caused. Committed figures restored. |
| Phase-tree producers without Drive | phase2 `python -m jspace_part2.figures` PASS (p2f01…p2f17 from the local mirror) · phase3 `python -m jspace_phase3.figures3` PASS (p3f01–03…) · phase1 `s19_figures_v2.py` FAIL(pre-existing) — it requires `metrics/descriptive_agg.json`, absent from the real Drive v2 run as well; unrelated to the reorg (phase1 is instead covered by the GPU smoke) |
| Papers | PASS — `kburtram_jspace.tex`, `olmo_lineage.tex` (after `0b7f08b`), `gemma4_nonlinear_jacobian_handout.tex`, plus both sideline TeX sources (`olmo_lineage_parallel_phase.tex`, `gemma_transport_development.tex`): **five for five, zero LaTeX errors, every `\includegraphics` found**; built into scratch, no build products committed |

### T6 — GPU smoke

| Check | Result |
|---|---|
| GPU hard gate in launch context | PASS (CUDA visible, tensor smoke on device) |
| Phase-1 Tier-A 7B battery (`s1_smoke_verify.py`, weights on local NVMe, `SL1_RUN_DIR` redirected to a scratch copy so the frozen Drive run stays read-only) | **PASS — 9/11 hits@20 (recorded: 9), median mid-min rank 3 (recorded: 3); per-probe ranks reproduce the stored battery.** Model loading + jlens hook plumbing proven under the new layout. |
| Phase-3 N8 Level-2 sentinel replay (32B) | OUT-OF-SCOPE — `n8_level2_sentinels.py` unconditionally `register()`s a methods-tier row into the phase3 registry at completion (`registry3.EVENTS` is not redirectable); ground rule 1 forbids invoking a registry-appending producer, and modifying frozen Phase-3 code to add a no-register mode is outside a validation's remit. The plan's fallback ("item 1 alone is acceptable; say so") is exercised: item 1 alone. |
| Lens fits / grids / wedges | not run (science blocks, per plan) |

## 3. Registry integrity statement

- 9 registry files moved **byte-identical** (sha256-checked against `pre-jspaces-reorg-v1`).
- 591 events across them; 1,511 hash-bearing output references.
- Git-resident: 119 references rehashed — 115 exact, 2 append-only-prefix (verified against historical bytes at the registering commits), 2 belonging to a superseded non-live event; **zero unexplained mismatches**.
- Drive-resident: `jspace-phase4 verify` hashed all 522 live-output references (521 verified, 1 permanent known deficit); gemma and olmo verifiers hashed **all** their live outputs clean (23 and 37 live events; 163 outputs on the olmo side) — this exceeds the plan's ≥5-per-side-track spot-check.
- Durability pass `reorg-validation-20260807` against `…/special-lab-1/phase4_20260731` with `KNOWN_DURABILITY_DEFICITS_PHASE4.json`: **522 references, 521 verified, 1 failure = the 1 known deficit, 0 unexpected failures** — the exact shape of the frozen freeze-era pass (521/520/1/1/0; both carry the tool's strict `ok:false` for the permanent deficit). Comparison to the frozen pass: **zero drifts** among common references; the reference sets differ only by the freeze-release event registered after that frozen pass (65 vs 64 live events). Output kept out of git at the validation scratch root.

## 4. Regeneration diffs (summary)

| Artifact class | Verdict |
|---|---|
| Analysis tables (`recon_*` consumers), reports, parquets | identical |
| Analysis figure PNGs (claim survival, cross-model matrix) | identical |
| Analysis figure PDFs (2) | equivalent (embedded CreationDate only; same byte length) |
| Paper figures figA–E, olmoL*, gmT1 | content-equivalent (visual + numeric annotations); byte deltas are rendering-environment; olmoL1–L4 title wording differs between the two producer scripts that share output names (pre-existing) |
| Papers (5 TeX documents) | build clean; products not committed |

## 5. Fixes made (all mechanical, one logical fix per commit)

1. `84e61d9` — **restore 125 frozen hash-pinned artifacts to pre-reorg bytes.** The consolidation commit's path rewrite had modified import bundles (breaking their internal `payload_sha256` self-hashes), validation snapshots, release bundles and configs (whose sha256 pin-lists cover other frozen sources), registered outputs pinned in the registries, hash-pinned governing memos (`jspace_lab_nextsteps_4_3.md` et al.), frozen replay experiment sources under byte/AST contracts, and the phase-1 exploratory mirror. Frozen artifacts legitimately print historical paths; the alias tables are the designed remedy. Every restored file verified byte-identical to its pre-reorg blob. Deliberately **not** restored (living code whose pins are only descriptive freeze-era inventory): the paths modules, `import_bundle.py`, `durability.py`, `manifests.py`, `pre_freeze_inventory.py`, `pilot_snapshot.py`, both `repro.sh`, phase4 tests, paper-analysis plot scripts.
2. `2b4ce4b` — **alias-aware materialization in registered-artifact consumers** (phase4 durability + import validator + canonical-lens decision; gemma/olmo repro and study2-release modules). Consumers resolve recorded historical paths through the package alias tables for reading while echoing recorded strings verbatim, so frozen snapshots replay byte-equal. The study-2 `_generation_protocol_unchanged` guard now handles pre-reorg generation commits by requiring renderer+config untouched from generation to the `pre-jspaces-reorg-v1` boundary at their old paths **plus** byte-exact frozen configs on the current tree (post-reorg renders keep the original strict rule).
3. `931e438` — **tests**: replace `ROOT.parents[N]` repo-root arithmetic (broken by the deeper tree) with `REPO_ROOT` + alias rewrite; revert `BUNDLE_RELATIVE` constants to the frozen recorded strings; accept historical `repo://` prefixes in deficit-URI assertions.
4. `0b7f08b` — `olmo_lineage.tex` graphicspath entry for `p4f03` (figure lives in phase4's reports/figures; build was in draft mode pre-reorg too).
5. `16ae358` — **component-wise old-path joins** the string rewrite could not see (`Path("interpretability")/"jspace_part2"`): reconstruct_phase2/3, olmo_lineage_plots_2 output dir, pilot_snapshot roots, repro_v2 constraints path; repro_v2's historical-worktree replay now picks whichever layout the checked-out commit carries.
6. `258f72f` — **durability append-only-prefix rule fires when the literal registry path is gone** (import rows pin side-registry prefixes under old absolute worktree paths; the materialized-candidate mismatch now falls through to the prefix rule for `evidence_events.jsonl` only; tampered ordinary files keep their original `missing` semantics, as `test_worktree_absolute_path_resolves_to_identical_tracked_file` pins).

## 6. Issues found, not fixed

1. **Registry-appending sentinel replay** — `n8_level2_sentinels.py` has no no-register mode (`registry3.EVENTS` hardcoded). If a post-freeze validation-grade replay is ever wanted, the campaign should add an explicit `--no-register` path to it at a science boundary, not during validation.
2. **olmoL filename collision** — `olmo_lineage_plots.py` and `olmo_lineage_plots_2.py` both write `olmoL1–L4` with different title wording; whichever runs last wins. Pre-existing; worth a rename at the next paper-analysis touch.
3. **`s19_figures_v2.py`** expects `metrics/descriptive_agg.json`, absent from the v2 Drive run; pre-existing.
4. **Plan-file command corrections** (for the next editor of `jspace_lab_reorg_validation.md`): phase2's suite is `jspace-part2 selftest` (its `tests/` are not pytest collectible — pytest collection aborts on their `sys.exit`); phase3 has no `repo://` resolver (nothing to round-trip); the same-named `test_study2_release.py` modules mean the suites must be pytest-run per package, never combined in one invocation.
5. **Frozen-era expectations vs living tree** — the plan's suite counts (279/59/48) and the phase2 README's "27 self-tests" are stale labels for grown suites; harmless.

## 7. Verdict

**VALIDATED WITH NOTES.**

Against §0's definition of working: a fresh clone installs from nothing with
no `PYTHONPATH`; every CPU conformance suite passes (534 pytest tests + 156
selftest checks, zero failures); all 9 registries moved byte-identical and
every git-resident registered byte rehashes exactly; Drive-resident outputs
verify clean with the single documented permanent deficit still (correctly)
red; frozen import bundles replay byte-equal; figures and reports regenerate
from registered data matching the committed record; five TeX documents build
with zero errors; and a deterministic model-backed sentinel (the 7B lens
battery) reproduces its recorded values exactly on GPU under the new layout.

The notes: the reorg commit itself had rewritten 148 hash-pinned frozen
files (restored here — the largest single finding of this validation), and
the residual items in §6 are pre-existing or advisory.

**The branch is safe to merge to `main` from a layout-integrity standpoint**,
provided the `pre-jspaces-reorg-v1` tag is pushed alongside it (the study-2
protocol guard references it).
