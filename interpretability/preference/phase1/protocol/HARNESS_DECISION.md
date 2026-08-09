# HARNESS_DECISION.md — how Lab 38 executes (isolated CLI + bench-as-library + thin adapter)

Decision date: 2026-08-07. Authority: PI session instruction ("special labs
may need a special benchmark … assess and decide"), which supersedes plan
§1.3's mandatory-bench framing. PI further licensed plan modifications that
improve implementability/quality, documented here and in
`preregistration/DEVIATIONS.md`.

## Decision

1. **Primary instrument = isolated campaign package** `preference_phase1`
   with its own CLI (`pref1`), owning: schema, canonical scientific hashing,
   bank generation + audits, codebook selection, strict parsing, binding
   resolution, the behavioral runner (immutable per-item JSONL, atomic
   resume cursor, sharding, ≤10-min-loss checkpointing, config-hash resume
   refusal), analysis/stats, evidence registry, figures, TeX reports.
   This follows the jspaces pattern (self-contained installable package,
   `repro.sh`, own registry) — the validated way multi-session campaigns
   survive VM churn in this repo.

2. **`interp_bench` is reused as a library, not a framework.** The package
   imports it (`interpretability/` on `sys.path`) for its validated,
   course-standard machinery: `load_model_and_tokenizer`, `ModelBundle`,
   anatomy resolution, `apply_chat_template`, `set_determinism`,
   `run_with_residual_cache`, `run_with_residual_patch(_batched)`,
   `steering_hooks`, `generate_continuous`, hook-parity self-checks,
   `save_figure` styling. A minimal `RunContext` is constructed by the
   package for library calls. Rationale: these helpers embody the course's
   audited residual-stream convention ("block k writes stream k+1", stream
   k = pre-norm residual after k blocks) and reimplementing them would be
   regression risk for zero gain.

3. **Thin bench adapter retained** (`labs/lab38_revealed_preference_report_channel.py`
   + `LAB_PROFILES["lab38"]` + `CHAT_TEMPLATE_LABS`): the course entry point
   `python interp_bench.py --lab lab38 --tier a --mode bank_audit|smoke`
   works and delegates to the package. The adapter aliases the bench's
   `--mode` default `"lora"` → `smoke` (bench CLI quirk, cf. lab36).
   Campaign-scale stages (dev pilot, frozen battery, mechanism) run through
   `pref1` for resume/shard control; the adapter refuses stages that
   require the campaign workflow, pointing at the `pref1` command.

## Model pinning (addendum E8)

The repo has **no** central revision-pinning machinery (`interpkit/pins.py`
was never built; `--model-revision` defaults to None). Therefore the
campaign pins revisions itself in `preference_phase1/models.py::PINS`,
resolved once against the HF Hub at implementation time and frozen; every
manifest records model id, revision SHA, tokenizer revision, and
chat-template hash. Tier map (course-standard, = lab36):

```text
model_tier_a: HuggingFaceTB/SmolLM2-135M-Instruct   (plumbing smoke only)
model_tier_b: allenai/Olmo-3-7B-Instruct             (primary)
model_tier_c: allenai/Olmo-3.1-32B-Instruct          (replication, drop-order gated)
```

## Consequences

- Course artifacts (run dirs, method_card, ledger_suggestions, evidence
  tags OBS/DECODE/CAUSAL/AUDIT, minimum-artifact contract) are still
  emitted — by the package, in the standard shapes.
- The bench's global `runs/` gitignore stands: live run dirs are Drive-
  mirrored; registered evidence is copied under `phase1/reports/` and
  hash-pinned in the registry.
- If `interp_bench` internals change upstream, the package's conformance
  tests (`tests/test_bench_conformance.py`) catch the drift — the same
  guard jspaces used for its pinned engine.
