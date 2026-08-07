# Calibrated exact-JVP transport gate protocol V2

This methods protocol incorporates the Gemma Study-2 repair. Its purpose is to
separate exact-backend mixed-precision disagreement from tangent-versus-finite-
response error before a target result is interpreted.

## 1. Isolate the calibration from the target

Create a new evidence chain, run root, config, and registry endpoint. Before
any calibration model data, freeze the checkpoints and revisions, prompt and
tokenizer hashes, layers, batches and slots, delivered direction families and
seeds, exact autodiff implementations, dtype, full-suffix estimand, optional
nested diagnostic screen, threshold formula, bootstrap, nuisance routers, and
all terminal paper sentences.

The calibration process must be unable to read the historical target config,
row table, summary, raw vector artifact, discrepancy, or classifier. Freeze
the threshold file atomically before opening the evidence registry. Treat any
target read or target-derived threshold change as a blocking incident.

## 2. Establish exact model identity and path parity

- Pin each model and tokenizer to an immutable revision and fully verify every
  downloaded file.
- Match the actual suffix, batch shape, attention state, position IDs, source
  slot, dtype, and target map.
- Require both exact-JVP primals to match the clean suffix.
- Require every tangent and summary value to be finite.
- Never substitute a finite difference for an exact JVP. Finite differences
  are secants.

Use at least one target architecture and one exact positive-control
architecture when the claim compares estimator behavior across models.

## 3. Measure backend disagreement prospectively

For every frozen full-suffix pair, record tangent cosine, relative error,
maximum absolute difference, difference in local dtype quanta, and the relative
equivalent of a frozen number of dtype quanta. Preserve enough dot products and
norms to reconstruct every summary.

The Study-2 reference grid uses two checkpoints, three layers per checkpoint,
four prompts, batches 1/4/8, three delivered direction families, and three
seeds per family: 216 full pairs. A nested attention/MLP screen is diagnostic
only and may not replace or enlarge the full-suffix ceiling estimand.

## 4. Audit independently

Before freezing a ceiling, require:

1. full primal parity and finite exact backends;
2. deterministic same-process replay and a fresh-process replay;
3. pair reconstruction from preserved sufficient statistics;
4. dtype-quantum reconstruction for every row;
5. canonical-result invariance to a row-order permutation;
6. an explicit audit of plotting floors and display-only clipping.

Accumulation precision is part of the audit contract. If a stored float32
reduction is reconstructed in float64, compare the residual to a prospectively
justified dimension/roundoff bound; do not demand decimal identity between two
different reduction algorithms.

## 5. Freeze the backend envelope

The Study-2 reference rule is:

```text
ceiling = max(
  3 * q99(full-suffix exact-backend tangent relative error),
  q99(relative equivalent of ten target-dtype quanta)
)
```

Resample prompts, not individual tensor elements, for the descriptive
bootstrap. The point rule—not a favorable bootstrap endpoint—is the decision
ceiling.

Before looking at the target, route separately for:

- **path ambiguity:** nonfinite backends or reproducible severe batch-1
  disagreement;
- **batch-composition nuisance:** the frozen batch-8 versus batch-1 tail rule;
- **architecture-dependent floor:** the frozen normalized per-model ratio and
  bootstrap-sign rule;
- **benign scheduling floor:** fallback only when no preceding route is active.

If the architecture router is active, use prospectively defined per-model
ceilings. Otherwise use the pooled all-frozen-batches ceiling. Never select a
route from the target outcome.

## 6. Register before unblinding

Write and independently hash the raw state, row table, pair summaries,
calibration summary, threshold, and any figure. Append the calibration event
only after the threshold bytes exist. Record that the target was not read and
that the threshold preceded the registry read.

## 7. Apply a mechanical target router

In a new process, first verify the registered calibration event and threshold
hash, then open the exact historical target sources. Verify every source hash
and the selected-slot identity assertion. Choose only a prewritten branch:

| Condition | Action |
|---|---|
| Stable applicable ceiling; target error at or below it; selected slot exact | relicense the preserved scoped result without recompute |
| Frozen batch-nuisance replay route active | run only the preregistered matched batch-1 declared-dose replay |
| Path ambiguity, source drift, unstable ceiling, or target above ceiling | remain blocked and register the reason |

The target router may not recompute or select rows unless its frozen branch
explicitly requires that replay. Missing effects remain missing, not zero.

## 8. Preserve historical state

A later calibrated license must not edit or withdraw the earlier failed event.
Publish a V2 state that states exactly which handoff classification is
superseded, why the original stop was correct under its own contract, and how
the new threshold was isolated from the target. Importers retain both records.

## 9. Publish the claim boundary

The exported object is a scoped methods result about the accuracy of a chosen
prompt-specific first-order source-to-target tangent at tested finite scales.
It is not evidence of nondifferentiability, Jacobian absence, missing
information, a named nonlinear mechanism, a workspace absence, or general
intervention invalidity. Preserve development/methods tiers and open no
confirmatory cell, independent-review field, or PI sign-off.

At release, publish a self-verifying registry prefix, exact V2 report and claim
ledger, TeX/PDF bytes, calibration figure, frozen design/configs, and a partial
status for every conditional or unopened item.
