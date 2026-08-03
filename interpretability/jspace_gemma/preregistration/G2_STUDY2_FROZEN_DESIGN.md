# Gemma transport study 2 — frozen G2 design

Status: `FROZEN_PRE_G2_1`. Tier: methods/development. Shared source parent: `901fb4fc7578a913088c7947a2e6240f7fc45aeb`. Branch: `interp_jspace_gemma_transport_2`. Drive root: `gemma_transport_2_20260803`.

## Question and boundary

G2.1 estimates exact-backend disagreement, not tangent-versus-response accuracy. Its ceiling is derived from 216 prospectively enumerated Gemma/OLMo backend pairs and cannot read, join, display, or condition on any study-1 Stage-1 outcome. The historical Stage-1 target is isolated in `gm2_stage1_relicense.yaml`, which the G2.1 run/freeze code is forbidden to open. The ceiling file is atomically frozen before registry code may read the existing Gemma registry. Only the later G2.2 process opens the historical target.

The calibration uses three delivered direction families—Rademacher, Gaussian, and sphere-tangent—and excludes activation-radial because study 1 produced no delivery/SNR-evaluable radial rows. Models, layers, prompts, batch sizes, direction seeds, backends, dtype, row fields, ceiling formula, architecture/batch/path routers, bootstrap, and stop rules are exact in `gm2_backend_parity_calibration.yaml`.

## Ceiling

The full-suffix finite/primal-parity rows define:

```text
ceiling = max(
  3 * q99(tangent relative disagreement),
  q99(relative equivalent of ten bfloat16 quanta)
)
```

The dtype-quantum conversion and per-model route are frozen in the YAML. If normalized model floors differ by the frozen factor with a nonzero log-ratio interval, per-model ceilings replace a pooled ceiling. No observed study-1 number may alter this rule.

## G2.2

G2.2 is a mechanical router over the frozen calibration event and exact study-1 hashes. It chooses only: relicensing without recompute, the declared-dose batch-1 replay, or remains blocked. The three paper-facing candidate sentences were committed in `protocol/G2_STAGE1_CANDIDATE_SENTENCES.md` before G2.1. The study remains methods/development tier under every route.

## Isolation and stops

This branch writes only `interpretability/jspace_gemma/`, the allowed Gemma paper paths, and its dedicated Drive root. It never writes Phase 4 or OLMo registries. Any target leakage during ceiling derivation, model/hash mismatch, material primal failure, nonfinite exact backend, unreconstructable raw row, or post-unblinding threshold change stops downstream Gemma science and is registered as an incident rather than repaired in place.
