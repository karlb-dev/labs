# Runtime-identity reproducibility incident: registered methods synthesis

**NAMED METHODS OBJECT — INCIDENT → CONTRACT → DIAGNOSIS → PROSPECTIVE
RESOLUTION → SCIENTIFIC CONSEQUENCE**

State date: 2026-08-04. This note elevates the prompt-112/323 backward-
semantics incident from an operational block record to a first-class Phase 4
methods finding, as directed by `jspace_lab_nextsteps_4_5_addendum.md` §2.1.
It creates no new measurement and changes no registered number.

## 1. The incident

During the Phase 4.4 sealed queue, the frozen prompt-323 influence stage
stopped itself before writing any contribution, layer result, or materiality
verdict. The producer's precommitted runtime control recomputed
`max ||J|| / sqrt(d_model)` for prompt 323 and obtained `181.826310` against
the frozen fit-log value `173.345` with absolute tolerance `0.5`.

Two independent clean current processes agreed with each other to `0.040794`
(`181.826310` versus `181.785516`) while both missed the historical value by
more than `8.4`. Replaying prompt 322 first did not restore the historical
computation (`59.545969` versus historical `52.150`; prompt 323 unchanged at
`181.854247`).

The registered prompt-112 control made this a broader backward-semantics
finding: its earlier influence event contains a successful historical clean
recomputation (`160.070954` against fit-log `159.952`), whereas two clean
current processes returned `55.544060` and `55.587600` — mutually consistent
to `0.043540`, but `104.5` below the registered recompute. All evaluations
were finite, shape-correct (sequence length 128, 111 valid tokens), and
maximal at source layer 0.

## 2. What the detection contract did

The `0.5` absolute runtime-control tolerance was frozen prospectively, before
any influence outcome. It caught the mismatch **before** any contribution
was written: both blocked attempts contain `contribution: null`, zero
completed layers, and one identical 1,146-byte state (SHA-256
`4eef3124...239eb3`), preserved in separately named blocked directories and
never promoted to evidence.

## 3. The diagnostic boundary

Every nominal identity matched the historical manifests: GPU model (NVIDIA
RTX PRO 6000 Blackwell Server Edition, capability 12.0), driver `580.82.07`,
CUDA `12.8`, Torch `2.11.0+cu128`, Transformers `5.13.1`, Triton `3.6.0`,
FLA `0.5.2`, the 48 fused Qwen bindings, exact Qwen snapshot revision, exact
`jlens` commit, and model/corpus/lens/fit-config/producer hashes.

What the historical record did **not** preserve: hashes of the installed
distribution contents (wheels/builds) and the compiled Triton/FLA kernel
caches. The incident record therefore asserts an inference boundary, not a
root cause: a build/content difference, compiled-kernel/cache difference, or
another unrecorded runtime state remains possible.
`PHASE4_PART4_PROMPT323_RUNTIME_IDENTITY.json` now locks the current
non-pyc content inventories (file counts, byte counts, `METADATA`/`RECORD`
hashes, and whole-content inventory hashes) for Torch, Transformers, Triton,
`fla-core`, and `flash-linear-attention`.

## 4. The prospective resolution

Because the frozen canonical producer computes Q-L1–Q-L5 solely from the
registered structural and functional gates — and retains prompt 323 under
every allowed influence label — the influence label could not alter the
Q-L branch. PI direction therefore authorized
`PROMPT323_RUNTIME_CONTRACT_AMENDMENT.md` (committed and pushed at
`92831d31e37e...` before any new influence output): the first amended
computation is the sole estimator; a second same-process computation is a
discarded repeatability diagnostic; the unchanged `0.5` value acts as a
technical repeatability gate; and the exact current distribution-content
inventories must match before model load. Banks, model, endpoints, lenses,
samples, thresholds, SESOIs, wording, and retention were unchanged.

The amended stage then registered `p4-qwen-lens-influence-prompt323-dev-v1`:
primary and discarded-repeat maxima `181.776618` / `181.777423`, worst
per-layer normalized-norm repeat difference `0.004572` against `0.5`, all
tensors finite, and every frozen A500/A1000 materiality metric negligible —
the closest more than 3,800-fold below its threshold. The historical
discrepancy is retained prominently as a non-gating limitation.

## 5. The scientific consequence

> **Fit-era and current-era backward semantics cannot be assumed
> interchangeable under version-level pinning alone.** Two runtimes that
> agree on GPU model, driver, CUDA, framework versions, kernels' nominal
> identities, model bytes, and corpus bytes can still produce materially
> different Jacobian norms, while each runtime remains internally repeatable
> to better than one part in ten thousand. Gradient-based interpretability
> pipelines that fit instruments in one era and audit them in another must
> therefore (i) pin installed distribution contents and compiled-kernel
> caches by hash at fit time, (ii) carry a prospective runtime control with
> a frozen tolerance in every consumer, and (iii) scope any cross-era
> reproduction claim to what the preserved identities actually support.

Phase 4's influence result is accordingly scoped as a **current-runtime
sensitivity shape**; historical-runtime reproducibility is explicitly not
claimed. The two preserved null states remain diagnostics and can never be
promoted.

## 6. Bound evidence

| Object | Reference |
|---|---|
| Block record | `reports/PHASE4_PART4_PROMPT323_RUNTIME_BLOCK.md` / `.json` |
| Runtime identity lock | `reports/PHASE4_PART4_PROMPT323_RUNTIME_IDENTITY.json` |
| Prospective amendment | `preregistration/PROMPT323_RUNTIME_CONTRACT_AMENDMENT.md` @ `92831d31e37e...` |
| Registered influence event | `p4-qwen-lens-influence-prompt323-dev-v1` @ `01236a3f9246...` |
| Diagnostics archive | Drive `diagnostics/prompt323_runtime_identity_20260803/` |
| Preserved null states | blocked directories with SHA-256 `4eef3124...239eb3` |

This synthesis licenses no new number, no historical-runtime identity claim,
and no change to Q-L4, and it does not reopen the influence stage.
