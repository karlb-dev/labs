# Gemma 4 31B transport autopsy — development report

Status: foundation, exact-JVP goldens, and paper-band convention registered;
all 56 OLMo control cells are hash-verified, but final calibration output is
blocked on a post-compute JSON scalar repair. All findings in this document
are development or methods evidence. Nothing here is a Phase 4 confirmatory
model cell.

## Scope and non-claims

The study tests the validity and locality of a transport instrument. A failed
fixed J-lens is not evidence that Gemma lacks a workspace, that information is
absent, or that a late readable representation is causally shared. Readout
opacity, within-context finite-radius curvature, and between-context tangent
heterogeneity remain separate hypotheses throughout.

## Concurrent campaign boundary

Mainline Phase 4 Qwen work, the OLMo lineage side track, and this Gemma
transport side track fork independently from `3b04173`. This report reads
prior registered artifacts by hash but does not edit either concurrent
package, run root, or registry. The completed side tracks join the Part-2
branch only at the Phase 5 handoff.

## Historical evidence (read-only; immutable import audit registered)

- The 120-prompt Gemma lens is identified at L30 and later, but not at L22.
- Mid-band accepted-answer ranks remain opaque; the fitted J readout does not
  rescue them.
- Historical finite-response tests report broad Gemma superposition defects.
- Historical fitted-mean-J faithfulness is poor, while OLMo becomes strongly
  faithful late.
- These results motivate G1 but do not substitute for an exact prompt-specific
  autodiff JVP.

All nine required historical result artifacts, including the Gemma deep-band
result, are now imported in the isolated registry. Their source registry,
outputs, and exact historical producer Git blobs pass SHA-256 verification.
No historical artifact was copied, edited, or superseded.

## Architecture audit

The exact pinned config has 60 text-decoder layers of width 5,376, with a
21,504-wide gated MLP. Five sliding-attention blocks followed by one global
block repeat ten times. Local attention uses 32 query/16 KV heads of width
256; global attention uses 32 query/4 KV heads of width 512 and reuses the
normalized keys as values. Every block has four RMSNorm sites and QKNorm;
there is no MoE, KV-shared tail, or per-layer embedding input. Embedding and
unembedding are tied. The monotone `30*tanh(z/30)` cap follows the LM head and
cannot change token ranks or pre-unembedding residual transport.

## Frozen G1 design

Stage 1 uses four fixed task strata, Gemma layers L22/L30/L37/L44/L52,
matched OLMo control layers, single-final-position and uniform-valid-position
modes, four non-lens direction families, and the relative epsilon ladder
0.0025–0.20. Lens-derived directions enter only when their exact hashes and
token targets are bound. Final residual is primary; normalized residual and
selected pre/post-softcap logits are secondary audits.

Numeric tangent, SNR, and curvature-partition thresholds remain behind the
pre-target firewall. They will be calibrated from OLMo and committed before
the first Gemma number.

The original-paper band ambiguity is resolved directly from the primary
Methods. The paper says its 25 sampled residual-stream layers are reindexed to
`[0,100]`, so its layer labels are percentages; its structural analysis places
the workspace at approximately L38--L92 on that axis. The transferable range
is therefore 38--92% depth, approximately Gemma L23--L55. The 37--62% range in
the later Gemma handout is not the governing paper convention. Consequently,
G6's L44/L48/L52 candidates are in the paper-relative band, although late
readability by itself remains neither workspace nor causal-channel evidence.
The primary page, byte hash, HTTP metadata, later conflicting Git object, and
mapping are frozen in `gm_g6_band_convention.yaml` and registered as
`gm-band-convention-v1`. Its manifest SHA-256 is
`04e56c9bffc02d9a9d2580a1982f520c1069e87e0898fe52a80b911b210e5c8f`.

Before model staging, the OLMo design was corrected to include L4 as the
shallow negative-control layer and L60 as the late identity anchor around the
five matched layers L24/L32/L40/L47/L56. This is required for the frozen
later-versus-shallow contrast and was made without observing any current-study
model number.

## OLMo control harness

The resumable producer uses a clean-parity-checked explicit no-cache decoder
suffix. A pre-result audit found that its first version subtracted a batch-one
clean target from batched perturbed outputs and assumed exact algebra among
the separately rounded negative/double/sum inputs. No model had been loaded.
The repaired path runs an all-clean baseline, perturbed forward, and direct
exact JVP with the same batch shape and request slot. It applies JVPs to every
separately realized fp32-to-model-dtype perturbation. Raw homogeneity,
odd-symmetry, and additivity defects remain reported, alongside nonlinear
remainders after subtracting exact first-order delivery mismatch. SNR uses the
larger of in-batch clean-repeat noise and the local target-dtype half-step
norm. Raw response and JVP vectors remain durable per cell. A deliberately
wrong source layer supplies the pre-threshold mismatch baseline. Each
prompt/layer/mode cell checkpoints to Drive and must finish within ten
minutes.

The staging verifier reused only complete content-addressed Drive blobs,
downloaded missing content into an isolated local HF cache, verified every LFS
SHA-256 and every ordinary Git blob ID, and confirmed the safetensor shard set
against the exact remote index before model load. Staging passed from clean
commit `42b34b1`: 26/26 files, 14/14 weight shards, 64,476,964,249 bytes,
zero failures. The Drive manifest file SHA-256 is
`fa620d543b8cc80d3545aee177343036b04a5a0de84b8e65c8dcfa15dec1776c`.
At that staging boundary, no current-study model cell had run.

The subsequent control run completed all 56/56 cells (1,568 metric rows and
28 bit-exact clean-parity checks) from clean pushed commit `06b2a3d`. Every
state-listed metric/raw hash verifies. Final summary serialization then
stopped because pandas produced `numpy.int64` values for curvature-fit
`source_layer` fields, which the standard JSON encoder refuses. No summary,
Parquet, raw inventory, or calibration registry event was created. This is a
post-compute serialization incident rather than a JVP/model failure. The
frozen state SHA-256 is
`f696f28cecc44d3a3d925308dd10226f1f7fa84e09e6e63ff37913ea3960278c`;
the full-run log SHA-256 is
`28a6aecdff750821603e5355bf0776ff38bc069181ad83d6edf4677249225dfe`.
The recovery path preserves all cells and uses a pure finalizer with distinct
compute and finalization commits. The immutable incident audit is registered
from clean commit `a196c4f`; it reread all raw tensors and bound 112 metric/raw
files under inventory SHA-256
`78a7a53cd626a55e98eb4a9e4a95ee0d00ae9c4e97192cdfb0d5a3ae997752de`.
Its manifest SHA-256 is
`78d53fca50b2a8ac2e114f71a7900a3581214e5367b0892dadf624ec736e8e25`.
A pre-finalizer probe also caught the mixed JSON union in `source_position`
(`-1` integer versus `all_valid` string) before any Parquet was written. The
derived Parquet uses a canonical string plus an explicit original runtime-type
column; the per-cell JSON remains untouched. A 1,568-row write/read probe
passes with this lossless storage normalization.

## Live evidence ledger

| Evidence | Tier | State | Result |
|---|---|---|---|
| `gm-foundation-diagnostic-v1` | methods | registered | no scientific result; the first foundation attempt found that the governing TeX is a Git object in a later shared commit, not a file in the exact side fork |
| `gm-foundation-v1` | methods | registered | 22 tests pass; exact architecture/runtime/governance/package inventories and nine historical imports verified |
| `gm-jvp-goldens-v1` | methods | registered | both autodiff backends exactly match the analytic derivative; forward/fallback/reverse derivatives agree on the nonlinear tiny transformer |
| `gm-band-convention-v1` | methods | registered from clean `3a599f7`; no model opened | primary Methods resolve the transferable band to 38--92% (Gemma approximately L23--L55) |
| `gm-olmo-calibration-finalize-diagnostic-v1` | methods | registered from clean `a196c4f` | immutable incident/inventory record for all 56 cells and the post-compute serialization failure |
| `gm-jvp-olmo-calibration-v1` | methods | 56 cells complete; not registered | summary serialization repair/finalization required |
| `gm-jvp-olmo-positive-control-v1` | methods | blocked on preceding boundaries | threshold calibration |
| `gm-jvp-gemma-stage1-v1` | methods | forbidden until thresholds commit | exact target gate |

## G1 decision table

| Observation | Classification | Route |
|---|---|---|
| Both models miss tiny faithful secants | harness/path defect | stop and repair |
| OLMo passes; Gemma tiny secant misses | Gemma path/nondifferentiable op | unfused parity audit |
| Gemma tiny secant matches; error grows with epsilon | finite curvature | G2/G3/G4 |
| Prompt tangent predicts; mean fitted J fails | context/position averaging | G5 |
| Late passes; mid-band fails | relocated transport regime | G6 |
| All faithful cells pass | historical harness suspect | audit and supersede only with new evidence |

## Exact-JVP goldens

Both `torch.func.jvp` and `torch.autograd.functional.jvp` match the analytic
polynomial derivative at zero recorded error. On a deterministic nonlinear
attention/RMSNorm/gated-MLP suffix, forward mode and fallback differ by
`4.81e-17`; forward mode and an independently materialized reverse Jacobian
vector product differ by `8.33e-17`. Central-secant errors over epsilon
`0.01, 0.005, 0.0025, 0.00125` are `1.09e-6, 2.72e-7, 6.80e-8,
1.70e-8`, the expected approximately quadratic decrease. The implementation
manifest states `finite_difference_exact_fallback: false`.

These are methods goldens, not model evidence. No current-study OLMo or Gemma
model result has been produced.

The foundation inventory resolves the exact Gemma revision to two remote
weight shards; zero are staged locally. The historical OLMo Drive cache has
11 of 14 expected weight shards complete by content-address and size. Shards
6, 9, and 14 remain broken/stale partials. The cache is explicitly marked
not load-ready by itself. The isolated local staging pass filled those files
and fully verified all 14 before the positive-control model can open.

## Infrastructure incidents

The first clean foundation attempt at commit `11501b8` stopped before any
foundation output or model result because it addressed the governing TeX as a
worktree file. The side fork predates that file; the source had been read from
later shared commit `4ea7a9b`. `gm-foundation-diagnostic-v1` records the
failure. The repaired producer pins the TeX by exact commit, Git blob, SHA-256,
and byte size, while continuing to pin the physical Drive PDF. This does not
change the scientific design or expose a target outcome.

## Next boundary

Commit/publish the pure finalizer, which verifies the frozen state and every
cell/raw hash, performs the native-integer and explicit Parquet storage
repairs, and records compute/finalization commits separately without rerunning
model cells. Numeric thresholds remain forbidden until the finalized
calibration is registered.
