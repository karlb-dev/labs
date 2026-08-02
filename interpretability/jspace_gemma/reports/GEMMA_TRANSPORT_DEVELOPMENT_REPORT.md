# Gemma 4 31B transport autopsy — development report

Status: infrastructure/foundation in progress. All findings in this document
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

## Historical evidence (read-only; import audit pending registration)

- The 120-prompt Gemma lens is identified at L30 and later, but not at L22.
- Mid-band accepted-answer ranks remain opaque; the fitted J readout does not
  rescue them.
- Historical finite-response tests report broad Gemma superposition defects.
- Historical fitted-mean-J faithfulness is poor, while OLMo becomes strongly
  faithful late.
- These results motivate G1 but do not substitute for an exact prompt-specific
  autodiff JVP.

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

## Live evidence ledger

| Evidence | Tier | State | Result |
|---|---|---|---|
| `gm-foundation-diagnostic-v1` | methods | registered | no scientific result; the first foundation attempt found that the governing TeX is a Git object in a later shared commit, not a file in the exact side fork |
| `gm-foundation-v1` | methods | pending | package/import/runtime boundary |
| `gm-jvp-goldens-v1` | methods | pending | analytic and tiny-transformer exact-JVP tests |
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

## Results

No current-study model result has been produced.

## Infrastructure incidents

The first clean foundation attempt at commit `11501b8` stopped before any
foundation output or model result because it addressed the governing TeX as a
worktree file. The side fork predates that file; the source had been read from
later shared commit `4ea7a9b`. `gm-foundation-diagnostic-v1` records the
failure. The repaired producer pins the TeX by exact commit, Git blob, SHA-256,
and byte size, while continuing to pin the physical Drive PDF. This does not
change the scientific design or expose a target outcome.

## Next boundary

Register foundation/import provenance, pass the independent exact-JVP
goldens, then stage and run OLMo. Update this report and the Drive handoff at
every evidence commit.
