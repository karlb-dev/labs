# Gemma 4 31B transport autopsy — state of record

Status: **terminal methods blocker**. This isolated side track is complete at
the G1 hard-stop branch defined by `jspace_lab_gemma_1.md` and its binding
addendum. It is not a Phase 4 confirmatory model result.

## Outcome

The exact-JVP harness, tiny-model goldens, and OLMo positive control pass.
Gemma Stage 1 then reports an operational local-tangent mismatch at all five
tested layers under thresholds frozen from OLMo before any target result.
However, the separately frozen actual-Gemma backend-parity check fails its
full-batch relative-error gate. Consequently, Stage 1 remains diagnostic and
no mechanism, nondifferentiability, curvature, heterogeneity, late-band, or
workspace conclusion is licensed.

The methods-only sentence earned by this track is:

> On Gemma 4 31B, an exact-JVP transport audit stopped at a precommitted
> cross-backend consistency gate: the selected replay agreed exactly, but the
> full matched batch exceeded the relative-error tolerance, so downstream
> mechanism and workspace interpretation were not licensed.

## Identity and isolation

| Field | Value |
|---|---|
| Study | `jspace-gemma-transport` |
| Branch | `interp_jspace_gemma_transport` |
| Fork | `3b041735d8b842de46a9c0a474fccd0c44e0841a` |
| Package | `interpretability/jspaces/sidelines/gemma/` |
| Registry | `interpretability/jspaces/sidelines/gemma/reports/evidence_events.jsonl` |
| Evidence prefix | `gm-` |
| Drive root | `gemma_transport_20260802/` |
| Native tiers | development and methods only |

No Phase 4 or OLMo-lineage package, registry, or run root was written. Prior
campaign evidence was imported read-only by exact hash. The branch is eligible
for an ancestry-preserving merge into `interp_jspace_part2` only after the
release event and import bundle verify.

## Evidence chain

1. `gm-foundation-v1` verifies the isolated package, architecture contracts,
   runtime, model inventories, and nine historical imports.
2. `gm-jvp-goldens-v1` verifies both PyTorch exact-JVP implementations against
   analytic and nonlinear tiny-transformer derivatives. Finite differences
   remain labeled secants.
3. `gm-jvp-olmo-calibration-v1` contains 56 immutable cells and 1,568 rows.
   Its positive shallow-to-late control gradient is finalized without
   recomputing a cell.
4. `gm-jvp-olmo-positive-control-v1` passes all 14 frozen control criteria and
   freezes the Gemma gates before target execution.
5. `gm-jvp-gemma-stage1-v1` contains 40 cells and 1,120 rows from clean commit
   `036e55233babcabacae061ab41d1410a35715aea`.
6. `gm-jvp-gemma-backend-parity-v1` is the terminal failed gate from clean
   commit `af21c2068508a28871f541c82b8dd1ff0f59916b`.

The detailed development report retains the foundation-path and OLMo
finalization incidents. Neither incident changes the terminal classification.

## Frozen G1 findings

The Stage-1 grid uses four fixed prompts, Gemma layers L22/L30/L37/L44/L52,
single-final-position and uniform-valid-position perturbations, four non-lens
directions, and seven relative epsilons from 0.0025 through 0.20. J-selected
directions were correctly deferred until their exact lens and token hashes
could be bound.

All 20 full-forward/suffix checks and all exact-JVP primal comparisons are
bit-exact. The wrong-hook sentinel has relative error 0.3355 against the 0.10
floor. Of 1,120 rows, 538 pass delivery, 508 clear measurement SNR 12, and 477
clear decision SNR 20. Smallest-primary pass counts are 0/12, 0/13, 0/12,
0/13, and 1/14 from L22 through L52. At the declared epsilon 0.10 dose,
12/16 primary rows per layer are evaluable and none pass. The frozen Stage-1
classifier therefore returns `local_tangent_mismatch` at every layer.

Those values describe the output of one forward-mode instrument. They do not
survive the independent full-batch backend gate as a mechanism result.

## Terminal backend-parity gate

The diagnostic freezes one registered mismatch row:
`gm-p001-L52-single_position`, random-Rademacher direction 0, epsilon 0.05.
It reconstructs the original eight-request batch beginning at request index 8
and the selected fifth slot. The exact Gemma snapshot is rehashed before load.

Both `torch.func.jvp` and `torch.autograd.functional.jvp` succeed. Both
primals match the same-batch clean forward exactly. At the selected slot, the
two tangents are bit-identical and the stored source activation, clean target,
finite response, forward tangent, direction and realized-vector hashes, and
five transport metrics all replay at zero error. The selected Stage-1
mismatch is reproduced: tangent cosine -0.00044545 and relative error 2.7718.

Across all eight original slots, backend tangent cosine is 0.99999958 and
relative error is 0.002458, with maximum absolute difference 0.0390625. The
cosine clears 0.999999, but the relative error exceeds the precommitted 1e-5
ceiling. `backend_tangent_all_slots` is the sole failed criterion. No backend
raised, no tensor was non-finite, and no finite difference was substituted.

The gate is not weakened after observing Gemma. Under the plan’s hard-stop
rule, this is a registered methods blocker, not permission to select the one
agreeing slot and continue.

## Licensed claims and non-claims

Licensed:

- The exact-JVP software passes analytic and tiny-transformer goldens.
- The OLMo positive control passes the frozen transport gate in this runtime.
- Gemma Stage 1 operationally reports mismatch under its forward-mode path.
- The actual-Gemma full-batch cross-backend consistency requirement fails.
- This Gemma transport assay is not licensed for mechanism interpretation at
  the tested boundary.

Not licensed:

- Gemma is nondifferentiable.
- The observed mismatch is finite curvature or tangent heterogeneity.
- Information or a workspace is absent.
- Late readout fails, or the L44--L52 band lacks a workspace.
- A fixed J-space intervention is or is not causally valid on Gemma generally.
- The result upgrades a Phase 4 confirmatory conclusion.

## Work intentionally not run

The hard stop cancels Gemma Stage 2, G2 layer/sublayer localization, G3
routing interventions, G4 norm/MLP factorials, G5 context atlas, G6 late-band
assay, and G7/G8 nonlinear recovery. Their absence is required incident
handling, not missing silent computation. No outcome-dependent figure suite
was generated. The development TeX source is updated; this VM has no TeX
engine, so no PDF is claimed as a release artifact.

## Reproduction and recovery

Run:

```bash
bash interpretability/jspaces/sidelines/gemma/repro.sh
cd interpretability/jspaces/sidelines/gemma
python -m jspace_gemma verify
```

Then verify the release JSON envelope, its payload hash, the recorded registry
prefix, and every inventory output hash. Do not rerun or overwrite the OLMo,
Stage-1, or backend-parity evidence. The stable restart file is
`/content/drive/MyDrive/interpret/gemma_transport_resume.md`; the canonical
dynamic handoff is `gemma_transport_inprogress.md` beside it.

## Phase 5 handoff

Phase 5 may import the methods blocker and the transport-gate protocol. It
must not turn this side study into a Phase 4 model cell, claim independent
review or PI sign-off, or infer a biological/mechanistic absence. Any future
repair must use a new evidence ID and prospectively frozen criterion on a new
workstream; it must preserve this failed artifact and explain supersession.
