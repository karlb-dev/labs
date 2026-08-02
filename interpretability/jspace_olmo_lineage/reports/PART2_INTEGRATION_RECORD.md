# OLMo lineage Part 2 integration record

Integration date: 2026-08-02T06:32:01Z

Status: merged into the current Part 2 campaign tree. OLMo, Gemma, Part 2, and
Phase 3 verification is complete; Phase 4's separate pre-existing Drive
qualification is recorded below. The recovery index records the exact pushed
Part 2 source commit for the six mutable Drive mirrors.

## Exact Git boundary

- Target branch: `interp_jspace_part2`.
- Part 2 pre-merge tip: `c9021e5c2405eaa34a16a07ba0ea903c53b81ee6`.
- Complete OLMo source tip:
  `a28cdd54dda335daf55f468e5be8cc65b2fc5253`.
- Two-parent merge commit:
  `65a787583d657e77f95ce379e3723c4d66a682ab`.
- Common pre-parallel ancestor:
  `4ea7a9ba7a534daa61e0d8c9960763b921a1b80b`.
- The Part 2 parent already contains terminal Gemma tip
  `b0425a441f1b87c33d9bb0b4d08d221942f11923`.
- The concurrent Phase 4 A1000-preparation tip
  `443c4771abb6033812d7b8f3b971107f4c4bdc7f` is not part of this merge.

The merge adds all 69 tracked files under
`interpretability/jspace_olmo_lineage/` without modifying the Phase 3, Phase 4,
Gemma, Part 2 evidence registries, or shared paper directories. It preserves
the complete OLMo commit graph rather than copying or squashing the results.

## Scientific state imported into Part 2

- O1: complete baseline-capability service; the exact OLMo/Qwen common support
  is 16/20, so Bank-W version 1 remains blocked and no O4 intervention exists.
- O2: complete four-checkpoint symmetric capacity analysis; Base-to-3.0 Think
  equal-layer centered-excess difference is +0.0154 percentage points with
  paired 90% interval [-0.0121, +0.0434] percentage points.
- O3: complete same-recipe/corpus provenance and geometry analysis; the frozen
  router is `dictionary-formation-pattern`.
- O4: resolved as gate-blocked, with sentence 4 explicitly pending.
- H5: official SFT/DPO stages are eligible, but the wedge is
  `queued-not-started`.
- O5: `defer-no-identifiable-crossed-intervention-estimand` /
  `not-executed-no-proxy-substitution`.
- Independent reconstruction: pass, including the exact frozen model row.
- OLMo-3.1 Instruct remains a sibling endpoint, never a fourth Think timepoint.

This repository merge does not raise any OLMo result above development or
methods tier, resolve the blocked externalization claim, execute queued Phase 5
work, or combine OLMo/Gemma/Phase 4 into a new scientific analysis.

## Reports, plots, and immutable release

The complete mutable source reports are:

- `OLMO_LINEAGE_STATE_OF_RECORD.md`;
- `OLMO_LINEAGE_CLAIMS_TABLE.md`;
- `OLMO_LINEAGE_DEVELOPMENT_REPORT.md`;
- `OLMO_LINEAGE_RESUME.md`;
- `INPROGRESS_OLMO_LINEAGE.md`;
- this integration record.

The isolated paper is 13 pages. Its source SHA-256 is
`33e88825ce67d158e83328ee378b7847674d2ba234b541839a10b287ea75c656`;
PDF SHA-256 is
`02a81b87fff5fdce07726af341fdc80b0fe010e42c41ddc47c06a2e8d3240ec7`.
The paper was rebuilt from the merged Part 2 tree and remained byte-identical:
13 letter-sized pages with no TeX layout/reference warnings. It contains all
five registered O3 figure PDFs. The registered figure event also retains the
corresponding five PNGs in the OLMo Drive run root.

| Paper figure | PDF SHA-256 |
|---|---|
| `olf01_operator_similarity_heatmap.pdf` | `eb909560821e7eebbacabbe45e379d848165e52df012609fb166b106771c3abd` |
| `olf02_token_row_similarity.pdf` | `5a89c91d065ad72b2a99bb235c2e40269ea32d5c76e532212cca1021141a2d90` |
| `olf03_selected_span_trajectory.pdf` | `9ad83919a5e7c423ff0766d813a4199192a36134941c95b606e0228ee0ce30e3` |
| `olf04_capacity_causal_state_space.pdf` | `0fde6be99cbc11ee09945370862c38aeb65b3e7f4d773eeb12110a592ca78f47` |
| `olf05_readout_transport_decomposition.pdf` | `e60174c7c0af4468ffc918b6e86caa0e331de5b1498d339bb04bd8f5112f5c6d` |

The immutable final handoff is under
`/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801/release`:

- `IMPORT_BUNDLE_PHASE4.json` — SHA-256
  `a2486ec5a4759a1f5b21643e7c60766824c48f13ff43240d458ba72147165a2a`;
- `IMPORT_BUNDLE_PHASE4.md` — SHA-256
  `36ed8f773836d2610a37254a81bee7748e928cd700b398f8bac7466a4a2d9468`.

The bundle registers 13 final artifacts. The OLMo registry has 25 origins, 24
live events, and 101 live outputs; inventory v1 is the only superseded origin.

## Combined verification

| Scope | Result |
|---|---|
| OLMo package | 58 tests pass; 24 live events / 101 outputs hash-verify; final bundle verifies |
| Gemma package | 48 tests pass; 19 live events / 43 effective outputs verify |
| Part 2 package | conformance self-tests and environment audit pass with `JLENS_ROOT=/tmp/jacobian-lens` |
| Phase 3 package | 104 tests pass; environment audit resolves the Drive run root |
| Phase 4 package | 153 tests and exact dependency lock pass; registry rehash checks 220 output references and reports two absent Drive files in one unrelated Phase 4 event |

The required Jacobian Lens checkout is exact commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.

The two Phase 4 absences belong to
`p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1`:
`state.json` and `capacity_reconstructions_a120.pt` under that event's Phase 4
Drive metric directory. They predate and are path-disjoint from this merge.
This OLMo integration neither opens, reconstructs, copies, nor changes those
concurrent Phase 4 outputs. The Phase 4 owner must reconcile them before a
campaign-wide evidence-clean declaration.

## Recovery and VM cleanup boundary

The six mutable reports are mirrored to the isolated Drive `reports/`
directory only from a clean committed tree. The immutable release, evidence
outputs, paper, and figures must verify before local scratch is removed.

At integration time `/content/olmo_lineage_work` occupies approximately 21
GiB: about 17 GiB of readout-source shards, 4.2 GiB of local lens copies, and
small metadata/figure staging. These are local conveniences, not the durable
record. They may be deleted after the integration commit is pushed, the six
Drive mirrors match, and the final bundle verifier passes from the pushed Part
2 tree. The pinned Jacobian Lens checkout may also be discarded with the VM.

## Reverification commands

```bash
cd /content/labs
git switch interp_jspace_part2
git pull --ff-only origin interp_jspace_part2
export JLENS_ROOT=/tmp/jacobian-lens
bash interpretability/jspace_part2/repro.sh
bash interpretability/jspace_phase3/repro.sh
bash interpretability/jspace_phase4/repro.sh
bash interpretability/jspace_gemma/repro.sh
bash interpretability/jspace_olmo_lineage/repro.sh
python -m jspace_olmo_lineage.recovery --verify
```
