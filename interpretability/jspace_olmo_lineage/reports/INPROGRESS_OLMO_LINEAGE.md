# OLMo lineage live in-progress state

Last updated: 2026-08-02T02:38:34Z

This is the volatile restart pointer for the OLMo-only parallel workstream.
The stable recovery procedure is `OLMO_LINEAGE_RESUME.md`; the scientific
narrative is `OLMO_LINEAGE_DEVELOPMENT_REPORT.md`. Canonical heavy outputs
live in Drive under `olmo_lineage_20260801`.

## Current durable state

- Branch: `interp_jspace_olmo_lineage`.
- Remote tracking branch: `origin/interp_jspace_olmo_lineage`.
- Branch parent before this package: `4ea7a9ba7a534daa61e0d8c9960763b921a1b80b`.
- Immutable scientific import boundary:
  `3b041735d8b842de46a9c0a474fccd0c44e0841a`.
- Foundation source commit:
  `dc5f62336ac83481f09dafebaad666a93157f812`.
- Durable Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801`.
- Registry:
  `interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl`.
- Active model job: none; Base O2 is complete and GPU memory is free.
- Last registered native evidence: `ol-capacity-olmo3-base-dev-v1`
  (development), created at 2026-08-02T02:38:05Z from clean commit `f67efcd`.
  Its six immutable outputs and registry event verify; the event/report Git
  checkpoint is the current publish action.
- Both OLMo Bank-W baseline capability outcomes and the joint analysis are
  open. No Bank-W intervention, confirmatory, or replication outcome has been
  opened.

## Work currently in progress

The isolated foundation and O1 service obligation are complete. Four immutable
foundation manifests and nine live evidence events verify cleanly; 31 package
tests pass. No model job is active.

Both OLMo models completed 384/384 unique rows with all 4,608 checked numeric
values finite per model and eight candidate-sequence scores per row. Think
low/high accuracy is 0.7135417/0.7187500 with high-low 0.0052083 and 90% CI
[-0.03125, 0.0416667]. Instruct low/high is 0.7395833/0.7187500 with high-low
-0.0208333 and 90% CI [-0.0572917, 0.0208333]. Both pass independently and
each has 17/24 capable families.

The exact Think/Instruct/Qwen common intersection is 16 families, below the
prospective minimum of 20. `olmo_phase4_service_ready=false`; the source Phase
4 aggregation and stricter side service decision both say BLOCKED. The
required early import bundle is complete at Drive
`release/IMPORT_BUNDLE_PHASE4_EARLY.json` (SHA-256
`debb29ef67ffa8741a4971ec2b0b21340bd5b48dc5729ac12f74f78839bf4f2b`)
with its readable companion (SHA-256
`a7e6faf9ad412bd965cfbc9f7b1e9e98c9194cc5019e669d0695b5852a55159d`).
Its embedded registry prefix covers the first 8,651 bytes through the joint
event and hashes to
`dcaca5a819a070f006a8534b820bfd476e0ebe63cd0b583412bfbfd050a79f10`.
Envelope and live-prefix verification pass. No Bank-W intervention is
authorized on this failed service set. O3 provenance is complete and O2
symmetric capacity is now the active work; any Bank-W redesign must be a
separate, prospectively frozen protocol.

The O3 audit is complete. All six pairwise comparisons are formally
`EXACT_SAME_RECIPE_CORPUS`; all final-lens, 4 x 30 slice, and sampled merge
checks pass. The three post-trained tokenizers produce identical token
sequences on all 120 ordered fit texts. Base uses the same raw texts and order
but exposes no BOS token under the shared model-aware `jlens.from_hf` policy.
The registered decision is `no_refit_run_geometry_analysis`; geometry and O2
capacity analyses are authorized, while intervention outcomes remain closed.
The immutable JSON is 38,818 bytes with SHA-256
`0912d223018accf2b5dfd33a44f7c74da63d9a912f0ef6b465dcfbb1d3581105`;
its Markdown companion is 2,583 bytes with SHA-256
`9f6c84784c15e4dc9288900892ddb6e349ebe1fbf0b46d66054addf779cf5f89`.

The prospective O2 config, centered-target estimator, raw-target sensitivity,
resumable per-layer model runner, paired four-model joint analysis, and tests
are published at `0e5800b`. The outcome-blind O2 protocol is now registered.
Its corpus selects 120
existing prompts in declared order (30 each factual, arithmetic, SQL, prose),
with 7,481 retained content positions. All four pinned tokenizers produce the
same content-token manifest
`03edeb51f61bc64d8bb42dfca1cf69808773d45d28810be8085b327936a1052e`.
BOS is explicit and the mask is content-relative. The corpus is 45,237 bytes,
SHA-256 `695d29f9057f39420e9cbb9f0a6c8dc3862aef42d808abe466736352e58a7948`;
the protocol is 12,124 bytes, SHA-256
`909c07d32af9c7a94239c431344d18fd1088c10dba640f76119bf7152b78c9a0`.
The protocol event itself opened no model outcome.

Base O2 is now complete on all 120 prompts and 7,481 positions. Primary
own-frame centered excess at layers 24/32/40 is respectively -0.0001367,
0.0031512, and 0.0058964, with lower-median occupancy 2 at every layer and 90%
prompt-bootstrap intervals [-0.0007022, 0.0004372], [0.0026555, 0.0036407],
and [0.0053421, 0.0064975]. There is no censoring and no solver-error increase.
The raw-target sensitivities are -0.0015276, -0.0000084, and -0.0000333 with
occupancies 1/1/2. Base own and Base-common frames are exactly identical. The
result is 103,058 bytes, SHA-256
`3708447c125b4932e4163c929ead4f5e744ce717d2d4b7756a383172fc4069c0`.
Do not assign a lineage shift label until the prospectively paired joint
analysis has all four registered model results.

## Exact next actions

From `/content/labs`:

```bash
git switch interp_jspace_olmo_lineage
git status --short --branch
python -m pip install -q -e interpretability/jspace_part2
python -m pip install -q -e interpretability/jspace_phase3
python -m pip install -q -e interpretability/jspace_phase4
python -m pip install -q -e interpretability/jspace_olmo_lineage
python -m pytest interpretability/jspace_olmo_lineage/tests -q
```

Restore the exact lens engine if `/tmp` was reclaimed:

```bash
git clone https://github.com/anthropics/jacobian-lens.git /tmp/jacobian-lens
git -C /tmp/jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
python -m pip install -q -e /tmp/jacobian-lens
```

If the Base event/report checkpoint is still dirty after a reclaim,
publish and mirror it first:

```bash
git add interpretability/jspace_olmo_lineage
git commit -m 'olmo: register Base symmetric capacity'
git pull --rebase origin interp_jspace_olmo_lineage
bash interpretability/jspace_olmo_lineage/repro.sh
git push origin interp_jspace_olmo_lineage
python -m jspace_olmo_lineage.recovery
```

After Base is durable in GitHub and the recovery mirrors, remove only its local
Hub cache snapshot to make room, then stage 3.0 Think directly:

```bash
hf download allenai/Olmo-3-32B-Think \
  --revision ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7 \
  --cache-dir /content/hf_local
```

Run 3.0 Think from a clean published checkpoint:

```bash
python -m jspace_olmo_lineage.experiments.capacity \
  --config interpretability/jspace_olmo_lineage/configs/ol_capacity_v1.yaml \
  --model-slug olmo3-think
```

Each completed layer is a resumable Drive checkpoint. Register, publish, and
mirror 3.0 Think before the next rotation. Do not open an O4 Bank-W
intervention under the failed 20-family protocol.

## Hardware and weight state at last update

- GPU observed: NVIDIA RTX PRO 6000 Blackwell Server Edition, approximately
  96 GiB VRAM, CUDA working.
- OLMo-3.1 Think and Instruct snapshots are complete in the Drive HF cache.
- The exact Think snapshot is also complete on local NVMe. Its result is
  durable in Drive, the side registry, and GitHub; GPU memory is free.
- The exact Instruct snapshot was removed from local NVMe after its result,
  bundle, Git checkpoint, and recovery mirrors became durable. It remains
  complete in Drive and directly recoverable from its pinned Hub revision.
- Direct pinned Hub staging was substantially faster than DriveFS for Instruct
  and is the preferred staging path while authentication/network remain healthy;
  the complete Drive snapshots remain recovery fallbacks.
- OLMo-3 Think is also present in Drive.
- The exact Base revision is complete in `/content/hf_local`; direct transfer
  was selected over DriveFS. Rotate it only after the current Base result is
  pushed and mirrored; it remains recoverable from the pinned Hub revision and
  the complete Drive cache.
- Local disk has approximately 56 GiB free with Base and OLMo-3.1 Think both
  resident.
- The exact `anthropics/jacobian-lens` checkout is installed from `/tmp` at
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`; recreate it after a reclaim.
- Never load model weights through DriveFS. Copy one pinned snapshot at a time
  to local NVMe and load from there.

## Safety boundary

Do not write Phase 3, Phase 4, Gemma, or main-paper registries/run roots. Do
not open untouched Phase 4 confirmatory or replication intervention outcomes.
The OLMo O1 outputs are baseline capability only and exist to service the
parallel Phase 4 Bank-W design.
