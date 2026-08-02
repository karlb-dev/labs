# OLMo lineage live in-progress state

Last updated: 2026-08-02T02:29:26Z

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
- Active model job: none; both OLMo baseline jobs finished and GPU memory is
  free.
- Last registered native evidence: `ol-capacity-protocol-v1` (methods),
  created at 2026-08-02T02:29:02Z from clean commit `0e5800b`. Its immutable
  corpus/protocol outputs and registry event verify; the event/report Git
  checkpoint is the current publish action.
- Both OLMo Bank-W baseline capability outcomes and the joint analysis are
  open. No Bank-W intervention, confirmatory, or replication outcome has been
  opened.

## Work currently in progress

The isolated foundation and O1 service obligation are complete. Four immutable
foundation manifests and eight live evidence events verify cleanly; 31 package
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
No O2 model capacity result has been opened.

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

If the protocol event/report checkpoint is still dirty after a reclaim,
publish and mirror it first:

```bash
git add interpretability/jspace_olmo_lineage
git commit -m 'olmo: register symmetric capacity protocol'
git pull --rebase origin interp_jspace_olmo_lineage
bash interpretability/jspace_olmo_lineage/repro.sh
git push origin interp_jspace_olmo_lineage
python -m jspace_olmo_lineage.recovery
```

Then finish or verify the direct pinned Base download:

```bash
hf download allenai/Olmo-3-1125-32B \
  --revision c2b61dae89a1ad10e4ad5653d0e46b590902607b \
  --cache-dir /content/hf_local
```

Run Base from a clean published source/protocol checkpoint:

```bash
python -m jspace_olmo_lineage.experiments.capacity \
  --config interpretability/jspace_olmo_lineage/configs/ol_capacity_v1.yaml \
  --model-slug olmo3-base
```

Each completed layer is a resumable Drive checkpoint. Register, publish, and
mirror Base before rotating to 3.0 Think. O2 includes Base even though Base
capability is not required. Do not open an O4 Bank-W intervention under the
failed 20-family protocol.

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
- The exact Base revision is being downloaded directly from the Hub into
  `/content/hf_local`; direct transfer was selected over DriveFS.
- Local disk had approximately 103 GiB free shortly after Base staging began;
  the exact OLMo-3.1 Think snapshot also remains local.
- The exact `anthropics/jacobian-lens` checkout is installed from `/tmp` at
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`; recreate it after a reclaim.
- Never load model weights through DriveFS. Copy one pinned snapshot at a time
  to local NVMe and load from there.

## Safety boundary

Do not write Phase 3, Phase 4, Gemma, or main-paper registries/run roots. Do
not open untouched Phase 4 confirmatory or replication intervention outcomes.
The OLMo O1 outputs are baseline capability only and exist to service the
parallel Phase 4 Bank-W design.
