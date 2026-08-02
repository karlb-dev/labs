# Resume the OLMo lineage workstream

This is the stable restart guide for the OLMo-only parallel phase. Start by
reading the short live pointer `INPROGRESS_OLMO_LINEAGE.md`; use this document
to reconstruct the environment and verify the last durable evidence after a
Colab reclaim.

## Governing documents

The campaign-wide bootstrap is:

```text
/content/drive/MyDrive/interpret/special_lab_resume.md
```

The operative OLMo plan and accepted addendum are:

```text
/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_olmo_lineage_1.md
/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_olmo_lineage_1_addendum.md
```

The Phase 4 parallel-namespace contract and addendum are:

```text
/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_nextsteps_4_3.md
/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_nextsteps_4_3_addendum.md
```

Their exact hashes are frozen in `configs/ol_foundation_v1.yaml` and verified
by `ol-foundation-v1`. Re-read those source documents if their hashes change;
never silently adopt drift.

## Immutable boundaries

- Git branch: `interp_jspace_olmo_lineage`.
- Scientific import boundary:
  `3b041735d8b842de46a9c0a474fccd0c44e0841a`.
- OLMo repository namespace:
  `interpretability/jspace_olmo_lineage/`.
- OLMo Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801`.
- Native evidence IDs begin with `ol-` and have only `development` or
  `methods` tier.
- Phase 4, Gemma, and OLMo remain separate until the later integration phase.
- Instruct is a sibling endpoint, not a fourth point on the Think trajectory.

The track may read only hash-pinned imported evidence. It may not write any
Phase 3, Phase 4, Gemma, or main-paper registry/run root, and may not open
untouched Phase 4 confirmatory or replication intervention outcomes.

## Restart from a fresh VM

Mount Drive, authenticate GitHub and Hugging Face as described in the main
resume, then:

```bash
cd /content/labs
git fetch origin
git switch interp_jspace_olmo_lineage
git pull --rebase origin interp_jspace_olmo_lineage
bash interpretability/jspace_olmo_lineage/repro.sh
```

O2 and later lens-backed work also requires the exact reference engine:

```bash
git clone https://github.com/anthropics/jacobian-lens.git /tmp/jacobian-lens
git -C /tmp/jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
python -m pip install -q -e /tmp/jacobian-lens
```

If the repository is absent, clone the same repository into `/content/labs`,
fetch all branches, and switch to the branch above. Do not recreate the
package from a mainline snapshot if the OLMo branch exists.

The default environment variables are already encoded in the package. To set
them explicitly:

```bash
export JSPACE_OLMO_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801
export JSPACE_OLMO_LOCAL_WORK=/content/olmo_lineage_work
export HF_HUB_CACHE=/content/hf_local
```

The local work and HF cache paths must not be under `/content/drive`.

## Establish the last durable checkpoint

Read, in this order:

1. `reports/INPROGRESS_OLMO_LINEAGE.md` on the branch.
2. The mirrored copy under the Drive run root `reports/`.
3. `reports/evidence_events.jsonl` in Git.
4. `manifests/ol_foundation_manifest.json` and the latest experiment state in
   Drive.
5. `reports/OLMO_LINEAGE_DEVELOPMENT_REPORT.md`.

Verify the registry and Drive files:

```bash
python -m jspace_olmo_lineage verify
python -m jspace_olmo_lineage.recovery --verify
```

The Git copy is authoritative for registry ordering and committed source. The
Drive copy is authoritative for large outputs and resumable state. If a raw
output exists in Drive but no registry event exists, treat it as an
unregistered partial and resume or audit it; do not cite it as evidence. If a
registry output hash fails, stop and diagnose rather than overwriting it.
The three live report mirrors and their recovery index are deliberately
mutable and are never registered as immutable evidence outputs; their source
Git commits and content hashes are recorded in the recovery index.

The completed O1 handoff is under Drive `release/` as
`IMPORT_BUNDLE_PHASE4_EARLY.json` (SHA-256
`debb29ef67ffa8741a4971ec2b0b21340bd5b48dc5729ac12f74f78839bf4f2b`)
and `IMPORT_BUNDLE_PHASE4_EARLY.md` (SHA-256
`a7e6faf9ad412bd965cfbc9f7b1e9e98c9194cc5019e669d0695b5852a55159d`).
It records a failed 16/20 common-support service gate. Do not open an O4
Bank-W intervention under that version-1 protocol.

## Detect and resume model work

Check for a surviving process and state without modifying either:

```bash
nvidia-smi
ps -eo pid,etime,cmd | rg 'jspace_olmo_lineage|transformers|python'
find /content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801 \
  -path '*state*' -o -path '*progress*'
```

Experiment commands are printed verbatim in the live in-progress file and in
the registry event once registered. Resumable producers validate the input
manifest before accepting prior state. Never combine state from different
model revisions, configs, branches, or code commits.

O2 model progress is checkpointed after each completed layer as
`metrics/<slug>/capacity/<evidence-id>/capacity_layer_<layer>.npz` in the OLMo
Drive root. An unregistered layer file is resumable state, not citable
evidence; the runner independently reconstructs its point estimates and
validates its input-manifest hash before reuse.

At the 2026-08-02T04:49:30Z evidence checkpoint, all four O2 model events and
the paired joint event are complete and registered. The live registry therefore
contains 22 origin events, of which 21 are live, with 73 live immutable
outputs, all of which verify. Inventory v1 is the one superseded origin. The
frozen joint verdict is `broadly_conserved_capacity_recruitment_consistent`;
OLMo-3.1 32B Instruct remains a sibling endpoint, not a fourth trajectory
point. The O3 protocol/extractor/aggregate/figure implementation is published,
39 tests pass, and `ol-geometry-protocol-v1` has frozen 13,319 exact readout
rows before any new geometry outcome. All four frozen readout inputs and the
four-checkpoint aggregate are registered and hash-verified. The O3 router
verdict is `dictionary-formation-pattern`; five PNG/PDF figure pairs and their
manifest are also registered. The live in-progress file identifies the pending
bounded O5 feasibility and release boundary.

O3 is staged so a fresh VM never needs four complete 32B snapshots at once.
After `ol-geometry-protocol-v1` exists, obtain only `config.json`,
`model.safetensors.index.json`, `model-00013-of-00014.safetensors`, and
`model-00014-of-00014.safetensors` at the exact revision for one model. Then
run:

```bash
python -m jspace_olmo_lineage.experiments.geometry extract-readout \
  --config interpretability/jspace_olmo_lineage/configs/ol_geometry_v1.yaml \
  --slug <model-slug> --snapshot <local-exact-revision-snapshot>
```

Commit, pull/rebase, reproduce, push, and recover after every evidence event.
The extracts contain only the frozen unembedding rows and final norm. The four
readouts, `ol-geometry-joint-dev-v1` aggregate, and
`ol-geometry-figures-dev-v1` now verify. Figures were rendered only from
registered tables. Exact kth/k+1 candidate-score gaps require a future
compatible replay because O2 did not retain the candidate-correlation log.
Causal core/fringe dose remains blocked by the O1 service gate.

The official intermediate-stage inventory is metadata-only and downloads no
weights. Version 1 has already been run and is immutable:

```bash
python -m jspace_olmo_lineage.experiments.checkpoint_inventory \
  --config interpretability/jspace_olmo_lineage/configs/ol_checkpoint_inventory_v1.yaml
```

It freezes exact official repository commits, small-file hashes, tokenizer and
architecture compatibility, weight-shard LFS manifests, post-training refs,
and the H5 queue decision. It does not itself authorize or run a stage wedge.
Its conservative byte-identity tokenizer predicate returned stated-unresolvable
because Think-SFT uses a different JSON serialization. Do not rerun or
overwrite v1. A read-only audit found an identical 100,278-entry token-ID map
and identical encodings on the complete 200-text frozen fitting corpus plus
seven edge cases. Version 2 is registered and explicitly supersedes v1. It
finds both official Think SFT and DPO cells eligible and queues a bounded H5
stage wedge without opening weights or model outcomes. Keep BOS/chat-template
differences and repository-only parent declarations attached to that queue.
The completed v2 command was:

```bash
python -m jspace_olmo_lineage.experiments.checkpoint_inventory \
  --config interpretability/jspace_olmo_lineage/configs/ol_checkpoint_inventory_v2.yaml
```

The model command is:

```bash
python -m jspace_olmo_lineage.experiments.capacity \
  --config interpretability/jspace_olmo_lineage/configs/ol_capacity_v1.yaml \
  --model-slug <olmo3-base|olmo3-think|olmo31-think|olmo31-instruct>
```

After all four registered model events, use the same module with
`--aggregate-joint`. Never aggregate an unregistered partial checkpoint.

## Weight staging and rotation

The O2 model and lens revisions are in `configs/ol_capacity_v1.yaml` (the
foundation retains the broader model inventory). OLMo-3.1 Think,
OLMo-3.1 Instruct, and OLMo-3 Think snapshots were present in the Drive cache
when this track began. Base was incomplete. Prefer a direct exact-revision Hub
download to local NVMe when authenticated and healthy; the Instruct transfer
showed this can be much faster than DriveFS. Otherwise copy the pinned Drive
snapshot. Verify the completed snapshot and load only from local storage. Keep
no more than two full 32B checkpoints resident. Remove a local copy only after
its result, raw rows, manifest, registry event, reports, and Git push are
durable; the Drive snapshot remains the recoverable source.

## Publish every material checkpoint

Use the dedicated branch. Before every push, pull/rebase the remote branch as
the user requested:

```bash
git status --short --branch
git add interpretability/jspace_olmo_lineage
git commit -m '<OLMo-only checkpoint message>'
git pull --rebase origin interp_jspace_olmo_lineage
bash interpretability/jspace_olmo_lineage/repro.sh
git push origin interp_jspace_olmo_lineage
python -m jspace_olmo_lineage.recovery
```

Do not add foreign namespace changes to an OLMo commit. If the pull introduces
a conflict, preserve both append-only registry histories and audit the merged
events before continuing. Never force-push or rewrite evidence history.

## Scientific execution order

The mandatory spine is O1 -> O2 -> O4:

- O1: exact Bank-W baseline capability gates for OLMo-3.1 Think and Instruct,
  then an immediate early Phase 4 import bundle.
- O2: symmetric Base/3.0 Think/3.1 Think/3.1 Instruct capacity table on the
  frozen 120-prompt corpus with the corrected estimator.
- O3: completed provenance/comparability audit; all six pairs are exact same
  recipe/corpus and the registered decision is to use the existing lenses.
- O4: Bank-W development load-by-derivation-by-redundancy intervention grid,
  using the frozen predictions and matched controls. Version 1 is currently
  gated out by the failed O1 common-support test; do not run it unless a new
  prospective protocol independently authorizes a redesigned service set.
- O5 and receiver/temporal follow-ups are bounded by the operative plan.

Base capability failure can gate Base out of O4 but cannot remove it from O2.
All results remain development/methods and are reported regardless of sign.

## Recovery-document maintenance rule

After every model completion, gate decision, weight rotation, long job start,
or interpretation change:

1. Update `INPROGRESS_OLMO_LINEAGE.md` with UTC time, active process, state
   path, last registered evidence, exact resume command, and next action.
2. Update `OLMO_LINEAGE_DEVELOPMENT_REPORT.md` with results and limitations.
3. Update this file only when the stable recovery procedure changes.
4. Commit, pull/rebase, test, push, and run
   `python -m jspace_olmo_lineage.recovery`.

In an emergency before a reclaim, a dirty report draft can be mirrored with
`python -m jspace_olmo_lineage.recovery --allow-dirty`; mark that mirror as a
draft and reconcile it with Git immediately after restart.
