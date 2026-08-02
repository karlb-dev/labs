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

## Weight staging and rotation

The pinned revisions are in `configs/ol_foundation_v1.yaml`. OLMo-3.1 Think,
OLMo-3.1 Instruct, and OLMo-3 Think snapshots were present in the Drive cache
when this track began. Base was incomplete. Copy only the active pinned
snapshot to local NVMe, verify it, and load it from local storage. Keep no more
than two full 32B checkpoints resident. Remove a local copy only after its
result, raw rows, manifest, registry event, reports, and Git push are durable;
the Drive snapshot remains the recoverable source.

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
- O3: audit provenance and comparability of all four lenses before any refit.
- O4: Bank-W development load-by-derivation-by-redundancy intervention grid,
  using the frozen predictions and matched controls.
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
