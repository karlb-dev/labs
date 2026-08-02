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
- Integrated campaign branch: `interp_jspace_part2`; the dedicated branch
  remains the immutable source history for the release.
- Scientific import boundary:
  `3b041735d8b842de46a9c0a474fccd0c44e0841a`.
- OLMo repository namespace:
  `interpretability/jspace_olmo_lineage/`.
- OLMo Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801`.
- Native evidence IDs begin with `ol-` and have only `development` or
  `methods` tier.
- Their evidence namespaces remain separate after repository integration;
  scientific synthesis still requires an explicit later Phase 5 router.
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
git switch interp_jspace_part2
git pull --ff-only origin interp_jspace_part2
bash interpretability/jspace_olmo_lineage/repro.sh
```

O2 and later lens-backed work also requires the exact reference engine:

```bash
git clone https://github.com/anthropics/jacobian-lens.git /tmp/jacobian-lens
git -C /tmp/jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
python -m pip install -q -e /tmp/jacobian-lens
```

If the repository is absent, clone the same repository into `/content/labs`,
fetch all branches, and switch to the Part 2 branch above. If Part 2 does not
yet contain merge commit `65a787583d657e77f95ce379e3723c4d66a682ab`, use
the dedicated OLMo branch rather than recreating the package from another
snapshot.

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
5. `reports/OLMO_LINEAGE_STATE_OF_RECORD.md`.
6. `reports/OLMO_LINEAGE_CLAIMS_TABLE.md`.
7. `reports/OLMO_LINEAGE_DEVELOPMENT_REPORT.md`.
8. `reports/PART2_INTEGRATION_RECORD.md`.

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
The five live report mirrors and their recovery index are deliberately
mutable and are never registered as immutable evidence outputs; their source
Git commits and content hashes are recorded in the recovery index.

The completed O1 handoff is under Drive `release/` as
`IMPORT_BUNDLE_PHASE4_EARLY.json` (SHA-256
`debb29ef67ffa8741a4971ec2b0b21340bd5b48dc5729ac12f74f78839bf4f2b`)
and `IMPORT_BUNDLE_PHASE4_EARLY.md` (SHA-256
`a7e6faf9ad412bd965cfbc9f7b1e9e98c9194cc5019e669d0695b5852a55159d`).
It records a failed 16/20 common-support service gate. Do not open an O4
Bank-W intervention under that version-1 protocol.

The final stopped-workstream handoff is also under Drive `release/`:

```text
IMPORT_BUNDLE_PHASE4.json     a2486ec5a4759a1f5b21643e7c60766824c48f13ff43240d458ba72147165a2a
IMPORT_BUNDLE_PHASE4.md       36ed8f773836d2610a37254a81bee7748e928cd700b398f8bac7466a4a2d9468
```

It embeds the 53,719-byte pre-release registry prefix through
`ol-independent-reconstruction-v1` with SHA-256
`db3fe202026e5cad019ca90a3dceb74efce3b248c02710cfd849dcdbf843e80a`.
Verify it, never re-emit or overwrite it:

```bash
python -m jspace_olmo_lineage.experiments.final_release \
  --config interpretability/jspace_olmo_lineage/configs/ol_final_release_v1.yaml \
  --verify
```

The first post-write verifier call returned nonzero only because YAML folding
placed whitespace after a line-ending hyphen in the machine-rendered
sentence-4 wording. The event and all 13 files already existed and matched
their registered hashes. The current verifier canonicalizes that prose-wrap
case; the claims-ledger artifact retains the exact licensed wording. Treat an
existing bundle plus event as immutable and verify it—never retry `--emit`.

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

At the final 2026-08-02T06:06:57Z evidence checkpoint, all four O2 model events,
the paired joint event, and the final methods handoff are complete and
registered. The registry contains 25 origin events, of which 24 are live, with
101 live immutable outputs, all of which verify. Inventory v1 is the one
superseded origin. The
frozen joint verdict is `broadly_conserved_capacity_recruitment_consistent`;
OLMo-3.1 32B Instruct remains a sibling endpoint, not a fourth trajectory
point. The O3 protocol/extractor/aggregate/figure implementation is published,
39 tests pass, and `ol-geometry-protocol-v1` has frozen 13,319 exact readout
rows before any new geometry outcome. All four frozen readout inputs and the
four-checkpoint aggregate are registered and hash-verified. The O3 router
verdict is `dictionary-formation-pattern`; five PNG/PDF figure pairs and their
manifest are also registered. The independent reconstruction and exact model
sentinel pass and are registered. Claims, state of record, isolated paper, and
the final import/restart bundle are complete; there is no OLMo job to resume.

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

The O5 methods audit has completed. Its command is retained for provenance but
must not be rerun because its outputs are immutable:

```bash
python -m jspace_olmo_lineage.experiments.o5_feasibility \
  --config interpretability/jspace_olmo_lineage/configs/ol_o5_feasibility_v1.yaml
```

Its decision is `defer-no-identifiable-crossed-intervention-estimand` /
`not-executed-no-proxy-substitution`. The claims ledger, independent
reconstruction, OLMo-run-specific paper, state of record, and final import
bundle are complete. Do not turn O2/O3 structural tables into an O5
intervention proxy.

The independent reconstruction is methods-only and completed from clean source
commit `12f21ad`. Its provenance command was:

```bash
python -m jspace_olmo_lineage.experiments.independent_reconstruction \
  --config interpretability/jspace_olmo_lineage/configs/ol_independent_reconstruction_v1.yaml \
  --snapshot /content/olmo_lineage_work/sentinel_olmo31_think
```

It reconstructs O1/O2/O3 solely from registered sufficient statistics,
regenerates figures outside the evidence namespace before an atomic publish,
verifies all fourteen weight-shard hashes, and repeats one frozen registered
Bank-W row. All eight candidate scores and the margin reproduced with zero
drift; five PNGs match byte-for-byte and five PDFs were regenerated. Its JSON
SHA-256 is `e159f01d...20542`, Markdown is `48647b44...abb5`, and payload hash
is `ee94e446...619f`. Its event exists and verifies; do not rerun it.
If Drive contains a reconstruction directory without a registry event, treat
it as an unregistered partial and audit it before any retry.

The isolated run-specific paper is under `reports/paper/`. Its deterministic
build command is:

```bash
bash interpretability/jspace_olmo_lineage/reports/paper/compile.sh
```

The 13-page source SHA-256 is `33e88825...c656`; the PDF SHA-256 is
`02a81b87...0ec7`. The build was repeated byte-for-byte, the TeX log has no
warnings, and selected title/table/figure/claim/appendix pages were visually
inspected. It deliberately does not edit `interpretability/jspace_paper`.

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
