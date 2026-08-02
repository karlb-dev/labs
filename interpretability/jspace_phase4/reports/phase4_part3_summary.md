# Phase 4 part 3 summary and Phase 4.4 handoff

**LIVE DRAFT — A1000 FIT IN PROGRESS — NOT A FREEZE RECORD**

Last updated: 2026-08-02 20:22 UTC. This document will be finalized after the
current A1000 producer exits and all completed outputs are independently
rehash-checked. Phase 4 remains development-only. No confirmatory or
replication intervention outcome may be opened, and this implementation agent
does not sign independent-review or PI fields.

## Executive state

Phase 4.3 has reduced the project to one active compute dependency and a
small, explicit set of governance/durability dependencies. The exact nested
Qwen draw-A fit is continuing from A500 to A1000 under the unchanged producer.
The last exact boundary at this draft is A950, checkpoint SHA-256
`1c32a45106648830ec5b89a853fcb14182d92d552f05725b8d7507281ec9dc01`.
The fit contract remains
`bf4caff4ff7c389d29f235a91062ae86e3a37dfc526c42bbd9af7c5d7e1f3b00`.
A950 is recovery state, not registered scientific evidence.

The expected VM reclaim is approximately 2026-08-02 23:38 UTC. At the
observed end-to-end rate, the A1000 checkpoint should land near 22:50 UTC,
with final serialization, hashing, registration, backup, integration, and
push near 22:55--23:05 UTC. There is not a reliable margin for the full
postfit queue. At 23:25 UTC no new long-running stage should be launched.

The isolated preparation branch already ancestry-contains the completed
Gemma and OLMo side-track merges from remote main. After the fitter exits, all
mainline, side-track, preparation, and A1000 work must be merged without
squashing into the single `interp_jspace_part2` branch, validated, pulled once
more, and pushed.

The new fit-diagnostic audit found and corrected an important wording error:
prompt 323 is not the overall archived maximum. Through n=950, the largest
retained finite rows are prompt 616 at 231.101, prompt 233 at 189.182, prompt
323 at 173.345, prompt 612 at 151.626, and prompt 660 at 113.855. The current
VM transcript recovers prompts 555--716 that were absent from the repaired
Drive log. Only the immutable first 7,073 transcript lines are admissible:
that 18,232,092-byte prefix hashes to
`e142b280e01622e2d4e4214083804de26f204c4fdfe6db05f656d0218536a943`
and yields exactly 162 unique diagnostic prompts. Later transcript lines can
display test fixtures and are intentionally excluded. Raw rows 1--180 were
emitted on the prior VM but are not materialized here, so the final report
must distinguish the recoverable
820-row raw archive from the final checkpoint's `n_done == next_idx == 1000`
acceptance invariant. No row is trimmed. The prospectively frozen prompt-323
audit remains in the decision queue; a multi-row retained-extremes influence
audit belongs in reviewed Phase 4.4 follow-up and cannot retroactively change
the A1000 decision rule.

## Binding scientific and governance boundary

- Phase 3 and every registered Phase 4 event are immutable.
- A1000 is the last automatic Qwen fit-size escalation.
- Partial fit counts are recovery state only.
- Native `ol-*` and `gm-*` events never enter the Phase 4 registry; side-track
  releases enter through one strict `p4-import-*` event apiece.
- No Bank-W intervention is authorized: exact three-model common support is
  16/20.
- P4-P1 is estimation-only in candidate 0.11. Its mandated consumed-data
  orthogonal feasibility shot remains independently review-gated and cannot
  restore P4-P1 as a Phase 4 primary.
- P4-P2 remains conditional on Q-L1/Q-L2, independent producer review, the
  consumed-family pilot, exact power, untouched-bank review, and PI approval.
- No confirmatory or replication intervention result may be opened before a
  real independent review, PI sign-off, freeze commit, and freeze tag.
- The implementation agent must not self-sign any reviewer or PI field.

## Completed Phase 4.3 preparation

Before any A1000 result existed, the project committed and tested:

- the A500–A1000 structural comparison contract;
- the A500/A1000/published functional comparison contract;
- the selection-margin audit;
- the retained prompt-323 paired influence audit;
- the mechanical Q-L1 through Q-L5 canonical decision;
- the conditional Q-L2 span-level estimand amendment;
- the P4-P2 GPU pilot and fixed exact/Monte-Carlo sign-flip power ruler;
- the one-shot Bank-B answer-direction-orthogonal feasibility producer and
  external-review packet;
- strict Gemma and OLMo import envelopes and a restart-safe admission queue;
- exact-hash-gated recovery for the missing historical A120 capacity tensor;
- a known-deficit-aware whole-registry inventory and two-pass Drive durability
  plan;
- a reclaim-safe `JSPACE4_STOP_AFTER` boundary for the A1000 postfit queue.

The test baseline before A1000 completion is 279 Phase 4 tests, 59
OLMo-lineage tests, and 48 Gemma tests: 386 total. The sole Gemma Torch FX
warning is unchanged. The preparation branch is clean and pushed.

### Phase 4.3 completion audit

This table separates implemented preparation from registered scientific
evidence and from approvals that this implementation agent cannot supply.

| boundary | state at this draft | closure evidence |
|---|---|---|
| governing Part 3/addendum instructions | complete | read into the operational handoffs and prospective contracts |
| Gemma and OLMo branch ancestry | complete | both terminal branches merged into the remote-main ancestry; native side namespaces excluded from the Phase 4 registry |
| prospective A1000 successor contracts | complete | structural, functional, margin, influence, canonical, import, recovery, durability, and reclaim-stop code committed before the A1000 result |
| A1000 fit | **RUNNING** | final append-only event, exact backup, normalized diagnostic archive, and tensor audit still required |
| frozen post-A1000 decision queue | **NOT YET RUN** | structural through canonical rows below remain pending |
| side-track Phase 4 admission | intentionally queued | may run only after the registered canonical boundary |
| whole-registry durability | blocked | two known historical A120--A250 outputs remain absent; a fresh materialization pass is also required |
| independent review / PI / freeze | not supplied | external reviewers must complete these gates; no untouched outcome is authorized |

## Side-track state already merged into the ancestry

### Gemma transport

The terminal Gemma bundle is a methods blocker. The exact-backend parity gate
failed at all frozen slots, with maximum relative error 0.002458 above the
precommitted `1e-5` tolerance. G2/G3 stopped. The strict Phase 4 envelope
validates five source events and 21 outputs. It licenses no mechanism,
nondifferentiability, late-band, workspace, confirmatory, or replication
claim.

### OLMo lineage and Bank W

Both OLMo models pass the independent Bank-W capability gate, as does Qwen,
but the exact common support across all eligible models is 16/20. P4-P3 is
therefore blocked and no Bank-W intervention may run. The early capability
envelope validates five source events and 11 outputs. The terminal OLMo
methods envelope validates one event and 13 outputs; the source verifier
covers 24 live events and 101 outputs. O5 remains
`not-executed-no-proxy-substitution` because no crossed causal estimand is
identifiable.

Mainline admission remains intentionally queued after the registered Qwen
canonical boundary. Admission does not change the already-frozen 16/20
decision.

## A1000 completion and integrity record

Fill this section only from registered and freshly rehashed bytes.

| field | final value |
|---|---|
| evidence ID | `p4-qwen-lens-fit-drawA-n1000-dev-v1` |
| prompts | **PENDING: must equal 1000** |
| final lens SHA-256 | **PENDING** |
| final lens bytes | **PENDING** |
| final checkpoint SHA-256 | **PENDING** |
| checkpoint-state SHA-256 | **PENDING** |
| result SHA-256 | **PENDING** |
| input-manifest SHA-256 | **PENDING** |
| registration commit | **PENDING** |
| local backup manifest | **PENDING** |
| cloud durability | **PENDING; cache presence is not proof** |

Final integrity review must record:

1. the normalized 820-row recoverable raw archive (prompts 181--1000), with
   exact source coverage and no unclassified omission inside that archive;
   prompts 1--180 are an explicit prior-VM raw-log deficit, while final
   `n_done == next_idx == 1000` proves that every nested-prefix prompt was
   accepted and that none was silently skipped; disposition of the distinct
   archival criterion is documented in
   `reviews/QWEN_A1000_RAW_DIAGNOSTIC_ARCHIVE_REVIEW_20260802.md`;
2. all source layers 0–62 and target layer 63;
3. expected tensor shapes and dtypes, with every tensor finite;
4. exact nested-corpus, model, runtime, `jlens`, and fit-contract hashes;
5. final checkpoint/header size and SHA agreement;
6. final fp16 lens/result/input-manifest hashes matching the append-only event;
7. an exact content-addressed local backup of every registered output;
8. prompt-norm and `max_d_mean` distributions, top finite rows, sequence-length
   distribution, and explicit confirmation that no row was trimmed post hoc;
9. clean focused verification of the A1000 event and its local backup after
   registration; whole-registry verification must continue to report the two
   known historical A120--A250 deficits rather than being called clean;
10. final recovery bytes republished after removing the temporary bind, with
    cloud durability stated only after a fresh remount or new-VM rehash.

## Frozen post-A1000 decision record

Only completed, registered results belong in this table.

| stage | evidence ID | status / key result | commit |
|---|---|---|---|
| A500–A1000 structural | `p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1` | **PENDING** | **PENDING** |
| A500/A1000/published functional | `p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1` | **PENDING** | **PENDING** |
| selection margin | `p4-qwen-selection-margin-a500-a1000-dev-v1` | **PENDING** | **PENDING** |
| prompt-323 influence | `p4-qwen-lens-influence-prompt323-dev-v1` | **PENDING** | **PENDING** |
| canonical decision | `p4-qwen-canonical-lens-decision-a1000-dev-v1` | **PENDING: Q-L1–Q-L5** | **PENDING** |

The queue order remains structural → functional → selection margin →
prompt-323 influence → canonical decision. Near reclaim, the queue may stop
only after a fully registered, pushed, and locally backed-up stage by setting
`JSPACE4_STOP_AFTER`. A later default invocation verifies existing stages and
continues in the same order.

## Reclaim-aware end-of-VM ladder

After A1000 exits:

1. Preserve the final registered fit outputs locally before any Drive or Git
   transition.
2. Remove the temporary recovery bind only after verifying the final local
   checkpoint/header pair; republish checkpoint first and header last to the
   underlying Drive directory.
3. Merge remote main, the A1000 registry commit, and the preparation branch
   into `interp_jspace_part2` with ancestry preserved. Resolve the registry as
   append-only and keep native side events outside it.
4. Bind only the three pre-authorized A1000 hash slots, commit, pull, and push.
5. If at least 70 minutes remain, run the default full postfit queue. If the
   margin is shorter, run `JSPACE4_STOP_AFTER=structural` only. Do not begin
   the roughly 52-minute functional gate without a safe completion margin.
6. At 23:25 UTC stop launching compute. Rehash outputs/backups, update this
   document and both operational handoffs, run proportionate tests, pull, and
   push the single mainline branch.

## Phase 4.4 exact resume order

On the next VM, first read this file plus `inprogress.md`,
`resume-phase-4-2.md`, `jspace_lab_nextsteps_4_3.md`, and its addendum. Then:

1. Pull `interp_jspace_part2` with `--ff-only`; require a clean tree.
2. Verify the registered A1000 fit and every local backup manifest before any
   model or analysis work. Materialize the frozen Qwen model revision
   `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` and the external published
   lens before starting the queue. The latter is repository
   `neuronpedia/jacobian-lens`, revision
   `a4114d7752d11eb546e6cf372213d7e75526d3a1`, file
   `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`,
   and must rehash to
   `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1`.
3. Run the A1000 postfit queue from its first incomplete stage. Default:

   ```bash
   bash interpretability/jspace_phase4/run_qwen_a1000_postfit_queue.sh
   ```

   Preserve the normalized 820-row diagnostic archive and tensor-integrity
   audit before any new VM replaces the local execution transcript. The
   diagnostic command must use `--codex-max-line 7073`; a broader read is
   invalid because later tool output contains displayed unit-test fixtures.
   Because the recovered maxima show that prompt 323 is not the global archived
   maximum, prospectively author an omnibus retained-extremes sensitivity for
   prompts 233, 612, 616, and 660 alongside the existing prompt-112 and
   frozen prompt-323 records. It is a no-trimming robustness appendix and
   cannot revise the already-frozen Q-L branch rule.

4. Read the registered canonical decision. If Q-L1/Q-L2, bind the registered
   A1000 lens SHA and canonical-result SHA into the Bank-B geometry and P4-P2
   configs. If Q-L3/Q-L4/Q-L5, do not run either lens-dependent follow-up.
5. Run the methods-only side admission queue after all six Qwen boundary
   events exist:

   ```bash
   bash interpretability/jspace_phase4/run_phase4_post_a1000_import_queue.sh
   ```

   It admits early OLMo capability, recomputes the 16/20 joint result, then
   admits terminal Gemma and terminal OLMo. It never runs an intervention.
6. After GPU release, run the A120 capacity recovery preflight and exact
   recovery:

   ```bash
   python -m jspace_phase4.experiments.p4_qwen_historical_capacity_recovery \
     --config interpretability/jspace_phase4/configs/p4_qwen_historical_capacity_recovery.yaml \
     --preflight
   python -m jspace_phase4.experiments.p4_qwen_historical_capacity_recovery \
     --config interpretability/jspace_phase4/configs/p4_qwen_historical_capacity_recovery.yaml \
     --recover
   ```

   Install output bytes only if SHA-256 equals the immutable pin
   `6b0399df2c57158e7fdad24274e50f8c1058021d233412afdcc5177f6c651b6f`.
7. Update the known-deficit manifest. The missing historical `state.json`
   cannot be synthesized; it still requires a genuinely exact backup or an
   independently/PI-approved append-only archival resolution.
8. Run the whole-registry durability audit and pre-freeze inventory. A known
   deficit remains red. Cloud proof requires a fresh materialization boundary.
9. Assemble the independent review packet. Do not freeze or tag, and do not
   open confirmatory/replication outcomes, until the actual reviewer and PI
   complete their fields.

### Phase 4.4 compute ladder

Budget the frozen continuation in atomic stages. Measured on this VM, the
A500/A1000/published functional gate takes about 52 minutes and the analogous
single-prompt influence producer takes about 12 minutes; allow at least 70
minutes for the complete structural-through-canonical queue. Structural,
selection-margin, and canonical synthesis are short CPU/IO stages. After the
canonical event, the side-import queue is CPU/IO-only. The exact A120 recovery
requires an otherwise free GPU with at least 75 GB available, but it is a
durability repair rather than a reason to delay banking a completed Qwen
decision boundary.

## Branch-dependent follow-ups

### Q-L1 or Q-L2

- A1000 may be nominated as the canonical Qwen instrument.
- Run and register the outcome-blind Bank-B geometry gate; this exposes no
  response outcome.
- Do not run the consumed Bank-B orthogonal shot until its exact producer has
  a separately registered independent review.
- Do not run the P4-P2 CUDA smoke or 160-row consumed-family pilot until its
  exact producer has a separately registered independent review.
- After a lawful P4-P2 pilot, use only the frozen mean-masked power ruler; an
  independently reviewed untouched bank/split is still required.

### Q-L3, Q-L4, or Q-L5

- Do not bind or run the current lens-dependent Bank-B or P4-P2 follow-ups.
- Preserve the methods conclusion: operator, sparse-span, or causal stability
  failed at the precommitted level.
- The next plan must choose a reviewed consensus/cross-fit instrument,
  published-lens comparability, or removal of the affected primary; it must
  not trigger an automatic A2000 fit.

## Durability deficits and retained backups

The historical A120–A250 functional event currently has two missing outputs:

- `capacity_reconstructions_a120.pt`: exact GPU recovery prepared;
- `state.json`: exhaustive current-VM exact-copy search negative; external
  governance resolution required.

Preserve until fresh cloud proof and release review:

- `/content/sl4_work/qwen_nested_lens_fit/`;
- `/content/sl4_work/postfit_registered_backups/`;
- `/content/olmo_terminal_recovery_stage_20260802/final_exact`;
- `/content/drivefs_stale_metadata_20260802T0832Z`;
- all final A1000 recovery and queue logs under the Phase 4 Drive root.

After all model writers stop, use these exact inventory commands from the
repository root. The first durability pass and inventory are expected to stay
red while either registered historical output is missing; that is an honest
result, not a reason to edit the old registry.

```bash
export JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731
python -m jspace_phase4.durability \
  --known-deficits interpretability/jspace_phase4/protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json \
  --pass-label local-plus-mounted-drive \
  --output /content/phase4_durability_pass1.json
python -m jspace_phase4.pre_freeze_inventory \
  --pass-label phase4-part3-final
```

Only after a remount or new VM may the independent second pass run:

```bash
python -m jspace_phase4.durability \
  --known-deficits interpretability/jspace_phase4/protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json \
  --pass-label fresh-drive-materialization \
  --previous /content/phase4_durability_pass1.json \
  --output /content/phase4_durability_pass2.json
```

## Required review before Phase 4.4 execution

The project review should decide only the items that genuinely require an
external scientific choice:

1. accept or challenge the mechanical Q-L branch after reading all five
   registered A1000 stages;
2. approve or reject the exact P4-P2 producer for a consumed-data pilot;
3. approve or reject the exact Bank-B orthogonal producer for its one
   consumed-data shot;
4. choose the archival treatment for the unrecoverable historical
   `state.json` if no exact backup appears;
5. confirm the final narrowed primary family and whether Phase 4 should freeze
   as a development/methods study with no confirmatory primary;
6. decide whether Addendum M1 requires a separately named synthesis registry
   event beyond the existing methods decision record and sentence-6 wording;
   do not fabricate or rush such an event during operational closeout;
7. accept the explicit A1000 raw-diagnostic archival deviation or identify a
   genuinely exact source for prompts 1--180; checkpoint proof must not be
   mislabeled as recovered terminal text;
8. provide real independent-review and PI signatures only after the complete
   packet is examined.

This document is a handoff and analysis record, not approval, a freeze commit,
or a freeze tag.
