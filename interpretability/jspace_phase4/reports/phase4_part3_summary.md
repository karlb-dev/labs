# Phase 4 part 3 summary and Phase 4.4 handoff

**PHASE 4.3 COMPUTE CLOSEOUT — DEVELOPMENT ONLY — NOT A FREEZE RECORD**

Finalized: 2026-08-02 23:00 UTC. The A1000 fit and its first frozen successor,
the A500--A1000 structural event, are registered, locally backed up, merged,
and pushed. The later functional, selection-margin, influence, and canonical
stages remain intentionally unrun for Phase 4.4. No confirmatory or
replication intervention outcome was opened, and this implementation agent
does not sign independent-review or PI fields.

## Executive state

Phase 4.3 ended at a clean structural boundary. The exact cumulative Qwen
draw-A estimator reached 1,000 accepted prompts under unchanged fit contract
`bf4caff4ff7c389d29f235a91062ae86e3a37dfc526c42bbd9af7c5d7e1f3b00`.
The final checkpoint is 6,606,047,399 bytes with SHA-256
`fd5a4ae614eef46002cc987a038d9a391016b7fbc91a754eed2adff83f6abf20`;
the registered fp16 lens is 3,303,034,078 bytes with SHA-256
`6e48c7731501d0fc6030f1d60eff6f19b211756f40ef0cd6e499e414f08f6bd6`.
The A500--A1000 structural event then passed both prospectively frozen gates.
The roughly 52-minute functional stage was not started because it lacked a
safe margin before the approximately 23:38 UTC reclaim.

The completed Gemma and OLMo side tracks, prospective preparation, A1000 fit,
exact binding, and structural event now share ancestry on the single
`interp_jspace_part2` branch. Merge commit `3d1f1bf` preserves history without
squashing. The 69-row Phase 4 registry contains no native `gm-*` or `ol-*`
event.

The final fit-diagnostic audit found and corrected an important wording error:
prompt 323 is not the overall archived maximum. Across the recoverable archive,
the largest
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

Final integrated validation is 279 Phase 4 tests, 59 OLMo-lineage tests, and
48 Gemma tests: 386 total. The sole Gemma Torch FX warning is unchanged. The
focused exact-binding suite additionally passes 12/12. Both the unified
mainline and the preparation branch are pushed.

### Phase 4.3 completion audit

This table separates implemented preparation from registered scientific
evidence and from approvals that this implementation agent cannot supply.

| boundary | state at this draft | closure evidence |
|---|---|---|
| governing Part 3/addendum instructions | complete | read into the operational handoffs and prospective contracts |
| Gemma and OLMo branch ancestry | complete | both terminal branches merged into the remote-main ancestry; native side namespaces excluded from the Phase 4 registry |
| prospective A1000 successor contracts | complete | structural, functional, margin, influence, canonical, import, recovery, durability, and reclaim-stop code committed before the A1000 result |
| A1000 fit | **COMPLETE / REGISTERED** | event at `c80017d`; exact registered-output and recovery backups; normalized 820-row archive; 63-layer tensor audit passes |
| frozen post-A1000 decision queue | **STRUCTURAL COMPLETE; LATER STAGES PENDING** | structural event at `d144ac1`; functional through canonical remain sealed for Phase 4.4 |
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

These values come only from registered and freshly rehashed bytes.

| field | final value |
|---|---|
| evidence ID | `p4-qwen-lens-fit-drawA-n1000-dev-v1` |
| prompts | 1,000; `n_done == next_idx == 1000` |
| final lens SHA-256 | `6e48c7731501d0fc6030f1d60eff6f19b211756f40ef0cd6e499e414f08f6bd6` |
| final lens bytes | 3,303,034,078 |
| final checkpoint SHA-256 | `fd5a4ae614eef46002cc987a038d9a391016b7fbc91a754eed2adff83f6abf20` |
| checkpoint-state SHA-256 | `b0cf4c8d7e6debc20d78a9ed49ba97025b27a2cee9a18ce517d67396801d6d2a` |
| result SHA-256 | `555569478f759f68ae205a2a495a04490a9ee9aa8cf5a875644935a9a6d06ef8` |
| input-manifest SHA-256 | `ce7e057e1298f8bc4217df3ce0a62afb916f4e3edf54466d1c120eaa9d9138f1` |
| registration commit | `c80017d` (ancestry integrated by `3d1f1bf`) |
| registered-output backup | `/content/sl4_work/postfit_registered_backups/p4-qwen-lens-fit-drawA-n1000-dev-v1/backup_manifest.json`, SHA-256 `cdaae0129c6f2ff55650481bf74713ccfb0af61039ceb4bf8d0dcc0a6ab5397c` |
| recovery backup | content-addressed under `final_backups/.../fd5a4ae6...abf20/`, manifest SHA-256 `cf8c7308bd6b9e50459ba75878b098d496e271a453625f41d20581062fda812a` |
| cloud durability | final checkpoint/progress/header republished checkpoint-first/header-last; cache-local rehash passes, but fresh cloud materialization remains pending |

The engineering QA archive contains 820 finite rows spanning prompts 181--1000,
zero skip rows, prompt-norm median/q95/max 7.8615/26.8911/231.101, and
`max_d_mean` median/q95/max 0.00758/0.03013/0.425. A log-log regression gives
coefficients +1.01888 for prompt magnitude and -0.94767 for prompt index
(R-squared 0.99714): estimator movement falls approximately as 1/n conditional
on contribution magnitude, while the per-prompt magnitude distribution remains
heavy-tailed. This is an equal-weight recurrence consistency check, not by
itself proof of structural or causal convergence. No row was trimmed.

The mmap tensor audit verifies 63 source layers (0--62), target layer 63,
5,120-by-5,120 shapes, fp32 checkpoint sums, fp16 registered lens tensors,
full finiteness, and bit-exact equality of every lens layer to
`(checkpoint_sum / 1000).to(float16)`.

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
| A500–A1000 structural | `p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1` | **PASS**: conservative assay L20--L44 task q50 0.998702 >= 0.95; q05 0.998122 >= 0.90 | `d144ac1` |
| A500/A1000/published functional | `p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1` | **PENDING** | **PENDING** |
| selection margin | `p4-qwen-selection-margin-a500-a1000-dev-v1` | **PENDING** | **PENDING** |
| prompt-323 influence | `p4-qwen-lens-influence-prompt323-dev-v1` | **PENDING** | **PENDING** |
| canonical decision | `p4-qwen-canonical-lens-decision-a1000-dev-v1` | **PENDING: Q-L1–Q-L5** | **PENDING** |

The structural result SHA-256 is
`eaf8a63ed038eedaabbbd037be26984a454cddd4182bb42c3e37270a91ec193d`.
Its six-output local backup manifest hashes to
`5bff1706c7684c38d8f103a2d725059a9a1bca8f0b8aaad3a528047390c57e09`.
Within assay L20--L44, raw/minus-identity/minus-scaled-identity operator
cosines are 0.998739/0.999610/0.998632; residual principal-subspace similarity
is 0.998196, and the assay-band medians of the per-layer median/max
principal-angle summaries are 1.824/8.853 degrees. These support
the registered operator-level pass while leaving sparse and causal stability
to the still-sealed successors. The prompts-501--1000 incremental block is
also highly aligned to A500 (conservative task q50/q05
0.994787/0.992476), whereas A1000 versus the partially specified external
published lens is only 0.906757/0.835442. The correct interpretation is strong
same-corpus estimator convergence, not published-recipe interchangeability.
The development-only cross-event synthesis in
`figures/p4qa02_qwen_structural_progression.png` shows conservative task
q50 0.995223 → 0.997715 → 0.998702 and q05
0.993079 → 0.996838 → 0.998122 across the three registered nested increments.
This is descriptive progression, not a fitted convergence-rate claim. The
registered structural figure and both engineering-QA figures were visually
inspected after copying into the report tree; no clipping, missing panel, or
mislabeled boundary was observed.

The queue order remains structural → functional → selection margin →
prompt-323 influence → canonical decision. Near reclaim, the queue may stop
only after a fully registered, pushed, and locally backed-up stage by setting
`JSPACE4_STOP_AFTER`. A later default invocation verifies existing stages and
continues in the same order.

## Completed reclaim-aware end-of-VM ladder

The fit outputs and recovery state were copied to two exact local backup
families; the temporary bind was removed; final recovery bytes were republished
checkpoint-first/header-last; all branches were ancestry-merged; the three
pre-authorized bindings were set to the registered lens hash; focused binding
tests passed 12/12; and the structural-only queue registered, backed up, pulled,
and pushed its result. No unsafe long stage was launched.

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
3. Resume the A1000 postfit queue from its first incomplete stage, the
   functional gate. The queue first verifies the already registered fit and
   structural event, then continues in frozen order. Default:

   ```bash
   bash interpretability/jspace_phase4/run_qwen_a1000_postfit_queue.sh
   ```

   The normalized 820-row diagnostic archive, its QA figure, and the passing
   tensor-integrity audit are tracked under `reports/` and mirrored under the
   Drive diagnostic root. Any regeneration must use `--codex-max-line 7073`;
   a broader read is invalid because later tool output contains displayed
   unit-test fixtures. Because the recovered maxima show that prompt 323 is not the global archived
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
