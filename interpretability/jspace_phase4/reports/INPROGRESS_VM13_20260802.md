# LIVE — Phase 4.3 continuation, VM13

Last updated: 2026-08-02 10:11 UTC. This is the canonical dynamic handoff.
Phase 4 remains **development-only**. Never open confirmatory or replication
intervention outcomes, and never self-sign independent-review or PI fields.

## Current boundary

- Main worktree: `/content/labs`, branch `interp_jspace_part2`, clean at
  `4ea7a9ba7a534daa61e0d8c9960763b921a1b80b` when the continuation launched.
- Remote mainline has since advanced through the ancestry-preserving terminal
  Gemma and OLMo merges plus their integration/cleanup records at `aa6663a`;
  leave the live local worktree untouched until the frozen A1000 wrapper
  finishes.
- Last registered compute commit: `f73614d6288179e73a496283dcd10090f76f2815`;
  it is pushed to origin.
- The isolated preparation branch was merged without flattening its ancestry.
- The append-only registry has 63 events. The merge preserves the preparation
  branch's exact 58-event prefix and the exact registered A500 event once.
- The merged suite passes: **153 tests passed**.
- Phase 4 Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731`.
- A1000 resumed from the exact n=554 handoff checkpoint under the frozen
  router. Atomic checkpoints through n=743 exist as a valid exact pair. Its
  first n=716 DriveFS atomic copy failed for local-cache exhaustion; the pair
  was recovered and the unchanged producer subsequently banked n=719 and
  n=743. Cloud upload remains rate-limited. A temporary recovery-directory
  bind now prevents redundant 6.6-GB DriveFS cache generations while keeping
  the producer's verified-temporary-copy/atomic-replace path unchanged. The
  wrapper is live in chunk 743:746. No partial checkpoint is registered
  evidence.

## VM13 live continuation

The frozen wrapper was relaunched from the clean main worktree. It reverified
the exact registered Branch-B functional-result hash, resolved Branch B,
verified the frozen CUDA/package/model/fit contracts and all 48 fused FLA
modules, and printed `resuming from checkpoint: 554/557 prompts processed`.
No partial checkpoint has been registered as evidence.

Newest durable resumed boundary:

| field | value |
|---|---|
| prompts banked | 743 / 1000 |
| durability | exact local + DriveFS-cache pair; cloud upload pending |
| checkpoint SHA-256 | `1601546fa7dbdee77b720dfad5d090e2adfb33b82247b7da47bab781adce555e` |
| checkpoint-state SHA-256 | `8215658433bb6a32b0bd7bb1670877830abb1af7a5df52efd91c2abedaf3da17` |
| checkpoint bytes | 6,606,047,399 |
| fit contract | `bf4caff4ff7c389d29f235a91062ae86e3a37dfc526c42bbd9af7c5d7e1f3b00` |
| checkpoint sync UTC | 2026-08-02 10:10:26 UTC |
| peak allocated VRAM | 62.846 GB |
| process state | live; active atomic chunk 743:746; unified session 83477 |

Prompt 660 is a retained finite heavy-tail row at 113.855, below the earlier
prompt-323 maximum 173.345; no outcome-dependent trimming or refit occurred.

### DriveFS durability incident at n=716

Prompt 716 completed with finite diagnostics, but the atomic Drive checkpoint
copy stopped with `ENOSPC` before publishing an n=716 Drive header. The local
checkpoint and header are complete and independently rehash to the values
above. The fitter exited and released the GPU; no scientific row or partial
milestone was registered.

The storage audit found 130 GiB of obsolete 6.6-GiB checkpoint copies in the
local DriveFS content cache. After the supported flush/unmount timed out, the
DriveFS core exited; its stale profile was quarantined intact at
`/content/drivefs_stale_metadata_20260802T0832Z`, and only its recreatable
content cache was evicted. A fresh read-only profile showed that cloud Drive
held an inconsistent recovery pair: an n=692 header naming checkpoint
`1785ae28...a4b53`, but payload `4d91f9dc...b1c4` from n=683. It also confirmed
that the OLMo terminal release files are absent in cloud storage, not merely a
directory-cache artifact.

A fresh writable DriveFS profile now contains the exact republished n=716 pair
and is attempting the correct 6,606,047,399-byte and 673-byte uploads. Google
currently returns `403 userRateLimitExceeded`; therefore DriveFS-cache presence
is not yet a cloud-durability claim. Preserve the local pair and quarantine,
and require a fresh read-only/remount rehash before calling n=716 cloud-durable.

The unchanged producer then reverified every model shard, CUDA/runtime, and all
48 fused modules and banked finite prompts through n=722. At that atomic
boundary the local and DriveFS-cache checkpoint paths independently hashed to
`47953edc...5f92b`; their 673-byte headers were byte-identical. The wrapper was
interrupted only after the n=722 sync had returned and before prompt 723 was
processed. The complete 164-record progress ledger was copied byte-for-byte
beside the local recovery pair.

To prevent the still-rate-limited DriveFS cache from accumulating another
6.6-GB generation every three prompts, the exact local fit-contract directory
is temporarily bind-mounted over only the Drive `draw_a/recovery` directory.
`findmnt` reports the overlay source, and both visible `fit.ckpt` paths resolve
to device/inode `55:7350255`. The producer still hashes a complete temporary
checkpoint and atomically replaces the recovery file; this changes neither fit
bytes nor estimator/config/runtime contracts. The pre-bind Drive recovery
directory, including n=195/n=198 contract-check files, remains preserved and
hidden beneath the mount. The unchanged wrapper restarted at clean commit
`4ea7a9b`, printed `recovered_next_idx: 722` and `resuming from checkpoint:
722/725 prompts processed`, and is live. Bound syncs through n=743 completed
and resumed normally without growing another DriveFS checkpoint generation.
Remove the bind after A1000, republish the final exact recovery
pair, and require a fresh cloud remount/rehash before making any cloud-
durability claim.

Prospective A1000 successor preparation is isolated from the live clean
worktree on branch `codex/phase4-a1000-prep-20260802`. Before any A1000
result existed, the conditional Q-L2 span-level estimand draft was committed
and pushed at `3a92492`; the A500--A1000 structural, functional,
selection-margin, and prompt-323 contracts were committed and pushed at
`542ed98`. Their A1000 hashes remain explicit placeholders. They may be bound
only after the registered n=1000 lens exists.

The preparation branch now contains remote main's terminal Gemma/OLMo ancestry
through `aa6663a` via two-parent merge `b0f74d6`; integration maintenance is
pushed through `82163fb`. In addition to the
frozen A1000 decision queue, it now contains the complete prospective
Bank-B answer-direction-orthogonal rescue: outcome-blind geometry, a
partial-dictionary prompt-only intervention core, atomic fact-level resume,
fixed variance/SESOI/power routing, focused tests, and an external-review
packet. The rescue remains
unexecuted and cannot open outcomes until a registered Q-L1/Q-L2 canonical
A1000 decision, passing geometry event, and separately registered independent
review all exist. The implementation agent has not signed that review.

It also contains the prospective P4-P2 exact/Monte-Carlo family-sign-flip
power ruler. Complete enumeration is used through 20 families and fixed
plus-one Monte Carlo signs above 20; the pilot mean is centered away before
simulation, the registered conservative planning SD is used, and a Wilson
lower-bound rule selects the first powered count. The ruler cannot authorize
an untouched bank or split. Candidate preregistration 0.11 and the development
report now reflect these decisions. The branch also contains an exact-hash-
gated A120 capacity recovery tool and the historical-state governance packet.
The required sixth conclusion-skeleton sentence, candidate freeze-gate ledger,
and methods decision record are now assembled with every unresolved gate
explicit. A prospective pre-freeze inventory generator also refuses a clean
status on known deficits, unreachable commits, namespace leakage, or
unreviewed temporary/recovery paths. The whole Phase-4 suite passes **270
tests**.

The combined OLMo-lineage and Phase-4 suites pass **329 tests** after the
terminal import normalization and admission-queue checks. The OLMo Bank-W compatibility
reader now resolves the exact registered Phase-4 registry byte prefix while
accepting only valid JSONL appended after it. The frozen side config and its
registered hash are unchanged; a regression test proves mutation within the
prefix is still rejected. This is integration maintenance for an append-only
mainline registry, not a rewrite of side-track evidence.

Preparation commit `a0a7b8f` also makes source-worktree repo outputs portable
at admission: exact merged bytes are rehashed in the current worktree and
registered by repo-relative path, while external Drive artifacts remain
absolute. Disposable clean-worktree rehearsals completed without retaining
events: OLMo admitted 5 selected / 11 selected / 14 durability outputs,
including its immutable registry snapshot; Gemma admitted 5 / 21 / 24, with
its source config materialized portably. The fresh imported Bank-W preflight
still returns 16/20 and the frozen blocked disposition.

Preparation commit `b912e37` adds a resume-safe post-A1000 admission queue. It
requires the complete six-event A1000 boundary, refuses dirty/non-mainline
execution and native `ol-*`/`gm-*` leakage, pulls before every push, and banks
early OLMo capability, the mainline 16/20 joint replay, terminal Gemma, and
terminal OLMo in that order. It also preserves every registered output set
under the local content-backup root and freshly validates existing events on
restart; it cannot run a lens-dependent pilot or open an intervention.
Preparation commit `3737b23` additionally makes the A1000 post-fit queue
rehash and locally preserve the registered fit lens, result, and manifest
before its first successor comparison. This closes the producer's intentional
local-lens deletion gap while Drive uploads remain rate-limited.

The OLMo capability boundary is now integrated into the preparation branch
with ancestry preserved through source commit `d76e937`. Both OLMo models
pass independently, but the frozen three-model intersection is only 16
families against the required 20, so P4-P3 is blocked and no Bank-W
intervention is authorized. The source-native release was normalized into
the strict mainline envelope and independently validated (5 events, 11
outputs). Preparation commit `e0d0d31` pins the exact six-line source registry
at source commit `d76e937`, includes that immutable snapshot in import-event
durability, and updates the downstream joint hashes; later OLMo registry
appends therefore cannot invalidate the early admission record. Mainline
registration and a fresh joint calculation remain queued in the prescribed
freeze order. The terminal
Gemma methods-blocker source is ancestry-merged through `b0425a4` / mainline
`c9021e5`; its normalized envelope strictly validates 5 events and 21 outputs
at preparation commit `65bfebe`, but its single Phase-4 import event remains
queued after A1000. Fetch before every push and never write native `gm-*` or
`ol-*` events from this Qwen/mainline continuation.

The isolated OLMo branch has additionally completed and registered all four
symmetric O2 capacity cells, their paired joint aggregation, the O3
provenance audit, four exact readout extracts, the joint geometry verdict,
and five figure pairs through `3b8edf0`. The O2 verdict is broadly conserved
capacity recruitment. O3 reports a descriptive dictionary-formation pattern:
operator continuity is high, but mapped-token continuity fails and sparse
selection diverges at the first released Think transition. This cannot alter
the already-failed Phase-4 O1 service decision, does not identify a causal
training effect, and does not license Bank-W intervention. Methods event
`ol-independent-reconstruction-v1` registered at source commit `471a48f` and
verifies thirteen outputs: all registered O1/O2/O3 summaries, all five PNGs
byte-for-byte, all fourteen pinned weight shards, and one frozen eight-score
Think row with zero numerical drift. It opens no new item, intervention,
model, stage, O5, confirmatory, or replication outcome. The 61-GiB snapshot
was deleted only after Git and 88-output recovery verification. The claims
ledger and state of record are durable through `ad0041b`. The 13-page paper
at `f2d4d37` passed an independent render audit. The terminal OLMo handoff
registered at `a28cdd5` from clean producer `7148d01`: 25 origins / 24 live
events / 101 live outputs, 58 tests, and one 13-output methods release event.
No scientific cell or authorization was added. The complete side branch is
ancestry-merged into remote main at `65a7875`, with integration/cleanup records
through `aa6663a`.

This VM initially found all thirteen terminal release files absent from cloud.
The only two missing pre-release live outputs were the bounded O5 decision
pair. They were deterministically rebuilt at original clean commit `843eabd`
and accepted only after matching registered hashes `d31d23e...4499` and
`9e3fbb45...675a` byte-for-byte, restoring the 23-event/88-output prefix. The
terminal producer was then replayed into isolated local staging at clean
commit `7148d01` and its original generation second
`2026-08-02T06:06:56Z`. The release environment was recovered exactly from
the registered foundation lock plus `jlens==0.1.0` and the recorded TeX
toolchain; all thirteen staged files matched the registered event before any
publication. They were atomically restored to their registered Drive paths,
and the ancestry-merged terminal verifier now passes 24 live events / 101
outputs with exact registry prefix `db3fe202...e80a`. Exact local recovery
staging remains at `/content/olmo_terminal_recovery_stage_20260802/final_exact`.
Because Drive uploads are still rate-limited, call this local/DriveFS-cache
recovery, not fresh cloud durability. Preparation commit `67c0637` now carries
the strict terminal envelope and a fresh saved validation over exactly one
methods event and thirteen outputs; the source-native and normalized bundle
SHA-256 values are `a2486ec5...a2a` and `be1870d0...a09`. Mainline admission
remains queued until the Qwen A1000 branch resolves.

The same OLMo branch registered semantic checkpoint inventory v2 at `2afb010`:
inventory v1 remains immutable but is explicitly superseded after identical
100,278 token-ID mappings and identical encodings on the frozen corpus. The
official SFT/DPO intermediates are semantically eligible, but the bounded O5
audit/decision at `4c6617e` found no identifiable crossed-intervention
estimand and records `not-executed-no-proxy-substitution`. No intermediate-
model behavior or proxy O5 outcome has been opened.

The Gemma 40-cell Stage-1 result found local tangent mismatch at every frozen
layer. Its precommitted forward-versus-fallback JVP parity diagnostic failed
the strict all-slot relative-error gate (0.002458 > 1e-5), despite exact
selected-row replay. G2/G3 were stopped and the terminal release is a
methods-only blocker: no mechanism, nondifferentiability, late-band,
workspace, confirmatory, or replication conclusion is licensed.

No confirmatory or replication intervention outcome has been opened.

## Completed Qwen draw-A n=500 fit

Live evidence: `p4-qwen-lens-fit-drawA-n500-dev-v1`.

The exact n=250 estimator continued on the unchanged nested draw-A corpus and
fit contract to n=500. The n=499 recovery handoff terminated only the active
Python process, removed the exact temporary recovery bind, restored the
registered A250 bytes, and launched the unchanged normal producer for prompt
500. The final registered event was committed at `9eac945`.

| field | value |
|---|---|
| prompts | 500 |
| lens SHA-256 | `84404956fb71a84f5af7fa22c34a8f07761777d048b3db314b1330037e4168a8` |
| lens bytes | 3,303,034,078 |
| final checkpoint SHA-256 | `f1f5c3eebca93bd0f8d00cbd8794848df569bc67be475d3ab4ef319d8aaa61b1` |
| checkpoint bytes | 6,606,047,399 |
| fit contract | `bf4caff4ff7c389d29f235a91062ae86e3a37dfc526c42bbd9af7c5d7e1f3b00` |
| peak allocated VRAM | 62.846 GB |
| model | Qwen/Qwen3.6-27B at `6a9e13bd...` |

All 250 continuation prompt records 251--500 have finite diagnostics. Prompt
323 is the retained heavy-tail maximum
(`max||J||/sqrt(d)=173.345`, `max_d_mean=0.271`), comparable to the
registered prompt-112 sensitivity. Every later prompt is at or below 79.579;
prompt 500 is ordinary at 13.380. No row was trimmed or refit.

The VM12 handoff named two ephemeral content-addressed paths that did not
carry onto VM13:

```text
/content/phase4_recovery_local_20260801_w4Pv6g/qwen36-27b_jlens_drawA_n0500.preserved.pt
/content/sl4_work/inputs/84404956fb71a84f5af7fa22c34a8f07761777d048b3db314b1330037e4168a8/qwen36-27b_jlens_drawA_n0500.pt
```

On VM13, the mounted A500 lens, result, and manifest first rehashed to their
registered values, then all three were atomically recopied and rehashed under
`/content/sl4_work/postfit_registered_backups/p4-qwen-lens-fit-drawA-n500-dev-v1/`.
The four post-A500 evidence sets were restored the same way. The five current
backup manifests cover 35 outputs / 3,462,433,852 bytes exactly.

## Completed post-A500 queue

The exact A500 hash was bound in only the two frozen successor configs and
pushed at `e6fea8e`. The four-stage queue completed at 20:10 UTC. Each event
was registered, committed, pushed, then copied into a hash-verified local
backup under `/content/sl4_work/postfit_registered_backups/<evidence-id>/`.

### 1. A250--A500 structural convergence

Evidence: `p4-qwen-lens-convergence-drawA-n250-n500-dev-v1`.
Registry commit: `bbbd6a7`.

- Frozen L20--L44 median raw/minus-identity/minus-scaled-identity matrix
  cosines: 0.99791 / 0.99936 / 0.99774.
- Conservative task-token median cosine: 0.997715 (floor 0.95).
- Conservative task-token q05: 0.996838 (floor 0.90).
- Both structural gates pass.
- p4f19 was visually checked and is legible.

### 2. A250--A500 functional gate

Evidence:
`p4-qwen-multilens-functional-gate-a250-a500-published-dev-v1`.
Registry commit: `d54688e`.

The frozen decision is **Branch B**.

| gate | result |
|---|---:|
| normalized projector overlap | 0.70302 < 0.85 — fail |
| selected-ID Jaccard | 0.53846 < 0.75 — fail |
| bridge rescue difference | 0.55891 > 0.25 — fail |
| occupancy | pass |
| centered excess | pass |
| span-safe-specific point estimate | pass |
| tail rate | pass |
| G4 positive control | pass |
| bridge preference | pass |
| structural dependency | pass |

The span-safe-specific mean difference is -0.02432 nat, but its 95% interval
[-0.25248, 0.16762] does not establish formal TOST equivalence. The
precommitted interpretation is functional instability; continue draw A to
n=1000. Neither A250 nor A500 is canonical. p4f20 was visually checked.

### 3. Official-Qwen mode-v2 baseline

Evidence: `p4-qwen-mode-gate-dev-v2`.
Registry commit: `08d5773`.

All model-backed development gates pass:

- thinking-on/off accuracy: 0.90 / 0.85;
- parse failure: 0 / 0;
- truncation: 0 / 0;
- parse-valid in both modes: 20/20;
- correct in both modes: 17/20, above floor 12;
- median generated tokens: 323.5 on / 4 off;
- thinking-on minus off accuracy: +0.05, 90% family-bootstrap interval
  [0.00, 0.15].

This is baseline parser/capability evidence only. No P4-P2 intervention ran.
p4f21 was visually checked.

### 4. Qwen Bank-W capability

Evidence: `p4-bank-w-capability-qwen36-27b-dev-v1`.
Registry commit: `f73614d`.

Qwen is independently capability-eligible:

- complete grid: 384 rows;
- accuracy: 0.83333 at both low and high load;
- high-minus-low accuracy: 0.00000;
- 90% family-bootstrap interval: [-0.02083, 0.02083], wholly inside
  [-0.08, 0.08];
- capable at both loads: 20/24 families, meeting the floor.

Both OLMo capability gates later passed independently, but the all-model
joint-support intersection is 16/20 and fails the frozen floor. P4-P3 is
blocked and no Bank-W intervention outcome ran.

## Report and figure integration

The exact p4f19/p4f20/p4f21 PNG/PDF pairs were copied from their registered
local backups into `reports/figures/` and rehashed to the registry. All
three PNGs were visually inspected. The Markdown development report,
preregistration candidate 0.9, TeX handout, this handoff, and the restart
handoff were integrated, rebuilt, visually checked, and pushed at
`3b041735d8b842de46a9c0a474fccd0c44e0841a`.

The isolated successor branch has since advanced the prospective scientific
preregistration to candidate 0.11 and refreshed the Markdown development
report at `3396469`, with a P4-P2 gate-wording clarification at `e74a1c7`;
no freeze or PI approval is implied.

## Paused Branch-B A1000 continuation

The frozen router verified registered result SHA-256
`c3290abc85c93b0fc8144432a2b674b886e3173609379598bd8c95a96ec471e4`,
resolved Branch B without reinterpretation, and launched:

```bash
bash interpretability/jspace_phase4/run_qwen_continuation_fit.sh draw_a_n1000
```

The run started from clean commit `3b04173`, recovered exact n=500, and
reverified the model, CUDA, and all 48 fused modules. It banked atomic
three-prompt checkpoints through the intentional handoff boundary:

| field | value |
|---|---|
| prompts banked | 554 / 1000 |
| checkpoint SHA-256 | `bf992067d690123109198c182a21169379e5752d89e73e96514fab7127fba74d` |
| checkpoint-state SHA-256 | `cc5478c836023183ec974d30a2c6eae7e1f7856d50fedac96c24ef35cfa3590d` |
| checkpoint bytes | 6,606,047,399 |
| seconds/new prompt | 178.48 |
| peak allocated VRAM | 62.846 GB |
| process state | intentionally paused; no owner |
| next chunk | 554:557 |

Authoritative recovery files and prior router log:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup_20260801.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_frozen_branch_followup.lock
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json
```

All continuation diagnostics through prompt 554 are finite. Prompts 541 and
542 are retained local-tail rows at 25.727 and 36.435 respectively, still
below the later-A500 maximum 79.579 and the retained prompt-323 maximum
173.345; no outcome-dependent trimming occurred.

At the measured rate, n=554--1000 alone requires about 22.1 compute hours,
before successor gates and report integration. n=554 is a recovery boundary,
not a registered evidence milestone and not a canonical-lens decision.

The wrapper was intentionally interrupted with exit code 130 only after the
n=554 checkpoint was mirrored and rehashed. Its next loop announced 554:557
and loaded n=554, but prompt 555 was never processed. The zero-byte lockfile
paths remain on Drive as normal `flock` targets; no process holds them.

### Exact next-VM restart

1. Mount this same Drive root; check out/pull clean branch
   `interp_jspace_part2` in `/content/labs` and read both handoff files.
2. Verify `checkpoint_state.json` reports `n_done=554`; rehash
   `recovery/fit.ckpt` to the checkpoint SHA above; verify no fitter, router,
   lock holder, or GPU process already owns the run.
3. Provide the frozen runtime and Qwen cache (default
   `HF_HUB_CACHE=/content/hf_local`) for model revision
   `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
4. From the clean repository run:

```bash
bash interpretability/jspace_phase4/run_qwen_frozen_branch_followup.sh
```

The router must reverify the registered functional-result hash, resolve
Branch B, and print `resuming from checkpoint: 554/...`. Do not launch the
lower-level fitter in parallel, delete the recovery pair, or register n=554
as a partial pseudo-milestone.

Do not delete the Qwen cache while A1000 needs it. Run the two OLMo Bank-W
gates only after A1000 or after a safe, explicit cache/staging transition.

## Drive durability audit

The 23:25 UTC whole-registry pass checked 49 live events and 220 output
references. It found no wrong hashes, but three paths from the older
`p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1` event were
absent. The published reconstruction was restored byte-for-byte from the
hash-verified A250--A500 backup and now matches registered SHA-256
`ebe0c641115b40db089412fb969e8c37eef90a7992a40996c4277d9dac43d703`.

Thus 218/220 live outputs are hash-accounted. These two older registered paths
remain missing in that event directory:

```text
state.json
expected 361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8

capacity_reconstructions_a120.pt
expected 6b0399df2c57158e7fdad24274e50f8c1058021d233412afdcc5177f6c651b6f
```

No preserved exact copy was found on this VM. Do not fabricate bytes, edit the
registry, or rerun model-scale work on CPU. Recover exact bytes from backup or
version history if available; otherwise use a reviewed append-only correction
plan, then rerun `jspace-phase4 verify`.

An exact-hash-gated GPU recovery tool is now prepared on the successor branch
for `capacity_reconstructions_a120.pt`. It verifies the original event,
config, cache, lens, dependency, function, shard, and runtime hashes and only
restores the registered path if the rebuilt bytes match the expected SHA.
Run it only after A1000 releases the GPU. The missing historical `state.json`
includes irrecoverable timing fields and requires an exact backup or an
independently/PI-approved append-only archival correction; do not synthesize
it.

Preparation commit `2e602d7` records an exhaustive exact-copy search. The live
Drive API exposes the target folder and thirteen current children but no
target `state.json` cloud ID; sixty exact-name objects and 1,246 revisions
contain no target surface. The quarantined pre-incident metadata independently
contains the same folder/children but no state child, tombstone, deleted-item,
or pending operation. No bytes were restored. The negative audit strengthens,
but does not replace, the required independent/PI governance decision.

## Freeze blockers

Phase 4 is **not frozen**. Remaining blockers include:

1. completion of the Branch-B A1000 fit and its frozen structural,
   functional, selection-margin, prompt-323, and canonical-decision queue;
2. canonical Q-L1/Q-L2 binding, a passing Bank-B geometry event, and
   independent review before the addendum-required consumed-development
   orthogonal shot; candidate 0.11 already makes P4-P1 estimation-only and
   the shot cannot restore it;
3. canonical Q-L1/Q-L2 binding and independent review before the P4-P2 GPU
   pilot, followed by its fixed exact power ruler and an independently
   reviewed untouched bank/split if feasible;
4. mainline import of the already decisive OLMo early bundle; P4-P3 is
   blocked at 16/20 and no Bank-W intervention is authorized;
5. recovery of the two missing older A120--A250 outputs and a clean
   whole-registry verification after DriveFS durability is confirmed;
6. independent protocol review;
7. PI sign-off, freeze commit, and freeze tag.

Confirmatory and replication jobs remain forbidden.
