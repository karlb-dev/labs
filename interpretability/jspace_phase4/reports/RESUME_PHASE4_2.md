# Resume Phase 4.2 — restart-safe handoff

Generated 2026-08-01 23:26 UTC on VM12. Phase 4 is development-only. Never
open confirmatory or replication intervention outcomes before independent
review, PI sign-off, a freeze commit, and a freeze tag.

Canonical mirrors of this file must remain byte-identical:

```text
/content/resume-phase-4-2.md
/content/drive/MyDrive/interpret/resume-phase-4-2.md
```

Read the fuller live ledger at:

```text
/content/drive/MyDrive/interpret/inprogress.md
/content/labs/interpretability/jspace_phase4/reports/INPROGRESS_VM12_20260801.md
```

## VM13 final closeout — 2026-08-02 23:00 UTC

A1000 and its structural successor are complete, registered, backed up, merged
with the terminal Gemma/OLMo ancestry, and pushed on `interp_jspace_part2`.
Exact lens/checkpoint/header SHA-256 values are respectively
`6e48c7731501d0fc6030f1d60eff6f19b211756f40ef0cd6e499e414f08f6bd6`,
`fd5a4ae614eef46002cc987a038d9a391016b7fbc91a754eed2adff83f6abf20`,
and `b0cf4c8d7e6debc20d78a9ed49ba97025b27a2cee9a18ce517d67396801d6d2a`.
The 63-layer exact-mean tensor audit passes. The recoverable QA archive has
820 finite rows (181--1000), no skips, and an explicit missing-raw-text
deviation for 1--180. Structural conservative q50/q05 are
0.998702/0.998122, both passing.

The next exact command after a clean `git pull --ff-only` is:

```bash
bash interpretability/jspace_phase4/run_qwen_a1000_postfit_queue.sh
```

It verifies the registered fit and structural stages and resumes at the
roughly 52-minute functional gate. Do not rerun the fit or structural producer.
The complete Phase 4.4 ladder, backup hashes, archival deviation, cache pins,
and external governance decisions are in `reports/phase4_part3_summary.md`.

## Historical VM13 mid-run delta — 2026-08-02 21:51 UTC

The operational n=554 restart below succeeded under the unchanged frozen
producer. A1000 is live; the newest valid local atomic boundary is n=980,
checkpoint SHA-256
`2b8c3e97fe9a74340f7504f6bcf8cfea32ac088e05dfceaf78640f0bac52ce70`,
checkpoint-state SHA-256
`3b5775ffb95cd80c6be08e70ca86510414a6f90622b726e5493ef664125b1956`,
and the unchanged wrapper is live in chunk 980:983 under unified session
83477. It reverified every model shard, CUDA/runtime, and all 48 fused modules,
then printed `recovered_next_idx: 722` and `resuming from checkpoint: 722/725
prompts processed`. A new raw-log audit corrects the earlier description of
prompt 323 as the overall maximum. Among archived rows through n=980, prompts
233, 323, 612, 616, and 660 are retained finite extremes at 189.182, 173.345,
151.626, 231.101, and 113.855, respectively; no trimming or refit occurred.
The current VM transcript recovers the n=555--716 Drive-log gap, while raw
terminal rows 1--180 from the prior VM are unavailable here. Record raw-log
coverage separately from the checkpoint's equal `n_done`/`next_idx` proof of
accepted prompts. Keep the frozen prompt-323 audit unchanged and route any
broader retained-top-row influence audit to Phase 4.4. Use the canonical
dynamic handoff above for the newest checkpoint before any restart.

The expected VM reclaim is approximately 2026-08-02 23:38 UTC. At the
observed rate, the A1000 checkpoint should land near 22:50 UTC and
final serialization, registration, backup, integration, and push near
22:55--23:05 UTC. Do not launch new long-running compute after 23:25 UTC. If
the full postfit queue lacks a safe margin, stop after its registered and
locally backed-up structural stage with `JSPACE4_STOP_AFTER=structural`.

The first n=716 Drive atomic copy failed with `ENOSPC` after computation but
before publishing its header. The fitter exited cleanly and the complete local
pair retained the hashes above. A storage audit found 130 GiB of obsolete
DriveFS checkpoint cache; the stale profile is quarantined intact at
`/content/drivefs_stale_metadata_20260802T0832Z`, its recreatable content cache
was evicted, and a fresh profile republished the exact n=716 pair. A read-only
cloud audit found an inconsistent older pair (n=692 header, n=683 payload), and
Google currently rejects the correct n=716 uploads with
`403 userRateLimitExceeded`. Treat n=716 as local/DriveFS-cache durable but not
cloud-durable until a fresh read-only/remount rehash succeeds. Do not delete
the local pair or quarantine.

After recovery, finite prompts through n=722 banked under the unchanged fit
contract. At n=722, both recovery checkpoint paths hashed to the value above,
the headers were byte-identical, and the progress ledger held 164 records. The
producer was interrupted only after that atomic sync returned and before
prompt 723. Because Google remained rate-limited, the exact local fit-contract
directory was then temporarily bind-mounted over only the Drive recovery
directory. Both visible checkpoint paths resolve to device/inode `55:7349719`;
the pre-bind Drive directory and its older contract-check files remain intact
underneath. This keeps the producer's verified temporary copy and atomic
replace behavior but prevents a new 6.6-GB DriveFS cache generation every
three prompts. Bound syncs through n=980 completed and resumed normally without
another DriveFS checkpoint generation. Remove the bind after A1000,
republish the final exact pair, and prove cloud durability only by a fresh
remount and rehash.

The isolated successor-preparation branch is
`codex/phase4-a1000-prep-20260802`. It contains the prospective A1000 queue,
selection-margin and prompt-323 audits, canonical decision producer, P4-P2
and Bank-B methods decisions, durability tooling, Bank-W v3/power work, and
the ancestry-preserving OLMo capability merge. The strict OLMo import envelope
validates 5 source events and 11 outputs. Both OLMo models pass independently,
but the three-model common support is 16/20, so P4-P3 remains blocked and no
Bank-W intervention is authorized. Native `ol-*`/`gm-*` events remain outside
the Phase 4 registry.

The preparation branch now ancestry-contains terminal remote main `aa6663a`
through two-parent merge `b0f74d6`, with append-only source-registry
integration maintenance at `82163fb`. The combined OLMo/Phase-4 suite passes
338 tests after terminal import normalization, admission-queue checks, and
durability-deficit record binding. The frozen OLMo compatibility config is
unchanged: its reader verifies the exact registered Phase-4 registry prefix
and permits only valid JSONL appended after that byte boundary.

The preparation branch is clean and pushed; its full combined Phase-4/OLMo
suite passes 338 tests. The ancestry-merged terminal Gemma suite adds 48
passing tests, for 386 total; its one existing Torch FX warning is unchanged.
The branch now includes the complete prospective one-shot
Bank-B answer-orthogonal producer and external-review instructions. That shot
has not run: exact A1000/canonical-decision binding, outcome-blind geometry,
and separately registered independent review are hard prerequisites.
It also includes the prospective exact/Monte-Carlo P4-P2 sign-flip power
ruler, which masks the pilot mean and cannot authorize an untouched split,
candidate preregistration 0.11, the refreshed development report, and an
exact-hash-gated A120 capacity recovery tool. Its early OLMo import now pins
the exact six-event source-registry boundary at `d76e937` in Git and includes
that snapshot in import durability, so concurrent later `ol-*` appends cannot
invalidate the strict 5-event/11-output validation. OLMo O2 and O3 geometry/figures
are complete through `3b8edf0`; the descriptive verdict is a dictionary-
formation pattern, while the failed O1 service gate still forbids Bank-W
intervention. Semantic checkpoint inventory v2 is registered, and the bounded
O5 decision is `not-executed-no-proxy-substitution` because no crossed causal
estimand is identifiable. Methods event `ol-independent-reconstruction-v1`
registered at `471a48f`: its thirteen outputs reproduce O1/O2/O3 summaries,
five PNGs byte-for-byte, fourteen weight-shard hashes, and one frozen eight-
score Think row with zero drift. The 61-GiB snapshot was deleted only after
Git and 88-output recovery verification. Claims/state are durable through
`ad0041b`; the 13-page paper at `f2d4d37` passed an independent render audit.
The terminal OLMo handoff registered at `a28cdd5` from clean producer
`7148d01`: 25 origins / 24 live events / 101 live outputs and one 13-output
methods event. The complete side branch is ancestry-merged into remote main
through `65a7875` / `aa6663a`. This VM recovered the two missing pre-release
O5 decision files at original clean commit `843eabd`, with both registered
hashes exact, then reproduced all thirteen terminal outputs in isolated local
staging at the original release second and environment. Every file matched its
registered hash before atomic restoration. The merged terminal verifier now
passes 24 live events / 101 outputs and registry prefix `db3fe202...e80a`.
Preserve `/content/olmo_terminal_recovery_stage_20260802/final_exact` until a
fresh cloud remount proves durability; current DriveFS presence is cache-local
while uploads are rate-limited. Preparation commit `67c0637` carries a strict
normalized terminal envelope and fresh saved validation over exactly one
methods event and thirteen outputs; the source-native and normalized bundle
SHA-256 values are `a2486ec5...a2a` and `be1870d0...a09`. Admit it only after
A1000 resolves.
Gemma Stage 1 reports
local tangent mismatch at all frozen layers; its precommitted exact-backend
parity diagnostic failed the all-slot relative-error gate (0.002458 > 1e-5),
so G2/G3 stopped. The terminal methods-blocker branch is ancestry-merged to
mainline at `c9021e5`, and the strict Phase-4 envelope validates 5 events / 21
outputs at preparation commit `65bfebe`; registration remains queued until
after A1000. No OLMo intermediate-model behavior or proxy O5 outcome has been
opened.
Both normalized side imports now have passing disposable clean-worktree
registration rehearsals. OLMo registers 14 durability outputs and Gemma 24;
tracked source-worktree files are rehashed and recorded portably, while Drive
artifacts remain absolute.
The resume-safe post-A1000 queue at `b912e37` requires all six registered Qwen
boundary events, refuses a dirty/non-mainline tree and native `ol-*`/`gm-*`
leakage, pulls before every push, and banks early OLMo capability, the 16/20
joint replay, terminal Gemma, and terminal OLMo in that order. It preserves
every registered output set locally and cannot run a pilot or intervention.
Preparation commit `3737b23` also makes the A1000 post-fit queue preserve the
registered fit lens, result, and manifest before its first comparison.
The required sixth paper-conclusion sentence, candidate freeze-gate ledger,
and methods decision record are also assembled with unresolved fields still
explicitly pending. The pre-freeze inventory generator is prepared and will
remain red on any known durability deficit.

## Current state

- Repository: `/content/labs`
- Branch: `interp_jspace_part2`
- Last registered compute commit: `f73614d6288179e73a496283dcd10090f76f2815`
- Registry: 63 append-only events
- Tests after merge: 153 passed
- A1000 was intentionally paused at the clean n=554 checkpoint for the VM12
  handoff and has now resumed under the frozen wrapper. The historical pause
  section below records that restart boundary; the canonical dynamic handoff
  contains the newest live checkpoint.

The CPU preparation branch is merged. The A500 event, four post-fit events,
and their commits are pushed. All registered post-fit outputs have verified
local backups under:

```text
/content/sl4_work/postfit_registered_backups/
```

## Completed A500 boundary

Evidence: `p4-qwen-lens-fit-drawA-n500-dev-v1`.

```text
lens SHA-256:
84404956fb71a84f5af7fa22c34a8f07761777d048b3db314b1330037e4168a8

checkpoint SHA-256:
f1f5c3eebca93bd0f8d00cbd8794848df569bc67be475d3ab4ef319d8aaa61b1

current VM13 local lens backup:
/content/sl4_work/postfit_registered_backups/p4-qwen-lens-fit-drawA-n500-dev-v1/00_qwen36-27b_jlens_drawA_n0500.pt
```

The two older VM12 content-addressed paths did not carry onto VM13. The A500
event and all four post-A500 events were rehashed from their registered paths
and atomically recopied locally; five manifests now cover 35 exact outputs /
3,462,433,852 bytes.

All prompts 251--500 logged finite diagnostics. Prompt 323 is a retained
heavy-tail row; no outcome-dependent trimming occurred.

## Completed post-fit decisions

1. `p4-qwen-lens-convergence-drawA-n250-n500-dev-v1`
   - structural median/q05: 0.997715 / 0.996838;
   - both frozen structural gates pass;
   - commit `bbbd6a7`.
2. `p4-qwen-multilens-functional-gate-a250-a500-published-dev-v1`
   - projector overlap 0.70302 < 0.85;
   - selected-ID Jaccard 0.53846 < 0.75;
   - bridge-rescue difference 0.55891 > 0.25;
   - frozen result **Branch B: continue draw A to n=1000**;
   - commit `d54688e`.
3. `p4-qwen-mode-gate-dev-v2`
   - all development gates pass;
   - on/off accuracy 0.90/0.85, zero parse failures and truncations,
     17/20 common-correct;
   - commit `08d5773`.
4. `p4-bank-w-capability-qwen36-27b-dev-v1`
   - Qwen independently eligible;
   - both-load accuracy 0.83333, difference 0.00000 with 90% interval
     [-0.02083, 0.02083], 20 capable families;
   - commit `f73614d`.

p4f19, p4f20, and p4f21 were visually inspected and copied into the report
figure directory with hashes matching their registered outputs.

## Paused A1000 continuation

The report/PDF integration is pushed at
`3b041735d8b842de46a9c0a474fccd0c44e0841a`. The frozen router verified the
registered functional-result hash, resolved Branch B, and launched:

```bash
bash interpretability/jspace_phase4/run_qwen_continuation_fit.sh draw_a_n1000
```

The producer recovered exact n=500 and banked atomic three-prompt
checkpoints through the intentional handoff boundary:

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

1. Mount the same Drive root and check out/pull clean branch
   `interp_jspace_part2` in `/content/labs`.
2. Read this file and `/content/drive/MyDrive/interpret/inprogress.md`.
3. Verify `checkpoint_state.json` reports `n_done=554`, then rehash
   `recovery/fit.ckpt` to the checkpoint SHA above. Check that no fitter,
   router, lock holder, or GPU process already owns the run.
4. Provide the frozen runtime and Qwen cache (default
   `HF_HUB_CACHE=/content/hf_local`) for exact model revision
   `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
5. From the clean repository run:

```bash
bash interpretability/jspace_phase4/run_qwen_frozen_branch_followup.sh
```

The router must reverify the registered functional-result hash, resolve
Branch B, and print `resuming from checkpoint: 554/...`. Do not launch the
lower-level fitter in parallel, delete the recovery pair, or register n=554
as a partial pseudo-milestone.

## Drive durability audit

The 23:25 UTC whole-registry pass checked 49 live events and 220 output
references. It found no wrong hashes, but three paths from the older
`p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1` event were
absent. The published reconstruction was restored byte-for-byte from the
hash-verified A250--A500 backup and now matches registered SHA-256
`ebe0c641115b40db089412fb969e8c37eef90a7992a40996c4277d9dac43d703`.

Thus 218/220 live outputs are hash-accounted. These two older registered paths
remain missing and keep whole-registry verification red:

```text
state.json
expected 361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8

capacity_reconstructions_a120.pt
expected 6b0399df2c57158e7fdad24274e50f8c1058021d233412afdcc5177f6c651b6f
```

Both belong under the registered A120--A250 functional-gate result directory.
No preserved exact copy was found on this VM. Do not fabricate bytes, edit the
registry, or rerun model-scale work on CPU. Recover exact bytes from backup or
version history if available; otherwise use a reviewed append-only correction
plan, then rerun `jspace-phase4 verify`.

The successor branch now contains an exact-hash-gated GPU recovery tool for
the A120 capacity artifact. It verifies the historical event, config, inputs,
dependencies, selected capacity functions, runtime, and target hash; run it
only after A1000 releases the GPU. The missing historical `state.json` has
irrecoverable timing fields and must come from an exact backup or an
independently/PI-approved append-only archival correction. Do not synthesize
it.

Preparation commit `2e602d7` records the exhaustive exact-copy search. The
live Drive target folder has thirteen children but no target `state.json`
cloud ID; sixty exact-name objects and 1,246 revisions expose no target. The
quarantined pre-incident DriveFS metadata has the same folder/children but no
state child, tombstone, deleted-item, or pending operation. No bytes were
restored, and external governance remains required.

## Other remaining work

- Complete A1000 and the frozen structural, functional, selection-margin,
  prompt-323, and canonical-decision queue.
- Import the already decisive OLMo early bundle on mainline after the A1000
  branch resolves. P4-P3 is blocked at 16/20; no Bank-W intervention is
  authorized.
- If and only if Q-L1/Q-L2 selects a canonical lens, bind it into the P4-P2
  and Bank-B configs. Their consumed-development interventions still require
  separately registered independent reviews; Q-L3/Q-L4/Q-L5 blocks them.
- Candidate 0.11 makes P4-P1 estimation-only. The addendum-required one-shot
  orthogonal assay is prepared but unrun and cannot restore the primary.
- Recover the two missing older A120--A250 functional-gate outputs above,
  confirm Drive cloud durability, and rerun whole-registry verification.
- Obtain independent protocol review and PI sign-off.

## Hard governance boundary

- No confirmatory/replication intervention outcomes.
- No self-signing reviewer or PI fields.
- Registered evidence is immutable; corrections append new events.
- Model-scale CPU fallback is forbidden.
- Phase 4 is not frozen until review, sign-off, freeze commit, and freeze tag
  all exist.
