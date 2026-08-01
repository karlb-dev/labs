# LIVE — Phase 4.2 block 2, VM12

Last updated: 2026-08-01 04:00 UTC. This is the canonical dynamic handoff.
The older Drive ledger was archived before this file replaced
`/content/drive/MyDrive/interpret/inprogress.md`.

## Non-negotiable boundary

- Repository: `/content/labs`
- Branch: `interp_jspace_part2`; use the remote tip, then inspect
  `git log --oneline -25` for the exact current commit.
- Phase 4 run root:
  `/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731`
- Governing review:
  `interpretability/jspace_phase4/reviews/jspace_lab_nextsteps_4_2.md`
  and its accepted addendum.
- Candidate preregistration:
  `interpretability/jspace_phase4/preregistration/SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md`.
- Phase 4 is still **development-only**. Confirmatory and replication
  outcomes remain forbidden until independent review, PI sign-off, a freeze
  commit, and a freeze tag.
- Model-scale work is GPU-only. A sandboxed `torch.cuda.is_available()==False`
  is not a CPU-fallback license; launch the exact command in host context.
- Every result boundary must be clean-tree, registered, committed, pushed,
  and mirrored to Drive. Never overwrite registered evidence; supersede it.

## Restart-safe GPU boundary

The Qwen draw-A nested fit finished normally at n=250. The scientific producer
exited, and the post-fit queue was deliberately not launched so Codex can be
restarted with a real Full Access profile. The canonical restart instructions
are in `interpretability/jspace_phase4/reports/RESUME_PHASE4_2.md` and the
byte-identical `/content/resume-phase-4-2.md` and Drive mirror.

Registered evidence: `p4-qwen-lens-fit-drawA-n250-dev-v1`. The final recovery
checkpoint SHA-256 is
`723844116788c73800e219c048c165d904873e0d7676ab4739d7763a605166d6`;
the materialized lens SHA-256 is
`b78427c84ddd3b9f7f4361b952b5169cf49335e77fe364e584b6abd799f79006`.
All three outputs were independently rehashed and matched. The registry event
was pushed at `829d82b`; both downstream configs were hash-bound, 20 targeted
tests passed, and the manifest-only boundary was pushed at `eb91a0b`.

Fit log:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_fit_drawA_n250_20260801.log
```

Checkpoint state and completed-fit watchdog log:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/WATCHDOG_PHASE4.log
```

Do not rerun the completed fit. In the replacement session, verify the GPU and
immediately launch `bash interpretability/jspace_phase4/run_qwen_postfit_queue.sh`
from a clean `eb91a0b`-or-later tree. The queue has its own Drive heartbeat.

## Completed during Phase 4.2 block 2

### OLMo common support

- Registered `p4-lineage-common-cohort-analysis-olmo-dev-v1`.
- All-four intersection: 79 facts / 25 families. Adjacent pairs: 92/29,
  119/34, and 97/29.
- Development-only Figure p4f11 and its report/TeX/PDF boundary are complete.

### Qwen functional gate preparation

- Registered leakage-safe fixed subset
  `p4-qwen-multilens-functional-subset-dev-v1`.
- Manifest file SHA-256:
  `bae4b0c0013513067da8b5ac2d1a6c37e1f0a0e6b1cf3c4c8129789075443d3e`;
  payload SHA-256:
  `fb227be17dcad06eafd93fc75bddba05cb63c1134f30c99b31da8dcf3d531564`.
- Frozen subset: 30 paired facts / 25 families, 20 bridge items / 20
  families, 40 prose prompts, 16 capacity prompts, and exact G4 n=60.
- Resumable multi-lens runner and tests are committed. It evaluates in frozen
  order published -> A120 -> A250 and checkpoints raw rows every five items.
- Both configs contain the exact registered n=250 lens hash at `eb91a0b`.
- The durable post-fit entrypoint
  `interpretability/jspace_phase4/run_qwen_postfit_queue.sh` is committed and
  has a persisted host/GPU approval. It runs convergence, prompt-112
  influence, the functional gate, and the model-backed mode gate. Between
  stages it permits exactly one dirty file (the evidence registry), commits
  and pushes that event, and refuses any unexpected tree change.

### Bank B

- Registered `p4-bank-b-candidate-v1`: 40 families, 160 facts, four facts per
  family, 160 unique true bridges, two second-hop relations and two
  counterfactual bridges per fact, exact Qwen token IDs, split 20/10/10.
- Prior overlap and cross-partition entity overlap are zero.
- Registered independent audit
  `p4-bank-b-restcountries-verification-dev-v1`: 121 exact/unambiguous rows,
  21 independent matches requiring manual ambiguity review, and 18
  independent-source mismatches. Figure p4f16 is registered in PNG/PDF.
- **Freeze blocker:** prospectively version and correct the root discrepancies,
  reverify the entire candidate, resolve multi-valued/manual ambiguities, and
  rerun power. Never edit the registered audit output or call Bank B
  freeze-ready yet.

### Bank W and max-T

- `p4-bank-w-candidate-v1` was superseded by registered
  `p4-bank-w-candidate-v2` after outcome-blind power calibration raised the
  common-support floor from 16 to 20. Row and partition hashes are unchanged.
- Bank W contains 72 template families, eight seeds/family, and 4,608 fully
  crossed load (2/6) x derivation x redundancy rows; split 24/24/24.
- Row payload SHA-256:
  `078e7726027ea61a09040b9030344c2c28868593695e650b66df527fefa8ab49`.
  Partition payload SHA-256:
  `361acad0f38f0662d8f7cb2648689915dcefdce5f2f778f4067fcf348a784238`.
- Real Qwen-tokenizer length span is exactly zero. Answer leakage is zero.
  First/last/frequency heuristics equal chance and target positions are exact.
- Registered `p4-bank-w-power-dev-v1`; type I is 0.040--0.050. Frozen SESOI
  is 0.10 nat/doubling = 0.15849625 nat for load 2 -> 6. Conservative
  minimum power is 0.703/0.806/0.858 at 16/20/24 common families. Figure
  p4f14 is committed in PNG and PDF.
- Candidate preregistration 0.3 now contains P4-P1's intersection-union,
  P4-P3's shared-family max-T, and P4-P2's exact common-support interaction
  and directional decision rule. It remains a candidate, not a freeze.

### Official Qwen mode parser

- Registered methods gate `p4-qwen-mode-parser-gate-dev-v1`.
- Exact chat-template SHA-256:
  `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`.
- Parser v2 correctly recognizes that official thinking-on opens `<think>`
  in prefill while thinking-off closes an empty block in prefill.
- Structural discovery: thinking-off x generated-reasoning does not exist.
  The frozen common-support primary interaction therefore uses prefill and
  final-answer phases; thinking-on reasoning-only is secondary.
- Truncation, EOS-inside-reasoning, parse failure, answer omission, and an
  answer before closure remain separate outcomes. Four rationale controls
  are exactly token matched in the golden.
- The model-backed development producer/config are committed as
  `p4-qwen-mode-gate-dev-v1`. CPU preflight resolves an outcome-blind paired
  subset of exactly 20 consumed Phase 3 facts / 20 canonical families; eight
  focused mode tests and the full Phase 4 suite (120 tests) pass. The producer
  checkpoints every real completion and is queued after the multi-lens gate.
- **Remaining mode blocker:** run that model-backed parser/correctness gate,
  then author prospective untouched families plus the SESOI/power ruler and
  obtain independent review.

## Immediate queue in the replacement session

1. Confirm the replacement session has host GPU visibility, the worktree is
   clean, and the remote contains `eb91a0b`.
2. Confirm no queue process is already alive, then launch the heartbeat-logged
   host GPU queue:

   ```bash
   bash interpretability/jspace_phase4/run_qwen_postfit_queue.sh
   ```

   It runs and banks convergence -> prompt-112 influence -> multi-lens
   functional gate -> model-backed official-mode gate. Durable stdout and
   five-minute GPU heartbeats are written at the Phase 4 run root.
3. Execute the functional runner's frozen branch without reinterpretation:
   - branch A/C: draw-B n=120;
   - branch B or any load-bearing functional failure: continue draw-A n=500;
   - structural-only failure with every functional/capacity endpoint inside
     SESOI: retain the functionally equivalent canonical lens and label the
     structural nonconvergence.
4. Keep the GPU occupied with the selected continuation and any remaining
   Phase 4 development/freeze blockers that fit before reclaim. Never start
   confirmatory/replication outcomes.

## Reporting and verification queue

- Bank B/W/mode protocol and p4f14 are integrated into the development MD,
  TeX, and visually inspected 13-page PDF at commit `139b512`; byte-identical
  TeX/PDF/PNG/PDF copies are on Drive.
- Rebuild and visually inspect the PDF after n=250, convergence/influence,
  functional gate, and final branch decision.
- Run `bash interpretability/jspace_phase4/repro.sh` and require all tests and
  `python -m jspace_phase4 verify` to pass.
- Refresh this ledger and the static resume mirror, archive the previous Drive
  copy, then copy byte-identical versions to Drive.
- Finish only at a truthful Phase 4 freeze point or VM reclaim boundary. PI
  and independent-review fields must never be self-signed by the agent.

## Permission/session note

This active Codex session was launched with a managed `workspace-write`
profile even though Full Access was enabled in the UI. Official Codex guidance
states that enabling a mode only makes it available; it does not select it or
change an existing chat, and managed/launch-time policy can override user
config. Long GPU/watchdog processes run independently and scoped queue
approval is persisted. `/root/.codex/config.toml` was set to
`approval_policy="never"` and `sandbox_mode="danger-full-access"` as a future
launch default, but the new session must still be launched with Full Access
and may remain constrained by an injected managed profile.
