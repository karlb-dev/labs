# LIVE — Phase 4.2 block 2, VM12

Last updated: 2026-08-01 02:52 UTC. This is the canonical dynamic handoff.
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

## Live GPU job

The Qwen draw-A nested fit to n=250 is active. At this ledger boundary the
durable Drive state is n=228, checkpoint SHA-256
`e65dd79b377fa46561bb27c67764d74c464b2717ae07f01a35637ba7ebef066f`,
6,606,047,399 bytes, and the process is computing chunk 228:231. Throughput
is about 180.16 seconds per new prompt including atomic sync; peak allocated
VRAM is 62,832,854,016 bytes. The expected finish is about one hour after
this update.

Fit log:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_fit_drawA_n250_20260801.log
```

Checkpoint state and watchdog log:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/WATCHDOG_PHASE4.log
```

Exact host launch (rerun safely to resume after a reclaim; first restore or
hard-link the latest verified Drive checkpoint to the producer's local
contract directory if local NVMe was lost):

```bash
cd /content/labs
env \
  JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731 \
  JSPACE4_LOCAL_WORK=/content/sl4_work \
  HF_HUB_CACHE=/content/hf_local \
  MPLCONFIGDIR=/tmp/matplotlib-phase4 \
  CUDA_VISIBLE_DEVICES=0 \
  script -q -a -c \
  "python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
    --config interpretability/jspace_phase4/configs/p4_qwen_nested_lens_fit_dev.yaml \
    --draw draw_a --stop-at 250" \
  /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_fit_drawA_n250_20260801.log
```

Watchdog:

```bash
bash interpretability/jspace_phase4/watchdog_phase4.sh \
  /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/checkpoint_state.json \
  250 \
  /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/WATCHDOG_PHASE4.log
```

The producer must finish from a clean tree so it can create
`p4-qwen-lens-fit-drawA-n250-dev-v1`. Do not leave edits uncommitted near
n=250.

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
- Config still contains `PENDING_REGISTERED_N250_LENS_SHA256`; fill it only
  from the registered n=250 event, then commit before execution.

### Bank B

- Registered `p4-bank-b-candidate-v1`: 40 families, 160 facts, four facts per
  family, 160 unique true bridges, two second-hop relations and two
  counterfactual bridges per fact, exact Qwen token IDs, split 20/10/10.
- Prior overlap and cross-partition entity overlap are zero.
- **Freeze blocker:** all 160 source rows are pinned single-source candidate
  records and still require independent verification. Do not call Bank B
  freeze-ready and do not expose untouched outcomes.

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
- Candidate preregistration 0.2 now contains P4-P1's intersection-union
  endpoint and P4-P3's shared-family max-T endpoint.

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
- **Remaining mode blocker:** model-backed development parser/correctness
  gate, item families, SESOI/power, and independent review.

## Immediate queue after n=250

1. Confirm the fit created `p4-qwen-lens-fit-drawA-n250-dev-v1`, verify every
   output hash, and record the registered lens SHA-256.
2. Replace `PENDING_REGISTERED_N250_LENS_SHA256` in both:
   - `configs/p4_qwen_lens_convergence_drawA_dev.yaml`
   - `configs/p4_qwen_multilens_functional_gate_dev.yaml`
   Commit and push that manifest-only boundary.
3. Run convergence on GPU:

   ```bash
   env JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731 \
     JSPACE4_LOCAL_WORK=/content/sl4_work HF_HUB_CACHE=/content/hf_local \
     MPLCONFIGDIR=/tmp/matplotlib-phase4 CUDA_VISIBLE_DEVICES=0 \
     python -m jspace_phase4.experiments.p4_qwen_lens_convergence \
       --config interpretability/jspace_phase4/configs/p4_qwen_lens_convergence_drawA_dev.yaml
   ```

4. Run prompt-112 influence on GPU:

   ```bash
   env JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731 \
     JSPACE4_LOCAL_WORK=/content/sl4_work HF_HUB_CACHE=/content/hf_local \
     MPLCONFIGDIR=/tmp/matplotlib-phase4 CUDA_VISIBLE_DEVICES=0 \
     python -m jspace_phase4.experiments.p4_qwen_lens_influence \
       --config interpretability/jspace_phase4/configs/p4_qwen_lens_influence_prompt112_dev.yaml
   ```

5. Run the model-backed multi-lens functional gate:

   ```bash
   env JSPACE4_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731 \
     JSPACE4_LOCAL_WORK=/content/sl4_work HF_HUB_CACHE=/content/hf_local \
     MPLCONFIGDIR=/tmp/matplotlib-phase4 CUDA_VISIBLE_DEVICES=0 \
     python -m jspace_phase4.experiments.p4_qwen_multilens_functional_gate \
       --config interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_dev.yaml
   ```

6. Execute the runner's frozen branch without reinterpretation:
   - branch A/C: draw-B n=120;
   - branch B or any load-bearing functional failure: continue draw-A n=500;
   - structural-only failure with every functional/capacity endpoint inside
     SESOI: retain the functionally equivalent canonical lens and label the
     structural nonconvergence.
7. Keep the GPU occupied with the selected continuation, then finish the
   model-backed mode development gate and any Phase 4 freeze blockers that
   fit before reclaim. Never start confirmatory/replication outcomes.

## Reporting and verification queue

- Add Bank B/W/mode protocol and p4f14 to
  `reports/PHASE4_DEVELOPMENT_REPORT.md` and
  `reports/handout/jspace_phase4_development.tex`.
- Rebuild and visually inspect the PDF after n=250, convergence/influence,
  functional gate, and final branch decision.
- Run `bash interpretability/jspace_phase4/repro.sh` and require all tests and
  `python -m jspace_phase4 verify` to pass.
- Refresh this ledger and the static resume mirror, archive the previous Drive
  copy, then copy byte-identical versions to Drive.
- Finish only at a truthful Phase 4 freeze point or VM reclaim boundary. PI
  and independent-review fields must never be self-signed by the agent.

## Permission/session note

This active Codex session was launched with a restricted profile even though
the UI showed Full Access; that profile cannot be hot-reloaded. Long GPU and
watchdog processes run independently, and scoped command approvals were
persisted. `/root/.codex/config.toml` was set to `approval_policy="never"`
and `sandbox_mode="danger-full-access"` for a future Codex launch, but a new
VM must recreate that file before starting the session. If Full Access still
does not propagate, relaunch the agent rather than leaving a long job paused
at an approval prompt.
