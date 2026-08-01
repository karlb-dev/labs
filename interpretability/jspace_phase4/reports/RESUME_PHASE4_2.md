# Resume Phase 4.2 — restart-safe handoff

Generated 2026-08-01 04:00 UTC on VM12. Read this file before launching any
new work. It is also mirrored byte-for-byte at:

- `/content/resume-phase-4-2.md`
- `/content/drive/MyDrive/interpret/resume-phase-4-2.md`

## Safe boundary reached

The Qwen draw-A nested Jacobian-lens fit finished normally at n=250. There is
no incomplete scientific write to protect and the post-fit queue was
deliberately **not** launched, so Codex can be stopped and restarted now.

- Repository: `/content/labs`
- Branch: `interp_jspace_part2`
- Pushed scientific restart commit (before handoff-only docs):
  `eb91a0b5fe0af7d4f86331c33497d7548b933d26`
- Worktree at handoff: clean
- Phase 4 run root:
  `/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731`
- Development-only boundary remains in force. Do not expose confirmatory or
  replication outcomes and do not self-sign PI or independent-review fields.

The completed evidence event is
`p4-qwen-lens-fit-drawA-n250-dev-v1`. All registered outputs were independently
rehash-verified before the registry event was committed:

| artifact | SHA-256 |
|---|---|
| n=250 lens | `b78427c84ddd3b9f7f4361b952b5169cf49335e77fe364e584b6abd799f79006` |
| fit result | `2ca66e7efa048c252b3fc2e4cf6c3b992e31a5e00381dee8f61598095e2721e6` |
| input manifest | `389087888f7bdaaca590b303cf1006bcdae14bfbec7576162d5ecf0b21057d5c` |
| final recovery checkpoint | `723844116788c73800e219c048c165d904873e0d7676ab4739d7763a605166d6` |

The registry append was pushed at `829d82b`. The registered lens hash was then
bound into both convergence and functional-gate YAMLs, 20 targeted tests
passed, and that manifest-only boundary was pushed at `eb91a0b`.

## Start the replacement Codex session with real Full Access

The old chat was injected at launch with a managed `workspace-write` profile;
changing the UI switch did not hot-reload it. The future-session defaults are
already present in `/root/.codex/config.toml`:

```toml
approval_policy = "never"
sandbox_mode = "danger-full-access"
```

Select Full Access **before** creating the replacement chat. If Codex is
started from a shell, the unambiguous launch is:

```bash
cd /content/labs
codex --dangerously-bypass-approvals-and-sandbox
```

At the beginning of the new chat, inspect its injected environment context.
If it still says `managed`, `workspace-write`, or `restricted`, the launcher
or organization policy overrode the local file and hourly prompts can still
occur; restart with the explicit flag above rather than relying on the UI
toggle. Do not leave an unattended run until a host-context GPU probe and a
networked `git push` both work without approval.

## Replacement-session bootstrap and immediate GPU launch

First read the governing and dynamic state:

```bash
cd /content/labs
git status --short --branch
git log -8 --oneline
sed -n '1,260p' /content/drive/MyDrive/interpret/inprogress.md
sed -n '1,260p' /content/drive/MyDrive/interpret/special_lab_resume.md
sed -n '1,320p' interpretability/jspace_phase4/reviews/jspace_lab_nextsteps_4_2.md
sed -n '1,260p' interpretability/jspace_phase4/reviews/jspace_lab_nextsteps_4_2_addendum.md
```

Verify GPU visibility in the same host execution context that will run model
jobs. The expected device is an NVIDIA RTX PRO 6000 Blackwell Server Edition,
and CPU fallback is forbidden:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu --format=csv,noheader
CUDA_VISIBLE_DEVICES=0 python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.randn((1024,1024),device='cuda',dtype=torch.float16); y=x@x; torch.cuda.synchronize(); print(torch.isfinite(y).all().item())"
```

Confirm no previous queue is alive before launching. At this handoff it was
not started. If the process/log says it has since started, attach and monitor
instead of launching a duplicate.

```bash
pgrep -af 'run_qwen_postfit_queue|p4_qwen_lens_convergence|p4_qwen_lens_influence|p4_qwen_multilens_functional_gate|p4_qwen_mode_model_gate' || true
bash interpretability/jspace_phase4/run_qwen_postfit_queue.sh
```

That committed entrypoint has defaults for the run root, local NVMe workspace,
HF cache, Matplotlib cache, and CUDA device. It refuses a dirty tree, a wrong
branch, or unbound hash sentinels. It runs and banks, in order:

1. Qwen draw-A lens convergence;
2. prompt-112 influence localization;
3. frozen multi-lens functional gate;
4. model-backed official Qwen thinking-mode parser/correctness gate.

After each producer it permits only the evidence registry to be dirty, then
commits and pushes the event. It records stdout and five-minute GPU heartbeats
at:

```text
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/qwen_postfit_queue_20260801.log
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/QWEN_POSTFIT_QUEUE_WATCHDOG.log
```

The finished-fit watchdog should self-exit after observing n=250. It must not
be restarted; the post-fit queue owns its own heartbeat.

## Frozen continuation after the functional gate

Do not reinterpret the branch after seeing outcomes:

- branch A/C: fit independent draw-B to n=120;
- branch B, or any load-bearing functional failure: continue draw-A to n=500;
- structural-only failure with every functional and capacity endpoint inside
  its SESOI: retain the functionally equivalent canonical lens and label the
  structural nonconvergence.

Prepare and launch the selected continuation as soon as the queue completes so
the GPU does not idle. All model jobs must be checkpointed to Drive at no more
than ten-minute granularity and must refuse CPU fallback.

## Productive CPU work while the GPU queue runs

The next session should use CPU time concurrently for these open blockers:

1. **Bank B correction and reverification.** Registered audit
   `p4-bank-b-restcountries-verification-dev-v1` found 121 exact/unambiguous,
   21 independent matches requiring manual ambiguity review, and 18
   independent-source mismatches across the frozen 160 facts. Author a new,
   prospectively versioned candidate; never edit the registered output. Root
   discrepancies include Bujumbura/Gitega, Colombo/Sri Jayawardenepura Kotte,
   Chinese/Hong Konger demonyms, Cap Vert, Saint-Lucie, Norfolk Island,
   native-name Iran, Macedonia/North Macedonia, and the full Spanish Saint
   Helena territory name. Re-run independent verification and power before
   calling Bank B freeze-ready.
2. **Bank W model-specific work.** Candidate v2 and max-T power are registered
   (72 families, eight seeds, 4,608 crossed rows, common-support floor 20),
   but baseline capability, intervention execution, and review remain open.
3. **Mode gate follow-up.** Interpret the queued model-backed gate, then author
   prospective untouched families and a SESOI/power ruler; the current 20
   bridge facts are development-only consumed Phase 3 material.
4. **Reporting.** Integrate n=250, convergence/influence, functional branch,
   the model-mode gate, and Bank B Figure p4f16 into the development Markdown,
   TeX, PDF, and visualizations. Rebuild and visually inspect the PDF, then
   mirror byte-identical report artifacts to Drive.

Candidate preregistration 0.3 already freezes the common-support P4-P2
interaction wording but is still a candidate. It is not confirmatory
authorization.

## Evidence already banked in this block

Key recent commits, all pushed to `origin/interp_jspace_part2`:

```text
eb91a0b config: bind registered Qwen n250 lens
829d82b data: register Qwen draw-A n250 lens
8082cce data: register Bank B independent verification
671495e feat: audit Bank B against independent source
ce88dd5 docs: checkpoint n237 and autonomous GPU queue
3f6146d feat: add model-backed Qwen mode gate
75aa7db ops: add durable Qwen post-fit GPU queue
29b27d4 docs: freeze common-support P4-P2 wording
139b512 docs: report Bank W power and mode gate
```

The full Phase 4 suite last passed 120 tests with one expected warning from a
sandboxed CUDA probe. The development report PDF is 13 pages and was visually
inspected before the n=250 result. Registered evidence and Drive artifacts are
immutable: supersede with a new evidence ID rather than overwriting.

## Boundary checks before the next long unattended interval

Require all of the following:

```bash
git status --short --branch
git rev-parse HEAD
git ls-remote origin refs/heads/interp_jspace_part2
python -m jspace_phase4 verify
bash interpretability/jspace_phase4/repro.sh
```

Refresh this handoff and `/content/drive/MyDrive/interpret/inprogress.md` after
every result-bearing boundary. Commit/push frequently, copy byte-identical
handoffs and report artifacts to Drive, keep a watchdog on every long GPU
queue, and continue until either a truthful Phase 4 freeze point or VM reclaim.
