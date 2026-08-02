# Gemma transport workstream — restart bootstrap

This is the short, stable bootstrap for the isolated Gemma transport
workstream. The canonical live state is in:

```text
/content/drive/MyDrive/interpret/gemma_transport_inprogress.md
```

Read these documents in order on a replacement VM:

1. `/content/drive/MyDrive/interpret/special_lab_resume.md` for shared J-space
   environment and conduct rules.
2. `/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_gemma_1.md` in
   full.
3. `/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_gemma_1_addendum.md`.
4. `/content/drive/MyDrive/interpret/gemma_transport_inprogress.md` for the
   current checkpoint, process ownership, exact next commands, and evidence
   queue.

## Fixed identity and isolation contract

```text
branch:        interp_jspace_gemma_transport
fork commit:   3b041735d8b842de46a9c0a474fccd0c44e0841a
package:       /content/labs/interpretability/jspace_gemma/
Drive root:    /content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802/
registry:      interpretability/jspace_gemma/reports/evidence_events.jsonl
evidence IDs:  gm-...
tier:          development/methods only
```

Never write to `interpretability/jspace_phase4/`,
`interpretability/jspace_olmo_lineage/`, either side track's registry, or the
Phase 4/OLMo Drive roots. Imports are read-only and hash-pinned. The Gemma
branch is merged into `interp_jspace_part2` only after the state-of-record and
Phase 4 import bundle are complete.

## Minimal replacement-VM bootstrap

Use the shared resume for secrets and full setup, then:

```bash
cd /content/labs
git fetch origin --prune
git switch interp_jspace_gemma_transport
git pull --ff-only origin interp_jspace_gemma_transport

test -d /content/jacobian-lens || \
  git clone https://github.com/anthropics/jacobian-lens.git /content/jacobian-lens
git -C /content/jacobian-lens checkout 581d3986
pip install -e /content/jacobian-lens
pip install -e interpretability/jspace_gemma

nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "HARD STOP: CUDA is not visible"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
print("CUDA smoke PASS", x.device)
PY

cat /content/drive/MyDrive/interpret/gemma_transport_inprogress.md
```

Before launching any producer, confirm the worktree is clean, no prior
producer owns the recorded lock, the model snapshot and config hashes match
the live handoff, and the output path is either absent or a valid resumable
checkpoint. Never run model-scale work on CPU.

## Terminal state

The scientific path stopped at the registered actual-Gemma backend-parity
gate. `gm-state-of-record-v1` is the terminal methods-only release from clean
producer `b80004843a5bbe57536e4da18297f7c52cf201a3`; it opens no Phase-4 model
cell and licenses no mechanism or workspace conclusion. G2--G8 must not be
resumed on this branch. The canonical import bundle is Drive
`gemma_transport_20260802/release/IMPORT_BUNDLE_PHASE4.json`.

Before every push, fetch and pull/rebase the Gemma branch, re-run the relevant
tests, reconcile any concurrent Gemma-branch commits, then push. Once the live
handoff records the release registry/report commit, this completed blocker
fork is eligible for the requested ancestry-preserving Part-2 merge.
