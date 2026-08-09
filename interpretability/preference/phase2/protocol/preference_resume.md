# preference_resume.md — portable bootstrap for the Lab 38 preference campaign

**Paste this to the agent on any new VM and you're done:**

> Read `/content/drive/MyDrive/preference/preference_resume.md` and resume
> the Lab 38 preference campaign. Static setup is in that file; the live
> state, queue, and exact next commands are in `inprogress.md` next to it.

This file is STATIC (how any VM gets from zero to working). DYNAMIC state
lives in `inprogress.md` (same folder, agent-maintained, newest entry
first). Update this file only when the *setup* changes, not when the
science moves. Last static revision: 2026-08-07 (campaign opened).

## 1 · What this campaign is (one paragraph)

Lab 38 Phase 1 asks whether an open chat model (primary: pinned OLMo 7B
Instruct) shows content-tracking revealed-preference asymmetries under
fully counterbalanced, action-binding forced-choice menus — and, only for
scenarios that graduate a strict behavioral gate, whether a scenario-
specific residual "choice-margin" direction causally transfers to a matched
report-only channel. Claim ceiling is functional coupling only (never
wants/welfare/consent/experience). The governing documents are, in
precedence order: `preference_1_1_addendum.md` > `preference_1_1.md` >
`lab38_revealed_preference_report_channel.md` (all in this Drive folder;
hash-pinned copies in the repo under `interpretability/preference/plans/`).
Preferred outcome: a clean instrument, even if every result is null. The
single human approval gate is the pre-battery freeze review (addendum §I);
stop-and-ask conditions are addendum §M.

## 2 · Ten-minute VM bootstrap (copy-paste, in order)

```bash
# 0. sanity
nvidia-smi; df -h /; ls /content/drive/MyDrive/preference || echo "MOUNT DRIVE FIRST"

# 0a. GPU HARD GATE for model-scale stages (dev pilot, frozen battery,
# mechanism). Run in the SAME execution context that will launch model jobs.
# CPU is only for: bank/tests/hashing/analysis/plots/TeX and the SmolLM2
# Tier-A smoke. Never silently fall back to CPU for 7B/32B work.
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import sys, torch
print("torch", torch.__version__, "CUDA", torch.version.cuda,
      "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("HARD STOP: GPU not visible; relaunch with host GPU access")
print("GPU", torch.cuda.get_device_name(0))
PY

# 1. secrets (SSH key tarball lives at /content/key.tar.gz on Colab VMs;
#    a Drive copy is at MyDrive/interpret/misc/key.tar.gz)
cd /content && tar xzf key.tar.gz && mkdir -p ~/.ssh \
  && cp key/id_ed25519 key/id_ed25519.pub ~/.ssh/ \
  && chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519
git config --global user.name "karlb-dev"
git config --global user.email "kburtram@live.com"
ssh -o StrictHostKeyChecking=accept-new -T git@github.com   # expect "Hi karlb-dev!"
# HF token (public models still throttle unauthenticated):
[ -f /content/drive/MyDrive/interpret/misc/hf_token.txt ] && \
  mkdir -p ~/.cache/huggingface && \
  cp /content/drive/MyDrive/interpret/misc/hf_token.txt ~/.cache/huggingface/token
python -c "from huggingface_hub import whoami; print('HF auth:', whoami()['name'])"

# 2. repo + branch
cd /content && { [ -d labs ] || git clone git@github.com:karlb-dev/labs.git; } \
  && cd labs && git checkout interp_preference_phase1 && git pull \
  && git log --oneline -10          # recent history = what happened lately

# 3. package install + no-model tests (fast; proves the instrument)
pip install -e interpretability/preference/phase1[dev] 2>/dev/null \
  || pip install -e interpretability/preference/phase1
python -m pytest interpretability/preference/phase1/tests -q

# 4. TeX toolchain (only needed at reporting boundaries; apt update FIRST —
#    stale-URL 404s otherwise; a few unrelated 404s scroll by harmlessly)
apt-get update -qq && apt-get install -y -qq texlive-latex-base \
  texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra

# 5. NOW read the dynamic state and do what it says:
#    /content/drive/MyDrive/preference/inprogress.md
```

**Model weights rule (inherited from the jspaces campaign, still true):**
hub-download to local NVMe (`HF_HUB_CACHE=/content/hf_local`), never
model-LOAD through DriveFS. 7B re-downloads in minutes; delete local
weights between model sets. If a hub download stalls mid-file with the
token installed, kill it and rsync any partial snapshot from
`MyDrive/hf_cache/hub` to local disk with `--size-only`, then verify every
shard in `model.safetensors.index.json` exists.

**Colab agent-sandbox note:** a restricted sandbox can hide the GPU while
the host GPU is healthy. A sandboxed CUDA failure means "relaunch that
command with host GPU access", never "continue on CPU".

## 3 · Where everything lives

| what | where |
|---|---|
| DYNAMIC state (read after bootstrap) | `MyDrive/preference/inprogress.md` |
| Governing plan + addendum (authoritative copies) | repo `interpretability/preference/plans/`; originals in this Drive folder |
| Campaign home in repo | `interpretability/preference/` (README maps the tree) |
| Phase 1 package + tests | `interpretability/preference/phase1/` |
| Evidence registry (append-only) | `interpretability/preference/phase1/reports/evidence_events.jsonl` |
| Intake record (hashes, missing inputs, departures) | `interpretability/preference/phase1/SOURCE_INTAKE.md` |
| Live run dirs (gitignored) | `/content/labs/interpretability/runs/lab38_*` |
| Drive mirror of runs/reports (per phase/part) | `MyDrive/preference/phase1/part1/` |
| TeX/PDF development handout | repo `interpretability/preference/phase1/reports/handout/` (+ PDF mirrored to Drive part folder) |
| Branch | `interp_preference_phase1` (from `main` @ `78b58f3`) |

## 4 · Conduct rules (digest; full text = plan §1.6, §2.3, addendum §C/§I/§M)

- Commit + push at every evidence boundary and ≥ every 10 min of model-run
  progress (per-item JSONL is small; push satisfies the ≤10-min-loss rule).
- Every long run: immutable per-item JSONL, atomic resume cursor,
  same-command resume, refuse resume on config/bank/model hash mismatch.
- Registry: append-only; supersede, never edit.
- Claim ceiling verbatim everywhere, including commit messages.
- Development vs frozen vocabulary mandatory (`scientific_tier` on every
  event). Nothing model-derived is interpreted before its gate passes.
- The freeze review is the ONLY human gate; addendum §M lists the
  stop-and-ask conditions. A behavioral null is a publishable success.
- Never: generate from DG-SAFE prompts, ablate refusal, pool a universal
  preference direction, or let disengagement displace the primary assay.
