# special_lab_resume.md — portable bootstrap for the J-space investigation

**Paste this to the agent on any new VM and you're done:**

> Read `/content/drive/MyDrive/interpret/special_lab_resume.md` and resume
> the J-space investigation. Static setup is in that file; the live state,
> queue, and next commands are in `inprogress.md` next to it. Follow the
> repro contract for anything new.

This file is STATIC (how any VM gets from zero to working). DYNAMIC state
lives in `inprogress.md` (same folder, agent-maintained: status, queue,
what's running, exact next commands) and in the plan files it points to.
Last static revision: 2026-07-27 (VM6, repair era). Update this file only
when the *setup* changes, not when the science moves.

---

## 1 · What this investigation is (one paragraph)

We are replicating and stress-testing Anthropic's July-2026 paper
"Verbalizable Representations Form a Global Workspace in Language Models"
(transformer-circuits.pub/2026/workspace) on open models with the
reference `jacobian-lens` package. **Part 1** (Lab 37, merged to `main`
via PR #9) was the exploratory campaign on `Olmo-3-32B-Think` +
`Qwen3.6-27B` — its headline claims are DOWNGRADED by a forensic review;
read `interpretability/jspace/REPORT_v2_ERRATA.md` before trusting any
Part-1 wording. **Part 2** (current) is the confirmatory campaign: first
repair the assay (Workstream R: the paper's output-protected dynamic
ablation, the paper's occupancy estimand, rank-safe projectors,
phase-resolved hooks, full-sequence scoring, paired clustered stats),
pass gates G0–G6, then run the OLMo lineage (base → SFT → DPO →
3.0/3.1-Think | 3.1-Instruct) as the primary study, Qwen thinking-on/off
as the mode contrast, Gemma-4-31B as the architecture leg. Evidence tiers
(exploratory/pilot/confirmatory) are mandatory vocabulary. Timebox: ≤200 h
total; VMs arrive in ~24 h GPU blocks; everything checkpoints ≤10 min.

## 2 · Ten-minute VM bootstrap (copy-paste, in order)

```bash
# 0. sanity
nvidia-smi; df -h /; ls /content/drive/MyDrive/interpret || echo "MOUNT DRIVE FIRST"

# 1. secrets
tar xzf /content/drive/MyDrive/interpret/misc/key.tar.gz -C ~/ && chmod 600 ~/.ssh/id_ed25519
git config --global user.name "karlb-dev"; git config --global user.email "kburtram@live.com"
ssh -o StrictHostKeyChecking=accept-new -T git@github.com   # expect "Hi karlb-dev!"
# HF token — pick whichever exists:
[ -f /content/drive/MyDrive/interpret/misc/hf_token.txt ] && \
  mkdir -p ~/.cache/huggingface && cp /content/drive/MyDrive/interpret/misc/hf_token.txt ~/.cache/huggingface/token
# otherwise run this in a NOTEBOOK cell (kernel env doesn't reach agent shells; the file does):
#   import os, pathlib; p = pathlib.Path("~/.cache/huggingface/token").expanduser()
#   p.parent.mkdir(parents=True, exist_ok=True); p.write_text(os.environ["HF_TOKEN"].strip())
python -c "from huggingface_hub import whoami; print('HF auth:', whoami()['name'])"

# 2. repo + branch (the campaign branch; `main` holds merged Part 1)
cd /content && git clone git@github.com:karlb-dev/labs.git && cd labs
git checkout interp_jspace_part2        # current campaign branch
git log --oneline -15                   # recent history = what happened lately

# 3. jlens engine, pinned + byte-verified
git clone https://github.com/anthropics/jacobian-lens.git /content/jacobian-lens
cd /content/jacobian-lens && git checkout 581d3986 && pip install -e . && cd /content/labs
diff -rq /content/jacobian-lens/jlens \
  /content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726/code/jacobian-lens-pkg/jlens  # expect silence

# 4. the Part-2 package: install + conformance tests + env audit, one command
bash interpretability/jspace_part2/repro.sh

# 5. exploratory-era live dir (only if inprogress.md says its jobs still matter)
[ -d interpretability/special_lab2 ] || cp -r \
  /content/drive/MyDrive/interpret/special-lab-1/part2_20260727/code/special_lab2 \
  interpretability/special_lab2

# 6. handout toolchain (only needed at reporting boundaries)
# NOTE (VM8): `apt-get install` WITHOUT a preceding `apt-get update` fails on
# this image with 404s on stale package URLs. Run update first; a few
# unrelated 404s (libidn12, ruby3.0) still scroll past and are harmless —
# texlive installs and pdflatex works.
apt-get update -qq && apt-get install -y -qq texlive-latex-base texlive-latex-recommended \
  texlive-fonts-recommended texlive-latex-extra

# 7. NOW read the dynamic state and do what it says:
#    /content/drive/MyDrive/interpret/inprogress.md
```

**Model weights rule:** hub-download to local NVMe (`/content/hf_local`
for HF-cache layout, or a plain dir under `/content/models/`) — measured
~320 MB/s with the token. NEVER stream a 32B model through DriveFS for a
load (it stalls under page-cache pressure), and NEVER delete DriveFS's
chunks.db (wedges the mount; recovery = kill the `--single_process`
`/opt/google/drive/drive` worker; supervisor respawns it). Delete local
weights between model sets; ~236 G disk total.

## 3 · Where everything lives

| what | where |
|---|---|
| DYNAMIC state (read after bootstrap) | `MyDrive/interpret/inprogress.md` |
| **THE CURRENT GOVERNING PLAN (VM8+)** | `special-lab-1/jspace_lab_nextsteps_2_2.md` (forensic review + stages N0–N8) **and** `..._addendum.md` §3 (PI decisions D1–D7, which GOVERN on conflict). Git mirror: `jspace_part2/reviews/` with `NEXTSTEPS_2_2_ACCEPTED.md` |
| **what the campaign is waiting on** | `jspace_part2/READY_FOR_FREEZE.md` (gate ledger) + `preregistration/SCIENTIFIC_PREREGISTRATION_CANDIDATE.md`; copies on Drive at `special-lab-1/` |
| static rules of conduct (the 12 standing directives) | `special-lab-1/experiment_reset_instructions.md` (copy synced to git mirror) |
| earlier governing review (Part-2 addendum, still in force where not superseded) | `special-lab-1/jspace_part2_plan1_addendum.md` (+ copies in `special_lab2/` and `jspace/part2/code/`) |
| repair preregistration (Workstream R + gates) | `special_lab2/REPAIR_PREREGISTRATION.md` (+ git mirror) |
| operative plan | `special_lab2/PLAN_PART2.md` — REVISION-1 header supersedes its own §0/§8 |
| Part-2 run dir (all outputs, per-model metrics, manifests) | `special-lab-1/part2_20260727/` |
| Part-1 run dirs (frozen) | `special-lab-1/{2026-07-25_1726, 2026-07-26_v2}/` |
| repro package (in git) | `labs/interpretability/jspace_part2/` — CLI `jspace-part2`, contract in `protocol/REPRO_CONTRACT.md` |
| evidence registry (what results exist, tiers, hashes, repro) | `jspace_part2/reports/evidence_registry.jsonl` · `jspace-part2 registry-list` |
| published mirror of exploratory era | `labs/interpretability/jspace/part2/` |
| Part-1 errata (read before citing Part 1) | `labs/interpretability/jspace/REPORT_v2_ERRATA.md` |
| Drive HF cache (persistent; WikiText + old models) | `MyDrive/hf_cache/hub` |
| secrets | `interpret/misc/key.tar.gz` (SSH); optional `interpret/misc/hf_token.txt` (place manually; agent tooling won't copy secrets to Drive) |

## 4 · Orientation: branch + commit history

- `main` = course repo; Lab 37 / Part 1 merged via **PR #9** (`4097c44`).
- **`interp_jspace_part2`** = the campaign branch (all Part-2 work).
  Milestones so far: `27c6d3c` bootstrap + original prereg → `20879fe` A0
  transfer gate → `062cff5` output-alignment traces → `a4e2654` forensic
  addendum adopted (Workstream R, errata, tiers) → `8b90728` repro
  contract + `jspace_part2` package. `git log --oneline -20` on the branch
  is always the true recent history; every phase boundary is one commit
  with a result-bearing message.
- Part-1 history (`interp_jspace` branch, merged) is archaeology; its live
  doc snapshot is `special-lab-1/inprogress_part1_final.md`.

## 5 · The repro contract (digest — full text in protocol/REPRO_CONTRACT.md)

1. **No result exists unless** a clean VM + `git clone` + basic deps can
   recreate it with one command AND the artifact carries its provenance
   (code commit, config hash, input hashes, model revision SHAs).
2. Every claim-bearing artifact has an **evidence_id** in the registry
   with tier, producing command, commit, output hashes, repro notes.
   Verify/re-run: `bash interpretability/jspace_part2/repro.sh <evidence-id>`.
3. Producers **refuse dirty git trees** (`--allow-dirty` = dev only,
   disqualifies from confirmatory tier).
4. Supersede by NEW evidence ids + `superseded_by` links; never edit old
   artifacts. Heavy artifacts live on Drive hash-pinned in
   `part2_20260727/manifests/artifact_inventory.jsonl`; weights pin to HF
   revision SHAs.
5. Confirmatory science runs from the `jspace_part2` package only.
   The gitignored `special_lab2/` dir is the frozen exploratory era.

## 6 · Conduct rules the agent must keep (authority: experiment_reset_instructions.md)

Commit + push at every phase boundary (preemptible VMs) · every GPU phase
resumable ≤10-min granularity, same-command resume · queue prefixes sized
to 24 h blocks · `inprogress.md` updated at every boundary and before/after
any long launch · handout-first reporting; figures regenerate from
registered metrics only; fixed entity→hue palette · evidence-tier labels
on everything until gates pass · stats per addendum §12 (paired, clustered,
equivalence for nulls, Holm primary family, power-simulated n) · honest
nulls welcome; forbidden claims per lab37 header · never thin every
workstream; characterize a positive before adding models.

## 7 · Troubleshooting quick hits

- **HF 429/slow**: token missing — step 1. Without it the hub throttles
  after ~90 GB/day unauthenticated.
- **Hub download STALLS mid-file even WITH the token** (seen 3× now: VM6
  twice, VM7 once on `Olmo-3-32B-Think` — bytes stop, process alive, no
  error). Don't wait it out and don't retry the hub. Kill it and rsync
  the snapshot out of the Drive HF cache to local NVMe instead:
  ```bash
  pkill -f "hf download"; rm -rf /content/models/<slug>/.cache
  rsync -rL --size-only \
    /content/drive/MyDrive/hf_cache/hub/models--<org>--<name>/snapshots/<rev>/ \
    /content/models/<slug>/
  ```
  `--size-only` reuses whatever the partial download already landed
  (VM7: 25 G of 61 G reused, full copy ~6 min). This is a Drive→disk
  COPY, which is fine — the forbidden thing is model-LOADING through
  DriveFS. Verify before use: every shard in `model.safetensors.index.json`
  must exist locally, and `config.json`'s sha256 must match the
  `config_sha256` recorded in prior runs' provenance blocks.
- **Model load hangs at N%**: you streamed via DriveFS — kill it,
  hub-download locally (§2 rule), rerun; loads then take ~15 s warm.
- **Push rejected**: ssh key not installed (step 1) or wrong branch
  (`push_lab2.sh` and package producers refuse off-branch/dirty states —
  that's by design).
- **`jspace-part2` not found**: rerun step 4 (`repro.sh` installs it).
- **A fit/grid died mid-run**: rerun the exact same command from
  `inprogress.md`; everything no-ops finished work and resumes from the
  last checkpoint (fits: per-prompt local ckpt, Drive copy every 10).
- **Disk full**: delete `/content/hf_local` models from a FINISHED set
  (never mid-set), keep `/content/models/*` for the active set only.
