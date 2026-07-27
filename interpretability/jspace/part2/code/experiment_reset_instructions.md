# experiment_reset_instructions.md — J-space lab: resume/reset any session

Authored 2026-07-27 (VM6, Part-2 bootstrap). Agent-maintained: update this
file whenever a convention changes so the user never has to retype the rules.
Canonical copy lives in git (`interpretability/special_lab2/`, mirrored to
`jspace/part2/code/`); a copy is kept at
`/content/drive/MyDrive/interpret/special-lab-1/` by `sync_code.sh`.

## The 60-second resume

1. Read `/content/drive/MyDrive/interpret/inprogress.md` — the live map:
   status, queue, what's running, exact next commands. It is the single
   source of truth for "where did we stop".
2. Read this file (rules + restore recipe).
3. **Read `jspace_part2_plan1_addendum.md` (special-lab-1 root; versioned
   copy in the live dir) — the forensic review that GOVERNS Part 2**: it
   sets the evidence tiers, the R-workstream (assay repair before any
   model matrix), the statistics plan (§12), the priority order (§15.1),
   and the stop rules (§15.2). Then `REPAIR_PREREGISTRATION.md` +
   `PLAN_PART2.md` (REVISION-1 header) in the live dir.
4. Part-1 background when needed: `2026-07-26_v2/report/REPORT_v2.md`
   **together with `interpretability/jspace/REPORT_v2_ERRATA.md`** —
   REPORT_v2's headline wording is downgraded there; do not inherit it.

## Durable state map

| what | where |
|---|---|
| live map / resume queue | `MyDrive/interpret/inprogress.md` (agent-maintained) |
| Part-2 run dir (all outputs) | `MyDrive/interpret/special-lab-1/part2_20260727/` (`config lens metrics/<model-slug> figures logs report code`) |
| Part-1 run dirs (frozen) | `.../special-lab-1/{2026-07-25_1726, 2026-07-26_v2}/` |
| repo + branch | `github.com/karlb-dev/labs`, branch `interp_jspace_part2` (part 1 merged via PR #9 on `main`) |
| live working dir (gitignored) | `/content/labs/interpretability/special_lab2/` (part-1 code vendored at `part1/`) |
| published mirror (in git) | `interpretability/jspace/part2/` via `push_lab2.sh` |
| Drive HF cache (persistent) | `MyDrive/hf_cache/hub` — OLMo-3-32B-Think, OLMo 7Bs, gemma-4-31B-it, WikiText |
| local HF cache (per-VM) | `/content/hf_local` — new big models; delete between sets |
| jlens engine | `anthropics/jacobian-lens @ 581d3986`, editable install; byte-checked vs Drive mirror |
| claim ledger (Part 2) | `interpretability/jspace/part2/README.md` (SL2-C*) |
| master results dataset | `part2_20260727/report/matrix_master.{csv,json}` |

## Fresh-VM restore recipe

```bash
# 0. sanity
nvidia-smi; df -h /; ls /content/drive/MyDrive/interpret || echo "MOUNT DRIVE FIRST"
# 1. ssh key (only if ~/.ssh is empty)
tar xzf /content/drive/MyDrive/interpret/misc/key.tar.gz -C ~/ && chmod 600 ~/.ssh/id_ed25519
# 1b. HF token (fresh VM): from the NOTEBOOK cell (os.environ there does NOT
#     reach these shells — separate processes; the shared filesystem does):
#     import os, pathlib; p = pathlib.Path("~/.cache/huggingface/token").expanduser()
#     p.parent.mkdir(parents=True, exist_ok=True); p.write_text(os.environ["HF_TOKEN"].strip())
#     verify here:  python -c "from huggingface_hub import whoami; print(whoami()['name'])"
git config --global user.name "karlb-dev"; git config --global user.email "kburtram@live.com"
ssh -o StrictHostKeyChecking=accept-new -T git@github.com   # expect "Hi karlb-dev!"
# 2. repo
cd /content && git clone git@github.com:karlb-dev/labs.git && cd labs && git checkout interp_jspace_part2
# 3. jlens, pinned + verified
git clone https://github.com/anthropics/jacobian-lens.git /content/jacobian-lens && cd /content/jacobian-lens
git checkout $(cat /content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726/code/jacobian-lens-pkg/GIT_REVISION.txt)
diff -rq jlens /content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726/code/jacobian-lens-pkg/jlens  # expect: identical
pip install -e /content/jacobian-lens
# 4. live code — repo is canonical; Drive code mirror is the fallback of record
#    (if the repo clone already has interpretability/special_lab2/, skip)
cp -r /content/drive/MyDrive/interpret/special-lab-1/part2_20260727/code/special_lab2 \
      /content/labs/interpretability/special_lab2
# 5. handout toolchain (needed at reporting boundaries)
apt-get install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra
# 6. resume whatever inprogress.md says was running; every script no-ops on
#    finished outputs and resumes per item/condition. pipefail + tee logs.
```

Models: do NOT copy weights from Drive for new models — hub download to
`/content/hf_local` measured 320 MB/s (61 GB in ~3 min), faster than DriveFS.
Drive-cached models (Think, gemma-4, WikiText) load in place via the default
`HF_HUB_CACHE`; first Think load through DriveFS takes ~10 min, then RAM page
cache makes reloads cheap.

## Standing directives (the rules, so nobody retypes them)

1. **Preemptible VMs**: commit early, commit often, push at every phase
   boundary (`bash refresh_handout.sh && bash push_lab2.sh "<phase>: <one-line result>"`).
   Everything durable goes to the Drive run dir incrementally: atomic JSON
   writes, per-item/per-condition checkpoints, logs tee'd locally and rsync'd
   by `sync_code.sh`.
2. **Every script**: no-op if its outputs exist, resumable mid-phase,
   `--force` to override. GPU scripts self-check free VRAM/disk before load.
3. **inprogress.md is agent-maintained.** Update at every phase boundary and
   before/after launching anything long. Must always contain: status line
   with UTC time + VM tag, the queue table with per-cell status, what is
   running right now + its log path, and the exact resume command.
4. **Handout-first reporting**: after every result-bearing phase, regenerate
   figures from metrics, recompile the living LaTeX handout (tex+pdf), keep
   `REPORT_PART2.md` and `summary/matrix` datasets current. The user reads
   the PDF, not the JSON.
5. **Visualization discipline**: figures regenerate from metrics JSONs alone
   via committed scripts — no hand-made plots. The master dataset
   (`matrix_master.csv/json`) carries every cell with n/seed/CI/dose/decoding
   /provenance so everything regenerates as probes, models, and evals
   improve. Fixed entity→hue map across all part-1+2 figures: J=#2a78d6,
   random=#eb6834, non-J=#1baf7a, baseline=ink; new condition hues chosen
   once in the figure script and never changed.
6. **Disk strategy**: run model SETS (one residency each): OLMo pass 1
   (A0,B3,A1,C3,C1,D) → Qwen (A2*) → OLMo pass 2 (B1,C2,B4,B2) → Gemma (A3)
   → E. Delete `/content/hf_local` models between sets. **NEVER delete the
   DriveFS chunks.db** — it wedges the mount into serving empty content;
   recovery = kill the `--single_process` `/opt/google/drive/drive` worker
   and let the supervisor respawn it.
   **VM6 lesson (2026-07-27): don't model-load through DriveFS at all.**
   Streaming Think via the Drive HF cache stalled mid-load (~50 GB in, GPU
   idle, page cache saturated by a prior 61 GB local download; 10+ min no
   progress). Hub download to `/content/hf_local` is ~320 MB/s (61 GB ≈ 3
   min) — ALWAYS hub-download big models to local NVMe and run with
   `HF_HUB_CACHE=/content/hf_local` (or `p2_load_model`, cache="local").
   The Drive HF cache remains the persistent fallback + WikiText source.
7. **Claim discipline**: evidence vocabulary is binding — established /
   provisional / exploratory / not-currently-identified (addendum §1.4);
   every metrics row, figure, and report line carries a tier label until
   the R-gates pass. Ledger namespaces: SL2-M* methods, SL2-O* OLMo
   lineage, SL2-Q* Qwen mode, SL2-G* Gemma, SL2-C* causal/task, SL2-T*
   temporal (addendum §19.6); each claim records estimand, tier, scope,
   effect+interval, artifact path, prereg link, falsifier, boundary,
   rerun status. Nothing inherits SL1 wording without R7 re-audit.
   Forbidden claims from the lab37 header still carry over.
8. **Budget**: ≤200 h campaign cap (user, 2026-07-27). Priority order =
   addendum §15.1 (R-workstream first; OLMo lineage before breadth);
   hard stop rules §15.2 (no models added while the positive control
   fails; no "null" without equivalence bounds; no "matched" without
   measured energy+effective-rank on the evaluated items; no
   answer-anywhere as a behavioral outcome; never thin every workstream).
   Compute envelope: repair 8–15 GPU-h; minimal publishable OLMo study
   35–60; strong cross-model 70–110; full package 100–160 (§16).
9. **Stats (addendum §12 replaces the old habits)**: paired item-level
   deltas; clustering by the true generation unit (template/schema/fact
   family); equivalence tests for nulls (planning margins: 0.5 nats
   answer-seq lp, 10 pp accuracy, 0.10 nats/token prose NLL, 5 pp top-1
   agreement, 20% occupancy — pilot-calibrated before confirmatory);
   power simulation instead of habitual n=60; small Holm-corrected
   primary family + separate secondary families; test interactions
   directly (never "significant in A, not in B"); teacher-forced
   deterministic primary endpoints; full-answer-sequence scoring.
10. **pipefail + tee on every long run** (a tee-masked exit code cost VM4 an
    evening). Long GPU runs launch detached with logs in the run dir.

## Model pins (hub-checked 2026-07-27)

| slug | HF id | cache | note |
|---|---|---|---|
| olmo3-think | `allenai/Olmo-3-32B-Think` | Drive | part-1 anchor + lens donor |
| olmo31-instruct | `allenai/Olmo-3.1-32B-Instruct` | local | A1. `Olmo-3-32B-Instruct` DOES NOT EXIST; 3.1 shares base `Olmo-3-1125-32B` with Think |
| olmo3-base | `allenai/Olmo-3-1125-32B` | local | optional H1 anchor |
| qwen36-27b | `Qwen/Qwen3.6-27B` | local | A2; no non-think dense sibling ships — A2 = same-model matrix completion |
| gemma4-31b | `google/gemma-4-31B-it` | Drive | A3; adaptation gate first (norm placement, logit softcap, fit memory) |

## History pointers

Part 1 (COMPLETE, PR #9): `inprogress_part1_final.md` (frozen copy of the
part-1 live doc), run dirs `2026-07-25_1726/` + `2026-07-26_v2/`,
`REPORT.md`/`REPORT_v2.md`, handout PDF, claim ledger SL1-C1..C7 in
`labs/lab37_jspace_workspace.md` (one open box: human label pass).
