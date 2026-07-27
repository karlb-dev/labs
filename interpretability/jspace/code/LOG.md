# special-lab-1 running log

## 2026-07-25 ~17:26–17:41 (VM1, prior session)
- Created run dir `2026-07-25_1726`, wrote `config/environment.json` +
  `pip_freeze.txt` (RTX PRO 6000 Blackwell 95GB, torch 2.11.0+cu128,
  transformers 5.13.1, HF cache on Drive).
- Smoke test on `allenai/Olmo-3-7B-Instruct`: lens fit on 6 prompts,
  source layers 8..26 step 2, 195s. Boot-probe battery (11 single-token
  factual probes): J-lens 9/11 hits@20 at mid layers vs logit-lens 7/11;
  median mid-layer best rank 3 vs 10. Mini ablation on boot→lira:
  J-space band ablation Δlogprob −2.45 vs random +1.22. Artifacts:
  `lens/smoke7b_lens.pt`, `metrics/smoke_7b.json`,
  `figures/smoke_boot_rank_vs_layer.png`.
- **VM died with `code/` empty — the smoke pipeline code was lost.**

## 2026-07-25 ~20:00 (VM2, this session)
- Session resumed on fresh VM, same GPU class (RTX PRO 6000, 97.9GB free).
  Repo intact (clean @ origin/main 74863a2). Drive artifacts intact.
- Verified references: paper (published 2026-07-06) fetched; `jacobian-lens`
  cloned to /content/jacobian-lens and `pip install -e`'d (works with
  transformers 5.13.1); paper's own experiment/eval prompt sets found in
  its `data/`. Nanda replication details from LessWrong review: Qwen 3.6
  27B, 25 Pile prompts × 128 tok, skip first 4, target penultimate layer;
  multilingual/typo evals replicated strongly, poetry/arithmetic failed,
  CKA bands "less clean"; flags: noisy directions, possibly confounded
  interventions.
- **Model decision**: `allenai/Olmo-3-32B-Think` (used by repo tier-c
  validation runs; 70GB already in Drive HF cache). `Olmo-3.1-32B-Think`
  (64.5GB, never run by any lab config) does not fit Drive (62GB free) and
  leaves ~1.5GB margin locally — recorded as untested-newer-option.
- **Corpus decision**: WikiText-103 (cached; matches neuronpedia reference
  lens recipe "Salesforce-wikitext") over Dolma (thematic ideal, fragile
  download, no methodological gain).
- Rule adopted after VM1's loss: `sync_code.sh` mirrors the lab dir to
  Drive `code/` after every edit session; fits copy checkpoints to Drive
  every 10 prompts; all JSON writes atomic.
- 7B weights pre-warm from Drive into page cache started in background.

## 2026-07-25 ~20:15–20:35 (VM2) — pipeline rebuilt, fit launched
- s0 env audit (vm2) + s2 corpus (200 seeded WikiText-103 records,
  sha 481be0c6...) written to Drive.
- s1 smoke verify: **exact reproduction** of VM1's battery with the stored
  7B lens — 9/11 hits@20, median mid rank 3 (identical). Pipeline proven.
- s3 fit launched 20:15 on Olmo-3-32B-Think: 21 source layers
  [4,8,12,16,20..44 step2,48,52,56,60], target L63, dim_batch 8,
  128-token prompts. Model load 426s (65.1GB VRAM); fitting at 96% GPU
  util, 77.6GB peak. **First prompt 153s**, max||J||/sqrt(d)=0.95 →
  ~77 min/slice, ~5.5h for all 120 prompts. Local ckpt every prompt,
  Drive copy every 10.
- Scripts s4–s9 written, compile-checked, synced to Drive code/ (sanity,
  descriptive w/ gradient-pursuit — unit-tested rel_err 3e-4 / 100% atom
  recovery on synthetic — broadcast, ablation, CoT, report+figures).
- Methodology pinned from paper fetch: active concepts = sparse nonneg
  gradient-pursuit decomposition onto J-lens dictionary (k~25 typical);
  variance share = pursuit reconstruction's share (<10% ceiling);
  ablation = zero projections of top-10 most-active J-dirs over the band;
  dictionary at layer l = row-normalized (W_U ⊙ g) @ J_l (RMSNorm gain
  linearization — report limitation).
- Tokenizer findings: Think chat template ALREADY ends with an open
  "<think>" (suppressed-CoT = immediate "\n\n</think>"); ' lira' has no
  single token → all rank tracking uses min over case/space variant
  first-tokens (paper-style synonym sets); 'lira'-like words degrade to
  1-char fragments — noted as limitation.

## 2026-07-26 04:45 (VM3, fresh session — VM2 reclaimed)
- VM2's last Drive write was 03:47 (ablation_results.json); reclaimed
  shortly after. Everything durable was on Drive — restore was clean:
  code mirror -> /content/labs/interpretability/special_lab1, jacobian-lens
  re-cloned at pinned 581d3986 (source byte-identical to Drive mirror),
  `pip install -e` OK. torch 2.11.0+cu128 / transformers 5.13.1 /
  datasets 4.0.0 match VM2 exactly. GPU same class (RTX PRO 6000, 97.9GB
  free). The inprogress.md resume recipe worked verbatim except one gap,
  now fixed there: the Drive jlens mirror has no pyproject/data, so the
  GitHub clone (at GIT_REVISION.txt) is required, then diff against the
  mirror to rule out local patches (there were none).
- NB: this LOG's VM2 entries stop at 20:35; VM2's later milestones (s4
  PASS 02:04, s5 02:16, s6 02:23, s7 03:08 + addendum) live in
  inprogress.md's status table — left there, not reconstructed here.
- s7 addendum state on restore: 98/99 (condition,task) blocks done — only
  nonJ_pca_k40/arithmetic_v2 missing (VM died inside the final block).
  Relaunched s7 (skips finished blocks) chained into s8 CoT (fresh,
  per-item resumable, 90 items), logs -> logs_local/ -> Drive via sync.

## 2026-07-26 05:04–06:05 (VM3) — pipeline completed, report written
- s7 addendum finished 05:04 (99/99). Verdict: twohop_lp confirms the causal
  null — static J ≈ random at all doses; nonJ-PCA does the real damage
  (register destruction, SQL→0, NLL 2.71→3.45/4.02); dyn10 = live-computation
  confound (asterisk collapse). arithmetic_v2 rescoring adopted for figures
  (v1 two-line window was format-biased: baseline 0.47 -> 0.87).
- s8 ran 05:15–05:51, 90/90 items @ ~25s, zero failures. Pre-CoT anticipation
  NULL (answer med rank 5613, 10% <=20; logit 11610); suppressed-CoT readout
  at answer point: med rank 211, 23% <=20, monotone collapse L30->L60
  (2048->6); think-vs-suppressed accuracy: twohop 0.95->0.82, arith
  1.00->0.90, sql 1.00->1.00. Divergence: 116 events / 59 items, only 4
  answer-matching; dominant divergent words are discourse adverbs
  ('famously', 'respectively', 'etc') during recall/enumeration spans.
- NEW s8b_cot_lead.py (CPU): workspace leads first textual answer statement
  by median 46 steps, 91% of 34 traced items (twohop 48.5 / sql 38 / arith
  9.5); per-layer anticipation profiles; top-20 divergence events verbatim
  -> metrics/cot_lead.json.
- s9 patched (arithmetic_v2 in f4/f5; cot_lead folded into summary.json),
  7 figures regenerated; REPORT.md authored to report/ with full
  paper/Nanda side-by-side, 9 limitations, 8 next steps.
- Phase 5 decision: run the MODEST version (readout differencing only, 20
  matched pairs, no ablation unless a clear consistent direction emerges) —
  GPU idle, all prior phases durable on Drive. s10_evalaware.py.

## 2026-07-26 06:13–06:20 (VM3) — Phase 5 done; LAB COMPLETE
- s10: eval-framing direction emerges far beyond criterion (p<0.001 all 7
  layers, cos 0.48–0.71, |meanD| 22->134 with depth; top tokens from L32 =
  literal evaluation lexicon; L24–28 obliquer: 'monitored','internally',
  'careful'). Gated mini-ablation ran and was behaviorally INERT (0 test
  mentions both conditions, near-identical generations). Recorded with the
  semantic-echo caveat: cue words are literally in the prompt, so this is a
  positive context-verbalization demo, NOT evidence of latent eval-awareness;
  confound-broken variant queued in report §7.
- REPORT.md updated (verdict slug, headline bullet, §4.4, limitation #10,
  next-step #8, read-next #6). All phases 0–5 complete; run dir is the
  archive. Remaining known-open: variance-matched control, frozen-top-k
  ablation, late-band fit, Dolma lens, CoT-rescue — all in report §7.

## 2026-07-26 08:1x–08:2x (VM3) — published as draft Lab 37 (user lifted no-git rule)
- User directive supersedes v1's "never git add": work is now a draft
  proposed lab. SSH key from interpret/misc/key.tar.gz -> ~/.ssh (auth OK
  as karlb-dev); repo-local identity karlb-dev <kburtram@live.com>
  (matches lab-commit history); origin switched to SSH.
- Branch interp_jspace: labs/lab37_jspace_workspace.md (lab36-style header,
  protocol table, operationalization-audit section, claim templates
  SL1-C1..C5 with falsifiers, promotion criteria) + jspace/ (README with
  findings table + artifact map, code mirror s0-s19, report, living
  handout, figures f1-f10, small metrics v1_/v2_ prefixed; >5MB Drive-only)
  + SPECIAL_TOPICS.MD proposed-extension stanza + .gitignore for
  special_lab1/. Pushed 61b4ada.
- New phase-boundary routine: refresh_handout.sh then push_lab.sh "<msg>"
  (mirrors live dir into jspace/, commits with trailer, pushes; refuses to
  run off-branch). Drive remains the durable archive; handout remains the
  primary read.

## 2026-07-26 06:30–06:39 (VM3) — workshop handout (user request)
- Wrote handout/olmo32b_jspace_handout.tex — 5-page workshop handout
  (pdflatex, article 10pt: abstract, setup, descriptive, causal table,
  thinking-model results, Phase 5, side-by-side scorecard, limitations,
  next steps, refs; figures f1/f2/f4/f5/f6/f8). Compiled clean on VM
  (texlive-latex-base+recommended), visually verified pp.1–2.
- Shipped self-contained to Drive report/handout/ (tex + PDF + figures/) —
  Overleaf-ready; author line left as placeholder for the user.

## 2026-07-26 07:0x–09:30 (VM3) — v2 P4, P1, P2 landed; died mid-P3
(Backfilled on VM4 from inprogress.md + metrics; VM3 was reclaimed before
this file got the entries.)
- P4 s13 (07:57, CPU): cot-lead foil calibration — answer det 0.92 / median
  lead 46 vs freq-matched noise floor det 0.06; family foils det 0.32 but
  lead ≈3 (concurrent with text = candidate enumeration, the faithful-readout
  signature); answer earlier than ALL its foils in 66% of items, median det
  rank 1. v1 lead claim SURVIVES with a floor. -> metrics/cot_foils.json, f9.
- P1 s11 (08:35): energy-matched grid. Measured first: v1's rank-matched
  controls were energy-mismatched ~10x in OPPOSITE directions (random_k ~3x
  lighter than J-span; top-k nonJ PCs far heavier — windowed deep-PC fallback
  needed at all 13 layers). At matched energy (0.97–1.01 per layer/dose):
  ALL groups = baseline at every dose ≤40 dims/layer; SQL 0.667 at all doses
  (v1 unmatched: 0.000); twohop_lp Δ −0.01/−0.17/−0.15 CIs⊃0. v1's non-J
  "selective damage" was 100% energy artifact; causal null now CLEAN and
  v1's J≈random was conservative (J carries 2–5x more energy per rank).
  -> metrics/energy_match.json + ablation_v2.json, f10/f11. Pushed 53fc1c2.
- P2 s12 (09:10): frozen per-item ablation — MARQUEE. frozen_j10 deletes the
  retrieved fact (twohop_lp −4.63 vs base −1.74, Δ−2.9 nats CI[−5.5,−3.8];
  recall 0.58→0.23 on single AND multi-hop) while frozen_rand10 = baseline
  everywhere, generations coherent (samples audited). NOT the paper's
  dissociation (shallow collapses equally → recall content channel, not
  scratchpad). live_rand10 = baseline → v1's live-J lobotomy was J-specific
  (pool-size caveat: 5120-row dict vs full vocab). -> frozen_ablation.json,
  f12. Pushed f036a60.
- P3 s14 launched 09:10; VM3 reclaimed ~09:30 with slice0 at 10/30 on Drive.

## 2026-07-26 15:39–15:50 (VM4, fresh session — VM3 reclaimed)
- Restored via inprogress.md recipe: jlens clone @581d3986 byte-identical to
  Drive mirror; code from the v2 mirror (sync_code.sh excludes handout/ from
  the v2 mirror — restored handout/ from 2026-07-26_v2/report/handout/, tex
  matches the repo copy from push f036a60). SSH key already present on this
  VM (~/.ssh mtime 15:33, pre-session); push access verified by dry-run.
- GPU: RTX PRO 6000 Blackwell 97 GB again; torch 2.11.0+cu128, transformers
  5.13.1 (same as VM3); bf16 matmul sanity pass (new s20_vm4_sanity.py).
- texlive reinstalled (fresh VM had none; needed apt-get update first, two
  404s otherwise); handout recompiles clean, 8 pages.
- 15:45 s14 resumed from slice0 10/30 (log: logs/s14_v2_vm4.log). Plan
  unchanged: s14 -> s15 (a,b + --with-cot c) -> s16 -> s17 -> phase Q
  decision -> REPORT_v2 + claim-ledger instantiation.
- Prep while fitting: s18_qwen_instruments.py WRITTEN (gated; hub pins
  verified — see PLAN_v2 phase-Q row); memory of Neuronpedia lens layout
  recorded there. f11 legend moved lower-left (was clipping on the
  grammaticality row).

## 2026-07-26 17:22–17:4x (VM4) — s14 done; s15 phase a+b landed (P3 verdict forming)
- s14 DONE 17:22: merged 120-prompt late lens {46,50,54,58,62} -> v2
  lens/olmo32bthink_late.pt. 45s/prompt (backward spans only L46->63),
  peak 69.3GB. Raw fitlate_*.ckpt deleted from Drive+local after merge
  (2GB; slice lenses retained) — user flagged disk pressure.
- s15 phase a DONE 17:34 (fast: 5 layers, ~1 min + PCA): **no late-shifted
  workspace.** Var share L46/50/54/58 = 0.64/0.61/0.65/0.72% (flat, same
  ~10x-thinner capacity as mid band), L62 = 1.60% (unembed-adjacent, like
  v1's L60 0.91%); act@0.01 = 4-7; top-1 persistence DECLINES 0.153->0.026
  monotonically — the inverted-U's right side. -> descriptive_late.json +
  layer_state_late/.
- s15 phase b data complete (90 items on Drive) but the run CRASHED in the
  aggregation: fresh in-memory items keep INT layer keys, aggregation
  indexes str(l) -> KeyError '46'. Fix: reload PROF_OUT from disk before
  computing median_profile (JSON round-trip normalizes keys). Also: the
  `| tee` pipeline masked the non-zero exit (task reported exit 0) —
  relaunched under `set -o pipefail`; earlier runs' completion claims were
  all verified against metrics, not exit codes, so nothing else affected.
- Phase b numbers (computed from the saved items, rerun will confirm):
  think-mode answer median rank 3916-7291 in the late band (best-layer
  median 1670, 0% <=20) — pre-CoT null SURVIVES the dedicated late lens
  (C5 falsifier refuted). Suppressed: per-layer medians 61/58/8/6/5
  (L46..62), best-late-layer median 3, 76% <=20, 58% <=5. Logit lens reads
  the late band equally well (30/18/6/4/4) — J-over-logit is a MID-band
  phenomenon; both lenses converge near the unembedding (final median 4).
- Handout §[v2-P3] subsection written (phase a+b); phase c (cot-lead-late)
  folds in when it lands.

## 2026-07-26 17:44–18:20 (VM4) — P3 complete and pushed; s16 launched
- 17:46 pushed 1dba39d (P3 a+b: handout §P3, f13/f14, late metrics; git
  identity had to be re-set repo-local on this fresh clone first).
- s15 rerun (pipefail) confirmed phase-b medians and ran phase c to done
  18:17: **late-band cot-lead = 49.5 steps median, 86.4% ws-first,
  det 98.9%, n=88** (mid-band was 46/91% — same phenomenon, independent
  lens, disjoint layers). Per-item ~23s. -> cot_lead_late.json.
- P3 phase c folded into handout §P3 as item (c); s16_cot_rescue launched
  18:19 (resumable per condition x item; ~2.5h full, VM horizon ~19:00 —
  cells bank to Drive as they complete).

## 2026-07-26 22:47–23:0x (VM5) — restore, disk incident, Q-first flip
- VM5 up (RTX PRO 6000 Blackwell 97GB, torch 2.11+cu128 — same as VM4's
  validated combo; standalone bf16 sanity skipped on that basis, s16's
  banked-item reproduction serves as the check). Restore recipe verbatim:
  jlens clone byte-identical at pin 581d398; code from v2 mirror; repo
  already on interp_jspace @ d6ba8ca clean. texlive installed (apt lists
  were stale -> update + retry).
- PLAN_v3 executed from T+0:00=22:47. s16 trimmed to the capped grid
  (frozen_j10/frozen_rand10 x twohop/onehop/arith, SQL + static-rescue
  cells dropped) and relaunched; Qwen weights (55 GB) + Neuronpedia lens
  fetched to /content/hf_local in parallel.
- **Disk incident (root-caused):** DriveFS FUSE caches streamed reads on
  local disk; Qwen-local (55 GB) + OLMo-from-Drive (64 GB stream) cannot
  coexist on the 113 GB disk. tee-to-Drive died ENOSPC at 23% of the OLMo
  load and pipefail killed s16. Fixes: logs now tee LOCALLY (logs_local/,
  rsynced at boundaries); phase order flipped Q-FIRST (PLAN_v3's own
  priority rule: banked Q1 outranks finished s16); Qwen blobs get deleted
  before the OLMo rerun; a watchdog will age-evict DriveFS content_cache
  during the OLMo stream.
- s18 restaged before first grid cell: (1) marquee frozen cells now run
  BEFORE the static cells (old order had them last -> a mid-grid death
  would have banked controls but not the verdict cells); (2) build_dicts
  switched to fp16 staging — Qwen's 248k vocab made the fp32 recipe peak
  ~99 GB (OOM); fp16 peaks 92.6 GB measured. First (aborted) s18 launch
  cost ~4 min, phase a banked.
- Q0 GATE PASS (60s): Qwen multihop with the published Neuronpedia lens
  under our harness — jlens pass@1 0.350 / @5 0.517 / @20 0.739 vs logit
  0.317/0.500/0.600, n=60. Readout advantage replicates on Qwen.
- Q1 in flight; first cells: none twohop 0.867, frozen_j10 twohop 0.367
  [0.200,0.533] — the frozen deletion transfers to Qwen.

## 2026-07-26 23:0x–23:33 (VM5) — phase Q COMPLETE; two numerics fixes en route
- s18 pass 1 killed deliberately pre-grid (old cond order + fp32 build_dicts
  would OOM at 248k vocab); pass 2 landed Q1+Q2 but phase-b variance shares
  went inf/nan at L20-38 — Qwen's outlier activations (a) overflow fp16 in
  the pursuit correlation (fixed: bf16) and (b) diverge the fixed-lr=0.25
  refit when near-duplicate atoms push the active-set Gram past 2/lr
  (fixed: step = min(lr, 1/(k+2)), provably contractive for unit-norm
  atoms). Stable layers L40-44 moved <=0.2pp under the fix (0.0668/0.0679/
  0.0642 -> 0.0651/0.0675/0.0639) — commensurability preserved. Static
  Q2 cells re-ran per pass: -0.87/-0.77/-0.87 (broken selection) vs
  -0.85/-0.72/-0.80 (clean) — the static null is insensitive even to
  broken direction selection (recorded as a robustness note).
- FINAL Q numbers: sanity jlens 0.350/0.517/0.739 vs logit 0.317/0.500/
  0.600 (n=60). Descriptive: var share 4.3->6.8% rising L20->L44,
  act@0.01 32-42, act@0.02 14-21, persistence 0.03-0.09 (paper-range
  capacity; OLMo is the outlier). Causal: frozen_j10 twohop 0.87->0.37,
  lp -0.77->-3.19 (delta -2.42); frozen_rand10 = baseline; onehop nearly
  intact on Qwen (0.90->0.83) unlike OLMo (collapsed equally) — the
  composed-task bias is the sharpest follow-up target. Statics all within
  +-0.08 nats of baseline. Handout §Q + f15; pushed 2d245fd.

## 2026-07-26 23:35–23:45 (VM5) — DriveFS incident and recovery (ops)
- The swap script's age-gated cache eviction deleted DriveFS's chunks.db
  (the cache is ONE SQLite db, not per-file chunks) -> the fuse.drive
  mount served EMPTY content for every cached file while stat sizes stayed
  correct; a first integrity audit looked like total metrics corruption.
- Recovery: kill -9 the --single_process /opt/google/drive/drive worker;
  its supervisor respawns it; reads recover fully. Server-side audit after
  restart: ALL v2 metrics, layer_state_qwen, v1 spot-checks parse — zero
  data loss (uploads had completed before the sweep; only local FUSE state
  was wedged).
- Standing fixes: never delete DriveFS cache files; watchdog LOGS only;
  local mirror of all v2 metrics refreshed at every boundary
  (/content/metrics_backup); logs tee locally. Disk headroom for the OLMo
  stream came from deleting Qwen blobs (55 GB); rm of base-image packages
  frees nothing (overlayfs lower layer).

## 2026-07-26 23:44–… (VM5) — P5+P6 via one-load driver
- s21 driver: single OLMo load (750 s through DriveFS) shared by s16
  (capped rescue grid) and s17 (P6 trim: fresh probe-swap twohop [60:90]
  n=30, readout tasks, static cells ride along). Cells flowing ~44 s/item.
- Early P5 shape (n=8 partial): frozen_j10 any-rate ~0.6 vs no-think 0.23
  vs frozen_rand10 ~1.0 — partial rescue; closed-</think> rate ~0-0.25 so
  post-segment scoring is degenerate (any/think metrics lead, decided
  before final numbers).

## 2026-07-27 00:32–00:5x (VM5) — P5+P6 COMPLETE; GPU program done T+1:46
- s16 (capped): twohop any-rate frozen_j10 0.80 vs no-think 0.23 vs
  frozen_rand10 0.93; onehop fully rescued 1.00; arith 0.93 both conds
  (token-cap scorer floor); closure rate HALVED under frozen-J (0.20 vs
  0.40). Rescue prediction holds in content-channel form -> SL1-C6.
- s17 (P6 trim): fresh probe-swap [60:90] items, seed-1 pools: frozen_j10
  acc 0.233 (== seed-0 exactly, disjoint items) dlp -2.82 (seed-0 -2.89);
  frozen_rand d-0.11; statics all CIs superset baseline. Handle is item-,
  pool-, and seed-robust; static null generalizes to never-seen items.
- Reporting finalized: REPORT_v2.md, README, PR_BODY.md, lab37 spec
  (allowed claim, status COMPLETE-proposed, checklist 3/4 boxes, C1-C7),
  handout §§P5/P6/Q + f15/f16/f17, inprogress.md flipped to FINAL.
