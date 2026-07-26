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

## 2026-07-26 06:30–06:39 (VM3) — workshop handout (user request)
- Wrote handout/olmo32b_jspace_handout.tex — 5-page workshop handout
  (pdflatex, article 10pt: abstract, setup, descriptive, causal table,
  thinking-model results, Phase 5, side-by-side scorecard, limitations,
  next steps, refs; figures f1/f2/f4/f5/f6/f8). Compiled clean on VM
  (texlive-latex-base+recommended), visually verified pp.1–2.
- Shipped self-contained to Drive report/handout/ (tex + PDF + figures/) —
  Overleaf-ready; author line left as placeholder for the user.
