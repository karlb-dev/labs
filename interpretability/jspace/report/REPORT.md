# special-lab-1 — J-space Replication on OLMo-3-32B-Think

Run `2026-07-25_1726` · model `allenai/Olmo-3-32B-Think` (64 layers, d=5120, bf16)
· lens `jlens` @ `581d3986` · fit corpus WikiText-103 · seeds 0 everywhere.
Executed across three Colab VMs (VM1 smoke → died; VM2 fit + phases 2–3 → died at
the 99th of 99 ablation cells; VM3 phase 4 + this report). Zero data loss:
every phase checkpointed incrementally to this Drive run dir.

References: Anthropic, "Verbalizable Representations Form a Global Workspace in
Language Models" (transformer-circuits.pub/2026/workspace, Jul 2026); Nanda's
Qwen 3.6 27B replication (LessWrong review; neuronpedia/jacobian-lens);
`anthropics/jacobian-lens` reference implementation. Paper/Nanda comparison
numbers below are as pinned in PLAN.md/LOG.md from the 2026-07-25 fetches.

## Verdict

`geometry_replicates__capacity_10x_thinner__causal_dissociation_null__workspace_leads_cot_46_steps__eval_direction_legible_but_inert`

## Headline numbers

- Lens beats logit lens where it should: multihop bridge-entity pass@1 **0.283 vs
  0.200** (+41% rel., n=60); easy factual probes both 17/21 @rank≤20 (ceiling).
- Active concepts per (position, layer): **median 6** @θ=0.01 (3 @θ=0.02, k90≈4),
  peaking L36–42. Paper: ~10–25.
- Variance share of the J-space: **max 0.67%** in the band (0.91% at L60,
  unembedding-adjacent). Paper: 6–10%. → structure present, ~10× thinner.
- Mid-band localization replicates qualitatively: top-1 persistence 0.19–0.20 at
  L32–40 vs 0.03–0.08 at both ends (cleanest inverted-U we measured).
- Broadcast: top J-directions read by **73–94** downstream components vs **0**
  for random (p≈1e-24) — but non-J top PCs are read by 77–125 → **not J-specific**.
- Causal dissociation: **does not replicate at doses 10/20/40 dims/layer.**
  Static J-span ablation ≈ random control on every task and on answer-logprob
  (Δ −0.15…−0.18 vs −0.01…−0.11, CIs straddle 0). The non-J high-variance
  control does the real damage (twohop_lp −0.86/−0.93, CIs<0; SQL→0 via
  register destruction; prose NLL 2.71→3.45/4.02). Paper-faithful dynamic
  top-10 destroys everything (NLL 2.71→24.34; asterisk degeneration) — the
  live-computation confound Nanda flagged; uninterpretable, not evidence.
- Thinking-model novel result: **pre-CoT anticipation is null** (answer median
  J-rank 5613 at last prompt token; 10% ≤20; logit lens 11610, 0%), but the
  **workspace leads the CoT text by a median 46 steps** (91% of traced items),
  and forcing an immediate answer (closed `</think>`, zero reasoning tokens)
  pulls the answer to median rank **211** (23% ≤20) with a clean monotone
  collapse L30→L60 (2048→6). Suppressing CoT costs accuracy: twohop 0.95→0.82,
  arithmetic 1.00→0.90, SQL 1.00→1.00.
- CoT faithfulness: 116 divergence events over 59/90 items, but only 4 involve
  the final answer; the dominant divergent words are discourse-mode adverbs
  (`famously`, `respectively`, `etc`) during recall-flavored passages.
- Phase 5 (modest): an **eval-framing direction emerges decisively** in the
  J-readout (sign-flip p<0.001 on all 7 probed layers; mean pairwise cos
  0.48–0.71; top decoded tokens from L32 up are literally `evaluation` /
  `assessed` / `assessing`), but projecting it out during generation leaves
  behavior essentially unchanged — legible, behaviorally inert at this scale,
  and possibly a lexical echo of the framing text (see §4.4).

## 1 · Environment & feasibility tier

**Tier landed: the top tier — full bf16, single GPU, no quantization, no
gradient checkpointing, full-model backward at fit shapes.** No silent
downgrades. RTX PRO 6000 Blackwell (97.9 GB) on all three VMs; Python 3.12.13,
torch 2.11.0+cu128, CUDA 12.8, transformers 5.13.1; HF cache on Drive.
Fit cost: 4 slices × 30 prompts × ~157 s/prompt ≈ **5.2 h GPU**, peak VRAM
**80.6 GB** (dim_batch 8, seq 128, skip_first 16, target L63). The one planned
reduction, recorded upfront in PLAN.md: 21 source layers of 64 (early controls
4–16, dense band 20–44, late controls 48–60) and 120 fitting prompts (jlens
README: quality saturates ~100; paper used 1000).

Model choice note: `Olmo-3.1-32B-Think` exists but has never been run by any
repo lab config and does not fit next to the cached weights on Drive (62 GB
free); we used the repo-standard `Olmo-3-32B-Think` (recorded as an
untested-newer-option, not a downgrade).

## 2 · Lens fit quality

- Per-prompt max ‖J‖/√d stable at 0.94–0.95 across all 120 prompts; running-mean
  deltas fell ~1/n → converged; slice merge via `JacobianLens.merge` (4×30).
- VM-to-VM determinism: VM1's 7B smoke battery reproduced **exactly** on VM2
  from the stored lens (9/11 hits@20, identical median ranks).
- Sanity gate (32B): 17/21 single-token factual probes surface the answer at
  rank ≤20 in the mid band for *both* J-lens and logit lens (easy probes are
  at ceiling and don't discriminate). The discriminative eval — multihop
  bridge-entity readout at the final prompt token (n=60, min over layers):
  J-lens pass@1 **0.283** vs logit **0.200**, pass@5 0.406 vs 0.414, pass@20
  0.547 vs 0.586. J-lens sharpens the rank-1 readout but does not expand the
  top-20 shortlist — the same "replicates, but less clean than Claude" flavor
  Nanda reported on Qwen. Two harness bugs were found and fixed en route
  (readout position off-by-one for prompts ending right before the target).

## 3 · Findings vs paper vs Qwen replication

| Measure | Claude (paper) | Qwen 3.6 27B (Nanda) | OLMo-3-32B-Think (this lab) |
|---|---|---|---|
| Lens recipe | 1000 prompts | 25 Pile prompts, penultimate target | 120 WikiText prompts, 21 layers, target L63 |
| Active concepts / position | ~10–25 | not headline ("less clean" overall) | median 6 @θ=0.01; 3 @θ=0.02; k90≈4 |
| J-space variance share | 6–10% | — | ≤0.67% in band (~10× thinner) |
| Layer localization | band ≈33–92% depth | CKA bands "less clean" | inverted-U, peak L32–42 (50–66%); persistence 0.19–0.20 vs 0.03–0.08 at ends |
| Broadcast (fan-out) | yes, workspace is read widely | — | vs chance: strong (73–94 readers vs 0). vs non-J high-variance: **not dissociated** (77–125) |
| Causal dissociation | top-10 ablation kills multi-hop, spares shallow/fluency | interventions flagged as possibly confounded | **null at our doses**: static J ≈ random everywhere; non-J PCs cause the selective damage; dynamic top-10 = uninterpretable collapse |
| Workspace ↔ CoT | identity claim; CoT partially rescues ablation | poetry/arithmetic evals failed | pre-CoT null; intra-CoT workspace leads text by 46 steps; answer-time loading under suppression (rank 211 vs 5613); suppression costs 0–13 pp accuracy |

## 4 · Phase detail

### 4.1 Descriptive geometry (s5, s6 → f1, f2, f8)

Sparse nonnegative gradient-pursuit decomposition onto the row-normalized
(W_U⊙g)·J_l dictionary, 200 diverse prompts (factual QA, arithmetic, code/SQL,
prose). Raw dictionary energy, active-concept counts, and top-1 persistence all
peak in the same mid band (L36–42; persistence peak L34–36), with early/late
control layers flat — the paper's qualitative signature. Quantitatively the
workspace is far thinner than Claude's: peak variance share 0.67% vs 6–10%,
median active concepts 6 vs ~25. Broadcast: each band layer's top J-directions
are read (input-projection alignment z>3) by 73–94 later components vs 0 for
matched random directions — but matched-dimension top *non-J* PCs are read by
77–125, so wide readout is a property of high-variance directions generally,
not of verbalizable ones specifically.

### 4.2 Causal ablation (s7 → f4, f5)

Battery: 2-hop probe-swap (n=60), chained arithmetic (n=30, v2 scorer — v1's
two-line window was format-biased, baseline 0.47→0.87 after fix), 3-table SQL
join-key checks (n=30), vs one-hop facts, grammatical minimal pairs, held-out
prose NLL. Conditions per dose k∈{10,20,40} dims/layer over band L20–44:
static top-k J-span; matched random subspace; matched top non-J PCs; plus
paper-faithful per-token dynamic top-10. 2000-resample bootstrap CIs.

| condition | twohop | twohop_lp Δ | arith(v2) | sql | onehop | grammar | prose NLL |
|---|---|---|---|---|---|---|---|
| none | 0.58 | — | 0.87 | 0.67 | 0.60 | 1.00 | 2.71 |
| jspace_dyn10 | 0.00 | −23.0 | 0.00 | 0.00 | 0.00 | 0.30 | 24.34 |
| jspace_k20 | 0.52 | −0.17 | 0.93 | 0.67 | 0.60 | 1.00 | 2.85 |
| random_k20 | 0.52 | −0.08 | 0.90 | 0.67 | 0.60 | 1.00 | 2.75 |
| nonJ_pca_k20 | 0.28 | **−0.86** | 1.00 | **0.00** | 0.70 | 1.00 | 3.45 |

(k=10/40 rows follow the same pattern; full grid in
`metrics/ablation_results.json`.) Generation audit (stored verbatim under each
condition's `samples`): under `jspace_k*` arithmetic and SQL are intact; under
`nonJ_pca_k*` SQL loses the formal register entirely — *"SELECT the user's city
and the user's country, and the user's name, and the user's id of the thing,
and…"* — while arithmetic survives; under `jspace_dyn10` output degenerates to
asterisks. Reading: at these doses, the causal load OLMo carries in
high-variance directions is generic (register/format, mild NLL), and removing
the *verbalizable* subspace specifically is no worse than removing random
dimensions. The paper's headline dissociation does not replicate here; the one
instrument that produced it (dynamic top-k of *currently active* directions)
removes the live computation itself and cannot distinguish "workspace" from
"whatever the model is computing right now" (Nanda's confound, reproduced).
The 7B smoke's single-item J-specific effect (−2.45 vs +1.22 logprob) did not
generalize to the 32B battery.

### 4.3 Thinking-model angle (s8, s8b → f6)

90 items (40 two-hop, 30 chained arithmetic, 20 SQL join-key), chat template
with native open `<think>`; readout dictionaries at L24/32/40 during greedy
generation (≤400 think tokens); suppressed variant closes `</think>`
immediately.

- **Pre-CoT anticipation: null.** At the final prompt token the answer's
  median J-rank is 5613 (10% of items ≤20); logit lens 11610 (0%). The J-lens
  advantage over logit lens is real (≈2–4× in the 20–26 band) but neither holds
  the answer. The paper-flavored claim "the workspace already contains the
  answer before reasoning" is not supported on this model.
- **Intra-CoT anticipation: strong.** The answer enters some read layer's
  top-8 in 82–100% of items during thinking, and on traced items (n=34) the
  workspace leads the first textual statement of the answer by a **median 46
  steps** (twohop 48.5, sql 38, arithmetic 9.5; workspace first in 91%).
- **Answer-time loading.** With thinking suppressed, the same readout at the
  (identical-question) answer position gives median rank 211, 23% ≤20 — a ~27×
  improvement over the open-`<think>` position from closing one tag. Per-layer
  medians collapse monotonically L30→L60: 2048→1549→519→305→118→63→10→6, with
  J-lens ahead of logit lens through L28–40 (e.g. L36: 519 vs 840). The
  workspace is populated on demand at answer time, not precomputed at prompt
  end — consistent with the substrate-sharing claim but not with anticipation.
- **Suppression cost:** twohop 0.95→0.82, arithmetic 1.00→0.90, SQL 1.00→1.00
  (tasks near ceiling; deltas compressed).
- **Faithfulness probe:** 116 stable-workspace-word-absent-from-CoT events
  across 59/90 items, only 4 matching the final answer. The recurring divergent
  words are discourse adverbs — `famously` (top-1 for 7–10 consecutive steps
  while the CoT recalls e.g. *"Springfield, Massachusetts is known for being
  the birthplace of bask[etball]"*), `respectively`, `etc` (during enumeration
  runs). We find no case of "workspace holds X, CoT argues Y, answer is X" —
  instead the workspace tracks something like the *discourse mode* of the
  current reasoning span. Top-20 events verbatim: `metrics/cot_lead.json`.

### 4.4 Phase 5 stretch — eval-awareness probe (s10, kept modest)

20 tasks rendered neutrally vs with an explicit "formal evaluation, you are
being tested and scored" prefix; J-readout differencing at the final prompt
token over L{20,24,28,32,36,40,44}. A consistent direction emerges far beyond
the pre-registered criterion: sign-flip permutation p<0.001 at every layer,
mean pairwise cosine of paired deltas 0.48–0.71, mean-delta norm growing
monotonically with depth (22→134). The direction is semantically legible: from
L32 upward its top decoded tokens are the evaluation lexicon (`evaluation`,
`evaluated`, `assessment`, `assessing`, `eval`); at L24–28 the associates are
more oblique (`assessed`, `monitored`, `administrative`, `internally`,
`careful`). Mean h-space shift ≈ 0.28–0.34 of typical ‖h‖.

The gated mini-ablation ran (criterion met): projecting the per-layer mean-Δh
direction out during 120-token generations on 6 eval-framed tasks changed
essentially nothing — no test/eval mentions appear in either condition's
thinking, and outputs are near-identical up to occasional mild rephrasing.

Interpretation, honestly bounded: because the framing text literally contains
evaluation vocabulary, a final-token readout difference of this shape is
exactly what a context-verbalizing workspace *should* show (a positive lens
demonstration), and cannot by itself distinguish "lexical echo of the prompt"
from a latent being-tested state. The inert ablation favors the echo reading
at these layers/doses. The confound-breaking version (implied-but-unstated
eval cues) is queued in §7. Data: `metrics/evalaware.json` (top-30 tokens
both directions per layer, permutation stats, verbatim generation pairs).

## 5 · Figures

- `f1_variance_share_by_layer.png` — J-space variance share vs layer, paper band shaded.
- `f2_active_concepts_by_layer.png` — active-concept medians, θ ∈ {0.01, 0.02, 0.05}.
- `f3_lens_vs_logit_rank.png` — probe-answer rank by layer, J-lens vs logit.
- `f4_ablation_dissociation.png` — accuracy ± CI per condition × task.
- `f5_dose_response.png` — dose–response, multi-step vs fluency panels.
- `f6_precot_anticipation.png` — pre-CoT answer-rank ECDF, J-lens vs logit.
- `f8_broadcast_fanout.png` — downstream readers per direction group.
  (No f7 — number reserved during planning, never assigned.)

All regenerable from `metrics/` alone via `scripts/s9_report.py`; machine
headline numbers in `report/summary.json`.

## 6 · Limitations

1. **Dictionary linearization.** Concepts read through (W_U⊙g)·J_l — RMSNorm
   gain only; the final block's full nonlinearity is not inverted.
2. **Layer subset.** Fitted band 20–44 (33–70% depth) plus sparse controls; the
   paper's workspace extends to ~92%. The suppressed-CoT profile (answer rank
   still collapsing L44→60) suggests real late-band structure we did not fit.
3. **Threshold commensurability.** "6 vs paper's ~25 active concepts" depends
   on our θ convention on normalized-dictionary correlations; the variance-share
   comparison (0.67% vs 6–10%) is the firmer capacity statement.
4. **Control mismatch.** Non-J PCs are top-variance but not variance-*matched*
   to the J-span; their larger causal effect partly reflects more energy
   removed. A variance-matched control is the single most important follow-up.
5. **Dynamic-ablation confound.** Per-token top-k removal deletes live
   computation (asterisk collapse); our static spans avoid this but may miss
   position-specific workspace structure. A frozen prompt-selected top-k is the
   clean middle instrument and was not run.
6. **Corpus & scale.** WikiText-103 (matches neuronpedia reference recipe), not
   Dolma; 120 fitting prompts vs paper's 1000 (convergence was measured, but
   rare-concept coverage may differ).
7. **Multi-token answers** tracked via min-rank over case/space variant first
   tokens; 1-char first-token fragments degrade a few items.
8. **CoT ceiling effects.** Think-mode accuracy 0.95–1.00 compresses the
   suppression contrast; divergence detector only sees single-token top-1 runs
   of word-like tokens (≥3 alpha chars, ≥5 steps).
9. Single seed (0); modest ns (30–60/task); band ablation only (no single-layer
   dose mapping).
10. **Phase 5 semantic-echo confound.** The eval framing states the cue words
    outright, so the (very significant) readout direction may verbalize the
    prompt rather than a latent state; its ablation was behaviorally inert.
    Treat §4.4 as a lens demonstration, not evidence of eval-awareness.

## 7 · What to run next

1. Variance-matched non-J control for s6/s7 (match removed energy per layer) —
   decides whether the broadcast/causal story is about energy or content.
2. Frozen prompt-selected top-10 ablation during generation (dynamic selection,
   static application) — the confound-free version of the paper's intervention.
3. CoT-rescue test: rerun static ablations under think-mode prompting (paper
   claims externalization rescues; our suppression asymmetry predicts it here).
4. Fit the late band (L46–62) and redo s5/s8 readouts there — the suppressed
   profile says the answer materializes precisely where we stopped fitting.
5. Dolma-corpus lens vs WikiText lens dictionary comparison (OLMo's actual
   pretraining distribution; scientific-legibility angle of this replication).
6. `famously`-attractor forensics: is discourse-adverb lock-in a lens artifact
   or a real mode representation? Cross-check on neuronpedia's Qwen lenses.
7. Paper's capacity and directed-modulation experiments (prompt sets already in
   `jacobian-lens/data/experiments/`).
8. Confound-broken eval-awareness: implied-but-unstated test cues (e.g. "your
   answer will be compared against other assistants'" vs a synonymous neutral
   sentence, no eval lexicon) — does the §4.4 direction still appear? Then
   ablate during a task where eval-framing measurably shifts behavior.

## Read next

1. `report/summary.json` — headline numbers, machine-readable.
2. `metrics/ablation_results.json` — full causal grid incl. verbatim samples.
3. `metrics/cot_lead.json` — lead stats, per-layer anticipation, top-20
   divergence events verbatim.
4. `code/special_lab1/PLAN.md` + `LOG.md` — decisions, pins, dead ends.
5. `figures/f4`, `f5`, `f6` — the three figures the verdict rests on.
6. `metrics/evalaware.json` — Phase 5 direction tokens + inert-ablation pairs.
