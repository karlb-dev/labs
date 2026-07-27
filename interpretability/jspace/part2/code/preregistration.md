# Preregistration — J-space Part 2 model-matrix campaign

> **STATUS NOTE (2026-07-27 evening): superseded for confirmatory
> purposes** by `REPAIR_PREREGISTRATION.md` following the forensic review
> (`jspace_part2_plan1_addendum.md`). This file remains the unaltered
> record of what was committed before any Part-2 data collection. Known
> supersessions: the A0 pass rule was a fit-shortcut heuristic (transfer
> is a geometry experiment; recipient fits are mandatory — addendum
> §18.4); B1's fixed cosine/Jaccard/nats thresholds await pilot-calibrated
> equivalence margins (§18.8); the §6-derived stats standards are replaced
> by addendum §12 (paired clustered inference, equivalence tests, Holm
> primary family, power simulation); "occupancy index" is renamed
> output-alignment (§18.12); model pins updated to the 3.1 primary pair
> (§18.5). The scientific preregistration will be authored after gates
> G0–G6 pass.

Committed to git **before any Workstream A–D data collection** on the pinned
models (the A0 gate runs only after this file's first commit). Campaign plan:
`PLAN_PART2.md` (= the user's `jspace_interp_part2_plan1.md` + launch
addendum). Part-1 evidence base: `jspace/report/REPORT_v2.md` (repo, merged
PR #9). Date: 2026-07-27. Author of record: karlb-dev; agent-executed.

## Pinned models (hub-checked 2026-07-27)

| role | HF id | notes |
|---|---|---|
| donor / part-1 anchor | `allenai/Olmo-3-32B-Think` | 64L, d5120, vocab 100278; base `allenai/Olmo-3-1125-32B` |
| A1 matched-pretraining instruct | `allenai/Olmo-3.1-32B-Instruct` | same base + arch + tokenizer as donor (model card); `Olmo-3-32B-Instruct` does not exist |
| A2 | `Qwen/Qwen3.6-27B` | no separate non-think dense sibling ships in the 3.6 line; A2 = same-model matrix completion (think cells, C3, C2, D, robustness) |
| A3 | `google/gemma-4-31B-it` | third architecture family; adaptation gate first |
| optional H1 anchor | `allenai/Olmo-3-1125-32B` | shared base; raw-completion cells only; run on slack or A1 flip |

## Hypothesis ladder (verbatim from the campaign plan §0)

- **H1 Externalization**: think-trained models put the workspace in tokens;
  the static dissociation lives in non-externalizing models. Discriminator:
  Workstream A.
- **H2 Occupancy**: the live verbalizable subspace in our models is occupied
  by the output stream itself, not deliberative content; dissociation
  requires workspace/output separation. Discriminator: Workstream D.
- **H3 Training-lab specifics**: unfalsifiable from outside; only shrinkable
  by exhausting H1/H2/H4/H5.
- **H4 Scale**: weakened already (Qwen has paper-range capacity, still null);
  state as untestable at our tier.
- **H5 Instruments**: largely closed by v2; residuals are task battery
  (Workstream C) and dose/selection (B4).

## Directional predictions and decision rules

- **A0 transfer gate.** PASS ⇔ recipient-with-donor-lens sanity within ~15%
  relative of the donor's own gate: probe hits@20 ≥ 15/21 (donor 17/21) AND
  multihop J-lens pass@1 ≥ 0.85 × donor's 0.283 = 0.240 (n=60, same items).
  Transfer semantics: donor J, recipient unembedding (the way a transferred
  lens would actually be used). Secondary (reported, non-gating): does J-lens
  still beat the recipient's own logit lens? Weak prior: PASS (post-training
  shifts the readout basis less than it shifts behavior). Either outcome is a
  standalone mini-result. PASS ⇒ A1 runs on the transferred lens (fit saved);
  FAIL ⇒ fit a 120-prompt Instruct lens with the part-1 recipe.
- **A1 (the H1 discriminator).** If H1, the static dissociation — or at
  minimum a one-hop/two-hop frozen asymmetry AND a nonzero static
  dose-response — appears or strengthens on Instruct relative to Think;
  descriptive capacity may shift. If Instruct is as null as Think, H1 is
  substantially weakened and H2/H3 rise. **Flip rule (binding):** if A1 flips
  the causal result, immediately pull B4 + C1-load forward on Instruct before
  spending anywhere else.
- **A2.** Under H1 the already-visible Qwen one-hop/two-hop asymmetry
  (0.90→0.83 vs 0.87→0.37) sharpens toward the paper's full dissociation and
  survives the C3 hard one-hop set (kills the ceiling confound).
- **A3.** Under H1, Gemma-non-think patterns with A1/A2 independent of
  architecture; a Gemma-specific deviation instead implicates architecture
  and is its own finding. Gate failure recorded as `GEMMA_BLOCKED_<reason>`.
- **B3 frozen-logit control.** If frozen-logit ≈ frozen-J on fact deletion,
  the Jacobian pullback is not doing causal work on these models and the
  method claim narrows honestly; if markedly weaker, the J-space framing
  earns its name causally. Required column in every Core Battery grid.
- **B1 fit-size decision rule (verbatim).** If directions are stable (median
  per-token cos > ~0.9 vs the 120-lens), frozen top-10 selections overlap
  (Jaccard ≥ 0.7 per item), and the frozen effect size moves < 0.5 nats under
  the 500-lens, the 120-prompt recipe is validated and every prior result
  inherits the robustness; otherwise the 500-lens becomes canonical and
  affected v1/v2 cells are rerun and flagged.
- **B4.** Flat null through k∈{80,160} at matched energy AND under the
  persistence-selected span (the last principled static selection rule)
  closes the dose/selection escape hatches.
- **C1 load interaction.** If the J-space is functionally a working set,
  behavioral capacity curves (accuracy vs k∈{2,3,4,6}) decline more shallowly
  on Qwen (fat workspace) than OLMo (thin); and any workspace that matters
  only under load shows static/persistence-selected ablation damage on
  high-load items specifically — this is the experiment that flushes out or
  closes the paper's dissociation.
- **D occupancy index.** Occupancy (rank of the imminently-emitted token in
  the live top-k J-readout; cosine of its dictionary direction to the top-k
  span) is high on think models and lower on any model where a dissociation
  appears; across the matrix, occupancy correlates inversely with
  static-ablation sensitivity.

## Statistics standards (every new cell)

n ≥ 60 on headline two-hop cells; 2 seeds on decisive grids; bootstrap CIs
everywhere; greedy + one temperature-0.7 replicate of frozen grids; BH-FDR
across the model×instrument matrix at campaign end; every cell carries
{model, instrument, condition, task, dose, n, seed, decoding, CI, provenance}
in `report/matrix_master.{csv,json}`.

## Priority order and drop rules (binding)

A0 → B3 → A1 → C3 → C1 → D → A2 → B1 → C2 → B4 → A3 → B2 → E, executed with
model-set disk grouping (ops only; see PLAN addendum — grouping reorders GPU
residency, not the bank-complete-cells rule). Never thin every workstream to
fit a budget; bank complete cells in order and state exactly where the money
ran out. Budget: user cap ≤200 h; plan estimate 45–55 GPU-h; minimal
publishable set = A0 + A1 + B3 + C1(load×ablation, OLMo pair) + C3 + D + the
A2 frozen grid.
