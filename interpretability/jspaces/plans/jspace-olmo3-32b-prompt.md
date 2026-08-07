# Claude Code Prompt — special-lab-1: J-space Replication on OLMo 3 32B (Thinking)

Paste everything below the line into Claude Code from the `interpretability/` folder of the labs repo.

---

## Mission

Replicate the core findings of Anthropic's July 2026 "Verbalizable Representations Form a Global Workspace in Language Models" paper on **OLMo 3 32B (the thinking/reasoning variant)** — the exact HF model id is whatever this repo already uses for the newest, largest OLmo thinking model; find it in the existing lab configs and reuse it verbatim rather than guessing.

Primary references (fetch and read before writing any code):
- Paper: https://transformer-circuits.pub/2026/workspace/index.html (especially `#methods-jlens`)
- Research post: https://www.anthropic.com/research/global-workspace
- Reference implementation: https://github.com/anthropics/jacobian-lens — clone it, read the README and `walkthrough.ipynb` end to end. Use their `jlens` package as the lens engine; do not reimplement the lens from scratch.

Context you should know: Neel Nanda already replicated the core phenomena on Qwen 3.6 27B, so a clean OLMo 3 32B result is a genuinely new data point — and because OLMo is fully open (weights + training data), any workspace structure we find is more scientifically legible than in closed models. Treat this as a real replication attempt, not a demo.

## Lab structure and conventions

- This is **special-lab-1**. Create it as a new lab directory following the exact structure, config style, runner harness, metrics collectors, and reporting conventions of the existing labs in this `interpretability/` folder. Study 2–3 of the most recent labs first and mirror them.
- **Do NOT** update the repo README, any lab index/registry files, or anything that would register this as an official lab. Do not `git add` or commit anything. This lab lives outside version control (I'll archive it to Google Drive myself). If the repo has tooling that auto-registers labs, bypass it.
- You **may** modify the shared lab infra (metrics collectors, harness, plotting utilities) where needed to capture the new metric types below — make those changes surgical and backward compatible with existing labs.

## Output location

- All analysis outputs go under `/content/drive/MyDrive/interpret/special-lab-1/<run_id>/` where `<run_id>` is a timestamp like `2026-07-25_1430`. Create the run directory at start.
- Inside the run dir: `config/` (frozen config, seeds, `pip freeze`, GPU info), `lens/` (fitted lens checkpoints), `metrics/` (JSON/parquet per experiment), `figures/`, `logs/`, `report/`.
- **Checkpoint everything incrementally.** This is a Colab VM; assume it can disconnect at any time. Every phase must be resumable: check for existing artifacts in the run dir before recomputing, and save partial results as you go (after every fitting chunk and every ablation batch, not just at phase end).

## Phase 0 — Environment audit and feasibility (do this before anything else)

1. Inventory the VM: GPU model, VRAM, RAM, disk, CUDA/torch versions. Write to `config/environment.json`.
2. Compute the memory budget honestly. The J-lens **fit requires backward passes through the full model**, which for a 32B model is far heavier than inference. Plan for:
   - bf16 weights + gradient checkpointing during fitting;
   - fitting the lens only on a **layer subset** (the paper found workspace structure concentrated in the middle block — target roughly the middle third of layers, plus a few early/late layers as controls);
   - the jlens README's guidance that lens quality saturates quickly: start with **~100–200 fitting prompts** of 128 tokens, not the paper's 1000; use `jlens.fit()` on disjoint slices and `JacobianLens.merge()` to parallelize/chunk;
   - if backward through 32B genuinely does not fit even with checkpointing, fall back in this order: (a) reduce sequence length, (b) fit on fewer layers, (c) 8-bit weights if jlens tolerates it, (d) as a last resort run the whole pipeline on the smaller OLMo thinking model in the repo's configs as a pilot and report the constraint clearly. Do not silently downgrade — record which tier we landed on in the report.
3. Smoke test: before touching the 32B, run the full pipeline end to end on the smallest model already configured in this repo's labs (fit a tiny lens on ~20 prompts, one ablation, one figure). Only proceed to 32B once the pipeline is proven.

## Phase 1 — Fit the lens

- Fitting corpus: pretraining-like text per the jlens README. If this repo already has a corpus utility, use it; otherwise pull a small slice of a standard open pretraining corpus (Dolma is thematically perfect for OLMo).
- Fit per the reference implementation, save the lens under `lens/`, and record fitting metrics: prompts used, layers covered, wall time, loss/quality curves, VRAM peak.
- Lens sanity check: run the README's "currency of the country shaped like a boot" style probe plus 10 similar single-token-answer factual prompts. At mid layers the lens top-1 should surface the answer concept before the final layer does. If the lens output is pure noise, stop and debug before proceeding.

## Phase 2 — Descriptive replication: does a J-space exist in OLMo 3 32B?

Paper targets to compare against: ~25 active concepts, <10% of activation variance, concentrated in middle layers.

For a diverse prompt set (~200 prompts spanning factual QA, multi-step arithmetic, code/SQL completion, and open-ended prose — build it in `config/prompts/`, reusing repo prompt infra where it exists):
1. **Sparsity**: at each (position, layer), how many lens concepts are meaningfully active (define an activation threshold; report sensitivity to it)? Distribution across prompts; compare median active-concept count to the paper's ~25.
2. **Variance share**: fraction of residual-stream activation variance captured by the verbalizable subspace, per layer. Plot the layer profile; test for the mid-layer concentration the paper reports.
3. **Broadcast**: for top J-space directions, measure downstream fan-out — how many later-layer components read from them (attention/MLP input projections with significant alignment) — versus matched random directions and matched non-verbalizable high-variance directions. The paper's claim is broadcast, not point-to-point.
4. **Verbalizable vs merely present**: compare J-lens readouts against a plain logit-lens baseline at the same layers to show what the Jacobian weighting adds.

## Phase 3 — Causal replication: ablation dissociation

The paper's headline causal result: ablating the workspace collapses multi-step reasoning while fluency survives.

1. Task battery (reuse/extend repo eval infra):
   - **Multi-step**: 2–3 hop factual chains, chained arithmetic, and — our domain twist — multi-clause SQL generation tasks requiring schema tracking (e.g., correct join keys across 3 tables).
   - **Fluency controls**: next-token perplexity on held-out prose, simple single-hop completions, grammaticality.
2. Interventions, each vs. matched controls (random subspace of equal dimension; top non-verbalizable variance directions):
   - project out the J-space at mid layers during generation;
   - graded ablation (25% / 50% / 100% of identified directions) for a dose-response curve.
3. Success pattern to test for: multi-step accuracy drops substantially under J-space ablation but not control ablation, while fluency metrics stay near baseline under both. Report effect sizes with bootstrap CIs, not just point estimates.

## Phase 4 — Thinking-model novel angle: workspace vs. chain-of-thought

This is the part nobody has published and why the thinking variant matters. OLMo 3 32B Think emits explicit reasoning tokens; the paper's identity claim says verbalizable workspace content and silent reasoning share substrate. So:
1. **Pre-CoT anticipation**: on reasoning tasks, read the J-space at the final prompt token *before any thinking tokens are generated*. How often does the eventual answer (or key intermediate, e.g., the correct join column) already appear in the workspace? Log rank trajectories across layers.
2. **CoT faithfulness probe**: during thinking-token generation, compare workspace contents against the stated reasoning at aligned positions. Flag divergences — cases where the workspace holds concept X while the CoT discusses Y, especially when the final answer matches X. Quantify divergence rate; save the top-20 most divergent examples verbatim for the report.
3. **Suppressed-CoT comparison**: run matched tasks with thinking disabled/short-circuited (whatever mechanism the model supports) and compare workspace richness and answer accuracy vs. full-CoT runs. Does the workspace carry more load when the model can't think out loud?

## Phase 5 — Stretch (only if time/compute remain): evaluation-awareness probe

Construct matched prompt pairs (identical task; one framed neutrally, one framed with explicit eval/test cues). Difference the J-space readouts to look for a "being tested" direction. If one emerges, run a small ablation on a behavioral task to see if outputs shift. Keep this modest — it's the flakiest part of the original paper and Nanda flagged the interventions as possibly confounded.

## Metrics infra updates

Extend the lab metrics collectors to capture, as first-class metric types: per-(position,layer) lens readouts (top-k tokens + activations, compressed), active-concept counts, variance-share-by-layer curves, ablation deltas with CIs, fan-out scores, CoT-divergence events. Keep raw tensors out of gdrive except where needed for the top divergent examples; store aggregates.

## Report

Generate `report/REPORT.md` in the style of the repo's existing lab reports, containing: environment + feasibility tier landed on; lens fit quality; a findings table with OLMo 3 32B numbers side by side against the paper's Claude numbers and Nanda's Qwen 3.6 27B numbers (active concepts, variance share, layer localization, ablation dissociation, plus our CoT results); all figures; an honest limitations section (lens noise, false positives, chunked fitting, layer subset, anything downgraded in Phase 0); and a "what to run next" list. Also emit `report/summary.json` with the headline numbers for machine consumption.

## Working style

- Read the paper's methods section and the jlens walkthrough *before* writing code; don't code from the summary above alone.
- Plan first: write `PLAN.md` in the lab dir mapping phases to scripts before implementing, and keep a running `LOG.md` of decisions, surprises, and dead ends.
- Prefer many small resumable scripts over one monolith. Seeds fixed and recorded everywhere. Every figure regenerable from saved metrics without re-running the model.
- If a paper claim can't be tested at our compute tier, say so explicitly in the report rather than substituting a weaker proxy silently.
