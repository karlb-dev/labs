# PR #9 — Lab 37: J-space Global Workspace Replication (final)

> Suggested PR body — `gh` is not authenticated on the lab VMs, so paste
> this over the PR description when convenient. The branch itself is
> final; everything below is in-repo.

## What this PR adds

A complete draft lab (Lab 37) replicating Anthropic's July 2026 global-
workspace paper on `allenai/Olmo-3-32B-Think` with the reference
`jacobian-lens` (pinned), plus an instrument-audit round (v2) and a
cross-model leg on `Qwen/Qwen3.6-27B` using Neuronpedia's published lens.
Layout follows the Lab 36 special-case mold: spec + claim ledger in
`labs/lab37_jspace_workspace.md`, everything else under `jspace/`
(README, code mirror s0–s23, living LaTeX handout + PDF, reports, figures,
small metrics as `results/v{1,2}_*.json`; heavy artifacts on Drive).

## Headline results

- **Geometry replicates; capacity is a real model property.** OLMo's
  J-space carries 0.61–0.72% of residual variance (~6 active concepts);
  the same harness reads Qwen3.6-27B at 4.3–6.8% with paper-range
  active-concept counts (paper/Claude: 6–10%, 10–25). A dedicated
  late-band lens (L46–62) rules out "the workspace was just past the
  fitted band" on OLMo.
- **The paper's static causal dissociation does not survive clean
  instruments on either model.** At per-layer energy-matched doses,
  J-span = matched-random = matched-non-J = baseline on OLMo (k≤40) and
  on Qwen (k=20). v1's apparent non-J selectivity was an energy artifact.
- **A control-clean causal handle exists and transfers:** freezing each
  item's top-10 J-directions deletes the retrieved fact on both models
  (answer logprob −2.9 nats OLMo / −2.4 Qwen; random-dictionary twins on
  baseline; fluency intact). On OLMo it deletes 1-hop and 2-hop recall
  alike (content channel); on Qwen the composed task is hit much harder
  (0.87→0.37) than 1-hop (0.90→0.83) — the closest either model came to
  the paper's dissociation signature, flagged as the sharpest follow-up.
- **The workspace leads the chain of thought** by a foil-calibrated
  median 46 steps (mid-band) / 49.5 steps (independent late-band lens);
  frequency-matched detector floor 0.06 vs 0.92 on answers. Pre-CoT
  anticipation stays null under every lens; answers load on demand.
- **CoT-rescue (P5):** externalized reasoning largely bypasses the
  frozen deletion — the same projectors that cut silent two-hop recall
  to 0.23 leave think-mode recovery at 0.80 (control 0.93; one-hop fully
  rescued), and frozen-J halves the `</think>`-closure rate. The paper's
  rescue prediction holds in content-channel form.
- **Seed-1 robustness (P6):** the frozen effect replicates on a second
  seed with fresh, never-before-used 2-hop items (`REPORT_v2.md` §P6 for
  the grid).

## Verdict

The v1-vs-paper disagreement was **instruments all along**: with energy
matching and matched-live controls, both models agree — static span
removal does nothing; per-item frozen selection deletes retrieved
content. The capacity difference (10×) between OLMo and Qwen/Claude is
real and now measured under a single harness.

## Reading order

1. `jspace/handout/olmo32b_jspace_handout.pdf` — the primary writeup
2. `jspace/report/REPORT_v2.md` — v2 verdict + claims table
3. `labs/lab37_jspace_workspace.md` — spec, claim ledger SL1-C1..C7,
   promotion checklist
4. `jspace/report/REPORT.md` — v1 baseline run

## Status / promotion

Proposed for promotion to canon: promotion criteria (1)–(3) met
(instrument audit landed, claims instantiated, Qwen leg run); (4) human
label pass over divergence + eval-awareness generations remains open and
is marked as such in the spec.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
