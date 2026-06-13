# Lab 15 - Multi-Turn Instrumentation

## Question

Can we trust turn-indexed internal measurements in a chat conversation?

This lab is not trying to prove a claim about persona, belief revision, roleplay, or self-report. It is a harness lab. Later labs will make multi-turn claims, so this lab checks the plumbing that those claims ride on:

- chat-template-aware turn segmentation;
- user/assistant token-span boundaries;
- KV-cache boundary reads versus full recompute;
- self-patching a turn-boundary state as a no-op;
- topic traces with random and length-matched null controls.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab15 --tier a
python interp_bench.py --lab lab15 --tier b
```

Useful while debugging:

```bash
python interp_bench.py --lab lab15 --tier a --no-plots
```

Lab 15 uses instruct models and chat templates. Tier A uses `HuggingFaceTB/SmolLM2-135M-Instruct`; Tier B uses `allenai/Olmo-3-7B-Instruct`.

## What It Does

The lab renders two scripted harmless conversations:

- `orchid_topic`: an accumulating orchid-greenhouse planning conversation;
- `archive_length_control`: a length- and structure-matched folder/archive conversation.

For each conversation, the lab derives message spans by rendering chat-template prefixes with the tokenizer's own template. It then traces three directions:

- `topic_orchid_minus_archive`;
- `random_null`;
- `length_matched_null`.

The topic trace is a demo. The graded object is the self-check stack.

## Main Artifacts

| Path | What it contains |
|---|---|
| `diagnostics/turn_boundary_check.json` | template parity, span coverage, no-gap, no-leak checks |
| `diagnostics/cache_recompute_parity.json` | KV-cache boundary states versus full recompute |
| `diagnostics/cache_recompute_parity_by_boundary.csv` | per-boundary parity details |
| `diagnostics/patch_noop_check.json` | self-patching a turn-boundary stream is a no-op |
| `diagnostics/null_trace_check.json` | finite-slope sanity check for topic/null traces |
| `diagnostics/null_trace_slopes.csv` | projection slopes by conversation and direction |
| `tables/turn_projection_trace.csv` | per-message span, boundary, and cumulative projections |
| `plots/demo_turn_trace.png` | topic trace with null controls |
| `multiturn_harness_report.md` | human-readable self-check report |
| `operationalization_audit.md` | what this harness does and does not license |
| `results.csv` | alias of `tables/turn_projection_trace.csv` |

## Evidence Discipline

The ledger claim is `OBS`, and only about instrumentation:

> The harness reproduces turn-boundary residual projections under KV-cache prefill versus full recompute to tolerance epsilon.

Do not write:

- "The model has a persistent persona."
- "The model changes its mind over turns."
- "The orchid trace proves semantic accumulation."

Those are later-lab claims. Lab 15 only says the measurement apparatus is or is not trustworthy enough to try them.

## Writeup Questions

1. Did template string tokenization match direct chat-template tokenization?
2. Did prefix-derived spans cover the full rendered prompt without gaps?
3. What was the maximum cache-vs-recompute residual difference?
4. What was the maximum logit change under self-patching?
5. Did the random and length-matched null traces look flat enough to make the topic trace interpretable?
6. Name one later multi-turn result this harness could fake if the cache parity check were removed.

## Common Failure Modes

If template parity fails, stop. The lab is measuring a different prompt than generation sees.

If assistant text appears inside a user span, every "user-turn state" claim is contaminated.

If cache parity fails, cached turn traces may be cache artifacts rather than prefix states.

If self-patching changes logits, the hook target is not the named residual stream.

If null traces drift, later persona or belief-revision traces need stronger length/template controls before they get claims.
