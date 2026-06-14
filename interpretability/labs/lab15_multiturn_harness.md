# Lab 15: Multi-Turn Instrumentation Harness

**Evidence level targeted:** `OBS`, but only for instrumentation. This lab does not make a model-science claim about persona, belief, roleplay, memory, capitulation, eval awareness, or self-report.

## The question

Can we trust turn-indexed internal measurements in a chat conversation?

Single-turn activations already have sharp edges. Multi-turn chat adds more little trapdoors: chat templates add role tokens, rendered prefixes can shift, user and assistant spans can be off by one, cached-prefix states can differ from full recompute, and a projection can rise just because the prompt got longer. Lab 15 exists so later labs do not build castles on a token-boundary bog.

The demo topic is deliberately harmless: an orchid-greenhouse planning conversation and a matched archive-folder control. The orchid trace is not the result. The self-check stack is the result.

## What this lab validates

| Instrument piece | What is checked | Why later labs care |
|---|---|---|
| Chat-template rendering | rendered-string tokenization matches direct `apply_chat_template(..., tokenize=True)` | Later labs must measure the same prompt the model sees. |
| Turn segmentation | prefix-derived message spans cover the full rendered prompt with no gaps | A turn-indexed state is meaningless if the span is shifted. |
| Content segmentation | content spans are mapped separately from template/role tokens | A “user content state” is not the same object as a “message span including separators.” |
| Assistant-generation boundary | user-ended prefixes with `add_generation_prompt=True` are checked | Generation-time reads often happen after an assistant header, not after raw user text. |
| Exact-chat hook parity | block hooks are compared against `streams[k+1]` on already-rendered chat token IDs | This catches BOS drift and stream-depth off-by-one errors. |
| KV-cache parity | cached prefix states match full recompute at turn boundaries | Cache artifacts can counterfeit multi-turn drift. |
| Self-patching no-op | replacing a boundary block output with the same vector leaves logits unchanged | Cross-turn patching later only means anything if this hook target is the named stream. |
| Null traces | topic projection is compared with length/template and random null directions | A climbing projection is not useful if the nulls climb too. |

## Stream and layer convention

The shared bench uses this residual-stream convention:

```text
streams[k] = pre-norm residual stream after k decoder blocks
streams[0] = embedding output
streams[L] = final norm input
```

A decoder block with index `layer` writes `streams[layer + 1]`. Lab 15 records `layer` and `stream_depth` together in the patch no-op artifacts so later cross-turn patching does not inherit the classic stream-depth off-by-one error.

## Run

From the course root:

```bash
python interp_bench.py --lab lab15 --tier a
python interp_bench.py --lab lab15 --tier b
```

Useful while debugging:

```bash
python interp_bench.py --lab lab15 --tier a --no-plots
python interp_bench.py --lab lab15 --tier b --allow-hook-mismatch
```

Lab 15 requires an instruct/chat model with a tokenizer chat template. In the course registry, Tier A should use a small instruct model such as `HuggingFaceTB/SmolLM2-135M-Instruct`; Tier B should use the standard advanced-course instruct model such as `allenai/Olmo-3-7B-Instruct`.

The lab file also writes `diagnostics/bench_integration_note.json`. If that file says `bench_chat_template_labs_has_lab15 = false`, add `lab15` to the bench’s `CHAT_TEMPLATE_LABS` set so `diagnostics/tokenizer_info.json` tells the same story as the lab.

## What it does

The lab renders two scripted conversations:

- `orchid_topic`: a harmless accumulating orchid-greenhouse planning dialogue;
- `archive_length_control`: a role- and structure-matched archive-folder dialogue.

For each conversation, it derives token spans by repeatedly rendering chat-template prefixes. It records:

- message spans, including role/template scaffolding;
- content spans, mapped through tokenizer offset mapping or exact token-subsequence fallback;
- assistant-generation prompt boundaries for user-ended prefixes;
- rendered-text and token-ID hashes.

It then measures boundary states in two ways:

1. full recompute of the whole prefix;
2. incremental prefill with `past_key_values`.

Those residual streams must match within tolerance. The lab then self-patches representative boundary states into the same forward pass. That should be an identity operation. If it is not, later cross-turn interventions are not standing on named ground.

Two real-template subtleties this lab is built to handle, both of which show up the moment you leave the CPU smoke model:

- **Content spans come from the full render, not from prefix renders.** Some chat templates close the *final* assistant turn with a different stop token than a non-final one — Olmo-3-Instruct uses `<|endoftext|>` for the last turn and `<|im_end|>` once another turn follows. So re-rendering a prefix is *not* a byte-prefix of the full conversation, and prefix-derived character offsets drift. The lab therefore locates each message's content in the one string the model is actually scored on and validates the span by decoding it back to the message text. Incremental prefix byte-identity is recorded as a diagnostic, but it is **not** a trust gate — failing it on Olmo is correct template behavior, not a bug.
- **Cache parity is dtype-aware.** Full recompute and cached prefill are bitwise identical only in fp32 (measured max relative-L2 error 6e-6 here). In bf16 the two attention kernels round differently and a single outlier dimension can diverge while the rest of the vector agrees, giving ~2–4% relative-L2 error and cosine ≈ 0.999. The gate is therefore the whole-vector relative-L2 error with a dtype-aware threshold, not a per-dimension absolute max: a real off-by-one in the cache window perturbs the *whole* vector (relative-L2 ≳ 0.3), so it still fails loudly, while harmless bf16 rounding passes.

Finally, the lab builds demonstration directions at every stream depth:

- `topic_orchid_minus_archive`: mean direction from short orchid-vs-archive chat contrasts;
- `length_matched_null`: mean direction from neutral table/ledger contrasts;
- `random_null_00` through `random_null_07`: seeded random unit directions.

It traces these directions over message boundaries, content boundaries, content means, and cumulative prefix means. It also sweeps stream depth so students can see how easy it is to pick a pretty layer after the fact.

## Artifact tree

```text
runs/lab15_multiturn_harness-<timestamp>-<id>/
  run_summary.md
  multiturn_harness_report.md
  multiturn_harness_card.md
  operationalization_audit.md
  ledger_suggestions.md
  metrics.json
  results.csv

  diagnostics/
    bench_integration_note.json
    rendered_conversation_preview.csv
    turn_boundary_check.json
    think_span_report.csv
    generation_prompt_boundary_check.csv
    generation_prompt_boundary_check.json
    chat_exact_hook_parity.json
    chat_exact_hook_parity_by_layer.csv
    chat_exact_lens_self_check.json
    cache_recompute_parity.json
    cache_recompute_parity_by_boundary.csv
    patch_noop_check.json
    patch_noop_sites.csv
    trace_direction_manifest.json
    trace_direction_depth_manifest.csv
    null_trace_check.json
    null_trace_slopes.csv

  tables/
    turn_segments.csv
    generation_prompt_boundaries.csv
    turn_projection_trace.csv
    trace_depth_sweep.csv
    trace_direction_cosines.csv
    harness_evidence_matrix.csv
    boundary_diagnostic_matrix.csv
    trace_slope_summary.csv
    downstream_readiness_card.csv
    plot_reading_guide.csv

  plots/
    harness_evidence_dashboard.png
    harness_evidence_matrix.png
    downstream_readiness_card.png
    demo_turn_trace.png
    trace_depth_sweep.png
    trace_slope_ledger.png
    depth_selection_atlas.png
    turn_span_map.png
    generation_boundary_audit.png
    cache_patch_diagnostics.png
```

## Start here

Open `plots/harness_evidence_dashboard.png` first. It is the run cockpit: self-check verdicts, content/template load, numeric parity gates, and null-trace pressure in one view. Then open `multiturn_harness_card.md` for the same verdict in prose.

Next inspect `tables/harness_evidence_matrix.csv` and `plots/harness_evidence_matrix.png`. These tell you which downstream permission each gate earns. The plot is not decorative: it decides whether later labs may trust turn-indexed projections, cache-efficient traces, generation-time reads, or cross-turn patching.

Then inspect `tables/boundary_diagnostic_matrix.csv`, `tables/turn_segments.csv`, and `plots/turn_span_map.png`. Confirm that user and assistant content spans make sense. Pay special attention to `content_span_method`: `offset_mapping` is ideal, `token_subsequence_fallback` is acceptable, and broad message-span fallback is a warning that later labs should avoid content-specific claims until fixed.

Next open `plots/generation_boundary_audit.png`, `tables/generation_prompt_boundaries.csv`, and `diagnostics/generation_prompt_boundary_check.json`. These tell you where the assistant-generation prompt boundary lands after a user message. That boundary is not always the same thing as “the last token of the user’s raw text.” Tiny hinge, giant door.

Then inspect `plots/cache_patch_diagnostics.png`, `diagnostics/cache_recompute_parity_by_boundary.csv`, and `diagnostics/patch_noop_sites.csv`. These are the boring diagnostics that matter. If cache parity or patch no-op fails, the projection plot is not interpretable.

Finally inspect `plots/demo_turn_trace.png`, `plots/trace_slope_ledger.png`, and `plots/depth_selection_atlas.png`. Read them as instrumentation demos only. If the topic trace rises but the null traces rise too, the correct conclusion is not “orchid state found.” The correct conclusion is “future multi-turn labs need stricter length/template controls.”

## Main diagnostics

| Artifact | What to look for |
|---|---|
| `diagnostics/turn_boundary_check.json` | Template parity, stable prefixes, complete span coverage, content spans, generation prompt checks, and no assistant leakage into user content spans. |
| `tables/turn_segments.csv` | Message spans versus content spans, boundary tokens, and template-token load. |
| `tables/generation_prompt_boundaries.csv` | User-ended prefixes rendered with `add_generation_prompt=True`. |
| `diagnostics/chat_exact_hook_parity.json` | Decoder-block hooks match assembled `streams[k+1]` on exact chat-rendered token IDs. |
| `diagnostics/chat_exact_lens_self_check.json` | Final-depth lens parity on the exact rendered chat prompt. |
| `diagnostics/cache_recompute_parity.json` | Maximum residual and logit differences between cached prefix reads and full recompute. |
| `diagnostics/patch_noop_check.json` | Maximum logit change after self-patching the same boundary vector back into the model. |
| `diagnostics/null_trace_check.json` | Whether null/control slopes are finite and whether there is a null-drift warning. |
| `diagnostics/bench_integration_note.json` | Whether the registry and chat-template lab set know about Lab 15. |

## How to read the plots

`plots/harness_evidence_dashboard.png` is the start-here plot. Panel A shows the self-check stack. Panel B shows whether the topic trace is carried by content or by template scaffolding. Panel C shows cache, patch, and hook numeric gates against their tolerances. Panel D asks whether null traces are trying to impersonate topic accumulation.

`plots/harness_evidence_matrix.png` and `plots/downstream_readiness_card.png` convert diagnostics into permissions: what later labs may inherit, what is blocked, and what needs caution.

`plots/demo_turn_trace.png` compares the orchid topic direction against the archive control conversation, the length/template null, and random-null band. Lines show cumulative prefix means; content-boundary markers show the narrower readout. Treat the figure as a trace-readout rehearsal, not a semantic result.

`plots/trace_slope_ledger.png` sorts all trace slopes by direction family. It is the quick way to see whether a random or length/template null is too loud for the demo trace to be comforting.

`plots/trace_depth_sweep.png` and `plots/depth_selection_atlas.png` show how topic and null slopes vary over stream depth. They train suspicion: if you choose the depth after seeing the curve, the depth choice is now part of the hypothesis and needs its own held-out check in later labs.

`plots/turn_span_map.png` shows where system, user, assistant, message, and content spans land in the rendered token sequence. This is the map that saves later labs from saying “user state” when they measured role scaffolding.

`plots/generation_boundary_audit.png` shows how many tokens the assistant-generation header adds after user-ended prefixes. `plots/cache_patch_diagnostics.png` shows cache-vs-recompute parity and self-patching no-op errors. These two plots are the tiny hinge pins for later generation-time probes and cross-turn interventions.

## Visualization upgrade

The upgraded Lab 15 plot suite treats the harness as a measurement instrument with its own evidence ladder. The old three plots remain, but they now sit behind a stronger audit board:

- `harness_evidence_dashboard.png`: one-screen verdict for self-checks, span load, numeric parity gates, and null drift.
- `harness_evidence_matrix.png`: which diagnostic licenses which downstream use.
- `downstream_readiness_card.png`: ready/blocked/caution statuses for turn traces, cached traces, generation-time reads, patching, and semantic drift claims.
- `boundary_diagnostic_matrix.csv`: joined per-segment spans, cache parity, patch no-op, and generation-boundary data.
- `trace_slope_summary.csv`: topic, control, length-null, and random-null slopes with risk flags.

The point is to make later Labs 16, 17, 22, 24, and 25 inherit an explicit contract instead of a visual-smoothness permission slip.

## Evidence discipline

The only default ledger claim is an instrumentation claim:

```text
[L15-C1] OBS | For model M, the Lab 15 harness segments chat-template turns,
validates cached boundary reads against full recompute to tolerance epsilon,
and self-patches representative boundary states as a no-op. This licenses later
multi-turn measurement under the same tokenizer/model/template conventions.
```

Do not write:

```text
The model has a persistent persona.
The model changed its mind over turns.
The orchid trace proves semantic accumulation.
The assistant remembers the orchid topic internally.
```

Those are later-lab claims. Lab 15 only decides whether the rails are straight enough to run the train.

## Writeup questions

1. Did rendered string tokenization match direct chat-template tokenization for every conversation?
2. Did content spans make sense for system, user, and assistant messages?
3. What is the difference between a message boundary, a content boundary, and an assistant-generation prompt boundary?
4. What was the maximum cache-vs-recompute residual difference, and at what boundary did it occur?
5. What was the maximum logit change under self-patching, and which layer/site produced it?
6. Did random or length/template null traces drift enough to trigger the warning?
7. At which stream depths did topic and null slopes become largest, and why is that dangerous for later science claims?
8. Name one later multi-turn result this harness could fake if each self-check were removed.

## Common failure modes

If template parity fails, stop. You are measuring a different prompt than the model’s chat interface uses.

If content spans are missing or fall back to whole message spans, later labs should avoid content-specific claims until they fix span mapping for that tokenizer.

If assistant text appears inside a user content span, every user-turn state claim is contaminated.

If generation-prompt boundaries fail, downstream generation-time reads may be shifted by the assistant header.

If cache parity fails, cached turn traces may be cache artifacts rather than prefix states.

If self-patching changes logits, the hook target is not the named residual stream.

If null traces drift, future persona, sycophancy, eval-awareness, or belief-revision traces need stronger length/template controls before they get ledger claims.

## How later labs should consume Lab 15

Lab 16, Lab 17, Lab 22 in multi-turn mode, Lab 24, and Lab 25 should inherit this contract:

1. render with the tokenizer’s chat template;
2. derive spans from prefix renders, not string guesses;
3. distinguish content spans from message/template spans;
4. distinguish message-end states from assistant-generation-prompt states;
5. verify cached reads against full recompute on representative boundaries;
6. self-patch a boundary state as a no-op before cross-turn patching;
7. include random and length/template null traces beside every exciting projection;
8. pre-register or hold out any stream-depth choice used for a science claim.

A later lab can still be exploratory without this checklist, but it should not write a strong ledger claim.

## Extensions

A manageable extension is a per-head attention-back-to-prior-turn visualization. Label it `OBS`: attention back to an earlier user turn is routing evidence, not proof that the content was used.

An ambitious extension is cache-efficient cross-turn patching: patch a turn-2 boundary state into a turn-4 forward pass, then verify a self-patch and a wrong-turn control before any behavioral claim. That is the first step toward the belief-revision interventions in Lab 24.
