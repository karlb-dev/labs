# Implementation addendum — severance experiment guide v2

**How to use this.** Attach this alongside the v2 guide for the build. The guide
is correct and shovel-ready; this file is the layer it doesn't carry: the
implementation traps that silently corrupt results, the places where a faithful
literal build produces garbage that looks like data, and a few interpretation
cautions for the headline tracks. Items are tiered. **Tier 1 will produce
wrong numbers that look real if skipped — do these first.** Tier 2 changes what
a result means. Tier 3 is engineering hygiene and cost.

---

## Tier 1 — Must verify before trusting any number

### 1.1 Add a KV-replay parity test (the missing parity check for B4)

B4 is now the headline, and its KV-cache teacher-forced replay (§13.3) is the
single most corruption-prone piece of code in the project. The guide has hook
parity and no-op parity; it has **no replay parity**, and a broken replay
silently produces plausible garbage.

**Required test, before any B4 run:** for a fixed prompt + answer + attribution
question, compute logits two ways — (a) one full forward pass over the
concatenated `input_ids`, and (b) the incremental replay (prefill → token-by-
token answer with `past_key_values` → append attribution question). With **no
injection**, the attribution-position logits must match within bf16 tolerance
(≤2e-2). If they don't, the cache stepping is wrong and every B4 number is
meaningless. Run this on OLMo and again on gpt-oss (MoE cache behavior differs).

Three things that break replay parity in practice:
- **Position IDs / `cache_position` desync.** When stepping one token at a time
  with a cache, position ids must stay contiguous across prefill → answer →
  question. Assert the attribution question's first-token position equals
  `prefill_len + len(canonical_answer_ids)`. A desync corrupts the KV silently.
- **Cache class.** Newer `transformers` may default to a static/preallocated
  cache that does not append incrementally the way the pseudocode assumes. Pass
  an explicit `DynamicCache` (or the model's documented incremental cache) and
  verify round-trip of `out.past_key_values`.
- **Hook lifetime.** The pseudocode creates an `ActivationAdder` *inside* the
  per-token loop. Creating/removing a forward hook every token is slow and leaks
  handles if any step raises. Register one hook with a toggled `active` flag (or
  a step-counter closure) and remove it in a `finally`.

### 1.2 Resolve yes/no and label token IDs at runtime — do not hard-code

Every logit-margin metric in the guide — B5 `logit(" yes") - logit(" no")`, the
patch-recovery `logit(target_marker_first_token)`, confidence parsing — assumes a
clean single-token mapping. Tokenizers disagree: `" yes"`, `"yes"`, `"Yes"`,
`" Yes"` may be different tokens, multi-token, or vary between OLMo and gpt-oss.

**Required:** at load time, resolve the actual token id(s) per tokenizer for
each label, check the leading-space and capitalization variants the chat
template will actually emit, and assert single-token or handle multi-token
explicitly (first-token logit or sequence score). Log the resolved ids once per
model. This is a classic silent bug: a wrong id makes a metric read a near-zero
logit and every condition looks null.

### 1.3 Verify the injection/capture position is the token you think it is

The "report_query_position = final token before first generated token" (§16.5)
is computed after the chat template adds assistant-role/generation-prompt tokens.
Position `-1` of your rendered text may be a role header token, not your
prompt's last content token, so you'd inject into the wrong place.

**Required:** for each chat template, render one example, decode the token at the
intended capture/injection index, and log it. Confirm it is the content token you
mean (e.g., the last token of the report query, or the concept span), not a
template artifact. This is the self-report analogue of the Lab 12 token-alignment
check — do it once per template, per model.

### 1.4 Batch-invariance is a discipline, not a setting

bf16 + flash-attention + batching means the **same** prompt yields different
logits at batch-size 1 vs inside a batch (non-associative reductions, kernel
selection). §5.5/§8.4 acknowledge this; the concrete rule the coder must follow:

**Compute every two conditions you compare directly under identical batch shape
and padding.** Either batch-size 1 for all compared conditions, or the same batch
with identical left-padding. Never compare a metric computed in a batch-of-8 to
one computed batch-of-1. The parity tests (1.1, hook parity) must run at the
same batch config as the science runs, or they certify a config you don't use.

### 1.5 Express dose in residual-RMS units, and normalize the direction once

Two latent inconsistencies that make dose-response uninterpretable:
- `ActivationAdder` adds `alpha * d` with **raw** `d` (difference-of-means), but
  `project_out` (§15.2) and the direction-math test unit-normalize. Pick one:
  **unit-normalize the direction everywhere**, so `alpha` has consistent meaning
  across construction, injection, and ablation.
- Residual norms grow with depth and vary by token, so a fixed `alpha` means
  wildly different perturbation strengths across layers, concepts, and models.
  **Express dose as a multiple of the per-(layer,position) residual RMS**, and
  log `injected_norm / residual_norm` for every condition. Without this the
  cross-layer dose-response and the gpt-oss-vs-OLMo comparison (§20.5) compare
  incommensurable quantities. The B5 perturbation-size calibration (§14.5) is the
  right idea applied locally; make it the **global** dose convention.

---

## Tier 2 — Changes what a result means

### 2.1 B4 canonical answer must be a high-probability default completion

Teacher-forcing a token the injected model "didn't want" to emit creates a
prediction-error signal in the KV that the model may later detect. That's a real
form of introspection, but if the canonical answer is *also* non-preferred under
`matched_default`, both conditions carry forcing-surprise and the
activation-vs-matched contrast muddies.

**Fix:** choose `canonical_answer` to be a natural, high-probability completion
under the **default** (no-injection) model, so only the activation condition
introduces an internal perturbation. Add a check: the default model's mean
log-prob on the canonical answer should be high (it's a plausible continuation),
and log it.

### 2.2 Interpret a B4 positive precisely — source-ID vs prediction-error

A B4 positive shows the model can attribute to a hidden factor when output is
matched. But the *mechanism* may be generic prediction-error detection ("my
next-token distribution was surprised during that span") rather than identifying
the cause as an injection specifically. Both are introspection; they are not the
same claim. The report template's "source-aware" line should distinguish
"detected an internal anomaly" (weaker) from "identified the anomaly's source"
(stronger), not collapse them. The fresh-transcript control separates
introspection from text-priors; it does **not** separate these two.

### 2.3 Inject at a mid layer for B4 so the trace reaches downstream KV

Injection is applied to the block **output** (after block k), so the perturbation
enters the cached KV only at layers **> k**. Inject too late (near final) and
there's little downstream KV to carry the trace into the attribution question's
attention, weakening B4 for reasons unrelated to whether the channel exists.
**Pick the B4 injection layer mid-stack** (not the latest decodable layer), and
note this rationale so the coder doesn't reuse the B2 "earliest plateau" layer
uncritically.

### 2.4 B3: confidence and entropy must be measured under identical injection

The entropy-dissociation test (§12.5) compares reported confidence against answer
entropy. These come from two prompts (report vs behavior), and the **injection
condition must be identical across both** — same layer, dose, position — or you're
correlating confidence and entropy measured under different interventions. State
this. Also: compute entropy on the **answer** tokens of the behavior generation,
and exclude rows where the confidence string is unparseable (NaN-and-hand-label,
don't coerce).

### 2.5 gpt-oss-120b is a reasoning model — it is not a clean scale target

This is the one the guide under-flags. gpt-oss uses harmony formatting with
analysis/final channels — i.e., it reasons. So it has the **same CoT-laundering
confound as the OLMo Think models**, and by the guide's own logic it belongs on
the reasoning axis, not as a clean scale target for the introspection claim. The
self-report must be read from the **final** channel; the analysis channel is a
confound, not free signal.

Consequence to state plainly so Karl doesn't over-read a gpt-oss result: there may
be **no** clean non-reasoning open model at 100B+ scale, which means the clean
introspection claim is effectively capped at 32B-Instruct, and any 120B number is
entangled with reasoning. A "gpt-oss stronger than OLMo-Instruct" result is then
confounded (scale and reasoning move together) and cannot attribute the gain to
scale. Treat the 120B run as reasoning-axis-at-scale, not clean-scale.

### 2.6 Be explicit about what each seed varies

Greedy decoding is deterministic given model + input, so re-running greedy with a
new `torch.manual_seed` produces **identical** generations — it is not an
independent sample. "3 seeds" meaningfully varies only the **direction-fit
subsample and the split assignment**. Document that, so "headline holds across
seeds" is understood as robustness to the fitting sample, not to generation
noise. Generation-noise robustness lives entirely in the temperature-sampling
panel (§18.4).

---

## Tier 3 — Engineering, reproducibility, cost

### 3.1 Fresh-transcript control: enforce the exact-token invariant

The B4 fresh-transcript control (§13.3) is only valid if the model sees the
**identical token sequence** for the attribution question in both paths; only the
KV *provenance* should differ (incrementally built with injection vs freshly
computed). It is easy to render them differently — e.g., the replay wraps the
answer as mid-generation assistant tokens while the fresh path re-wraps it in a
fresh assistant turn with role headers. **Assert: concatenated `input_ids` are
identical across replay and fresh-transcript paths.** Hash the full token
sequence, not just the answer text.

### 3.2 Pin the semantic judge; never self-judge; gate on agreement

The judge is co-primary (§11.5, §18.5). Make it reproducible and honest:
- Pin judge model + revision + prompt + `temperature=0`; store raw judge outputs.
- **Do not use a model under test as its own judge** (self-preference, shared
  blind spots). Use a separate model.
- Compute Cohen's κ between judge and the blind human labels **before** trusting
  it. If κ is below threshold, fall back to human-primary for that family rather
  than reporting the judge number. The guide mentions reliability; make the
  fallback mechanical.

### 3.3 Enforce anti-forking by construction, not by intention

§6.1 freezes tuning before heldout. Make it impossible to cheat: the heldout
split lives in a file the selection code **cannot read**, selection functions
receive only train/val `item_id`s (assert this), and heldout evaluation is a
single CLI call that loads frozen configs from a manifest. A coder will otherwise
write one convenience function that sees everything and quietly peeks.

### 3.4 B5: add a demand-characteristics control

"Did anything seem like an unusual inserted internal signal?" is a suggestive
question; an RLHF'd model may lean "yes" under any salient condition. The
clean-trial false-alarm gate catches most of it, but add a **neutral-framing
variant** of the question (or flip the polarity on a fraction of trials) to
separate "detects anomaly" from "says yes when asked to look for one." Report
both framings.

### 3.5 Residual storage: cast for math, store small, stream to disk

Patchscope captures and patch-recovery save residuals across the layer grid ×
items × conditions; at d_model ~5–8k on 32B/120B this is large. Store bf16/fp16,
cast to fp32 only for the metric math, save only selected layers/positions, and
stream to disk rather than accumulating captures in RAM. Easy OOM otherwise.

### 3.6 Make the negative result a first-class artifact

The most likely clean outcome is "potent injection, behavior moves, B4/B5 flat" —
functional shallowness — and the pipeline must render its tables and plots with
the same prominence as a positive. The **single most important artifact for a
negative is the behavioral-potency table** (§17.3): a flat B4/B5 means nothing
without it, and means a lot with it. Ensure a flat headline still auto-generates
the full "why this is not X" report sections, not an empty run.

---

## One-line build order delta

The guide's build order is fine. The only reordering this addendum forces:
**implement and pass the KV-replay parity test (1.1) before writing any B4
scoring**, and **resolve label token ids (1.2) and verify positions (1.3) before
any logit-margin metric**. Those three are the difference between a B4/B5 number
that means something and one that is an artifact of plumbing. Everything else can
follow the guide's first-week plan.
