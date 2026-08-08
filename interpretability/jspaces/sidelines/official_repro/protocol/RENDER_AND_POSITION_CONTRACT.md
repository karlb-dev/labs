# Render and position contract (frozen pre-data)

Governs every prompt render and every scored token position. Addendum §3.3
folded in. The rendered-token manifests produced under this contract are
hashed into each evidence event.

## 1. Prompt-form routing (plan §6.1)

- **Raw string** (all six lens evals; `probe-swap` prompts;
  `flexible-generalization` templates; `ignition` carriers; `capacity`
  lists; `selectivity-linecount` assembled texts): the UTF-8 string
  verbatim as completion text. No system prompt, no chat tokens, no
  generation prefix. Tokenizer BOS policy per §3.
- **Role dictionary ending in a user turn** (`verbal-report`,
  `selectivity-language` questions, `top-down-summoning` questions,
  `dual-task`/`directed-modulation` instructions before prefill): render
  through the exact checkpoint tokenizer chat template with
  `add_generation_prompt=True`.
- **Role dictionary ending in an assistant prefill**
  (`verbal-introspection` — final assistant turn is the (empty) released
  turn plus the released prefill; `directed-modulation` and `dual-task`
  carrier copies; `selectivity-linecount` prefill conditions): render with
  `continue_final_message=True`. The open quote / partial sentence stays
  open; the assistant turn is never closed and reopened.
- **Fallback** (addendum §3.3): if the pinned `transformers` revision
  cannot `continue_final_message` for a lane, manually concatenate
  template-rendered history + verbatim prefill, byte-diff-proven against
  the native path on a supported configuration before use. Recorded here
  if ever exercised.

## 2. Qwen mode

Role-dict cells use the checkpoint's official direct/non-thinking template
mode. The rendered token stream must contain **no** automatically opened
thinking span; the exact template kwargs (`enable_thinking=False` or the
revision's actual mechanism — recorded as a fact at admission) and chat
template file hash go into the render manifest. Qwen thinking-on is
outside Study 1.

## 3. BOS policy

Upstream `HFLensModel` forces `add_bos_token=True` when the tokenizer has
that attribute. Study rule: raw-string prompts use the upstream default
(force_bos on); chat-template renders use exactly the tokens the template
produces (no second BOS injected — asserted per lane at admission by token
dump). The per-lane observed BOS behavior is recorded in the render
manifest as fact.

## 4. Position rules per set

Let `ids` be the rendered token IDs; "final position" = `len(ids) - 1`.

| Set | Scored / intervention positions |
|---|---|
| lens evals with `target` (multihop, multilingual, order-ops) | readout at the **final prompt token** — the pinned bytes show `target` is the expected continuation, so the prompt's last token *is* "the token immediately preceding target" (D7) |
| lens-eval-poetry | readout at the last newline token in `ids` |
| lens-eval-typo / association | readout at the final position |
| probe-swap | baseline + swap answer scored at the final position (prompt "ends just before the answer"); swap applied at **every prompt position** |
| verbal-report | scored at the final rendered position — the open-model mapping of the paper's "final colon" (the colon belongs to the raw-format `Assistant:` marker, which chat templates realize as the generation prompt; divergence D1). Swap applied at every prompt position |
| flexible-generalization | scored at final position; swap at every prompt position |
| verbal-introspection | scored at the last prefill token (the open quote); steering added over **every token of the second user turn** (the "Trial 1" question — the released rule says "the user's question turn", and the questionless first turn is setup; divergence D2 records this reading), at every band layer |
| selectivity-language | label tokens tracked over the question-token span (tokens after the passage, i.e. the rendered question suffix following `{text}`) |
| selectivity-linecount | number-canon tokens tracked at any prompt position in the band |
| directed-modulation / dual-task | tracked over the teacher-forced carrier response span (the assistant-prefill tokens) |
| ignition | readout at the `{W}` token position (its final token if multi-token; single-token pairs preferred per canon rules) |
| top-down-summoning | readout fraction over the stimulus span; swaps at every stimulus position; answers scored at the final position |

## 5. Span location method

Spans (question, stimulus, carrier-response, assistant-prefill) are
located by rendering the pre-span and full texts and matching **token
subsequences**, never by character offsets alone. Every location is
asserted: decoding the located span must reproduce the span text modulo
the frozen whitespace normalization (§6).

## 6. Whitespace normalization for target-token matching

A candidate token matches a released target string iff
`decode([token]).strip() == target.strip()` under NFC, case-sensitive,
where `strip()` removes ASCII whitespace only. The leading-space variant
(`" France"`) and the bare variant (`"France"`) are both derived; the
**in-context** variant (what the tokenizer produces where the target
would appear) is preferred for intervention vectors; both are recorded.

## 7. Generation cells

Only cells whose released spec samples a continuation generate
(`top-down-summoning` causal answers; `selectivity-language` automatic
continuation control if scored by generation — it is not: it is scored by
lens readout over question tokens, so no generation; the linecount
`continue` condition scores lens readout only). Generation is greedy
(`do_sample=False`), max 24 new tokens, recorded verbatim.

## 8. Position audit (per schema × lane, before science)

Five fixed sentinel items per distinct JSON schema: render, save text +
token IDs, prove the scored position decodes to the expected boundary,
assert no truncation (`max_seq_len` 2048 for renders; fit uses upstream
128), hash into the render manifest. Truncated-render rows would be
`TOKENIZATION_GATED`, never silently truncated.

## 9. Hook-fire accounting

Every intervention forward counts hook fires per (layer, position); the
count must equal the frozen plan for that cell (plan §15 stop rule;
addendum §3.1 batching proof). Batch renders may group same-shape items
only when per-item fire counts remain provable.
