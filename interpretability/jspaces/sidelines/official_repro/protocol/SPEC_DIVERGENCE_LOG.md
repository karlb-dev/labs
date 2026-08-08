# Spec divergence log

Every ambiguity between the plan, the addendum, the pinned upstream
bytes, and the paper — resolved **before** the affected outcome, never
silently. Append-only; entries numbered D1….

## D1 — verbal-report "final colon" under chat templates (pre-data)

The pinned experiment README scores "the model's greedy next token at the
final `:`". In the paper's raw Human/Assistant format that colon is the
`Assistant:` marker's colon — i.e. the last prompt token before the
model's reply. Open-model chat templates realize the same boundary as the
generation prompt (e.g. `<|im_start|>assistant\n`). **Frozen mapping:**
scored position = final rendered token under
`add_generation_prompt=True`; the swap-out token is the greedy next token
there. R1, recorded in RENDER_AND_POSITION_CONTRACT §4. No literal ":"
token is required to exist.

## D2 — introspection "user's question turn" (pre-data)

The released `intro_prompt` has two user turns (setup; "Trial 1: Do you
detect an injected thought? …"). The README says steering is applied over
"every token of the user's question turn". **Frozen reading:** the second
user turn (the trial question). The setup turn is not steered. R1.

## D3 — released concept count 101 vs paper n=100 (pre-data)

`verbal-introspection.json` ships 101 concepts; the paper reports n=100.
All 101 released concepts run; no row is dropped to match the paper
count; figures state n=101 (released).

## D4 — `qwen-n1000` Hub revision label (pre-data; addendum §2.3)

The pinned walkthrough sets `LENS_REVISION = "qwen-n1000"`. The stronger
identity — Hub commit `a4114d7…` + sha256 + byte length — is primary. At
lens download the study additionally attempts to resolve `qwen-n1000`
and records the commit it points to or the exact resolution error here:

> RESOLVED at launch (2026-08-08): see
> `configs/qwen_lens_manifest.json` → `qwen_n1000_ref_resolution`.
> Outcome recorded there verbatim; science never branches on it.

## D5 — capacity `targets_per_family` carries unused families (pre-data)

The released file lists 8 families in `targets_per_family` but only 4 in
`block_families`/`candidate_pools` (colors/animals/body/clothes have no
pools). Frozen reading: the 4 pooled `block_families` are the experiment;
the extra keys are vestigial release metadata. Recorded, not "fixed".

## D6 — modulation `group` → `group_kind` collapse (pre-data)

`phrasings[*].group` ∈ {focus, mention, dismissal, negated-think};
`group_kind` maps these to {focus, control, suppress, suppress}. The
contrast is computed over the collapsed kinds per the README; per-group
curves retained.
