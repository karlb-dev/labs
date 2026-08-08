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

> RESOLVED at launch (2026-08-08): `qwen-n1000` resolves to Hub commit
> `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`, whose lens file has LFS
> sha256 `1718c8c5…11e1` and size 3,303,032,772 — **byte-identical to the
> frozen pin** (equivalence per addendum §2.3). Recorded verbatim in
> `configs/qwen_lens_manifest.json`; science never branches on it.

## D5 — capacity `targets_per_family` carries unused families (pre-data)

The released file lists 8 families in `targets_per_family` but only 4 in
`block_families`/`candidate_pools` (colors/animals/body/clothes have no
pools). Frozen reading: the 4 pooled `block_families` are the experiment;
the extra keys are vestigial release metadata. Recorded, not "fixed".

## D7 — lens-eval `target` marks the continuation, not a substring (pre-data)

Verified against the pinned bytes: in the three targeted eval sets the
`target` is the expected next word — prompts end immediately before it
(93/107/55 items; incidental substring collisions only, e.g. "15" inside
an arithmetic expression). "The token immediately preceding `target`"
therefore = the **final prompt token**. Frozen positions: multihop /
multilingual / order-ops / typo / association → final token; poetry →
last newline token. RENDER_AND_POSITION_CONTRACT §4 corrected pre-data.

## D8 — order-ops synonym expansion table (pre-data)

The README states the rule ("numbers → digit and word forms; operations →
symbol and word forms") but not the table. Frozen in
`jspace_official_repro/targets.py`: digits 0–25 ↔ English words;
addition {addition, plus, +} · subtraction {subtraction, minus, -} ·
multiplication {multiplication, times, *, ×} · division {division,
divided, /, ÷} · mod {mod, modulo, %} · squared {squared, square, ²}.
Rank = min over single-token members (space + bare variants). R1.

## D6 — modulation `group` → `group_kind` collapse (pre-data)

`phrasings[*].group` ∈ {focus, mention, dismissal, negated-think};
`group_kind` maps these to {focus, control, suppress, suppress}. The
contrast is computed over the collapsed kinds per the README; per-group
curves retained.
