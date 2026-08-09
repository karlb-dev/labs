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

## D9 — g-folding audit is MATERIAL on Qwen; paper-literal unfolded v_t
stays primary (decided before any scientific intervention outcome)

The contract §5 audit on the Qwen lane returned min cosine **0.3821**
(worst cell: layer 29, token 57867) over 26 battery tokens × 14 band
layers — far below the 0.99 immateriality bar. Decision, logged before
any registered intervention outcome: the **paper-literal unfolded**
``v_t = J_ℓᵀ u_t`` (paper §2.1 defines v_t from W_U rows; the RMSNorm
gain belongs to final_norm, not W_U) remains the one primary
intervention basis, exactly as the contract froze it. Folding materiality
is recorded as a lane fact and explains part of any probe-vs-readout
divergence; a folded-v_t variant is a permissible after-core sensitivity
under the plan §14 drop order (first candidates to drop), not a second
primary. The smoke admission (scratch, unregistered) surfaced this;
the registered admission carries the same audit values.

**OLMo addendum (2026-08-09, logged before any OLMo intervention):**
the OLMo lane audit returned min cosine **0.9886** — marginally under
the 0.99 bar, i.e. essentially immaterial, in sharp contrast to Qwen's
0.382. The same decision applies: paper-literal unfolded ``v_t`` is the
one primary on both lanes. The Qwen-vs-OLMo folding asymmetry is
recorded as a lane fact and is itself OR-Q4 evidence: the campaign's
g-folded ablation dictionaries and the paper's unfolded swap basis are
nearly the same object on OLMo but materially different objects on
Qwen.

## D12 — modulation/ignition pairing dimensions (pre-data)

The release fixes the item sets but not the trial-pairing counts.
Frozen deterministic rules: directed-modulation pairs phrasing i ×
target j with carrier ``(i+j) mod 20`` (all 24 phrasings × all 46
targets, one carrier each); ignition runs the complete released pair ×
α grid (α ∈ {0.0,0.1,…,1.0}) with carrier rotation ``pair_index mod
len(carriers)`` plus a 5-carrier variability subset on the 12-country
pairs. R1 (content exact, pairing reconstructed).

## D13 — carrier-task instruction templates (pre-data)

Single-task templates are taken verbatim from the released
``jlens/examples.py`` (``Write "{carrier}" {instruction} Don't write
anything else.`` with assistant prefill = carrier). Dual-task combined
clauses ("X and Y while you write the sentence") are a reconstruction —
the release ships conditions and pair keys but no combined phrasing. R1.

## D10 — selectivity-linecount assembly and hit-k (pre-data)

The README fixes content and order ("question before the wrapped passage,
ends with the matching prefill") but not separators or the tracked-set k.
Frozen: ``question + "\n\n" + wrapped + "\n\n" + prefill`` (no leading
separator when the question is empty; `continue` = ``explicit_q + "\n\n"
+ wrapped``); hit at rank 1 primary (the conventions' default), rank ≤5
recorded sensitivity. R1.

## D11 — introspection steering normalization scope (pre-data)

"Scaled by the layer's mean residual norm" — corpus- vs prompt-level is
unspecified. Frozen: per (prompt, layer) mean over the steered-turn
positions from the clean forward. R1.

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
