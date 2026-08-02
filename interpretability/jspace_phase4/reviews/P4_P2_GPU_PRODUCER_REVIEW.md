# P4-P2 GPU producer review

**Status: engineering implementation audit complete; development-only;
execution is not yet authorized.**

Reviewed 2026-08-02 before any P4-P2 phase-intervention pilot outcome was
generated. This review does not constitute independent protocol review, PI
approval, a freeze commit, or a freeze tag.

## Disposition

The registered module
`jspace_phase4.experiments.p4_qwen_mode_variance_pilot` remains the frozen
protocol and pure analyzer. The new, unexecuted producer
`jspace_phase4.experiments.p4_qwen_mode_variance_gpu` now implements the GPU
path without mutating that methods event. The initial implementation was
banked at commit `41e23e7`; the review-contract successor has producer and
lower-level hook SHA-256 values
`961b392a4659a23cd5489a24ef298660138295434c9c889783460f2b360092b9`
and
`fe3483447dee3d476b42e2da23dfb3e036f6292f57ce2d8e7c0278a492c3d6e7`.
Its still-unbound config SHA-256 is
`22346644ffe5c6d2234132367f4b7fb3060cbf35c9a6a3627d4c301891c23b37`;
that hash must change when the two A1000 placeholders are lawfully bound.

The implementation binds the live protocol, passing v2 baseline, exact model
revision and template/parser contract; refuses any lens other than a
registered A1000 canonical decision on Q-L1/Q-L2; loads exactly the consumed
20-family subset; and emits the exact 160-row grid. It maintains distinct
clean and intervened cache streams, computes clean top-10 plus all accepted
alias-piece protection at each owned predictor position, derives the J rank
and energy profile on the same hidden state used by its matched control, and
records unpooled per-layer/per-position geometry. Phase ownership is checked
against an explicit position mask, including the reasoning-end delimiter
position that predicts the first thinking-on answer token. Missing hooks,
wrong-phase ownership, overlap, rank mismatch, energy mismatch, and protected
span leakage are hard failures.

The producer also records exact prompt and completion token sequences,
per-token parser phases, deterministic replay sentinels, arm-specific
teacher-forced accepted-answer log probability, immutable row/profile parts,
and a manifest-bound restart state. A resumed run refuses code, config, model,
lens, bank, protocol, seed, review, smoke, sentinel, row, or profile-part
drift. The entire Phase 4 suite passes at this boundary (216 tests), including
new goldens for all four common mode/phase boundaries, both arms through the
lower-level hook tests, cache isolation, predictor masks, parser grading,
missing/wrong-phase hooks, duplicate profiles, and resume mismatch.

This is an engineering audit by the implementation agent, **not** the
independent review required by Section 6.1. No P4-P2 intervention or smoke
outcome has been generated. Full execution remains blocked on all three of:

1. a registered A1000 canonical-lens decision on Q-L1 or Q-L2 and exact
   replacement of both hash placeholders in the producer config;
2. a separately authored and registered independent-review envelope matching
   `P4_P2_GPU_INDEPENDENT_REVIEW_INSTRUCTIONS.md`;
3. a passing one-family CUDA smoke generated only after the canonical lens is
   bound.

## Required producer contract

A reviewed development producer must satisfy all of the following before a
pilot launch:

1. Require the exact Qwen revision, tokenizer/chat-template hash, parser
   version, deterministic decoding rule, shared v2 instruction, and
   2,048-token cap already used by the passing v2 baseline.
2. Resolve a live canonical-lens event and bind the exact lens path and
   SHA-256 in the input manifest. A recovery checkpoint or unregistered fit
   milestone is not a lens.
3. Resolve exactly the 20 consumed development facts and one canonical family
   per fact from `p4-qwen-multilens-functional-subset-dev-v1`. Confirmatory
   and replication Bank B/W families are forbidden.
4. Execute exactly
   `thinking_on/thinking_off × prefill/final_answer ×
   matched_control/span_safe_j` in the registered cell order. The absent
   thinking-off reasoning cell must never be imputed.
5. Use the same clean, per-position accepted-alias protection for both arms.
   The matched arm must consume the J arm's realized per-position rank and
   removed-energy profile under the frozen stable seed namespace.
6. Prove phase isolation per row: at least one intended-phase hook fire and
   zero wrong-phase fires. Log selected/protected overlap, requested and
   delivered rank, and energy-relative error without pooling positions.
7. Grade only normalized exact accepted-alias accuracy from the parser's final
   answer span. Record parse validity, EOS/length/error stop, truncation,
   generated-token count, and raw completion hash. Parse failure counts as
   incorrect.
8. Write an immutable per-row table with one row per family/cell, an
   input-manifest envelope, restart state keyed by that manifest, and a stop
   record. A resumed run must refuse any model, lens, bank, protocol, seed, or
   code mismatch.
9. Run the existing pure analyzer only after the complete 160-row grid passes
   all mechanical gates. Register the rows, manifest, analysis, and state as
   a new `phase4-development` event; never mutate the methods event.

## Review tests required before launch

- tiny-model golden tests for both phase boundaries and both arms;
- exact matched-profile parity and protection-overlap tests;
- wrong-phase and missing-hook negative tests;
- parser/stop-reason and parse-failure grading tests;
- state-resume mismatch and duplicate-row refusal tests;
- a one-family CUDA smoke that produces all eight cells and passes the pure
  analyzer without reading any untouched family.

The implementation now satisfies the engineering checklist above, subject to
independent verification. Until the three remaining blockers are complete,
`execution_authorized_at_this_boundary` remains `false`.
