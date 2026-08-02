# P4-P2 GPU producer review

**Status: prospective engineering review; development-only; execution is not
yet authorized.**

Reviewed 2026-08-02 before any P4-P2 phase-intervention pilot outcome was
generated. This review does not constitute independent protocol review, PI
approval, a freeze commit, or a freeze tag.

## Disposition

The registered module
`jspace_phase4.experiments.p4_qwen_mode_variance_pilot` correctly freezes and
validates the consumed-family selection, eight-cell interaction, mechanical
gates, and variance-only analyzer. Its analyzer is deterministic, treats
parse failure as incorrect, refuses incomplete family grids, and cannot use
the pilot mean to choose a SESOI.

It is deliberately **not a GPU intervention producer**. It never loads Qwen,
binds a lens, installs phase hooks, generates completions, or emits pilot
rows. Therefore the registered methods event is sound but execution remains
blocked. Calling the pure analyzer with hand-authored rows would not satisfy
the protocol.

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

Until those requirements and the registered canonical-lens decision are
complete, `execution_authorized_at_this_boundary` remains `false`.
