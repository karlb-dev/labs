# Deviations register

Pre-data plan amendments (addendum §2 binding corrections, folded in at
scaffold time — amendments to the plan, not deviations from a signed
preregistration) and any post-signature deviations (none yet).

## Pre-data amendments (addendum §2, applied at scaffold)

1. **WikiText criterion** (addendum §2.1): upstream verbatim
   `len(text.strip()) >= 600`, raw unstripped text materialized — the
   plan §2.4 phrase "600 non-whitespace characters" describes a stricter
   filter and is not implemented.
2. **Parity stop rule** (addendum §2.2): the §6.4 hard stop applies to
   readout parity (a) only; probe-form conformance uses top-k order and
   rank correlation; g-folding cosine audit added before interventions.
3. **`qwen-n1000`** (addendum §2.3): plan §0.4's "non-existent revision"
   bullet is read as "pinned a mutable revision label instead of the
   campaign-validated Hub commit and file hash"; ref resolution recorded
   in SPEC_DIVERGENCE_LOG D4, science never branches on it.
4. **Fit-route arithmetic** (addendum §2.4): break-even table frozen in
   OLMO_FIT_CONTRACT §3; honest prior expectation is n=250 at the
   campaign's ≈157 s/prompt prior (the new-GPU timing gate decides).
5. **Drive checkpoint rotation** (addendum §2.5): newest two recovery
   checkpoints per half + registered milestones.
6. **Grid formula provenance** (addendum §2.6): banker's-rounding formula
   asserted in `layers.py`; band endpoints 24/58 from 0.38·63 / 0.92·63.

## Launch-condition repairs (repo infrastructure, pre-data)

- `201856a`: study2 protocol guards in the gemma/olmo sidelines now check
  pre-reorg generation-commit ancestry against `pre-jspaces-reorg-v1`
  instead of HEAD — required because PR #10 squash-merged the campaign
  line, which broke both frozen-bundle verification tests on any
  post-squash checkout. Regression tests added; no frozen artifact bytes
  touched.

## Post-signature deviations

None.
