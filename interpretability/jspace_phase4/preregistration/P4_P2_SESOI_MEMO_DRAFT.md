# P4-P2 SESOI memo

**PROSPECTIVE CANDIDATE — NOT FROZEN — INDEPENDENT REVIEW AND PI APPROVAL
PENDING**

Drafted 2026-08-02 before any P4-P2 phase-intervention pilot outcome existed.
The pilot may estimate variability only; its observed mean is forbidden from
selecting or revising this value.

## Candidate substantive threshold

For the registered family-level interaction

```text
I = [damage(thinking_on, final_answer)
     - damage(thinking_on, prefill)]
    - [damage(thinking_off, final_answer)
       - damage(thinking_off, prefill)],
```

the candidate smallest effect of substantive interest is **0.20 accuracy
points** in the positive direction. Damage is matched-control accuracy minus
span-safe-J accuracy, so a positive interaction means that moving the lesion
from prefill to the final-answer phase has a more harmful effect under
official thinking-on than under official thinking-off.

The 0.20 threshold is an operational claim boundary, not an estimate from the
pilot. On a 20-family descriptive grid it corresponds to a net four-family
shift across the eight binary cells. Smaller interactions would be too
dependent on a few discrete family flips to carry the proposed mode-by-phase
mechanism claim, even if a much larger sample made a null test significant.

## Power and decision rule

- The alternative remains one-sided: `I > 0`.
- Planning alpha remains the conservative Holm value `0.05/3`.
- The variance pilot contributes only
  `max(sample SD, family-bootstrap 90% upper SD)` under its registered
  20,000-draw rule. Its family-interaction mean is masked for planning.
- The eventual untouched-family count must achieve at least 0.80 power at
  `I = 0.20` under the exact frozen family-level sign-flip procedure and the
  conservative planning SD. Scenario-envelope or Gaussian power alone is
  insufficient.
- The family split and its hash must be frozen before any untouched outcome
  is opened. Confirmatory and replication families must be disjoint from the
  20 consumed pilot families and from all prior Phase 2/3 facts.
- If an adequately powered untouched split is infeasible, P4-P2 remains
  estimation-only or outside the Phase 4 primary family. The SESOI may not be
  enlarged after the pilot merely to manufacture power.

This memo fixes a candidate ruler for independent review. It does not
authorize the pilot, select a canonical lens, open untouched data, or satisfy
the Phase 4 freeze conditions.
