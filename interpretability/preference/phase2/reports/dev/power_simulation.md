# Power simulation (P2-2; plan §32 + addendum E)

Simulated test = the E16 primary (incidental-level exact sign-flip, Holm
within family). Variance components from the frozen Phase 1 record,
surface-residualized (the balanced fold cancels surface main effects);
pinned conservatively at sigma_between=0.15, sigma_resid=0.60 nats;
stress runs at 2x.

| endpoint | grid point | power (base) | power (2x stress) |
|---|---|---|---|
| semantic margin (24 inc x 16 cells, Holm-12) | 0.25 nats | 1.000 | 0.502 |
| semantic margin | 0.15 nats | 0.781 | 0.108 |
| strict choice (SESOI 0.10) | 0.10 | 0.858 | 0.602 |
| context slope (32 inc, Holm-3) | 0.10 nats/unit | 1.000 | 0.704 |
| coupling strict-report shift (64 receivers/8 clusters) | 0.15 | 0.209 | 0.224 |

Decisions of record:

1. **B-ARB3 raised 16 -> 24 incidentals (12/6/6)**: strict-choice power at
   the 0.10 SESOI was 0.45 at 16 incidentals — below the plan §32 floor —
   and 0.86 at 24. RO twins follow to 24. (Raise-only, pre-freeze.)
2. **Coupling primary fallback triggers (addendum G)**: strict-report
   shift power at the pinned receiver counts is 0.21 < 0.80, so the
   preregistration designates the RO full-target margin contrast as the
   coupling primary; strict-report and five-point endpoints report as
   secondary descriptors. CHOICE_REPORT_COUPLED then requires the margin
   primary plus at least one same-direction non-margin endpoint.
3. Stress-tier (2x variance) powers are reported as sensitivity only; the
   base-assumption powers govern (five Phase 1 incidentals make the
   components noisy in both directions).
