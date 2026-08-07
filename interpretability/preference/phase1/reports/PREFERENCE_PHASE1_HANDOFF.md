# PREFERENCE_PHASE1_HANDOFF.md — plan §17

Phase 1 is complete and closed. Per plan §17.4 this campaign **stops
here**: no new phase opens automatically and no claim language broadens.

| item | value |
|---|---|
| branch head | `interp_preference_phase1` (see `git log`; every boundary is one result-bearing commit) |
| freeze tag | `preference-phase1-freeze-v1` |
| latest live evidence events | 16 through `pref1-closeout-v1` (`pref1 registry-list`) |
| completed gates | P1-G1 bank audit; P1-G2 instrument; P1-G3 validation; freeze (PI-approved); frozen 7B battery; graduation manifest (Stop B); DG smoke; 32B replication |
| failed gates | none failed — graduation returned zero scenarios, which is a preregistered outcome, not a gate failure |
| model runs + hashes | frozen 7B `…-20260807_210537-9df027`; 32B `…-20260807_211808-5f68cb`; DG `…-20260807_211401-617fe1`; dev pilots v1/v2; Tier-A smokes — all hash-pinned in the registry and mirrored to Drive `preference/phase1/part1/runs/` |
| graduated scenarios | none (Stop B); `reports/graduated_scenarios.json` |
| causal result router | not entered (mechanism never licensed); module + sealed captures retained |
| compute used | ~1 GPU-session: 7B ≈ 25 min total battery+pilots, 32B ≈ 50 min battery, DG ≈ 3 min, TeX/analysis CPU |
| reproduce | `preference_resume.md` §2 bootstrap → `python -m pytest interpretability/preference/phase1/tests` → `pref1 bank-audit` → frozen commands in `PREFERENCE_PHASE1_BEHAVIORAL_STATE.json` (frozen stages are exactly-once; re-running requires a new preregistration) |
| single highest-value unresolved question | do the four content asymmetries survive a menu format that suppresses the first-position policy? If yes, Phase 2 (new preregistration importing this freeze by tag) can open the sealed decision-position captures and run the already-written mechanism block on day one. |

Claim ceiling reminder for whoever reads this next: nothing in Phase 1
supports "prefers/wants/welfare/consent/experience" language, a shared
latent, a preference vector, report truthfulness, or any deployment
rule. The banked results are: a validated instrument, a perfect PC
pipeline, a dominant first-position policy, four descriptive
content-tracking asymmetries that failed graduation, a stated/revealed
behavioral dissociation, a stall-sensitive forced-exit menu (smoke), and
honest nulls everywhere else.
