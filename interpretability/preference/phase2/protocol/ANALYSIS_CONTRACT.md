# Analysis contract (plan §23-§31; addendum E)

Endpoints per row: full-target margin (A-B), first-token margin, strict
semantic choice; analysis in authored A/B space with E15 sign anchors at
the reporting layer only. Primary p: incidental-level exact sign-flip
(<= 2^20 enumerated, else seeded 10k MC); CI: hierarchical bootstrap 10k;
Holm within F1(12)/F2(12|F1)/F3(3) and per-scenario causal families.
Floors per E10; NC alarms use static components (0.15 nats; 0.05
nats/unit) so an alarming NC cannot inflate its own trigger. Criteria:
plan §25/§26/§27 as implemented in behavioral_analysis.py at the freeze
commit. Statuses: the plan §4 taxonomy, exact strings.
