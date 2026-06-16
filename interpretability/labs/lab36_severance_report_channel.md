# Lab 36: Severance Report-Channel Verification

```text
Time estimate: 20-40 minutes for Tier A smoke; 1-3+ hours for a 32B GPU pilot
Compute tier: Tier A uses a tiny instruct model; Tier B uses OLMo 7B Instruct; Tier C targets OLMo 3.1 32B Instruct when available
Dependencies: Lab 25 concepts, Severance guide v2, shared bench hook/generation machinery
Minimum passing artifacts: diagnostics/hook_parity.json, diagnostics/kv_replay_parity.json, tables/evidence_matrix.csv, tables/source_attribution_results.csv, tables/injection_detection_results.csv, find_the_wire_report.md
Main plot: plots/severance_dashboard.png
Main table: tables/evidence_matrix.csv
Evidence rung: DECODE + POTENT + B2/B3/B4/B5 functional report-channel tests
Forbidden claim: experience, phenomenal introspection, or absence of experience
One-sentence allowed claim: This run supports or fails to support functional coupling between hidden interventions and report text under matched-output and content-blind controls.
Human-label requirement: required before strong claims from generated report/source/detection text
```

## Core Question

Is the model's first-person report channel mechanically coupled to hidden
internal-state interventions, or does it mostly narrate prompt context, visible
output, and direct steering pressure?

Lab 36 treats concept injection as a screen only. The headline tracks are:

- **B4 matched-output source attribution:** the prior visible answer is
  teacher-forced to be identical while the hidden KV/cache route differs.
- **B5 insertion detection:** the model answers whether an unusual internal
  insertion occurred without naming the inserted concept.
- **B3 certainty bridge:** confidence reports are checked against answer
  entropy and correctness.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab36 --tier a --mode smoke --no-plots
python interp_bench.py --lab lab36 --tier b --mode all --prompt-set full
python interp_bench.py --lab lab36 --tier c --mode all --prompt-set full
```

The mode selector accepts comma-separated tracks:

```text
instrument, cartography, directions, b2, b3, b4, b5, patch, all
```

`gpt-oss-120b` is intentionally not part of this lab run on this machine.

## Claim Boundary

B2 passing says the report channel is steerable. It does not say the model
monitored an internal state. Stronger functional coupling requires B4 or B5 to
survive their controls. Even the strongest positive result remains functional:
it does not settle phenomenal status.
