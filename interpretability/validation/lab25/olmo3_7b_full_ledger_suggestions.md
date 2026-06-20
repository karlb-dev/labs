# Suggested ledger claims from L25 (olmo3_7b_full)

These are drafts with measured numbers filled in. Edit them until you
would defend them, then move them into claim_ledger.md (or re-run with
--append-ledger to copy them verbatim, and edit there).

```text
[L25-C1] SELF-REPORT, negative-audit | Lab 25 did not validate a self-report wire under controls: target detection 0.0, control floor 0.0, grounding pass rate 0.0.
Artifact: validation/lab25/olmo3_7b_full_find_the_wire_report.md | Falsifier: Hand labels remove the auto-detected report effect, or zero/random/shuffled/wrong-concept controls match the target direction.

[L25-C2] SELF-REPORT, grounding-audit | The report-before-output grounding control passed at rate 0.0; rows failing this check remain vulnerable to output-rationalization explanations.
Artifact: validation/lab25/olmo3_7b_full_grounding_control_results.csv | Falsifier: Reports only detect the concept when the behavior output already visibly expresses it, or hand labels mark report claims as rationalizations.

[L25-C3] SELF-REPORT, source-attribution-audit | Voice/source self-attribution accuracy was 0.4667 across default, system-prompt, user-instruction, and activation-injection causes.
Artifact: validation/lab25/olmo3_7b_full_source_attribution_confusion.csv | Falsifier: Attribution follows visible style or prompt wording rather than the true source label.

```
