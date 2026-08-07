# REPORT_v2 ERRATA — retrospective evidence-status corrections (2026-07-27)

A forensic external review of the merged Lab 37 campaign
(`jspace_part2_plan1_addendum.md`, archived with the Part-2 code at
`jspace/part2/code/`) found that several REPORT_v2.md headline claims
overstate what the implemented instruments measured. **REPORT_v2.md is
preserved unchanged as the exploratory-campaign record; read it together
with this errata.** Part 2 begins with an assay-repair phase
(`REPAIR_PREREGISTRATION.md`) and a Stage-2 re-audit that will upgrade or
retire each row below under repaired instruments.

Three root causes, in brief:

1. **The live ablation was not the paper's intervention.** The paper's
   dynamic top-10 removal PROTECTS any direction among the clean pass's
   top-10 output tokens; `jspace_dyn10` did not implement this safeguard,
   so the asterisk collapse does not bear on the paper's protocol.
2. **The capacity numbers are not the paper's estimand.** The paper
   defines occupancy by a marginal-gain crossing against equal-size random
   dictionaries and reports variance in EXCESS of the matched random
   control at median occupancy; the lab reported thresholded coefficient
   counts and raw reconstruction share at a fixed pursuit budget.
3. **Causal cells used separate per-condition bootstraps, CI-overlap null
   logic, raw-QR projectors without rank tests, controls unmatched per
   item for energy/effective-rank/geometry, and first-token scoring.**

## Claim-by-claim status (addendum §6, adopted verbatim)

| REPORT_v2 claim | Revised status | Publication-safe wording now |
|---|---|---|
| "The disagreement with the paper's causal story was instruments all along" | Not identified | Several alternative interventions produced different outcomes, but the paper's output-protected dynamic ablation has not yet been reproduced. |
| "Static J-span causal dissociation is null on both models" | Provisional auxiliary null | A selected static corpus-level J span produced no large effect at tested doses under the current tasks and aggregate-energy controls. |
| "Frozen per-item J ablation is control-clean" | Provisional | Prompt-selected J-aligned projection reduces factual-answer scores relative to the implemented random controls; geometry- and energy-matched controls remain. |
| "OLMo capacity is ten times thinner than the paper" | Not identified | The lab's fixed-threshold sparse-reconstruction summaries are much smaller for OLMo than Qwen; paper-comparable occupancy has not been computed. |
| "Qwen is paper-range under the same harness" | Not identified | Qwen is substantially larger than OLMo under the lab's current estimator; comparison to the paper requires the paper's occupancy and random-adjusted variance definitions. |
| "Live per-token deletion measures computation deletion" | Exploratory | Unprotected live J-direction deletion catastrophically degrades output, unlike a random live control. The paper's intended-output protection is missing. |
| "Workspace leads CoT by 46 steps" | Exploratory mid-band, provisional late-band | Answer-related J readouts often precede answer strings. A complete-sample, preregistered, foil-calibrated onset estimate is pending. (Mid-band trace saving was outcome-selected.) |
| "Externalization rescues frozen deletion" | Provisional mention-recovery | Extended thinking often re-expresses the target answer under the J intervention; recovery of final task performance has not been established (post-`</think>` accuracy was much lower; windows were unmatched). |
| "Second seed exact replication" | Fresh-sample robustness | The effect reappeared after changing item set, random pools, and seed — a useful bundled replication, not a seed-only sensitivity test. |
| "Broadcast non-dissociation stands" | Not paper-comparable | The lab's linear fan-out metric did not distinguish J directions from structured high-variance controls. The paper's MLP-gain and attention OV/label-preservation assays remain untested. |

## Also corrected

- **"Same harness" (Qwen leg):** the Qwen run changed model, tokenizer,
  lens provenance, layer coverage, vocab size, random-dictionary size,
  numerical precision, pursuit update, n's, and prompt rendering — a
  valuable engineering extension, not a one-variable contrast. Raw
  completion prompting is additionally NOT Qwen's official non-thinking
  mode (that is a chat-template toggle on the same checkpoint).
- **OLMo/Qwen pursuit divergence:** the banked OLMo capacity numbers
  predate the numerics-fixed pursuit used for Qwen; both must be
  recomputed with one final validated solver before any cross-model
  capacity statement.
- **s5 descriptive accumulators:** centered variance summed batch-local
  means (omits between-batch mean variance) and PCA moments were not
  resume-safe; k90 treated coefficient energy as additive despite
  dictionary correlation. Affected quantities are exploratory.
- **SL1 claim ledger:** wording is frozen-historical. Nothing inherits it
  without Stage-2 re-audit (`REPAIR_PREREGISTRATION.md` R7).

The Part-1 phenomena remain valuable exploratory priors — several are
promising (frozen content-channel effect, output-alignment of the live
readout, late-band all-item lead traces). The corrections above are about
evidence tier, not about deleting observations.
