#!/usr/bin/env python3
"""A7: campaign missingness / gate register (T6).

Every absent cell in the campaign matrix, with its named data state from
the frozen vocabulary (EVIDENCE_TIER_RULES.md rule 3) and gate evidence.
No cell here may ever be rendered as a zero or pooled as a null.

Outputs: data/missingness_register.parquet, data/capability_gating.parquet,
reports/A7_MISSINGNESS_AND_GATES.md
"""
from pathlib import Path

import pandas as pd

A = Path("/content/labs/interpretability/jspaces/phases/paper_analysis/analysis")

M = [
    # (study, model_scope, cell, state, reason, gate_evidence_id)
    ("olmo_study2", "Think-SFT", "Bank-S seven-condition stage effects (both frames)",
     "capability_gated_missing",
     "972-row battery: overall capable 0.00617, Bank-S 0.00833, 0 Bank-S facts capable on direct+composed; prospective floors 72 facts / 20 families",
     "ol2-stage-wedge-think-sft-tier1-v1"),
    ("olmo_study2", "Think-DPO", "Bank-S seven-condition stage effects (both frames)",
     "capability_gated_missing",
     "972-row battery: overall capable 0.00309, Bank-S 0.00278, 0 Bank-S facts capable on direct+composed",
     "ol2-stage-wedge-think-dpo-tier1-v1"),
    ("olmo_study2", "SFT vs DPO", "adjacent-boundary stage localization",
     "capability_gated_missing",
     "both cohorts empty; route null_or_unresolved; neither SFT- nor DPO-boundary evidence",
     "ol2-stage-wedge-joint-analysis-v1"),
    ("olmo_study2", "SFT/DPO", "Tier-2 wedge lens refit",
     "protocol_gated_missing",
     "adjacent-boundary and frame-agreement preconditions unmet; Tier 2 forbidden",
     "ol2-stage-wedge-joint-analysis-v1"),
    ("phase4", "Qwen+OLMo pair", "Bank-W cross-model load intervention (P4-P3)",
     "blocked",
     "16/20 common-capable families under strict import + fresh mainline replay",
     "p4-bank-w-capability-joint-imported-dev-v1"),
    ("olmo_study2", "Think/Instruct pair", "Bank-W pair intervention (redesign)",
     "underpowered",
     "outcome-blind power 0.7788 at 16 shared capable families vs 0.80 target; first passing count 18",
     "ol2-bank-w-olmo-pair-power-v1"),
    ("phase4", "Qwen", "P4-P1 bridge-vs-answer-direction orthogonal shot",
     "not_applicable",
     "estimation-only: Bank B underpowered and orthogonal shot not applicable under Q-L4; never run",
     "p4-bank-b-power-dev-v1"),
    ("phase4", "Qwen", "P4-P2 candidate primary",
     "not_applicable",
     "removed by Q-L4 before pilot, review, or outcome; SESOI 0.20 unchanged and unused",
     "p4-qwen-canonical-lens-decision-a1000-dev-v1"),
    ("phase4", "Qwen", "Phase 4 Holm confirmatory family",
     "not_run_by_stop_rule",
     "zero opened tests; no alpha transfer or post-hoc replacement",
     "p4-qwen-canonical-lens-decision-a1000-dev-v1"),
    ("phase4", "Qwen", "A2000 fit extension / retained-extremes producer / M3 / M4",
     "not_applicable",
     "no A2000 branch exists; M3/M4 never opened",
     "p4-qwen-canonical-lens-decision-a1000-dev-v1"),
    ("olmo_study2", "Base+Think", "causal-assay dose placement on H6 ladder",
     "archive_unavailable",
     "six registered source tables audited; none contains exact (model,item,layer,position) total-dose + residual-norm records; coverage is unavailable, NOT 0%",
     "ol2-transport-validation-joint-v1"),
    ("phase4", "Qwen", "A120-A250 operational state.json",
     "known_deficit",
     "permanent historical deficit (SHA-256 361bda08...); 16/17 outputs of the event verify; decision role superseded by later live gates",
     "p4-qwen-a120-a250-state-permanent-deficit-v1"),
    ("phase3", "Qwen", "P3-P3 held-out-family replication",
     "protocol_gated_missing",
     "replication cell never authorized by the frozen Phase 3 protocol; confirmatory contrast stands unreplicated",
     "p3-inference-audit-v1"),
    ("phase2", "all", "HM1 context-specific transport arm",
     "not_applicable",
     "pre-run gate failed (-0.04 median band improvement, 0 layers at 0.80); arm not admitted",
     "n6-confirmatory-analysis-v2"),
    ("phase2", "all", "HP5 load-interaction outcome",
     "protocol_gated_missing",
     "G5 bank built at dev tier; load intervention never opened in Phase 2; thread routed to Bank-W lineage",
     "g5-item-manifest-v5"),
    ("phase2", "Gemma-4-31B", "HP4 occupancy cell",
     "not_applicable",
     "excluded by J-lens validity premise failure (PI amendment); explicitly not a below-boundary data point",
     "gm-state-of-record-v1"),
    ("olmo_study1", "lineage", "O5 grid",
     "not_run_by_stop_rule",
     "feasibility decision closed O5 without opening it",
     "ol-o5-feasibility-decision-v1"),
    ("phase4", "Qwen", "prompt-323 historical-runtime influence shape",
     "archive_unavailable",
     "distribution-content and compiled-kernel identities not preserved at fit time; current-runtime shape only",
     "p4-runtime-identity-synthesis-v1"),
]

CAP = [
    ("olmo_study2", "Think-SFT", "overall_capable_rate", 0.00617, "ol2-stage-wedge-think-sft-tier1-v1"),
    ("olmo_study2", "Think-SFT", "bank_s_capable_rate", 0.00833, "ol2-stage-wedge-think-sft-tier1-v1"),
    ("olmo_study2", "Think-SFT", "bank_s_facts_direct_plus_composed", 0, "ol2-stage-wedge-think-sft-tier1-v1"),
    ("olmo_study2", "Think-DPO", "overall_capable_rate", 0.00309, "ol2-stage-wedge-think-dpo-tier1-v1"),
    ("olmo_study2", "Think-DPO", "bank_s_capable_rate", 0.00278, "ol2-stage-wedge-think-dpo-tier1-v1"),
    ("olmo_study2", "Think-DPO", "bank_s_facts_direct_plus_composed", 0, "ol2-stage-wedge-think-dpo-tier1-v1"),
    ("olmo_study2", "Think", "bank_w_capable_families", 17, "ol2-bank-w-olmo-pair-power-v1"),
    ("olmo_study2", "Instruct", "bank_w_capable_families", 17, "ol2-bank-w-olmo-pair-power-v1"),
    ("olmo_study2", "Think∩Instruct", "bank_w_shared_capable_families", 16, "ol2-bank-w-olmo-pair-power-v1"),
    ("olmo_study2", "Think∩Instruct", "bank_w_power_at_16", 0.7788, "ol2-bank-w-olmo-pair-power-v1"),
    ("olmo_study2", "Think∩Instruct", "bank_w_first_passing_count", 18, "ol2-bank-w-olmo-pair-power-v1"),
    ("phase4", "cross-model", "bank_w_common_capable_families", 16, "p4-bank-w-capability-joint-imported-dev-v1"),
    ("phase4", "cross-model", "bank_w_required_families", 20, "p4-bank-w-capability-joint-imported-dev-v1"),
]


def main():
    (A / "data").mkdir(exist_ok=True)
    miss = pd.DataFrame(M, columns=["study", "model_scope", "cell", "state",
                                    "reason", "gate_evidence_id"])
    miss.to_parquet(A / "data/missingness_register.parquet", index=False)
    cap = pd.DataFrame(CAP, columns=["study", "model_scope", "metric", "value",
                                     "gate_evidence_id"])
    cap.to_parquet(A / "data/capability_gating.parquet", index=False)

    by_state = miss.groupby("state").size().sort_values(ascending=False)
    L = ["# A7 — missingness and gate audit", "",
         "Source: `data/missingness_register.parquet` + "
         "`data/capability_gating.parquet` "
         "(`scripts/build_missingness_register.py`). States use the frozen "
         "vocabulary; none of these cells may be rendered as zero, pooled "
         "as a null, or dropped silently from a figure.", "",
         "## Register", "",
         "| Study | Scope | Absent cell | State | Gate evidence |",
         "|---|---|---|---|---|"]
    for _, r in miss.iterrows():
        L.append(f"| {r.study} | {r.model_scope} | {r.cell} | `{r.state}` | "
                 f"`{r.gate_evidence_id}` |")
    L += ["", "## Reasons (verbatim boundary facts)", ""]
    for _, r in miss.iterrows():
        L.append(f"- **{r.cell}** ({r.model_scope}): {r.reason}.")
    L += ["", "## State tally", ""]
    for s, n in by_state.items():
        L.append(f"- `{s}`: {n}")
    L += ["", "## The canonical example (enshrined by the addendum)", "",
          "The OLMo SFT/DPO wedge capability table — capable rates "
          "0.617%/0.309% and **zero** Bank-S facts capable on direct + "
          "composed at either checkpoint — is the campaign's canonical "
          "demonstration that missing, gated, and not-applicable are data "
          "states: the seven-condition stage effects exist as *questions* "
          "with empty prospective cohorts, not as nulls. Any figure "
          "rendering these cells must show a gate glyph, never a zero bar.",
          ""]
    (A / "reports/A7_MISSINGNESS_AND_GATES.md").write_text("\n".join(L))
    print(f"missingness rows: {len(miss)}; capability rows: {len(cap)}")
    print(by_state.to_string())


if __name__ == "__main__":
    main()
