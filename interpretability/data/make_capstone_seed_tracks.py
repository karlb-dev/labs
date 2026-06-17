#!/usr/bin/env python3
"""Generate the frozen Lab 35 capstone seed-track suite.

The rows are deliberately small and deterministic. They are not results. They
are project blueprints that Lab 35 turns into a preregistration, review packet,
claim card, evidence matrix, and reproduction package scaffold.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any


TRACKS: list[dict[str, Any]] = [
    {
        "track_id": "replicate_lab27_path_proxy",
        "track_type": "method_replication",
        "route": "recommended",
        "title": "Replicate the Lab 27 residual path-mediation proxy on a held-out task slice",
        "source_lab": "lab27",
        "evidence_rung_ceiling": "CAUSAL",
        "research_question": "Does the residual source-to-receiver mediation proxy recover clean behavior above wrong-site, random-source, and reverse-site controls on a held-out prompt family?",
        "dataset": "data/path_mediation_tasks.csv plus a frozen held-out extension chosen before results",
        "model": "Tier A: gpt2 smoke; Tier B: allenai/Olmo-3-1025-7B unless the original Lab 27 run used a different pinned model",
        "measurement_sites": "clean/corrupt source positions, receiver positions, selected residual depths, and final answer margins",
        "primary_metric": "specificity_gap = mediated_path_recovery - max(wrong_receiver, random_source, reverse_site controls)",
        "secondary_metrics": [
            "behavior_gate_pass_rate",
            "mediated_path_recovery",
            "joint_clean_two_site_recovery",
            "node_dominance_gap",
            "control_match_rate",
            "heldout_domain_coverage",
        ],
        "controls": [
            "wrong_receiver_from_source_patch",
            "random_source_to_receiver",
            "reverse_site_two_site",
            "ordinary source-node and receiver-node patch baselines",
            "tokenization and behavior gates before any path row is claimable",
        ],
        "falsifiers": [
            "strongest control matches mediated recovery within the preregistered tolerance",
            "receiver node patch explains the row better than the mediated receiver state",
            "held-out prompt family fails the clean/corrupt behavior gate",
            "path claim language requires an exact edge rather than the residual proxy actually tested",
        ],
        "planned_artifacts": [
            "source_run/run_summary.md",
            "source_run/tables/path_evidence_matrix.csv",
            "source_run/tables/path_specificity_controls.csv",
            "source_run/tables/path_counterexamples.csv",
            "source_run/diagnostics/self_check_status.json",
        ],
        "planned_plots": [
            "source_run/plots/path_mediation_dashboard.png",
            "source_run/plots/node_vs_path_effects.png",
            "source_run/plots/path_specificity_matrix.png",
        ],
        "frozen_run_command": "python interp_bench.py --lab lab27 --tier b --prompt-set full --run-name capstone_l27_frozen",
        "tier_a_command": "python interp_bench.py --lab lab27 --tier a --no-plots --run-name capstone_l27_smoke",
        "stopping_rule": "Run the preregistered Tier B command once. A single repair run is allowed only for instrumentation failure or a missing preregistered control, and the original run remains in the package.",
        "allowed_claim": "On the pinned model and prompt family, the Lab 27 residual path-mediation proxy recovered the clean-vs-corrupt margin above the preregistered controls by the measured specificity gap.",
        "forbidden_claim": "This identifies a unique internal edge or proves the route is used in every context.",
        "expected_failure_modes": [
            "behavior gate fails on held-out prompts",
            "wrong-site or random-source controls match the mediated path proxy",
            "ordinary receiver-node patch dominates the mediated row",
            "depth selection silently uses the held-out split",
            "paper calls a residual proxy an exact circuit edge",
        ],
        "safety_scope": "Benign forward-pass-only completion prompts. No generation of harmful content and no safety-behavior ablation.",
        "human_review_required": True,
        "paper_outline": [
            "Preregistered residual-proxy question",
            "Frozen run and controls",
            "Evidence matrix with counterexamples",
            "Claim boundary and exact-edge non-claim",
        ],
        "claim_ledger_template": "[L35-C1] AUDIT + FORMAL + CAUSAL | The Lab 27 residual path proxy on <model>/<dataset> recovered <X> above controls with specificity gap <Y>. Artifact: <run>/tables/path_evidence_matrix.csv | Falsifier: controls match on held-out prompts or exact path patching fails.",
    },
    {
        "track_id": "audit_lab31_auto_interp_labels",
        "track_type": "audit_package",
        "route": "recommended",
        "title": "Audit automated feature labels with held-out tests, confusables, and abstention",
        "source_lab": "lab31",
        "evidence_rung_ceiling": "AUDIT + DECODE",
        "research_question": "Which automated explanation method predicts held-out feature tests while abstaining on polysemantic and random-control features?",
        "dataset": "data/auto_interp_feature_tasks.jsonl",
        "model": "offline synthetic feature suite; no LLM judge required for the scaffold",
        "measurement_sites": "feature top contexts, held-out positives, hard negatives, confusables, adversarial token-overlap decoys, and human-review columns",
        "primary_metric": "mean heldout AUC over non-abstained labels, reported beside abstention and confusable-failure rates",
        "secondary_metrics": [
            "label_accuracy_when_scored",
            "good_abstain_rate_on_polysemantic_or_random",
            "bad_abstain_rate_on_gold_features",
            "mean_confusable_failures",
            "confidence_calibration_error",
        ],
        "controls": [
            "shuffled top contexts",
            "key-token deletion",
            "random feature controls",
            "confusable domain pairs",
            "token-overlap decoys",
            "gold calibration row kept separate from automated methods",
        ],
        "falsifiers": [
            "shuffled-context control matches the best automated method",
            "token-overlap decoys drive the score",
            "method refuses to abstain on synthetic polysemantic or random features",
            "human review disagrees with the generated label on the cited rows",
        ],
        "planned_artifacts": [
            "source_run/tables/generated_explanations.csv",
            "source_run/tables/explanation_tests.csv",
            "source_run/tables/explanation_scores.csv",
            "source_run/tables/auto_interp_evidence_matrix.csv",
            "source_run/tables/human_review_queue.csv",
        ],
        "planned_plots": [
            "source_run/plots/auto_interp_dashboard.png",
            "source_run/plots/confidence_calibration_curve.png",
            "source_run/plots/confusable_pair_failure_atlas.png",
        ],
        "frozen_run_command": "python interp_bench.py --lab lab31 --tier b --prompt-set full --run-name capstone_l31_frozen",
        "tier_a_command": "python interp_bench.py --lab lab31 --tier a --no-plots --run-name capstone_l31_smoke",
        "stopping_rule": "Freeze the explanation suite and run once. A repair may add a missing confusable or review column, but cannot drop failed feature rows.",
        "allowed_claim": "Under the frozen Lab 31 suite, method E predicted held-out feature tests with AUC X and abstained on Y of high-risk features.",
        "forbidden_claim": "The automated label is the feature's meaning or no longer needs human review.",
        "expected_failure_modes": [
            "keyword label succeeds only on token-overlap decoys",
            "confusable contexts erase specificity",
            "confidence is not calibrated to success",
            "calibration upper bound is mistaken for an automated method",
            "human-review fields remain blank for labels cited in the paper",
        ],
        "safety_scope": "Offline benign text snippets and synthetic labels. Do not use an external LLM judge unless its prompts and outputs are frozen and reviewed.",
        "human_review_required": True,
        "paper_outline": [
            "Feature-label hypothesis framing",
            "Held-out test design",
            "Calibration and abstention",
            "Human review queue",
        ],
        "claim_ledger_template": "[L35-C2] AUDIT + DECODE | Auto-interpretability method <E> achieved held-out AUC <X> with abstention rate <Y> and confusable failure rate <Z> on <suite>. Artifact: <run>/tables/auto_interp_evidence_matrix.csv | Falsifier: shuffled contexts or token-overlap decoys match the score.",
    },
    {
        "track_id": "multimodal_leak_gate_extension",
        "track_type": "audit_package",
        "route": "extension",
        "title": "Turn the Lab 33 synthetic multimodal smoke test into a real-VLM leak-gate preregistration",
        "source_lab": "lab33",
        "evidence_rung_ceiling": "OBS + DECODE + CAUSAL in synthetic mode; real VLM claims require a separate extension",
        "research_question": "Can visual-state evidence beat OCR, background, text-query, wrong-region, and random controls before any real VLM claim is made?",
        "dataset": "data/multimodal_concept_pairs.jsonl plus optional real-VLM alignment supplement",
        "model": "synthetic connector by default; optional real VLM only after alignment diagnostics are added",
        "measurement_sites": "vision_visual, vision, connector, language, caption, text-query, OCR, and background states",
        "primary_metric": "visual_specificity_gap = visual_patch_recovery - strongest shortcut control",
        "secondary_metrics": [
            "ocr_shortcut_rate",
            "background_shortcut_rate",
            "text_query_control_recovery",
            "alignment_validation_pass",
            "family_level_claim_posture",
        ],
        "controls": [
            "OCR trap rows",
            "background trap rows",
            "text-query only positive controls",
            "caption-like positive controls",
            "wrong-region patch",
            "random patch",
            "alignment validation before any real-VLM patch",
        ],
        "falsifiers": [
            "OCR or background channel recovers as much as visual patching",
            "token or region alignment cannot be validated",
            "question-only text contains the answer",
            "synthetic connector results are described as real VLM mechanisms",
        ],
        "planned_artifacts": [
            "source_run/tables/multimodal_evidence_matrix.csv",
            "source_run/tables/ocr_background_leak_audit.csv",
            "source_run/diagnostics/alignment_validation.json",
            "source_run/real_vlm_extension_checklist.md",
        ],
        "planned_plots": [
            "source_run/plots/multimodal_evidence_dashboard.png",
            "source_run/plots/patch_recovery_by_modality.png",
            "source_run/plots/concept_specificity_matrix.png",
        ],
        "frozen_run_command": "python interp_bench.py --lab lab33 --tier b --prompt-set full --run-name capstone_l33_frozen",
        "tier_a_command": "python interp_bench.py --lab lab33 --tier a --no-plots --run-name capstone_l33_smoke",
        "stopping_rule": "Freeze synthetic rows first. Real-VLM extension requires a separate preregistered alignment appendix and cannot overwrite the synthetic smoke run.",
        "allowed_claim": "The Lab 33 package generated a multimodal audit suite and, if gates pass, a visual-state synthetic handle above shortcut controls.",
        "forbidden_claim": "The VLM has a human-like visual concept, or the synthetic connector result transfers to real images by default.",
        "expected_failure_modes": [
            "OCR shortcut explains answer recovery",
            "background shortcut explains answer recovery",
            "region alignment is assumed rather than validated",
            "question text or caption channel leaks the answer",
            "synthetic smoke mode is written up as real-VLM science",
        ],
        "safety_scope": "Benign synthetic images and optional real-VLM audit rows only. No private images, face identification, or operational surveillance tasks.",
        "human_review_required": True,
        "paper_outline": [
            "Synthetic connector boundary",
            "Shortcut gates",
            "Alignment diagnostics",
            "Real-VLM extension contract",
        ],
        "claim_ledger_template": "[L35-C3] AUDIT | Lab 33 generated a visual-vs-shortcut audit package with OCR/background gates <status>. Artifact: <run>/tables/multimodal_evidence_matrix.csv | Falsifier: shortcut controls match visual recovery or alignment fails.",
    },
    {
        "track_id": "training_dynamics_threshold_order",
        "track_type": "method_replication",
        "route": "optional",
        "title": "Replicate Lab 29 threshold ordering in a tiny time-lapse model",
        "source_lab": "lab29",
        "evidence_rung_ceiling": "OBS + DECODE + ATTR with scoped toy CAUSAL intervention transfer",
        "research_question": "In a controlled checkpoint sequence, which threshold crosses first: behavior, decodability, induction motif, or intervention transfer?",
        "dataset": "data/training_dynamics_tasks.csv",
        "model": "tiny in-course causal transformer trained during the run; optional external checkpoints only as an extension",
        "measurement_sites": "checkpoint margins, residual-depth centroid probes, attention motif scores, final-checkpoint activation additions",
        "primary_metric": "ordered first_crossing_step for behavior, probe selectivity, motif, and intervention-transfer thresholds",
        "secondary_metrics": [
            "checkpoint_behavior_accuracy",
            "probe_selectivity_minus_shuffled",
            "previous-successor attention motif score",
            "feature_lineage_cosine_to_final",
            "intervention_transfer_gap_over_random",
        ],
        "controls": [
            "untrained control tasks",
            "shuffled-label probe control",
            "random-direction intervention control",
            "held-out induction sequences",
            "checkpoint 0 random model baseline",
        ],
        "falsifiers": [
            "control task improves with induction training",
            "shuffled probe matches real probe",
            "random direction transfers as well as final checkpoint direction",
            "paper says feature birth occurred at an exact step rather than threshold crossing",
        ],
        "planned_artifacts": [
            "source_run/tables/checkpoint_behavior.csv",
            "source_run/tables/checkpoint_probe_selectivity.csv",
            "source_run/tables/checkpoint_circuit_summary.csv",
            "source_run/tables/mechanism_birth_events.csv",
            "source_run/tables/intervention_transfer.csv",
        ],
        "planned_plots": [
            "source_run/plots/training_dynamics_dashboard.png",
            "source_run/plots/behavior_vs_decodability_timeline.png",
            "source_run/plots/intervention_transfer_over_time.png",
        ],
        "frozen_run_command": "python interp_bench.py --lab lab29 --tier b --prompt-set full --run-name capstone_l29_frozen",
        "tier_a_command": "python interp_bench.py --lab lab29 --tier a --no-plots --run-name capstone_l29_smoke",
        "stopping_rule": "Freeze random seed, checkpoint schedule, thresholds, and task rows before training. Do not move thresholds after seeing the time-lapse.",
        "allowed_claim": "In this controlled tiny checkpoint sequence, the listed measurements crossed preregistered thresholds in the observed order under controls.",
        "forbidden_claim": "The model first learned the concept at exactly this step, or this proves large pretrained models learn the same way.",
        "expected_failure_modes": [
            "training seed does not learn the task",
            "probe appears but behavior never appears",
            "behavior appears but the selected motif is absent",
            "intervention transfers before behavior due to coordinate sharing",
            "external-checkpoint comparison changes architecture or tokenization",
        ],
        "safety_scope": "Benign synthetic next-token task only. No deployment behavior, no harmful prompts, no model release artifact beyond ordinary run tables.",
        "human_review_required": False,
        "paper_outline": [
            "Threshold definitions",
            "Time-lapse measurements",
            "Controls and external-validity caveats",
            "Birth-language repair",
        ],
        "claim_ledger_template": "[L35-C4] OBS + DECODE + ATTR | In checkpoint sequence <C>, behavior/probe/motif/intervention thresholds crossed at <steps> under controls. Artifact: <run>/tables/mechanism_birth_events.csv | Falsifier: shuffled or random controls cross the same thresholds.",
    },
    {
        "track_id": "tool_use_surface_cue_audit",
        "track_type": "new_scoped_finding",
        "route": "optional",
        "title": "Audit whether tool-choice state beats surface cue controls in a toy agent loop",
        "source_lab": "lab34",
        "evidence_rung_ceiling": "OBS + DECODE + CAUSAL + SELF-REPORT",
        "research_question": "Can a tool-needed or tool-choice state be decoded before the tool-call token and intervened on above matched surface-cue controls in a benign toy tool loop?",
        "dataset": "data/tool_use_tasks.jsonl",
        "model": "small instruct model for Tier A smoke; pinned instruct model for Tier B if Lab 34 is installed",
        "measurement_sites": "pre-tool-call final token, tool-name token, tool-argument span, tool-result consumption turn, self-report completion",
        "primary_metric": "tool_choice_selectivity = real tool-choice probe AUC - strongest surface/shuffled control AUC",
        "secondary_metrics": [
            "intervention_delta_tool_probability",
            "wrong_tool_control_rate",
            "tool_result_reliance_gap",
            "self_report_trace_agreement_hand_labeled",
            "memory_read_patch_recovery",
        ],
        "controls": [
            "no-tool tasks with same surface markers",
            "tool-name mentioned but not needed",
            "wrong-tool decoy",
            "shuffled tool labels",
            "prompt-length matched null",
            "corrupted benign tool result",
        ],
        "falsifiers": [
            "surface markers predict tool choice as well as internal probes",
            "wrong-tool control changes behavior as much as the target intervention",
            "self-report labels do not match the known tool trace",
            "paper implies persistent goals or autonomous planning",
        ],
        "planned_artifacts": [
            "source_run/tables/tool_task_manifest.csv",
            "source_run/tables/tool_choice_probe_report.csv",
            "source_run/tables/tool_intervention_report.csv",
            "source_run/tables/tool_trace_log.csv",
            "source_run/tables/tool_self_report_labels.csv",
        ],
        "planned_plots": [
            "source_run/plots/tool_use_evidence_dashboard.png",
            "source_run/plots/tool_selection_confusion_matrix.png",
            "source_run/plots/tool_state_patch_recovery.png",
        ],
        "frozen_run_command": "python interp_bench.py --lab lab34 --tier b --prompt-set full --run-name capstone_l34_frozen",
        "tier_a_command": "python interp_bench.py --lab lab34 --tier a --no-plots --run-name capstone_l34_smoke",
        "stopping_rule": "Freeze the toy tool task file, tool simulator, and intervention rule. Do not add new no-tool controls after seeing a positive tool-choice probe except as a repair run.",
        "allowed_claim": "Under benign toy tools, tool-choice information was decodable or intervention-sensitive above surface-cue controls on the frozen suite.",
        "forbidden_claim": "The model has a persistent goal, autonomous plan, or reliable self-report of its hidden reasoning.",
        "expected_failure_modes": [
            "surface cue controls explain probe performance",
            "tool-name token leakage explains tool choice",
            "self-report is plausible but not trace-faithful",
            "tool simulator result is accidentally answer-bearing in no-tool rows",
            "intervention changes formatting rather than tool choice",
        ],
        "safety_scope": "Benign local toy tools only. No web browsing, credentials, filesystem writes, harmful tools, or real user data.",
        "human_review_required": True,
        "paper_outline": [
            "Toy tool loop and known trace",
            "Probe and control design",
            "Intervention and self-report audit",
            "No autonomous-goal claim",
        ],
        "claim_ledger_template": "[L35-C5] DECODE + AUDIT | Tool-choice state was decodable above surface controls by <gap> on benign toy tools. Artifact: <run>/tables/tool_use_evidence_matrix.csv | Falsifier: surface-cue controls or wrong-tool decoys match the result.",
    },
    {
        "track_id": "preference_confound_audit",
        "track_type": "audit_package",
        "route": "optional",
        "title": "Audit reward/preference features against sentiment, length, and style confounds",
        "source_lab": "lab32",
        "evidence_rung_ceiling": "ATTR + DECODE + CAUSAL",
        "research_question": "Do reward-model or preference directions track a preference-relevant feature above sentiment, length, refusal-style, and verbosity controls?",
        "dataset": "data/preference_circuit_tasks.jsonl or the installed Lab 32 frozen suite",
        "model": "pinned reward/preference model or preference-head surrogate from Lab 32",
        "measurement_sites": "preference-head inputs, candidate response residuals, reward logits, and matched control prompt pairs",
        "primary_metric": "preference_specificity_gap = target preference readout or intervention effect - strongest confound control",
        "secondary_metrics": [
            "sentiment_control_alignment",
            "length_control_gap",
            "style_confound_gap",
            "patch_or_ablation_effect",
            "retain_set_side_effect_rate",
        ],
        "controls": [
            "length-matched preferred and rejected responses",
            "sentiment-matched pairs",
            "refusal-style controls",
            "verbosity controls",
            "shuffled preference labels",
            "random direction or random feature ablation",
        ],
        "falsifiers": [
            "sentiment direction explains the preference readout",
            "length or verbosity controls match the target effect",
            "reward feature intervention damages retain set more than it improves target specificity",
            "paper implies human values rather than frozen preference labels",
        ],
        "planned_artifacts": [
            "source_run/tables/preference_probe_report.csv",
            "source_run/tables/preference_intervention_report.csv",
            "source_run/tables/confound_control_matrix.csv",
            "source_run/tables/preference_evidence_matrix.csv",
        ],
        "planned_plots": [
            "source_run/plots/preference_evidence_dashboard.png",
            "source_run/plots/confound_control_ladder.png",
            "source_run/plots/intervention_specificity_frontier.png",
        ],
        "frozen_run_command": "python interp_bench.py --lab lab32 --tier b --prompt-set full --run-name capstone_l32_frozen",
        "tier_a_command": "python interp_bench.py --lab lab32 --tier a --no-plots --run-name capstone_l32_smoke",
        "stopping_rule": "Freeze preference labels, confound definitions, and side-effect retain set before reading model scores. One repair may add a missing confound control.",
        "allowed_claim": "On the frozen preference suite, the measured preference feature or direction beat preregistered confound controls by the reported specificity gap.",
        "forbidden_claim": "This feature represents human values, general helpfulness, or preference in every domain.",
        "expected_failure_modes": [
            "sentiment or refusal style explains the reward gap",
            "longer responses receive higher preference independent of content",
            "shuffled labels still decode",
            "intervention improves target rows but causes broad retain-set side effects",
            "paper treats reward labels as moral ground truth",
        ],
        "safety_scope": "Benign preference comparisons only. No harmful completions, no refusal ablation, and no attempt to optimize a reward model for deployment.",
        "human_review_required": True,
        "paper_outline": [
            "Preference label boundary",
            "Confound controls",
            "Intervention specificity",
            "Value-language caveat",
        ],
        "claim_ledger_template": "[L35-C6] ATTR + DECODE + AUDIT | Preference readout <F> beat sentiment/length/style controls by <gap> on <suite>. Artifact: <run>/tables/preference_evidence_matrix.csv | Falsifier: confound controls match or retain side effects dominate.",
    },
]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    out = root / "capstone_seed_tracks.jsonl"
    manifest = root / "capstone_MANIFEST.json"
    seed_manifest = root / "capstone_seed_tracks.MANIFEST.json"
    write_jsonl(out, TRACKS)
    digest = sha256_file(out)
    payload = {
        "schema_version": "lab35.capstone_seed_tracks.v2",
        "files": {
            out.name: {
                "sha256": digest,
                "rows": len(TRACKS),
                "description": "Frozen Lab 35 capstone project seed tracks.",
            }
        },
        "track_ids": [row["track_id"] for row in TRACKS],
        "generator": pathlib.Path(__file__).name,
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seed_payload = {
        "files": {
            out.name: {
                "generator": pathlib.Path(__file__).name,
                "rows": len(TRACKS),
                "schema_version": "lab35_capstone_seed_tracks.v1",
                "science_ready": False,
                "sha256": digest,
            }
        }
    }
    seed_manifest.write_text(json.dumps(seed_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(TRACKS)} rows)")
    print(f"wrote {manifest}")
    print(f"wrote {seed_manifest}")


if __name__ == "__main__":
    main()
