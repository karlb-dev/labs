"""Register OR1-B/C/D boundaries as their outputs land (idempotent).

Covers: OLMo fit (route, halves, split-half, merged), OLMo admission +
evals + core, battery groups per lane, VR v2 (+ v1 supersession per
INCIDENT or1-001), and both cross-overs. Run repeatedly; each event
registers once when its outputs exist.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import registry
from ..lens_eval import EVAL_SETS
from ..paths import DRIVE_ROOT

FIT = DRIVE_ROOT / "olmo_fit"
OLMO = DRIVE_ROOT / "olmo_lane"
QWEN = DRIVE_ROOT / "qwen_lane"


def _existing() -> set[str]:
    return {row["evidence_id"] for row in registry.read_events()
            if row["event"] in {"evidence_created", "release_created"}}


def _maybe(evidence_id: str, outputs: list[Path], *, tier: str, what: str,
           command: str, **extra) -> bool:
    if evidence_id in _existing():
        return False
    if not all(path.exists() for path in outputs):
        return False
    registry.create(evidence_id, tier=tier, what=what, command=command,
                    outputs=outputs, **extra)
    print("registered:", evidence_id)
    return True


def main() -> None:
    route = FIT / "fit_route.json"
    if route.exists():
        route_data = json.loads(route.read_text())
        _maybe("or1-olmo-fit-route-v1", [route],
               tier="methods",
               what=(f"OLMo fit timing gate: dim_batch={route_data['dim_batch']}, "
                     f"{route_data['measured_s_per_prompt']:.1f} s/prompt, "
                     f"frozen route n_per_half={route_data['n_per_half']} "
                     f"(merged n={route_data['merged_n']}); chosen from "
                     "timing only, before any outcome."),
               command="python -m jspace_official_repro.experiments.olmo_fit")
    for half in ("A", "B"):
        marker = FIT / f"half_{half}_complete.json"
        lens = FIT / f"olmo_or1_half_{half}.pt"
        _maybe(f"or1-olmo-fit-half-{half.lower()}-v1", [marker, lens],
               tier="development",
               what=(f"OLMo official-estimator half-{half} fit (upstream "
                     "jacobian_for_prompt semantics; hardened checkpointing "
                     "proven bit-identical to upstream on tiny model)."),
               command="python -m jspace_official_repro.experiments.olmo_fit")
    audit = FIT / "splithalf_operator_audit.json"
    _maybe("or1-olmo-fit-splithalf-audit-v1", [audit],
           tier="methods",
           what="Split-half operator-layer audit (cosines, sym-rel-Frobenius, "
                "identity fractions, principal-subspace overlap).",
           command="python -m jspace_official_repro.splithalf")
    _maybe("or1-olmo-fit-merged-v1",
           [FIT / "merged_complete.json", FIT / "olmo_or1_merged.pt"],
           tier="development",
           what="Merged OLMo OR1 lens (prompt-count-weighted merge of "
                "halves; the Study-1 OLMo primary instrument).",
           command="python -m jspace_official_repro.experiments.olmo_fit")

    _maybe("or1-olmo-lens-admission-v1", [OLMO / "olmo_admission.json"],
           tier="methods",
           what="OLMo lane admission with merged OR1 lens: readout parity, "
                "alpha-0 no-op, g-folding audit, sentinels, repeatability.",
           command="python -m jspace_official_repro.experiments.olmo_lane")
    eval_views = {
        "or1-olmo-lens-evals-v1": ["eval_", "eval_halfA_", "eval_halfB_",
                                   "eval_campaign_", "eval_merged9_",
                                   "eval_campaign9_"],
    }
    for evidence_id, prefixes in eval_views.items():
        outputs = [OLMO / f"{prefix}{set_name}.json"
                   for prefix in prefixes for set_name in EVAL_SETS]
        _maybe(evidence_id, outputs, tier="development",
               what="Six released lens evaluations on OLMo: merged primary "
                    "(24-layer paper grid), half A/B stability views, frozen "
                    "campaign lens full set + 9-layer intersection "
                    "concordance, logit baseline.",
               command="python -m jspace_official_repro.experiments.olmo_lane")
    _maybe("or1-olmo-verbal-report-v1",
           [OLMO / "verbal_report_olmo.json",
            OLMO / "verbal_report_olmo_raw.jsonl"],
           tier="development",
           what="Verbal report on OLMo, merged OR1 lens (v2 scoring from "
                "first run; INCIDENT or1-001 fix pre-applied).",
           command="python -m jspace_official_repro.experiments.olmo_lane")
    _maybe("or1-olmo-flexible-generalization-v1",
           [OLMO / "flexible_generalization_olmo.json",
            OLMO / "flexible_generalization_olmo_raw.jsonl"],
           tier="development",
           what="Flexible generalization on OLMo, merged OR1 lens.",
           command="python -m jspace_official_repro.experiments.olmo_lane")
    _maybe("or1-olmo-probe-swap-v1",
           [OLMO / "probe_swap_olmo.json", OLMO / "probe_swap_olmo_raw.jsonl"],
           tier="development",
           what="Probe-swap raw J-lens token arm on OLMo "
                "(prompt_exact_representation_adapted_raw_jlens).",
           command="python -m jspace_official_repro.experiments.olmo_lane")

    for lane in ("qwen", "olmo"):
        battery = DRIVE_ROOT / f"{lane}_battery"
        groups = {
            "a": [f"selectivity_language_{lane}.json",
                  f"selectivity_linecount_{lane}.json"],
            "b": [f"verbal_introspection_{lane}.json",
                  f"directed_modulation_{lane}.json",
                  f"dual_task_{lane}.json", f"linebreak_{lane}.json"],
            "c": [f"capacity_{lane}.json", f"ignition_{lane}.json",
                  f"top_down_{lane}.json"],
        }
        for group, names in groups.items():
            _maybe(f"or1-{lane}-battery-group-{group}-v1",
                   [battery / name for name in names],
                   tier="development",
                   what=f"Extended battery group {group.upper()} on {lane} "
                        "(complete-group banking; R2 line-break labeled).",
                   command=f"python -m jspace_official_repro.experiments.battery {lane}")

    vr2 = QWEN / "verbal_report_qwen_v2.json"
    if vr2.exists() and "or1-qwen-verbal-report-v2" not in _existing():
        registry.create(
            "or1-qwen-verbal-report-v2", tier="development",
            what="Verbal report on Qwen, v2 min-over-forms scoring with "
                 "boundary-in-context swap vectors (INCIDENT or1-001 "
                 "correction; supersedes v1).",
            command="python -m jspace_official_repro.experiments.qwen_followup",
            outputs=[vr2, QWEN / "verbal_report_qwen_v2_raw.jsonl"],
        )
        registry.supersede(
            "or1-qwen-verbal-report-v1", "or1-qwen-verbal-report-v2",
            reason="token-form scoring defect at chat boundary "
                   "(INCIDENT or1-001); v2 scores min over single-token "
                   "forms and uses the boundary in-context swap vector",
        )
        print("registered: or1-qwen-verbal-report-v2 (+ superseded v1)")

    for lane in ("qwen", "olmo"):
        _maybe(f"or1-instrument-crossover-{lane}-v1",
               [DRIVE_ROOT / f"crossover_{lane}" / f"crossover_{lane}.json"],
               tier="development",
               what=f"Bounded instrument cross-over, {lane} lane: paper "
                    "coordinate swap vs campaign protected dynamic top-10 J "
                    "ablation vs exact rank/energy matched control on the "
                    "frozen subsets"
                    + (" (new merged vs frozen campaign lens)" if lane == "olmo"
                       else "") + ".",
               command=f"python -m jspace_official_repro.experiments."
                       f"{'qwen_followup' if lane == 'qwen' else 'olmo_crossover'}")
    print("registration sweep complete")


if __name__ == "__main__":
    main()
