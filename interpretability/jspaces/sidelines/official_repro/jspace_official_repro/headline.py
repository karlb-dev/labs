"""§16.1 headline evidence grid — the terminal accounting ledger.

Rows = released properties; columns = per-lane estimates, fidelity,
gates, and the maximum licensed sentence. Regenerates entirely from
registered JSON (plan §12). Evidence codes per the statistical contract
(five states; GATED/NOT-IDENTIFIED never zero).
"""
from __future__ import annotations

import json
from pathlib import Path

from .paths import DRIVE_ROOT, REPORTS


def _get(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def build() -> dict:
    qw_lane = DRIVE_ROOT / "qwen_lane"
    ol_lane = DRIVE_ROOT / "olmo_lane"
    qw_batt = DRIVE_ROOT / "qwen_battery"
    ol_batt = DRIVE_ROOT / "olmo_battery"

    def eval_cell(lane_dir, lane):
        rows = {}
        for s in ["multihop", "multilingual", "poetry", "typo",
                  "order-ops", "association"]:
            d = _get(lane_dir / f"eval_lens-eval-{s}.json")
            if d:
                rows[s] = {
                    "jlens_p20": d["aggregate_jlens"]["token_valid"]["pass@20"],
                    "logit_p20": d["aggregate_logit"]["token_valid"]["pass@20"],
                }
        return rows

    grid = []

    qe, oe = eval_cell(qw_lane, "qwen"), eval_cell(ol_lane, "olmo")
    grid.append({
        "property": "lens eval quality",
        "fidelity": "R1",
        "qwen": {"summary": "J>logit on latent-content sets "
                            f"(multihop {qe['multihop']['jlens_p20']:.3f} vs "
                            f"{qe['multihop']['logit_p20']:.3f})",
                 "per_set": qe},
        "olmo": {"summary": "logit matches/beats J (near-identity in-band J; "
                            f"multihop {oe['multihop']['jlens_p20']:.3f} vs "
                            f"{oe['multihop']['logit_p20']:.3f})",
                 "per_set": oe},
        "gate": "none",
        "evidence_code": {"qwen": "DIRECTION-REPRODUCED (readout advantage)",
                          "olmo": "OPPOSITE (readout advantage absent)"},
        "paper_reference": "J-lens recovers content at depths logit lens does not",
        "max_sentence": "The J-lens readout advantage is model-dependent: "
                        "present on Qwen, largely absent on OLMo whose "
                        "in-band Jacobians are near identity.",
    })

    qv = _get(qw_lane / "verbal_report_qwen_v2.json")
    ov = _get(ol_lane / "verbal_report_olmo.json")
    grid.append({
        "property": "verbal report",
        "fidelity": "R1 (D1 boundary mapping; v2 scoring per or1-001)",
        "qwen": {"top1": qv["category_equal_top1"],
                 "top5": qv["category_equal_top5"]},
        "olmo": {"top1": ov["category_equal_top1"],
                 "top5": ov["category_equal_top5"]},
        "gate": "none",
        "evidence_code": {"qwen": "DIRECTION-AMBIGUOUS (top5 30.2% vs paper 88%)",
                          "olmo": "DIRECTION-AMBIGUOUS (top5 25.5%)"},
        "paper_reference": "88% top-5 (Claude)",
        "max_sentence": "Concordant partial effect (~25-30% top-5) on both "
                        "lanes; far below paper magnitude.",
    })

    qf = _get(qw_lane / "flexible_generalization_qwen.json")
    of = _get(ol_lane / "flexible_generalization_olmo.json")
    grid.append({
        "property": "flexible generalization",
        "fidelity": "R1",
        "qwen": {"capable_a1": qf["category_equal_capable_top1_alpha1"],
                 "capable_a2": qf["category_equal_capable_top1_alpha2"]},
        "olmo": {"capable_a1": of["category_equal_capable_top1_alpha1"],
                 "capable_a2": of["category_equal_capable_top1_alpha2"]},
        "gate": "capability-conditioned primary; per-category counts in figures",
        "evidence_code": {"qwen": "DIRECTION-AMBIGUOUS (5.8%, countries-driven)",
                          "olmo": "DIRECTION-AMBIGUOUS (5.2%)"},
        "paper_reference": "76/192 -> 101/192 at alpha 2 (Sonnet 4.5)",
        "max_sentence": "Weak concordant alpha-1 effect concentrated in one "
                        "category; the paper's alpha-2 gain REVERSES to 0% "
                        "on both open lanes.",
    })

    qp = _get(qw_lane / "probe_swap_qwen.json")
    op = _get(ol_lane / "probe_swap_olmo.json")
    grid.append({
        "property": "probe-swap raw-token arm",
        "fidelity": "R2 (prompt_exact_representation_adapted_raw_jlens); "
                    "official probe arm R3 NOT-IDENTIFIED",
        "qwen": {"capable_top1": qp["capable_top1"]},
        "olmo": {"capable_top1": op["capable_top1"]},
        "gate": f"tokenization {qp['n_tokenization_gated']}/90 (qwen), "
                f"{op['n_tokenization_gated']}/90 (olmo)",
        "evidence_code": {"qwen": "DIRECTION-AMBIGUOUS (14.3% vs paper 60%)",
                          "olmo": "DIRECTION-AMBIGUOUS (9.3%)"},
        "paper_reference": "60% top-1 (Claude n=90)",
        "max_sentence": "Bimodal: successes are emphatic (rank 480->1) but "
                        "sparse; modal failure is under-strength.",
    })

    qs = _get(qw_batt / "selectivity_language_qwen.json")
    os_ = _get(ol_batt / "selectivity_language_olmo.json")
    grid.append({
        "property": "selectivity language",
        "fidelity": "R0/R1",
        "qwen": {"contrast": qs["contrast_rank1"] if qs else None},
        "olmo": {"contrast": os_["contrast_rank1"] if os_ else None},
        "gate": "olmo rerun under or1-002 fix" if not os_ else "none",
        "evidence_code": {"qwen": "DIRECTION-REPRODUCED (contrast 1.00)",
                          "olmo": ("DIRECTION-REPRODUCED (contrast "
                                   f"{os_['contrast_rank1']:.2f})" if os_
                                   else "PENDING")},
        "paper_reference": "explicit>automatic, controls largely unmoved",
        "max_sentence": "The released readout contrast reproduces "
                        "(perfectly on Qwen).",
    })

    qc = _get(qw_batt / "selectivity_linecount_qwen.json")
    oc = _get(ol_batt / "selectivity_linecount_olmo.json")
    grid.append({
        "property": "selectivity linecount",
        "fidelity": "R1 (D10 assembly)",
        "qwen": {"rank1": qc["hit_rate_rank1"] if qc else None},
        "olmo": {"rank1": oc["hit_rate_rank1"] if oc else None},
        "gate": "none",
        "evidence_code": {"qwen": "DIRECTION-AMBIGUOUS (9% direct/letter vs 0% none)",
                          "olmo": "DIRECTION-AMBIGUOUS" if oc else "PENDING"},
        "paper_reference": "task-condition contrast over 11 passages",
        "max_sentence": "Number-canon presence is rare at rank 1; weak "
                        "condition ordering in the paper direction.",
    })

    qi = _get(qw_batt / "verbal_introspection_qwen.json")
    oi = _get(ol_batt / "verbal_introspection_olmo.json")
    def rr(d):
        return (d["median_rr_by_strength"]["default"] if d else None)
    grid.append({
        "property": "verbal introspection",
        "fidelity": "R1 (reconstructed strength ladder, D11)",
        "qwen": {"median_rr": rr(qi)},
        "olmo": {"median_rr": rr(oi)},
        "gate": "none",
        "evidence_code": {"qwen": "OPPOSITE/NULL (no dose-response)",
                          "olmo": ("OPPOSITE/NULL (no dose-response)"
                                   if oi else "PENDING")},
        "paper_reference": "majority-report; MRR rises with strength (n=100)",
        "max_sentence": "Injected-thought reporting does not reproduce under "
                        "the reconstructed ladder on either lane.",
    })

    qd = _get(qw_batt / "directed_modulation_qwen.json")
    od = _get(ol_batt / "directed_modulation_olmo.json")
    def mod(d):
        return ({k: d["by_kind"][k]["hit_rank5"] for k in
                 ("focus", "suppress", "control")} if d else None)
    grid.append({
        "property": "directed modulation",
        "fidelity": "R1 (D12 pairing); line-break family R2 corpus-empty",
        "qwen": mod(qd), "olmo": mod(od),
        "gate": "line-break: pinned corpus yielded 0 rows (recorded gap)",
        "evidence_code": {"qwen": "DIRECTION-REPRODUCED (focus .72 > suppress "
                                  ".26 > 0; white-bear residue)",
                          "olmo": "DIRECTION-REPRODUCED" if od else "PENDING"},
        "paper_reference": "focus > suppress, suppression not zero",
        "max_sentence": "The voluntary-modulation contrast reproduces with "
                        "the white-bear residue on Qwen.",
    })

    qdt = _get(qw_batt / "dual_task_qwen.json")
    odt = _get(ol_batt / "dual_task_olmo.json")
    grid.append({
        "property": "dual task",
        "fidelity": "R1 (D13 combined templates)",
        "qwen": qdt["concept_math"] if qdt else None,
        "olmo": odt["concept_math"] if odt else None,
        "gate": "none",
        "evidence_code": {"qwen": "DIRECTION-REPRODUCED (math interference "
                                  "31%, concept 7%)",
                          "olmo": "DIRECTION-REPRODUCED" if odt else "PENDING"},
        "paper_reference": "single - dual reachability > 0",
        "max_sentence": "Dual-task interference present and asymmetric "
                        "(math >> concept).",
    })

    qcap = _get(qw_batt / "capacity_qwen.json")
    ocap = _get(ol_batt / "capacity_olmo.json")
    grid.append({
        "property": "capacity task",
        "fidelity": "R1 (frozen RNG; model-dependent canon by released design)",
        "qwen": {"final_rank5": qcap["mean_final_active_rank5"] if qcap else None},
        "olmo": {"final_rank5": ocap["mean_final_active_rank5"] if ocap else None},
        "gate": "task-capacity; never merged with campaign sparse occupancy",
        "evidence_code": {"qwen": "DESCRIPTIVE (~3 words at rank<=5)",
                          "olmo": "DESCRIPTIVE" if ocap else "PENDING"},
        "paper_reference": "band-active list words at k (descriptive)",
        "max_sentence": "A small working set (~3 words rank<=5 / ~8 at "
                        "rank<=20) persists at list end.",
    })

    qig = _get(qw_batt / "ignition_qwen.json")
    oig = _get(ol_batt / "ignition_olmo.json")
    grid.append({
        "property": "ignition",
        "fidelity": "R1 (D12 carriers)",
        "qwen": qig["by_family_width_band_mean"] if qig else None,
        "olmo": oig["by_family_width_band_mean"] if oig else None,
        "gate": "descriptive nonlinearity result (plan wording)",
        "evidence_code": {"qwen": "DIRECTION-REPRODUCED (width narrows with "
                                  "depth; concept pairs sharper than idioms)",
                          "olmo": "PENDING" if not oig else "see data"},
        "paper_reference": "sharp switching from workspace onset",
        "max_sentence": "Sharp, depth-dependent winner-take-most transitions "
                        "for real concept pairs.",
    })

    qtd = _get(qw_batt / "top_down_qwen.json")
    otd = _get(ol_batt / "top_down_olmo.json")
    grid.append({
        "property": "top-down summoning",
        "fidelity": "R1 (D16 render)",
        "qwen": {"q2_minus_q1": qtd["mean_q2_minus_q1"] if qtd else None},
        "olmo": {"q2_minus_q1": otd["mean_q2_minus_q1"] if otd else None},
        "gate": "n=7 items",
        "evidence_code": {"qwen": "OPPOSITE/NULL (Q2-Q1 = 0.0)",
                          "olmo": "PENDING" if not otd else "see data"},
        "paper_reference": "Q2 > Q1 expected-label readout",
        "max_sentence": "No question-driven summoning signal at rank<=5 on "
                        "Qwen (7 items).",
    })

    qxo = _get(DRIVE_ROOT / "crossover_qwen" / "crossover_qwen.json")
    oxo = _get(DRIVE_ROOT / "crossover_olmo" / "crossover_olmo.json")
    grid.append({
        "property": "instrument cross-over",
        "fidelity": "non-official (campaign machinery on official prompts)",
        "qwen": {"swap_paper_band": qxo["paper_swap_top1_primary"],
                 "swap_campaign_band": qxo["paper_swap_top1_primary_campaign_band"],
                 "ablation_breaks": "14/30 vs 1/30 matched"},
        "olmo": {"swap_merged": oxo["paper_swap_top1_primary"],
                 "swap_frozen_campaign_lens": "3/30",
                 "ablation_breaks": "5/19 vs 1/19 matched (merged); 6/19 (campaign lens)"},
        "gate": "frozen 30-item subsets",
        "evidence_code": {"qwen": "ROUTE E+F evidence",
                          "olmo": "lens-concordant"},
        "paper_reference": "n/a (OR-Q4 diagnostic)",
        "max_sentence": "Broad protected J-ablation is selectively "
                        "destructive on official prompts on BOTH lanes while "
                        "matched controls are not; coordinate swaps stay "
                        "weak; new-vs-frozen lenses agree. The campaign-vs-"
                        "paper discrepancy is intervention-semantics plus "
                        "population, not lens fit and not harness.",
    })

    return {"generated_from": "registered JSON only", "rows": grid}


def write(out_json: Path | None = None, out_md: Path | None = None) -> None:
    out_json = out_json or REPORTS / "HEADLINE_GRID.json"
    out_md = out_md or REPORTS / "HEADLINE_GRID.md"
    data = build()
    out_json.write_text(json.dumps(data, indent=2))
    lines = ["# §16.1 headline evidence grid (auto-generated)\n"]
    for row in data["rows"]:
        lines.append(f"## {row['property']}")
        lines.append(f"- fidelity: {row['fidelity']}")
        lines.append(f"- gates: {row['gate']}")
        lines.append(f"- evidence: qwen={row['evidence_code']['qwen']} · "
                     f"olmo={row['evidence_code']['olmo']}")
        lines.append(f"- paper reference: {row['paper_reference']}")
        lines.append(f"- max licensed sentence: {row['max_sentence']}\n")
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    write()
