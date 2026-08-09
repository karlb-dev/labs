"""Extract report numbers from registered JSON into TeX macros.

Every number the report prints regenerates from these inputs (plan §12);
the TeX never hard-codes a result value. Macro prefixes: ``qw`` = Qwen
lane, ``ol`` = OLMo lane; ``qwvrv2*`` = the INCIDENT or1-001 v2 rerun.
"""
from __future__ import annotations

import json
from pathlib import Path

from .paths import DRIVE_ROOT, REPORTS

_K_WORDS = {1: "one", 5: "five", 20: "twenty"}

EVAL_SHORT = {"lens-eval-multihop": "multihop",
              "lens-eval-multilingual": "multilingual",
              "lens-eval-poetry": "poetry",
              "lens-eval-typo": "typo",
              "lens-eval-order-ops": "orderops",
              "lens-eval-association": "association"}


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value):
    return "—" if value is None else f"{100 * value:.1f}\\%"


def lane_macros(lane_dir: Path, lane: str, prefix: str) -> dict[str, str]:
    macros: dict[str, str] = {}
    eval_prefixes = [("eval_", ""), ("eval_alllayers_", "all")]
    if lane == "olmo":
        eval_prefixes += [("eval_halfA_", "halfa"), ("eval_halfB_", "halfb"),
                          ("eval_campaign_", "camp"),
                          ("eval_merged9_", "mnine"),
                          ("eval_campaign9_", "cnine")]
    for set_name, short in EVAL_SHORT.items():
        for file_prefix, tag in eval_prefixes:
            path = lane_dir / f"{file_prefix}{set_name}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for which in ("jlens", "logit"):
                agg = data[f"aggregate_{which}"]["token_valid"]
                for k in (1, 5, 20):
                    macros[f"{prefix}{tag}{short}{which}p{_K_WORDS[k]}"] = (
                        _fmt(agg[f"pass@{k}"]))
            macros[f"{prefix}{tag}{short}n"] = str(data["n_items"])
            att = data["aggregate_jlens"]["attrition"]
            macros[f"{prefix}{tag}{short}gated"] = str(
                att["gated_intermediates"])

    def _vr(path, vr_prefix):
        if not path.exists():
            return
        data = json.loads(path.read_text())
        macros[f"{vr_prefix}cattopone"] = _pct(data["category_equal_top1"])
        macros[f"{vr_prefix}cattopfive"] = _pct(data["category_equal_top5"])
        macros[f"{vr_prefix}trialtopone"] = _pct(data["trial_weighted_top1"])
        macros[f"{vr_prefix}ncat"] = str(data["n_categories"])
        executed = sum(c["n_executed"] for c in data["categories"])
        macros[f"{vr_prefix}nexec"] = str(executed)
        improved = worsened = 0
        for c in data["categories"]:
            for t in c["trials"]:
                if t.get("state") == "EXECUTED":
                    if t["rank_after"] < t["rank_before"]:
                        improved += 1
                    elif t["rank_after"] > t["rank_before"]:
                        worsened += 1
        macros[f"{vr_prefix}improved"] = str(improved)
        macros[f"{vr_prefix}worsened"] = str(worsened)

    _vr(lane_dir / f"verbal_report_{lane}.json", f"{prefix}vr")
    if lane == "qwen":
        _vr(lane_dir / "verbal_report_qwen_v2.json", "qwvrvtwo")

    fg = lane_dir / f"flexible_generalization_{lane}.json"
    if fg.exists():
        data = json.loads(fg.read_text())
        macros[f"{prefix}fgcapaone"] = _pct(
            data["category_equal_capable_top1_alpha1"])
        macros[f"{prefix}fgcapatwo"] = _pct(
            data["category_equal_capable_top1_alpha2"])
        macros[f"{prefix}fgdiagaone"] = _pct(
            data["category_equal_diagnostic_top1_alpha1"])
        macros[f"{prefix}fgncapable"] = str(
            sum(c["n_capable"] for c in data["categories"]))
        macros[f"{prefix}fgnexec"] = str(data["total_executed_alpha1"])
        base = [c["baseline_correct_fraction"] for c in data["categories"]]
        macros[f"{prefix}fgbaseline"] = _pct(sum(base) / len(base))
        for c in data["categories"]:
            key = c["category"][:5]
            macros[f"{prefix}fg{key}capaone"] = _pct(c["capable_top1_alpha1"])
            macros[f"{prefix}fg{key}capatwo"] = _pct(c["capable_top1_alpha2"])
            macros[f"{prefix}fg{key}ncap"] = str(c["n_capable"])

    ps = lane_dir / f"probe_swap_{lane}.json"
    if ps.exists():
        data = json.loads(ps.read_text())
        macros[f"{prefix}psncap"] = str(data["n_baseline_capable"])
        macros[f"{prefix}psnexec"] = str(data["n_executed"])
        macros[f"{prefix}psntokgated"] = str(data["n_tokenization_gated"])
        macros[f"{prefix}pscaptopone"] = _pct(data["capable_top1"])
        macros[f"{prefix}psdiagtopone"] = _pct(data["diagnostic_top1"])
        macros[f"{prefix}psmulti"] = _pct(data["multihop_capable_top1"])
        macros[f"{prefix}psnonmulti"] = _pct(data["non_multihop_capable_top1"])

    adm = lane_dir / f"{lane}_admission.json"
    if adm.exists():
        data = json.loads(adm.read_text())
        macros[f"{prefix}parity"] = (
            f"{data['readout_parity']['max_abs_diff']:.1e}")
        macros[f"{prefix}gfold"] = _fmt(data["g_folding"]["min_cosine"])
    return macros


def splithalf_macros() -> dict[str, str]:
    macros: dict[str, str] = {}
    path = DRIVE_ROOT / "olmo_fit" / "splithalf_operator_audit.json"
    if path.exists():
        data = json.loads(path.read_text())
        macros["olshmedfrob"] = _pct(data["median_sym_rel_frobenius"])
        band = [r for r in data["per_layer"] if 24 <= r["layer"] <= 58]
        if band:
            frobs = sorted(r["sym_rel_frobenius"] for r in band)
            macros["olshbandfrob"] = _pct(frobs[len(frobs) // 2])
    route = DRIVE_ROOT / "olmo_fit" / "fit_route.json"
    if route.exists():
        data = json.loads(route.read_text())
        macros["olfitdimbatch"] = str(data["dim_batch"])
        macros["olfitsperprompt"] = _fmt(data["measured_s_per_prompt"], 1)
        macros["olfitmerged"] = str(data["merged_n"])
    return macros


def write_numbers_tex(out: Path | None = None) -> Path:
    out = out or (REPORTS / "tex" / "numbers.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    macros = lane_macros(DRIVE_ROOT / "qwen_lane", "qwen", "qw")
    macros.update(lane_macros(DRIVE_ROOT / "olmo_lane", "olmo", "ol"))
    macros.update(splithalf_macros())
    lines = ["% AUTO-GENERATED by jspace_official_repro.report_data — do not edit"]
    for key, value in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{key}}}{{{value}}}")
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    path = write_numbers_tex()
    print("wrote", path)
