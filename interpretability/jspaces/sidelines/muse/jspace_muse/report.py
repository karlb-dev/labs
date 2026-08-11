"""State-of-record markdown + TeX/PDF handout."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .figures import generate_all
from .paths import DRIVE_ROOT, REPORTS, ensure_dirs
from .registry import register
from .util import log, utc_now


def _load_metrics() -> dict:
    out = {}
    mdir = DRIVE_ROOT / "metrics"
    if not mdir.exists():
        return out
    for p in mdir.glob("*.json"):
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception:
            pass
    return out


def write_state_of_record(metrics: dict) -> Path:
    h = metrics.get("battery_summary", {}).get("headline", {})
    pre = metrics.get("admission_pre_fit", {})
    post = metrics.get("admission_post_fit", {})
    fit = metrics.get("fit", {})
    gates = pre.get("gates", {})

    lines = [
        "# Muse Glimmer 30B — J-space sideline state of record",
        "",
        f"Generated: {utc_now()}  ",
        "Tier: **development / methods only**  ",
        "Branch: `jspace_muse`  ",
        "Model: `meta-models/Muse-Glimmer-30B` @ `97c77dff50b2797bcc558fa2d909761dbc575c59`",
        "",
        "## Question",
        "",
        "Does the Phase-2-style Jacobian-lens instrument work on Muse Glimmer,",
        "and is there any indication of a workspace-like structure worth chasing?",
        "",
        "## Geometry admission (pre-fit)",
        "",
        f"- Shape: {pre.get('facts', {}).get('n_layers')} × {pre.get('facts', {}).get('d_model')}",
        f"- Softcap: {pre.get('facts', {}).get('logit_softcap')}",
        f"- Identity readout parity: `{gates.get('identity_parity_ok')}`",
        f"- Finite activations: `{gates.get('finite_acts')}`",
        f"- Linearity canary odd-symmetry ok: `{gates.get('linearity_odd_symmetry_ok')}`",
        f"- **all_pass: `{gates.get('all_pass')}`**",
        "",
        "## Lens fit",
        "",
        f"- n_prompts: {fit.get('n_prompts') or fit.get('merged', {}).get('n_prompts')}",
        f"- dim_batch: {fit.get('dim_batch')}",
        f"- source layers: {fit.get('source_layers')}",
        f"- target: {fit.get('target_layer')}",
        f"- merged sha256: `{fit.get('merged', {}).get('sha256', 'n/a')}`",
        "",
        "## Post-fit admission",
        "",
        f"- readout parity ok: `{post.get('readout_parity', {}).get('ok')}`",
        f"- g-fold min cosine: `{post.get('g_folding', {}).get('min_cosine')}`",
        f"- g-fold immaterial (≥0.99): `{post.get('g_folding', {}).get('immaterial')}`",
        "",
        "## Battery headline",
        "",
        "| cell | value |",
        "|---|---|",
        f"| J-lens advantage (band, mean ranks) | {h.get('depth_j_advantage_band')} |",
        f"| Selectivity-language contrast | {h.get('selectivity_contrast')} |",
        f"| Modulation focus improves | {h.get('modulation_focus_improves')} |",
        f"| Dual-task math interference | {h.get('dual_math_interference')} |",
        f"| Capacity mean active@rank≤5 | {h.get('capacity_mean_r5')} |",
        f"| Ignition median α@rank≤5 | {h.get('ignition_median_alpha')} |",
        f"| Verbal-report top5 hit rate | {h.get('vr_top5_hit_rate')} |",
        f"| Ablation mean J damage | {h.get('ablation_j_damage')} |",
        f"| Ablation mean random damage | {h.get('ablation_random_damage')} |",
        f"| Ablation n selective | {h.get('ablation_n_selective')} |",
        "",
        "## Reading (provisional)",
        "",
        _reading(h, gates),
        "",
        "## Paths",
        "",
        f"- Drive root: `{DRIVE_ROOT}`",
        "- Package: `interpretability/jspaces/sidelines/muse/`",
        "",
        "No confirmatory claims. Frozen campaign registries untouched.",
        "",
    ]
    text = "\n".join(lines)
    for dest in (
        DRIVE_ROOT / "reports" / "MUSE_STATE_OF_RECORD.md",
        REPORTS / "MUSE_STATE_OF_RECORD.md",
    ):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        log(f"wrote {dest}")
    return DRIVE_ROOT / "reports" / "MUSE_STATE_OF_RECORD.md"


def _reading(h: dict, gates: dict) -> str:
    if not gates.get("all_pass"):
        return (
            "Geometry admission did **not** fully pass. Treat all battery "
            "numbers as instrument diagnostics only; do not chase workspace "
            "mechanism claims on this branch without a repair."
        )
    bits = []
    adv = h.get("depth_j_advantage_band")
    if adv is not None:
        if adv > 5:
            bits.append(
                f"J-lens shows a **band rank advantage** of ~{adv:.1f} over logit lens "
                "(direction of the paper's depth-recovery claim)."
            )
        elif adv > 0:
            bits.append(
                f"J-lens advantage is **weak** (~{adv:.1f} rank); closer to OLMo "
                "near-identity behaviour than Qwen."
            )
        else:
            bits.append(
                "J-lens does **not** beat logit lens in the band (OLMo-like / opposite)."
            )
    sc = h.get("selectivity_contrast")
    if sc is not None:
        bits.append(
            f"Selectivity-language contrast = {sc:.2f} "
            f"({'reproduces direction' if sc > 0.2 else 'weak/null'})."
        )
    if h.get("modulation_focus_improves"):
        bits.append("Directed modulation: focus improves rank (workspace-compatible).")
    elif h.get("modulation_focus_improves") is False:
        bits.append("Directed modulation: focus did **not** improve rank.")
    jd, rd = h.get("ablation_j_damage"), h.get("ablation_random_damage")
    if jd is not None and rd is not None:
        if jd > rd + 5:
            bits.append(
                f"Protected ablation: J-direction damage ({jd:.1f}) exceeds "
                f"random ({rd:.1f}) — **promising causal selectivity**."
            )
        else:
            bits.append(
                f"Protected ablation: J ({jd:.1f}) vs random ({rd:.1f}) — "
                "no clear selective causal edge."
            )
    vr = h.get("vr_top5_hit_rate")
    if vr is not None:
        bits.append(
            f"Verbal-report top5 hit rate {vr:.0%} "
            f"(open models were ~25–30% in OR1; paper 88%)."
        )
    if not bits:
        return "Insufficient banked metrics for a reading yet."
    bits.append(
        "**Bottom line:** development/methods only. If geometry + selective "
        "ablation + modulation co-occur, a larger confirmatory sideline is "
        "licensed to preregister; otherwise park as a negative/weak instrument result."
    )
    return " ".join(bits)


def write_tex(metrics: dict) -> Path:
    h = metrics.get("battery_summary", {}).get("headline", {})
    pre = metrics.get("admission_pre_fit", {})
    gates = pre.get("gates", {})
    fit = metrics.get("fit", {})

    def fmt(x):
        if x is None:
            return "n/a"
        if isinstance(x, float):
            return f"{x:.3g}"
        return str(x)

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\title{Muse Glimmer 30B \\ J-space instrument probe}
\author{jspace\_muse sideline \\ development/methods tier}
\date{\today}
\begin{document}
\maketitle

\begin{abstract}
We port the Phase-2 Jacobian-lens assay to Meta's Muse Glimmer 30B
(52 layers $\times$ $d=6656$), test residual-stack geometry, fit a
120-prompt WikiText lens, and run a compact battery of the strongest
open-model cells from prior OLMo/Qwen work. Tier: development/methods only.
\end{abstract}

\section{Geometry}
\begin{itemize}
  \item Shape: %(n_layers)s $\times$ %(d_model)s; softcap=%(softcap)s
  \item Identity parity: %(parity)s
  \item All pre-fit gates: %(all_pass)s
\end{itemize}

\section{Lens fit}
\begin{itemize}
  \item $n=120$ WikiText (Phase-2 draw-A recipe)
  \item dim\_batch=%(dim_batch)s; target layer %(target)s
  \item sources: %(n_src)s layers
\end{itemize}

\section{Battery headline}
\begin{center}
\begin{tabular}{lr}
\toprule
Cell & Value \\
\midrule
J-lens band advantage (ranks) & %(jadv)s \\
Selectivity-language contrast & %(sel)s \\
Modulation focus improves & %(mod)s \\
Dual-task math interference & %(dual)s \\
Capacity mean active@$\le$5 & %(cap)s \\
Ignition median $\alpha$@$\le$5 & %(ign)s \\
Verbal-report top5 hit rate & %(vr)s \\
Ablation J / random damage & %(ablj)s / %(ablr)s \\
\bottomrule
\end{tabular}
\end{center}

\section{Figures}
""" % {
        "n_layers": pre.get("facts", {}).get("n_layers", "?"),
        "d_model": pre.get("facts", {}).get("d_model", "?"),
        "softcap": pre.get("facts", {}).get("logit_softcap", "?"),
        "parity": gates.get("identity_parity_ok", "?"),
        "all_pass": gates.get("all_pass", "?"),
        "dim_batch": fit.get("dim_batch", "?"),
        "target": fit.get("target_layer", 51),
        "n_src": len(fit.get("source_layers") or []),
        "jadv": fmt(h.get("depth_j_advantage_band")),
        "sel": fmt(h.get("selectivity_contrast")),
        "mod": fmt(h.get("modulation_focus_improves")),
        "dual": fmt(h.get("dual_math_interference")),
        "cap": fmt(h.get("capacity_mean_r5")),
        "ign": fmt(h.get("ignition_median_alpha")),
        "vr": fmt(h.get("vr_top5_hit_rate")),
        "ablj": fmt(h.get("ablation_j_damage")),
        "ablr": fmt(h.get("ablation_random_damage")),
    }

    fig_dir = DRIVE_ROOT / "figures"
    for name, cap in [
        ("fig_depth_profile.png", "J-lens vs logit-lens depth profile"),
        ("fig_modulation.png", "Directed modulation"),
        ("fig_ignition.png", "Ignition curves"),
        ("fig_headline_bars.png", "Headline metrics"),
    ]:
        p = fig_dir / name
        if p.exists():
            # copy next to tex for pdflatex
            tex += (
                f"\n\\begin{{figure}}[h]\\centering\\includegraphics[width=0.92\\textwidth]"
                f"{{{name}}}\\caption{{{cap}}}\\end{{figure}}\n"
            )

    tex += r"""
\section{Reading}
See \texttt{MUSE\_STATE\_OF\_RECORD.md} for the live narrative. No confirmatory
claims; frozen campaign registries were not modified.

\end{document}
"""
    tex_path = DRIVE_ROOT / "reports" / "muse_jspace_handout.tex"
    tex_path.write_text(tex)
    # copy figures beside tex
    import shutil
    for fig in (DRIVE_ROOT / "figures").glob("fig_*.png"):
        shutil.copy2(fig, tex_path.parent / fig.name)
    # also to repo reports
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "muse_jspace_handout.tex").write_text(tex)
    return tex_path


def compile_pdf(tex_path: Path) -> Path | None:
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=str(tex_path.parent),
            check=False,
            capture_output=True,
            timeout=120,
        )
        pdf = tex_path.with_suffix(".pdf")
        if pdf.exists():
            log(f"wrote {pdf}")
            return pdf
    except FileNotFoundError:
        log("pdflatex not installed; skipping PDF (tex written)")
    except Exception as e:
        log(f"pdflatex failed: {e}")
    return None


def build_all() -> dict:
    ensure_dirs()
    generate_all()
    metrics = _load_metrics()
    sor = write_state_of_record(metrics)
    tex = write_tex(metrics)
    pdf = compile_pdf(tex)
    outs = [sor, tex]
    if pdf:
        outs.append(pdf)
    register({
        "evidence_id": "muse-report-v1",
        "what": "Muse state-of-record + TeX/PDF handout + figures",
        "command": "python -m jspace_muse.report / jspace-muse report",
        "outputs": outs,
    })
    return {"state_of_record": str(sor), "tex": str(tex), "pdf": str(pdf) if pdf else None}


if __name__ == "__main__":
    print(build_all())
