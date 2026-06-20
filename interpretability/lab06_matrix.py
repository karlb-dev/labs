#!/usr/bin/env python
"""Lab 6 validation matrix driver.

Loads ONE model and runs every (behavior, scope) cell IN-PROCESS, so a 32B
model is loaded once rather than once per cell. After each cell it:
  * copies the cell's run directory to Google Drive
     (MyDrive/interpret/lab06_matrix_<date>/<model>/<behavior>_<scope>/),
  * regenerates FINDINGS.md + matrix_results.json (Drive AND a git-tracked
    repo copy under interpretability/lab06_matrix/<date>/),
so partial progress is never lost and a confirmed NO is recorded as a result.

Usage:
  python lab06_matrix.py --model gpt2 --dtype float32 --tier a
  python lab06_matrix.py --model allenai/Olmo-3-1125-32B --dtype bfloat16 --tier b
  python lab06_matrix.py --model gpt2 --behaviors induction_p3,successor \
      --heads-only-behaviors induction_p3,recall
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import time
import traceback

import interp_bench as bench
from labs import lab06_circuit_discovery as lab06

ALL_BEHAVIORS = [
    "induction_p3",
    "induction_p2",
    "successor",
    "ioi",
    "agreement",
    "agreement_long",
    "taskvec",
    "recall",
]
# Default behaviors that additionally get a heads_only contrast cell: the
# MLP-mediated recall behavior and the canonical induction case, so the
# "not heads-only" determination has both columns.
DEFAULT_HEADS_ONLY = ["induction_p3", "recall"]


def parse_matrix_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16"))
    p.add_argument("--tier", default="auto", choices=("auto", "a", "b", "c"))
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--behaviors", default=",".join(ALL_BEHAVIORS), help="comma-separated heads_and_mlps behaviors")
    p.add_argument("--heads-only-behaviors", default=",".join(DEFAULT_HEADS_ONLY),
                   help="comma-separated behaviors to ALSO run in heads_only scope (the 'not heads-only' contrast)")
    p.add_argument("--resample-draws", type=int, default=5)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--allow-mixed-period", action="store_true")
    p.add_argument("--swa-lengths", default="1024,4096,5120")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--attn-implementation", default="auto")
    p.add_argument("--date", default=None, help="matrix date slug (default: now). Reuse to resume into the same matrix.")
    p.add_argument("--drive-root", default="/content/drive/MyDrive/interpret")
    p.add_argument("--repo-root", default=str(bench.COURSE_ROOT))
    p.add_argument("--skip-existing", action="store_true",
                   help="skip cells whose metrics.json already exists (resume a partial matrix)")
    return p.parse_args(argv)


def _base_args(m: argparse.Namespace) -> argparse.Namespace:
    """A fully-resolved bench arg namespace for lab6, then tier defaults."""
    argv = [
        "--lab", "lab6",
        "--model", m.model,
        "--dtype", m.dtype,
        "--tier", m.tier,
        "--device", m.device,
        "--max-examples", str(m.max_examples),
        "--resample-draws", str(m.resample_draws),
        "--swa-lengths", m.swa_lengths,
    ]
    if m.no_plots:
        argv.append("--no-plots")
    if m.allow_mixed_period:
        argv.append("--allow-mixed-period")
    if getattr(m, "trust_remote_code", False):
        argv.append("--trust-remote-code")
    if getattr(m, "attn_implementation", "auto") and m.attn_implementation != "auto":
        argv += ["--attn-implementation", m.attn_implementation]
    base = bench.parse_args(argv)
    bench.apply_tier_defaults(base)
    base.max_examples = m.max_examples  # never let tier defaults shrink the matrix below the >=8 gate
    return base


def _copy_to_drive(cell_dir: pathlib.Path, dst: pathlib.Path) -> str:
    try:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cell_dir, dst, dirs_exist_ok=True)
        return "ok"
    except Exception as exc:  # Drive FUSE can be flaky; never abort the matrix on a copy
        return f"copy-failed: {type(exc).__name__}: {exc}"


def _cell_summary(metrics: dict) -> dict:
    circuits = metrics.get("circuits", {}) or {}
    knee = circuits.get("knee", {}) or {}
    kdisc = knee.get("discovery") or {}
    kheld = knee.get("heldout") or {}
    return {
        "behavior": metrics.get("behavior"),
        "scope": metrics.get("scope"),
        "verdict": metrics.get("verdict"),
        "verdict_reason": metrics.get("verdict_reason"),
        "aborted": metrics.get("aborted", False),
        "base_metric": metrics.get("base_metric"),
        "n_discovery": metrics.get("n_discovery"),
        "n_heldout": metrics.get("n_heldout"),
        "knee_n_nodes": metrics.get("knee_n_nodes"),
        "floor_n_nodes": metrics.get("floor_n_nodes"),
        "disc_faith_mean": kdisc.get("faith_mean"),
        "disc_faith_resample": kdisc.get("faith_resample"),
        "held_faith_mean": kheld.get("faith_mean"),
        "held_faith_resample": metrics.get("headline_heldout_faith_resample"),
        "motif_core_held_resample": metrics.get("motif_core_heldout_faith_resample"),
        "mean_minus_resample_gap": metrics.get("mean_minus_resample_gap_discovery"),
        "knee_minus_floor": metrics.get("knee_minus_floor_faithfulness"),
        "induction_motif_present": metrics.get("induction_motif_present"),
        "n_suppression_heads": len(metrics.get("suppression_heads", []) or []),
        "n_mlp_positive": len(metrics.get("mlp_positive_causal", []) or []),
        "mlp_in_knee": metrics.get("mlp_in_knee_circuit", []),
        "edge_claimed": bool((metrics.get("edge") or {}).get("claimed")),
        "edge": (metrics.get("edge") or {}).get("edge"),
    }


def _fnum(x) -> str:
    return "—" if x is None else (f"{x:+.2f}" if isinstance(x, (int, float)) else str(x))


def _reconcile_scope(rows: list[dict]) -> dict:
    """For each behavior, decide NOT-HEADS-ONLY by comparing scopes."""
    by_beh: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_beh.setdefault(r["behavior"], {})[r["scope"]] = r
    notes = {}
    for beh, scopes in by_beh.items():
        ho = scopes.get("heads_only")
        hm = scopes.get("heads_and_mlps")
        if ho and hm and ho.get("held_faith_resample") is not None and hm.get("held_faith_resample") is not None:
            ho_pass = (ho["held_faith_resample"] or -9) >= lab06.FAITHFULNESS_FLOOR
            hm_pass = (hm["held_faith_resample"] or -9) >= lab06.FAITHFULNESS_FLOOR
            if hm_pass and not ho_pass and hm.get("mlp_in_knee"):
                notes[beh] = "NOT HEADS-ONLY (heads_and_mlps passes, heads_only fails; MLPs in knee: %s)" % ", ".join(hm["mlp_in_knee"])
    return notes


def write_findings(out_dirs: list[pathlib.Path], model: str, date: str, rows: list[dict]) -> None:
    scope_notes = _reconcile_scope(rows)
    lines = [
        f"# Lab 6 validation matrix — `{model}`",
        "",
        f"Matrix `lab06_matrix_{date}`. One question per cell: does this model implement a clean, "
        "transferable circuit for this behavior — yes, no, or not-as-a-heads-only-graph? "
        "**A confirmed NO is a success.** Headline faithfulness is RESAMPLE (interchange) ablation; "
        "mean ablation is shown for the inflation comparison.",
        "",
        "## Results matrix",
        "",
        "| behavior | scope | verdict | held-out F (resample) | disc F (resample / mean) | motif-core held F | knee/floor nodes | mean−resample gap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["behavior"], x["scope"])):
        lines.append(
            f"| {r['behavior']} | {r['scope']} | **{r['verdict']}** | {_fnum(r['held_faith_resample'])} | "
            f"{_fnum(r['disc_faith_resample'])} / {_fnum(r['disc_faith_mean'])} | {_fnum(r['motif_core_held_resample'])} | "
            f"{r.get('knee_n_nodes','—')}/{r.get('floor_n_nodes','—')} | {_fnum(r['mean_minus_resample_gap'])} |"
        )
    if scope_notes:
        lines += ["", "## Scope reconciliation (NOT HEADS-ONLY)", ""]
        for beh, note in sorted(scope_notes.items()):
            lines.append(f"- **{beh}**: {note}")

    lines += ["", "## Per-cell detail", ""]
    for r in sorted(rows, key=lambda x: (x["behavior"], x["scope"])):
        lines += [
            f"### {r['behavior']} ({r['scope']}) — {r['verdict']}",
            "",
            f"- {r.get('verdict_reason','')}",
            f"- base metric {_fnum(r['base_metric'])}; n_discovery {r['n_discovery']}, n_heldout {r['n_heldout']}.",
            f"- knee {r.get('knee_n_nodes')} nodes; floor {r.get('floor_n_nodes')} nodes; knee−floor gap {_fnum(r['knee_minus_floor'])}.",
            f"- faithfulness — discovery: resample {_fnum(r['disc_faith_resample'])}, mean {_fnum(r['disc_faith_mean'])}; "
            f"held-out: resample {_fnum(r['held_faith_resample'])}, mean {_fnum(r['held_faith_mean'])}.",
            f"- motif-core held-out (resample) {_fnum(r['motif_core_held_resample'])}; induction motif present: {r['induction_motif_present']}.",
            f"- suppression heads: {r['n_suppression_heads']}; positive-causal MLPs: {r['n_mlp_positive']}; MLPs in knee: {', '.join(r['mlp_in_knee']) or 'none'}.",
            f"- edge: {r['edge'] if r['edge_claimed'] else 'none claimed'}.",
            "",
        ]

    negatives = [r for r in rows if r["verdict"] in ("OVERFIT / NO CLEAN CIRCUIT", "MECHANISM ABSENT", "INSUFFICIENT PROMPTS", "MIXED_PERIOD")]
    lines += ["## Negative / absent verdicts (each a successful result)", ""]
    for r in sorted(negatives, key=lambda x: (x["behavior"], x["scope"])):
        lines.append(f"- `{r['behavior']}/{r['scope']}`: {r['verdict']} — {r.get('verdict_reason','')}")
    lines += [
        "",
        "## Cross-cutting reading",
        "",
        "- **mean-ablation inflation:** the mean−resample gap column quantifies how much faithfulness was a mean-ablation "
        "artifact (positive = mean inflated). Large gaps with suppression heads present support the brake-removal explanation.",
        "- **prev-token→induction core:** see each induction/successor cell's edge claim and `induction_motif_present`.",
        "- **not-heads-only:** see the scope reconciliation section above.",
        "- **successor:** expected MECHANISM ABSENT for the induction edge — a successful negative if so.",
        "",
        "_Generated incrementally by lab06_matrix.py; updated after every cell._",
        "",
    ]
    text = "\n".join(lines)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        bench.write_text(d / "FINDINGS.md", text)
        bench.write_json(d / "matrix_results.json", {"model": model, "date": date, "cells": rows})


def main(argv=None) -> None:
    m = parse_matrix_args(argv)
    date = m.date or bench.now_slug()
    matrix_name = f"lab06_matrix_{date}"
    model_slug = bench.sanitize_tag(m.model.replace("/", "__"))

    local_matrix = pathlib.Path(m.repo_root) / "runs" / matrix_name / model_slug
    drive_matrix = pathlib.Path(m.drive_root) / matrix_name / model_slug
    repo_findings = pathlib.Path(m.repo_root) / "lab06_matrix" / date / model_slug  # git-tracked text only, per model

    base = _base_args(m)

    # env + model load (once) -------------------------------------------------
    load_dir = local_matrix / "_load"
    load_dir.mkdir(parents=True, exist_ok=True)
    bench.configure_env(load_dir)
    bench._ensure_plot_style()
    bench.ensure_ledger()
    import torch

    base_ctx = bench.RunContext(run_dir=load_dir, args=base)
    bench.set_determinism(torch, base.seed)
    print(f"[matrix] loading {m.model} (dtype={base.dtype}) ...")
    t_load = time.perf_counter()
    bundle = bench.load_model_and_tokenizer(base_ctx)
    base_ctx.bind_model(bundle)
    print(f"[matrix] model loaded in {time.perf_counter()-t_load:.1f}s; {bundle.anatomy.n_layers} layers")

    cells = [(b, "heads_and_mlps") for b in m.behaviors.split(",") if b.strip()]
    cells += [(b, "heads_only") for b in m.heads_only_behaviors.split(",") if b.strip()]

    # resume: load prior matrix_results.json if present
    rows: list[dict] = []
    prior = drive_matrix / "matrix_results.json"
    if prior.exists():
        try:
            rows = json.load(open(prior)).get("cells", [])
        except Exception:
            rows = []

    def have_cell(b, s):
        return any(r["behavior"] == b and r["scope"] == s for r in rows)

    for behavior, scope in cells:
        cell_dir = local_matrix / f"{behavior}_{scope}"
        if m.skip_existing and (cell_dir / "metrics.json").exists():
            print(f"[matrix] skip existing {behavior}/{scope}")
            continue
        cell_dir.mkdir(parents=True, exist_ok=True)
        args_cell = copy.deepcopy(base)
        args_cell.behavior = behavior
        args_cell.scope = scope
        args_cell.run_dir = str(cell_dir)
        ctx = bench.RunContext(run_dir=cell_dir, args=args_cell)
        ctx.bind_model(bundle)
        bench.write_json(ctx.path("run_config.json"), vars(args_cell))
        ctx.register_artifact(ctx.path("run_config.json"), "config", "Resolved args for this matrix cell.")
        try:
            bench.write_json(ctx.path("run_metadata.json"), bench.collect_run_metadata(torch))
        except Exception:
            pass

        print(f"\n[matrix] ===== {behavior} / {scope} =====")
        t0 = time.perf_counter()
        try:
            lab06.run(ctx, bundle)  # catches CellAbort internally -> verdict card
        except Exception as exc:
            traceback.print_exc()
            bench.write_json(
                ctx.path("metrics.json"),
                {"behavior": behavior, "scope": scope, "verdict": "ERROR", "verdict_reason": f"{type(exc).__name__}: {exc}", "aborted": True},
            )
        finally:
            bench.write_json(ctx.path("artifact_index.json"), {"artifacts": ctx.artifacts})
        elapsed = time.perf_counter() - t0

        try:
            metrics = json.load(open(ctx.path("metrics.json")))
        except Exception:
            metrics = {"behavior": behavior, "scope": scope, "verdict": "ERROR", "verdict_reason": "no metrics.json"}
        summary = _cell_summary(metrics)
        summary["elapsed_s"] = round(elapsed, 1)
        rows = [r for r in rows if not (r["behavior"] == behavior and r["scope"] == scope)] + [summary]
        print(f"[matrix] {behavior}/{scope}: {summary['verdict']} in {elapsed:.1f}s")

        copy_status = _copy_to_drive(cell_dir, drive_matrix / f"{behavior}_{scope}")
        print(f"[matrix] drive copy: {copy_status}")
        write_findings([drive_matrix, repo_findings], m.model, date, rows)

    print(f"\n[matrix] DONE. {len(rows)} cells. FINDINGS at {drive_matrix/'FINDINGS.md'} and {repo_findings/'FINDINGS.md'}")


if __name__ == "__main__":
    main()
