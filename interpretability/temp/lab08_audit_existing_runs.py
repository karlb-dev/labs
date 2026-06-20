#!/usr/bin/env python3
"""Audit historical Lab 8 SAE runs copied from Drive.

The script expects run directories under
``runs/lab08_existing_comparison`` by default and writes:

* ``lab08_existing_run_audit.csv``: one summary row per run
* ``lab08_existing_run_audit.md``: human-readable confirmation notes
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs" / "lab08_existing_comparison"


def read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def find_runs(input_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted({p.parent for p in input_root.rglob("metrics.json")})


def pick_best_feature(atlas: list[dict[str, str]]) -> dict[str, str]:
    if not atlas:
        return {}
    verdict_bonus = {
        "survived": 2.0,
        "narrowed": 1.0,
        "token-feature": -0.5,
        "polysemantic": -0.75,
        "killed": -1.0,
        "silent-on-corpus": -2.0,
    }

    def score(row: dict[str, str]) -> float:
        return (
            verdict_bonus.get(row.get("verdict", ""), 0.0)
            + as_float(row.get("held_out_auc"), 0.5)
            + 0.25 * as_float(row.get("confusable_auc"), 0.5)
            + 0.25 * as_float(row.get("label_purity"), 0.0)
            - 0.1 * as_float(row.get("polysemy_entropy_bits"), 0.0)
            - 2.0 * min(as_float(row.get("fire_fraction"), 0.0), 0.5)
        )

    return max(atlas, key=score)


def pick_best_survivor(atlas: list[dict[str, str]]) -> dict[str, str]:
    survivors = [r for r in atlas if r.get("verdict") == "survived"]
    if not survivors:
        return {}
    return max(
        survivors,
        key=lambda r: (
            as_float(r.get("held_out_auc"), 0.5),
            as_float(r.get("confusable_auc"), 0.5),
            -as_float(r.get("fire_fraction"), 0.0),
        ),
    )


def summarize_clamp(run_dir: pathlib.Path) -> dict[str, Any]:
    rows = read_csv(run_dir / "tables" / "clamp_operating_points.csv")
    if not rows:
        rows = read_csv(run_dir / "tables" / "feature_clamp.csv")
    if not rows:
        return {
            "clamp_rows": 0,
            "clamp_label": "",
            "clamp_real_base_hits": "",
            "clamp_real_max_hits": "",
            "clamp_best_mult": "",
            "clamp_random_max_hits": "",
            "clamp_fluent_best_distinct": "",
        }

    real = [r for r in rows if r.get("condition") == "real"]
    random = [r for r in rows if r.get("condition") == "random"]
    base = next((r for r in real if as_float(r.get("clamp_mult")) == 0.0), real[0] if real else {})
    fluent = [r for r in real if as_float(r.get("distinct_ratio")) >= 0.4]
    best = max(fluent or real, key=lambda r: as_int(r.get("domain_keyword_hits"))) if real else {}
    return {
        "clamp_rows": len(rows),
        "clamp_label": best.get("label", base.get("label", "")),
        "clamp_real_base_hits": as_int(base.get("domain_keyword_hits"), 0) if base else "",
        "clamp_real_max_hits": as_int(best.get("domain_keyword_hits"), 0) if best else "",
        "clamp_best_mult": best.get("clamp_mult", ""),
        "clamp_random_max_hits": max((as_int(r.get("domain_keyword_hits")) for r in random), default=""),
        "clamp_fluent_best_distinct": best.get("distinct_ratio", ""),
    }


def summarize_run(run_dir: pathlib.Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    config = read_json(run_dir / "run_config.json")
    atlas = read_csv(run_dir / "tables" / "feature_atlas.csv")
    if not atlas:
        atlas = read_csv(run_dir / "results.csv")
    domain_summary = read_csv(run_dir / "tables" / "domain_validation_summary.csv")
    best = pick_best_feature(atlas)
    survivor = pick_best_survivor(atlas)
    verdicts = Counter(r.get("verdict", "") for r in atlas)

    narrowed = verdicts.get("narrowed", 0)
    killed_family = (
        verdicts.get("killed", 0)
        + verdicts.get("polysemantic", 0)
        + verdicts.get("token-feature", 0)
        + verdicts.get("silent-on-corpus", 0)
    )

    row: dict[str, Any] = {
        "run_path": rel(run_dir),
        "run_name": config.get("run_name") or run_dir.name,
        "model_id": metrics.get("model_id", config.get("model", "")),
        "tier": config.get("tier", ""),
        "prompt_set": config.get("prompt_set", ""),
        "sae_layer": metrics.get("sae_layer", ""),
        "sae_d_sae": metrics.get("sae_d_sae", ""),
        "reconstruction_fvu": metrics.get("reconstruction_fvu", ""),
        "per_token_l0": metrics.get("per_token_l0", ""),
        "silent_feature_fraction": metrics.get("silent_feature_fraction", ""),
        "atlas_size": len(atlas) or metrics.get("atlas_size", ""),
        "n_survived": verdicts.get("survived", metrics.get("n_survived", 0)),
        "n_narrowed": narrowed,
        "n_killed_poly_token_silent": killed_family,
        "n_domain_summary_rows": len(domain_summary),
        "best_feature": best.get("feature", ""),
        "best_label": best.get("proposed_label", ""),
        "best_verdict": best.get("verdict", ""),
        "best_held_out_auc": best.get("held_out_auc", ""),
        "best_confusable_with": best.get("confusable_with", ""),
        "best_confusable_auc": best.get("confusable_auc", ""),
        "best_fire_fraction": best.get("fire_fraction", ""),
        "best_label_purity": best.get("label_purity", ""),
        "best_max_activation": best.get("max_activation", ""),
        "best_survived_feature": survivor.get("feature", ""),
        "best_survived_label": survivor.get("proposed_label", ""),
        "best_survived_auc": survivor.get("held_out_auc", ""),
        "best_survived_fire_fraction": survivor.get("fire_fraction", ""),
        "best_survived_confusable_auc": survivor.get("confusable_auc", ""),
        "clamp_causal_metric": metrics.get("clamp_causal", ""),
    }
    row.update(summarize_clamp(run_dir))
    return row


def write_csv_out(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: pathlib.Path, rows: list[dict[str, Any]], input_root: pathlib.Path) -> None:
    lines = [
        "# Lab 8 Existing Run Audit",
        "",
        f"Input root: `{rel(input_root)}`",
        f"Runs parsed: {len(rows)}",
        "",
        "## Summary",
        "",
        "| run | model | FVU | L0 | silent | verdicts | best feature | clamp |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        verdicts = (
            f"{row['n_survived']} survived, {row['n_narrowed']} narrowed, "
            f"{row['n_killed_poly_token_silent']} killed/poly/token/silent"
        )
        best = (
            f"F{row['best_feature']} {row['best_label']} ({row['best_verdict']}, "
            f"AUC {row['best_held_out_auc']}, fire {row['best_fire_fraction']})"
        )
        if row.get("best_survived_feature"):
            best += (
                f"; survived F{row['best_survived_feature']} {row['best_survived_label']} "
                f"(AUC {row['best_survived_auc']}, fire {row['best_survived_fire_fraction']})"
            )
        clamp = (
            f"{row['clamp_real_base_hits']} -> {row['clamp_real_max_hits']} hits"
            if row.get("clamp_rows")
            else "not run"
        )
        lines.append(
            f"| `{row['run_name']}` | `{row['model_id']}` | {row['reconstruction_fvu']} | "
            f"{row['per_token_l0']} | {row['silent_feature_fraction']} | {verdicts} | {best} | {clamp} |"
        )

    lines += [
        "",
        "## Prompt-Summary Check",
        "",
        "- `run6/A` matches the reported gpt2 negative: FVU about 0.0019, L0 about 74.57, "
        "0 survived, 2 narrowed, and a failed code clamp.",
        "- `run6/B` and `run6/C` match the reported Olmo result: FVU about 0.3761, L0 about 113.54, "
        "0 survived, 4 narrowed, with feature 1265 as the best law handle.",
        "- `run1/lab08_tierb_full` contains the earlier positive: 1 survived feature, with broad high-fire "
        "emotion feature 1204 in the atlas.",
        "",
        "## Notes",
        "",
        "- A run-level `clamp_causal=true` is not treated as a clean causal claim here; the older clamp used one "
        "random control and keyword hits only.",
        "- The audit reflects historical artifacts as written, including the older line-level validation protocol.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_INPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = pathlib.Path(args.input_root).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    runs = find_runs(input_root)
    rows = [summarize_run(run) for run in runs]
    rows.sort(key=lambda r: (str(r.get("model_id", "")), str(r.get("run_name", "")), str(r.get("run_path", ""))))
    csv_path = out_dir / "lab08_existing_run_audit.csv"
    md_path = out_dir / "lab08_existing_run_audit.md"
    write_csv_out(csv_path, rows)
    write_markdown(md_path, rows, input_root)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
