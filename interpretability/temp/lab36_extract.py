#!/usr/bin/env python
"""Extract Lab 36 headline metrics from one or more run dirs for comparison.

Usage: python temp/lab36_extract.py <run_dir> [<run_dir> ...]
Prints a markdown row per run plus the B5 headline-row detail when present.
"""
import csv
import json
import os
import sys

HEADLINE = "headline_content_blind_logit_only"


def load_metrics(run_dir):
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def b5_headline_row(run_dir):
    p = os.path.join(run_dir, "tables", "injection_detection_summary.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    # Prefer the v3 content-blind headline row; fall back to all_insertions_vs_clean.
    for key in (HEADLINE, "all_insertions_vs_clean"):
        for r in rows:
            if r.get("comparison") == key or r.get("b5_summary_row_id", "").startswith(key):
                return key, r
    # Some versions tag the headline via a dedicated column.
    for r in rows:
        if r.get("is_headline") in ("True", "true", "1"):
            return r.get("comparison", "?"), r
    return ("first", rows[0]) if rows else None


def g(m, k, default="n/a"):
    v = m.get(k, default)
    if isinstance(v, float):
        return f"{v:.4f}"
    return v


def main(dirs):
    print("| run | model | verdict | n_items | n_dir | B4 act acc | B4 fresh | B5 d' | B5 FA | B5 leak | B5 pass | B2 tgt-floor | maxpatch | warn | fails |")
    print("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for d in dirs:
        m = load_metrics(d)
        name = os.path.basename(d.rstrip("/"))
        if m is None:
            print(f"| {name} | MISSING metrics.json | | | | | | | | | | | | | |")
            continue
        hr = b5_headline_row(d)
        b5_pass = ""
        if hr:
            _, row = hr
            b5_pass = row.get("pass_gate", "")
        print(
            f"| {name} | `{m.get('model_id','?')}` | `{m.get('verdict','?')}` | "
            f"{g(m,'n_items')} | {g(m,'n_directions')} | {g(m,'b4_activation_source_accuracy')} | "
            f"{g(m,'b4_activation_fresh_accuracy')} | {g(m,'b5_d_prime_all_insertions')} | "
            f"{g(m,'b5_false_alarm_rate')} | {g(m,'b5_content_leak_rate')} | {b5_pass} | "
            f"{g(m,'mean_b2_target_minus_floor')} | {g(m,'max_patch_recovery')} | "
            f"{g(m,'warning_count')} | {g(m,'failure_specimens')} |"
        )
    # Detail: B5 headline row per run
    print("\n### B5 headline rows (injection_detection_summary.csv)\n")
    for d in dirs:
        hr = b5_headline_row(d)
        name = os.path.basename(d.rstrip("/"))
        if not hr:
            print(f"- {name}: no injection_detection_summary.csv")
            continue
        key, row = hr
        keep = {k: row.get(k) for k in (
            "comparison", "d_prime", "false_alarm_rate", "content_leak_rate",
            "behavior_task_success_rate", "hit_rate", "pass_gate",
            "decision_source", "dose") if k in row}
        print(f"- {name} [{key}]: {keep}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
