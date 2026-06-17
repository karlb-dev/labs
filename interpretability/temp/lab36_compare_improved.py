#!/usr/bin/env python
"""Compare improved-instrument Lab 36 runs (B4 model-canonical + B5 sentinel)
against the earlier v3 runs.  Prints markdown tables for the validation report.
"""
import json
import os
import sys

RUNS_DIR = "runs"


def load(run):
    p = os.path.join(RUNS_DIR, run, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def f(m, k):
    v = m.get(k, "") if m else ""
    if v in ("", None):
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


# (label, improved_run, prior_v3_run_or_None)
ROWS = [
    ("A SmolLM2-135M", "lab36_smollm", "lab36_tierA_smollm_full"),
    ("B Olmo-3-7B-Instruct", "lab36_olmo7b_instruct", "lab36_tierB_olmo7b_full"),
    ("B Olmo-3-7B-Think", "lab36_olmo7b_think", None),
    ("Gemma-4-E4B", "lab36_gemma4_e4b", "lab36_gemma4_e4b_full"),
    ("C Olmo-3.1-32B-Instruct", "lab36_olmo31_32b_instruct", "lab36_tierC_olmo31_32b_full"),
    ("C Olmo-3.1-32B-Think", "lab36_olmo31_32b_think", None),
]

print("## Improved-instrument headline (B4 model-canonical answer, B5 sentinel control)\n")
print("| Model | Verdict | B4 acc | B4 plaus (default route) | B5 d′ report-query | B5 d′ sentinel | B5 FA | B5 leak |")
print("|---|---|--:|--:|--:|--:|--:|--:|")
for label, run, _prior in ROWS:
    m = load(run)
    if not m:
        print(f"| {label} | (missing) | | | | | | |")
        continue
    print(f"| {label} | `{m.get('verdict','?')}` | {f(m,'b4_activation_source_accuracy')} | "
          f"{f(m,'b4_activation_canonical_plausibility_rate')} | {f(m,'b5_d_prime_all_insertions')} | "
          f"{f(m,'b5_sentinel_d_prime')} | {f(m,'b5_false_alarm_rate')} | {f(m,'b5_content_leak_rate')} |")

print("\n## B4 plausibility: before (v3, CSV answer) vs after (model greedy answer)\n")
print("| Model | v3 B4 plaus | improved B4 plaus | v3 B4 acc | improved B4 acc |")
print("|---|--:|--:|--:|--:|")
for label, run, prior in ROWS:
    if not prior:
        continue
    m, p = load(run), load(prior)
    print(f"| {label} | {f(p,'b4_activation_canonical_plausibility_rate')} | {f(m,'b4_activation_canonical_plausibility_rate')} | "
          f"{f(p,'b4_activation_source_accuracy')} | {f(m,'b4_activation_source_accuracy')} |")

print("\n## B5 sentinel control: report-query d′ vs sentinel d′ (does the signal survive upstream injection?)\n")
print("| Model | report-query d′ | sentinel d′ | Δ | report-query FA | sentinel FA |")
print("|---|--:|--:|--:|--:|--:|")
for label, run, _prior in ROWS:
    m = load(run)
    if not m:
        continue
    try:
        delta = float(m.get("b5_d_prime_all_insertions")) - float(m.get("b5_sentinel_d_prime"))
        delta = f"{delta:.3f}"
    except (TypeError, ValueError):
        delta = "—"
    print(f"| {label} | {f(m,'b5_d_prime_all_insertions')} | {f(m,'b5_sentinel_d_prime')} | {delta} | "
          f"{f(m,'b5_false_alarm_rate')} | {f(m,'b5_sentinel_false_alarm_rate')} |")

print("\n## Think vs Instruct (reasoning axis)\n")
print("| Metric | 7B-Instruct | 7B-Think | 32B-Instruct | 32B-Think |")
print("|---|--:|--:|--:|--:|")
pairs = {"7BI": load("lab36_olmo7b_instruct"), "7BT": load("lab36_olmo7b_think"),
         "32BI": load("lab36_olmo31_32b_instruct"), "32BT": load("lab36_olmo31_32b_think")}
for metric, key in [("B4 acc", "b4_activation_source_accuracy"),
                    ("B5 d′ report-query", "b5_d_prime_all_insertions"),
                    ("B5 d′ sentinel", "b5_sentinel_d_prime"),
                    ("B5 false alarm", "b5_false_alarm_rate"),
                    ("B5 content leak", "b5_content_leak_rate"),
                    ("verdict", "verdict")]:
    print(f"| {metric} | {f(pairs['7BI'],key)} | {f(pairs['7BT'],key)} | {f(pairs['32BI'],key)} | {f(pairs['32BT'],key)} |")
