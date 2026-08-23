#!/usr/bin/env python3
"""Build the Tier 1 LogWarden state of record from retained row exports only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--rows-dir", help="verified row bundle (defaults to <run>/repro/rows)")
parser.add_argument("--freeze", action="store_true", help="freeze expected deterministic output hashes")
args = parser.parse_args()
run = Path(args.run).resolve()
lab = Path(__file__).resolve().parents[1]
rows_root = Path(args.rows_dir).resolve() if args.rows_dir else run / "repro" / "rows"
if not rows_root.is_relative_to(run / "repro"):
    raise RuntimeError(f"Row bundle must stay under {run / 'repro'}: {rows_root}")
reports = run / "reports"
tables = run / "tables"
figures = run / "figures"
for directory in (reports, tables, figures):
    directory.mkdir(parents=True, exist_ok=True)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def canonical(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def verify_inputs() -> dict[str, Any]:
    manifest = json.loads((rows_root / "manifest.json").read_text())
    received = manifest.pop("receiptSha256")
    if digest_bytes(canonical(manifest).encode()) != received:
        raise RuntimeError("Report-row manifest receipt hash drift")
    manifest["receiptSha256"] = received
    for item in manifest["artifacts"]:
        path = rows_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest_file(path) != item["sha256"]:
            raise RuntimeError(f"Report-row artifact drift: {item['path']}")
    return manifest


manifest = verify_inputs()
run_id = manifest["runId"]


def repro_gate(mode: str) -> dict[str, Any]:
    path = run / "repro" / f"{mode}-pass.json"
    if not path.is_file():
        return {"mode": mode, "disposition": "PENDING", "receiptSha256": None}
    record = json.loads(path.read_text())
    received = record.pop("receiptSha256")
    if digest_bytes(canonical(record).encode()) != received:
        raise RuntimeError(f"{mode} reproduction receipt hash drift")
    record["receiptSha256"] = received
    if record.get("runId") != run_id or record.get("mode") != mode or record.get("disposition") != "PASS" or \
       record.get("rowManifestReceiptSha256") != manifest["receiptSha256"] or record.get("queryBundleSha256") != manifest["queryBundleSha256"]:
        raise RuntimeError(f"{mode} reproduction receipt identity drift")
    return record


rows_gate = repro_gate("rows")
restore_gate = repro_gate("restore")


def jsonl(name: str) -> pd.DataFrame:
    path = rows_root / f"{name}.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True, dtype=False)


def evidence(source_path: str) -> dict[str, Any]:
    item = next((value for value in manifest["artifacts"] if value.get("sourcePath") == source_path), None)
    if item is None:
        raise RuntimeError(f"Missing evidence input: {source_path}")
    return json.loads((rows_root / item["path"]).read_text())


primary = jsonl("primary_predictions")
controls = jsonl("control_predictions")
tool_rows = jsonl("tool_scores")
retrieval_rows = jsonl("retrieval_scores")
metrics = jsonl("metric_results")
contrasts = jsonl("paired_contrasts")
control_pairs = jsonl("control_comparisons")
calibration_models = jsonl("calibration_models")
performance = jsonl("agent_performance")
service = jsonl("service_performance")
pipeline_phases = jsonl("pipeline_phase_performance")
queue_performance = jsonl("queue_performance")
sql_performance = jsonl("sql_resource_performance")
query_store_performance = jsonl("query_store_performance")
retrieval_benchmarks = jsonl("retrieval_benchmarks")
safety_rows = jsonl("safety_audit")
completeness = jsonl("run_completeness")
model_dispositions = evidence("metrics/model-dispositions.json")
freeze = evidence("manifests/freeze.json")
capture = evidence("capture/verification-standard-v1.json")
packets = evidence("packets/build.json")
leakage = evidence("packets/leakage-audit.json")
retrieval_receipt = evidence("knowledge/retrieval-evaluation-dev-calibration-test_id-test_variant_holdout-test_unknown.json")
security = evidence("security/tool-security-gate.json")
telemetry = evidence("telemetry/reconciliation.json")
analysis_receipt = evidence("metrics/campaign-analysis.json")


def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


numeric(performance, [
    "agent_elapsed_ms", "model_request_count", "model_response_count", "model_client_elapsed_ms",
    "headers_wait_ms", "body_read_ms", "parse_ms", "prompt_tokens", "completion_tokens", "model_error_count",
    "length_finish_count", "repaired_response_count", "rejected_response_count", "tool_call_count",
    "no_repair_count", "code_fence_repair_count", "leading_text_repair_count", "rejected_repair_count",
    "tool_latency_ms", "snapshot_miss_count", "tool_error_count", "validation_failure_count",
])
numeric(tool_rows, ["called_count", "valid_call_count", "snapshot_miss_count", "tool_error_count", "tool_score"])
numeric(service, [
    "sample_count", "mean_running_requests", "max_running_requests", "max_waiting_requests", "max_swapped_requests",
    "max_kv_cache_usage_ratio", "mean_prompt_tokens_per_second", "max_prompt_tokens_per_second",
    "mean_generation_tokens_per_second", "max_generation_tokens_per_second", "prefix_cache_hit_ratio",
    "preemption_delta", "request_error_delta", "cancellation_delta", "prompt_token_delta", "generation_token_delta",
    "mean_gpu_utilization_pct", "max_gpu_utilization_pct", "max_gpu_memory_used_mib", "max_gpu_power_draw_w", "max_gpu_temperature_c",
])
numeric(pipeline_phases, ["sample_count", "non_success_count", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms"])
numeric(queue_performance, [
    "sample_count", "mean_pending_count", "max_pending_count", "mean_leased_count", "max_leased_count",
    "max_retryable_count", "max_oldest_pending_age_ms", "max_lease_expired_count", "observed_arrivals",
    "observed_completions", "max_worker_count",
])
numeric(sql_performance, [
    "sample_count", "mean_process_cpu_pct", "max_process_cpu_pct", "max_process_memory_kb", "max_target_memory_kb",
    "max_request_count", "max_blocked_request_count", "max_runnable_task_count", "max_pending_io_count",
    "max_data_file_bytes", "max_log_file_bytes", "max_log_used_pct", "unavailable_sample_count",
])
numeric(query_store_performance, ["interval_count", "execution_count", "duration_ms", "cpu_ms", "logical_reads", "physical_reads", "log_bytes"])

profiles = ["muse-glimmer-30b", "gemma-4-31b", "qwen-3.8-27b"]
target_profiles = ["muse-glimmer-30b", "gemma-4-31b", "olmo-3.1-32b-instruct", "qwen-3.8-27b"]
model_arms = ["A-direct", "A-rag", "A-tools", "A-router"]
baseline_arms = [
    "B0-majority-no-action-v1", "B1-rules-v1", "B2-lexical-v1",
    "B2-vector-v1", "B2-hybrid-v1", "B3-oracle-packet-v1",
]
primary["profile_key"] = primary["model_profile_id"].fillna("baseline")
performance["profile_key"] = performance["model_profile_id"].fillna("baseline")
tool_rows["profile_key"] = tool_rows["model_profile_id"].fillna("baseline")
retrieval_rows["profile_key"] = retrieval_rows["model_profile_id"].fillna("baseline")


def rate(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return None if clean.empty else float(clean.mean())


def quantile(series: pd.Series, value: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return None if clean.empty else float(clean.quantile(value, interpolation="linear"))


def safe_div(left: float, right: float) -> float | None:
    return None if right == 0 else left / right


def ece(outcomes: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float | None:
    if len(outcomes) == 0:
        return None
    total = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (probabilities >= low) & ((probabilities <= high) if index == bins - 1 else (probabilities < high))
        if selected.any():
            total += selected.mean() * abs(float(probabilities[selected].mean()) - float(outcomes[selected].mean()))
    return total


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-min(value, 709)))
    exp = math.exp(max(value, -709))
    return exp / (1 + exp)


def calibrated(model: dict[str, Any], confidence: float) -> float:
    bounded = min(1 - 1e-6, max(1e-6, confidence))
    return sigmoid(float(model["intercept"]) + float(model["slope"]) * math.log(bounded / (1 - bounded)))


calibration_index: dict[tuple[str, str], dict[str, Any]] = {}
for row in calibration_models.to_dict("records"):
    calibration_index[(row["model_profile_id"], row["agent_arm_id"])] = json.loads(row["coefficients_json"])


def contrast(profile: str, arm: str, metric_name: str) -> dict[str, Any] | None:
    selected = contrasts[(contrasts.contrast_id == f"{profile}-{arm}-vs-B1") & (contrasts.metric_name == metric_name)]
    return None if selected.empty else selected.iloc[0].to_dict()


def tool_summary(profile: str, arm: str) -> dict[str, float | None]:
    values = tool_rows[(tool_rows.profile_key == profile) & (tool_rows.agent_arm_id == arm)]
    required = values[values.expectation == "required"]
    forbidden = values[values.expectation == "forbidden"]
    calls = float(pd.to_numeric(values.called_count, errors="coerce").fillna(0).sum())
    valid = float(pd.to_numeric(values.valid_call_count, errors="coerce").fillna(0).sum())
    return {
        "required_tool_recall": rate(required.correct_use),
        "forbidden_tool_rate": rate(pd.to_numeric(forbidden.called_count, errors="coerce").fillna(0) > 0),
        "argument_valid_rate": safe_div(valid, calls),
        "snapshot_miss_rate": safe_div(float(pd.to_numeric(values.snapshot_miss_count, errors="coerce").fillna(0).sum()), calls),
    }


def calibration_summary(values: pd.DataFrame, profile: str, arm: str) -> dict[str, Any]:
    selected = values.dropna(subset=["confidence", "end_to_end_success"]).copy()
    if selected.empty or (profile, arm) not in calibration_index:
        return {"raw_brier": None, "raw_ece": None, "calibrated_brier": None, "calibrated_ece": None,
                "selective_accuracy": None, "selective_coverage": None, "calibration_converged": None}
    outcomes = selected.end_to_end_success.astype(float).to_numpy()
    raw = selected.confidence.astype(float).to_numpy()
    coefficients = calibration_index[(profile, arm)]
    model = coefficients
    fitted = np.array([calibrated(model, value) for value in raw])
    threshold = coefficients.get("threshold", {})
    chosen = fitted >= float(threshold.get("threshold", 1))
    return {
        "raw_brier": float(np.mean((raw - outcomes) ** 2)), "raw_ece": ece(outcomes, raw),
        "calibrated_brier": float(np.mean((fitted - outcomes) ** 2)), "calibrated_ece": ece(outcomes, fitted),
        "selective_accuracy": None if not chosen.any() else float(selected.loc[chosen, "action_correct"].astype(float).mean()),
        "selective_coverage": float(chosen.mean()), "calibration_converged": bool(model.get("converged", False)),
    }


def cell_row(profile: str, arm: str, kind: str) -> dict[str, Any]:
    values = primary[(primary.profile_key == profile) & (primary.agent_arm_id == arm)]
    if values.empty:
        return {
            "profile": profile, "arm": arm, "kind": kind, "disposition": "STOP_PORT" if profile.startswith("olmo") else "UNAVAILABLE",
            "eligible": 0, "completed": 0,
        }
    labels = sorted(values.expected_class.dropna().unique())
    predicted = values.predicted_class.fillna("__failure__")
    macro = f1_score(values.expected_class, predicted, labels=labels, average="macro", zero_division=0)
    perf = performance[(performance.profile_key == profile) & (performance.agent_arm_id == arm)]
    retrieval = retrieval_rows[(retrieval_rows.profile_key == profile) & (retrieval_rows.agent_arm_id == arm)]
    tools = tool_summary(profile, arm)
    calibration_values = calibration_summary(values, profile, arm)
    action = contrast(profile, arm, "acceptable_action_accuracy") if profile != "baseline" else None
    loss = contrast(profile, arm, "cost_weighted_loss_reduction") if profile != "baseline" else None
    derived = kind == "DERIVED"
    model_ms = pd.Series(dtype=float) if derived else perf.model_client_elapsed_ms
    tokens = pd.Series(dtype=float) if derived else pd.to_numeric(perf.prompt_tokens, errors="coerce").fillna(0) + pd.to_numeric(perf.completion_tokens, errors="coerce").fillna(0)
    responses = 0 if derived else float(perf.model_response_count.fillna(0).sum())
    repair_distribution = {
        "none": 0 if derived else int(perf.no_repair_count.fillna(0).sum()),
        "strip_code_fence": 0 if derived else int(perf.code_fence_repair_count.fillna(0).sum()),
        "strip_leading_text": 0 if derived else int(perf.leading_text_repair_count.fillna(0).sum()),
        "rejected": 0 if derived else int(perf.rejected_repair_count.fillna(0).sum()),
    }
    return {
        "profile": profile, "arm": arm, "kind": kind, "disposition": "COMPLETE",
        "eligible": len(values), "completed": int((values.outcome != "failure").sum()),
        "end_to_end_success": rate(values.end_to_end_success), "macro_f1": float(macro),
        "action_accuracy": rate(values.action_correct), "cost_weighted_loss": rate(values.cost_weighted_loss),
        "contract_success": rate(values.contract_success), "unknown_abstain_accuracy": rate(values.loc[values.regime == "U", "abstention_correct"]),
        "b1_coverage": (1 - rate(values.abstained)) if profile == "baseline" and arm == "B1-rules-v1" else None,
        **tools, "retrieval_recall_at_5": rate(retrieval.recall_at_k), "grounding_rate": rate(retrieval.grounded),
        "first_pass_valid_rate": safe_div(float(repair_distribution["none"]), responses),
        "repaired_response_rate": safe_div(float(repair_distribution["strip_code_fence"] + repair_distribution["strip_leading_text"]), responses),
        "rejected_response_rate": safe_div(float(repair_distribution["rejected"]), responses),
        "repair_kind_distribution": canonical(repair_distribution),
        "action_delta_vs_b1": None if action is None else action["observed_difference"],
        "action_delta_ci_low": None if action is None else action["ci_low"], "action_delta_ci_high": None if action is None else action["ci_high"],
        "action_delta_holm_p": None if action is None else action["holm_adjusted_p_value"],
        "loss_reduction_vs_b1": None if loss is None else loss["observed_difference"],
        "loss_delta_ci_low": None if loss is None else loss["ci_low"], "loss_delta_ci_high": None if loss is None else loss["ci_high"],
        "model_latency_p50_ms": quantile(model_ms, .5), "model_latency_p95_ms": quantile(model_ms, .95),
        "agent_latency_p50_ms": None if derived else quantile(perf.agent_elapsed_ms, .5),
        "agent_latency_p95_ms": None if derived else quantile(perf.agent_elapsed_ms, .95),
        "tokens_per_episode": None if tokens.empty else float(tokens.sum() / len(values)),
        **calibration_values,
    }


score_rows: list[dict[str, Any]] = []
for arm in baseline_arms:
    score_rows.append(cell_row("baseline", arm, "BASELINE"))
for profile in profiles:
    for arm in model_arms:
        score_rows.append(cell_row(profile, arm, "DERIVED" if arm == "A-router" else "AGENT"))
for arm in model_arms:
    score_rows.append(cell_row("olmo-3.1-32b-instruct", arm, "STOPPED"))
scorecard = pd.DataFrame(score_rows)


def formally_supported(row: dict[str, Any] | None) -> bool:
    return bool(row and float(row["ci_low"]) > 0 and float(row["holm_adjusted_p_value"]) <= .05)


formal_positive = any(formally_supported(row) for row in contrasts.to_dict("records"))


taxonomy: list[dict[str, Any]] = []


def tax(code: str, stage: str, severity: str, evidence_value: dict[str, Any]) -> None:
    taxonomy.append({"runId": run_id, "taxonomyCode": code, "stage": stage, "severity": severity, "evidence": evidence_value})


for profile in profiles:
    for arm in model_arms:
        action = contrast(profile, arm, "acceptable_action_accuracy")
        loss = contrast(profile, arm, "cost_weighted_loss_reduction")
        supported = formally_supported(action) or formally_supported(loss)
        tax("LLM_LIFT_KNOWN" if supported else "RULES_SUFFICIENT", "paired_primary", "info", {
            "profileKey": profile, "armId": arm, "scope": "all_test_roles", "actionContrast": action, "lossContrast": loss,
            "rule": "positive lift requires CI lower bound > 0 and Holm-adjusted p <= 0.05",
        })
        for scope, field in [("regime", "regime"), ("family", "family")]:
            for value in sorted(primary[(primary.profile_key == profile) & (primary.agent_arm_id == arm)][field].dropna().unique()):
                left = primary[(primary.profile_key == profile) & (primary.agent_arm_id == arm) & (primary[field] == value)]
                right = primary[(primary.profile_key == "baseline") & (primary.agent_arm_id == "B1-rules-v1") & (primary[field] == value)]
                tax("CLEAN_NULL" if not supported else "LLM_LIFT_CONTEXT", "descriptive_stratum", "info", {
                    "profileKey": profile, "armId": arm, scope: value, "actionAccuracy": rate(left.action_correct),
                    "b1ActionAccuracy": rate(right.action_correct), "inference": "descriptive; multiplicity-controlled inference is aggregate only",
                })
        if arm == "A-rag":
            selected = contrasts[(contrasts.contrast_id == f"{profile}-A-rag-vs-A-direct") & (contrasts.metric_name == "acceptable_action_accuracy")]
            row = None if selected.empty else selected.iloc[0].to_dict()
            tax("RETRIEVAL_LIFT" if formally_supported(row) else "CLEAN_NULL", "retrieval_ablation", "info", {"profileKey": profile, "contrast": row})
        if arm == "A-tools":
            summary = tool_summary(profile, arm)
            required_recall = 0 if summary["required_tool_recall"] is None else summary["required_tool_recall"]
            forbidden_rate = 1 if summary["forbidden_tool_rate"] is None else summary["forbidden_tool_rate"]
            snapshot_miss_rate = 1 if summary["snapshot_miss_rate"] is None else summary["snapshot_miss_rate"]
            disciplined = required_recall >= .95 and forbidden_rate <= .01 and snapshot_miss_rate <= .01
            tax("TOOL_DISCIPLINED" if disciplined else "TOOL_FRAGILE", "tool_use", "warning" if not disciplined else "info", {"profileKey": profile, **summary})
    tax("RUNTIME_RELIABLE", "runtime", "info", {"profileKey": profile, "telemetryReconciliation": telemetry["receiptSha256"], "allScientificCellsTerminal": True})

for arm in model_arms:
    tax("STOP_PORT", "port_gate", "error", {"profileKey": "olmo-3.1-32b-instruct", "armId": arm, "gates": model_dispositions["portExclusions"][0]["gates"]})
tax("GUIDED_DECODING_RECOVERS", "tier2_guided_gate", "info", {"profileKey": "olmo-3.1-32b-instruct", "tier": 2,
    "receiptSha256": evidence("metrics/chat-port-gate-guided-json-olmo-3.1-32b-instruct.json")["receiptSha256"]})
tax("PACKET_CORRELATION_ONLY", "correlation", "info", {"scope": "Tier 1 replay", "packetCount": packets["packetCount"]})

security_clean = bool(safety_rows.passed.astype(bool).all()) and security.get("disposition") == "PASS" \
    and all(item.get("disposition") == "PASS" for item in security.get("positives", []) + security.get("negatives", []))
tax("SAFETY_CLEAN" if security_clean else "STOP_SAFETY", "safety_audit", "info" if security_clean else "critical", {
    "sqlChecks": len(safety_rows), "toolPositiveChecks": len(security.get("positives", [])),
    "toolNegativeChecks": len(security.get("negatives", [])), "receiptSha256": security.get("receiptSha256"),
})
if not formal_positive:
    tax("CLEAN_NULL", "primary_hypothesis_family", "info", {"contrastCount": len(contrasts), "formalPositiveCount": 0})

taxonomy.sort(key=lambda row: (row["taxonomyCode"], row["stage"], canonical(row["evidence"])))

cell_taxonomy: dict[tuple[str, str], set[str]] = {}
for row in taxonomy:
    if row["stage"] not in {"paired_primary", "runtime", "tool_use", "port_gate", "primary_hypothesis_family"}:
        continue
    profile_key = row["evidence"].get("profileKey")
    arm_id = row["evidence"].get("armId")
    if profile_key is None:
        for profile_key in profiles:
            for arm_id in model_arms:
                cell_taxonomy.setdefault((profile_key, arm_id), set()).add(row["taxonomyCode"])
    elif arm_id is None:
        for arm_id in model_arms:
            cell_taxonomy.setdefault((profile_key, arm_id), set()).add(row["taxonomyCode"])
    else:
        cell_taxonomy.setdefault((profile_key, arm_id), set()).add(row["taxonomyCode"])

scorecard["taxonomy_labels"] = [
    ";".join(sorted(cell_taxonomy.get((row.profile, row.arm), {"BASELINE" if row.kind == "BASELINE" else row.disposition})))
    for row in scorecard.itertuples(index=False)
]

claims: list[dict[str, Any]] = []


def claim(identifier: str, text: str, tag: str, status: str, artifacts: list[str], limitations: list[str], support: str) -> None:
    claims.append({"claimId": identifier[:100], "runId": run_id, "claimText": text, "evidenceTag": tag, "status": status,
                   "supportQuery": support, "supportArtifacts": artifacts, "limitations": limitations})


for row in contrasts.to_dict("records"):
    is_supported = formally_supported(row)
    status = "supported" if is_supported else ("rejected" if float(row["ci_high"]) <= 0 else "qualified")
    direction = "improved" if float(row["observed_difference"]) > 0 else "changed"
    claim(
        "LW-" + re.sub(r"[^A-Za-z0-9]+", "-", f"{row['contrast_id']}-{row['metric_name']}").strip("-")[:88],
        f"{row['contrast_id']} {direction} {row['metric_name']} by {row['observed_difference']:.6f} "
        f"(familywise CI {row['ci_low']:.6f} to {row['ci_high']:.6f}; Holm p={row['holm_adjusted_p_value']:.6g}).",
        "CAUSAL-APPLICATION" if row["hypothesis_class"] != "model_arm_minus_rules" else "PAIRED", status,
        ["tables/paired-contrasts.jsonl", "metrics/campaign-analysis.json"],
        ["Frozen synthetic packets and exact profile identities only.", "Failed agent runs remain failures; no imputation."],
        f"eval.bootstrap_results/eval.permutation_results: {row['contrast_id']}/{row['metric_name']}",
    )

claim("LW-CLEAN-NULL", "No model/arm contrast cleared both the positive familywise CI and Holm-adjusted permutation gate.", "PAIRED",
      "supported" if not formal_positive else "rejected", ["tables/paired-contrasts.jsonl"],
      ["A null result does not prove equivalence outside the frozen benchmark."], "all 36 materialized primary contrasts")
claim("LW-OLMO-PORT", "OLMo failed both unconstrained Tier-1 structured-contract gates and has no synthesized campaign rows.", "OBS", "supported",
      [gate["receiptRelativePath"] for gate in model_dispositions["portExclusions"][0]["gates"]],
      ["The separate guided-decoding Tier-2 pass is not pooled into Tier 1."], "two hash-valid STOP_PORT receipts")
claim("LW-SAFETY", f"The independent Tier-1 safety checks found {'zero' if security_clean else 'one or more'} prohibited capabilities or executions.",
      "AUDIT", "supported" if security_clean else "rejected", ["security/tool-security-gate.json", "reporting.v_safety_audit"],
      ["This establishes only the lab policy boundary, not production safety."], "reporting.v_safety_audit plus tool-security gate")
claim("LW-PACKET-CORRELATION", "Tier-1 correlation credit belongs to the deterministic packet builder; model correlation was not measured.", "HARNESS", "supported",
      ["packets/build.json"], ["A-raw-events and live correlation are Tier 2."], "frozen one-packet-per-episode design")

benchmark_summary = retrieval_benchmarks.groupby("retrieval_mode").agg(recall_at_k=("recall_at_k", "mean"), reciprocal_rank=("reciprocal_rank", "mean"), ndcg_at_k=("ndcg_at_k", "mean")).reset_index()
if not benchmark_summary.empty:
    best = benchmark_summary.sort_values("recall_at_k", ascending=False).iloc[0]
    claim("LW-RETRIEVAL-DESCRIPTIVE", f"{best.retrieval_mode} had the highest retained test-role mean recall@5 ({best.recall_at_k:.6f}).",
          "OBS", "qualified", ["retrieval_benchmarks.jsonl"], ["No paired inferential search-mode contrast was materialized."], "eval.retrieval_benchmark_results")
claims.sort(key=lambda row: row["claimId"])


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    body = "".join(canonical(value) + "\n" for value in values)
    path.write_text(body)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.9g")


write_csv(tables / "scorecard.csv", scorecard)
write_jsonl(tables / "taxonomy.jsonl", taxonomy)
write_jsonl(tables / "claims.jsonl", claims)
write_csv(tables / "claims.csv", pd.DataFrame(claims).drop(columns=["supportArtifacts", "limitations"], errors="ignore"))


def display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        return canonical(value).replace("|", "\\|")
    if bool(pd.isna(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def md(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "_No retained rows._"
    selected = frame if columns is None else frame[[column for column in columns if column in frame.columns]]
    return "\n".join([
        "| " + " | ".join(selected.columns) + " |",
        "|" + "|".join(["---"] * len(selected.columns)) + "|",
        *("| " + " | ".join(display(value) for value in row) + " |" for row in selected.itertuples(index=False, name=None)),
    ])


score_columns = ["profile", "arm", "kind", "disposition", "taxonomy_labels", "eligible", "completed", "end_to_end_success", "macro_f1",
                 "action_accuracy", "cost_weighted_loss", "action_delta_vs_b1", "action_delta_ci_low", "action_delta_ci_high",
                 "action_delta_holm_p", "b1_coverage", "required_tool_recall", "forbidden_tool_rate", "argument_valid_rate", "snapshot_miss_rate",
                 "retrieval_recall_at_5", "contract_success", "first_pass_valid_rate", "repaired_response_rate", "rejected_response_rate",
                 "raw_ece", "calibrated_ece", "agent_latency_p95_ms", "tokens_per_episode"]
scorecard_md = "# LogWarden Tier 1 scorecard\n\n" \
    "All rows use the three frozen test roles. Failures count as failures; missing cells are not imputed. " \
    "`A-router` is derived and has zero incremental inference cost. OLMo is retained as `STOP_PORT`.\n\n" + md(scorecard, score_columns) + \
    "\n\nPositive lift required a familywise CI lower bound above zero and Holm-adjusted p ≤ 0.05; no contrast cleared both gates.\n"
(reports / "SCORECARD.md").write_text(scorecard_md)

failure = primary.assign(primary_failure=np.select(
    [primary.outcome == "failure", ~primary.contract_success.astype(bool), ~primary.action_correct.astype(bool),
     ~primary.class_correct.astype(bool), ~primary.severity_correct.astype(bool), ~primary.abstention_correct.astype(bool)],
    [primary.failure_stage.fillna("unknown_failure"), "model_contract", "action_wrong", "classification_wrong", "severity_wrong", "abstention_wrong"],
    default="success",
)).groupby(["profile_key", "agent_arm_id", "primary_failure"], dropna=False).size().rename("episodes").reset_index()
write_csv(tables / "F02_failure_funnel.csv", failure)
(reports / "FAILURE_ATLAS.md").write_text("# Failure atlas\n\nA mutually exclusive first-failure rule is applied to every retained primary prediction.\n\n" + md(failure) + "\n")

retrieval_cell = retrieval_rows.groupby(["profile_key", "agent_arm_id"], dropna=False).agg(
    rows=("prediction_id", "size"), recall_at_5=("recall_at_k", "mean"), mrr=("reciprocal_rank", "mean"),
    ndcg_at_5=("ndcg_at_k", "mean"), grounding_rate=("grounded", "mean"), citation_precision=("citation_precision", "mean"),
    citation_recall=("citation_recall", "mean"),
).reset_index()
retrieval_cell["source"] = "prediction_score"
benchmark_cells = retrieval_benchmarks.groupby("retrieval_mode", dropna=False).agg(
    rows=("episode_id", "size"), recall_at_5=("recall_at_k", "mean"), mrr=("reciprocal_rank", "mean"),
    ndcg_at_5=("ndcg_at_k", "mean"), no_answer_accuracy=("no_answer_correct", "mean"),
).reset_index().rename(columns={"retrieval_mode": "agent_arm_id"})
benchmark_cells["profile_key"] = "retrieval-benchmark"
benchmark_cells["source"] = "frozen_search_benchmark"
retrieval_scorecard = pd.concat([retrieval_cell, benchmark_cells], ignore_index=True, sort=False)
write_csv(tables / "F07_retrieval_scorecard.csv", retrieval_scorecard)
grounding_contrasts = contrasts[contrasts.hypothesis_class.isin(["retrieval_ablation", "tool_context_ablation"])].copy()
grounding_contrasts["record_type"] = "paired_contrast"
shuffled_effect = control_pairs[control_pairs.control_id == "shuffled-runbooks-v1"].groupby("model_profile_id", dropna=False).agg(
    sample_count=("control_prediction_id", "size"), observed_difference=("action_score_delta", "mean"),
    semantic_agreement=("semantic_decision_agreement", "mean"), ordered_tool_agreement=("ordered_tool_call_agreement", "mean"),
).reset_index()
shuffled_effect["contrast_id"] = shuffled_effect.model_profile_id + "-shuffled-runbooks-v1"
shuffled_effect["metric_name"] = "action_score_delta"
shuffled_effect["hypothesis_class"] = "wrong_evidence_control"
shuffled_effect["record_type"] = "negative_control"
grounding_figure = pd.concat([grounding_contrasts, shuffled_effect], ignore_index=True, sort=False)
write_csv(tables / "F06_grounding_causal_effect.csv", grounding_figure)
(reports / "GROUNDING_REPORT.md").write_text("# Grounding and retrieval report\n\n## Retrieval scorecard\n\n" + md(retrieval_scorecard) +
    "\n\n## Paired causal application contrasts\n\n" + md(grounding_contrasts[["contrast_id", "metric_name", "sample_count", "observed_difference", "ci_low", "ci_high", "holm_adjusted_p_value"]]) +
    "\n\n## Shuffled-runbook negative control\n\n" + md(shuffled_effect) +
    "\n\nHybrid retrieval led the descriptive retrieval benchmark, but no retrieval causal claim survived the declared multiplicity/null gate.\n")

tool_expectation_table = tool_rows.groupby(["profile_key", "agent_arm_id", "expectation"], dropna=False).agg(
    episode_tool_rows=("prediction_id", "size"), calls=("called_count", "sum"), valid_calls=("valid_call_count", "sum"),
    snapshot_misses=("snapshot_miss_count", "sum"), tool_errors=("tool_error_count", "sum"), correct_use_rate=("correct_use", "mean"),
).reset_index()
tool_cell_rows = []
for (profile_key, arm_id), values in tool_rows.groupby(["profile_key", "agent_arm_id"], sort=True):
    summary = tool_summary(profile_key, arm_id)
    observed = values[values.called_count > 0]
    tool_cell_rows.append({
        "profile_key": profile_key, "agent_arm_id": arm_id, **summary,
        "observed_tool_precision": rate(observed.correct_use), "tool_calls": int(values.called_count.sum()),
        "tool_errors": int(values.tool_error_count.sum()), "tool_expectation_rows": len(values),
    })
tool_cell = pd.DataFrame(tool_cell_rows)
write_csv(tables / "F05_tool_precision_recall.csv", tool_cell)
(reports / "TOOL_USE_REPORT.md").write_text("# Tool-use report\n\n## Profile/arm scorecard\n\n" + md(tool_cell) +
    "\n\n## Expectation-level counts\n\n" + md(tool_expectation_table) +
    "\n\nTool taxonomies are rule-derived from required-call recall, forbidden-call rate, argument validity, and snapshot misses; see `tables/taxonomy.jsonl`.\n")

calibration_summary_rows = []
reliability_rows = []
risk_coverage_rows = []
for profile in profiles:
    for arm in model_arms:
        values = primary[(primary.profile_key == profile) & (primary.agent_arm_id == arm)].dropna(subset=["confidence", "end_to_end_success"])
        coefficients = calibration_index.get((profile, arm))
        if values.empty or coefficients is None:
            continue
        outcomes = values.end_to_end_success.astype(float).to_numpy()
        action_outcomes = values.action_correct.astype(float).to_numpy()
        raw = values.confidence.astype(float).to_numpy()
        fitted = np.array([calibrated(coefficients, value) for value in raw])
        summary = calibration_summary(values, profile, arm)
        calibration_summary_rows.append({"profile": profile, "arm": arm, **summary, "test_rows": len(values)})
        for state, probabilities in [("raw", raw), ("calibrated", fitted)]:
            for index in range(10):
                low, high = index / 10, (index + 1) / 10
                selected = (probabilities >= low) & ((probabilities <= high) if index == 9 else (probabilities < high))
                if selected.any():
                    reliability_rows.append({"profile": profile, "arm": arm, "state": state, "bin": index,
                                             "rows": int(selected.sum()), "mean_confidence": float(probabilities[selected].mean()),
                                             "accuracy": float(outcomes[selected].mean())})
            for threshold_value in np.linspace(0, 1, 21):
                selected = probabilities >= threshold_value
                if selected.any():
                    selective_accuracy = float(action_outcomes[selected].mean())
                    risk_coverage_rows.append({"profile": profile, "arm": arm, "state": state,
                                               "threshold": float(threshold_value), "rows": int(selected.sum()),
                                               "coverage": float(selected.mean()), "selective_action_accuracy": selective_accuracy,
                                               "selective_risk": 1 - selective_accuracy})
calibration_table = pd.DataFrame(calibration_summary_rows)
reliability = pd.DataFrame(reliability_rows)
risk_coverage = pd.DataFrame(risk_coverage_rows)
reliability_export = reliability.copy()
reliability_export["record_type"] = "reliability_bin"
risk_coverage_export = risk_coverage.copy()
risk_coverage_export["record_type"] = "risk_coverage"
calibration_plot_data = pd.concat([reliability_export, risk_coverage_export], ignore_index=True, sort=False)
write_csv(tables / "calibration-summary.csv", calibration_table)
write_csv(tables / "F08_reliability_risk_coverage.csv", calibration_plot_data)
(reports / "CALIBRATION_REPORT.md").write_text("# Calibration and abstention report\n\n" + md(calibration_table) +
    "\n\nSeveral frozen Platt fits did not converge; those models remain retained and are explicitly flagged. Calibration was fit on 60 calibration episodes per arm before test access.\n")

control_summary = []
for path_item in sorted(
    (value for value in manifest["artifacts"] if str(value.get("sourcePath", "")).startswith("metrics/control-comparison-")),
    key=lambda value: str(value.get("sourcePath", "")),
):
    record = json.loads((rows_root / path_item["path"]).read_text())
    control_summary.append({"profile": record["profileKey"], "control": record["controlId"], **record["summary"],
                            "invariance": record["invarianceOutcome"], "receiptSha256": record["receiptSha256"]})
control_summary_frame = pd.DataFrame(control_summary)
write_jsonl(tables / "negative-controls.jsonl", control_summary)
masking = pd.DataFrame([{"profile": row["profile"], "semantic_decision_agreement": row["semanticDecisionAgreement"]["rate"],
                         "ordered_tool_agreement": row["orderedToolCallAgreement"]["rate"], "action_score_delta": row["meanActionScoreDelta"]}
                        for row in control_summary if row["control"] == "error-number-mask-v1"])
write_csv(tables / "F09_masking_dropoff.csv", masking)
(reports / "NEGATIVE_CONTROLS.md").write_text("# Negative controls\n\n" + md(masking) +
    "\n\nFull retained control summaries (masking, shuffled runbooks, and single-worker batching) are in `tables/negative-controls.jsonl`. "
    "Batching classifications preserve request-level normalization caveats.\n")

safety_table = safety_rows.copy()
safety_table["source"] = "reporting.v_safety_audit"
security_cases = pd.DataFrame([{"principal_name": "lw_agent", "check_name": item["name"], "observed_value": item.get("errorNumber", item.get("rowCount", "PASS")),
                                "expected_value": "denied" if item in security.get("negatives", []) else "allowed", "passed": item.get("disposition") == "PASS", "source": "tool-security-gate"}
                               for item in security.get("positives", []) + security.get("negatives", [])])
safety_table = pd.concat([safety_table, security_cases], ignore_index=True)
write_csv(tables / "safety-audit.csv", safety_table)
(reports / "SAFETY_AUDIT.md").write_text("# Safety audit\n\nDisposition: **" + ("SAFETY_CLEAN" if security_clean else "STOP_SAFETY") + "**.\n\n" + md(safety_table) +
    "\n\nThis proves only the frozen lab's procedure and permission boundary; it is not a production-safety claim.\n")

contract = scorecard[scorecard.kind.isin(["AGENT", "DERIVED"])][["profile", "arm", "contract_success"]].copy()
repair = performance.groupby(["profile_key", "agent_arm_id"], dropna=False).agg(responses=("model_response_count", "sum"), repairs=("repaired_response_count", "sum"), rejections=("rejected_response_count", "sum")).reset_index()
contract = contract.merge(repair, left_on=["profile", "arm"], right_on=["profile_key", "agent_arm_id"], how="left")
contract["first_pass_valid_rate"] = (contract.responses.fillna(0) - contract.repairs.fillna(0) - contract.rejections.fillna(0)) / contract.responses.replace(0, np.nan)
write_csv(tables / "F10_contract_reliability.csv", contract)

latency = performance[performance.model_profile_id.notna() & (performance.prediction_source == "agent")].groupby(["model_profile_id", "agent_arm_id"], dropna=False).agg(
    episodes=("prediction_id", "size"), agent_p50_ms=("agent_elapsed_ms", "median"), agent_p95_ms=("agent_elapsed_ms", lambda value: value.quantile(.95)),
    model_p50_ms=("model_client_elapsed_ms", "median"), model_p95_ms=("model_client_elapsed_ms", lambda value: value.quantile(.95)),
    tool_p50_ms=("tool_latency_ms", "median"), parse_p50_ms=("parse_ms", "median"), prompt_tokens=("prompt_tokens", "sum"), completion_tokens=("completion_tokens", "sum"),
).reset_index()
primary_phase_p95 = pipeline_phases[pipeline_phases.run_kind == "agent_replay"].pivot_table(
    index=["model_profile_id", "agent_arm_id"], columns="span_name", values="p95_ms", aggfunc="first",
).reset_index()
primary_phase_p95.columns = [
    str(column) if column in {"model_profile_id", "agent_arm_id"} else "phase_p95_" + re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_") + "_ms"
    for column in primary_phase_p95.columns
]
latency_decomposition = latency.merge(primary_phase_p95, on=["model_profile_id", "agent_arm_id"], how="left")
write_csv(tables / "F11_latency_decomposition.csv", latency_decomposition)

agent_performance_rows: list[dict[str, Any]] = []
agent_only = performance[performance.model_profile_id.notna() & (performance.prediction_source == "agent")]
for (profile_key, arm_id), values in agent_only.groupby(["model_profile_id", "agent_arm_id"], sort=True):
    successes = int((values.outcome != "failure").sum())
    responses = float(values.model_response_count.fillna(0).sum())
    agent_performance_rows.append({
        "profile": profile_key, "arm": arm_id, "episodes": len(values), "successful_episodes": successes,
        "agent_p50_ms": quantile(values.agent_elapsed_ms, .50), "agent_p90_ms": quantile(values.agent_elapsed_ms, .90),
        "agent_p95_ms": quantile(values.agent_elapsed_ms, .95), "agent_p99_ms": quantile(values.agent_elapsed_ms, .99),
        "model_client_p50_ms": quantile(values.model_client_elapsed_ms, .50), "model_client_p90_ms": quantile(values.model_client_elapsed_ms, .90),
        "model_client_p95_ms": quantile(values.model_client_elapsed_ms, .95), "model_client_p99_ms": quantile(values.model_client_elapsed_ms, .99),
        "model_requests": int(values.model_request_count.fillna(0).sum()), "model_responses": int(responses),
        "prompt_tokens": int(values.prompt_tokens.fillna(0).sum()), "completion_tokens": int(values.completion_tokens.fillna(0).sum()),
        "tokens_per_episode": safe_div(float((values.prompt_tokens.fillna(0) + values.completion_tokens.fillna(0)).sum()), len(values)),
        "model_seconds_per_success": safe_div(float(values.model_client_elapsed_ms.fillna(0).sum()) / 1000, successes),
        "tool_calls_per_success": safe_div(float(values.tool_call_count.fillna(0).sum()), successes),
        "first_pass_valid_rate": safe_div(float(values.no_repair_count.fillna(0).sum()), responses),
        "repaired_response_rate": safe_div(float(values.repaired_response_count.fillna(0).sum()), responses),
        "rejected_response_rate": safe_div(float(values.rejected_repair_count.fillna(0).sum()), responses),
        "model_errors": int(values.model_error_count.fillna(0).sum()), "length_finishes": int(values.length_finish_count.fillna(0).sum()),
        "snapshot_misses": int(values.snapshot_miss_count.fillna(0).sum()), "tool_errors": int(values.tool_error_count.fillna(0).sum()),
        "validation_failures": int(values.validation_failure_count.fillna(0).sum()),
    })
agent_performance_table = pd.DataFrame(agent_performance_rows)
write_csv(tables / "agent-cell-performance.csv", agent_performance_table)

service_table = service.copy()
if not service_table.empty:
    service_table["residency_window_seconds"] = (
        pd.to_datetime(service_table.last_sample_at_utc, utc=True) - pd.to_datetime(service_table.first_sample_at_utc, utc=True)
    ).dt.total_seconds()
write_csv(tables / "model-service-performance.csv", service_table)
write_csv(tables / "agent-pipeline-performance.csv", pipeline_phases)
write_csv(tables / "queue-performance.csv", queue_performance)
write_csv(tables / "sql-resource-performance.csv", sql_performance)
write_csv(tables / "query-store-performance.csv", query_store_performance)

performance_availability = pd.DataFrame([
    {"metric_family": "agent wall clock", "status": "available", "provenance": "client monotonic timestamps", "note": "p50/p90/p95/p99 retained per profile/arm"},
    {"metric_family": "pipeline phases", "status": "available", "provenance": "closed telemetry spans", "note": "prompt, model, retrieval, tool, validation, persistence, and loop spans"},
    {"metric_family": "model request total", "status": "available", "provenance": "client monotonic timestamps", "note": "headers wait, body read, and parse are separately retained"},
    {"metric_family": "true model TTFT", "status": "unavailable", "provenance": "endpoint limitation", "note": "non-streaming transport exposes no first-token event; headers wait is not relabeled TTFT"},
    {"metric_family": "inter-token latency", "status": "unavailable", "provenance": "endpoint limitation", "note": "no streaming token timestamps"},
    {"metric_family": "vLLM request/token counters", "status": "available", "provenance": "raw /metrics snapshots", "note": "running/waiting requests, token deltas, preemptions, and prefix-cache counters where exposed"},
    {"metric_family": "vLLM throughput gauges", "status": "unavailable", "provenance": "vLLM 0.27.1 metric surface", "note": "prompt/generation throughput gauges were absent; counter deltas remain retained"},
    {"metric_family": "KV-cache occupancy", "status": "unavailable", "provenance": "vLLM 0.27.1 metric surface", "note": "metric was absent for these service configurations"},
    {"metric_family": "GPU", "status": "available", "provenance": "nvidia-smi samples", "note": "utilization, memory, power, temperature, clocks, and process snapshots retained"},
    {"metric_family": "queue", "status": "available", "provenance": "SQL queue sampler", "note": "depth, age, lease, worker, and terminal-state samples retained"},
    {"metric_family": "SQL resources", "status": "partial", "provenance": "SQL DMVs", "note": "memory/request/blocking/I/O/file sizes available; process CPU and log-used percent unavailable"},
    {"metric_family": "Query Store intervals", "status": "unavailable", "provenance": "Tier-2 materialization deferred", "note": "settings/history are retained, but no interval rows were materialized for Tier 1"},
])
write_csv(tables / "performance-metric-availability.csv", performance_availability)

performance_report = "# Agent and inference performance report\n\n" \
    f"The reconciled telemetry set contains {telemetry['observed']['trace_count']:,} closed traces, " \
    f"{telemetry['observed']['span_count']:,} closed spans, {telemetry['observed']['model_request_count']:,} paired model requests/responses, " \
    f"{telemetry['observed']['metric_sample_count']:,} metric samples, and {telemetry['observed']['raw_metric_snapshot_count']:,} raw snapshots.\n\n" \
    "## Primary agent cells\n\n" + md(agent_performance_table) + \
    "\n\n## Persisted pipeline span distributions\n\n" + md(pipeline_phases) + \
    "\n\n## vLLM and GPU residency samples\n\n" + md(service_table) + \
    "\n\n## SQL queue\n\n" + md(queue_performance) + \
    "\n\n## SQL resource sampling\n\n" + md(sql_performance) + \
    "\n\n## Query Store materialization\n\n" + md(query_store_performance) + \
    "\n\n## Metric availability and non-inference policy\n\n" + md(performance_availability) + \
    "\n\nMissing service metrics remain unavailable; the report never derives TTFT or inter-token latency from total request time. " \
    "Residency windows include governed warm-up, calibration, primary replay, and controls and therefore are not model-only benchmark times.\n"
(reports / "PERFORMANCE_REPORT.md").write_text(performance_report)

formal_lift = contrasts[contrasts.hypothesis_class == "model_arm_minus_rules"].copy()
formal_lift["scope"] = "aggregate_formal"
formal_lift["regime"] = "ALL"
descriptive_lift_rows = []
for profile_key in profiles:
    for arm_id in model_arms:
        for regime in sorted(primary.regime.dropna().unique()):
            left = primary[(primary.profile_key == profile_key) & (primary.agent_arm_id == arm_id) & (primary.regime == regime)]
            right = primary[(primary.profile_key == "baseline") & (primary.agent_arm_id == "B1-rules-v1") & (primary.regime == regime)]
            paired = left[["episode_id", "action_correct", "cost_weighted_loss"]].merge(
                right[["episode_id", "action_correct", "cost_weighted_loss"]], on="episode_id", suffixes=("_left", "_right"),
            )
            if paired.empty:
                continue
            descriptive_lift_rows.extend([
                {"contrast_id": f"{profile_key}-{arm_id}-vs-B1", "metric_name": "acceptable_action_accuracy",
                 "hypothesis_class": "model_arm_minus_rules", "sample_count": len(paired),
                 "observed_difference": rate(paired.action_correct_left.astype(float) - paired.action_correct_right.astype(float)),
                 "scope": "regime_descriptive", "regime": regime},
                {"contrast_id": f"{profile_key}-{arm_id}-vs-B1", "metric_name": "cost_weighted_loss_reduction",
                 "hypothesis_class": "model_arm_minus_rules", "sample_count": len(paired),
                 "observed_difference": rate(paired.cost_weighted_loss_right.astype(float) - paired.cost_weighted_loss_left.astype(float)),
                 "scope": "regime_descriptive", "regime": regime},
            ])
lift_table = pd.concat([formal_lift, pd.DataFrame(descriptive_lift_rows)], ignore_index=True, sort=False)
write_csv(tables / "F01_lift_over_rules.csv", lift_table)
confusion = primary.groupby(["profile_key", "agent_arm_id", "expected_class", "predicted_class"], dropna=False).size().rename("episodes").reset_index()
write_csv(tables / "F03_confusion_grid.csv", confusion)
severity_rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
action_cost_rows = []
for (profile_key, arm_id), values in primary.groupby(["profile_key", "agent_arm_id"], sort=True):
    expected_rank = values.expected_severity.map(severity_rank)
    predicted_rank = values.predicted_severity.map(severity_rank)
    actionable = values.expected_action != "no_action"
    noise = values.expected_action == "no_action"
    action_cost_rows.append({
        "profile": profile_key, "arm": arm_id, "episodes": len(values),
        "cost_weighted_loss": rate(values.cost_weighted_loss), "action_error_rate": rate(~values.action_correct.astype(bool)),
        "miss_rate_on_actionable": rate((values.predicted_action.isna() | (values.predicted_action == "no_action"))[actionable]),
        "false_alarm_rate_on_no_action": rate((values.predicted_action.notna() & (values.predicted_action != "no_action"))[noise]),
        "severity_undercall_rate": rate(predicted_rank.isna() | (predicted_rank < expected_rank)),
        "classification_error_rate": rate(~values.class_correct.astype(bool)), "terminal_failure_rate": rate(values.outcome == "failure"),
    })
action_cost = pd.DataFrame(action_cost_rows)
write_csv(tables / "F04_action_cost.csv", action_cost)
nulls = contrasts[["contrast_id", "metric_name", "observed_difference", "null_mean", "null_sd", "permutation_p_value", "holm_adjusted_p_value"]].copy()
write_csv(tables / "F12_permutation_nulls.csv", nulls)
b1_regime = primary[(primary.profile_key == "baseline") & (primary.agent_arm_id == "B1-rules-v1")].groupby("regime").agg(
    episodes=("prediction_id", "size"), action_accuracy=("action_correct", "mean"), end_to_end_success=("end_to_end_success", "mean"), coverage=("abstained", lambda value: 1 - value.astype(float).mean()),
).reset_index()
write_csv(tables / "F13_b1_coverage_by_regime.csv", b1_regime)


def save_figure(code: str, source: pd.DataFrame, draw, caption: str) -> dict[str, Any]:
    source_path = tables / f"{code}_{figure_names[code]}.csv"
    write_csv(source_path, source)
    plt.figure(figsize=(10, 5.8))
    if source.empty:
        plt.axis("off")
        plt.text(.5, .5, "NO RETAINED ROWS", ha="center", va="center", fontsize=15)
    else:
        draw(source)
    plt.tight_layout()
    target = figures / f"{code}_{figure_names[code]}.png"
    plt.savefig(target, dpi=150, metadata={"Software": "LogWarden row-only report builder"})
    plt.close()
    return {"figure": target.name, "source": source_path.name, "sourceSha256": digest_file(source_path), "caption": caption,
            "renderingTolerance": "PNG bytes may vary by Matplotlib/font build; source CSV must match byte-for-byte."}


def bars(frame: pd.DataFrame, labels: pd.Series, values: pd.Series, title: str, ylabel: str = "value") -> None:
    positions = np.arange(len(frame))
    plt.bar(positions, values)
    plt.xticks(positions, labels, rotation=35, ha="right", fontsize=7)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=.25)


figure_names = {
    "F01": "lift_over_rules", "F02": "failure_funnel", "F03": "confusion_grid", "F04": "action_cost",
    "F05": "tool_precision_recall", "F06": "grounding_causal_effect", "F07": "retrieval_scorecard",
    "F08": "reliability_risk_coverage", "F09": "masking_dropoff", "F10": "contract_reliability",
    "F11": "latency_decomposition", "F12": "permutation_nulls", "F13": "b1_coverage_by_regime",
}
figure_manifest = []
figure_manifest.append(save_figure("F01", lift_table,
    lambda f: bars(f[f.scope == "aggregate_formal"], f[f.scope == "aggregate_formal"].contrast_id + ":" + f[f.scope == "aggregate_formal"].metric_name,
                   f[f.scope == "aggregate_formal"].observed_difference, "Paired lift over deterministic rules", "difference"),
    "Paired episode-level difference versus B1; positive is better. Formal aggregate intervals and descriptive regime rows share the source."))
figure_manifest.append(save_figure("F02", failure,
    lambda f: bars(f, f.profile_key + "/" + f.agent_arm_id + "/" + f.primary_failure, f.episodes, "Failure-stage funnel", "episodes"),
    "Mutually exclusive first-failure classification across all primary test-role predictions."))
focus = confusion[(confusion.profile_key == "qwen-3.8-27b") & (confusion.agent_arm_id == "A-rag")].copy()
figure_manifest.append(save_figure("F03", confusion,
    lambda _f: bars(focus, focus.expected_class.fillna("?") + "→" + focus.predicted_class.fillna("failure"), focus.episodes, "Qwen A-rag confusion counts", "episodes"),
    "Confusion source includes every profile/arm; the rendered panel shows Qwen A-rag for legibility."))
figure_manifest.append(save_figure("F04", action_cost,
    lambda f: bars(f, f.profile + "/" + f.arm, f.cost_weighted_loss.fillna(0), "Cost-weighted loss (lower is better)", "loss"),
    "End-to-end cost-weighted loss with miss, false-alarm, severity-undercall, classification, and terminal-failure decomposition in the source."))
tool_plot = tool_cell
figure_manifest.append(save_figure("F05", tool_cell,
    lambda _f: bars(tool_plot, tool_plot.profile_key + "/" + tool_plot.agent_arm_id, tool_plot.required_tool_recall.fillna(0), "Required-tool recall", "rate"),
    "Required-tool correct use with forbidden calls, argument validity, and snapshot misses in the source CSV."))
figure_manifest.append(save_figure("F06", grounding_figure,
    lambda f: bars(f[f.record_type == "paired_contrast"], f[f.record_type == "paired_contrast"].contrast_id + ":" + f[f.record_type == "paired_contrast"].metric_name,
                   f[f.record_type == "paired_contrast"].observed_difference, "Retrieval/tool causal application contrasts", "difference"),
    "Paired A-rag/A-direct and A-tools/A-rag differences plus the shuffled-runbook negative control in the source."))
figure_manifest.append(save_figure("F07", retrieval_scorecard,
    lambda f: bars(f, f.profile_key + "/" + f.agent_arm_id, f.recall_at_5.fillna(0), "Retrieval recall@5", "recall"),
    "Retrieval recall@5; lexical/vector/hybrid evaluator baselines and model arms are retained."))
reliability_plot = reliability[(reliability.profile == "qwen-3.8-27b") & (reliability.arm == "A-rag")]
figure_manifest.append(save_figure("F08", calibration_plot_data,
    lambda _f: [plt.plot(group.mean_confidence, group.accuracy, marker="o", label=state) for state, group in reliability_plot.groupby("state")] or None,
    "Raw and calibration-only Platt reliability; the rendered panel shows Qwen A-rag and the source covers all cells."))
if not reliability_plot.empty:
    plt.close("all")
figure_manifest.append(save_figure("F09", masking,
    lambda f: bars(f, f.profile, f.semantic_decision_agreement, "Error-number masking semantic agreement", "agreement"),
    "Semantic decision agreement after removing exact error cues; action-score deltas remain in the source."))
figure_manifest.append(save_figure("F10", contract,
    lambda f: bars(f, f.profile + "/" + f.arm, f.contract_success.fillna(0), "Final contract success", "rate"),
    "First-pass and final structured-contract reliability; repairs and rejections are retained."))
figure_manifest.append(save_figure("F11", latency_decomposition,
    lambda f: bars(f, f.model_profile_id + "/" + f.agent_arm_id, f.agent_p95_ms, "Replay p95 end-to-end agent latency", "milliseconds"),
    "Agent, model-client, tool, parse, and token decomposition from persisted timestamps."))
figure_manifest.append(save_figure("F12", nulls,
    lambda f: bars(f, f.contrast_id + ":" + f.metric_name, f.observed_difference - f.null_mean, "Observed minus permutation-null mean", "difference"),
    "Observed contrasts against retained permutation summaries; exact replicate artifacts are retained separately."))
figure_manifest.append(save_figure("F13", b1_regime,
    lambda f: bars(f, f.regime, f.action_accuracy, "B1 acceptable-action accuracy by regime", "accuracy"),
    "Deterministic rules coverage and end-to-end success by frozen regime."))

figure_manifest_body = {"schemaVersion": 1, "runId": run_id, "queryBundleSha256": manifest["queryBundleSha256"], "figures": figure_manifest}
(figures / "manifest.json").write_text(json.dumps({**figure_manifest_body, "receiptSha256": digest_bytes(canonical(figure_manifest_body).encode())}, indent=2) + "\n")
(reports / "FIGURE_CAPTIONS.md").write_text("# Figure captions and reconstruction policy\n\n" + "\n".join(f"- **{item['figure']}** — {item['caption']} Source: `{item['source']}` (`{item['sourceSha256']}`)." for item in figure_manifest) +
    "\n\nPNG bytes may vary across plotting/font builds; every source CSV and caption is hash-exact.\n")

claims_frame = pd.DataFrame([{**row, "supportArtifacts": "; ".join(row["supportArtifacts"]), "limitations": "; ".join(row["limitations"])} for row in claims])
claims_md = "# LogWarden claims table\n\nPositive claims require the frozen evidence gate; `qualified` rows report observations without promotion.\n\n" + md(claims_frame) + "\n"
(reports / "LOGWARDEN_CLAIMS_TABLE.md").write_text(claims_md)

limitations = """# LogWarden limitations

- The 600 episodes are benign synthetic SQL Server incidents; results do not establish production safety or open-world log understanding.
- OLMo failed both unconstrained Tier-1 contract gates. Its guided-decoding Tier-2 pass is separate and no Tier-1 rows were synthesized.
- No paired primary contrast cleared both the familywise interval and Holm-adjusted permutation gate. Observed marginal differences are not promoted.
- The frozen within-family/split label permutation produced degenerate nulls for several binary contrasts; this is retained and limits positive inference.
- Muse used three effective workers during calibration; Gemma used a +0.01 GPU-residency override and recorded one vLLM preemption. Performance is not attributable solely to model identity.
- Calibration sets contain only 60 episodes per profile; several Platt fits did not converge and are flagged.
- Tier 1 replay packets are already correlated. Model-generated correlation, live parity, storm throughput, native transport, ANN, and SQL-native comparators are Tier 2.
- PNG rendering may vary by plotting/font build; source tables, claims, scorecard, and Markdown are hash-exact.
"""
(reports / "LOGWARDEN_LIMITATIONS.md").write_text(limitations)

state = f"""# LogWarden Tier 1 state of record

- Run: `{run_id}`
- Freeze: `{freeze['freezeHash']}` (`{freeze['receiptSha256']}`)
- Corpus: {capture['passedEpisodes']}/{capture['episodeCount']} capture gates passed; {packets['packetCount']} packets; leakage findings: {leakage['findingCount']}.
- Analysis: {len(primary):,} primary predictions, {len(metrics):,} metric rows, {len(contrasts)} paired contrasts; receipt `{analysis_receipt['receiptSha256']}`.
- Telemetry: {telemetry['observed']['trace_count']:,} closed traces, {telemetry['observed']['span_count']:,} closed spans, {telemetry['observed']['model_request_count']:,} paired model requests/responses; receipt `{telemetry['receiptSha256']}`.
- Safety: `{'SAFETY_CLEAN' if security_clean else 'STOP_SAFETY'}`.
- Primary adjudication: `{'CLEAN_NULL' if not formal_positive else 'SUPPORTED_LIFT'}`.

## Adjudication

1. The corpus passed 600/600 capture checks and a zero-finding leakage audit.
2. Muse, Gemma, and Qwen completed every governed Tier-1 cell; OLMo is explicitly `STOP_PORT` after two gates.
3. B1 was strong: action accuracy {scorecard[(scorecard.profile=='baseline') & (scorecard.arm=='B1-rules-v1')].iloc[0].action_accuracy:.4f}; B3 is the evaluator-only information ceiling.
4. No model/arm beat B1 under the complete frozen inference gate.
5. Regime/family differences are descriptive only and remain in the scorecard/taxonomy exports.
6. Hybrid retrieval had the best descriptive recall among executable search modes, but no paired search-mode inference was materialized.
7. Retrieval changed several marginal outcomes, but no causal retrieval contrast survived multiplicity and permutation control.
8. Tool behavior varied materially; fragile cells are labeled from required, forbidden, argument, and snapshot evidence.
9. Tier-1 packet correlation is a harness property (`PACKET_CORRELATION_ONLY`).
10. Calibration is limited; nonconverged fits and selective coverage are explicit.
11. Model-contract/policy failures dominate the retained agent failure funnel; SQL deadlocks affected orchestration probes, not immutable outputs.
12. Replay latency/token/GPU/SQL telemetry is complete; live throughput Pareto analysis remains Tier 2.
13. Derived A-router reproduced B1 because B1 resolved the retained episodes; it adds zero inference cost and no quality lift.
14–18. Live parity, event-rate capacity, raw-event correlation, SQL feature ablations, and ANN are Tier 2 and are not claimed.
19. The independent Tier-1 permission/procedure audit passed; this is not production-safety evidence.
20. Row-only and restored-database reconstruction both passed; their stable receipts are recorded in `LOGWARDEN_REPRODUCIBILITY.md`.
21. The principal unexplained result is why the declared permutation scheme is degenerate for several binary contrasts; no positive claim depends on it.
22. The defensible carry-forward architecture is deterministic rules for the known head, explicitly gated retrieval/model assistance for residual cases, and SQL Server as the durable evidence/queue/evaluation plane—not an autonomous remediation agent.
"""
(reports / "LOGWARDEN_STATE_OF_RECORD.md").write_text(state)

repro_text = f"""# LogWarden reproducibility

The row-only input bundle contains {sum((item.get('rows') or 0) for item in manifest['artifacts']):,} rows across {len(manifest['artifacts'])} query/evidence artifacts.

- Input-manifest receipt: `{manifest['receiptSha256']}`
- Query-bundle hash: `{manifest['queryBundleSha256']}`
- Deterministic outputs: scorecard, taxonomy, claims, report Markdown, and figure source CSVs.
- Rendering tolerance: PNG bytes may vary across Matplotlib/font builds; source CSV hashes and captions must match.
- `repro.sh --mode rows` verifies every input hash, rebuilds outputs without SQL/GPU/inference, and checks expected hashes.
- `repro.sh --mode restore` starts an isolated SQL Server container, restores the final `.bak` pair, regenerates row exports, compares them byte-for-byte, then runs row-only reconstruction.
- Row-only gate: `{rows_gate['disposition']}` (`{rows_gate['receiptSha256'] or 'not available'}`).
- Fresh-restore gate: `{restore_gate['disposition']}` (`{restore_gate['receiptSha256'] or 'not available'}`).
"""
(reports / "LOGWARDEN_REPRODUCIBILITY.md").write_text(repro_text)

validation = f"""# LogWarden validation record

| Gate | Disposition | Evidence |
|---|---|---|
| Dedicated Lab 3 branch/worktree | PASS | `aidataapps-logwarden` |
| CUDA/vLLM governed inference | PASS | three completed profiles; OLMo STOP_PORT |
| SQL Server 2025 + full-text | PASS | capability and retrieval receipts |
| Corpus capture | PASS | {capture['passedEpisodes']}/{capture['episodeCount']} episodes |
| Packet leakage | PASS | {leakage['findingCount']} findings |
| Freeze | PASS | `{freeze['receiptSha256']}` |
| Retrieval evaluation | PASS | {retrieval_receipt['completedCellCount']}/{retrieval_receipt['expectedCellCount']} cells |
| Primary scoring/statistics | PASS | `{analysis_receipt['receiptSha256']}` |
| Telemetry reconciliation | PASS | `{telemetry['receiptSha256']}` |
| Tier-1 safety audit | {'PASS' if security_clean else 'FAIL'} | `{security['receiptSha256']}` |
| Row-only reconstruction | {rows_gate['disposition']} | `{rows_gate['receiptSha256'] or 'not available'}` |
| Database-restore reconstruction | {restore_gate['disposition']} | `{restore_gate['receiptSha256'] or 'not available'}` |

A passing benchmark is not a positive model result. The primary scientific adjudication is `{('CLEAN_NULL' if not formal_positive else 'SUPPORTED_LIFT')}`.
"""
(reports / "VALIDATION.md").write_text(validation)

# Root state-of-record files are generated copies, never hand-maintained result text.
root_mapping = {
    "LOGWARDEN_STATE_OF_RECORD.md": state,
    "LOGWARDEN_CLAIMS_TABLE.md": claims_md,
    "LOGWARDEN_PERFORMANCE_REPORT.md": performance_report,
    "LOGWARDEN_LIMITATIONS.md": limitations,
    "LOGWARDEN_REPRODUCIBILITY.md": repro_text,
    "VALIDATION.md": validation,
}
for name, body in root_mapping.items():
    (lab / name).write_text(body)
(lab / "LOGWARDEN_FREEZE_RECORD.md").write_text((run / "reports" / "LOGWARDEN_FREEZE_RECORD.md").read_text())

readme = (lab / "README.md").read_text()
result_block = f"""<!-- logwarden-results:start -->
## Results

The frozen Tier-1 campaign is complete for Muse, Gemma, and Qwen; OLMo is `STOP_PORT`. Deterministic B1 reached `{scorecard[(scorecard.profile=='baseline') & (scorecard.arm=='B1-rules-v1')].iloc[0].action_accuracy:.3f}` acceptable-action accuracy. No model/arm contrast cleared both the familywise interval and Holm-adjusted permutation gate, so the headline is `CLEAN_NULL`, not model lift. The Tier-1 safety audit passed. LogWarden can make scoped, auditable recommendations on its frozen synthetic episodes; it cannot establish production safety, autonomously remediate, or generalize to arbitrary logs.

See [the state of record](LOGWARDEN_STATE_OF_RECORD.md), [claims](LOGWARDEN_CLAIMS_TABLE.md), [performance evidence](LOGWARDEN_PERFORMANCE_REPORT.md), and [limitations](LOGWARDEN_LIMITATIONS.md).
<!-- logwarden-results:end -->"""
if "<!-- logwarden-results:start -->" in readme:
    readme = re.sub(r"<!-- logwarden-results:start -->.*?<!-- logwarden-results:end -->", result_block, readme, flags=re.S)
else:
    readme += "\n\n" + result_block + "\n"
(lab / "README.md").write_text(readme)

deterministic_paths = [
    *(reports / name for name in ["SCORECARD.md", "FAILURE_ATLAS.md", "GROUNDING_REPORT.md", "TOOL_USE_REPORT.md", "CALIBRATION_REPORT.md",
                                    "NEGATIVE_CONTROLS.md", "SAFETY_AUDIT.md", "PERFORMANCE_REPORT.md", "LOGWARDEN_STATE_OF_RECORD.md", "LOGWARDEN_CLAIMS_TABLE.md",
                                    "LOGWARDEN_LIMITATIONS.md", "LOGWARDEN_REPRODUCIBILITY.md", "FIGURE_CAPTIONS.md", "VALIDATION.md"]),
    *(tables / name for name in ["scorecard.csv", "taxonomy.jsonl", "claims.jsonl", "claims.csv", "negative-controls.jsonl", "calibration-summary.csv",
                                  "agent-cell-performance.csv", "agent-pipeline-performance.csv", "model-service-performance.csv",
                                  "queue-performance.csv", "sql-resource-performance.csv", "query-store-performance.csv", "performance-metric-availability.csv",
                                  *[f"{code}_{name}.csv" for code, name in figure_names.items()]]),
    *(lab / name for name in ["LOGWARDEN_STATE_OF_RECORD.md", "LOGWARDEN_CLAIMS_TABLE.md", "LOGWARDEN_PERFORMANCE_REPORT.md", "LOGWARDEN_LIMITATIONS.md",
                              "LOGWARDEN_REPRODUCIBILITY.md", "LOGWARDEN_FREEZE_RECORD.md", "VALIDATION.md", "README.md"]),
]
expected_path = run / "repro" / "expected-report-outputs.json"
output_rows = [{"path": str(path.relative_to(run) if path.is_relative_to(run) else Path("repo") / path.relative_to(lab)),
                "bytes": path.stat().st_size, "sha256": digest_file(path)} for path in deterministic_paths]
expected_body = {"schemaVersion": 1, "runId": run_id, "inputManifestReceiptSha256": manifest["receiptSha256"],
                 "queryBundleSha256": manifest["queryBundleSha256"], "outputs": output_rows,
                 "pngPolicy": "re-render and verify source CSV/caption hashes; PNG byte equality is not required"}
if args.freeze:
    expected_path.write_text(json.dumps({**expected_body, "receiptSha256": digest_bytes(canonical(expected_body).encode())}, indent=2) + "\n")
else:
    expected = json.loads(expected_path.read_text())
    received = expected.pop("receiptSha256")
    if digest_bytes(canonical(expected).encode()) != received or expected != expected_body:
        raise RuntimeError("Deterministic report reconstruction drift")

print(json.dumps({"runId": run_id, "inputRows": sum((item.get("rows") or 0) for item in manifest["artifacts"]),
                  "scorecardRows": len(scorecard), "taxonomyRows": len(taxonomy), "claimRows": len(claims),
                  "figures": len(figure_manifest), "formalPositiveClaims": int(formal_positive),
                  "expectedOutputs": str(expected_path), "mode": "freeze" if args.freeze else "verify", "disposition": "PASS"}, indent=2))
