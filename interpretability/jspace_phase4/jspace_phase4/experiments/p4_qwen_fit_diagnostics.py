"""Audit and plot the per-prompt diagnostics from a nested Qwen lens fit.

The frozen ``jlens.fit`` implementation prints, but does not store in its
checkpoint, one diagnostic line per accepted prompt.  This utility combines
plain producer logs and locally retained Codex execution transcripts,
deduplicates repeated observations, rejects conflicting copies, and records
exactly which prompt rows remain unavailable.  It is an engineering audit;
it neither creates evidence nor changes the append-only registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIAGNOSTIC_RE = re.compile(
    r"prompt\s+(?P<prompt>\d+)/(?P<target>\d+)\s+"
    r"seq_len=(?P<seq_len>\d+)\s+n_valid=(?P<n_valid>\d+)\s+"
    r"(?P<seconds>\d+(?:\.\d+)?)s\s+"
    r"max\|\|J\|\|/sqrt\(d\)=(?P<prompt_norm>[-+\w.]+)\s+"
    r"max_d_mean=(?P<max_d_mean>[-+\w.]+)",
)
SKIP_RE = re.compile(r"skipping prompt\s+(?P<prompt>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Diagnostic:
    prompt: int
    seq_len: int
    n_valid: int
    seconds: float
    prompt_norm: float
    max_d_mean: float

    def comparison_key(self) -> tuple[int, int, float, float]:
        return (
            self.seq_len,
            self.n_valid,
            self.prompt_norm,
            self.max_d_mean,
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def codex_output_text(path: Path) -> Iterator[str]:
    """Yield only tool outputs, never user/assistant text or tool inputs."""

    with path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid Codex JSONL at {path}:{line_number}"
                ) from exc
            payload = envelope.get("payload")
            if not isinstance(payload, dict) or "output" not in payload:
                continue
            payload_type = str(payload.get("type", ""))
            if "output" not in payload_type:
                continue
            yield from strings(payload["output"])


def diagnostics_in_text(text: str) -> Iterator[Diagnostic]:
    for match in DIAGNOSTIC_RE.finditer(text):
        yield Diagnostic(
            prompt=int(match.group("prompt")),
            seq_len=int(match.group("seq_len")),
            n_valid=int(match.group("n_valid")),
            seconds=float(match.group("seconds")),
            prompt_norm=float(match.group("prompt_norm")),
            max_d_mean=float(match.group("max_d_mean")),
        )


def merge_diagnostics(
    sources: Iterable[tuple[str, Iterable[str]]],
) -> tuple[dict[int, Diagnostic], dict[int, set[str]], set[int]]:
    rows: dict[int, Diagnostic] = {}
    provenance: dict[int, set[str]] = {}
    skipped: set[int] = set()
    for source_name, texts in sources:
        for text in texts:
            skipped.update(int(x.group("prompt")) for x in SKIP_RE.finditer(text))
            for row in diagnostics_in_text(text):
                previous = rows.get(row.prompt)
                if previous is not None and (
                    previous.comparison_key() != row.comparison_key()
                ):
                    raise RuntimeError(
                        "conflicting diagnostic copies for prompt "
                        f"{row.prompt}: {previous} versus {row}"
                    )
                if row.seq_len <= 0 or not 0 < row.n_valid <= row.seq_len:
                    raise RuntimeError(f"invalid token counts: {row}")
                if not math.isfinite(row.prompt_norm) or row.prompt_norm <= 0:
                    raise RuntimeError(f"invalid prompt norm: {row}")
                if not math.isfinite(row.max_d_mean):
                    if not (row.prompt == 1 and math.isnan(row.max_d_mean)):
                        raise RuntimeError(f"invalid running-mean change: {row}")
                rows[row.prompt] = previous or row
                provenance.setdefault(row.prompt, set()).add(source_name)
    return rows, provenance, skipped


def numeric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise RuntimeError("numeric summary requires a nonempty finite vector")
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "q90": float(np.quantile(array, 0.9)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def top_rows(
    rows: Sequence[Diagnostic], attribute: str, limit: int = 15,
) -> list[dict[str, object]]:
    ranked = sorted(rows, key=lambda row: getattr(row, attribute), reverse=True)
    return [asdict(row) for row in ranked[:limit]]


def rolling_median(values: np.ndarray, window: int = 31) -> np.ndarray:
    radius = window // 2
    return np.asarray([
        np.median(values[max(0, index - radius):index + radius + 1])
        for index in range(len(values))
    ])


def build_summary(
    rows_by_prompt: Mapping[int, Diagnostic], *, expected_prompts: int,
    skipped: set[int], checkpoint_state: Mapping[str, object] | None,
) -> dict[str, object]:
    expected = set(range(1, expected_prompts + 1))
    observed = set(rows_by_prompt)
    out_of_range = sorted(observed - expected)
    if out_of_range:
        raise RuntimeError(f"diagnostic prompt indices out of range: {out_of_range}")
    if skipped:
        raise RuntimeError(f"producer reported skipped prompts: {sorted(skipped)}")

    rows = [rows_by_prompt[index] for index in sorted(rows_by_prompt)]
    finite_d_mean = [row.max_d_mean for row in rows if math.isfinite(row.max_d_mean)]
    prompts = np.asarray([row.prompt for row in rows], dtype=np.float64)
    d_mean = np.asarray(finite_d_mean, dtype=np.float64)
    d_mean_prompts = np.asarray([
        row.prompt for row in rows if math.isfinite(row.max_d_mean)
    ], dtype=np.float64)
    slope_mask = d_mean_prompts >= 250
    log_log_slope = None
    if int(slope_mask.sum()) >= 2:
        log_log_slope = float(np.polyfit(
            np.log(d_mean_prompts[slope_mask]),
            np.log(d_mean[slope_mask]),
            1,
        )[0])

    checkpoint_audit = None
    if checkpoint_state is not None:
        checkpoint_audit = {
            key: checkpoint_state.get(key)
            for key in (
                "schema_version", "next_idx", "n_done", "checkpoint_sha256",
                "checkpoint_bytes", "fit_contract_sha256", "synced_utc",
            )
        }
        for key in ("next_idx", "n_done"):
            value = checkpoint_audit.get(key)
            if value is not None and not 0 <= int(value) <= expected_prompts:
                raise RuntimeError(f"checkpoint {key} is out of range: {value}")

    return {
        "schema_version": 1,
        "audit_tier": "engineering-diagnostic-not-registered-evidence",
        "expected_prompts": expected_prompts,
        "archived_diagnostic_rows": len(rows),
        "archived_coverage_fraction": len(rows) / expected_prompts,
        "archived_first_prompt": int(prompts.min()) if len(prompts) else None,
        "archived_last_prompt": int(prompts.max()) if len(prompts) else None,
        "missing_prompt_indices": sorted(expected - observed),
        "skipped_prompt_indices": [],
        "raw_row_archive_complete": observed == expected,
        "checkpoint_state": checkpoint_audit,
        "sequence_length_counts": dict(sorted(Counter(
            row.seq_len for row in rows).items()
        )),
        "valid_token_counts": dict(sorted(Counter(
            row.n_valid for row in rows).items()
        )),
        "seconds": numeric_summary([row.seconds for row in rows]),
        "max_jacobian_norm_over_sqrt_d": numeric_summary([
            row.prompt_norm for row in rows
        ]),
        "max_d_mean": numeric_summary(finite_d_mean),
        "max_d_mean_log_log_slope_from_prompt_250": log_log_slope,
        "top_prompt_norm_rows": top_rows(rows, "prompt_norm"),
        "top_max_d_mean_rows": top_rows(
            [row for row in rows if math.isfinite(row.max_d_mean)],
            "max_d_mean",
        ),
    }


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_csv(
    path: Path, rows: Sequence[Diagnostic], provenance: Mapping[int, set[str]],
) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "prompt", "seq_len", "n_valid", "seconds",
            "max_jacobian_norm_over_sqrt_d", "max_d_mean", "sources",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "prompt": row.prompt,
                "seq_len": row.seq_len,
                "n_valid": row.n_valid,
                "seconds": row.seconds,
                "max_jacobian_norm_over_sqrt_d": row.prompt_norm,
                "max_d_mean": row.max_d_mean,
                "sources": ";".join(sorted(provenance[row.prompt])),
            })
    os.replace(temporary, path)


def write_figure(path: Path, rows: Sequence[Diagnostic]) -> None:
    prompts = np.asarray([row.prompt for row in rows])
    norms = np.asarray([row.prompt_norm for row in rows])
    d_mean = np.asarray([row.max_d_mean for row in rows])
    seconds = np.asarray([row.seconds for row in rows])
    finite_d = np.isfinite(d_mean)

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    axis = axes[0, 0]
    axis.scatter(prompts, norms, s=10, alpha=0.55, color="#276FBF")
    axis.set_yscale("log")
    axis.set(title="Per-prompt Jacobian-norm diagnostic",
             xlabel="nested prompt index", ylabel=r"max $||J||/\sqrt{d}$")
    for index in np.argsort(norms)[-5:]:
        axis.annotate(str(int(prompts[index])), (prompts[index], norms[index]),
                      xytext=(3, 3), textcoords="offset points", fontsize=8)

    axis = axes[0, 1]
    axis.scatter(prompts[finite_d], d_mean[finite_d], s=9, alpha=0.35,
                 color="#D95F02", label="prompt")
    axis.plot(prompts[finite_d], rolling_median(d_mean[finite_d]),
              color="#7F2704", linewidth=1.5, label="31-row median")
    axis.set_yscale("log")
    axis.set(title="Running-mean change declines with fit size",
             xlabel="nested prompt index", ylabel="max_d_mean")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    axis.hist(np.log10(norms), bins=35, color="#276FBF", alpha=0.85)
    axis.axvline(np.log10(np.median(norms)), color="black", linestyle="--",
                 linewidth=1, label="median")
    axis.set(title="Heavy-tailed prompt-norm distribution",
             xlabel=r"$\log_{10}$(max $||J||/\sqrt{d}$)", ylabel="rows")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    points = axis.scatter(norms[finite_d], d_mean[finite_d], c=prompts[finite_d],
                          s=12, alpha=0.6, cmap="viridis")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(title="Prompt magnitude versus estimator movement",
             xlabel=r"max $||J||/\sqrt{d}$", ylabel="max_d_mean")
    figure.colorbar(points, ax=axis, label="prompt index")
    figure.suptitle(
        f"Qwen draw-A fit diagnostics: {len(rows)} archived rows; "
        f"median prompt time {np.median(seconds):.0f}s",
        fontsize=13,
    )

    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    figure.savefig(temporary, format=path.suffix.lstrip("."), dpi=180)
    plt.close(figure)
    os.replace(temporary, path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain-log", action="append", default=[], type=Path)
    parser.add_argument("--codex-jsonl", action="append", default=[], type=Path)
    parser.add_argument("--checkpoint-state", type=Path)
    parser.add_argument("--expected-prompts", type=int, default=1000)
    parser.add_argument("--require-complete-raw-log", action="store_true")
    parser.add_argument("--require-final-checkpoint", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--figure-stem", default="p4qa01_qwen_a1000_fit_diagnostics")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.expected_prompts <= 0:
        raise SystemExit("--expected-prompts must be positive")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, Iterable[str]]] = []
    source_records: list[dict[str, object]] = []
    for path in arguments.plain_log:
        resolved = path.resolve()
        label = f"plain:{resolved.name}"
        sources.append((label, [resolved.read_text(errors="replace")]))
        source_records.append({
            "kind": "plain-log", "name": resolved.name,
            "sha256": file_sha256(resolved), "bytes": resolved.stat().st_size,
        })
    for path in arguments.codex_jsonl:
        resolved = path.resolve()
        digest = file_sha256(resolved)
        label = f"codex-session:{digest[:12]}"
        sources.append((label, codex_output_text(resolved)))
        source_records.append({
            "kind": "codex-tool-output-jsonl", "name": resolved.name,
            "sha256": digest, "bytes": resolved.stat().st_size,
        })
    if not sources:
        raise SystemExit("at least one diagnostic source is required")

    rows_by_prompt, provenance, skipped = merge_diagnostics(sources)
    checkpoint_state = None
    if arguments.checkpoint_state is not None:
        checkpoint_state = json.loads(arguments.checkpoint_state.read_text())
    summary = build_summary(
        rows_by_prompt, expected_prompts=arguments.expected_prompts,
        skipped=skipped, checkpoint_state=checkpoint_state,
    )
    summary["sources"] = source_records

    if arguments.require_complete_raw_log and not summary["raw_row_archive_complete"]:
        raise SystemExit(
            "raw diagnostic archive is incomplete: missing prompts "
            f"{summary['missing_prompt_indices']}"
        )
    if arguments.require_final_checkpoint:
        state = summary["checkpoint_state"]
        if not isinstance(state, dict) or any(
            int(state.get(key, -1)) != arguments.expected_prompts
            for key in ("next_idx", "n_done")
        ):
            raise SystemExit("checkpoint does not prove the expected final boundary")

    rows = [rows_by_prompt[index] for index in sorted(rows_by_prompt)]
    csv_path = arguments.output_dir / "qwen_a1000_fit_diagnostics_rows.csv"
    json_path = arguments.output_dir / "qwen_a1000_fit_diagnostics_summary.json"
    write_csv(csv_path, rows, provenance)
    atomic_text(json_path, json.dumps(summary, sort_keys=True, indent=1) + "\n")
    for suffix in ("png", "pdf"):
        write_figure(
            arguments.output_dir / f"{arguments.figure_stem}.{suffix}", rows)

    print(json.dumps({
        "archived_diagnostic_rows": len(rows),
        "missing_prompt_count": len(summary["missing_prompt_indices"]),
        "raw_row_archive_complete": summary["raw_row_archive_complete"],
        "checkpoint_n_done": (
            summary["checkpoint_state"].get("n_done")
            if isinstance(summary["checkpoint_state"], dict) else None
        ),
        "output_dir": str(arguments.output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
