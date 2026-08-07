import json

import pytest

from jspace_phase4.experiments.p4_qwen_fit_diagnostics import (
    build_summary,
    codex_output_text,
    merge_diagnostics,
)


def line(prompt, *, norm=2.5, d_mean="1.0e-2"):
    return (
        f"prompt {prompt}/{prompt + 2}  seq_len=128 n_valid=111  164s  "
        f"max||J||/sqrt(d)={norm}  max_d_mean={d_mean}\n"
    )


def test_merge_diagnostics_deduplicates_consistent_copies():
    rows, provenance, skipped = merge_diagnostics([
        ("one", [line(2)]),
        ("two", [line(2), line(3, norm=7.0)]),
    ])
    assert sorted(rows) == [2, 3]
    assert provenance[2] == {"one", "two"}
    assert skipped == set()


def test_merge_diagnostics_rejects_conflicts_and_skips():
    with pytest.raises(RuntimeError, match="conflicting diagnostic copies"):
        merge_diagnostics([
            ("one", [line(2)]),
            ("two", [line(2, norm=3.5)]),
        ])
    rows, _, skipped = merge_diagnostics([
        ("one", [line(2) + "skipping prompt 7: too short\n"]),
    ])
    with pytest.raises(RuntimeError, match="reported skipped prompts"):
        build_summary(
            rows, expected_prompts=2, skipped=skipped, checkpoint_state=None)


def test_codex_reader_uses_only_tool_outputs(tmp_path):
    transcript = tmp_path / "session.jsonl"
    envelopes = [
        {"payload": {"type": "custom_tool_call", "input": line(1)}},
        {"payload": {
            "type": "custom_tool_call_output",
            "output": [{"type": "input_text", "text": line(2)}],
        }},
    ]
    transcript.write_text("".join(json.dumps(row) + "\n" for row in envelopes))
    rows, _, _ = merge_diagnostics([
        ("session", codex_output_text(transcript, max_line=2)),
    ])
    assert sorted(rows) == [2]


def test_codex_reader_stops_before_later_displayed_fixture(tmp_path):
    transcript = tmp_path / "session.jsonl"
    envelopes = [
        {"payload": {
            "type": "function_call_output", "output": line(5),
        }},
        {"payload": {
            "type": "custom_tool_call_output", "output": line(7),
        }},
    ]
    transcript.write_text("".join(json.dumps(row) + "\n" for row in envelopes))
    rows, _, skipped = merge_diagnostics([
        ("session", codex_output_text(transcript, max_line=1)),
    ])
    assert sorted(rows) == [5]
    assert skipped == set()


def test_summary_distinguishes_raw_coverage_from_checkpoint_boundary():
    rows, _, skipped = merge_diagnostics([
        ("one", [line(2), line(3, norm=7.0)]),
    ])
    summary = build_summary(
        rows,
        expected_prompts=3,
        skipped=skipped,
        checkpoint_state={
            "schema_version": 1,
            "next_idx": 3,
            "n_done": 3,
            "checkpoint_sha256": "a" * 64,
        },
    )
    assert summary["archived_diagnostic_rows"] == 2
    assert summary["missing_prompt_indices"] == [1]
    assert summary["raw_row_archive_complete"] is False
    assert summary["archived_rows_all_finite"] is True
    assert summary["checkpoint_proves_all_prompts_accepted"] is True
    assert summary["checkpoint_state"]["n_done"] == 3


def test_summary_reports_equal_weight_log_scaling_diagnostic():
    observations = [
        line(2, norm=2.0, d_mean="3.0"),
        line(3, norm=4.0, d_mean="4.0"),
        line(4, norm=3.0, d_mean="2.25"),
        line(5, norm=6.0, d_mean="3.6"),
    ]
    rows, _, skipped = merge_diagnostics([("one", observations)])
    summary = build_summary(
        rows, expected_prompts=5, skipped=skipped, checkpoint_state=None)
    regression = summary["max_d_mean_log_scaling_regression"]
    assert regression["n_rows"] == 4
    assert regression["prompt_norm_log_coefficient"] == pytest.approx(1.0)
    assert regression["prompt_index_log_coefficient"] == pytest.approx(-1.0)
    assert regression["r_squared"] == pytest.approx(1.0)
