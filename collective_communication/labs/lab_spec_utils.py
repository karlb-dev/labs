"""Helpers for scaffold labs that emit course-path artifacts."""

from __future__ import annotations

from typing import Any


def build_spec(
    lab_spec: dict[str, Any],
    *,
    args: Any,
    payload_bytes: int,
    n_devices: int,
) -> dict[str, Any]:
    spec = dict(lab_spec)
    spec.update(
        {
            "axis_name": getattr(args, "axis_name", None),
            "dtype": getattr(args, "dtype", None),
            "payload_bytes": payload_bytes,
            "n_devices": n_devices,
        }
    )
    return spec


def render_markdown(spec: dict[str, Any]) -> str:
    lines = [
        f"# {spec['title']}",
        "",
        spec["goal"],
        "",
    ]
    for key, title in (
        ("implemented_ops", "Implemented Happy Path"),
        ("deferred_ops", "Deferred Custom Work"),
        ("byte_model", "Byte Model"),
        ("pass_condition", "Pass Condition"),
        ("artifacts", "Artifacts To Inspect"),
        ("next_steps", "Next Steps"),
    ):
        values = spec.get(key) or []
        if not values:
            continue
        lines.extend([f"## {title}", ""])
        for value in values:
            lines.append(f"- {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
