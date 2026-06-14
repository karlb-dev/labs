"""Lab 33: Multimodal mechanistic interpretability.

The default lab is a synthetic connector smoke path. It renders deterministic
image specs, builds known visual/text/shortcut states, and audits whether the
multimodal artifact contract is strong enough to prevent caption, OCR,
background, and alignment shortcuts from becoming fake VLM mechanism claims.

Evidence level: OBS + DECODE + CAUSAL + AUDIT, scoped to synthetic connector
mode only. The default run intentionally writes science_ready=false for real
VLM claims.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L33"
DATA_FILE = "multimodal_concept_pairs.jsonl"
LAB_MANIFEST_FILE = "multimodal_MANIFEST.json"
PROMPT_SET_CAPS = {"small": 16, "medium": 24, "full": 0}
STATE_DIM = 128
STATE_TYPES = ("vision_visual", "vision", "connector", "language", "caption", "text_query")
CONTROL_PATCH_TYPES = (
    "ocr_channel_patch",
    "background_channel_patch",
    "text_query_patch",
    "wrong_region_or_ocr_patch",
    "random_patch_control",
)
VISUAL_PATCH_TYPE = "visual_region_patch"
MIN_AUC_GATE = 0.75
MIN_RECOVERY_GATE = 0.35
MIN_SPECIFICITY_GAP = 0.15
CONTROL_CLOSE_TOL = 0.05
REQUIRED_FIELDS = {
    "item_id",
    "image_path",
    "image_control_path",
    "question",
    "target",
    "distractor",
    "concept_family",
    "text_control_prompt",
    "split",
    "image_spec",
    "control_spec",
    "region",
    "notes",
}
FAMILIES = ("color", "shape", "count", "spatial", "chart", "ocr_control", "background_control")
COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


@dataclasses.dataclass
class MultiItem:
    item_id: str
    image_path: str
    image_control_path: str
    question: str
    target: str
    distractor: str
    concept_family: str
    text_control_prompt: str
    split: str
    image_spec: dict[str, Any]
    control_spec: dict[str, Any]
    region: str
    notes: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MultiItem":
        data = {key: payload.get(key) for key in REQUIRED_FIELDS}
        for key in ("image_spec", "control_spec"):
            if isinstance(data[key], str):
                data[key] = json.loads(data[key])
        return cls(
            item_id=str(data["item_id"]).strip(),
            image_path=str(data["image_path"]).strip(),
            image_control_path=str(data["image_control_path"]).strip(),
            question=str(data["question"]).strip(),
            target=str(data["target"]).strip().lower(),
            distractor=str(data["distractor"]).strip().lower(),
            concept_family=str(data["concept_family"]).strip(),
            text_control_prompt=str(data["text_control_prompt"]).strip(),
            split=str(data["split"]).strip() or "train",
            image_spec=dict(data["image_spec"] or {}),
            control_spec=dict(data["control_spec"] or {}),
            region=str(data["region"]).strip(),
            notes=str(data["notes"]).strip(),
        )


# ---------------------------------------------------------------------------
# Basic numeric and file helpers
# ---------------------------------------------------------------------------


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def rounded(value: Any, digits: int = 4) -> Any:
    val = fnum(value)
    return round(val, digits) if math.isfinite(val) else ""


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.fmean(vals)) if vals else default


def safe_max(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else default


def auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = [float(s) for y, s in zip(labels, scores) if int(y) == 1 and math.isfinite(float(s))]
    neg = [float(s) for y, s in zip(labels, scores) if int(y) == 0 and math.isfinite(float(s))]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for ps in pos:
        for ns in neg:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / (len(pos) * len(neg))


def dot(xs: Sequence[float], ys: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(xs, ys))


def cosine(xs: Sequence[float], ys: Sequence[float]) -> float:
    nx = math.sqrt(sum(float(x) * float(x) for x in xs))
    ny = math.sqrt(sum(float(y) * float(y) for y in ys))
    return dot(xs, ys) / max(nx * ny, 1e-9)


# ---------------------------------------------------------------------------
# Data loading, manifest checks, schema audit
# ---------------------------------------------------------------------------


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".jsonl", ".json"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def manifest_expected_hash(path: pathlib.Path) -> tuple[str | None, str]:
    notes: list[str] = []
    for manifest_path in (path.parent / "MANIFEST.json", path.parent / LAB_MANIFEST_FILE):
        if not manifest_path.exists():
            notes.append(f"{manifest_path.name} not found")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            notes.append(f"{manifest_path.name} unreadable: {exc}")
            continue
        candidates: list[Any] = []
        if isinstance(manifest, dict):
            candidates.extend([
                manifest.get(path.name),
                manifest.get(str(path)),
                manifest.get("files", {}).get(path.name) if isinstance(manifest.get("files"), dict) else None,
            ])
        for entry in candidates:
            if isinstance(entry, str):
                return entry, f"found string entry in {manifest_path.name}"
            if isinstance(entry, dict):
                for key in ("sha256", "hash", "sha256_hex"):
                    val = entry.get(key)
                    if isinstance(val, str):
                        return val, f"found {key} entry in {manifest_path.name}"
        notes.append(f"no usable sha256 entry in {manifest_path.name}")
    return None, "; ".join(notes)


def balanced_cap(items: Sequence[MultiItem], cap: int) -> list[MultiItem]:
    if cap <= 0 or len(items) <= cap:
        return list(items)
    by_family: dict[str, list[MultiItem]] = defaultdict(list)
    for item in items:
        by_family[item.concept_family].append(item)
    out: list[MultiItem] = []
    cursor = 0
    families = [f for f in FAMILIES if by_family.get(f)] + sorted(set(by_family) - set(FAMILIES))
    while len(out) < cap:
        made_progress = False
        for family in families:
            if cursor < len(by_family[family]):
                out.append(by_family[family][cursor])
                made_progress = True
                if len(out) >= cap:
                    break
        if not made_progress:
            break
        cursor += 1
    return out


def load_items(ctx: bench.RunContext) -> tuple[list[MultiItem], dict[str, Any]]:
    path = data_path(ctx.args)
    if not path.exists():
        raise FileNotFoundError(
            f"Lab 33 data file not found: {path}. Run data/make_multimodal_concept_pairs.py or install the frozen JSONL."
        )
    payloads: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            payloads.append(payload)
    if not payloads:
        raise ValueError(f"{path} contains no JSONL rows")
    items = [MultiItem.from_payload(payload) for payload in payloads]
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    items = balanced_cap(items, cap)
    if int(ctx.args.max_examples or 0) > 0:
        items = balanced_cap(items, int(ctx.args.max_examples))
    actual_sha = file_sha256(path)
    expected_sha, manifest_note = manifest_expected_hash(path)
    info = {
        "data_file": DATA_FILE,
        "data_path": str(path),
        "data_sha256": actual_sha,
        "manifest_expected_sha256": expected_sha,
        "manifest_note": manifest_note,
        "manifest_ok": (actual_sha == expected_sha) if expected_sha else None,
        "n_rows_file": len(payloads),
        "n_rows_selected": len(items),
        "families": dict(Counter(item.concept_family for item in items)),
        "splits": dict(Counter(item.split for item in items)),
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
        "science_ready": False,
        "science_scope": "synthetic connector smoke mode; no real VLM hooks loaded",
        "safety_scope": "benign synthetic shapes, charts, positions, OCR traps, and background controls only",
        "fallback_data": False,
    }
    return items, info


def spec_answer_value(item: MultiItem, spec: Mapping[str, Any]) -> str:
    family = item.concept_family
    if family in {"color", "ocr_control", "background_control"}:
        return str(spec.get("color", "none")).lower()
    if family == "shape":
        return str(spec.get("shape", "none")).lower()
    if family == "count":
        return COUNT_WORDS.get(int(spec.get("count", 1) or 1), str(spec.get("count", "none"))).lower()
    if family == "spatial":
        return str(spec.get("position", "none")).lower()
    if family == "chart":
        value = str(spec.get("chart_value", "none")).lower()
        if value.startswith("left"):
            return "left"
        if value.startswith("right"):
            return "right"
        return value
    return str(spec.get("label", "none")).lower()


def schema_audit(ctx: bench.RunContext, items: Sequence[MultiItem]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for item in items:
        problems: list[str] = []
        if not item.item_id:
            problems.append("blank_item_id")
        if item.item_id in seen:
            problems.append("duplicate_item_id")
        seen.add(item.item_id)
        if item.concept_family not in FAMILIES:
            problems.append(f"unknown_family:{item.concept_family}")
        if item.split not in {"train", "test", "eval", "heldout"}:
            problems.append(f"unknown_split:{item.split}")
        if not isinstance(item.image_spec, dict) or not isinstance(item.control_spec, dict):
            problems.append("spec_not_object")
        if item.target == item.distractor:
            problems.append("target_equals_distractor")
        clean_value = spec_answer_value(item, item.image_spec)
        corrupt_value = spec_answer_value(item, item.control_spec)
        if clean_value != item.target:
            problems.append(f"clean_spec_value={clean_value}_not_target={item.target}")
        if corrupt_value != item.distractor:
            problems.append(f"control_spec_value={corrupt_value}_not_distractor={item.distractor}")
        if pathlib.Path(item.image_path).name == pathlib.Path(item.image_control_path).name:
            problems.append("clean_control_image_paths_match")
        if item.concept_family == "ocr_control" and not str(item.image_spec.get("ocr_text", "")).strip():
            problems.append("ocr_control_without_ocr_text")
        if item.concept_family == "background_control" and str(item.image_spec.get("background", "plain")).lower() in {"plain", "none", ""}:
            problems.append("background_control_without_background")
        rows.append({
            "item_id": item.item_id,
            "concept_family": item.concept_family,
            "split": item.split,
            "target": item.target,
            "distractor": item.distractor,
            "clean_spec_value": clean_value,
            "control_spec_value": corrupt_value,
            "schema_ok": not problems,
            "problems": ";".join(problems),
            "image_spec_json": json.dumps(item.image_spec, sort_keys=True),
            "control_spec_json": json.dumps(item.control_spec, sort_keys=True),
        })
    path = ctx.path("diagnostics", "schema_audit.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Schema and clean/corrupt target-value audit for Lab 33 data rows.")
    summary = {
        "n_schema_rows": len(rows),
        "n_schema_ok": sum(1 for r in rows if r["schema_ok"]),
        "n_schema_failed": sum(1 for r in rows if not r["schema_ok"]),
    }
    if summary["n_schema_ok"] == 0:
        raise RuntimeError("Lab 33 schema audit failed every row.")
    return rows, summary


# ---------------------------------------------------------------------------
# Synthetic state construction
# ---------------------------------------------------------------------------


def feature_vector(name: str) -> list[float]:
    rng = random.Random(stable_int("lab33:" + name))
    vec = [rng.gauss(0.0, 1.0) for _ in range(STATE_DIM)]
    return normalize(vec)


def add_scaled(dst: list[float], name: str, scale: float = 1.0) -> None:
    vec = feature_vector(name)
    for i, v in enumerate(vec):
        dst[i] += scale * float(v)


def normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if norm < 1e-12:
        return [0.0 for _ in vec]
    return [float(v) / norm for v in vec]


def state_from_features(features: Sequence[tuple[str, float]]) -> list[float]:
    vec = [0.0 for _ in range(STATE_DIM)]
    for name, scale in features:
        add_scaled(vec, name, scale)
    return normalize(vec)


def mix_states(parts: Sequence[tuple[Sequence[float], float]]) -> list[float]:
    vec = [0.0 for _ in range(STATE_DIM)]
    for state, weight in parts:
        for i, v in enumerate(state):
            vec[i] += float(weight) * float(v)
    return normalize(vec)


def value_direction(value: str) -> list[float]:
    return feature_vector("value:" + str(value).lower())


def answer_margin(state: Sequence[float], target: str, distractor: str) -> float:
    target_vec = value_direction(target)
    distractor_vec = value_direction(distractor)
    return dot(state, [a - b for a, b in zip(target_vec, distractor_vec)])


def count_word(spec: Mapping[str, Any]) -> str:
    return COUNT_WORDS.get(int(spec.get("count", 1) or 1), str(spec.get("count", 1))).lower()


def visual_features(item: MultiItem, spec: Mapping[str, Any]) -> list[tuple[str, float]]:
    feats: list[tuple[str, float]] = []
    family = item.concept_family
    answer = spec_answer_value(item, spec)
    feats.append((f"family:{family}", 0.35))
    feats.append((f"region:{item.region}", 0.25))
    for key in ("color", "shape", "position"):
        val = str(spec.get(key, "")).lower()
        if val and val not in {"none", "plain"}:
            feats.append((f"{key}:{val}", 0.8 if key == "position" else 0.55))
    count = count_word(spec)
    feats.append((f"count:{count}", 0.55))
    if "chart_value" in spec:
        chart = str(spec.get("chart_value", "")).lower()
        feats.append((f"chart_value:{chart}", 0.9))
        if chart.startswith("left"):
            feats.append(("chart_taller:left", 1.0))
        elif chart.startswith("right"):
            feats.append(("chart_taller:right", 1.0))
    feats.append((f"visual_answer:{family}:{answer}", 1.35))
    feats.append((f"value:{answer}", 1.25))
    return feats


def ocr_features(spec: Mapping[str, Any]) -> list[tuple[str, float]]:
    text = str(spec.get("ocr_text", "") or "").strip().lower()
    if not text:
        return []
    return [(f"ocr:{text}", 1.0), (f"value:{text}", 0.8)]


def background_features(spec: Mapping[str, Any]) -> list[tuple[str, float]]:
    bg = str(spec.get("background", "") or "").lower()
    if not bg or bg in {"plain", "none", "chart", "line"}:
        return []
    feats = [(f"background:{bg}", 1.0)]
    if bg in {"red", "blue", "green", "yellow", "purple", "orange"}:
        feats.append((f"value:{bg}", 0.65))
    return feats


def query_features(item: MultiItem) -> list[tuple[str, float]]:
    family = item.concept_family
    coarse = "color" if family in {"ocr_control", "background_control"} else family
    return [
        (f"question_family:{coarse}", 1.0),
        (f"question_region:{item.region}", 0.4),
        ("instruction:use_image", 0.5),
        ("instruction:ignore_shortcuts", 0.3 if family in {"ocr_control", "background_control"} else 0.1),
    ]


def caption_features(item: MultiItem, spec: Mapping[str, Any]) -> list[tuple[str, float]]:
    value = spec_answer_value(item, spec)
    coarse = "color" if item.concept_family in {"ocr_control", "background_control"} else item.concept_family
    return [(f"caption_family:{coarse}", 0.7), (f"caption_value:{value}", 1.0), (f"value:{value}", 1.15)]


def build_states(items: Sequence[MultiItem]) -> dict[tuple[str, str, str], list[float]]:
    states: dict[tuple[str, str, str], list[float]] = {}
    for item in items:
        for variant, spec in (("clean", item.image_spec), ("corrupt", item.control_spec)):
            visual = state_from_features(visual_features(item, spec))
            ocr = state_from_features(ocr_features(spec))
            background = state_from_features(background_features(spec))
            query = state_from_features(query_features(item))
            caption = state_from_features(caption_features(item, spec))
            vision = mix_states([(visual, 0.82), (ocr, 0.20), (background, 0.16)])
            connector = mix_states([(vision, 0.62), (caption, 0.25), (query, 0.13)])
            language = mix_states([(connector, 0.48), (caption, 0.32), (query, 0.20)])
            states[(item.item_id, variant, "vision_visual")] = visual
            states[(item.item_id, variant, "vision")] = vision
            states[(item.item_id, variant, "connector")] = connector
            states[(item.item_id, variant, "language")] = language
            states[(item.item_id, variant, "caption")] = caption
            states[(item.item_id, variant, "text_query")] = query
            states[(item.item_id, variant, "ocr_channel")] = ocr
            states[(item.item_id, variant, "background_channel")] = background
    return states


def state_vector_audit(ctx: bench.RunContext, states: Mapping[tuple[str, str, str], Sequence[float]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (item_id, variant, state_type), state in sorted(states.items()):
        vals = [float(x) for x in state]
        norm = math.sqrt(sum(v * v for v in vals))
        rows.append({
            "item_id": item_id,
            "variant": variant,
            "state_type": state_type,
            "dim": len(vals),
            "norm": rounded(norm),
            "finite": all(math.isfinite(v) for v in vals),
            "near_unit_or_zero": bool(norm < 1e-9 or abs(norm - 1.0) <= 1e-5),
        })
    path = ctx.path("diagnostics", "state_vector_audit.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Synthetic state vector finite/norm audit.")
    return {
        "n_state_vectors": len(rows),
        "state_vectors_finite": all(r["finite"] for r in rows),
        "state_vectors_norm_ok": all(r["near_unit_or_zero"] for r in rows),
    }


# ---------------------------------------------------------------------------
# Rendering and prompt manifests
# ---------------------------------------------------------------------------


def color_to_hex(name: str) -> str:
    return {
        "red": "#d62728",
        "blue": "#1f77b4",
        "green": "#2ca02c",
        "yellow": "#f1c40f",
        "purple": "#9467bd",
        "orange": "#ff7f0e",
        "black": "#111111",
        "white": "#ffffff",
    }.get(str(name).lower(), "#777777")


def render_spec_image(path: pathlib.Path, spec: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    bg = str(spec.get("background", "plain")).lower()
    face = color_to_hex(bg) if bg in {"red", "green", "blue", "yellow", "purple", "orange"} else "white"
    ax.set_facecolor(face)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if bg == "line":
        ax.plot([0.15, 0.85], [0.5, 0.5], color="#333333", linewidth=2)
    if bg == "chart" or str(spec.get("shape", "")).lower() == "bar":
        chart = str(spec.get("chart_value", "left_taller")).lower()
        left_h, right_h = (0.74, 0.36) if chart.startswith("left") else (0.36, 0.74)
        ax.add_patch(patches.Rectangle((0.24, 0.12), 0.18, left_h, color="#4C78A8"))
        ax.add_patch(patches.Rectangle((0.58, 0.12), 0.18, right_h, color="#F58518"))
        ax.text(0.33, 0.05, "L", ha="center", va="center", fontsize=8)
        ax.text(0.67, 0.05, "R", ha="center", va="center", fontsize=8)
    else:
        count = int(spec.get("count", 1) or 1)
        pos = str(spec.get("position", "center")).lower()
        centers = {
            "center": [(0.5, 0.5)],
            "left": [(0.30, 0.5)],
            "right": [(0.70, 0.5)],
            "above": [(0.5, 0.72)],
            "below": [(0.5, 0.28)],
        }.get(pos, [(0.5, 0.5)])
        if count > 1:
            centers = [(0.30 + 0.20 * (i % 3), 0.66 - 0.24 * (i // 3)) for i in range(count)]
        color = color_to_hex(str(spec.get("color", "black")))
        shape = str(spec.get("shape", "circle")).lower()
        for cx, cy in centers:
            if shape == "square":
                ax.add_patch(patches.Rectangle((cx - 0.14, cy - 0.14), 0.28, 0.28, color=color))
            elif shape == "triangle":
                ax.add_patch(patches.RegularPolygon((cx, cy), 3, radius=0.19, orientation=math.pi / 2, color=color))
            elif shape == "star":
                ax.scatter([cx], [cy], marker="*", s=760, color=color)
            else:
                ax.add_patch(patches.Circle((cx, cy), 0.14, color=color))
    ocr = str(spec.get("ocr_text", "") or "").strip()
    if ocr:
        ax.text(0.5, 0.5, ocr.upper(), ha="center", va="center", fontsize=15, weight="bold", color="white")
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.02, facecolor=ax.get_facecolor())
    plt.close(fig)


def render_images(ctx: bench.RunContext, items: Sequence[MultiItem]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        for variant, rel, spec in (
            ("clean", item.image_path, item.image_spec),
            ("corrupt", item.image_control_path, item.control_spec),
        ):
            out = ctx.path("state", "images", pathlib.Path(rel).name)
            render_spec_image(out, spec)
            rows.append({
                "item_id": item.item_id,
                "variant": variant,
                "rendered_path": str(out.relative_to(ctx.run_dir)),
                "logical_path": rel,
                "exists": out.exists(),
                "size_bytes": out.stat().st_size if out.exists() else 0,
                "spec_json": json.dumps(spec, sort_keys=True),
            })
    path = ctx.path("state", "rendered_images_manifest.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "state", "Rendered synthetic image manifest.")
    diag = ctx.path("diagnostics", "render_validation.csv")
    bench.write_csv_with_context(ctx, diag, rows)
    ctx.register_artifact(diag, "diagnostic", "Rendered image existence and size audit.")
    return rows, {
        "n_rendered_images": len(rows),
        "all_images_rendered": all(bool(r["exists"]) and int(r["size_bytes"]) > 0 for r in rows),
    }


def prompt_manifest(items: Sequence[MultiItem], render_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rendered = {(r["item_id"], r["variant"]): r["rendered_path"] for r in render_rows}
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append({
            "item_id": item.item_id,
            "concept_family": item.concept_family,
            "question": item.question,
            "target": item.target,
            "distractor": item.distractor,
            "split": item.split,
            "region": item.region,
            "clean_image": rendered.get((item.item_id, "clean"), ""),
            "corrupt_image": rendered.get((item.item_id, "corrupt"), ""),
            "text_control_prompt": item.text_control_prompt,
            "clean_value": spec_answer_value(item, item.image_spec),
            "corrupt_value": spec_answer_value(item, item.control_spec),
            "clean_ocr_text": str(item.image_spec.get("ocr_text", "") or ""),
            "corrupt_ocr_text": str(item.control_spec.get("ocr_text", "") or ""),
            "clean_background": str(item.image_spec.get("background", "")),
            "corrupt_background": str(item.control_spec.get("background", "")),
            "image_spec_json": json.dumps(item.image_spec, sort_keys=True),
            "control_spec_json": json.dumps(item.control_spec, sort_keys=True),
            "notes": item.notes,
        })
    return rows


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def modality_probe_report(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in sorted({i.concept_family for i in items}):
        fam_items = [i for i in items if i.concept_family == family]
        for state_type in STATE_TYPES:
            labels: list[int] = []
            scores: list[float] = []
            margins: list[float] = []
            for item in fam_items:
                clean = answer_margin(states[(item.item_id, "clean", state_type)], item.target, item.distractor)
                corrupt = answer_margin(states[(item.item_id, "corrupt", state_type)], item.target, item.distractor)
                labels.extend([1, 0])
                scores.extend([clean, corrupt])
                margins.append(clean - corrupt)
            clean_acc = safe_mean([1.0 if s > 0 else 0.0 for s in scores[::2]], default=0.0)
            corrupt_acc = safe_mean([1.0 if s < 0 else 0.0 for s in scores[1::2]], default=0.0)
            rows.append({
                "concept_family": family,
                "state_type": state_type,
                "n_items": len(fam_items),
                "auc": rounded(auc_binary(labels, scores)),
                "clean_margin_positive_rate": rounded(clean_acc),
                "corrupt_margin_negative_rate": rounded(corrupt_acc),
                "balanced_margin_accuracy": rounded((clean_acc + corrupt_acc) / 2.0),
                "mean_clean_minus_corrupt_margin": rounded(safe_mean(margins, default=0.0)),
                "claim_scope": "synthetic_connector_only",
            })
    return rows


def patch_report(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    patch_rows: list[dict[str, Any]] = []
    specificity_rows: list[dict[str, Any]] = []
    for item in items:
        clean_connector = states[(item.item_id, "clean", "connector")]
        corrupt_connector = states[(item.item_id, "corrupt", "connector")]
        clean_margin = answer_margin(clean_connector, item.target, item.distractor)
        corrupt_margin = answer_margin(corrupt_connector, item.target, item.distractor)
        denom = clean_margin - corrupt_margin
        if abs(denom) < 1e-6:
            denom = 1e-6
        random_state = mix_states([(corrupt_connector, 0.85), (state_from_features([(f"random_patch:{item.item_id}", 1.0)]), 0.15)])
        ocr_mix = mix_states([(corrupt_connector, 0.86), (states[(item.item_id, "clean", "ocr_channel")], 0.14)])
        bg_mix = mix_states([(corrupt_connector, 0.86), (states[(item.item_id, "clean", "background_channel")], 0.14)])
        wrong_mix = mix_states([
            (corrupt_connector, 0.78),
            (states[(item.item_id, "clean", "ocr_channel")], 0.11),
            (states[(item.item_id, "clean", "background_channel")], 0.11),
        ])
        patches = {
            "clean_reference": clean_connector,
            "no_patch_corrupt": corrupt_connector,
            "connector_clean_to_corrupt": clean_connector,
            VISUAL_PATCH_TYPE: mix_states([(corrupt_connector, 0.38), (states[(item.item_id, "clean", "vision_visual")], 0.62)]),
            "language_state_patch": mix_states([(corrupt_connector, 0.45), (states[(item.item_id, "clean", "language")], 0.55)]),
            "caption_patch": mix_states([(corrupt_connector, 0.55), (states[(item.item_id, "clean", "caption")], 0.45)]),
            "ocr_channel_patch": ocr_mix,
            "background_channel_patch": bg_mix,
            "text_query_patch": mix_states([(corrupt_connector, 0.86), (states[(item.item_id, "clean", "text_query")], 0.14)]),
            "wrong_region_or_ocr_patch": wrong_mix,
            "random_patch_control": random_state,
        }
        per_patch_recovery: dict[str, float] = {}
        for patch_name, state in patches.items():
            margin = answer_margin(state, item.target, item.distractor)
            rec = (margin - corrupt_margin) / denom
            per_patch_recovery[patch_name] = rec
            patch_rows.append({
                "item_id": item.item_id,
                "concept_family": item.concept_family,
                "split": item.split,
                "region": item.region,
                "patch_type": patch_name,
                "patch_kind": "target" if patch_name in {"connector_clean_to_corrupt", VISUAL_PATCH_TYPE, "language_state_patch", "caption_patch"} else "control",
                "clean_margin": rounded(clean_margin),
                "corrupt_margin": rounded(corrupt_margin),
                "denominator": rounded(denom),
                "patched_margin": rounded(margin),
                "recovery": rounded(rec),
                "target": item.target,
                "distractor": item.distractor,
            })
        control_floor = safe_max([per_patch_recovery.get(name) for name in CONTROL_PATCH_TYPES])
        visual_rec = per_patch_recovery.get(VISUAL_PATCH_TYPE, float("nan"))
        connector_rec = per_patch_recovery.get("connector_clean_to_corrupt", float("nan"))
        specificity_rows.append({
            "item_id": item.item_id,
            "concept_family": item.concept_family,
            "split": item.split,
            "region": item.region,
            "visual_region_recovery": rounded(visual_rec),
            "connector_recovery": rounded(connector_rec),
            "caption_recovery": rounded(per_patch_recovery.get("caption_patch")),
            "ocr_control_recovery": rounded(per_patch_recovery.get("ocr_channel_patch")),
            "background_control_recovery": rounded(per_patch_recovery.get("background_channel_patch")),
            "text_query_control_recovery": rounded(per_patch_recovery.get("text_query_patch")),
            "wrong_region_or_ocr_recovery": rounded(per_patch_recovery.get("wrong_region_or_ocr_patch")),
            "random_control_recovery": rounded(per_patch_recovery.get("random_patch_control")),
            "control_floor": rounded(control_floor),
            "specificity_gap": rounded(visual_rec - control_floor if math.isfinite(control_floor) else float("nan")),
            "visual_beats_controls": bool(math.isfinite(control_floor) and visual_rec >= control_floor + MIN_SPECIFICITY_GAP),
        })
    return patch_rows, specificity_rows


def direction_from_pairs(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]], state_type: str) -> list[float]:
    terms: list[list[float]] = []
    for item in items:
        clean = states[(item.item_id, "clean", state_type)]
        corrupt = states[(item.item_id, "corrupt", state_type)]
        terms.append([a - b for a, b in zip(clean, corrupt)])
    if not terms:
        return [0.0 for _ in range(STATE_DIM)]
    return normalize([safe_mean([term[j] for term in terms], default=0.0) for j in range(STATE_DIM)])


def auc_for_direction(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]], direction: Sequence[float], state_type: str) -> float:
    labels: list[int] = []
    scores: list[float] = []
    for item in items:
        labels.extend([1, 0])
        scores.append(dot(states[(item.item_id, "clean", state_type)], direction))
        scores.append(dot(states[(item.item_id, "corrupt", state_type)], direction))
    return auc_binary(labels, scores)


def cross_modal_transfer(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in sorted({i.concept_family for i in items}):
        fam_items = [i for i in items if i.concept_family == family]
        train = [i for i in fam_items if i.split == "train"] or fam_items
        image_dir = direction_from_pairs(train, states, "vision_visual")
        caption_dir = direction_from_pairs(train, states, "caption")
        connector_dir = direction_from_pairs(train, states, "connector")
        rows.append({
            "concept_family": family,
            "n_items": len(fam_items),
            "n_train_items": len(train),
            "image_direction_on_caption_auc": rounded(auc_for_direction(fam_items, states, image_dir, "caption")),
            "caption_direction_on_image_auc": rounded(auc_for_direction(fam_items, states, caption_dir, "vision_visual")),
            "image_direction_on_text_query_auc": rounded(auc_for_direction(fam_items, states, image_dir, "text_query")),
            "connector_direction_on_visual_auc": rounded(auc_for_direction(fam_items, states, connector_dir, "vision_visual")),
            "image_caption_direction_cosine": rounded(cosine(image_dir, caption_dir)),
            "image_connector_direction_cosine": rounded(cosine(image_dir, connector_dir)),
            "caption_connector_direction_cosine": rounded(cosine(caption_dir, connector_dir)),
            "claim_scope": "synthetic_feature_basis_only",
        })
    return rows


def leak_audit(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        ocr = str(item.image_spec.get("ocr_text", "") or "").strip().lower()
        background = str(item.image_spec.get("background", "") or "").strip().lower()
        visual_margin = answer_margin(states[(item.item_id, "clean", "vision_visual")], item.target, item.distractor)
        full_margin = answer_margin(states[(item.item_id, "clean", "vision")], item.target, item.distractor)
        connector_margin = answer_margin(states[(item.item_id, "clean", "connector")], item.target, item.distractor)
        ocr_proj = dot(states[(item.item_id, "clean", "ocr_channel")], value_direction(item.distractor)) if ocr else 0.0
        bg_proj = dot(states[(item.item_id, "clean", "background_channel")], value_direction(item.distractor)) if background else 0.0
        ocr_names_distractor = bool(ocr and ocr == item.distractor.lower())
        background_names_distractor = bool(background and background == item.distractor.lower())
        shortcut_risk = bool((ocr_names_distractor or background_names_distractor) and full_margin < visual_margin - 0.02)
        rows.append({
            "item_id": item.item_id,
            "concept_family": item.concept_family,
            "ocr_text": ocr,
            "background": background,
            "ocr_names_target": bool(ocr and ocr == item.target.lower()),
            "ocr_names_distractor": ocr_names_distractor,
            "background_names_target": bool(background and background == item.target.lower()),
            "background_names_distractor": background_names_distractor,
            "visual_only_target_margin": rounded(visual_margin),
            "vision_with_shortcuts_target_margin": rounded(full_margin),
            "connector_target_margin": rounded(connector_margin),
            "shortcut_margin_delta": rounded(full_margin - visual_margin),
            "ocr_distractor_projection": rounded(ocr_proj),
            "background_distractor_projection": rounded(bg_proj),
            "shortcut_risk": shortcut_risk,
            "leak_status": "shortcut_control_triggered" if ocr_names_distractor or background_names_distractor else "no_shortcut_marker",
        })
    triggered = [r for r in rows if r["leak_status"] == "shortcut_control_triggered"]
    risky = [r for r in rows if r["shortcut_risk"]]
    summary = {
        "n_leak_controls_triggered": len(triggered),
        "n_shortcut_risk_rows": len(risky),
        "ocr_or_background_gate": "failed_for_real_vlm_claims" if triggered else "passed",
        "science_ready": False,
    }
    return rows, summary


def text_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def text_leak_audit(items: Sequence[MultiItem]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        question = item.question.lower()
        prompt = item.text_control_prompt.lower()
        toks = text_tokens(question + " " + prompt)
        target = item.target.lower()
        distractor = item.distractor.lower()
        target_in_text = target in toks
        distractor_in_text = distractor in toks
        rows.append({
            "item_id": item.item_id,
            "concept_family": item.concept_family,
            "target": target,
            "distractor": distractor,
            "target_in_question_or_text_control": target_in_text,
            "distractor_in_question_or_text_control": distractor_in_text,
            "text_leak_status": "label_text_leak" if target_in_text or distractor_in_text else "label_blind_text_control",
            "question": item.question,
            "text_control_prompt": item.text_control_prompt,
        })
    n_leaks = sum(1 for r in rows if r["text_leak_status"] == "label_text_leak")
    return rows, {"n_text_label_leaks": n_leaks, "text_leak_gate": "failed" if n_leaks else "passed"}


def alignment_validation(items: Sequence[MultiItem]) -> dict[str, Any]:
    return {
        "mode": "synthetic_connector",
        "connector_token_count": 4,
        "all_items_fixed_connector_tokens": True,
        "image_sizes_fixed_by_renderer": True,
        "region_inventory": sorted({item.region for item in items}),
        "real_vlm_required_before_science_claims": True,
        "real_vlm_required_diagnostics": [
            "image token count per example",
            "image patch grid after resize/crop",
            "region to token index mapping",
            "clean/corrupt image-token compatibility",
            "caption/text-only/OCR/background controls",
            "hook parity for vision, connector, and language modules",
        ],
        "alignment_hazard_note": "Real VLMs may emit variable image token counts. A plot cannot validate region alignment; the run must write explicit token-region mappings.",
        "n_items": len(items),
        "science_ready": False,
    }


# ---------------------------------------------------------------------------
# Evidence matrix, counterexamples, summaries
# ---------------------------------------------------------------------------


def by_family_value(rows: Sequence[Mapping[str, Any]], family: str, *, key: str, filters: Mapping[str, Any] | None = None) -> list[float]:
    vals: list[float] = []
    filters = filters or {}
    for row in rows:
        if row.get("concept_family") != family:
            continue
        if any(row.get(k) != v for k, v in filters.items()):
            continue
        val = fnum(row.get(key))
        if math.isfinite(val):
            vals.append(val)
    return vals


def evidence_matrix(
    data_info: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    specificity_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    leak_rows: Sequence[Mapping[str, Any]],
    text_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    transfer_by_family = {r["concept_family"]: r for r in transfer_rows}
    leak_by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in leak_rows:
        leak_by_family[str(row["concept_family"])].append(row)

    for family in sorted({r["concept_family"] for r in probe_rows}):
        connector_auc = safe_mean(by_family_value(probe_rows, family, key="auc", filters={"state_type": "connector"}))
        visual_auc = safe_mean(by_family_value(probe_rows, family, key="auc", filters={"state_type": "vision_visual"}))
        text_auc = safe_mean(by_family_value(probe_rows, family, key="auc", filters={"state_type": "text_query"}))
        caption_auc = safe_mean(by_family_value(probe_rows, family, key="auc", filters={"state_type": "caption"}))
        visual_patch = safe_mean(by_family_value(specificity_rows, family, key="visual_region_recovery"))
        connector_patch = safe_mean(by_family_value(specificity_rows, family, key="connector_recovery"))
        control_floor = safe_mean(by_family_value(specificity_rows, family, key="control_floor"))
        specificity_gap = safe_mean(by_family_value(specificity_rows, family, key="specificity_gap"))
        leak_rate = safe_mean([1.0 if r.get("leak_status") == "shortcut_control_triggered" else 0.0 for r in leak_by_family.get(family, [])], default=0.0)
        shortcut_risk_rate = safe_mean([1.0 if r.get("shortcut_risk") else 0.0 for r in leak_by_family.get(family, [])], default=0.0)
        transfer = transfer_by_family.get(family, {})
        image_caption_auc = fnum(transfer.get("image_direction_on_caption_auc"))
        image_text_query_auc = fnum(transfer.get("image_direction_on_text_query_auc"))
        if leak_rate > 0:
            posture = "shortcut_control_triggered_real_vlm_claim_blocked"
            claim = "Synthetic shortcut rows intentionally block real-VLM visual claims."
        elif connector_auc >= MIN_AUC_GATE and visual_patch >= MIN_RECOVERY_GATE and specificity_gap >= MIN_SPECIFICITY_GAP:
            posture = "synthetic_visual_specificity_gate_passed_not_real_vlm_evidence"
            claim = "Synthetic visual patch beats shortcut controls for this family; still synthetic only."
        elif connector_auc >= MIN_AUC_GATE and connector_patch >= MIN_RECOVERY_GATE:
            posture = "synthetic_connector_only_node_or_caption_explanation"
            claim = "Connector patch works, but visual specificity is not established."
        else:
            posture = "synthetic_gate_needs_debugging_or_negative"
            claim = "Do not write modality-mechanism language for this family."
        evidence.append({
            "concept_family": family,
            "n_items": data_info.get("families", {}).get(family, ""),
            "visual_auc": rounded(visual_auc),
            "connector_auc": rounded(connector_auc),
            "caption_auc": rounded(caption_auc),
            "text_query_auc": rounded(text_auc),
            "visual_region_recovery": rounded(visual_patch),
            "connector_recovery": rounded(connector_patch),
            "control_floor": rounded(control_floor),
            "specificity_gap": rounded(specificity_gap),
            "image_direction_on_caption_auc": rounded(image_caption_auc),
            "image_direction_on_text_query_auc": rounded(image_text_query_auc),
            "shortcut_leak_rate": rounded(leak_rate),
            "shortcut_risk_rate": rounded(shortcut_risk_rate),
            "text_label_leak_gate": text_summary["text_leak_gate"],
            "science_ready_for_real_vlm": False,
            "claim_posture": posture,
            "smallest_supported_claim": claim,
        })
        for row in [r for r in specificity_rows if r["concept_family"] == family]:
            if fnum(row.get("control_floor")) >= fnum(row.get("visual_region_recovery")) - CONTROL_CLOSE_TOL:
                counterexamples.append({
                    "concept_family": family,
                    "item_id": row["item_id"],
                    "kind": "control_matches_visual_patch",
                    "visual_region_recovery": row.get("visual_region_recovery", ""),
                    "control_floor": row.get("control_floor", ""),
                    "specificity_gap": row.get("specificity_gap", ""),
                    "lesson": "A shortcut, wrong-region, text, or random control is too close to the visual patch.",
                })
        for row in leak_by_family.get(family, []):
            if row.get("leak_status") == "shortcut_control_triggered":
                counterexamples.append({
                    "concept_family": family,
                    "item_id": row["item_id"],
                    "kind": "ocr_or_background_shortcut_present",
                    "visual_region_recovery": "",
                    "control_floor": "",
                    "specificity_gap": "",
                    "lesson": "This row intentionally contains OCR/background shortcut evidence; real-VLM claims must survive it.",
                })
    metrics = {
        "science_ready": False,
        "n_families": len(evidence),
        "n_items": data_info.get("n_rows_selected", 0),
        "mean_connector_auc": rounded(safe_mean([r["connector_auc"] for r in evidence])),
        "mean_visual_auc": rounded(safe_mean([r["visual_auc"] for r in evidence])),
        "mean_text_query_auc": rounded(safe_mean([r["text_query_auc"] for r in evidence])),
        "mean_visual_region_recovery": rounded(safe_mean([r["visual_region_recovery"] for r in evidence])),
        "mean_control_floor": rounded(safe_mean([r["control_floor"] for r in evidence])),
        "mean_specificity_gap": rounded(safe_mean([r["specificity_gap"] for r in evidence])),
        "synthetic_gate_pass_families": sum(1 for r in evidence if str(r["claim_posture"]).startswith("synthetic_visual_specificity")),
        "shortcut_blocked_families": sum(1 for r in evidence if str(r["claim_posture"]).startswith("shortcut_control")),
        "n_counterexamples": len(counterexamples),
        "thresholds": {
            "min_auc_gate": MIN_AUC_GATE,
            "min_recovery_gate": MIN_RECOVERY_GATE,
            "min_specificity_gap": MIN_SPECIFICITY_GAP,
            "control_close_tolerance": CONTROL_CLOSE_TOL,
        },
    }
    return evidence, counterexamples, metrics


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_tables(
    ctx: bench.RunContext,
    prompt_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    specificity_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    leak_rows: Sequence[Mapping[str, Any]],
    text_leak_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> None:
    specs = [
        ("tables/multimodal_prompt_manifest.csv", prompt_rows, "Synthetic multimodal prompts, rendered images, specs, and shortcut metadata."),
        ("tables/modality_probe_report.csv", probe_rows, "Synthetic readout report by family and state type."),
        ("tables/multimodal_patch_report.csv", patch_rows, "Clean/corrupt synthetic patch recovery rows."),
        ("tables/multimodal_specificity_controls.csv", specificity_rows, "Visual patch recovery versus OCR/background/text/wrong/random controls."),
        ("tables/cross_modal_transfer.csv", transfer_rows, "Synthetic image, caption, connector, and text-query direction transfer report."),
        ("tables/ocr_background_leak_audit.csv", leak_rows, "OCR and background shortcut gate."),
        ("tables/text_leak_audit.csv", text_leak_rows, "Question and text-control label leakage audit."),
        ("tables/multimodal_evidence_matrix.csv", evidence_rows, "Family-level evidence matrix and claim posture."),
        ("tables/multimodal_counterexamples.csv", counterexamples, "Counterexamples that block or narrow modality claims."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)
    results = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results, evidence_rows)
    ctx.register_artifact(results, "table", "Alias of tables/multimodal_evidence_matrix.csv for dashboards.")


def write_state(ctx: bench.RunContext, items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]], transfer_rows: Sequence[Mapping[str, Any]]) -> None:
    import torch

    directions: dict[str, Any] = {
        "feature_basis": "deterministic_hash_vectors",
        "state_dim": STATE_DIM,
        "families": sorted({item.concept_family for item in items}),
        "transfer_rows": list(transfer_rows),
    }
    for family in sorted({item.concept_family for item in items}):
        fam_items = [item for item in items if item.concept_family == family and item.split == "train"] or [item for item in items if item.concept_family == family]
        directions[f"{family}:image_direction"] = direction_from_pairs(fam_items, states, "vision_visual")
        directions[f"{family}:caption_direction"] = direction_from_pairs(fam_items, states, "caption")
        directions[f"{family}:connector_direction"] = direction_from_pairs(fam_items, states, "connector")
    path = ctx.path("state", "multimodal_directions.pt")
    torch.save(directions, path)
    ctx.register_artifact(path, "state", "Synthetic multimodal directions and metadata.")
    meta = {
        "lab": LAB_ID,
        "state_dim": STATE_DIM,
        "state_types": STATE_TYPES,
        "basis": "deterministic hash-normalized vectors keyed by feature names",
        "non_claim": "These are synthetic states, not captured VLM activations.",
        "families": sorted({item.concept_family for item in items}),
        "n_items": len(items),
    }
    meta_path = ctx.path("state", "multimodal_state_metadata.json")
    bench.write_json(meta_path, meta)
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for synthetic multimodal states.")


def write_status_files(
    ctx: bench.RunContext,
    data_info: Mapping[str, Any],
    hook_check: Mapping[str, Any],
    lens_check: Mapping[str, Any],
    patch_noop: Mapping[str, Any],
    schema_summary: Mapping[str, Any],
    render_summary: Mapping[str, Any],
    state_summary: Mapping[str, Any],
    alignment: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    safety = {
        "lab": LAB_ID,
        "safe_scope": data_info["safety_scope"],
        "blocked_activities": [
            "face recognition",
            "identity inference",
            "surveillance or private-image analysis",
            "medical-image interpretation",
            "real VLM claims from synthetic connector mode",
        ],
        "generated_harmful_content": False,
        "uses_private_images": False,
        "science_ready": False,
        "note": "Default run uses benign rendered shapes, counts, positions, charts, OCR traps, and background controls.",
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, safety)
    ctx.register_artifact(path, "diagnostic", "Safety and scope status for Lab 33.")

    checks = {
        "hook_parity_ok": bool(hook_check.get("ok")),
        "lens_self_check_ok": bool(lens_check.get("ok", lens_check.get("top1_matches", False) or lens_check.get("near_tie_accepted", False))),
        "patch_noop_ok": bool(patch_noop.get("ok")),
        "schema_ok_rows": schema_summary.get("n_schema_ok", 0),
        "schema_failed_rows": schema_summary.get("n_schema_failed", 0),
        "all_images_rendered": bool(render_summary.get("all_images_rendered")),
        "state_vectors_finite": bool(state_summary.get("state_vectors_finite")),
        "state_vectors_norm_ok": bool(state_summary.get("state_vectors_norm_ok")),
        "alignment_mode": alignment.get("mode"),
        "real_vlm_required_before_science_claims": bool(alignment.get("real_vlm_required_before_science_claims")),
        "evidence_rows": metrics.get("n_families", 0),
        "counterexamples": metrics.get("n_counterexamples", 0),
        "ok_for_synthetic_audit": bool(hook_check.get("ok")) and bool(patch_noop.get("ok")) and bool(render_summary.get("all_images_rendered")) and bool(state_summary.get("state_vectors_finite")),
        "ok_for_real_vlm_claims": False,
    }
    path = ctx.path("diagnostics", "self_check_status.json")
    bench.write_json(path, checks)
    ctx.register_artifact(path, "diagnostic", "Aggregated self-check status for Lab 33.")


def write_method_card(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 33 method card",
        "",
        "This run is a synthetic connector smoke test. It is not a real VLM mechanistic result.",
        "",
        f"- model loaded by bench for microscope checks: `{ctx.model_id or ctx.args.model}`",
        "- image source: deterministic rendered shapes, counts, positions, charts, OCR traps, and background controls",
        "- connector: synthetic feature-basis mixture of visual, caption-like, and question states",
        "- evidence rung: `OBS + DECODE + CAUSAL + AUDIT`, scoped to synthetic mode",
        "- non-claim: no real vision tokens, connector modules, or VLM language states were captured",
        "- required before real-VLM science: token-region alignment, real VLM hooks, image/text/caption controls, and hand review of ambiguous images",
        "",
        f"- science_ready_for_real_vlm: `{metrics['science_ready']}`",
        f"- mean connector AUC: `{metrics['mean_connector_auc']}`",
        f"- mean visual patch recovery: `{metrics['mean_visual_region_recovery']}`",
        f"- mean specificity gap: `{metrics['mean_specificity_gap']}`",
        f"- counterexamples: `{metrics['n_counterexamples']}`",
        "",
        "| family | connector AUC | visual recovery | control floor | specificity gap | posture |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| {row['concept_family']} | {row['connector_auc']} | {row['visual_region_recovery']} | {row['control_floor']} | {row['specificity_gap']} | {row['claim_posture']} |"
        )
    lines += [
        "",
        "Safe sentence: `The synthetic connector suite exercised multimodal readout, patch, shortcut, and alignment audits.`",
        "",
        "Unsafe sentence: `The VLM has a visual concept or sees the object.`",
    ]
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Method card and real-VLM non-claim boundary for Lab 33.")


def write_operationalization_audit(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]], text_summary: Mapping[str, Any], leak_summary: Mapping[str, Any]) -> None:
    passed = sum(1 for row in evidence if str(row.get("claim_posture", "")).startswith("synthetic_visual_specificity"))
    audit_result = "mixed" if passed and counterexamples else "passed" if passed else "failed_or_smoke_only"
    lines = [
        "# Lab 33 operationalization audit",
        "",
        "```yaml",
        "headline_claim: \"image-derived features and text-derived features meet in a shared mechanism\"",
        "cheap_explanation: \"OCR text, background color, caption leakage, text prompt leakage, random disruption, or bad region alignment explains recovery\"",
        "killer_control: \"OCR/background/text-query/wrong-region/random controls plus explicit alignment validation\"",
        f"result: \"{audit_result}\"",
        "claim_allowed: \"synthetic audit package only; no real VLM mechanism claim\"",
        "```",
        "",
        "## What the default run can say",
        "",
        "It can say that the synthetic connector suite produced coherent readout, patch, shortcut, alignment, and counterexample artifacts.",
        "",
        "## What it cannot say",
        "",
        "It cannot say that a real VLM saw an object, aligned a region, represented a visual concept, or used an image-derived feature causally.",
        "",
        "## Shortcut gates",
        "",
        f"- OCR/background gate: `{leak_summary['ocr_or_background_gate']}` with `{leak_summary['n_leak_controls_triggered']}` triggered rows.",
        f"- Text-label leak gate: `{text_summary['text_leak_gate']}` with `{text_summary['n_text_label_leaks']}` label-leak rows.",
        "",
        "## Family verdicts",
        "",
    ]
    lines += [f"- `{row['concept_family']}`: `{row['claim_posture']}`. {row['smallest_supported_claim']}" for row in evidence]
    lines += ["", "## Counterexamples", ""]
    if counterexamples:
        lines += [f"- `{row['concept_family']}` `{row['item_id']}` `{row['kind']}`: {row['lesson']}" for row in counterexamples[:40]]
    else:
        lines.append("- No automatic counterexample crossed the configured synthetic thresholds. Real-VLM claims are still blocked by default.")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Operationalization audit for Lab 33 shortcut and alignment risks.")


def write_real_vlm_checklist(ctx: bench.RunContext) -> None:
    lines = [
        "# Real VLM extension checklist",
        "",
        "Do not set `science_ready=true` until each item below has an artifact.",
        "",
        "| gate | required artifact | why it matters |",
        "|---|---|---|",
        "| model state capture | diagnostics/real_vlm_hook_parity.json | proves vision, connector, and language states are the tensors named by the lab |",
        "| image-token alignment | diagnostics/image_token_alignment.csv | maps rendered regions to token indices after resize/crop |",
        "| clean/corrupt compatibility | diagnostics/image_pair_alignment.csv | ensures patching swaps comparable positions |",
        "| OCR trap audit | tables/ocr_background_leak_audit.csv | blocks rendered-text shortcuts |",
        "| background audit | tables/ocr_background_leak_audit.csv | blocks background-label shortcuts |",
        "| caption/text controls | tables/text_leak_audit.csv | prevents answer-bearing text from masquerading as vision |",
        "| random/wrong-region controls | tables/multimodal_specificity_controls.csv | separates content restoration from disruption |",
        "| hand review | tables/human_image_review_queue.csv | catches ambiguous images and OCR/background cases |",
        "",
        "The default synthetic connector run intentionally fails the real-VLM readiness gate because it has no real VLM states.",
    ]
    path = ctx.path("real_vlm_extension_checklist.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Checklist for upgrading Lab 33 from synthetic connector audit to real VLM science.")


def write_run_summary(ctx: bench.RunContext, data_info: Mapping[str, Any], metrics: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]]) -> None:
    strongest = max(evidence, key=lambda r: fnum(r.get("specificity_gap"), -999.0)) if evidence else None
    smallest = "The synthetic connector audit package ran; no real-VLM claim is licensed."
    if strongest:
        smallest = f"Best synthetic family `{strongest['concept_family']}` had specificity gap {strongest['specificity_gap']}, but the run remains synthetic and science_ready=false."
    main_counter = counterexamples[0]["lesson"] if counterexamples else "No automatic synthetic counterexample crossed thresholds; real-VLM claims remain blocked."
    lines = [
        "# Lab 33 run summary: multimodal mechanistic interpretability",
        "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- families: `{data_info['families']}`",
        f"- science_ready_for_real_vlm: `{metrics['science_ready']}`",
        f"- mean connector AUC: `{metrics['mean_connector_auc']}`",
        f"- mean visual patch recovery: `{metrics['mean_visual_region_recovery']}`",
        f"- mean specificity gap: `{metrics['mean_specificity_gap']}`",
        f"- smallest surviving claim: {smallest}",
        f"- main counterexample: {main_counter}",
        "",
        "## Evidence matrix",
        "",
        "| family | connector AUC | visual recovery | control floor | gap | posture |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| `{row['concept_family']}` | {row['connector_auc']} | {row['visual_region_recovery']} | {row['control_floor']} | {row['specificity_gap']} | {row['claim_posture']} |")
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for the synthetic-mode boundary.",
        "2. `diagnostics/schema_audit.csv` and `diagnostics/alignment_validation.json` for setup gates.",
        "3. `tables/modality_probe_report.csv` for readout semantics.",
        "4. `tables/multimodal_patch_report.csv` and `tables/multimodal_specificity_controls.csv` for patch evidence and controls.",
        "5. `tables/ocr_background_leak_audit.csv` and `tables/text_leak_audit.csv` before writing visual language.",
        "6. `operationalization_audit.md` and `real_vlm_extension_checklist.md` before touching the ledger.",
        "",
        "## Caveats",
        "",
        "- The connector is synthetic and uses deterministic hash-basis feature vectors.",
        "- Caption-like state is a positive control, not evidence that a real VLM used vision.",
        "- OCR/background traps intentionally block broad real-VLM claims.",
        "- Region alignment is declared fixed only in synthetic mode.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Run summary with verdicts, caveats, and reading order.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "AUDIT,CONSTRUCTION",
            "text": f"Lab 33 rendered a synthetic multimodal concept suite and produced modality readout, patch, shortcut, alignment, and evidence-matrix artifacts. The run is not real-VLM evidence because science_ready=false.",
            "artifact": f"runs/{ctx.run_dir.name}/tables/multimodal_evidence_matrix.csv",
            "falsifier": "Schema, rendering, self-check, or shortcut audit fails; or real-VLM hook/alignment artifacts are absent while real-VLM claims are made.",
        }
    ]
    for idx, row in enumerate(evidence, start=2):
        claims.append({
            "id": f"{LAB_ID}-C{idx}",
            "tag": "AUDIT,DECODE,CAUSAL",
            "text": f"In synthetic connector mode, family `{row['concept_family']}` had connector AUC {row['connector_auc']}, visual patch recovery {row['visual_region_recovery']}, and specificity gap {row['specificity_gap']}; posture `{row['claim_posture']}`. This is a synthetic audit result, not a real VLM mechanism claim.",
            "artifact": f"runs/{ctx.run_dir.name}/tables/multimodal_evidence_matrix.csv",
            "falsifier": "Shortcut controls match the visual patch, text prompts leak labels, or real-VLM alignment diagnostics fail.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "multimodal_evidence_dashboard.png", "first_question": "Did readout, patch specificity, and shortcut gates tell the same story?", "non_claim": "Synthetic connector is not a VLM."},
        {"plot": "modality_handoff_atlas.png", "first_question": "Which synthetic states decode each concept family?", "non_claim": "AUC is on synthetic states."},
        {"plot": "image_text_probe_transfer.png", "first_question": "Does image direction transfer to caption state more than text-query state?", "non_claim": "Caption transfer is a positive control, not vision evidence."},
        {"plot": "patch_recovery_by_modality.png", "first_question": "Do visual patches beat OCR/background/text/random controls?", "non_claim": "Patch is not a real VLM intervention."},
        {"plot": "concept_specificity_matrix.png", "first_question": "Which families have positive specificity gaps?", "non_claim": "Bright cells are synthetic audit passes."},
        {"plot": "spatial_region_patch_map.png", "first_question": "Are spatial rows specific or carried by wrong-region controls?", "non_claim": "Region alignment is synthetic."},
        {"plot": "cross_modal_feature_bridge.png", "first_question": "How do synthetic image, caption, and connector directions align?", "non_claim": "Bridge is hash-basis synthetic."},
        {"plot": "ocr_background_shortcut_panel.png", "first_question": "Which shortcut rows block visual claims?", "non_claim": "Shortcut traps are audit gates, not failures to hide."},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 33.")


def write_placeholder(ctx: bench.RunContext, name: str, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=10, wrap=True)
    bench.save_figure(ctx, fig, name, title)


def write_plots(
    ctx: bench.RunContext,
    probe_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    specificity_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    leak_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    if not evidence_rows:
        for name in (
            "multimodal_evidence_dashboard.png",
            "modality_handoff_atlas.png",
            "image_text_probe_transfer.png",
            "patch_recovery_by_modality.png",
            "concept_specificity_matrix.png",
            "spatial_region_patch_map.png",
            "cross_modal_feature_bridge.png",
            "ocr_background_shortcut_panel.png",
        ):
            write_placeholder(ctx, name, name.replace("_", " ").replace(".png", ""), "No evidence rows were produced.")
        return

    families = [str(r["concept_family"]) for r in evidence_rows]
    x = np.arange(len(families))
    connector_auc = [fnum(r.get("connector_auc"), 0.0) for r in evidence_rows]
    visual_patch = [fnum(r.get("visual_region_recovery"), 0.0) for r in evidence_rows]
    control_floor = [fnum(r.get("control_floor"), 0.0) for r in evidence_rows]
    gaps = [fnum(r.get("specificity_gap"), 0.0) for r in evidence_rows]
    leak_rates = [fnum(r.get("shortcut_leak_rate"), 0.0) for r in evidence_rows]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle("Lab 33 multimodal synthetic connector dashboard", fontsize=14, fontweight="bold")
    state_means = {s: safe_mean([r["auc"] for r in probe_rows if r["state_type"] == s], default=0.0) for s in STATE_TYPES}
    axes[0, 0].bar(list(state_means), list(state_means.values()))
    axes[0, 0].axhline(MIN_AUC_GATE, linestyle="--", linewidth=1, label="AUC gate")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Mean readout AUC by synthetic state")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].bar(x - 0.18, visual_patch, 0.36, label="visual patch")
    axes[0, 1].bar(x + 0.18, control_floor, 0.36, label="strongest control")
    axes[0, 1].axhline(MIN_RECOVERY_GATE, linestyle="--", linewidth=1, label="recovery gate")
    axes[0, 1].set_xticks(x, families, rotation=20, ha="right")
    axes[0, 1].set_title("Visual patch versus shortcut controls")
    axes[0, 1].set_ylabel("recovery")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].bar(families, gaps)
    axes[1, 0].axhline(MIN_SPECIFICITY_GAP, linestyle="--", linewidth=1, label="specificity gate")
    axes[1, 0].axhline(0, linewidth=0.8)
    axes[1, 0].set_xticks(x, families, rotation=20, ha="right")
    axes[1, 0].set_ylabel("visual minus control")
    axes[1, 0].set_title("Specificity gap")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(families, leak_rates)
    axes[1, 1].set_xticks(x, families, rotation=20, ha="right")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("OCR/background shortcut trigger rate")
    axes[1, 1].set_ylabel("rate")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "multimodal_evidence_dashboard.png", "Lab 33 synthetic multimodal dashboard.")

    mat = np.zeros((len(STATE_TYPES), len(families)))
    for i, state_type in enumerate(STATE_TYPES):
        for j, family in enumerate(families):
            mat[i, j] = safe_mean([r["auc"] for r in probe_rows if r["concept_family"] == family and r["state_type"] == state_type], default=0.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(families) + 3), 5.0))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(STATE_TYPES)), STATE_TYPES)
    ax.set_xticks(range(len(families)), families, rotation=25, ha="right")
    ax.set_title("Modality handoff atlas")
    for i in range(len(STATE_TYPES)):
        for j in range(len(families)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUC")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "modality_handoff_atlas.png", "State-type by concept-family synthetic readout heatmap.")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x2 = np.arange(len(transfer_rows))
    labels = [r["concept_family"] for r in transfer_rows]
    ax.bar(x2 - 0.25, [fnum(r["image_direction_on_caption_auc"], 0.0) for r in transfer_rows], 0.25, label="image dir on caption")
    ax.bar(x2, [fnum(r["caption_direction_on_image_auc"], 0.0) for r in transfer_rows], 0.25, label="caption dir on image")
    ax.bar(x2 + 0.25, [fnum(r["image_direction_on_text_query_auc"], 0.0) for r in transfer_rows], 0.25, label="image dir on text-query")
    ax.axhline(0.5, linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x2, labels, rotation=25, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Image, caption, and text-query transfer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "image_text_probe_transfer.png", "Synthetic image/caption/text-query transfer AUCs.")

    patch_means: dict[str, list[float]] = defaultdict(list)
    for row in patch_rows:
        patch_means[str(row["patch_type"])].append(fnum(row["recovery"]))
    patch_order = [
        "connector_clean_to_corrupt",
        VISUAL_PATCH_TYPE,
        "language_state_patch",
        "caption_patch",
        "ocr_channel_patch",
        "background_channel_patch",
        "text_query_patch",
        "wrong_region_or_ocr_patch",
        "random_patch_control",
    ]
    patch_order = [p for p in patch_order if p in patch_means]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(patch_order, [safe_mean(patch_means[p], default=0.0) for p in patch_order])
    ax.axhline(0, linewidth=0.8)
    ax.axhline(1, linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(patch_order)), patch_order, rotation=35, ha="right")
    ax.set_ylabel("recovery")
    ax.set_title("Patch recovery by modality/control")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "patch_recovery_by_modality.png", "Patch recovery by synthetic modality or shortcut control.")

    fam_mat = np.zeros((len(families), 4))
    for i, row in enumerate(evidence_rows):
        fam_mat[i, :] = [
            fnum(row.get("connector_auc"), 0.0),
            fnum(row.get("visual_region_recovery"), 0.0),
            fnum(row.get("control_floor"), 0.0),
            fnum(row.get("specificity_gap"), 0.0),
        ]
    fig, ax = plt.subplots(figsize=(7.5, max(4.8, 0.45 * len(families) + 2)))
    im = ax.imshow(fam_mat, aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=1.0)
    ax.set_yticks(range(len(families)), families)
    ax.set_xticks(range(4), ["connector AUC", "visual rec", "control", "gap"], rotation=20, ha="right")
    ax.set_title("Concept specificity matrix")
    for i in range(fam_mat.shape[0]):
        for j in range(fam_mat.shape[1]):
            ax.text(j, i, f"{fam_mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "concept_specificity_matrix.png", "Family-level multimodal specificity matrix.")

    spatial_rows = [r for r in specificity_rows if r.get("concept_family") == "spatial"]
    if spatial_rows:
        names = [str(r["item_id"]).replace("spatial_", "") for r in spatial_rows]
        vals = np.array([
            [fnum(r.get("visual_region_recovery"), 0.0), fnum(r.get("wrong_region_or_ocr_recovery"), 0.0), fnum(r.get("random_control_recovery"), 0.0)]
            for r in spatial_rows
        ])
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        im = ax.imshow(vals.T, aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=1.0)
        ax.set_yticks(range(3), ["visual", "wrong/shortcut", "random"])
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
        ax.set_title("Spatial region patch map")
        fig.colorbar(im, ax=ax, shrink=0.8, label="recovery")
        fig.tight_layout()
        bench.save_figure(ctx, fig, "spatial_region_patch_map.png", "Synthetic spatial row patch recovery map.")
    else:
        write_placeholder(ctx, "spatial_region_patch_map.png", "Spatial region patch map", "No spatial rows selected by this prompt-set cap.")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x2 - 0.18, [fnum(r["image_caption_direction_cosine"], 0.0) for r in transfer_rows], 0.36, label="image-caption cosine")
    ax.bar(x2 + 0.18, [fnum(r["image_connector_direction_cosine"], 0.0) for r in transfer_rows], 0.36, label="image-connector cosine")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x2, labels, rotation=25, ha="right")
    ax.set_ylabel("cosine")
    ax.set_title("Synthetic cross-modal feature bridge")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "cross_modal_feature_bridge.png", "Synthetic direction cosine bridge by family.")

    shortcut_rows = [r for r in leak_rows if r.get("leak_status") == "shortcut_control_triggered"]
    if shortcut_rows:
        names = [str(r["item_id"]).replace("background_trap_", "bg_").replace("ocr_trap_", "ocr_") for r in shortcut_rows]
        deltas = [fnum(r.get("shortcut_margin_delta"), 0.0) for r in shortcut_rows]
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(names) + 3), 4.8))
        ax.bar(range(len(names)), deltas)
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_ylabel("vision full minus visual-only margin")
        ax.set_title("OCR/background shortcut panel")
        fig.tight_layout()
        bench.save_figure(ctx, fig, "ocr_background_shortcut_panel.png", "OCR/background shortcut effect on visual margin.")
    else:
        write_placeholder(ctx, "ocr_background_shortcut_panel.png", "OCR/background shortcut panel", "No OCR or background shortcut rows were selected.")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    items, data_info = load_items(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 33 data manifest, hash, and synthetic-mode scope.")

    schema_rows, schema_summary = schema_audit(ctx, items)
    usable_items = [item for item, row in zip(items, schema_rows) if row["schema_ok"]]
    if not usable_items:
        raise RuntimeError("Lab 33 has no schema-valid rows after audit.")

    # The bench-loaded model is not the object of multimodal science here. It
    # still proves that the standard microscope checks run in the same way as
    # other labs, which keeps the artifact contract uniform.
    hook_check = bench.run_hook_parity_check(ctx, bundle, usable_items[0].text_control_prompt)
    first = bench.run_with_residual_cache(bundle, usable_items[0].text_control_prompt)
    lens_check = bench.run_lens_self_check(ctx, bundle, first)
    patch_noop = bench.run_patch_noop_check(ctx, bundle, usable_items[0].text_control_prompt)

    alignment = alignment_validation(usable_items)
    alignment_path = ctx.path("diagnostics", "alignment_validation.json")
    bench.write_json(alignment_path, alignment)
    ctx.register_artifact(alignment_path, "diagnostic", "Synthetic connector alignment validation and real-VLM requirements.")

    render_rows, render_summary = render_images(ctx, usable_items)
    states = build_states(usable_items)
    state_summary = state_vector_audit(ctx, states)
    prompt_rows = prompt_manifest(usable_items, render_rows)
    probe_rows = modality_probe_report(usable_items, states)
    patch_rows, specificity_rows = patch_report(usable_items, states)
    transfer_rows = cross_modal_transfer(usable_items, states)
    leak_rows, leak_summary = leak_audit(usable_items, states)
    text_leak_rows, text_summary = text_leak_audit(usable_items)
    evidence_rows, counterexamples, metrics = evidence_matrix(
        data_info,
        probe_rows,
        patch_rows,
        specificity_rows,
        transfer_rows,
        leak_rows,
        text_summary,
    )
    metrics = {
        **metrics,
        "data": data_info,
        "schema": schema_summary,
        "render": render_summary,
        "state_vectors": state_summary,
        "alignment": alignment,
        "leak_summary": leak_summary,
        "text_leak_summary": text_summary,
    }

    write_tables(
        ctx,
        prompt_rows,
        probe_rows,
        patch_rows,
        specificity_rows,
        transfer_rows,
        leak_rows,
        text_leak_rows,
        evidence_rows,
        counterexamples,
    )
    write_state(ctx, usable_items, states, transfer_rows)
    write_status_files(ctx, data_info, hook_check, lens_check, patch_noop, schema_summary, render_summary, state_summary, alignment, metrics)

    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 33 metrics, gates, and thresholds.")

    write_method_card(ctx, evidence_rows, metrics)
    write_operationalization_audit(ctx, evidence_rows, counterexamples, text_summary, leak_summary)
    write_real_vlm_checklist(ctx)
    write_run_summary(ctx, data_info, metrics, evidence_rows, counterexamples)
    write_claims(ctx, evidence_rows)
    write_plots(ctx, probe_rows, patch_rows, specificity_rows, transfer_rows, leak_rows, evidence_rows)
    print(f"[lab33] wrote {len(prompt_rows)} prompts, {len(probe_rows)} probe rows, {len(patch_rows)} patch rows, and {len(evidence_rows)} evidence rows")

# ---------------------------------------------------------------------------
# Second-pass visualization/data-quality overrides
# ---------------------------------------------------------------------------
#
# This block deliberately sits after the first-pass implementation. Python name
# binding means these definitions replace the earlier writer/plot/run functions
# without disturbing the proven synthetic connector measurement code above.

PATCH_DOSE_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
MAX_FAILURE_SPECIMENS = 40


def row_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "passed", "pass"}


def safe_stderr(values: Sequence[Any]) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) <= 1:
        return float("nan")
    return float(statistics.stdev(vals) / math.sqrt(len(vals)))


def safe_min(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return min(vals) if vals else default


def item_uid(item: MultiItem) -> str:
    return row_id("mm_item", item.item_id, item.concept_family, item.split)


def patch_metadata(patch_type: str) -> tuple[str, str, str]:
    table = {
        "clean_reference": ("reference", "connector", "none"),
        "no_patch_corrupt": ("baseline", "connector", "none"),
        "connector_clean_to_corrupt": ("target", "connector", "none"),
        VISUAL_PATCH_TYPE: ("target", "vision_visual", "visual"),
        "language_state_patch": ("target", "language", "language"),
        "caption_patch": ("positive_control", "caption", "caption"),
        "ocr_channel_patch": ("shortcut_control", "ocr_channel", "ocr"),
        "background_channel_patch": ("shortcut_control", "background_channel", "background"),
        "text_query_patch": ("text_control", "text_query", "text"),
        "wrong_region_or_ocr_patch": ("wrong_site_control", "ocr+background", "wrong_region_or_shortcut"),
        "random_patch_control": ("random_control", "random_hash_source", "random"),
    }
    return table.get(str(patch_type), ("other", "", ""))


_ORIGINAL_LOAD_ITEMS = load_items


def builtin_smoke_payloads() -> list[dict[str, Any]]:
    """Tiny built-in Tier A fallback so artifact plumbing is still testable.

    The frozen JSONL is the science source. This fallback is intentionally
    marked smoke-only and low-row-count in the data manifest.
    """
    def row(item_id: str, family: str, target: str, distractor: str, spec: dict[str, Any], control: dict[str, Any], region: str = "object", split: str = "train") -> dict[str, Any]:
        return {
            "item_id": item_id,
            "image_path": f"generated/{item_id}_clean.png",
            "image_control_path": f"generated/{item_id}_corrupt.png",
            "question": "What is the answer shown by the rendered image?",
            "target": target,
            "distractor": distractor,
            "concept_family": family,
            "text_control_prompt": "Answer the visual question using only the rendered image.",
            "split": split,
            "image_spec": spec,
            "control_spec": control,
            "region": region,
            "notes": "built-in Tier A smoke fallback row",
        }
    return [
        row("smoke_color_red_blue", "color", "red", "blue", {"color": "red", "shape": "circle", "background": "plain"}, {"color": "blue", "shape": "circle", "background": "plain"}),
        row("smoke_shape_square_circle", "shape", "square", "circle", {"color": "green", "shape": "square", "background": "plain"}, {"color": "green", "shape": "circle", "background": "plain"}),
        row("smoke_count_two_three", "count", "two", "three", {"color": "orange", "shape": "circle", "count": 2, "background": "plain"}, {"color": "orange", "shape": "circle", "count": 3, "background": "plain"}),
        row("smoke_spatial_left_right", "spatial", "left", "right", {"color": "purple", "shape": "circle", "position": "left", "background": "plain"}, {"color": "purple", "shape": "circle", "position": "right", "background": "plain"}, region="left"),
        row("smoke_chart_left_right", "chart", "left", "right", {"shape": "bar", "chart_value": "left_taller", "background": "chart"}, {"shape": "bar", "chart_value": "right_taller", "background": "chart"}, region="bars"),
        row("smoke_ocr_trap_red_blue", "ocr_control", "red", "blue", {"color": "red", "shape": "circle", "ocr_text": "blue", "background": "plain"}, {"color": "blue", "shape": "circle", "ocr_text": "blue", "background": "plain"}),
        row("smoke_background_trap_red_blue", "background_control", "red", "blue", {"color": "red", "shape": "circle", "background": "blue"}, {"color": "blue", "shape": "circle", "background": "blue"}),
    ]


def load_items(ctx: bench.RunContext) -> tuple[list[MultiItem], dict[str, Any]]:  # type: ignore[override]
    path = data_path(ctx.args)
    if path.exists():
        return _ORIGINAL_LOAD_ITEMS(ctx)
    if str(getattr(ctx.args, "tier", "")).lower() != "a":
        raise FileNotFoundError(
            f"Lab 33 data file not found: {path}. Run data/make_multimodal_concept_pairs.py or install the frozen JSONL."
        )
    print("[lab33] frozen multimodal JSONL missing; using built-in Tier A fallback. This is plumbing only.")
    payloads = builtin_smoke_payloads()
    items = [MultiItem.from_payload(payload) for payload in payloads]
    if int(ctx.args.max_examples or 0) > 0:
        items = balanced_cap(items, int(ctx.args.max_examples))
    digest = hashlib.sha256("\n".join(item.item_id for item in items).encode("utf-8")).hexdigest()
    info = {
        "data_file": DATA_FILE,
        "data_path": str(path),
        "data_source": "builtin_tier_a_smoke_fallback",
        "data_sha256": digest,
        "manifest_expected_sha256": None,
        "manifest_note": f"{path} not found; built-in smoke fallback used",
        "manifest_ok": False,
        "n_rows_file": len(payloads),
        "n_rows_selected": len(items),
        "families": dict(Counter(item.concept_family for item in items)),
        "splits": dict(Counter(item.split for item in items)),
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
        "science_ready": False,
        "science_scope": "synthetic connector smoke mode; no real VLM hooks loaded",
        "safety_scope": "benign synthetic shapes, charts, positions, OCR traps, and background controls only",
        "fallback_data": True,
    }
    return items, info


def state_score_long(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        for variant in ("clean", "corrupt"):
            for state_type in (*STATE_TYPES, "ocr_channel", "background_channel"):
                state = states[(item.item_id, variant, state_type)]
                target_projection = dot(state, value_direction(item.target))
                distractor_projection = dot(state, value_direction(item.distractor))
                rows.append({
                    "score_row_id": row_id("score", item.item_id, variant, state_type),
                    "item_uid": item_uid(item),
                    "item_id": item.item_id,
                    "concept_family": item.concept_family,
                    "split": item.split,
                    "region": item.region,
                    "variant": variant,
                    "state_type": state_type,
                    "target": item.target,
                    "distractor": item.distractor,
                    "target_projection": rounded(target_projection),
                    "distractor_projection": rounded(distractor_projection),
                    "answer_margin": rounded(target_projection - distractor_projection),
                    "claim_scope": "synthetic_feature_basis_only",
                })
    return rows


def interpolate_state(base: Sequence[float], source: Sequence[float], dose: float) -> list[float]:
    return normalize([(1.0 - float(dose)) * float(a) + float(dose) * float(b) for a, b in zip(base, source)])


def patch_dose_response(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        clean_connector = states[(item.item_id, "clean", "connector")]
        corrupt_connector = states[(item.item_id, "corrupt", "connector")]
        clean_margin = answer_margin(clean_connector, item.target, item.distractor)
        corrupt_margin = answer_margin(corrupt_connector, item.target, item.distractor)
        denominator = clean_margin - corrupt_margin
        if abs(denominator) < 1e-6:
            denominator = 1e-6
        random_source = state_from_features([(f"dose_random_source:{item.item_id}", 1.0)])
        source_specs = [
            (VISUAL_PATCH_TYPE, "vision_visual", states[(item.item_id, "clean", "vision_visual")], "target_visual"),
            ("language_state_patch", "language", states[(item.item_id, "clean", "language")], "target_language"),
            ("caption_patch", "caption", states[(item.item_id, "clean", "caption")], "positive_caption_control"),
            ("ocr_channel_patch", "ocr_channel", states[(item.item_id, "clean", "ocr_channel")], "shortcut_control"),
            ("background_channel_patch", "background_channel", states[(item.item_id, "clean", "background_channel")], "shortcut_control"),
            ("text_query_patch", "text_query", states[(item.item_id, "clean", "text_query")], "text_control"),
            ("random_patch_control", "random_hash_source", random_source, "random_control"),
        ]
        for patch_type, source_state_type, source, control_role in source_specs:
            for dose in PATCH_DOSE_GRID:
                state = interpolate_state(corrupt_connector, source, dose)
                patched_margin = answer_margin(state, item.target, item.distractor)
                recovery_value = (patched_margin - corrupt_margin) / denominator
                rows.append({
                    "dose_response_id": row_id("dose", item.item_id, patch_type, dose),
                    "item_uid": item_uid(item),
                    "item_id": item.item_id,
                    "concept_family": item.concept_family,
                    "split": item.split,
                    "region": item.region,
                    "patch_type": patch_type,
                    "source_state_type": source_state_type,
                    "control_role": control_role,
                    "dose": dose,
                    "clean_margin": rounded(clean_margin),
                    "corrupt_margin": rounded(corrupt_margin),
                    "denominator": rounded(denominator),
                    "patched_margin": rounded(patched_margin),
                    "recovery": rounded(recovery_value),
                    "target": item.target,
                    "distractor": item.distractor,
                    "claim_scope": "synthetic_dose_sweep_only",
                })
    return rows


def patch_recovery_summary(patch_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in patch_rows:
        grouped[str(row.get("patch_type", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for patch_type, members in sorted(grouped.items()):
        vals = [fnum(r.get("recovery")) for r in members]
        role, source, control = patch_metadata(patch_type)
        rows.append({
            "patch_type": patch_type,
            "method_role": role,
            "source_state_type": source,
            "control_family": control,
            "n_rows": len(members),
            "mean_recovery": rounded(safe_mean(vals)),
            "stderr_recovery": rounded(safe_stderr(vals)),
            "min_recovery": rounded(safe_min(vals)),
            "max_recovery": rounded(safe_max(vals)),
            "n_negative_recovery": sum(1 for v in vals if math.isfinite(v) and v < 0),
            "n_above_recovery_gate": sum(1 for v in vals if math.isfinite(v) and v >= MIN_RECOVERY_GATE),
        })
    return rows


def concept_family_summary(
    evidence_rows: Sequence[Mapping[str, Any]],
    specificity_rows: Sequence[Mapping[str, Any]],
    leak_rows: Sequence[Mapping[str, Any]],
    text_leak_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    spec_by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    leak_by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    text_by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in specificity_rows:
        spec_by_family[str(row.get("concept_family", ""))].append(row)
    for row in leak_rows:
        leak_by_family[str(row.get("concept_family", ""))].append(row)
    for row in text_leak_rows:
        text_by_family[str(row.get("concept_family", ""))].append(row)
    out: list[dict[str, Any]] = []
    for row in evidence_rows:
        family = str(row.get("concept_family", ""))
        spec = spec_by_family.get(family, [])
        leaks = leak_by_family.get(family, [])
        text = text_by_family.get(family, [])
        out.append({
            "concept_family": family,
            "n_items": row.get("n_items", ""),
            "mean_visual_recovery": row.get("visual_region_recovery", ""),
            "mean_control_floor": row.get("control_floor", ""),
            "mean_specificity_gap": row.get("specificity_gap", ""),
            "n_control_close_to_visual": sum(1 for r in spec if boolish(r.get("control_close_to_visual"))),
            "n_visual_beats_controls": sum(1 for r in spec if boolish(r.get("visual_beats_controls"))),
            "n_shortcut_triggered": sum(1 for r in leaks if r.get("leak_status") == "shortcut_control_triggered"),
            "n_text_label_leaks": sum(1 for r in text if r.get("text_leak_status") == "label_text_leak"),
            "claim_posture": row.get("claim_posture", ""),
        })
    return out


def augment_rows(
    prompt_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    specificity_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    leak_rows: Sequence[Mapping[str, Any]],
    text_leak_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> None:
    state_roles = {
        "vision_visual": "target_visual_only",
        "vision": "target_visual_plus_shortcuts",
        "connector": "target_connector_mixture",
        "language": "downstream_language_proxy",
        "caption": "positive_caption_control",
        "text_query": "negative_text_query_control",
    }
    for row in prompt_rows:
        if isinstance(row, dict):
            row.setdefault("prompt_row_id", row_id("prompt", row.get("item_id")))
            row.setdefault("item_uid", row_id("mm_item", row.get("item_id"), row.get("concept_family"), row.get("split")))
    for row in probe_rows:
        if isinstance(row, dict):
            row.setdefault("probe_row_id", row_id("probe", row.get("concept_family"), row.get("state_type")))
            row.setdefault("method_role", state_roles.get(str(row.get("state_type")), "unknown"))
            row.setdefault("is_control_state", str(row.get("state_type")) in {"caption", "text_query"})
            auc = fnum(row.get("auc"))
            row.setdefault("auc_minus_chance", rounded(auc - 0.5 if math.isfinite(auc) else float("nan")))
    for row in patch_rows:
        if isinstance(row, dict):
            role, source, control = patch_metadata(str(row.get("patch_type", "")))
            row.setdefault("patch_row_id", row_id("patch", row.get("item_id"), row.get("patch_type")))
            row.setdefault("intervention_id", row_id("intervention", row.get("patch_type"), source, control))
            row.setdefault("method_role", role)
            row.setdefault("source_state_type", source)
            row.setdefault("control_family", control)
    for row in specificity_rows:
        if not isinstance(row, dict):
            continue
        candidates = [
            ("ocr_channel_patch", fnum(row.get("ocr_control_recovery"))),
            ("background_channel_patch", fnum(row.get("background_control_recovery"))),
            ("text_query_patch", fnum(row.get("text_query_control_recovery"))),
            ("wrong_region_or_ocr_patch", fnum(row.get("wrong_region_or_ocr_recovery"))),
            ("random_patch_control", fnum(row.get("random_control_recovery"))),
        ]
        finite = [(name, val) for name, val in candidates if math.isfinite(val)]
        strongest_name, strongest_val = max(finite, key=lambda kv: kv[1]) if finite else ("", float("nan"))
        visual = fnum(row.get("visual_region_recovery"))
        row.setdefault("specificity_row_id", row_id("specificity", row.get("item_id")))
        row["strongest_control_type"] = strongest_name
        row["strongest_control_recovery"] = rounded(strongest_val)
        row["control_close_to_visual"] = bool(math.isfinite(visual) and math.isfinite(strongest_val) and strongest_val >= visual - CONTROL_CLOSE_TOL)
        row["negative_specificity_gap"] = bool(fnum(row.get("specificity_gap")) < 0)
    for row in transfer_rows:
        if isinstance(row, dict):
            row.setdefault("transfer_row_id", row_id("transfer", row.get("concept_family")))
            a = fnum(row.get("image_direction_on_caption_auc"))
            b = fnum(row.get("image_direction_on_text_query_auc"))
            row.setdefault("caption_minus_text_query_auc_gap", rounded(a - b if math.isfinite(a) and math.isfinite(b) else float("nan")))
    for row in leak_rows:
        if isinstance(row, dict):
            row.setdefault("leak_row_id", row_id("leak", row.get("item_id")))
            flags = []
            if boolish(row.get("ocr_names_distractor")):
                flags.append("ocr_names_distractor")
            if boolish(row.get("background_names_distractor")):
                flags.append("background_names_distractor")
            if boolish(row.get("shortcut_risk")):
                flags.append("shortcut_reduces_visual_margin")
            row.setdefault("risk_flags", ";".join(flags))
    for row in text_leak_rows:
        if isinstance(row, dict):
            row.setdefault("text_leak_row_id", row_id("textleak", row.get("item_id")))
            row.setdefault("question_contains_label", boolish(row.get("target_in_question_or_text_control")) or boolish(row.get("distractor_in_question_or_text_control")))
    for row in evidence_rows:
        if isinstance(row, dict):
            row.setdefault("evidence_row_id", row_id("evidence", row.get("concept_family")))
    for i, row in enumerate(counterexamples):
        if isinstance(row, dict):
            row.setdefault("counterexample_id", row_id("counterexample", row.get("concept_family"), row.get("item_id"), row.get("kind"), i))


def failure_specimens(
    specificity_rows: Sequence[Mapping[str, Any]],
    leak_rows: Sequence[Mapping[str, Any]],
    text_leak_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(row: Mapping[str, Any], kind: str, severity: str, lesson: str, source_table: str) -> None:
        item_id = str(row.get("item_id", ""))
        key = (item_id, kind)
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "specimen_id": row_id("specimen", item_id, kind),
            "item_id": item_id,
            "concept_family": row.get("concept_family", ""),
            "kind": kind,
            "severity": severity,
            "source_table": source_table,
            "visual_region_recovery": row.get("visual_region_recovery", ""),
            "control_floor": row.get("control_floor", row.get("strongest_control_recovery", "")),
            "specificity_gap": row.get("specificity_gap", ""),
            "leak_status": row.get("leak_status", ""),
            "lesson": lesson,
            "student_action": "Inspect this specimen before using broad visual-mechanism language.",
        })

    for row in counterexamples:
        add(row, str(row.get("kind", "counterexample")), "warning", str(row.get("lesson", "Counterexample narrows the claim.")), "tables/multimodal_counterexamples.csv")
    for row in specificity_rows:
        visual = fnum(row.get("visual_region_recovery"))
        gap = fnum(row.get("specificity_gap"))
        control = fnum(row.get("control_floor"))
        if math.isfinite(gap) and gap < MIN_SPECIFICITY_GAP:
            add(row, "weak_or_negative_specificity_gap", "warning", "Visual patch did not beat the strongest control by the configured gap.", "tables/multimodal_specificity_controls.csv")
        elif math.isfinite(visual) and visual < MIN_RECOVERY_GATE:
            add(row, "weak_visual_recovery", "info", "Visual patch recovery stayed below the recovery gate.", "tables/multimodal_specificity_controls.csv")
        elif math.isfinite(control) and math.isfinite(visual) and control >= visual - CONTROL_CLOSE_TOL:
            add(row, "control_close_to_visual", "warning", "A control is close enough to the visual patch to block overbroad language.", "tables/multimodal_specificity_controls.csv")
    for row in leak_rows:
        if row.get("leak_status") == "shortcut_control_triggered":
            add(row, "ocr_background_shortcut", "blocker", "OCR or background shortcut evidence is intentionally present.", "tables/ocr_background_leak_audit.csv")
    for row in text_leak_rows:
        if row.get("text_leak_status") == "label_text_leak":
            add(row, "text_label_leak", "blocker", "Question or text-control prompt contains target or distractor label text.", "tables/text_leak_audit.csv")

    def rank(row: Mapping[str, Any]) -> tuple[int, float]:
        severity = {"blocker": 0, "warning": 1, "info": 2}.get(str(row.get("severity")), 3)
        gap = fnum(row.get("specificity_gap"), 999.0)
        return severity, gap

    return sorted(rows, key=rank)[:MAX_FAILURE_SPECIMENS]


def warning_summary_rows(
    data_info: Mapping[str, Any],
    schema_summary: Mapping[str, Any],
    render_summary: Mapping[str, Any],
    state_summary: Mapping[str, Any],
    alignment: Mapping[str, Any],
    metrics: Mapping[str, Any],
    leak_summary: Mapping[str, Any],
    text_summary: Mapping[str, Any],
    specificity_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(code: str, severity: str, n_rows: int, artifact: str, message: str) -> None:
        rows.append({
            "warning_id": row_id("warn", code, n_rows, artifact),
            "code": code,
            "severity": severity,
            "n_rows": int(n_rows),
            "artifact": artifact,
            "message": message,
        })

    add("synthetic_mode_blocks_real_vlm_claims", "blocker", 1, "diagnostics/alignment_validation.json", "Default Lab 33 uses synthetic connector states; science_ready for real-VLM claims is false.")
    if data_info.get("fallback_data"):
        add("fallback_data_used", "warning", 1, "diagnostics/data_manifest.json", "Builtin Tier A fallback data was used; this is plumbing only.")
    if data_info.get("manifest_ok") is False:
        add("data_manifest_hash_mismatch", "warning", 1, "diagnostics/data_manifest.json", "Frozen data hash did not match the manifest entry.")
    elif data_info.get("manifest_ok") is None:
        add("data_manifest_hash_missing", "info", 1, "diagnostics/data_manifest.json", "No usable manifest hash was found for the selected JSONL.")
    failed_schema = int(schema_summary.get("n_schema_failed", 0) or 0)
    if failed_schema:
        add("schema_rows_failed", "warning", failed_schema, "diagnostics/schema_audit.csv", "Some rows failed schema or clean/corrupt answer-value checks.")
    if not render_summary.get("all_images_rendered"):
        add("render_validation_failed", "blocker", 1, "diagnostics/render_validation.csv", "At least one rendered image is missing or empty.")
    if not state_summary.get("state_vectors_finite"):
        add("nonfinite_state_vectors", "blocker", 1, "diagnostics/state_vector_audit.csv", "At least one synthetic state vector has a non-finite value.")
    if alignment.get("real_vlm_required_before_science_claims"):
        add("real_vlm_alignment_missing", "blocker", 1, "real_vlm_extension_checklist.md", "A real VLM extension must prove image-token and region alignment before changing the claim posture.")
    shortcut_count = int(leak_summary.get("n_leak_controls_triggered", 0) or 0)
    if shortcut_count:
        add("shortcut_controls_triggered", "blocker", shortcut_count, "tables/ocr_background_leak_audit.csv", "OCR/background shortcuts triggered and must remain visible.")
    text_leaks = int(text_summary.get("n_text_label_leaks", 0) or 0)
    if text_leaks:
        add("text_label_leaks", "blocker", text_leaks, "tables/text_leak_audit.csv", "Question or text-control labels leaked target/distractor text.")
    close_controls = sum(1 for r in specificity_rows if fnum(r.get("control_floor")) >= fnum(r.get("visual_region_recovery")) - CONTROL_CLOSE_TOL)
    if close_controls:
        add("controls_close_to_visual_patch", "warning", close_controls, "tables/multimodal_specificity_controls.csv", "At least one strongest control is too close to visual patch recovery.")
    low_n = sum(1 for n in (data_info.get("families", {}) or {}).values() if int(n) < 2)
    if low_n:
        add("low_family_sample_count", "info", low_n, "diagnostics/data_manifest.json", "Some concept families have fewer than two selected rows; Tier A plots should be read as plumbing evidence.")
    if failure_rows:
        add("failure_specimens_available", "info", len(failure_rows), "cards/failure_specimens.md", "Specimen-level failures or caveats were exported for reading before claims.")
    if metrics.get("n_dose_response_rows", 0) == 0:
        add("dose_response_empty", "warning", 1, "tables/patch_dose_response.csv", "Dose-response source table is empty.")
    return rows


def write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def write_run_config_snapshot(ctx: bench.RunContext, data_info: Mapping[str, Any], alignment: Mapping[str, Any]) -> None:
    snapshot = {
        "lab": LAB_ID,
        "run_name": ctx.run_dir.name,
        "model_id": ctx.model_id or ctx.args.model,
        "model_revision": ctx.model_revision,
        "tier": ctx.args.tier,
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
        "seed": ctx.args.seed,
        "dtype": ctx.args.dtype,
        "quantization": ctx.args.quantization,
        "decoding": "none; synthetic connector plus bench microscope checks only",
        "synthetic_connector": {
            "state_dim": STATE_DIM,
            "state_types": list(STATE_TYPES),
            "dose_grid": list(PATCH_DOSE_GRID),
            "control_patch_types": list(CONTROL_PATCH_TYPES),
            "visual_patch_type": VISUAL_PATCH_TYPE,
        },
        "thresholds": {
            "min_auc_gate": MIN_AUC_GATE,
            "min_recovery_gate": MIN_RECOVERY_GATE,
            "min_specificity_gap": MIN_SPECIFICITY_GAP,
            "control_close_tolerance": CONTROL_CLOSE_TOL,
        },
        "data": dict(data_info),
        "alignment": dict(alignment),
        "science_ready_for_real_vlm": False,
    }
    path = ctx.path("diagnostics", "run_config_snapshot.json")
    bench.write_json(path, snapshot)
    ctx.register_artifact(path, "diagnostic", "Run configuration snapshot for Lab 33 plotting and data artifacts.")


def write_warning_artifacts(ctx: bench.RunContext, warning_rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path = ctx.path("diagnostics", "warning_summary.csv")
    bench.write_csv_with_context(ctx, csv_path, warning_rows)
    ctx.register_artifact(csv_path, "diagnostic", "Warnings and blockers emitted by the Lab 33 audit.")
    json_path = ctx.path("diagnostics", "warning_summary.json")
    bench.write_json(json_path, {"n_warnings": len(warning_rows), "warnings": list(warning_rows)})
    ctx.register_artifact(json_path, "diagnostic", "Machine-readable warning and blocker summary for Lab 33.")


def write_failure_specimens(ctx: bench.RunContext, failure_rows: Sequence[Mapping[str, Any]]) -> None:  # type: ignore[override]
    csv_path = ctx.path("tables", "failure_specimens.csv")
    bench.write_csv_with_context(ctx, csv_path, failure_rows)
    ctx.register_artifact(csv_path, "table", "Specimen-level rows that block or narrow visual-mechanism claims.")
    jsonl_path = ctx.path("tables", "failure_specimens.jsonl")
    write_jsonl(jsonl_path, failure_rows)
    ctx.register_artifact(jsonl_path, "table", "JSONL failure specimens for downstream review or notebooks.")
    lines = [
        "# Lab 33 failure specimens",
        "",
        "These specimens are not cleanup chores. They are the audit teeth: controls that match the visual patch, shortcut rows, text-label leaks, or weak specificity rows.",
        "",
    ]
    if not failure_rows:
        lines.append("No automatic failure specimen crossed the configured synthetic thresholds. The real-VLM non-claim still holds.")
    else:
        for row in failure_rows:
            lines += [
                f"## `{row.get('item_id', '')}` - {row.get('kind', '')}",
                "",
                f"- family: `{row.get('concept_family', '')}`",
                f"- severity: `{row.get('severity', '')}`",
                f"- source table: `{row.get('source_table', '')}`",
                f"- visual recovery: `{row.get('visual_region_recovery', '')}`; control floor: `{row.get('control_floor', '')}`; gap: `{row.get('specificity_gap', '')}`",
                f"- lesson: {row.get('lesson', '')}",
                f"- student action: {row.get('student_action', '')}",
                "",
            ]
    card_path = ctx.path("cards", "failure_specimens.md")
    bench.write_text(card_path, "\n".join(lines).rstrip() + "\n")
    ctx.register_artifact(card_path, "card", "Markdown reading cards for Lab 33 failure specimens.")


def write_upgrade_tables(
    ctx: bench.RunContext,
    score_rows: Sequence[Mapping[str, Any]],
    dose_rows: Sequence[Mapping[str, Any]],
    patch_summary_rows: Sequence[Mapping[str, Any]],
    family_summary_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    warning_rows: Sequence[Mapping[str, Any]],
) -> None:
    specs = [
        ("tables/state_score_long.csv", score_rows, "Per-item clean/corrupt synthetic state projections and answer margins."),
        ("tables/patch_dose_response.csv", dose_rows, "Synthetic patch dose-response rows for visual/caption/shortcut/random sources."),
        ("tables/multimodal_dose_response.csv", dose_rows, "Alias of patch_dose_response.csv for plot readers."),
        ("tables/patch_recovery_summary.csv", patch_summary_rows, "Aggregate recovery summary by patch type with raw-count flags."),
        ("tables/patch_type_summary.csv", patch_summary_rows, "Alias of patch_recovery_summary.csv."),
        ("tables/concept_family_summary.csv", family_summary_rows, "Family-level specificity, shortcut, text-leak, and posture summary."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)
    write_failure_specimens(ctx, failure_rows)
    write_warning_artifacts(ctx, warning_rows)


def write_plot_guide(ctx: bench.RunContext) -> None:  # type: ignore[override]
    rows = [
        {"plot": "multimodal_evidence_dashboard.png", "first_question": "Did readout, patch specificity, and shortcut gates tell the same story?", "source_table": "tables/figure_sources/multimodal_evidence_dashboard.csv", "non_claim": "Synthetic connector is not a VLM."},
        {"plot": "target_vs_control.png", "first_question": "Does each visual patch beat its strongest same-row control?", "source_table": "tables/figure_sources/target_vs_control.csv", "non_claim": "A paired visual-control gap is still synthetic-only."},
        {"plot": "dose_response.png", "first_question": "Does recovery rise with visual-patch dose, or only at one convenient scale?", "source_table": "tables/figure_sources/dose_response.csv", "non_claim": "Dose is synthetic vector interpolation, not a real VLM intervention strength."},
        {"plot": "modality_handoff_atlas.png", "first_question": "Which synthetic states decode each concept family?", "source_table": "tables/figure_sources/modality_handoff_atlas.csv", "non_claim": "AUC is on synthetic states."},
        {"plot": "image_text_probe_transfer.png", "first_question": "Does image direction transfer to caption state more than text-query state?", "source_table": "tables/figure_sources/image_text_probe_transfer.csv", "non_claim": "Caption transfer is a positive control, not vision evidence."},
        {"plot": "patch_recovery_by_modality.png", "first_question": "Do visual patches beat OCR/background/text/random controls on aggregate and in raw spread?", "source_table": "tables/figure_sources/patch_recovery_by_modality.csv", "non_claim": "Patch is not a real VLM intervention."},
        {"plot": "concept_specificity_matrix.png", "first_question": "Which families have positive specificity gaps?", "source_table": "tables/figure_sources/concept_specificity_matrix.csv", "non_claim": "Bright cells are synthetic audit passes."},
        {"plot": "spatial_region_patch_map.png", "first_question": "Are spatial rows specific or carried by wrong-region controls?", "source_table": "tables/figure_sources/spatial_region_patch_map.csv", "non_claim": "Region alignment is synthetic."},
        {"plot": "cross_modal_feature_bridge.png", "first_question": "How do synthetic image, caption, and connector directions align?", "source_table": "tables/figure_sources/cross_modal_feature_bridge.csv", "non_claim": "Bridge is hash-basis synthetic."},
        {"plot": "ocr_background_shortcut_panel.png", "first_question": "Which shortcut rows block visual claims?", "source_table": "tables/figure_sources/ocr_background_shortcut_panel.csv", "non_claim": "Shortcut traps are audit gates, not failures to hide."},
        {"plot": "paired_examples.png", "first_question": "Which specimens most strongly warn against the aggregate story?", "source_table": "tables/figure_sources/paired_examples.csv", "non_claim": "Specimen cards narrow the claim; they do not establish broad evidence."},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 33.")


def write_figure_source_tables(
    ctx: bench.RunContext,
    probe_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    specificity_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    leak_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    dose_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> None:
    patch_summary = patch_recovery_summary(patch_rows)
    entries: list[tuple[str, Sequence[Mapping[str, Any]], str, str, str]] = [
        ("multimodal_evidence_dashboard.png", evidence_rows, "connector_auc, visual_region_recovery, control_floor, specificity_gap, shortcut_leak_rate", "strongest shortcut/random/wrong-region control", "Synthetic audit package status by family."),
        ("target_vs_control.png", specificity_rows, "visual_region_recovery minus control_floor", "paired strongest control", "Visual patch must beat controls on the same row."),
        ("dose_response.png", dose_rows, "recovery across dose", "caption/shortcut/random dose curves", "Visual dose curve should separate from controls."),
        ("modality_handoff_atlas.png", probe_rows, "AUC", "state-type comparisons", "Readout semantics differ by synthetic state."),
        ("image_text_probe_transfer.png", transfer_rows, "cross-modal AUC", "text_query transfer", "Caption/text transfer are controls for visual evidence."),
        ("patch_recovery_by_modality.png", patch_summary, "mean recovery and raw item counts", "patch-type controls", "Aggregate patch recovery should not hide controls."),
        ("concept_specificity_matrix.png", evidence_rows, "family-level gates", "control_floor", "Family-level posture and caveats."),
        ("spatial_region_patch_map.png", [r for r in specificity_rows if r.get("concept_family") == "spatial"], "spatial recovery", "wrong_region_or_ocr and random", "Spatial claim requires region-specific recovery."),
        ("cross_modal_feature_bridge.png", transfer_rows, "direction cosine", "caption/text transfer", "Bridge geometry is debug signal only."),
        ("ocr_background_shortcut_panel.png", [r for r in leak_rows if r.get("leak_status") == "shortcut_control_triggered"], "shortcut_margin_delta", "visual-only margin", "Shortcut rows should block overclaims."),
        ("paired_examples.png", failure_rows, "specimen recovery/control/leak fields", "specimen-specific controls", "Failure specimens must stay visible."),
    ]
    manifest_rows: list[dict[str, Any]] = []
    for fig_name, rows, metric, control, claim in entries:
        source_rows = [dict(r) for r in rows] if rows else [{"note": f"No source rows for {fig_name} in this run."}]
        rel = f"tables/figure_sources/{fig_name.replace('.png', '.csv')}"
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, source_rows)
        ctx.register_artifact(path, "table", f"Source table for {fig_name}.")
        manifest_rows.append({
            "figure_path": f"plots/{fig_name}",
            "source_table": rel,
            "source_row_count": len(rows),
            "metric": metric,
            "control": control,
            "claim_supported": claim,
            "generated": not bool(getattr(ctx.args, "no_plots", False)),
            "scope": "synthetic connector only; no real VLM evidence",
        })
    csv_path = ctx.path("tables", "plot_manifest.csv")
    bench.write_csv_with_context(ctx, csv_path, manifest_rows)
    ctx.register_artifact(csv_path, "table", "Figure manifest with source tables, row counts, metrics, controls, and claim scope.")
    json_path = ctx.path("plot_manifest.json")
    bench.write_json(json_path, {"figures": manifest_rows, "scope": "synthetic connector only", "science_ready_for_real_vlm": False})
    ctx.register_artifact(json_path, "manifest", "Machine-readable Lab 33 plot manifest.")


def write_plots(
    ctx: bench.RunContext,
    probe_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    specificity_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    leak_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    dose_rows: Sequence[Mapping[str, Any]] | None = None,
    failure_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:  # type: ignore[override]
    dose_rows = list(dose_rows or [])
    failure_rows = list(failure_rows or [])
    write_plot_guide(ctx)
    write_figure_source_tables(ctx, probe_rows, patch_rows, specificity_rows, transfer_rows, leak_rows, evidence_rows, dose_rows, failure_rows)
    if ctx.args.no_plots:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    expected = [
        "multimodal_evidence_dashboard.png",
        "target_vs_control.png",
        "dose_response.png",
        "modality_handoff_atlas.png",
        "image_text_probe_transfer.png",
        "patch_recovery_by_modality.png",
        "concept_specificity_matrix.png",
        "spatial_region_patch_map.png",
        "cross_modal_feature_bridge.png",
        "ocr_background_shortcut_panel.png",
        "paired_examples.png",
    ]
    if not evidence_rows:
        for name in expected:
            write_placeholder(ctx, name, name.replace("_", " ").replace(".png", ""), "No evidence rows were produced.")
        return

    families = [str(r["concept_family"]) for r in evidence_rows]
    x = np.arange(len(families))
    connector_auc = [fnum(r.get("connector_auc"), 0.0) for r in evidence_rows]
    visual_patch = [fnum(r.get("visual_region_recovery"), 0.0) for r in evidence_rows]
    control_floor = [fnum(r.get("control_floor"), 0.0) for r in evidence_rows]
    gaps = [fnum(r.get("specificity_gap"), 0.0) for r in evidence_rows]
    leak_rates = [fnum(r.get("shortcut_leak_rate"), 0.0) for r in evidence_rows]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle("Lab 33 multimodal synthetic connector dashboard\nreal-VLM science_ready=false; controls are evidence, not decoration", fontsize=13, fontweight="bold")
    state_means = {s: safe_mean([r["auc"] for r in probe_rows if r["state_type"] == s], default=0.0) for s in STATE_TYPES}
    axes[0, 0].bar(list(state_means), list(state_means.values()))
    axes[0, 0].axhline(MIN_AUC_GATE, linestyle="--", linewidth=1, label="AUC gate")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Mean readout AUC by synthetic state")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].bar(x - 0.18, visual_patch, 0.36, label="visual patch")
    axes[0, 1].bar(x + 0.18, control_floor, 0.36, label="strongest control")
    axes[0, 1].axhline(MIN_RECOVERY_GATE, linestyle="--", linewidth=1, label="recovery gate")
    axes[0, 1].set_xticks(x, families, rotation=20, ha="right")
    axes[0, 1].set_title("Visual patch versus strongest control")
    axes[0, 1].set_ylabel("recovery")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].bar(families, gaps)
    axes[1, 0].axhline(MIN_SPECIFICITY_GAP, linestyle="--", linewidth=1, label="specificity gate")
    axes[1, 0].axhline(0, linewidth=0.8)
    axes[1, 0].set_xticks(x, families, rotation=20, ha="right")
    axes[1, 0].set_ylabel("visual minus control")
    axes[1, 0].set_title("Specificity gap")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(families, leak_rates)
    axes[1, 1].set_xticks(x, families, rotation=20, ha="right")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("OCR/background shortcut trigger rate")
    axes[1, 1].set_ylabel("rate")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    bench.save_figure(ctx, fig, "multimodal_evidence_dashboard.png", "Lab 33 synthetic multimodal dashboard.")

    if specificity_rows:
        order = sorted(range(len(specificity_rows)), key=lambda i: (str(specificity_rows[i].get("concept_family")), str(specificity_rows[i].get("item_id"))))
        labels = [str(specificity_rows[i].get("item_id", "")) for i in order]
        y_visual = [fnum(specificity_rows[i].get("visual_region_recovery"), 0.0) for i in order]
        y_control = [fnum(specificity_rows[i].get("control_floor"), 0.0) for i in order]
        fig, ax = plt.subplots(figsize=(max(9.5, 0.38 * len(labels) + 4), 5.2))
        xx = np.arange(len(labels))
        for xi, a, b in zip(xx, y_visual, y_control):
            ax.plot([xi, xi], [b, a], linewidth=0.8, alpha=0.55)
        ax.scatter(xx - 0.05, y_visual, label="visual patch", s=28)
        ax.scatter(xx + 0.05, y_control, label="strongest control", marker="x", s=34)
        ax.axhline(MIN_RECOVERY_GATE, linestyle="--", linewidth=1, label="recovery gate")
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(xx, labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("recovery")
        ax.set_title("Target visual patch versus strongest control, item by item")
        ax.legend(fontsize=8)
        fig.tight_layout()
        bench.save_figure(ctx, fig, "target_vs_control.png", "Paired target/control recovery for each Lab 33 item.")
    else:
        write_placeholder(ctx, "target_vs_control.png", "Target vs control", "No specificity rows were produced.")

    if dose_rows:
        keep = [VISUAL_PATCH_TYPE, "caption_patch", "ocr_channel_patch", "background_channel_patch", "text_query_patch", "random_patch_control"]
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        for patch_type in keep:
            rows = [r for r in dose_rows if r.get("patch_type") == patch_type]
            if not rows:
                continue
            doses = sorted({fnum(r.get("dose")) for r in rows if math.isfinite(fnum(r.get("dose")))})
            means = [safe_mean([r.get("recovery") for r in rows if fnum(r.get("dose")) == dose], default=0.0) for dose in doses]
            ax.plot(doses, means, marker="o", label=patch_type)
            for r in rows:
                d = fnum(r.get("dose"))
                if math.isfinite(d):
                    jitter = ((stable_int(str(r.get("dose_response_id", r.get("item_id", "")))) % 100) - 50) / 9000.0
                    ax.scatter([d + jitter], [fnum(r.get("recovery"), 0.0)], s=12, alpha=0.28)
        ax.axhline(0, linewidth=0.8)
        ax.axhline(MIN_RECOVERY_GATE, linestyle="--", linewidth=1, label="recovery gate")
        ax.set_xlabel("synthetic dose toward source state")
        ax.set_ylabel("recovery")
        ax.set_title("Patch dose-response: visual signal against caption, shortcut, and random controls")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        bench.save_figure(ctx, fig, "dose_response.png", "Synthetic dose-response curves for visual and control sources.")
    else:
        write_placeholder(ctx, "dose_response.png", "Dose response", "No dose-response rows were produced.")

    mat = np.zeros((len(STATE_TYPES), len(families)))
    for i, state_type in enumerate(STATE_TYPES):
        for j, family in enumerate(families):
            mat[i, j] = safe_mean([r["auc"] for r in probe_rows if r["concept_family"] == family and r["state_type"] == state_type], default=0.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(families) + 3), 5.0))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(STATE_TYPES)), STATE_TYPES)
    ax.set_xticks(range(len(families)), families, rotation=25, ha="right")
    ax.set_title("Modality handoff atlas")
    for i in range(len(STATE_TYPES)):
        for j in range(len(families)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUC")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "modality_handoff_atlas.png", "State-type by concept-family synthetic readout heatmap.")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x2 = np.arange(len(transfer_rows))
    labels = [r["concept_family"] for r in transfer_rows]
    ax.bar(x2 - 0.25, [fnum(r["image_direction_on_caption_auc"], 0.0) for r in transfer_rows], 0.25, label="image dir on caption")
    ax.bar(x2, [fnum(r["caption_direction_on_image_auc"], 0.0) for r in transfer_rows], 0.25, label="caption dir on image")
    ax.bar(x2 + 0.25, [fnum(r["image_direction_on_text_query_auc"], 0.0) for r in transfer_rows], 0.25, label="image dir on text-query")
    ax.axhline(0.5, linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x2, labels, rotation=25, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Image, caption, and text-query transfer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "image_text_probe_transfer.png", "Synthetic image/caption/text-query transfer AUCs.")

    patch_order = [
        "connector_clean_to_corrupt",
        VISUAL_PATCH_TYPE,
        "language_state_patch",
        "caption_patch",
        "ocr_channel_patch",
        "background_channel_patch",
        "text_query_patch",
        "wrong_region_or_ocr_patch",
        "random_patch_control",
    ]
    patch_order = [p for p in patch_order if any(r.get("patch_type") == p for r in patch_rows)]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    means = [safe_mean([r.get("recovery") for r in patch_rows if r.get("patch_type") == p], default=0.0) for p in patch_order]
    errs = [safe_stderr([r.get("recovery") for r in patch_rows if r.get("patch_type") == p]) for p in patch_order]
    ax.bar(range(len(patch_order)), means, yerr=[0 if not math.isfinite(e) else e for e in errs], capsize=3)
    for i, p in enumerate(patch_order):
        vals = [fnum(r.get("recovery")) for r in patch_rows if r.get("patch_type") == p]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, alpha=0.65)
    ax.axhline(0, linewidth=0.8)
    ax.axhline(1, linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(patch_order)), patch_order, rotation=35, ha="right")
    ax.set_ylabel("recovery")
    ax.set_title("Patch recovery by modality/control, with raw rows")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "patch_recovery_by_modality.png", "Patch recovery by synthetic modality or shortcut control, including raw-point spread.")

    fam_mat = np.zeros((len(families), 5))
    for i, row in enumerate(evidence_rows):
        fam_mat[i, :] = [
            fnum(row.get("connector_auc"), 0.0),
            fnum(row.get("visual_region_recovery"), 0.0),
            fnum(row.get("control_floor"), 0.0),
            fnum(row.get("specificity_gap"), 0.0),
            fnum(row.get("shortcut_leak_rate"), 0.0),
        ]
    fig, ax = plt.subplots(figsize=(8.4, max(4.8, 0.45 * len(families) + 2)))
    im = ax.imshow(fam_mat, aspect="auto", vmin=-0.5, vmax=1.0, cmap="coolwarm")
    ax.set_yticks(range(len(families)), families)
    ax.set_xticks(range(5), ["connector AUC", "visual rec", "control", "gap", "shortcut"], rotation=25, ha="right")
    ax.set_title("Concept specificity matrix")
    for i in range(fam_mat.shape[0]):
        for j in range(fam_mat.shape[1]):
            ax.text(j, i, f"{fam_mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "concept_specificity_matrix.png", "Family-level multimodal specificity matrix.")

    spatial_rows = [r for r in specificity_rows if r.get("concept_family") == "spatial"]
    if spatial_rows:
        names = [str(r["item_id"]).replace("spatial_", "") for r in spatial_rows]
        vals = np.array([
            [fnum(r.get("visual_region_recovery"), 0.0), fnum(r.get("wrong_region_or_ocr_recovery"), 0.0), fnum(r.get("random_control_recovery"), 0.0)]
            for r in spatial_rows
        ])
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        im = ax.imshow(vals.T, aspect="auto", vmin=-0.5, vmax=1.0, cmap="coolwarm")
        ax.set_yticks(range(3), ["visual", "wrong/shortcut", "random"])
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
        ax.set_title("Spatial region patch map")
        fig.colorbar(im, ax=ax, shrink=0.8, label="recovery")
        fig.tight_layout()
        bench.save_figure(ctx, fig, "spatial_region_patch_map.png", "Synthetic spatial row patch recovery map.")
    else:
        write_placeholder(ctx, "spatial_region_patch_map.png", "Spatial region patch map", "No spatial rows selected by this prompt-set cap.")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x2 - 0.18, [fnum(r["image_caption_direction_cosine"], 0.0) for r in transfer_rows], 0.36, label="image-caption cosine")
    ax.bar(x2 + 0.18, [fnum(r["image_connector_direction_cosine"], 0.0) for r in transfer_rows], 0.36, label="image-connector cosine")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x2, labels, rotation=25, ha="right")
    ax.set_ylabel("cosine")
    ax.set_title("Synthetic cross-modal feature bridge")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "cross_modal_feature_bridge.png", "Synthetic direction cosine bridge by family.")

    shortcut_rows = [r for r in leak_rows if r.get("leak_status") == "shortcut_control_triggered"]
    if shortcut_rows:
        names = [str(r["item_id"]).replace("background_trap_", "bg_").replace("ocr_trap_", "ocr_") for r in shortcut_rows]
        deltas = [fnum(r.get("shortcut_margin_delta"), 0.0) for r in shortcut_rows]
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(names) + 3), 4.8))
        ax.bar(range(len(names)), deltas)
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_ylabel("vision full minus visual-only margin")
        ax.set_title("OCR/background shortcut panel")
        fig.tight_layout()
        bench.save_figure(ctx, fig, "ocr_background_shortcut_panel.png", "OCR/background shortcut effect on visual margin.")
    else:
        write_placeholder(ctx, "ocr_background_shortcut_panel.png", "OCR/background shortcut panel", "No OCR or background shortcut rows were selected.")

    if failure_rows:
        rows = list(failure_rows)[:14]
        labels = [str(r.get("item_id", ""))[:28] for r in rows]
        visual_vals = [fnum(r.get("visual_region_recovery"), 0.0) for r in rows]
        control_vals = [fnum(r.get("control_floor"), 0.0) for r in rows]
        y = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.38 * len(rows) + 1.5)))
        ax.hlines(y, control_vals, visual_vals, linewidth=1)
        ax.scatter(control_vals, y, label="control floor", s=30)
        ax.scatter(visual_vals, y, label="visual patch", s=30)
        ax.axvline(MIN_RECOVERY_GATE, linestyle="--", linewidth=1, label="recovery gate")
        ax.set_yticks(y, labels)
        ax.set_xlabel("recovery")
        ax.set_title("Paired specimens: visual patch versus strongest control")
        ax.legend(fontsize=8)
        fig.tight_layout()
        bench.save_figure(ctx, fig, "paired_examples.png", "Specimen-level visual/control paired examples.")
    else:
        write_placeholder(ctx, "paired_examples.png", "Paired examples", "No failure specimens were produced. Check diagnostics/warning_summary.csv before trusting the run.")


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:  # type: ignore[override]
    items, data_info = load_items(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 33 data manifest, hash, and synthetic-mode scope.")

    schema_rows, schema_summary = schema_audit(ctx, items)
    usable_items = [item for item, row in zip(items, schema_rows) if row["schema_ok"]]
    if not usable_items:
        raise RuntimeError("Lab 33 has no schema-valid rows after audit.")

    # The bench-loaded model is not the object of multimodal science here. It
    # still proves that the standard microscope checks run in the same way as
    # other labs, which keeps the artifact contract uniform.
    hook_check = bench.run_hook_parity_check(ctx, bundle, usable_items[0].text_control_prompt)
    first = bench.run_with_residual_cache(bundle, usable_items[0].text_control_prompt)
    lens_check = bench.run_lens_self_check(ctx, bundle, first)
    patch_noop = bench.run_patch_noop_check(ctx, bundle, usable_items[0].text_control_prompt)

    alignment = alignment_validation(usable_items)
    alignment_path = ctx.path("diagnostics", "alignment_validation.json")
    bench.write_json(alignment_path, alignment)
    ctx.register_artifact(alignment_path, "diagnostic", "Synthetic connector alignment validation and real-VLM requirements.")
    write_run_config_snapshot(ctx, data_info, alignment)

    render_rows, render_summary = render_images(ctx, usable_items)
    states = build_states(usable_items)
    state_summary = state_vector_audit(ctx, states)
    prompt_rows = prompt_manifest(usable_items, render_rows)
    probe_rows = modality_probe_report(usable_items, states)
    patch_rows, specificity_rows = patch_report(usable_items, states)
    transfer_rows = cross_modal_transfer(usable_items, states)
    leak_rows, leak_summary = leak_audit(usable_items, states)
    text_leak_rows, text_summary = text_leak_audit(usable_items)
    evidence_rows, counterexamples, metrics = evidence_matrix(
        data_info,
        probe_rows,
        patch_rows,
        specificity_rows,
        transfer_rows,
        leak_rows,
        text_summary,
    )
    augment_rows(prompt_rows, probe_rows, patch_rows, specificity_rows, transfer_rows, leak_rows, text_leak_rows, evidence_rows, counterexamples)
    score_rows = state_score_long(usable_items, states)
    dose_rows = patch_dose_response(usable_items, states)
    patch_summary_rows = patch_recovery_summary(patch_rows)
    family_summary_rows = concept_family_summary(evidence_rows, specificity_rows, leak_rows, text_leak_rows)
    failure_rows = failure_specimens(specificity_rows, leak_rows, text_leak_rows, counterexamples)

    metrics = {
        **metrics,
        "data": data_info,
        "schema": schema_summary,
        "render": render_summary,
        "state_vectors": state_summary,
        "alignment": alignment,
        "leak_summary": leak_summary,
        "text_leak_summary": text_summary,
        "n_state_score_rows": len(score_rows),
        "n_dose_response_rows": len(dose_rows),
        "n_failure_specimens": len(failure_rows),
    }
    warning_rows = warning_summary_rows(data_info, schema_summary, render_summary, state_summary, alignment, metrics, leak_summary, text_summary, specificity_rows, failure_rows)

    write_tables(
        ctx,
        prompt_rows,
        probe_rows,
        patch_rows,
        specificity_rows,
        transfer_rows,
        leak_rows,
        text_leak_rows,
        evidence_rows,
        counterexamples,
    )
    write_upgrade_tables(ctx, score_rows, dose_rows, patch_summary_rows, family_summary_rows, failure_rows, warning_rows)
    write_state(ctx, usable_items, states, transfer_rows)
    write_status_files(ctx, data_info, hook_check, lens_check, patch_noop, schema_summary, render_summary, state_summary, alignment, metrics)

    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 33 metrics, gates, and thresholds.")

    write_method_card(ctx, evidence_rows, metrics)
    write_operationalization_audit(ctx, evidence_rows, counterexamples, text_summary, leak_summary)
    write_real_vlm_checklist(ctx)
    write_run_summary(ctx, data_info, metrics, evidence_rows, counterexamples)
    write_claims(ctx, evidence_rows)
    write_plots(ctx, probe_rows, patch_rows, specificity_rows, transfer_rows, leak_rows, evidence_rows, dose_rows, failure_rows)
    print(f"[lab33] wrote {len(prompt_rows)} prompts, {len(probe_rows)} probe rows, {len(patch_rows)} patch rows, {len(dose_rows)} dose rows, {len(evidence_rows)} evidence rows, and {len(failure_rows)} failure specimens")
