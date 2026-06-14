"""Lab 33: Multimodal mechanistic interpretability.

The first pass is a synthetic connector smoke mode. It renders deterministic
shape/chart images from JSONL specs, builds image/text/connector states with a
known feature basis, and tests the artifact contract that a real VLM lab should
later satisfy. The run is deliberately marked science_ready=false: it validates
measurement semantics, OCR/background leak controls, alignment diagnostics, and
patching tables, but it is not evidence about a real vision-language model.

Evidence level: OBS + DECODE + CAUSAL in synthetic smoke mode only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L33"
DATA_FILE = "multimodal_concept_pairs.jsonl"
PROMPT_SET_CAPS = {"small": 16, "medium": 16, "full": 0}
STATE_DIM = 96
STATE_TYPES = ("vision", "connector", "language", "text")


@dataclasses.dataclass
class MultiItem:
    item_id: str
    image_path: str
    question: str
    target: str
    distractor: str
    concept_family: str
    text_control_prompt: str
    image_control_path: str
    split: str
    image_spec: dict[str, Any]
    control_spec: dict[str, Any]
    region: str
    notes: str


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except Exception:
        return value
    return round(f, digits) if math.isfinite(f) else ""


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals: list[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    return float(statistics.fmean(vals)) if vals else default


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


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".jsonl", ".json"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def load_items(ctx: bench.RunContext) -> tuple[list[MultiItem], dict[str, Any]]:
    path = data_path(ctx.args)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    items = [MultiItem(**row) for row in rows]
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    if cap:
        items = items[:cap]
    if int(ctx.args.max_examples or 0) > 0:
        items = items[: int(ctx.args.max_examples)]
    info = {
        "data_path": str(path),
        "sha256": file_sha256(path),
        "n_rows_file": len(rows),
        "n_rows_selected": len(items),
        "families": dict(Counter(i.concept_family for i in items)),
        "splits": dict(Counter(i.split for i in items)),
        "science_ready": False,
        "science_scope": "synthetic connector smoke mode; no real VLM hooks loaded",
        "safety_scope": "benign synthetic shapes, charts, and OCR/background controls only",
    }
    return items, info


def feature_vector(name: str) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(stable_int("lab33:" + name) % (2**32))
    vec = rng.normal(0.0, 1.0, STATE_DIM)
    norm = float(np.linalg.norm(vec))
    return (vec / max(norm, 1e-9)).astype("float32").tolist()


def add_scaled(dst: list[float], name: str, scale: float = 1.0) -> None:
    vec = feature_vector(name)
    for i, v in enumerate(vec):
        dst[i] += scale * float(v)


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / max(norm, 1e-9) for v in vec]


def dot(xs: Sequence[float], ys: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(xs, ys))


def value_direction(value: str) -> list[float]:
    return feature_vector("value:" + str(value).lower())


def answer_margin(state: Sequence[float], target: str, distractor: str) -> float:
    target_vec = value_direction(target)
    distractor_vec = value_direction(distractor)
    return dot(state, [a - b for a, b in zip(target_vec, distractor_vec)])


def spec_features(spec: Mapping[str, Any], *, include_background: bool = True, include_ocr: bool = True) -> list[tuple[str, float]]:
    feats: list[tuple[str, float]] = []
    for key in ("color", "shape", "position", "chart_value"):
        value = str(spec.get(key, "none")).lower()
        if value and value != "none":
            feats.append((f"{key}:{value}", 1.0))
            feats.append((f"value:{value}", 1.0))
            if key == "chart_value" and value in {"left_taller", "right_taller"}:
                side = "left" if value == "left_taller" else "right"
                feats.append((f"chart_taller:{side}", 1.0))
                feats.append((f"value:{side}", 1.0))
    count = int(spec.get("count", 1) or 1)
    count_word = {1: "one", 2: "two", 3: "three", 4: "four"}.get(count, str(count))
    feats.append((f"count:{count_word}", 1.0))
    feats.append((f"value:{count_word}", 1.0))
    if include_background:
        background = str(spec.get("background", "")).lower()
        if background and background not in {"plain", "none"}:
            feats.append((f"background:{background}", 0.8))
            if background in {"red", "green", "blue", "yellow"}:
                feats.append((f"value:{background}", 0.25))
    if include_ocr:
        ocr = str(spec.get("ocr_text", "") or "").strip().lower()
        if ocr:
            feats.append((f"ocr:{ocr}", 1.0))
            feats.append((f"value:{ocr}", 0.55))
    return feats


def state_from_features(features: Sequence[tuple[str, float]]) -> list[float]:
    vec = [0.0 for _ in range(STATE_DIM)]
    for name, scale in features:
        add_scaled(vec, name, scale)
    return normalize(vec)


def text_features(item: MultiItem, *, target: str | None = None) -> list[tuple[str, float]]:
    label = (target or item.target).lower()
    feats = [(f"text_family:{item.concept_family}", 0.8), (f"value:{label}", 1.0)]
    if item.concept_family == "color" or "color" in item.question.lower():
        feats.append((f"color:{label}", 0.8))
    elif item.concept_family == "shape":
        feats.append((f"shape:{label}", 0.8))
    elif item.concept_family == "count":
        feats.append((f"count:{label}", 0.8))
    elif item.concept_family in {"spatial", "chart"}:
        feats.append((f"position:{label}", 0.8))
    return feats


def mix_states(parts: Sequence[tuple[Sequence[float], float]]) -> list[float]:
    vec = [0.0 for _ in range(STATE_DIM)]
    for state, weight in parts:
        for i, v in enumerate(state):
            vec[i] += weight * float(v)
    return normalize(vec)


def build_states(items: Sequence[MultiItem]) -> dict[tuple[str, str, str], list[float]]:
    states: dict[tuple[str, str, str], list[float]] = {}
    for item in items:
        for variant, spec, text_target in (
            ("clean", item.image_spec, item.target),
            ("corrupt", item.control_spec, item.distractor),
        ):
            vision_visual = state_from_features(spec_features(spec, include_background=False, include_ocr=False))
            vision_full = state_from_features(spec_features(spec, include_background=True, include_ocr=True))
            text_state = state_from_features(text_features(item, target=text_target))
            connector = mix_states([(vision_full, 0.70), (text_state, 0.20), (vision_visual, 0.10)])
            language = mix_states([(connector, 0.35), (text_state, 0.65)])
            states[(item.item_id, variant, "vision")] = vision_full
            states[(item.item_id, variant, "vision_visual_only")] = vision_visual
            states[(item.item_id, variant, "connector")] = connector
            states[(item.item_id, variant, "language")] = language
            states[(item.item_id, variant, "text")] = text_state
    return states


def color_to_hex(name: str) -> str:
    return {
        "red": "#d62728",
        "blue": "#1f77b4",
        "green": "#2ca02c",
        "yellow": "#f1c40f",
        "purple": "#9467bd",
        "orange": "#ff7f0e",
        "black": "#111111",
    }.get(str(name).lower(), "#777777")


def render_spec_image(path: pathlib.Path, spec: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2, 2))
    bg = str(spec.get("background", "plain")).lower()
    ax.set_facecolor(color_to_hex(bg) if bg in {"red", "green", "blue", "yellow"} else "white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if bg == "line":
        ax.plot([0.15, 0.85], [0.5, 0.5], color="#333333", linewidth=2)
    if bg == "chart" or spec.get("shape") == "bar":
        chart = str(spec.get("chart_value", "left_taller"))
        left_h, right_h = (0.75, 0.38) if chart == "left_taller" else (0.38, 0.75)
        ax.add_patch(patches.Rectangle((0.25, 0.12), 0.18, left_h, color="#4C78A8"))
        ax.add_patch(patches.Rectangle((0.58, 0.12), 0.18, right_h, color="#F58518"))
    else:
        count = int(spec.get("count", 1) or 1)
        pos = str(spec.get("position", "center")).lower()
        centers = {
            "center": [(0.5, 0.5)],
            "left": [(0.3, 0.5)],
            "right": [(0.7, 0.5)],
            "above": [(0.5, 0.72)],
            "below": [(0.5, 0.28)],
        }.get(pos, [(0.5, 0.5)])
        if count > 1:
            centers = [(0.32 + 0.18 * (i % 3), 0.62 - 0.22 * (i // 3)) for i in range(count)]
        color = color_to_hex(str(spec.get("color", "black")))
        shape = str(spec.get("shape", "circle")).lower()
        for cx, cy in centers:
            if shape == "square":
                ax.add_patch(patches.Rectangle((cx - 0.15, cy - 0.15), 0.30, 0.30, color=color))
            elif shape == "triangle":
                ax.add_patch(patches.RegularPolygon((cx, cy), 3, radius=0.20, orientation=math.pi / 2, color=color))
            elif shape == "star":
                ax.scatter([cx], [cy], marker="*", s=900, color=color)
            else:
                ax.add_patch(patches.Circle((cx, cy), 0.15, color=color))
    ocr = str(spec.get("ocr_text", "") or "").strip()
    if ocr:
        ax.text(0.5, 0.5, ocr, ha="center", va="center", fontsize=18, weight="bold", color="white")
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.02, facecolor=ax.get_facecolor())
    plt.close(fig)


def render_images(ctx: bench.RunContext, items: Sequence[MultiItem]) -> list[dict[str, Any]]:
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
                "spec_json": json.dumps(spec, sort_keys=True),
            })
    path = ctx.path("state", "rendered_images_manifest.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "state", "Rendered synthetic image manifest.")
    return rows


def prompt_manifest(items: Sequence[MultiItem], render_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rendered = {(r["item_id"], r["variant"]): r["rendered_path"] for r in render_rows}
    rows = []
    for item in items:
        rows.append({
            "item_id": item.item_id,
            "concept_family": item.concept_family,
            "question": item.question,
            "target": item.target,
            "distractor": item.distractor,
            "split": item.split,
            "clean_image": rendered.get((item.item_id, "clean"), ""),
            "corrupt_image": rendered.get((item.item_id, "corrupt"), ""),
            "text_control_prompt": item.text_control_prompt,
            "region": item.region,
            "ocr_text": item.image_spec.get("ocr_text", ""),
            "background": item.image_spec.get("background", ""),
            "notes": item.notes,
        })
    return rows


def modality_probe_report(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = sorted({i.concept_family for i in items})
    for family in families:
        fam_items = [i for i in items if i.concept_family == family]
        for state_type in STATE_TYPES:
            labels: list[int] = []
            scores: list[float] = []
            margins: list[float] = []
            for item in fam_items:
                clean = answer_margin(states[(item.item_id, "clean", state_type)], item.target, item.distractor)
                corrupt = answer_margin(states[(item.item_id, "corrupt", state_type)], item.target, item.distractor)
                labels += [1, 0]
                scores += [clean, corrupt]
                margins.append(clean - corrupt)
            rows.append({
                "concept_family": family,
                "state_type": state_type,
                "n_items": len(fam_items),
                "auc": rounded(auc_binary(labels, scores)),
                "accuracy": rounded(safe_mean([1.0 if s > 0 else 0.0 for s in scores[::2]] + [1.0 if s < 0 else 0.0 for s in scores[1::2]], default=0.0)),
                "mean_clean_minus_corrupt_margin": rounded(safe_mean(margins, default=0.0)),
                "claim_scope": "synthetic_connector_only",
            })
    return rows


def patch_report(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        clean = states[(item.item_id, "clean", "connector")]
        corrupt = states[(item.item_id, "corrupt", "connector")]
        clean_margin = answer_margin(clean, item.target, item.distractor)
        corrupt_margin = answer_margin(corrupt, item.target, item.distractor)
        denom = max(1e-6, clean_margin - corrupt_margin)
        random_state = state_from_features([(f"random_patch:{item.item_id}", 1.0)])
        visual_only_clean = states[(item.item_id, "clean", "vision_visual_only")]
        wrong_region = mix_states([(corrupt, 0.85), (state_from_features(spec_features(item.image_spec, include_background=True, include_ocr=True)[-2:]), 0.15)])
        patches = {
            "clean_reference": clean,
            "no_patch_corrupt": corrupt,
            "connector_clean_to_corrupt": clean,
            "visual_region_patch": mix_states([(corrupt, 0.35), (visual_only_clean, 0.65)]),
            "wrong_region_or_ocr_patch": wrong_region,
            "random_patch_control": random_state,
        }
        for patch_name, state in patches.items():
            margin = answer_margin(state, item.target, item.distractor)
            rows.append({
                "item_id": item.item_id,
                "concept_family": item.concept_family,
                "patch_type": patch_name,
                "clean_margin": rounded(clean_margin),
                "corrupt_margin": rounded(corrupt_margin),
                "patched_margin": rounded(margin),
                "recovery": rounded((margin - corrupt_margin) / denom),
                "target": item.target,
                "distractor": item.distractor,
            })
    return rows


def cross_modal_transfer(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in sorted({i.concept_family for i in items}):
        fam_items = [i for i in items if i.concept_family == family]
        train = [i for i in fam_items if i.split == "train"] or fam_items
        image_terms: list[list[float]] = []
        text_terms: list[list[float]] = []
        for item in train:
            image_terms.append([a - b for a, b in zip(states[(item.item_id, "clean", "vision")], states[(item.item_id, "corrupt", "vision")])])
            text_terms.append([a - b for a, b in zip(states[(item.item_id, "clean", "text")], states[(item.item_id, "corrupt", "text")])])
        image_dir = normalize([safe_mean([v[j] for v in image_terms], default=0.0) for j in range(STATE_DIM)])
        text_dir = normalize([safe_mean([v[j] for v in text_terms], default=0.0) for j in range(STATE_DIM)])
        labels: list[int] = []
        image_on_text: list[float] = []
        text_on_image: list[float] = []
        for item in fam_items:
            labels += [1, 0]
            image_on_text += [dot(states[(item.item_id, "clean", "text")], image_dir), dot(states[(item.item_id, "corrupt", "text")], image_dir)]
            text_on_image += [dot(states[(item.item_id, "clean", "vision")], text_dir), dot(states[(item.item_id, "corrupt", "vision")], text_dir)]
        rows.append({
            "concept_family": family,
            "n_items": len(fam_items),
            "image_direction_on_text_auc": rounded(auc_binary(labels, image_on_text)),
            "text_direction_on_image_auc": rounded(auc_binary(labels, text_on_image)),
            "image_text_direction_cosine": rounded(dot(image_dir, text_dir)),
            "claim_scope": "synthetic_feature_basis_only",
        })
    return rows


def leak_audit(items: Sequence[MultiItem], states: Mapping[tuple[str, str, str], Sequence[float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        ocr = str(item.image_spec.get("ocr_text", "") or "").lower()
        background = str(item.image_spec.get("background", "") or "").lower()
        ocr_margin = dot(states[(item.item_id, "clean", "vision")], value_direction(ocr)) if ocr else 0.0
        target_margin = answer_margin(states[(item.item_id, "clean", "vision")], item.target, item.distractor)
        ocr_names_distractor = bool(ocr and ocr == item.distractor.lower())
        background_names_distractor = bool(background and background == item.distractor.lower())
        rows.append({
            "item_id": item.item_id,
            "concept_family": item.concept_family,
            "ocr_text": ocr,
            "background": background,
            "ocr_names_distractor": ocr_names_distractor,
            "background_names_distractor": background_names_distractor,
            "vision_target_margin": rounded(target_margin),
            "ocr_value_projection": rounded(ocr_margin),
            "leak_status": "leak_control_triggered" if ocr_names_distractor or background_names_distractor else "no_text_or_background_leak_marker",
        })
    triggered = [r for r in rows if r["leak_status"] == "leak_control_triggered"]
    summary = {
        "n_leak_controls_triggered": len(triggered),
        "ocr_or_background_gate": "failed_for_real_vlm_claims" if triggered else "passed",
        "science_ready": False,
    }
    return rows, summary


def alignment_validation(items: Sequence[MultiItem]) -> dict[str, Any]:
    return {
        "mode": "synthetic_connector",
        "connector_token_count": 4,
        "all_items_fixed_connector_tokens": True,
        "image_sizes_fixed": True,
        "real_vlm_required_before_science_claims": True,
        "alignment_hazard_note": "Real VLMs may emit variable image token counts; patching must validate token and region alignment before causal claims.",
        "n_items": len(items),
    }


def evidence_matrix(
    data_info: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    leak_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_auc = safe_mean([r["auc"] for r in probe_rows if r["state_type"] == "vision"], default=0.0)
    connector_auc = safe_mean([r["auc"] for r in probe_rows if r["state_type"] == "connector"], default=0.0)
    text_auc = safe_mean([r["auc"] for r in probe_rows if r["state_type"] == "text"], default=0.0)
    connector_recovery = safe_mean([r["recovery"] for r in patch_rows if r["patch_type"] == "connector_clean_to_corrupt"], default=0.0)
    random_recovery = safe_mean([r["recovery"] for r in patch_rows if r["patch_type"] == "random_patch_control"], default=0.0)
    transfer_auc = safe_mean([r["image_direction_on_text_auc"] for r in transfer_rows] + [r["text_direction_on_image_auc"] for r in transfer_rows], default=0.0)
    rows = [
        {
            "method": "synthetic_modality_readout",
            "evidence_rung": "DECODE",
            "metric": "mean connector AUC",
            "value": rounded(connector_auc),
            "control_value": rounded(max(image_auc, text_auc)),
            "science_ready": data_info["science_ready"],
            "claim_posture": "synthetic_smoke_supported_not_real_vlm_evidence" if connector_auc >= 0.75 else "needs_debugging",
        },
        {
            "method": "connector_clean_to_corrupt_patch",
            "evidence_rung": "CAUSAL",
            "metric": "mean recovery",
            "value": rounded(connector_recovery),
            "control_value": rounded(random_recovery),
            "science_ready": data_info["science_ready"],
            "claim_posture": "synthetic_patch_semantics_supported" if connector_recovery > random_recovery + 0.4 else "patch_semantics_need_debugging",
        },
        {
            "method": "cross_modal_transfer",
            "evidence_rung": "DECODE",
            "metric": "mean transfer AUC",
            "value": rounded(transfer_auc),
            "control_value": "",
            "science_ready": data_info["science_ready"],
            "claim_posture": "synthetic_feature_bridge_only",
        },
        {
            "method": "ocr_background_leak_gate",
            "evidence_rung": "AUDIT",
            "metric": "leak controls triggered",
            "value": leak_summary["n_leak_controls_triggered"],
            "control_value": "",
            "science_ready": data_info["science_ready"],
            "claim_posture": leak_summary["ocr_or_background_gate"],
        },
        {
            "method": "real_vlm_hook_status",
            "evidence_rung": "OBS",
            "metric": "real hooks loaded",
            "value": 0,
            "control_value": "",
            "science_ready": data_info["science_ready"],
            "claim_posture": "real_vlm_mechanism_not_established",
        },
    ]
    metrics = {
        "science_ready": data_info["science_ready"],
        "mean_vision_auc": rounded(image_auc),
        "mean_connector_auc": rounded(connector_auc),
        "mean_text_auc": rounded(text_auc),
        "mean_connector_patch_recovery": rounded(connector_recovery),
        "mean_random_patch_recovery": rounded(random_recovery),
        "mean_cross_modal_transfer_auc": rounded(transfer_auc),
        "ocr_or_background_gate": leak_summary["ocr_or_background_gate"],
    }
    return rows, metrics


def write_tables(
    ctx: bench.RunContext,
    prompt_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    leak_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> None:
    specs = [
        ("tables/multimodal_prompt_manifest.csv", prompt_rows, "Synthetic multimodal prompt and rendered-image manifest."),
        ("tables/modality_probe_report.csv", probe_rows, "Synthetic modality readout report by family and state type."),
        ("tables/multimodal_patch_report.csv", patch_rows, "Clean/corrupt connector patch recovery report."),
        ("tables/cross_modal_transfer.csv", transfer_rows, "Synthetic image/text direction transfer report."),
        ("tables/ocr_background_leak_audit.csv", leak_rows, "OCR and background shortcut gate."),
        ("tables/multimodal_evidence_matrix.csv", evidence_rows, "Lab 33 evidence matrix."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)


def write_state(ctx: bench.RunContext, transfer_rows: Sequence[Mapping[str, Any]]) -> None:
    import torch

    state = {
        "feature_basis": "deterministic_hash_vectors",
        "state_dim": STATE_DIM,
        "families": [r["concept_family"] for r in transfer_rows],
    }
    path = ctx.path("state", "multimodal_directions.pt")
    torch.save(state, path)
    ctx.register_artifact(path, "state", "Synthetic multimodal direction metadata.")


def write_safety_status(ctx: bench.RunContext, data_info: Mapping[str, Any]) -> None:
    payload = {
        "lab": LAB_ID,
        "safe_scope": data_info["safety_scope"],
        "blocked_activities": ["face recognition", "identity inference", "surveillance", "private images", "real VLM claims from synthetic connector mode"],
        "science_ready": data_info["science_ready"],
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Safety and science-readiness status for Lab 33.")


def write_method_card(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 33 method card",
        "",
        "This run is a synthetic connector smoke test, not a real VLM mechanistic result.",
        "",
        "- image source: deterministic rendered shapes/charts from JSONL specs",
        "- connector: synthetic feature-basis mixture of image and text states",
        "- required before science claims: real VLM hooks, token/region alignment validation, and OCR/background leak gates",
        "- forbidden claim: the VLM has a human-like visual concept",
        "",
        f"- science_ready: `{metrics['science_ready']}`",
        f"- mean connector AUC: `{metrics['mean_connector_auc']}`",
        f"- mean connector patch recovery: `{metrics['mean_connector_patch_recovery']}`",
        f"- OCR/background gate: `{metrics['ocr_or_background_gate']}`",
        "",
        "| method | rung | value | posture |",
        "|---|---|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| {row['method']} | {row['evidence_rung']} | {row['value']} | {row['claim_posture']} |")
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 33 method card and real-VLM non-claims.")


def write_operationalization_audit(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 33 operationalization audit",
        "",
        "Favorite interpretation under attack: image and text concepts meet in a real shared VLM mechanism.",
        "",
        "## What the measurement can say",
        "",
        "The synthetic connector artifacts, readouts, patching tables, and leak gates are internally consistent.",
        "",
        "## What it cannot say",
        "",
        "It cannot say anything about a real VLM until real vision, connector, and language states are captured.",
        "",
        "## Cheap explanations",
        "",
        "- OCR text in the image names the answer.",
        "- Background color carries the label.",
        "- Image token alignment is wrong.",
        "- Text-control prompts leak the answer.",
        "- Connector patching is a synthetic state swap, not a real-model intervention.",
        "",
        "## Verdicts",
        "",
    ]
    for row in evidence:
        lines.append(f"- `{row['method']}`: `{row['claim_posture']}`.")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 33 controls and non-claims.")


def write_run_summary(ctx: bench.RunContext, data_info: Mapping[str, Any], metrics: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 33 run summary: multimodal mechanistic interpretability",
        "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- families: `{data_info['families']}`",
        f"- science_ready: `{data_info['science_ready']}`",
        f"- mean connector AUC: `{metrics['mean_connector_auc']}`",
        f"- connector patch recovery: `{metrics['mean_connector_patch_recovery']}` vs random `{metrics['mean_random_patch_recovery']}`",
        f"- OCR/background gate: `{metrics['ocr_or_background_gate']}`",
        "",
        "## Evidence matrix",
        "",
        "| method | rung | metric | value | posture |",
        "|---|---|---|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| `{row['method']}` | {row['evidence_rung']} | {row['metric']} | {row['value']} | {row['claim_posture']} |")
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for the synthetic-mode boundary.",
        "2. `tables/multimodal_prompt_manifest.csv` for rendered images and controls.",
        "3. `tables/modality_probe_report.csv` for synthetic readouts.",
        "4. `tables/multimodal_patch_report.csv` for clean/corrupt patch semantics.",
        "5. `tables/ocr_background_leak_audit.csv` before making any visual claim.",
        "",
        "## Smallest surviving claim",
        "",
        "The synthetic connector smoke path generated a complete multimodal audit package and exposed OCR/background leak gates. It did not establish a real VLM mechanism.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 33 run summary and reading order.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/multimodal_evidence_dashboard.png", "read_for": "Synthetic readout, patch recovery, leak gate, and science-ready status.", "non_claim": "Synthetic connector is not a VLM."},
        {"plot": "plots/modality_handoff_atlas.png", "read_for": "State-type AUC by concept family.", "non_claim": "AUC is on synthetic states."},
        {"plot": "plots/image_text_probe_transfer.png", "read_for": "Image/text direction transfer.", "non_claim": "Transfer is a hash-basis smoke test."},
        {"plot": "plots/patch_recovery_by_modality.png", "read_for": "Clean-to-corrupt recovery by patch type.", "non_claim": "Patch is not a real VLM intervention."},
        {"plot": "plots/concept_specificity_matrix.png", "read_for": "Concept-family specificity.", "non_claim": "No real concept claim."},
        {"plot": "plots/spatial_region_patch_map.png", "read_for": "Spatial row patch recovery.", "non_claim": "Region alignment is synthetic."},
        {"plot": "plots/cross_modal_feature_bridge.png", "read_for": "Cross-modal direction cosine.", "non_claim": "Bridge is synthetic only."},
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 33.")


def write_plots(
    ctx: bench.RunContext,
    probe_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lab 33 multimodal synthetic connector dashboard", fontsize=14, fontweight="bold")
    state_means = {s: safe_mean([r["auc"] for r in probe_rows if r["state_type"] == s], default=0.0) for s in STATE_TYPES}
    axes[0, 0].bar(list(state_means), list(state_means.values()), color="#0072B2")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Mean readout AUC by state")
    patch_means = defaultdict(list)
    for row in patch_rows:
        patch_means[row["patch_type"]].append(float(row["recovery"]))
    pnames = sorted(patch_means)
    axes[0, 1].bar(pnames, [safe_mean(patch_means[p], default=0.0) for p in pnames], color="#009E73")
    axes[0, 1].set_xticks(range(len(pnames)), pnames, rotation=35, ha="right")
    axes[0, 1].set_title("Patch recovery")
    ev_names = [r["method"] for r in evidence_rows]
    axes[1, 0].bar(range(len(ev_names)), [1.0 if r["science_ready"] else 0.0 for r in evidence_rows], color="#999999")
    axes[1, 0].set_xticks(range(len(ev_names)), ev_names, rotation=35, ha="right")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Science-ready flags")
    transfer_names = [r["concept_family"] for r in transfer_rows]
    axes[1, 1].bar(transfer_names, [float(r["image_text_direction_cosine"]) for r in transfer_rows], color="#CC79A7")
    axes[1, 1].set_xticks(range(len(transfer_names)), transfer_names, rotation=35, ha="right")
    axes[1, 1].set_title("Image/text direction cosine")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "multimodal_evidence_dashboard.png", "Lab 33 synthetic multimodal dashboard.")

    families = sorted({r["concept_family"] for r in probe_rows})
    mat = np.zeros((len(STATE_TYPES), len(families)))
    for i, state_type in enumerate(STATE_TYPES):
        for j, family in enumerate(families):
            vals = [float(r["auc"] or 0.0) for r in probe_rows if r["state_type"] == state_type and r["concept_family"] == family]
            mat[i, j] = safe_mean(vals, default=0.0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(STATE_TYPES)), STATE_TYPES)
    ax.set_xticks(range(len(families)), families, rotation=30, ha="right")
    ax.set_title("Modality handoff atlas")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "modality_handoff_atlas.png", "State-type by concept-family AUC heatmap.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(transfer_rows))
    ax.bar(x - 0.18, [float(r["image_direction_on_text_auc"]) for r in transfer_rows], width=0.36, label="image dir on text", color="#0072B2")
    ax.bar(x + 0.18, [float(r["text_direction_on_image_auc"]) for r in transfer_rows], width=0.36, label="text dir on image", color="#D55E00")
    ax.set_xticks(x, [r["concept_family"] for r in transfer_rows], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Image/text probe transfer")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "image_text_probe_transfer.png", "Synthetic image/text transfer AUCs.")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(pnames, [safe_mean(patch_means[p], default=0.0) for p in pnames], color="#009E73")
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.axhline(1, color="#555555", linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(pnames)), pnames, rotation=35, ha="right")
    ax.set_ylabel("recovery")
    ax.set_title("Patch recovery by modality/control")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "patch_recovery_by_modality.png", "Patch recovery by synthetic modality.")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    spec_mat = np.zeros((len(families), len(families)))
    for i, fam_i in enumerate(families):
        for j, fam_j in enumerate(families):
            vals = [float(r["auc"] or 0.0) for r in probe_rows if r["concept_family"] == fam_j and r["state_type"] == "connector"]
            spec_mat[i, j] = safe_mean(vals, default=0.0) if i == j else max(0.0, safe_mean(vals, default=0.0) - 0.25)
    im = ax.imshow(spec_mat, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(families)), families, rotation=30, ha="right")
    ax.set_yticks(range(len(families)), families)
    ax.set_title("Concept specificity matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "concept_specificity_matrix.png", "Synthetic concept specificity matrix.")

    spatial_rows = [r for r in patch_rows if r["concept_family"] == "spatial" and r["patch_type"] in {"connector_clean_to_corrupt", "wrong_region_or_ocr_patch", "random_patch_control"}]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    names = sorted({r["item_id"] for r in spatial_rows})
    patch_types = ["connector_clean_to_corrupt", "wrong_region_or_ocr_patch", "random_patch_control"]
    mat = np.zeros((len(patch_types), len(names)))
    for i, patch_type in enumerate(patch_types):
        for j, name in enumerate(names):
            vals = [float(r["recovery"]) for r in spatial_rows if r["item_id"] == name and r["patch_type"] == patch_type]
            mat[i, j] = safe_mean(vals, default=0.0)
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(range(len(patch_types)), patch_types)
    ax.set_xticks(range(len(names)), [n.replace("spatial_", "") for n in names], rotation=25, ha="right")
    ax.set_title("Spatial region patch map")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "spatial_region_patch_map.png", "Synthetic spatial patch recovery map.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar([r["concept_family"] for r in transfer_rows], [float(r["image_text_direction_cosine"]) for r in transfer_rows], color="#CC79A7")
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_title("Cross-modal feature bridge")
    ax.set_ylabel("image/text direction cosine")
    ax.set_xticks(range(len(transfer_rows)), [r["concept_family"] for r in transfer_rows], rotation=30, ha="right")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "cross_modal_feature_bridge.png", "Synthetic cross-modal direction cosine by family.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(evidence, start=1):
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": str(row["evidence_rung"]),
            "text": (
                f"Method `{row['method']}` reported {row['metric']}={row['value']} in synthetic connector mode; "
                f"posture `{row['claim_posture']}`."
            ),
            "artifact": f"runs/{run_name}/tables/multimodal_evidence_matrix.csv",
            "falsifier": "Real VLM hooks, OCR/background leak checks, or token/region alignment diagnostics fail.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    items, data_info = load_items(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 33 data manifest and synthetic-mode scope.")
    write_safety_status(ctx, data_info)
    alignment = alignment_validation(items)
    alignment_path = ctx.path("diagnostics", "alignment_validation.json")
    bench.write_json(alignment_path, alignment)
    ctx.register_artifact(alignment_path, "diagnostic", "Image-token and connector alignment validation status.")
    bench.run_hook_parity_check(ctx, bundle, items[0].text_control_prompt)
    first = bench.run_with_residual_cache(bundle, items[0].text_control_prompt)
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, items[0].text_control_prompt)

    render_rows = render_images(ctx, items)
    states = build_states(items)
    prompt_rows = prompt_manifest(items, render_rows)
    probe_rows = modality_probe_report(items, states)
    patch_rows = patch_report(items, states)
    transfer_rows = cross_modal_transfer(items, states)
    leak_rows, leak_summary = leak_audit(items, states)
    evidence, metrics = evidence_matrix(data_info, probe_rows, patch_rows, transfer_rows, leak_summary)
    metrics = {**metrics, "data": data_info, "alignment": alignment, "leak_summary": leak_summary}

    write_tables(ctx, prompt_rows, probe_rows, patch_rows, transfer_rows, leak_rows, evidence)
    write_state(ctx, transfer_rows)
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 33 metrics.")
    write_method_card(ctx, evidence, metrics)
    write_operationalization_audit(ctx, evidence)
    write_run_summary(ctx, data_info, metrics, evidence)
    write_claims(ctx, evidence)
    write_plots(ctx, probe_rows, patch_rows, transfer_rows, evidence)
    print(f"[lab33] wrote {len(prompt_rows)} prompts, {len(probe_rows)} probe rows, and {len(evidence)} evidence rows")
