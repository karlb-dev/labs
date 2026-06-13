"""Lab 13: Emotion geometry, read/write transfer, and affect steering.

Advanced course Group I: reusable instruments for slippery concepts. The lab
asks a deliberately narrow version of an emotionally loaded question:

    When an instruct model reads emotional text, and when it is prompted to
    write emotional text, do paired affect contrasts expose related residual
    stream directions?

The object of study is not feeling. It is an operational contrast:

    emotional text or prompt  minus  neutral matched text or prompt

for the same cause. The lab extracts directions at chat-templated final-prompt
positions and tests three things:

* DECODE: do comprehension-derived and write-intent-derived directions decode
  the paired emotion/neutral contrast on held-out examples?
* TRANSFER: does a direction extracted from reading emotion decode write-intent
  prompts, and does a write-intent direction decode reading prompts?
* CAUSAL HANDLE: does adding the input-derived direction during generation
  shift generated affect more than random, shuffled, and sentiment controls?

The favorite interpretation is fragile by design. Topic, valence, arousal,
surprise, positive calmness, prompt style, and generic sentiment all get a
chance to explain the result first. The run writes an operationalization audit
that decides whether the supported claim is an emotion-specific handle, a
broader affect/valence handle, or no defended claim.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L13"
DATA_FILE = "affect_emotion_pairs.csv"
VALENCE_FILE = "affect_valence.csv"

EMOTIONS = ("joy", "sadness", "anger", "fear")
SOURCES = ("comprehension", "generation")
FEATURE_SUFFIXES = ("emotion", "neutral")

# --prompt-set controls the target row cap per emotion. --max-examples > 0
# overrides the per-emotion cap. 0 means no cap.
PROMPT_SET_LIMITS: dict[str, dict[str, int]] = {
    "small": {"per_emotion": 3, "confounds": 4, "sentiment_pairs": 6, "steering_per_emotion": 1},
    "medium": {"per_emotion": 5, "confounds": 8, "sentiment_pairs": 12, "steering_per_emotion": 2},
    "full": {"per_emotion": 0, "confounds": 0, "sentiment_pairs": 24, "steering_per_emotion": 3},
}

TRAIN_FRACTION = 0.67
MAX_NEW_TOKENS = 48
ENGINE_MAX_CONCURRENT = 16

# Scales are fractions of the median train activation norm at the selected
# stream depth. Dose 0 is generated only once as the unsteered baseline.
STEERING_DOSES = (0.0, 0.4, 0.8, 1.2)
HEADLINE_STEERING_DOSE = 0.8

# Control budgets. These are vector-space operations, not additional model
# forwards, except for the sentiment direction cache.
N_DEPTH_RANDOM_CONTROLS = 3
N_DEPTH_SHUFFLES = 2

# Thresholds used only for the run's audit verdict. They are intentionally
# conservative and visible; students may argue with them, but not hide them.
CROSS_TRANSFER_AUC_BAR = 0.60
CONTROL_ADJUSTED_AUC_BAR = 0.05
SPECIFICITY_AUC_BAR = 0.55
CROSS_CAUSE_AUC_BAR = 0.55
SENTIMENT_COSINE_WARN = 0.70
STEERING_DELTA_BAR = 0.05

COMPREHENSION_SYSTEM = "You are a careful reader. Answer exactly as requested."
GENERATION_SYSTEM = "You write one sentence in the requested style."

# Small transparent lexicons. These are not judges. They are cheap automatic
# rulers for the plots, paired with blank hand-label columns for the writeup.
EMOTION_WORDS: dict[str, set[str]] = {
    "joy": set(
        """
        joy joyful joyous happy happiness delighted delight delightedly cheerful cheer
        cheers cheered grin grinned smile smiled smiling laughter laugh laughed
        thrilled thrill wonderful bright warm glad grateful proud celebrate celebrated
        celebration relief relieved pleasant pleased fun excited excitement hopeful hope
        """.split()
    ),
    "sadness": set(
        """
        sad sadness sorrow sorrowful grief grieving grieved mourn mourning tears tear
        lonely loneliness alone loss lost empty heavy disappointed disappointment
        painful quietly quiet sorrowfully unhappy miserable melancholy regret regretted
        ache aching heartbreak heartbroken
        """.split()
    ),
    "anger": set(
        """
        anger angry angrily furious fury rage enraged outraged unfair injustice insulted
        insulting bitter bristle bristled irritation irritated frustrating frustrated
        frustration clenched clench sharp resentful resentment annoyed annoyedly fuming
        protest protested slammed
        """.split()
    ),
    "fear": set(
        """
        fear fearful afraid anxiety anxious panic panicked terrified terror dread
        dreadful nervous danger dangerous worried worry alarm alarmed froze frozen
        trembling tense uneasy scared fright frightened risk risky threatened threat
        """.split()
    ),
}

POSITIVE_WORDS = EMOTION_WORDS["joy"] | {
    "calm", "peaceful", "pleasant", "orderly", "safe", "comfortable", "gentle",
    "steady", "quiet", "soft", "balanced", "settled",
}
NEGATIVE_WORDS = EMOTION_WORDS["sadness"] | EMOTION_WORDS["anger"] | EMOTION_WORDS["fear"]
AROUSAL_WORDS = {
    "rapidly", "quickly", "alarm", "flared", "rush", "rushed", "froze", "panic",
    "urgent", "shook", "thunder", "cheered", "furious", "terror", "excited",
    "sudden", "suddenly", "burst", "shouted", "shout", "slammed", "trembling",
}


@dataclasses.dataclass(frozen=True)
class EmotionItem:
    item_id: str
    emotion: str
    cause: str
    arousal: str
    valence: str
    content_text: str
    neutral_text: str
    generation_prompt: str
    neutral_generation_prompt: str
    confound: str = ""
    note: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "EmotionItem":
        return cls(
            item_id=str(row.get("item_id", "")).strip(),
            emotion=str(row.get("emotion", "")).strip(),
            cause=str(row.get("cause", "")).strip(),
            arousal=str(row.get("arousal", "")).strip(),
            valence=str(row.get("valence", "")).strip(),
            content_text=str(row.get("content_text", "")).strip(),
            neutral_text=str(row.get("neutral_text", "")).strip(),
            generation_prompt=str(row.get("generation_prompt", "")).strip(),
            neutral_generation_prompt=str(row.get("neutral_generation_prompt", "")).strip(),
            confound=str(row.get("confound", "")).strip(),
            note=str(row.get("note", "")).strip(),
        )


# ---------------------------------------------------------------------------
# Tiny Tier A fallback data. Science runs must use data/affect_emotion_pairs.csv.
# ---------------------------------------------------------------------------


def _builtin_smoke_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        item_id: str,
        emotion: str,
        cause: str,
        arousal: str,
        valence: str,
        content: str,
        neutral: str,
        gen: str,
        neutral_gen: str,
        *,
        confound: str = "",
        note: str = "builtin Tier A smoke row",
    ) -> None:
        rows.append({
            "item_id": item_id,
            "emotion": emotion,
            "cause": cause,
            "arousal": arousal,
            "valence": valence,
            "content_text": content,
            "neutral_text": neutral,
            "generation_prompt": gen,
            "neutral_generation_prompt": neutral_gen,
            "confound": confound,
            "note": note,
        })

    add(
        "smoke_joy_reunion", "joy", "reunion", "high", "positive",
        "Maya saw her brother step off the train and laughed with bright relief.",
        "Maya saw her brother step off the train after the scheduled trip.",
        "Write one sentence about Maya seeing her brother at the station with clear joy.",
        "Write one factual sentence about Maya seeing her brother at the station.",
    )
    add(
        "smoke_joy_exam", "joy", "exam_result", "medium", "positive",
        "The score email arrived, and Jonah grinned because he had passed.",
        "The score email arrived, and Jonah learned that he had passed.",
        "Write one sentence about Jonah passing an exam with joyful affect.",
        "Write one factual sentence about Jonah passing an exam.",
    )
    add(
        "smoke_joy_pet", "joy", "pet_return", "medium", "positive",
        "The missing dog bounded through the gate, and the family cheered.",
        "The missing dog returned through the gate while the family watched.",
        "Write one sentence about a missing dog returning in a joyful style.",
        "Write one factual sentence about a missing dog returning.",
    )
    add(
        "smoke_joy_rain", "joy", "needed_rain", "low", "positive",
        "Rain finally softened the dry garden, and Lena felt grateful and glad.",
        "Rain finally reached the dry garden after several weeks.",
        "Write one sentence about needed rain arriving with quiet joy.",
        "Write one factual sentence about needed rain arriving.",
    )

    add(
        "smoke_sadness_loss", "sadness", "bereavement", "low", "negative",
        "After the funeral, Tom folded the black coat and sat in heavy silence.",
        "After the funeral, Tom folded the black coat and sat at the table.",
        "Write one sentence about Tom after a funeral with sadness.",
        "Write one factual sentence about Tom after a funeral.",
    )
    add(
        "smoke_sadness_departure", "sadness", "friend_moves", "low", "negative",
        "When Priya's friend moved away, the apartment felt lonely and empty.",
        "When Priya's friend moved away, the apartment had one fewer resident.",
        "Write one sentence about a friend moving away with sadness.",
        "Write one factual sentence about a friend moving away.",
    )
    add(
        "smoke_sadness_letter", "sadness", "missed_chance", "medium", "negative",
        "The unopened letter explained the chance he had missed, and regret ached in him.",
        "The unopened letter explained the opportunity that was no longer available.",
        "Write one sentence about a missed chance with a sad tone.",
        "Write one factual sentence about a missed chance.",
    )
    add(
        "smoke_sadness_rain", "sadness", "ruined_picnic", "low", "negative",
        "The picnic blanket stayed packed, and the children looked quietly disappointed.",
        "The picnic blanket stayed packed because the outing was cancelled.",
        "Write one sentence about a cancelled picnic with sadness.",
        "Write one factual sentence about a cancelled picnic.",
    )

    add(
        "smoke_anger_ticket", "anger", "unfair_ticket", "high", "negative",
        "The unjust parking ticket made Rosa furious, and she slammed the notice down.",
        "The parking ticket was placed on Rosa's windshield near noon.",
        "Write one sentence about Rosa receiving an unfair ticket with anger.",
        "Write one factual sentence about Rosa receiving a ticket.",
    )
    add(
        "smoke_anger_queue", "anger", "queue_cut", "medium", "negative",
        "After the man cut the line, Omar clenched his jaw at the insult.",
        "After the man entered the line, Omar remained near the counter.",
        "Write one sentence about someone cutting in line with angry affect.",
        "Write one factual sentence about someone entering a line.",
    )
    add(
        "smoke_anger_broken_promise", "anger", "broken_promise", "medium", "negative",
        "The broken promise felt unfair, and Nina's voice sharpened with resentment.",
        "The promise was not completed by the agreed time.",
        "Write one sentence about a broken promise with anger.",
        "Write one factual sentence about a promise not being completed.",
    )
    add(
        "smoke_anger_noise", "anger", "night_noise", "medium", "negative",
        "The pounding music at midnight left Elias irritated and fuming.",
        "The music continued at midnight in the adjacent apartment.",
        "Write one sentence about midnight noise with anger.",
        "Write one factual sentence about midnight noise.",
    )

    add(
        "smoke_fear_storm", "fear", "storm", "high", "negative",
        "Thunder shook the windows, and Imani froze in sudden fear.",
        "Thunder sounded outside, and the windows moved in their frames.",
        "Write one sentence about a storm with fear.",
        "Write one factual sentence about a storm.",
    )
    add(
        "smoke_fear_hallway", "fear", "dark_hallway", "medium", "negative",
        "The dark hallway creaked, and Mateo felt anxious before taking a step.",
        "The hallway was dark, and Mateo paused before taking a step.",
        "Write one sentence about a dark hallway with fear.",
        "Write one factual sentence about a dark hallway.",
    )
    add(
        "smoke_fear_phone", "fear", "late_call", "medium", "negative",
        "The late-night phone call made Sara's stomach tighten with worry.",
        "The phone rang late at night while Sara was awake.",
        "Write one sentence about a late-night phone call with fear.",
        "Write one factual sentence about a late-night phone call.",
    )
    add(
        "smoke_fear_exam", "fear", "medical_test", "low", "negative",
        "Before the test results arrived, Leon waited in tense dread.",
        "Before the test results arrived, Leon waited in the clinic.",
        "Write one sentence about waiting for medical test results with fear.",
        "Write one factual sentence about waiting for medical test results.",
    )

    add(
        "smoke_confound_surprise_neutral", "fear", "surprising_schedule_change", "high", "neutral",
        "The red card in the envelope unexpectedly changed the meeting room to 4B.",
        "The card in the envelope changed the meeting room to 4B.",
        "Write one sentence about an unexpected meeting-room change without emotional language.",
        "Write one factual sentence about a meeting-room change.",
        confound="surprising_neutral",
    )
    add(
        "smoke_confound_positive_calm", "joy", "calm_garden", "low", "positive",
        "The garden path was pleasant, orderly, quiet, and calm.",
        "The garden path was paved, bordered, and swept.",
        "Write one calm positive sentence about a garden path without excitement.",
        "Write one factual sentence about a garden path.",
        confound="positive_calm",
    )
    add(
        "smoke_confound_high_arousal_neutral", "anger", "machine_alert", "high", "neutral",
        "The machine alarm flashed rapidly because the scheduled test began.",
        "The machine signal appeared because the scheduled test began.",
        "Write one high-arousal but neutral sentence about a machine test.",
        "Write one factual sentence about a machine test.",
        confound="high_arousal_neutral",
    )
    add(
        "smoke_confound_negative_calm", "sadness", "rain_delay", "low", "negative",
        "The delay was inconvenient, dull, and quietly unpleasant, but no one was upset.",
        "The delay lasted for forty minutes before boarding resumed.",
        "Write one mildly negative but unemotional sentence about a delay.",
        "Write one factual sentence about a delay.",
        confound="negative_calm",
    )
    return rows


def _builtin_valence_pairs() -> list[tuple[str, str]]:
    return [
        ("The room felt warm, safe, and pleasant.", "The room felt cold, unsafe, and unpleasant."),
        ("The team celebrated the successful launch.", "The team regretted the failed launch."),
        ("The message brought relief and gratitude.", "The message brought worry and disappointment."),
        ("The walk was peaceful and comfortable.", "The walk was tense and uncomfortable."),
        ("The result was welcome and hopeful.", "The result was unwelcome and discouraging."),
        ("The dinner ended with laughter and smiles.", "The dinner ended with tears and silence."),
        ("The plan made everyone glad.", "The plan made everyone anxious."),
        ("The letter was cheerful and kind.", "The letter was bitter and frightening."),
    ]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def rounded(x: Any, ndigits: int = 4) -> Any:
    try:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return round(float(x), ndigits)
    except Exception:
        pass
    return x


def none_if_nan(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def safe_fmean(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(statistics.fmean(finite)) if finite else default


def safe_median(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(statistics.median(finite)) if finite else default


def safe_stderr(vals: Sequence[float]) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if len(finite) <= 1:
        return float("nan")
    return float(statistics.stdev(finite) / math.sqrt(len(finite)))


def finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            vals.append(float(val))
    return vals


def auc_from_scores(pos: Sequence[float], neg: Sequence[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def emotion_counts(text: str) -> dict[str, int]:
    toks = words(text)
    return {emo: sum(1 for w in toks if w in lex) for emo, lex in EMOTION_WORDS.items()}


def target_margin(text: str, emotion: str) -> tuple[float, int, int]:
    counts = emotion_counts(text)
    target = counts.get(emotion, 0)
    other = max((v for k, v in counts.items() if k != emotion), default=0)
    length_norm = math.sqrt(max(1, len(words(text))))
    return (target - other) / length_norm, target, other


def valence_score(text: str) -> float:
    toks = words(text)
    pos = sum(1 for w in toks if w in POSITIVE_WORDS)
    neg = sum(1 for w in toks if w in NEGATIVE_WORDS)
    return (pos - neg) / math.sqrt(max(1, len(toks)))


def arousal_score(text: str) -> float:
    toks = words(text)
    return sum(1 for w in toks if w in AROUSAL_WORDS) / math.sqrt(max(1, len(toks)))


def text_quality_stats(text: str) -> dict[str, Any]:
    toks = words(text)
    if not toks:
        return {
            "word_count": 0,
            "distinct_1": 0.0,
            "repetition_rate": 0.0,
            "empty_output": True,
        }
    distinct_1 = len(set(toks)) / len(toks)
    repeated_bigram = 0
    bigrams = list(zip(toks, toks[1:]))
    seen_bigrams: set[tuple[str, str]] = set()
    for bg in bigrams:
        if bg in seen_bigrams:
            repeated_bigram += 1
        seen_bigrams.add(bg)
    repetition_rate = repeated_bigram / max(1, len(bigrams))
    return {
        "word_count": len(toks),
        "distinct_1": rounded(distinct_1),
        "repetition_rate": rounded(repetition_rate),
        "empty_output": False,
    }


def kl_bits(base_logits: Any, shifted_logits: Any) -> float:
    """KL(P_base || P_shifted) in bits for next-token side-effect reporting."""
    import torch

    p_log = torch.nn.functional.log_softmax(base_logits.float(), dim=-1)
    q_log = torch.nn.functional.log_softmax(shifted_logits.float(), dim=-1)
    p = p_log.exp()
    return float((p * (p_log - q_log)).sum() / math.log(2.0))


def unit(v: Any) -> Any:
    norm = v.norm().clamp_min(1e-9)
    if not bool(norm.isfinite()):
        raise RuntimeError("Direction norm was not finite.")
    return v / norm


def cosine(a: Any, b: Any) -> float:
    denom = (a.norm() * b.norm()).clamp_min(1e-9)
    return float((a @ b) / denom)


def random_unit(d_model: int, seed: int) -> Any:
    import torch

    gen = torch.Generator().manual_seed(seed)
    return unit(torch.randn(d_model, generator=gen))


def maybe_negate_sentiment_for_emotion(sentiment_direction: Any | None, emotion: str) -> Any | None:
    if sentiment_direction is None:
        return None
    return sentiment_direction if emotion == "joy" else -sentiment_direction


# ---------------------------------------------------------------------------
# Data loading and prompt rendering
# ---------------------------------------------------------------------------


def _manifest_expected_hash(path: pathlib.Path) -> tuple[str | None, str]:
    manifest_path = path.parent / "MANIFEST.json"
    if not manifest_path.exists():
        return None, "data/MANIFEST.json not found"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"data/MANIFEST.json unreadable: {exc}"
    candidates: list[Any] = []
    if isinstance(manifest, dict):
        candidates.extend([
            manifest.get(path.name),
            manifest.get(str(path)),
            manifest.get("files", {}).get(path.name) if isinstance(manifest.get("files"), dict) else None,
        ])
    for entry in candidates:
        if isinstance(entry, str):
            return entry, "found string entry"
        if isinstance(entry, dict):
            for key in ("sha256", "hash", "sha256_hex"):
                val = entry.get(key)
                if isinstance(val, str):
                    return val, f"found {key} entry"
    return None, f"no usable sha256 entry for {path.name}"


def load_items(args: Any) -> tuple[list[EmotionItem], list[EmotionItem], dict[str, Any]]:
    path = bench.COURSE_ROOT / "data" / DATA_FILE
    expected_sha, manifest_note = _manifest_expected_hash(path)
    data_source = "frozen_csv"
    actual_sha: str | None = None
    manifest_ok: bool | None = None

    if path.exists():
        actual_sha = bench.sha256_file(path)
        manifest_ok = (expected_sha == actual_sha) if expected_sha else None
        with path.open(newline="", encoding="utf-8") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
    else:
        tier = str(getattr(args, "tier", "")).lower()
        if tier != "a":
            raise RuntimeError(
                f"Frozen dataset missing: {path}. Science runs must use the committed "
                "data/affect_emotion_pairs.csv generated by data/make_emotion_pairs.py."
            )
        print("[lab13] frozen emotion CSV missing; using builtin Tier A smoke fallback. "
              "This run is plumbing only and should not enter the claim ledger.")
        rows = _builtin_smoke_rows()
        data_source = "builtin_tier_a_smoke_fallback"
        actual_sha = hashlib.sha256("\n".join(r["item_id"] for r in rows).encode("utf-8")).hexdigest()
        manifest_note = "frozen CSV absent; builtin fallback has no manifest entry"
        manifest_ok = False

    required = {
        "item_id", "emotion", "cause", "arousal", "valence", "content_text", "neutral_text",
        "generation_prompt", "neutral_generation_prompt",
    }
    if rows:
        missing = sorted(required - set(rows[0]))
        if missing:
            raise ValueError(f"{DATA_FILE} is missing required columns: {missing}")
    all_items = [EmotionItem.from_row(row) for row in rows]
    bad = [it.item_id or "(blank)" for it in all_items if not it.item_id or not it.emotion]
    if bad:
        raise ValueError(f"{DATA_FILE} contains rows without item_id/emotion: {bad[:10]}")

    selected = tuple(e.strip() for e in str(getattr(args, "emotions", "")).split(",") if e.strip()) or EMOTIONS
    target_emotions = {it.emotion for it in all_items if not it.confound}
    unknown = set(selected) - target_emotions
    if unknown:
        raise ValueError(f"--emotions included unknown labels: {sorted(unknown)}; available: {sorted(target_emotions)}")

    limits = PROMPT_SET_LIMITS.get(str(getattr(args, "prompt_set", "small")))
    if limits is None:
        raise ValueError("Lab 13 uses --prompt-set small|medium|full; custom files are not used here.")
    per_emotion = limits["per_emotion"]
    confound_cap = limits["confounds"]
    max_examples = int(getattr(args, "max_examples", 0) or 0)
    if max_examples > 0:
        per_emotion = max_examples
        confound_cap = min(confound_cap or max_examples, max_examples)

    targets: list[EmotionItem] = []
    by_emotion: dict[str, list[EmotionItem]] = defaultdict(list)
    for item in all_items:
        if item.emotion in selected and not item.confound:
            by_emotion[item.emotion].append(item)
    for emotion in selected:
        rows_for_emotion = by_emotion[emotion]
        if per_emotion > 0:
            rows_for_emotion = rows_for_emotion[:per_emotion]
        if len(rows_for_emotion) < 2:
            raise RuntimeError(
                f"Lab 13 needs at least two target items for {emotion!r}; got {len(rows_for_emotion)}. "
                "Use a larger --max-examples or prompt set."
            )
        targets.extend(rows_for_emotion)

    confounds = [it for it in all_items if it.confound]
    if confound_cap > 0:
        confounds = confounds[:confound_cap]

    counts_by_emotion = {e: sum(1 for it in targets if it.emotion == e) for e in selected}
    causes_by_emotion = {
        e: sorted({it.cause for it in targets if it.emotion == e})
        for e in selected
    }
    info = {
        "data_file": DATA_FILE,
        "data_path": str(path),
        "data_source": data_source,
        "data_sha256": actual_sha,
        "manifest_expected_sha256": expected_sha,
        "manifest_note": manifest_note,
        "manifest_ok": manifest_ok,
        "selected_emotions": list(selected),
        "per_emotion_cap": per_emotion,
        "confound_cap": confound_cap,
        "n_target_items": len(targets),
        "n_confound_items": len(confounds),
        "counts_by_emotion": counts_by_emotion,
        "causes_by_emotion": causes_by_emotion,
        "confound_types": sorted({it.confound for it in confounds if it.confound}),
        "tier_a_fallback_warning": data_source.startswith("builtin"),
    }
    return targets, confounds, info


def load_valence_pairs(args: Any) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    path = bench.COURSE_ROOT / "data" / VALENCE_FILE
    expected_sha, manifest_note = _manifest_expected_hash(path)
    if path.exists():
        positives: list[str] = []
        negatives: list[str] = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = str(row.get("label", "")).strip().lower()
                statement = str(row.get("statement", "")).strip()
                if not statement:
                    continue
                if label in {"1", "positive", "pos", "+"}:
                    positives.append(statement)
                elif label in {"0", "negative", "neg", "-"}:
                    negatives.append(statement)
        n = min(len(positives), len(negatives))
        pairs = list(zip(positives[:n], negatives[:n]))
        info = {
            "valence_file": VALENCE_FILE,
            "valence_path": str(path),
            "valence_source": "frozen_csv",
            "valence_sha256": bench.sha256_file(path),
            "manifest_expected_sha256": expected_sha,
            "manifest_note": manifest_note,
            "manifest_ok": (bench.sha256_file(path) == expected_sha) if expected_sha else None,
            "n_pairs_available": n,
        }
        return pairs, info

    tier = str(getattr(args, "tier", "")).lower()
    if tier != "a":
        raise RuntimeError(
            f"Sentiment-control dataset missing: {path}. Science runs need data/{VALENCE_FILE} "
            "so the emotion directions can be audited against generic sentiment."
        )
    pairs = _builtin_valence_pairs()
    return pairs, {
        "valence_file": VALENCE_FILE,
        "valence_path": str(path),
        "valence_source": "builtin_tier_a_smoke_fallback",
        "valence_sha256": hashlib.sha256("\n".join(p + n for p, n in pairs).encode("utf-8")).hexdigest(),
        "manifest_expected_sha256": expected_sha,
        "manifest_note": "frozen valence CSV absent; builtin fallback has no manifest entry",
        "manifest_ok": False,
        "n_pairs_available": len(pairs),
    }


def comprehension_user_message(text: str) -> str:
    return (
        "Read this sentence and keep its emotional content in mind.\n"
        f"Sentence: {text}\n"
        "Reply with exactly one word: done"
    )


def render_comprehension(bundle: bench.ModelBundle, text: str) -> str:
    return bench.apply_chat_template(
        bundle,
        comprehension_user_message(text),
        system=COMPREHENSION_SYSTEM,
        add_generation_prompt=True,
    )


def render_generation(bundle: bench.ModelBundle, prompt: str) -> str:
    return bench.apply_chat_template(
        bundle,
        prompt,
        system=GENERATION_SYSTEM,
        add_generation_prompt=True,
    )


def feature_key(source: str, suffix: str) -> str:
    return f"{source}_{suffix}"


def last_token_streams(bundle: bench.ModelBundle, templated_prompt: str) -> Any:
    cap = bench.run_with_residual_cache(bundle, templated_prompt, add_special_tokens=False)
    return cap.streams[:, -1, :]


def run_rendered_hook_parity_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    templated_prompt: str,
) -> dict[str, Any]:
    """Hook parity for already-rendered chat prompts.

    The shared bench hook-parity helper tokenizes raw prompts with the tokenizer
    defaults. Lab 13 caches chat-templated strings with
    ``add_special_tokens=False`` so generation, feature extraction, and this
    parity check all inspect the exact same token sequence.
    """
    import torch

    block_outputs: dict[int, Any] = {}

    def make_hook(idx: int):
        def hook(module: Any, hook_args: tuple, output: Any) -> None:
            del module, hook_args
            out = output[0] if isinstance(output, tuple) else output
            block_outputs[idx] = bench.tensor_cpu_float(out)

        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(bundle.blocks)]
    try:
        capture = bench.run_with_residual_cache(bundle, templated_prompt, add_special_tokens=False)
    finally:
        for handle in handles:
            handle.remove()

    n_layers = bundle.anatomy.n_layers
    by_layer_rows: list[dict[str, Any]] = []
    max_diff = 0.0
    max_mean_diff = 0.0
    compared = 0
    missing_layers: list[int] = []
    for k in range(n_layers):
        if k not in block_outputs:
            missing_layers.append(k)
            continue
        hook_out = block_outputs[k][0]
        expected = capture.streams[k + 1]
        abs_diff = (hook_out - expected).abs()
        layer_max = float(abs_diff.max())
        layer_mean = float(abs_diff.mean())
        max_diff = max(max_diff, layer_max)
        max_mean_diff = max(max_mean_diff, layer_mean)
        compared += 1
        by_layer_rows.append({
            "layer": k,
            "max_abs_diff": layer_max,
            "mean_abs_diff": layer_mean,
            "hook_l2": float(hook_out.norm()),
            "expected_l2": float(expected.norm()),
            "shape": "x".join(str(x) for x in hook_out.shape),
            "ok_at_tolerance": layer_max <= float(ctx.args.hook_tolerance),
        })

    by_layer_path = ctx.path("diagnostics", "hook_parity_by_layer.csv")
    bench.write_csv_with_context(ctx, by_layer_path, by_layer_rows)
    ctx.register_artifact(by_layer_path, "diagnostic", "Rendered-prompt hook parity by layer for Lab 13 chat-template capture.")

    ok = (compared == n_layers and not missing_layers and max_diff <= float(ctx.args.hook_tolerance))
    result = {
        "ok": ok,
        "prompt_rendered": True,
        "add_special_tokens": False,
        "compared_layers": compared,
        "n_layers": n_layers,
        "missing_layers": missing_layers,
        "max_abs_diff": max_diff,
        "max_mean_abs_diff": max_mean_diff,
        "tolerance": float(ctx.args.hook_tolerance),
        "token_count": len(capture.input_ids),
        "note": "Lab 13 parity uses the same already-rendered chat prompt and add_special_tokens=False convention as feature caching and generation.",
    }
    result_path = ctx.path("diagnostics", "hook_parity.json")
    bench.write_json(result_path, result)
    ctx.register_artifact(result_path, "diagnostic", "Rendered-prompt hook parity self-check for Lab 13.")

    status = "OK" if ok else "FAILED"
    print(f"[lab13] rendered hook parity: {status} (max abs diff = {max_diff:.2e})")
    if not ok and not ctx.args.allow_hook_mismatch:
        raise RuntimeError(
            "Rendered-prompt hook parity failed. See diagnostics/hook_parity.json and "
            "diagnostics/hook_parity_by_layer.csv. Pass --allow-hook-mismatch only for architecture bring-up."
        )
    return result


def make_split(items: Sequence[EmotionItem], seed: int) -> dict[str, bool]:
    """Deterministic train/eval split grouped by emotion and cause.

    Cause grouping prevents the common leakage where one prompt about a cause
    trains the direction and a paraphrase of the same cause evaluates it.
    """
    split: dict[str, bool] = {}
    by_emotion: dict[str, dict[str, list[EmotionItem]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_emotion[item.emotion][item.cause].append(item)
    for emotion, cause_map in by_emotion.items():
        causes = sorted(cause_map, key=lambda c: stable_hash_int(f"{seed}:{emotion}:{c}"))
        n_train = int(round(TRAIN_FRACTION * len(causes)))
        n_train = max(1, min(len(causes) - 1, n_train)) if len(causes) > 1 else 1
        train_causes = set(causes[:n_train])
        for cause, rows in cause_map.items():
            for item in rows:
                split[item.item_id] = cause in train_causes
    return split


def cache_features(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    targets: Sequence[EmotionItem],
    confounds: Sequence[EmotionItem],
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    all_items = list(targets) + list(confounds)
    report_every = max(1, len(all_items) // 5)
    for i, item in enumerate(all_items):
        templated = {
            "comprehension_emotion": render_comprehension(bundle, item.content_text),
            "comprehension_neutral": render_comprehension(bundle, item.neutral_text),
            "generation_emotion": render_generation(bundle, item.generation_prompt),
            "generation_neutral": render_generation(bundle, item.neutral_generation_prompt),
        }
        features[item.item_id] = {key: last_token_streams(bundle, prompt) for key, prompt in templated.items()}
        for key, prompt in templated.items():
            ids = bundle.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            rows.append({
                "item_id": item.item_id,
                "emotion": item.emotion,
                "cause": item.cause,
                "arousal": item.arousal,
                "valence": item.valence,
                "confound": item.confound,
                "feature_key": key,
                "n_tokens": len(ids),
                "final_token_id": ids[-1] if ids else "",
                "final_token_text": bundle.tokenizer.decode([ids[-1]]) if ids else "",
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
                "prompt_tail": prompt[-180:].replace("\n", "\\n"),
            })
        if (i + 1) % report_every == 0:
            print(f"[lab13] cached {i + 1}/{len(all_items)} items")

    path = ctx.path("diagnostics", "prompt_render_audit.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Chat-rendered prompt token counts and final-token audit.")
    return features


# ---------------------------------------------------------------------------
# Directions and decoding evaluation
# ---------------------------------------------------------------------------


def paired_delta(features: Mapping[str, dict[str, Any]], item: EmotionItem, source: str, depth: int) -> Any:
    return features[item.item_id][feature_key(source, "emotion")][depth] - features[item.item_id][feature_key(source, "neutral")][depth]


def direction_for(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    source: str,
    emotion: str,
    depth: int,
    *,
    train_only: bool = True,
) -> Any | None:
    import torch

    deltas = [
        paired_delta(features, item, source, depth)
        for item in items
        if item.emotion == emotion and ((not train_only) or split[item.item_id])
    ]
    if not deltas:
        return None
    return unit(torch.stack(deltas).mean(dim=0))


def shuffled_direction_for(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    source: str,
    emotion: str,
    depth: int,
    seed: int,
) -> Any | None:
    import torch

    deltas = [
        paired_delta(features, item, source, depth)
        for item in items
        if item.emotion == emotion and split[item.item_id]
    ]
    if len(deltas) < 2:
        return direction_for(items, features, split, source, emotion, depth)
    order = sorted(range(len(deltas)), key=lambda i: stable_hash_int(f"{seed}:shuffle:{emotion}:{source}:{i}"))
    flip = set(order[: len(order) // 2])
    signed = [(-d if i in flip else d) for i, d in enumerate(deltas)]
    return unit(torch.stack(signed).mean(dim=0))


def projection_scores(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    direction: Any,
    target_source: str,
    emotion: str,
    depth: int,
    *,
    train: bool,
    cause: str | None = None,
) -> tuple[list[float], list[float]]:
    pos: list[float] = []
    neg: list[float] = []
    for item in items:
        if item.emotion != emotion:
            continue
        if cause is not None and item.cause != cause:
            continue
        if split[item.item_id] != train:
            continue
        pos.append(float(features[item.item_id][feature_key(target_source, "emotion")][depth] @ direction))
        neg.append(float(features[item.item_id][feature_key(target_source, "neutral")][depth] @ direction))
    return pos, neg


def delta_scores_for_items(
    rows: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    direction: Any,
    target_source: str,
    depth: int,
) -> list[float]:
    return [float(paired_delta(features, item, target_source, depth) @ direction) for item in rows]


def evaluate_direction(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    direction: Any,
    target_source: str,
    emotion: str,
    depth: int,
    *,
    train: bool = False,
    cause: str | None = None,
) -> dict[str, Any]:
    pos, neg = projection_scores(
        items, features, split, direction, target_source, emotion, depth, train=train, cause=cause
    )
    return {
        "auc": auc_from_scores(pos, neg),
        "mean_pos": safe_fmean(pos),
        "mean_neg": safe_fmean(neg),
        "margin": safe_fmean(pos) - safe_fmean(neg),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def orient_on_train(
    direction: Any,
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    source: str,
    emotion: str,
    depth: int,
) -> Any:
    pos, neg = projection_scores(items, features, split, direction, source, emotion, depth, train=True)
    if pos and neg and safe_fmean(pos, 0.0) < safe_fmean(neg, 0.0):
        return -direction
    return direction


def build_directions_at_depth(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    depth: int,
) -> dict[tuple[str, str], Any]:
    dirs: dict[tuple[str, str], Any] = {}
    for source in SOURCES:
        for emotion in sorted({item.emotion for item in items}):
            d = direction_for(items, features, split, source, emotion, depth)
            if d is not None:
                dirs[(source, emotion)] = orient_on_train(d, items, features, split, source, emotion, depth)
    return dirs


def summarize_real_at_depth(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    depth: int,
    *,
    train: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate real directions at one depth on either train or eval rows.

    Directions are always FIT on train rows. The ``train`` flag only chooses
    the rows used for evaluation. Depth selection uses train evaluation, while
    the transfer table and headline metrics use eval evaluation. This avoids
    the easy but poisonous mistake of picking the best depth on the held-out
    rows and then reporting those same rows as evidence.
    """
    rows: list[dict[str, Any]] = []
    cross: list[float] = []
    same: list[float] = []
    dirs = build_directions_at_depth(items, features, split, depth)
    eval_split = "train" if train else "eval"
    for (source, emotion), d in dirs.items():
        for target_source in SOURCES:
            ev = evaluate_direction(items, features, split, d, target_source, emotion, depth, train=train)
            auc = ev["auc"]
            if math.isfinite(float(auc)):
                if source == target_source:
                    same.append(float(auc))
                else:
                    cross.append(float(auc))
            rows.append({
                "depth": depth,
                "eval_split": eval_split,
                "direction_kind": "real",
                "control_seed": "",
                "direction_source": source,
                "eval_target": target_source,
                "emotion": emotion,
                "auc": rounded(ev["auc"]),
                "selectivity_vs_chance": rounded(ev["auc"] - 0.5) if math.isfinite(ev["auc"]) else "",
                "mean_pos": rounded(ev["mean_pos"]),
                "mean_neg": rounded(ev["mean_neg"]),
                "margin": rounded(ev["margin"]),
                "n_pos": ev["n_pos"],
                "n_neg": ev["n_neg"],
            })
    summary = {
        "real_cross_auc": safe_fmean(cross),
        "real_same_source_auc": safe_fmean(same),
        "real_cross_auc_stderr": safe_stderr(cross),
        "n_real_cross_cells": len(cross),
    }
    return rows, summary


def summarize_control_cross_auc(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    depth: int,
    d_model: int,
    seed: int,
    *,
    train: bool = False,
) -> dict[str, float]:
    shuffled_vals: list[float] = []
    random_vals: list[float] = []
    emotions = sorted({item.emotion for item in items})
    for source in SOURCES:
        for emotion in emotions:
            for j in range(N_DEPTH_SHUFFLES):
                d = shuffled_direction_for(items, features, split, source, emotion, depth, seed + 101 * j)
                if d is None:
                    continue
                d = orient_on_train(d, items, features, split, source, emotion, depth)
                for target_source in SOURCES:
                    if target_source == source:
                        continue
                    ev = evaluate_direction(items, features, split, d, target_source, emotion, depth, train=train)
                    if math.isfinite(ev["auc"]):
                        shuffled_vals.append(float(ev["auc"]))
            for j in range(N_DEPTH_RANDOM_CONTROLS):
                d = random_unit(d_model, seed + stable_hash_int(f"depth-random:{depth}:{source}:{emotion}:{j}") % 10_000_000)
                d = orient_on_train(d, items, features, split, source, emotion, depth)
                for target_source in SOURCES:
                    if target_source == source:
                        continue
                    ev = evaluate_direction(items, features, split, d, target_source, emotion, depth, train=train)
                    if math.isfinite(ev["auc"]):
                        random_vals.append(float(ev["auc"]))
    return {
        "shuffled_cross_auc": safe_fmean(shuffled_vals),
        "random_cross_auc": safe_fmean(random_vals),
        "n_shuffled_cross_cells": len(shuffled_vals),
        "n_random_cross_cells": len(random_vals),
    }


def scan_transfer_depths(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    n_depths: int,
    d_model: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    transfer_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    for depth in range(1, n_depths):
        eval_rows, eval_summary = summarize_real_at_depth(items, features, split, depth, train=False)
        train_rows, train_summary = summarize_real_at_depth(items, features, split, depth, train=True)
        # The main transfer table includes eval rows first for backward-friendly plotting,
        # plus train rows so the depth decision is auditable without opening JSON.
        transfer_rows.extend(eval_rows)
        transfer_rows.extend(train_rows)

        eval_ctrl = summarize_control_cross_auc(items, features, split, depth, d_model, seed, train=False)
        train_ctrl = summarize_control_cross_auc(items, features, split, depth, d_model, seed, train=True)

        eval_strongest = max(
            [v for v in (eval_ctrl["shuffled_cross_auc"], eval_ctrl["random_cross_auc"], 0.5) if math.isfinite(float(v))],
            default=0.5,
        )
        train_strongest = max(
            [v for v in (train_ctrl["shuffled_cross_auc"], train_ctrl["random_cross_auc"], 0.5) if math.isfinite(float(v))],
            default=0.5,
        )
        eval_adjusted = eval_summary["real_cross_auc"] - eval_strongest if math.isfinite(eval_summary["real_cross_auc"]) else float("nan")
        train_adjusted = train_summary["real_cross_auc"] - train_strongest if math.isfinite(train_summary["real_cross_auc"]) else float("nan")
        depth_rows.append({
            "depth": depth,
            "selection_metric": "train_control_adjusted_cross_auc",
            "train_real_cross_auc": rounded(train_summary["real_cross_auc"]),
            "train_real_cross_auc_stderr": rounded(train_summary["real_cross_auc_stderr"]),
            "train_real_same_source_auc": rounded(train_summary["real_same_source_auc"]),
            "train_shuffled_cross_auc": rounded(train_ctrl["shuffled_cross_auc"]),
            "train_random_cross_auc": rounded(train_ctrl["random_cross_auc"]),
            "train_strongest_control_cross_auc": rounded(train_strongest),
            "train_control_adjusted_cross_auc": rounded(train_adjusted),
            "real_cross_auc": rounded(eval_summary["real_cross_auc"]),
            "real_cross_auc_stderr": rounded(eval_summary["real_cross_auc_stderr"]),
            "real_same_source_auc": rounded(eval_summary["real_same_source_auc"]),
            "shuffled_cross_auc": rounded(eval_ctrl["shuffled_cross_auc"]),
            "random_cross_auc": rounded(eval_ctrl["random_cross_auc"]),
            "strongest_control_cross_auc": rounded(eval_strongest),
            "control_adjusted_cross_auc": rounded(eval_adjusted),
            "n_real_cross_cells": eval_summary["n_real_cross_cells"],
            "n_shuffled_cross_cells": eval_ctrl["n_shuffled_cross_cells"],
            "n_random_cross_cells": eval_ctrl["n_random_cross_cells"],
        })

    def score(row: Mapping[str, Any]) -> float:
        val = row.get("train_control_adjusted_cross_auc")
        return float(val) if isinstance(val, (int, float)) and math.isfinite(float(val)) else -999.0

    valid_rows = [r for r in depth_rows if score(r) > -999.0]
    best_depth = int(max(valid_rows, key=score)["depth"]) if valid_rows else max(1, (n_depths - 1) // 2)
    return transfer_rows, depth_rows, best_depth


def add_controls_at_depth(
    rows: list[dict[str, Any]],
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    depth: int,
    seed: int,
    sentiment_direction: Any | None,
    d_model: int,
) -> None:
    emotions = sorted({item.emotion for item in items})
    for source in SOURCES:
        for emotion in emotions:
            controls: list[tuple[str, str, Any | None, str]] = []
            for j in range(N_DEPTH_SHUFFLES):
                d = shuffled_direction_for(items, features, split, source, emotion, depth, seed + 211 * j)
                if d is not None:
                    d = orient_on_train(d, items, features, split, source, emotion, depth)
                controls.append(("shuffled", source, d, str(j)))
            for j in range(N_DEPTH_RANDOM_CONTROLS):
                d = random_unit(d_model, seed + stable_hash_int(f"detail-random:{source}:{emotion}:{depth}:{j}") % 10_000_000)
                d = orient_on_train(d, items, features, split, source, emotion, depth)
                controls.append(("random", source, d, str(j)))
            signed_sentiment = maybe_negate_sentiment_for_emotion(sentiment_direction, emotion)
            if signed_sentiment is not None:
                controls.append(("sentiment_control", "lab7_style_signed_sentiment", signed_sentiment, ""))
            for kind, direction_source, d, control_seed in controls:
                if d is None:
                    continue
                for target_source in SOURCES:
                    ev = evaluate_direction(items, features, split, d, target_source, emotion, depth, train=False)
                    rows.append({
                        "depth": depth,
                        "eval_split": "eval",
                        "direction_kind": kind,
                        "control_seed": control_seed,
                        "direction_source": direction_source,
                        "eval_target": target_source,
                        "emotion": emotion,
                        "auc": rounded(ev["auc"]),
                        "selectivity_vs_chance": rounded(ev["auc"] - 0.5) if math.isfinite(ev["auc"]) else "",
                        "mean_pos": rounded(ev["mean_pos"]),
                        "mean_neg": rounded(ev["mean_neg"]),
                        "margin": rounded(ev["margin"]),
                        "n_pos": ev["n_pos"],
                        "n_neg": ev["n_neg"],
                    })


def build_sentiment_direction(
    args: Any,
    bundle: bench.ModelBundle,
    depth: int,
    cap_pairs: int,
) -> tuple[Any | None, int, dict[str, Any]]:
    import torch

    pairs, info = load_valence_pairs(args)
    n = min(cap_pairs, len(pairs)) if cap_pairs > 0 else len(pairs)
    if n < 2:
        info["n_pairs_used"] = 0
        return None, 0, info
    diffs = []
    for pos, neg in pairs[:n]:
        pos_vec = last_token_streams(bundle, render_comprehension(bundle, pos))[depth]
        neg_vec = last_token_streams(bundle, render_comprehension(bundle, neg))[depth]
        diffs.append(pos_vec - neg_vec)
    info["n_pairs_used"] = n
    info["direction_semantics"] = "positive-valence minus negative-valence, comprehension prompt state"
    return unit(torch.stack(diffs).mean(dim=0)), n, info


def cross_cause_generalization(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    depth: int,
) -> list[dict[str, Any]]:
    del split  # This analysis uses its own leave-one-cause-out split.
    rows: list[dict[str, Any]] = []
    for emotion in sorted({item.emotion for item in items}):
        causes = sorted({item.cause for item in items if item.emotion == emotion})
        for held_cause in causes:
            pseudo_split = {item.item_id: item.emotion == emotion and item.cause != held_cause for item in items}
            n_train_causes = sum(1 for c in causes if c != held_cause)
            for source in SOURCES:
                d = direction_for(items, features, pseudo_split, source, emotion, depth)
                if d is None:
                    continue
                d = orient_on_train(d, items, features, pseudo_split, source, emotion, depth)
                eval_split = {item.item_id: False for item in items}
                for target_source in SOURCES:
                    ev = evaluate_direction(
                        items, features, eval_split, d, target_source, emotion, depth,
                        train=False, cause=held_cause,
                    )
                    rows.append({
                        "emotion": emotion,
                        "held_out_cause": held_cause,
                        "direction_source": source,
                        "eval_target": target_source,
                        "depth": depth,
                        "auc": rounded(ev["auc"]),
                        "margin": rounded(ev["margin"]),
                        "n_pos": ev["n_pos"],
                        "n_neg": ev["n_neg"],
                        "n_train_causes": n_train_causes,
                    })
    return rows


def emotion_specificity_rows(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    dirs: Mapping[tuple[str, str], Any],
    depth: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_items = [it for it in items if not split[it.item_id]]
    train_items = [it for it in items if split[it.item_id]]
    # Tiny smoke runs sometimes leave one eval item per emotion; if a pool is
    # empty, the AUC is reported as blank rather than silently using train.
    for (source, emotion), d in dirs.items():
        for target_source in SOURCES:
            target = [it for it in eval_items if it.emotion == emotion]
            if not target:
                target = [it for it in train_items if it.emotion == emotion]
                used_split = "train_fallback"
            else:
                used_split = "eval"
            pools = {
                "all_other_emotions": [it for it in eval_items if it.emotion != emotion],
                "same_valence_other_emotions": [
                    it for it in eval_items
                    if it.emotion != emotion and it.valence and target and it.valence == target[0].valence
                ],
                "same_arousal_other_emotions": [
                    it for it in eval_items
                    if it.emotion != emotion and it.arousal and target and it.arousal == target[0].arousal
                ],
            }
            target_scores = delta_scores_for_items(target, features, d, target_source, depth)
            for pool_name, pool_items in pools.items():
                other_scores = delta_scores_for_items(pool_items, features, d, target_source, depth)
                rows.append({
                    "direction_source": source,
                    "eval_target": target_source,
                    "emotion": emotion,
                    "comparison_pool": pool_name,
                    "depth": depth,
                    "specificity_auc": rounded(auc_from_scores(target_scores, other_scores)),
                    "target_delta_mean": rounded(safe_fmean(target_scores)),
                    "comparison_delta_mean": rounded(safe_fmean(other_scores)),
                    "margin": rounded(safe_fmean(target_scores) - safe_fmean(other_scores)),
                    "n_target": len(target_scores),
                    "n_comparison": len(other_scores),
                    "target_split_used": used_split,
                })
    return rows


def cosine_rows(
    dirs: Mapping[tuple[str, str], Any],
    sentiment_direction: Any | None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    vectors: dict[str, Any] = {f"{source}_{emotion}": d for (source, emotion), d in dirs.items()}
    if sentiment_direction is not None:
        vectors["sentiment_lab7_style_positive_minus_negative"] = sentiment_direction
    labels = sorted(vectors)
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                rows.append({
                    "direction_a": a,
                    "direction_b": b,
                    "cosine": rounded(cosine(vectors[a], vectors[b])),
                    "abs_cosine": rounded(abs(cosine(vectors[a], vectors[b]))),
                })
    return rows, labels, vectors


def confound_projection_audit(
    confounds: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    dirs: Mapping[tuple[str, str], Any],
    depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for item in confounds:
        for (source, emotion), d in dirs.items():
            pos = float(features[item.item_id][feature_key(source, "emotion")][depth] @ d)
            neg = float(features[item.item_id][feature_key(source, "neutral")][depth] @ d)
            delta = pos - neg
            rows.append({
                "item_id": item.item_id,
                "confound": item.confound,
                "label": item.emotion,
                "cause": item.cause,
                "arousal": item.arousal,
                "valence": item.valence,
                "direction_source": source,
                "direction_emotion": emotion,
                "direction": f"{source}_{emotion}",
                "projection_delta": rounded(delta),
                "abs_projection_delta": rounded(abs(delta)),
                "content_projection": rounded(pos),
                "neutral_projection": rounded(neg),
                "valence_score_content": rounded(valence_score(item.content_text)),
                "arousal_score_content": rounded(arousal_score(item.content_text)),
            })
    summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["confound"]), str(row["direction_source"]), str(row["direction_emotion"]))].append(
            float(row["projection_delta"])
        )
    for (confound, source, emotion), vals in sorted(grouped.items()):
        summary.append({
            "confound": confound,
            "direction_source": source,
            "direction_emotion": emotion,
            "mean_projection_delta": rounded(safe_fmean(vals)),
            "mean_abs_projection_delta": rounded(safe_fmean([abs(v) for v in vals])),
            "max_abs_projection_delta": rounded(max((abs(v) for v in vals), default=float("nan"))),
            "n_rows": len(vals),
        })
    return rows, summary


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------


def selected_steering_items(
    items: Sequence[EmotionItem],
    split: Mapping[str, bool],
    per_emotion: int,
) -> list[EmotionItem]:
    selected: list[EmotionItem] = []
    for emotion in sorted({item.emotion for item in items}):
        eval_rows = [item for item in items if item.emotion == emotion and not split[item.item_id]]
        train_rows = [item for item in items if item.emotion == emotion and split[item.item_id]]
        selected.extend((eval_rows or train_rows)[:per_emotion])
    return selected


def append_generation_row(
    generation_rows: list[dict[str, Any]],
    item: EmotionItem,
    emotion: str,
    condition: str,
    dose_fraction: float,
    abs_scale: float,
    ref_norm: float,
    next_token_kl: float,
    prompt: str,
    text: str,
    depth: int,
    injection_layer: int,
) -> None:
    margin, target_hits, other_hits = target_margin(text, emotion)
    q = text_quality_stats(text)
    generation_rows.append({
        "item_id": item.item_id,
        "emotion": emotion,
        "cause": item.cause,
        "condition": condition,
        "dose_fraction_of_median_norm": rounded(dose_fraction),
        "absolute_scale": rounded(abs_scale),
        "median_reference_norm": rounded(ref_norm),
        "next_token_kl_to_baseline_bits": rounded(next_token_kl),
        "target_margin": rounded(margin),
        "target_hits": target_hits,
        "other_emotion_hits": other_hits,
        "valence_score": rounded(valence_score(text)),
        "arousal_score": rounded(arousal_score(text)),
        "stream_depth": depth,
        "injection_layer": injection_layer,
        **q,
        "hand_target_affect_0_2": "",
        "hand_off_target_affect_0_2": "",
        "hand_degenerated_0_1": "",
        "labeler_notes": "",
        "prompt": item.neutral_generation_prompt,
        "steering_prompt_kind": "neutral_generation_prompt",
        "rendered_prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "generation": text,
    })


def run_steering(
    bundle: bench.ModelBundle,
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    dirs: Mapping[tuple[str, str], Any],
    split: Mapping[str, bool],
    sentiment_direction: Any | None,
    depth: int,
    d_model: int,
    seed: int,
    per_emotion: int,
    ref_norm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generation_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    injection_layer = depth - 1
    chosen = selected_steering_items(items, split, per_emotion)
    by_emotion: dict[str, list[EmotionItem]] = defaultdict(list)
    for item in chosen:
        by_emotion[item.emotion].append(item)

    nonzero_doses = [float(d) for d in STEERING_DOSES if float(d) > 0.0]

    for emotion, rows in sorted(by_emotion.items()):
        prompts = [render_generation(bundle, item.neutral_generation_prompt) for item in rows]
        base_logits_by_item: dict[str, Any] = {}
        for item, prompt in zip(rows, prompts):
            base_logits_by_item[item.item_id] = bench.next_token_logits(bundle, prompt)
        baseline_outs = bench.generate_continuous(
            bundle,
            prompts,
            MAX_NEW_TOKENS,
            max_concurrent=ENGINE_MAX_CONCURRENT,
            skip_special_tokens=True,
            progress_label=f"lab13-{emotion}-baseline",
        )
        for item, prompt, text in zip(rows, prompts, baseline_outs):
            append_generation_row(
                generation_rows, item, emotion, "baseline", 0.0, 0.0, ref_norm, 0.0,
                prompt, text, depth, injection_layer,
            )

        input_dir = dirs.get(("comprehension", emotion))
        write_dir = dirs.get(("generation", emotion))
        shuffled_input = shuffled_direction_for(items, features, split, "comprehension", emotion, depth, seed + 391)
        if shuffled_input is not None:
            shuffled_input = orient_on_train(shuffled_input, items, features, split, "comprehension", emotion, depth)
        random_dir = random_unit(d_model, seed + stable_hash_int(f"steer:{emotion}:random") % 10_000_000)
        random_dir = orient_on_train(random_dir, items, features, split, "generation", emotion, depth)
        signed_sentiment = maybe_negate_sentiment_for_emotion(sentiment_direction, emotion)

        conditions: list[tuple[str, Any | None]] = [
            ("input_direction", input_dir),
            ("write_intent_direction", write_dir),
            ("shuffled_input_direction", shuffled_input),
            ("random_oriented", random_dir),
            ("sentiment_control", signed_sentiment),
        ]
        for condition, direction in conditions:
            if direction is None:
                continue
            job_prompts: list[str] = []
            job_items: list[EmotionItem] = []
            job_doses: list[float] = []
            job_scales: list[float] = []
            for dose in nonzero_doses:
                for item, prompt in zip(rows, prompts):
                    job_prompts.append(prompt)
                    job_items.append(item)
                    job_doses.append(dose)
                    job_scales.append(dose * ref_norm)
            outs = bench.generate_continuous(
                bundle,
                job_prompts,
                MAX_NEW_TOKENS,
                max_concurrent=ENGINE_MAX_CONCURRENT,
                skip_special_tokens=True,
                progress_label=f"lab13-{emotion}-{condition}",
                steer=(injection_layer, direction, job_scales),
            )
            for item, prompt, dose, abs_scale, text in zip(job_items, job_prompts, job_doses, job_scales, outs):
                try:
                    shifted_logits = bench.next_token_logits(bundle, prompt, steer=(injection_layer, direction, abs_scale))
                    next_kl = kl_bits(base_logits_by_item[item.item_id], shifted_logits)
                except Exception:
                    next_kl = float("nan")
                append_generation_row(
                    generation_rows, item, emotion, condition, dose, abs_scale, ref_norm, next_kl,
                    prompt, text, depth, injection_layer,
                )

    for emotion in sorted(by_emotion):
        baseline = [
            float(r["target_margin"]) for r in generation_rows
            if r["emotion"] == emotion and r["condition"] == "baseline"
            and isinstance(r.get("target_margin"), (int, float))
        ]
        base_mean = safe_fmean(baseline)
        for dose in nonzero_doses:
            row: dict[str, Any] = {
                "emotion": emotion,
                "dose_fraction_of_median_norm": rounded(dose),
                "baseline_mean": rounded(base_mean),
                "n_baseline_prompts": len(baseline),
                "stream_depth": depth,
                "injection_layer": injection_layer,
            }
            means: dict[str, float] = {}
            for condition in ("input_direction", "write_intent_direction", "random_oriented", "shuffled_input_direction", "sentiment_control"):
                vals = [
                    float(r["target_margin"]) for r in generation_rows
                    if r["emotion"] == emotion
                    and r["condition"] == condition
                    and abs(float(r["dose_fraction_of_median_norm"]) - dose) < 1e-9
                    and isinstance(r.get("target_margin"), (int, float))
                ]
                means[condition] = safe_fmean(vals)
                row[f"{condition}_mean"] = rounded(means[condition])
                row[f"{condition}_delta_vs_baseline"] = rounded(means[condition] - base_mean)
                row[f"{condition}_n"] = len(vals)
            row["input_over_random_delta"] = rounded((means["input_direction"] - base_mean) - (means["random_oriented"] - base_mean))
            row["input_over_shuffled_delta"] = rounded((means["input_direction"] - base_mean) - (means["shuffled_input_direction"] - base_mean))
            row["input_over_sentiment_delta"] = rounded((means["input_direction"] - base_mean) - (means["sentiment_control"] - base_mean))
            row["write_intent_over_random_delta"] = rounded((means["write_intent_direction"] - base_mean) - (means["random_oriented"] - base_mean))
            effect_rows.append(row)
    return generation_rows, effect_rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_depth_curve(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]], best_depth: int) -> None:
    if not rows:
        return
    depths = [int(r["depth"]) for r in rows]
    fig, ax = bench.new_figure(figsize=(8.8, 4.8))
    for key, label in [
        ("real_cross_auc", "real cross read/write"),
        ("shuffled_cross_auc", "shuffled-label control"),
        ("random_cross_auc", "random-direction control"),
    ]:
        xs, ys = [], []
        for r in rows:
            val = r.get(key)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                xs.append(int(r["depth"]))
                ys.append(float(val))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=2.5, label=label)
    ax.axhline(0.5, linestyle=":", linewidth=1.0, label="chance")
    ax.axvline(best_depth, linestyle="--", linewidth=1.0, label=f"selected depth {best_depth}")
    ax.set_xlabel("stream depth k (after k blocks; 0 is embeddings)")
    ax.set_ylabel("held-out cross-source AUC")
    ax.set_title("Depth selection: read/write transfer versus controls")
    ax.legend(loc="best")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "emotion_depth_curve.png", "Depth sweep used to choose the transfer depth.")


def plot_transfer_matrix(
    ctx: bench.RunContext,
    rows: Sequence[Mapping[str, Any]],
    emotions: Sequence[str],
    depth: int,
) -> None:
    import numpy as np

    labels = [f"{a}->{b}" for a in SOURCES for b in SOURCES]
    grid = np.full((len(emotions), len(labels)), np.nan)
    for i, emotion in enumerate(emotions):
        for j, label in enumerate(labels):
            source, target = label.split("->")
            vals = [
                float(r["auc"]) for r in rows
                if r.get("depth") == depth
                and r.get("direction_kind") == "real"
                and r.get("eval_split", "eval") == "eval"
                and r.get("emotion") == emotion
                and r.get("direction_source") == source
                and r.get("eval_target") == target
                and isinstance(r.get("auc"), (int, float))
            ]
            if vals:
                grid[i, j] = vals[0]
    fig, ax = bench.new_figure(figsize=(8.8, 4.8))
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions)
    ax.set_title(f"Emotion read/write transfer at stream depth {depth}")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, label="AUC: emotion state vs neutral match")
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        "emotion_transfer_matrix.png",
        "Comprehension and write-intent direction transfer matrix by emotion at the selected depth.",
    )


def plot_cosines(ctx: bench.RunContext, labels: Sequence[str], vectors: Mapping[str, Any], depth: int) -> None:
    import numpy as np

    if not labels:
        return
    grid = np.full((len(labels), len(labels)), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            grid[i, j] = cosine(vectors[a], vectors[b])
    fig, ax = bench.new_figure(figsize=(0.48 * len(labels) + 5.0, 0.48 * len(labels) + 4.8))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Emotion direction cosines at stream depth {depth}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, fraction=0.035, label="cosine")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "emotion_direction_cosines.png", "Pairwise cosines among emotion directions and sentiment control.")


def plot_specificity(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    vals_by_emotion: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("comparison_pool") != "all_other_emotions":
            continue
        if r.get("direction_source") == r.get("eval_target"):
            continue
        val = r.get("specificity_auc")
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            vals_by_emotion[str(r["emotion"])].append(float(val))
    if not vals_by_emotion:
        return
    labels = sorted(vals_by_emotion)
    means = [safe_fmean(vals_by_emotion[e]) for e in labels]
    fig, ax = bench.new_figure(figsize=(7.8, 4.6))
    ax.bar(labels, means)
    ax.axhline(0.5, linestyle=":", linewidth=1.0, label="chance")
    ax.axhline(SPECIFICITY_AUC_BAR, linestyle="--", linewidth=1.0, label="audit bar")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC: target emotion delta vs other-emotion deltas")
    ax.set_title("Emotion specificity beyond generic affect")
    ax.legend(loc="best")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "emotion_specificity.png", "Specificity of each emotion direction against other emotion deltas.")


def plot_cross_cause(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    vals_by_emotion: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        val = r.get("auc")
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            vals_by_emotion[str(r["emotion"])].append(float(val))
    if not vals_by_emotion:
        return
    labels = sorted(vals_by_emotion)
    means = [safe_fmean(vals_by_emotion[e]) for e in labels]
    errs = [safe_stderr(vals_by_emotion[e]) for e in labels]
    fig, ax = bench.new_figure(figsize=(7.8, 4.6))
    ax.bar(labels, means, yerr=[0.0 if not math.isfinite(e) else e for e in errs], capsize=3)
    ax.axhline(0.5, linestyle=":", linewidth=1.0, label="chance")
    ax.axhline(CROSS_CAUSE_AUC_BAR, linestyle="--", linewidth=1.0, label="audit bar")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("leave-one-cause-out AUC")
    ax.set_title("Cross-cause generalization")
    ax.legend(loc="best")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "cross_cause_generalization.png", "Cause-held-out transfer summary by emotion.")


def plot_confound_audit(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    finite_rows = [r for r in rows if isinstance(r.get("abs_projection_delta"), (int, float))]
    if not finite_rows:
        return
    top = sorted(finite_rows, key=lambda r: float(r["abs_projection_delta"]), reverse=True)[:14]
    labels = [f"{r['confound']}\n{r['direction']}" for r in top]
    vals = [float(r["projection_delta"]) for r in top]
    fig, ax = bench.new_figure(figsize=(9.0, 5.6))
    y = list(range(len(top)))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.0, linewidth=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("projection delta: confound text minus neutral match")
    ax.set_title("Largest confound projections onto emotion directions")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confound_projection_audit.png", "Largest surprise/calm/arousal confound projections.")


def plot_steering_effects(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    doses = sorted({float(r["dose_fraction_of_median_norm"]) for r in rows if isinstance(r.get("dose_fraction_of_median_norm"), (int, float))})
    conditions = [
        ("input_direction_delta_vs_baseline", "input-derived direction"),
        ("write_intent_direction_delta_vs_baseline", "write-intent direction"),
        ("random_oriented_delta_vs_baseline", "random control"),
        ("shuffled_input_direction_delta_vs_baseline", "shuffled control"),
        ("sentiment_control_delta_vs_baseline", "sentiment control"),
    ]
    fig, ax = bench.new_figure(figsize=(8.8, 4.8))
    for key, label in conditions:
        ys: list[float] = []
        xs: list[float] = []
        for dose in doses:
            vals = [
                float(r[key]) for r in rows
                if abs(float(r["dose_fraction_of_median_norm"]) - dose) < 1e-9
                and isinstance(r.get(key), (int, float))
            ]
            if vals:
                xs.append(dose)
                ys.append(safe_fmean(vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.axvline(HEADLINE_STEERING_DOSE, linestyle="--", linewidth=1.0, label="headline dose")
    ax.set_xlabel("dose as fraction of median train activation norm")
    ax.set_ylabel("generated target-affect margin delta vs baseline")
    ax.set_title("Affect steering dose response")
    ax.legend(loc="best")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "affect_steering_effects.png", "Dose response for input-derived emotion steering and controls.")


# ---------------------------------------------------------------------------
# Cards, audits, summaries, and labels
# ---------------------------------------------------------------------------


def compute_audit_verdict(metrics: Mapping[str, Any]) -> tuple[str, str]:
    cross = metrics.get("mean_cross_transfer_auc")
    adjusted = metrics.get("selected_depth_control_adjusted_cross_auc")
    specificity = metrics.get("mean_specificity_auc")
    cross_cause = metrics.get("mean_cross_cause_auc")
    sentiment_cos = metrics.get("max_abs_sentiment_cosine")
    steer = metrics.get("steering_input_over_random_delta")

    def ge(x: Any, bar: float) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) >= bar

    def le(x: Any, bar: float) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) <= bar

    transfer_ok = ge(cross, CROSS_TRANSFER_AUC_BAR) and ge(adjusted, CONTROL_ADJUSTED_AUC_BAR)
    specificity_ok = ge(specificity, SPECIFICITY_AUC_BAR)
    cause_ok = ge(cross_cause, CROSS_CAUSE_AUC_BAR)
    sentiment_ok = le(sentiment_cos, SENTIMENT_COSINE_WARN)
    steering_ok = ge(steer, STEERING_DELTA_BAR)

    if transfer_ok and specificity_ok and cause_ok and sentiment_ok and steering_ok:
        # The steering gate above is the automatic lexicon ruler only. The lab's
        # own rule (generation_labeling_guide.md) is that a steering effect that
        # lives only in the lexicon score is not a finished causal claim until it
        # survives hand-labeled generations -- which the harness cannot do. So the
        # auto-verdict certifies the decode + confound result and defers the causal
        # handle to hand labels rather than asserting a completed causal handle.
        return "passed", "emotion-specific read/write transfer; causal steering pending hand-labeled generations"
    if transfer_ok and steering_ok and not sentiment_ok:
        return "failed", "generic sentiment or valence handle"
    if transfer_ok or steering_ok:
        return "mixed", "affect handle with unresolved confounds"
    return "failed", "no defended emotion-geometry claim"


def write_generation_labeling_guide(ctx: bench.RunContext) -> None:
    lines = [
        "# Lab 13 Generation Labeling Guide",
        "",
        "The automatic target-affect margin is a lexicon ruler. It is useful for plotting, but it is not a judge.",
        "Hand-label at least the rows used in your writeup, especially cases where the input direction beats or loses to a control.",
        "",
        "## Suggested columns",
        "",
        "- `hand_target_affect_0_2`: 0 = no target affect, 1 = weak or ambiguous target affect, 2 = clear target affect.",
        "- `hand_off_target_affect_0_2`: 0 = no other emotion, 1 = weak other emotion, 2 = clear other emotion.",
        "- `hand_degenerated_0_1`: 1 if the output is repetitive, malformed, prompt-echoing, or otherwise not a usable sentence.",
        "- `labeler_notes`: short reason for surprising labels.",
        "",
        "## Rule of thumb",
        "",
        "A steering effect that exists only in the lexicon score and vanishes under hand labels is a failed causal claim, not a weaker success.",
        "",
    ]
    path = ctx.path("tables", "generation_labeling_guide.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "guide", "Manual labeling guide for steering generations.")


def write_operationalization_audit(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
    confound_rows: Sequence[Mapping[str, Any]],
) -> None:
    top_confound = sorted(
        [r for r in confound_rows if isinstance(r.get("abs_projection_delta"), (int, float))],
        key=lambda r: float(r["abs_projection_delta"]),
        reverse=True,
    )[:5]
    result = str(metrics.get("audit_result", "mixed"))
    claim_allowed = str(metrics.get("claim_allowed", "affect handle with unresolved confounds"))
    yaml_block = {
        "headline_claim": "reading-derived emotion directions share geometry with write-intent directions and can steer generated affect",
        "cheap_explanation": "topic, cause, valence, arousal, surprise, calm positivity, or generic sentiment",
        "killer_control": "cross-cause generalization, emotion specificity against other emotions, signed sentiment-direction cosine, confound projections, and random/shuffled steering controls",
        "result": result,
        "claim_allowed": claim_allowed,
    }
    lines = [
        "# Lab 13 Operationalization Audit",
        "",
        "```yaml",
        *[f"{k}: {json.dumps(v)}" for k, v in yaml_block.items()],
        "```",
        "",
        "## What Was Measured",
        "",
        "The lab measures paired residual-stream contrasts: emotional content or write prompt minus a neutral match about the same cause. A positive result is a property of this operationalization, not evidence that the model feels the emotion.",
        "",
        "## Cheap Explanations Under Audit",
        "",
        "- Topic or cause: every target item is paired with a neutral paraphrase about the same cause; cross-cause rows are held out by cause.",
        "- Valence or sentiment: the run computes a Lab-7-style positive-minus-negative sentiment direction and reports its cosine with each emotion direction.",
        "- Generic affect: `emotion_specificity_vs_other_emotions.csv` compares each target emotion delta with other-emotion deltas.",
        "- Arousal, surprise, and calm positivity: confound rows are projected onto every emotion direction.",
        "- Prompt-style steering: random, shuffled, write-intent, and sentiment controls ride the same dose schedule as the input-derived direction.",
        "",
        "## Current Run Headline",
        "",
        f"- Best stream depth: {metrics.get('best_depth')}",
        f"- Selected-depth control-adjusted cross AUC: {metrics.get('selected_depth_control_adjusted_cross_auc')}",
        f"- Mean cross input/output transfer AUC: {metrics.get('mean_cross_transfer_auc')}",
        f"- Mean emotion specificity AUC: {metrics.get('mean_specificity_auc')}",
        f"- Mean cross-cause AUC: {metrics.get('mean_cross_cause_auc')}",
        f"- Max absolute sentiment-control cosine: {metrics.get('max_abs_sentiment_cosine')}",
        f"- Headline-dose input-over-random steering delta: {metrics.get('steering_input_over_random_delta')}",
        f"- Audit result: `{result}`",
        f"- Claim allowed: `{claim_allowed}`",
        "",
        "## Largest Confound Projection Deltas",
        "",
    ]
    if not top_confound:
        lines.append("No confound projection rows were available.")
    else:
        for row in top_confound:
            lines.append(
                f"- `{row['item_id']}` ({row['confound']}) on `{row['direction']}`: "
                f"delta {row['projection_delta']}"
            )
    lines += [
        "",
        "## Non-claims",
        "",
        "This audit does not support claims about felt emotion, consciousness, a true personality, or a localized emotion module. Even a passed audit supports a handle under this paired data design, not an ontology of emotion inside the model.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits and cheap-explanation audit for Lab 13.")


def write_emotion_geometry_card(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 13 Emotion Geometry Card",
        "",
        "Read this before the heatmaps. It says what the run can and cannot defend.",
        "",
        "## Verdict",
        "",
        f"- Audit result: `{metrics.get('audit_result')}`",
        f"- Claim allowed: `{metrics.get('claim_allowed')}`",
        f"- Data source: `{metrics.get('data', {}).get('data_source')}`",
        f"- Model: `{metrics.get('model_id')}`",
        "",
        "## Headline metrics",
        "",
        f"- Best stream depth: {metrics.get('best_depth')} (steering injection layer {metrics.get('injection_layer')})",
        f"- Mean cross read/write AUC: {metrics.get('mean_cross_transfer_auc')}",
        f"- Control-adjusted cross AUC at selected depth: {metrics.get('selected_depth_control_adjusted_cross_auc')}",
        f"- Mean same-emotion comprehension/write cosine: {metrics.get('mean_comp_gen_cosine')}",
        f"- Mean specificity AUC against other emotions: {metrics.get('mean_specificity_auc')}",
        f"- Mean leave-one-cause-out AUC: {metrics.get('mean_cross_cause_auc')}",
        f"- Max abs sentiment cosine: {metrics.get('max_abs_sentiment_cosine')}",
        f"- Input-derived steering over random at dose {HEADLINE_STEERING_DOSE}: {metrics.get('steering_input_over_random_delta')}",
        "",
        "## Read next",
        "",
        "1. `operationalization_audit.md` for the cheap-explanation verdict.",
        "2. `tables/emotion_probe_transfer.csv` and `plots/emotion_transfer_matrix.png` for decodability and transfer.",
        "3. `tables/emotion_specificity_vs_other_emotions.csv` for generic-affect controls.",
        "4. `tables/steering_generations.csv` plus `tables/generation_labeling_guide.md` before making a causal steering claim.",
        "",
        "## Non-claims",
        "",
        "The run does not show that the model feels anything, that emotion is localized at one layer, or that the direction is a mechanism rather than a usable handle.",
        "",
    ]
    path = ctx.path("emotion_geometry_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "card", "First-read verdict card for Lab 13.")


def write_run_summary(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        f"# Lab 13 Run Summary: {ctx.run_dir.name}",
        "",
        "## 1. Behavior studied",
        "",
        "Reading emotion-laden text, prompting the model to write in an emotion, and generation under activation addition.",
        "",
        "## 2. Internal object measured",
        "",
        f"Paired difference-in-means directions at stream depth {metrics.get('best_depth')}: emotion state minus neutral matched state.",
        "",
        "## 3. Intervention and controls",
        "",
        "Held-out decoding, cross-source transfer, leave-one-cause-out tests, specificity against other emotions, sentiment/cosine controls, confound projection rows, and activation-addition steering with random and shuffled controls.",
        "",
        "## 4. Metric changes",
        "",
        f"- Mean cross read/write AUC: {metrics.get('mean_cross_transfer_auc')}",
        f"- Selected-depth control-adjusted cross AUC: {metrics.get('selected_depth_control_adjusted_cross_auc')}",
        f"- Mean specificity AUC: {metrics.get('mean_specificity_auc')}",
        f"- Mean cross-cause AUC: {metrics.get('mean_cross_cause_auc')}",
        f"- Input-over-random steering delta at dose {HEADLINE_STEERING_DOSE}: {metrics.get('steering_input_over_random_delta')}",
        "",
        "## 5. Claim supported",
        "",
        f"Audit `{metrics.get('audit_result')}`; allowed claim: `{metrics.get('claim_allowed')}`. See `emotion_geometry_card.md` and `operationalization_audit.md`.",
        "",
        "## 6. Claim not supported",
        "",
        "No claim about felt emotion, consciousness, a stable personality, or a unique emotion module is supported by this lab.",
        "",
        "## 7. Falsifier",
        "",
        "A new cause-balanced dataset, hand labels, or stronger confound controls that reduce transfer and steering effects to random/shuffled/sentiment baselines would falsify the favored interpretation.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable summary following the standard seven-question contract.")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 13 requires an instruct model with a chat template.")

    targets, confounds, data_info = load_items(args)
    emotions = sorted({item.emotion for item in targets})
    print(
        f"[lab13] {len(targets)} target items across {len(emotions)} emotions; "
        f"{len(confounds)} confounds; prompt_set={args.prompt_set}"
    )
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Frozen Lab 13 data hash, filters, counts, and fallback status.")

    item_manifest_rows = [
        {
            "item_id": item.item_id,
            "role": "target" if not item.confound else "confound",
            "emotion": item.emotion,
            "cause": item.cause,
            "arousal": item.arousal,
            "valence": item.valence,
            "confound": item.confound,
            "content_len_words": len(words(item.content_text)),
            "neutral_len_words": len(words(item.neutral_text)),
            "generation_prompt_len_words": len(words(item.generation_prompt)),
            "neutral_generation_prompt_len_words": len(words(item.neutral_generation_prompt)),
            "note": item.note,
        }
        for item in list(targets) + list(confounds)
    ]
    item_manifest_path = ctx.path("tables", "item_manifest.csv")
    bench.write_csv_with_context(ctx, item_manifest_path, item_manifest_rows)
    ctx.register_artifact(item_manifest_path, "table", "Selected target and confound rows with labels, word counts, and provenance notes.")

    first_prompt = render_comprehension(bundle, targets[0].content_text)
    run_rendered_hook_parity_check(ctx, bundle, first_prompt)
    first_capture = bench.run_with_residual_cache(bundle, first_prompt, add_special_tokens=False)
    bench.run_lens_self_check(ctx, bundle, first_capture)

    split = make_split(targets, int(args.seed))
    split_rows = [
        {
            "item_id": item.item_id,
            "emotion": item.emotion,
            "cause": item.cause,
            "arousal": item.arousal,
            "valence": item.valence,
            "split": "train" if split[item.item_id] else "eval",
        }
        for item in targets
    ]
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows)
    ctx.register_artifact(split_path, "diagnostic", "Deterministic cause-grouped train/eval split.")

    family_manifest_rows = []
    for emotion in emotions:
        rows = [it for it in targets if it.emotion == emotion]
        family_manifest_rows.append({
            "emotion": emotion,
            "n_items": len(rows),
            "n_train": sum(1 for it in rows if split[it.item_id]),
            "n_eval": sum(1 for it in rows if not split[it.item_id]),
            "causes": ";".join(sorted({it.cause for it in rows})),
            "arousal_labels": ";".join(sorted({it.arousal for it in rows if it.arousal})),
            "valence_labels": ";".join(sorted({it.valence for it in rows if it.valence})),
        })
    family_path = ctx.path("tables", "emotion_family_manifest.csv")
    bench.write_csv_with_context(ctx, family_path, family_manifest_rows)
    ctx.register_artifact(family_path, "table", "Selected emotion families, causes, and split counts.")

    features = cache_features(ctx, bundle, targets, confounds)
    n_depths = bundle.anatomy.n_layers + 1
    transfer_rows, depth_rows, best_depth = scan_transfer_depths(
        targets, features, split, n_depths, bundle.anatomy.d_model, int(args.seed)
    )
    print(f"[lab13] selected stream depth {best_depth} by train-split control-adjusted cross-source transfer")

    depth_path = ctx.path("tables", "depth_selection.csv")
    bench.write_csv_with_context(ctx, depth_path, depth_rows)
    ctx.register_artifact(depth_path, "table", "Depth sweep summary with train-selected and eval-reported real/control cross-source AUCs.")
    depth_diag_path = ctx.path("diagnostics", "depth_selection.json")
    bench.write_json(depth_diag_path, {
        "selected_depth": best_depth,
        "selection_split": "train",
        "selection_rule": "maximize train_control_adjusted_cross_auc over depths 1..L; eval rows are reported after selection",
        "stream_depth_convention": "streams[k] is the pre-norm residual after k blocks; depth 0 embeddings excluded",
        "injection_layer_for_steering": best_depth - 1,
        "selected_depth_row": next((r for r in depth_rows if int(r["depth"]) == best_depth), {}),
    })
    ctx.register_artifact(depth_diag_path, "diagnostic", "Depth-selection rule, split, selected stream depth, and steering layer mapping.")

    limits = PROMPT_SET_LIMITS[str(args.prompt_set)]
    sentiment_cap = limits["sentiment_pairs"]
    if int(getattr(args, "max_examples", 0) or 0) > 0:
        sentiment_cap = max(2, min(sentiment_cap, int(args.max_examples) * 2))
    sentiment_direction, n_sentiment_pairs, valence_info = build_sentiment_direction(args, bundle, best_depth, sentiment_cap)
    valence_manifest_path = ctx.path("diagnostics", "sentiment_control_manifest.json")
    bench.write_json(valence_manifest_path, valence_info)
    ctx.register_artifact(valence_manifest_path, "diagnostic", "Data and hash information for the Lab-7-style sentiment control.")

    add_controls_at_depth(
        transfer_rows,
        targets,
        features,
        split,
        best_depth,
        int(args.seed),
        sentiment_direction,
        bundle.anatomy.d_model,
    )

    transfer_path = ctx.path("tables", "emotion_probe_transfer.csv")
    bench.write_csv_with_context(ctx, transfer_path, transfer_rows)
    ctx.register_artifact(
        transfer_path,
        "table",
        "Emotion/neutral decoding and comprehension-write transfer by depth, with controls at the selected depth.",
    )
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, transfer_rows)
    ctx.register_artifact(results_path, "results", "Alias of emotion_probe_transfer.csv for the standard run contract.")

    best_dirs = build_directions_at_depth(targets, features, split, best_depth)

    cross_cause_rows = cross_cause_generalization(targets, features, split, best_depth)
    cross_path = ctx.path("tables", "cross_cause_generalization.csv")
    bench.write_csv_with_context(ctx, cross_path, cross_cause_rows)
    ctx.register_artifact(cross_path, "table", "Leave-one-cause-out transfer for each emotion and source/target pair.")

    specificity = emotion_specificity_rows(targets, features, split, best_dirs, best_depth)
    specificity_path = ctx.path("tables", "emotion_specificity_vs_other_emotions.csv")
    bench.write_csv_with_context(ctx, specificity_path, specificity)
    ctx.register_artifact(specificity_path, "table", "Emotion-specificity checks against other-emotion, same-valence, and same-arousal deltas.")

    cos_rows, labels, vectors = cosine_rows(best_dirs, sentiment_direction)
    cos_path = ctx.path("tables", "emotion_direction_cosines.csv")
    bench.write_csv_with_context(ctx, cos_path, cos_rows)
    ctx.register_artifact(cos_path, "table", "Pairwise cosines among emotion directions and the sentiment control.")

    confound_rows, confound_summary = confound_projection_audit(confounds, features, best_dirs, best_depth)
    confound_path = ctx.path("tables", "confound_projection_audit.csv")
    bench.write_csv_with_context(ctx, confound_path, confound_rows)
    ctx.register_artifact(confound_path, "table", "Projection of surprise/calm/arousal confounds onto target emotion directions.")
    confound_summary_path = ctx.path("tables", "confound_projection_summary.csv")
    bench.write_csv_with_context(ctx, confound_summary_path, confound_summary)
    ctx.register_artifact(confound_summary_path, "table", "Aggregate confound projection magnitudes by confound and direction.")

    ref_norm_vals = [
        float(features[item.item_id][feature_key(source, suffix)][best_depth].norm())
        for item in targets
        if split[item.item_id]
        for source in SOURCES
        for suffix in FEATURE_SUFFIXES
    ]
    ref_norm = safe_median(ref_norm_vals, default=1.0)
    steering_per_emotion = limits["steering_per_emotion"]
    if int(getattr(args, "max_examples", 0) or 0) > 0:
        steering_per_emotion = max(1, min(steering_per_emotion, int(args.max_examples)))
    generation_rows, steering_rows = run_steering(
        bundle,
        targets,
        features,
        best_dirs,
        split,
        sentiment_direction,
        best_depth,
        bundle.anatomy.d_model,
        int(args.seed),
        steering_per_emotion,
        ref_norm,
    )
    generations_path = ctx.path("tables", "steering_generations.csv")
    bench.write_csv_with_context(ctx, generations_path, generation_rows)
    ctx.register_artifact(
        generations_path,
        "table",
        "Generated outputs for input/write/random/shuffled/sentiment steering with lexicon scores and blank hand-label columns.",
    )
    steering_path = ctx.path("tables", "steering_effects.csv")
    bench.write_csv_with_context(ctx, steering_path, steering_rows)
    ctx.register_artifact(steering_path, "table", "Per-emotion steering effect over baseline and controls across a dose sweep.")
    write_generation_labeling_guide(ctx)

    state_payload: dict[str, Any] = {
        "depth": best_depth,
        "injection_layer": best_depth - 1,
        "depth_convention": "bench streams[k]: 0 = embeddings, k = residual after block k; injection at layer k-1 targets stream depth k",
        "read_site": "chat-templated final prompt token before assistant generation",
        "directions": {f"{source}_{emotion}": direction for (source, emotion), direction in best_dirs.items()},
        "sentiment_lab7_style_positive_minus_negative": sentiment_direction,
        "n_sentiment_pairs": n_sentiment_pairs,
        "method": "paired difference-in-means, cause-grouped train split only, unit normalized",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "evidence": "DECODE artifact unless injected in the steering table.",
    }
    state_path = ctx.path("state", "emotion_directions.pt")
    torch.save(state_payload, state_path)
    ctx.register_artifact(state_path, "tensor", "Comprehension/write-intent emotion directions and sentiment control at selected depth.")
    state_metadata = {k: v for k, v in state_payload.items() if k not in {"directions", "sentiment_lab7_style_positive_minus_negative"}}
    state_metadata["direction_names"] = sorted(state_payload["directions"])
    state_metadata_path = ctx.path("state", "emotion_directions_metadata.json")
    bench.write_json(state_metadata_path, state_metadata)
    ctx.register_artifact(state_metadata_path, "metadata", "Human-readable metadata for saved emotion directions.")

    if not args.no_plots:
        plot_depth_curve(ctx, depth_rows, best_depth)
        plot_transfer_matrix(ctx, transfer_rows, emotions, best_depth)
        plot_cosines(ctx, labels, vectors, best_depth)
        plot_specificity(ctx, specificity)
        plot_cross_cause(ctx, cross_cause_rows)
        plot_confound_audit(ctx, confound_rows)
        plot_steering_effects(ctx, steering_rows)

    selected_depth_row = next((r for r in depth_rows if int(r["depth"]) == best_depth), {})
    real_best = [
        float(r["auc"]) for r in transfer_rows
        if r.get("depth") == best_depth
        and r.get("direction_kind") == "real"
        and r.get("eval_split", "eval") == "eval"
        and isinstance(r.get("auc"), (int, float))
        and math.isfinite(float(r["auc"]))
    ]
    cross_best = [
        float(r["auc"]) for r in transfer_rows
        if r.get("depth") == best_depth
        and r.get("direction_kind") == "real"
        and r.get("eval_split", "eval") == "eval"
        and r.get("direction_source") != r.get("eval_target")
        and isinstance(r.get("auc"), (int, float))
        and math.isfinite(float(r["auc"]))
    ]
    comp_gen_cos = [
        cosine(best_dirs[("comprehension", emotion)], best_dirs[("generation", emotion)])
        for emotion in emotions
        if ("comprehension", emotion) in best_dirs and ("generation", emotion) in best_dirs
    ]
    sentiment_cos = [
        abs(float(r["cosine"])) for r in cos_rows
        if "sentiment_lab7_style" in str(r["direction_a"]) or "sentiment_lab7_style" in str(r["direction_b"])
    ]
    specificity_vals = [
        float(r["specificity_auc"]) for r in specificity
        if r.get("comparison_pool") == "all_other_emotions"
        and r.get("direction_source") != r.get("eval_target")
        and isinstance(r.get("specificity_auc"), (int, float))
        and math.isfinite(float(r["specificity_auc"]))
    ]
    cross_cause_vals = [
        float(r["auc"]) for r in cross_cause_rows
        if isinstance(r.get("auc"), (int, float)) and math.isfinite(float(r["auc"]))
    ]
    cross_cause_cross_vals = [
        float(r["auc"]) for r in cross_cause_rows
        if r.get("direction_source") != r.get("eval_target")
        and isinstance(r.get("auc"), (int, float)) and math.isfinite(float(r["auc"]))
    ]
    confound_abs_vals = [
        float(r["abs_projection_delta"]) for r in confound_rows
        if isinstance(r.get("abs_projection_delta"), (int, float))
    ]
    headline_rows = [
        r for r in steering_rows
        if abs(float(r.get("dose_fraction_of_median_norm", -999)) - HEADLINE_STEERING_DOSE) < 1e-9
    ]
    steering_over_random = finite_values(headline_rows, "input_over_random_delta")
    steering_over_shuffled = finite_values(headline_rows, "input_over_shuffled_delta")
    steering_over_sentiment = finite_values(headline_rows, "input_over_sentiment_delta")

    metrics: dict[str, Any] = {
        "model_id": bundle.anatomy.model_id,
        "n_target_items": len(targets),
        "n_confound_items": len(confounds),
        "emotions": emotions,
        "best_depth": best_depth,
        "injection_layer": best_depth - 1,
        "n_depths": n_depths,
        "n_sentiment_pairs": n_sentiment_pairs,
        "mean_real_auc_at_best_depth": none_if_nan(rounded(safe_fmean(real_best))),
        "mean_cross_transfer_auc": none_if_nan(rounded(safe_fmean(cross_best))),
        "selected_depth_real_cross_auc": selected_depth_row.get("real_cross_auc"),
        "selected_depth_random_cross_auc": selected_depth_row.get("random_cross_auc"),
        "selected_depth_shuffled_cross_auc": selected_depth_row.get("shuffled_cross_auc"),
        "selected_depth_control_adjusted_cross_auc": selected_depth_row.get("control_adjusted_cross_auc"),
        "mean_comp_gen_cosine": none_if_nan(rounded(safe_fmean(comp_gen_cos))),
        "max_abs_sentiment_cosine": none_if_nan(rounded(max(sentiment_cos, default=float("nan")))),
        "mean_specificity_auc": none_if_nan(rounded(safe_fmean(specificity_vals))),
        "mean_cross_cause_auc": none_if_nan(rounded(safe_fmean(cross_cause_cross_vals))),
        "mean_cross_cause_auc_all_cells": none_if_nan(rounded(safe_fmean(cross_cause_vals))),
        "mean_abs_confound_projection_delta": none_if_nan(rounded(safe_fmean(confound_abs_vals))),
        "steering_input_over_random_delta": none_if_nan(rounded(safe_fmean(steering_over_random))),
        "steering_input_over_shuffled_delta": none_if_nan(rounded(safe_fmean(steering_over_shuffled))),
        "steering_input_over_sentiment_delta": none_if_nan(rounded(safe_fmean(steering_over_sentiment))),
        "reference_activation_norm_median": rounded(ref_norm),
        "steering_doses_fraction_of_median_norm": list(STEERING_DOSES),
        "headline_steering_dose": HEADLINE_STEERING_DOSE,
        "data": data_info,
        "sentiment_control_data": valence_info,
        "audit_thresholds": {
            "cross_transfer_auc_bar": CROSS_TRANSFER_AUC_BAR,
            "control_adjusted_auc_bar": CONTROL_ADJUSTED_AUC_BAR,
            "specificity_auc_bar": SPECIFICITY_AUC_BAR,
            "cross_cause_auc_bar": CROSS_CAUSE_AUC_BAR,
            "sentiment_cosine_warn": SENTIMENT_COSINE_WARN,
            "steering_delta_bar": STEERING_DELTA_BAR,
        },
    }
    audit_result, claim_allowed = compute_audit_verdict(metrics)
    if data_info.get("tier_a_fallback_warning"):
        audit_result = "not_science_run"
        claim_allowed = "plumbing only; do not ledger"
    metrics["audit_result"] = audit_result
    metrics["claim_allowed"] = claim_allowed

    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 13 metrics and audit verdict.")

    write_operationalization_audit(ctx, metrics, confound_rows)
    write_emotion_geometry_card(ctx, metrics)
    write_run_summary(ctx, metrics)

    run_name = ctx.run_dir.name
    fallback_note = " This was a Tier A fallback plumbing run and should not be moved into the claim ledger." if data_info.get("tier_a_fallback_warning") else ""
    if isinstance(metrics.get("steering_input_over_random_delta"), (int, float)) and float(metrics["steering_input_over_random_delta"]) >= STEERING_DELTA_BAR:
        causal_text = (
            f"Input-derived emotion directions on {bundle.anatomy.model_id} shifted generated target-affect "
            f"scores by {metrics['steering_input_over_random_delta']} over an oriented random direction at "
            f"dose {HEADLINE_STEERING_DOSE} of the median train activation norm, injecting at layer "
            f"{best_depth - 1}. Audit: {audit_result}; allowed claim: {claim_allowed}. This is a scoped "
            f"activation-addition effect on this prompt family, not evidence of felt emotion.{fallback_note}"
        )
    else:
        causal_text = (
            f"Input-derived emotion directions on {bundle.anatomy.model_id} did not validate as a strong "
            f"causal affect-steering handle in this run: the headline input-over-random delta was "
            f"{metrics.get('steering_input_over_random_delta')} at dose {HEADLINE_STEERING_DOSE}. "
            f"Audit: {audit_result}; allowed claim: {claim_allowed}.{fallback_note}"
        )

    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "CAUSAL",
            "text": causal_text,
            "artifact": f"runs/{run_name}/tables/steering_effects.csv",
            "falsifier": (
                "A random, shuffled, or sentiment control matches the effect; hand labels disagree with the "
                "lexicon score; next-token KL or degeneration shows the behavior changed by damage rather "
                "than affect; or the effect vanishes on held-out causes."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "DECODE",
            "text": (
                f"Comprehension and write-intent emotion directions share measurable geometry on "
                f"{bundle.anatomy.model_id}: mean cross read/write AUC is "
                f"{metrics['mean_cross_transfer_auc']}, mean same-emotion cosine is "
                f"{metrics['mean_comp_gen_cosine']}, and the selected-depth control-adjusted cross AUC is "
                f"{metrics['selected_depth_control_adjusted_cross_auc']}. Audit: {audit_result}; "
                f"allowed claim: {claim_allowed}. This is a read/write affect-handle claim, not a claim "
                f"about felt emotion or a unique emotion mechanism.{fallback_note}"
            ),
            "artifact": f"runs/{run_name}/tables/emotion_probe_transfer.csv",
            "falsifier": (
                "Cross-source AUC falls to chance on new causes, same-valence controls explain the effect, "
                "or the directions are collinear with the Lab-7-style sentiment direction."
            ),
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
