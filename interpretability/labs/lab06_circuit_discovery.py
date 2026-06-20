"""Lab 6: Circuit discovery and validation, the manual way.

This lab composes the previous instrumentation labs (plus the residual-stream
and self-check habits from Lab 1) into a small, earned circuit claim:
- Lab 2: direct-logit attribution (cheap screen)
- Lab 3: attention motifs (previous-token, induction, sink patterns)
- Lab 5: intervention discipline (mean-ablation instead of zero-ablation)

The deliverable is a circuit card with three earned numbers and an explicitly
scoped mechanism sketch (heads-only routing subgraph):

* faithfulness: with every non-circuit head mean-ablated, how much of the
  original behavior remains?
* completeness: with the circuit heads mean-ablated, how much behavior remains?
* minimality: how much faithfulness is lost if each kept node is removed?

The target behavior is induction completion on fixed-length repeating patterns.
The circuit claim is intentionally heads-only: it is a routing subgraph. MLPs
are ranked and reported as supporting infrastructure, but they are not part of
the faithfulness complement. This is the manual baseline that Lab 9 will
confront with an automated attribution graph. Keep the card.

Evidence level: CAUSAL at heads-only circuit scope, on a stated prompt
population and a stated off-distribution (dataset-mean ablation).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import os
import statistics
from typing import Any, Iterable, Sequence

import interp_bench as bench
from labs.lab02_direct_logit_attribution import compute_direct_logit_attribution
from labs.lab03_attention_routing import (
    head_attribution_scores,
    induction_score,
    prev_token_score,
)

LAB_ID = "L06"

# --- Selection / verdict thresholds ----------------------------------------
# A circuit "passes" when it preserves at least this ratio of base behavior.
FAITHFULNESS_FLOOR = 0.70
# Knee (FIX 1): the smallest circuit whose faithfulness is within KNEE_EPSILON
# of the prune trajectory's PEAK. The knee is the honest headline; the floor
# circuit (smallest still above FAITHFULNESS_FLOOR) is the overfit comparison.
KNEE_EPSILON = 0.02
# Minimality honesty (FIX 7): a kept node whose marginal faithfulness is below
# this pays no rent -- it is likely overfit filler, reported with and without.
MARGINAL_RENT_THRESHOLD = 0.02
# Hygiene gate (FIX 6): fewer than this many baseline-positive discovery
# prompts ABORTS the cell. Tiny-n cards are forbidden.
MIN_POSITIVE = 8
# Suppression heads (FIX 4): a head whose single-head ablation RAISES the
# metric by more than this fraction of base is a brake / anti-circuit head.
SUPPRESSION_REL_THRESHOLD = 0.05
# Resample / interchange ablation (FIX 2): default within-distribution draws.
DEFAULT_RESAMPLE_DRAWS = 5

# Screening is deliberately broader than a minimal demo. A too-thin screen can
# make the pruning stage fail before the students learn anything about circuits.
SCREEN_TOP_ATTRIBUTION_MIN = 20
SCREEN_TOP_INDUCTION_MIN = 8
SCREEN_TOP_PREV_MIN = 8
SCREEN_TOP_ATTRIBUTION_FRAC = 0.035
SCREEN_TOP_MOTIF_FRAC = 0.012
N_MLP_CANDIDATES = 8

MOTIF_STRONG_THRESHOLD = 0.35
FIRST_TOKEN_SINK_THRESHOLD = 0.45
EDGE_MIN_SOURCE_EFFECT = 1e-6
# Olmo 7B's strongest ordered interaction in the validation run is small but
# real enough to teach redundancy. Report it as weak at 2%, strong at 5%.
EDGE_MIN_ROUTED_FRACTION = 0.02
EDGE_STRONG_ROUTED_FRACTION = 0.05

# The labeled motif heads that constitute each behavior's "core". The
# motif-core-only circuit (these heads alone) is evaluated alongside the knee
# circuit: if the full circuit only beats the motif core via marginal-zero
# filler, the circuit is overfit, not real.
CORE_MOTIFS_BY_BEHAVIOR: dict[str, tuple[str, ...]] = {
    "induction_p3": ("induction", "previous_token"),
    "induction_p2": ("induction", "previous_token"),
    "successor": ("induction", "previous_token"),  # expected ABSENT: that IS the finding
    "swa": ("induction", "previous_token"),
}


@dataclasses.dataclass(frozen=True)
class CircuitPrompt:
    example_id: str
    family: str          # discovery or heldout
    prompt: str
    target: str
    distractor: str
    period: int | None = None   # cycle period for induction-family prompts (FIX 5)


# ===========================================================================
# Behavior families (PART 3). Each behavior is its OWN run and its OWN circuit
# card. Discovery and held-out within a behavior must share ONE token length;
# the hygiene gate (FIX 6) enforces that per model and itemizes any prompt it
# could not use. Pools are generous (>=12 discovery) so >=8 survive the
# baseline-positive gate on each tokenizer.
# ===========================================================================


def _induction_prompts(rows: Sequence[tuple[str, str, tuple[str, ...]]], period: int) -> list[CircuitPrompt]:
    """Build fixed-length repeating-cycle induction prompts.

    ``rows`` is (example_id, family, cycle_words). The 8-token window is the
    cycle repeated; the target is the induction continuation, the distractor is
    the most plausible wrong continuation -- for period-3 the cycle restart
    (word 0), for period-2 the copy-the-current-token error (word 1).
    """
    out: list[CircuitPrompt] = []
    for example_id, family, words in rows:
        seq = [words[i % period] for i in range(8)]
        prompt = " ".join(seq)
        target = " " + words[8 % period]
        distractor = " " + (words[0] if period == 3 else words[1])
        out.append(CircuitPrompt(example_id, family, prompt, target, distractor, period=period))
    return out


_P3_ROWS = [
    ("d_colors", "discovery", ("red", "blue", "green")),
    ("d_animals", "discovery", ("dog", "cat", "bird")),
    ("d_letters", "discovery", ("B", "F", "Q")),
    ("d_numbers", "discovery", ("seven", "three", "nine")),
    ("d_fruit", "discovery", ("apple", "pear", "banana")),
    ("d_shapes", "discovery", ("circle", "square", "oval")),
    ("d_weather", "discovery", ("rain", "snow", "wind")),
    ("d_seasons", "discovery", ("spring", "autumn", "summer")),
    ("d_birds", "discovery", ("hawk", "crow", "dove")),
    ("d_drinks", "discovery", ("tea", "milk", "juice")),
    ("d_planets", "discovery", ("mars", "earth", "venus")),
    ("d_trees", "discovery", ("oak", "pine", "elm")),
    ("d_body", "discovery", ("head", "hand", "foot")),
    ("d_music", "discovery", ("drum", "flute", "harp")),
    ("d_clothes", "discovery", ("hat", "coat", "shoe")),
    ("h_beasts", "heldout", ("wolf", "bear", "fox")),
    ("h_matter", "heldout", ("glass", "stone", "iron")),
    ("h_tools", "heldout", ("hammer", "saw", "drill")),
    ("h_vehicles", "heldout", ("car", "bus", "train")),
    ("h_foods", "heldout", ("wine", "fish", "bread")),
    ("h_metals3", "heldout", ("tin", "lead", "zinc")),
]

_P2_ROWS = [
    ("d_moon", "discovery", ("moon", "star")),
    ("d_sun", "discovery", ("sun", "rain")),
    ("d_day", "discovery", ("day", "night")),
    ("d_hot", "discovery", ("hot", "cold")),
    ("d_up", "discovery", ("up", "down")),
    ("d_yes", "discovery", ("yes", "no")),
    ("d_left", "discovery", ("left", "right")),
    ("d_black", "discovery", ("black", "white")),
    ("d_in", "discovery", ("in", "out")),
    ("d_win", "discovery", ("win", "lose")),
    ("d_salt", "discovery", ("salt", "pepper")),
    ("d_king", "discovery", ("king", "queen")),
    ("h_metals", "heldout", ("gold", "silver")),
    ("h_north", "heldout", ("north", "south")),
    ("h_open", "heldout", ("open", "shut")),
    ("h_true", "heldout", ("true", "false")),
]


def _successor_prompts(rows: Sequence[tuple[str, str, str, str]]) -> list[CircuitPrompt]:
    """Successor / counting. Distractor = copy-the-last-token, which subtracts
    induction/copying out: an induction head would echo the final token, the
    successor mechanism emits the next item in the order."""
    return [CircuitPrompt(eid, fam, prompt, " " + nxt, " " + prompt.split()[-1]) for eid, fam, prompt, nxt in rows]


_SUCCESSOR_ROWS = [
    ("s_count", "discovery", "one two three four five six seven eight", "nine"),
    ("s_count_b", "discovery", "two three four five six seven eight nine", "ten"),
    ("s_count_c", "discovery", "four five six seven eight nine ten eleven", "twelve"),
    ("s_count_d", "discovery", "three four five six seven eight nine ten", "eleven"),
    ("s_months", "discovery", "January February March April May June July August", "September"),
    ("s_months_b", "discovery", "February March April May June July August September", "October"),
    ("s_months_c", "discovery", "March April May June July August September October", "November"),
    ("s_letters", "discovery", "A B C D E F G H", "I"),
    ("s_letters_b", "discovery", "C D E F G H I J", "K"),
    ("s_letters_d", "discovery", "H I J K L M N O", "P"),
    ("s_letters_e", "discovery", "K L M N O P Q R", "S"),
    ("s_letters_f", "discovery", "M N O P Q R S T", "U"),
    ("s_letters2", "heldout", "P Q R S T U V W", "X"),
    ("s_letters_h", "heldout", "R S T U V W X Y", "Z"),
    ("s_ordinal", "heldout", "first second third fourth fifth sixth seventh eighth", "ninth"),
    ("s_count_h", "heldout", "five six seven eight nine ten eleven twelve", "thirteen"),
]


def _ioi_prompts(rows: Sequence[tuple[str, str, str, str]]) -> list[CircuitPrompt]:
    """Single fixed IOI template; the held-out axis is the name pair."""
    tmpl = "When {a} and {b} went to the store {b} gave the key to"
    return [
        CircuitPrompt(eid, fam, tmpl.format(a=a, b=b), " " + a, " " + b)
        for eid, fam, a, b in rows
    ]


_IOI_ROWS = [
    ("ioi_1", "discovery", "Anna", "Mark"),
    ("ioi_2", "discovery", "Paul", "Sara"),
    ("ioi_3", "discovery", "Lisa", "John"),
    ("ioi_4", "discovery", "Tom", "Kate"),
    ("ioi_5", "discovery", "Mary", "Fred"),
    ("ioi_6", "discovery", "Jane", "Carl"),
    ("ioi_7", "discovery", "Lucy", "Dave"),
    ("ioi_8", "discovery", "Anne", "Greg"),
    ("ioi_9", "discovery", "Sam", "Beth"),
    ("ioi_10", "discovery", "Mike", "Lara"),
    ("ioi_11", "discovery", "Adam", "Ruth"),
    ("ioi_12", "discovery", "Neil", "Joan"),
    ("ioi_h1", "heldout", "Emma", "Jack"),
    ("ioi_h2", "heldout", "Ben", "Rose"),
    ("ioi_h3", "heldout", "Cole", "Faye"),
    ("ioi_h4", "heldout", "Erik", "Gail"),
]


_AGREEMENT_ROWS = [
    ("ag_1", "discovery", "The dogs near the fence", " are", " is"),
    ("ag_2", "discovery", "The dog near the fences", " is", " are"),
    ("ag_3", "discovery", "The books near the shelf", " are", " is"),
    ("ag_4", "discovery", "The book near the shelves", " is", " are"),
    ("ag_5", "discovery", "The cups near the tray", " are", " is"),
    ("ag_6", "discovery", "The cup near the trays", " is", " are"),
    ("ag_7", "discovery", "The keys near the door", " are", " is"),
    ("ag_8", "discovery", "The key near the doors", " is", " are"),
    ("ag_9", "discovery", "The cars near the gate", " are", " is"),
    ("ag_10", "discovery", "The car near the gates", " is", " are"),
    ("ag_11", "discovery", "The birds near the nest", " are", " is"),
    ("ag_12", "discovery", "The bird near the nests", " is", " are"),
    ("ag_h1", "heldout", "The plates near the sink", " are", " is"),
    ("ag_h2", "heldout", "The plate near the sinks", " is", " are"),
    ("ag_h3", "heldout", "The lamps near the wall", " are", " is"),
    ("ag_h4", "heldout", "The lamp near the walls", " is", " are"),
]

# Long-gap agreement: subject and number-attractor are far apart, a real test
# of whether a number-transport circuit appears once the adversary is wide.
_AGREEMENT_LONG_ROWS = [
    ("al_1", "discovery", "The dogs that lived near the old fence", " are", " is"),
    ("al_2", "discovery", "The dog that lived near the old fences", " is", " are"),
    ("al_3", "discovery", "The books that sat near the tall shelf", " are", " is"),
    ("al_4", "discovery", "The book that sat near the tall shelves", " is", " are"),
    ("al_5", "discovery", "The cups that stood near the round tray", " are", " is"),
    ("al_6", "discovery", "The cup that stood near the round trays", " is", " are"),
    ("al_7", "discovery", "The keys that hung near the front door", " are", " is"),
    ("al_8", "discovery", "The key that hung near the front doors", " is", " are"),
    ("al_9", "discovery", "The cars that parked near the wide gate", " are", " is"),
    ("al_10", "discovery", "The car that parked near the wide gates", " is", " are"),
    ("al_11", "discovery", "The birds that nested near the green hedge", " are", " is"),
    ("al_12", "discovery", "The bird that nested near the green hedges", " is", " are"),
    ("al_h1", "heldout", "The plates that dried near the deep sink", " are", " is"),
    ("al_h2", "heldout", "The plate that dried near the deep sinks", " is", " are"),
    ("al_h3", "heldout", "The lamps that glowed near the bare wall", " are", " is"),
    ("al_h4", "heldout", "The lamp that glowed near the bare walls", " is", " are"),
]


def _taskvec_prompts(prefix: str, rows: Sequence[tuple[str, str, str, str]]) -> list[CircuitPrompt]:
    """Antonym function vector. Few-shot antonym demos then a query word; the
    target is its antonym, the distractor is echoing the query."""
    return [
        CircuitPrompt(eid, fam, f"{prefix} {query}", " " + anto, " " + query)
        for eid, fam, query, anto in rows
    ]


_TASKVEC_PREFIX_D = "hot cold up down wet dry fast slow"
_TASKVEC_PREFIX_H = "open shut light dark hard soft rich poor"
_TASKVEC_ROWS = [
    ("tv_1", "discovery", "big", "small"),
    ("tv_2", "discovery", "tall", "short"),
    ("tv_3", "discovery", "happy", "sad"),
    ("tv_4", "discovery", "rich", "poor"),
    ("tv_5", "discovery", "high", "low"),
    ("tv_6", "discovery", "hard", "soft"),
    ("tv_7", "discovery", "light", "dark"),
    ("tv_8", "discovery", "full", "empty"),
    ("tv_9", "discovery", "young", "old"),
    ("tv_10", "discovery", "near", "far"),
    ("tv_11", "discovery", "loud", "quiet"),
    ("tv_12", "discovery", "weak", "strong"),
]
_TASKVEC_HELDOUT = [
    ("tv_h1", "heldout", "high", "low"),
    ("tv_h2", "heldout", "left", "right"),
    ("tv_h3", "heldout", "fast", "slow"),
    ("tv_h4", "heldout", "big", "small"),
]


def _recall_prompts(rows: Sequence[tuple[str, str, str, str]]) -> list[CircuitPrompt]:
    """Taxonomic recall (MLP-mediated): the heads-only contrast behavior."""
    tmpl = "In biology a {x} is a kind of"
    return [
        CircuitPrompt(eid, fam, tmpl.format(x=x), " " + cat, " " + wrong)
        for eid, fam, x, cat, wrong in rows
    ]


_RECALL_ROWS = [
    ("rc_1", "discovery", "sparrow", "bird", "fish"),
    ("rc_2", "discovery", "salmon", "fish", "bird"),
    ("rc_3", "discovery", "maple", "tree", "fish"),
    ("rc_4", "discovery", "poodle", "dog", "tree"),
    ("rc_5", "discovery", "eagle", "bird", "fish"),
    ("rc_6", "discovery", "tuna", "fish", "tree"),
    ("rc_7", "discovery", "oak", "tree", "dog"),
    ("rc_8", "discovery", "beagle", "dog", "bird"),
    ("rc_9", "discovery", "hawk", "bird", "dog"),
    ("rc_10", "discovery", "shark", "fish", "tree"),
    ("rc_11", "discovery", "birch", "tree", "fish"),
    ("rc_12", "discovery", "terrier", "dog", "bird"),
    ("rc_h1", "heldout", "trout", "fish", "tree"),
    ("rc_h2", "heldout", "robin", "bird", "dog"),
    ("rc_h3", "heldout", "willow", "tree", "fish"),
    ("rc_h4", "heldout", "boxer", "dog", "bird"),
]


def behavior_prompts(behavior: str) -> list[CircuitPrompt]:
    if behavior == "induction_p3":
        return _induction_prompts(_P3_ROWS, period=3)
    if behavior == "induction_p2":
        return _induction_prompts(_P2_ROWS, period=2)
    if behavior == "successor":
        return _successor_prompts(_SUCCESSOR_ROWS)
    if behavior == "ioi":
        return _ioi_prompts(_IOI_ROWS)
    if behavior == "agreement":
        return [CircuitPrompt(e, f, p, t, d) for e, f, p, t, d in _AGREEMENT_ROWS]
    if behavior == "agreement_long":
        return [CircuitPrompt(e, f, p, t, d) for e, f, p, t, d in _AGREEMENT_LONG_ROWS]
    if behavior == "taskvec":
        return _taskvec_prompts(_TASKVEC_PREFIX_D, _TASKVEC_ROWS) + _taskvec_prompts(
            _TASKVEC_PREFIX_H, _TASKVEC_HELDOUT
        )
    if behavior == "recall":
        return _recall_prompts(_RECALL_ROWS)
    raise ValueError(f"unknown behavior {behavior!r}")


def swa_prompts(length_words: int) -> list[CircuitPrompt]:
    """PART 4: programmatically build a POPULATION of long period-3 induction
    prompts at one shared length, to test whether the circuit reorganizes once
    prompts exceed Olmo-3's 4096-token sliding window. Each period-3 cycle from
    INDUCTION_P3 is repeated to ``length_words`` words (one short of a full
    cycle so the continuation is word 2); target/distractor as in INDUCTION_P3.

    Built in code, never hand-typed. NOTE: full attention-pattern capture at
    >4k tokens is memory-infeasible (eager attention is O(seq^2) per head per
    layer), so the SWA cell runs a reduced pipeline -- the caller is expected to
    keep ``length_words`` within what the GPU can capture, or to drive SWA
    through the ablation-only path.
    """
    period = 3
    n_words = (length_words // period) * period + (period - 1)
    out: list[CircuitPrompt] = []
    for example_id, family, words in _P3_ROWS:
        seq = [words[i % period] for i in range(n_words)]
        prompt = " ".join(seq)
        target = " " + words[n_words % period]
        distractor = " " + words[0]
        out.append(CircuitPrompt(f"swa{length_words}_{example_id}", family, prompt, target, distractor, period=period))
    return out


@dataclasses.dataclass
class TaskExample:
    prompt: CircuitPrompt
    target_id: int
    distractor_id: int
    prompt_len: int
    base_diff: float = 0.0


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def node_name(kind: str, layer: int, head: int | None = None) -> str:
    return f"L{layer}H{head}" if kind == "head" else f"MLP{layer}"


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def safe_ratio(num: float, denom: float) -> float | None:
    if abs(denom) < 1e-9:
        return None
    return float(num) / float(denom)


def rank_map(scores: dict[Any, float], *, reverse: bool = True, key_abs: bool = False) -> dict[Any, int]:
    def sort_key(item: tuple[Any, float]) -> float:
        return abs(item[1]) if key_abs else item[1]

    return {
        key: i + 1
        for i, (key, _score) in enumerate(sorted(scores.items(), key=sort_key, reverse=reverse))
    }


def screen_budgets(n_layers: int, n_heads: int) -> tuple[int, int, int]:
    """Broad but finite screen budgets so students see the disagreement between
    cheap screens (Lab 2 attribution + Lab 3 motifs) and actual causal effect
    (the central pedagogical payload of this lab). Too narrow a screen can make
    the whole exercise look like the model only needed three heads.
    """
    total = n_layers * n_heads
    top_attr = min(total, max(SCREEN_TOP_ATTRIBUTION_MIN, math.ceil(total * SCREEN_TOP_ATTRIBUTION_FRAC)))
    top_ind = min(total, max(SCREEN_TOP_INDUCTION_MIN, math.ceil(total * SCREEN_TOP_MOTIF_FRAC)))
    top_prev = min(total, max(SCREEN_TOP_PREV_MIN, math.ceil(total * SCREEN_TOP_MOTIF_FRAC)))
    return top_attr, top_ind, top_prev


def first_token_score(pattern: Any) -> float:
    """Mean attention mass assigned to token 0, excluding the trivial first row."""
    if pattern.shape[-1] <= 1:
        return 0.0
    return float(pattern[1:, 0].mean())


def mean_logit_diff(bundle: bench.ModelBundle, examples: Sequence[TaskExample]) -> float:
    diffs = []
    for ex in examples:
        logits = bench.run_with_residual_cache(bundle, ex.prompt.prompt).final_logits_last
        diffs.append(float(logits[ex.target_id] - logits[ex.distractor_id]))
    return statistics.fmean(diffs)


def describe_head_list(heads: Sequence[tuple[int, int]]) -> str:
    return ", ".join(node_name("head", *h) for h in heads) or "none"


# Node abstraction (FIX 3): a circuit node is a head or an MLP layer. Heads are
# ("head", layer, head); MLP layers are ("mlp", layer, -1). Heads-only scope
# never admits MLP nodes; heads_and_mlps makes them first-class.
Node = tuple[str, int, int]


def head_node(layer: int, head: int) -> Node:
    return ("head", layer, head)


def mlp_node(layer: int) -> Node:
    return ("mlp", layer, -1)


def node_label(node: Node) -> str:
    kind, layer, head = node
    return node_name(kind, layer, head if kind == "head" else None)


def split_nodes(nodes: Sequence[Node]) -> tuple[list[tuple[int, int]], list[int]]:
    heads = [(layer, head) for (kind, layer, head) in nodes if kind == "head"]
    mlps = [layer for (kind, layer, _h) in nodes if kind == "mlp"]
    return heads, mlps


def describe_nodes(nodes: Sequence[Node]) -> str:
    return ", ".join(node_label(n) for n in nodes) or "none"


def edge_strength_label(raw_fraction: float | None) -> str:
    if raw_fraction is None or raw_fraction <= 0:
        return "none"
    if raw_fraction < EDGE_MIN_ROUTED_FRACTION:
        return "below_threshold"
    if raw_fraction < EDGE_STRONG_ROUTED_FRACTION:
        return "weak"
    return "strong"


# ---------------------------------------------------------------------------
# Dataset validation and baseline
# ---------------------------------------------------------------------------


class CellAbort(Exception):
    """Abort a behavior cell with an explicit, recordable verdict rather than a
    crash. The matrix driver catches this and writes the verdict to the card so
    a refusal (INSUFFICIENT PROMPTS, MIXED_PERIOD) is a first-class result."""

    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


def build_dataset(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    max_examples: int,
    *,
    behavior: str,
    prompts: Sequence[CircuitPrompt] | None = None,
    allow_mixed_period: bool = False,
    min_positive: int = MIN_POSITIVE,
) -> tuple[list[TaskExample], list[TaskExample], dict[str, Any], int]:
    """Hard prompt-hygiene gate (FIX 6 + FIX 5).

    For this model+tokenizer:
    * group prompts by token length and adopt the modal length as the run
      length; every prompt of a different length, or with a multi-token
      target/distractor, is EXCLUDED and itemized (never silently used);
    * compute the base logit gap per usable prompt and count baseline-positive
      discovery prompts; if fewer than ``min_positive`` survive, ABORT the cell
      (no tiny-n cards);
    * refuse a run whose induction prompts mix cycle periods unless
      ``allow_mixed_period`` (the base metric must never average two
      mechanisms).

    Emits ``prompt_hygiene_report.md`` and ``tokenization_and_baseline.csv``.
    """
    tokenizer = bundle.tokenizer
    pool = list(prompts) if prompts is not None else behavior_prompts(behavior)

    rows: list[dict[str, Any]] = []
    for cp in pool:
        target_ids = tokenizer.encode(cp.target, add_special_tokens=False)
        distractor_ids = tokenizer.encode(cp.distractor, add_special_tokens=False)
        prompt_ids = tokenizer.encode(cp.prompt, add_special_tokens=False)
        rows.append(
            {
                "cp": cp,
                "example_id": cp.example_id,
                "family": cp.family,
                "prompt": cp.prompt,
                "period": cp.period if cp.period is not None else "",
                "n_prompt_tokens": len(prompt_ids),
                "prompt_tokens": " ".join(bench.visible_token(tokenizer.decode([i])) for i in prompt_ids),
                "target": bench.visible_token(cp.target),
                "distractor": bench.visible_token(cp.distractor),
                "target_ids": target_ids,
                "distractor_ids": distractor_ids,
                "single_token_answers": len(target_ids) == 1 and len(distractor_ids) == 1,
            }
        )

    # Adopt the modal prompt length as the run length. A prompt of any other
    # length is an offender to be reported, not silently coerced.
    length_counts: dict[int, int] = {}
    for row in rows:
        length_counts[row["n_prompt_tokens"]] = length_counts.get(row["n_prompt_tokens"], 0) + 1
    run_len = max(length_counts, key=lambda L: (length_counts[L], -L))

    discovery: list[TaskExample] = []
    heldout: list[TaskExample] = []
    excluded: list[dict[str, Any]] = []
    periods_used: set[int] = set()

    for row in rows:
        cp = row["cp"]
        reasons: list[str] = []
        if row["n_prompt_tokens"] != run_len:
            reasons.append(f"length {row['n_prompt_tokens']} != run length {run_len}")
        if len(row["target_ids"]) != 1:
            reasons.append(f"target is {len(row['target_ids'])} tokens")
        if len(row["distractor_ids"]) != 1:
            reasons.append(f"distractor is {len(row['distractor_ids'])} tokens")

        row["usable"] = not reasons
        row["exclude_reason"] = "; ".join(reasons)
        if reasons:
            excluded.append({"example_id": cp.example_id, "family": cp.family, "reason": row["exclude_reason"]})
            row["baseline_logit_diff"] = ""
            row["baseline_pass"] = ""
            continue

        ex = TaskExample(cp, row["target_ids"][0], row["distractor_ids"][0], run_len)
        logits = bench.run_with_residual_cache(bundle, cp.prompt).final_logits_last
        ex.base_diff = float(logits[ex.target_id] - logits[ex.distractor_id])
        row["baseline_logit_diff"] = round(ex.base_diff, 4)
        row["baseline_pass"] = ex.base_diff > 0
        if cp.period is not None:
            periods_used.add(cp.period)
        if ex.base_diff > 0:
            (discovery if cp.family == "discovery" else heldout).append(ex)
        elif cp.family == "heldout":
            pass  # baseline-negative held-out prompts are simply not scored
        else:
            excluded.append(
                {"example_id": cp.example_id, "family": cp.family, "reason": f"baseline gap {ex.base_diff:+.3f} <= 0"}
            )

    if max_examples and max_examples > 0:
        discovery = discovery[:max_examples]

    # --- write diagnostics before any abort, so refusals are auditable -------
    csv_rows = [
        {k: v for k, v in row.items() if k not in ("cp", "target_ids", "distractor_ids")}
        for row in rows
    ]
    report = ctx.path("diagnostics", "tokenization_and_baseline.csv")
    bench.write_csv(report, csv_rows)
    ctx.register_artifact(report, "diagnostic", "Tokenization contract and baseline logit gap for every Lab 6 prompt.")

    info: dict[str, Any] = {
        "behavior": behavior,
        "run_length_tokens": run_len,
        "n_pool": len(pool),
        "n_discovery_positive": len(discovery),
        "n_heldout_positive": len(heldout),
        "n_excluded": len(excluded),
        "periods_used": sorted(periods_used),
        "excluded": excluded,
        "min_positive": min_positive,
    }
    _write_hygiene_report(ctx, behavior, run_len, rows, discovery, heldout, excluded, periods_used, min_positive)

    if len(periods_used) > 1 and not allow_mixed_period:
        raise CellAbort(
            "MIXED_PERIOD",
            f"prompts mix cycle periods {sorted(periods_used)}; refuse to average two mechanisms "
            "into one base metric (pass --allow-mixed-period to override).",
        )
    if len(discovery) < min_positive:
        raise CellAbort(
            "INSUFFICIENT PROMPTS",
            f"only {len(discovery)} baseline-positive discovery prompts survive at run length {run_len} "
            f"(need >= {min_positive}); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.",
        )
    return discovery, heldout, info, run_len


def _write_hygiene_report(
    ctx: bench.RunContext,
    behavior: str,
    run_len: int,
    rows: Sequence[dict[str, Any]],
    discovery: Sequence[TaskExample],
    heldout: Sequence[TaskExample],
    excluded: Sequence[dict[str, Any]],
    periods_used: set[int],
    min_positive: int,
) -> None:
    lines = [
        f"# Prompt hygiene report: `{behavior}`",
        "",
        f"- Model: `{getattr(ctx, 'model_id', '') or 'unknown'}`",
        f"- Run length (modal token length adopted): **{run_len}**",
        f"- Cycle periods present: {sorted(periods_used) or 'n/a (non-cyclic behavior)'}",
        f"- Baseline-positive discovery prompts: **{len(discovery)}** (gate requires >= {min_positive})",
        f"- Baseline-positive held-out prompts: **{len(heldout)}**",
        f"- Excluded / unusable prompts: **{len(excluded)}**",
        "",
        "## Excluded prompts (itemized — nothing is silently dropped)",
        "",
    ]
    if excluded:
        lines += ["| example | family | reason |", "|---|---|---|"]
        lines += [f"| `{e['example_id']}` | {e['family']} | {e['reason']} |" for e in excluded]
    else:
        lines.append("None — every pooled prompt was usable at the run length.")
    lines += [
        "",
        "## Usable prompts and baseline gaps",
        "",
        "| example | family | period | n_tok | base logit gap | positive |",
        "|---|---|---|---:|---:|:---:|",
    ]
    for row in rows:
        if not row.get("usable"):
            continue
        lines.append(
            f"| `{row['example_id']}` | {row['family']} | {row['period']} | {row['n_prompt_tokens']} | "
            f"{row.get('baseline_logit_diff', '')} | {'yes' if row.get('baseline_pass') else 'no'} |"
        )
    path = ctx.path("prompt_hygiene_report.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Per-model prompt-hygiene gate: run length, exclusions, and baseline gaps.")


# ---------------------------------------------------------------------------
# Metric under intervention
# ---------------------------------------------------------------------------


def metric_under_ablation(
    bundle: bench.ModelBundle,
    examples: Sequence[TaskExample],
    head_anatomy: bench.HeadAnatomy,
    comp_anatomy: bench.ComponentAnatomy,
    heads: Sequence[tuple[int, int]],
    mlps: Sequence[int],
    head_means: Any,
    mlp_means: Any,
) -> float:
    """Mean logit(target) minus logit(distractor) with a node set mean-ablated.

    Dataset-mean ablation (not zero-ablation) keeps the intervention closer to
    the data manifold while still removing the prompt-specific computation of
    the ablated heads. This is the "off switch" that defines the circuit for
    this particular off-distribution. A different mean (or zero) defines a
    different circuit.
    """
    diffs: list[float] = []
    for ex in examples:
        logits = bench.run_with_node_set_ablation(
            bundle,
            ex.prompt.prompt,
            head_anatomy,
            comp_anatomy,
            heads=heads,
            mlps=mlps,
            head_means=head_means,
            mlp_means=mlp_means,
        )
        diffs.append(float(logits[ex.target_id] - logits[ex.distractor_id]))
    return statistics.fmean(diffs)


def metric_under_resample(
    bundle: bench.ModelBundle,
    examples: Sequence[TaskExample],
    head_anatomy: bench.HeadAnatomy,
    comp_anatomy: bench.ComponentAnatomy,
    heads: Sequence[tuple[int, int]],
    mlps: Sequence[int],
    caps_by_id: dict[str, tuple[Any, Any]],
    draws: int,
) -> float:
    """Resample / interchange ablation (FIX 2): the primary off-distribution.

    Instead of replacing an ablated node's activation with the dataset MEAN
    (which removes the model's own suppression structure and inflates
    faithfulness above 1.0), replace it with the activation that node took on a
    DIFFERENT in-distribution prompt. This is exactly the mean-ablation
    machinery fed one other prompt's captured (o_in_last, mlp_contrib) instead
    of the average. We average over ``draws`` deterministic source rotations for
    stability. ``caps_by_id`` maps every resample-source example id to its
    full-position captures.

    The examples and the resample sources come from the SAME family, so the
    intervention stays within distribution. Returns the mean target-distractor
    logit gap.
    """
    source_ids = [ex.prompt.example_id for ex in examples]
    n = len(source_ids)
    if n < 2 or draws <= 0:
        # Degenerate: cannot resample from a single prompt. Fall back to the
        # mean of the captured sources so the call still returns a number; the
        # hygiene gate already forbids tiny-n, so this only guards edge cases.
        import torch

        head_means = torch.stack([caps_by_id[i][0] for i in source_ids]).mean(dim=0) if caps_by_id else None
        mlp_means = torch.stack([caps_by_id[i][1] for i in source_ids]).mean(dim=0) if caps_by_id else None
        return metric_under_ablation(bundle, examples, head_anatomy, comp_anatomy, heads, mlps, head_means, mlp_means)

    diffs: list[float] = []
    for i, ex in enumerate(examples):
        n_draws = min(draws, n - 1)
        for k in range(n_draws):
            src_id = source_ids[(i + 1 + k) % n]
            head_src, mlp_src = caps_by_id[src_id]
            logits = bench.run_with_node_set_ablation(
                bundle,
                ex.prompt.prompt,
                head_anatomy,
                comp_anatomy,
                heads=heads,
                mlps=mlps,
                head_means=head_src,
                mlp_means=mlp_src,
            )
            diffs.append(float(logits[ex.target_id] - logits[ex.distractor_id]))
    return statistics.fmean(diffs)


# ---------------------------------------------------------------------------
# Plotting and evidence-table helpers
# ---------------------------------------------------------------------------

MOTIF_COLOR_FALLBACKS = {
    "induction": "#E69F00",
    "previous_token": "#0072B2",
    "first_token_sink": "#7E57C2",
    "diffuse": "#999999",
    "other": "#555555",
    "support_mlp": "#CC79A7",
}

MOTIF_MARKER_FALLBACKS = {
    "induction": "*",
    "previous_token": "o",
    "first_token_sink": "s",
    "diffuse": ".",
    "other": "x",
    "support_mlp": "D",
}

FAMILY_COLOR_FALLBACKS = {
    "discovery": "#009E73",
    "heldout": "#0072B2",
    "undefined": "#888888",
}


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _motif_color(label: str) -> str:
    func = getattr(bench, "plot_motif_color", None)
    if callable(func):
        return func(label, MOTIF_COLOR_FALLBACKS.get(label, "#555555"))
    return MOTIF_COLOR_FALLBACKS.get(str(label), "#555555")


def _motif_marker(label: str) -> str:
    func = getattr(bench, "plot_motif_marker", None)
    marker = func(label, MOTIF_MARKER_FALLBACKS.get(label, "o")) if callable(func) else MOTIF_MARKER_FALLBACKS.get(str(label), "o")
    # Filled markers keep dense scatterplots readable and avoid Matplotlib edgecolor warnings.
    return "o" if marker in {"x", ".", ","} else marker


def _family_color(family: str) -> str:
    func = getattr(bench, "plot_category_color", None)
    if callable(func):
        return func(family, FAMILY_COLOR_FALLBACKS.get(family, "#555555"))
    return FAMILY_COLOR_FALLBACKS.get(str(family), "#555555")


def _component_color(component: str) -> str:
    func = getattr(bench, "plot_component_color", None)
    if callable(func):
        return func(component, "#555555")
    return {"head": "#0072B2", "mlp": "#E69F00", "attn": "#0072B2"}.get(str(component), "#555555")


def _lighten(color: str, amount: float = 0.55) -> str:
    func = getattr(bench, "lighten_color", None)
    if callable(func):
        return func(color, amount)
    try:
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        return mcolors.to_hex(tuple(c + (1.0 - c) * amount for c in rgb))
    except Exception:
        return color


def _panel_label(ax: Any, label: str) -> None:
    func = getattr(bench, "add_panel_label", None)
    if callable(func):
        func(ax, label)
    else:
        ax.text(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def _zero_line(ax: Any, axis: str = "y", **kwargs: Any) -> None:
    func = getattr(bench, "add_zero_line", None)
    if callable(func):
        func(ax, axis=axis, **kwargs)
    else:
        if axis == "y":
            ax.axhline(0, color=kwargs.get("color", "black"), linewidth=kwargs.get("linewidth", 0.8), alpha=kwargs.get("alpha", 0.7))
        else:
            ax.axvline(0, color=kwargs.get("color", "black"), linewidth=kwargs.get("linewidth", 0.8), alpha=kwargs.get("alpha", 0.7))


def _style(ax: Any, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    func = getattr(bench, "style_ax", None)
    if callable(func):
        func(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    else:
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def plot_reading_guide_rows() -> list[dict[str, str]]:
    """A small map from artifact to the concept it teaches."""
    return [
        {
            "artifact": "plots/circuit_discovery_dashboard.png",
            "question": "What is the overall manual circuit claim?",
            "read_for": "F/C/M, pruning, candidate quality, and prompt-level failures in one place.",
        },
        {
            "artifact": "plots/candidate_evidence_matrix.png",
            "question": "Which heads survived the evidence ladder?",
            "read_for": "cheap screen signals, motif labels, causal drop, and final membership side by side.",
        },
        {
            "artifact": "plots/causal_motif_atlas.png",
            "question": "Where are causally useful heads in layer/head space?",
            "read_for": "screened heads, signed causal effects, motif labels, and final circuit membership.",
        },
        {
            "artifact": "plots/screen_vs_causal.png",
            "question": "Where did cheap screening lie?",
            "read_for": "heads with high attribution or motif scores but near-zero or negative causal drop.",
        },
        {
            "artifact": "plots/minimality_ledger.png",
            "question": "Did every final head earn its rent?",
            "read_for": "marginal faithfulness loss when each kept head is removed.",
        },
        {
            "artifact": "plots/prompt_failure_scatter.png",
            "question": "Where does the circuit fail or over-recover?",
            "read_for": "base behavior strength versus circuit-only faithfulness for each prompt.",
        },
        {
            "artifact": "plots/edge_interaction_map.png",
            "question": "Was an ordered edge earned?",
            "read_for": "previous-token to induction interaction size, fraction, and layer ordering.",
        },
    ]


def build_circuit_evidence_matrix(
    cand_rows: Sequence[dict[str, Any]],
    minimality_rows: Sequence[dict[str, Any]],
    circuit: Sequence[tuple[int, int]],
    edge: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Join cheap screen, motif, ablation, pruning, and edge status into one table."""
    circuit_set = {node_name("head", *key) for key in circuit}
    marginal = {row["node"]: row for row in minimality_rows}
    edge_roles: dict[str, str] = {}
    if edge and edge.get("claimed"):
        edge_roles[node_name("head", *edge["from"])] = "edge_source"
        edge_roles[node_name("head", *edge["to"])] = "edge_target"

    out: list[dict[str, Any]] = []
    for row in cand_rows:
        node = str(row.get("node", ""))
        if row.get("kind") != "head":
            continue
        cheap_rank = _to_int(row.get("cheap_rank"), 10**9)
        causal_rank = _to_int(row.get("causal_rank"), 10**9)
        causal_drop = _to_float(row.get("causal_drop"), 0.0)
        tags: list[str] = []
        if node in circuit_set:
            tags.append("final_circuit")
        if causal_drop > 0:
            tags.append("positive_causal_drop")
        elif causal_drop < 0:
            tags.append("negative_causal_drop")
        motif = str(row.get("motif_label", "other"))
        if motif not in ("", "other"):
            tags.append(f"motif:{motif}")
        if abs(cheap_rank - causal_rank) >= 8 and causal_rank < 10**9:
            tags.append("screen_causal_disagreement")
        if edge_roles.get(node):
            tags.append(edge_roles[node])
        mrow = marginal.get(node, {})
        out.append(
            {
                "node": node,
                "layer": row.get("layer", ""),
                "head": row.get("head", ""),
                "motif_label": motif,
                "screen_reason": row.get("screen_reason", ""),
                "cheap_rank": row.get("cheap_rank", ""),
                "causal_rank": row.get("causal_rank", ""),
                "rank_gap_cheap_minus_causal": row.get("rank_gap_cheap_minus_causal", ""),
                "mean_attr": row.get("mean_attr", ""),
                "abs_attr": round(abs(_to_float(row.get("mean_attr"), 0.0)), 6),
                "induction_score": row.get("induction_score", ""),
                "prev_token_score": row.get("prev_token_score", ""),
                "first_token_score": row.get("first_token_score", ""),
                "single_ablated_metric": row.get("single_ablated_metric", ""),
                "causal_drop": row.get("causal_drop", ""),
                "in_final_circuit": node in circuit_set,
                "marginal_value": mrow.get("marginal_value", ""),
                "minimality_passes_positive_marginal": mrow.get("minimality_passes_positive_marginal", ""),
                "edge_role": edge_roles.get(node, ""),
                "evidence_tags": ";".join(tags),
            }
        )
    return sorted(out, key=lambda r: (not bool(r["in_final_circuit"]), -_to_float(r.get("causal_drop"), 0.0), _to_int(r.get("cheap_rank"), 10**9)))


def build_prompt_failure_modes(rows: Sequence[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        faith = row.get("faithfulness")
        faith_float = _to_float(faith, float("nan"))
        base = _to_float(row.get("base_diff"), float("nan"))
        circuit_diff = _to_float(row.get("circuit_diff"), float("nan"))
        if not _is_finite(faith_float):
            mode = "undefined_base_not_target_preferring"
            gap_to_floor = ""
            over_recovery = ""
        elif faith_float < floor:
            mode = "under_floor"
            gap_to_floor = round(faith_float - floor, 5)
            over_recovery = round(max(0.0, faith_float - 1.0), 5)
        elif faith_float > 1.05:
            mode = "over_recovery"
            gap_to_floor = round(faith_float - floor, 5)
            over_recovery = round(faith_float - 1.0, 5)
        else:
            mode = "preserved"
            gap_to_floor = round(faith_float - floor, 5)
            over_recovery = round(max(0.0, faith_float - 1.0), 5)
        out.append(
            {
                **row,
                "faithfulness": round(faith_float, 5) if _is_finite(faith_float) else None,
                "base_diff": round(base, 5) if _is_finite(base) else "",
                "circuit_diff": round(circuit_diff, 5) if _is_finite(circuit_diff) else "",
                "failure_mode": mode,
                "gap_to_floor": gap_to_floor,
                "over_recovery_above_full_model": over_recovery,
            }
        )
    return out


def plot_screen_vs_causal(ctx: bench.RunContext, cand_rows: Sequence[dict[str, Any]]) -> None:
    """Show the central lesson: cheap screens and causal effects disagree."""
    import matplotlib.pyplot as plt

    heads = [r for r in cand_rows if r.get("kind") == "head"]
    if not heads:
        return

    fig, (ax_rank, ax_attr) = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    for ax in (ax_rank, ax_attr):
        _style(ax)
        _zero_line(ax)

    for r in heads:
        motif = str(r.get("motif_label", "other"))
        color = _motif_color(motif)
        marker = _motif_marker(motif)
        drop = _to_float(r.get("causal_drop"), 0.0)
        cheap = _to_float(r.get("cheap_rank"), 0.0)
        attr = abs(_to_float(r.get("mean_attr"), 0.0))
        alpha = 0.92 if drop > 0 else 0.45
        size = 78 if drop > 0 else 44
        ax_rank.scatter(cheap, drop, s=size, color=color, marker=marker, alpha=alpha, edgecolors="black", linewidths=0.35)
        ax_attr.scatter(attr, drop, s=size, color=color, marker=marker, alpha=alpha, edgecolors="black", linewidths=0.35)

    label_nodes: set[str] = set()
    label_nodes.update(str(r["node"]) for r in sorted(heads, key=lambda x: -_to_float(x.get("causal_drop"), 0.0))[:8])
    label_nodes.update(str(r["node"]) for r in sorted(heads, key=lambda x: _to_float(x.get("cheap_rank"), 10**9))[:6])
    label_nodes.update(str(r["node"]) for r in sorted(heads, key=lambda x: abs(_to_float(x.get("rank_gap_cheap_minus_causal"), 0.0)), reverse=True)[:4])
    for r in heads:
        if str(r.get("node")) in label_nodes:
            ax_rank.annotate(str(r["node"]), (_to_float(r.get("cheap_rank"), 0.0), _to_float(r.get("causal_drop"), 0.0)), textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax_attr.annotate(str(r["node"]), (abs(_to_float(r.get("mean_attr"), 0.0)), _to_float(r.get("causal_drop"), 0.0)), textcoords="offset points", xytext=(4, 4), fontsize=7)

    _style(ax_rank, title="Cheap screen rank vs causal effect", xlabel="cheap screen rank, lower is better", ylabel="single-head causal drop")
    _style(ax_attr, title="Frozen attribution magnitude vs causal effect", xlabel="|direct-logit attribution|", ylabel="single-head causal drop")
    for motif in ("previous_token", "induction", "first_token_sink", "other"):
        if any(str(r.get("motif_label", "other")) == motif for r in heads):
            ax_attr.scatter([], [], color=_motif_color(motif), marker=_motif_marker(motif), label=motif)
    ax_attr.legend(fontsize=8, loc="best")
    fig.suptitle("Cheap screening is a hypothesis generator, not a circuit claim")
    bench.save_figure(ctx, fig, "screen_vs_causal.png", "Cheap screen rank and attribution magnitude against single-head mean-ablation effect.")


def plot_prune_trajectory(ctx: bench.RunContext, trajectory: Sequence[dict[str, Any]], *, floor: float) -> None:
    if not trajectory:
        return
    fig, ax = bench.new_figure(figsize=(8.8, 5.3))
    xs = [t["n_nodes"] for t in trajectory]
    ys = [_to_float(t["faithfulness"], 0.0) for t in trajectory]
    ax.plot(xs, ys, marker="o", linewidth=2.3, color=_family_color("discovery"))
    ax.fill_between(xs, [floor for _ in xs], ys, where=[y >= floor for y in ys], alpha=0.10, color=_family_color("discovery"), interpolate=True)
    ax.fill_between(xs, ys, [floor for _ in xs], where=[y < floor for y in ys], alpha=0.10, color="#D55E00", interpolate=True)
    for t in trajectory:
        if t.get("removed"):
            ax.annotate(f"-{t['removed']}", (t["n_nodes"], _to_float(t["faithfulness"], 0.0)), textcoords="offset points", xytext=(2, 8), fontsize=7, rotation=30)
    ax.axhline(floor, color="#D55E00", linewidth=1.1, linestyle="--", label=f"floor = {floor:.2f}")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.45, label="full-model baseline")
    ax.invert_xaxis()
    _style(ax, title="Greedy pruning: what the behavior costs, node by node", xlabel="circuit size, heads kept", ylabel="faithfulness, complement mean-ablated")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "prune_trajectory.png", "Faithfulness at each greedy pruning step, with floor and full-model reference.")


def plot_circuit_graph(
    ctx: bench.RunContext,
    circuit: Sequence[tuple[int, int]],
    head_labels: dict[tuple[int, int], str],
    mlp_support: Sequence[dict[str, Any]],
    edge: dict[str, Any] | None,
    n_layers: int,
    n_heads: int,
) -> None:
    fig, ax = bench.new_figure(figsize=(10.8, 6.2))
    for layer, head in circuit:
        label = head_labels.get((layer, head), "other")
        ax.scatter(layer, head, s=210, color=_motif_color(label), marker=_motif_marker(label), zorder=3, edgecolors="black", linewidths=0.8)
        ax.annotate(f"L{layer}H{head}\n{label}", (layer, head), textcoords="offset points", xytext=(6, 6), fontsize=7)

    shown_mlps = list(mlp_support[:5])
    for i, r in enumerate(shown_mlps):
        y = n_heads + 1 + 0.70 * (i % 3)
        ax.scatter(_to_float(r.get("layer"), 0.0), y, s=145, marker="s", color=_component_color("mlp"), zorder=3, edgecolors="black", linewidths=0.8)
        ax.annotate(f"MLP{r['layer']}\ndrop {_to_float(r.get('causal_drop'), 0.0):+.2f}", (_to_float(r.get("layer"), 0.0), y), textcoords="offset points", xytext=(4, 6 + 4 * (i % 2)), fontsize=7)
    if len(mlp_support) > len(shown_mlps):
        ax.text(0.99, 0.98, f"+{len(mlp_support) - len(shown_mlps)} support MLPs in card", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=_component_color("mlp"))

    if edge is not None and edge.get("claimed"):
        l1, h1 = edge["from"]
        l2, h2 = edge["to"]
        if (l1, h1) not in circuit:
            ax.scatter(l1, h1, s=200, facecolors="none", edgecolors=_motif_color("previous_token"), linewidths=1.8, zorder=3)
            ax.annotate(f"L{l1}H{h1}\nscreened, pruned", (l1, h1), textcoords="offset points", xytext=(6, -18), fontsize=7, color=_motif_color("previous_token"))
        ax.annotate("", xy=(l2, h2), xytext=(l1, h1), arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2.0, "shrinkA": 13, "shrinkB": 13})
        mid_x, mid_y = (l1 + l2) / 2, (h1 + h2) / 2
        ax.annotate(f"{edge['strength']} interaction\n{edge['raw_interaction_fraction']:.0%}", (mid_x, mid_y), textcoords="offset points", xytext=(0, 12), fontsize=8, ha="center")

    for motif in ("previous_token", "induction", "first_token_sink", "other"):
        if any(head_labels.get(key, "other") == motif for key in circuit):
            ax.scatter([], [], color=_motif_color(motif), marker=_motif_marker(motif), label=motif, edgecolors="black", linewidths=0.6)
    if shown_mlps:
        ax.scatter([], [], color=_component_color("mlp"), marker="s", label="support MLP", edgecolors="black", linewidths=0.6)
    ax.set_xlim(-1, n_layers)
    ax.set_ylim(-1.5, n_heads + 4.8)
    _style(ax, title="Validated heads-only routing circuit", xlabel="layer", ylabel="head index; squares above are supporting MLPs")
    ax.legend(fontsize=8, loc="upper left", ncols=2)
    bench.save_figure(ctx, fig, "circuit_graph.png", "Circuit heads, motif labels, support MLPs, and the claimed edge if any.")


def plot_fcm(ctx: bench.RunContext, fcm: dict[str, Any], *, floor: float) -> None:
    fig, ax = bench.new_figure(figsize=(8.2, 5.2))
    groups: list[tuple[str, str, float]] = []
    for family in ("discovery", "heldout"):
        if family in fcm:
            groups.append((family, "faithfulness", _to_float(fcm[family].get("faithfulness"), 0.0)))
            groups.append((family, "completeness_effect", _to_float(fcm[family].get("completeness_effect"), 0.0)))
    xs = list(range(len(groups)))
    colors = [_family_color(fam) if metric == "faithfulness" else _lighten(_family_color(fam), 0.35) for fam, metric, _ in groups]
    bars = ax.bar(xs, [v for _, _, v in groups], color=colors, alpha=0.92)
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.axhline(floor, color="#D55E00", linewidth=1.1, linestyle="--", label="faithfulness floor")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{metric.replace('_', ' ')}\n{family}" for family, metric, _ in groups], fontsize=8)
    _style(ax, title="Circuit scorecard: preservation and destruction of behavior", ylabel="fraction of base behavior")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "circuit_scorecard.png", "Faithfulness and completeness effect on discovery and held-out families.")


def plot_prompt_faithfulness(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(9.6, 5.2))
    ordered = sorted(rows, key=lambda r: (_to_float(r.get("faithfulness"), -999.0)))
    xs = list(range(len(ordered)))
    ys = [_to_float(r.get("faithfulness"), 0.0) for r in ordered]
    colors = [_family_color(str(r.get("family", "undefined"))) for r in ordered]
    bars = ax.bar(xs, ys, color=colors, alpha=0.88)
    ax.axhline(FAITHFULNESS_FLOOR, color="#D55E00", linestyle="--", linewidth=1.1, label="floor")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.35, label="full model")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r.get("example_id", "")) for r in ordered], rotation=35, ha="right", fontsize=8)
    for bar, row in zip(bars, ordered):
        faith = _to_float(row.get("faithfulness"), float("nan"))
        if math.isfinite(faith):
            ax.annotate(f"{faith:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=7, rotation=90)
    _style(ax, title="Failure cases and over-recovery: per-prompt circuit faithfulness", ylabel="per-prompt faithfulness")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "per_prompt_faithfulness.png", "Per-prompt faithfulness sorted from weakest to strongest.")


def plot_edge_interactions(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(8.8, 5.0))
    top = sorted(rows, key=lambda r: _to_float(r.get("interaction"), -999.0), reverse=True)[:10]
    xs = list(range(len(top)))
    ys = [_to_float(r.get("interaction"), 0.0) for r in top]
    colors = [
        _motif_color("induction") if str(r.get("edge_strength")) in {"strong", "weak"} else _lighten(_motif_color("induction"), 0.55)
        for r in top
    ]
    bars = ax.bar(xs, ys, color=colors, alpha=0.9)
    _zero_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r.get("edge", "")) for r in top], rotation=35, ha="right", fontsize=8)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    _style(ax, title="Ordered previous-token to induction interaction checks", ylabel="interaction = effect(prev) - effect(prev | induction ablated)")
    bench.save_figure(ctx, fig, "edge_interactions.png", "Ablation-interaction evidence for the one edge claim.")


def plot_minimality_ledger(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(8.4, max(3.8, 0.35 * len(rows) + 1.8)))
    ordered = sorted(rows, key=lambda r: _to_float(r.get("marginal_value"), 0.0))
    ys = list(range(len(ordered)))
    vals = [_to_float(r.get("marginal_value"), 0.0) for r in ordered]
    labels = [str(r.get("node", "")) for r in ordered]
    colors = [_motif_color(str(r.get("motif_label", "other"))) for r in ordered]
    ax.hlines(ys, 0, vals, color=colors, alpha=0.72, linewidth=2.8)
    ax.scatter(vals, ys, color=colors, s=85, edgecolors="black", linewidths=0.45, zorder=3)
    _zero_line(ax, axis="x")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for y, v in zip(ys, vals):
        ax.annotate(f"{v:+.3f}", (v, y), textcoords="offset points", xytext=(5 if v >= 0 else -5, 0), ha="left" if v >= 0 else "right", va="center", fontsize=8)
    _style(ax, title="Minimality ledger: marginal faithfulness of each kept head", xlabel="faithfulness lost when this head is removed", ylabel="final circuit head")
    bench.save_figure(ctx, fig, "minimality_ledger.png", "Marginal value of each kept circuit head under the pruning rule.")


def plot_prompt_failure_scatter(ctx: bench.RunContext, rows: Sequence[dict[str, Any]], *, floor: float) -> None:
    valid = [r for r in rows if _is_finite(r.get("faithfulness")) and _is_finite(r.get("base_diff"))]
    if not valid:
        return
    fig, ax = bench.new_figure(figsize=(8.6, 5.5))
    for family in sorted({str(r.get("family", "")) for r in valid}):
        fam_rows = [r for r in valid if str(r.get("family", "")) == family]
        ax.scatter(
            [_to_float(r.get("base_diff"), 0.0) for r in fam_rows],
            [_to_float(r.get("faithfulness"), 0.0) for r in fam_rows],
            s=[54 + 22 * max(0.0, min(2.0, _to_float(r.get("over_recovery_above_full_model"), 0.0))) for r in fam_rows],
            color=_family_color(family),
            alpha=0.85,
            label=family,
            edgecolors="black",
            linewidths=0.35,
        )
    ax.axhline(floor, color="#D55E00", linestyle="--", linewidth=1.1, label="faithfulness floor")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.4, label="full-model baseline")
    label_rows = sorted(valid, key=lambda r: _to_float(r.get("faithfulness"), 0.0))[:3] + sorted(valid, key=lambda r: _to_float(r.get("faithfulness"), 0.0), reverse=True)[:2]
    seen: set[str] = set()
    for r in label_rows:
        node = str(r.get("example_id", ""))
        if node in seen:
            continue
        seen.add(node)
        ax.annotate(node, (_to_float(r.get("base_diff"), 0.0), _to_float(r.get("faithfulness"), 0.0)), textcoords="offset points", xytext=(5, 5), fontsize=8)
    _style(ax, title="Prompt-level audit: weak behavior, failures, and over-recovery", xlabel="full-model base logit diff", ylabel="circuit faithfulness")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "prompt_failure_scatter.png", "Per-prompt base behavior against circuit-only faithfulness.")


def plot_candidate_evidence_matrix(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    heads = [r for r in rows if r.get("node")]
    if not heads:
        return
    # Keep the final circuit and the strongest disagreements visible without making a postage stamp atlas.
    chosen = sorted(
        heads,
        key=lambda r: (
            not bool(r.get("in_final_circuit")),
            -abs(_to_float(r.get("rank_gap_cheap_minus_causal"), 0.0)),
            -_to_float(r.get("causal_drop"), 0.0),
        ),
    )[:28]
    columns = [
        ("|attr|", "abs_attr", False),
        ("induction", "induction_score", False),
        ("prev-token", "prev_token_score", False),
        ("sink", "first_token_score", False),
        ("causal drop", "causal_drop", True),
        ("marginal", "marginal_value", True),
    ]
    raw_cols: list[list[float]] = []
    for _, key, signed in columns:
        vals = [_to_float(r.get(key), 0.0) for r in chosen]
        if signed:
            scale = max([abs(v) for v in vals] + [1e-9])
            raw_cols.append([v / scale for v in vals])
        else:
            vmin, vmax = min(vals), max(vals)
            denom = max(vmax - vmin, 1e-9)
            raw_cols.append([((v - vmin) / denom) for v in vals])
    data = [[raw_cols[j][i] for j in range(len(columns))] for i in range(len(chosen))]

    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(data, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, max(4.2, 0.36 * len(chosen) + 1.7)), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[0] for c in columns], rotation=30, ha="right")
    labels = [f"{'★ ' if r.get('in_final_circuit') else ''}{r['node']} · {r.get('motif_label', 'other')}" for r in chosen]
    ax.set_yticks(range(len(chosen)))
    ax.set_yticklabels(labels, fontsize=7.5)
    for i, r in enumerate(chosen):
        for j, (_label, key, _signed) in enumerate(columns):
            v = _to_float(r.get(key), float("nan"))
            txt = "" if not math.isfinite(v) else (f"{v:+.2f}" if key in {"causal_drop", "marginal_value"} else f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8, color="black")
    for i, r in enumerate(chosen):
        if r.get("edge_role"):
            ax.text(len(columns) - 0.1, i, str(r["edge_role"]).replace("edge_", ""), ha="left", va="center", fontsize=7, color="black")
    ax.set_title("Candidate evidence matrix: screen signals, causal tests, and final membership")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("column-normalized evidence score")
    bench.save_figure(ctx, fig, "candidate_evidence_matrix.png", "Candidate heads with cheap scores, motif labels, causal drops, final membership, and marginality.")


def plot_causal_motif_atlas(ctx: bench.RunContext, rows: Sequence[dict[str, Any]], n_layers: int, n_heads: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    if not rows:
        return
    grid = np.full((n_heads, n_layers), np.nan, dtype=float)
    in_circuit: list[tuple[int, int, str]] = []
    for r in rows:
        layer = _to_int(r.get("layer"), -1)
        head = _to_int(r.get("head"), -1)
        if 0 <= layer < n_layers and 0 <= head < n_heads:
            grid[head, layer] = _to_float(r.get("causal_drop"), 0.0)
            if r.get("in_final_circuit"):
                in_circuit.append((layer, head, str(r.get("motif_label", "other"))))
    if np.all(np.isnan(grid)):
        return
    vmax = float(np.nanmax(np.abs(grid))) if np.any(~np.isnan(grid)) else 1.0
    vmax = max(vmax, 1e-6)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    for layer, head, motif in in_circuit:
        ax.scatter(layer, head, marker=_motif_marker(motif), s=95, facecolors="none", edgecolors="black", linewidths=1.6)
        ax.annotate(f"L{layer}H{head}", (layer, head), textcoords="offset points", xytext=(3, 3), fontsize=6.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("head")
    ax.set_title("Causal motif atlas: screened heads on the layer-head grid")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("single-head causal drop")
    bench.save_figure(ctx, fig, "causal_motif_atlas.png", "Layer-head atlas of screened-head causal drops with final circuit heads outlined.")


def plot_edge_interaction_map(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    valid = [r for r in rows if _is_finite(r.get("interaction"))]
    if not valid:
        return
    fig, ax = bench.new_figure(figsize=(8.3, 5.5))
    for r in valid:
        frac = _to_float(r.get("raw_interaction_fraction"), 0.0)
        inter = _to_float(r.get("interaction"), 0.0)
        strength = str(r.get("edge_strength", "none"))
        color = _motif_color("induction") if strength in {"weak", "strong"} else _lighten(_motif_color("previous_token"), 0.35)
        ax.scatter(
            _to_float(r.get("from_layer"), 0.0),
            _to_float(r.get("to_layer"), 0.0),
            s=80 + 420 * max(0.0, min(0.12, frac)) / 0.12,
            color=color,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
        )
        if inter >= sorted([_to_float(x.get("interaction"), 0.0) for x in valid], reverse=True)[min(4, len(valid) - 1)]:
            ax.annotate(str(r.get("edge", "")), (_to_float(r.get("from_layer"), 0.0), _to_float(r.get("to_layer"), 0.0)), textcoords="offset points", xytext=(4, 4), fontsize=7)
    lo = min([_to_float(r.get("from_layer"), 0.0) for r in valid] + [_to_float(r.get("to_layer"), 0.0) for r in valid]) - 1
    hi = max([_to_float(r.get("from_layer"), 0.0) for r in valid] + [_to_float(r.get("to_layer"), 0.0) for r in valid]) + 1
    ax.plot([lo, hi], [lo, hi], color="black", alpha=0.25, linewidth=0.9, label="same layer")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _style(ax, title="Edge interaction map: ordered previous-token to induction pairs", xlabel="source layer", ylabel="target layer")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "edge_interaction_map.png", "Ordered edge checks by source and target layer, sized by routed-fraction proxy.")


def plot_circuit_discovery_dashboard(
    ctx: bench.RunContext,
    fcm: dict[str, Any],
    trajectory: Sequence[dict[str, Any]],
    cand_rows: Sequence[dict[str, Any]],
    prompt_rows: Sequence[dict[str, Any]],
    *,
    floor: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()

    # A. F/C/M scorecard.
    groups: list[tuple[str, str, float]] = []
    for family in ("discovery", "heldout"):
        if family in fcm:
            groups.append((family, "faithfulness", _to_float(fcm[family].get("faithfulness"), 0.0)))
            groups.append((family, "completeness effect", _to_float(fcm[family].get("completeness_effect"), 0.0)))
    xs = list(range(len(groups)))
    ax0.bar(xs, [v for _, _, v in groups], color=[_family_color(f) if m == "faithfulness" else _lighten(_family_color(f), 0.35) for f, m, _ in groups])
    ax0.axhline(floor, color="#D55E00", linestyle="--", linewidth=1.0)
    ax0.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
    ax0.set_xticks(xs)
    ax0.set_xticklabels([f"{m}\n{f}" for f, m, _ in groups], fontsize=8)
    _style(ax0, title="F/C scorecard", ylabel="fraction of base behavior")
    _panel_label(ax0, "A")

    # B. Pruning trajectory.
    if trajectory:
        x = [t["n_nodes"] for t in trajectory]
        y = [_to_float(t["faithfulness"], 0.0) for t in trajectory]
        ax1.plot(x, y, marker="o", color=_family_color("discovery"), linewidth=2.0)
        ax1.axhline(floor, color="#D55E00", linestyle="--", linewidth=1.0)
        ax1.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
        ax1.invert_xaxis()
    _style(ax1, title="Greedy pruning path", xlabel="heads kept", ylabel="faithfulness")
    _panel_label(ax1, "B")

    # C. Candidate quality.
    heads = [r for r in cand_rows if r.get("kind") == "head"]
    for r in heads:
        motif = str(r.get("motif_label", "other"))
        ax2.scatter(
            _to_float(r.get("cheap_rank"), 0.0),
            _to_float(r.get("causal_drop"), 0.0),
            color=_motif_color(motif),
            marker=_motif_marker(motif),
            s=54,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.35,
        )
    _zero_line(ax2)
    _style(ax2, title="Cheap screen vs causal test", xlabel="cheap rank, lower is better", ylabel="causal drop")
    _panel_label(ax2, "C")

    # D. Prompt audit.
    valid = [r for r in prompt_rows if _is_finite(r.get("faithfulness"))]
    ordered = sorted(valid, key=lambda r: _to_float(r.get("faithfulness"), 0.0))
    xs2 = list(range(len(ordered)))
    ax3.bar(xs2, [_to_float(r.get("faithfulness"), 0.0) for r in ordered], color=[_family_color(str(r.get("family", ""))) for r in ordered], alpha=0.85)
    ax3.axhline(floor, color="#D55E00", linestyle="--", linewidth=1.0)
    ax3.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
    ax3.set_xticks(xs2)
    ax3.set_xticklabels([str(r.get("example_id", "")) for r in ordered], rotation=35, ha="right", fontsize=7)
    _style(ax3, title="Prompt-level failures and over-recovery", ylabel="faithfulness")
    _panel_label(ax3, "D")

    fig.suptitle("Manual circuit discovery: from suspects to a scoped heads-only claim", fontsize=14)
    bench.save_figure(ctx, fig, "circuit_discovery_dashboard.png", "One-screen summary of Lab 6 F/C/M, pruning, screening, and prompt-level audit.")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


# Initial circuit seed is capped so greedy-prune-to-1 stays O(n^2)-tractable on
# 64-layer x 40-head models. Disclosed (printed + recorded), never silent.
MAX_SEED_NODES = 64


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    """Bench entrypoint for ONE (behavior, scope) cell. A hygiene/period refusal
    is caught and written as a first-class verdict card rather than crashing."""
    try:
        _run_cell(ctx, bundle)
    except CellAbort as abort:
        _write_abort_card(ctx, bundle, abort)


def _write_abort_card(ctx: bench.RunContext, bundle: bench.ModelBundle, abort: CellAbort) -> None:
    behavior = getattr(ctx.args, "behavior", "induction_p3")
    scope = getattr(ctx.args, "scope", "heads_and_mlps")
    print(f"[lab6] CELL ABORTED ({abort.verdict}): {abort.message}")
    metrics = {
        "behavior": behavior,
        "scope": scope,
        "verdict": abort.verdict,
        "verdict_reason": abort.message,
        "aborted": True,
    }
    bench.write_json(ctx.path("metrics.json"), metrics)
    ctx.register_artifact(ctx.path("metrics.json"), "metrics", "Aborted-cell metrics with verdict.")
    card = [
        f"# Circuit card: {behavior} ({scope})",
        "",
        f"- **Model:** `{bundle.anatomy.model_id}` | run `{ctx.run_dir.name}`",
        f"- **Verdict:** **{abort.verdict}**",
        "",
        abort.message,
        "",
        "A refusal here is a first-class result. The lab will not manufacture a tiny-n or "
        "mixed-mechanism circuit. See `prompt_hygiene_report.md` for per-prompt detail.",
    ]
    bench.write_text(ctx.path("circuit_card.md"), "\n".join(card))
    ctx.register_artifact(ctx.path("circuit_card.md"), "summary", "Aborted-cell circuit card.")


def _run_cell(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    behavior = getattr(args, "behavior", "induction_p3")
    scope = getattr(args, "scope", "heads_and_mlps")
    resample_draws = int(getattr(args, "resample_draws", DEFAULT_RESAMPLE_DRAWS))
    allow_mixed = bool(getattr(args, "allow_mixed_period", False))
    n_layers = bundle.anatomy.n_layers

    prompts: list[CircuitPrompt] | None = None
    if behavior == "swa":
        length_words = int(str(getattr(args, "swa_lengths", "1024")).split(",")[0].strip())
        prompts = swa_prompts(length_words)

    discovery, heldout, hygiene, seq_len = build_dataset(
        ctx,
        bundle,
        args.max_examples,
        behavior=behavior,
        prompts=prompts,
        allow_mixed_period=allow_mixed,
        min_positive=MIN_POSITIVE,
    )
    base_metric = statistics.fmean(ex.base_diff for ex in discovery)
    heldout_base = statistics.fmean(ex.base_diff for ex in heldout) if heldout else None
    print(
        f"[lab6] behavior={behavior} scope={scope}: discovery {len(discovery)} prompts, "
        f"held-out {len(heldout)}; seq {seq_len}; base metric {base_metric:+.3f}; "
        f"resample draws {resample_draws}"
    )

    # ----- instrument verification (load-bearing self-checks) ----------------
    probe = discovery[0].prompt.prompt
    bench.run_hook_parity_check(ctx, bundle, probe)
    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, probe, rel_tolerance=args.dla_tolerance)
    head_anatomy = bench.resolve_head_anatomy(ctx, bundle)
    first_att = bench.run_with_attention_cache(bundle, probe)
    bench.run_lens_self_check(ctx, bundle, first_att.capture)
    first_comp = bench.run_with_component_cache(bundle, probe, comp_anatomy, all_positions=False)
    bench.run_decomposition_check(ctx, bundle, first_comp, rel_tolerance=args.dla_tolerance)
    bench.run_head_decomposition_check(ctx, bundle, head_anatomy, first_att, rel_tolerance=args.dla_tolerance)
    n_heads = head_anatomy.n_heads

    # ----- captures and dataset means (discovery AND held-out for resample) ---
    att_caps: dict[str, Any] = {}
    comp_caps: dict[str, Any] = {}
    for ex in list(discovery) + list(heldout):
        att_caps[ex.prompt.example_id] = bench.run_with_attention_cache(bundle, ex.prompt.prompt, all_positions=True)
        comp_caps[ex.prompt.example_id] = bench.run_with_component_cache(
            bundle, ex.prompt.prompt, comp_anatomy, all_positions=True
        )

    head_means = torch.stack([att_caps[ex.prompt.example_id].o_in_last for ex in discovery]).mean(dim=0)
    mlp_means = torch.stack([comp_caps[ex.prompt.example_id].mlp_contrib for ex in discovery]).mean(dim=0)
    caps_disc = {
        ex.prompt.example_id: (att_caps[ex.prompt.example_id].o_in_last, comp_caps[ex.prompt.example_id].mlp_contrib)
        for ex in discovery
    }
    caps_held = {
        ex.prompt.example_id: (att_caps[ex.prompt.example_id].o_in_last, comp_caps[ex.prompt.example_id].mlp_contrib)
        for ex in heldout
    }

    manifest_path = ctx.path("diagnostics", "ablation_manifest.json")
    bench.write_json(
        manifest_path,
        {
            "primary_off_distribution": "resample / interchange ablation (within-distribution, FIX 2)",
            "comparison_off_distribution": "dataset mean over discovery prompts",
            "resample_draws": resample_draws,
            "prompt_length_tokens": seq_len,
            "discovery_examples": [ex.prompt.example_id for ex in discovery],
            "heldout_examples": [ex.prompt.example_id for ex in heldout],
            "head_means_shape": list(head_means.shape),
            "mlp_means_shape": list(mlp_means.shape),
            "scope": scope,
        },
    )
    ctx.register_artifact(manifest_path, "diagnostic", "Definition of the resample and mean ablation off distributions.")

    # ----- screening (cheap: attribution + motifs) ---------------------------
    head_attr: dict[tuple[int, int], list[float]] = {}
    head_induct: dict[tuple[int, int], list[float]] = {}
    head_prev: dict[tuple[int, int], list[float]] = {}
    head_first: dict[tuple[int, int], list[float]] = {}
    mlp_attr: dict[int, list[float]] = {}

    for ex in discovery:
        att = att_caps[ex.prompt.example_id]
        att_final = bench.AttentionCapture(
            capture=att.capture,
            attentions=att.attentions,
            o_in_last=att.o_in_last[:, -1],
            attn_out_last=att.attn_out_last[:, -1],
        )
        attr = head_attribution_scores(bundle, comp_anatomy, head_anatomy, att_final, ex.target_id, ex.distractor_id)

        comp = comp_caps[ex.prompt.example_id]
        comp_final = bench.ComponentCapture(
            capture=comp.capture,
            attn_contrib=comp.attn_contrib[:, -1],
            mlp_contrib=comp.mlp_contrib[:, -1],
        )
        dla = compute_direct_logit_attribution(bundle, comp_final, ex.target_id, ex.distractor_id)

        for layer in range(n_layers):
            mlp_attr.setdefault(layer, []).append(float(dla["mlp_scores"][layer]))
            for head in range(n_heads):
                key = (layer, head)
                pattern = att.attentions[layer, head]
                head_attr.setdefault(key, []).append(float(attr["scores"][layer][head]))
                head_prev.setdefault(key, []).append(prev_token_score(pattern))
                head_first.setdefault(key, []).append(first_token_score(pattern))
                ind = induction_score(pattern, att.capture.input_ids)
                head_induct.setdefault(key, []).append(0.0 if ind is None else float(ind))

    mean_attr = {k: statistics.fmean(v) for k, v in head_attr.items()}
    mean_induct = {k: statistics.fmean(v) for k, v in head_induct.items()}
    mean_prev = {k: statistics.fmean(v) for k, v in head_prev.items()}
    mean_first = {k: statistics.fmean(v) for k, v in head_first.items()}
    mean_mlp = {k: statistics.fmean(v) for k, v in mlp_attr.items()}

    attr_rank = rank_map(mean_attr, key_abs=True)
    induct_rank = rank_map(mean_induct)
    prev_rank = rank_map(mean_prev)
    top_attr, top_ind, top_prev = screen_budgets(n_layers, n_heads)

    screen_reasons: dict[tuple[int, int], set[str]] = {}
    for key in sorted(mean_attr, key=lambda k: -abs(mean_attr[k]))[:top_attr]:
        screen_reasons.setdefault(key, set()).add("attribution")
    for key in sorted(mean_induct, key=lambda k: -mean_induct[k])[:top_ind]:
        screen_reasons.setdefault(key, set()).add("induction")
    for key in sorted(mean_prev, key=lambda k: -mean_prev[k])[:top_prev]:
        screen_reasons.setdefault(key, set()).add("prev_token")

    def motif_label(key: tuple[int, int]) -> str:
        if mean_induct.get(key, 0.0) >= MOTIF_STRONG_THRESHOLD:
            return "induction"
        if mean_prev.get(key, 0.0) >= MOTIF_STRONG_THRESHOLD:
            return "previous_token"
        if mean_first.get(key, 0.0) >= FIRST_TOKEN_SINK_THRESHOLD:
            return "first_token_sink"
        return "other"

    head_labels = {key: motif_label(key) for key in screen_reasons}
    # MLP screening: in heads_and_mlps scope, MLP layers are first-class circuit
    # nodes, so EVERY layer is causally screened (cheap: n_layers passes). This is
    # what lets load-bearing-but-low-attribution layers -- gpt2 MLP0 acts like an
    # extended embedding and is catastrophic to ablate yet writes nothing to the
    # logit-diff directly -- be seeded and kept instead of silently ablated. In
    # heads_only scope MLPs are never circuit nodes, so we only rank the top few
    # by attribution for the supporting-MLP report.
    if scope == "heads_and_mlps":
        mlp_candidates = list(range(n_layers))
    else:
        mlp_candidates = sorted(mean_mlp, key=lambda layer: -abs(mean_mlp[layer]))[:N_MLP_CANDIDATES]
    print(
        f"[lab6] screened {len(screen_reasons)} candidate heads "
        f"(attr {top_attr}, induction {top_ind}, previous-token {top_prev}) "
        f"+ {len(mlp_candidates)} candidate MLPs (scope={scope})"
    )

    # ----- single-node causal ranking (drop = base - metric_with_node_ablated) -
    cand_rows: list[dict[str, Any]] = []
    head_causal_drop: dict[tuple[int, int], float] = {}
    head_single_metric: dict[tuple[int, int], float] = {}
    mlp_causal_drop: dict[int, float] = {}
    mlp_single_metric: dict[int, float] = {}

    for key in sorted(screen_reasons, key=lambda k: min(attr_rank[k], induct_rank[k], prev_rank[k])):
        layer, head = key
        ablated = metric_under_ablation(
            bundle, discovery, head_anatomy, comp_anatomy, heads=[key], mlps=[], head_means=head_means, mlp_means=mlp_means
        )
        drop = base_metric - ablated
        head_causal_drop[key] = drop
        head_single_metric[key] = ablated
        reason = "+".join(sorted(screen_reasons[key]))
        cheap_rank = min(attr_rank[key], induct_rank[key], prev_rank[key])
        cand_rows.append(
            {
                "node": node_name("head", layer, head),
                "kind": "head",
                "layer": layer,
                "head": head,
                "screen_reason": reason,
                "cheap_rank": cheap_rank,
                "mean_attr": round(mean_attr[key], 5),
                "abs_attr_rank": attr_rank[key],
                "induction_score": round(mean_induct[key], 5),
                "induction_rank": induct_rank[key],
                "prev_token_score": round(mean_prev[key], 5),
                "prev_token_rank": prev_rank[key],
                "first_token_score": round(mean_first[key], 5),
                "motif_label": head_labels[key],
                "single_ablated_metric": round(ablated, 5),
                "causal_drop": round(drop, 5),
            }
        )

    for layer in mlp_candidates:
        ablated = metric_under_ablation(
            bundle, discovery, head_anatomy, comp_anatomy, heads=[], mlps=[layer], head_means=head_means, mlp_means=mlp_means
        )
        drop = base_metric - ablated
        mlp_causal_drop[layer] = drop
        mlp_single_metric[layer] = ablated
        cand_rows.append(
            {
                "node": node_name("mlp", layer),
                "kind": "mlp",
                "layer": layer,
                "head": "",
                "screen_reason": "attribution",
                "cheap_rank": "",
                "mean_attr": round(mean_mlp[layer], 5),
                "abs_attr_rank": "",
                "induction_score": "",
                "induction_rank": "",
                "prev_token_score": "",
                "prev_token_rank": "",
                "first_token_score": "",
                "motif_label": "support_mlp",
                "single_ablated_metric": round(ablated, 5),
                "causal_drop": round(drop, 5),
            }
        )

    head_rows = [r for r in cand_rows if r["kind"] == "head"]
    causal_rank_by_node = {
        r["node"]: i + 1 for i, r in enumerate(sorted(head_rows, key=lambda row: -row["causal_drop"]))
    }
    for row in cand_rows:
        row["causal_rank"] = causal_rank_by_node.get(row["node"], "")
        if row["kind"] == "head" and isinstance(row["cheap_rank"], int) and row["causal_rank"]:
            row["rank_gap_cheap_minus_causal"] = int(row["cheap_rank"]) - int(row["causal_rank"])
        else:
            row["rank_gap_cheap_minus_causal"] = ""

    cand_rows_sorted = sorted(cand_rows, key=lambda r: (r["kind"] != "head", -float(r["causal_drop"])))
    cand_path = ctx.path("tables", "candidate_components.csv")
    bench.write_csv_with_context(ctx, cand_path, cand_rows_sorted)
    ctx.register_artifact(cand_path, "table", "Screened candidates with cheap scores, motif labels, and causal drops.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, cand_rows_sorted)
    ctx.register_artifact(results_path, "results", "Alias of candidate_components.csv for the standard run contract.")

    # ----- node universe and faithfulness over scope -------------------------
    all_head_nodes = [head_node(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    all_mlp_nodes = [mlp_node(layer) for layer in range(n_layers)]
    universe = list(all_head_nodes) + (list(all_mlp_nodes) if scope == "heads_and_mlps" else [])

    def node_drop(node: Node) -> float:
        kind, layer, head = node
        return head_causal_drop.get((layer, head), 0.0) if kind == "head" else mlp_causal_drop.get(layer, 0.0)

    def metric_of(nodes_to_ablate: Sequence[Node], examples: Sequence[TaskExample], ablation: str, caps: dict) -> float:
        heads_ab, mlps_ab = split_nodes(nodes_to_ablate)
        if ablation == "mean":
            return metric_under_ablation(
                bundle, examples, head_anatomy, comp_anatomy, heads_ab, mlps_ab, head_means, mlp_means
            )
        return metric_under_resample(
            bundle, examples, head_anatomy, comp_anatomy, heads_ab, mlps_ab, caps, resample_draws
        )

    def faithfulness_of(
        circuit_nodes: Sequence[Node], examples: Sequence[TaskExample], base: float, ablation: str, caps: dict
    ) -> float:
        circuit_set = set(circuit_nodes)
        complement = [n for n in universe if n not in circuit_set]
        metric = metric_of(complement, examples, ablation, caps)
        ratio = safe_ratio(metric, base)
        if ratio is None:
            raise RuntimeError("Cannot compute faithfulness ratio because the base metric is zero.")
        return ratio

    # ----- seed circuit: positive single-node causal drop, capped ------------
    seed_nodes = [head_node(*k) for k, d in head_causal_drop.items() if d > 0]
    if scope == "heads_and_mlps":
        seed_nodes += [mlp_node(layer) for layer, d in mlp_causal_drop.items() if d > 0]
    if not seed_nodes:
        raise CellAbort(
            "MECHANISM ABSENT",
            "no screened node has a positive single-node causal drop, so there is nothing to assemble into a "
            "circuit for this behavior under this off distribution. See tables/candidate_components.csv.",
        )
    seed_nodes.sort(key=lambda n: -node_drop(n))
    seed_capped = False
    if len(seed_nodes) > MAX_SEED_NODES:
        seed_capped = True
        print(f"[lab6] seed circuit capped from {len(seed_nodes)} to {MAX_SEED_NODES} highest-causal nodes (disclosed).")
        seed_nodes = seed_nodes[:MAX_SEED_NODES]

    # ----- greedy prune ALL the way to 1 node, full trajectory + snapshots ----
    circuit = list(seed_nodes)
    current_faith = faithfulness_of(circuit, discovery, base_metric, "mean", caps_disc)
    trajectory: list[dict[str, Any]] = [
        {
            "step": 0,
            "n_nodes": len(circuit),
            "faithfulness": round(current_faith, 5),
            "removed": "",
            "rule": "start with every positive-causal screened node (capped)",
            "_snapshot": list(circuit),
        }
    ]
    print(f"[lab6] seed circuit: {len(circuit)} nodes, faithfulness {current_faith:.3f}")
    while len(circuit) > 1:
        options: list[tuple[float, Node]] = []
        for node in circuit:
            reduced = [x for x in circuit if x != node]
            options.append((faithfulness_of(reduced, discovery, base_metric, "mean", caps_disc), node))
        best_faith, best_node = max(options, key=lambda item: item[0])
        circuit = [x for x in circuit if x != best_node]
        current_faith = best_faith
        trajectory.append(
            {
                "step": len(trajectory),
                "n_nodes": len(circuit),
                "faithfulness": round(best_faith, 5),
                "removed": node_label(best_node),
                "rule": "removed least costly node (greedy, to 1)",
                "_snapshot": list(circuit),
            }
        )

    # ----- knee + floor selection (FIX 1) ------------------------------------
    peak_faith = max(r["faithfulness"] for r in trajectory)
    within_peak = [r for r in trajectory if r["faithfulness"] >= peak_faith - KNEE_EPSILON]
    knee_row = min(within_peak, key=lambda r: (r["n_nodes"], -r["faithfulness"]))
    knee_circuit = list(knee_row["_snapshot"])
    floor_rows = [r for r in trajectory if r["faithfulness"] >= FAITHFULNESS_FLOOR]
    meets_floor = bool(floor_rows)
    if floor_rows:
        floor_row = min(floor_rows, key=lambda r: (r["n_nodes"], -r["faithfulness"]))
    else:
        floor_row = max(trajectory, key=lambda r: r["faithfulness"])
    floor_circuit = list(floor_row["_snapshot"])
    knee_vs_floor_gap = round(knee_row["faithfulness"] - floor_row["faithfulness"], 5)
    print(
        f"[lab6] peak faithfulness {peak_faith:.3f}; knee = {knee_row['n_nodes']} nodes "
        f"(faith {knee_row['faithfulness']:.3f}); floor = {floor_row['n_nodes']} nodes "
        f"(faith {floor_row['faithfulness']:.3f}); meets_floor={meets_floor}"
    )

    # headline circuit is the KNEE (FIX 1)
    circuit = list(knee_circuit)
    circuit_heads = [(layer, head) for (kind, layer, head) in circuit if kind == "head"]
    circuit_mlps = [layer for (kind, layer, _h) in circuit if kind == "mlp"]

    traj_csv = [{k: v for k, v in r.items() if k != "_snapshot"} for r in trajectory]
    traj_path = ctx.path("tables", "prune_trajectory.csv")
    bench.write_csv_with_context(ctx, traj_path, traj_csv)
    ctx.register_artifact(traj_path, "table", "Faithfulness at each greedy pruning step (pruned to 1 node).")

    knee_floor = {
        "peak_faithfulness": round(peak_faith, 5),
        "knee": {"n_nodes": knee_row["n_nodes"], "faithfulness": knee_row["faithfulness"], "nodes": [node_label(n) for n in knee_circuit]},
        "floor": {"n_nodes": floor_row["n_nodes"], "faithfulness": floor_row["faithfulness"], "nodes": [node_label(n) for n in floor_circuit]},
        "knee_epsilon": KNEE_EPSILON,
        "knee_minus_floor_faithfulness": knee_vs_floor_gap,
        "seed_capped": seed_capped,
        "meets_faithfulness_floor": meets_floor,
    }
    bench.write_json(ctx.path("tables", "knee_floor_selection.json"), knee_floor)
    ctx.register_artifact(ctx.path("tables", "knee_floor_selection.json"), "metrics", "Knee vs floor circuit selection (FIX 1).")

    # ----- suppression heads (FIX 4) -----------------------------------------
    supp_threshold = SUPPRESSION_REL_THRESHOLD * abs(base_metric)
    suppression_heads = sorted(
        ((k, d) for k, d in head_causal_drop.items() if d < -supp_threshold), key=lambda kv: kv[1]
    )
    suppression_set = {head_node(*k) for k, _d in suppression_heads}

    def faithfulness_brake_intact(circuit_nodes: Sequence[Node], examples: Sequence[TaskExample], base: float, ablation: str, caps: dict) -> float | None:
        circuit_set = set(circuit_nodes)
        complement = [n for n in universe if n not in circuit_set and n not in suppression_set]
        metric = metric_of(complement, examples, ablation, caps)
        return safe_ratio(metric, base)

    # ----- motif-core-only circuit -------------------------------------------
    core_motifs = CORE_MOTIFS_BY_BEHAVIOR.get(behavior, ())
    labeled_core = [head_node(*k) for k in screen_reasons if head_labels.get(k) in core_motifs and head_causal_drop.get(k, 0.0) > 0]
    induction_motif_present = bool(labeled_core)
    if labeled_core:
        motif_core = labeled_core
        motif_core_def = "labeled motif heads: " + ", ".join(core_motifs)
    else:
        top_heads = sorted(((k, d) for k, d in head_causal_drop.items() if d > 0), key=lambda kv: -kv[1])[:2]
        motif_core = [head_node(*k) for k, _d in top_heads]
        motif_core_def = "fallback: top-2 single-head-causal heads (no labeled motif core survived)"

    # ----- F/C/M for knee / floor / motif-core under mean AND resample -------
    def evaluate_circuit(nodes: Sequence[Node], examples: Sequence[TaskExample], base: float | None, caps: dict) -> dict[str, Any] | None:
        if not examples or base is None or base <= 0:
            return None
        heads_ab, mlps_ab = split_nodes(nodes)
        comp_mean = metric_under_ablation(bundle, examples, head_anatomy, comp_anatomy, heads_ab, mlps_ab, head_means, mlp_means)
        comp_res = metric_under_resample(bundle, examples, head_anatomy, comp_anatomy, heads_ab, mlps_ab, caps, resample_draws)
        return {
            "n_nodes": len(nodes),
            "n_prompts": len(examples),
            "faith_mean": round(faithfulness_of(nodes, examples, base, "mean", caps), 5),
            "faith_resample": round(faithfulness_of(nodes, examples, base, "resample", caps), 5),
            "completeness_ratio_mean": round_or_none(safe_ratio(comp_mean, base), 5),
            "completeness_ratio_resample": round_or_none(safe_ratio(comp_res, base), 5),
        }

    circuits: dict[str, Any] = {}
    for cname, nodes in (("knee", knee_circuit), ("floor", floor_circuit), ("motif_core", motif_core)):
        entry: dict[str, Any] = {"nodes": [node_label(n) for n in nodes]}
        entry["discovery"] = evaluate_circuit(nodes, discovery, base_metric, caps_disc)
        entry["heldout"] = evaluate_circuit(nodes, heldout, heldout_base, caps_held)
        circuits[cname] = entry

    # brake-intact comparison on the knee circuit
    brake_intact = {
        "discovery_faith_mean": round_or_none(faithfulness_brake_intact(knee_circuit, discovery, base_metric, "mean", caps_disc), 5),
        "discovery_faith_resample": round_or_none(faithfulness_brake_intact(knee_circuit, discovery, base_metric, "resample", caps_disc), 5),
        "n_suppression_heads_held_out": len(suppression_heads),
    }
    if heldout:
        brake_intact["heldout_faith_mean"] = round_or_none(faithfulness_brake_intact(knee_circuit, heldout, heldout_base, "mean", caps_held), 5)
        brake_intact["heldout_faith_resample"] = round_or_none(faithfulness_brake_intact(knee_circuit, heldout, heldout_base, "resample", caps_held), 5)

    # ----- legacy F/C/M dict for the existing plots (knee, mean ablation) ----
    fcm: dict[str, Any] = {
        "faithfulness_floor": FAITHFULNESS_FLOOR,
        "meets_faithfulness_floor": meets_floor,
        "prune_stop_reason": "pruned to 1 node; knee selected within KNEE_EPSILON of peak",
    }

    def legacy_family(name: str, examples: Sequence[TaskExample], base: float | None) -> None:
        if not examples or base is None or base <= 0:
            return
        faith = faithfulness_of(knee_circuit, examples, base, "mean", caps_disc if name == "discovery" else caps_held)
        heads_ab, mlps_ab = split_nodes(knee_circuit)
        circuit_ablated = metric_under_ablation(bundle, examples, head_anatomy, comp_anatomy, heads_ab, mlps_ab, head_means, mlp_means)
        completeness_ratio = safe_ratio(circuit_ablated, base)
        if completeness_ratio is None:
            return
        fcm[name] = {
            "n_prompts": len(examples),
            "base_metric": round(base, 5),
            "faithfulness": round(faith, 5),
            "circuit_ablated_metric": round(circuit_ablated, 5),
            "completeness_ratio": round(completeness_ratio, 5),
            "completeness_effect": round(1.0 - completeness_ratio, 5),
        }

    legacy_family("discovery", discovery, base_metric)
    legacy_family("heldout", heldout, heldout_base)

    # ----- minimality ledger with rent-unpaid flags (FIX 7) ------------------
    minimality_rows: list[dict[str, Any]] = []
    knee_disc_faith_mean = fcm["discovery"]["faithfulness"]
    for node in knee_circuit:
        reduced = [x for x in knee_circuit if x != node]
        f_without = faithfulness_of(reduced, discovery, base_metric, "mean", caps_disc)
        marginal = knee_disc_faith_mean - f_without
        kind, layer, head = node
        minimality_rows.append(
            {
                "node": node_label(node),
                "layer": layer,
                "head": head if kind == "head" else "",
                "kind": kind,
                "motif_label": head_labels.get((layer, head), "support_mlp") if kind == "head" else "support_mlp",
                "single_head_causal_drop": round(node_drop(node), 5),
                "faithfulness_without": round(f_without, 5),
                "marginal_value": round(marginal, 5),
                "minimality_passes_positive_marginal": marginal > 0,
                "rent_unpaid": marginal < MARGINAL_RENT_THRESHOLD,
            }
        )
    minimality_rows.sort(key=lambda row: row["marginal_value"])
    rent_unpaid_labels = {r["node"] for r in minimality_rows if r["rent_unpaid"]}
    rent_paid_circuit = [n for n in knee_circuit if node_label(n) not in rent_unpaid_labels]

    fcm["minimality_worst_marginal"] = min((r["marginal_value"] for r in minimality_rows), default=None)
    fcm["minimality_all_positive"] = all(r["minimality_passes_positive_marginal"] for r in minimality_rows)

    # held-out transfer WITH and WITHOUT the rent-unpaid filler nodes (FIX 7)
    rent_compare = {"n_rent_unpaid": len(rent_unpaid_labels)}
    if heldout:
        rent_compare["knee_heldout_faith_resample"] = circuits["knee"]["heldout"]["faith_resample"] if circuits["knee"]["heldout"] else None
        if rent_paid_circuit and len(rent_paid_circuit) != len(knee_circuit):
            rc = evaluate_circuit(rent_paid_circuit, heldout, heldout_base, caps_held)
            rent_compare["rent_paid_heldout_faith_resample"] = rc["faith_resample"] if rc else None
        else:
            rent_compare["rent_paid_heldout_faith_resample"] = rent_compare["knee_heldout_faith_resample"]

    fcm_path = ctx.path("faithfulness_completeness_minimality.json")
    bench.write_json(
        fcm_path,
        {
            "behavior": behavior,
            "scope": scope,
            "headline_circuit": "knee",
            "knee_circuit": [node_label(n) for n in knee_circuit],
            "floor_circuit": [node_label(n) for n in floor_circuit],
            "motif_core": {"definition": motif_core_def, "nodes": [node_label(n) for n in motif_core]},
            **fcm,
            "circuits": circuits,
            "brake_intact": brake_intact,
            "rent_compare": rent_compare,
            "minimality": minimality_rows,
            "interpretation": {
                "faith_resample": "PRIMARY: complement resample-ablated; >=floor and transferring is a real circuit",
                "faith_mean": "comparison: complement dataset-mean ablated; >1.0 usually means suppression-head removal",
                "completeness_ratio": "circuit ablated; lower means the circuit was more necessary",
                "minimality": "loss in faithfulness when each kept node is removed; rent_unpaid flags filler",
            },
        },
    )
    ctx.register_artifact(fcm_path, "metrics", "Faithfulness, completeness, minimality (mean+resample, knee/floor/motif-core).")
    min_path = ctx.path("tables", "pruned_circuit.csv")
    bench.write_csv_with_context(ctx, min_path, minimality_rows)
    ctx.register_artifact(min_path, "table", "Every kept node with its marginal faithfulness value and rent-unpaid flag.")

    # ----- the one edge claim: ordered prev-token -> induction interaction ---
    induction_heads = [(layer, head) for (kind, layer, head) in knee_circuit if kind == "head" and head_labels.get((layer, head)) == "induction"]
    prev_heads = [
        key
        for key in screen_reasons
        if head_labels.get(key) == "previous_token" and head_causal_drop.get(key, 0.0) > EDGE_MIN_SOURCE_EFFECT
    ]
    edge_rows: list[dict[str, Any]] = []
    for h_prev in prev_heads:
        for h_ind in induction_heads:
            if h_prev == h_ind or h_prev[0] >= h_ind[0]:
                continue
            m_i = head_single_metric.get(h_ind)
            if m_i is None:
                m_i = metric_under_ablation(bundle, discovery, head_anatomy, comp_anatomy, heads=[h_ind], mlps=[], head_means=head_means, mlp_means=mlp_means)
            m_p = head_single_metric[h_prev]
            m_ip = metric_under_ablation(bundle, discovery, head_anatomy, comp_anatomy, heads=[h_ind, h_prev], mlps=[], head_means=head_means, mlp_means=mlp_means)
            effect_p = base_metric - m_p
            effect_p_given_i = m_i - m_ip
            interaction = effect_p - effect_p_given_i
            raw_frac = safe_ratio(interaction, effect_p)
            strength = edge_strength_label(raw_frac)
            edge_rows.append(
                {
                    "edge": f"{node_name('head', *h_prev)} -> {node_name('head', *h_ind)}",
                    "from_layer": h_prev[0],
                    "from_head": h_prev[1],
                    "to_layer": h_ind[0],
                    "to_head": h_ind[1],
                    "effect_prev_alone": round(effect_p, 5),
                    "effect_prev_given_induction_ablated": round(effect_p_given_i, 5),
                    "interaction": round(interaction, 5),
                    "raw_interaction_fraction": round_or_none(raw_frac, 5),
                    "edge_strength": strength,
                    "claimable_fraction": raw_frac is not None and EDGE_MIN_ROUTED_FRACTION <= raw_frac <= 1.0,
                }
            )

    edge_csv_path = ctx.path("tables", "edge_interactions.csv")
    bench.write_csv_with_context(ctx, edge_csv_path, sorted(edge_rows, key=lambda r: r["interaction"], reverse=True))
    ctx.register_artifact(edge_csv_path, "table", "All ordered previous-token to induction ablation-interaction checks.")

    if edge_rows:
        best_edge = max(edge_rows, key=lambda r: r["interaction"])
        h_prev = (best_edge["from_layer"], best_edge["from_head"])
        h_ind = (best_edge["to_layer"], best_edge["to_head"])
        raw_fraction = best_edge["raw_interaction_fraction"]
        claimed = bool(best_edge["claimable_fraction"] and best_edge["interaction"] > 0)
        if claimed:
            reason = f"ordered pair has positive interaction above the reporting threshold; strength={best_edge['edge_strength']}"
        elif raw_fraction is not None and raw_fraction > 1.0:
            reason = "interaction is positive but larger than the source effect, so it is not a literal routed fraction"
        elif best_edge["interaction"] <= 0:
            reason = "best ordered pair has no positive interaction"
        else:
            reason = "best ordered pair is below the routed-fraction threshold"
        edge = {
            "claimed": claimed,
            "from": h_prev,
            "to": h_ind,
            "edge": best_edge["edge"],
            "effect_prev_alone": best_edge["effect_prev_alone"],
            "effect_prev_given_induction_ablated": best_edge["effect_prev_given_induction_ablated"],
            "interaction": best_edge["interaction"],
            "raw_interaction_fraction": raw_fraction,
            "strength": best_edge["edge_strength"],
            "reason": reason,
        }
    else:
        edge = {
            "claimed": False,
            "edge": None,
            "reason": (
                "No ordered previous-token head before an induction head survived the causal and motif checks. "
                "The lab therefore makes no edge claim (expected for non-induction behaviors such as successor)."
            ),
        }

    edge_path = ctx.path("tables", "edge_claim.json")
    bench.write_json(
        edge_path,
        {
            **edge,
            "thresholds": {
                "min_source_effect": EDGE_MIN_SOURCE_EFFECT,
                "min_routed_fraction": EDGE_MIN_ROUTED_FRACTION,
                "strong_routed_fraction": EDGE_STRONG_ROUTED_FRACTION,
                "requires_source_layer_lt_target_layer": True,
            },
            "explanation": (
                "Ablation interaction asks whether a previous-token head's effect shrinks when the induction head "
                "is already ablated. This licenses only an interaction-granularity edge, not a path-patched route."
            ),
        },
    )
    ctx.register_artifact(edge_path, "metrics", "The one edge claim, or the reason no edge was claimed.")
    if edge.get("claimed"):
        print(f"[lab6] {edge['strength']} edge claimed: {edge['edge']} ({edge['raw_interaction_fraction']:.0%} interaction fraction)")
    else:
        print(f"[lab6] no edge claimed: {edge['reason']}")

    # ----- per-prompt faithfulness (knee circuit, mean ablation, for plots) --
    per_prompt_rows: list[dict[str, Any]] = []
    heads_ab, mlps_ab = split_nodes(knee_circuit)
    complement_heads = [h for h in all_head_nodes if h not in set(knee_circuit)]
    comp_h, comp_m = split_nodes([n for n in universe if n not in set(knee_circuit)])
    for ex in list(discovery) + list(heldout):
        logits = bench.run_with_node_set_ablation(
            bundle, ex.prompt.prompt, head_anatomy, comp_anatomy, heads=comp_h, mlps=comp_m, head_means=head_means, mlp_means=mlp_means
        )
        circuit_diff = float(logits[ex.target_id] - logits[ex.distractor_id])
        per_prompt_rows.append(
            {
                "example_id": ex.prompt.example_id,
                "family": ex.prompt.family,
                "prompt": ex.prompt.prompt,
                "base_diff": round(ex.base_diff, 5),
                "circuit_diff": round(circuit_diff, 5),
                "faithfulness": round(circuit_diff / ex.base_diff, 5) if ex.base_diff != 0 else None,
                "note": "",
            }
        )
    per_prompt_path = ctx.path("tables", "per_prompt_faithfulness.csv")
    bench.write_csv_with_context(ctx, per_prompt_path, per_prompt_rows)
    ctx.register_artifact(per_prompt_path, "table", "Per-prompt faithfulness (knee circuit, mean ablation).")
    prompt_failure_rows = build_prompt_failure_modes(per_prompt_rows, floor=FAITHFULNESS_FLOOR)
    prompt_failure_path = ctx.path("tables", "prompt_failure_modes.csv")
    bench.write_csv_with_context(ctx, prompt_failure_path, prompt_failure_rows)
    ctx.register_artifact(prompt_failure_path, "table", "Prompt-level failure modes, including over-recovery and under-floor cases.")
    failures = sorted(
        (row for row in prompt_failure_rows if row["faithfulness"] is not None), key=lambda row: row["faithfulness"]
    )[:2]

    evidence_rows = build_circuit_evidence_matrix(cand_rows_sorted, minimality_rows, circuit_heads, edge)
    evidence_path = ctx.path("tables", "circuit_evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence_rows)
    ctx.register_artifact(evidence_path, "table", "Joined evidence for every screened head.")
    guide_path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, guide_path, plot_reading_guide_rows())
    ctx.register_artifact(guide_path, "table", "Short guide mapping each Lab 6 visualization to the concept it teaches.")

    # ----- decide the verdict (PART 6) ---------------------------------------
    knee_held = circuits["knee"]["heldout"]
    motif_held = circuits["motif_core"]["heldout"]
    knee_disc = circuits["knee"]["discovery"]
    knee_held_res = knee_held["faith_resample"] if knee_held else None
    motif_held_res = motif_held["faith_resample"] if motif_held else None
    mean_resample_gap_disc = round(knee_disc["faith_mean"] - knee_disc["faith_resample"], 5) if knee_disc else None
    mlp_positive = sorted(((l, d) for l, d in mlp_causal_drop.items() if d > 0), key=lambda kv: -kv[1])
    mlp_in_knee = [node_label(n) for n in knee_circuit if n[0] == "mlp"]

    def decide_verdict() -> tuple[str, str]:
        if not heldout or knee_held_res is None:
            return "INSUFFICIENT PROMPTS", "no baseline-positive held-out prompts available to test transfer"
        comparable = (
            motif_held_res is not None
            and motif_held_res >= knee_held_res - 0.15
            and motif_held_res >= FAITHFULNESS_FLOOR - 0.10
        )
        if knee_held_res >= FAITHFULNESS_FLOOR and comparable:
            return (
                "CIRCUIT CONFIRMED",
                f"knee held-out resample faithfulness {knee_held_res:.2f} >= {FAITHFULNESS_FLOOR:.2f}; "
                f"motif-core-only transfers comparably ({motif_held_res:.2f}).",
            )
        if knee_held_res >= FAITHFULNESS_FLOOR and not comparable:
            return (
                "OVERFIT / NO CLEAN CIRCUIT",
                f"knee transfers ({knee_held_res:.2f}) but the motif core alone does not "
                f"({motif_held_res}); the extra heads are filler, not mechanism.",
            )
        if (knee_disc and knee_disc["faith_resample"] >= FAITHFULNESS_FLOOR) or fcm.get("discovery", {}).get("faithfulness", 0.0) >= FAITHFULNESS_FLOOR:
            return (
                "OVERFIT / NO CLEAN CIRCUIT",
                f"discovery passes but held-out resample faithfulness {knee_held_res:.2f} < {FAITHFULNESS_FLOOR:.2f}.",
            )
        if behavior in CORE_MOTIFS_BY_BEHAVIOR and not induction_motif_present and not edge.get("claimed"):
            return (
                "MECHANISM ABSENT",
                "the expected previous-token -> induction motif does not survive screening/causal checks, "
                "and no transferable subgraph was found. Reported as a successful negative.",
            )
        return (
            "OVERFIT / NO CLEAN CIRCUIT",
            f"no transferable subgraph: knee held-out resample {knee_held_res:.2f} < {FAITHFULNESS_FLOOR:.2f}.",
        )

    verdict, verdict_reason = decide_verdict()
    print(f"[lab6] VERDICT ({behavior}/{scope}): {verdict} -- {verdict_reason}")

    # ----- plots -------------------------------------------------------------
    if not args.no_plots:
        plot_circuit_discovery_dashboard(ctx, fcm, traj_csv, cand_rows_sorted, prompt_failure_rows, floor=FAITHFULNESS_FLOOR)
        plot_screen_vs_causal(ctx, cand_rows_sorted)
        plot_prune_trajectory(ctx, traj_csv, floor=FAITHFULNESS_FLOOR)
        mlp_support = [r for r in cand_rows_sorted if r["kind"] == "mlp" and float(r["causal_drop"]) > 0]
        plot_circuit_graph(ctx, circuit_heads, head_labels, mlp_support, edge, n_layers, n_heads)
        if "discovery" in fcm:
            plot_fcm(ctx, fcm, floor=FAITHFULNESS_FLOOR)
        plot_prompt_faithfulness(ctx, prompt_failure_rows)
        plot_prompt_failure_scatter(ctx, prompt_failure_rows, floor=FAITHFULNESS_FLOOR)
        plot_minimality_ledger(ctx, minimality_rows)
        plot_candidate_evidence_matrix(ctx, evidence_rows)
        plot_causal_motif_atlas(ctx, evidence_rows, n_layers, n_heads)
        plot_edge_interactions(ctx, edge_rows)
        plot_edge_interaction_map(ctx, edge_rows)

    # ----- circuit card ------------------------------------------------------
    supporting_mlps = sorted(
        (r for r in cand_rows_sorted if r["kind"] == "mlp" and float(r["causal_drop"]) > 0),
        key=lambda row: -float(row["causal_drop"]),
    )
    _write_card(
        ctx, bundle, behavior, scope, base_metric, discovery, heldout, hygiene, seq_len,
        knee_circuit, floor_circuit, knee_row, floor_row, peak_faith, knee_vs_floor_gap, meets_floor,
        circuits, motif_core, motif_core_def, induction_motif_present, fcm, minimality_rows,
        rent_unpaid_labels, rent_compare, suppression_heads, head_causal_drop, brake_intact,
        supporting_mlps, mlp_in_knee, mean_resample_gap_disc, edge, failures, verdict, verdict_reason,
        seed_capped, resample_draws,
    )

    # ----- metrics + ledger + run summary ------------------------------------
    metrics = {
        "behavior": behavior,
        "scope": scope,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "aborted": False,
        "base_metric": base_metric,
        "run_length_tokens": seq_len,
        "n_discovery": len(discovery),
        "n_heldout": len(heldout),
        "resample_draws": resample_draws,
        "knee_circuit": [node_label(n) for n in knee_circuit],
        "floor_circuit": [node_label(n) for n in floor_circuit],
        "knee_n_nodes": knee_row["n_nodes"],
        "floor_n_nodes": floor_row["n_nodes"],
        "peak_faithfulness": round(peak_faith, 5),
        "knee_minus_floor_faithfulness": knee_vs_floor_gap,
        "meets_faithfulness_floor": meets_floor,
        "headline_heldout_faith_resample": knee_held_res,
        "headline_discovery_faith_resample": knee_disc["faith_resample"] if knee_disc else None,
        "mean_minus_resample_gap_discovery": mean_resample_gap_disc,
        "motif_core_def": motif_core_def,
        "motif_core_heldout_faith_resample": motif_held_res,
        "induction_motif_present": induction_motif_present,
        "circuits": circuits,
        "brake_intact": brake_intact,
        "suppression_heads": [{"node": node_name("head", *k), "single_drop": round(d, 5)} for k, d in suppression_heads],
        "mlp_positive_causal": [{"node": node_name("mlp", l), "drop": round(d, 5)} for l, d in mlp_positive],
        "mlp_in_knee_circuit": mlp_in_knee,
        "edge": edge,
        "minimality_worst_marginal": fcm["minimality_worst_marginal"],
        "n_rent_unpaid_nodes": len(rent_unpaid_labels),
        "seed_capped": seed_capped,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 6 metrics, including the per-cell verdict.")

    _write_ledger_and_summary(
        ctx, bundle, behavior, scope, n_layers, n_heads, base_metric, discovery, heldout, seq_len,
        knee_circuit, floor_circuit, circuits, fcm, edge, verdict, verdict_reason, knee_vs_floor_gap,
        mean_resample_gap_disc, motif_held_res, suppression_heads, mlp_positive,
    )
    print(f"[lab6] wrote circuit_card.md, run_summary.md, metrics.json; verdict {verdict}")


def _fmt(value: Any) -> str:
    return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def _write_card(
    ctx, bundle, behavior, scope, base_metric, discovery, heldout, hygiene, seq_len,
    knee_circuit, floor_circuit, knee_row, floor_row, peak_faith, knee_vs_floor_gap, meets_floor,
    circuits, motif_core, motif_core_def, induction_motif_present, fcm, minimality_rows,
    rent_unpaid_labels, rent_compare, suppression_heads, head_causal_drop, brake_intact,
    supporting_mlps, mlp_in_knee, mean_resample_gap_disc, edge, failures, verdict, verdict_reason,
    seed_capped, resample_draws,
):
    knee_disc = circuits["knee"]["discovery"] or {}
    knee_held = circuits["knee"]["heldout"] or {}
    motif_held = circuits["motif_core"]["heldout"] or {}
    card = [
        f"# Circuit card: {behavior} ({scope})",
        "",
        f"- **Verdict:** **{verdict}**",
        f"  - {verdict_reason}",
        f"- **Model:** `{bundle.anatomy.model_id}` | run `{ctx.run_dir.name}`",
        f"- **Behavior:** `{behavior}`; **scope:** `{scope}`; run length {seq_len} tokens.",
        f"- **Metric:** mean logit(target) - logit(distractor).",
        f"- **Base metric:** {base_metric:+.3f} on {len(discovery)} discovery prompts "
        f"({len(heldout)} baseline-positive held-out).",
        f"- **Primary off distribution:** resample / interchange ablation ({resample_draws} within-distribution draws). "
        f"Mean ablation reported alongside as comparison.",
        "",
        "## Headline numbers (knee circuit)",
        "",
        "| family | faithfulness (resample) | faithfulness (mean) | completeness (resample) |",
        "|---|---:|---:|---:|",
        f"| discovery | {_fmt(knee_disc.get('faith_resample'))} | {_fmt(knee_disc.get('faith_mean'))} | "
        f"{_fmt(knee_disc.get('completeness_ratio_resample'))} |",
        f"| held-out | {_fmt(knee_held.get('faith_resample'))} | {_fmt(knee_held.get('faith_mean'))} | "
        f"{_fmt(knee_held.get('completeness_ratio_resample'))} |",
        "",
        f"- Mean-minus-resample gap on discovery: **{_fmt(mean_resample_gap_disc)}** "
        "(large positive = mean-ablation inflation, usually suppression-head removal).",
        "",
        "## Knee vs floor (FIX 1)",
        "",
        f"- Peak trajectory faithfulness: {peak_faith:.3f}.",
        f"- **Knee** (headline): {knee_row['n_nodes']} nodes, faithfulness {knee_row['faithfulness']:.3f} "
        f"-> `{describe_nodes(knee_circuit)}`.",
        f"- **Floor** (overfit comparison): {floor_row['n_nodes']} nodes, faithfulness {floor_row['faithfulness']:.3f}.",
        f"- Knee-minus-floor faithfulness gap: {knee_vs_floor_gap:+.3f}. The floor circuit is the floor-hugger; "
        "the knee is the honest circuit.",
        f"- Meets {FAITHFULNESS_FLOOR:.2f} faithfulness floor (mean, discovery): {'yes' if meets_floor else 'no'}."
        + ("  Seed circuit was capped (disclosed)." if seed_capped else ""),
        "",
        "## Motif-core-only transfer (FIX 7)",
        "",
        f"- Motif core: {motif_core_def} -> `{describe_nodes(motif_core)}`.",
        f"- Motif-core held-out faithfulness (resample): **{_fmt(motif_held.get('faith_resample'))}** "
        f"vs knee held-out **{_fmt(knee_held.get('faith_resample'))}**.",
        f"- Induction motif present among causal heads: {'yes' if induction_motif_present else 'no'}.",
        "",
        "## Suppression / anti-circuit heads (FIX 4)",
        "",
    ]
    if suppression_heads:
        card += ["| head | single-head causal drop (negative = brake) |", "|---|---:|"]
        card += [f"| {node_name('head', *k)} | {d:+.3f} |" for k, d in suppression_heads]
        card += [
            "",
            f"- Brake-intact faithfulness (those {len(suppression_heads)} heads left LIVE in the complement): "
            f"discovery resample {_fmt(brake_intact.get('discovery_faith_resample'))} "
            f"(vs {_fmt(knee_disc.get('faith_resample'))}), "
            f"discovery mean {_fmt(brake_intact.get('discovery_faith_mean'))} "
            f"(vs {_fmt(knee_disc.get('faith_mean'))}). "
            "The drop when brakes are held out shows how much faithfulness was just brake removal.",
        ]
    else:
        card.append("- None: no screened head raised the metric by more than "
                    f"{SUPPRESSION_REL_THRESHOLD:.0%} of base when ablated.")
    card += [
        "",
        "## MLP contribution (FIX 3)",
        "",
        f"- Scope `{scope}`. MLP nodes inside the knee circuit: "
        + (", ".join(mlp_in_knee) if mlp_in_knee else "none"),
        "- Supporting MLPs by single-MLP causal drop: "
        + (", ".join(f"MLP{r['layer']} ({float(r['causal_drop']):+.2f})" for r in supporting_mlps) if supporting_mlps else "none with positive drop"),
        "",
        "## Minimality (FIX 7)",
        "",
        f"- Worst marginal value: `{fcm['minimality_worst_marginal']}`; rent-unpaid (filler) nodes "
        f"(marginal < {MARGINAL_RENT_THRESHOLD}): {len(rent_unpaid_labels)}"
        + (f" -> {', '.join(sorted(rent_unpaid_labels))}" if rent_unpaid_labels else "") + ".",
    ]
    if "rent_paid_heldout_faith_resample" in rent_compare:
        card.append(
            f"- Held-out resample with vs without filler: "
            f"{_fmt(rent_compare.get('knee_heldout_faith_resample'))} (knee) vs "
            f"{_fmt(rent_compare.get('rent_paid_heldout_faith_resample'))} (rent-paid only)."
        )
    card += ["", "## Edge claim", ""]
    if edge and edge.get("claimed"):
        card.append(
            f"Claimed {edge['strength']} edge `{edge['edge']}`: raw interaction fraction "
            f"{edge['raw_interaction_fraction']:.0%}. Ablation-interaction granularity, not path patching."
        )
    else:
        card.append(f"No edge claimed. Reason: {edge['reason'] if edge else 'no edge diagnostic produced'}")
    card += ["", "## Failure cases the circuit least explains", ""]
    if failures:
        for row in failures:
            card.append(f"- `{row['example_id']}` ({row['family']}): faithfulness {row['faithfulness']} "
                        f"with base diff {row['base_diff']}.")
    else:
        card.append("- No ratio-defined failure cases were available.")
    card += [
        "",
        "## Scope and honesty",
        "",
        f"- Population: `{behavior}` prompts at run length {seq_len}; discovery+held-out share one token length.",
        f"- Circuit nodes: {'attention heads + MLP layers' if scope == 'heads_and_mlps' else 'attention heads only (MLPs intact)'}.",
        "- Primary intervention: resample/interchange ablation. Mean ablation defines a different (often inflated) circuit.",
        "- Edge evidence: ablation interaction only. Keys/values/subpaths are filler terms unless path patching is added.",
        "- A confirmed NO (OVERFIT / MECHANISM ABSENT / NOT HEADS-ONLY) is a successful result, not a failure.",
        "",
    ]
    bench.write_text(ctx.path("circuit_card.md"), "\n".join(card))
    ctx.register_artifact(ctx.path("circuit_card.md"), "summary", "The Lab 6 circuit card deliverable.")


def _write_ledger_and_summary(
    ctx, bundle, behavior, scope, n_layers, n_heads, base_metric, discovery, heldout, seq_len,
    knee_circuit, floor_circuit, circuits, fcm, edge, verdict, verdict_reason, knee_vs_floor_gap,
    mean_resample_gap_disc, motif_held_res, suppression_heads, mlp_positive,
):
    run_name = ctx.run_dir.name
    knee_disc = circuits["knee"]["discovery"] or {}
    knee_held = circuits["knee"]["heldout"] or {}
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "CAUSAL",
            "text": (
                f"On {behavior} in {bundle.anatomy.model_id} ({scope}), the knee circuit "
                f"({describe_nodes(knee_circuit)}) has discovery faithfulness "
                f"{_fmt(knee_disc.get('faith_resample'))} (resample) / {_fmt(knee_disc.get('faith_mean'))} (mean) and "
                f"held-out {_fmt(knee_held.get('faith_resample'))} (resample). Verdict: {verdict}."
            ),
            "artifact": f"runs/{run_name}/faithfulness_completeness_minimality.json",
            "falsifier": (
                "Held-out resample faithfulness below the floor, a motif-core that transfers as well as the full "
                "knee, or a different off distribution changing the verdict."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": (
                f"Mean-minus-resample faithfulness gap on discovery is {_fmt(mean_resample_gap_disc)} with "
                f"{len(suppression_heads)} suppression heads detected: evidence on whether mean ablation inflates "
                "faithfulness via brake removal."
            ),
            "artifact": f"runs/{run_name}/metrics.json",
            "falsifier": "Mean and resample agree and no suppression heads exist; then the inflation claim is refuted.",
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)

    lines = [
        f"# Lab 6 run summary: {behavior} ({scope})",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({n_layers} blocks x {n_heads} heads)",
        f"- behavior `{behavior}`, scope `{scope}`, run length {seq_len}",
        f"- {len(discovery)} discovery + {len(heldout)} held-out prompts",
        "- primary intervention: resample/interchange ablation; mean ablation reported as comparison",
        "",
        "## Verdict",
        "",
        f"- **{verdict}** -- {verdict_reason}",
        "",
        "## Headline",
        "",
        f"- base metric {base_metric:+.3f}; knee circuit {len(knee_circuit)} nodes: {describe_nodes(knee_circuit)}",
        f"- discovery faithfulness: resample {_fmt(knee_disc.get('faith_resample'))}, mean {_fmt(knee_disc.get('faith_mean'))}",
        f"- held-out faithfulness: resample {_fmt(knee_held.get('faith_resample'))}, mean {_fmt(knee_held.get('faith_mean'))}",
        f"- motif-core held-out (resample): {_fmt(motif_held_res)}",
        f"- knee-minus-floor faithfulness gap: {knee_vs_floor_gap:+.3f}",
        f"- mean-minus-resample gap (discovery): {_fmt(mean_resample_gap_disc)}",
        f"- suppression heads: {len(suppression_heads)}; positive-causal MLPs: {len(mlp_positive)}",
        f"- edge: {edge['edge'] + ' (' + edge['strength'] + ')' if edge and edge.get('claimed') else 'none claimed'}",
        "",
        "## Claims",
        "",
    ]
    for claim in claims:
        lines.append(f"- `{claim['id']}` {claim['tag']}: {claim['text']}")
        lines.append(f"  - falsifier: {claim['falsifier']}")
    lines += [
        "",
        "## Reading order",
        "",
        "1. `circuit_card.md` - the deliverable and the verdict.",
        "2. `prompt_hygiene_report.md` - which prompts the gate kept or excluded.",
        "3. `faithfulness_completeness_minimality.json` - knee/floor/motif-core under mean and resample.",
        "4. `tables/knee_floor_selection.json` and `plots/prune_trajectory.png` - knee vs floor.",
        "5. `metrics.json` - verdict, suppression heads, MLP contribution, mean-vs-resample gap.",
        "",
    ]
    bench.write_text(ctx.path("run_summary.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("run_summary.md"), "summary", "Lab 6 run summary with the per-cell verdict.")
