"""Lab 12: Relation geometry and method validation.

First lab of the advanced course (Group I: bridge and instruments). The intro
course localized a few factual-recall cases; this lab points the SAME toolkit
(Lab 4 probes-with-controls, Lab 5 interchange patching) at a scaled-up,
deliberately confound-controlled relation dataset, and asks:

    Do relation classes share measurable internal geometry, or is each
    relation its own small trick wearing one probe?

The dataset (data/advanced_relation_geometry.csv) is built so the two
cheapest explanations of a "relation direction" are controlled by
construction, via three RELATION-SWAP GROUPS:

    country_sem: capital_of / language_of / continent_of over the SAME
                 countries with the SAME template skeleton;
    adj_morph:   opposite_of / comparative_of over the SAME adjectives;
    month_seq:   month_after / month_before over the SAME months.

Inside a swap group, prompts for one subject differ at EXACTLY the
relation-word token. So if a probe separates the families inside a group,
the separation cannot be entity class and cannot be template syntax. What
remains possible — and the operationalization audit says so loudly — is a
relation-word token echo; the patching phase and the direction-cosine atlas
are how that residual cheap explanation gets pressure-tested.

Two intervention families, both Lab 5 interchange patching:

  * SUBJECT-SWAP (within relation): clean "The capital of France is" vs
    corrupt "The capital of Germany is". Recovery by depth x role per
    relation family. The cross-family comparison of these localization
    profiles is the "shared machinery" HANDLE (not mechanism).
  * RELATION-SWAP (within swap group, same subject): clean "The capital of
    France is" (Paris) vs corrupt "The language of France is" (French),
    patched at the relation-word token. Recovery here is causal evidence
    that the residual at the relation token carries relation identity that
    the answer pathway actually uses.

Controls riding the same rails: shuffled relation labels, random directions,
within-group common-subject restriction (entity control), wrong-position
patching (the shared " of"/template token), and mismatched-vector patching.

Evidence levels: DECODE for the probe phase, CAUSAL (scoped) for the
patching phase, OBS for margin stability and profile similarity. Nothing in
this lab shows HOW relation information is computed; positive results are
handles for later labs, and the audit tries to kill the favorite reading.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any

import interp_bench as bench

LAB_ID = "L12"

DATA_FILE = "advanced_relation_geometry.csv"

# --prompt-set (or --relation-set) controls the per-family item cap;
# --max-examples > 0 overrides it. 0 = no cap.
PROMPT_SET_FAMILY_CAPS = {"small": 8, "medium": 16, "full": 0}

# Patch-pair budgets per prompt-set size: (subject-swap pairs per family,
# relation-swap subjects per ordered family pair).
PATCH_BUDGETS = {"small": (3, 3), "medium": (5, 5), "full": (8, 8)}

TRAIN_FRACTION = 0.7
N_SHUFFLES = 2          # shuffled-label refits per (scope, role, depth)
N_RANDOM_DIRS = 3       # random-direction AUC baselines per (role, depth)
NORMALIZE_ROWS = True   # unit-normalize activation rows, as in Lab 4
LOGISTIC_L2 = 1e-2
MIN_ITEMS_FOR_PROBES = 24

# A positive result should beat controls by a visible amount before the summary
# writes positive causal language. The numbers are deliberately conservative
# teaching thresholds, not universal scientific constants.
MIN_SELECTIVITY_FOR_DECODE_CLAIM = 0.10
MIN_PATCH_GAP_FOR_CAUSAL_CLAIM = 0.15

# Behavioral gate margins (logits), as in Lab 5: a patching pair is usable
# only if the clean prompt prefers the clean answer and the corrupt prompt
# prefers the corrupt answer by enough margin for a meaningful denominator.
CLEAN_MARGIN = 0.5
CORRUPT_MARGIN = 0.5

ROLES = ("relword", "subject", "final")
SWAP_GROUPS = ("country_sem", "adj_morph", "month_seq")


@dataclasses.dataclass
class Item:
    """One frozen relation item plus its runtime tokenization facts."""

    item_id: str
    family: str
    swap_group: str
    entity_group: str
    prompt: str
    relword: str
    subject: str
    target: str
    hard_distractor: str
    easy_distractor: str
    note: str
    # Filled by the tokenization gate:
    input_ids: list[int] = dataclasses.field(default_factory=list)
    subject_pos: int = -1
    relword_pos: int = -1
    target_id: int = -1
    hard_id: int = -1
    easy_id: int = -1
    # Filled after caching:
    margin_hard: float = 0.0
    margin_easy: float = 0.0


# ---------------------------------------------------------------------------
# Small utilities
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


def auc_from_scores(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUC (Mann-Whitney). Ties get half credit."""
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


def safe_fmean(vals: list[float], default: float = float("nan")) -> float:
    finite = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(statistics.fmean(finite)) if finite else default


# ---------------------------------------------------------------------------
# Data loading and the tokenization gate
# ---------------------------------------------------------------------------


def _builtin_smoke_rows() -> list[dict[str, str]]:
    """Tiny deterministic fallback used only for Tier A plumbing when the
    frozen CSV is absent. It is intentionally simple and is labeled loudly in
    every manifest; Tier B/C science runs must use the committed dataset."""
    rows: list[dict[str, str]] = []

    def add(item_id: str, family: str, swap_group: str, entity_group: str,
            relword: str, subject: str, target: str, hard: str, easy: str,
            prompt: str, note: str = "builtin_smoke_fallback") -> None:
        rows.append({
            "item_id": item_id,
            "family": family,
            "swap_group": swap_group,
            "entity_group": entity_group,
            "prompt": prompt,
            "relword": relword,
            "subject": subject,
            "target": target,
            "hard_distractor": hard,
            "easy_distractor": easy,
            "note": note,
        })

    countries = [
        ("france", "France", " Paris", " French", " Europe"),
        ("germany", "Germany", " Berlin", " German", " Europe"),
        ("italy", "Italy", " Rome", " Italian", " Europe"),
        ("spain", "Spain", " Madrid", " Spanish", " Europe"),
        ("japan", "Japan", " Tokyo", " Japanese", " Asia"),
        ("china", "China", " Beijing", " Chinese", " Asia"),
        ("india", "India", " Delhi", " Hindi", " Asia"),
        ("brazil", "Brazil", " Brasilia", " Portuguese", " America"),
        ("canada", "Canada", " Ottawa", " English", " America"),
        ("egypt", "Egypt", " Cairo", " Arabic", " Africa"),
    ]
    for slug, country, capital, language, continent in countries:
        add(f"smoke_capital_{slug}", "capital_of", "country_sem", "country", "capital", country,
            capital, language, continent, f"The capital of {country} is")
        add(f"smoke_language_{slug}", "language_of", "country_sem", "country", "language", country,
            language, capital, continent, f"The language of {country} is")
        add(f"smoke_continent_{slug}", "continent_of", "country_sem", "country", "continent", country,
            continent, capital, language, f"The continent of {country} is")

    adjectives = [
        ("hot", "hot", " cold", " hotter"),
        ("cold", "cold", " hot", " colder"),
        ("big", "big", " small", " bigger"),
        ("small", "small", " big", " smaller"),
        ("fast", "fast", " slow", " faster"),
        ("slow", "slow", " fast", " slower"),
        ("high", "high", " low", " higher"),
        ("low", "low", " high", " lower"),
        ("young", "young", " old", " younger"),
        ("old", "old", " young", " older"),
    ]
    for slug, adj, opposite, comparative in adjectives:
        add(f"smoke_opposite_{slug}", "opposite_of", "adj_morph", "adjective", "opposite", adj,
            opposite, comparative, " Paris", f"The opposite of {adj} is")
        add(f"smoke_comparative_{slug}", "comparative_of", "adj_morph", "adjective", "comparative", adj,
            comparative, opposite, " Paris", f"The comparative of {adj} is")

    months = [
        ("january", "January", " February", " December"),
        ("february", "February", " March", " January"),
        ("march", "March", " April", " February"),
        ("april", "April", " May", " March"),
        ("may", "May", " June", " April"),
        ("june", "June", " July", " May"),
        ("july", "July", " August", " June"),
        ("august", "August", " September", " July"),
        ("september", "September", " October", " August"),
        ("october", "October", " November", " September"),
        ("november", "November", " December", " October"),
        ("december", "December", " January", " November"),
    ]
    for slug, month, after, before in months:
        add(f"smoke_after_{slug}", "month_after", "month_seq", "month", "after", month,
            after, before, " Paris", f"The month after {month} is")
        add(f"smoke_before_{slug}", "month_before", "month_seq", "month", "before", month,
            before, after, " Paris", f"The month before {month} is")
    return rows


def _rows_to_items(rows: list[dict[str, str]]) -> list[Item]:
    return [Item(
        item_id=row["item_id"],
        family=row["family"],
        swap_group=row.get("swap_group", ""),
        entity_group=row.get("entity_group", ""),
        prompt=row["prompt"],
        relword=row["relword"],
        subject=row["subject"],
        target=row["target"],
        hard_distractor=row["hard_distractor"],
        easy_distractor=row["easy_distractor"],
        note=row.get("note", ""),
    ) for row in rows]


def _manifest_expected_hash(path: pathlib.Path) -> tuple[str | None, str]:
    """Best-effort lookup in data/MANIFEST.json. The generator/manifest format
    may evolve, so failure returns (None, reason) instead of aborting."""
    manifest_path = path.parent / "MANIFEST.json"
    if not manifest_path.exists():
        return None, "data/MANIFEST.json not found"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"data/MANIFEST.json unreadable: {exc}"
    candidates: list[Any] = []
    if isinstance(manifest, dict):
        candidates.extend([manifest.get(path.name), manifest.get(str(path)), manifest.get("files", {}).get(path.name)])
    for entry in candidates:
        if isinstance(entry, str):
            return entry, "found string entry"
        if isinstance(entry, dict):
            for key in ("sha256", "hash", "sha256_hex"):
                val = entry.get(key)
                if isinstance(val, str):
                    return val, f"found {key} entry"
    return None, "no usable sha256 entry for advanced_relation_geometry.csv"


def load_items(args: Any) -> tuple[list[Item], dict[str, Any]]:
    path = bench.COURSE_ROOT / "data" / DATA_FILE
    data_source = "frozen_csv"
    expected_sha, manifest_note = _manifest_expected_hash(path)
    actual_sha = None
    manifest_ok: bool | None = None
    if path.exists():
        actual_sha = bench.sha256_file(path)
        manifest_ok = (expected_sha == actual_sha) if expected_sha else None
        rows: list[dict[str, str]] = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
    else:
        tier = str(getattr(args, "tier", "")).lower()
        if tier != "a":
            raise RuntimeError(
                f"Frozen dataset missing: {path}. Science runs must use the committed "
                "data/advanced_relation_geometry.csv generated by data/make_advanced_relation_sets.py."
            )
        print("[lab12] frozen CSV missing; using builtin Tier A smoke fallback. "
              "This run is plumbing only and should not enter the claim ledger.")
        data_source = "builtin_tier_a_smoke_fallback"
        rows = _builtin_smoke_rows()
        actual_sha = hashlib.sha256("\n".join(r["item_id"] for r in rows).encode("utf-8")).hexdigest()
        manifest_note = "frozen CSV absent; builtin fallback has no manifest entry"
        manifest_ok = False

    required = {"item_id", "family", "swap_group", "entity_group", "prompt", "relword", "subject",
                "target", "hard_distractor", "easy_distractor"}
    if rows:
        missing = sorted(required - set(rows[0]))
        if missing:
            raise ValueError(f"{DATA_FILE} is missing columns: {missing}")
    items = _rows_to_items(rows)

    # --relation-set is the Lab 12 alias; it wins over --prompt-set when given.
    set_name = getattr(args, "relation_set", "") or args.prompt_set
    if set_name not in PROMPT_SET_FAMILY_CAPS:
        raise ValueError(
            f"Lab 12 uses the frozen relation CSV; --prompt-set/--relation-set "
            f"selects size only (small|medium|full), got {set_name!r}."
        )
    cap = PROMPT_SET_FAMILY_CAPS[set_name]
    if args.max_examples > 0:
        cap = args.max_examples

    wanted = {f.strip() for f in str(getattr(args, "relations", "")).split(",") if f.strip()}
    families_present = {it.family for it in items}
    if wanted:
        unknown = wanted - families_present
        if unknown:
            raise ValueError(f"--relations names unknown families: {sorted(unknown)}; "
                             f"available: {sorted(families_present)}")
        items = [it for it in items if it.family in wanted]

    if cap > 0:
        capped: list[Item] = []
        seen: dict[str, int] = defaultdict(int)
        for it in items:  # CSV order is roster order, shared across swap families.
            if seen[it.family] < cap:
                capped.append(it)
                seen[it.family] += 1
        items = capped

    family_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    for it in items:
        family_counts[it.family] += 1
        group_counts[it.swap_group or "ungrouped"] += 1
    info = {
        "relation_set": set_name,
        "per_family_cap": cap,
        "relations_filter": sorted(wanted),
        "families": sorted(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "swap_group_counts": dict(sorted(group_counts.items())),
        "n_items": len(items),
        "data_source": data_source,
        "data_path": str(path),
        "data_sha256": actual_sha,
        "manifest_expected_sha256": expected_sha,
        "manifest_note": manifest_note,
        "manifest_ok": manifest_ok,
    }
    return items, info


def tokenization_gate(
    ctx: bench.RunContext, bundle: bench.ModelBundle, items: list[Item]
) -> list[Item]:
    """Runtime tokenization validation for every measured object.

    The CSV is authoring-time verified, but this lab never trusts authoring-time
    facts about a tokenizer it did not see. Dropped rows are written as data.
    The role positions are located in the exact encoding used by the bench's
    residual cache (add_special_tokens=True for base-model labs)."""
    tokenizer = bundle.tokenizer
    kept: list[Item] = []
    rows: list[dict[str, Any]] = []
    for it in items:
        problems: list[str] = []
        enc = tokenizer(it.prompt, add_special_tokens=True)["input_ids"]
        tokens_text = [tokenizer.decode([i]) for i in enc]

        def one_token_id(text: str, label: str) -> int:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                problems.append(f"{label} {text!r} is {len(ids)} tokens: {ids}")
                return -1
            return int(ids[0])

        subj_id = one_token_id(" " + it.subject, "subject")
        rel_id = one_token_id(" " + it.relword, "relword")
        target_id = one_token_id(it.target, "target")
        hard_id = one_token_id(it.hard_distractor, "hard_distractor")
        easy_id = one_token_id(it.easy_distractor, "easy_distractor")

        if target_id >= 0 and hard_id >= 0 and target_id == hard_id:
            problems.append("target and hard_distractor have the same token id")
        if target_id >= 0 and easy_id >= 0 and target_id == easy_id:
            problems.append("target and easy_distractor have the same token id")
        if hard_id >= 0 and easy_id >= 0 and hard_id == easy_id:
            problems.append("hard_distractor and easy_distractor have the same token id")

        subject_pos = relword_pos = -1
        subj_hits: list[int] = []
        rel_hits: list[int] = []
        if subj_id >= 0:
            subj_hits = [p for p, t in enumerate(enc) if t == subj_id]
            if len(subj_hits) == 1:
                subject_pos = subj_hits[0]
            else:
                problems.append(f"subject token occurs {len(subj_hits)}x in prompt")
        if rel_id >= 0:
            rel_hits = [p for p, t in enumerate(enc) if t == rel_id]
            if len(rel_hits) == 1:
                relword_pos = rel_hits[0]
            else:
                problems.append(f"relword token occurs {len(rel_hits)}x in prompt")
        if subject_pos >= 0 and relword_pos >= 0 and subject_pos == relword_pos:
            problems.append("subject and relation word resolved to the same position")

        rows.append({
            "item_id": it.item_id,
            "family": it.family,
            "swap_group": it.swap_group,
            "entity_group": it.entity_group,
            "prompt": it.prompt,
            "n_tokens": len(enc),
            "input_ids": " ".join(str(i) for i in enc),
            "tokens_text": " | ".join(repr(t) for t in tokens_text),
            "subject": it.subject,
            "subject_id": subj_id,
            "subject_hits": " ".join(map(str, subj_hits)),
            "subject_pos": subject_pos,
            "relword": it.relword,
            "relword_id": rel_id,
            "relword_hits": " ".join(map(str, rel_hits)),
            "relword_pos": relword_pos,
            "target": it.target,
            "target_id": target_id,
            "hard_distractor": it.hard_distractor,
            "hard_id": hard_id,
            "easy_distractor": it.easy_distractor,
            "easy_id": easy_id,
            "kept": not problems,
            "problems": "; ".join(problems),
        })
        if not problems:
            it.input_ids = list(map(int, enc))
            it.subject_pos = subject_pos
            it.relword_pos = relword_pos
            it.target_id = target_id
            it.hard_id = hard_id
            it.easy_id = easy_id
            kept.append(it)

    path = ctx.path("diagnostics", "tokenization_audit.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic",
                          "Runtime single-token and role-position verification for every item.")
    n_drop = len(items) - len(kept)
    if n_drop:
        print(f"[lab12] tokenization gate dropped {n_drop}/{len(items)} items "
              "(see diagnostics/tokenization_audit.csv)")
    if len(kept) < MIN_ITEMS_FOR_PROBES:
        raise RuntimeError(
            f"Only {len(kept)} items survived the tokenization gate; the probe phase needs "
            f"at least {MIN_ITEMS_FOR_PROBES}. Check diagnostics/tokenization_audit.csv."
        )
    return kept


def role_position(it: Item, role: str) -> int:
    if role == "relword":
        return it.relword_pos
    if role == "subject":
        return it.subject_pos
    if role == "final":
        return len(it.input_ids) - 1
    raise ValueError(f"unknown role {role!r}")


# ---------------------------------------------------------------------------
# Subject-grouped split (entity leakage hygiene, as in Lab 4)
# ---------------------------------------------------------------------------


def make_subject_split(items: list[Item], seed: int) -> dict[str, bool]:
    """Train/eval split BY SUBJECT: every item sharing a subject string lands
    on the same side, across families. Otherwise a relation probe can ride
    memorized entities (France-in-train, France-in-eval) instead of relation
    context. Deterministic repair guarantees each family keeps at least two
    train and one eval item where the data allows."""
    subjects = sorted({it.subject for it in items})
    ranked = sorted(subjects, key=lambda s: stable_hash_int(f"{seed}:{s}"))
    n_train = max(1, int(round(TRAIN_FRACTION * len(ranked))))
    train_subjects = set(ranked[:n_train])

    def counts(train: set[str]) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for it in items:
            tr, ev = out.get(it.family, (0, 0))
            if it.subject in train:
                out[it.family] = (tr + 1, ev)
            else:
                out[it.family] = (tr, ev + 1)
        return out

    for _ in range(len(ranked) * 2):
        bad = [(f, c) for f, c in counts(train_subjects).items() if c[0] < 2 or c[1] < 1]
        if not bad:
            break
        family, (n_tr, n_ev) = bad[0]
        fam_subjects = [it.subject for it in items if it.family == family]
        moved = False
        for s in ranked:
            if s not in fam_subjects:
                continue
            if n_tr < 2 and s not in train_subjects:
                train_subjects.add(s)
                moved = True
                break
            if n_ev < 1 and s in train_subjects and len(train_subjects) > 1:
                train_subjects.remove(s)
                moved = True
                break
        if not moved:
            break
    return {it.item_id: it.subject in train_subjects for it in items}


# ---------------------------------------------------------------------------
# Probe phase (DECODE)
# ---------------------------------------------------------------------------


def fit_logistic(X: Any, y: Any, l2: float = LOGISTIC_L2) -> dict[str, Any]:
    """L2 logistic regression via torch LBFGS, train-set standardization only
    (same hygiene as Lab 4: leaking eval scale inflates decodability)."""
    import torch

    mu = X.mean(dim=0)
    sigma = X.std(dim=0).clamp_min(1e-6)
    Xs = (X - mu) / sigma
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=50, line_search_fn="strong_wolfe")
    yf = y.float()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, yf)
        loss = loss + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return {"w": w.detach(), "b": b.detach(), "mu": mu, "sigma": sigma}


def logistic_scores(probe: dict[str, Any], X: Any) -> list[float]:
    Xs = (X - probe["mu"]) / probe["sigma"]
    return (Xs @ probe["w"] + probe["b"]).tolist()


def ovr_mass_mean_direction(X: Any, is_family: Any) -> Any:
    """One-vs-rest mass-mean: family mean minus rest mean (train rows only)."""
    return X[is_family].mean(dim=0) - X[~is_family].mean(dim=0)


def centroid_accuracy(X_train: Any, fams_train: list[str], X_eval: Any,
                      fams_eval: list[str], families: list[str]) -> float:
    """Nearest-class-mean multiclass accuracy with explicit missing-class
    guards. Missing eval families make the scope unearned, not zero."""
    import torch

    if X_train.shape[0] == 0 or X_eval.shape[0] == 0 or not families:
        return float("nan")
    centroids = []
    used_fams = []
    for fam in families:
        mask = torch.tensor([f == fam for f in fams_train], dtype=torch.bool)
        if bool(mask.any()):
            centroids.append(X_train[mask].mean(dim=0))
            used_fams.append(fam)
    if len(used_fams) < 2:
        return float("nan")
    eval_keep = [i for i, f in enumerate(fams_eval) if f in used_fams]
    if not eval_keep or len({fams_eval[i] for i in eval_keep}) < 2:
        return float("nan")
    Xev = X_eval[torch.tensor(eval_keep)]
    cents = torch.stack(centroids)
    d2 = torch.cdist(Xev, cents)
    pred = d2.argmin(dim=1)
    truth = torch.tensor([used_fams.index(fams_eval[i]) for i in eval_keep])
    return float((pred == truth).float().mean())


def shuffled(labels: list[str], seed: int) -> list[str]:
    import torch

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(labels), generator=gen).tolist()
    return [labels[i] for i in perm]


def run_probe_phase(
    ctx: bench.RunContext,
    items: list[Item],
    feats: Any,                 # [n, L+1, 3, d] float32 (roles in ROLES order)
    split_lookup: dict[str, bool],
    n_depths: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    report: list[dict[str, Any]] = []
    families_all = sorted({it.family for it in items})

    # Scopes: the full relation problem, plus each swap group restricted to the
    # subjects its families share. Only the latter supports the entity/template
    # controlled relation-identity claim.
    scopes: list[tuple[str, list[int], list[str], str]] = []
    scopes.append(("all", list(range(len(items))), families_all, "not_entity_controlled"))
    for group in SWAP_GROUPS:
        fams = sorted({it.family for it in items if it.swap_group == group})
        if len(fams) < 2:
            continue
        subj_sets = [{it.subject for it in items if it.family == f} for f in fams]
        common = set.intersection(*subj_sets) if subj_sets else set()
        idx = [i for i, it in enumerate(items)
               if it.swap_group == group and it.subject in common]
        if len(idx) >= 3 * len(fams):
            scopes.append((group, idx, fams, "entity_and_template_controlled"))

    train_mask = torch.tensor([split_lookup[it.item_id] for it in items])

    for scope_name, idx_list, fams, control_scope in scopes:
        idx = torch.tensor(idx_list, dtype=torch.long)
        scope_train = idx[train_mask[idx]]
        scope_eval = idx[~train_mask[idx]]
        fams_train = [items[int(i)].family for i in scope_train]
        fams_eval = [items[int(i)].family for i in scope_eval]
        train_fams_present = set(fams_train)
        eval_fams_present = set(fams_eval)
        scope_status = "ok" if (len(train_fams_present) >= 2 and len(eval_fams_present) >= 2) else "insufficient_split"
        if set(fams) - train_fams_present or set(fams) - eval_fams_present:
            print(f"[lab12] scope {scope_name}: split missing train/eval families; "
                  "some rows will be NaN and the scope is not claim-ready")
        for role_i, role in enumerate(ROLES):
            for depth in range(n_depths):
                Xtr = feats[scope_train, depth, role_i, :]
                Xev = feats[scope_eval, depth, role_i, :]

                # Multiclass centroid accuracy + shuffled-label control.
                acc = centroid_accuracy(Xtr, fams_train, Xev, fams_eval, fams)
                report.append({"scope": scope_name, "control_scope": control_scope,
                               "role": role, "depth": depth,
                               "method": "centroid", "family": "multiclass",
                               "eval_kind": "real", "metric": "accuracy",
                               "value": rounded(acc), "n_train": len(scope_train),
                               "n_eval": len(scope_eval), "chance": rounded(1.0 / len(fams)),
                               "status": scope_status})
                ctrl_accs = []
                if len(scope_train) and len(scope_eval):
                    for s in range(N_SHUFFLES):
                        fams_shuf = shuffled(fams_train, seed * 1009 + depth * 31 + role_i * 7 + s)
                        ctrl_accs.append(centroid_accuracy(Xtr, fams_shuf, Xev, fams_eval, fams))
                report.append({"scope": scope_name, "control_scope": control_scope,
                               "role": role, "depth": depth,
                               "method": "centroid", "family": "multiclass",
                               "eval_kind": "shuffled", "metric": "accuracy",
                               "value": rounded(safe_fmean(ctrl_accs)),
                               "n_train": len(scope_train), "n_eval": len(scope_eval),
                               "chance": rounded(1.0 / len(fams)), "status": scope_status})

                # One-vs-rest mass-mean AUC per family.
                macro_real_aucs: list[float] = []
                for fam in fams:
                    is_fam_tr = torch.tensor([f == fam for f in fams_train], dtype=torch.bool)
                    is_fam_ev = [f == fam for f in fams_eval]
                    if not bool(is_fam_tr.any()) or not bool((~is_fam_tr).any()) or True not in is_fam_ev or False not in is_fam_ev:
                        continue
                    d = ovr_mass_mean_direction(Xtr, is_fam_tr)
                    d = d / d.norm().clamp_min(1e-9)
                    scores = (Xev @ d).tolist()
                    pos = [s for s, m in zip(scores, is_fam_ev) if m]
                    neg = [s for s, m in zip(scores, is_fam_ev) if not m]
                    auc = auc_from_scores(pos, neg)
                    macro_real_aucs.append(auc)
                    report.append({"scope": scope_name, "control_scope": control_scope,
                                   "role": role, "depth": depth,
                                   "method": "mass_mean_ovr", "family": fam,
                                   "eval_kind": "real", "metric": "auc",
                                   "value": rounded(auc), "n_train": len(scope_train),
                                   "n_eval": len(scope_eval), "chance": 0.5,
                                   "status": scope_status})
                if macro_real_aucs:
                    report.append({"scope": scope_name, "control_scope": control_scope,
                                   "role": role, "depth": depth,
                                   "method": "mass_mean_ovr", "family": "macro_mean",
                                   "eval_kind": "real", "metric": "auc",
                                   "value": rounded(safe_fmean(macro_real_aucs)),
                                   "n_train": len(scope_train), "n_eval": len(scope_eval),
                                   "chance": 0.5, "status": scope_status})

                # Random-direction AUC baseline, macro-averaged over families.
                gen = torch.Generator().manual_seed(seed * 7919 + depth * 101 + role_i)
                rnd_aucs = []
                for _ in range(N_RANDOM_DIRS):
                    d = torch.randn(Xtr.shape[1], generator=gen)
                    d = d / d.norm().clamp_min(1e-9)
                    per_fam = []
                    for fam in fams:
                        is_fam_tr = torch.tensor([f == fam for f in fams_train], dtype=torch.bool)
                        is_fam_ev = [f == fam for f in fams_eval]
                        if not bool(is_fam_tr.any()) or not bool((~is_fam_tr).any()) or True not in is_fam_ev or False not in is_fam_ev:
                            continue
                        tr_scores = Xtr @ d
                        sign = 1.0
                        if float(tr_scores[is_fam_tr].mean()) < float(tr_scores[~is_fam_tr].mean()):
                            sign = -1.0
                        scores = (sign * (Xev @ d)).tolist()
                        pos = [s for s, m in zip(scores, is_fam_ev) if m]
                        neg = [s for s, m in zip(scores, is_fam_ev) if not m]
                        per_fam.append(auc_from_scores(pos, neg))
                    if per_fam:
                        rnd_aucs.append(safe_fmean(per_fam))
                report.append({"scope": scope_name, "control_scope": control_scope,
                               "role": role, "depth": depth,
                               "method": "random_direction", "family": "macro_mean",
                               "eval_kind": "random", "metric": "auc",
                               "value": rounded(safe_fmean(rnd_aucs)),
                               "n_train": len(scope_train), "n_eval": len(scope_eval),
                               "chance": 0.5, "status": scope_status})

        # Logistic OvR at the readout position only (max-margin upper bound;
        # closed-form probes carry the full sweep to keep runtime tame).
        role_i = ROLES.index("final")
        for depth in range(n_depths):
            Xtr = feats[scope_train, depth, role_i, :]
            Xev = feats[scope_eval, depth, role_i, :]
            for fam in fams:
                y = torch.tensor([1 if f == fam else 0 for f in fams_train])
                is_fam_ev = [f == fam for f in fams_eval]
                if not bool((y == 1).any()) or not bool((y == 0).any()) or True not in is_fam_ev or False not in is_fam_ev:
                    continue
                probe = fit_logistic(Xtr, y)
                scores = logistic_scores(probe, Xev)
                pos = [s for s, m in zip(scores, is_fam_ev) if m]
                neg = [s for s, m in zip(scores, is_fam_ev) if not m]
                report.append({"scope": scope_name, "control_scope": control_scope,
                               "role": "final", "depth": depth,
                               "method": "logistic_ovr", "family": fam,
                               "eval_kind": "real", "metric": "auc",
                               "value": rounded(auc_from_scores(pos, neg)),
                               "n_train": len(scope_train), "n_eval": len(scope_eval),
                               "chance": 0.5, "status": scope_status})

    scope_info = {name: {"n_items": len(idx), "families": fams, "control_scope": control_scope}
                  for name, idx, fams, control_scope in scopes}
    return report, scope_info


# ---------------------------------------------------------------------------
# Direction geometry
# ---------------------------------------------------------------------------


def _probe_value(report: list[dict[str, Any]], scope: str, role: str,
                 depth: int, eval_kind: str) -> float:
    vals = [float(r["value"]) for r in report
            if r["scope"] == scope and r["role"] == role and r["depth"] == depth
            and r["method"] == "centroid" and r["eval_kind"] == eval_kind
            and isinstance(r.get("value"), (int, float))]
    return vals[0] if vals else float("nan")


def selectivity_by_depth(report: list[dict[str, Any]], n_depths: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        for depth in range(n_depths):
            group_vals: list[float] = []
            for group in SWAP_GROUPS:
                real = _probe_value(report, group, role, depth, "real")
                ctrl = _probe_value(report, group, role, depth, "shuffled")
                if math.isfinite(real) and math.isfinite(ctrl):
                    group_vals.append(real - ctrl)
                    rows.append({"role": role, "scope": group, "depth": depth,
                                 "real_accuracy": rounded(real), "shuffled_accuracy": rounded(ctrl),
                                 "selectivity": rounded(real - ctrl), "aggregation": "group"})
            rows.append({"role": role, "scope": "swap_group_macro", "depth": depth,
                         "real_accuracy": "", "shuffled_accuracy": "",
                         "selectivity": rounded(safe_fmean(group_vals, default=float("nan"))),
                         "aggregation": "macro"})
    return rows


def best_depths_from_report(report: list[dict[str, Any]], n_depths: int) -> dict[str, int]:
    """Pick a depth separately for each role using entity/template-controlled
    selectivity. Depth 0 is excluded for subject/final summaries where possible;
    relword depth 0 is kept as a calibration trap, not as a discovery."""
    sel_rows = selectivity_by_depth(report, n_depths)

    def score(role: str, depth: int) -> float:
        vals = [float(r["selectivity"]) for r in sel_rows
                if r["role"] == role and r["scope"] == "swap_group_macro" and r["depth"] == depth
                and isinstance(r.get("selectivity"), (int, float)) and math.isfinite(float(r["selectivity"]))]
        return vals[0] if vals else float("-inf")

    depths: dict[str, int] = {}
    for role in ROLES:
        candidates = list(range(n_depths))
        if role in ("subject", "final") and n_depths > 2:
            candidates = list(range(1, n_depths - 1))
        depths[role] = max(candidates, key=lambda d: score(role, d)) if candidates else 0
    return depths


def build_relation_directions(
    items: list[Item], feats: Any, split_lookup: dict[str, bool],
    depth_by_role: dict[str, int], families: list[str],
) -> dict[str, dict[str, Any]]:
    """Per-family OvR mass-mean directions using train rows only. Subject and
    final roles get their own selected depths so the saved handle says where
    each direction actually came from."""
    import torch

    train_idx = [i for i, it in enumerate(items) if split_lookup[it.item_id]]
    fams_train = [items[i].family for i in train_idx]
    out: dict[str, dict[str, Any]] = {}
    for role in ("subject", "final"):
        depth = depth_by_role[role]
        role_i = ROLES.index(role)
        X = feats[torch.tensor(train_idx), depth, role_i, :]
        dirs: dict[str, Any] = {}
        for fam in families:
            is_fam = torch.tensor([f == fam for f in fams_train], dtype=torch.bool)
            if bool(is_fam.any()) and bool((~is_fam).any()):
                d = ovr_mass_mean_direction(X, is_fam)
                dirs[fam] = d / d.norm().clamp_min(1e-9)
        out[role] = dirs
    return out


def cosine_matrix(dirs: dict[str, Any], families: list[str]) -> list[list[float]]:
    import torch

    mat = []
    for fa in families:
        row = []
        for fb in families:
            if fa in dirs and fb in dirs:
                a, b = dirs[fa], dirs[fb]
                row.append(float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-9)))
            else:
                row.append(float("nan"))
        mat.append(row)
    return mat


# ---------------------------------------------------------------------------
# Patching phase (CAUSAL)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PatchPair:
    """One aligned clean/corrupt pair. kind is subject_swap or relation_swap."""

    kind: str
    family_clean: str
    family_corrupt: str
    clean: Item
    corrupt: Item
    swap_pos: int            # the single position where the prompts differ
    target_id: int           # clean answer token
    distractor_id: int       # corrupt answer token
    clean_diff: float = 0.0
    corrupt_diff: float = 0.0


def aligned_at_one_position(a: Item, b: Item) -> int:
    """Return the single differing position, or -1 if not exactly one."""
    if len(a.input_ids) != len(b.input_ids):
        return -1
    diff = [p for p in range(len(a.input_ids)) if a.input_ids[p] != b.input_ids[p]]
    return diff[0] if len(diff) == 1 else -1


def build_subject_swap_pairs(items: list[Item], max_pairs: int) -> list[PatchPair]:
    """Cyclic clean/corrupt pairing inside each family (Lab 5 style)."""
    pairs: list[PatchPair] = []
    by_family: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_family[it.family].append(it)
    for family in sorted(by_family):
        fam_items = by_family[family]
        made = 0
        for i, clean in enumerate(fam_items):
            if made >= max_pairs:
                break
            for step in range(1, len(fam_items)):
                corrupt = fam_items[(i + step) % len(fam_items)]
                if corrupt.target_id == clean.target_id:
                    continue
                pos = aligned_at_one_position(clean, corrupt)
                if pos != clean.subject_pos or pos != corrupt.subject_pos:
                    continue
                pairs.append(PatchPair(
                    kind="subject_swap", family_clean=family, family_corrupt=family,
                    clean=clean, corrupt=corrupt, swap_pos=pos,
                    target_id=clean.target_id, distractor_id=corrupt.target_id,
                ))
                made += 1
                break
    return pairs


def build_relation_swap_pairs(items: list[Item], max_subjects: int) -> list[PatchPair]:
    """Same subject, different relation, inside each swap group. The prompts
    differ at exactly the relation-word token; the clean answer and the
    corrupt answer are the two relations' answers for that subject."""
    pairs: list[PatchPair] = []
    by_group: dict[str, dict[str, dict[str, Item]]] = defaultdict(lambda: defaultdict(dict))
    for it in items:
        if it.swap_group:
            by_group[it.swap_group][it.family][it.subject] = it
    for group in sorted(by_group):
        fams = sorted(by_group[group])
        for fa in fams:
            for fb in fams:
                if fa == fb:
                    continue
                common = sorted(set(by_group[group][fa]) & set(by_group[group][fb]))
                made = 0
                for subject in common:
                    if made >= max_subjects:
                        break
                    clean, corrupt = by_group[group][fa][subject], by_group[group][fb][subject]
                    if clean.target_id == corrupt.target_id:
                        continue
                    pos = aligned_at_one_position(clean, corrupt)
                    if pos != clean.relword_pos or pos != corrupt.relword_pos:
                        continue
                    pairs.append(PatchPair(
                        kind="relation_swap", family_clean=fa, family_corrupt=fb,
                        clean=clean, corrupt=corrupt, swap_pos=pos,
                        target_id=clean.target_id, distractor_id=corrupt.target_id,
                    ))
                    made += 1
    return pairs


def pair_logit_diff(logits: Any, pair: PatchPair) -> float:
    return float(logits[pair.target_id] - logits[pair.distractor_id])


def gate_patch_pairs(
    bundle: bench.ModelBundle, pairs: list[PatchPair], captures: dict[str, Any]
) -> tuple[list[PatchPair], list[dict[str, Any]]]:
    kept: list[PatchPair] = []
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        clean_cap = captures[pair.clean.item_id]
        corrupt_cap = captures[pair.corrupt.item_id]
        pair.clean_diff = pair_logit_diff(clean_cap.final_logits_last, pair)
        pair.corrupt_diff = pair_logit_diff(corrupt_cap.final_logits_last, pair)
        ok = pair.clean_diff > CLEAN_MARGIN and pair.corrupt_diff < -CORRUPT_MARGIN
        rows.append({
            "kind": pair.kind, "family_clean": pair.family_clean,
            "family_corrupt": pair.family_corrupt,
            "clean_item": pair.clean.item_id, "corrupt_item": pair.corrupt.item_id,
            "clean_diff": rounded(pair.clean_diff),
            "corrupt_diff": rounded(pair.corrupt_diff),
            "kept": ok,
            "drop_reason": "" if ok else
            (f"clean_diff {pair.clean_diff:.2f} <= {CLEAN_MARGIN}"
             if pair.clean_diff <= CLEAN_MARGIN
             else f"corrupt_diff {pair.corrupt_diff:.2f} >= -{CORRUPT_MARGIN}"),
        })
        if ok:
            kept.append(pair)
    return kept, rows


def recovery(patched_diff: float, pair: PatchPair) -> float:
    denom = pair.clean_diff - pair.corrupt_diff
    if abs(denom) < 1e-9:
        raise ValueError(f"tiny recovery denominator for {pair.clean.item_id}")
    return (patched_diff - pair.corrupt_diff) / denom


def patch_positions_for(pair: PatchPair, grid_roles: list[str]) -> list[tuple[str, int]]:
    """Map --patch-grid role names to positions on this pair's corrupt prompt.
    Positions other than the swap position hold IDENTICAL tokens in clean and
    corrupt, so patching them measures migrated information, not token
    substitution — the handout leans on this distinction."""
    out: list[tuple[str, int]] = []
    it = pair.corrupt
    for role in grid_roles:
        if role == "subject":
            out.append(("subject", it.subject_pos))
        elif role == "relation":
            out.append(("relation", it.relword_pos))
        elif role == "last":
            out.append(("last", len(it.input_ids) - 1))
        else:
            raise ValueError(f"--patch-grid role {role!r} not in subject|relation|last")
    return out


def control_position(pair: PatchPair) -> int:
    """A position whose token is identical in clean and corrupt and which is
    neither the subject nor the relation word nor the final token: the
    wrong-position control. Falls back to position 0."""
    it = pair.corrupt
    banned = {it.subject_pos, it.relword_pos, len(it.input_ids) - 1, pair.swap_pos}
    for p in range(len(it.input_ids)):
        if p not in banned and it.input_ids[p] == pair.clean.input_ids[p]:
            return p
    return 0


def run_patch_phase(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    pairs: list[PatchPair],
    captures: dict[str, Any],
    grid_roles: list[str],
    n_depths: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    control_done: set[tuple[str, str, str]] = set()
    total = len(pairs)
    for pi, pair in enumerate(pairs):
        clean_cap = captures[pair.clean.item_id]
        sites = patch_positions_for(pair, grid_roles)
        ctrl_key = (pair.kind, pair.family_clean, pair.family_corrupt)
        run_controls = ctrl_key not in control_done
        if run_controls:
            control_done.add(ctrl_key)
            sites = sites + [("wrong_position", control_position(pair))]
        for depth in range(n_depths):
            for role, pos in sites:
                logits = bench.run_with_residual_patch(
                    bundle, pair.corrupt.prompt, depth, pos,
                    clean_cap.streams[depth, pos],
                )
                rows.append({
                    "kind": pair.kind, "family_clean": pair.family_clean,
                    "family_corrupt": pair.family_corrupt,
                    "clean_item": pair.clean.item_id,
                    "corrupt_item": pair.corrupt.item_id,
                    "depth": depth, "patch_role": role, "position": pos,
                    "clean_diff": rounded(pair.clean_diff),
                    "corrupt_diff": rounded(pair.corrupt_diff),
                    "patched_diff": rounded(pair_logit_diff(logits, pair)),
                    "recovery": rounded(recovery(pair_logit_diff(logits, pair), pair)),
                })
        if (pi + 1) % max(1, total // 5) == 0:
            print(f"[lab12] patched {pi + 1}/{total} pairs")
    return rows


def mismatched_vector_controls(
    bundle: bench.ModelBundle,
    pairs: list[PatchPair],
    captures: dict[str, Any],
    items: list[Item],
    n_depths: int,
) -> list[dict[str, Any]]:
    """Patch the corrupt prompt's swapped position with a vector from an
    unrelated family. This controls for margin movement caused by smashing the
    corrupt evidence rather than restoring clean relation/subject content."""
    rows: list[dict[str, Any]] = []
    by_family: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_family[it.family].append(it)
    families = sorted(by_family)
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        key = (pair.kind, pair.family_clean, pair.family_corrupt)
        if key in seen:
            continue
        seen.add(key)
        forbidden = {pair.family_clean, pair.family_corrupt}
        donor_family = next((f for f in families if f not in forbidden), None)
        if donor_family is None:
            continue
        donor = next((it for it in by_family[donor_family] if it.item_id in captures), None)
        if donor is None:
            continue
        donor_cap = captures[donor.item_id]
        donor_pos = donor.subject_pos if pair.kind == "subject_swap" else donor.relword_pos
        patch_role = "subject" if pair.kind == "subject_swap" else "relation"
        for depth in range(n_depths):
            logits = bench.run_with_residual_patch(
                bundle, pair.corrupt.prompt, depth, pair.swap_pos,
                donor_cap.streams[depth, donor_pos],
            )
            patched = pair_logit_diff(logits, pair)
            rows.append({
                "kind": "mismatched_control", "source_kind": pair.kind,
                "family_clean": pair.family_clean,
                "family_corrupt": pair.family_corrupt,
                "donor_family": donor_family,
                "clean_item": donor.item_id, "corrupt_item": pair.corrupt.item_id,
                "depth": depth, "patch_role": patch_role, "position": pair.swap_pos,
                "clean_diff": rounded(pair.clean_diff),
                "corrupt_diff": rounded(pair.corrupt_diff),
                "patched_diff": rounded(patched),
                "recovery": rounded(recovery(patched, pair)),
            })
    return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def mean_recovery(rows: list[dict[str, Any]], **filters: Any) -> float:
    vals = [float(r["recovery"]) for r in rows
            if all(r.get(k) == v for k, v in filters.items())]
    return safe_fmean(vals)


def family_depth_curve(rows: list[dict[str, Any]], family: str, patch_role: str,
                       n_depths: int, kind: str = "subject_swap") -> list[float]:
    return [mean_recovery(rows, kind=kind, family_clean=family,
                          patch_role=patch_role, depth=d) for d in range(n_depths)]


def band_stats(curve: list[float], n_layers: int) -> dict[str, Any]:
    """Summaries over the NON-TRIVIAL depth band 1..n_layers-1.

    Depth 0 is token substitution (patching the embedding row swaps the word
    — recovery ~1.0 tells you the rails are aligned, nothing more), and depth
    n_layers is the final-norm input, where a non-final-position patch cannot
    reach the final logits at all. Both are kept in the raw tables as sanity
    rows but excluded from every causal summary, exactly as Lab 5 excludes
    them from its localization decision."""
    band = [(d, v) for d, v in enumerate(curve)
            if 1 <= d <= n_layers - 1 and math.isfinite(v)]
    if not band:
        return {"band_mean": None, "persistence_depth": None, "depth0_sanity": none_if_nan(curve[0])}
    persistence = max((d for d, v in band if v >= 0.5), default=None)
    return {
        "band_mean": rounded(statistics.fmean([v for _, v in band])),
        "persistence_depth": persistence,
        "depth0_sanity": none_if_nan(curve[0]),
    }


def none_if_nan(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return rounded(x)


def profile_correlation(a: list[float], b: list[float]) -> float:
    xs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(xs) < 3:
        return float("nan")
    ax = [x for x, _ in xs]
    by = [y for _, y in xs]
    ma, mb = statistics.fmean(ax), statistics.fmean(by)
    num = sum((x - ma) * (y - mb) for x, y in xs)
    da = math.sqrt(sum((x - ma) ** 2 for x in ax))
    db = math.sqrt(sum((y - mb) ** 2 for y in by))
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return num / (da * db)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_probe_by_layer(ctx: bench.RunContext, report: list[dict[str, Any]],
                        n_depths: int, best_depth: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, role in zip(axes, ROLES):
        def curve(scope: str, eval_kind: str) -> tuple[list[int], list[float]]:
            rows = sorted([r for r in report
                           if r["scope"] == scope and r["role"] == role
                           and r["method"] == "centroid" and r["eval_kind"] == eval_kind],
                          key=lambda r: r["depth"])
            return [r["depth"] for r in rows], [float(r["value"]) for r in rows]

        xs, ys = curve("all", "real")
        if xs:
            ax.plot(xs, ys, color="tab:gray", linewidth=2.2, label="12-way (entities differ)")
        colors = {"country_sem": "tab:red", "adj_morph": "tab:blue", "month_seq": "tab:green"}
        for group, color in colors.items():
            xs, ys = curve(group, "real")
            if xs:
                ax.plot(xs, ys, color=color, linewidth=2.2, label=f"{group} (entity-controlled)")
            xs, ys = curve(group, "shuffled")
            if xs:
                ax.plot(xs, ys, color=color, linewidth=1.1, linestyle=":", alpha=0.8)
        ax.axvline(best_depth, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title(f"role: {role}")
        ax.set_xlabel("residual-stream depth")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("centroid accuracy (dotted = shuffled-label control)")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Relation identity by depth and token role; swap-group curves carry the controlled claim")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "relation_probe_by_layer.png",
                      "Centroid relation-identity accuracy by depth/role with shuffled controls; "
                      "swap-group scopes hold entities and template fixed.")


def plot_patch_heatmap(ctx: bench.RunContext, patch_rows: list[dict[str, Any]],
                       families: list[str], grid_roles: list[str], n_depths: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    roles = [r for r in ("subject", "relation", "last") if r in grid_roles]
    fig, axes = plt.subplots(1, len(roles), figsize=(4.6 * len(roles), 0.42 * len(families) + 2.4),
                             squeeze=False)
    im = None
    for ax, role in zip(axes[0], roles):
        grid = np.full((len(families), n_depths), np.nan)
        for i, fam in enumerate(families):
            for d in range(n_depths):
                grid[i, d] = mean_recovery(patch_rows, kind="subject_swap",
                                           family_clean=fam, patch_role=role, depth=d)
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families, fontsize=7)
        ax.set_xlabel("stream depth")
        ax.set_title(f"patched at {role}")
    if im is not None:
        fig.colorbar(im, ax=list(axes[0]), fraction=0.025, label="mean recovery")
    fig.suptitle("Subject-swap patching recovery by relation family (per-family causal localization profile)")
    bench.save_figure(ctx, fig, "relation_patch_heatmap.png",
                      "Mean interchange-patch recovery by family x depth at each patched role.")


def plot_relation_swap_curves(ctx: bench.RunContext, patch_rows: list[dict[str, Any]],
                              n_depths: int) -> None:
    fig, ax = bench.new_figure(figsize=(9.6, 5.2))
    pairs = sorted({(r["family_clean"], r["family_corrupt"]) for r in patch_rows
                    if r["kind"] == "relation_swap"})
    for fa, fb in pairs:
        ys = [mean_recovery(patch_rows, kind="relation_swap", family_clean=fa,
                            family_corrupt=fb, patch_role="relation", depth=d)
              for d in range(n_depths)]
        ax.plot(range(n_depths), ys, linewidth=1.8, label=f"{fa} → {fb}")
    ctrl = [mean_recovery(patch_rows, kind="relation_swap", patch_role="wrong_position",
                          depth=d) for d in range(n_depths)]
    if any(math.isfinite(v) for v in ctrl):
        ax.plot(range(n_depths), ctrl, color="black", linestyle=":", linewidth=1.6,
                label="wrong-position control")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axhline(1.0, color="black", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("stream depth")
    ax.set_ylabel("mean recovery of the relation-flip margin")
    ax.set_title("Relation-swap patching at the relation-word token (same subject, relation changed)")
    ax.legend(fontsize=7, ncol=2)
    bench.save_figure(ctx, fig, "relation_swap_recovery.png",
                      "Recovery curves for relation-swap pairs patched at the relation token, with wrong-position control.")


def plot_cosines(ctx: bench.RunContext, mats: dict[str, list[list[float]]],
                 families: list[str], depth_by_role: dict[str, int] | int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    roles = sorted(mats)
    fig, axes = plt.subplots(1, len(roles), figsize=(6.4 * len(roles), 5.4), squeeze=False)
    im = None
    for ax, role in zip(axes[0], roles):
        grid = np.array(mats[role])
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families, fontsize=7)
        depth = depth_by_role.get(role, "?") if isinstance(depth_by_role, dict) else depth_by_role
        ax.set_title(f"OvR mass-mean directions @ depth {depth}, role {role}")
        for i in range(len(families)):
            for j in range(len(families)):
                if np.isfinite(grid[i, j]) and i != j:
                    ax.annotate(f"{grid[i, j]:.2f}", (j, i), ha="center", va="center", fontsize=5.5)
    if im is not None:
        fig.colorbar(im, ax=list(axes[0]), fraction=0.03, label="cosine")
    fig.suptitle("Relation-direction cosine atlas (block structure = candidate shared geometry; verify against entity groups)")
    bench.save_figure(ctx, fig, "relation_direction_cosines.png",
                      "Pairwise cosines among per-family relation directions at the selected role-specific depths.")




def plot_probe_selectivity(ctx: bench.RunContext, sel_rows: list[dict[str, Any]],
                           depth_by_role: dict[str, int], n_depths: int) -> None:
    fig, ax = bench.new_figure(figsize=(9.4, 5.0))
    for role in ROLES:
        ys = []
        for d in range(n_depths):
            vals = [float(r["selectivity"]) for r in sel_rows
                    if r["role"] == role and r["scope"] == "swap_group_macro" and r["depth"] == d
                    and isinstance(r.get("selectivity"), (int, float))]
            ys.append(vals[0] if vals else float("nan"))
        ax.plot(range(n_depths), ys, linewidth=2.0, label=f"{role} macro selectivity")
        ax.axvline(depth_by_role[role], linewidth=0.8, alpha=0.35)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("residual-stream depth")
    ax.set_ylabel("centroid accuracy minus shuffled control")
    ax.set_title("Entity/template-controlled relation selectivity by role")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "relation_probe_selectivity.png",
                      "Swap-group macro selectivity curves used to choose saved direction depths.")


def plot_transfer_matrix(ctx: bench.RunContext, transfer_rows: list[dict[str, Any]],
                         families: list[str]) -> None:
    import numpy as np

    fig, ax = bench.new_figure(figsize=(8.2, 7.2))
    grid = np.full((len(families), len(families)), np.nan)
    kind_grid: dict[tuple[int, int], str] = {}
    for r in transfer_rows:
        fa, fb = r["family_clean"], r["family_corrupt"]
        if fa not in families or fb not in families:
            continue
        i, j = families.index(fa), families.index(fb)
        kind_grid[(i, j)] = r["kind"]
        val = r.get("band_mean_recovery")
        if isinstance(val, (int, float)):
            grid[i, j] = float(val)
    im = ax.imshow(grid, aspect="auto", vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=7)
    for i in range(len(families)):
        for j in range(len(families)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6)
            elif kind_grid.get((i, j)) == "no_aligned_pair":
                ax.text(j, i, "·", ha="center", va="center", fontsize=8)
    ax.set_title("Relation transfer matrix: diagonal subject-swap, within-swap-group relation-swap")
    fig.colorbar(im, ax=ax, fraction=0.035, label="band mean recovery")
    bench.save_figure(ctx, fig, "relation_transfer_matrix.png",
                      "Visual version of relation_transfer_matrix.csv; empty cells lack token-aligned controls.")


def plot_patch_control_gaps(ctx: bench.RunContext, metrics: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    labels = ["subject matched", "relation matched", "wrong position", "mismatched vector"]
    vals = [
        metrics["patching"]["subject_swap"].get("band_mean"),
        metrics["patching"]["relation_swap"].get("band_mean"),
        metrics["patching"].get("wrong_position_mean_recovery"),
        metrics["patching"].get("mismatched_vector_mean_recovery"),
    ]
    numeric = [float(v) if isinstance(v, (int, float)) else float("nan") for v in vals]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(range(len(labels)), [0.0 if not math.isfinite(v) else v for v in numeric])
    ax.axhline(0.0, linewidth=0.8)
    ax.axhline(1.0, linewidth=0.8, alpha=0.35)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mean recovery (non-trivial depth band)")
    ax.set_title("Matched patching must beat controls before causal language earns its badge")
    for i, v in enumerate(numeric):
        if math.isfinite(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        else:
            ax.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "patch_control_gaps.png",
                      "Matched subject/relation patch recovery compared with wrong-position and mismatched-vector controls.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    grid_roles = [r.strip() for r in str(getattr(args, "patch_grid", "subject,relation,last")).split(",") if r.strip()]
    if not grid_roles:
        raise ValueError("--patch-grid must include at least one of subject,relation,last")
    for r in grid_roles:
        if r not in ("subject", "relation", "last"):
            raise ValueError(f"--patch-grid accepts subject,relation,last; got {r!r}")

    items, data_info = load_items(args)
    set_name = data_info["relation_set"]
    print(f"[lab12] {data_info['n_items']} items, {len(data_info['families'])} families, "
          f"relation-set {set_name!r} (per-family cap {data_info['per_family_cap'] or 'none'})")
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic",
                          "Frozen relation CSV hash, manifest check, filters, and counts.")

    # Instrument checks before science (hook indexing, lens, patch no-op).
    bench.run_hook_parity_check(ctx, bundle, items[0].prompt)
    first_capture = bench.run_with_residual_cache(bundle, items[0].prompt)
    bench.run_lens_self_check(ctx, bundle, first_capture)
    bench.run_patch_noop_check(ctx, bundle, items[0].prompt)

    items = tokenization_gate(ctx, bundle, items)
    families = sorted({it.family for it in items})
    n_depths = bundle.anatomy.n_layers + 1

    # Family/data card after tokenizer gate, not before.
    family_rows: list[dict[str, Any]] = []
    for fam in families:
        fam_items = [it for it in items if it.family == fam]
        family_rows.append({
            "family": fam,
            "n_items": len(fam_items),
            "swap_group": fam_items[0].swap_group,
            "entity_group": fam_items[0].entity_group,
            "example_prompt": fam_items[0].prompt,
            "target_examples": "; ".join(sorted({it.target.strip() for it in fam_items})[:5]),
        })
    family_path = ctx.path("tables", "relation_family_manifest.csv")
    bench.write_csv_with_context(ctx, family_path, family_rows)
    ctx.register_artifact(family_path, "table", "Relation family counts and example prompts after tokenization gating.")

    # ----- forward passes: cache role-position streams + final logits -------
    captures: dict[str, Any] = {}
    feats_list = []
    margin_rows: list[dict[str, Any]] = []
    t_report = max(1, len(items) // 5)
    for i, it in enumerate(items):
        cap = bench.run_with_residual_cache(bundle, it.prompt)
        if cap.input_ids != it.input_ids:
            raise RuntimeError(f"{it.item_id}: capture tokenization differs from gate tokenization")
        captures[it.item_id] = cap
        positions = [role_position(it, role) for role in ROLES]
        feats_list.append(cap.streams[:, positions, :])      # [L+1, 3, d]
        it.margin_hard = float(cap.final_logits_last[it.target_id] - cap.final_logits_last[it.hard_id])
        it.margin_easy = float(cap.final_logits_last[it.target_id] - cap.final_logits_last[it.easy_id])
        top_id = int(cap.final_logits_last.argmax())
        margin_rows.append({
            "item_id": it.item_id, "family": it.family, "swap_group": it.swap_group,
            "subject": it.subject, "target": it.target.strip(),
            "hard_distractor": it.hard_distractor.strip(), "easy_distractor": it.easy_distractor.strip(),
            "margin_hard": rounded(it.margin_hard), "margin_easy": rounded(it.margin_easy),
            "knows_hard": it.margin_hard > 0, "knows_easy": it.margin_easy > 0,
            "passes_patch_clean_gate_hard": it.margin_hard > CLEAN_MARGIN,
            "top_token_id": top_id, "top_token_text": bundle.tokenizer.decode([top_id]),
            "near_tie": "near_tie=1" in it.note,
        })
        if (i + 1) % t_report == 0:
            print(f"[lab12] cached {i + 1}/{len(items)} items")

    feats_raw = torch.stack(feats_list)                       # [n, L+1, 3, d]
    row_norms = feats_raw.norm(dim=-1)                        # [n, L+1, 3]
    feats = feats_raw / row_norms[..., None].clamp_min(1e-9) if NORMALIZE_ROWS else feats_raw

    norm_rows = []
    for role_i, role in enumerate(ROLES):
        for depth in range(n_depths):
            vals = row_norms[:, depth, role_i].tolist()
            norm_rows.append({"role": role, "depth": depth, "mean_norm": rounded(safe_fmean(vals)),
                              "min_norm": rounded(min(vals)), "max_norm": rounded(max(vals))})
    norm_path = ctx.path("diagnostics", "activation_norms_by_role_depth.csv")
    bench.write_csv_with_context(ctx, norm_path, norm_rows)
    ctx.register_artifact(norm_path, "diagnostic", "Residual-stream norm audit before row normalization.")

    margin_path = ctx.path("tables", "margin_report.csv")
    bench.write_csv_with_context(ctx, margin_path, margin_rows)
    ctx.register_artifact(margin_path, "table",
                          "Dual-distractor logit margins per item (hard = same class, easy = cross class).")

    fam_margin_rows = []
    for fam in families:
        fam_rows = [r for r in margin_rows if r["family"] == fam]
        fam_margin_rows.append({
            "family": fam, "swap_group": next(it.swap_group for it in items if it.family == fam),
            "n": len(fam_rows),
            "mean_margin_hard": rounded(safe_fmean([float(r["margin_hard"]) for r in fam_rows])),
            "mean_margin_easy": rounded(safe_fmean([float(r["margin_easy"]) for r in fam_rows])),
            "frac_knows_hard": rounded(statistics.fmean([1.0 if r["knows_hard"] else 0.0 for r in fam_rows])),
            "frac_knows_easy": rounded(statistics.fmean([1.0 if r["knows_easy"] else 0.0 for r in fam_rows])),
            "frac_passes_clean_gate_hard": rounded(statistics.fmean([1.0 if r["passes_patch_clean_gate_hard"] else 0.0 for r in fam_rows])),
        })
    fam_margin_path = ctx.path("tables", "margin_by_family.csv")
    bench.write_csv_with_context(ctx, fam_margin_path, fam_margin_rows)
    ctx.register_artifact(fam_margin_path, "table", "Per-family margin stability aggregates.")

    # ----- probes ------------------------------------------------------------
    split_lookup = make_subject_split(items, args.seed)
    split_rows = [{"item_id": it.item_id, "family": it.family, "swap_group": it.swap_group,
                   "subject": it.subject, "split": "train" if split_lookup[it.item_id] else "eval"}
                  for it in items]
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows)
    ctx.register_artifact(split_path, "diagnostic",
                          "Subject-grouped train/eval split (entity leakage hygiene).")
    split_balance: list[dict[str, Any]] = []
    for fam in families:
        fam_items = [it for it in items if it.family == fam]
        split_balance.append({"family": fam, "swap_group": fam_items[0].swap_group,
                              "n_train": sum(1 for it in fam_items if split_lookup[it.item_id]),
                              "n_eval": sum(1 for it in fam_items if not split_lookup[it.item_id])})
    split_balance_path = ctx.path("diagnostics", "split_balance.csv")
    bench.write_csv_with_context(ctx, split_balance_path, split_balance)
    ctx.register_artifact(split_balance_path, "diagnostic", "Train/eval counts per family after subject-grouped split.")

    print(f"[lab12] probing {n_depths} depths x {len(ROLES)} roles "
          f"(centroid + OvR mass-mean everywhere, logistic at final)")
    probe_report, scope_info = run_probe_phase(ctx, items, feats, split_lookup, n_depths, args.seed)
    probe_path = ctx.path("tables", "probe_report.csv")
    bench.write_csv_with_context(ctx, probe_path, probe_report)
    ctx.register_artifact(probe_path, "table",
                          "Every probe evaluation: scope, role, depth, method, controls.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, probe_report)
    ctx.register_artifact(results_path, "results", "Alias of probe_report.csv for the standard run contract.")

    sel_rows = selectivity_by_depth(probe_report, n_depths)
    sel_path = ctx.path("tables", "probe_selectivity_by_depth.csv")
    bench.write_csv_with_context(ctx, sel_path, sel_rows)
    ctx.register_artifact(sel_path, "table", "Centroid accuracy minus shuffled control by depth, role, and swap-group scope.")
    depth_by_role = best_depths_from_report(probe_report, n_depths)
    best_depth = depth_by_role["final"]
    depth_decision_path = ctx.path("diagnostics", "relation_depth_selection.json")
    bench.write_json(depth_decision_path, {
        "depth_by_role": depth_by_role,
        "selection_rule": "max swap-group macro centroid selectivity; subject/final exclude depth 0 and final-norm depth when possible",
        "stream_convention": "streams[k] is the pre-norm residual after k blocks; k=0 embedding output",
    })
    ctx.register_artifact(depth_decision_path, "diagnostic", "Why the saved relation-direction depths were chosen.")
    print(f"[lab12] selected relation-direction depths: {depth_by_role}")

    # ----- relation directions + cosine atlas --------------------------------
    dirs_by_role = build_relation_directions(items, feats, split_lookup, depth_by_role, families)
    cos_mats = {role: cosine_matrix(dirs, families) for role, dirs in dirs_by_role.items()}
    swap_group_of = {it.family: it.swap_group for it in items}
    cos_rows = []
    for role, mat in cos_mats.items():
        for i, fa in enumerate(families):
            for j, fb in enumerate(families):
                if i < j:
                    cos_rows.append({"role": role, "depth": depth_by_role[role],
                                     "family_a": fa, "family_b": fb,
                                     "same_swap_group": bool(swap_group_of.get(fa)) and swap_group_of.get(fa) == swap_group_of.get(fb),
                                     "cosine": rounded(mat[i][j])})
    cos_path = ctx.path("tables", "relation_cosine_matrix.csv")
    bench.write_csv_with_context(ctx, cos_path, cos_rows)
    ctx.register_artifact(cos_path, "table", "Pairwise cosines among OvR relation directions.")

    state_payload: dict[str, Any] = {
        "depth_by_role": depth_by_role,
        "depth_convention": "bench streams[k]: 0 = embeddings, k = residual after block k",
        "roles": {role: {fam: d for fam, d in dirs.items()} for role, dirs in dirs_by_role.items()},
        "normalization": "row_unit_norm" if NORMALIZE_ROWS else "raw_streams",
        "method": "one-vs-rest mass-mean (family mean minus rest mean), train split only",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "families": families,
        "evidence": "DECODE artifact. Saving a direction is not evidence the model uses it.",
    }
    state_path = ctx.path("state", "relation_directions.pt")
    torch.save(state_payload, state_path)
    ctx.register_artifact(state_path, "tensor",
                          "Per-family OvR relation directions at selected depths (subject and final roles).")
    meta_path = ctx.path("state", "relation_directions_metadata.json")
    bench.write_json(meta_path, {k: v for k, v in state_payload.items() if k != "roles"})
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for relation_directions.pt.")

    # ----- patching -----------------------------------------------------------
    max_pairs, max_swap_subjects = PATCH_BUDGETS.get(set_name, PATCH_BUDGETS["small"])
    subject_pairs = build_subject_swap_pairs(items, max_pairs)
    relation_pairs = build_relation_swap_pairs(items, max_swap_subjects)
    gated, gate_rows = gate_patch_pairs(bundle, subject_pairs + relation_pairs, captures)
    gate_path = ctx.path("diagnostics", "patch_pair_gate.csv")
    bench.write_csv_with_context(ctx, gate_path, gate_rows)
    ctx.register_artifact(gate_path, "diagnostic",
                          "Behavioral baseline gate for every candidate patch pair, with drop reasons.")
    n_subject = sum(1 for p in gated if p.kind == "subject_swap")
    n_relation = sum(1 for p in gated if p.kind == "relation_swap")
    print(f"[lab12] patch pairs after gate: {n_subject} subject-swap, {n_relation} relation-swap "
          f"(of {len(subject_pairs)}+{len(relation_pairs)} candidates)")

    patch_rows: list[dict[str, Any]] = []
    if gated:
        patch_rows = run_patch_phase(ctx, bundle, gated, captures, grid_roles, n_depths)
        patch_rows += mismatched_vector_controls(bundle, gated, captures, items, n_depths)
    patch_path = ctx.path("tables", "patch_report.csv")
    bench.write_csv_with_context(ctx, patch_path, patch_rows)
    ctx.register_artifact(patch_path, "table",
                          "Every interchange patch: pair, depth, patched role, recovery, controls.")

    # ----- transfer matrix ----------------------------------------------------
    n_layers = bundle.anatomy.n_layers
    transfer_rows: list[dict[str, Any]] = []
    stats_by_family: dict[str, dict[str, Any]] = {}
    for fam in families:
        curve = family_depth_curve(patch_rows, fam, "subject", n_depths)
        stats_by_family[fam] = band_stats(curve, n_layers)
    for fa in families:
        for fb in families:
            if fa == fb:
                s = stats_by_family[fa]
                transfer_rows.append({"family_clean": fa, "family_corrupt": fb,
                                      "kind": "subject_swap",
                                      "band_mean_recovery": s["band_mean"],
                                      "persistence_depth": s["persistence_depth"],
                                      "depth0_sanity": s["depth0_sanity"]})
            elif swap_group_of.get(fa) and swap_group_of.get(fa) == swap_group_of.get(fb):
                ys = [mean_recovery(patch_rows, kind="relation_swap", family_clean=fa,
                                    family_corrupt=fb, patch_role="relation", depth=d)
                      for d in range(n_depths)]
                s = band_stats(ys, n_layers)
                transfer_rows.append({"family_clean": fa, "family_corrupt": fb,
                                      "kind": "relation_swap",
                                      "band_mean_recovery": s["band_mean"],
                                      "persistence_depth": s["persistence_depth"],
                                      "depth0_sanity": s["depth0_sanity"]})
            else:
                transfer_rows.append({"family_clean": fa, "family_corrupt": fb,
                                      "kind": "no_aligned_pair",
                                      "band_mean_recovery": "",
                                      "persistence_depth": "", "depth0_sanity": ""})
    transfer_path = ctx.path("tables", "relation_transfer_matrix.csv")
    bench.write_csv_with_context(ctx, transfer_path, transfer_rows)
    ctx.register_artifact(transfer_path, "table",
                          "Within-relation diagonal vs cross-relation swap-group patch recovery; unaligned cells empty by design.")

    # Cross-family localization-profile similarity (the shared-machinery HANDLE).
    profile_rows: list[dict[str, Any]] = []
    profile_corrs: list[float] = []
    fam_curves = {fam: family_depth_curve(patch_rows, fam, "subject", n_depths)[1:n_layers]
                  for fam in families}
    for i, fa in enumerate(families):
        for fb in families[i + 1:]:
            c = profile_correlation(fam_curves[fa], fam_curves[fb])
            profile_rows.append({"family_a": fa, "family_b": fb,
                                 "same_swap_group": bool(swap_group_of.get(fa)) and swap_group_of.get(fa) == swap_group_of.get(fb),
                                 "profile_correlation": rounded(c)})
            if math.isfinite(c):
                profile_corrs.append(c)
    profile_path = ctx.path("tables", "localization_profile_similarity.csv")
    bench.write_csv_with_context(ctx, profile_path, profile_rows)
    ctx.register_artifact(profile_path, "table", "Pairwise correlations among subject-swap localization profiles.")

    # ----- headline numbers ------------------------------------------------------
    def group_centroid(group: str, eval_kind: str, depth: int, role: str = "final") -> float:
        return _probe_value(probe_report, group, role, depth, eval_kind)

    group_stats = {}
    for group in SWAP_GROUPS:
        if group in scope_info:
            real = group_centroid(group, "real", depth_by_role["final"])
            ctrl = group_centroid(group, "shuffled", depth_by_role["final"])
            group_stats[group] = {"accuracy": rounded(real), "shuffled": rounded(ctrl),
                                  "selectivity": rounded(real - ctrl if math.isfinite(real) and math.isfinite(ctrl) else float("nan")),
                                  "chance": rounded(1.0 / len(scope_info[group]["families"])),
                                  "n_families": len(scope_info[group]["families"])}
    all_real = group_centroid("all", "real", depth_by_role["final"])
    all_ctrl = group_centroid("all", "shuffled", depth_by_role["final"])

    rel_swap_curve = [mean_recovery(patch_rows, kind="relation_swap",
                                    patch_role="relation", depth=d) for d in range(n_depths)]
    subj_swap_curve = [mean_recovery(patch_rows, kind="subject_swap",
                                     patch_role="subject", depth=d) for d in range(n_depths)]
    rel_swap_stats = band_stats(rel_swap_curve, n_layers)
    subj_swap_stats = band_stats(subj_swap_curve, n_layers)
    wrong_pos_vals = [float(r["recovery"]) for r in patch_rows
                      if r["patch_role"] == "wrong_position" and 1 <= int(r["depth"]) <= n_layers - 1]
    mismatch_vals = [float(r["recovery"]) for r in patch_rows
                     if r["kind"] == "mismatched_control" and 1 <= int(r["depth"]) <= n_layers - 1]

    same_group_cos = [float(r["cosine"]) for r in cos_rows if r["role"] == "final" and r["same_swap_group"]]
    diff_group_cos = [float(r["cosine"]) for r in cos_rows if r["role"] == "final" and not r["same_swap_group"]]

    within_selectivities = [float(s["selectivity"]) for s in group_stats.values()
                            if isinstance(s.get("selectivity"), (int, float)) and math.isfinite(float(s["selectivity"]))]
    max_within_selectivity = max(within_selectivities, default=float("nan"))
    wrong_mean = safe_fmean(wrong_pos_vals)
    mismatch_mean = safe_fmean(mismatch_vals)
    control_floor = max(v for v in (wrong_mean, mismatch_mean, 0.0) if math.isfinite(v))
    relation_gap = (rel_swap_stats["band_mean"] - control_floor
                    if isinstance(rel_swap_stats["band_mean"], (int, float)) else float("nan"))
    subject_gap = (subj_swap_stats["band_mean"] - control_floor
                   if isinstance(subj_swap_stats["band_mean"], (int, float)) else float("nan"))
    decode_verdict = "validated_selective" if math.isfinite(max_within_selectivity) and max_within_selectivity >= MIN_SELECTIVITY_FOR_DECODE_CLAIM else "not_validated_or_entity_confounded"
    causal_verdict = "validated_position_specific" if math.isfinite(relation_gap) and relation_gap >= MIN_PATCH_GAP_FOR_CAUSAL_CLAIM else "not_validated_by_controls"

    metrics = {
        "n_items": len(items),
        "families": families,
        "relation_set": set_name,
        "data_source": data_info["data_source"],
        "data_manifest_ok": data_info.get("manifest_ok"),
        "depth_by_role": depth_by_role,
        "best_depth": best_depth,
        "n_depths": n_depths,
        "normalization": "row_unit_norm" if NORMALIZE_ROWS else "raw_streams",
        "verdicts": {
            "decode": decode_verdict,
            "causal": causal_verdict,
            "max_within_group_selectivity": none_if_nan(max_within_selectivity),
            "relation_patch_specificity_gap": none_if_nan(relation_gap),
            "subject_patch_specificity_gap": none_if_nan(subject_gap),
            "claim_thresholds": {
                "min_selectivity_for_decode_claim": MIN_SELECTIVITY_FOR_DECODE_CLAIM,
                "min_patch_gap_for_causal_claim": MIN_PATCH_GAP_FOR_CAUSAL_CLAIM,
            },
        },
        "probe": {
            "all_relation_accuracy_at_best_final_depth": rounded(all_real),
            "all_relation_shuffled": rounded(all_ctrl),
            "within_group": group_stats,
        },
        "patching": {
            "n_subject_swap_pairs": n_subject,
            "n_relation_swap_pairs": n_relation,
            "band_convention": "all causal summaries cover depths 1..n_layers-1; depth 0 is token substitution and depth n_layers cannot reach the readout",
            "subject_swap": subj_swap_stats,
            "relation_swap": rel_swap_stats,
            "wrong_position_mean_recovery": none_if_nan(wrong_mean),
            "wrong_position_max_recovery": none_if_nan(max(wrong_pos_vals, default=float("nan"))),
            "mismatched_vector_mean_recovery": none_if_nan(mismatch_mean),
            "mismatched_vector_max_recovery": none_if_nan(max(mismatch_vals, default=float("nan"))),
            "specificity_gap_subject": none_if_nan(subject_gap),
            "specificity_gap_relation": none_if_nan(relation_gap),
            "per_family_subject_swap": stats_by_family,
        },
        "geometry": {
            "mean_profile_correlation": none_if_nan(safe_fmean(profile_corrs)),
            "cosine_same_group_mean_final": none_if_nan(safe_fmean(same_group_cos)),
            "cosine_diff_group_mean_final": none_if_nan(safe_fmean(diff_group_cos)),
        },
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 12 metrics and dynamic verdicts.")

    # ----- plots ----------------------------------------------------------------
    if not args.no_plots:
        plot_probe_by_layer(ctx, probe_report, n_depths, best_depth)
        plot_probe_selectivity(ctx, sel_rows, depth_by_role, n_depths)
        if patch_rows:
            plot_patch_heatmap(ctx, patch_rows, families, grid_roles, n_depths)
            plot_relation_swap_curves(ctx, patch_rows, n_depths)
            plot_transfer_matrix(ctx, transfer_rows, families)
            plot_patch_control_gaps(ctx, metrics)
        plot_cosines(ctx, cos_mats, families, depth_by_role)

    # ----- operationalization audit + method card -------------------------------
    write_operationalization_audit(ctx, metrics, scope_info)
    write_method_validation_card(ctx, bundle, metrics, data_info)

    # ----- claims + run summary -------------------------------------------------
    run_name = ctx.run_dir.name
    group_text = "; ".join(
        f"{g}: {s['accuracy']} vs shuffled {s['shuffled']} (chance {s['chance']})"
        for g, s in group_stats.items()
    ) or "no swap-group scope survived"
    if decode_verdict == "validated_selective":
        c1_text = (
            f"Relation identity is decodable from {bundle.anatomy.model_id}'s residual stream at "
            f"final-role depth {depth_by_role['final']} inside entity- and template-matched swap "
            f"groups — {group_text}. The claim excludes entity class and template syntax, but not "
            f"a relation-word token echo."
        )
    else:
        c1_text = (
            f"This run did NOT validate an entity/template-controlled relation-identity probe: the "
            f"largest within-swap-group selectivity was {none_if_nan(max_within_selectivity)} "
            f"at the final-role selection threshold {MIN_SELECTIVITY_FOR_DECODE_CLAIM}. Any high "
            f"uncontrolled all-relation accuracy should be treated as entity/template geometry until "
            f"the swap-group controls pass."
        )
    if causal_verdict == "validated_position_specific":
        c2_text = (
            f"Relation-swap interchange patching is position-specific on {bundle.anatomy.model_id}: "
            f"the relation-word residual recovers the clean-vs-corrupt answer margin with band mean "
            f"{rel_swap_stats['band_mean']} and persistence depth {rel_swap_stats['persistence_depth']} "
            f"of {n_layers}, beating the larger control mean by gap {none_if_nan(relation_gap)} "
            f"over {n_relation} gated relation-swap pairs. Subject-swap persistence is "
            f"{subj_swap_stats['persistence_depth']} with band mean {subj_swap_stats['band_mean']}."
        )
    else:
        c2_text = (
            f"This run did NOT earn a positive relation-swap causal claim: matched relation-token "
            f"band mean was {rel_swap_stats['band_mean']} and the gap over wrong-position/mismatched "
            f"controls was {none_if_nan(relation_gap)} (threshold {MIN_PATCH_GAP_FOR_CAUSAL_CLAIM}, "
            f"{n_relation} gated relation-swap pairs). The safe claim is a failed or inconclusive "
            f"intervention audit, not relation use."
        )
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE",
            "text": c1_text,
            "artifact": f"runs/{run_name}/tables/probe_report.csv",
            "falsifier": (
                "Within-group accuracy collapses to its shuffled control on a new entity roster, or a "
                "probe trained only on relation-word token embeddings matches it."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": c2_text,
            "artifact": f"runs/{run_name}/tables/relation_transfer_matrix.csv",
            "falsifier": (
                "Wrong-position or mismatched-vector controls recover comparably to matched patches, "
                "or recovery does not localize to relation-relevant positions on a held-out family."
            ),
        },
        {
            "id": f"{LAB_ID}-C3",
            "tag": "OBS",
            "text": (
                f"Cross-family similarity of causal localization profiles (mean pairwise correlation "
                f"{metrics['geometry']['mean_profile_correlation']}) and the direction-cosine atlas "
                f"(same-group mean {metrics['geometry']['cosine_same_group_mean_final']} vs cross-group "
                f"{metrics['geometry']['cosine_diff_group_mean_final']}) are handles on shared relation "
                f"machinery, not mechanism evidence. Profile similarity does not identify shared components."
            ),
            "artifact": f"runs/{run_name}/tables/localization_profile_similarity.csv",
            "falsifier": (
                "Component-level analysis (Lab 6 toolkit) shows disjoint heads/MLPs across families "
                "despite similar depth profiles."
            ),
        },
    ]
    if data_info.get("data_source") != "frozen_csv":
        for c in claims:
            c["text"] = "SMOKE-FALLBACK DATA ONLY, DO NOT LEDGER AS SCIENCE: " + c["text"]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    write_run_summary(ctx, bundle, metrics, claims, scope_info, data_info)
    print(f"[lab12] wrote run_summary.md, method_validation_card.md, operationalization_audit.md, and {len(claims)} drafted claims")


# ---------------------------------------------------------------------------
# Audit and summary writers
# ---------------------------------------------------------------------------




def write_method_validation_card(ctx: bench.RunContext, bundle: bench.ModelBundle,
                                 metrics: dict[str, Any], data_info: dict[str, Any]) -> None:
    verdicts = metrics["verdicts"]
    lines = [
        "# Lab 12 method validation card",
        "",
        "This card is the one-page answer to: *did the instrument survive its own controls?*",
        "",
        "| gate | verdict | evidence |",
        "|---|---|---|",
        f"| frozen data | `{data_info.get('data_source')}` / manifest ok `{data_info.get('manifest_ok')}` | `diagnostics/frozen_data_manifest.json` |",
        f"| tokenizer and role positions | see audit | `diagnostics/tokenization_audit.csv` |",
        f"| split hygiene | subject-grouped | `diagnostics/split_audit.csv`, `diagnostics/split_balance.csv` |",
        f"| probe selectivity | `{verdicts['decode']}` | max within-group selectivity `{verdicts['max_within_group_selectivity']}` |",
        f"| causal specificity | `{verdicts['causal']}` | relation patch gap `{verdicts['relation_patch_specificity_gap']}` |",
        f"| saved direction scope | model `{bundle.anatomy.model_id}`, depth by role `{metrics['depth_by_role']}` | `state/relation_directions_metadata.json` |",
        "",
        "## Claim posture",
        "",
    ]
    if verdicts["decode"] == "validated_selective":
        lines.append("- Probe result: claimable at `DECODE`, with the relation-word-token-echo caveat attached.")
    else:
        lines.append("- Probe result: not claimable as relation geometry. Treat uncontrolled accuracy as a confound candidate.")
    if verdicts["causal"] == "validated_position_specific":
        lines.append("- Patching result: claimable at narrow `CAUSAL` scope for token-aligned relation-swap pairs.")
    else:
        lines.append("- Patching result: not claimable as positive causal use; report the failed control audit.")
    lines += [
        "",
        "## The thing not learned",
        "",
        "No artifact in this lab identifies the head, MLP, SAE feature, or algorithm that computes a relation. The lab produces controlled handles for later labs, not a mechanism trophy.",
        "",
    ]
    path = ctx.path("method_validation_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "One-page verdict on whether Lab 12's probe/patch instruments survived their controls.")

def write_operationalization_audit(ctx: bench.RunContext, metrics: dict[str, Any],
                                   scope_info: dict[str, Any]) -> None:
    p = metrics["probe"]
    pa = metrics["patching"]
    geo = metrics["geometry"]
    verdicts = metrics["verdicts"]
    lines = [
        "# Lab 12 operationalization audit",
        "",
        "Favorite interpretation under attack: *\"relation families have shared,",
        "measurable internal geometry that the model actually uses.\"*",
        "",
        "## Verdict before story",
        "",
        f"- Probe gate: `{verdicts['decode']}` with max within-group selectivity `{verdicts['max_within_group_selectivity']}`.",
        f"- Patching gate: `{verdicts['causal']}` with relation-control gap `{verdicts['relation_patch_specificity_gap']}`.",
        "- Read the rest of this document as the prosecutor's brief, not as supplementary decoration.",
        "",
        "## Cheap explanation 1: the 'relation direction' is an entity-class direction",
        "",
        "- Test: swap-group scopes restrict probing to families sharing the SAME subjects.",
        f"- Numbers: all-relation accuracy {p['all_relation_accuracy_at_best_final_depth']} (entities differ, partly entity-classifiable)",
        f"  vs within-group accuracies { {g: s['accuracy'] for g, s in p['within_group'].items()} }.",
        "- Verdict: if the within-group numbers sit near chance or shuffled while the all-relation",
        "  number is high, the lab found ENTITY geometry, not relation geometry. That negative is",
        "  a clean instrument result, not a failed lab.",
        "",
        "## Cheap explanation 2: it is a template/syntax direction",
        "",
        "- Test: inside a swap group the template skeleton is identical except the relation word,",
        "  so template syntax cannot separate families there. Cross-group comparisons are not",
        "  template-controlled and cannot carry the headline claim alone.",
        "- Verdict: within-group separation survives this control by construction; cross-group",
        "  separation remains a breadth diagnostic, not a proof.",
        "",
        "## Cheap explanation 3: it is a relation-word token echo",
        "",
        "- The within-group probe could be reading a lingering copy of the relation-word token",
        "  (`capital` vs `language`) rather than reusable relation geometry. Probes cannot kill",
        "  this reading here; the lab says that out loud.",
        f"- Pressure test: relation-swap patching has band mean `{pa['relation_swap']['band_mean']}` and",
        f"  persists to depth `{pa['relation_swap']['persistence_depth']}` of `{metrics['n_depths'] - 1}`.",
        "  If this beats controls, the honest causal phrase is: the relation-word residual carries",
        "  relation identity that this answer pathway uses. Calling that a shared algorithm is an overclaim.",
        "",
        "## Cheap explanation 4: the patch machinery, not the content, moves the margin",
        "",
        f"- Wrong-position control: mean `{pa['wrong_position_mean_recovery']}`, max `{pa['wrong_position_max_recovery']}`.",
        f"- Mismatched-vector control: mean `{pa['mismatched_vector_mean_recovery']}`, max `{pa['mismatched_vector_max_recovery']}`.",
        "- Nonzero mismatched recovery is the Lab 5 lesson restated: stomping corrupt evidence can",
        "  raise the target margin without restoring clean content. The causal claim lives in the",
        f"  gap, not in the matched number alone. Subject gap `{pa['specificity_gap_subject']}`, relation gap `{pa['specificity_gap_relation']}`.",
        "",
        "## What 'shared geometry' is allowed to mean after this lab",
        "",
        f"- Mean cross-family localization-profile correlation: `{geo['mean_profile_correlation']}`.",
        f"- Direction cosines: same-group `{geo['cosine_same_group_mean_final']}` vs cross-group `{geo['cosine_diff_group_mean_final']}` (final role).",
        "- These are handles. Similar depth profiles and cosine block structure do not show shared",
        "  heads, shared MLPs, or a shared algorithm. Twelve unrelated tricks wearing one probe",
        "  remains live until component-level work says otherwise.",
        "",
        "## Scope notes",
        "",
        f"- Swap-group scopes measured: {sorted(scope_info)}.",
        f"- Data source: `{metrics['data_source']}`. If this says builtin fallback, the run is plumbing only.",
        "- All claims are scoped to this model, these frozen single-token items, and these templates.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary",
                          "The deflationary twin: cheap explanations, controls, and dynamic verdicts.")


def write_run_summary(ctx: bench.RunContext, bundle: bench.ModelBundle,
                      metrics: dict[str, Any], claims: list[dict[str, str]],
                      scope_info: dict[str, Any], data_info: dict[str, Any]) -> None:
    p = metrics["probe"]
    pa = metrics["patching"]
    lines = [
        "# Lab 12 run summary: relation geometry and method validation",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- items: {metrics['n_items']} across {len(metrics['families'])} relation families "
        f"(relation-set {metrics['relation_set']!r}, data source `{metrics['data_source']}`, "
        f"data sha256 {str(data_info.get('data_sha256'))[:16]}…)",
        f"- direction depths: {metrics['depth_by_role']}",
        f"- probes: centroid + OvR mass-mean at roles {ROLES}, logistic at final; activations {metrics['normalization']}",
        "- evidence levels: DECODE (probes), CAUSAL scoped (patching), OBS (margins, profiles)",
        f"- dynamic verdicts: probe `{metrics['verdicts']['decode']}`, patching `{metrics['verdicts']['causal']}`",
        "",
        "## 1. What behavior was measured?",
        "",
        "Next-token preference margins between each relation's answer and two distractors",
        "(same-class and cross-class), on frozen single-token relation prompts.",
        "",
        "## 2. What internal object was measured?",
        "",
        "Pre-final-norm residual streams using the bench `streams[k]` convention at three token roles:",
        "relation word, subject, final. Probes decode relation identity from those streams;",
        "interchange patches replace one stream vector at a time.",
        "",
        "## 3. What intervention was used, and what controls?",
        "",
        "Lab 5 interchange patching in two designs: subject-swap within relation and relation-swap",
        "inside swap groups where subject and template are held fixed. Controls: shuffled labels,",
        "random directions, entity/template-matched scopes, wrong-position patches, mismatched vectors,",
        "and a behavioral baseline gate with a drop audit.",
        "",
        "## 4. Headline numbers",
        "",
        f"- all-relation accuracy at final-role depth {metrics['depth_by_role']['final']}: "
        f"{p['all_relation_accuracy_at_best_final_depth']} (shuffled {p['all_relation_shuffled']})",
    ]
    for g, s in p["within_group"].items():
        lines.append(f"- {g} ({s['n_families']} families, same subjects): {s['accuracy']} "
                     f"vs shuffled {s['shuffled']} (chance {s['chance']}, selectivity {s['selectivity']})")
    lines += [
        f"- subject-swap patching at the subject token (depths 1..L-1): band mean "
        f"{pa['subject_swap']['band_mean']}, persists to depth {pa['subject_swap']['persistence_depth']} "
        f"({pa['n_subject_swap_pairs']} pairs)",
        f"- relation-swap patching at the relation token: band mean {pa['relation_swap']['band_mean']}, "
        f"persists to depth {pa['relation_swap']['persistence_depth']} ({pa['n_relation_swap_pairs']} pairs)",
        f"- wrong-position control mean: {pa['wrong_position_mean_recovery']}; mismatched-vector "
        f"control mean: {pa['mismatched_vector_mean_recovery']}; relation specificity gap "
        f"{pa['specificity_gap_relation']}",
        f"- mean cross-family localization-profile correlation: {metrics['geometry']['mean_profile_correlation']}",
        "",
        "## 5. What claim is supported, at what rung?",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    patch_plots_line = (
        "4. `plots/relation_patch_heatmap.png`, `plots/relation_swap_recovery.png`, "
        "and `plots/patch_control_gaps.png`."
        if pa["n_subject_swap_pairs"] or pa["n_relation_swap_pairs"]
        else "4. No patch plots are written if the behavioral patch-pair gate drops every pair; inspect `diagnostics/patch_pair_gate.csv` instead."
    )
    transfer_line = (
        "5. `tables/relation_transfer_matrix.csv` and `plots/relation_transfer_matrix.png`."
        if pa["n_subject_swap_pairs"] or pa["n_relation_swap_pairs"]
        else "5. `tables/relation_transfer_matrix.csv`; the matching plot is skipped when there are no patch rows."
    )
    lines += [
        "",
        "## 6. The reading order",
        "",
        "1. `method_validation_card.md` — the pass/fail posture before the plot feast.",
        "2. `diagnostics/tokenization_audit.csv`, `diagnostics/split_balance.csv`, and `diagnostics/patch_pair_gate.csv`.",
        "3. `plots/relation_probe_by_layer.png` and `plots/relation_probe_selectivity.png`.",
        patch_plots_line,
        transfer_line,
        "6. `plots/relation_direction_cosines.png`, then `operationalization_audit.md` before naming anything shared geometry.",
        "",
        "## 7. Caveats students must carry forward",
        "",
        "- Within-group probes are entity- and template-controlled, but relation-word token echo remains alive.",
        "- Relation-swap patching can make that echo a causal handle; it still does not identify a mechanism.",
        "- Families outside swap groups provide breadth, not the headline controlled claim.",
        "- This is method validation: trust calibration plus saved handles, not a discovery trophy.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "The seven standard questions answered with this run's numbers.")
