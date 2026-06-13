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
import math
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


def load_items(args: Any) -> tuple[list[Item], dict[str, Any]]:
    path = bench.COURSE_ROOT / "data" / DATA_FILE
    if not path.exists():
        raise RuntimeError(
            f"Frozen dataset missing: {path}. Re-checkout the repo or run "
            "data/make_advanced_relation_sets.py at authoring time; do not "
            "regenerate per-run."
        )
    items: list[Item] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append(Item(
                item_id=row["item_id"],
                family=row["family"],
                swap_group=row["swap_group"],
                entity_group=row["entity_group"],
                prompt=row["prompt"],
                relword=row["relword"],
                subject=row["subject"],
                target=row["target"],
                hard_distractor=row["hard_distractor"],
                easy_distractor=row["easy_distractor"],
                note=row.get("note", ""),
            ))

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
        for it in items:  # CSV order is roster order, shared across swap families
            if seen[it.family] < cap:
                capped.append(it)
                seen[it.family] += 1
        items = capped

    info = {"relation_set": set_name, "per_family_cap": cap,
            "families": sorted({it.family for it in items}), "n_items": len(items),
            "data_sha256": bench.sha256_file(path)}
    return items, info


def tokenization_gate(
    ctx: bench.RunContext, bundle: bench.ModelBundle, items: list[Item]
) -> list[Item]:
    """Re-verify every measured token against the RUNTIME tokenizer and
    locate role positions inside the exact encoding the capture will use.

    The CSV was verified against gpt2 and Olmo-3 at authoring time, but the
    lab never trusts authoring-time facts about a tokenizer it did not see:
    anything that breaks here is dropped with an audit row, not patched over.
    """
    tokenizer = bundle.tokenizer
    kept: list[Item] = []
    rows: list[dict[str, Any]] = []
    for it in items:
        problems: list[str] = []
        enc = tokenizer(it.prompt, add_special_tokens=True)["input_ids"]

        def one_token_id(text: str, label: str) -> int:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                problems.append(f"{label} {text!r} is {len(ids)} tokens")
                return -1
            return ids[0]

        subj_id = one_token_id(" " + it.subject, "subject")
        rel_id = one_token_id(" " + it.relword, "relword")
        target_id = one_token_id(it.target, "target")
        hard_id = one_token_id(it.hard_distractor, "hard_distractor")
        easy_id = one_token_id(it.easy_distractor, "easy_distractor")

        subject_pos = relword_pos = -1
        if subj_id >= 0:
            hits = [p for p, t in enumerate(enc) if t == subj_id]
            if len(hits) == 1:
                subject_pos = hits[0]
            else:
                problems.append(f"subject token occurs {len(hits)}x in prompt")
        if rel_id >= 0:
            hits = [p for p, t in enumerate(enc) if t == rel_id]
            if len(hits) == 1:
                relword_pos = hits[0]
            else:
                problems.append(f"relword token occurs {len(hits)}x in prompt")

        rows.append({
            "item_id": it.item_id, "family": it.family, "prompt": it.prompt,
            "n_tokens": len(enc), "subject_pos": subject_pos,
            "relword_pos": relword_pos, "kept": not problems,
            "problems": "; ".join(problems),
        })
        if not problems:
            it.input_ids = list(enc)
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
    if len(kept) < 24:
        raise RuntimeError(
            f"Only {len(kept)} items survived the tokenization gate; the probe "
            "phase needs more. Check diagnostics/tokenization_audit.csv."
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
    """Nearest-class-mean multiclass accuracy: the simplest geometry claim
    ("family centroids are mutually separated"), no per-class threshold."""
    import torch

    cents = torch.stack([
        X_train[[f == fam for f in fams_train]].mean(dim=0) for fam in families
    ])
    d2 = torch.cdist(X_eval, cents)
    pred = d2.argmin(dim=1)
    truth = torch.tensor([families.index(f) for f in fams_eval])
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

    # Scopes: the full 12-way problem, plus each swap group restricted to the
    # subjects its families SHARE (the entity/template-controlled scope).
    scopes: list[tuple[str, list[int], list[str]]] = []
    scopes.append(("all", list(range(len(items))), families_all))
    for group in SWAP_GROUPS:
        fams = sorted({it.family for it in items if it.swap_group == group})
        if len(fams) < 2:
            continue
        subj_sets = [{it.subject for it in items if it.family == f} for f in fams]
        common = set.intersection(*subj_sets)
        idx = [i for i, it in enumerate(items)
               if it.swap_group == group and it.subject in common]
        if len(idx) >= 3 * len(fams):
            scopes.append((group, idx, fams))

    train_mask = torch.tensor([split_lookup[it.item_id] for it in items])

    for scope_name, idx_list, fams in scopes:
        idx = torch.tensor(idx_list)
        scope_train = idx[train_mask[idx]]
        scope_eval = idx[~train_mask[idx]]
        fam_of = [items[i].family for i in idx_list]
        fams_train = [items[int(i)].family for i in scope_train]
        fams_eval = [items[int(i)].family for i in scope_eval]
        if len(set(fams_train)) < len(fams) or len(set(fams_eval)) < len(fams):
            print(f"[lab12] scope {scope_name}: a family is missing from train or "
                  "eval after the subject split; rows for it will be partial")
        for role_i, role in enumerate(ROLES):
            for depth in range(n_depths):
                Xtr = feats[scope_train, depth, role_i, :]
                Xev = feats[scope_eval, depth, role_i, :]

                # Multiclass centroid accuracy + shuffled-label control.
                acc = centroid_accuracy(Xtr, fams_train, Xev, fams_eval, fams)
                report.append({"scope": scope_name, "role": role, "depth": depth,
                               "method": "centroid", "family": "multiclass",
                               "eval_kind": "real", "metric": "accuracy",
                               "value": rounded(acc), "n_train": len(scope_train),
                               "n_eval": len(scope_eval), "chance": rounded(1.0 / len(fams))})
                ctrl_accs = []
                for s in range(N_SHUFFLES):
                    fams_shuf = shuffled(fams_train, seed * 1009 + depth * 31 + role_i * 7 + s)
                    ctrl_accs.append(centroid_accuracy(Xtr, fams_shuf, Xev, fams_eval, fams))
                report.append({"scope": scope_name, "role": role, "depth": depth,
                               "method": "centroid", "family": "multiclass",
                               "eval_kind": "shuffled", "metric": "accuracy",
                               "value": rounded(safe_fmean(ctrl_accs)),
                               "n_train": len(scope_train), "n_eval": len(scope_eval),
                               "chance": rounded(1.0 / len(fams))})

                # One-vs-rest mass-mean AUC per family + random-direction AUC.
                for fam in fams:
                    is_fam_tr = torch.tensor([f == fam for f in fams_train])
                    is_fam_ev = [f == fam for f in fams_eval]
                    if not is_fam_tr.any() or not (~is_fam_tr).any() or True not in is_fam_ev:
                        continue
                    d = ovr_mass_mean_direction(Xtr, is_fam_tr)
                    scores = (Xev @ d).tolist()
                    pos = [s for s, m in zip(scores, is_fam_ev) if m]
                    neg = [s for s, m in zip(scores, is_fam_ev) if not m]
                    report.append({"scope": scope_name, "role": role, "depth": depth,
                                   "method": "mass_mean_ovr", "family": fam,
                                   "eval_kind": "real", "metric": "auc",
                                   "value": rounded(auc_from_scores(pos, neg)),
                                   "n_train": len(scope_train), "n_eval": len(scope_eval),
                                   "chance": 0.5})
                gen = torch.Generator().manual_seed(seed * 7919 + depth * 101 + role_i)
                rnd_aucs = []
                for _ in range(N_RANDOM_DIRS):
                    d = torch.randn(Xtr.shape[1], generator=gen)
                    d = d / d.norm().clamp_min(1e-9)
                    fam = fams[0]
                    is_fam_tr = torch.tensor([f == fam for f in fams_train])
                    is_fam_ev = [f == fam for f in fams_eval]
                    sign = 1.0
                    tr_scores = Xtr @ d
                    if is_fam_tr.any() and (~is_fam_tr).any():
                        if float(tr_scores[is_fam_tr].mean()) < float(tr_scores[~is_fam_tr].mean()):
                            sign = -1.0
                    scores = (sign * (Xev @ d)).tolist()
                    pos = [s for s, m in zip(scores, is_fam_ev) if m]
                    neg = [s for s, m in zip(scores, is_fam_ev) if not m]
                    rnd_aucs.append(auc_from_scores(pos, neg))
                report.append({"scope": scope_name, "role": role, "depth": depth,
                               "method": "random_direction", "family": fams[0],
                               "eval_kind": "random", "metric": "auc",
                               "value": rounded(safe_fmean(rnd_aucs)),
                               "n_train": len(scope_train), "n_eval": len(scope_eval),
                               "chance": 0.5})

        # Logistic OvR at the readout position only (max-margin upper bound;
        # closed-form probes carry the rest of the sweep — runtime discipline).
        role_i = ROLES.index("final")
        for depth in range(n_depths):
            Xtr = feats[scope_train, depth, role_i, :]
            Xev = feats[scope_eval, depth, role_i, :]
            for fam in fams:
                y = torch.tensor([1 if f == fam else 0 for f in fams_train])
                if not bool((y == 1).any()) or not bool((y == 0).any()):
                    continue
                probe = fit_logistic(Xtr, y)
                scores = logistic_scores(probe, Xev)
                is_fam_ev = [f == fam for f in fams_eval]
                pos = [s for s, m in zip(scores, is_fam_ev) if m]
                neg = [s for s, m in zip(scores, is_fam_ev) if not m]
                report.append({"scope": scope_name, "role": "final", "depth": depth,
                               "method": "logistic_ovr", "family": fam,
                               "eval_kind": "real", "metric": "auc",
                               "value": rounded(auc_from_scores(pos, neg)),
                               "n_train": len(scope_train), "n_eval": len(scope_eval),
                               "chance": 0.5})

    scope_info = {name: {"n_items": len(idx), "families": fams}
                  for name, idx, fams in scopes}
    return report, scope_info


# ---------------------------------------------------------------------------
# Direction geometry
# ---------------------------------------------------------------------------


def best_depth_from_report(report: list[dict[str, Any]], n_depths: int) -> int:
    """Depth where the entity/template-controlled evidence is strongest:
    mean within-swap-group centroid accuracy minus its shuffled control,
    averaged over groups, at the final role."""
    def selectivity(depth: int) -> float:
        vals = []
        for group in SWAP_GROUPS:
            real = [r["value"] for r in report
                    if r["scope"] == group and r["role"] == "final" and r["depth"] == depth
                    and r["method"] == "centroid" and r["eval_kind"] == "real"]
            ctrl = [r["value"] for r in report
                    if r["scope"] == group and r["role"] == "final" and r["depth"] == depth
                    and r["method"] == "centroid" and r["eval_kind"] == "shuffled"]
            if real and ctrl:
                vals.append(float(real[0]) - float(ctrl[0]))
        return safe_fmean(vals, default=0.0)

    return max(range(n_depths), key=selectivity)


def build_relation_directions(
    items: list[Item], feats: Any, split_lookup: dict[str, bool],
    depth: int, families: list[str],
) -> dict[str, dict[str, Any]]:
    """Per-family OvR mass-mean directions at the chosen depth (train rows
    only), for the subject and final roles."""
    import torch

    train_idx = [i for i, it in enumerate(items) if split_lookup[it.item_id]]
    fams_train = [items[i].family for i in train_idx]
    out: dict[str, dict[str, Any]] = {}
    for role in ("subject", "final"):
        role_i = ROLES.index(role)
        X = feats[torch.tensor(train_idx), depth, role_i, :]
        dirs: dict[str, Any] = {}
        for fam in families:
            is_fam = torch.tensor([f == fam for f in fams_train])
            if is_fam.any() and (~is_fam).any():
                dirs[fam] = ovr_mass_mean_direction(X, is_fam)
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
    """Patch the corrupt prompt's swap position with a vector from an
    UNRELATED family's clean run (same depth, that prompt's own swap-role
    position). If recovery here rivals the matched patch, the matched result
    was never about relation/subject content."""
    rows: list[dict[str, Any]] = []
    by_family: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_family[it.family].append(it)
    families = sorted(by_family)
    seen: set[str] = set()
    for pair in pairs:
        if pair.family_clean in seen or pair.kind != "subject_swap":
            continue
        seen.add(pair.family_clean)
        donor_family = families[(families.index(pair.family_clean) + 1) % len(families)]
        donor = next((it for it in by_family[donor_family]
                      if it.item_id in captures), None)
        if donor is None:
            continue
        donor_cap = captures[donor.item_id]
        for depth in range(n_depths):
            logits = bench.run_with_residual_patch(
                bundle, pair.corrupt.prompt, depth, pair.swap_pos,
                donor_cap.streams[depth, donor.subject_pos],
            )
            rows.append({
                "kind": "mismatched_control", "family_clean": pair.family_clean,
                "family_corrupt": pair.family_corrupt,
                "clean_item": donor.item_id, "corrupt_item": pair.corrupt.item_id,
                "depth": depth, "patch_role": "subject", "position": pair.swap_pos,
                "clean_diff": rounded(pair.clean_diff),
                "corrupt_diff": rounded(pair.corrupt_diff),
                "patched_diff": rounded(pair_logit_diff(logits, pair)),
                "recovery": rounded(recovery(pair_logit_diff(logits, pair), pair)),
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
                 families: list[str], depth: int) -> None:
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
        ax.set_title(f"OvR mass-mean directions @ depth {depth}, role {role}")
        for i in range(len(families)):
            for j in range(len(families)):
                if np.isfinite(grid[i, j]) and i != j:
                    ax.annotate(f"{grid[i, j]:.2f}", (j, i), ha="center", va="center", fontsize=5.5)
    if im is not None:
        fig.colorbar(im, ax=list(axes[0]), fraction=0.03, label="cosine")
    fig.suptitle("Relation-direction cosine atlas (block structure = candidate shared geometry; verify against entity groups)")
    bench.save_figure(ctx, fig, "relation_direction_cosines.png",
                      "Pairwise cosines among per-family relation directions at the selected depth.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    grid_roles = [r.strip() for r in str(getattr(args, "patch_grid", "subject,relation,last")).split(",") if r.strip()]
    for r in grid_roles:
        if r not in ("subject", "relation", "last"):
            raise ValueError(f"--patch-grid accepts subject,relation,last; got {r!r}")

    items, data_info = load_items(args)
    set_name = data_info["relation_set"]
    print(f"[lab12] {data_info['n_items']} items, {len(data_info['families'])} families, "
          f"relation-set {set_name!r} (per-family cap {data_info['per_family_cap'] or 'none'})")
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Frozen relation CSV hash, filters, and counts.")

    # Instrument checks before science (hook indexing, lens, patch no-op).
    bench.run_hook_parity_check(ctx, bundle, items[0].prompt)
    first_capture = bench.run_with_residual_cache(bundle, items[0].prompt)
    bench.run_lens_self_check(ctx, bundle, first_capture)
    bench.run_patch_noop_check(ctx, bundle, items[0].prompt)

    items = tokenization_gate(ctx, bundle, items)
    families = sorted({it.family for it in items})
    n_depths = bundle.anatomy.n_layers + 1

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
        margin_rows.append({
            "item_id": it.item_id, "family": it.family, "subject": it.subject,
            "target": it.target.strip(), "hard_distractor": it.hard_distractor.strip(),
            "easy_distractor": it.easy_distractor.strip(),
            "margin_hard": rounded(it.margin_hard), "margin_easy": rounded(it.margin_easy),
            "knows_hard": it.margin_hard > 0, "knows_easy": it.margin_easy > 0,
            "near_tie": "near_tie=1" in it.note,
        })
        if (i + 1) % t_report == 0:
            print(f"[lab12] cached {i + 1}/{len(items)} items")

    feats_raw = torch.stack(feats_list)                       # [n, L+1, 3, d]
    row_norms = feats_raw.norm(dim=-1)                        # [n, L+1, 3]
    feats = feats_raw / row_norms[..., None].clamp_min(1e-9) if NORMALIZE_ROWS else feats_raw

    margin_path = ctx.path("tables", "margin_report.csv")
    bench.write_csv_with_context(ctx, margin_path, margin_rows)
    ctx.register_artifact(margin_path, "table",
                          "Dual-distractor logit margins per item (hard = same class, easy = cross class).")

    fam_margin_rows = []
    for fam in families:
        fam_rows = [r for r in margin_rows if r["family"] == fam]
        fam_margin_rows.append({
            "family": fam, "n": len(fam_rows),
            "mean_margin_hard": rounded(safe_fmean([float(r["margin_hard"]) for r in fam_rows])),
            "mean_margin_easy": rounded(safe_fmean([float(r["margin_easy"]) for r in fam_rows])),
            "frac_knows_hard": rounded(statistics.fmean([1.0 if r["knows_hard"] else 0.0 for r in fam_rows])),
            "frac_knows_easy": rounded(statistics.fmean([1.0 if r["knows_easy"] else 0.0 for r in fam_rows])),
        })
    fam_margin_path = ctx.path("tables", "margin_by_family.csv")
    bench.write_csv_with_context(ctx, fam_margin_path, fam_margin_rows)
    ctx.register_artifact(fam_margin_path, "table", "Per-family margin stability aggregates.")

    # ----- probes ------------------------------------------------------------
    split_lookup = make_subject_split(items, args.seed)
    split_rows = [{"item_id": it.item_id, "family": it.family, "subject": it.subject,
                   "split": "train" if split_lookup[it.item_id] else "eval"} for it in items]
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows)
    ctx.register_artifact(split_path, "diagnostic",
                          "Subject-grouped train/eval split (entity leakage hygiene).")

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

    best_depth = best_depth_from_report(probe_report, n_depths)
    print(f"[lab12] best entity-controlled depth (within-group selectivity, final role): {best_depth}")

    # ----- relation directions + cosine atlas --------------------------------
    dirs_by_role = build_relation_directions(items, feats, split_lookup, best_depth, families)
    cos_mats = {role: cosine_matrix(dirs, families) for role, dirs in dirs_by_role.items()}
    cos_rows = []
    for role, mat in cos_mats.items():
        for i, fa in enumerate(families):
            for j, fb in enumerate(families):
                if i < j:
                    cos_rows.append({"role": role, "family_a": fa, "family_b": fb,
                                     "same_swap_group": next(it.swap_group for it in items if it.family == fa) != "" and
                                     next(it.swap_group for it in items if it.family == fa) ==
                                     next(it.swap_group for it in items if it.family == fb),
                                     "cosine": rounded(mat[i][j])})
    cos_path = ctx.path("tables", "relation_cosine_matrix.csv")
    bench.write_csv_with_context(ctx, cos_path, cos_rows)
    ctx.register_artifact(cos_path, "table", "Pairwise cosines among OvR relation directions.")

    state_payload: dict[str, Any] = {
        "depth": best_depth,
        "depth_convention": "bench streams[k]: 0 = embeddings, k = residual after block k",
        "roles": {role: {fam: d for fam, d in dirs.items()} for role, dirs in dirs_by_role.items()},
        "normalization": "row_unit_norm" if NORMALIZE_ROWS else "raw_streams",
        "method": "one-vs-rest mass-mean (family mean minus rest mean), train split only",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "evidence": "DECODE artifact. Saving a direction is not evidence the model uses it.",
    }
    state_path = ctx.path("state", "relation_directions.pt")
    torch.save(state_payload, state_path)
    ctx.register_artifact(state_path, "tensor",
                          "Per-family OvR relation directions at the selected depth (subject and final roles).")

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
    swap_group_of = {it.family: it.swap_group for it in items}
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
                          "Within-relation (diagonal) vs cross-relation (swap-group cells) peak patch recovery; "
                          "cells without token-aligned pairs are honestly empty.")

    # Cross-family localization-profile similarity (the shared-machinery HANDLE).
    # Correlations use the same non-trivial band as every causal summary;
    # including depth 0 (~1.0 for everyone) and depth L (0 for everyone) would
    # manufacture similarity out of the conventions.
    profile_corrs: list[float] = []
    fam_curves = {fam: family_depth_curve(patch_rows, fam, "subject", n_depths)[1:n_layers]
                  for fam in families}
    for i, fa in enumerate(families):
        for fb in families[i + 1:]:
            c = profile_correlation(fam_curves[fa], fam_curves[fb])
            if math.isfinite(c):
                profile_corrs.append(c)

    # ----- plots ----------------------------------------------------------------
    if not args.no_plots:
        plot_probe_by_layer(ctx, probe_report, n_depths, best_depth)
        if patch_rows:
            plot_patch_heatmap(ctx, patch_rows, families, grid_roles, n_depths)
            plot_relation_swap_curves(ctx, patch_rows, n_depths)
        plot_cosines(ctx, cos_mats, families, best_depth)

    # ----- headline numbers ------------------------------------------------------
    def group_centroid(group: str, eval_kind: str, depth: int) -> float:
        vals = [float(r["value"]) for r in probe_report
                if r["scope"] == group and r["role"] == "final" and r["depth"] == depth
                and r["method"] == "centroid" and r["eval_kind"] == eval_kind]
        return vals[0] if vals else float("nan")

    group_stats = {}
    for group in SWAP_GROUPS:
        if group in scope_info:
            real = group_centroid(group, "real", best_depth)
            ctrl = group_centroid(group, "shuffled", best_depth)
            group_stats[group] = {"accuracy": rounded(real), "shuffled": rounded(ctrl),
                                  "selectivity": rounded(real - ctrl),
                                  "chance": rounded(1.0 / len(scope_info[group]["families"])),
                                  "n_families": len(scope_info[group]["families"])}
    all_real = group_centroid("all", "real", best_depth)
    all_ctrl = group_centroid("all", "shuffled", best_depth)

    rel_swap_curve = [mean_recovery(patch_rows, kind="relation_swap",
                                    patch_role="relation", depth=d) for d in range(n_depths)]
    subj_swap_curve = [mean_recovery(patch_rows, kind="subject_swap",
                                     patch_role="subject", depth=d) for d in range(n_depths)]
    rel_swap_stats = band_stats(rel_swap_curve, n_layers)
    subj_swap_stats = band_stats(subj_swap_curve, n_layers)
    wrong_pos_vals = [float(r["recovery"]) for r in patch_rows
                      if r["patch_role"] == "wrong_position"]
    mismatch_vals = [float(r["recovery"]) for r in patch_rows
                     if r["kind"] == "mismatched_control" and 1 <= int(r["depth"]) <= n_layers - 1]

    same_group_cos = [float(r["cosine"]) for r in cos_rows if r["role"] == "final" and r["same_swap_group"]]
    diff_group_cos = [float(r["cosine"]) for r in cos_rows if r["role"] == "final" and not r["same_swap_group"]]

    metrics = {
        "n_items": len(items),
        "families": families,
        "relation_set": set_name,
        "best_depth": best_depth,
        "n_depths": n_depths,
        "normalization": "row_unit_norm" if NORMALIZE_ROWS else "raw_streams",
        "probe": {
            "all_12way_accuracy_at_best_depth": rounded(all_real),
            "all_12way_shuffled": rounded(all_ctrl),
            "within_group": group_stats,
        },
        "patching": {
            "n_subject_swap_pairs": n_subject,
            "n_relation_swap_pairs": n_relation,
            "band_convention": "all causal summaries cover depths 1..n_layers-1; depth 0 is "
                               "token substitution and depth n_layers cannot reach the readout",
            "subject_swap": subj_swap_stats,
            "relation_swap": rel_swap_stats,
            "wrong_position_mean_recovery": none_if_nan(safe_fmean(wrong_pos_vals)),
            "wrong_position_max_recovery": none_if_nan(max(wrong_pos_vals, default=float("nan"))),
            "mismatched_vector_mean_recovery": none_if_nan(safe_fmean(mismatch_vals)),
            "mismatched_vector_max_recovery": none_if_nan(max(mismatch_vals, default=float("nan"))),
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
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 12 metrics.")

    # ----- operationalization audit ----------------------------------------------
    write_operationalization_audit(ctx, metrics, scope_info)

    # ----- claims + run summary -----------------------------------------------------
    run_name = ctx.run_dir.name
    gs = group_stats
    group_text = "; ".join(
        f"{g}: {s['accuracy']} vs shuffled {s['shuffled']} (chance {s['chance']})"
        for g, s in gs.items()
    ) or "no swap-group scope survived"
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE",
            "text": (
                f"Relation identity is decodable from {bundle.anatomy.model_id}'s residual stream at "
                f"depth {best_depth} (final token, centroid classifier) inside entity- and "
                f"template-matched swap groups — {group_text} — so the separation is not entity class "
                f"or template syntax. A relation-word token echo is NOT excluded by this claim."
            ),
            "artifact": f"runs/{run_name}/tables/probe_report.csv",
            "falsifier": (
                "Within-group accuracy collapses to its shuffled control on a new entity roster, or a "
                "probe trained only on relation-word token embeddings matches it."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": (
                f"Interchange patching is position- and content-specific for relations on "
                f"{bundle.anatomy.model_id}: the relation-word residual carries causally usable "
                f"relation identity through depth {rel_swap_stats['persistence_depth']} of "
                f"{n_layers} (recovery >= 0.5; band mean {rel_swap_stats['band_mean']}, "
                f"{n_relation} gated relation-swap pairs), while subject identity persists at the "
                f"subject token through depth {subj_swap_stats['persistence_depth']} (band mean "
                f"{subj_swap_stats['band_mean']}, {n_subject} pairs). Wrong-position patches stay at "
                f"{none_if_nan(safe_fmean(wrong_pos_vals))} mean; mismatched-vector patches reach "
                f"{none_if_nan(safe_fmean(mismatch_vals))} mean — margin destruction, not content "
                f"restoration, and the gap to matched patches is the causal evidence."
            ),
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
                f"{metrics['geometry']['cosine_diff_group_mean_final']}) are a HANDLE on shared relation "
                f"machinery, not a mechanism: profile similarity does not identify shared components."
            ),
            "artifact": f"runs/{run_name}/plots/relation_patch_heatmap.png",
            "falsifier": (
                "Component-level analysis (Lab 6 toolkit) shows disjoint heads/MLPs across families "
                "despite similar depth profiles."
            ),
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    write_run_summary(ctx, bundle, metrics, claims, scope_info, data_info)
    print(f"[lab12] wrote run_summary.md, operationalization_audit.md, and {len(claims)} drafted claims")


# ---------------------------------------------------------------------------
# Audit and summary writers
# ---------------------------------------------------------------------------


def write_operationalization_audit(ctx: bench.RunContext, metrics: dict[str, Any],
                                   scope_info: dict[str, Any]) -> None:
    p = metrics["probe"]
    pa = metrics["patching"]
    geo = metrics["geometry"]
    lines = [
        "# Lab 12 operationalization audit",
        "",
        "Favorite interpretation under attack: *\"relation families have shared,",
        "measurable internal geometry that the model actually uses.\"*",
        "",
        "## Cheap explanation 1: the 'relation direction' is an entity-class direction",
        "",
        f"- Test: swap-group scopes restrict probing to families sharing the SAME subjects.",
        f"- Numbers: 12-way accuracy {p['all_12way_accuracy_at_best_depth']} (entities differ, partly entity-classifiable)",
        f"  vs within-group accuracies { {g: s['accuracy'] for g, s in p['within_group'].items()} }.",
        "- Verdict: if the within-group numbers sit near their chance values while the 12-way",
        "  number is high, the lab found ENTITY geometry, not relation geometry. Write that down",
        "  as the result; it is a valid negative.",
        "",
        "## Cheap explanation 2: it is a template/syntax direction",
        "",
        "- Test: inside a swap group the template skeleton is IDENTICAL except the relation word,",
        "  so template syntax cannot separate the families there. Cross-group comparisons are",
        "  NOT template-controlled (currency_of/home_of even end in 'is the'); the handout forbids",
        "  building claims on cross-group separations alone.",
        "- Verdict: within-group separation survives this control by construction.",
        "",
        "## Cheap explanation 3: it is a relation-word token echo",
        "",
        "- The within-group probe COULD be reading a lingering copy of the relation-word token",
        "  (capital vs language) rather than reusable relation geometry. This audit cannot fully",
        "  kill that reading with probes — by design we say so instead of hiding it.",
        f"- Pressure test: relation-swap patching moves behavior (band mean "
        f"{pa['relation_swap']['band_mean']}, persists to depth "
        f"{pa['relation_swap']['persistence_depth']} of {metrics['n_depths'] - 1})",
        "  — a token echo that the answer pathway consults IS a causal handle, but calling it",
        "  'relation geometry' would overclaim. The honest claim is positional and causal:",
        "  the relation-word residual carries relation identity that the readout uses.",
        "",
        "## Cheap explanation 4: the patch machinery, not the content, moves the margin",
        "",
        f"- Wrong-position control: mean {pa['wrong_position_mean_recovery']}, "
        f"max {pa['wrong_position_max_recovery']} — should be ~0.",
        f"- Mismatched-vector control: mean {pa['mismatched_vector_mean_recovery']}, "
        f"max {pa['mismatched_vector_max_recovery']}. Nonzero recovery here is the Lab 5",
        "  lesson restated: stomping the corrupt evidence raises the margin without restoring",
        "  the clean content. The causal claim lives in the GAP between matched patches",
        f"  (subject band mean {pa['subject_swap']['band_mean']}) and this control,",
        "  not in the matched number alone.",
        "- Verdict: matched patches must beat both controls by a wide margin or the causal",
        "  rows are void.",
        "",
        "## What 'shared geometry' is allowed to mean after this lab",
        "",
        f"- Mean cross-family localization-profile correlation: {geo['mean_profile_correlation']}.",
        f"- Direction cosines: same-group {geo['cosine_same_group_mean_final']} vs",
        f"  cross-group {geo['cosine_diff_group_mean_final']} (final role).",
        "- These are HANDLES. Similar depth profiles and cosine block structure do not show",
        "  shared heads, shared MLPs, or a shared algorithm. 'Twelve unrelated tricks wearing",
        "  one probe' remains live until component-level work (Lab 6 toolkit) says otherwise.",
        "",
        "## Scope notes",
        "",
        f"- Swap-group scopes measured: {sorted(scope_info)}.",
        "- All claims are scoped to this model, these frozen single-token items, and these",
        "  templates. The authorship relation from the outline was dropped at authoring time",
        "  (book titles are multi-token); that absence is a frame limitation, not evidence.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary",
                          "The deflationary twin: cheap explanations, the controls aimed at each, verdicts.")


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
        f"(relation-set {metrics['relation_set']!r}, data sha256 {str(data_info['data_sha256'])[:16]}…)",
        f"- probes: centroid + OvR mass-mean at roles {ROLES}, logistic at final; activations "
        f"{metrics['normalization']}",
        "- evidence levels: DECODE (probes), CAUSAL scoped (patching), OBS (margins, profiles)",
        "",
        "## 1. What behavior was measured?",
        "",
        "Next-token preference margins between the relation's answer and two distractors",
        "(same-class and cross-class), on frozen single-token relation prompts.",
        "",
        "## 2. What internal object was measured?",
        "",
        "Pre-final-norm residual streams (bench streams[k] convention) at three token roles:",
        "relation word, subject, final. Probes decode relation identity from them; interchange",
        "patches replace them one (depth, position) at a time.",
        "",
        "## 3. What intervention was used, and what controls?",
        "",
        "Lab 5 interchange patching in two designs: subject-swap (within relation) and",
        "relation-swap (same subject, relation word swapped — only token-aligned by construction",
        "inside swap groups). Controls: shuffled relation labels, random directions,",
        "entity/template-matched swap-group scopes, wrong-position patches, mismatched vectors,",
        "behavioral baseline gate with drop audit.",
        "",
        "## 4. Headline numbers",
        "",
        f"- 12-way relation accuracy at depth {metrics['best_depth']} (final role): "
        f"{p['all_12way_accuracy_at_best_depth']} (shuffled {p['all_12way_shuffled']})",
    ]
    for g, s in p["within_group"].items():
        lines.append(f"- {g} ({s['n_families']} families, same subjects): {s['accuracy']} "
                     f"vs shuffled {s['shuffled']} (chance {s['chance']}, selectivity {s['selectivity']})")
    lines += [
        f"- subject-swap patching (subject token, depths 1..L-1): band mean "
        f"{pa['subject_swap']['band_mean']}, persists to depth "
        f"{pa['subject_swap']['persistence_depth']} ({pa['n_subject_swap_pairs']} pairs)",
        f"- relation-swap patching (relation token): band mean {pa['relation_swap']['band_mean']}, "
        f"persists to depth {pa['relation_swap']['persistence_depth']} "
        f"({pa['n_relation_swap_pairs']} pairs)",
        f"- wrong-position control mean: {pa['wrong_position_mean_recovery']}; "
        f"mismatched-vector control mean: {pa['mismatched_vector_mean_recovery']} "
        f"(margin destruction, not restoration — see audit)",
        f"- mean cross-family localization-profile correlation: "
        f"{metrics['geometry']['mean_profile_correlation']}",
        "",
        "## 5. What claim is supported, at what rung?",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## 6. The reading order",
        "",
        "1. `diagnostics/tokenization_audit.csv` and `diagnostics/patch_pair_gate.csv` — what",
        "   survived the gates, before believing any number.",
        "2. `plots/relation_probe_by_layer.png` — swap-group curves vs 12-way curve vs controls.",
        "3. `plots/relation_patch_heatmap.png` and `plots/relation_swap_recovery.png` — where the",
        "   causal recovery lives, and whether the wrong-position control stays flat.",
        "4. `tables/relation_transfer_matrix.csv` — diagonal (within-relation) vs swap-group cells.",
        "5. `plots/relation_direction_cosines.png` — block structure, then immediately reread the",
        "   audit's token-echo section before naming it 'shared geometry'.",
        "6. `operationalization_audit.md` — the deflationary twin; it overrides your enthusiasm.",
        "",
        "## 7. Caveats students must carry forward",
        "",
        "- The within-group probe result is entity- and template-controlled, but a relation-word",
        "  token echo is not excluded by probes alone; the relation-swap patch makes it causal",
        "  but still positional. 'Relation geometry' as a mechanism claim is NOT earned here.",
        "- Families outside swap groups (currency, plural, color, material, home) contribute",
        "  breadth and margins, but their separations are not template-controlled.",
        "- This is method validation: its product is trust in the instruments on richer data,",
        "  plus saved directions and localization handles for later labs — not a discovery.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "The seven standard questions answered with this run's numbers.")
