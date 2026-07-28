# N1.5 — canonical family accessor. NO analysis may derive a family from
# a string; every clustered statistic obtains it from the audited map
# (data/probe_swap_family_map.json, authored in family_authoring.py).
#
# nextsteps_2_2 §8.2: `attach_family` validates many-to-one and refuses
# unmapped items rather than silently producing singleton clusters.
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "probe_swap_family_map.json"


class FamilyMapError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_map(path: str | Path = MAP_PATH) -> dict:
    m = json.loads(Path(path).read_text())
    by_item = {r["item_id"]: r for r in m["items"]}
    if len(by_item) != len(m["items"]):
        raise FamilyMapError("duplicate item_id in family map")
    m["_by_item"] = by_item
    return m


def family_of(item_id: str) -> str:
    m = load_map()
    row = m["_by_item"].get(item_id)
    if row is None:
        raise FamilyMapError(f"no canonical family for item {item_id!r}")
    return row["canonical_family"]


def attach_family(rows, *, item_column: str = "item_id", strict: bool = True):
    """Join canonical_family/template onto a per-item frame. Overwrites any
    pre-existing `family` column (which, for pilot artifacts, is the
    defective prefix field) and preserves it as `family_legacy`."""
    import pandas as pd

    m = load_map()
    fm = pd.DataFrame([{k: r[k] for k in
                        ("item_id", "canonical_family", "template_id", "template_hash")}
                       for r in m["items"]])
    out = rows.copy()
    if "family" in out.columns:
        out = out.rename(columns={"family": "family_legacy"})
    out = out.merge(fm, left_on=item_column, right_on="item_id",
                    how="left", validate="many_to_one",
                    suffixes=("", "_fm"))
    if "item_id_fm" in out.columns:
        out = out.drop(columns=["item_id_fm"])
    missing = out["canonical_family"].isna()
    if missing.any():
        ids = sorted(out.loc[missing, item_column].unique())
        if strict:
            raise FamilyMapError(
                f"missing canonical family for {len(ids)} items: {ids[:10]!r}")
        out.loc[missing, "canonical_family"] = "UNMAPPED"
    return out


def audit() -> dict:
    """Map-level invariants (the review's failure mode: two nominally
    distinct families sharing one template)."""
    m = load_map()
    items = m["items"]
    by_hash: dict[str, set] = {}
    by_template: dict[str, set] = {}
    for r in items:
        by_hash.setdefault(r["template_hash"], set()).add(r["canonical_family"])
        by_template.setdefault(r["template_id"], set()).add(r["canonical_family"])
    shared_hash = {h: sorted(f) for h, f in by_hash.items() if len(f) > 1}
    shared_tid = {t: sorted(f) for t, f in by_template.items() if len(f) > 1}
    fam_sizes: dict[str, int] = {}
    for r in items:
        fam_sizes[r["canonical_family"]] = fam_sizes.get(r["canonical_family"], 0) + 1
    return {
        "n_items": len(items),
        "n_families": len(fam_sizes),
        "n_templates": len(by_template),
        "families_sharing_a_template_hash": shared_hash,
        "families_sharing_a_template_id": shared_tid,
        "largest_families": sorted(fam_sizes.items(), key=lambda kv: -kv[1])[:10],
        "singletons": sum(1 for v in fam_sizes.values() if v == 1),
        "ok": not shared_hash and not shared_tid,
    }
