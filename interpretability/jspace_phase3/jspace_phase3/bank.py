# Bank F / Bank S schema + validators (nextsteps §5, addendum §4.2–4.3).
#
# THE POINT OF THE THICK BANK: Phase 2 compared a released two-hop
# battery against a separate one-hop battery, so "composed vs direct"
# was confounded with answer identity, relation family, surface form and
# difficulty — and the confirmatory two-hop leg collapsed to 9 items in 2
# families. Phase 3 pairs variants WITHIN a fact bundle, so the primary
# contrast cancels answer identity and most of relation difficulty:
#
#   direct           bridge -> answer          ("The language spoken in
#                                               Brazil is")
#   composed         source -> bridge -> answer ("The language spoken in
#                                               the country where the
#                                               Amazon River ends is")
#   bridge_supplied  composed prompt + bridge stated (mediation control)
#   three_hop        optional deeper chain
#
# Every variant of a bundle shares the SAME final answer and the SAME
# frozen alias set, which is what makes the paired contrast legitimate.
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

VARIANTS = ("direct", "composed", "bridge_supplied", "three_hop")


@dataclass(frozen=True)
class FactBundle:
    fact_id: str
    canonical_family: str          # (second-hop relation, answer type)
    relation_group: str            # coarser pooling, sensitivity unit
    bank: str                      # "F" (factual) | "S" (synthetic)
    source: str
    bridge: str
    answer: str
    accepted_answers: list[str]
    prompts: dict                  # variant -> prompt text
    counterfactual_bridge: str | None = None
    counterfactual_answer: str | None = None
    counterfactual_accepted: list[str] = field(default_factory=list)
    distractor_bridge: str | None = None
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------ derived
    @property
    def template_hash(self) -> str:
        """Surface-template identity: prompts with entities masked out.
        Partition disjointness is enforced on this, not just on text."""
        masked = {}
        for v, p in sorted(self.prompts.items()):
            m = p
            for ent in (self.source, self.bridge, self.answer):
                if ent:
                    m = m.replace(ent, "<E>")
            masked[v] = re.sub(r"\s+", " ", m).strip()
        return hashlib.sha256(
            json.dumps(masked, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def triple_key(self) -> str:
        """(bridge, answer) identity for cross-PHASE dedup (addendum
        §4.2a): no Phase 3 bundle may share this with a Phase 2 item."""
        return f"{_norm(self.bridge)}|{_norm(self.answer)}"

    def as_items(self) -> list[dict]:
        out = []
        for v, prompt in self.prompts.items():
            out.append({
                "item_id": f"{self.fact_id}#{v}",
                "fact_id": self.fact_id, "variant": v, "bank": self.bank,
                "canonical_family": self.canonical_family,
                "relation_group": self.relation_group,
                "prompt": prompt, "canonical_answer": self.answer,
                "accepted_answers": list(self.accepted_answers),
                "source_entity": self.source, "bridge_entity": self.bridge,
                "counterfactual_bridge": self.counterfactual_bridge,
                "counterfactual_answer": self.counterfactual_answer,
                "counterfactual_accepted": list(self.counterfactual_accepted),
                "distractor_bridge": self.distractor_bridge,
                "template_hash": self.template_hash,
                "provenance": dict(self.provenance),
            })
        return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# ------------------------------------------------------------ validators
class BankError(ValueError):
    pass


def validate_bundle(b: FactBundle) -> list[str]:
    """§5.4 authoring constraints. Returns a list of violations (empty =
    clean) rather than raising, so an authoring pass can report all."""
    v: list[str] = []
    if "direct" not in b.prompts or "composed" not in b.prompts:
        v.append("missing direct or composed variant")
    for name, p in b.prompts.items():
        if name not in VARIANTS:
            v.append(f"unknown variant {name!r}")
        if p != p.rstrip():
            v.append(f"{name}: trailing whitespace (scoring boundary)")
        if not p.endswith(("is", "are", "was", "were", "be", "of", "the",
                           "to", "in", "by", "as", ":")) and \
                not p[-1].isalnum():
            v.append(f"{name}: prompt does not end at a scoring boundary")
        pn = _norm(p)
        if _norm(b.answer) and _norm(b.answer) in pn:
            v.append(f"{name}: ANSWER leaks into the prompt")
        if name == "composed" and _norm(b.bridge) and _norm(b.bridge) in pn:
            v.append("composed: BRIDGE leaks into the primary prompt")
        if name == "bridge_supplied" and _norm(b.bridge) not in pn:
            v.append("bridge_supplied: bridge is not actually supplied")
        if name == "direct" and _norm(b.bridge) and _norm(b.bridge) not in pn:
            v.append("direct: prompt must ask the second hop from the bridge")
    if not b.accepted_answers:
        v.append("empty alias set")
    for a in b.accepted_answers:
        if a.strip() == "":
            v.append("blank alias")
    if _norm(b.answer) not in {_norm(a) for a in b.accepted_answers}:
        v.append("canonical answer absent from its own alias set")
    if b.counterfactual_bridge and not b.counterfactual_answer:
        v.append("counterfactual bridge without counterfactual answer")
    if b.counterfactual_answer and \
            _norm(b.counterfactual_answer) == _norm(b.answer):
        v.append("counterfactual answer equals the true answer")
    if b.counterfactual_answer and not b.counterfactual_accepted:
        v.append("counterfactual answer has no alias set")
    if b.bank == "F" and not b.provenance.get("source"):
        v.append("Bank F bundle without a factual source in provenance")
    return v


def check_alias_prefix_free(aliases: list[str]) -> list[list[str]]:
    """Surface-level prefix audit (the tokenizer-level audit lives in
    scoring.ScoringSession.alias_audit and is the binding one)."""
    bad = []
    ns = [(a, _norm(a)) for a in aliases]
    for i, (a1, n1) in enumerate(ns):
        for a2, n2 in ns[i + 1:]:
            short, long_ = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
            if short and long_.startswith(short):
                bad.append([a1, a2])
    return bad


def validate_bank(bundles: list[FactBundle], *,
                  phase2_triples: set[str] | None = None) -> dict:
    """Bank-level audits: duplicate facts, cross-phase triple leakage,
    family sizes, counterfactual availability, alias prefix-freedom."""
    report: dict = {"n_bundles": len(bundles), "violations": {},
                    "duplicate_fact_ids": [], "duplicate_triples": [],
                    "phase2_triple_collisions": [], "alias_prefix_issues": {},
                    "family_counts": {}, "relation_group_counts": {},
                    "template_reuse": {}}
    seen_ids, seen_triples, templates = set(), {}, {}
    for b in bundles:
        viol = validate_bundle(b)
        if viol:
            report["violations"][b.fact_id] = viol
        if b.fact_id in seen_ids:
            report["duplicate_fact_ids"].append(b.fact_id)
        seen_ids.add(b.fact_id)
        if b.triple_key in seen_triples:
            report["duplicate_triples"].append(
                [seen_triples[b.triple_key], b.fact_id])
        seen_triples[b.triple_key] = b.fact_id
        if phase2_triples and b.triple_key in phase2_triples:
            report["phase2_triple_collisions"].append(b.fact_id)
        pref = check_alias_prefix_free(b.accepted_answers)
        if pref:
            report["alias_prefix_issues"][b.fact_id] = pref
        report["family_counts"][b.canonical_family] = \
            report["family_counts"].get(b.canonical_family, 0) + 1
        report["relation_group_counts"][b.relation_group] = \
            report["relation_group_counts"].get(b.relation_group, 0) + 1
        templates.setdefault(b.template_hash, []).append(b.canonical_family)
    report["template_reuse"] = {t: sorted(set(f)) for t, f in templates.items()
                                if len(set(f)) > 1}
    report["n_families"] = len(report["family_counts"])
    report["n_with_counterfactual"] = sum(
        1 for b in bundles if b.counterfactual_bridge)
    report["ok"] = not (report["violations"] or report["duplicate_fact_ids"]
                        or report["duplicate_triples"]
                        or report["phase2_triple_collisions"]
                        or report["alias_prefix_issues"]
                        or report["template_reuse"])
    return report


def phase2_triple_keys(manifest_payload: dict) -> set[str]:
    """(bridge, answer) keys of every Phase 2 item, for §4.2a dedup."""
    out = set()
    for r in manifest_payload["items"]:
        bridge = r.get("bridge_entity") or ""
        out.add(f"{_norm(bridge)}|{_norm(r.get('canonical_answer', ''))}")
        # answer-only key too: famous two-hop answers are few
        out.add(f"|{_norm(r.get('canonical_answer', ''))}")
    return out


# --------------------------------------------------------------- storage
def save_bank(bundles: list[FactBundle], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(
        json.dumps(asdict(b), sort_keys=True) for b in bundles) + "\n")
    return path


def load_bank(path: Path) -> list[FactBundle]:
    return [FactBundle(**json.loads(l))
            for l in Path(path).read_text().splitlines() if l.strip()]
