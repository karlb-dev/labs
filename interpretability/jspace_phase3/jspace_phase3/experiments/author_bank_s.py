# Bank S authoring — synthetic in-context composition (nextsteps §5.2,
# addendum §4.3: the only bank testing WORKING MEMORY rather than
# parametric recall — the distinction on which the word "workspace"
# turns; its composed-minus-direct contrast is preregistered as the
# first member of Family B).
#
# Every bundle defines arbitrary mappings INSIDE the prompt, so the
# model must compose two in-context hops. Variant semantics (validated
# by bank.py's S-bank rules, which bind the final query sentence):
#   direct           only the second mapping is defined; read it back
#   composed         both mappings defined; the query names ONLY the
#                    source (bridge never appears in the query)
#   bridge_supplied  both mappings defined; the query routes through the
#                    bridge explicitly (mediation control)
# Counterfactual = rotated sibling (different bridge+answer, same world).
#
# Entities are fixed nonce strings (reviewable, deterministic, and
# guaranteed disjoint from every Phase 2 answer — asserted anyway).
#
# Usage: python -m jspace_phase3.experiments.author_bank_s
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ..bank import FactBundle, phase2_triple_keys, save_bank, validate_bank
from ..paths3 import resolve_uri
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

EVIDENCE_ID = "p3-bank-s-v2"
SUPERSEDES = "p3-bank-s-tranche1-v1"  # tranche 2 adds the remaining §5.2
# template types (symbol-operator, alias-with-distractor, reversible
# mapping) plus five more worlds
TIER = "phase3-development"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
P2_MANIFEST_URI = "drive://metrics/cross_model/g5_item_manifest_v5.json"

# nonce entity pools; each family draws (source, bridge, answer) rows
W1 = ["Varnel", "Toskin", "Merrid", "Quolby", "Zarnex", "Fenlow",
      "Dratchet", "Kilvorn", "Sabreth", "Umbrix", "Playden", "Norvick",
      "Tessic", "Gormund", "Ryelot"]
W2 = ["Ashgrove", "Bellmore", "Cinderfall", "Dunwick", "Eastmere",
      "Foxhollow", "Grimsby Cross", "Harrowgate", "Ivorden", "Juniper Row",
      "Kestrel Point", "Larkspur", "Mossvale", "Nightfen", "Oakhurst"]


def rows(pool, n, k=3):
    """n disjoint (source, bridge, answer) triples from a pool."""
    out = []
    for i in range(n):
        out.append(tuple(pool[i * k + j] for j in range(k)))
    return out


FAMILIES = [
    dict(
        family="s_two_codes", group="synthetic_lookup",
        templates=dict(
            direct="In code B, the word {bridge} stands for {answer}. "
                   "According to code B, {bridge} translates to",
            composed="In code A, the word {source} stands for {bridge}. "
                     "In code B, the word {bridge} stands for {answer}. "
                     "Translating {source} through code A and then code B "
                     "gives",
            bridge_supplied="In code A, the word {source} stands for "
                            "{bridge}. In code B, the word {bridge} stands "
                            "for {answer}. Code A turns {source} into "
                            "{bridge}, and applying code B to {bridge} "
                            "gives"),
        instances=rows(W1, 5)),
    dict(
        family="s_box_owner", group="synthetic_binding",
        templates=dict(
            direct="The {bridge} box belongs to {answer}. The owner of "
                   "the {bridge} box is",
            composed="The {source} key is kept inside the {bridge} box. "
                     "The {bridge} box belongs to {answer}. The person "
                     "who owns the box holding the {source} key is",
            bridge_supplied="The {source} key is kept inside the {bridge} "
                            "box. The {bridge} box belongs to {answer}. "
                            "The {source} key is in the {bridge} box, so "
                            "the owner of the {bridge} box is"),
        instances=[("brass", "crimson", "Halvern"),
                   ("copper", "indigo", "Mistrell"),
                   ("iron", "scarlet", "Doverel"),
                   ("silver", "violet", "Kranmoor"),
                   ("pewter", "amber", "Selwick")]),
    dict(
        family="s_route_terminal", group="synthetic_path",
        templates=dict(
            direct="Platform {bridge} connects only to the {answer} exit. "
                   "From platform {bridge}, the connecting exit is",
            composed="Line {source} ends at platform {bridge}. Platform "
                     "{bridge} connects only to the {answer} exit. Riding "
                     "line {source} to its final stop and taking the "
                     "connection, one reaches the exit called",
            bridge_supplied="Line {source} ends at platform {bridge}. "
                            "Platform {bridge} connects only to the "
                            "{answer} exit. Line {source} terminates at "
                            "platform {bridge}, and from {bridge} the "
                            "connecting exit is"),
        instances=[("K4", "P9", "Wrenfield"),
                   ("R7", "P12", "Copperline"),
                   ("M2", "P6", "Thistledown"),
                   ("B9", "P15", "Galehaven"),
                   ("T5", "P11", "Marrowick")]),
    dict(
        family="s_ledger_alias", group="synthetic_binding",
        templates=dict(
            direct="The alias {bridge} is assigned to desk {answer}. "
                   "Desk assignment for the alias {bridge}:",
            composed="In the ledger, {source} is recorded under the alias "
                     "{bridge}. The alias {bridge} is assigned to desk "
                     "{answer}. The desk assigned to the alias under "
                     "which {source} is recorded is desk",
            bridge_supplied="In the ledger, {source} is recorded under "
                            "the alias {bridge}. The alias {bridge} is "
                            "assigned to desk {answer}. {source}'s alias "
                            "is {bridge}, and the desk assigned to "
                            "{bridge} is desk"),
        instances=[("Ostrander", "Bluejay", "R14"),
                   ("Pemberton", "Foxglove", "Q73"),
                   ("Quillfeather", "Marlin", "N28"),
                   ("Ravensworth", "Petrel", "V56"),
                   ("Silverstein", "Curlew", "J91")]),
    dict(
        family="s_graph_hop", group="synthetic_path",
        templates=dict(
            direct="From node {bridge} there is exactly one edge, leading "
                   "to node {answer}. One step from node {bridge} reaches "
                   "node",
            composed="From node {source} there is exactly one edge, "
                     "leading to node {bridge}. From node {bridge} there "
                     "is exactly one edge, leading to node {answer}. "
                     "Starting at node {source} and following exactly two "
                     "edges, one arrives at node",
            bridge_supplied="From node {source} there is exactly one "
                            "edge, leading to node {bridge}. From node "
                            "{bridge} there is exactly one edge, leading "
                            "to node {answer}. Two steps from {source} "
                            "means passing through {bridge} and stopping "
                            "at node"),
        instances=[("QV", "LN", "RX"), ("HK", "PW", "ZC"),
                   ("DM", "BT", "GF"), ("SY", "JQ", "VL"),
                   ("XE", "WU", "KH")]),
    dict(
        family="s_parcel_floor", group="synthetic_binding",
        templates=dict(
            direct="Locker {bridge} is on the {answer} floor. The floor "
                   "for locker {bridge} is the",
            composed="The parcel for {source} was placed in locker "
                     "{bridge}. Locker {bridge} is on the {answer} floor. "
                     "To collect the parcel for {source}, go to the floor "
                     "called the",
            bridge_supplied="The parcel for {source} was placed in locker "
                            "{bridge}. Locker {bridge} is on the {answer} "
                            "floor. {source}'s parcel sits in locker "
                            "{bridge}, which is on the floor called the"),
        instances=[("Marisol", "C12", "mezzanine"),
                   ("Thaddeus", "F03", "garret"),
                   ("Yolanda", "K88", "rooftop"),
                   ("Bartholomew", "A51", "basement"),
                   ("Genevieve", "H27", "annex")]),
    dict(
        family="s_recipe_station", group="synthetic_lookup",
        templates=dict(
            direct="The {bridge} sauce is prepared at the {answer} "
                   "station. The station for the {bridge} sauce is the",
            composed="The dish {source} is finished with the {bridge} "
                     "sauce. The {bridge} sauce is prepared at the "
                     "{answer} station. The station where the sauce for "
                     "{source} is prepared is the",
            bridge_supplied="The dish {source} is finished with the "
                            "{bridge} sauce. The {bridge} sauce is "
                            "prepared at the {answer} station. {source} "
                            "takes the {bridge} sauce, which is prepared "
                            "at the station called the"),
        instances=[("Karvella", "ember", "Halcyon"),
                   ("Dulcinet", "frost", "Bramble"),
                   ("Brindlewick", "cinder", "Foxfire"),
                   ("Solmarine", "juniper", "Quarry"),
                   ("Tarragonda", "smoke", "Rushlight")]),
    dict(
        family="s_badge_gate", group="synthetic_path",
        templates=dict(
            direct="Badge color {bridge} opens only gate {answer}. The "
                   "gate opened by a {bridge} badge is gate",
            composed="Every member of team {source} carries a {bridge} "
                     "badge. Badge color {bridge} opens only gate "
                     "{answer}. A member of team {source} can therefore "
                     "open gate",
            bridge_supplied="Every member of team {source} carries a "
                            "{bridge} badge. Badge color {bridge} opens "
                            "only gate {answer}. Team {source} carries "
                            "{bridge} badges, and a {bridge} badge opens "
                            "gate"),
        instances=[("Osprey", "teal", "G7"),
                   ("Lynx", "maroon", "K3"),
                   ("Heron", "ochre", "M9"),
                   ("Marten", "cobalt", "T4"),
                   ("Plover", "sable", "W2")]),
]


FAMILIES += [
    dict(
        family="s_rule_operator", group="synthetic_lookup",
        templates=dict(
            direct="Under the merge rule, {bridge} becomes {answer}. "
                   "Applying the merge rule to {bridge} produces",
            composed="Under the flip rule, {source} becomes {bridge}. "
                     "Under the merge rule, {bridge} becomes {answer}. "
                     "Applying the flip rule and then the merge rule to "
                     "{source} produces",
            bridge_supplied="Under the flip rule, {source} becomes "
                            "{bridge}. Under the merge rule, {bridge} "
                            "becomes {answer}. The flip rule sends "
                            "{source} to {bridge}, and the merge rule "
                            "then produces"),
        instances=[("Drovak", "Plimset", "Corvane"),
                   ("Ilbeck", "Trosmit", "Vandrel"),
                   ("Osprel", "Klindor", "Marbeth"),
                   ("Ungrath", "Selpix", "Dorwin"),
                   ("Yarnold", "Brelqua", "Fenwick")]),
    dict(
        family="s_alias_distractor", group="synthetic_binding",
        templates=dict(
            direct="The alias {bridge} unlocks cabinet {answer}. The "
                   "cabinet unlocked by the alias {bridge} is cabinet",
            composed="In the registry, {source} appears under the alias "
                     "{bridge}. In the registry, Welfort appears under "
                     "the alias Squall. The alias {bridge} unlocks "
                     "cabinet {answer}. The cabinet unlocked by the "
                     "alias of {source} is cabinet",
            bridge_supplied="In the registry, {source} appears under the "
                            "alias {bridge}. In the registry, Welfort "
                            "appears under the alias Squall. The alias "
                            "{bridge} unlocks cabinet {answer}. "
                            "{source}'s alias is {bridge}, and {bridge} "
                            "unlocks cabinet"),
        instances=[("Ambrell", "Falcon", "D2"),
                   ("Brockway", "Osier", "F8"),
                   ("Crandell", "Wren", "H5"),
                   ("Dunmore", "Teal", "L3"),
                   ("Everhart", "Skua", "N7")]),
    dict(
        family="s_same_specimen", group="synthetic_binding",
        templates=dict(
            direct="The specimen called {bridge} is stored in drawer "
                   "{answer}. The drawer holding {bridge} is drawer",
            composed="{source} and {bridge} are two names for the same "
                     "specimen. The specimen called {bridge} is stored "
                     "in drawer {answer}. The drawer holding {source} is "
                     "drawer",
            bridge_supplied="{source} and {bridge} are two names for the "
                            "same specimen. The specimen called {bridge} "
                            "is stored in drawer {answer}. Since {source} "
                            "is {bridge}, it sits in drawer"),
        instances=[("Kelvarite", "Bluntstone", "C4"),
                   ("Mornadine", "Ashglass", "E9"),
                   ("Purlquartz", "Greyshard", "G6"),
                   ("Rindspar", "Duskflint", "J2"),
                   ("Tavermica", "Palegrit", "K8")]),
    dict(
        family="s_warehouse_camera", group="synthetic_path",
        templates=dict(
            direct="Aisle {bridge} is monitored by camera {answer}. The "
                   "camera covering aisle {bridge} is camera",
            composed="Crate {source} sits in aisle {bridge}. Aisle "
                     "{bridge} is monitored by camera {answer}. The "
                     "camera that can see crate {source} is camera",
            bridge_supplied="Crate {source} sits in aisle {bridge}. "
                            "Aisle {bridge} is monitored by camera "
                            "{answer}. Crate {source} is in aisle "
                            "{bridge}, watched by camera"),
        instances=[("V19", "Delta", "M1"),
                   ("W44", "Sigma", "M6"),
                   ("X73", "Omega", "M3"),
                   ("Y28", "Kappa", "M9"),
                   ("Z56", "Theta", "M4")]),
    dict(
        family="s_signal_relay", group="synthetic_path",
        templates=dict(
            direct="Relay {bridge} broadcasts on channel {answer}. The "
                   "channel used by relay {bridge} is channel",
            composed="Signal {source} is forwarded to relay {bridge}. "
                     "Relay {bridge} broadcasts on channel {answer}. "
                     "Listeners hear signal {source} on channel",
            bridge_supplied="Signal {source} is forwarded to relay "
                            "{bridge}. Relay {bridge} broadcasts on "
                            "channel {answer}. Signal {source} passes "
                            "through {bridge} and airs on channel"),
        instances=[("Aurora", "Northgate", "C11"),
                   ("Bramblewood", "Eastspire", "C47"),
                   ("Cindertrail", "Westhollow", "C29"),
                   ("Dawnmere", "Southcrag", "C83"),
                   ("Emberlyn", "Midvane", "C65")]),
    dict(
        family="s_tonic_label", group="synthetic_lookup",
        templates=dict(
            direct="The {bridge} tonic is bottled under a {answer} "
                   "label. The label on the {bridge} tonic reads",
            composed="Powdered {source} brews into the {bridge} tonic. "
                     "The {bridge} tonic is bottled under a {answer} "
                     "label. A brew that starts from {source} ends under "
                     "a label reading",
            bridge_supplied="Powdered {source} brews into the {bridge} "
                            "tonic. The {bridge} tonic is bottled under "
                            "a {answer} label. {source} makes the "
                            "{bridge} tonic, whose label reads"),
        instances=[("wolfsbark", "amberleaf", "Quillmark"),
                   ("frostroot", "duskpetal", "Hartwood"),
                   ("emberfern", "palebriar", "Selvane"),
                   ("mirthweed", "coldbloom", "Ternbury"),
                   ("gloamsage", "wrymoss", "Vexhall")]),
    dict(
        family="s_parcel_kiosk", group="synthetic_path",
        templates=dict(
            direct="Courier {bridge} finishes every round at kiosk "
                   "{answer}. The end-of-day kiosk for courier {bridge} "
                   "is kiosk",
            composed="Parcel {source} is carried by courier {bridge}. "
                     "Courier {bridge} finishes every round at kiosk "
                     "{answer}. At day's end, parcel {source} sits at "
                     "kiosk",
            bridge_supplied="Parcel {source} is carried by courier "
                            "{bridge}. Courier {bridge} finishes every "
                            "round at kiosk {answer}. {bridge} carries "
                            "parcel {source}, ending at kiosk"),
        instances=[("P501", "Hobbes", "K12"),
                   ("P322", "Marlow", "K30"),
                   ("P874", "Ferris", "K25"),
                   ("P169", "Quimby", "K41"),
                   ("P648", "Ashby", "K17")]),
    dict(
        family="s_choir_hall", group="synthetic_binding",
        templates=dict(
            direct="The {bridge} ensemble rehearses in hall {answer}. "
                   "The rehearsal hall of the {bridge} ensemble is hall",
            composed="Singer {source} belongs to the {bridge} ensemble. "
                     "The {bridge} ensemble rehearses in hall {answer}. "
                     "Singer {source} rehearses in hall",
            bridge_supplied="Singer {source} belongs to the {bridge} "
                            "ensemble. The {bridge} ensemble rehearses "
                            "in hall {answer}. {source} sings with the "
                            "{bridge} ensemble in hall"),
        instances=[("Corvina", "Larkrise", "B6"),
                   ("Desmonda", "Nightreed", "B2"),
                   ("Elspetha", "Goldenmere", "B9"),
                   ("Fiorella", "Stormvale", "B4"),
                   ("Gwendolyn", "Ashenfell", "B7")]),
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def build_bundles() -> list[FactBundle]:
    bundles = []
    for fam in FAMILIES:
        inst = fam["instances"]
        for i, (src, bridge, ans) in enumerate(inst):
            cf = inst[(i + 1) % len(inst)]
            prompts = {v: t.format(source=src, bridge=bridge, answer=ans)
                       for v, t in fam["templates"].items()}
            slug = re.sub(r"[^a-z0-9]+", "-", src.lower()).strip("-")[:40]
            bundles.append(FactBundle(
                fact_id=f"{fam['family']}:{slug}",
                canonical_family=fam["family"],
                relation_group=fam["group"], bank="S",
                source=src, bridge=bridge, answer=ans,
                accepted_answers=[f" {ans}"], prompts=prompts,
                counterfactual_bridge=cf[1], counterfactual_answer=cf[2],
                counterfactual_accepted=[f" {cf[2]}"],
                provenance={"source": "synthetic",
                            "generator": "author_bank_s tranche 1"}))
    return bundles


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    bundles = build_bundles()
    p2 = phase2_triple_keys(json.load(open(resolve_uri(P2_MANIFEST_URI)))["payload"])
    p2_answers = {k for k in p2 if k.startswith("|")}
    for b in bundles:
        assert f"|{norm(b.answer)}" not in p2_answers, \
            f"{b.fact_id}: nonce answer collides with a Phase 2 answer"
    val = validate_bank(bundles, phase2_triples=p2)
    out = REPO_DATA / "bank_s_v2.jsonl"
    save_bank(bundles, out)
    payload = {"n_bundles": len(bundles),
               "n_families": val["n_families"],
               "family_counts": val["family_counts"],
               "validation": val}
    cmd = "python -m jspace_phase3.experiments.author_bank_s"
    meta = REPO_DATA / "bank_s_v2.meta.json"
    write_result3(payload, meta, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=0))
    register(EVIDENCE_ID, tier=TIER, command=cmd, supersedes=SUPERSEDES,
             what=(f"Bank S v2: {len(bundles)} synthetic in-context "
                   f"composition bundles / {val['n_families']} template "
                   f"families, direct/composed/bridge_supplied + rotated "
                   f"counterfactuals, nonce entities disjoint from Phase 2"),
             outputs=[out, meta])
    print(json.dumps({k: v for k, v in payload.items() if k != "validation"},
                     indent=1))
    print("validation ok:", val["ok"], "| violations:",
          json.dumps(val["violations"], indent=1)[:1500])


if __name__ == "__main__":
    main()
