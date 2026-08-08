"""P2-0 bootstrap: import boundary + hygiene events (plan §6, §57; addendum H).

Idempotent: each event is appended at most once (the registry refuses
duplicate origin ids). Run AFTER committing the package skeleton so the
events register from a clean tree:

    python -m preference_phase2.bootstrap

Hygiene items 6 (language-wall raise fix) and 7 (portable CPU paths) are
appended by their producing stages (P2-4 and P2-1 respectively), per
addendum H "all precede the freeze".
"""

from __future__ import annotations

import pathlib

from . import paths, registry
from .canonical import sha256_file

# Imported Phase 1 identity (plan §6.1), verified against the Phase 1
# registry and reports at intake.
PHASE1_FREEZE_TAG = "preference-phase1-freeze-v1"
PHASE1_CLOSEOUT_COMMIT = "3f090218ecb2c721d4d3b486e428119c67a58b4a"
PHASE2_REVIEW_TARGET = "5038315affb180ebf2ffb6d792a7ee48bc7cec5e"
PHASE1_RUN_7B = "lab38_revealed_preference_report_channel-20260807_210537-9df027"
PHASE1_RUN_32B = "lab38_revealed_preference_report_channel-20260807_211808-5f68cb"
PHASE1_BANK_CONTENT_HASH = (
    "8d5039af581204a5a276ae71c7fd50f8a9911e22ad81a69933939344a2fc9f64"
)
PHASE1_BANK_JSONL_SHA = (
    "1eea2c6017b536533ccac7b8de5ca1099ac2d3a699f0f6c04d48184fb2dca1f2"
)
PHASE1_FREEZE_RECORD_STALE_JSONL_SHA_PREFIX = "634aded4b467f0e3"
PHASE1_MECHANISM_EVENT_IDS = ["pref1-mechanism-case-study-v1"]


def _art(rel: str) -> dict:
    p = paths.repo_root() / rel
    return {"path": rel, "sha256": sha256_file(p) if p.exists() else None}


def _reanalysis_dir_artifacts() -> list[dict]:
    root = paths.reports_root() / "dev_cpu_reanalysis_20260808"
    arts = []
    for p in sorted(root.iterdir()):
        if p.is_file():
            rel = str(p.relative_to(paths.repo_root()))
            arts.append({"path": rel, "sha256": sha256_file(p)})
    return arts


def _have(event_id: str) -> bool:
    return any(
        r.get("event_id") == event_id and r.get("event") in (None, "evidence_created")
        for r in registry.read_events()
    )


def _register(**kw) -> None:
    if _have(kw["event_id"]):
        print(f"skip (exists): {kw['event_id']}")
        return
    registry.register(**kw)
    print(f"registered: {kw['event_id']}")


def main() -> None:
    plans = "interpretability/preference/plans"
    ph1 = "interpretability/preference/phase1"

    _register(
        event_id="pref2-addendum-intake-v1",
        event_type="governance",
        scientific_tier="instrument",
        claim_summary=(
            "Phase 2 governing replacement plan and execution addendum adopted: "
            "addendum §B errata E1-E17 supersede the corresponding replacement-plan "
            "text; precedence Phase 1 record < handout < Phase 1 plan/addendum < "
            "replacement plan < addendum. preference_2_1.md and the 2026-08-08 "
            "opener preference_2_2.md are superseded drafts preserved in Git history."
        ),
        input_artifacts=[
            _art(f"{plans}/preference_2_2.md"),
            _art(f"{plans}/preference_2_2_addendum.md"),
        ],
        parent_event_ids=[],
    )

    _register(
        event_id="pref2-import-phase1-v1",
        event_type="import_boundary",
        scientific_tier="instrument",
        claim_summary=(
            f"Imported Phase 1 scientific boundary pinned: freeze tag "
            f"{PHASE1_FREEZE_TAG}; closeout commit {PHASE1_CLOSEOUT_COMMIT}; "
            f"7B frozen run {PHASE1_RUN_7B}; 32B frozen run {PHASE1_RUN_32B}; "
            f"bank content hash {PHASE1_BANK_CONTENT_HASH[:16]}...; bank jsonl sha "
            f"{PHASE1_BANK_JSONL_SHA[:16]}...; mechanism events "
            f"{PHASE1_MECHANISM_EVENT_IDS}. Phase 2 review target branch head "
            f"{PHASE2_REVIEW_TARGET}. Every Phase 1 file and evidence event is "
            f"immutable for Phase 2."
        ),
        input_artifacts=[
            _art(f"{ph1}/reports/evidence_events.jsonl"),
            _art(f"{ph1}/reports/frozen_7b/results.jsonl"),
            _art(f"{ph1}/reports/frozen_32b/results.jsonl"),
            _art(f"{ph1}/preregistration/PREFERENCE_PHASE1_FREEZE_RECORD.md"),
            _art("interpretability/preference/data/lab38_preference_bank.jsonl"),
            _art("interpretability/preference/data/lab38_preference_bank.meta.json"),
        ],
        parent_event_ids=["pref2-addendum-intake-v1"],
    )

    _register(
        event_id="pref2-phase1-reanalysis-v1",
        event_type="cpu_reanalysis_intake",
        scientific_tier="development",
        claim_summary=(
            "Pre-VM CPU reanalysis of the frozen Phase 1 record registered as "
            "development-tier evidence: pipeline validation exact 20/20 scenarios "
            "both models; censoring identity position+|content|=0.500 exact 14/14 "
            "at 32B; folded margins -0.2..-2.2 nats toward pole_0 in 23/24 cells; "
            "label-rank/reply-order aliased to display position 2320/2320; 7B RO "
            "constant-code 9/12; G-LEX refutes unconditional standalone string-"
            "probability explanations at development tier. Narrowed language per "
            "plan §0.5: contextual lexical or task priors remain untested."
        ),
        input_artifacts=_reanalysis_dir_artifacts(),
        parent_event_ids=["pref2-import-phase1-v1"],
        limitations=(
            "Development tier; scripts carry machine-specific absolute paths "
            "(portable refactor = P2-1); hidden pole labels are sign anchors, "
            "not visible causal slots."
        ),
    )

    hygiene = [
        (
            "pref2-hygiene-stale-bank-sha-v1",
            "Phase 1 freeze record cites bank jsonl sha "
            f"{PHASE1_FREEZE_RECORD_STALE_JSONL_SHA_PREFIX}... while the frozen "
            f"bank on disk and its meta.json carry {PHASE1_BANK_JSONL_SHA[:16]}...; "
            "scientific content hash matches "
            f"({PHASE1_BANK_CONTENT_HASH[:16]}...) so identity is intact — the "
            "record's jsonl byte-sha is stale (pre-final regeneration). Annotated "
            "here append-only; the historical record is never edited.",
            [_art(f"{ph1}/preregistration/PREFERENCE_PHASE1_FREEZE_RECORD.md"),
             _art("interpretability/preference/data/lab38_preference_bank.jsonl")],
        ),
        (
            "pref2-hygiene-7b-capture-seal-v1",
            "capture_seal_status=unverifiable_null_sha: the 7B sealed capture "
            "(decision_residuals.pt on Drive, phase1/part1) was registered with a "
            "null SHA and no committed manifest; the Drive object is retained "
            "as-is. Phase 2 captures are the first verifiable seal (per-shard "
            "SHA256 + committed manifests, addendum C2).",
            [],
        ),
        (
            "pref2-hygiene-32b-captures-v1",
            "32B frozen captures: absent_by_design — the Phase 1 case study "
            "captured at run time only and no frozen 32B capture manifest exists. "
            "Phase 2 recaptures at aligned sites with manifests.",
            [],
        ),
        (
            "pref2-hygiene-readme-status-v1",
            "Campaign README refreshed at Phase 2 intake: records the plan "
            "supersession chain, Phase 2 governance, and non-frozen status "
            "(commit e9dc057). Stale 'Phase 2 decision optional' language from "
            "the Phase 1 closeout is superseded by the Phase 2 opening.",
            [_art("interpretability/preference/README.md")],
        ),
        (
            "pref2-hygiene-mechanism-controls-v1",
            "Mechanism-control disclosure for the Phase 1 PC case study, quoted "
            "from the frozen record: wrong-scenario addition moved the AR margin "
            "~62% of the primary contrast (0.49 vs 0.787 nats); the RO-fitted "
            "source was never given the identifiability gate yet moved AR "
            "strongly while nearly orthogonal to the AR direction; zero strict "
            "output flips anywhere; intervention site was the final prompt token "
            "(most output-adjacent). Phase 1 language stands as margin-moving "
            "instrument pilot; Phase 2 must earn semantic specificity, upstream "
            "propagation, codebook transfer, and strict-choice movement "
            "separately.",
            [_art(f"{ph1}/reports/frozen_32b/mechanism/mechanism_summary.json"),
             _art(f"{ph1}/reports/frozen_32b/mechanism/mech_pc_control.json")],
        ),
        (
            "pref2-hygiene-old-phase2-status-v1",
            "The pre-replacement preference_2_2.md simultaneously declared the "
            "pre-VM CPU program complete and scheduled P2-0..P2-4 as remaining. "
            "Accurate boundary of record: Phase 1 CPU reanalysis complete; Phase "
            "2 assay implementation, preregistration/freeze, and GPU campaign "
            "not complete at branch intake. Both superseded proposals preserved "
            "in Git history.",
            [],
        ),
        (
            "pref2-hygiene-package-absent-v1",
            "At branch intake (review target " + PHASE2_REVIEW_TARGET[:12] +
            "...), the Phase 2 delta contained only proposals, one CPU "
            "reanalysis, one lexical probe, one tokenizer-only port audit, and "
            "README/gitignore edits — no phase2 package, CLI, tests, v3 banks, "
            "review packets, render audits, registry, preregistration, freeze, "
            "frozen configs, or Phase 2 model output. This event marks the "
            "package-construction start.",
            [],
        ),
    ]
    for event_id, summary, arts in hygiene:
        _register(
            event_id=event_id,
            event_type="hygiene",
            scientific_tier="instrument",
            claim_summary=summary,
            input_artifacts=arts,
            parent_event_ids=["pref2-import-phase1-v1"],
        )

    _register(
        event_id="pref2-forensic-review-v1",
        event_type="forensic_review",
        scientific_tier="instrument",
        claim_summary=(
            "Part I forensic review adopted as the phase decision: Phase 1 was a "
            "successful instrument campaign whose generated choice is well "
            "approximated by a thresholded surface policy (position + |semantic "
            "choice effect| = 0.500 at 32B in saturated cells); the folded margin "
            "is an output-distribution endpoint, not yet twelve stable defaults; "
            "hidden pole labels are sign anchors, not causal slots — the "
            "load-bearing Phase 2 deconfound is a randomized context reversal. "
            "Proceed with E1 surface decomposition, E2 semantic defaults + "
            "context ladders, E3 scenario-local handles, E4 disjoint-surface "
            "report coupling; behavioral map across four models, causal work "
            "conditional; completeness = OLMo-32B spine (addendum E17)."
        ),
        input_artifacts=[
            _art(f"{plans}/preference_2_2.md"),
            _art(f"{plans}/preference_2_2_addendum.md"),
        ],
        parent_event_ids=["pref2-phase1-reanalysis-v1"],
    )

    print("P2-0 bootstrap events complete.")


if __name__ == "__main__":
    main()
