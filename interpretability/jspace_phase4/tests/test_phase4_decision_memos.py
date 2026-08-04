import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_p4p2_review_refuses_to_treat_analyzer_as_gpu_producer():
    text = (ROOT / "reviews" / "P4_P2_GPU_PRODUCER_REVIEW.md").read_text()
    assert "not a GPU intervention producer" in text
    assert "complete 160-row grid" in text
    assert "execution_authorized_at_this_boundary` remains `false`" in text


def test_p4p2_sesoi_is_prospective_and_cannot_follow_pilot_mean():
    text = (
        ROOT / "preregistration" / "P4_P2_SESOI_MEMO_DRAFT.md"
    ).read_text()
    assert "before any P4-P2 phase-intervention pilot outcome existed" in text
    assert "0.20 accuracy" in text
    assert "observed mean is forbidden" in text
    assert "at least 0.80 power" in text


def test_bank_b_decision_keeps_untouched_partitions_sealed():
    text = (ROOT / "reviews" / "BANK_B_PHASE4_DECISION.md").read_text()
    assert "estimation-only future instrument-development resource" in text
    assert "confirmatory and replication partitions remain sealed" in text
    assert "primary family contains P4-P2 and P4-P3 only" in text
    assert "3,562 independent families" in text


def test_candidate_013_records_terminal_branch_without_claiming_a_freeze():
    text = (
        ROOT / "preregistration"
        / "SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md"
    ).read_text()
    normalized = " ".join(text.split())
    assert "Version: candidate 0.13" in normalized
    assert "supersedes candidate 0.12" in normalized
    assert "CANDIDATE — NOT FROZEN" in normalized
    assert "Q-L4" in normalized
    assert "no single sparse Qwen lens is nominated" in normalized
    assert "P4-P1 estimation-only" in normalized
    assert "removes P4-P2" in normalized
    assert "P4-P3 blocked" in normalized
    assert "zero opened tests" in normalized
    assert "untouched partitions remain sealed" in normalized
    assert "planning alpha `0.05/3`" in normalized
    assert "No A2000 branch exists" in normalized
    assert "fresh independent Drive rematerialization" in normalized


def test_freeze_ledger_is_explicitly_unfrozen_and_routes_all_primaries():
    text = (
        ROOT / "preregistration" / "FREEZE_GATE_LEDGER_PHASE4.md"
    ).read_text()
    normalized = " ".join(text.split())
    assert "CANDIDATE LEDGER — NOT FROZEN" in normalized
    assert "P4-P1" in text and "ESTIMATION-ONLY" in text
    assert "P4-P2" in text and "REMOVED / BLOCKED BY Q-L4" in text
    assert "P4-P3" in text and "BLOCKED" in text
    assert "418/419" in normalized
    assert "one known deficit" in normalized
    assert "zero unexpected deficits" in normalized
    assert "Exact A120 capacity" in normalized and "RECOVERED" in normalized
    assert "Historical A120--A250 `state.json`" in normalized
    assert "The implementation agent has not self-signed" in text
    assert "Freeze commit / tag" in text and "NOT CREATED" in text


def test_methods_record_and_paper_skeleton_include_invariance_boundary():
    record = (
        ROOT / "paper" / "PHASE4_METHODS_DECISION_RECORD.md"
    ).read_text()
    skeleton = (ROOT / "paper" / "PAPER_CONCLUSION_SKELETON.md").read_text()
    assert "Operator convergence is not instrument invariance" in record
    assert "A1000 is the terminal automatic fit size" in record
    assert "P4-P3 is capability-blocked" in record
    assert "6. **Structural convergence of an averaged transport operator" \
        in skeleton
    assert "mechanical result is Q-L4" in skeleton


def test_parallel_import_inventory_records_admission_without_tier_upgrade():
    text = (ROOT / "manifests" / "parallel_import_inventory.md").read_text()
    normalized = " ".join(text.split())
    assert "p4-import-olmo-bank-w-capability-v1" in text
    assert "p4-import-gemma-transport-v1" in text
    assert "p4-import-olmo-lineage-final-v1" in text
    assert "Registered" in text
    assert "side-development-import" in normalized
    assert "O5 has no identifiable estimand" in text
    assert ("contains no native `ol-*`, `ol2-*`, `gm-*`, or `gm2-*` "
            "evidence IDs" in " ".join(text.split()))
    assert "Methods blocker" in text
    assert "do not" in text and "upgrade tiers" in text


def test_a120_state_search_records_negative_evidence_without_synthesis():
    text = (
        ROOT / "reviews" / "A120_STATE_EXACT_COPY_SEARCH_20260802.md"
    ).read_text()
    normalized = " ".join(text.split())
    assert "No exact bytes were found and no file was restored" in normalized
    assert "sixty `state.json` files" in normalized
    assert "1,246 revisions" in text
    assert "`deleted_items` table is empty" in text
    assert "does not authorize reconstruction" in normalized
    assert "independent reviewer and the PI" in normalized


def test_phase43_governing_sources_are_hash_pinned_and_adopted():
    expected = {
        "jspace_lab_nextsteps_4_3.md": (
            "1edc4f13201ea2fc9d866fbb5ebe6588194b1b8496e6a6872c7044228d2afc16"),
        "jspace_lab_nextsteps_4_3_addendum.md": (
            "79816a5ee5fb9cda72be1bbc510aa4937c685717191318334ca579fd21cd96c7"),
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / "reviews" / name).read_bytes()).hexdigest() \
            == digest

    adoption = (ROOT / "reviews" / "PHASE4_PLAN_ACCEPTED.md").read_text()
    assert "Phase 4.3 accepted" in adoption
    assert "Q-L1 through Q-L5" in adoption
    assert "16/20 common-support" in adoption
    assert all(digest in adoption for digest in expected.values())


def test_vm13_restart_snapshot_preserves_development_boundary():
    text = (ROOT / "reports" / "INPROGRESS_VM13_20260802.md").read_text()
    assert "Phase 4 remains **development-only**" in text
    latest_match = re.search(
        r"\| prompts banked \| (\d+) / 1000 \|", text)
    assert latest_match is not None
    latest = int(latest_match.group(1))
    assert 641 <= latest <= 1000
    assert f"Atomic checkpoints through n={latest}" in text
    if latest < 1000:
        assert f"active atomic chunk {latest}:{min(latest + 3, 1000)}" in text
    else:
        assert "registered scientific evidence" in text
        assert "process state | complete" in text
    assert "e0d0d31" in text
    assert "No confirmatory or replication intervention outcome" in text
