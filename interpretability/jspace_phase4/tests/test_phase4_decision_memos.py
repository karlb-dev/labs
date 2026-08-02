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


def test_candidate_011_integrates_decisions_without_claiming_a_freeze():
    text = (
        ROOT / "preregistration"
        / "SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md"
    ).read_text()
    normalized = " ".join(text.split())
    assert "Version: candidate 0.11" in normalized
    assert "CANDIDATE — NOT FROZEN" in normalized
    assert "P4-P1 is removed from the Phase 4 primary family" in normalized
    assert "confirmatory and replication sides remain sealed" in normalized
    assert "P4-P1 is removed and P4-P3 is capability-blocked" in normalized
    assert "precommitted alpha `0.05/3`" in normalized
    assert "No branch may be selected while" in normalized
    assert "the A1000 hash is a placeholder" in normalized
    assert "P4-P3 is blocked" in normalized
    assert "only remaining conditional candidate primary" in normalized
    assert "one-shot consumed-development orthogonal" in normalized


def test_freeze_ledger_is_explicitly_unfrozen_and_routes_all_primaries():
    text = (
        ROOT / "preregistration" / "FREEZE_GATE_LEDGER_PHASE4.md"
    ).read_text()
    normalized = " ".join(text.split())
    assert "CANDIDATE LEDGER — NOT FROZEN" in normalized
    assert "P4-P1" in text and "REMOVED FROM PRIMARY" in text
    assert "P4-P2" in text and "ONLY CONDITIONAL CANDIDATE PRIMARY" in text
    assert "P4-P3" in text and "BLOCKED" in text
    assert "218/220" in text
    assert "The implementation agent has not and will not self-sign" in text
    assert "Freeze commit / tag" in text and "NOT CREATED" in text


def test_methods_record_and_paper_skeleton_include_invariance_boundary():
    record = (
        ROOT / "paper" / "PHASE4_METHODS_DECISION_RECORD.md"
    ).read_text()
    skeleton = (ROOT / "paper" / "PAPER_CONCLUSION_SKELETON.md").read_text()
    assert "Operator convergence is not instrument invariance" in record
    assert "A1000 is the last automatic fit-size escalation" in record
    assert "P4-P3 is capability-blocked" in record
    assert "6. **Structural convergence of the averaged transport operator" \
        in skeleton
    assert "instrument invariance must be tested" in skeleton


def test_parallel_import_inventory_keeps_validation_distinct_from_admission():
    text = (ROOT / "manifests" / "parallel_import_inventory.md").read_text()
    normalized = " ".join(text.split())
    assert "p4-import-olmo-bank-w-capability-v1" in text
    assert "p4-import-gemma-transport-v1" in text
    assert "VALIDATED / NOT REGISTERED" in text
    assert "O2 and O3 geometry/figures complete" in normalized
    assert "No final release bundle yet" in text
    assert "methods blocker" in text
    assert "never a license" in text
