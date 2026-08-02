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


def test_candidate_010_integrates_decisions_without_claiming_a_freeze():
    text = (
        ROOT / "preregistration"
        / "SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md"
    ).read_text()
    normalized = " ".join(text.split())
    assert "Version: candidate 0.10" in normalized
    assert "CANDIDATE — NOT FROZEN" in normalized
    assert "P4-P1 is removed from the Phase 4 primary family" in normalized
    assert "confirmatory and replication sides remain sealed" in normalized
    assert (
        "Holm correction covers the two active P4 primary p-values"
        in normalized
    )
    assert "No branch may be selected while" in normalized
    assert "the A1000 hash is a placeholder" in normalized
