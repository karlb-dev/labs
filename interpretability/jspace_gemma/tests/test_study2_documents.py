import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_study1_release_documents_remain_byte_immutable():
    expected = {
        "GEMMA_TRANSPORT_STATE_OF_RECORD.md": (
            "0eeab02938c32839a8ca19e2446bfc3b69bfcb18769428afb6904133363fa344"
        ),
        "gemma_transport_claim_ledger.md": (
            "812a45a5044069baf41d8db276b1d59f0c89dd76acb790e9e0e6506c5cccd779"
        ),
        "TRANSPORT_GATE_PROTOCOL.md": (
            "f1d83baa36a41623ba4a5990cd83f09bf8e9e09690c1b95c5d5ce7e716e8b768"
        ),
    }
    for name, digest in expected.items():
        assert _sha(ROOT / "release" / name) == digest


def test_study2_documents_preserve_relicense_and_claim_boundaries():
    paths = [
        ROOT / "reports/GEMMA_TRANSPORT_STUDY2_REPORT.md",
        ROOT / "release/GEMMA_TRANSPORT_STATE_OF_RECORD_V2.md",
        ROOT / "release/gemma_transport_claim_ledger_v2.md",
    ]
    required = (
        "0.07870368901355948",
        "0.0024581113830208778",
        "branch_1_relicense_without_recompute",
        "local_tangent_mismatch",
    )
    for path in paths:
        text = path.read_text()
        for value in required:
            assert value in text
        assert "nondifferentiability" in text
        assert "workspace" in text

    report = " ".join(paths[0].read_text().split())
    assert "No G2.2 model compute" in report
    assert "target-isolated" in report
    state = paths[1].read_text()
    assert "neither edits nor retroactively passes" in state
    ledger = paths[2].read_text()
    assert "Study 1 correctly stopped" in ledger


def test_v2_protocol_preserves_target_firewall_and_secant_label():
    text = " ".join(
        (ROOT / "release/TRANSPORT_GATE_PROTOCOL_V2.md").read_text().split()
    )
    assert "threshold file atomically before opening the evidence registry" in text
    assert "Finite differences are secants" in text
    assert "Missing effects remain missing, not zero" in text
    assert "must not edit or withdraw the earlier failed event" in text


def test_study2_handout_is_deterministic_and_contains_registered_figure():
    handout = ROOT / "reports/handout"
    expected = {
        "gemma_transport_development.tex": (
            "afaf90af4f3b96f5e1b267607e52839725a38cbf25b4713e343863d0383187e2"
        ),
        "gemma_transport_development.pdf": (
            "518b1fac1997469e364ebe641fcc40b99b513c0a57144a47668793a3882fd5c9"
        ),
        "figures/gm2_backend_disagreement_by_model_batch.png": (
            "9c21f13c90a8b9d3d0325a699e025f1267dde8548bc3e74eb6f592ff0bb773ff"
        ),
    }
    for relative, digest in expected.items():
        assert _sha(handout / relative) == digest
