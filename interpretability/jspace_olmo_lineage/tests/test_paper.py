import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports/paper"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_isolated_paper_source_and_pdf_are_complete():
    tex = PAPER / "olmo_lineage_parallel_phase.tex"
    pdf = PAPER / "olmo_lineage_parallel_phase.pdf"
    assert tex.is_file()
    assert pdf.read_bytes().startswith(b"%PDF")
    assert pdf.stat().st_size > 300_000
    assert _sha256(tex) == (
        "33e88825ce67d158e83328ee378b7847674d2ba234b541839a10b287ea75c656")
    assert _sha256(pdf) == (
        "02a81b87fff5fdce07726af341fdc80b0fe010e42c41ddc47c06a2e8d3240ec7")


def test_paper_uses_registered_geometry_figures():
    expected = {
        "olf01_operator_similarity_heatmap.pdf":
            "eb909560821e7eebbacabbe45e379d848165e52df012609fb166b106771c3abd",
        "olf02_token_row_similarity.pdf":
            "5a89c91d065ad72b2a99bb235c2e40269ea32d5c76e532212cca1021141a2d90",
        "olf03_selected_span_trajectory.pdf":
            "9ad83919a5e7c423ff0766d813a4199192a36134941c95b606e0228ee0ce30e3",
        "olf04_capacity_causal_state_space.pdf":
            "0fde6be99cbc11ee09945370862c38aeb65b3e7f4d773eeb12110a592ca78f47",
        "olf05_readout_transport_decomposition.pdf":
            "e60174c7c0af4468ffc918b6e86caa0e331de5b1498d339bb04bd8f5112f5c6d",
    }
    assert {path.name for path in (PAPER / "figures").glob("*.pdf")} == set(
        expected)
    for name, digest in expected.items():
        assert _sha256(PAPER / "figures" / name) == digest


def test_paper_preserves_claim_and_namespace_boundaries():
    tex = (PAPER / "olmo_lineage_parallel_phase.tex").read_text()
    script = (PAPER / "compile.sh").read_text()
    assert "16/20" in tex
    assert "dictionary-formation-pattern" in tex
    assert "not-executed-no-proxy-substitution" in tex
    assert "shared Phase 4 or Gemma papers" in tex
    assert "SOURCE_DATE_EPOCH=1785648410" in script
    assert "interpretability/jspace_paper" not in script
