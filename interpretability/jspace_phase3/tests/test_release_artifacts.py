import pytest

from jspace_phase3.experiments import p3_release_manifest as manifest
from jspace_phase3.experiments import p3_release_publication as publication


def test_pdf_pages_parses_pdfinfo(monkeypatch, tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF")
    monkeypatch.setattr(
        publication.subprocess,
        "check_output",
        lambda *args, **kwargs: "Title: report\nPages:          7\n",
    )
    assert publication.pdf_pages(path) == 7


def test_pdf_pages_rejects_missing_page_count(monkeypatch, tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF")
    monkeypatch.setattr(
        publication.subprocess,
        "check_output",
        lambda *args, **kwargs: "Title: report\n",
    )
    with pytest.raises(RuntimeError, match="page count"):
        publication.pdf_pages(path)


def test_model_ref_requires_and_splits_revision_pin():
    assert manifest.model_ref("model://org/name@abc123") == {
        "hub_id": "org/name",
        "revision": "abc123",
    }
    with pytest.raises(ValueError, match="revision-pinned"):
        manifest.model_ref("model://org/name")


def test_file_record_hashes_existing_file(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"release artifact")
    record = manifest.file_record(path)
    assert record["path"] == str(path)
    assert record["bytes"] == len(b"release artifact")
    assert len(record["sha256"]) == 64
