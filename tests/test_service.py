from pathlib import Path

import pymupdf
import pytest

from factlayer.config import Config
from factlayer.schema import DocumentStatus, FailureStage, Predicate
from factlayer.service import FactLayerService, discover_pdfs, write_json


def offline_config(tmp_path: Path) -> Config:
    return Config(
        db_path=tmp_path / "facts.db",
        upload_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
        llm_model="unused-offline-model",
        api_key=None,
        no_llm=True,
    )


def save_fact_pdf(path: Path, value: int = 120) -> None:
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text(
        (50, 100),
        f"Revenue from operations for FY24 stood at INR {value} million.",
        fontsize=11,
    )
    document.save(path)
    document.close()


def test_discover_pdfs_expands_directories_in_stable_order(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    save_fact_pdf(tmp_path / "b.PDF")
    save_fact_pdf(nested / "a.pdf")
    (tmp_path / "notes.txt").write_text("not a PDF")

    discovered = discover_pdfs([tmp_path, nested / "a.pdf"])

    assert {item.name for item in discovered} == {"a.pdf", "b.PDF"}
    assert len(discovered) == 2


def test_discover_pdfs_rejects_missing_and_empty_inputs(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="does not exist"):
        discover_pdfs([tmp_path / "missing.pdf"])
    with pytest.raises(ValueError, match="contains no PDF"):
        discover_pdfs([empty])


def test_offline_pipeline_stores_grounded_facts_and_reports_duplicates(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "report.pdf"
    save_fact_pdf(pdf)
    service = FactLayerService(offline_config(tmp_path))
    progress: list[str] = []

    first = service.ingest_paths([pdf], progress=progress.append)
    repeated = service.ingest_paths([pdf])

    assert first.failed is False
    assert first.documents[0].status == DocumentStatus.COMPLETE.value
    assert first.documents[0].extraction_mode == "offline"
    assert first.documents[0].counts.passages == 1
    assert first.documents[0].counts.facts_created >= 1
    assert first.documents[0].counts.provider_calls == 0
    assert repeated.documents[0].duplicate is True
    assert repeated.documents[0].counts.facts_reused >= 1
    assert service.status()["documents"] == 1
    assert service.status()["facts"] == first.documents[0].counts.facts_created
    assert any(message.startswith("Finished report.pdf") for message in progress)

    fact = service.store.list_facts()[0]
    evidence = fact.evidence[0]
    passage = service.store.get_passage(evidence.passage_id)
    assert passage is not None
    assert evidence.quote in passage.text


def test_reprocess_replaces_results_without_duplicating_the_document(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    save_fact_pdf(pdf)
    service = FactLayerService(offline_config(tmp_path))
    first = service.ingest_paths([pdf]).documents[0]

    rerun = service.ingest_paths([pdf], reprocess=True).documents[0]

    assert rerun.reprocessed is True
    assert rerun.document_id == first.document_id
    assert len(service.store.list_documents()) == 1
    assert rerun.counts.facts_created == first.counts.facts_created


def test_report_is_json_serializable_and_can_be_written(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    save_fact_pdf(pdf)
    service = FactLayerService(offline_config(tmp_path))
    service.ingest_paths([pdf])

    report = service.report()
    target = write_json(report, tmp_path / "output" / "report.json")

    assert report["summary"]["documents"] == 1
    assert report["summary"]["facts"] >= 1
    assert target.is_file()
    assert '"verified": true' in target.read_text()


def test_invalid_pdf_returns_a_failed_run_and_non_sensitive_error(tmp_path: Path) -> None:
    invalid = tmp_path / "broken.pdf"
    invalid.write_bytes(b"not a PDF")
    service = FactLayerService(offline_config(tmp_path))

    run = service.ingest_paths([invalid])

    assert run.failed is True
    assert run.documents[0].status == DocumentStatus.FAILED.value
    assert run.documents[0].document_id is None
    assert "PDF header" in run.documents[0].error


def test_safe_error_redacts_credential_shaped_values() -> None:
    message = FactLayerService._safe_error(ValueError("api_key=should-not-appear timeout"))

    assert "should-not-appear" not in message
    assert "[redacted]" in message


def test_one_predicate_conflict_does_not_discard_the_whole_document(
    tmp_path: Path,
) -> None:
    """A single mismatched fact should cost that fact, not every fact after it.

    A real hundred page annual report hit this: a measure normally reported as a number
    turned up once as a category, the store refused the write to protect the invariant,
    and the unguarded exception aborted the ingest. The document came out marked Failed
    with the facts extracted before that point silently thrown away.
    """
    pdf = tmp_path / "report.pdf"
    save_fact_pdf(pdf)
    service = FactLayerService(offline_config(tmp_path))

    # Claim the predicate as a different value kind than the PDF will produce, so the
    # conflict the store guards against is guaranteed to fire during ingest.
    service.store.put_predicate(
        Predicate(
            key="revenue_from_operations",
            display_name="Revenue from operations",
            value_kind="category",
        )
    )

    run = service.ingest_paths([pdf]).documents[0]

    assert run.status != DocumentStatus.FAILED.value
    assert run.error is None
    conflicts = [
        failure
        for failure in service.store.list_failures()
        if "already stored as" in failure.reason
    ]
    assert conflicts, "the refused fact should be recorded as a failure, not swallowed"
    assert conflicts[0].stage is FailureStage.NORMALIZATION
    # The invariant the store was protecting still holds.
    assert service.store.get_predicate("revenue_from_operations").value_kind == "category"
