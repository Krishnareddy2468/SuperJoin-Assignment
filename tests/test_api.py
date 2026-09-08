import hashlib
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from factlayer.api import create_app
from factlayer.config import Config
from factlayer.service import FactLayerService


def offline_config(tmp_path: Path) -> Config:
    return Config(
        db_path=tmp_path / "facts.db",
        upload_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
        llm_model="unused-offline-model",
        api_key=None,
        no_llm=True,
    )


def pdf_bytes(tmp_path: Path, name: str, text: str) -> bytes:
    path = tmp_path / name
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_textbox(
        pymupdf.Rect(50, 60, 550, 400),
        text,
        fontsize=11,
    )
    document.save(path)
    document.close()
    return path.read_bytes()


@pytest.fixture
def api(tmp_path: Path) -> tuple[TestClient, FactLayerService, Config]:
    config = offline_config(tmp_path)
    service = FactLayerService(config)
    app = create_app(config, service=service, max_upload_bytes=2_000_000)
    return TestClient(app, raise_server_exceptions=False), service, config


def upload(client: TestClient, filename: str, content: bytes, content_type: str = "application/pdf"):
    return client.post(
        "/documents",
        files={"file": (filename, content, content_type)},
    )


def test_health_docs_and_browser_ui_are_served(api) -> None:
    client, _, _ = api

    health = client.get("/health")
    home = client.get("/")
    docs = client.get("/docs")
    schema = client.get("/openapi.json")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["extraction_mode"] == "offline"
    assert home.status_code == 200
    assert "Turn scattered PDFs" in home.text
    assert docs.status_code == 200
    assert "/documents" in schema.json()["paths"]
    assert "/facts/{fact_id}/evidence" in schema.json()["paths"]


def test_upload_stores_generated_name_and_duplicate_bytes_are_reused(
    api,
    tmp_path: Path,
) -> None:
    client, service, config = api
    content = pdf_bytes(
        tmp_path,
        "source.pdf",
        "Revenue from operations for FY24 stood at INR 120 million.",
    )

    first = upload(client, "../../Quarterly Report.pdf", content)
    repeated = upload(client, "renamed.pdf", content)

    assert first.status_code == 201
    assert first.json()["original_filename"] == "Quarterly Report.pdf"
    assert first.json()["duplicate"] is False
    assert first.json()["fact_count"] >= 1
    assert first.json()["extraction_mode"] == "offline"
    assert repeated.status_code == 200
    assert repeated.json()["duplicate"] is True
    assert repeated.json()["id"] == first.json()["id"]
    assert len(service.store.list_documents()) == 1
    stored = config.upload_dir / f"{hashlib.sha256(content).hexdigest()}.pdf"
    assert stored.read_bytes() == content
    assert "source.pdf" not in {item.name for item in config.upload_dir.iterdir()}
    assert str(tmp_path) not in first.text


def test_facts_evidence_download_and_reprocess_are_connected(
    api,
    tmp_path: Path,
) -> None:
    client, _, _ = api
    content = pdf_bytes(
        tmp_path,
        "report.pdf",
        "Revenue from operations for FY24 stood at INR 120 million.",
    )
    uploaded = upload(client, "report.pdf", content).json()
    document_id = uploaded["id"]

    documents = client.get("/documents", params={"limit": 1})
    detail = client.get(f"/documents/{document_id}")
    facts = client.get(
        f"/documents/{document_id}/facts",
        params={"predicate": "revenue_from_operations", "minimum_confidence": 0.5},
    )

    assert documents.json()["total"] == 1
    assert detail.json()["fact_count"] >= 1
    assert facts.status_code == 200
    assert facts.json()["total"] >= 1
    fact = facts.json()["items"][0]
    fact_detail = client.get(f"/facts/{fact['id']}")
    evidence = client.get(f"/facts/{fact['id']}/evidence")

    assert fact_detail.json()["confidence"] >= 0.5
    assert fact_detail.json()["raw_value"]
    assert evidence.json()["total"] >= 1
    citation = evidence.json()["items"][0]
    assert citation["verified"] is True
    assert citation["page_index"] == 0
    assert citation["quote_end"] > citation["quote_start"]

    downloaded = client.get(f"/documents/{document_id}/file")
    rerun = client.post(f"/documents/{document_id}/reprocess")
    assert downloaded.status_code == 200
    assert downloaded.content == content
    assert rerun.status_code == 200
    assert rerun.json()["reprocessed"] is True
    assert rerun.json()["id"] == document_id


@pytest.mark.parametrize(
    ("filename", "content", "content_type", "status", "code"),
    [
        ("notes.txt", b"plain text", "text/plain", 415, "unsupported_media_type"),
        ("fake.pdf", b"plain text", "application/pdf", 415, "unsupported_media_type"),
        ("fake.pdf", b"%PDF-1.7\nbroken", "text/plain", 415, "unsupported_media_type"),
    ],
)
def test_invalid_uploads_return_structured_safe_errors(
    api,
    filename: str,
    content: bytes,
    content_type: str,
    status: int,
    code: str,
) -> None:
    client, _, config = api

    response = upload(client, filename, content, content_type)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["detail"]
    assert not list(config.upload_dir.glob("*.part"))


def test_upload_limit_and_query_validation_return_clear_4xx(api) -> None:
    client, _, config = api
    limited = create_app(config, service=FactLayerService(config), max_upload_bytes=20)
    limited_client = TestClient(limited, raise_server_exceptions=False)

    too_large = upload(
        limited_client,
        "large.pdf",
        b"%PDF-1.7\n" + b"x" * 100,
    )
    bad_page = client.get("/documents", params={"page": 0, "limit": 900})

    assert too_large.status_code == 413
    assert too_large.json()["error"]["code"] == "upload_too_large"
    assert bad_page.status_code == 422
    assert bad_page.json()["error"]["code"] == "validation_error"
    assert "input" not in bad_page.text.casefold()


def test_broken_pdf_body_is_rejected_without_exposing_its_storage_path(api) -> None:
    client, _, config = api

    response = upload(client, "broken.pdf", b"%PDF-1.7\nthis is not a complete PDF")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unprocessable_document"
    assert str(config.upload_dir) not in response.text
    assert "traceback" not in response.text.casefold()
    assert not list(config.upload_dir.glob("*.pdf"))


def test_relation_filters_and_detail_include_both_grounded_facts(
    api,
    tmp_path: Path,
) -> None:
    client, _, _ = api
    statement = (
        "Mr. Suvir Suren Sujan, Non-Executive Director (DIN: 01173669), "
        "resigned from the Board with effect from August 24, 2023."
    )
    first = pdf_bytes(tmp_path, "first.pdf", statement)
    second = pdf_bytes(tmp_path, "second.pdf", statement + " This filing confirms the change.")
    first_document = upload(client, "first.pdf", first).json()
    upload(client, "second.pdf", second)

    relations = client.get(
        "/relations",
        params={
            "document_id": first_document["id"],
            "entity_id": "din:01173669",
            "minimum_confidence": 0.5,
            "limit": 10,
        },
    )

    assert relations.status_code == 200
    assert relations.json()["total"] >= 1
    relation = relations.json()["items"][0]
    assert relation["fact_a"]["evidence"][0]["verified"] is True
    assert relation["fact_b"]["evidence"][0]["verified"] is True
    detail = client.get(f"/relations/{relation['id']}")
    assert detail.status_code == 200
    assert detail.json()["explanation"]
    assert detail.json()["rule_version"]


def test_missing_resources_never_expose_server_paths(api, tmp_path: Path) -> None:
    client, _, _ = api

    response = client.get("/facts/not-a-real-fact")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert str(tmp_path) not in response.text
    assert "traceback" not in response.text.casefold()


def test_uploaded_facts_credit_the_filename_the_user_sent(api, tmp_path: Path) -> None:
    """Publisher is part of every fact's context, so it has to be the real name.

    An upload is stored under a generated content-hash name. Re-ingesting that path
    reported the hash as the document's filename, which then got stamped onto every
    fact the upload produced and shown in the UI as the publisher.
    """
    client, _, _ = api
    content = pdf_bytes(
        tmp_path,
        "quarterly-update.pdf",
        "Revenue from operations stood at 100 million for FY24.",
    )

    created = client.post(
        "/documents",
        files={"file": ("quarterly-update.pdf", content, "application/pdf")},
    )
    assert created.status_code == 201
    document_id = created.json()["id"]

    body = client.get(f"/documents/{document_id}/facts?limit=10").json()
    facts = body.get("items", body)
    assert facts, "the upload should produce at least one fact"
    publishers = {(fact.get("context") or {}).get("publisher") for fact in facts}
    assert publishers == {"quarterly-update.pdf"}


def test_the_browser_cannot_cache_the_page_or_its_assets(api) -> None:
    """A fixed UI must not keep showing old behaviour after a reload.

    The page is served by its own route rather than by the mounted static files, so it
    does not inherit their headers. It went out with only an ETag, which lets a browser
    apply heuristic freshness and reuse the cached copy without asking the server. The
    result looks like the fix never landed, and the wasted debugging goes into the code
    rather than into the cache. Assert every document the browser loads opts out.
    """
    client, _service, _config = api

    for path in ("/", "/static/app.js", "/static/index.html", "/static/styles.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers.get("cache-control") == "no-store", path
