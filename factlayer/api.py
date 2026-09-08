"""FastAPI interface for uploads and evidence-first result inspection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Any, Iterable
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from factlayer import __version__
from factlayer.config import Config, load_config
from factlayer.ingest import DEFAULT_MAX_PDF_BYTES
from factlayer.schema import (
    Document,
    DocumentStatus,
    Fact,
    FailureStage,
    Relation,
    RelationType,
    ReviewState,
)
from factlayer.service import DocumentRun, FactLayerService


STATIC_DIR = Path(__file__).with_name("static")
UPLOAD_CHUNK_BYTES = 1024 * 1024
ALLOWED_PDF_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}


class _NoCacheStaticFiles(StaticFiles):
    """Serve app.js and friends without letting the browser cache a stale copy.

    Starlette's default StaticFiles sends Last-Modified and ETag but no Cache-Control,
    which lets a browser apply its own heuristic caching and reuse an old response
    without even asking the server. That is invisible and confusing during active
    development: the file on disk and the running server are both correct, but the
    page still shows old behaviour after a plain reload. This project has no build
    step and no CDN in front of it, so there is nothing to gain from caching here.
    """

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response


def create_app(
    config: Config | None = None,
    *,
    service: FactLayerService | None = None,
    max_upload_bytes: int = DEFAULT_MAX_PDF_BYTES,
) -> FastAPI:
    """Build an isolated application, which keeps API tests and deployments configurable."""
    if max_upload_bytes <= 0:
        raise ValueError("The upload limit must be positive")
    runtime = service or FactLayerService(config or load_config())
    app = FastAPI(
        title="FactLayer API",
        version=__version__,
        description="Upload PDFs and inspect grounded facts, evidence, and relationships.",
    )
    app.state.service = runtime
    app.state.max_upload_bytes = max_upload_bytes
    app.mount("/static", _NoCacheStaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError):
        fields = [
            ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            for item in error.errors()
        ]
        message = "One or more request values are invalid."
        if visible := [field for field in fields if field]:
            message = f"Invalid request value: {', '.join(dict.fromkeys(visible))}."
        return _error_response(422, "validation_error", message)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, error: HTTPException):
        detail = error.detail if isinstance(error.detail, str) else "The request could not be completed."
        return _error_response(error.status_code, _error_code(error.status_code), detail)

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, _error: Exception):
        return _error_response(
            500,
            "internal_error",
            "The request could not be completed. Check the server logs for details.",
        )

    @app.get("/", include_in_schema=False)
    async def browser_ui():
        # The page is served by this route, not by the mounted static files, so it does
        # not inherit their no-store header and was going out with only an ETag. A browser
        # is then free to reuse the cached page without asking, which is how a fixed UI
        # keeps showing old behaviour after a normal reload. Same reasoning as
        # _NoCacheStaticFiles; the shell has to opt in the same way the assets do.
        return FileResponse(
            STATIC_DIR / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "schema_version": runtime.store.schema_version,
            "llm_enabled": runtime.config.llm_enabled,
            "extraction_mode": "hybrid" if runtime.config.llm_enabled else "offline",
        }

    @app.post("/documents")
    async def upload_document(
        response: Response,
        file: Annotated[UploadFile, File(description="A PDF document")],
    ) -> dict[str, Any]:
        original_filename = _safe_filename(file.filename)
        _validate_upload_metadata(file, original_filename)
        stored_path = await _save_upload(
            file,
            runtime.config.upload_dir,
            max_upload_bytes=max_upload_bytes,
        )
        run = await run_in_threadpool(
            runtime.ingest_document,
            stored_path,
            original_filename=original_filename,
        )
        if run.error:
            if not run.document_id:
                stored_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=422,
                    detail="The uploaded PDF could not be opened or validated.",
                )
            raise HTTPException(
                status_code=500,
                detail="Document processing failed. Its failure record is available for review.",
            )
        response.status_code = 200 if run.duplicate else 201
        return _document_run_payload(runtime, run)

    @app.get("/documents")
    async def documents(
        page: Annotated[int, Query(ge=1)] = 1,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        status: DocumentStatus | None = None,
        extraction_mode: Annotated[str | None, Query(pattern="^(offline|hybrid)$")] = None,
    ) -> dict[str, Any]:
        items = runtime.store.list_documents(status=status)
        if extraction_mode:
            items = [item for item in items if item.extraction_mode == extraction_mode]
        return _page(
            [_document_payload(runtime, item) for item in items],
            page,
            limit,
        )

    @app.get("/documents/{document_id}")
    async def document_detail(document_id: str) -> dict[str, Any]:
        return _document_payload(runtime, _document_or_404(runtime, document_id))

    @app.get("/documents/{document_id}/facts")
    async def document_facts(
        document_id: str,
        page: Annotated[int, Query(ge=1)] = 1,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        entity_id: str | None = None,
        predicate: str | None = None,
        review_state: ReviewState | None = None,
        minimum_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
    ) -> dict[str, Any]:
        _document_or_404(runtime, document_id)
        items = runtime.store.list_facts(
            document_id=document_id,
            entity_id=entity_id,
            predicate_key=predicate,
            review_state=review_state,
            minimum_confidence=minimum_confidence,
        )
        return _page([_fact_payload(item) for item in items], page, limit)

    @app.get("/facts")
    async def facts(
        page: Annotated[int, Query(ge=1)] = 1,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        document_id: str | None = None,
        entity_id: str | None = None,
        predicate: str | None = None,
        review_state: ReviewState | None = None,
        minimum_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
    ) -> dict[str, Any]:
        if document_id:
            _document_or_404(runtime, document_id)
        items = runtime.store.list_facts(
            document_id=document_id,
            entity_id=entity_id,
            predicate_key=predicate,
            review_state=review_state,
            minimum_confidence=minimum_confidence,
        )
        return _page([_fact_payload(item) for item in items], page, limit)

    @app.get("/facts/{fact_id}")
    async def fact_detail(fact_id: str) -> dict[str, Any]:
        fact = runtime.store.get_fact(fact_id)
        if not fact:
            raise HTTPException(status_code=404, detail="Fact not found.")
        return _fact_payload(fact)

    @app.get("/facts/{fact_id}/evidence")
    async def fact_evidence(
        fact_id: str,
        page: Annotated[int, Query(ge=1)] = 1,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        if not runtime.store.get_fact(fact_id):
            raise HTTPException(status_code=404, detail="Fact not found.")
        evidence = runtime.store.list_evidence(fact_id=fact_id)
        return _page([item.model_dump(mode="json") for item in evidence], page, limit)

    @app.get("/relations")
    async def relations(
        page: Annotated[int, Query(ge=1)] = 1,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        document_id: str | None = None,
        entity_id: str | None = None,
        predicate: str | None = None,
        relation_type: RelationType | None = None,
        review_state: ReviewState | None = None,
        minimum_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
    ) -> dict[str, Any]:
        if document_id:
            _document_or_404(runtime, document_id)
        items = runtime.store.list_relations(
            relation_type=relation_type,
            review_state=review_state,
            minimum_confidence=minimum_confidence,
        )
        selected = _filter_relations(
            runtime,
            items,
            document_id=document_id,
            entity_id=entity_id,
            predicate=predicate,
        )
        return _page([_relation_payload(runtime, item) for item in selected], page, limit)

    @app.get("/relations/{relation_id}")
    async def relation_detail(relation_id: str) -> dict[str, Any]:
        relation = runtime.store.get_relation(relation_id)
        if not relation:
            raise HTTPException(status_code=404, detail="Relationship not found.")
        return _relation_payload(runtime, relation)

    @app.get("/failures")
    async def failures(
        page: Annotated[int, Query(ge=1)] = 1,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        document_id: str | None = None,
        stage: FailureStage | None = None,
    ) -> dict[str, Any]:
        if document_id:
            _document_or_404(runtime, document_id)
        items = runtime.store.list_failures(
            document_id=document_id,
            stage=stage.value if stage else None,
        )
        return _page([item.model_dump(mode="json") for item in items], page, limit)

    @app.post("/documents/{document_id}/reprocess")
    async def reprocess_document(document_id: str) -> dict[str, Any]:
        document = _document_or_404(runtime, document_id)
        source = _stored_pdf(runtime.config, document)
        if not source.is_file():
            raise HTTPException(
                status_code=409,
                detail="The original uploaded PDF is unavailable for reprocessing.",
            )
        run = await run_in_threadpool(
            runtime.ingest_document,
            source,
            reprocess=True,
            original_filename=document.original_filename,
        )
        if run.error:
            raise HTTPException(
                status_code=500,
                detail="Document reprocessing failed. Its failure record is available for review.",
            )
        return _document_run_payload(runtime, run)

    @app.get("/documents/{document_id}/file", include_in_schema=False)
    async def document_file(document_id: str):
        document = _document_or_404(runtime, document_id)
        source = _stored_pdf(runtime.config, document)
        if not source.is_file():
            raise HTTPException(status_code=404, detail="The uploaded PDF file is unavailable.")
        return FileResponse(
            source,
            media_type="application/pdf",
            filename=document.original_filename,
        )

    return app


def _document_or_404(service: FactLayerService, document_id: str) -> Document:
    document = service.store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


def _document_payload(service: FactLayerService, document: Document) -> dict[str, Any]:
    facts = service.store.list_facts(document_id=document.id)
    fact_ids = {item.id for item in facts}
    relations = [
        item
        for item in service.store.list_relations()
        if item.fact_a_id in fact_ids or item.fact_b_id in fact_ids
    ]
    payload = document.model_dump(mode="json")
    payload.update(
        {
            "fact_count": len(facts),
            "relation_count": len(relations),
            "failure_count": len(service.store.list_failures(document_id=document.id)),
        }
    )
    return payload


def _document_run_payload(service: FactLayerService, run: DocumentRun) -> dict[str, Any]:
    if not run.document_id:
        raise HTTPException(status_code=422, detail=run.error or "Document processing failed.")
    document = _document_or_404(service, run.document_id)
    payload = _document_payload(service, document)
    payload.update(
        {
            "duplicate": run.duplicate,
            "reprocessed": run.reprocessed,
            "elapsed_ms": run.elapsed_ms,
            "run_counts": run.counts.__dict__,
        }
    )
    return payload


def _fact_payload(fact: Fact) -> dict[str, Any]:
    payload = fact.model_dump(mode="json")
    payload.update(
        {
            "raw_value": fact.value.raw,
            "normalized_value": _normalized_value(fact),
            "confidence": min(fact.extraction_confidence, fact.normalization_confidence),
        }
    )
    return payload


def _normalized_value(fact: Fact) -> Any:
    value = fact.value.model_dump(mode="json")
    for key in ("number", "value", "state", "text"):
        if key in value:
            return value[key]
    return fact.value.raw


def _relation_payload(service: FactLayerService, relation: Relation) -> dict[str, Any]:
    left = service.store.get_fact(relation.fact_a_id)
    right = service.store.get_fact(relation.fact_b_id)
    payload = relation.model_dump(mode="json")
    payload["fact_a"] = _fact_payload(left) if left else None
    payload["fact_b"] = _fact_payload(right) if right else None
    return payload


def _filter_relations(
    service: FactLayerService,
    relations: Iterable[Relation],
    *,
    document_id: str | None,
    entity_id: str | None,
    predicate: str | None,
) -> list[Relation]:
    selected: list[Relation] = []
    for relation in relations:
        facts = [
            fact
            for fact in (
                service.store.get_fact(relation.fact_a_id),
                service.store.get_fact(relation.fact_b_id),
            )
            if fact
        ]
        if document_id and not any(
            evidence.document_id == document_id
            for fact in facts
            for evidence in fact.evidence
        ):
            continue
        if entity_id and not any(fact.subject.id == entity_id for fact in facts):
            continue
        if predicate and not any(fact.predicate.key == predicate for fact in facts):
            continue
        selected.append(relation)
    return selected


def _page(items: list[Any], page: int, limit: int) -> dict[str, Any]:
    start = (page - 1) * limit
    return {
        "items": items[start : start + limit],
        "page": page,
        "limit": limit,
        "total": len(items),
    }


def _safe_filename(filename: str | None) -> str:
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name:
        raise HTTPException(status_code=400, detail="A PDF filename is required.")
    return name[:500]


def _validate_upload_metadata(file: UploadFile, filename: str) -> None:
    if not filename.casefold().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only .pdf files are accepted.")
    content_type = (file.content_type or "").casefold()
    if content_type not in ALLOWED_PDF_TYPES:
        raise HTTPException(status_code=415, detail="The upload content type must be PDF.")


async def _save_upload(
    upload: UploadFile,
    upload_dir: Path,
    *,
    max_upload_bytes: int,
) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary = upload_dir / f"upload-{uuid4().hex}.part"
    digest = hashlib.sha256()
    size = 0
    first_chunk = True
    try:
        with temporary.open("xb") as destination:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                if first_chunk:
                    first_chunk = False
                    if not chunk.startswith(b"%PDF-"):
                        raise HTTPException(
                            status_code=415,
                            detail="The uploaded content does not have a valid PDF header.",
                        )
                size += len(chunk)
                if size > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"The PDF exceeds the {max_upload_bytes}-byte upload limit.",
                    )
                digest.update(chunk)
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        stored = upload_dir / f"{digest.hexdigest()}.pdf"
        if stored.exists():
            temporary.unlink(missing_ok=True)
        else:
            temporary.replace(stored)
        return stored
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _stored_pdf(config: Config, document: Document) -> Path:
    return config.upload_dir / f"{document.content_hash}.pdf"


def _error_code(status_code: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        409: "conflict",
        413: "upload_too_large",
        415: "unsupported_media_type",
        422: "unprocessable_document",
    }.get(status_code, "request_error")


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": message,
            "error": {"code": code, "message": message},
        },
    )


app = create_app()


__all__ = ["app", "create_app"]
