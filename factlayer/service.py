"""Shared orchestration for document processing, linking, and reports."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from factlayer.config import Config
from factlayer.consistency import ConsistencyChecker
from factlayer.entities import EntityResolver
from factlayer.extract_llm import LlmExtractionStats, LlmExtractor, build_llm_extractor
from factlayer.extract_rules import (
    DeterministicExtractor,
    ExtractionStats,
    detect_reporting_entity,
)
from factlayer.ingest import IngestedPdf, PdfIngestor
from factlayer.link import FactLinker
from factlayer.schema import (
    Document,
    DocumentStatus,
    EntityReference,
    ExtractionFailure,
    FailureStage,
)
from factlayer.store import FactStore


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PipelineCounts:
    pages: int = 0
    passages: int = 0
    tables: int = 0
    candidates_seen: int = 0
    facts_extracted: int = 0
    facts_created: int = 0
    facts_reused: int = 0
    failures: int = 0
    relations_created: int = 0
    relations_updated: int = 0
    cache_hits: int = 0
    provider_calls: int = 0


@dataclass(frozen=True)
class DocumentRun:
    path: str
    document_id: str | None
    filename: str
    status: str
    duplicate: bool = False
    reprocessed: bool = False
    extraction_mode: str = "offline"
    elapsed_ms: float = 0
    counts: PipelineCounts = field(default_factory=PipelineCounts)
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchRun:
    documents: tuple[DocumentRun, ...]
    elapsed_ms: float

    @property
    def failed(self) -> bool:
        return any(item.status == DocumentStatus.FAILED.value for item in self.documents)

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": [item.as_dict() for item in self.documents],
            "elapsed_ms": self.elapsed_ms,
            "failed": self.failed,
        }


@dataclass(frozen=True)
class LinkRun:
    candidates: int
    created: int
    updated: int
    elapsed_ms: float
    consistency_findings: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FactLayerService:
    """Run the same fact pipeline for the CLI and the future HTTP API."""

    def __init__(
        self,
        config: Config,
        *,
        store: FactStore | None = None,
        ingestor: PdfIngestor | None = None,
        deterministic_extractor: DeterministicExtractor | None = None,
        llm_extractor: LlmExtractor | None = None,
        linker: FactLinker | None = None,
        checker: ConsistencyChecker | None = None,
    ):
        config.ensure_dirs()
        self.config = config
        self.store = store or FactStore(config.db_path)
        self.ingestor = ingestor or PdfIngestor()
        self.deterministic = deterministic_extractor or DeterministicExtractor()
        self.llm = llm_extractor or build_llm_extractor(config, cache=self.store)
        self.linker = linker or FactLinker()
        self.checker = checker or ConsistencyChecker(linker=self.linker)

    def ingest_paths(
        self,
        inputs: Iterable[Path | str],
        *,
        reprocess: bool = False,
        progress: ProgressCallback | None = None,
    ) -> BatchRun:
        started = time.perf_counter()
        paths = discover_pdfs(inputs)
        runs = tuple(
            self.ingest_document(path, reprocess=reprocess, progress=progress)
            for path in paths
        )
        return BatchRun(runs, (time.perf_counter() - started) * 1_000)

    def ingest_document(
        self,
        path: Path | str,
        *,
        reprocess: bool = False,
        original_filename: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> DocumentRun:
        started = time.perf_counter()
        source = Path(path)
        filename = original_filename or source.name or str(source)
        document: Document | None = None
        created: bool | None = None
        self._progress(progress, f"Inspecting {filename}")
        try:
            inspected = self.ingestor.inspect(source).model_copy(
                update={
                    "original_filename": filename,
                    "extraction_mode": self._extraction_mode,
                }
            )
            document, created = self.store.register_document(inspected)
            if not created and not reprocess:
                self._progress(progress, f"Skipping duplicate {filename}")
                return self._existing_run(source, document, started)
            if not created:
                self.store.clear_document_results(document.id)

            self.store.update_document_status(
                document.id,
                DocumentStatus.PROCESSING,
                extraction_mode=self._extraction_mode,
            )
            self._progress(progress, f"Reading pages from {filename}")
            ingested = self.ingestor.ingest(source)
            # An upload is stored under a generated content-hash name, so a second ingest
            # of that path reports the hash as the document's filename. Publisher is part
            # of every fact's context envelope, so leaving it would stamp a meaningless
            # name on everything the upload produced. Carry the registered document -
            # which already holds the name the user actually sent - into extraction.
            ingested = replace(ingested, document=document)
            for passage in ingested.passages:
                self.store.put_passage(passage)

            self._progress(progress, f"Extracting grounded facts from {filename}")
            # Prefer the organisation the document reports on. Falling back to the
            # document makes every fact its own island, because a document subject is
            # unique by construction and can never match anything in another file.
            reporting_entity = detect_reporting_entity(ingested)
            default_subject = reporting_entity or self._document_subject(document)
            if reporting_entity is not None:
                self._progress(
                    progress,
                    f"Attributing unqualified facts to {reporting_entity.canonical_name}",
                )
            deterministic = self.deterministic.extract_pdf(
                ingested,
                default_subject=default_subject,
            )
            llm = self.llm.extract_pdf(ingested, default_subject=default_subject)
            failures = (*deterministic.failures, *llm.failures)
            for failure in failures:
                self.store.put_failure(failure)

            resolver = EntityResolver.from_store(self.store)
            facts_created = 0
            facts_reused = 0
            relations_created = 0
            relations_updated = 0
            extracted_facts = (*deterministic.facts, *llm.facts)
            for extracted in extracted_facts:
                resolved, resolution = resolver.resolve_fact(extracted)
                self.store.put_entity(resolution.entity)
                try:
                    stored_fact, fact_created = self.store.put_fact(resolved)
                except ValueError as error:
                    # The store refuses to let one predicate mean two different kinds of
                    # value - a real annual report hit this after a hundred pages, when a
                    # measure normally reported as a number turned up once as a category,
                    # and the unguarded write aborted the whole document, discarding every
                    # fact after it even though nothing about them was wrong. One
                    # mismatched fact should cost only that fact.
                    evidence = resolved.evidence[0] if resolved.evidence else None
                    self.store.put_failure(
                        ExtractionFailure(
                            document_id=document.id,
                            passage_id=evidence.passage_id if evidence else None,
                            page_index=evidence.page_index if evidence else None,
                            stage=FailureStage.NORMALIZATION,
                            reason=str(error),
                            rejected_output={
                                "predicate": resolved.predicate.key,
                                "value_kind": resolved.value.kind,
                            },
                        )
                    )
                    continue
                facts_created += int(fact_created)
                facts_reused += int(not fact_created)
                if not fact_created:
                    continue
                for relation in self.linker.link_new_fact(self.store, stored_fact):
                    # link_new_fact persists each relation. A relation involving a newly
                    # created fact is necessarily new in a correctly maintained store.
                    if relation.id:
                        relations_created += 1

            warning_messages = tuple(
                dict.fromkeys(
                    [*ingested.document.warnings, *(failure.reason for failure in failures)]
                )
            )
            final_status = (
                DocumentStatus.PARTIAL
                if warning_messages
                else DocumentStatus.COMPLETE
            )
            self.store.update_document_status(
                document.id,
                final_status,
                page_count=ingested.document.page_count,
                warnings=list(warning_messages),
                extraction_mode=self._extraction_mode,
            )
            counts = self._pipeline_counts(
                ingested=ingested,
                deterministic=deterministic.stats,
                llm=llm.stats,
                facts_created=facts_created,
                facts_reused=facts_reused,
                failures=len(failures),
                relations_created=relations_created,
                relations_updated=relations_updated,
            )
            self._progress(
                progress,
                f"Finished {filename}: {counts.facts_created} new facts, "
                f"{counts.failures} failures",
            )
            return DocumentRun(
                path=str(source),
                document_id=document.id,
                filename=filename,
                status=final_status.value,
                reprocessed=not created,
                extraction_mode=self._extraction_mode,
                elapsed_ms=(time.perf_counter() - started) * 1_000,
                counts=counts,
                warnings=warning_messages,
            )
        except Exception as error:
            safe_error = self._safe_error(error)
            if document and document.id:
                failure = ExtractionFailure(
                    document_id=document.id,
                    stage=FailureStage.DOCUMENT,
                    reason=safe_error,
                    rejected_output={"error_type": type(error).__name__},
                    recoverable=False,
                )
                try:
                    self.store.put_failure(failure)
                    self.store.update_document_status(
                        document.id,
                        DocumentStatus.FAILED,
                        warnings=[safe_error],
                        extraction_mode=self._extraction_mode,
                    )
                except Exception:
                    pass
            self._progress(progress, f"Failed {filename}: {safe_error}")
            return DocumentRun(
                path=str(source),
                document_id=document.id if document else None,
                filename=filename,
                status=DocumentStatus.FAILED.value,
                reprocessed=created is False,
                extraction_mode=self._extraction_mode,
                elapsed_ms=(time.perf_counter() - started) * 1_000,
                counts=PipelineCounts(failures=1),
                error=safe_error,
            )

    def link_all(self, *, cross_document_only: bool = False) -> LinkRun:
        started = time.perf_counter()
        facts = self.store.list_facts()
        relations = self.linker.link_facts(
            facts,
            cross_document_only=cross_document_only,
        )
        created = 0
        updated = 0
        for relation in relations:
            _, was_created = self.store.put_relation(relation)
            created += int(was_created)
            updated += int(not was_created)

        # Recompute what the documents claim about their own numbers. Unlike linking,
        # which compares two statements, this checks a statement against arithmetic, so
        # it is the step that can surface a real internal contradiction.
        findings = (
            *self.checker.discover_growth_checks(facts),
            *self.checker.discover_component_total_checks(facts),
        )
        for finding in findings:
            self.checker.persist(self.store, finding)

        return LinkRun(
            candidates=len(relations),
            created=created,
            updated=updated,
            consistency_findings=len(findings),
            elapsed_ms=(time.perf_counter() - started) * 1_000,
        )

    def status(self) -> dict[str, Any]:
        documents = self.store.list_documents()
        facts = self.store.list_facts()
        relations = self.store.list_relations()
        failures = self.store.list_failures()
        findings = self.store.list_consistency_findings()
        return {
            "database": str(self.config.db_path),
            "documents": len(documents),
            "document_statuses": dict(Counter(item.status.value for item in documents)),
            "facts": len(facts),
            "relations": len(relations),
            "relation_types": dict(Counter(item.relation_type.value for item in relations)),
            "consistency_findings": len(findings),
            "failures": len(failures),
            "llm_enabled": self.config.llm_enabled,
            "extraction_mode": self._extraction_mode,
        }

    def report(self, *, document_id: str | None = None) -> dict[str, Any]:
        documents = self.store.list_documents()
        if document_id:
            selected = [item for item in documents if item.id == document_id]
            if not selected:
                raise KeyError(f"Document {document_id!r} was not found")
            documents = selected
        facts = self.store.list_facts(document_id=document_id)
        fact_ids = {fact.id for fact in facts}
        relations = [
            relation
            for relation in self.store.list_relations()
            if not document_id
            or relation.fact_a_id in fact_ids
            or relation.fact_b_id in fact_ids
        ]
        findings = [
            finding
            for finding in self.store.list_consistency_findings()
            if not document_id or any(item in fact_ids for item in finding.fact_ids)
        ]
        failures = self.store.list_failures(document_id=document_id)
        return {
            "summary": {
                "documents": len(documents),
                "facts": len(facts),
                "relations": len(relations),
                "consistency_findings": len(findings),
                "failures": len(failures),
            },
            "documents": [item.model_dump(mode="json") for item in documents],
            "facts": [item.model_dump(mode="json") for item in facts],
            "relations": [item.model_dump(mode="json") for item in relations],
            "consistency_findings": [
                item.model_dump(mode="json") for item in findings
            ],
            "failures": [item.model_dump(mode="json") for item in failures],
        }

    @property
    def _extraction_mode(self) -> str:
        return "hybrid" if self.config.llm_enabled else "offline"

    def _existing_run(
        self,
        path: Path,
        document: Document,
        started: float,
    ) -> DocumentRun:
        facts = self.store.list_facts(document_id=document.id)
        failures = self.store.list_failures(document_id=document.id)
        fact_ids = {fact.id for fact in facts}
        relations = [
            item
            for item in self.store.list_relations()
            if item.fact_a_id in fact_ids or item.fact_b_id in fact_ids
        ]
        return DocumentRun(
            path=str(path),
            document_id=document.id,
            filename=document.original_filename,
            status=document.status.value,
            duplicate=True,
            extraction_mode=document.extraction_mode,
            elapsed_ms=(time.perf_counter() - started) * 1_000,
            counts=PipelineCounts(
                pages=document.page_count,
                passages=len(self.store.list_passages(document.id)),
                facts_reused=len(facts),
                failures=len(failures),
                relations_updated=len(relations),
            ),
            warnings=tuple(document.warnings),
        )

    def _pipeline_counts(
        self,
        *,
        ingested: IngestedPdf,
        deterministic: ExtractionStats,
        llm: LlmExtractionStats,
        facts_created: int,
        facts_reused: int,
        failures: int,
        relations_created: int,
        relations_updated: int,
    ) -> PipelineCounts:
        return PipelineCounts(
            pages=ingested.document.page_count,
            passages=len(ingested.passages),
            tables=len(ingested.tables),
            candidates_seen=deterministic.candidates_seen + llm.candidates_seen,
            facts_extracted=deterministic.facts_accepted + llm.facts_accepted,
            facts_created=facts_created,
            facts_reused=facts_reused,
            failures=failures,
            relations_created=relations_created,
            relations_updated=relations_updated,
            cache_hits=llm.cache_hits,
            provider_calls=llm.provider_calls,
        )

    @staticmethod
    def _document_subject(document: Document) -> EntityReference:
        readable = re.sub(
            r"[-_]",
            " ",
            document.original_filename.rsplit(".", 1)[0],
        ).strip()
        return EntityReference(
            id=f"document:{document.id}",
            canonical_name=readable or document.original_filename,
            entity_type="document",
        )

    @staticmethod
    def _progress(callback: ProgressCallback | None, message: str) -> None:
        if callback:
            callback(message)

    @staticmethod
    def _safe_error(error: Exception) -> str:
        message = str(error).strip()
        if not message:
            message = type(error).__name__
        # Provider exceptions sometimes include request metadata. The CLI only needs a
        # useful category and bounded message, never credentials or a full response.
        message = re.sub(
            r"(?i)(api[_ -]?key|authorization|token)\s*[:=]\s*\S+",
            r"\1=[redacted]",
            message,
        )
        return message[:500]


def discover_pdfs(inputs: Iterable[Path | str]) -> tuple[Path, ...]:
    """Expand files and directories into a stable, duplicate-free PDF list."""
    discovered: dict[Path, Path] = {}
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            raise ValueError(f"Input path does not exist: {path}")
        if path.is_file() and path.suffix.casefold() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {path}")
        if path.is_dir():
            candidates = tuple(path.rglob("*"))
            if not any(
                candidate.is_file() and candidate.suffix.casefold() == ".pdf"
                for candidate in candidates
            ):
                raise ValueError(f"Directory contains no PDF files: {path}")
        else:
            candidates = (path,)
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.casefold() == ".pdf":
                resolved = candidate.resolve()
                discovered.setdefault(resolved, candidate)
    return tuple(discovered[key] for key in sorted(discovered, key=str))


def write_json(payload: dict[str, Any], output: Path | str) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "BatchRun",
    "DocumentRun",
    "FactLayerService",
    "LinkRun",
    "PipelineCounts",
    "discover_pdfs",
    "write_json",
]
