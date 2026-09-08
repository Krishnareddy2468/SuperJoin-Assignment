"""SQLite persistence for the fact knowledge layer.

The store keeps SQL details out of extraction and linking code. Public methods
accept and return domain models, while transactions protect facts from ever
being left behind without their evidence.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from factlayer.schema import (
    BoundingBox,
    ConsistencyCheckType,
    ConsistencyFinding,
    ContextEnvelope,
    Document,
    DocumentStatus,
    Entity,
    EntityReference,
    Evidence,
    ExtractionFailure,
    ExtractionMethod,
    Fact,
    Passage,
    Predicate,
    PredicateReference,
    Relation,
    RelationType,
    ReviewState,
    ValueComparison,
)


SCHEMA_VERSION = 4


DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    original_filename TEXT NOT NULL,
    page_count INTEGER NOT NULL CHECK (page_count >= 0),
    status TEXT NOT NULL,
    extraction_mode TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS passages (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL CHECK (page_index >= 0),
    reading_order INTEGER NOT NULL CHECK (reading_order >= 0),
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL CHECK (char_start >= 0),
    char_end INTEGER NOT NULL CHECK (char_end > char_start),
    bbox_json TEXT,
    text_hash TEXT,
    UNIQUE(document_id, page_index, reading_order)
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    identifiers_json TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    review_state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predicates (
    key TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    value_kind TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    subject_id TEXT NOT NULL REFERENCES entities(id),
    predicate_key TEXT NOT NULL REFERENCES predicates(key),
    value_kind TEXT NOT NULL,
    value_json TEXT NOT NULL,
    context_json TEXT NOT NULL,
    extraction_confidence REAL NOT NULL CHECK (extraction_confidence BETWEEN 0 AND 1),
    normalization_confidence REAL NOT NULL CHECK (normalization_confidence BETWEEN 0 AND 1),
    review_state TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    passage_id TEXT NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL CHECK (page_index >= 0),
    quote TEXT NOT NULL,
    quote_start INTEGER NOT NULL CHECK (quote_start >= 0),
    quote_end INTEGER NOT NULL CHECK (quote_end > quote_start),
    bbox_json TEXT,
    extractor TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    verified INTEGER NOT NULL CHECK (verified = 1),
    UNIQUE(fact_id, passage_id, quote_start, quote_end)
);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    fact_a_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    fact_b_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    value_comparison_json TEXT NOT NULL,
    context_diff_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    rule_version TEXT NOT NULL,
    review_state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK(fact_a_id < fact_b_id),
    UNIQUE(fact_a_id, fact_b_id)
);

CREATE TABLE IF NOT EXISTS consistency_findings (
    id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    check_type TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    formula TEXT NOT NULL,
    operands_json TEXT NOT NULL,
    stated_result TEXT,
    calculated_result TEXT,
    difference TEXT,
    tolerance TEXT,
    explanation TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    rule_version TEXT NOT NULL,
    review_state TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS consistency_finding_facts (
    finding_id TEXT NOT NULL REFERENCES consistency_findings(id) ON DELETE CASCADE,
    fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY(finding_id, fact_id),
    UNIQUE(finding_id, position)
);

CREATE TABLE IF NOT EXISTS extraction_failures (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    passage_id TEXT REFERENCES passages(id) ON DELETE CASCADE,
    page_index INTEGER CHECK (page_index >= 0),
    stage TEXT NOT NULL,
    reason TEXT NOT NULL,
    rejected_output_json TEXT,
    recoverable INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    passage_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    prompt_version INTEGER NOT NULL,
    response_json TEXT NOT NULL,
    validation_result TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(passage_hash, model, schema_version, prompt_version)
);

CREATE INDEX IF NOT EXISTS idx_passages_document_page
    ON passages(document_id, page_index, reading_order);
CREATE INDEX IF NOT EXISTS idx_facts_subject_predicate
    ON facts(subject_id, predicate_key);
CREATE INDEX IF NOT EXISTS idx_facts_predicate_review
    ON facts(predicate_key, review_state);
CREATE INDEX IF NOT EXISTS idx_evidence_document_fact
    ON evidence(document_id, fact_id);
CREATE INDEX IF NOT EXISTS idx_relations_type_confidence
    ON relations(relation_type, confidence);
CREATE INDEX IF NOT EXISTS idx_consistency_type_relation
    ON consistency_findings(check_type, relation_type, confidence);
CREATE INDEX IF NOT EXISTS idx_consistency_facts_fact
    ON consistency_finding_facts(fact_id, finding_id);
CREATE INDEX IF NOT EXISTS idx_failures_document_stage
    ON extraction_failures(document_id, stage);
"""


class StoreVersionError(RuntimeError):
    """Raised when a database was created by unsupported schema code."""


class FactStore:
    """Persist validated facts and their provenance in a local SQLite file."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Fail fast if the schema can't be built, rather than only discovering a version
        # mismatch on whatever request happens to run first.
        with self._connection():
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            # Checked on every connection, not only at construction. The database file
            # lives outside this process - a deleted data directory, a wiped demo db - and
            # sqlite3.connect() happily hands back a fresh, schema-less file for a path
            # that no longer holds the one this store was built against. Without this, a
            # store built before that deletion looks healthy right up until the first
            # query, which then fails with "no such table" on every single request until
            # the process restarts. The DDL and each migration step are already written to
            # be safe to repeat, so this costs a few cheap existence checks per request.
            self._ensure_schema(connection)
            yield connection
        finally:
            connection.close()

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(DDL)
        row = connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            connection.commit()
            return
        version = int(row["value"])
        if version == 1:
            columns = {
                item["name"]
                for item in connection.execute("PRAGMA table_info(passages)").fetchall()
            }
            if "role" not in columns:
                connection.execute(
                    "ALTER TABLE passages ADD COLUMN role TEXT NOT NULL DEFAULT 'body'"
                )
            connection.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                ("2",),
            )
            connection.commit()
            version = 2
        if version == 2:
            cache_columns = {
                item["name"]
                for item in connection.execute("PRAGMA table_info(llm_cache)").fetchall()
            }
            if "schema_version" not in cache_columns:
                connection.executescript(
                    """CREATE TABLE llm_cache_v3 (
                           cache_key TEXT PRIMARY KEY,
                           passage_hash TEXT NOT NULL,
                           model TEXT NOT NULL,
                           schema_version INTEGER NOT NULL,
                           prompt_version INTEGER NOT NULL,
                           response_json TEXT NOT NULL,
                           validation_result TEXT NOT NULL,
                           created_at TEXT NOT NULL,
                           UNIQUE(passage_hash, model, schema_version, prompt_version)
                       );
                       INSERT INTO llm_cache_v3(
                           cache_key, passage_hash, model, schema_version,
                           prompt_version, response_json, validation_result, created_at
                       )
                       SELECT cache_key, passage_hash, model, 1,
                              prompt_version, response_json, validation_result, created_at
                       FROM llm_cache;
                       DROP TABLE llm_cache;
                       ALTER TABLE llm_cache_v3 RENAME TO llm_cache;"""
                )
            connection.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                ("3",),
            )
            connection.commit()
            version = 3
        if version == 3:
            connection.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                (str(SCHEMA_VERSION),),
            )
            connection.commit()
            version = SCHEMA_VERSION
        if version != SCHEMA_VERSION:
            raise StoreVersionError(
                f"Database schema version {version} is not supported; expected {SCHEMA_VERSION}"
            )

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Commit a unit of work completely or leave the database unchanged."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @property
    def schema_version(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
        return int(row["value"])

    def foreign_keys_enabled(self) -> bool:
        with self._connection() as connection:
            return bool(connection.execute("PRAGMA foreign_keys").fetchone()[0])

    def register_document(self, document: Document) -> tuple[Document, bool]:
        """Return the existing document for duplicate bytes, otherwise create it."""
        with self.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (document.content_hash,)
            ).fetchone()
            if existing:
                return self._document_from_row(existing), False

            stored = document.model_copy(update={"id": document.id or self._new_id("doc")})
            connection.execute(
                """INSERT INTO documents(
                       id, content_hash, original_filename, page_count, status,
                       extraction_mode, warnings_json, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stored.id,
                    stored.content_hash,
                    stored.original_filename,
                    stored.page_count,
                    stored.status.value,
                    stored.extraction_mode,
                    self._json(stored.warnings),
                    stored.created_at.isoformat(),
                    stored.updated_at.isoformat(),
                ),
            )
            return stored, True

    def get_document(self, document_id: str) -> Document | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self, *, status: DocumentStatus | None = None) -> list[Document]:
        sql = "SELECT * FROM documents"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status.value)
        sql += " ORDER BY created_at, id"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._document_from_row(row) for row in rows]

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        page_count: int | None = None,
        warnings: list[str] | None = None,
        extraction_mode: str | None = None,
    ) -> Document:
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status.value, self._now()]
        if page_count is not None:
            assignments.append("page_count = ?")
            params.append(page_count)
        if warnings is not None:
            assignments.append("warnings_json = ?")
            params.append(self._json(warnings))
        if extraction_mode is not None:
            if extraction_mode not in {"offline", "hybrid"}:
                raise ValueError("Extraction mode must be 'offline' or 'hybrid'")
            assignments.append("extraction_mode = ?")
            params.append(extraction_mode)
        params.append(document_id)
        with self.transaction(immediate=True) as connection:
            cursor = connection.execute(
                f"UPDATE documents SET {', '.join(assignments)} WHERE id = ?", params
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Document {document_id!r} does not exist")
        return self.get_document(document_id)  # type: ignore[return-value]

    def put_passage(self, passage: Passage) -> Passage:
        with self.transaction(immediate=True) as connection:
            return self._put_passage(connection, passage)

    def _put_passage(self, connection: sqlite3.Connection, passage: Passage) -> Passage:
        existing = connection.execute(
            """SELECT id FROM passages
               WHERE document_id = ? AND page_index = ? AND reading_order = ?""",
            (passage.document_id, passage.page_index, passage.reading_order),
        ).fetchone()
        stored = passage.model_copy(update={"id": passage.id or (existing["id"] if existing else self._new_id("passage"))})
        connection.execute(
            """INSERT INTO passages(
                   id, document_id, page_index, reading_order, role, text,
                   char_start, char_end, bbox_json, text_hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   role = excluded.role, text = excluded.text, char_start = excluded.char_start,
                   char_end = excluded.char_end, bbox_json = excluded.bbox_json,
                   text_hash = excluded.text_hash""",
            (
                stored.id,
                stored.document_id,
                stored.page_index,
                stored.reading_order,
                stored.role.value,
                stored.text,
                stored.char_start,
                stored.char_end,
                self._model_json(stored.bbox),
                stored.text_hash,
            ),
        )
        return stored

    def get_passage(self, passage_id: str) -> Passage | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM passages WHERE id = ?", (passage_id,)).fetchone()
        return self._passage_from_row(row) if row else None

    def list_passages(self, document_id: str, *, page_index: int | None = None) -> list[Passage]:
        sql = "SELECT * FROM passages WHERE document_id = ?"
        params: list[Any] = [document_id]
        if page_index is not None:
            sql += " AND page_index = ?"
            params.append(page_index)
        sql += " ORDER BY page_index, reading_order"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._passage_from_row(row) for row in rows]

    def put_entity(self, entity: Entity) -> Entity:
        with self.transaction(immediate=True) as connection:
            return self._put_entity(connection, entity)

    def _put_entity(self, connection: sqlite3.Connection, entity: Entity) -> Entity:
        stored = entity.model_copy(update={"id": entity.id or self._new_id("entity")})
        connection.execute(
            """INSERT INTO entities(
                   id, canonical_name, entity_type, identifiers_json, aliases_json,
                   confidence, review_state
               ) VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   canonical_name = excluded.canonical_name,
                   entity_type = excluded.entity_type,
                   identifiers_json = excluded.identifiers_json,
                   aliases_json = excluded.aliases_json,
                   confidence = excluded.confidence,
                   review_state = excluded.review_state""",
            (
                stored.id,
                stored.canonical_name,
                stored.entity_type,
                self._json(stored.identifiers),
                self._json(stored.aliases),
                stored.confidence,
                stored.review_state.value,
            ),
        )
        return stored

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return self._entity_from_row(row) if row else None

    def list_entities(self, *, entity_type: str | None = None) -> list[Entity]:
        sql = "SELECT * FROM entities"
        params: list[Any] = []
        if entity_type:
            sql += " WHERE entity_type = ?"
            params.append(entity_type)
        sql += " ORDER BY canonical_name, id"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._entity_from_row(row) for row in rows]

    def put_predicate(self, predicate: Predicate) -> Predicate:
        with self.transaction(immediate=True) as connection:
            self._put_predicate(connection, predicate)
        return predicate

    def _put_predicate(self, connection: sqlite3.Connection, predicate: Predicate) -> None:
        existing = connection.execute(
            "SELECT value_kind FROM predicates WHERE key = ?", (predicate.key,)
        ).fetchone()
        if existing and existing["value_kind"] != predicate.value_kind:
            raise ValueError(
                f"Predicate {predicate.key!r} is already stored as {existing['value_kind']!r}, "
                f"not {predicate.value_kind!r}"
            )
        connection.execute(
            """INSERT INTO predicates(key, display_name, value_kind, aliases_json, description)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   display_name = excluded.display_name,
                   value_kind = excluded.value_kind,
                   aliases_json = excluded.aliases_json,
                   description = COALESCE(excluded.description, predicates.description)""",
            (
                predicate.key,
                predicate.display_name,
                predicate.value_kind,
                self._json(predicate.aliases),
                predicate.description,
            ),
        )

    def get_predicate(self, key: str) -> Predicate | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM predicates WHERE key = ?", (key,)).fetchone()
        return self._predicate_from_row(row) if row else None

    def list_predicates(self) -> list[Predicate]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM predicates ORDER BY key").fetchall()
        return [self._predicate_from_row(row) for row in rows]

    def put_fact(self, fact: Fact) -> tuple[Fact, bool]:
        """Store a fact and all its evidence in one transaction."""
        with self.transaction(immediate=True) as connection:
            entity = self._entity_for_fact(connection, fact)
            predicate = Predicate(
                key=fact.predicate.key,
                display_name=fact.predicate.display_name,
                value_kind=fact.value.kind,
            )
            self._put_predicate(connection, predicate)
            prepared = fact.model_copy(
                update={
                    "id": fact.id or self._new_id("fact"),
                    "subject": EntityReference(
                        id=entity.id,
                        canonical_name=entity.canonical_name,
                        entity_type=entity.entity_type,
                    ),
                }
            )
            fingerprint = self._fact_fingerprint(prepared)
            existing = connection.execute(
                "SELECT id FROM facts WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if existing:
                for evidence in prepared.evidence:
                    self._put_evidence(connection, existing["id"], evidence)
                return self._fact_from_id(connection, existing["id"]), False

            connection.execute(
                """INSERT INTO facts(
                       id, fingerprint, subject_id, predicate_key, value_kind,
                       value_json, context_json, extraction_confidence,
                       normalization_confidence, review_state, warnings_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prepared.id,
                    fingerprint,
                    entity.id,
                    prepared.predicate.key,
                    prepared.value.kind,
                    self._model_json(prepared.value),
                    self._model_json(prepared.context),
                    prepared.extraction_confidence,
                    prepared.normalization_confidence,
                    prepared.review_state.value,
                    self._json(prepared.warnings),
                    prepared.created_at.isoformat(),
                ),
            )
            for evidence in prepared.evidence:
                self._put_evidence(connection, prepared.id, evidence)
            return self._fact_from_id(connection, prepared.id), True

    def _entity_for_fact(self, connection: sqlite3.Connection, fact: Fact) -> Entity:
        if fact.subject.id:
            row = connection.execute("SELECT * FROM entities WHERE id = ?", (fact.subject.id,)).fetchone()
            if row:
                return self._entity_from_row(row)
        exact = connection.execute(
            """SELECT * FROM entities
               WHERE lower(canonical_name) = lower(?) AND entity_type = ?
               ORDER BY id LIMIT 1""",
            (fact.subject.canonical_name, fact.subject.entity_type or "unknown"),
        ).fetchone()
        if exact:
            return self._entity_from_row(exact)
        return self._put_entity(
            connection,
            Entity(
                id=fact.subject.id,
                canonical_name=fact.subject.canonical_name,
                entity_type=fact.subject.entity_type or "unknown",
                confidence=fact.normalization_confidence,
                review_state=fact.review_state,
            ),
        )

    def _put_evidence(self, connection: sqlite3.Connection, fact_id: str, evidence: Evidence) -> None:
        passage_row = connection.execute(
            "SELECT * FROM passages WHERE id = ?", (evidence.passage_id,)
        ).fetchone()
        if passage_row is None:
            raise ValueError(f"Evidence passage {evidence.passage_id!r} does not exist")
        verified = evidence.verified_against(self._passage_from_row(passage_row))
        existing_span = connection.execute(
            """SELECT id FROM evidence
               WHERE fact_id = ? AND passage_id = ? AND quote_start = ? AND quote_end = ?""",
            (fact_id, verified.passage_id, verified.quote_start, verified.quote_end),
        ).fetchone()
        if existing_span:
            return
        evidence_id = verified.id or self._new_id("evidence")
        id_owner = connection.execute(
            "SELECT fact_id FROM evidence WHERE id = ?",
            (evidence_id,),
        ).fetchone()
        if id_owner:
            suffix = hashlib.sha256(f"{evidence_id}\0{fact_id}".encode()).hexdigest()[:24]
            evidence_id = f"evidence_{suffix}"
        connection.execute(
            """INSERT INTO evidence(
                   id, fact_id, document_id, passage_id, page_index, quote,
                   quote_start, quote_end, bbox_json, extractor, confidence, verified
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                evidence_id,
                fact_id,
                verified.document_id,
                verified.passage_id,
                verified.page_index,
                verified.quote,
                verified.quote_start,
                verified.quote_end,
                self._model_json(verified.bbox),
                verified.extractor.value,
                verified.confidence,
            ),
        )

    def get_fact(self, fact_id: str) -> Fact | None:
        with self._connection() as connection:
            row = connection.execute("SELECT id FROM facts WHERE id = ?", (fact_id,)).fetchone()
            return self._fact_from_id(connection, fact_id) if row else None

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
        return self._evidence_from_row(row) if row else None

    def list_evidence(
        self, *, fact_id: str | None = None, document_id: str | None = None
    ) -> list[Evidence]:
        clauses: list[str] = []
        params: list[Any] = []
        if fact_id:
            clauses.append("fact_id = ?")
            params.append(fact_id)
        if document_id:
            clauses.append("document_id = ?")
            params.append(document_id)
        sql = "SELECT * FROM evidence"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY page_index, quote_start, id"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._evidence_from_row(row) for row in rows]

    def list_facts(
        self,
        *,
        document_id: str | None = None,
        entity_id: str | None = None,
        predicate_key: str | None = None,
        review_state: ReviewState | None = None,
        minimum_confidence: float | None = None,
    ) -> list[Fact]:
        joins = ""
        clauses: list[str] = []
        params: list[Any] = []
        if document_id:
            joins = " JOIN evidence e ON e.fact_id = f.id"
            clauses.append("e.document_id = ?")
            params.append(document_id)
        if entity_id:
            clauses.append("f.subject_id = ?")
            params.append(entity_id)
        if predicate_key:
            clauses.append("f.predicate_key = ?")
            params.append(predicate_key)
        if review_state:
            clauses.append("f.review_state = ?")
            params.append(review_state.value)
        if minimum_confidence is not None:
            clauses.append("MIN(f.extraction_confidence, f.normalization_confidence) >= ?")
            params.append(minimum_confidence)
        sql = f"SELECT DISTINCT f.id FROM facts f{joins}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY f.created_at, f.id"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
            return [self._fact_from_id(connection, row["id"]) for row in rows]

    def comparison_candidates(self, fact: Fact) -> list[Fact]:
        if not fact.subject.id:
            return []
        return [
            candidate
            for candidate in self.list_facts(
                entity_id=fact.subject.id,
                predicate_key=fact.predicate.key,
            )
            if candidate.id != fact.id
        ]

    def put_relation(self, relation: Relation) -> tuple[Relation, bool]:
        """Canonicalize a fact pair so A-B and B-A cannot both exist."""
        left, right = sorted((relation.fact_a_id, relation.fact_b_id))
        comparison = relation.value_comparison
        context_diff = relation.context_diff
        if relation.fact_a_id != left:
            comparison = ValueComparison(
                left=relation.value_comparison.right,
                right=relation.value_comparison.left,
                agrees=relation.value_comparison.agrees,
                difference=relation.value_comparison.difference,
                tolerance=relation.value_comparison.tolerance,
            )
            context_diff = [
                item.model_copy(update={"left": item.right, "right": item.left})
                for item in relation.context_diff
            ]
        prepared = relation.model_copy(
            update={
                "id": relation.id or self._new_id("relation"),
                "fact_a_id": left,
                "fact_b_id": right,
                "value_comparison": comparison,
                "context_diff": context_diff,
            }
        )
        with self.transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT id FROM relations WHERE fact_a_id = ? AND fact_b_id = ?", (left, right)
            ).fetchone()
            relation_id = existing["id"] if existing else prepared.id
            connection.execute(
                """INSERT INTO relations(
                       id, fact_a_id, fact_b_id, relation_type, value_comparison_json,
                       context_diff_json, explanation, confidence, rule_version,
                       review_state, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(fact_a_id, fact_b_id) DO UPDATE SET
                       relation_type = excluded.relation_type,
                       value_comparison_json = excluded.value_comparison_json,
                       context_diff_json = excluded.context_diff_json,
                       explanation = excluded.explanation,
                       confidence = excluded.confidence,
                       rule_version = excluded.rule_version,
                       review_state = excluded.review_state""",
                (
                    relation_id,
                    left,
                    right,
                    prepared.relation_type.value,
                    self._model_json(prepared.value_comparison),
                    self._json([item.model_dump(mode="json") for item in prepared.context_diff]),
                    prepared.explanation,
                    prepared.confidence,
                    prepared.rule_version,
                    prepared.review_state.value,
                    prepared.created_at.isoformat(),
                ),
            )
            return self._relation_from_id(connection, relation_id), existing is None

    def get_relation(self, relation_id: str) -> Relation | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM relations WHERE id = ?", (relation_id,)).fetchone()
        return self._relation_from_row(row) if row else None

    def list_relations(
        self,
        *,
        relation_type: RelationType | None = None,
        minimum_confidence: float | None = None,
        review_state: ReviewState | None = None,
    ) -> list[Relation]:
        clauses: list[str] = []
        params: list[Any] = []
        if relation_type:
            clauses.append("relation_type = ?")
            params.append(relation_type.value)
        if minimum_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(minimum_confidence)
        if review_state:
            clauses.append("review_state = ?")
            params.append(review_state.value)
        sql = "SELECT * FROM relations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, id"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._relation_from_row(row) for row in rows]

    def put_consistency_finding(
        self,
        finding: ConsistencyFinding,
    ) -> tuple[ConsistencyFinding, bool]:
        """Store a calculation and its ordered fact operands atomically."""
        fingerprint = self._consistency_fingerprint(finding)
        with self.transaction(immediate=True) as connection:
            placeholders = ",".join("?" for _ in finding.fact_ids)
            rows = connection.execute(
                f"SELECT id FROM facts WHERE id IN ({placeholders})",
                finding.fact_ids,
            ).fetchall()
            found = {row["id"] for row in rows}
            missing = [fact_id for fact_id in finding.fact_ids if fact_id not in found]
            if missing:
                raise ValueError(
                    f"Consistency finding references missing facts: {', '.join(missing)}"
                )
            existing = connection.execute(
                "SELECT id FROM consistency_findings WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            finding_id = existing["id"] if existing else (finding.id or self._new_id("check"))
            prepared = finding.model_copy(update={"id": finding_id})
            connection.execute(
                """INSERT INTO consistency_findings(
                       id, fingerprint, check_type, relation_type, formula, operands_json,
                       stated_result, calculated_result, difference, tolerance, explanation,
                       confidence, rule_version, review_state, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(fingerprint) DO UPDATE SET
                       relation_type = excluded.relation_type,
                       formula = excluded.formula,
                       operands_json = excluded.operands_json,
                       stated_result = excluded.stated_result,
                       calculated_result = excluded.calculated_result,
                       difference = excluded.difference,
                       tolerance = excluded.tolerance,
                       explanation = excluded.explanation,
                       confidence = excluded.confidence,
                       rule_version = excluded.rule_version,
                       review_state = excluded.review_state""",
                (
                    prepared.id,
                    fingerprint,
                    prepared.check_type.value,
                    prepared.relation_type.value,
                    prepared.formula,
                    self._json(prepared.operands),
                    self._decimal_text(prepared.stated_result),
                    self._decimal_text(prepared.calculated_result),
                    self._decimal_text(prepared.difference),
                    self._decimal_text(prepared.tolerance),
                    prepared.explanation,
                    prepared.confidence,
                    prepared.rule_version,
                    prepared.review_state.value,
                    prepared.created_at.isoformat(),
                ),
            )
            connection.execute(
                "DELETE FROM consistency_finding_facts WHERE finding_id = ?",
                (prepared.id,),
            )
            connection.executemany(
                """INSERT INTO consistency_finding_facts(finding_id, fact_id, position)
                   VALUES (?, ?, ?)""",
                [
                    (prepared.id, fact_id, position)
                    for position, fact_id in enumerate(prepared.fact_ids)
                ],
            )
            return prepared, existing is None

    def get_consistency_finding(self, finding_id: str) -> ConsistencyFinding | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM consistency_findings WHERE id = ?",
                (finding_id,),
            ).fetchone()
            return self._consistency_from_row(connection, row) if row else None

    def list_consistency_findings(
        self,
        *,
        check_type: ConsistencyCheckType | None = None,
        relation_type: RelationType | None = None,
        review_state: ReviewState | None = None,
        fact_id: str | None = None,
    ) -> list[ConsistencyFinding]:
        joins = ""
        clauses: list[str] = []
        params: list[Any] = []
        if fact_id:
            joins = " JOIN consistency_finding_facts cff ON cff.finding_id = cf.id"
            clauses.append("cff.fact_id = ?")
            params.append(fact_id)
        if check_type:
            clauses.append("cf.check_type = ?")
            params.append(check_type.value)
        if relation_type:
            clauses.append("cf.relation_type = ?")
            params.append(relation_type.value)
        if review_state:
            clauses.append("cf.review_state = ?")
            params.append(review_state.value)
        sql = f"SELECT DISTINCT cf.* FROM consistency_findings cf{joins}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY cf.created_at, cf.id"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
            return [self._consistency_from_row(connection, row) for row in rows]

    def put_failure(self, failure: ExtractionFailure) -> ExtractionFailure:
        stored = failure.model_copy(update={"id": failure.id or self._new_id("failure")})
        rejected = None if stored.rejected_output is None else self._json(stored.rejected_output)
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO extraction_failures(
                       id, document_id, passage_id, page_index, stage, reason,
                       rejected_output_json, recoverable, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stored.id,
                    stored.document_id,
                    stored.passage_id,
                    stored.page_index,
                    stored.stage.value,
                    stored.reason,
                    rejected,
                    int(stored.recoverable),
                    stored.created_at.isoformat(),
                ),
            )
        return stored

    def list_failures(
        self, *, document_id: str | None = None, stage: str | None = None
    ) -> list[ExtractionFailure]:
        clauses: list[str] = []
        params: list[Any] = []
        if document_id:
            clauses.append("document_id = ?")
            params.append(document_id)
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        sql = "SELECT * FROM extraction_failures"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, id"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._failure_from_row(row) for row in rows]

    def cache_llm_response(
        self,
        *,
        passage_hash: str,
        model: str,
        schema_version: int = 1,
        prompt_version: int,
        response: dict[str, Any] | list[Any],
        validation_result: str,
    ) -> str:
        cache_key = hashlib.sha256(
            f"{passage_hash}\0{model}\0{schema_version}\0{prompt_version}".encode()
        ).hexdigest()
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO llm_cache(
                       cache_key, passage_hash, model, schema_version, prompt_version,
                       response_json, validation_result, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(passage_hash, model, schema_version, prompt_version) DO UPDATE SET
                       cache_key = excluded.cache_key,
                       response_json = excluded.response_json,
                       validation_result = excluded.validation_result,
                       created_at = excluded.created_at""",
                (
                    cache_key,
                    passage_hash,
                    model,
                    schema_version,
                    prompt_version,
                    self._json(response),
                    validation_result,
                    self._now(),
                ),
            )
        return cache_key

    def get_cached_llm_response(
        self,
        *,
        passage_hash: str,
        model: str,
        schema_version: int = 1,
        prompt_version: int,
    ) -> dict[str, Any] | list[Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT response_json FROM llm_cache
                   WHERE passage_hash = ? AND model = ?
                     AND schema_version = ? AND prompt_version = ?""",
                (passage_hash, model, schema_version, prompt_version),
            ).fetchone()
        return json.loads(row["response_json"]) if row else None

    def clear_document_results(self, document_id: str) -> None:
        """Prepare one document for reprocessing without touching other documents."""
        with self.transaction(immediate=True) as connection:
            exists = connection.execute(
                "SELECT 1 FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if not exists:
                raise KeyError(f"Document {document_id!r} does not exist")
            fact_rows = connection.execute(
                "SELECT DISTINCT fact_id FROM evidence WHERE document_id = ?", (document_id,)
            ).fetchall()
            finding_rows = connection.execute(
                """SELECT DISTINCT cff.finding_id
                   FROM consistency_finding_facts cff
                   JOIN evidence e ON e.fact_id = cff.fact_id
                   WHERE e.document_id = ?""",
                (document_id,),
            ).fetchall()
            connection.executemany(
                "DELETE FROM consistency_findings WHERE id = ?",
                [(row["finding_id"],) for row in finding_rows],
            )
            connection.executemany(
                "DELETE FROM facts WHERE id = ?", [(row["fact_id"],) for row in fact_rows]
            )
            connection.execute("DELETE FROM extraction_failures WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM passages WHERE document_id = ?", (document_id,))
            connection.execute(
                """UPDATE documents
                   SET status = ?, warnings_json = '[]', updated_at = ? WHERE id = ?""",
                (DocumentStatus.REGISTERED.value, self._now(), document_id),
            )

    def _fact_from_id(self, connection: sqlite3.Connection, fact_id: str) -> Fact:
        row = connection.execute(
            """SELECT f.*, e.canonical_name, e.entity_type, p.display_name
               FROM facts f
               JOIN entities e ON e.id = f.subject_id
               JOIN predicates p ON p.key = f.predicate_key
               WHERE f.id = ?""",
            (fact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Fact {fact_id!r} does not exist")
        evidence_rows = connection.execute(
            "SELECT * FROM evidence WHERE fact_id = ? ORDER BY page_index, quote_start", (fact_id,)
        ).fetchall()
        return Fact.model_validate(
            {
                "id": row["id"],
                "subject": {
                    "id": row["subject_id"],
                    "canonical_name": row["canonical_name"],
                    "entity_type": row["entity_type"],
                },
                "predicate": {
                    "key": row["predicate_key"],
                    "display_name": row["display_name"],
                },
                "value": json.loads(row["value_json"]),
                "context": json.loads(row["context_json"]),
                "evidence": [self._evidence_from_row(item) for item in evidence_rows],
                "extraction_confidence": row["extraction_confidence"],
                "normalization_confidence": row["normalization_confidence"],
                "review_state": row["review_state"],
                "warnings": json.loads(row["warnings_json"]),
                "created_at": row["created_at"],
            }
        )

    def _relation_from_id(self, connection: sqlite3.Connection, relation_id: str) -> Relation:
        row = connection.execute("SELECT * FROM relations WHERE id = ?", (relation_id,)).fetchone()
        if row is None:
            raise KeyError(f"Relation {relation_id!r} does not exist")
        return self._relation_from_row(row)

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> Document:
        return Document.model_validate(
            {
                "id": row["id"],
                "content_hash": row["content_hash"],
                "original_filename": row["original_filename"],
                "page_count": row["page_count"],
                "status": row["status"],
                "extraction_mode": row["extraction_mode"],
                "warnings": json.loads(row["warnings_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    @staticmethod
    def _passage_from_row(row: sqlite3.Row) -> Passage:
        return Passage.model_validate(
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "page_index": row["page_index"],
                "reading_order": row["reading_order"],
                "role": row["role"],
                "text": row["text"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "bbox": json.loads(row["bbox_json"]) if row["bbox_json"] else None,
                "text_hash": row["text_hash"],
            }
        )

    @staticmethod
    def _entity_from_row(row: sqlite3.Row) -> Entity:
        return Entity.model_validate(
            {
                "id": row["id"],
                "canonical_name": row["canonical_name"],
                "entity_type": row["entity_type"],
                "identifiers": json.loads(row["identifiers_json"]),
                "aliases": json.loads(row["aliases_json"]),
                "confidence": row["confidence"],
                "review_state": row["review_state"],
            }
        )

    @staticmethod
    def _predicate_from_row(row: sqlite3.Row) -> Predicate:
        return Predicate.model_validate(
            {
                "key": row["key"],
                "display_name": row["display_name"],
                "value_kind": row["value_kind"],
                "aliases": json.loads(row["aliases_json"]),
                "description": row["description"],
            }
        )

    @staticmethod
    def _evidence_from_row(row: sqlite3.Row) -> Evidence:
        return Evidence.model_validate(
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "passage_id": row["passage_id"],
                "page_index": row["page_index"],
                "quote": row["quote"],
                "quote_start": row["quote_start"],
                "quote_end": row["quote_end"],
                "bbox": json.loads(row["bbox_json"]) if row["bbox_json"] else None,
                "extractor": row["extractor"],
                "confidence": row["confidence"],
                "verified": bool(row["verified"]),
            }
        )

    @staticmethod
    def _relation_from_row(row: sqlite3.Row) -> Relation:
        return Relation.model_validate(
            {
                "id": row["id"],
                "fact_a_id": row["fact_a_id"],
                "fact_b_id": row["fact_b_id"],
                "relation_type": row["relation_type"],
                "value_comparison": json.loads(row["value_comparison_json"]),
                "context_diff": json.loads(row["context_diff_json"]),
                "explanation": row["explanation"],
                "confidence": row["confidence"],
                "rule_version": row["rule_version"],
                "review_state": row["review_state"],
                "created_at": row["created_at"],
            }
        )

    @staticmethod
    def _consistency_from_row(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> ConsistencyFinding:
        fact_rows = connection.execute(
            """SELECT fact_id FROM consistency_finding_facts
               WHERE finding_id = ? ORDER BY position""",
            (row["id"],),
        ).fetchall()
        return ConsistencyFinding.model_validate(
            {
                "id": row["id"],
                "check_type": row["check_type"],
                "fact_ids": [item["fact_id"] for item in fact_rows],
                "relation_type": row["relation_type"],
                "formula": row["formula"],
                "operands": json.loads(row["operands_json"]),
                "stated_result": row["stated_result"],
                "calculated_result": row["calculated_result"],
                "difference": row["difference"],
                "tolerance": row["tolerance"],
                "explanation": row["explanation"],
                "confidence": row["confidence"],
                "rule_version": row["rule_version"],
                "review_state": row["review_state"],
                "created_at": row["created_at"],
            }
        )

    @staticmethod
    def _failure_from_row(row: sqlite3.Row) -> ExtractionFailure:
        rejected = row["rejected_output_json"]
        return ExtractionFailure.model_validate(
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "passage_id": row["passage_id"],
                "page_index": row["page_index"],
                "stage": row["stage"],
                "reason": row["reason"],
                "rejected_output": json.loads(rejected) if rejected else None,
                "recoverable": bool(row["recoverable"]),
                "created_at": row["created_at"],
            }
        )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _model_json(cls, model: Any) -> str | None:
        if model is None:
            return None
        return cls._json(model.model_dump(mode="json"))

    @classmethod
    def _fact_fingerprint(cls, fact: Fact) -> str:
        source_documents = sorted({item.document_id for item in fact.evidence})
        payload = {
            "subject_id": fact.subject.id,
            "predicate": fact.predicate.key,
            "value": fact.value.model_dump(mode="json"),
            "context": fact.context.model_dump(mode="json"),
            "documents": source_documents,
        }
        return hashlib.sha256(cls._json(payload).encode()).hexdigest()

    @classmethod
    def _consistency_fingerprint(cls, finding: ConsistencyFinding) -> str:
        payload = {
            "check_type": finding.check_type.value,
            "fact_ids": finding.fact_ids,
            "formula": finding.formula,
            "rule_version": finding.rule_version,
        }
        return hashlib.sha256(cls._json(payload).encode()).hexdigest()

    @staticmethod
    def _decimal_text(value) -> str | None:
        return None if value is None else str(value)

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = ["FactStore", "SCHEMA_VERSION", "StoreVersionError"]
