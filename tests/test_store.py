import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from factlayer.schema import (
    ContextDifference,
    ContextEnvelope,
    Document,
    DocumentStatus,
    EntityReference,
    Evidence,
    ExtractionFailure,
    ExtractionMethod,
    Fact,
    FailureStage,
    NumericValue,
    Passage,
    Predicate,
    PredicateReference,
    Relation,
    RelationType,
    ReportingPeriod,
    ReviewState,
    ValueComparison,
)
from factlayer.store import FactStore, SCHEMA_VERSION, StoreVersionError


def make_document(
    store: FactStore, name: str, hash_character: str, *, amount: str = "100"
) -> tuple[Document, Passage]:
    document, created = store.register_document(
        Document(
            content_hash=hash_character * 64,
            original_filename=name,
            page_count=1,
        )
    )
    assert created is True
    passage = store.put_passage(
        Passage(
            document_id=document.id,
            page_index=0,
            reading_order=0,
            text=f"Revenue from operations was ₹{amount} crore in FY24.",
        )
    )
    return document, passage


def make_fact(
    document: Document,
    passage: Passage,
    *,
    fact_id: str | None = None,
    amount: str = "100",
    confidence: float = 0.95,
) -> Fact:
    quote = f"₹{amount} crore"
    quote_start = passage.text.index(quote)
    evidence = Evidence(
        document_id=document.id,
        passage_id=passage.id,
        page_index=passage.page_index,
        quote=quote,
        quote_start=quote_start,
        quote_end=quote_start + len(quote),
        extractor=ExtractionMethod.DETERMINISTIC,
        confidence=0.99,
    ).verified_against(passage)
    return Fact(
        id=fact_id,
        subject=EntityReference(
            id="entity-delhivery",
            canonical_name="Delhivery Limited",
            entity_type="company",
        ),
        predicate=PredicateReference(
            key="revenue_from_operations",
            display_name="Revenue from operations",
        ),
        value=NumericValue(
            raw=quote,
            number=Decimal(amount) * Decimal("10000000"),
            reported_number=Decimal(amount),
            reported_unit="crore",
            unit="rupee",
            currency="INR",
            scale=Decimal("10000000"),
        ),
        context=ContextEnvelope(
            period=ReportingPeriod(label="FY24"),
            scope="consolidated",
        ),
        evidence=[evidence],
        extraction_confidence=confidence,
        normalization_confidence=0.97,
    )


@pytest.fixture
def store(tmp_path) -> FactStore:
    return FactStore(tmp_path / "factlayer.db")


def test_store_initializes_versioned_schema_and_foreign_keys(store: FactStore) -> None:
    assert store.schema_version == SCHEMA_VERSION
    assert store.foreign_keys_enabled() is True

    with sqlite3.connect(store.db_path) as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }

    assert "idx_facts_subject_predicate" in indexes
    assert "idx_evidence_document_fact" in indexes


def test_store_rejects_an_unknown_schema_version(tmp_path) -> None:
    database = tmp_path / "future.db"
    store = FactStore(database)
    with store.transaction() as connection:
        connection.execute(
            "UPDATE schema_meta SET value = '99' WHERE key = 'schema_version'"
        )

    with pytest.raises(StoreVersionError, match="not supported"):
        FactStore(database)


def test_version_one_database_adds_passage_roles_safely(tmp_path) -> None:
    database = tmp_path / "version-one.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_meta VALUES ('schema_version', '1')")
        connection.execute(
            """CREATE TABLE passages (
                   id TEXT PRIMARY KEY,
                   document_id TEXT NOT NULL,
                   page_index INTEGER NOT NULL,
                   reading_order INTEGER NOT NULL,
                   text TEXT NOT NULL,
                   char_start INTEGER NOT NULL,
                   char_end INTEGER NOT NULL,
                   bbox_json TEXT,
                   text_hash TEXT
               )"""
        )

    store = FactStore(database)

    with sqlite3.connect(database) as connection:
        columns = {
            row[1]: row[4]
            for row in connection.execute("PRAGMA table_info(passages)").fetchall()
        }
    assert store.schema_version == SCHEMA_VERSION
    assert columns["role"] == "'body'"


def test_version_two_database_adds_schema_aware_llm_cache_keys(tmp_path) -> None:
    database = tmp_path / "version-two.db"
    store = FactStore(database)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE llm_cache")
        connection.execute(
            """CREATE TABLE llm_cache (
                   cache_key TEXT PRIMARY KEY,
                   passage_hash TEXT NOT NULL,
                   model TEXT NOT NULL,
                   prompt_version INTEGER NOT NULL,
                   response_json TEXT NOT NULL,
                   validation_result TEXT NOT NULL,
                   created_at TEXT NOT NULL,
                   UNIQUE(passage_hash, model, prompt_version)
               )"""
        )
        connection.execute(
            """INSERT INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "old-cache-key",
                "a" * 64,
                "example-model",
                1,
                '{"facts": []}',
                "accepted",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.execute(
            "UPDATE schema_meta SET value = '2' WHERE key = 'schema_version'"
        )

    migrated = FactStore(database)

    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.get_cached_llm_response(
        passage_hash="a" * 64,
        model="example-model",
        schema_version=1,
        prompt_version=1,
    ) == {"facts": []}
    assert migrated.get_cached_llm_response(
        passage_hash="a" * 64,
        model="example-model",
        schema_version=2,
        prompt_version=1,
    ) is None


def test_document_registration_deduplicates_identical_bytes(store: FactStore) -> None:
    first, created = store.register_document(
        Document(content_hash="a" * 64, original_filename="first-name.pdf")
    )
    duplicate, duplicate_created = store.register_document(
        Document(content_hash="a" * 64, original_filename="renamed-copy.pdf")
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate.id == first.id
    assert duplicate.original_filename == "first-name.pdf"
    assert len(store.list_documents()) == 1


def test_passages_upsert_in_stable_reading_order(store: FactStore) -> None:
    document, passage = make_document(store, "report.pdf", "a")
    updated = store.put_passage(
        passage.model_copy(update={"text": "Revenue from operations was ₹100 crore in FY24."})
    )

    assert updated.id == passage.id
    assert store.get_passage(passage.id) == updated
    assert store.list_passages(document.id) == [updated]


def test_fact_and_verified_evidence_are_stored_atomically(store: FactStore) -> None:
    document, passage = make_document(store, "report.pdf", "a")
    fact = make_fact(document, passage, fact_id="fact-a")

    stored, created = store.put_fact(fact)

    assert created is True
    assert stored.id == "fact-a"
    assert stored.subject.id == "entity-delhivery"
    assert stored.evidence[0].verified is True
    assert store.get_fact("fact-a") == stored
    assert store.list_evidence(fact_id="fact-a") == stored.evidence
    assert store.get_evidence(stored.evidence[0].id) == stored.evidence[0]
    assert store.get_entity("entity-delhivery").canonical_name == "Delhivery Limited"
    assert store.get_predicate("revenue_from_operations").value_kind == "number"


def test_missing_evidence_passage_rolls_back_the_whole_fact(store: FactStore) -> None:
    document, passage = make_document(store, "report.pdf", "a")
    fact = make_fact(document, passage, fact_id="fact-invalid")
    invalid_evidence = fact.evidence[0].model_copy(update={"passage_id": "missing-passage"})
    invalid_fact = fact.model_copy(update={"evidence": [invalid_evidence]})

    with pytest.raises(ValueError, match="does not exist"):
        store.put_fact(invalid_fact)

    assert store.get_fact("fact-invalid") is None
    assert store.list_entities() == []
    assert store.list_predicates() == []


def test_duplicate_fact_adds_no_duplicate_rows(store: FactStore) -> None:
    document, passage = make_document(store, "report.pdf", "a")
    fact = make_fact(document, passage)

    first, first_created = store.put_fact(fact)
    duplicate, duplicate_created = store.put_fact(fact)

    assert first_created is True
    assert duplicate_created is False
    assert duplicate.id == first.id
    assert len(store.list_facts()) == 1
    assert len(store.list_evidence(fact_id=first.id)) == 1


def test_two_facts_can_cite_the_same_generated_evidence_id(store: FactStore) -> None:
    document, passage = make_document(store, "report.pdf", "a")
    first = make_fact(document, passage, fact_id="fact-revenue")
    shared = first.evidence[0].model_copy(update={"id": "evidence-shared-quote"})
    first = first.model_copy(update={"evidence": [shared]})
    second = first.model_copy(
        update={
            "id": "fact-income",
            "predicate": PredicateReference(
                key="reported_income",
                display_name="Reported income",
            ),
        }
    )

    stored_first, _ = store.put_fact(first)
    stored_second, _ = store.put_fact(second)

    assert stored_first.evidence[0].id == "evidence-shared-quote"
    assert stored_second.evidence[0].id != stored_first.evidence[0].id
    assert stored_second.evidence[0].quote == stored_first.evidence[0].quote


def test_fact_queries_filter_by_source_entity_predicate_and_confidence(store: FactStore) -> None:
    document, passage = make_document(store, "report.pdf", "a")
    stored, _ = store.put_fact(make_fact(document, passage, confidence=0.91))

    assert store.list_facts(document_id=document.id) == [stored]
    assert store.list_facts(entity_id="entity-delhivery") == [stored]
    assert store.list_facts(predicate_key="revenue_from_operations") == [stored]
    assert store.list_facts(review_state=ReviewState.READY) == [stored]
    assert store.list_facts(minimum_confidence=0.9) == [stored]
    assert store.list_facts(minimum_confidence=0.95) == []
    assert store.comparison_candidates(stored) == []


def test_dynamic_predicates_and_entity_queries_round_trip(store: FactStore) -> None:
    predicate = Predicate(
        key="urban_food_inflation",
        display_name="Urban food inflation",
        value_kind="number",
        aliases=["Urban food CPI"],
    )
    store.put_predicate(predicate)
    document, passage = make_document(store, "report.pdf", "a")
    store.put_fact(make_fact(document, passage))

    assert predicate in store.list_predicates()
    assert [entity.entity_type for entity in store.list_entities(entity_type="company")] == ["company"]


def test_predicate_kind_cannot_change_silently(store: FactStore) -> None:
    store.put_predicate(
        Predicate(key="revenue", display_name="Revenue", value_kind="number")
    )

    with pytest.raises(ValueError, match="already stored as 'number'"):
        store.put_predicate(
            Predicate(key="revenue", display_name="Revenue", value_kind="category")
        )

    assert store.get_predicate("revenue").value_kind == "number"


def test_reversed_relationships_share_one_canonical_pair(store: FactStore) -> None:
    first_doc, first_passage = make_document(store, "first.pdf", "a")
    second_doc, second_passage = make_document(store, "second.pdf", "b", amount="101")
    first, _ = store.put_fact(make_fact(first_doc, first_passage, fact_id="fact-a", amount="100"))
    second, _ = store.put_fact(make_fact(second_doc, second_passage, fact_id="fact-b", amount="101"))

    initial = Relation(
        fact_a_id=second.id,
        fact_b_id=first.id,
        relation_type=RelationType.CONTRADICTS,
        value_comparison=ValueComparison(left="101", right="100", agrees=False),
        context_diff=[ContextDifference(field="scope", left="group", right="company")],
        explanation="The normalized values differ under the same reporting context.",
        confidence=0.9,
        rule_version="1",
    )
    stored, created = store.put_relation(initial)
    replacement, replacement_created = store.put_relation(
        Relation(
            fact_a_id=first.id,
            fact_b_id=second.id,
            relation_type=RelationType.NEEDS_REVIEW,
            value_comparison=ValueComparison(left="100", right="101", agrees=False),
            explanation="The pair needs more context before classification.",
            confidence=0.6,
            rule_version="2",
            review_state=ReviewState.NEEDS_REVIEW,
        )
    )

    assert created is True
    assert replacement_created is False
    assert stored.fact_a_id == "fact-a"
    assert stored.value_comparison.left == "100"
    assert stored.context_diff[0].left == "company"
    assert replacement.id == stored.id
    assert len(store.list_relations()) == 1
    assert store.list_relations(
        relation_type=RelationType.NEEDS_REVIEW,
        minimum_confidence=0.5,
        review_state=ReviewState.NEEDS_REVIEW,
    ) == [replacement]


def test_failure_records_and_llm_cache_are_queryable(store: FactStore) -> None:
    document, passage = make_document(store, "report.pdf", "a")
    failure = store.put_failure(
        ExtractionFailure(
            document_id=document.id,
            passage_id=passage.id,
            page_index=0,
            stage=FailureStage.GROUNDING,
            reason="The quote did not occur in the source passage.",
            rejected_output={"quote": "invented"},
        )
    )
    response = {"facts": [{"predicate": "revenue"}]}
    key = store.cache_llm_response(
        passage_hash="c" * 64,
        model="example-model",
        prompt_version=1,
        response=response,
        validation_result="accepted",
    )

    assert key
    assert store.list_failures(document_id=document.id, stage="grounding") == [failure]
    assert store.get_cached_llm_response(
        passage_hash="c" * 64,
        model="example-model",
        prompt_version=1,
    ) == response
    assert store.get_cached_llm_response(
        passage_hash="c" * 64,
        model="different-model",
        prompt_version=1,
    ) is None
    assert store.get_cached_llm_response(
        passage_hash="c" * 64,
        model="example-model",
        schema_version=2,
        prompt_version=1,
    ) is None
    second_schema_response = {"facts": [{"predicate": "board_status"}]}
    store.cache_llm_response(
        passage_hash="c" * 64,
        model="example-model",
        schema_version=2,
        prompt_version=1,
        response=second_schema_response,
        validation_result="accepted",
    )
    assert store.get_cached_llm_response(
        passage_hash="c" * 64,
        model="example-model",
        schema_version=2,
        prompt_version=1,
    ) == second_schema_response
    assert store.get_cached_llm_response(
        passage_hash="c" * 64,
        model="example-model",
        schema_version=1,
        prompt_version=1,
    ) == response


def test_reprocessing_one_document_preserves_unrelated_knowledge(store: FactStore) -> None:
    first_doc, first_passage = make_document(store, "first.pdf", "a")
    second_doc, second_passage = make_document(store, "second.pdf", "b")
    first, _ = store.put_fact(make_fact(first_doc, first_passage, fact_id="fact-a"))
    second, _ = store.put_fact(make_fact(second_doc, second_passage, fact_id="fact-b"))
    store.put_relation(
        Relation(
            fact_a_id=first.id,
            fact_b_id=second.id,
            relation_type=RelationType.CORROBORATES,
            value_comparison=ValueComparison(left="100", right="100", agrees=True),
            explanation="The normalized values agree.",
            confidence=0.99,
            rule_version="1",
        )
    )

    store.clear_document_results(first_doc.id)

    assert store.get_document(first_doc.id).status is DocumentStatus.REGISTERED
    assert store.list_passages(first_doc.id) == []
    assert store.list_facts(document_id=first_doc.id) == []
    assert store.get_fact(first.id) is None
    assert store.get_fact(second.id) is not None
    assert store.list_passages(second_doc.id) == [second_passage]
    assert store.list_relations() == []


def test_store_recovers_when_its_database_file_is_wiped_from_under_it(
    tmp_path: Path,
) -> None:
    """A live server's database can be deleted by something outside the process.

    The schema used to be built once, at construction. If the underlying file was
    deleted and recreated empty afterwards - a wiped demo directory, a stray rm -rf -
    every following request failed with "no such table" until the process restarted,
    which looked exactly like the API being down. The store now checks its schema on
    every connection instead of only at startup, so the same live instance recovers
    without a restart.
    """
    db_path = tmp_path / "facts.db"
    store = FactStore(db_path)
    assert store.schema_version == SCHEMA_VERSION

    db_path.unlink()
    db_path.touch()
    assert db_path.stat().st_size == 0

    assert store.schema_version == SCHEMA_VERSION
