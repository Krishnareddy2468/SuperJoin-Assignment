from datetime import date
from pathlib import Path

import pytest

from factlayer.link import RULE_VERSION, FactLinker
from factlayer.normalize import normalize_number
from factlayer.schema import (
    BooleanValue,
    CategoricalValue,
    ContextEnvelope,
    DateValue,
    Document,
    EntityReference,
    Evidence,
    ExtractionMethod,
    Fact,
    IdentifierValue,
    Passage,
    PredicateReference,
    RelationType,
    ReportingPeriod,
    ReviewState,
    TextValue,
)
from factlayer.store import FactStore


DELHIVERY = EntityReference(
    id="company:delhivery",
    canonical_name="Delhivery Limited",
    entity_type="company",
)
REVENUE = PredicateReference(
    key="revenue_from_operations",
    display_name="Revenue from operations",
)
FY24 = ReportingPeriod(
    label="FY 2023-24",
    start=date(2023, 4, 1),
    end=date(2024, 3, 31),
)
FY23 = ReportingPeriod(
    label="FY 2022-23",
    start=date(2022, 4, 1),
    end=date(2023, 3, 31),
)


def make_fact(
    fact_id: str,
    document_id: str,
    value,
    *,
    context: ContextEnvelope | None = None,
    subject: EntityReference = DELHIVERY,
    predicate: PredicateReference = REVENUE,
    review_state: ReviewState = ReviewState.READY,
    passage_id: str | None = None,
) -> Fact:
    raw = value.raw
    passage = Passage(
        id=passage_id or f"passage:{fact_id}",
        document_id=document_id,
        page_index=0,
        reading_order=0,
        text=f"Reported value: {raw}",
    )
    start = passage.text.index(raw)
    evidence = Evidence(
        id=f"evidence:{fact_id}",
        document_id=document_id,
        passage_id=passage.id,
        page_index=0,
        quote=raw,
        quote_start=start,
        quote_end=start + len(raw),
        extractor=ExtractionMethod.DETERMINISTIC,
        confidence=0.98,
    ).verified_against(passage)
    return Fact(
        id=fact_id,
        subject=subject,
        predicate=predicate,
        value=value,
        context=context or ContextEnvelope(),
        evidence=[evidence],
        extraction_confidence=0.95,
        normalization_confidence=0.94,
        review_state=review_state,
    )


def revenue_context(
    *,
    period: ReportingPeriod = FY24,
    scope: str = "consolidated",
    basis: str = "audited",
    publisher: str = "Annual report",
    geography: str = "India",
) -> ContextEnvelope:
    return ContextEnvelope(
        period=period,
        scope=scope,
        basis=basis,
        geography=geography,
        publisher=publisher,
    )


def test_million_and_crore_revenue_corroborate_after_normalization() -> None:
    left = make_fact(
        "fact-a",
        "annual-report",
        normalize_number("₹81,415.38 million").value,
        context=revenue_context(publisher="Annual report"),
    )
    right = make_fact(
        "fact-b",
        "earnings-deck",
        normalize_number("₹8,142 crore").value,
        context=revenue_context(publisher="Earnings deck"),
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.CORROBORATES
    assert relation.value_comparison.agrees is True
    assert relation.value_comparison.difference is not None
    assert relation.value_comparison.tolerance is not None
    assert [item.field for item in relation.context_diff] == ["publisher"]
    assert relation.rule_version == RULE_VERSION
    assert "agree within a tolerance" in relation.explanation


def test_equivalent_period_dates_ignore_display_label_variants() -> None:
    alternate_fy24 = ReportingPeriod(
        label="FY24",
        start=date(2023, 4, 1),
        end=date(2024, 3, 31),
    )
    left = make_fact(
        "fact-a",
        "document-a",
        normalize_number("₹100 crore").value,
        context=revenue_context(period=FY24),
    )
    right = make_fact(
        "fact-b",
        "document-b",
        normalize_number("₹100 crore").value,
        context=revenue_context(period=alternate_fy24, publisher="Other report"),
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.CORROBORATES
    assert "period" not in {item.field for item in relation.context_diff}


def test_different_values_with_equivalent_context_contradict() -> None:
    left = make_fact(
        "fact-a",
        "document-a",
        normalize_number("₹100 crore").value,
        context=revenue_context(),
    )
    right = make_fact(
        "fact-b",
        "document-b",
        normalize_number("₹120 crore").value,
        context=revenue_context(publisher="Other report"),
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.CONTRADICTS
    assert relation.value_comparison.agrees is False
    assert relation.review_state is ReviewState.READY
    assert "while period, scope, basis" in relation.explanation


@pytest.mark.parametrize(
    ("left_context", "right_context", "expected"),
    [
        (
            revenue_context(scope="standalone"),
            revenue_context(scope="consolidated"),
            RelationType.RECONCILED_BY_SCOPE,
        ),
        (
            revenue_context(period=FY23),
            revenue_context(period=FY24),
            RelationType.RECONCILED_BY_PERIOD,
        ),
        (
            revenue_context(basis="audited"),
            revenue_context(basis="unaudited"),
            RelationType.RECONCILED_BY_BASIS,
        ),
    ],
)
def test_one_context_axis_explains_different_numeric_values(
    left_context: ContextEnvelope,
    right_context: ContextEnvelope,
    expected: RelationType,
) -> None:
    left = make_fact(
        "fact-a",
        "document-a",
        normalize_number("₹100 crore").value,
        context=left_context,
    )
    right = make_fact(
        "fact-b",
        "document-b",
        normalize_number("₹120 crore").value,
        context=right_context,
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is expected
    explaining = [item for item in relation.context_diff if item.explains_difference]
    assert len(explaining) == 1
    assert explaining[0].field in relation.explanation


def test_dated_status_change_reconciles_by_as_of_date() -> None:
    predicate = PredicateReference(key="board_membership_status", display_name="Board status")
    left = make_fact(
        "fact-a",
        "prospectus",
        CategoricalValue(raw="active", state="active"),
        predicate=predicate,
        context=ContextEnvelope(as_of=date(2022, 5, 1), publisher="Prospectus"),
    )
    right = make_fact(
        "fact-b",
        "annual-report",
        CategoricalValue(raw="resigned", state="resigned"),
        predicate=predicate,
        context=ContextEnvelope(as_of=date(2023, 8, 24), publisher="Annual report"),
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.RECONCILED_BY_AS_OF
    assert relation.value_comparison.agrees is False
    assert next(item for item in relation.context_diff if item.field == "as_of").left == (
        "2022-05-01"
    )


def test_percent_and_ratio_reconcile_as_equivalent_units() -> None:
    left = make_fact(
        "fact-a",
        "document-a",
        normalize_number("12.7%").value,
        context=revenue_context(),
    )
    right = make_fact(
        "fact-b",
        "document-b",
        normalize_number("0.127x").value,
        context=revenue_context(publisher="Other report"),
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.RECONCILED_BY_UNIT
    assert relation.value_comparison.agrees is True
    assert next(item for item in relation.context_diff if item.field == "unit").explains_difference


def test_multiple_context_differences_are_incomparable() -> None:
    left = make_fact(
        "fact-a",
        "document-a",
        normalize_number("₹100 crore").value,
        context=revenue_context(period=FY23, scope="standalone"),
    )
    right = make_fact(
        "fact-b",
        "document-b",
        normalize_number("₹120 crore").value,
        context=revenue_context(period=FY24, scope="consolidated"),
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.INCOMPARABLE
    assert relation.review_state is ReviewState.NEEDS_REVIEW
    assert "period" in relation.explanation and "scope" in relation.explanation


def test_missing_context_is_not_treated_as_an_explanation() -> None:
    left = make_fact(
        "fact-a",
        "document-a",
        normalize_number("₹100 crore").value,
        context=ContextEnvelope(period=FY24),
    )
    right = make_fact(
        "fact-b",
        "document-b",
        normalize_number("₹120 crore").value,
        context=ContextEnvelope(period=FY24, scope="consolidated"),
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.NEEDS_REVIEW
    scope = next(item for item in relation.context_diff if item.field == "scope")
    assert scope.left is None
    assert scope.explains_difference is False


def test_two_context_free_numeric_facts_require_review() -> None:
    left = make_fact(
        "fact-a",
        "document-a",
        normalize_number("₹100 crore").value,
    )
    right = make_fact(
        "fact-b",
        "document-b",
        normalize_number("₹120 crore").value,
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.NEEDS_REVIEW
    assert "neither numeric fact has material context" in relation.explanation


@pytest.mark.parametrize(
    ("left_value", "right_value"),
    [
        (
            CategoricalValue(raw="Active", state="active"),
            CategoricalValue(raw="active", state="ACTIVE"),
        ),
        (
            TextValue(raw="Strong growth", text="Strong growth"),
            TextValue(raw="strong-growth", text="strong-growth"),
        ),
        (
            DateValue(raw="2024-03-31", value=date(2024, 3, 31)),
            DateValue(raw="March 31, 2024", value=date(2024, 3, 31)),
        ),
        (BooleanValue(raw="Yes", value=True), BooleanValue(raw="true", value=True)),
        (
            IdentifierValue(raw="DIN 01173669", value="01173669", scheme="DIN"),
            IdentifierValue(raw="0117 3669", value="0117 3669", scheme="din"),
        ),
    ],
)
def test_semantic_value_kinds_compare_after_safe_normalization(
    left_value,
    right_value,
) -> None:
    predicate = PredicateReference(key="semantic_measure", display_name="Semantic measure")
    left = make_fact("fact-a", "document-a", left_value, predicate=predicate)
    right = make_fact("fact-b", "document-b", right_value, predicate=predicate)

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.CORROBORATES
    assert relation.value_comparison.agrees is True


def test_dates_with_different_precision_require_review() -> None:
    predicate = PredicateReference(key="event_date", display_name="Event date")
    left = make_fact(
        "fact-a",
        "document-a",
        DateValue(raw="2024", value=date(2024, 1, 1), precision="year"),
        predicate=predicate,
    )
    right = make_fact(
        "fact-b",
        "document-b",
        DateValue(raw="January 1, 2024", value=date(2024, 1, 1), precision="day"),
        predicate=predicate,
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.NEEDS_REVIEW
    assert relation.value_comparison.agrees is None


def test_uncertain_source_fact_forces_relation_review() -> None:
    left = make_fact(
        "fact-a",
        "document-a",
        normalize_number("₹100 crore").value,
        context=revenue_context(),
        review_state=ReviewState.NEEDS_REVIEW,
    )
    right = make_fact(
        "fact-b",
        "document-b",
        normalize_number("₹100 crore").value,
        context=revenue_context(),
    )

    relation = FactLinker().compare(left, right)

    assert relation.relation_type is RelationType.NEEDS_REVIEW
    assert relation.review_state is ReviewState.NEEDS_REVIEW


def test_candidate_generation_uses_subject_predicate_and_document() -> None:
    first = make_fact("fact-a", "document-a", normalize_number("₹100 crore").value)
    second = make_fact("fact-b", "document-b", normalize_number("₹101 crore").value)
    same_document = make_fact("fact-c", "document-a", normalize_number("₹102 crore").value)
    different_subject = make_fact(
        "fact-d",
        "document-d",
        normalize_number("₹103 crore").value,
        subject=EntityReference(id="company:other", canonical_name="Other", entity_type="company"),
    )
    different_predicate = make_fact(
        "fact-e",
        "document-e",
        normalize_number("₹104 crore").value,
        predicate=PredicateReference(key="profit", display_name="Profit"),
    )

    candidates = [first, second, same_document, different_subject, different_predicate, first]

    # Same-document pairs are compared by default: an annual report states revenue on
    # both a standalone and a consolidated basis on one page, and that reconciliation
    # is only reachable if facts from one document can be paired.
    pairs = FactLinker().candidate_pairs(candidates)
    assert {(left.id, right.id) for left, right in pairs} == {
        ("fact-a", "fact-b"),
        ("fact-a", "fact-c"),
        ("fact-b", "fact-c"),
    }

    restricted = FactLinker().candidate_pairs(candidates, cross_document_only=True)
    assert {(left.id, right.id) for left, right in restricted} == {
        ("fact-a", "fact-b"),
        ("fact-b", "fact-c"),
    }


def test_cells_quoted_from_one_passage_are_not_treated_as_rival_claims() -> None:
    """A repeated table row is many measures, not many answers to one question.

    An earnings deck prints a "% margin" row across eight quarterly columns. Every
    cell lands on the same passage under the same weak label, so pairing them would
    report dozens of contradictions that the document never stated.
    """
    shared = "passage:quarterly-margin-table"
    cells = [
        make_fact(
            f"fact-cell-{index}",
            "document-deck",
            normalize_number(raw).value,
            context=revenue_context(scope="segment"),
            predicate=PredicateReference(key="percentage_margin", display_name="% margin"),
            passage_id=shared,
        )
        for index, raw in enumerate(["18.4%", "6.8%", "8.9%"])
    ]

    assert FactLinker().candidate_pairs(cells) == ()

    # A figure quoted from its own sentence stays comparable against the table.
    standalone = make_fact(
        "fact-sentence",
        "document-deck",
        normalize_number("6.8%").value,
        context=revenue_context(scope="segment"),
        predicate=PredicateReference(key="percentage_margin", display_name="% margin"),
    )
    paired = FactLinker().candidate_pairs([*cells, standalone])
    assert {(left.id, right.id) for left, right in paired} == {
        ("fact-cell-0", "fact-sentence"),
        ("fact-cell-1", "fact-sentence"),
        ("fact-cell-2", "fact-sentence"),
    }


def test_incremental_linking_queries_and_stores_only_relevant_facts(tmp_path: Path) -> None:
    store = FactStore(tmp_path / "facts.db")
    stored_facts: list[Fact] = []
    for index, (amount, publisher) in enumerate(
        [("100", "First report"), ("120", "Second report")]
    ):
        document, _ = store.register_document(
            Document(
                content_hash=chr(ord("a") + index) * 64,
                original_filename=f"report-{index}.pdf",
            )
        )
        fact = make_fact(
            f"fact-{index}",
            document.id,
            normalize_number(f"₹{amount} crore").value,
            context=revenue_context(publisher=publisher),
        )
        source = Passage(
            id=fact.evidence[0].passage_id,
            document_id=document.id,
            page_index=0,
            reading_order=0,
            text=f"Reported value: {fact.value.raw}",
        )
        store.put_passage(source)
        stored, _ = store.put_fact(fact)
        stored_facts.append(stored)

    linker = FactLinker()
    relations = linker.link_new_fact(store, stored_facts[1])
    repeated = linker.link_new_fact(store, stored_facts[1])

    assert len(relations) == len(repeated) == 1
    assert relations[0].relation_type is RelationType.CONTRADICTS
    assert relations[0].id == repeated[0].id
    assert len(store.list_relations()) == 1
    assert store.get_relation(relations[0].id).rule_version == RULE_VERSION


def test_a_quarter_beside_a_full_year_is_not_a_period_reconciliation() -> None:
    """Different-length periods are apples to oranges, not an explained difference.

    A deck prints quarters and full years under one label. Calling that pair
    "reconciled by period" would dress up a comparison that should not be made.
    """
    q4 = ReportingPeriod(label="Q4 FY 2023-24", start=date(2024, 1, 1), end=date(2024, 3, 31))

    relation = FactLinker().compare(
        make_fact("fact-year", "doc-a", normalize_number("₹8,142 crore").value,
                  context=revenue_context(period=FY24)),
        make_fact("fact-quarter", "doc-b", normalize_number("₹2,076 crore").value,
                  context=revenue_context(period=q4)),
    )

    assert relation.relation_type is not RelationType.RECONCILED_BY_PERIOD

    # Two comparable years still reconcile normally.
    comparable = FactLinker().compare(
        make_fact("fact-fy24", "doc-a", normalize_number("₹8,142 crore").value,
                  context=revenue_context(period=FY24)),
        make_fact("fact-fy23", "doc-b", normalize_number("₹7,225 crore").value,
                  context=revenue_context(period=FY23)),
    )
    assert comparable.relation_type is RelationType.RECONCILED_BY_PERIOD


def test_context_only_one_source_states_lowers_confidence_but_still_reconciles() -> None:
    """Silence about one field should not bury an otherwise clear explanation.

    Almost no filing repeats its country in every sentence. The gap is recorded and
    costs confidence, rather than sending the whole comparison to review.
    """
    stated = revenue_context(period=FY24)
    silent = revenue_context(period=FY23).model_copy(update={"geography": None})

    relation = FactLinker().compare(
        make_fact("fact-a", "doc-a", normalize_number("₹8,142 crore").value, context=stated),
        make_fact("fact-b", "doc-b", normalize_number("₹7,225 crore").value, context=silent),
    )

    assert relation.relation_type is RelationType.RECONCILED_BY_PERIOD
    assert "geography" in relation.explanation
    assert any(item.field == "geography" for item in relation.context_diff)
