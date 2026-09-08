from datetime import date
from decimal import Decimal

import pytest
from pydantic import TypeAdapter, ValidationError

from factlayer.schema import (
    BooleanValue,
    BoundingBox,
    CategoricalValue,
    ContextEnvelope,
    DateValue,
    Document,
    EntityReference,
    Evidence,
    ExtractionFailure,
    ExtractionMethod,
    Fact,
    FactValue,
    FailureStage,
    IdentifierValue,
    NumericValue,
    Passage,
    Predicate,
    PredicateReference,
    Relation,
    RelationType,
    ReportingPeriod,
    TextValue,
    ValueComparison,
)


DOCUMENT_HASH = "a" * 64


def grounded_evidence() -> Evidence:
    passage = Passage(
        id="passage-1",
        document_id="document-1",
        page_index=5,
        reading_order=2,
        text="Revenue from operations was ₹81,415.38 million in FY24.",
        char_start=100,
        bbox=BoundingBox(x0=42, y0=80, x1=510, y1=112),
    )
    quote = "₹81,415.38 million"
    quote_start = passage.char_start + passage.text.index(quote)
    evidence = Evidence(
        document_id=passage.document_id,
        passage_id=passage.id,
        page_index=passage.page_index,
        quote=quote,
        quote_start=quote_start,
        quote_end=quote_start + len(quote),
        bbox=passage.bbox,
        extractor=ExtractionMethod.DETERMINISTIC,
        confidence=0.98,
    )
    return evidence.verified_against(passage)


def revenue_fact() -> Fact:
    return Fact(
        id="fact-1",
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
            raw="₹81,415.38 million",
            reported_number=Decimal("81415.38"),
            number=Decimal("81415380000"),
            reported_unit="million",
            unit="rupee",
            currency="INR",
            scale=Decimal("1000000"),
        ),
        context=ContextEnvelope(
            period=ReportingPeriod(
                label="FY 2023-24",
                start=date(2023, 4, 1),
                end=date(2024, 3, 31),
            ),
            scope="consolidated",
            basis="audited",
            publisher="Delhivery Limited",
        ),
        evidence=[grounded_evidence()],
        extraction_confidence=0.96,
        normalization_confidence=0.94,
        warnings=["Value is reported in millions", "Value is reported in millions"],
    )


def test_numeric_fact_round_trips_through_stable_json() -> None:
    fact = revenue_fact()

    restored = Fact.model_validate_json(fact.model_dump_json())

    assert restored == fact
    assert isinstance(restored.value, NumericValue)
    assert restored.value.number == Decimal("81415380000")
    assert restored.context.period.end == date(2024, 3, 31)
    assert restored.warnings == ["Value is reported in millions"]


@pytest.mark.parametrize(
    "value",
    [
        TextValue(raw="grew strongly", text="grew strongly"),
        CategoricalValue(raw="resigned", state="resigned"),
        DateValue(raw="August 24, 2023", value=date(2023, 8, 24)),
        BooleanValue(raw="Yes", value=True),
        IdentifierValue(raw="DIN 01173669", value="01173669", scheme="DIN"),
    ],
)
def test_semantic_value_types_keep_raw_and_normalized_forms(value: FactValue) -> None:
    adapter = TypeAdapter(FactValue)

    restored = adapter.validate_json(adapter.dump_json(value))

    assert restored == value
    assert restored.raw


def test_new_predicates_do_not_require_an_enum_change() -> None:
    predicate = Predicate(
        key="food_inflation_rural",
        display_name="Rural food inflation",
        value_kind="number",
        aliases=["Rural food CPI", "rural food cpi"],
    )

    assert predicate.key == "food_inflation_rural"
    assert predicate.aliases == ["Rural food CPI"]


def test_fact_without_evidence_is_rejected() -> None:
    payload = revenue_fact().model_dump()
    payload["evidence"] = []

    with pytest.raises(ValidationError, match="at least 1 item"):
        Fact.model_validate(payload)


def test_fact_with_unverified_evidence_is_rejected() -> None:
    payload = revenue_fact().model_dump()
    payload["evidence"][0]["verified"] = False

    with pytest.raises(ValidationError, match="verified source evidence"):
        Fact.model_validate(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"page_index": -1}, "greater than or equal to 0"),
        ({"quote_start": 20, "quote_end": 10}, "greater than quote_start"),
        ({"quote_start": 0, "quote_end": 2}, "span the exact quote"),
        ({"confidence": 1.2}, "less than or equal to 1"),
    ],
)
def test_invalid_evidence_locations_are_rejected(changes: dict, message: str) -> None:
    payload = grounded_evidence().model_dump()
    payload.update(changes)

    with pytest.raises(ValidationError, match=message):
        Evidence.model_validate(payload)


def test_quote_must_resolve_at_the_claimed_passage_offsets() -> None:
    passage = Passage(
        id="passage-1",
        document_id="document-1",
        page_index=0,
        reading_order=0,
        text="A grounded sentence.",
    )
    evidence = Evidence(
        document_id="document-1",
        passage_id="passage-1",
        page_index=0,
        quote="grounded",
        quote_start=0,
        quote_end=len("grounded"),
        extractor=ExtractionMethod.DETERMINISTIC,
    )

    with pytest.raises(ValueError, match="not found"):
        evidence.verified_against(passage)


def test_invalid_period_and_bounding_box_are_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot end before"):
        ReportingPeriod(label="Broken year", start=date(2025, 1, 1), end=date(2024, 1, 1))

    with pytest.raises(ValidationError, match="positive width and height"):
        BoundingBox(x0=100, y0=20, x1=50, y1=30)


def test_document_requires_a_sha256_hash_and_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="String should match pattern"):
        Document(content_hash="not-a-hash", original_filename="report.pdf")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Document(content_hash=DOCUMENT_HASH, original_filename="report.pdf", mystery=True)


def test_relation_cannot_compare_a_fact_with_itself() -> None:
    with pytest.raises(ValidationError, match="relationship with itself"):
        Relation(
            fact_a_id="fact-1",
            fact_b_id="fact-1",
            relation_type=RelationType.CORROBORATES,
            value_comparison=ValueComparison(left="10", right="10", agrees=True),
            explanation="Both facts report the same normalized value.",
            confidence=0.99,
            rule_version="1",
        )


def test_failures_are_serializable_and_keep_rejected_output() -> None:
    failure = ExtractionFailure(
        document_id="document-1",
        passage_id="passage-9",
        page_index=8,
        stage=FailureStage.GROUNDING,
        reason="The returned quote does not occur in the source passage.",
        rejected_output={"quote": "Invented evidence"},
        recoverable=True,
    )

    restored = ExtractionFailure.model_validate_json(failure.model_dump_json())

    assert restored == failure
    assert restored.rejected_output == {"quote": "Invented evidence"}
