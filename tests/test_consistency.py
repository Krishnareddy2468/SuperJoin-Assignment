from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from factlayer.consistency import CONSISTENCY_RULE_VERSION, ConsistencyChecker
from factlayer.normalize import normalize_number
from factlayer.schema import (
    ConsistencyCheckType,
    ConsistencyFinding,
    ContextEnvelope,
    Document,
    EntityReference,
    Evidence,
    ExtractionMethod,
    Fact,
    Passage,
    PredicateReference,
    RelationType,
    ReportingPeriod,
    ReviewState,
)
from factlayer.store import FactStore


SUBJECT = EntityReference(
    id="company:example",
    canonical_name="Example Limited",
    entity_type="company",
)
REVENUE = PredicateReference(key="revenue", display_name="Revenue")
GROWTH = PredicateReference(key="revenue_percentage_change", display_name="Revenue growth")
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


def context(period: ReportingPeriod = FY24, *, scope: str = "consolidated") -> ContextEnvelope:
    return ContextEnvelope(
        period=period,
        scope=scope,
        basis="audited",
        geography="India",
        publisher="Example report",
    )


def fact(
    fact_id: str,
    raw: str,
    *,
    predicate: PredicateReference = REVENUE,
    fact_context: ContextEnvelope | None = None,
    method: ExtractionMethod = ExtractionMethod.DETERMINISTIC,
    warnings: list[str] | None = None,
    review_state: ReviewState = ReviewState.READY,
) -> Fact:
    source = Passage(
        id=f"passage:{fact_id}",
        document_id=f"document:{fact_id}",
        page_index=0,
        reading_order=0,
        text=f"Reported value: {raw}",
    )
    start = source.text.index(raw)
    evidence = Evidence(
        id=f"evidence:{fact_id}",
        document_id=source.document_id,
        passage_id=source.id,
        page_index=0,
        quote=raw,
        quote_start=start,
        quote_end=start + len(raw),
        extractor=method,
        confidence=0.98,
    ).verified_against(source)
    normalized = normalize_number(raw)
    assert normalized.value is not None
    return Fact(
        id=fact_id,
        subject=SUBJECT,
        predicate=predicate,
        value=normalized.value,
        context=fact_context or context(),
        evidence=[evidence],
        extraction_confidence=0.96,
        normalization_confidence=normalized.confidence,
        warnings=warnings or [],
        review_state=review_state,
    )


def growth_facts(stated: str = "20%") -> tuple[Fact, Fact, Fact]:
    current = fact("current", "₹120 crore", fact_context=context(FY24))
    prior = fact("prior", "₹100 crore", fact_context=context(FY23))
    stated_fact = fact(
        "stated-growth",
        stated,
        predicate=GROWTH,
        fact_context=context(FY24),
    )
    return current, prior, stated_fact


def test_correct_stated_growth_corroborates_with_auditable_operands() -> None:
    current, prior, stated = growth_facts()

    finding = ConsistencyChecker().check_growth(
        current=current,
        prior=prior,
        stated_growth=stated,
    )

    assert finding.check_type is ConsistencyCheckType.GROWTH
    assert finding.relation_type is RelationType.CORROBORATES
    assert finding.stated_result == Decimal("20")
    assert finding.calculated_result == Decimal("20.0")
    assert finding.difference == 0
    assert finding.tolerance > 0
    assert finding.formula == "growth_percent = (current - prior) / abs(prior) * 100"
    assert finding.operands["current"]["fact_id"] == "current"
    assert finding.rule_version == CONSISTENCY_RULE_VERSION


def test_wrong_growth_is_a_definitive_contradiction_when_inputs_are_clear() -> None:
    current, prior, stated = growth_facts("10%")

    finding = ConsistencyChecker().check_growth(
        current=current,
        prior=prior,
        stated_growth=stated,
    )

    assert finding.relation_type is RelationType.CONTRADICTS
    assert finding.difference > finding.tolerance
    assert finding.review_state is ReviewState.READY
    assert "beyond" in finding.explanation


def test_reported_rounding_does_not_create_a_false_growth_conflict() -> None:
    current = fact("current", "₹8,142 crore", fact_context=context(FY24))
    prior = fact("prior", "₹7,225 crore", fact_context=context(FY23))
    stated = fact(
        "stated-growth",
        "12.7%",
        predicate=GROWTH,
        fact_context=context(FY24),
    )

    finding = ConsistencyChecker().check_growth(
        current=current,
        prior=prior,
        stated_growth=stated,
    )

    assert finding.relation_type is RelationType.CORROBORATES
    assert finding.difference <= finding.tolerance


def test_uncertain_growth_mismatch_is_only_a_likely_contradiction() -> None:
    current, prior, stated = growth_facts("40%")
    current = current.model_copy(
        update={
            "warnings": ["The current value is approximate."],
            "review_state": ReviewState.NEEDS_REVIEW,
        }
    )

    finding = ConsistencyChecker().check_growth(
        current=current,
        prior=prior,
        stated_growth=stated,
    )

    assert finding.relation_type is RelationType.LIKELY_CONTRADICTION
    assert finding.review_state is ReviewState.NEEDS_REVIEW
    assert "uncertainty prevents a definitive contradiction" in finding.explanation


def test_missing_growth_period_and_zero_prior_need_review() -> None:
    current, prior, stated = growth_facts()
    stated = stated.model_copy(update={"context": ContextEnvelope(scope="consolidated")})
    missing_period = ConsistencyChecker().check_growth(
        current=current,
        prior=prior,
        stated_growth=stated,
    )
    zero_prior = prior.model_copy(update={"value": normalize_number("₹0 crore").value})
    undefined = ConsistencyChecker().check_growth(
        current=current,
        prior=zero_prior,
        stated_growth=growth_facts()[2],
    )

    assert missing_period.relation_type is RelationType.NEEDS_REVIEW
    assert undefined.relation_type is RelationType.NEEDS_REVIEW
    assert undefined.calculated_result is None


def component_facts(
    total: str = "₹100 crore",
    first: str = "₹40 crore",
    second: str = "₹60 crore",
) -> tuple[Fact, list[Fact]]:
    total_fact = fact(
        "total",
        total,
        predicate=PredicateReference(key="total_income", display_name="Total income"),
    )
    components = [
        fact(
            "component-a",
            first,
            predicate=PredicateReference(key="service_income", display_name="Service income"),
        ),
        fact(
            "component-b",
            second,
            predicate=PredicateReference(key="other_income", display_name="Other income"),
        ),
    ]
    return total_fact, components


def test_components_that_sum_to_the_total_corroborate() -> None:
    total, components = component_facts()

    finding = ConsistencyChecker().check_component_total(
        total=total,
        components=components,
        components_complete=True,
    )

    assert finding.relation_type is RelationType.CORROBORATES
    assert finding.calculated_result == finding.stated_result
    assert len(finding.fact_ids) == 3
    assert finding.operands["components_complete"] is True


def test_incorrect_complete_total_is_a_contradiction() -> None:
    total, components = component_facts(second="₹50 crore")

    finding = ConsistencyChecker().check_component_total(
        total=total,
        components=components,
        components_complete=True,
    )

    assert finding.relation_type is RelationType.CONTRADICTS
    assert finding.difference > finding.tolerance


def test_normal_component_rounding_is_not_flagged() -> None:
    total, components = component_facts(first="₹33 crore", second="₹66 crore")

    finding = ConsistencyChecker().check_component_total(
        total=total,
        components=components,
        components_complete=True,
    )

    assert finding.difference == Decimal("10000000")
    assert finding.difference <= finding.tolerance
    assert finding.relation_type is RelationType.CORROBORATES


def test_incomplete_components_and_context_mismatch_need_review() -> None:
    total, components = component_facts(second="₹50 crore")
    incomplete = ConsistencyChecker().check_component_total(
        total=total,
        components=components,
        components_complete=False,
    )
    components[1] = components[1].model_copy(
        update={"context": context(scope="standalone")}
    )
    mismatched = ConsistencyChecker().check_component_total(
        total=total,
        components=components,
        components_complete=True,
    )

    assert incomplete.relation_type is RelationType.NEEDS_REVIEW
    assert mismatched.relation_type is RelationType.NEEDS_REVIEW


def test_uncertain_total_mismatch_is_a_likely_contradiction() -> None:
    total, components = component_facts(second="₹50 crore")
    components[1] = components[1].model_copy(
        update={
            "warnings": ["This component was extracted approximately."],
            "review_state": ReviewState.NEEDS_REVIEW,
        }
    )

    finding = ConsistencyChecker().check_component_total(
        total=total,
        components=components,
        components_complete=True,
    )

    assert finding.relation_type is RelationType.LIKELY_CONTRADICTION


def test_repeated_prose_and_table_metric_preserves_comparison_inputs() -> None:
    left = fact(
        "prose",
        "₹100 crore",
        method=ExtractionMethod.DETERMINISTIC,
    )
    right = fact(
        "table",
        "₹120 crore",
        method=ExtractionMethod.TABLE,
    )

    finding = ConsistencyChecker().check_repeated_metric(left, right)

    assert finding.check_type is ConsistencyCheckType.REPEATED_METRIC
    assert finding.relation_type is RelationType.CONTRADICTS
    assert finding.operands["extraction_methods"] == ["deterministic", "table"]
    assert finding.formula == "normalized(left_value) = normalized(right_value)"


def store_fact(store: FactStore, source_fact: Fact, hash_character: str) -> Fact:
    document, _ = store.register_document(
        Document(
            content_hash=hash_character * 64,
            original_filename=f"{source_fact.id}.pdf",
        )
    )
    raw = source_fact.value.raw
    source = Passage(
        id=source_fact.evidence[0].passage_id,
        document_id=document.id,
        page_index=0,
        reading_order=0,
        text=f"Reported value: {raw}",
    )
    stored_passage = store.put_passage(source)
    start = source.text.index(raw)
    evidence = source_fact.evidence[0].model_copy(
        update={
            "document_id": document.id,
            "passage_id": stored_passage.id,
            "quote_start": start,
            "quote_end": start + len(raw),
            "verified": False,
        }
    ).verified_against(stored_passage)
    prepared = source_fact.model_copy(update={"evidence": [evidence]})
    stored, _ = store.put_fact(prepared)
    return stored


def test_findings_round_trip_and_reprocessing_removes_stale_calculations(
    tmp_path: Path,
) -> None:
    store = FactStore(tmp_path / "facts.db")
    current, prior, stated = growth_facts("10%")
    stored = [
        store_fact(store, source, marker)
        for source, marker in zip((current, prior, stated), "abc")
    ]
    finding = ConsistencyChecker().check_growth(
        current=stored[0],
        prior=stored[1],
        stated_growth=stored[2],
    )

    first, created = store.put_consistency_finding(finding)
    repeated, repeated_created = store.put_consistency_finding(finding)

    assert created is True
    assert repeated_created is False
    assert repeated == first
    assert store.get_consistency_finding(first.id) == first
    assert store.list_consistency_findings(
        check_type=ConsistencyCheckType.GROWTH,
        relation_type=RelationType.CONTRADICTS,
        fact_id=stored[0].id,
    ) == [first]

    store.clear_document_results(stored[0].evidence[0].document_id)

    assert store.list_consistency_findings() == []


def test_finding_rejects_duplicate_fact_operands() -> None:
    with pytest.raises(ValidationError, match="cannot repeat"):
        ConsistencyFinding(
            check_type=ConsistencyCheckType.GROWTH,
            fact_ids=["fact-a", "fact-a"],
            relation_type=RelationType.NEEDS_REVIEW,
            formula="growth = change / prior",
            operands={},
            explanation="Needs review.",
            confidence=0.5,
            rule_version="test",
        )


def test_growth_checks_are_discovered_from_predicate_shape() -> None:
    """A stated growth percentage should find its own operands.

    Discovery keys off the shape of the predicate rather than any known measure, so an
    unfamiliar document contributes checks without new code.
    """
    findings = ConsistencyChecker().discover_growth_checks(
        [
            fact("fact-current", "8,142", fact_context=context(FY24)),
            fact("fact-prior", "7,225", fact_context=context(FY23)),
            fact("fact-growth", "12.68%", predicate=GROWTH, fact_context=context(FY24)),
        ]
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.check_type is ConsistencyCheckType.GROWTH
    assert finding.relation_type is RelationType.CORROBORATES
    assert finding.operands["current"]["raw"] == "8,142"
    assert finding.operands["prior"]["raw"] == "7,225"


def test_growth_discovery_ignores_a_quarter_that_merely_ends_before_the_year() -> None:
    """The prior period has to cover comparable ground, not just sit next to it.

    An earnings deck prints quarters and full years in one table. The quarter ending the
    day before a financial year starts is adjacent without being the prior period, and
    pairing them turns a correct 12.7% into a nonsense 338%.
    """
    q4_fy23 = ReportingPeriod(label="Q4 FY 2022-23", start=date(2023, 1, 1), end=date(2023, 3, 31))

    findings = ConsistencyChecker().discover_growth_checks(
        [
            fact("fact-current", "8,142", fact_context=context(FY24)),
            fact("fact-quarter", "1,860", fact_context=context(q4_fy23)),
            fact("fact-growth", "12.68%", predicate=GROWTH, fact_context=context(FY24)),
        ]
    )

    assert findings == ()


def test_growth_discovery_reports_a_mismatch_it_can_defend() -> None:
    """When the operands are sound and the arithmetic disagrees, say so."""
    findings = ConsistencyChecker().discover_growth_checks(
        [
            fact("fact-current", "8,142", fact_context=context(FY24)),
            fact("fact-prior", "7,225", fact_context=context(FY23)),
            fact("fact-growth", "45.00%", predicate=GROWTH, fact_context=context(FY24)),
        ]
    )

    assert len(findings) == 1
    assert findings[0].relation_type in {
        RelationType.CONTRADICTS,
        RelationType.LIKELY_CONTRADICTION,
    }


def _cell(fact_id: str, raw: str, *, predicate: str, row: int, column: int = 1) -> Fact:
    """A fact carrying the grid position a table extractor records."""
    built = fact(
        fact_id,
        raw,
        predicate=PredicateReference(key=predicate, display_name=predicate),
    )
    # Cells of one table share a document; the helper otherwise invents one per fact.
    evidence = [
        item.model_copy(update={"document_id": "document:statement"})
        for item in built.evidence
    ]
    return built.model_copy(
        update={
            "evidence": evidence,
            "context": built.context.model_copy(
                update={
                    "qualifiers": {
                        "table_index": "0",
                        "table_row": str(row),
                        "table_column": str(column),
                    }
                }
            ),
        }
    )


def test_component_totals_are_discovered_from_grid_position() -> None:
    """Row labels repeat down a statement, so position is what identifies components."""
    facts = [
        _cell("fact-a", "100", predicate="employee_benefit_expenses", row=1),
        _cell("fact-b", "50", predicate="finance_costs", row=2),
        _cell("fact-c", "160", predicate="total_expenses", row=3),
    ]

    findings = ConsistencyChecker().discover_component_total_checks(facts)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.check_type is ConsistencyCheckType.COMPONENT_TOTAL
    assert finding.stated_result == Decimal("160")
    assert finding.calculated_result == Decimal("150")


def test_a_total_is_never_auto_promoted_to_a_contradiction() -> None:
    """Rows we failed to parse are invisible, and subtotals double count.

    Both faults look exactly like a statement disagreeing with itself, so the check
    raises a review candidate instead of asserting a contradiction.
    """
    facts = [
        _cell("fact-a", "100", predicate="employee_benefit_expenses", row=1),
        _cell("fact-b", "50", predicate="finance_costs", row=2),
        _cell("fact-c", "9,999", predicate="total_expenses", row=3),
    ]

    findings = ConsistencyChecker().discover_component_total_checks(facts)

    assert len(findings) == 1
    assert findings[0].relation_type is RelationType.NEEDS_REVIEW


def test_percentage_rows_are_not_summed_into_a_currency_total() -> None:
    """A margin row inside a rupee table has no place in its sum."""
    facts = [
        _cell("fact-a", "100", predicate="employee_benefit_expenses", row=1),
        _cell("fact-b", "12.5%", predicate="margin", row=2),
        _cell("fact-c", "100", predicate="total_expenses", row=3),
    ]

    findings = ConsistencyChecker().discover_component_total_checks(facts)

    # Only one addable component remains, which is below the minimum for a check.
    assert findings == ()
