from decimal import Decimal
from pathlib import Path

import pytest

from factlayer.extract_rules import DeterministicExtractor, detect_reporting_entity
from factlayer.ingest import ExtractedTable, IngestedPage, PdfIngestor, TableCell
from factlayer.schema import BoundingBox, EntityReference, Passage, PassageRole, ReviewState


PROJECT_ROOT = Path(__file__).parents[1]
DELHIVERY_DATA = PROJECT_ROOT / "starter-datasets" / "delhivery"
DELHIVERY = EntityReference(
    id="company:delhivery",
    canonical_name="Delhivery Limited",
    entity_type="company",
)


def passage(text: str, *, passage_id: str = "passage_1", bbox: BoundingBox | None = None) -> Passage:
    return Passage(
        id=passage_id,
        document_id="document_1",
        page_index=0,
        reading_order=0,
        role=PassageRole.BODY,
        text=text,
        char_start=100,
        bbox=bbox,
    )


def page_with(*passages: Passage, tables: tuple[ExtractedTable, ...] = ()) -> IngestedPage:
    return IngestedPage(
        page_index=0,
        passages=passages,
        tables=tables,
        warnings=(),
        elapsed_ms=0,
    )


def extract_text(text: str):
    source = passage(text)
    outcome = DeterministicExtractor().extract_page(
        page_with(source),
        default_subject=DELHIVERY,
        publisher="Annual report",
    )
    return source, outcome


def test_prose_value_keeps_its_label_context_and_exact_evidence() -> None:
    source, outcome = extract_text(
        "The revenue from operations on consolidated basis for\n"
        "FY24 stood at ₹ 81,415.38 million as against ₹72,253.01 million for FY23."
    )

    fact = next(item for item in outcome.facts if item.value.raw == "₹ 81,415.38 million")

    assert fact.predicate.key == "revenue_from_operations"
    assert fact.value.number == Decimal("81415380000.00")
    assert fact.value.currency == "INR"
    assert fact.context.scope == "consolidated"
    assert fact.context.period.label == "FY 2023-24"
    assert fact.evidence[0].verified is True
    assert fact.evidence[0].quote in source.text
    assert "revenue from operations" in fact.evidence[0].quote


def test_percentage_change_is_not_stored_as_the_absolute_metric() -> None:
    _, outcome = extract_text("Revenue increased by 12.7% during FY24.")

    fact = outcome.facts[0]

    assert fact.predicate.key == "revenue_percentage_change"
    assert fact.value.unit == "percent"
    assert fact.value.number == Decimal("12.7")


def test_page_furniture_and_unsupported_numbers_are_ignored() -> None:
    _, outcome = extract_text("Annual Report 2024\nPage 47\n5\n(1)")

    assert outcome.facts == ()
    assert outcome.failures == ()
    assert outcome.stats.candidates_seen == 4
    assert outcome.stats.candidates_ignored == 4


def test_unlabelled_value_is_recorded_as_a_rejected_candidate() -> None:
    _, outcome = extract_text("Approximately ₹100 crore.")

    assert outcome.facts == ()
    assert len(outcome.failures) == 1
    assert outcome.failures[0].rejected_output == {"candidate": "₹100 crore"}
    assert "no reliable predicate label" in outcome.failures[0].reason


def test_din_and_dated_board_status_share_the_resolved_person() -> None:
    source, outcome = extract_text(
        "Mr. Suvir Suren Sujan, Non-Executive Director (DIN: 01173669), "
        "resigned from the Board with effect from August 24, 2023."
    )
    facts = {item.predicate.key: item for item in outcome.facts}

    identifier = facts["director_identification_number"]
    status = facts["board_membership_status"]

    assert identifier.subject.id == "din:01173669"
    assert identifier.subject.canonical_name == "Suvir Suren Sujan"
    assert identifier.value.value == "01173669"
    assert status.subject == identifier.subject
    assert status.value.state == "resigned"
    assert status.value.raw == "resigned from the Board"
    assert status.context.as_of.isoformat() == "2023-08-24"
    assert all(item.evidence[0].quote in source.text for item in facts.values())


def test_cin_is_extracted_as_a_company_identifier() -> None:
    _, outcome = extract_text("Delhivery Limited (CIN: U63090DL2011PLC221234)")

    fact = next(item for item in outcome.facts if item.predicate.key == "corporate_identity_number")

    assert fact.subject.id == "cin:U63090DL2011PLC221234"
    assert fact.subject.entity_type == "company"
    assert fact.value.value == "U63090DL2011PLC221234"


def test_table_row_and_column_context_produce_a_grounded_fact() -> None:
    box = BoundingBox(x0=10, y0=10, x1=80, y1=30)
    source = passage("100", bbox=box)
    table = ExtractedTable(
        page_index=0,
        table_index=0,
        bbox=box,
        rows=(("Metric", "FY24"), ("Revenue", "100")),
        cells=(
            TableCell(0, 0, "Metric", box),
            TableCell(0, 1, "FY24", box),
            TableCell(1, 0, "Revenue", box),
            TableCell(1, 1, "100", box),
        ),
    )

    outcome = DeterministicExtractor().extract_page(
        page_with(source, tables=(table,)),
        default_subject=DELHIVERY,
    )

    assert len(outcome.facts) == 1
    fact = outcome.facts[0]
    assert fact.predicate.key == "revenue"
    assert fact.context.period.label == "FY 2023-24"
    assert fact.evidence[0].quote == "100"
    assert fact.evidence[0].bbox == box


def test_repeated_extraction_has_stable_ids_and_no_duplicates() -> None:
    source = passage("Revenue was ₹100 crore in FY24.")
    extractor = DeterministicExtractor()

    first = extractor.extract_page(page_with(source), default_subject=DELHIVERY)
    second = extractor.extract_page(page_with(source), default_subject=DELHIVERY)

    assert [item.id for item in first.facts] == [item.id for item in second.facts]
    assert len({item.id for item in first.facts}) == len(first.facts)


@pytest.mark.parametrize(
    ("filename", "page_index", "expected_predicate"),
    [
        ("01-delhivery-prospectus-2022-excerpt.pdf", 85, "director_identification_number"),
        ("02-delhivery-annual-report-fy24-excerpt.pdf", 21, "revenue_from_operations"),
        ("03-delhivery-q4-fy24-earnings-presentation.pdf", 5, "revenue_from_services"),
    ],
)
def test_each_delhivery_document_yields_a_grounded_offline_fact(
    filename: str, page_index: int, expected_predicate: str
) -> None:
    ingested_page = PdfIngestor().extract_page(
        DELHIVERY_DATA / filename,
        page_index,
        include_tables=False,
    )
    outcome = DeterministicExtractor().extract_page(
        ingested_page,
        default_subject=DELHIVERY,
        publisher=filename,
    )

    assert expected_predicate in {item.predicate.key for item in outcome.facts}
    passages = {item.id: item for item in ingested_page.passages}
    for fact in outcome.facts:
        for evidence in fact.evidence:
            source = passages[evidence.passage_id]
            local_start = evidence.quote_start - source.char_start
            local_end = evidence.quote_end - source.char_start
            assert source.text[local_start:local_end] == evidence.quote


def test_a_table_states_its_unit_once_and_every_cell_inherits_it() -> None:
    """Financial tables print the unit in the corner cell, not in every number.

    Read without that hint a cell is just "8,142", which cannot be compared against
    "₹ 81,415.38 million" from another document even though both state the same amount.
    """
    extractor = DeterministicExtractor()
    table = ExtractedTable(
        page_index=0,
        table_index=0,
        bbox=BoundingBox(x0=0, y0=0, x1=100, y1=50),
        rows=(
            ("₹ Cr", "FY23", "FY24"),
            ("Revenue from customers", "7,225", "8,142"),
        ),
        cells=(),
    )

    assert extractor._table_unit_hint(table) == "₹ Cr"

    applied = extractor._normalize_table_value("8,142", "₹ Cr")
    assert applied.value is not None
    assert applied.value.currency == "INR"
    assert applied.value.number == Decimal("81420000000")
    # The cell's own text stays the reported value; only the reading comes from the header.
    assert applied.value.raw == "8,142"
    assert any(item.code == "unit_from_table_header" for item in applied.warnings)


def test_a_percentage_cell_does_not_inherit_a_currency_unit() -> None:
    """A "% margin" row inside a "₹ Cr" table is not measured in crores."""
    applied = DeterministicExtractor()._normalize_table_value("18.4%", "₹ Cr")

    assert applied.value is not None
    assert applied.value.unit == "percent"
    assert applied.value.currency is None


def test_column_headers_are_not_mistaken_for_a_unit_statement() -> None:
    table = ExtractedTable(
        page_index=0,
        table_index=0,
        bbox=BoundingBox(x0=0, y0=0, x1=100, y1=50),
        rows=(("Particulars", "Q1 FY23", "Q2 FY23"), ("Shipments", "94", "134")),
        cells=(),
    )

    assert DeterministicExtractor()._table_unit_hint(table) is None


def _pdf_named(*page_texts: str):
    """Build the smallest IngestedPdf that detect_reporting_entity can read."""
    from factlayer.ingest import IngestedPdf
    from factlayer.schema import Document

    pages = tuple(
        IngestedPage(
            page_index=index,
            passages=(
                Passage(
                    id=f"passage-{index}",
                    document_id="doc-1",
                    page_index=index,
                    reading_order=0,
                    text=text,
                ),
            ),
            tables=(),
            warnings=(),
            elapsed_ms=0.0,
        )
        for index, text in enumerate(page_texts)
    )
    document = Document(content_hash="a" * 64, original_filename="report.pdf")
    return IngestedPdf(document=document.model_copy(update={"id": "doc-1"}), pages=pages)


def test_the_company_named_on_the_cover_becomes_the_subject() -> None:
    pdf = _pdf_named(
        "Delhivery Limited\nAnnual Report",
        "Delhivery Limited reported growth.",
        "Delhivery Limited continued to invest.",
    )

    entity = detect_reporting_entity(pdf)

    assert entity is not None
    assert entity.canonical_name == "Delhivery Limited"
    assert entity.entity_type == "company"


def test_a_vendor_credited_inside_a_table_is_not_the_subject() -> None:
    """A macro report has no single reporting company, so nothing should be claimed.

    Statistical annexes credit data providers, and treating one as the document's
    subject would file a whole economy's figures under a vendor's name.
    """
    pdf = _pdf_named(
        "India: 2025 Article IV Consultation",
        "Staff Report",
        "Sources: CEIC Data Company Ltd; IMF staff calculations.",
        "Sources: CEIC Data Company Ltd.",
        "Sources: CEIC Data Company Ltd.",
    )

    assert detect_reporting_entity(pdf) is None


def test_no_subject_is_claimed_when_no_company_clearly_leads() -> None:
    """A prospectus cover lists bankers and registrars, none of them the issuer."""
    pdf = _pdf_named(
        "Kotak Mahindra Capital Company Limited\nLink Intime India Private Limited",
        "Offer details.",
    )

    assert detect_reporting_entity(pdf) is None


def _status_fact_for(text: str, *, subject_warning: str | None = None):
    import re as _re
    from factlayer.schema import EntityReference as _Ref

    passage = Passage(id="p1", document_id="d1", page_index=0, reading_order=0, text=text)
    match = _re.search(r"\bDIN\s*:?\s*(\d{8})\b", text)
    assert match is not None
    return DeterministicExtractor()._status_fact(
        passage,
        _Ref(id="din:01173669", canonical_name="Suvir Suren Sujan", entity_type="person"),
        match,
        publisher="prospectus.pdf",
        document_context=None,
        subject_warning=subject_warning,
    )


def test_a_tenure_statement_counts_as_still_serving() -> None:
    """Without this, only departures were recorded and nothing to compare them to.

    A prospectus says when a director's term began; the later report says they left.
    Both halves are needed before the as-of date can explain the difference.
    """
    from datetime import date as _date

    fact = _status_fact_for(
        "Term: Liable to retire by rotation Period of Directorship: "
        "Since March 7, 2019 DIN: 01173669"
    )

    assert fact is not None
    assert fact.value.state == "serving"
    assert fact.context.as_of == _date(2019, 3, 7)


def test_a_departure_outranks_a_tenure_note_in_the_same_paragraph() -> None:
    """Board-change paragraphs mention the joining date and then the exit."""
    fact = _status_fact_for(
        "DIN: 01173669 was a director since March 7, 2019 and resigned from the "
        "Board with effect from August 24, 2023."
    )

    assert fact is not None
    assert fact.value.state == "resigned"


def test_an_unresolved_name_makes_the_status_doubtful_too() -> None:
    """Board tables print in columns and bleed, so the doubt has to travel.

    A status is only as trustworthy as the person it is pinned to.
    """
    fact = _status_fact_for(
        "Period of Directorship: Since March 7, 2019 DIN: 01173669",
        subject_warning="A DIN was found, but the person's name could not be resolved nearby.",
    )

    assert fact is not None
    assert fact.review_state is ReviewState.NEEDS_REVIEW
    assert any("name could not be resolved" in w for w in fact.warnings)


MACRO_DATA = Path(__file__).resolve().parent.parent / "starter-datasets" / "india-macroeconomy"


@pytest.mark.parametrize(
    "filename",
    [
        "01-india-economic-survey-2024-25-excerpt.pdf",
        "02-rbi-annual-report-2024-25-excerpt.pdf",
        "03-imf-india-2025-article-iv-excerpt.pdf",
    ],
)
def test_macroeconomic_reports_yield_grounded_facts_without_new_rules(filename: str) -> None:
    """The pipeline was tuned on company filings; these reports were never used for it.

    Nothing here asserts a particular measure, because the point is the opposite: an
    unfamiliar document should produce grounded facts through the same generic path.
    """
    subject = EntityReference(id="place:india", canonical_name="India", entity_type="place")
    ingested_page = PdfIngestor().extract_page(MACRO_DATA / filename, 10, include_tables=False)

    outcome = DeterministicExtractor().extract_page(
        ingested_page,
        default_subject=subject,
        publisher=filename,
    )

    assert outcome.facts, "an unfamiliar report should still yield facts"
    passages = {item.id: item for item in ingested_page.passages}
    for fact in outcome.facts:
        for evidence in fact.evidence:
            source = passages[evidence.passage_id]
            local_start = evidence.quote_start - source.char_start
            local_end = evidence.quote_end - source.char_start
            assert source.text[local_start:local_end] == evidence.quote


def test_no_reporting_company_is_claimed_for_a_macroeconomic_report() -> None:
    """A report about an economy has no reporting company, and guessing one is harmful.

    Statistical annexes credit data vendors; treating one as the subject would file a
    whole economy's figures under its name.
    """
    pdf = PdfIngestor().ingest(MACRO_DATA / "03-imf-india-2025-article-iv-excerpt.pdf")

    assert detect_reporting_entity(pdf) is None
