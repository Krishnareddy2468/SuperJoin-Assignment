"""Deterministic fact extraction that works without an API key.

These rules favour precision and traceability over coverage. They extract facts
only when a value can be paired with a useful label and grounded back to the
exact passage or table cell that supplied it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date

from factlayer.ingest import ExtractedTable, IngestedPage, IngestedPdf, TableCell
from factlayer.normalize import (
    NormalizationWarning,
    PredicateRegistry,
    normalize_context,
    normalize_date,
    normalize_number,
    normalize_period,
)
from factlayer.schema import (
    CategoricalValue,
    ContextEnvelope,
    EntityReference,
    Evidence,
    ExtractionFailure,
    ExtractionMethod,
    Fact,
    FailureStage,
    IdentifierValue,
    Passage,
    PassageRole,
    PredicateReference,
    ReviewState,
)


_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?"
_SCALE = r"(?:thousand|lakh|lakhs|lac|lacs|million|millions|mn|crore|crores|cr|billion|billions|bn)"
_UNIT = r"(?:km|kilometres?|kilometers?|kg|kilograms?|metric\s+tonnes?|tonnes?|shipments?|employees?|days?|months?|years?)"
_CURRENCY = r"(?:₹|rs\.?|inr|us\$|usd|\$|eur|€|gbp|£)"
_NUMERIC_CANDIDATE = re.compile(
    rf"(?<![\w])(?:\(\s*)?(?:{_CURRENCY}\s*)?{_NUMBER}(?:\s*{_SCALE})?(?:\s*{_UNIT})?"
    rf"(?:\s*(?:%|percent|pct|basis\s+points?|bps|x))?(?:\s*\))?",
    flags=re.IGNORECASE,
)
_DATE_TEXT = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
)
_DIN = re.compile(r"\bDIN\s*:?\s*(\d{8})\b", re.IGNORECASE)
_CIN = re.compile(r"\bCIN\s*:?\s*([A-Z0-9]{21})\b", re.IGNORECASE)
_PERSON_NAME = re.compile(
    r"\b(?:Mr|Ms|Mrs|Dr)\.?\s+([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,5})"
)
# A name on its own line, the way a board table opens each director's record. Each word
# must be capitalised and then lower case, which is what keeps acronyms out: "Delhivery
# Robotics LLC" and "ICICI Prudential" are not people.
_BARE_PERSON_NAME = re.compile(r"[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+){1,3}\.?")
# Column headings and field labels in these tables read like Title Case names. Treating
# one as a person is worse than finding no name at all, because a wrong subject is
# asserted with confidence instead of being flagged.
_TABLE_VOCABULARY = {
    "directorship",
    "directorships",
    "director",
    "directors",
    "designation",
    "address",
    "occupation",
    "term",
    "period",
    "name",
    "age",
    "years",
    "companies",
    "company",
    "board",
    "date",
    "birth",
    "other",
    "nominee",
    "executive",
    "independent",
    "chairman",
    "particulars",
    "total",
}
_ORGANISATION_TAIL = re.compile(
    r"\b(?:Limited|Ltd|LLP|LLC|PLC|Inc|Incorporated|Corporation|Corp|Private|Pvt|"
    r"Pte|GmbH|BV|SA|Company|Bank|Fund|Trust|Capital|Partners|Ventures?|Holdings?|"
    r"Technologies|Solutions|Services|Industries|Systems)\b",
    re.IGNORECASE,
)
# Normalization notes that describe how a value was interpreted rather than raising a
# doubt about it. They stay visible on the fact, but they do not send it to review.
_INFORMATIONAL_NORMALIZATION_CODES = {"unit_from_table_header"}
# How much of a document counts as front matter when looking for the organisation it
# reports on. A filing names itself on the cover, so this stays deliberately tight: a
# wider window starts picking up data vendors and agencies credited in early tables.
# Two pages allows for a cover followed by a title or contents spread.
_FRONT_MATTER_PAGES = 2
# How much of the previous page can still belong to a record that carried over.
_PAGE_CARRYOVER_CHARS = 600
_INCORPORATED_NAME = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9&.'’-]*\s+){0,4}?"
    r"(?:Limited|Ltd\.?|Incorporated|Inc\.?|Corporation|Corp\.?|PLC|LLP))\b"
)
# Words that show up capitalised in front of a company suffix without being part of
# the registered name, usually because a sentence or heading ran into it.
_NAME_LEAD_NOISE = {
    "the",
    "and",
    "of",
    "for",
    "our",
    "its",
    "with",
    "by",
    "to",
    "in",
    "on",
    "company",
    "subsidiary",
    "subsidiaries",
    "associate",
    "holding",
    "private",
    "public",
}


def detect_reporting_entity(pdf: IngestedPdf) -> EntityReference | None:
    """Work out which organisation a document is reporting on.

    Facts lifted from prose rarely carry their own subject: an annual report writes
    "revenue from operations stood at ..." and trusts the reader to know whose revenue
    that is. Attributing such facts to the PDF itself makes them permanently
    incomparable, because every document is a unique subject, so the same measure
    published in two filings can never meet. Naming the reporting entity is what lets
    them meet.

    The registered name is the signal that travels between documents; a CIN is better
    still but often appears in only one of them, so it is attached as an identifier
    when present rather than being required. Returns ``None`` when no organisation
    stands out - a statistical bulletin about a whole economy has no single reporting
    company - which leaves the caller free to keep the document-level fallback instead
    of inventing a subject.
    """
    counts: dict[str, int] = {}
    introduced: set[str] = set()
    cin: str | None = None
    for page in pdf.pages:
        for passage in page.passages:
            if cin is None and (found := _CIN.search(passage.text)):
                cin = found.group(1).upper()
            for match in _INCORPORATED_NAME.finditer(passage.text):
                name = _trim_company_name(match.group(1))
                if not name:
                    continue
                counts[name] = counts.get(name, 0) + 1
                if page.page_index < _FRONT_MATTER_PAGES:
                    introduced.add(name)
    # A document that reports on an organisation introduces it up front, on the cover or
    # title page. Companies that only turn up deep inside are being cited, not reported
    # on: a central bank's annual report mentions agencies and a statistical annex
    # credits its data vendor, and neither is the subject of the document's figures.
    counts = {name: total for name, total in counts.items() if name in introduced}
    def geographic_subject() -> EntityReference | None:
        # Institutional and country reports do not have a reporting company, but their
        # figures still need a stable subject if several publications describe the same
        # place. Reuse the geography cues already extracted by the context normalizer;
        # this stays content-driven instead of depending on filenames or a report list.
        geography_counts: Counter[str] = Counter()
        for page in pdf.pages[:6]:
            for passage in page.passages:
                if passage.role is not PassageRole.BODY:
                    continue
                context = normalize_context(passage.text)
                geography = context.value.geography if context.value else None
                if geography and geography.casefold() != "global":
                    # Do not mistake the country token in a vendor's registered name
                    # (for example, ``... India Private Limited``) for the report's
                    # geographic subject.
                    if re.search(
                        rf"\b{re.escape(geography)}\s+(?:private\s+)?(?:limited|ltd\.?|incorporated|inc\.?)\b",
                        passage.text,
                        re.IGNORECASE,
                    ):
                        continue
                    geography_counts[geography] += 1
        if not geography_counts:
            return None
        geography, _ = geography_counts.most_common(1)[0]
        normalized = re.sub(r"[^a-z0-9]+", "_", geography.casefold()).strip("_")
        return EntityReference(
            id=f"place:{normalized}",
            canonical_name=geography,
            entity_type="place",
        )

    if not counts:
        return geographic_subject()
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    name, occurrences = ranked[0]
    if occurrences < 2:
        return geographic_subject()
    # Being the most mentioned company is not proof of being the reporting one. A
    # prospectus discusses acquisitions and shareholders at similar length to the
    # issuer, and picking the runner-up would file the whole document's figures under
    # a subsidiary. Insist on a clear margin and otherwise return None, which leaves
    # the facts attributed to the document and flagged for review. Admitting "we
    # cannot tell whose number this is" beats silently choosing the wrong company.
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if runner_up and occurrences < 2 * runner_up:
        return geographic_subject()
    return EntityReference(
        id=f"cin:{cin}" if cin else None,
        canonical_name=name,
        entity_type="company",
    )


def _trim_company_name(candidate: str) -> str:
    words = re.sub(r"\s+", " ", candidate).strip().split(" ")
    while words and words[0].casefold() in _NAME_LEAD_NOISE:
        words.pop(0)
    # A suffix on its own ("Limited") names nothing.
    if len(words) < 2:
        return ""
    return " ".join(words)


@dataclass(frozen=True)
class ExtractionStats:
    candidates_seen: int = 0
    facts_accepted: int = 0
    candidates_ignored: int = 0
    candidates_rejected: int = 0

    def combined(self, other: "ExtractionStats") -> "ExtractionStats":
        return ExtractionStats(
            candidates_seen=self.candidates_seen + other.candidates_seen,
            facts_accepted=self.facts_accepted + other.facts_accepted,
            candidates_ignored=self.candidates_ignored + other.candidates_ignored,
            candidates_rejected=self.candidates_rejected + other.candidates_rejected,
        )


@dataclass(frozen=True)
class ExtractionOutcome:
    facts: tuple[Fact, ...] = field(default_factory=tuple)
    failures: tuple[ExtractionFailure, ...] = field(default_factory=tuple)
    stats: ExtractionStats = field(default_factory=ExtractionStats)


class DeterministicExtractor:
    """Extract conservative numeric and identifier facts from ingested pages."""

    def __init__(self, predicate_registry: PredicateRegistry | None = None):
        self.predicates = predicate_registry or PredicateRegistry()

    def extract_pdf(
        self,
        pdf: IngestedPdf,
        *,
        default_subject: EntityReference | None = None,
        document_context: ContextEnvelope | None = None,
    ) -> ExtractionOutcome:
        subject = default_subject or self._document_subject(
            pdf.document.id, pdf.document.original_filename
        )
        fallback_subject = default_subject is None
        outcome = ExtractionOutcome()
        preceding_text = ""
        for page in pdf.pages:
            page_outcome = self.extract_page(
                page,
                default_subject=subject,
                publisher=pdf.document.original_filename,
                document_context=document_context,
                fallback_subject=fallback_subject,
                preceding_text=preceding_text,
            )
            # Keep only the tail: enough to carry a record over a page break, not enough
            # to let an unrelated name from higher up the previous page reach forward.
            preceding_text = "\n".join(
                item.text for item in page.passages
            )[-_PAGE_CARRYOVER_CHARS:]
            outcome = ExtractionOutcome(
                facts=(*outcome.facts, *page_outcome.facts),
                failures=(*outcome.failures, *page_outcome.failures),
                stats=outcome.stats.combined(page_outcome.stats),
            )
        return self._deduplicate(outcome)

    def extract_page(
        self,
        page: IngestedPage,
        *,
        default_subject: EntityReference,
        publisher: str | None = None,
        document_context: ContextEnvelope | None = None,
        fallback_subject: bool = False,
        preceding_text: str = "",
    ) -> ExtractionOutcome:
        facts: list[Fact] = []
        failures: list[ExtractionFailure] = []
        stats = ExtractionStats()
        table_passages = self._table_passage_ids(page)

        for passage in page.passages:
            if passage.role is not PassageRole.BODY:
                continue
            identifier_facts, identifier_failures = self._identifier_facts(
                passage,
                default_subject=default_subject,
                publisher=publisher,
                document_context=document_context,
                preceding_text=preceding_text,
            )
            facts.extend(identifier_facts)
            failures.extend(identifier_failures)
            stats = stats.combined(
                ExtractionStats(
                    candidates_seen=len(identifier_facts) + len(identifier_failures),
                    facts_accepted=len(identifier_facts),
                    candidates_rejected=len(identifier_failures),
                )
            )

            if passage.id in table_passages:
                continue
            numeric_facts, numeric_failures, ignored, seen = self._numeric_facts(
                passage,
                default_subject=default_subject,
                publisher=publisher,
                document_context=document_context,
                fallback_subject=fallback_subject,
            )
            facts.extend(numeric_facts)
            failures.extend(numeric_failures)
            stats = stats.combined(
                ExtractionStats(
                    candidates_seen=seen,
                    facts_accepted=len(numeric_facts),
                    candidates_ignored=ignored,
                    candidates_rejected=len(numeric_failures),
                )
            )

        for table in page.tables:
            table_facts, table_failures, seen = self._table_facts(
                table,
                page.passages,
                default_subject=default_subject,
                publisher=publisher,
                document_context=document_context,
                fallback_subject=fallback_subject,
            )
            facts.extend(table_facts)
            failures.extend(table_failures)
            stats = stats.combined(
                ExtractionStats(
                    candidates_seen=seen,
                    facts_accepted=len(table_facts),
                    candidates_rejected=len(table_failures),
                )
            )
        return self._deduplicate(
            ExtractionOutcome(tuple(facts), tuple(failures), stats)
        )

    def _numeric_facts(
        self,
        passage: Passage,
        *,
        default_subject: EntityReference,
        publisher: str | None,
        document_context: ContextEnvelope | None,
        fallback_subject: bool,
    ) -> tuple[list[Fact], list[ExtractionFailure], int, int]:
        facts: list[Fact] = []
        failures: list[ExtractionFailure] = []
        ignored = 0
        matches = list(_NUMERIC_CANDIDATE.finditer(passage.text))
        for match in matches:
            raw = match.group().strip()
            if self._is_unsupported_number(raw, passage):
                ignored += 1
                continue
            normalized = normalize_number(raw)
            if normalized.value is None:
                failures.append(
                    self._failure(
                        passage,
                        FailureStage.NORMALIZATION,
                        normalized.warnings[0].message,
                        {"candidate": raw},
                    )
                )
                continue
            label, label_confidence = self._infer_predicate(
                passage.text, match.start(), match.end()
            )
            if label is None:
                failures.append(
                    self._failure(
                        passage,
                        FailureStage.EXTRACTION,
                        "A numeric value was found, but no reliable predicate label was nearby.",
                        {"candidate": raw},
                    )
                )
                continue
            fact = self._numeric_fact(
                passage,
                match.start(),
                match.end(),
                raw,
                label,
                normalized,
                default_subject=default_subject,
                publisher=publisher,
                document_context=document_context,
                extraction_confidence=label_confidence,
                fallback_subject=fallback_subject,
            )
            if fact:
                facts.append(fact)
        return facts, failures, ignored, len(matches)

    def _numeric_fact(
        self,
        passage: Passage,
        value_start: int,
        value_end: int,
        raw: str,
        label: str,
        normalized,
        *,
        default_subject: EntityReference,
        publisher: str | None,
        document_context: ContextEnvelope | None,
        extraction_confidence: float,
        fallback_subject: bool,
        evidence_bbox=None,
        table_context: str | None = None,
    ) -> Fact | None:
        predicate = self.predicates.register(label, value_kind="number")
        evidence = self._evidence(
            passage,
            value_start,
            value_end,
            extractor=ExtractionMethod.TABLE if evidence_bbox else ExtractionMethod.DETERMINISTIC,
            bbox=evidence_bbox,
            quote_value_only=bool(evidence_bbox),
        )
        context_text = f"{passage.text}\n{table_context or ''}".strip()
        context, context_warnings, context_confidence = self._context(
            context_text, publisher=publisher, default=document_context
        )
        warnings = [warning.message for warning in normalized.warnings]
        warnings.extend(warning.message for warning in context_warnings)
        if fallback_subject:
            warnings.append("The subject falls back to the source document and needs review.")
        # Some notes explain how a value was read without casting doubt on it. Reading a
        # unit from the table header that governs the cell is ordinary, correct
        # behaviour, and recording it should not push an otherwise solid fact into
        # review - doing so would suppress every figure in a well-formed table.
        doubtful = [
            warning
            for warning in normalized.warnings
            if warning.code not in _INFORMATIONAL_NORMALIZATION_CODES
        ]
        review_state = (
            ReviewState.NEEDS_REVIEW
            if doubtful or context_warnings or fallback_subject
            else ReviewState.READY
        )
        fact_id = self._fact_id(
            passage.document_id,
            passage.page_index,
            predicate.key,
            raw,
            value_start,
        )
        return Fact(
            id=fact_id,
            subject=default_subject,
            predicate=PredicateReference(key=predicate.key, display_name=predicate.display_name),
            value=normalized.value,
            context=context,
            evidence=[evidence],
            extraction_confidence=min(extraction_confidence, evidence.confidence),
            normalization_confidence=min(normalized.confidence, context_confidence),
            review_state=review_state,
            warnings=warnings,
        )

    def _table_facts(
        self,
        table: ExtractedTable,
        passages: tuple[Passage, ...],
        *,
        default_subject: EntityReference,
        publisher: str | None,
        document_context: ContextEnvelope | None,
        fallback_subject: bool,
    ) -> tuple[list[Fact], list[ExtractionFailure], int]:
        if len(table.rows) < 2 or len(table.rows[0]) < 2:
            return [], [], 0
        headers = table.rows[0]
        cells = {(cell.row_index, cell.column_index): cell for cell in table.cells}
        unit_hint = self._table_unit_hint(table)
        facts: list[Fact] = []
        failures: list[ExtractionFailure] = []
        seen = 0
        for row_index, row in enumerate(table.rows[1:], start=1):
            if not row or not row[0].strip():
                continue
            label = self._clean_predicate_label(row[0])
            if not label:
                continue
            for column_index, raw in enumerate(row[1:], start=1):
                raw = raw.strip()
                if not raw or raw in {"-", "—", "–", "n/a", "NA"}:
                    continue
                if not _NUMERIC_CANDIDATE.fullmatch(raw):
                    continue
                seen += 1
                normalized = self._normalize_table_value(raw, unit_hint)
                cell = cells.get((row_index, column_index))
                passage_match = self._passage_for_table_value(passages, raw, cell)
                if normalized.value is None or passage_match is None:
                    reference = passage_match or self._nearest_passage(passages, cell)
                    if reference:
                        reason = (
                            normalized.warnings[0].message
                            if normalized.value is None
                            else "The table value could not be grounded to extracted page text."
                        )
                        failures.append(
                            self._failure(
                                reference,
                                FailureStage.GROUNDING,
                                reason,
                                {"table_value": raw, "row_label": label},
                            )
                        )
                    continue
                value_start = passage_match.text.find(raw)
                header = headers[column_index] if column_index < len(headers) else ""
                fact = self._numeric_fact(
                    passage_match,
                    value_start,
                    value_start + len(raw),
                    raw,
                    label,
                    normalized,
                    default_subject=default_subject,
                    publisher=publisher,
                    document_context=document_context,
                    extraction_confidence=0.9,
                    fallback_subject=fallback_subject,
                    evidence_bbox=cell.bbox if cell else table.bbox,
                    table_context=header,
                )
                # Record where the cell sat. Grid position is the only reliable way to
                # tell later which figures are the components of a stated total, and it
                # is knowable here and nowhere else: reconstructing it downstream from
                # bounding boxes mixes up rows whose labels repeat.
                fact = fact.model_copy(
                    update={
                        "context": fact.context.model_copy(
                            update={
                                "qualifiers": {
                                    **fact.context.qualifiers,
                                    "table_index": str(table.table_index),
                                    "table_row": str(row_index),
                                    "table_column": str(column_index),
                                }
                            }
                        )
                    }
                )
                facts.append(fact)
        return facts, failures, seen

    @staticmethod
    def _table_unit_hint(table: ExtractedTable) -> str | None:
        """Find a currency or scale that a table states once for all of its cells.

        Financial tables almost never repeat the unit in every cell. They print it once
        - usually in the corner cell of the header row, as "₹ Cr" or "(₹ in million)" -
        and every number below inherits it. Read without that hint a cell is just
        "8,142", which cannot be compared against "₹ 81,415.38 million" from another
        document even though both state the same amount.
        """
        for text in table.rows[0] if table.rows else ():
            if not text or not text.strip():
                continue
            # A header that carries a number of its own ("Q1 FY23") is a column label,
            # not a unit statement.
            if re.search(_NUMBER, text):
                continue
            # The surrounding guards matter: "rs" sits inside "Particulars", and a
            # header column called that is not a statement about currency.
            match = re.search(
                rf"(?<![A-Za-z])(?P<hint>{_CURRENCY}\s*(?:in\s+)?{_SCALE}?|{_SCALE})(?![A-Za-z])",
                text,
                re.IGNORECASE,
            )
            if match:
                hint = re.sub(r"\s+", " ", match.group("hint")).strip()
                if re.search(rf"{_CURRENCY}|{_SCALE}", hint, re.IGNORECASE):
                    return hint
        return None

    @staticmethod
    def _normalize_table_value(raw: str, unit_hint: str | None):
        """Normalize a cell, applying the table's shared unit when the cell omits it."""
        normalized = normalize_number(raw)
        if unit_hint is None or normalized.value is None:
            return normalized
        # Never override what the cell states for itself, and leave percentages and
        # ratios alone - a "% margin" row is not measured in crores.
        if normalized.value.currency is not None or normalized.value.unit is not None:
            return normalized
        if normalized.value.scale != 1:
            return normalized
        hinted = normalize_number(f"{unit_hint} {raw}")
        if hinted.value is None:
            return normalized
        return replace(
            normalized,
            # Keep the cell's own text as the reported value; only the interpretation
            # comes from the header, and the warning records that.
            value=hinted.value.model_copy(update={"raw": raw}),
            warnings=(
                *normalized.warnings,
                NormalizationWarning(
                    code="unit_from_table_header",
                    message=(
                        f"The unit {unit_hint!r} was taken from the table header "
                        "because the cell did not repeat it."
                    ),
                ),
            ),
        )

    def _identifier_facts(
        self,
        passage: Passage,
        *,
        default_subject: EntityReference,
        publisher: str | None,
        document_context: ContextEnvelope | None,
        preceding_text: str = "",
    ) -> tuple[list[Fact], list[ExtractionFailure]]:
        facts: list[Fact] = []
        failures: list[ExtractionFailure] = []
        din_matches = list(_DIN.finditer(passage.text))
        for index, match in enumerate(din_matches):
            # Each record ends at its own DIN, so the previous DIN is where it started.
            record_start = din_matches[index - 1].end() if index else 0
            subject = self._person_for_record(passage.text, record_start, match.start())
            if subject is None and index == 0 and preceding_text:
                # A director's record can begin on the previous page: the name, address
                # and occupation sit at the foot of one page and the DIN at the head of
                # the next. Only the first DIN on a page can be in that position.
                subject = self._person_for_record(
                    preceding_text + "\n" + passage.text[:match.start()],
                    0,
                    len(preceding_text) + 1 + match.start(),
                )
            subject = subject or default_subject
            din = match.group(1)
            if subject is default_subject:
                subject = subject.model_copy(update={"id": f"din:{din}"})
                name_warning = "A DIN was found, but the person's name could not be resolved nearby."
            else:
                subject = subject.model_copy(update={"id": f"din:{din}"})
                name_warning = None
            identifier = self._identifier_fact(
                passage,
                match,
                subject,
                scheme="DIN",
                predicate_label="Director identification number",
                publisher=publisher,
                document_context=document_context,
                warning=name_warning,
            )
            facts.append(identifier)
            status_fact = self._status_fact(
                passage,
                subject,
                match,
                publisher=publisher,
                document_context=document_context,
                subject_warning=name_warning,
                record_start=record_start,
                record_end=(
                    din_matches[index + 1].start()
                    if index + 1 < len(din_matches)
                    else len(passage.text)
                ),
            )
            if status_fact:
                facts.append(status_fact)

        for match in _CIN.finditer(passage.text):
            cin = match.group(1).upper()
            subject = default_subject.model_copy(update={"id": f"cin:{cin}", "entity_type": "company"})
            facts.append(
                self._identifier_fact(
                    passage,
                    match,
                    subject,
                    scheme="CIN",
                    predicate_label="Corporate identity number",
                    publisher=publisher,
                    document_context=document_context,
                )
            )
        return facts, failures

    def _identifier_fact(
        self,
        passage: Passage,
        match: re.Match[str],
        subject: EntityReference,
        *,
        scheme: str,
        predicate_label: str,
        publisher: str | None,
        document_context: ContextEnvelope | None,
        warning: str | None = None,
    ) -> Fact:
        predicate = self.predicates.register(predicate_label, value_kind="identifier")
        context, context_warnings, context_confidence = self._context(
            passage.text, publisher=publisher, default=document_context
        )
        warnings = [item.message for item in context_warnings]
        if warning:
            warnings.append(warning)
        raw = match.group()
        evidence = self._evidence(
            passage, match.start(), match.end(), extractor=ExtractionMethod.DETERMINISTIC
        )
        return Fact(
            id=self._fact_id(passage.document_id, passage.page_index, predicate.key, raw, match.start()),
            subject=subject,
            predicate=PredicateReference(key=predicate.key, display_name=predicate.display_name),
            value=IdentifierValue(raw=raw, value=match.group(1).upper(), scheme=scheme),
            context=context,
            evidence=[evidence],
            extraction_confidence=0.97 if warning is None else 0.75,
            normalization_confidence=context_confidence,
            review_state=ReviewState.NEEDS_REVIEW if warnings else ReviewState.READY,
            warnings=warnings,
        )

    def _status_fact(
        self,
        passage: Passage,
        subject: EntityReference,
        din_match: re.Match[str],
        *,
        publisher: str | None,
        document_context: ContextEnvelope | None,
        subject_warning: str | None = None,
        record_start: int | None = None,
        record_end: int | None = None,
    ) -> Fact | None:
        # Stay inside this director's record. A fixed character window reaches into the
        # rows on either side, and in a table listing four directors that means picking
        # up a neighbour's appointment date and reporting it as this person's.
        window_start = (
            max(0, din_match.start() - 160) if record_start is None else record_start
        )
        window_end = (
            min(len(passage.text), din_match.end() + 360)
            if record_end is None
            else min(len(passage.text), record_end)
        )
        nearby_text = passage.text[window_start:window_end]
        lowered = nearby_text.casefold()
        # Order matters. A departure is checked before a tenure statement because a
        # board-change paragraph often mentions when someone joined and then that they
        # left; the exit is the later state and the one worth recording.
        states = [
            (
                "resigned",
                r"\bresigned\s+from\s+the\s+(?:board|office)\b|\bresigned\s+with\s+effect\b",
            ),
            ("ceased", r"\bceased\s+to\s+be\b|\bceased\s+from\b"),
            (
                "serving",
                r"\bperiod\s+of\s+directorship\s*:?\s*since\b"
                r"|\bdirector\s+of\s+the\s+company\s+since\b"
                r"|\bcontinues?\s+(?:as|to\s+be)\s+(?:a\s+)?director\b",
            ),
            ("appointed", r"\bappointed\s+(?:as|to)\b|\bre-?appointed\b"),
        ]
        status_result = next(
            (
                (label, match)
                for label, pattern in states
                if (match := re.search(pattern, lowered)) is not None
            ),
            None,
        )
        if status_result is None:
            return None
        state, local_status_match = status_result
        status_start = window_start + local_status_match.start()
        status_end = window_start + local_status_match.end()
        date_match = re.search(
            _DATE_TEXT,
            passage.text[status_start:window_end],
            re.IGNORECASE,
        )
        effective_date: date | None = None
        warnings: list[str] = []
        # A status is only as trustworthy as the person it is pinned to. Board tables
        # print in columns and often bleed, so if the name beside this DIN could not be
        # resolved, that doubt has to travel with the status too - not just with the
        # identifier fact.
        if subject_warning:
            warnings.append(subject_warning)
        if date_match:
            parsed_date = normalize_date(date_match.group())
            if parsed_date.value:
                effective_date = parsed_date.value.value
            else:
                warnings.extend(item.message for item in parsed_date.warnings)
        else:
            warnings.append("The role change has no reliable effective date nearby.")

        # Read context from this director's record only. Using the whole passage drags in
        # the dates and terms of everyone else in the table, which shows up as an
        # "ambiguous reporting period" warning and pushes a clean fact into review.
        context, context_warnings, context_confidence = self._context(
            passage.text[window_start:window_end],
            publisher=publisher,
            default=document_context,
        )
        # Holding a board seat is a state on a date, not a quantity measured over a
        # reporting period. A director's record lists a term and a start date, so period
        # detection has plenty to be uncertain about and none of it changes the fact.
        # Dropping the period, and the doubts that came with it, keeps the as-of date as
        # the only time context - which is exactly what the comparison uses.
        context_warnings = tuple(
            warning
            for warning in context_warnings
            if "reporting period" not in warning.message
        )
        if effective_date:
            context = context.model_copy(
                update={"as_of": effective_date, "period": None}
            )
        warnings.extend(item.message for item in context_warnings)
        predicate = self.predicates.register("Board membership status", value_kind="category")
        date_end = status_start + date_match.end() if date_match else status_end
        quote_start = min(din_match.start(), status_start)
        quote_end = max(din_match.end(), status_end, date_end)
        evidence = self._evidence(
            passage, quote_start, quote_end, extractor=ExtractionMethod.DETERMINISTIC
        )
        return Fact(
            id=self._fact_id(
                passage.document_id, passage.page_index, predicate.key, state, quote_start
            ),
            subject=subject,
            predicate=PredicateReference(key=predicate.key, display_name=predicate.display_name),
            value=CategoricalValue(
                raw=passage.text[status_start:status_end],
                state=state,
            ),
            context=context,
            evidence=[evidence],
            extraction_confidence=0.94,
            normalization_confidence=context_confidence,
            review_state=ReviewState.NEEDS_REVIEW if warnings else ReviewState.READY,
            warnings=warnings,
        )

    def _infer_predicate(self, text: str, start: int, end: int) -> tuple[str | None, float]:
        before_value = text[max(0, start - 160) : start]
        if re.search(
            r"\b(?:real\s+)?gdp\b.{0,120}\b(?:growth|percentage\s+change|grow|grew|expanded|moderated|increased|declined)\b",
            before_value,
            re.IGNORECASE,
        ):
            return "real GDP growth", 0.9
        sentence_start = self._sentence_start(text, start)
        sentence_before = re.sub(r"\s+", " ", text[sentence_start:start])
        before = sentence_before.strip(" •:-\n\t")
        before = re.sub(r"^y\s+", "", before, flags=re.IGNORECASE)
        after_line = text[end : text.find("\n", end) if text.find("\n", end) != -1 else len(text)]
        following_line_end = text.find("\n", end + len(after_line) + 1)
        following = text[end : following_line_end if following_line_end != -1 else len(text)]

        verb_match = re.search(
            r"(?P<label>[A-Za-z][A-Za-z0-9 &'’()/.,-]{2,140}?)\s+"
            r"(?:was|were|is|are|stood\s+at|amounted\s+to|totalled|totaled|reached)\s*$",
            before,
            re.IGNORECASE,
        )
        if verb_match:
            label = self._clean_predicate_label(verb_match.group("label"))
            return (label, 0.92) if label else (None, 0.0)

        change_match = re.search(
            r"(?P<label>[A-Za-z][A-Za-z0-9 &'’()/.,-]{2,120}?)\s+"
            r"(?:increased|decreased|grew|declined|changed)\s+(?:by|of)\s*$",
            before,
            re.IGNORECASE,
        )
        if change_match:
            label = self._clean_predicate_label(change_match.group("label"))
            if label:
                is_percentage = re.search(
                    r"%|\bpercent(?:age)?\b|\bpct\b|\bbps?\b|\bbasis points?\b",
                    text[start:end],
                    re.IGNORECASE,
                )
                suffix = "percentage change" if is_percentage else "change"
                return f"{label} {suffix}", 0.9

        comparison_match = re.search(r"\bas\s+against\s*$", before, re.IGNORECASE)
        if comparison_match:
            earlier = re.search(
                r"(?:the\s+)?(?P<label>[A-Za-z][A-Za-z &'’/-]{2,100}?)\s+"
                r"(?:was|stood\s+at|amounted\s+to)",
                sentence_before,
                re.IGNORECASE,
            )
            if earlier:
                label = self._clean_predicate_label(earlier.group("label"))
                return (label, 0.82) if label else (None, 0.0)

        if ":" in before:
            colon_label = before.rsplit(":", 1)[-1].strip()
            if 3 <= len(colon_label) <= 80 and re.search(r"[A-Za-z]", colon_label):
                label = self._clean_predicate_label(colon_label)
                if label:
                    return label, 0.72

        following_lines = [line.strip(" •y:-\t") for line in following.splitlines() if line.strip()]
        for line in following_lines[:2]:
            line = re.sub(r"^(?:fy\s*\d{2,4}(?:[-–/]\d{2,4})?|q[1-4])\s+", "", line, flags=re.IGNORECASE)
            label_match = re.match(r"([A-Za-z][A-Za-z &'’/-]{2,80})", line)
            if label_match:
                label = self._clean_predicate_label(label_match.group(1))
                if label:
                    return label, 0.78
        return None, 0.0

    @staticmethod
    def _clean_predicate_label(label: str) -> str:
        cleaned = re.sub(r"^[•y\s]+", "", label).strip(" :;,.-")
        cleaned = re.sub(
            r"^(?:(?:the|a|an|and|our)\s+)+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        if re.search(r"\bour\s+", cleaned, re.IGNORECASE):
            cleaned = re.split(r"\bour\s+", cleaned, flags=re.IGNORECASE)[-1]
        cleaned = re.sub(
            r"\s+on\s+(?:a\s+)?(?:standalone|consolidated|segment|group)\s+basis\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s+(?:for|during|in)\s+(?:q[1-4]\s+)?fy\s*\d{2,4}(?:[-–/]\d{2,4})?\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Financial tables mark lines with footnote references such as "Revenue from
        # customers(1)" or "Revenue for services (a+b)". The marker is a pointer to a
        # note, not part of the measure name, and leaving it in splits one measure into
        # several predicates that can never be compared with each other.
        cleaned = re.sub(
            r"\s*\(\s*(?:\d{1,2}|[a-z](?:\s*[+,&]\s*[a-z])*)\s*\)\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.rstrip("*†‡¹²³⁴⁵⁶⁷⁸⁹⁰ ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" :;,.-")
        if re.search(r"\b(?:and|by|for|from|in|of|or|to|with)\s*$", cleaned, re.IGNORECASE):
            return ""
        # A measure name does not open with a function word. Labels like "for FY",
        # "in fiscal" or "whereas the loss" are sentence fragments swept up from the
        # surrounding prose, and they are actively harmful: unrelated quantities collapse
        # onto one key, so revenue ends up being compared against loss.
        if re.match(
            r"(?:and|as|at|because|before|but|by|during|for|from|however|if|in|into|"
            r"of|on|or|since|so|than|that|though|to|under|until|when|where|whereas|"
            r"which|while|with|within|without)\b",
            cleaned,
            re.IGNORECASE,
        ):
            return ""
        # Filler words that a table or sentence can leave behind as a whole "label".
        if cleaned.casefold() in {
            "each",
            "nil",
            "respectively",
            "same",
            "thereof",
            "above",
            "below",
            "note",
            "notes",
            "particulars",
            "year",
            "years",
            "period",
            "quarter",
            "amount",
            "amounts",
            "value",
            "values",
        }:
            return ""
        return cleaned if 2 < len(cleaned) <= 120 and len(cleaned.split()) <= 10 else ""

    @staticmethod
    def _is_unsupported_number(raw: str, passage: Passage) -> bool:
        compact = raw.strip("() ")
        if re.fullmatch(r"\(\s*\d{1,2}\s*\)", raw):
            return True
        if re.fullmatch(r"\d{4}", compact) and 1900 <= int(compact) <= 2100:
            return True
        has_signal = bool(
            re.search(
                rf"{_CURRENCY}|{_SCALE}|{_UNIT}|%|\bpercent\b|\bpct\b|\bbps?\b|\bbasis points?\b|\d:\d|\d+x\b",
                raw,
                re.IGNORECASE,
            )
            or "," in raw
            or "." in raw
            or (raw.startswith("(") and raw.endswith(")"))
        )
        if not has_signal:
            return True
        return passage.role in {PassageRole.HEADER, PassageRole.FOOTER}

    def _evidence(
        self,
        passage: Passage,
        value_start: int,
        value_end: int,
        *,
        extractor: ExtractionMethod,
        bbox=None,
        quote_value_only: bool = False,
    ) -> Evidence:
        if quote_value_only:
            quote_start, quote_end = value_start, value_end
        else:
            quote_start, quote_end = self._sentence_span(passage.text, value_start, value_end)
        quote = passage.text[quote_start:quote_end]
        evidence = Evidence(
            id=f"evidence_{self._short_hash(passage.id, quote_start, quote_end, quote)}",
            document_id=passage.document_id,
            passage_id=passage.id,
            page_index=passage.page_index,
            quote=quote,
            quote_start=passage.char_start + quote_start,
            quote_end=passage.char_start + quote_end,
            bbox=bbox or passage.bbox,
            extractor=extractor,
            confidence=0.99 if quote_value_only else 0.97,
        )
        return evidence.verified_against(passage)

    @staticmethod
    def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
        quote_start = DeterministicExtractor._sentence_start(text, start)
        while quote_start < len(text) and text[quote_start].isspace():
            quote_start += 1
        right_candidates = [
            position
            for mark in (". ", "; ", "? ", "! ")
            if (position := text.find(mark, end)) != -1
        ]
        quote_end = min(right_candidates) + 1 if right_candidates else len(text)
        while quote_end > quote_start and text[quote_end - 1].isspace():
            quote_end -= 1
        return quote_start, quote_end

    @staticmethod
    def _sentence_start(text: str, position: int) -> int:
        boundaries = [
            text.rfind(mark, 0, position) for mark in (". ", "; ", "? ", "! ")
        ]
        boundary = max(boundaries)
        return boundary + 1 if boundary >= 0 else 0

    def _context(
        self,
        text: str,
        *,
        publisher: str | None,
        default: ContextEnvelope | None,
    ) -> tuple[ContextEnvelope, tuple[NormalizationWarning, ...], float]:
        local = normalize_context(text, publisher=publisher)
        if default is None:
            return local.value, local.warnings, local.confidence
        qualifiers = {**default.qualifiers, **local.value.qualifiers}
        context = ContextEnvelope(
            period=local.value.period or default.period,
            scope=local.value.scope or default.scope,
            basis=local.value.basis or default.basis,
            as_of=local.value.as_of or default.as_of,
            geography=local.value.geography or default.geography,
            publisher=local.value.publisher or default.publisher,
            qualifiers=qualifiers,
        )
        warnings = tuple(
            warning for warning in local.warnings if warning.code != "missing_context"
        )
        return context, warnings, max(local.confidence, 0.9)

    @staticmethod
    def _person_before(text: str, position: int) -> EntityReference | None:
        matches = list(_PERSON_NAME.finditer(text[:position]))
        if not matches:
            return None
        match = matches[-1]
        if position - match.end() > 120:
            return None
        return EntityReference(
            canonical_name=match.group(1).strip(),
            entity_type="person",
        )

    def _person_for_record(
        self,
        text: str,
        record_start: int,
        din_start: int,
    ) -> EntityReference | None:
        """Find whose DIN this is, without straying into a neighbouring record.

        A board table lists each director as a run of fields that ends with their DIN,
        so the DIN before this one marks where this record began. Searching the whole
        page instead would attach every DIN in a four-director passage to whichever name
        happened to appear first, giving one person several DINs and leaving entity
        resolution no choice but to refuse the merge.

        Inside the record a titled name wins, because prose like "Mr. Sandeep Kumar
        Barasia (DIN: ...)" is unambiguous. Otherwise the record opens with the bare
        name, which is how these tables are laid out. When the record began on the
        previous page there is no name to find, and returning None is the honest answer.
        """
        segment = text[record_start:din_start]
        if not segment.strip():
            return None
        titled = list(_PERSON_NAME.finditer(segment))
        if titled:
            return EntityReference(
                canonical_name=titled[-1].group(1).strip(),
                entity_type="person",
            )
        # Structure beats vocabulary here. A record is a name followed by "Label: value"
        # lines, so the name is whatever sits directly above the first labelled field.
        # Company names bleeding in from the neighbouring directorships column are not
        # followed by a label, which is what rules them out - and no blocklist of words
        # like "Properties" or "Council" has to be maintained to do it.
        lines = [line.strip() for line in segment.splitlines()]
        lines = [line for line in lines if line]
        for current, following in zip(lines, lines[1:]):
            if ":" in current or ":" not in following:
                continue
            if not _BARE_PERSON_NAME.fullmatch(current):
                continue
            if _ORGANISATION_TAIL.search(current):
                continue
            if any(word.casefold() in _TABLE_VOCABULARY for word in current.split()):
                continue
            return EntityReference(canonical_name=current, entity_type="person")
        return None
        titled = list(_PERSON_NAME.finditer(segment))
        if titled:
            return EntityReference(
                canonical_name=titled[-1].group(1).strip(),
                entity_type="person",
            )
        for line in segment.splitlines():
            candidate = line.strip()
            if not candidate or ":" in candidate:
                continue
            if not _BARE_PERSON_NAME.fullmatch(candidate):
                continue
            # "Dr. Reddy's Laboratories Limited" reads like a name until you notice the
            # suffix; those come from the neighbouring directorships column.
            if _ORGANISATION_TAIL.search(candidate):
                continue
            if any(word.casefold() in _TABLE_VOCABULARY for word in candidate.split()):
                continue
            return EntityReference(canonical_name=candidate, entity_type="person")
        return None

    @staticmethod
    def _document_subject(document_id: str, filename: str) -> EntityReference:
        readable_name = re.sub(r"[-_]", " ", filename.rsplit(".", 1)[0]).strip()
        return EntityReference(
            id=f"document:{document_id}",
            canonical_name=readable_name or filename,
            entity_type="document",
        )

    @staticmethod
    def _table_passage_ids(page: IngestedPage) -> set[str]:
        return {
            passage.id
            for passage in page.passages
            for table in page.tables
            if passage.bbox and _overlap_ratio(passage.bbox, table.bbox) >= 0.5
        }

    @staticmethod
    def _passage_for_table_value(
        passages: tuple[Passage, ...], raw: str, cell: TableCell | None
    ) -> Passage | None:
        candidates = [passage for passage in passages if raw in passage.text]
        if cell and cell.bbox:
            candidates.sort(
                key=lambda passage: _overlap_ratio(passage.bbox, cell.bbox) if passage.bbox else 0,
                reverse=True,
            )
        return candidates[0] if candidates else None

    @staticmethod
    def _nearest_passage(
        passages: tuple[Passage, ...], cell: TableCell | None
    ) -> Passage | None:
        if not passages:
            return None
        if cell is None or cell.bbox is None:
            return passages[0]
        return max(
            passages,
            key=lambda passage: _overlap_ratio(passage.bbox, cell.bbox) if passage.bbox else 0,
        )

    @staticmethod
    def _failure(
        passage: Passage,
        stage: FailureStage,
        reason: str,
        rejected_output: dict,
    ) -> ExtractionFailure:
        return ExtractionFailure(
            id=f"failure_{DeterministicExtractor._short_hash(passage.id, stage.value, reason, rejected_output)}",
            document_id=passage.document_id,
            passage_id=passage.id,
            page_index=passage.page_index,
            stage=stage,
            reason=reason,
            rejected_output=rejected_output,
            recoverable=True,
        )

    @staticmethod
    def _fact_id(document_id: str, page_index: int, predicate: str, raw: str, start: int) -> str:
        return f"fact_{DeterministicExtractor._short_hash(document_id, page_index, predicate, raw, start)}"

    @staticmethod
    def _short_hash(*parts) -> str:
        payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    @staticmethod
    def _deduplicate(outcome: ExtractionOutcome) -> ExtractionOutcome:
        facts = {fact.id: fact for fact in outcome.facts}
        failures = {failure.id: failure for failure in outcome.failures}
        removed = len(outcome.facts) - len(facts)
        stats = ExtractionStats(
            candidates_seen=outcome.stats.candidates_seen,
            facts_accepted=len(facts),
            candidates_ignored=outcome.stats.candidates_ignored + removed,
            candidates_rejected=len(failures),
        )
        return ExtractionOutcome(tuple(facts.values()), tuple(failures.values()), stats)


def _overlap_ratio(left, right) -> float:
    if left is None or right is None:
        return 0.0
    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    intersection = width * height
    left_area = (left.x1 - left.x0) * (left.y1 - left.y0)
    return intersection / left_area if left_area else 0.0


__all__ = [
    "DeterministicExtractor",
    "ExtractionOutcome",
    "ExtractionStats",
]
