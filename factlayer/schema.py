"""Domain models shared by extraction, storage, linking, and the API.

The models are deliberately strict at the edges. A typo should fail close to
where it was introduced, and a stored fact should always carry enough evidence
for another person to verify it.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _unique_text(values: list[str]) -> list[str]:
    """Remove duplicate aliases without changing their useful display form."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            unique.append(cleaned)
            seen.add(key)
    return unique


class DomainModel(BaseModel):
    """Consistent validation rules for every public domain object."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class DocumentStatus(str, Enum):
    REGISTERED = "registered"
    PROCESSING = "processing"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ReviewState(str, Enum):
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class ExtractionMethod(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    TABLE = "table"
    DERIVED = "derived"
    MANUAL = "manual"


class PassageRole(str, Enum):
    BODY = "body"
    HEADER = "header"
    FOOTER = "footer"
    TABLE = "table"


class RelationType(str, Enum):
    CORROBORATES = "corroborates"
    CONTRADICTS = "contradicts"
    LIKELY_CONTRADICTION = "likely_contradiction"
    RECONCILED_BY_PERIOD = "reconciled_by_period"
    RECONCILED_BY_SCOPE = "reconciled_by_scope"
    RECONCILED_BY_BASIS = "reconciled_by_basis"
    RECONCILED_BY_AS_OF = "reconciled_by_as_of"
    RECONCILED_BY_UNIT = "reconciled_by_unit"
    INCOMPARABLE = "incomparable"
    NEEDS_REVIEW = "needs_review"


class ConsistencyCheckType(str, Enum):
    GROWTH = "growth"
    COMPONENT_TOTAL = "component_total"
    REPEATED_METRIC = "repeated_metric"


class FailureStage(str, Enum):
    DOCUMENT = "document"
    PAGE = "page"
    INGESTION = "ingestion"
    EXTRACTION = "extraction"
    GROUNDING = "grounding"
    NORMALIZATION = "normalization"
    ENTITY_RESOLUTION = "entity_resolution"
    LINKING = "linking"


class BoundingBox(DomainModel):
    """A rectangular source location in PDF coordinate space."""

    x0: Annotated[float, Field(ge=0)]
    y0: Annotated[float, Field(ge=0)]
    x1: Annotated[float, Field(gt=0)]
    y1: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def corners_are_in_reading_order(self) -> "BoundingBox":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("A bounding box must have a positive width and height")
        return self


class Document(DomainModel):
    id: Identifier | None = None
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_filename: ShortText
    page_count: Annotated[int, Field(ge=0)] = 0
    status: DocumentStatus = DocumentStatus.REGISTERED
    extraction_mode: Literal["offline", "hybrid"] = "offline"
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now_utc)
    updated_at: datetime = Field(default_factory=_now_utc)

    @field_validator("warnings")
    @classmethod
    def clean_warnings(cls, values: list[str]) -> list[str]:
        return _unique_text(values)


class Passage(DomainModel):
    id: Identifier | None = None
    document_id: Identifier
    page_index: Annotated[int, Field(ge=0)]
    reading_order: Annotated[int, Field(ge=0)]
    role: PassageRole = PassageRole.BODY
    text: Annotated[str, StringConstraints(min_length=1)]
    char_start: Annotated[int, Field(ge=0)] = 0
    char_end: Annotated[int, Field(gt=0)] | None = None
    bbox: BoundingBox | None = None
    text_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def finish_character_range(self) -> "Passage":
        if self.char_end is None:
            self.char_end = self.char_start + len(self.text)
        if self.char_end <= self.char_start:
            raise ValueError("Passage char_end must be greater than char_start")
        return self


class Entity(DomainModel):
    id: Identifier | None = None
    canonical_name: ShortText
    entity_type: ShortText
    identifiers: dict[str, str] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    confidence: Confidence = 1.0
    review_state: ReviewState = ReviewState.READY

    @field_validator("aliases")
    @classmethod
    def clean_aliases(cls, values: list[str]) -> list[str]:
        return _unique_text(values)

    @field_validator("identifiers")
    @classmethod
    def identifiers_have_names_and_values(cls, values: dict[str, str]) -> dict[str, str]:
        cleaned = {scheme.strip().lower(): value.strip() for scheme, value in values.items()}
        if any(not scheme or not value for scheme, value in cleaned.items()):
            raise ValueError("Entity identifiers need both a scheme and a value")
        return cleaned


class Predicate(DomainModel):
    """A measure discovered from documents, not a fixed application enum."""

    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: ShortText
    value_kind: Literal["number", "text", "category", "date", "boolean", "identifier"]
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None

    @field_validator("aliases")
    @classmethod
    def clean_aliases(cls, values: list[str]) -> list[str]:
        return _unique_text(values)


class EntityReference(DomainModel):
    id: Identifier | None = None
    canonical_name: ShortText
    entity_type: ShortText | None = None


class PredicateReference(DomainModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: ShortText


class NumericValue(DomainModel):
    kind: Literal["number"] = "number"
    raw: ShortText
    number: Decimal
    reported_number: Decimal | None = None
    unit: str | None = None
    reported_unit: str | None = None
    currency: str | None = None
    scale: Annotated[Decimal, Field(gt=0)] = Decimal("1")
    approximate: bool = False

    @model_validator(mode="after")
    def retain_reported_number(self) -> "NumericValue":
        if self.reported_number is None:
            self.reported_number = self.number
        return self


class TextValue(DomainModel):
    kind: Literal["text"] = "text"
    raw: ShortText
    text: ShortText


class CategoricalValue(DomainModel):
    kind: Literal["category"] = "category"
    raw: ShortText
    state: ShortText


class DateValue(DomainModel):
    kind: Literal["date"] = "date"
    raw: ShortText
    value: date
    precision: Literal["day", "month", "quarter", "year"] = "day"


class BooleanValue(DomainModel):
    kind: Literal["boolean"] = "boolean"
    raw: ShortText
    value: bool


class IdentifierValue(DomainModel):
    kind: Literal["identifier"] = "identifier"
    raw: ShortText
    value: Identifier
    scheme: ShortText


FactValue = Annotated[
    NumericValue | TextValue | CategoricalValue | DateValue | BooleanValue | IdentifierValue,
    Field(discriminator="kind"),
]


class ReportingPeriod(DomainModel):
    label: ShortText
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def dates_are_chronological(self) -> "ReportingPeriod":
        if self.start and self.end and self.end < self.start:
            raise ValueError("A reporting period cannot end before it starts")
        return self


class ContextEnvelope(DomainModel):
    period: ReportingPeriod | None = None
    scope: str | None = None
    basis: str | None = None
    as_of: date | None = None
    geography: str | None = None
    publisher: str | None = None
    qualifiers: dict[str, str] = Field(default_factory=dict)


class Evidence(DomainModel):
    id: Identifier | None = None
    document_id: Identifier
    passage_id: Identifier
    page_index: Annotated[int, Field(ge=0)]
    quote: Annotated[str, StringConstraints(min_length=1)]
    quote_start: Annotated[int, Field(ge=0)]
    quote_end: Annotated[int, Field(gt=0)]
    bbox: BoundingBox | None = None
    extractor: ExtractionMethod
    confidence: Confidence = 1.0
    verified: bool = False

    @model_validator(mode="after")
    def quote_range_is_valid(self) -> "Evidence":
        if self.quote_end <= self.quote_start:
            raise ValueError("Evidence quote_end must be greater than quote_start")
        if self.quote_end - self.quote_start != len(self.quote):
            raise ValueError("Evidence quote offsets must span the exact quote")
        return self

    def verified_against(self, passage: Passage) -> "Evidence":
        """Return verified evidence when the quote resolves exactly in its passage."""
        if passage.id != self.passage_id or passage.document_id != self.document_id:
            raise ValueError("Evidence points to a different passage or document")
        if passage.page_index != self.page_index:
            raise ValueError("Evidence page does not match its passage")
        local_start = self.quote_start - passage.char_start
        local_end = self.quote_end - passage.char_start
        if local_start < 0 or passage.text[local_start:local_end] != self.quote:
            raise ValueError("Evidence quote was not found at the stated passage offsets")
        return self.model_copy(update={"verified": True})


class Fact(DomainModel):
    id: Identifier | None = None
    subject: EntityReference
    predicate: PredicateReference
    value: FactValue
    context: ContextEnvelope = Field(default_factory=ContextEnvelope)
    evidence: Annotated[list[Evidence], Field(min_length=1)]
    extraction_confidence: Confidence
    normalization_confidence: Confidence
    review_state: ReviewState = ReviewState.READY
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now_utc)

    @field_validator("evidence")
    @classmethod
    def evidence_is_verified(cls, values: list[Evidence]) -> list[Evidence]:
        if any(not evidence.verified for evidence in values):
            raise ValueError("Every fact needs verified source evidence")
        return values

    @field_validator("warnings")
    @classmethod
    def clean_warnings(cls, values: list[str]) -> list[str]:
        return _unique_text(values)


class ValueComparison(DomainModel):
    left: str
    right: str
    agrees: bool | None = None
    difference: Decimal | None = None
    tolerance: Decimal | None = None


class ContextDifference(DomainModel):
    field: ShortText
    left: Any = None
    right: Any = None
    explains_difference: bool = False


class Relation(DomainModel):
    id: Identifier | None = None
    fact_a_id: Identifier
    fact_b_id: Identifier
    relation_type: RelationType
    value_comparison: ValueComparison
    context_diff: list[ContextDifference] = Field(default_factory=list)
    explanation: Annotated[str, StringConstraints(min_length=1)]
    confidence: Confidence
    rule_version: ShortText
    review_state: ReviewState = ReviewState.READY
    created_at: datetime = Field(default_factory=_now_utc)

    @model_validator(mode="after")
    def facts_are_distinct(self) -> "Relation":
        if self.fact_a_id == self.fact_b_id:
            raise ValueError("A fact cannot have a relationship with itself")
        return self


class ConsistencyFinding(DomainModel):
    """An auditable calculation over two or more grounded facts."""

    id: Identifier | None = None
    check_type: ConsistencyCheckType
    fact_ids: Annotated[list[Identifier], Field(min_length=2)]
    relation_type: RelationType
    formula: ShortText
    operands: dict[str, Any]
    stated_result: Decimal | None = None
    calculated_result: Decimal | None = None
    difference: Decimal | None = None
    tolerance: Annotated[Decimal | None, Field(ge=0)] = None
    explanation: Annotated[str, StringConstraints(min_length=1)]
    confidence: Confidence
    rule_version: ShortText
    review_state: ReviewState = ReviewState.READY
    created_at: datetime = Field(default_factory=_now_utc)

    @field_validator("fact_ids")
    @classmethod
    def facts_are_unique(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("A consistency finding cannot repeat the same fact")
        return values


class ExtractionFailure(DomainModel):
    id: Identifier | None = None
    document_id: Identifier
    passage_id: Identifier | None = None
    page_index: Annotated[int, Field(ge=0)] | None = None
    stage: FailureStage
    reason: ShortText
    rejected_output: dict[str, Any] | str | None = None
    recoverable: bool = True
    created_at: datetime = Field(default_factory=_now_utc)


__all__ = [
    "BooleanValue",
    "BoundingBox",
    "CategoricalValue",
    "ContextDifference",
    "ContextEnvelope",
    "ConsistencyCheckType",
    "ConsistencyFinding",
    "DateValue",
    "Document",
    "DocumentStatus",
    "Entity",
    "EntityReference",
    "Evidence",
    "ExtractionFailure",
    "ExtractionMethod",
    "Fact",
    "FactValue",
    "FailureStage",
    "IdentifierValue",
    "NumericValue",
    "Passage",
    "PassageRole",
    "Predicate",
    "PredicateReference",
    "Relation",
    "RelationType",
    "ReportingPeriod",
    "ReviewState",
    "TextValue",
    "ValueComparison",
]
