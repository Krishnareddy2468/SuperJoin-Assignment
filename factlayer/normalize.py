"""Normalize extracted values and context without guessing missing meaning.

Extraction answers “what did the document say?” This module answers “how can
that statement be represented consistently?” Every result retains the source
text and carries warnings when normalization is incomplete or ambiguous.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Generic, Literal, TypeVar

from factlayer.schema import ContextEnvelope, DateValue, NumericValue, Predicate, ReportingPeriod


T = TypeVar("T")


@dataclass(frozen=True)
class NormalizationWarning:
    code: str
    message: str


@dataclass(frozen=True)
class NormalizationResult(Generic[T]):
    value: T | None
    confidence: float
    warnings: tuple[NormalizationWarning, ...] = field(default_factory=tuple)

    @property
    def needs_review(self) -> bool:
        return self.value is None or bool(self.warnings)


@dataclass(frozen=True)
class NumericRange:
    raw: str
    lower: Decimal
    upper: Decimal
    scale: Decimal = Decimal("1")
    unit: str | None = None
    currency: str | None = None
    approximate: bool = False


@dataclass(frozen=True)
class NumericComparison:
    comparable: bool
    agrees: bool | None
    difference: Decimal | None
    allowed_difference: Decimal | None
    relative_difference: Decimal | None
    reason: str


class PeriodKind(str, Enum):
    POINT_IN_TIME = "point_in_time"
    FISCAL_YEAR = "fiscal_year"
    CALENDAR_YEAR = "calendar_year"
    QUARTER = "quarter"
    DATE_RANGE = "date_range"
    MONTH = "month"


@dataclass(frozen=True)
class NormalizedPeriod:
    kind: PeriodKind
    period: ReportingPeriod | None = None
    as_of: date | None = None


_SCALES = {
    "thousand": Decimal("1000"),
    "thousands": Decimal("1000"),
    "k": Decimal("1000"),
    "lakh": Decimal("100000"),
    "lakhs": Decimal("100000"),
    "lac": Decimal("100000"),
    "lacs": Decimal("100000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
    "mn": Decimal("1000000"),
    "crore": Decimal("10000000"),
    "crores": Decimal("10000000"),
    "cr": Decimal("10000000"),
    "billion": Decimal("1000000000"),
    "billions": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
}

_CURRENCIES = {
    "₹": ("INR", "rupee"),
    "inr": ("INR", "rupee"),
    "rs": ("INR", "rupee"),
    "rs.": ("INR", "rupee"),
    "$": ("USD", "US dollar"),
    "usd": ("USD", "US dollar"),
    "us$": ("USD", "US dollar"),
    "€": ("EUR", "euro"),
    "eur": ("EUR", "euro"),
    "£": ("GBP", "pound sterling"),
    "gbp": ("GBP", "pound sterling"),
}

_UNIT_ALIASES = {
    "km": "kilometre",
    "kilometer": "kilometre",
    "kilometers": "kilometre",
    "kilometre": "kilometre",
    "kilometres": "kilometre",
    "kg": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "ton": "tonne",
    "tons": "tonne",
    "tonne": "tonne",
    "tonnes": "tonne",
    "metric ton": "tonne",
    "metric tons": "tonne",
    "metric tonne": "tonne",
    "metric tonnes": "tonne",
    "shipment": "shipment",
    "shipments": "shipment",
    "employee": "employee",
    "employees": "employee",
    "day": "day",
    "days": "day",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
}

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?"
_SCALE_PATTERN = "|".join(sorted((re.escape(key) for key in _SCALES), key=len, reverse=True))
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))


def normalize_number(raw: str) -> NormalizationResult[NumericValue]:
    """Normalize one reported number, including scale, unit, and currency."""
    source = raw.strip()
    if not source:
        return _failure("missing_number", "No numeric text was supplied")

    ratio = re.fullmatch(rf"\s*({_NUMBER_PATTERN})\s*:\s*({_NUMBER_PATTERN})\s*", source)
    if ratio:
        left = _decimal(ratio.group(1))
        right = _decimal(ratio.group(2))
        if right == 0:
            return _failure("invalid_ratio", "A ratio cannot have zero as its second value")
        return NormalizationResult(
            NumericValue(raw=source, number=left / right, reported_number=left / right, unit="ratio"),
            confidence=0.99,
        )

    matches = list(re.finditer(_NUMBER_PATTERN, source))
    if not matches:
        return _failure("missing_number", f"No number was found in {source!r}")
    if len(matches) > 1:
        return _failure(
            "ambiguous_number",
            "More than one number was found; extract a smaller value span or parse it as a range",
        )

    match = matches[0]
    try:
        reported_number = _decimal(match.group())
    except InvalidOperation:
        return _failure("invalid_number", f"{match.group()!r} is not a valid decimal number")

    lowered = source.casefold()
    accounting_negative = source.startswith("(") and source.endswith(")")
    if accounting_negative and reported_number > 0:
        reported_number = -reported_number

    currency, currency_unit = _find_currency(source)
    scale_label, scale = _find_scale(lowered)
    unit, reported_unit = _find_unit(lowered, match.end(), scale_label)
    approximate = bool(re.search(r"(?:\babout\b|\bapprox(?:imately)?\.?\b|~|≈)", lowered))

    if re.search(r"%|\bpercent(?:age)?\b|\bpct\b", lowered):
        unit = "percent"
        reported_unit = "%"
    elif re.search(r"\bbps?\b|\bbasis points?\b", lowered):
        unit = "percentage_point"
        reported_unit = "basis point"
        scale = Decimal("0.01")
        scale_label = "basis point"
    elif re.search(r"(?:\d|\s)x\s*$", lowered):
        unit = "ratio"
        reported_unit = "x"

    if currency:
        unit = currency_unit
        if reported_unit is None:
            reported_unit = scale_label

    warnings: list[NormalizationWarning] = []
    confidence = 0.99
    suffix = lowered[match.end() :].strip(" .,)\n\t")
    known_suffix = bool(
        scale_label
        or unit
        or re.search(r"%|\bpercent(?:age)?\b|\bpct\b|\bbps?\b|\bbasis points?\b|x\s*$", lowered)
    )
    if suffix and not known_suffix:
        warnings.append(
            NormalizationWarning(
                "unknown_unit",
                f"The unit text {suffix!r} is not recognized; the raw value was preserved",
            )
        )
        reported_unit = suffix
        confidence = 0.72

    normalized = reported_number * scale
    return NormalizationResult(
        NumericValue(
            raw=source,
            number=normalized,
            reported_number=reported_number,
            unit=unit,
            reported_unit=reported_unit,
            currency=currency,
            scale=scale,
            approximate=approximate,
        ),
        confidence=confidence,
        warnings=tuple(warnings),
    )


def normalize_numeric_range(raw: str) -> NormalizationResult[NumericRange]:
    """Normalize ranges such as ``₹10–12 crore`` or ``5% to 7%``."""
    source = raw.strip()
    pattern = rf"({_NUMBER_PATTERN})\s*(?:-|–|—|to)\s*({_NUMBER_PATTERN})"
    match = re.search(pattern, source, flags=re.IGNORECASE)
    if not match:
        return _failure("missing_range", "No two-ended numeric range was found")
    lower = _decimal(match.group(1))
    upper = _decimal(match.group(2))
    if upper < lower:
        return _failure("reversed_range", "A numeric range cannot end below its starting value")

    lowered = source.casefold()
    currency, currency_unit = _find_currency(source)
    _, scale = _find_scale(lowered)
    unit, _ = _find_unit(lowered, match.end(), None)
    if re.search(r"%|\bpercent(?:age)?\b|\bpct\b", lowered):
        unit = "percent"
    elif currency:
        unit = currency_unit
    approximate = bool(re.search(r"(?:\babout\b|\bbetween\b|~|≈)", lowered))
    return NormalizationResult(
        NumericRange(
            raw=source,
            lower=lower * scale,
            upper=upper * scale,
            scale=scale,
            unit=unit,
            currency=currency,
            approximate=approximate,
        ),
        confidence=0.97,
    )


def normalize_date(raw: str) -> NormalizationResult[DateValue]:
    """Normalize a standalone date while retaining its reported precision."""
    source = raw.strip()
    lowered = source.casefold()
    if not source:
        return _failure("missing_date", "No date text was supplied")
    try:
        iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", lowered)
        if iso:
            value = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            return NormalizationResult(DateValue(raw=source, value=value), confidence=0.99)

        month_first = re.fullmatch(
            rf"({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})",
            lowered,
        )
        if month_first:
            return NormalizationResult(
                DateValue(raw=source, value=_date_from_match(month_first)), confidence=0.99
            )

        day_first = re.fullmatch(
            rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN}),?\s+(\d{{4}})",
            lowered,
        )
        if day_first:
            value = date(
                int(day_first.group(3)),
                _MONTHS[day_first.group(2)],
                int(day_first.group(1)),
            )
            return NormalizationResult(DateValue(raw=source, value=value), confidence=0.99)

        month_only = re.fullmatch(rf"({_MONTH_PATTERN})\s+(\d{{4}})", lowered)
        if month_only:
            value = date(int(month_only.group(2)), _MONTHS[month_only.group(1)], 1)
            return NormalizationResult(
                DateValue(raw=source, value=value, precision="month"), confidence=0.92
            )

        year_only = re.fullmatch(r"(\d{4})", lowered)
        if year_only:
            value = date(int(year_only.group(1)), 1, 1)
            return NormalizationResult(
                DateValue(raw=source, value=value, precision="year"),
                confidence=0.78,
                warnings=(
                    NormalizationWarning(
                        "date_precision_year",
                        "Only a year was reported, so month and day remain unknown",
                    ),
                ),
            )
    except ValueError:
        return _failure("invalid_date", f"{source!r} is not a valid calendar date")
    return _failure("unknown_date", f"The date in {source!r} could not be normalized safely")


def normalize_period(
    text: str, *, fiscal_year_start_month: int = 4
) -> NormalizationResult[NormalizedPeriod]:
    """Normalize common reporting periods while distinguishing dates from durations."""
    if not 1 <= fiscal_year_start_month <= 12:
        raise ValueError("fiscal_year_start_month must be between 1 and 12")
    try:
        return _normalize_period(text, fiscal_year_start_month=fiscal_year_start_month)
    except (ValueError, OverflowError):
        return _failure(
            "invalid_period",
            f"The date or reporting period in {text.strip()!r} is not valid",
        )


def _normalize_period(
    text: str, *, fiscal_year_start_month: int
) -> NormalizationResult[NormalizedPeriod]:
    source = text.strip()
    lowered = source.casefold()
    if not source:
        return _failure("missing_period", "No period text was supplied")

    as_of_match = re.search(rf"\bas\s+(?:of|at)\s+({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})", lowered)
    if as_of_match:
        as_of = _date_from_match(as_of_match)
        return NormalizationResult(
            NormalizedPeriod(kind=PeriodKind.POINT_IN_TIME, as_of=as_of),
            confidence=0.99,
        )

    iso_as_of = re.search(r"\bas\s+(?:of|at)\s+(\d{4})-(\d{2})-(\d{2})", lowered)
    if iso_as_of:
        as_of = date(int(iso_as_of.group(1)), int(iso_as_of.group(2)), int(iso_as_of.group(3)))
        return NormalizationResult(
            NormalizedPeriod(kind=PeriodKind.POINT_IN_TIME, as_of=as_of),
            confidence=0.99,
        )

    day_first_as_of = re.search(
        rf"\bas\s+(?:of|at)\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN}),?\s+(\d{{4}})",
        lowered,
    )
    if day_first_as_of:
        as_of = date(
            int(day_first_as_of.group(3)),
            _MONTHS[day_first_as_of.group(2)],
            int(day_first_as_of.group(1)),
        )
        return NormalizationResult(
            NormalizedPeriod(kind=PeriodKind.POINT_IN_TIME, as_of=as_of),
            confidence=0.99,
        )

    fy_match = re.search(r"\b(?:fy|fiscal year)\s*['’]?(\d{2,4})(?:\s*[-–/]\s*(\d{2,4}))?\b", lowered)
    if fy_match:
        first = _four_digit_year(fy_match.group(1))
        second_text = fy_match.group(2)
        end_year = _four_digit_year(second_text) if second_text else first
        if second_text and len(fy_match.group(1)) == 4:
            start_year = first
        else:
            start_year = end_year - 1
        start = date(start_year, fiscal_year_start_month, 1)
        end = _day_before_year_start(end_year, fiscal_year_start_month)
        label = f"FY {start_year}-{str(end_year)[-2:]}"

        quarter_match = re.search(r"\bq([1-4])\b", lowered)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            quarter_start_month = ((fiscal_year_start_month - 1 + (quarter - 1) * 3) % 12) + 1
            quarter_start_year = start_year if quarter_start_month >= fiscal_year_start_month else end_year
            quarter_start = date(quarter_start_year, quarter_start_month, 1)
            quarter_end = _add_months(quarter_start, 3)
            quarter_end = date.fromordinal(quarter_end.toordinal() - 1)
            return NormalizationResult(
                NormalizedPeriod(
                    kind=PeriodKind.QUARTER,
                    period=ReportingPeriod(
                        label=f"Q{quarter} {label}", start=quarter_start, end=quarter_end
                    ),
                ),
                confidence=0.99,
            )
        return NormalizationResult(
            NormalizedPeriod(
                kind=PeriodKind.FISCAL_YEAR,
                period=ReportingPeriod(label=label, start=start, end=end),
            ),
            confidence=0.99,
        )

    year_ended = re.search(
        rf"\byear ended\s+({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})", lowered
    )
    if year_ended:
        end = _date_from_match(year_ended)
        start = _year_period_start(end)
        return NormalizationResult(
            NormalizedPeriod(
                kind=PeriodKind.DATE_RANGE,
                period=ReportingPeriod(label=source, start=start, end=end),
            ),
            confidence=0.98,
        )

    range_match = re.search(
        rf"(?:from\s+)?({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})\s+(?:to|[-–—])\s+"
        rf"({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})",
        lowered,
    )
    if range_match:
        start = date(int(range_match.group(3)), _MONTHS[range_match.group(1)], int(range_match.group(2)))
        end = date(int(range_match.group(6)), _MONTHS[range_match.group(4)], int(range_match.group(5)))
        if end < start:
            return _failure("reversed_period", "The reporting period ends before it starts")
        return NormalizationResult(
            NormalizedPeriod(
                kind=PeriodKind.DATE_RANGE,
                period=ReportingPeriod(label=source, start=start, end=end),
            ),
            confidence=0.99,
        )

    month_match = re.search(rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b", lowered)
    if month_match and "ended" in lowered:
        year = int(month_match.group(2))
        month = _MONTHS[month_match.group(1)]
        start = date(year, month, 1)
        end = date.fromordinal(_add_months(start, 1).toordinal() - 1)
        return NormalizationResult(
            NormalizedPeriod(
                kind=PeriodKind.MONTH,
                period=ReportingPeriod(label=source, start=start, end=end),
            ),
            confidence=0.96,
        )

    calendar_quarter = re.search(r"\bq([1-4])\s+(\d{4})\b", lowered)
    if calendar_quarter:
        quarter = int(calendar_quarter.group(1))
        year = int(calendar_quarter.group(2))
        start = date(year, 1 + (quarter - 1) * 3, 1)
        end = date.fromordinal(_add_months(start, 3).toordinal() - 1)
        return NormalizationResult(
            NormalizedPeriod(
                kind=PeriodKind.QUARTER,
                period=ReportingPeriod(label=f"Q{quarter} {year}", start=start, end=end),
            ),
            confidence=0.98,
        )

    calendar_year = re.fullmatch(r"(?:calendar year\s+)?(\d{4})", lowered)
    if calendar_year:
        year = int(calendar_year.group(1))
        warning = NormalizationWarning(
            "year_assumed_calendar",
            "A bare year was interpreted as a calendar year because no fiscal-year cue was present",
        )
        return NormalizationResult(
            NormalizedPeriod(
                kind=PeriodKind.CALENDAR_YEAR,
                period=ReportingPeriod(
                    label=str(year), start=date(year, 1, 1), end=date(year, 12, 31)
                ),
            ),
            confidence=0.78,
            warnings=(warning,),
        )

    return _failure(
        "unknown_period",
        f"The reporting period in {source!r} could not be normalized safely",
    )


def normalize_context(text: str, *, publisher: str | None = None) -> NormalizationResult[ContextEnvelope]:
    """Collect period, scope, basis, and geography cues from nearby source text."""
    source = text.strip()
    lowered = source.casefold()
    warnings: list[NormalizationWarning] = []
    confidence = 0.95

    period_result = normalize_period(source)
    period = period_result.value.period if period_result.value else None
    as_of = period_result.value.as_of if period_result.value else None
    if period_result.value:
        warnings.extend(period_result.warnings)
    elif re.search(r"\b(?:fy|fiscal|quarter|period|year ended|as of|as at)\b", lowered):
        warnings.extend(period_result.warnings)
        confidence = min(confidence, 0.65)

    scopes = _matched_labels(
        lowered,
        {
            "standalone": r"\bstand[- ]?alone\b",
            "consolidated": r"\bconsolidated\b|\bgroup\s+(?:level|accounts?)\b",
            "segment": r"\bsegment(?:al)?\b",
            "company": r"\bcompany[- ]wide\b",
        },
    )
    scope = scopes[0] if len(scopes) == 1 else None
    if len(scopes) > 1:
        warnings.append(
            NormalizationWarning(
                "ambiguous_scope",
                f"Conflicting scope cues were found: {', '.join(scopes)}",
            )
        )
        confidence = min(confidence, 0.65)

    basis_groups = {
        "assurance": _matched_labels(lowered, {"audited": r"\baudited\b", "unaudited": r"\bunaudited\b"}),
        "value_status": _matched_labels(
            lowered,
            {
                "actual": r"\bactuals?\b|\brealised\b|\brealized\b",
                "forecast": r"\bforecast(?:ed)?\b|\bprojection\b|\bprojected\b",
                "estimate": r"\bestimate(?:d)?\b",
                "provisional": r"\bprovisional\b",
            },
        ),
        "price_basis": _matched_labels(lowered, {"nominal": r"\bnominal\b", "real": r"\breal\b"}),
        "adjustment": _matched_labels(
            lowered,
            {
                "adjusted": r"\badjusted\b|\bnon-gaap\b",
                "reported": r"\breported\b|\bgaap\b",
            },
        ),
    }
    basis_parts: list[str] = []
    qualifiers: dict[str, str] = {}
    for group, labels in basis_groups.items():
        if len(labels) == 1:
            qualifiers[group] = labels[0]
            basis_parts.append(labels[0])
        elif len(labels) > 1:
            warnings.append(
                NormalizationWarning(
                    f"ambiguous_{group}",
                    f"Conflicting {group.replace('_', ' ')} cues were found: {', '.join(labels)}",
                )
            )
            confidence = min(confidence, 0.65)

    geography_labels = _matched_labels(
        lowered,
        {
            "India": r"\bindia(?:n)?\b|\bdomestic\b",
            "global": r"\bglobal\b|\bworldwide\b",
        },
    )
    geography = geography_labels[0] if len(geography_labels) == 1 else None
    if len(geography_labels) > 1:
        warnings.append(
            NormalizationWarning(
                "ambiguous_geography",
                f"Conflicting geography cues were found: {', '.join(geography_labels)}",
            )
        )

    if period_result.value:
        qualifiers["period_kind"] = period_result.value.kind.value
    context = ContextEnvelope(
        period=period,
        scope=scope,
        basis="; ".join(basis_parts) or None,
        as_of=as_of,
        geography=geography,
        publisher=publisher,
        qualifiers=qualifiers,
    )
    if not any((period, as_of, scope, basis_parts, geography, publisher)):
        warnings.append(
            NormalizationWarning(
                "missing_context",
                "No reliable period, scope, basis, geography, or publisher context was found",
            )
        )
        confidence = 0.45
    return NormalizationResult(context, confidence=confidence, warnings=tuple(warnings))


class PredicateRegistry:
    """Grow canonical predicate keys as unfamiliar measures appear."""

    def __init__(self) -> None:
        self._predicates: dict[str, Predicate] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        label: str,
        *,
        value_kind: Literal["number", "text", "category", "date", "boolean", "identifier"],
        aliases: tuple[str, ...] | list[str] = (),
        key: str | None = None,
        description: str | None = None,
    ) -> Predicate:
        canonical_key = key or canonical_predicate_key(label)
        known = self._predicates.get(canonical_key)
        if known and known.value_kind != value_kind:
            raise ValueError(
                f"Predicate {canonical_key!r} is already registered as {known.value_kind!r}"
            )
        all_aliases = _unique_aliases([label, *aliases, *(known.aliases if known else [])])
        for alias in all_aliases:
            alias_key = _alias_key(alias)
            owner = self._aliases.get(alias_key)
            if owner and owner != canonical_key:
                raise ValueError(f"Predicate alias {alias!r} already belongs to {owner!r}")
        predicate = Predicate(
            key=canonical_key,
            display_name=known.display_name if known else label.strip(),
            value_kind=value_kind,
            aliases=all_aliases,
            description=description or (known.description if known else None),
        )
        self._predicates[canonical_key] = predicate
        for alias in all_aliases:
            self._aliases[_alias_key(alias)] = canonical_key
        return predicate

    def register_compatible(
        self,
        label: str,
        *,
        value_kind: Literal["number", "text", "category", "date", "boolean", "identifier"],
        aliases: tuple[str, ...] | list[str] = (),
    ) -> Predicate | None:
        """Register a predicate, or return None when the label already means something else.

        Two extractors work the same document, and they do not always agree on what a
        label holds: rules may read "at the end of FY24" as a number while a model reads
        it as a date. One predicate has one value kind, so the disagreement is real and
        the later claim has to be dropped - but only that one claim. Raising here would
        abandon every remaining fact in the document, which is far worse than losing the
        candidate that caused it.
        """
        try:
            return self.register(label, value_kind=value_kind, aliases=aliases)
        except ValueError:
            return None

    def resolve(self, label: str) -> Predicate | None:
        key = self._aliases.get(_alias_key(label)) or canonical_predicate_key(label)
        return self._predicates.get(key)

    def all(self) -> tuple[Predicate, ...]:
        return tuple(self._predicates[key] for key in sorted(self._predicates))


# Reporting vocabulary, not knowledge about any particular company. Financial
# statements name the same line item several ways: a balance sheet says "revenue from
# contracts with customers" (the Ind AS 115 / IFRS 15 wording), the directors' report
# says "revenue from operations", and an investor deck says "revenue from customers".
# Without this table those spellings become separate predicates and a figure repeated
# across two documents can never be recognised as the same measure.
#
# Only genuine synonyms belong here. "Revenue from services" is deliberately absent:
# decks define it as excluding traded goods, so it is a narrower measure even when the
# two numbers happen to land close together. "Total income" is absent for the same
# reason - it adds other income to the top line.
_MEASURE_SYNONYMS = {
    "revenue_from_contracts_with_customers": "revenue_from_operations",
    "total_revenue_from_contracts_with_customers": "revenue_from_operations",
    "revenue_from_customers": "revenue_from_operations",
    "total_revenue_from_customers": "revenue_from_operations",
    "revenues_from_customers": "revenue_from_operations",
    "total_revenue": "revenue_from_operations",
}
# Profit and loss are deliberately not aliased to each other. Statements report both as
# positive magnitudes, so merging them without sign normalization would compare a loss
# against a profit and call the result a contradiction.


def canonical_predicate_key(label: str) -> str:
    source = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    source = source.casefold().replace("&", " and ").replace("%", " percentage ")
    source = re.sub(
        r"\((?:in\s+)?(?:₹|rs\.?|inr|usd|\$)?\s*(?:thousand|lakh|lac|million|mn|crore|cr|billion|bn)s?\)",
        " ",
        source,
    )
    source = re.sub(r"\b(?:in|amounts? in)\s+(?:inr|usd|rs\.?)?\s*(?:million|crore|billion|lakh)s?\b", " ", source)
    key = re.sub(r"[^a-z0-9]+", "_", source).strip("_")
    if not key or not key[0].isalpha():
        key = f"measure_{key}" if key else "unnamed_measure"
    return _MEASURE_SYNONYMS.get(key, key)


def compare_numeric_values(left: NumericValue, right: NumericValue) -> NumericComparison:
    """Compare normalized numbers using the precision visible in each source."""
    if left.currency != right.currency and left.currency and right.currency:
        return NumericComparison(False, None, None, None, None, "Currencies differ and no exchange rate was supplied")
    if left.unit != right.unit:
        return NumericComparison(False, None, None, None, None, "Normalized units differ")

    difference = abs(left.number - right.number)
    largest = max(abs(left.number), abs(right.number))
    relative = difference / largest if largest else Decimal("0")
    rounding = _rounding_tolerance(left) + _rounding_tolerance(right)
    approximate = max(largest * Decimal("0.01"), rounding) if left.approximate or right.approximate else rounding
    agrees = difference == 0 or difference < approximate
    reason = (
        "Values agree within their reported rounding precision"
        if agrees
        else "Values differ beyond their reported rounding precision"
    )
    if left.approximate or right.approximate:
        reason += " and approximate-value tolerance"
    return NumericComparison(True, agrees, difference, approximate, relative, reason)


def _rounding_tolerance(value: NumericValue) -> Decimal:
    reported = value.reported_number if value.reported_number is not None else value.number / value.scale
    exponent = reported.as_tuple().exponent
    displayed_step = (Decimal(10) ** exponent) * value.scale
    return abs(displayed_step) / 2


def _find_currency(source: str) -> tuple[str | None, str | None]:
    lowered = source.casefold()
    for marker in sorted(_CURRENCIES, key=len, reverse=True):
        if marker in {"₹", "$", "€", "£"}:
            matched = marker in source
        else:
            matched = bool(re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", lowered))
        if matched:
            return _CURRENCIES[marker]
    return None, None


def _find_scale(lowered: str) -> tuple[str | None, Decimal]:
    match = re.search(rf"\b({_SCALE_PATTERN})\.?\b", lowered)
    if not match:
        return None, Decimal("1")
    label = match.group(1)
    return label, _SCALES[label]


def _find_unit(lowered: str, number_end: int, scale_label: str | None) -> tuple[str | None, str | None]:
    suffix = lowered[number_end:].strip(" .,)\n\t")
    if scale_label:
        suffix = re.sub(rf"^(?:{re.escape(scale_label)})\.?\s*", "", suffix).strip()
    for alias in sorted(_UNIT_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", suffix):
            reported = f"{scale_label} {alias}".strip() if scale_label else alias
            return _UNIT_ALIASES[alias], reported
    return None, scale_label


def _decimal(raw: str) -> Decimal:
    return Decimal(raw.replace(",", ""))


def _failure(code: str, message: str) -> NormalizationResult:
    return NormalizationResult(
        value=None,
        confidence=0.0,
        warnings=(NormalizationWarning(code, message),),
    )


def _four_digit_year(raw: str | None) -> int:
    if raw is None:
        raise ValueError("A year is required")
    year = int(raw)
    if len(raw) == 4:
        return year
    return 2000 + year if year <= 50 else 1900 + year


def _day_before_year_start(end_year: int, fiscal_year_start_month: int) -> date:
    next_start = date(end_year, fiscal_year_start_month, 1)
    return date.fromordinal(next_start.toordinal() - 1)


def _year_period_start(end: date) -> date:
    try:
        anniversary = date(end.year - 1, end.month, end.day)
    except ValueError:
        anniversary = date(end.year - 1, end.month, 28)
    return date.fromordinal(anniversary.toordinal() + 1)


def _add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)


def _date_from_match(match: re.Match[str]) -> date:
    return date(int(match.group(3)), _MONTHS[match.group(1)], int(match.group(2)))


def _matched_labels(text: str, patterns: dict[str, str]) -> list[str]:
    return [label for label, pattern in patterns.items() if re.search(pattern, text)]


def _alias_key(label: str) -> str:
    return re.sub(r"\s+", " ", label.casefold()).strip()


def _unique_aliases(aliases: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        cleaned = alias.strip()
        key = _alias_key(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


__all__ = [
    "NormalizedPeriod",
    "NormalizationResult",
    "NormalizationWarning",
    "NumericComparison",
    "NumericRange",
    "PeriodKind",
    "PredicateRegistry",
    "canonical_predicate_key",
    "compare_numeric_values",
    "normalize_context",
    "normalize_date",
    "normalize_number",
    "normalize_numeric_range",
    "normalize_period",
]
