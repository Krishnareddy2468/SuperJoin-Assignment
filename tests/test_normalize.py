from datetime import date
from decimal import Decimal

import pytest

from factlayer.normalize import (
    PeriodKind,
    PredicateRegistry,
    canonical_predicate_key,
    compare_numeric_values,
    normalize_context,
    normalize_date,
    normalize_number,
    normalize_numeric_range,
    normalize_period,
)


@pytest.mark.parametrize(
    ("raw", "expected", "scale"),
    [
        ("1 thousand", "1000", "1000"),
        ("1.5 lakh", "150000.0", "100000"),
        ("2 million", "2000000", "1000000"),
        ("3 crore", "30000000", "10000000"),
        ("1.2 billion", "1200000000.0", "1000000000"),
    ],
)
def test_indian_and_international_scales(raw: str, expected: str, scale: str) -> None:
    result = normalize_number(raw)

    assert result.value.number == Decimal(expected)
    assert result.value.scale == Decimal(scale)
    assert result.value.raw == raw
    assert result.warnings == ()


def test_indian_grouping_currency_and_accounting_negative() -> None:
    result = normalize_number("(₹1,23,456.75 crore)")

    assert result.value.reported_number == Decimal("-123456.75")
    assert result.value.number == Decimal("-1234567500000.00")
    assert result.value.currency == "INR"
    assert result.value.unit == "rupee"
    assert result.value.reported_unit == "crore"


@pytest.mark.parametrize(
    ("raw", "number", "unit"),
    [
        ("12.5%", "12.5", "percent"),
        ("75 basis points", "0.75", "percentage_point"),
        ("1.8x", "1.8", "ratio"),
        ("3:2", "1.5", "ratio"),
        ("2 million tonnes", "2000000", "tonne"),
        ("450 shipments", "450", "shipment"),
    ],
)
def test_percentages_ratios_and_ordinary_units(raw: str, number: str, unit: str) -> None:
    result = normalize_number(raw)

    assert result.value.number == Decimal(number)
    assert result.value.unit == unit


def test_approximate_values_are_marked_without_changing_the_number() -> None:
    result = normalize_number("approximately ₹8,142 crore")

    assert result.value.number == Decimal("81420000000")
    assert result.value.approximate is True


def test_numeric_ranges_keep_both_normalized_ends() -> None:
    result = normalize_numeric_range("between ₹10–12 crore")

    assert result.value.lower == Decimal("100000000")
    assert result.value.upper == Decimal("120000000")
    assert result.value.currency == "INR"
    assert result.value.unit == "rupee"
    assert result.value.approximate is True


@pytest.mark.parametrize(
    ("raw", "expected", "precision"),
    [
        ("2024-03-31", date(2024, 3, 31), "day"),
        ("March 31st, 2024", date(2024, 3, 31), "day"),
        ("31 March 2024", date(2024, 3, 31), "day"),
        ("March 2024", date(2024, 3, 1), "month"),
        ("2024", date(2024, 1, 1), "year"),
    ],
)
def test_standalone_dates_keep_their_reported_precision(
    raw: str, expected: date, precision: str
) -> None:
    result = normalize_date(raw)

    assert result.value.value == expected
    assert result.value.precision == precision


def test_invalid_standalone_date_is_not_guessed() -> None:
    result = normalize_date("February 30, 2024")

    assert result.value is None
    assert result.warnings[0].code == "invalid_date"


def test_reversed_and_missing_ranges_fail_visibly() -> None:
    reversed_result = normalize_numeric_range("12 to 10 percent")
    missing_result = normalize_numeric_range("about 10 percent")

    assert reversed_result.value is None
    assert reversed_result.warnings[0].code == "reversed_range"
    assert missing_result.value is None
    assert missing_result.warnings[0].code == "missing_range"


def test_unknown_units_and_ambiguous_numbers_preserve_uncertainty() -> None:
    unknown = normalize_number("42 widgets")
    abbreviation_trap = normalize_number("42 workers")
    ambiguous = normalize_number("Revenue changed from 10 to 12")

    assert unknown.value.raw == "42 widgets"
    assert unknown.value.reported_unit == "widgets"
    assert unknown.warnings[0].code == "unknown_unit"
    assert abbreviation_trap.warnings[0].code == "unknown_unit"
    assert unknown.needs_review is True
    assert ambiguous.value is None
    assert ambiguous.warnings[0].code == "ambiguous_number"


@pytest.mark.parametrize(
    ("raw", "start", "end", "kind"),
    [
        ("FY24", date(2023, 4, 1), date(2024, 3, 31), PeriodKind.FISCAL_YEAR),
        ("FY 2023-24", date(2023, 4, 1), date(2024, 3, 31), PeriodKind.FISCAL_YEAR),
        ("year ended March 31, 2024", date(2023, 4, 1), date(2024, 3, 31), PeriodKind.DATE_RANGE),
        ("Q4 FY24", date(2024, 1, 1), date(2024, 3, 31), PeriodKind.QUARTER),
        ("Q1 2024", date(2024, 1, 1), date(2024, 3, 31), PeriodKind.QUARTER),
        ("month ended February 2024", date(2024, 2, 1), date(2024, 2, 29), PeriodKind.MONTH),
    ],
)
def test_reporting_period_variants(
    raw: str, start: date, end: date, kind: PeriodKind
) -> None:
    result = normalize_period(raw)

    assert result.value.kind is kind
    assert result.value.period.start == start
    assert result.value.period.end == end


def test_point_in_time_and_explicit_date_range_remain_distinct() -> None:
    point = normalize_period("as of March 31, 2024")
    period = normalize_period("from April 1, 2023 to March 31, 2024")

    assert point.value.kind is PeriodKind.POINT_IN_TIME
    assert point.value.as_of == date(2024, 3, 31)
    assert point.value.period is None
    assert period.value.kind is PeriodKind.DATE_RANGE
    assert period.value.period.start == date(2023, 4, 1)
    assert period.value.period.end == date(2024, 3, 31)

    day_first = normalize_period("as at 31 March 2024")
    assert day_first.value.as_of == date(2024, 3, 31)


def test_bare_year_is_retained_with_an_explicit_assumption_warning() -> None:
    result = normalize_period("2024")

    assert result.value.kind is PeriodKind.CALENDAR_YEAR
    assert result.warnings[0].code == "year_assumed_calendar"
    assert result.needs_review is True


def test_unknown_period_is_not_invented() -> None:
    result = normalize_period("during the period under review")

    assert result.value is None
    assert result.warnings[0].code == "unknown_period"


def test_invalid_calendar_date_becomes_a_reviewable_failure() -> None:
    result = normalize_period("as of February 30, 2024")

    assert result.value is None
    assert result.warnings[0].code == "invalid_period"


def test_leap_day_year_end_starts_on_the_following_day_in_the_prior_year() -> None:
    result = normalize_period("year ended February 29, 2024")

    assert result.value.period.start == date(2023, 3, 1)
    assert result.value.period.end == date(2024, 2, 29)


def test_two_digit_historical_fiscal_year_uses_a_clear_century_pivot() -> None:
    result = normalize_period("FY99")

    assert result.value.period.start == date(1998, 4, 1)
    assert result.value.period.end == date(1999, 3, 31)


def test_context_collects_scope_basis_period_geography_and_publisher() -> None:
    result = normalize_context(
        "Audited consolidated actual results for India for FY 2023-24",
        publisher="Delhivery Limited",
    )

    assert result.value.period.start == date(2023, 4, 1)
    assert result.value.scope == "consolidated"
    assert result.value.basis == "audited; actual"
    assert result.value.geography == "India"
    assert result.value.publisher == "Delhivery Limited"
    assert result.value.qualifiers["value_status"] == "actual"
    assert result.value.qualifiers["period_kind"] == "fiscal_year"
    assert result.warnings == ()


def test_actual_forecast_and_standalone_consolidated_conflicts_are_not_guessed() -> None:
    result = normalize_context(
        "Standalone and consolidated actual and forecast values for FY24"
    )

    assert result.value.scope is None
    assert "value_status" not in result.value.qualifiers
    assert {warning.code for warning in result.warnings} == {
        "ambiguous_scope",
        "ambiguous_value_status",
    }
    assert result.confidence == 0.65


def test_missing_context_is_visible() -> None:
    result = normalize_context("Revenue increased significantly")

    assert result.value.period is None
    assert result.value.scope is None
    assert result.warnings[0].code == "missing_context"
    assert result.confidence == 0.45


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Revenue from Operations", "revenue_from_operations"),
        ("Revenue from operations (₹ million)", "revenue_from_operations"),
        ("Profit & Loss", "profit_and_loss"),
        ("EBITDA %", "ebitda_percentage"),
        ("2024 estimate", "measure_2024_estimate"),
    ],
)
def test_predicate_keys_are_stable_and_unit_free(label: str, expected: str) -> None:
    assert canonical_predicate_key(label) == expected


def test_dynamic_predicate_registry_resolves_aliases() -> None:
    registry = PredicateRegistry()
    predicate = registry.register(
        "Revenue from operations",
        value_kind="number",
        aliases=["Operating revenue", "Revenue from Operations"],
    )

    assert predicate.key == "revenue_from_operations"
    assert registry.resolve("operating REVENUE") == predicate
    assert registry.resolve("Revenue from operations") == predicate
    assert registry.all() == (predicate,)


def test_predicate_alias_and_value_kind_conflicts_are_rejected() -> None:
    registry = PredicateRegistry()
    registry.register("Revenue", value_kind="number", aliases=["Turnover"])
    registry.register("Employee count", value_kind="number")

    with pytest.raises(ValueError, match="already belongs"):
        registry.register("Sales", value_kind="number", aliases=["Turnover"])
    with pytest.raises(ValueError, match="already registered"):
        registry.register("Revenue", value_kind="category")


def test_million_and_crore_revenue_agree_within_visible_rounding() -> None:
    annual_report = normalize_number("₹81,415.38 million").value
    earnings_deck = normalize_number("₹8,142 crore").value

    comparison = compare_numeric_values(annual_report, earnings_deck)

    assert comparison.comparable is True
    assert comparison.agrees is True
    assert comparison.difference == Decimal("4620000.00")
    assert comparison.allowed_difference == Decimal("5005000.000")


def test_values_one_display_unit_apart_do_not_agree() -> None:
    left = normalize_number("₹100 crore").value
    right = normalize_number("₹101 crore").value

    comparison = compare_numeric_values(left, right)

    assert comparison.comparable is True
    assert comparison.agrees is False


def test_different_units_or_currencies_are_not_compared() -> None:
    tonnes = normalize_number("10 tonnes").value
    shipments = normalize_number("10 shipments").value
    dollars = normalize_number("$10 million").value
    rupees = normalize_number("₹10 million").value

    assert compare_numeric_values(tonnes, shipments).comparable is False
    assert compare_numeric_values(dollars, rupees).comparable is False
