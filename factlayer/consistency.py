"""Auditable consistency checks for growth, totals, and repeated metrics."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from typing import Iterable

from factlayer.link import FactLinker
from factlayer.normalize import canonical_predicate_key
from factlayer.schema import (
    ConsistencyCheckType,
    ConsistencyFinding,
    Fact,
    NumericValue,
    RelationType,
    ReportingPeriod,
    ReviewState,
)
from factlayer.store import FactStore


CONSISTENCY_RULE_VERSION = "consistency-v1"


class ConsistencyChecker:
    """Recompute claims using normalized, grounded fact values."""

    def __init__(
        self,
        *,
        rule_version: str = CONSISTENCY_RULE_VERSION,
        linker: FactLinker | None = None,
    ):
        if not rule_version.strip():
            raise ValueError("A consistency rule version is required")
        self.rule_version = rule_version
        self.linker = linker or FactLinker()

    def check_growth(
        self,
        *,
        current: Fact,
        prior: Fact,
        stated_growth: Fact,
    ) -> ConsistencyFinding:
        """Verify a reported percentage with ``(current-prior)/abs(prior)``."""
        facts = [current, prior, stated_growth]
        self._require_ids(facts)
        numeric = self._numeric_values(facts)
        formula = "growth_percent = (current - prior) / abs(prior) * 100"
        operands = {
            "current": self._operand(current),
            "prior": self._operand(prior),
            "stated_growth": self._operand(stated_growth),
        }
        issues = self._growth_issues(current, prior, stated_growth, numeric)
        current_value, prior_value, growth_value = numeric
        stated = self._as_percent(growth_value)
        if prior_value.number == 0:
            issues.append("The prior value is zero, so percentage growth is undefined.")
            return self._finding(
                check_type=ConsistencyCheckType.GROWTH,
                facts=facts,
                relation_type=RelationType.NEEDS_REVIEW,
                formula=formula,
                operands=operands,
                stated=stated,
                calculated=None,
                difference=None,
                tolerance=None,
                explanation="Growth cannot be recomputed because the prior value is zero.",
                confidence=self._source_confidence(facts) * 0.5,
            )

        calculated = (
            (current_value.number - prior_value.number)
            / abs(prior_value.number)
            * Decimal("100")
        )
        difference = abs(stated - calculated)
        tolerance = self._growth_tolerance(
            current_value,
            prior_value,
            growth_value,
            calculated,
        )
        relation_type = self._calculation_decision(
            difference=difference,
            tolerance=tolerance,
            facts=facts,
            issues=issues,
        )
        explanation = self._calculation_explanation(
            label="Reported growth",
            stated=stated,
            calculated=calculated,
            difference=difference,
            tolerance=tolerance,
            relation_type=relation_type,
            issues=issues,
            unit="percentage points",
        )
        return self._finding(
            check_type=ConsistencyCheckType.GROWTH,
            facts=facts,
            relation_type=relation_type,
            formula=formula,
            operands=operands,
            stated=stated,
            calculated=calculated,
            difference=difference,
            tolerance=tolerance,
            explanation=explanation,
            confidence=self._finding_confidence(facts, relation_type),
        )

    def check_component_total(
        self,
        *,
        total: Fact,
        components: Iterable[Fact],
        components_complete: bool,
    ) -> ConsistencyFinding:
        """Compare a stated total with a caller-confirmed component set."""
        component_list = sorted(components, key=lambda fact: fact.id or "")
        if not component_list:
            raise ValueError("A component-total check needs at least one component")
        facts = [total, *component_list]
        self._require_ids(facts)
        values = self._numeric_values(facts)
        total_value, *component_values = values
        formula = "stated_total = sum(components)"
        operands = {
            "stated_total": self._operand(total),
            "components": [
                self._operand(component) for component in component_list
            ],
            "components_complete": components_complete,
        }
        issues = self._total_issues(total, component_list, values)
        if not components_complete:
            issues.append("The supplied component list is not confirmed complete.")
        calculated = sum(
            (component.number for component in component_values),
            Decimal("0"),
        )
        stated = total_value.number
        difference = abs(stated - calculated)
        tolerance = self._visible_tolerance(total_value) + sum(
            (self._visible_tolerance(component) for component in component_values),
            Decimal("0"),
        )
        if not components_complete or self._blocking_total_issue(issues):
            relation_type = RelationType.NEEDS_REVIEW
        else:
            relation_type = self._calculation_decision(
                difference=difference,
                tolerance=tolerance,
                facts=facts,
                issues=issues,
            )
        explanation = self._calculation_explanation(
            label="Stated total",
            stated=stated,
            calculated=calculated,
            difference=difference,
            tolerance=tolerance,
            relation_type=relation_type,
            issues=issues,
            unit=total_value.currency or total_value.unit or "normalized units",
        )
        return self._finding(
            check_type=ConsistencyCheckType.COMPONENT_TOTAL,
            facts=facts,
            relation_type=relation_type,
            formula=formula,
            operands=operands,
            stated=stated,
            calculated=calculated,
            difference=difference,
            tolerance=tolerance,
            explanation=explanation,
            confidence=self._finding_confidence(facts, relation_type),
        )

    def check_repeated_metric(self, left: Fact, right: Fact) -> ConsistencyFinding:
        """Turn a normal fact comparison into a stored consistency audit."""
        self._require_ids([left, right])
        if left.id > right.id:
            left, right = right, left
        relation = self.linker.compare(left, right)
        numeric = (
            isinstance(left.value, NumericValue)
            and isinstance(right.value, NumericValue)
        )
        stated = left.value.number if numeric else None
        calculated = right.value.number if numeric else None
        difference = relation.value_comparison.difference if numeric else None
        tolerance = relation.value_comparison.tolerance if numeric else None
        methods = sorted(
            {
                evidence.extractor.value
                for fact in (left, right)
                for evidence in fact.evidence
            }
        )
        operands = {
            "left": self._operand(left),
            "right": self._operand(right),
            "extraction_methods": methods,
            "context_diff": [
                item.model_dump(mode="json") for item in relation.context_diff
            ],
        }
        return self._finding(
            check_type=ConsistencyCheckType.REPEATED_METRIC,
            facts=[left, right],
            relation_type=relation.relation_type,
            formula="normalized(left_value) = normalized(right_value)",
            operands=operands,
            stated=stated,
            calculated=calculated,
            difference=difference,
            tolerance=tolerance,
            explanation=relation.explanation,
            confidence=relation.confidence,
            review_state=relation.review_state,
        )

    def discover_growth_checks(self, facts: Iterable[Fact]) -> tuple[ConsistencyFinding, ...]:
        """Find stated growth percentages and recompute them from their own operands.

        This is where a genuine contradiction can come from. Corroboration and
        reconciliation compare what two documents said; here the system does its own
        arithmetic and checks the document against itself, so a claim like "revenue grew
        12.7%" is tested against the two revenue figures the same filing reports. The
        search is driven by predicate shape rather than by any known measure, so an
        unfamiliar document contributes checks without new code.
        """
        by_key: dict[tuple[str, str], list[Fact]] = {}
        for fact in facts:
            if fact.id and fact.subject.id:
                by_key.setdefault((fact.subject.id, fact.predicate.key), []).append(fact)

        findings: list[ConsistencyFinding] = []
        for (subject_id, key), stated_facts in sorted(by_key.items()):
            base_key = _growth_base_key(key)
            if base_key is None:
                continue
            operands = by_key.get((subject_id, base_key), ())
            for stated in stated_facts:
                if not _is_percentage(stated) or stated.context.period is None:
                    continue
                # Several figures can match the stated period - a filing reports the
                # same measure on a standalone and a consolidated basis - and only some
                # of them have a comparable prior-year figure to work from. Try each
                # until one yields a complete pair rather than giving up on the first.
                for current in _matching_values(operands, stated.context.period, stated):
                    prior = _preceding_value(operands, current)
                    if prior is None:
                        continue
                    findings.append(
                        self.check_growth(current=current, prior=prior, stated_growth=stated)
                    )
                    break
        return tuple(findings)

    def discover_component_total_checks(
        self,
        facts: Iterable[Fact],
    ) -> tuple[ConsistencyFinding, ...]:
        """Check stated totals against the figures printed above them.

        A statement column is a run of line items ending in a total, so the components
        are knowable from grid position rather than from wording. Rows are gathered
        upwards from the total and stop at the previous total, which is what keeps a
        sub-total from being counted twice inside a section.

        Completeness is asserted only for a plain, unbroken run. Anything else is passed
        as incomplete, which makes the checker report the arithmetic without calling a
        mismatch a contradiction - the right default when a missing line item looks
        exactly like a wrong one.
        """
        columns: dict[tuple, list[Fact]] = {}
        for fact in facts:
            if not fact.id or not isinstance(fact.value, NumericValue):
                continue
            marks = fact.context.qualifiers
            if not {"table_index", "table_row", "table_column"} <= marks.keys():
                continue
            evidence = fact.evidence[0] if fact.evidence else None
            if evidence is None:
                continue
            key = (
                fact.subject.id,
                evidence.document_id,
                evidence.page_index,
                marks["table_index"],
                marks["table_column"],
            )
            columns.setdefault(key, []).append(fact)

        findings: list[ConsistencyFinding] = []
        for column in columns.values():
            ordered = sorted(column, key=lambda item: int(item.context.qualifiers["table_row"]))
            section: list[Fact] = []
            for fact in ordered:
                if not _is_total_label(fact.predicate.key):
                    section.append(fact)
                    continue
                components = [
                    item
                    for item in section
                    if _comparable_component(item, fact)
                ]
                if len(components) >= 2:
                    findings.append(
                        self.check_component_total(
                            total=fact,
                            components=components,
                            # Never asserted, and the reason is worth keeping. Counting
                            # the rows we parsed cannot tell us about rows we failed to
                            # parse, and an income statement routinely prints a subtotal
                            # among its line items - "revenue from customers" already
                            # contains "revenue from services" - so a naive sum double
                            # counts. Both faults look exactly like a document
                            # contradicting itself. Findings are therefore raised as
                            # review candidates; promoting one to a contradiction is a
                            # judgement a person makes after reading the statement.
                            components_complete=False,
                        )
                    )
                section = []
        return tuple(findings)

    def check_repeated_metrics(
        self,
        facts: Iterable[Fact],
        *,
        cross_document_only: bool = False,
    ) -> tuple[ConsistencyFinding, ...]:
        return tuple(
            self.check_repeated_metric(left, right)
            for left, right in self.linker.candidate_pairs(
                facts,
                cross_document_only=cross_document_only,
            )
        )

    @staticmethod
    def persist(
        store: FactStore,
        finding: ConsistencyFinding,
    ) -> tuple[ConsistencyFinding, bool]:
        return store.put_consistency_finding(finding)

    def _growth_issues(
        self,
        current: Fact,
        prior: Fact,
        stated: Fact,
        values: list[NumericValue],
    ) -> list[str]:
        issues: list[str] = []
        if current.subject.id != prior.subject.id or current.subject.id != stated.subject.id:
            issues.append("Growth operands do not share the same resolved subject.")
        if current.predicate.key != prior.predicate.key:
            issues.append("Current and prior values use different predicates.")
        if not self._same_numeric_unit(values[0], values[1]):
            issues.append("Current and prior values use incompatible units or currencies.")
        if values[2].unit not in {"percent", "ratio"}:
            issues.append("The stated growth fact is not a percentage or ratio.")
        issues.extend(
            self._shared_context_issues(
                [current, prior, stated],
                include_period=False,
            )
        )
        if not self._ordered_comparable_periods(current, prior):
            issues.append("Current and prior reporting periods are missing or not comparable.")
        if stated.context.period and current.context.period:
            if self._period_key(stated) != self._period_key(current):
                issues.append("The stated growth period does not match the current value period.")
        elif stated.context.period is None:
            issues.append("The stated growth fact has no reporting period.")
        return issues

    def _total_issues(
        self,
        total: Fact,
        components: list[Fact],
        values: list[NumericValue],
    ) -> list[str]:
        issues: list[str] = []
        if any(component.subject.id != total.subject.id for component in components):
            issues.append("The total and components do not share the same resolved subject.")
        if any(not self._same_numeric_unit(values[0], value) for value in values[1:]):
            issues.append("The total and components use incompatible units or currencies.")
        issues.extend(self._shared_context_issues([total, *components], include_period=True))
        if not self._has_shared_context([total, *components]) and not self._co_located(
            [total, *components]
        ):
            issues.append("The operands have no shared reporting context or source location.")
        return issues

    @staticmethod
    def _blocking_total_issue(issues: list[str]) -> bool:
        return any(
            marker in issue
            for issue in issues
            for marker in (
                "do not share",
                "incompatible",
                "no shared reporting context",
                "not confirmed complete",
                "different period",
                "different scope",
                "different basis",
                "different as of",
                "different geography",
            )
        )

    def _calculation_decision(
        self,
        *,
        difference: Decimal,
        tolerance: Decimal,
        facts: list[Fact],
        issues: list[str],
    ) -> RelationType:
        blocking = any(
            text in issue
            for issue in issues
            for text in (
                "do not share",
                "different predicates",
                "incompatible",
                "not a percentage",
                "missing or not comparable",
                "does not match",
                "has no reporting period",
                "different scope",
                "different basis",
                "different as of",
                "different geography",
            )
        )
        if blocking:
            return RelationType.NEEDS_REVIEW
        if difference <= tolerance:
            return RelationType.CORROBORATES
        if self._has_uncertainty(facts) or issues:
            return RelationType.LIKELY_CONTRADICTION
        return RelationType.CONTRADICTS

    def _finding(
        self,
        *,
        check_type: ConsistencyCheckType,
        facts: list[Fact],
        relation_type: RelationType,
        formula: str,
        operands: dict,
        stated: Decimal | None,
        calculated: Decimal | None,
        difference: Decimal | None,
        tolerance: Decimal | None,
        explanation: str,
        confidence: float,
        review_state: ReviewState | None = None,
    ) -> ConsistencyFinding:
        fact_ids = [fact.id for fact in facts]
        state = review_state or (
            ReviewState.NEEDS_REVIEW
            if relation_type
            in {
                RelationType.LIKELY_CONTRADICTION,
                RelationType.NEEDS_REVIEW,
                RelationType.INCOMPARABLE,
            }
            else ReviewState.READY
        )
        return ConsistencyFinding(
            id=self._finding_id(check_type, fact_ids),
            check_type=check_type,
            fact_ids=fact_ids,
            relation_type=relation_type,
            formula=formula,
            operands=operands,
            stated_result=stated,
            calculated_result=calculated,
            difference=difference,
            tolerance=tolerance,
            explanation=explanation,
            confidence=max(0, min(1, round(confidence, 6))),
            rule_version=self.rule_version,
            review_state=state,
        )

    @staticmethod
    def _calculation_explanation(
        *,
        label: str,
        stated: Decimal,
        calculated: Decimal,
        difference: Decimal,
        tolerance: Decimal,
        relation_type: RelationType,
        issues: list[str],
        unit: str,
    ) -> str:
        if relation_type is RelationType.CORROBORATES:
            return (
                f"{label} {stated} agrees with the recomputed {calculated} within "
                f"the {tolerance} {unit} rounding tolerance."
            )
        issue_text = f" Review notes: {' '.join(issues)}" if issues else ""
        if relation_type is RelationType.CONTRADICTS:
            return (
                f"{label} {stated} differs from the recomputed {calculated} by "
                f"{difference} {unit}, beyond the {tolerance} tolerance."
            )
        if relation_type is RelationType.LIKELY_CONTRADICTION:
            return (
                f"{label} {stated} differs from the recomputed {calculated} by "
                f"{difference} {unit}, beyond the {tolerance} tolerance, but source "
                f"uncertainty prevents a definitive contradiction.{issue_text}"
            )
        return (
            f"{label} {stated} and recomputed {calculated} cannot be judged safely."
            f"{issue_text}"
        )

    @staticmethod
    def _numeric_values(facts: list[Fact]) -> list[NumericValue]:
        if any(not isinstance(fact.value, NumericValue) for fact in facts):
            raise ValueError("Consistency calculations require numeric facts")
        return [fact.value for fact in facts]  # type: ignore[misc]

    @staticmethod
    def _require_ids(facts: list[Fact]) -> None:
        if any(not fact.id for fact in facts):
            raise ValueError("Facts need stable IDs before a consistency check")
        if len({fact.id for fact in facts}) != len(facts):
            raise ValueError("A consistency check cannot use the same fact twice")

    @staticmethod
    def _as_percent(value: NumericValue) -> Decimal:
        if value.unit == "percent":
            return value.number
        if value.unit == "ratio":
            return value.number * Decimal("100")
        return value.number

    def _growth_tolerance(
        self,
        current: NumericValue,
        prior: NumericValue,
        stated: NumericValue,
        calculated: Decimal,
    ) -> Decimal:
        current_tolerance = self._visible_tolerance(current)
        prior_tolerance = self._visible_tolerance(prior)
        propagated = (
            current_tolerance / abs(prior.number) * Decimal("100")
            + abs(current.number)
            * prior_tolerance
            / (prior.number * prior.number)
            * Decimal("100")
        )
        stated_tolerance = self._visible_tolerance(stated)
        if stated.unit == "ratio":
            stated_tolerance *= Decimal("100")
        if current.approximate or prior.approximate or stated.approximate:
            propagated = max(propagated, abs(calculated) * Decimal("0.01"))
        return propagated + stated_tolerance

    @staticmethod
    def _visible_tolerance(value: NumericValue) -> Decimal:
        reported = (
            value.reported_number
            if value.reported_number is not None
            else value.number / value.scale
        )
        step = Decimal("10") ** reported.as_tuple().exponent * value.scale
        tolerance = abs(step) / Decimal("2")
        if value.approximate:
            tolerance = max(tolerance, abs(value.number) * Decimal("0.01"))
        return tolerance

    @staticmethod
    def _same_numeric_unit(left: NumericValue, right: NumericValue) -> bool:
        return left.unit == right.unit and (
            left.currency == right.currency
            or left.currency is None
            or right.currency is None
        )

    def _shared_context_issues(
        self,
        facts: list[Fact],
        *,
        include_period: bool,
    ) -> list[str]:
        fields = ["scope", "basis", "as_of", "geography"]
        if include_period:
            fields.insert(0, "period")
        issues: list[str] = []
        for field in fields:
            values = {self._context_value(fact, field) for fact in facts}
            if len(values) > 1:
                issues.append(f"The operands have different {field.replace('_', ' ')} context.")
        return issues

    @staticmethod
    def _context_value(fact: Fact, field: str):
        if field == "period":
            return ConsistencyChecker._period_key(fact)
        value = getattr(fact.context, field)
        return value.casefold().strip() if isinstance(value, str) else value

    @staticmethod
    def _period_key(fact: Fact):
        period = fact.context.period
        if period is None:
            return None
        if period.start or period.end:
            return period.start, period.end
        return period.label.casefold().strip()

    @staticmethod
    def _ordered_comparable_periods(current: Fact, prior: Fact) -> bool:
        current_period = current.context.period
        prior_period = prior.context.period
        if not current_period or not prior_period:
            return False
        if not current_period.end or not prior_period.end:
            return current_period.label != prior_period.label
        if current_period.end <= prior_period.end:
            return False
        if current_period.start and prior_period.start:
            current_days = (current_period.end - current_period.start).days
            prior_days = (prior_period.end - prior_period.start).days
            return abs(current_days - prior_days) <= 7
        return True

    @staticmethod
    def _has_shared_context(facts: list[Fact]) -> bool:
        return any(
            (
                facts[0].context.period,
                facts[0].context.scope,
                facts[0].context.basis,
                facts[0].context.as_of,
                facts[0].context.geography,
            )
        )

    @staticmethod
    def _co_located(facts: list[Fact]) -> bool:
        locations = [
            {
                (evidence.document_id, evidence.page_index, evidence.passage_id)
                for evidence in fact.evidence
            }
            for fact in facts
        ]
        return bool(set.intersection(*locations)) if locations else False

    @staticmethod
    def _has_uncertainty(facts: list[Fact]) -> bool:
        return any(
            fact.review_state is not ReviewState.READY
            or fact.warnings
            or (isinstance(fact.value, NumericValue) and fact.value.approximate)
            for fact in facts
        )

    @staticmethod
    def _source_confidence(facts: list[Fact]) -> float:
        return min(
            min(fact.extraction_confidence, fact.normalization_confidence)
            for fact in facts
        )

    def _finding_confidence(
        self,
        facts: list[Fact],
        relation_type: RelationType,
    ) -> float:
        multiplier = {
            RelationType.CORROBORATES: 0.97,
            RelationType.CONTRADICTS: 0.95,
            RelationType.LIKELY_CONTRADICTION: 0.72,
            RelationType.NEEDS_REVIEW: 0.5,
        }.get(relation_type, 0.65)
        return self._source_confidence(facts) * multiplier

    @staticmethod
    def _operand(fact: Fact) -> dict:
        value = fact.value
        if isinstance(value, NumericValue):
            normalized = str(value.number)
            unit = value.currency or value.unit
        else:
            normalized = value.model_dump(mode="json")
            unit = None
        return {
            "fact_id": fact.id,
            "predicate": fact.predicate.key,
            "raw": value.raw,
            "normalized": normalized,
            "unit": unit,
            "context": fact.context.model_dump(mode="json"),
        }

    def _finding_id(
        self,
        check_type: ConsistencyCheckType,
        fact_ids: list[str],
    ) -> str:
        payload = json.dumps(
            [check_type.value, fact_ids, self.rule_version],
            separators=(",", ":"),
        )
        return f"check_{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def _growth_base_key(predicate_key: str) -> str | None:
    """Return the measure a growth predicate is derived from, if it is one.

    "revenues_from_customers_percentage_change" describes the change in
    "revenues_from_customers", so the base name says which figures to recompute from.
    The result is re-canonicalized so that reporting synonyms line up with whatever the
    other document called the same measure.
    """
    for suffix in ("_percentage_change", "_percent_change", "_growth", "_change"):
        if predicate_key.endswith(suffix) and len(predicate_key) > len(suffix):
            return canonical_predicate_key(predicate_key[: -len(suffix)].replace("_", " "))
    return None


def _is_total_label(predicate_key: str) -> bool:
    """True when a row label reads as a total rather than a line item."""
    return bool(re.match(r"^(?:total|sub_?total|grand_total)(?:_|$)", predicate_key))


def _comparable_component(component: Fact, total: Fact) -> bool:
    """Only add up figures that are actually addable to the total."""
    if not isinstance(component.value, NumericValue) or not isinstance(total.value, NumericValue):
        return False
    if component.value.unit != total.value.unit:
        return False
    if component.value.currency != total.value.currency:
        return False
    # A percentage column has no meaningful sum, and a margin row inside a rupee table
    # would poison one.
    return component.value.unit != "percent"


def _is_percentage(fact: Fact) -> bool:
    return isinstance(fact.value, NumericValue) and fact.value.unit == "percent"


def _matching_values(
    candidates: Iterable[Fact],
    period: ReportingPeriod,
    stated: Fact,
) -> list[Fact]:
    """List the amounts a stated growth figure could be describing."""
    matches = []
    for fact in candidates:
        if not isinstance(fact.value, NumericValue) or _is_percentage(fact):
            continue
        if fact.context.period is None or fact.context.period.label != period.label:
            continue
        # Growth is only meaningful between figures reported on the same basis, so a
        # standalone amount is never used to explain a consolidated percentage.
        if fact.context.scope != stated.context.scope and None not in (
            fact.context.scope,
            stated.context.scope,
        ):
            continue
        matches.append(fact)
    return matches


def _preceding_value(candidates: Iterable[Fact], current: Fact) -> Fact | None:
    """Find the comparable amount for the period just before ``current``."""
    period = current.context.period
    if period is None or period.start is None or period.end is None:
        return None
    span = (period.end - period.start).days
    best: Fact | None = None
    for fact in candidates:
        other = fact.context.period
        if other is None or other.start is None or other.end is None:
            continue
        if other.label == period.label or fact.id == current.id:
            continue
        if not isinstance(fact.value, NumericValue) or _is_percentage(fact):
            continue
        if fact.context.scope != current.context.scope:
            continue
        gap = (period.start - other.end).days
        # The prior period should butt up against the current one. A wider window would
        # happily pair FY22 with FY24 and report a false mismatch.
        if not -1 <= gap <= 45:
            continue
        # It must also cover a comparable stretch of time. A deck prints quarters and
        # full years in one table, and the quarter ending the day before a financial
        # year starts is adjacent without being the prior period - pairing them turns a
        # correct 12.7% into a nonsense 338%.
        other_span = (other.end - other.start).days
        if span and abs(other_span - span) > max(31, span * 0.25):
            continue
        if best is None or other.start > best.context.period.start:
            best = fact
    return best


__all__ = ["CONSISTENCY_RULE_VERSION", "ConsistencyChecker"]
