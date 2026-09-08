"""Context-aware comparison of resolved facts.

The linker does not infer hidden context. It compares normalized values, records
each relevant context difference, and applies a small decision table whose
inputs are stored on the resulting relation.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import combinations
from typing import Any, Iterable

from factlayer.entities import normalize_identifier, normalize_identifier_scheme
from factlayer.normalize import compare_numeric_values
from factlayer.schema import (
    BooleanValue,
    CategoricalValue,
    ContextDifference,
    DateValue,
    Fact,
    IdentifierValue,
    NumericValue,
    Relation,
    RelationType,
    ReviewState,
    TextValue,
    ValueComparison,
)
from factlayer.store import FactStore


RULE_VERSION = "context-link-v1"

_RECONCILIATION_TYPES = {
    "period": RelationType.RECONCILED_BY_PERIOD,
    "scope": RelationType.RECONCILED_BY_SCOPE,
    "basis": RelationType.RECONCILED_BY_BASIS,
    "as_of": RelationType.RECONCILED_BY_AS_OF,
    "unit": RelationType.RECONCILED_BY_UNIT,
}
_EXPLANATORY_CONTEXT_FIELDS = {
    "period",
    "scope",
    "basis",
    "as_of",
    "unit",
    "geography",
}


@dataclass(frozen=True)
class _ComparedValue:
    comparison: ValueComparison
    comparable: bool
    reason: str
    unit_reconciled: bool = False
    uncertain: bool = False


class FactLinker:
    """Compare facts with the same resolved subject and dynamic predicate."""

    def __init__(self, *, rule_version: str = RULE_VERSION):
        if not rule_version.strip():
            raise ValueError("A relation rule version is required")
        self.rule_version = rule_version

    def candidate_pairs(
        self,
        facts: Iterable[Fact],
        *,
        cross_document_only: bool = False,
    ) -> tuple[tuple[Fact, Fact], ...]:
        """Group only relevant facts and return each canonical pair once.

        Same-document pairs are included by default because some of the most useful
        reconciliations live inside a single filing: one annual-report page states
        revenue on both a standalone and a consolidated basis, and skipping those
        pairs would hide the scope explanation entirely.
        """
        unique = {fact.id: fact for fact in facts if fact.id and fact.subject.id}
        groups: dict[tuple[str, str], list[Fact]] = {}
        for fact in unique.values():
            key = (fact.subject.id, fact.predicate.key)
            groups.setdefault(key, []).append(fact)

        pairs: list[tuple[Fact, Fact]] = []
        for group in groups.values():
            ordered = sorted(group, key=lambda fact: fact.id)
            for left, right in combinations(ordered, 2):
                if cross_document_only and not self._cross_document(left, right):
                    continue
                if not self._independent_claims(left, right):
                    continue
                pairs.append((left, right))
        return tuple(sorted(pairs, key=lambda pair: (pair[0].id, pair[1].id)))

    def compare(self, left: Fact, right: Fact) -> Relation:
        left, right = self._canonical_pair(left, right)
        self._validate_pair(left, right)
        value = self._compare_values(left, right)
        context_diff = self._context_diff(left, right, unit_reconciled=value.unit_reconciled)
        relation_type = self._decide(left, right, value, context_diff)
        confidence = self._confidence(left, right, relation_type, value, context_diff)
        review_state = (
            ReviewState.NEEDS_REVIEW
            if relation_type
            in {RelationType.NEEDS_REVIEW, RelationType.INCOMPARABLE}
            else ReviewState.READY
        )
        explanation = self._explanation(
            left,
            right,
            relation_type,
            value,
            context_diff,
        )
        return Relation(
            id=self._relation_id(left.id, right.id),
            fact_a_id=left.id,
            fact_b_id=right.id,
            relation_type=relation_type,
            value_comparison=value.comparison,
            context_diff=context_diff,
            explanation=explanation,
            confidence=confidence,
            rule_version=self.rule_version,
            review_state=review_state,
        )

    def link_facts(
        self,
        facts: Iterable[Fact],
        *,
        cross_document_only: bool = False,
    ) -> tuple[Relation, ...]:
        return tuple(
            self.compare(left, right)
            for left, right in self.candidate_pairs(
                facts,
                cross_document_only=cross_document_only,
            )
        )

    def link_new_fact(
        self,
        store: FactStore,
        fact: Fact,
        *,
        cross_document_only: bool = False,
    ) -> tuple[Relation, ...]:
        """Compare one stored fact only with indexed, relevant existing facts."""
        if not fact.id:
            raise ValueError("The fact must be stored before it can be linked incrementally")
        relations: list[Relation] = []
        for candidate in store.comparison_candidates(fact):
            if cross_document_only and not self._cross_document(fact, candidate):
                continue
            if not self._independent_claims(fact, candidate):
                continue
            relation = self.compare(fact, candidate)
            stored, _ = store.put_relation(relation)
            relations.append(stored)
        return tuple(sorted(relations, key=lambda relation: relation.id))

    def _compare_values(self, left: Fact, right: Fact) -> _ComparedValue:
        if type(left.value) is not type(right.value):
            if isinstance(left.value, NumericValue) and isinstance(right.value, NumericValue):
                return self._compare_numeric(left.value, right.value)
            return _ComparedValue(
                comparison=ValueComparison(
                    left=self._display_value(left.value),
                    right=self._display_value(right.value),
                    agrees=None,
                ),
                comparable=False,
                uncertain=True,
                reason="The facts use different value kinds.",
            )
        if isinstance(left.value, NumericValue):
            return self._compare_numeric(left.value, right.value)
        if isinstance(left.value, CategoricalValue):
            return self._compare_normalized_text(
                left.value.state,
                right.value.state,
                left_display=left.value.state,
                right_display=right.value.state,
                noun="category states",
            )
        if isinstance(left.value, TextValue):
            return self._compare_normalized_text(
                left.value.text,
                right.value.text,
                left_display=left.value.text,
                right_display=right.value.text,
                noun="text values",
            )
        if isinstance(left.value, DateValue):
            if left.value.precision != right.value.precision:
                return _ComparedValue(
                    comparison=ValueComparison(
                        left=left.value.value.isoformat(),
                        right=right.value.value.isoformat(),
                        agrees=None,
                    ),
                    comparable=False,
                    uncertain=True,
                    reason="The dates use different reporting precision.",
                )
            agrees = left.value.value == right.value.value
            return _ComparedValue(
                comparison=ValueComparison(
                    left=left.value.value.isoformat(),
                    right=right.value.value.isoformat(),
                    agrees=agrees,
                ),
                comparable=True,
                reason="The dates are equal." if agrees else "The dates are different.",
            )
        if isinstance(left.value, BooleanValue):
            agrees = left.value.value is right.value.value
            return _ComparedValue(
                comparison=ValueComparison(
                    left=str(left.value.value).lower(),
                    right=str(right.value.value).lower(),
                    agrees=agrees,
                ),
                comparable=True,
                reason=(
                    "The boolean values are equal."
                    if agrees
                    else "The boolean values are different."
                ),
            )
        if isinstance(left.value, IdentifierValue):
            left_scheme = normalize_identifier_scheme(left.value.scheme)
            right_scheme = normalize_identifier_scheme(right.value.scheme)
            left_value = normalize_identifier(left_scheme, left.value.value)
            right_value = normalize_identifier(right_scheme, right.value.value)
            if left_scheme != right_scheme:
                return _ComparedValue(
                    comparison=ValueComparison(
                        left=f"{left_scheme}:{left_value}",
                        right=f"{right_scheme}:{right_value}",
                        agrees=None,
                    ),
                    comparable=False,
                    reason="Identifier schemes differ.",
                )
            agrees = left_value == right_value
            return _ComparedValue(
                comparison=ValueComparison(
                    left=f"{left_scheme}:{left_value}",
                    right=f"{right_scheme}:{right_value}",
                    agrees=agrees,
                ),
                comparable=True,
                reason=(
                    "The normalized identifiers are equal."
                    if agrees
                    else "The normalized identifiers are different."
                ),
            )
        raise TypeError(f"Unsupported fact value type: {type(left.value).__name__}")

    def _compare_numeric(self, left: NumericValue, right: NumericValue) -> _ComparedValue:
        unit_reconciled = False
        compared_left = left
        compared_right = right
        if {left.unit, right.unit} == {"percent", "ratio"}:
            compared_left = self._as_ratio(left)
            compared_right = self._as_ratio(right)
            unit_reconciled = True
        result = compare_numeric_values(compared_left, compared_right)
        return _ComparedValue(
            comparison=ValueComparison(
                left=self._display_value(left),
                right=self._display_value(right),
                agrees=result.agrees,
                difference=result.difference,
                tolerance=result.allowed_difference,
            ),
            comparable=result.comparable,
            reason=result.reason,
            unit_reconciled=unit_reconciled and result.comparable,
            uncertain=(
                not result.comparable
                and (left.unit is None or right.unit is None)
            ),
        )

    @staticmethod
    def _as_ratio(value: NumericValue) -> NumericValue:
        if value.unit != "percent":
            return value.model_copy(update={"currency": None})
        return value.model_copy(
            update={
                "number": value.number / Decimal("100"),
                "unit": "ratio",
                "currency": None,
                "scale": value.scale / Decimal("100"),
            }
        )

    def _context_diff(
        self,
        left: Fact,
        right: Fact,
        *,
        unit_reconciled: bool,
    ) -> list[ContextDifference]:
        values = [
            ("period", self._period_value(left), self._period_value(right)),
            ("scope", self._plain(left.context.scope), self._plain(right.context.scope)),
            ("basis", self._plain(left.context.basis), self._plain(right.context.basis)),
            ("as_of", self._date_value(left.context.as_of), self._date_value(right.context.as_of)),
            ("unit", self._unit_value(left), self._unit_value(right)),
            (
                "geography",
                self._plain(left.context.geography),
                self._plain(right.context.geography),
            ),
            (
                "publisher",
                self._plain(left.context.publisher),
                self._plain(right.context.publisher),
            ),
        ]
        differences: list[ContextDifference] = []
        for field, left_value, right_value in values:
            if left_value == right_value:
                continue
            both_known = left_value is not None and right_value is not None
            explanatory = field in _EXPLANATORY_CONTEXT_FIELDS and both_known
            if field == "unit" and unit_reconciled:
                explanatory = True
            if field == "period" and explanatory and not self._periods_comparable(left, right):
                # A quarter set beside a full year is not the same measure observed in a
                # different period; the two cover different amounts of trading. Saying
                # "different period" would dress an apples-to-oranges pair up as an
                # explained one, so it stays an unresolved difference for review.
                explanatory = False
            differences.append(
                ContextDifference(
                    field=field,
                    left=left_value,
                    right=right_value,
                    explains_difference=explanatory,
                )
            )
        return differences

    def _decide(
        self,
        left: Fact,
        right: Fact,
        value: _ComparedValue,
        context_diff: list[ContextDifference],
    ) -> RelationType:
        if ReviewState.REJECTED in {left.review_state, right.review_state}:
            return RelationType.NEEDS_REVIEW
        if ReviewState.NEEDS_REVIEW in {left.review_state, right.review_state}:
            return RelationType.NEEDS_REVIEW
        if (
            isinstance(left.value, NumericValue)
            and not self._has_material_context(left)
            and not self._has_material_context(right)
        ):
            return RelationType.NEEDS_REVIEW

        material = [item for item in context_diff if item.field != "publisher"]
        unresolved = [item for item in material if not item.explains_difference]
        explanatory = [item for item in material if item.explains_difference]
        if value.unit_reconciled and not any(item.field == "unit" for item in explanatory):
            explanatory.append(
                ContextDifference(
                    field="unit",
                    left=self._unit_value(left),
                    right=self._unit_value(right),
                    explains_difference=True,
                )
            )
        if value.uncertain:
            return RelationType.NEEDS_REVIEW
        if not value.comparable:
            return RelationType.INCOMPARABLE

        # An unresolved context difference means two different things depending on
        # whether both sources actually said something. When both state a value and the
        # values clash, the facts may not be about the same observation at all. When
        # only one source states the field, the other is merely silent.
        conflicting = [
            item for item in unresolved if item.left is not None and item.right is not None
        ]
        unstated = [item for item in unresolved if item.left is None or item.right is None]
        if conflicting:
            return RelationType.NEEDS_REVIEW

        fields = {item.field for item in explanatory}
        if fields:
            # One axis explaining the difference is a usable answer even when another
            # field is simply unstated. Almost no filing repeats its country on every
            # sentence, and refusing to reconcile over that kind of silence buries
            # correct findings in the review queue. The gaps stay recorded in the
            # context diff and pull the confidence down, so a reviewer still sees them.
            if len(fields) == 1:
                return _RECONCILIATION_TYPES.get(
                    next(iter(fields)), RelationType.INCOMPARABLE
                )
            return RelationType.INCOMPARABLE

        if value.comparison.agrees is True:
            # Silence about scope or basis does not weaken an agreement. Two documents
            # reporting the same figure for the same measure, subject, and period still
            # corroborate each other when one of them never named its reporting scope;
            # the matching values are themselves evidence they describe one quantity.
            return RelationType.CORROBORATES
        if value.comparison.agrees is False:
            # Disagreement is the opposite case: an unstated field is exactly the kind
            # of thing that would explain the gap, so calling it a contradiction would
            # overreach.
            return (
                RelationType.NEEDS_REVIEW if unstated else RelationType.CONTRADICTS
            )
        return RelationType.NEEDS_REVIEW

    def _explanation(
        self,
        left: Fact,
        right: Fact,
        relation_type: RelationType,
        value: _ComparedValue,
        context_diff: list[ContextDifference],
    ) -> str:
        measure = left.predicate.display_name
        left_value = value.comparison.left
        right_value = value.comparison.right
        if relation_type is RelationType.CORROBORATES:
            detail = (
                f"agree within a tolerance of {value.comparison.tolerance}"
                if value.comparison.tolerance is not None
                else "agree after normalization"
            )
            return (
                f"{measure}: {left_value} and {right_value} {detail}; "
                "material context is compatible."
            )
        if relation_type is RelationType.CONTRADICTS:
            difference = (
                f" by {value.comparison.difference}"
                if value.comparison.difference is not None
                else ""
            )
            return (
                f"{measure}: {left_value} and {right_value} differ{difference} while period, "
                "scope, basis, as-of date, unit, and geography are compatible."
            )
        if relation_type in _RECONCILIATION_TYPES.values():
            field = next(
                item.field
                for item in context_diff
                if _RECONCILIATION_TYPES.get(item.field) is relation_type
            )
            difference = next(item for item in context_diff if item.field == field)
            explanation = (
                f"{measure}: {left_value} and {right_value} refer to different "
                f"{field.replace('_', ' ')} "
                f"contexts ({self._format_context(difference.left)} versus "
                f"{self._format_context(difference.right)}), which explains why they should not "
                "be treated as the same observation."
            )
            # Name whatever only one side stated, so the reader can judge how much of
            # the context this reconciliation actually rests on.
            gaps = sorted(
                item.field.replace("_", " ")
                for item in context_diff
                if item.field != "publisher"
                and not item.explains_difference
                and (item.left is None or item.right is None)
            )
            if gaps:
                explanation += (
                    f" Only one source states its {', '.join(gaps)}, "
                    "so that part of the context is unconfirmed."
                )
            return explanation
        if relation_type is RelationType.INCOMPARABLE:
            fields = [
                item.field.replace("_", " ")
                for item in context_diff
                if item.field != "publisher"
            ]
            reason = ", ".join(fields) if fields else value.reason
            return f"{measure}: the facts are not safely comparable because {reason}."
        # Say what actually stopped us. A doubtful source fact outranks a context gap:
        # blaming a missing geography when the real problem is an unverified subject
        # sends a reviewer to fix the wrong thing.
        doubtful = [
            fact
            for fact in (left, right)
            if fact.review_state in {ReviewState.NEEDS_REVIEW, ReviewState.REJECTED}
        ]
        if doubtful:
            reason = doubtful[0].warnings[0] if doubtful[0].warnings else "it needs review"
            return (
                f"{measure}: comparison needs review because a source fact is not "
                f"settled - {reason}"
            )
        unresolved = [
            item.field.replace("_", " ")
            for item in context_diff
            if item.field != "publisher" and not item.explains_difference
        ]
        if unresolved:
            return (
                f"{measure}: comparison needs review because "
                f"{', '.join(unresolved)} context is missing or uncertain."
            )
        if (
            isinstance(left.value, NumericValue)
            and not self._has_material_context(left)
            and not self._has_material_context(right)
        ):
            return (
                f"{measure}: comparison needs review because neither numeric fact "
                "has material context."
            )
        if ReviewState.NEEDS_REVIEW in {left.review_state, right.review_state}:
            return (
                f"{measure}: comparison needs review because at least one "
                "source fact needs review."
            )
        return f"{measure}: comparison needs review. {value.reason}"

    def _confidence(
        self,
        left: Fact,
        right: Fact,
        relation_type: RelationType,
        value: _ComparedValue,
        context_diff: list[ContextDifference] = (),
    ) -> float:
        source_confidence = min(
            left.extraction_confidence,
            left.normalization_confidence,
            right.extraction_confidence,
            right.normalization_confidence,
        )
        multiplier = {
            RelationType.CORROBORATES: 0.98,
            RelationType.CONTRADICTS: 0.96,
            RelationType.RECONCILED_BY_PERIOD: 0.92,
            RelationType.RECONCILED_BY_SCOPE: 0.92,
            RelationType.RECONCILED_BY_BASIS: 0.92,
            RelationType.RECONCILED_BY_AS_OF: 0.92,
            RelationType.RECONCILED_BY_UNIT: 0.92,
            RelationType.INCOMPARABLE: 0.72,
            RelationType.NEEDS_REVIEW: 0.55,
        }.get(relation_type, 0.6)
        if not value.comparable:
            multiplier = min(multiplier, 0.7)
        # Every field only one source stated is a piece of context we could not confirm,
        # so a verdict resting on partial context should not read as fully certain.
        gaps = sum(
            1
            for item in context_diff
            if item.field != "publisher"
            and not item.explains_difference
            and (item.left is None or item.right is None)
        )
        if gaps:
            multiplier *= max(0.75, 0.92**gaps)
        return round(source_confidence * multiplier, 6)

    @staticmethod
    def _compare_normalized_text(
        left: str,
        right: str,
        *,
        left_display: str,
        right_display: str,
        noun: str,
    ) -> _ComparedValue:
        agrees = _normalized_text(left) == _normalized_text(right)
        return _ComparedValue(
            comparison=ValueComparison(
                left=left_display,
                right=right_display,
                agrees=agrees,
            ),
            comparable=True,
            reason=f"The normalized {noun} are {'equal' if agrees else 'different'}.",
        )

    @staticmethod
    def _periods_comparable(left: Fact, right: Fact) -> bool:
        """True when two reporting periods cover a similar span of time.

        Only meaningful when both periods carry dates. Labels alone ("FY24" versus
        "Q4 FY24") are left to the caller's other checks.
        """
        first, second = left.context.period, right.context.period
        if first is None or second is None:
            return True
        if not (first.start and first.end and second.start and second.end):
            return True
        left_days = (first.end - first.start).days
        right_days = (second.end - second.start).days
        if not left_days or not right_days:
            return True
        longer, shorter = max(left_days, right_days), min(left_days, right_days)
        return longer <= shorter * 2

    @staticmethod
    def _period_value(fact: Fact) -> dict[str, str | None] | None:
        period = fact.context.period
        if period is None:
            return None
        if period.start or period.end:
            return {
                "start": period.start.isoformat() if period.start else None,
                "end": period.end.isoformat() if period.end else None,
            }
        return {
            "label": period.label,
        }

    @staticmethod
    def _has_material_context(fact: Fact) -> bool:
        return any(
            (
                fact.context.period,
                fact.context.scope,
                fact.context.basis,
                fact.context.as_of,
                fact.context.geography,
            )
        )

    @staticmethod
    def _unit_value(fact: Fact) -> dict[str, str | None] | None:
        if isinstance(fact.value, NumericValue):
            if fact.value.unit is None and fact.value.currency is None:
                return None
            return {"unit": fact.value.unit, "currency": fact.value.currency}
        if isinstance(fact.value, IdentifierValue):
            return {
                "unit": "identifier",
                "currency": normalize_identifier_scheme(fact.value.scheme),
            }
        return None

    @staticmethod
    def _display_value(value: Any) -> str:
        if isinstance(value, NumericValue):
            return f"{value.number} {value.currency or value.unit or ''}".strip()
        if isinstance(value, CategoricalValue):
            return value.state
        if isinstance(value, TextValue):
            return value.text
        if isinstance(value, DateValue):
            return value.value.isoformat()
        if isinstance(value, BooleanValue):
            return str(value.value).lower()
        if isinstance(value, IdentifierValue):
            return f"{value.scheme}:{value.value}"
        return str(value)

    @staticmethod
    def _plain(value: str | None) -> str | None:
        return re.sub(r"\s+", " ", value.casefold()).strip() if value else None

    @staticmethod
    def _date_value(value: date | None) -> str | None:
        return value.isoformat() if value else None

    @staticmethod
    def _format_context(value: Any) -> str:
        if isinstance(value, dict):
            return "/".join(str(item) for item in value.values() if item is not None)
        return "missing" if value is None else str(value)

    @staticmethod
    def _cross_document(left: Fact, right: Fact) -> bool:
        left_documents = {evidence.document_id for evidence in left.evidence}
        right_documents = {evidence.document_id for evidence in right.evidence}
        return bool(left_documents - right_documents or right_documents - left_documents)

    @staticmethod
    def _independent_claims(left: Fact, right: Fact) -> bool:
        """True when two facts are separate assertions, not neighbouring table cells.

        A table row such as "% margin" repeated down a column yields many cells that
        share one predicate label and one passage while describing different measures
        or different columns. The row label alone cannot tell them apart, so treating
        them as rival claims about a single quantity invents contradictions that the
        document never made. Requiring separate passages keeps genuine pairs such as
        standalone versus consolidated revenue, which are stated in separate
        sentences, and drops the sibling-cell noise.
        """
        left_passages = {evidence.passage_id for evidence in left.evidence}
        right_passages = {evidence.passage_id for evidence in right.evidence}
        return not (left_passages & right_passages)

    @staticmethod
    def _canonical_pair(left: Fact, right: Fact) -> tuple[Fact, Fact]:
        if not left.id or not right.id:
            raise ValueError("Both facts need stable IDs before comparison")
        return (left, right) if left.id < right.id else (right, left)

    @staticmethod
    def _validate_pair(left: Fact, right: Fact) -> None:
        if left.id == right.id:
            raise ValueError("A fact cannot be compared with itself")
        if not left.subject.id or left.subject.id != right.subject.id:
            raise ValueError("Facts need the same resolved subject before comparison")
        if left.predicate.key != right.predicate.key:
            raise ValueError("Facts need the same predicate before comparison")

    def _relation_id(self, left_id: str, right_id: str) -> str:
        payload = json.dumps(
            [left_id, right_id, self.rule_version],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"relation_{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def _normalized_text(value: str) -> str:
    source = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", source.casefold()).strip()


__all__ = ["FactLinker", "RULE_VERSION"]
