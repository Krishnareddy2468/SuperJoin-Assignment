"""Conservative entity and predicate resolution across extracted documents.

Identifiers settle identity when they are available. Names are useful fallback
signals, but they never override conflicting DINs, CINs, tickers, or registration
numbers. Ambiguous cases remain separate and are marked for review.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Literal

from rapidfuzz import fuzz

from factlayer.normalize import PredicateRegistry
from factlayer.schema import (
    Entity,
    EntityReference,
    Fact,
    IdentifierValue,
    Predicate,
    ReviewState,
)


_HONORIFICS = re.compile(r"^(?:(?:mr|mrs|ms|miss|dr|prof)\.?\s+)+", re.IGNORECASE)
_COMPANY_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "limited",
    "ltd",
    "llp",
    "plc",
    "private",
    "pvt",
}
_IDENTIFIER_PRIORITY = (
    "din",
    "cin",
    "lei",
    "ticker",
    "registration",
    "registration_number",
)


class ResolutionMethod(str, Enum):
    IDENTIFIER = "identifier"
    EXACT_NAME = "exact_name"
    FUZZY_NAME = "fuzzy_name"
    NEW_ENTITY = "new_entity"
    NEEDS_REVIEW = "needs_review"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class EntityMention:
    name: str
    entity_type: str
    identifiers: dict[str, str] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    review_state: ReviewState = ReviewState.READY

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("An entity mention needs a name")
        if not self.entity_type.strip():
            raise ValueError("An entity mention needs an entity type")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Entity mention confidence must be between zero and one")


@dataclass(frozen=True)
class AliasAttribution:
    alias: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class EntityResolution:
    entity: Entity
    method: ResolutionMethod
    confidence: float
    reason: str
    matched_entity_id: str | None = None
    candidate_entity_ids: tuple[str, ...] = field(default_factory=tuple)
    aliases_added: tuple[AliasAttribution, ...] = field(default_factory=tuple)

    @property
    def needs_review(self) -> bool:
        return self.entity.review_state is ReviewState.NEEDS_REVIEW


class EntityResolver:
    """Resolve mentions against a growing, in-memory entity catalogue."""

    def __init__(
        self,
        entities: Iterable[Entity] = (),
        *,
        fuzzy_auto_threshold: float = 97,
        fuzzy_review_threshold: float = 88,
        ambiguity_margin: float = 4,
        predicate_registry: PredicateRegistry | None = None,
    ):
        if not 0 <= fuzzy_review_threshold <= fuzzy_auto_threshold <= 100:
            raise ValueError("Fuzzy thresholds must satisfy 0 <= review <= auto <= 100")
        if ambiguity_margin < 0:
            raise ValueError("The fuzzy ambiguity margin cannot be negative")
        self.fuzzy_auto_threshold = fuzzy_auto_threshold
        self.fuzzy_review_threshold = fuzzy_review_threshold
        self.ambiguity_margin = ambiguity_margin
        self.predicates = predicate_registry or PredicateRegistry()
        self._entities: dict[str, Entity] = {}
        self._identifier_index: dict[tuple[str, str], set[str]] = {}
        self._name_index: dict[tuple[str, str], set[str]] = {}
        self._alias_sources: dict[tuple[str, str], set[str]] = {}
        for entity in entities:
            self.add(entity)

    @classmethod
    def from_store(cls, store, **kwargs) -> "EntityResolver":
        """Load the current catalogue without coupling resolution to SQLite internals."""
        return cls(store.list_entities(), **kwargs)

    def add(self, entity: Entity) -> Entity:
        stored = entity if entity.id else entity.model_copy(update={"id": self._entity_id(entity)})
        previous = self._entities.get(stored.id)
        if previous:
            self._remove_indexes(previous)
        self._entities[stored.id] = stored
        self._index(stored)
        return stored

    def all(self) -> tuple[Entity, ...]:
        return tuple(
            sorted(
                self._entities.values(),
                key=lambda entity: (entity.canonical_name.casefold(), entity.id or ""),
            )
        )

    def alias_evidence(self, entity_id: str, alias: str) -> tuple[str, ...]:
        """Return the source evidence recorded when an alias was learned."""
        key = (entity_id, normalize_entity_name(alias))
        return tuple(sorted(self._alias_sources.get(key, set())))

    def register_predicate(
        self,
        label: str,
        *,
        value_kind: Literal["number", "text", "category", "date", "boolean", "identifier"],
        aliases: Iterable[str] = (),
    ) -> Predicate:
        """Register a document-discovered predicate without changing a schema enum."""
        return self.predicates.register(
            label,
            value_kind=value_kind,
            aliases=tuple(aliases),
        )

    def resolve(self, mention: EntityMention) -> EntityResolution:
        prepared = self._prepare_mention(mention)
        identifier_matches = self._identifier_matches(prepared.identifiers)
        if len(identifier_matches) > 1:
            return self._review_entity(
                prepared,
                ResolutionMethod.CONFLICT,
                "The supplied identifiers point to different existing entities, "
                "so they were not merged.",
                identifier_matches,
            )
        if len(identifier_matches) == 1:
            matched = self._entities[next(iter(identifier_matches))]
            conflicts = identifier_conflicts(matched.identifiers, prepared.identifiers)
            names_differ = normalize_entity_name(
                matched.canonical_name,
                entity_type=matched.entity_type,
            ) != normalize_entity_name(
                prepared.name,
                entity_type=prepared.entity_type,
            )
            can_upgrade_identity = (
                matched.review_state is ReviewState.NEEDS_REVIEW
                and prepared.review_state is ReviewState.READY
                and names_differ
            )
            if conflicts or (
                not self._types_compatible(matched.entity_type, prepared.entity_type)
                and not can_upgrade_identity
            ):
                return self._review_entity(
                    prepared,
                    ResolutionMethod.CONFLICT,
                    "An identifier matched, but another identifier or entity type conflicted.",
                    {matched.id},
                )
            return self._merge(
                matched,
                prepared,
                ResolutionMethod.IDENTIFIER,
                0.99,
                "A stable identifier matched an existing entity.",
                replace_identity=can_upgrade_identity,
            )

        exact_matches = self._exact_name_matches(prepared)
        if exact_matches:
            compatible, conflicting = self._partition_identifier_compatibility(
                exact_matches,
                prepared.identifiers,
            )
            if len(compatible) == 1 and not conflicting:
                matched = self._entities[next(iter(compatible))]
                return self._merge(
                    matched,
                    prepared,
                    ResolutionMethod.EXACT_NAME,
                    0.96,
                    "The normalized name and entity type matched exactly.",
                )
            candidates = compatible | conflicting
            return self._review_entity(
                prepared,
                ResolutionMethod.CONFLICT if conflicting else ResolutionMethod.NEEDS_REVIEW,
                "The normalized name was ambiguous or carried a conflicting identifier.",
                candidates,
            )

        fuzzy_matches = self._fuzzy_matches(prepared)
        if fuzzy_matches:
            top_id, top_score = fuzzy_matches[0]
            close_ids = {
                entity_id
                for entity_id, score in fuzzy_matches
                if top_score - score < self.ambiguity_margin
            }
            conflicting_ids = {
                entity_id
                for entity_id, score in fuzzy_matches
                if score >= self.fuzzy_review_threshold
                and identifier_conflicts(
                    self._entities[entity_id].identifiers,
                    prepared.identifiers,
                )
            }
            if conflicting_ids:
                return self._review_entity(
                    prepared,
                    ResolutionMethod.CONFLICT,
                    "A similar name belonged to an entity with a conflicting identifier.",
                    conflicting_ids,
                )
            if (
                top_score >= self.fuzzy_auto_threshold
                and len(close_ids) == 1
            ):
                return self._merge(
                    self._entities[top_id],
                    prepared,
                    ResolutionMethod.FUZZY_NAME,
                    min(0.95, top_score / 100),
                    f"The name matched conservatively with a similarity score of {top_score:.1f}.",
                )
            if top_score >= self.fuzzy_review_threshold:
                return self._review_entity(
                    prepared,
                    ResolutionMethod.NEEDS_REVIEW,
                    f"The closest name match scored {top_score:.1f}, which needs human review.",
                    close_ids,
                )

        # An entity's review state records whether its *identity* is unsettled, which
        # is not the same question as whether the fact that mentioned it was solid.
        # Nothing competed with this name, so the identity is settled even when the
        # mention arrived from a shaky extraction. Copying the mention's review state
        # here used to mark the entity for review permanently, and _merge then handed
        # that flag to every later fact about the same subject — one weak fact was
        # enough to push a whole document's facts to needs_review.
        entity = self.add(self._new_entity(prepared, review=False))
        return EntityResolution(
            entity=entity,
            method=ResolutionMethod.NEW_ENTITY,
            confidence=entity.confidence,
            reason="No safe identifier or name match was found, so a separate entity was created.",
        )

    def resolve_fact(self, fact: Fact) -> tuple[Fact, EntityResolution]:
        identifiers: dict[str, str] = {}
        if fact.subject.id and ":" in fact.subject.id:
            scheme, value = fact.subject.id.split(":", 1)
            if scheme.casefold() in _IDENTIFIER_PRIORITY:
                identifiers[scheme] = value
        if isinstance(fact.value, IdentifierValue):
            identifiers[fact.value.scheme] = fact.value.value
        evidence_ids = tuple(
            evidence.id for evidence in fact.evidence if evidence.id is not None
        )
        mention = EntityMention(
            name=fact.subject.canonical_name,
            entity_type=fact.subject.entity_type or "unknown",
            identifiers=identifiers,
            evidence_ids=evidence_ids,
            confidence=fact.normalization_confidence,
            review_state=fact.review_state,
        )
        resolution = self.resolve(mention)
        warnings = list(fact.warnings)
        review_state = fact.review_state
        if resolution.needs_review:
            warnings.append(resolution.reason)
            review_state = ReviewState.NEEDS_REVIEW
        resolved_fact = fact.model_copy(
            update={
                "subject": EntityReference(
                    id=resolution.entity.id,
                    canonical_name=resolution.entity.canonical_name,
                    entity_type=resolution.entity.entity_type,
                ),
                "normalization_confidence": min(
                    fact.normalization_confidence,
                    resolution.confidence,
                ),
                "review_state": review_state,
                "warnings": warnings,
            }
        )
        return resolved_fact, resolution

    def _prepare_mention(self, mention: EntityMention) -> EntityMention:
        identifiers = {
            normalize_identifier_scheme(scheme): normalize_identifier(scheme, value)
            for scheme, value in mention.identifiers.items()
            if scheme.strip() and value.strip()
        }
        entity_type = normalize_entity_type(mention.entity_type)
        if "din" in identifiers:
            entity_type = "person"
        elif "cin" in identifiers:
            entity_type = "company"
        return EntityMention(
            name=clean_display_name(mention.name),
            entity_type=entity_type,
            identifiers=identifiers,
            evidence_ids=tuple(dict.fromkeys(mention.evidence_ids)),
            confidence=mention.confidence,
            review_state=mention.review_state,
        )

    def _identifier_matches(self, identifiers: dict[str, str]) -> set[str]:
        matches: set[str] = set()
        for item in identifiers.items():
            matches.update(self._identifier_index.get(item, set()))
        return matches

    def _exact_name_matches(self, mention: EntityMention) -> set[str]:
        key = (
            mention.entity_type,
            normalize_entity_name(mention.name, entity_type=mention.entity_type),
        )
        return set(self._name_index.get(key, set()))

    def _fuzzy_matches(self, mention: EntityMention) -> list[tuple[str, float]]:
        needle = normalize_entity_name(mention.name, entity_type=mention.entity_type)
        if len(needle) < 5:
            return []
        matches: list[tuple[str, float]] = []
        for entity in self._entities.values():
            if not self._types_compatible(entity.entity_type, mention.entity_type):
                continue
            names = [entity.canonical_name, *entity.aliases]
            score = max(
                max(
                    fuzz.ratio(
                        needle,
                        normalize_entity_name(name, entity_type=mention.entity_type),
                    ),
                    fuzz.token_sort_ratio(
                        needle,
                        normalize_entity_name(name, entity_type=mention.entity_type),
                    ),
                )
                for name in names
            )
            if score >= self.fuzzy_review_threshold:
                matches.append((entity.id, float(score)))
        return sorted(matches, key=lambda item: (-item[1], item[0]))

    def _partition_identifier_compatibility(
        self,
        entity_ids: set[str],
        identifiers: dict[str, str],
    ) -> tuple[set[str], set[str]]:
        compatible: set[str] = set()
        conflicting: set[str] = set()
        for entity_id in entity_ids:
            entity = self._entities[entity_id]
            if identifier_conflicts(entity.identifiers, identifiers):
                conflicting.add(entity_id)
            else:
                compatible.add(entity_id)
        return compatible, conflicting

    def _merge(
        self,
        entity: Entity,
        mention: EntityMention,
        method: ResolutionMethod,
        confidence: float,
        reason: str,
        *,
        replace_identity: bool = False,
    ) -> EntityResolution:
        aliases = list(entity.aliases)
        aliases_added: list[AliasAttribution] = []
        canonical_name = mention.name if replace_identity else entity.canonical_name
        entity_type = mention.entity_type if replace_identity else entity.entity_type
        known_names = {canonical_name.casefold(), *(alias.casefold() for alias in aliases)}
        if (
            not replace_identity
            and mention.name.casefold() not in known_names
            and mention.evidence_ids
        ):
            aliases.append(mention.name)
            aliases_added.append(AliasAttribution(mention.name, mention.evidence_ids))
        identifiers = {**entity.identifiers, **mention.identifiers}
        merged = entity.model_copy(
            update={
                "canonical_name": canonical_name,
                "entity_type": entity_type,
                "identifiers": identifiers,
                "aliases": aliases,
                "confidence": (
                    min(mention.confidence, confidence)
                    if replace_identity
                    else min(entity.confidence, mention.confidence, confidence)
                ),
                "review_state": (
                    mention.review_state if replace_identity else entity.review_state
                ),
            }
        )
        self.add(merged)
        for attribution in aliases_added:
            key = (merged.id, normalize_entity_name(attribution.alias))
            self._alias_sources.setdefault(key, set()).update(attribution.evidence_ids)
        return EntityResolution(
            entity=merged,
            method=method,
            confidence=min(mention.confidence, confidence),
            reason=reason,
            matched_entity_id=merged.id,
            aliases_added=tuple(aliases_added),
        )

    def _review_entity(
        self,
        mention: EntityMention,
        method: ResolutionMethod,
        reason: str,
        candidates: set[str | None],
    ) -> EntityResolution:
        entity = self.add(self._new_entity(mention, review=True, force_review_id=True))
        candidate_ids = tuple(sorted(item for item in candidates if item))
        return EntityResolution(
            entity=entity,
            method=method,
            confidence=min(mention.confidence, 0.6),
            reason=reason,
            candidate_entity_ids=candidate_ids,
        )

    def _new_entity(
        self,
        mention: EntityMention,
        *,
        review: bool,
        force_review_id: bool = False,
    ) -> Entity:
        entity = Entity(
            id=None,
            canonical_name=mention.name,
            entity_type=mention.entity_type,
            identifiers=mention.identifiers,
            confidence=min(mention.confidence, 0.6 if review else 0.9),
            review_state=ReviewState.NEEDS_REVIEW if review else ReviewState.READY,
        )
        if force_review_id:
            entity_id = f"entity-review:{self._short_hash(entity.model_dump(mode='json'))}"
        else:
            entity_id = self._entity_id(entity)
        return entity.model_copy(update={"id": entity_id})

    def _entity_id(self, entity: Entity) -> str:
        for scheme in _IDENTIFIER_PRIORITY:
            if value := entity.identifiers.get(scheme):
                return f"{scheme}:{value}"
        payload = {
            "name": normalize_entity_name(
                entity.canonical_name,
                entity_type=entity.entity_type,
            ),
            "type": normalize_entity_type(entity.entity_type),
        }
        return f"entity:{self._short_hash(payload)}"

    def _index(self, entity: Entity) -> None:
        entity_type = normalize_entity_type(entity.entity_type)
        for scheme, value in entity.identifiers.items():
            item = (
                normalize_identifier_scheme(scheme),
                normalize_identifier(scheme, value),
            )
            self._identifier_index.setdefault(item, set()).add(entity.id)
        for name in (entity.canonical_name, *entity.aliases):
            key = (
                entity_type,
                normalize_entity_name(name, entity_type=entity_type),
            )
            self._name_index.setdefault(key, set()).add(entity.id)

    def _remove_indexes(self, entity: Entity) -> None:
        for values in (*self._identifier_index.values(), *self._name_index.values()):
            values.discard(entity.id)

    @staticmethod
    def _types_compatible(left: str, right: str) -> bool:
        left_type = normalize_entity_type(left)
        right_type = normalize_entity_type(right)
        return left_type == right_type or "unknown" in {left_type, right_type}

    @staticmethod
    def _short_hash(payload) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:24]


def clean_display_name(name: str) -> str:
    cleaned = unicodedata.normalize("NFKC", name).strip()
    cleaned = _HONORIFICS.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")


def normalize_entity_name(name: str, *, entity_type: str | None = None) -> str:
    source = clean_display_name(name)
    source = unicodedata.normalize("NFKD", source).encode("ascii", "ignore").decode()
    source = source.casefold().replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+", source)
    if normalize_entity_type(entity_type or "unknown") == "company":
        while tokens and tokens[-1] in _COMPANY_SUFFIXES:
            tokens.pop()
    return " ".join(tokens)


def normalize_entity_type(entity_type: str) -> str:
    key = re.sub(r"[^a-z]+", "_", entity_type.casefold()).strip("_")
    aliases = {
        "business": "company",
        "corporation": "company",
        "organisation": "organization",
        "director": "person",
        "individual": "person",
    }
    return aliases.get(key, key or "unknown")


def normalize_identifier_scheme(scheme: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", scheme.casefold()).strip("_")
    aliases = {
        "director_identification_number": "din",
        "corporate_identity_number": "cin",
        "stock_symbol": "ticker",
        "registration_no": "registration_number",
    }
    return aliases.get(key, key)


def normalize_identifier(scheme: str, value: str) -> str:
    normalized_scheme = normalize_identifier_scheme(scheme)
    source = unicodedata.normalize("NFKC", value).strip().upper()
    if normalized_scheme in {"din", "cin", "lei"}:
        return re.sub(r"[^A-Z0-9]", "", source)
    if normalized_scheme in {"ticker", "registration", "registration_number"}:
        return re.sub(r"\s+", "", source)
    return re.sub(r"\s+", " ", source)


def identifier_conflicts(left: dict[str, str], right: dict[str, str]) -> tuple[str, ...]:
    normalized_left = {
        normalize_identifier_scheme(scheme): normalize_identifier(scheme, value)
        for scheme, value in left.items()
    }
    normalized_right = {
        normalize_identifier_scheme(scheme): normalize_identifier(scheme, value)
        for scheme, value in right.items()
    }
    return tuple(
        sorted(
            scheme
            for scheme in normalized_left.keys() & normalized_right.keys()
            if normalized_left[scheme] != normalized_right[scheme]
        )
    )


__all__ = [
    "AliasAttribution",
    "EntityMention",
    "EntityResolution",
    "EntityResolver",
    "ResolutionMethod",
    "clean_display_name",
    "identifier_conflicts",
    "normalize_entity_name",
    "normalize_entity_type",
    "normalize_identifier",
    "normalize_identifier_scheme",
]
