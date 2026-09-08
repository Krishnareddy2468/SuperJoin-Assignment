from pathlib import Path

from factlayer.entities import (
    EntityMention,
    EntityResolver,
    ResolutionMethod,
    identifier_conflicts,
    normalize_entity_name,
    normalize_identifier,
)
from factlayer.extract_rules import DeterministicExtractor
from factlayer.ingest import PdfIngestor
from factlayer.schema import Entity, EntityReference, ReviewState


PROJECT_ROOT = Path(__file__).parents[1]
DELHIVERY_DATA = PROJECT_ROOT / "starter-datasets" / "delhivery"
DELHIVERY = EntityReference(
    id="company:delhivery",
    canonical_name="Delhivery Limited",
    entity_type="company",
)


def test_names_remove_formatting_noise_without_losing_display_names() -> None:
    assert normalize_entity_name("Mr. Suvir Suren Sujan", entity_type="person") == (
        "suvir suren sujan"
    )
    assert normalize_entity_name("Delhivery Pvt. Ltd.", entity_type="company") == "delhivery"
    assert normalize_entity_name("DELHIVERY LIMITED", entity_type="company") == "delhivery"


def test_identifier_normalization_preserves_leading_zeroes() -> None:
    assert normalize_identifier("DIN", " 0117 3669 ") == "01173669"
    assert normalize_identifier("CIN", "u63090dl2011plc221234") == (
        "U63090DL2011PLC221234"
    )


def test_din_match_wins_and_records_an_evidence_backed_alias() -> None:
    existing = Entity(
        id="din:01173669",
        canonical_name="Suvir Suren Sujan",
        entity_type="person",
        identifiers={"din": "01173669"},
    )
    resolver = EntityResolver([existing])

    result = resolver.resolve(
        EntityMention(
            name="Suvir S. Sujan",
            entity_type="person",
            identifiers={"Director Identification Number": "01173669"},
            evidence_ids=("evidence-prospectus",),
        )
    )

    assert result.method is ResolutionMethod.IDENTIFIER
    assert result.entity.id == existing.id
    assert result.entity.aliases == ["Suvir S. Sujan"]
    assert result.aliases_added[0].evidence_ids == ("evidence-prospectus",)
    assert resolver.alias_evidence(existing.id, "Suvir S. Sujan") == (
        "evidence-prospectus",
    )


def test_alias_without_source_evidence_is_not_added() -> None:
    existing = Entity(
        id="din:01173669",
        canonical_name="Suvir Suren Sujan",
        entity_type="person",
        identifiers={"din": "01173669"},
    )

    result = EntityResolver([existing]).resolve(
        EntityMention(
            name="S. S. Sujan",
            entity_type="person",
            identifiers={"din": "01173669"},
        )
    )

    assert result.entity.aliases == []
    assert result.aliases_added == ()


def test_company_suffix_variants_match_before_fuzzy_resolution() -> None:
    resolver = EntityResolver(
        [
            Entity(
                id="company:delhivery",
                canonical_name="Delhivery Limited",
                entity_type="company",
            )
        ]
    )

    result = resolver.resolve(
        EntityMention(
            name="Delhivery Pvt. Ltd.",
            entity_type="company",
            identifiers={"cin": "U63090DL2011PLC221234"},
            evidence_ids=("evidence-cin",),
        )
    )

    assert result.method is ResolutionMethod.EXACT_NAME
    assert result.entity.id == "company:delhivery"
    assert result.entity.identifiers["cin"] == "U63090DL2011PLC221234"


def test_same_name_with_different_dins_stays_separate_for_review() -> None:
    existing = Entity(
        id="din:01173669",
        canonical_name="Suvir Suren Sujan",
        entity_type="person",
        identifiers={"din": "01173669"},
    )
    resolver = EntityResolver([existing])

    result = resolver.resolve(
        EntityMention(
            name="Suvir Suren Sujan",
            entity_type="person",
            identifiers={"din": "99999999"},
            evidence_ids=("evidence-conflict",),
        )
    )

    assert result.method is ResolutionMethod.CONFLICT
    assert result.entity.id != existing.id
    assert result.entity.review_state is ReviewState.NEEDS_REVIEW
    assert result.candidate_entity_ids == (existing.id,)
    assert identifier_conflicts(existing.identifiers, result.entity.identifiers) == ("din",)


def test_similar_names_cannot_override_conflicting_identifiers() -> None:
    existing = Entity(
        id="din:01173669",
        canonical_name="Suvir Suren Sujan",
        entity_type="person",
        identifiers={"din": "01173669"},
    )

    result = EntityResolver([existing]).resolve(
        EntityMention(
            name="Suvir Suren Sujam",
            entity_type="person",
            identifiers={"din": "99999999"},
        )
    )

    assert result.method is ResolutionMethod.CONFLICT
    assert result.entity.id != existing.id
    assert result.needs_review is True

    repeated = EntityResolver([existing, result.entity]).resolve(
        EntityMention(
            name="Suvir Suren Sujam",
            entity_type="person",
            identifiers={"din": "99999999"},
        )
    )
    assert repeated.entity.id == result.entity.id
    assert repeated.needs_review is True


def test_high_confidence_fuzzy_match_adds_a_sourced_alias() -> None:
    existing = Entity(
        id="organization:imf",
        canonical_name="International Monetary Fund",
        entity_type="organization",
    )
    resolver = EntityResolver([existing])

    result = resolver.resolve(
        EntityMention(
            name="International Monetary Funds",
            entity_type="organization",
            evidence_ids=("evidence-imf",),
        )
    )

    assert result.method is ResolutionMethod.FUZZY_NAME
    assert result.entity.id == existing.id
    assert "International Monetary Funds" in result.entity.aliases


def test_ambiguous_fuzzy_match_creates_a_reviewable_entity() -> None:
    resolver = EntityResolver(
        [
            Entity(
                id="person:one",
                canonical_name="Sahil Barua",
                entity_type="person",
            ),
            Entity(
                id="person:two",
                canonical_name="Sahil Baruah",
                entity_type="person",
            ),
        ],
        ambiguity_margin=5,
    )

    result = resolver.resolve(
        EntityMention(name="Sahil Baru", entity_type="person")
    )

    assert result.method is ResolutionMethod.NEEDS_REVIEW
    assert result.needs_review is True
    assert set(result.candidate_entity_ids) == {"person:one", "person:two"}


def test_unseen_macroeconomic_predicate_registers_without_a_schema_change() -> None:
    resolver = EntityResolver()

    predicate = resolver.register_predicate(
        "Urban food inflation",
        value_kind="number",
        aliases=["Urban food CPI"],
    )

    assert predicate.key == "urban_food_inflation"
    assert resolver.predicates.resolve("urban food CPI") == predicate


def test_director_facts_from_two_documents_resolve_to_the_same_din() -> None:
    extractor = DeterministicExtractor()
    resolver = EntityResolver()
    resolved_ids: list[str] = []
    sources = [
        ("01-delhivery-prospectus-2022-excerpt.pdf", 85),
        ("02-delhivery-annual-report-fy24-excerpt.pdf", 23),
    ]

    for filename, page_index in sources:
        page = PdfIngestor().extract_page(
            DELHIVERY_DATA / filename,
            page_index,
            include_tables=False,
        )
        outcome = extractor.extract_page(
            page,
            default_subject=DELHIVERY,
            publisher=filename,
        )
        din_fact = next(
            fact
            for fact in outcome.facts
            if fact.predicate.key == "director_identification_number"
            and fact.value.value == "01173669"
        )
        resolved, resolution = resolver.resolve_fact(din_fact)
        assert resolution.method in {
            ResolutionMethod.NEW_ENTITY,
            ResolutionMethod.IDENTIFIER,
        }
        resolved_ids.append(resolved.subject.id)

    assert resolved_ids == ["din:01173669", "din:01173669"]


def test_a_shaky_fact_does_not_taint_later_facts_about_the_same_subject() -> None:
    """One weak extraction must not send a whole document's facts to review.

    An entity's review state records whether its *identity* is unsettled, which is a
    different question from whether the fact that mentioned it was solid. Copying the
    mention's state onto the entity used to mark it permanently, and every later fact
    resolving to it inherited the flag - enough to bury an entire filing's figures.
    """
    resolver = EntityResolver()

    first = resolver.resolve(
        EntityMention(
            name="Delhivery Limited",
            entity_type="company",
            review_state=ReviewState.NEEDS_REVIEW,
        )
    )
    assert first.entity.review_state is ReviewState.READY

    second = resolver.resolve(
        EntityMention(name="Delhivery Limited", entity_type="company", confidence=0.95)
    )
    assert second.entity.id == first.entity.id
    assert second.needs_review is False


def test_a_disputed_identity_still_stays_flagged() -> None:
    """The fix above must not silence a genuine identifier conflict."""
    existing = Entity(
        id="din:01173669",
        canonical_name="Suvir Suren Sujan",
        entity_type="person",
        identifiers={"din": "01173669"},
    )
    resolver = EntityResolver([existing])

    clash = resolver.resolve(
        EntityMention(
            name="Suvir Suren Sujan",
            entity_type="person",
            identifiers={"din": "99999999"},
        )
    )

    assert clash.method is ResolutionMethod.CONFLICT
    assert clash.needs_review is True
