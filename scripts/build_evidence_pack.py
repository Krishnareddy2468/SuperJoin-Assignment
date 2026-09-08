"""Condense a full pipeline report into something a reviewer can read.

A complete report is tens of megabytes, which is fine as a working artefact and
useless as a thing to open in a browser. This keeps the run's headline numbers and a
worked example of every relationship kind, with both facts and their evidence inlined
so a claim can be checked without cross-referencing ids by hand.

    python scripts/build_evidence_pack.py data/delhivery-report.json \
        reports/delhivery-evidence-pack.json
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

# Relationship kinds worth showing even when they are rare; the demo cases live here.
EXAMPLES_PER_KIND = 3


def fact_view(fact: dict) -> dict:
    evidence = (fact.get("evidence") or [{}])[0]
    value = fact.get("value", {})
    return {
        "predicate": fact["predicate"]["key"],
        "subject": fact["subject"]["canonical_name"],
        "raw_value": value.get("raw") or value.get("state"),
        "normalized_value": value.get("number"),
        "unit": value.get("unit"),
        "context": {
            key: value_
            for key, value_ in fact.get("context", {}).items()
            if value_ not in (None, {}, "")
        },
        "review_state": fact.get("review_state"),
        "warnings": fact.get("warnings") or [],
        "evidence": {
            "document": evidence.get("document_id"),
            "pdf_page_index": evidence.get("page_index"),
            "quote": evidence.get("quote"),
            "extractor": evidence.get("extractor"),
            "verified": evidence.get("verified"),
        },
    }


def build(report: dict) -> dict:
    facts = {fact["id"]: fact for fact in report["facts"]}
    documents = {doc["id"]: doc["original_filename"] for doc in report["documents"]}
    by_kind: dict[str, list] = collections.defaultdict(list)
    for relation in report["relations"]:
        by_kind[relation["relation_type"]].append(relation)

    highlights = {}
    for kind, relations in sorted(by_kind.items()):
        chosen = sorted(relations, key=lambda item: -item.get("confidence", 0))
        highlights[kind] = [
            {
                "relation_type": relation["relation_type"],
                "confidence": relation.get("confidence"),
                "explanation": relation.get("explanation"),
                "context_diff": relation.get("context_diff"),
                "fact_a": fact_view(facts[relation["fact_a_id"]]),
                "fact_b": fact_view(facts[relation["fact_b_id"]]),
            }
            for relation in chosen[:EXAMPLES_PER_KIND]
            if relation["fact_a_id"] in facts and relation["fact_b_id"] in facts
        ]

    return {
        "summary": {
            "documents": len(report["documents"]),
            "facts": len(report["facts"]),
            "relations": len(report["relations"]),
            "consistency_findings": len(report.get("consistency_findings", [])),
            "recorded_failures": len(report["failures"]),
            "relations_by_kind": dict(
                collections.Counter(r["relation_type"] for r in report["relations"])
            ),
            "facts_by_extractor": dict(
                collections.Counter(
                    evidence["extractor"]
                    for fact in report["facts"]
                    for evidence in fact.get("evidence", [])
                )
            ),
            "distinct_predicates": len({f["predicate"]["key"] for f in report["facts"]}),
        },
        "documents": [
            {
                "id": doc["id"],
                "filename": doc["original_filename"],
                "pages": doc.get("page_count"),
                "status": doc.get("status"),
                "extraction_mode": doc.get("extraction_mode"),
            }
            for doc in report["documents"]
        ],
        "document_names": documents,
        "consistency_findings": report.get("consistency_findings", []),
        "top_failure_reasons": collections.Counter(
            failure["reason"] for failure in report["failures"]
        ).most_common(10),
        "relationship_examples": highlights,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    source, target = Path(argv[1]), Path(argv[2])
    pack = build(json.loads(source.read_text()))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {target} ({target.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
