"""Command-line access to the shared FactLayer pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from factlayer import __version__
from factlayer.config import Config, load_config
from factlayer.service import FactLayerService, write_json


def build_parser() -> argparse.ArgumentParser:
    """Create the parser without reading environment variables or credentials."""
    parser = argparse.ArgumentParser(
        prog="factlayer",
        description="Extract grounded facts from PDFs and compare their context.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--no-llm",
        "--offline",
        dest="no_llm",
        action="store_true",
        help="Run only local deterministic extraction, even when an API key exists.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="Use this SQLite database instead of the configured default.",
    )
    commands = parser.add_subparsers(dest="command")

    ingest = commands.add_parser(
        "ingest",
        help="Process PDFs or directories through the complete pipeline.",
    )
    ingest.add_argument("paths", nargs="+", type=Path, help="PDF files or directories to process.")
    ingest.add_argument(
        "--reprocess",
        action="store_true",
        help="Replace stored results when the same PDF bytes are seen again.",
    )
    ingest.add_argument("--output", type=Path, help="Write the run summary as JSON.")

    link = commands.add_parser("link", help="Rebuild relationships between stored facts.")
    link.add_argument(
        "--cross-document-only",
        action="store_true",
        help=(
            "Only compare facts that come from different documents. "
            "By default facts are also compared inside one document, which is how "
            "standalone and consolidated figures on the same page get reconciled."
        ),
    )
    link.add_argument("--output", type=Path, help="Write the link summary as JSON.")

    report = commands.add_parser("report", help="Export facts, evidence, and relationships as JSON.")
    report.add_argument("--document", dest="document_id", help="Limit output to one document ID.")
    report.add_argument("--output", type=Path, help="Write JSON to a file instead of standard output.")

    status = commands.add_parser("status", help="Show database and processing counts.")
    status.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    models = commands.add_parser("models", help="Show optional model configuration safely.")
    models.add_argument(
        "--check",
        action="store_true",
        help="Contact Gemini and verify that the configured model is visible to this key.",
    )

    commands.add_parser(
        "llm-status",
        help="Show optional LLM status without making a network request.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return zero only when its requested work succeeds."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0

    config = load_config(db_path=arguments.database, no_llm=arguments.no_llm)
    if arguments.command == "llm-status":
        _print_llm_status(config)
        return 0
    if arguments.command == "models":
        return _run_models(config, check=arguments.check)

    try:
        service = FactLayerService(config)
        if arguments.command == "ingest":
            run = service.ingest_paths(
                arguments.paths,
                reprocess=arguments.reprocess,
                progress=lambda message: print(message, file=sys.stderr),
            )
            payload = run.as_dict()
            if arguments.output:
                target = write_json(payload, arguments.output)
                print(f"Run summary written to {target}")
            else:
                _print_ingest_summary(run.documents, run.elapsed_ms)
            return 1 if run.failed else 0

        if arguments.command == "link":
            run = service.link_all(
                cross_document_only=arguments.cross_document_only
            )
            payload = run.as_dict()
            if arguments.output:
                target = write_json(payload, arguments.output)
                print(f"Link summary written to {target}")
            else:
                print(
                    f"Compared {run.candidates} fact pairs: {run.created} new, "
                    f"{run.updated} refreshed ({run.elapsed_ms:.1f} ms)."
                )
            return 0

        if arguments.command == "report":
            payload = service.report(document_id=arguments.document_id)
            if arguments.output:
                target = write_json(payload, arguments.output)
                print(f"Report written to {target}")
            else:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if arguments.command == "status":
            payload = service.status()
            if arguments.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                _print_status(payload)
            return 0
    except (KeyError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: {FactLayerService._safe_error(error)}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {arguments.command}")
    return 2


def _print_ingest_summary(documents, elapsed_ms: float) -> None:
    if not documents:
        print("No PDFs were processed.")
        return
    for item in documents:
        if item.error:
            print(f"{item.filename}: failed — {item.error}")
            continue
        if item.duplicate:
            print(
                f"{item.filename}: duplicate, reused {item.counts.facts_reused} stored facts."
            )
            continue
        print(
            f"{item.filename}: {item.status}, {item.counts.pages} pages, "
            f"{item.counts.facts_created} new facts, "
            f"{item.counts.relations_created} new relationships, "
            f"{item.counts.failures} failures ({item.elapsed_ms:.1f} ms)."
        )
        if item.counts.cache_hits or item.counts.provider_calls:
            print(
                f"  LLM cache hits: {item.counts.cache_hits}; "
                f"provider calls: {item.counts.provider_calls}."
            )
        for warning in item.warnings:
            print(f"  Warning: {warning}")
    print(f"Finished {len(documents)} document(s) in {elapsed_ms:.1f} ms.")


def _print_status(payload: dict) -> None:
    statuses = payload["document_statuses"]
    status_text = ", ".join(f"{key}={value}" for key, value in sorted(statuses.items()))
    print(f"Database: {payload['database']}")
    print(f"Mode: {payload['extraction_mode']}")
    print(f"Documents: {payload['documents']}" + (f" ({status_text})" if status_text else ""))
    print(f"Facts: {payload['facts']}")
    print(f"Relationships: {payload['relations']}")
    print(f"Consistency findings: {payload['consistency_findings']}")
    print(f"Failures: {payload['failures']}")


def _print_llm_status(config: Config) -> None:
    if config.llm_enabled:
        print(f"LLM extraction: enabled ({config.llm_model})")
        fallbacks = config.llm_models[1:]
        if fallbacks:
            print(f"Fallback models: {', '.join(fallbacks)}")
        else:
            print("Fallback models: none configured")
    else:
        print(f"LLM extraction: disabled ({config.why_llm_disabled()})")


def _run_models(config: Config, *, check: bool) -> int:
    _print_llm_status(config)
    if not check:
        print("No network request was made. Add --check to verify model access.")
        return 0
    if not config.llm_enabled:
        print("Model access cannot be checked while LLM extraction is disabled.", file=sys.stderr)
        return 1
    try:
        from google import genai

        client = genai.Client(api_key=config.api_key or "")
        names = {
            str(getattr(model, "name", "")).removeprefix("models/")
            for model in client.models.list()
        }
    except Exception as error:
        print(
            f"Model check failed ({type(error).__name__}). No credential details were logged.",
            file=sys.stderr,
        )
        return 1
    reachable = [model for model in config.llm_models if model in names]
    missing = [model for model in config.llm_models if model not in names]
    for model in config.llm_models:
        label = "available" if model in names else "not listed for this key"
        print(f"  {model}: {label}")
    if not reachable:
        print(
            "None of the configured models are reachable. Set FACTLAYER_LLM_MODEL or "
            "FACTLAYER_LLM_FALLBACK_MODELS to names from the list above.",
            file=sys.stderr,
        )
        return 1
    # A missing fallback is worth saying out loud but does not make the run unusable,
    # because at least one model in the chain answered.
    if missing:
        print(f"Usable chain: {', '.join(reachable)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
