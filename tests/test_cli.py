import os
import json
import subprocess
import sys
from pathlib import Path

import pymupdf

from factlayer import __version__
from factlayer.cli import main


def save_cli_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text(
        (50, 100),
        "Revenue from operations for FY24 stood at INR 120 million.",
        fontsize=11,
    )
    document.save(path)
    document.close()


def test_main_without_arguments_prints_help(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "usage: factlayer" in output
    assert "Extract grounded facts from PDFs" in output
    assert "ingest" in output
    assert "report" in output
    assert "status" in output


def test_module_help_does_not_require_an_api_key() -> None:
    environment = os.environ.copy()
    environment.pop("GEMINI_API_KEY", None)

    result = subprocess.run(
        [sys.executable, "-m", "factlayer.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0
    assert "usage: factlayer" in result.stdout
    assert "GEMINI_API_KEY" not in result.stderr


def test_module_prints_package_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "factlayer.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"factlayer {__version__}"


def test_llm_status_explains_hard_offline_mode(capsys, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "configured-but-not-called")

    assert main(["--no-llm", "llm-status"]) == 0

    assert capsys.readouterr().out.strip() == (
        "LLM extraction: disabled (FACTLAYER_NO_LLM is set)"
    )


def test_llm_status_shows_the_configured_model_without_calling_it(capsys, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "configured-but-not-called")
    monkeypatch.setenv("FACTLAYER_LLM_MODEL", "gemini-test-model")
    # Set explicitly so the assertion does not depend on whatever a developer happens to
    # have in their own .env file.
    monkeypatch.setenv("FACTLAYER_LLM_FALLBACK_MODELS", "backup-one,backup-two")

    assert main(["llm-status"]) == 0

    assert capsys.readouterr().out.strip().splitlines() == [
        "LLM extraction: enabled (gemini-test-model)",
        "Fallback models: backup-one, backup-two",
    ]


def test_llm_status_says_so_when_fallbacks_are_switched_off(capsys, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "configured-but-not-called")
    monkeypatch.setenv("FACTLAYER_LLM_MODEL", "gemini-test-model")
    monkeypatch.setenv("FACTLAYER_LLM_FALLBACK_MODELS", "")

    assert main(["llm-status"]) == 0

    assert "Fallback models: none configured" in capsys.readouterr().out


def test_models_command_does_not_make_a_request_by_default(capsys, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "configured-but-not-called")

    assert main(["models"]) == 0

    output = capsys.readouterr().out
    assert "LLM extraction: enabled" in output
    assert "No network request was made" in output
    assert "configured-but-not-called" not in output


def test_cli_ingest_status_and_report_share_one_database(
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "report.pdf"
    database = tmp_path / "facts.db"
    report = tmp_path / "result.json"
    save_cli_pdf(pdf)
    prefix = ["--offline", "--database", str(database)]

    assert main([*prefix, "ingest", str(pdf)]) == 0
    first_output = capsys.readouterr()
    assert "new facts" in first_output.out
    assert "Inspecting report.pdf" in first_output.err

    assert main([*prefix, "ingest", str(pdf)]) == 0
    assert "duplicate" in capsys.readouterr().out

    assert main([*prefix, "status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["documents"] == 1
    assert status["facts"] >= 1
    assert status["extraction_mode"] == "offline"

    assert main([*prefix, "report", "--output", str(report)]) == 0
    capsys.readouterr()
    payload = json.loads(report.read_text())
    assert payload["summary"]["documents"] == 1
    assert payload["summary"]["facts"] == status["facts"]


def test_cli_returns_nonzero_for_a_fatal_document_error(
    tmp_path: Path,
    capsys,
) -> None:
    invalid = tmp_path / "broken.pdf"
    invalid.write_bytes(b"plain text")

    exit_code = main(
        [
            "--offline",
            "--database",
            str(tmp_path / "facts.db"),
            "ingest",
            str(invalid),
        ]
    )

    output = capsys.readouterr()
    assert exit_code == 1
    assert "failed" in output.out
    assert "PDF header" in output.out
