from pathlib import Path

import pymupdf
import pytest

from factlayer.ingest import PdfIngestor, PdfValidationError
from factlayer.schema import DocumentStatus, PassageRole


PROJECT_ROOT = Path(__file__).parents[1]
DELHIVERY_REPORT = (
    PROJECT_ROOT
    / "starter-datasets"
    / "delhivery"
    / "02-delhivery-annual-report-fy24-excerpt.pdf"
)


def save_text_pdf(path: Path, pages: list[list[tuple[float, float, str]]]) -> None:
    document = pymupdf.open()
    for items in pages:
        page = document.new_page(width=600, height=800)
        for x, y, text in items:
            page.insert_text((x, y), text, fontsize=10)
    document.save(path)
    document.close()


def test_inspect_uses_content_hash_and_pdf_metadata(tmp_path) -> None:
    path = tmp_path / "small.pdf"
    save_text_pdf(path, [[(50, 100, "A useful statement")]])

    document = PdfIngestor().inspect(path)

    assert document.original_filename == "small.pdf"
    assert document.page_count == 1
    assert len(document.content_hash) == 64
    assert document.id == f"doc_{document.content_hash[:24]}"
    assert document.status is DocumentStatus.REGISTERED


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("empty.pdf", b"", "empty"),
        ("not-a-pdf.pdf", b"plain text", "PDF header"),
        ("broken.pdf", b"%PDF-1.7\nthis is damaged", "could not be opened"),
    ],
)
def test_invalid_files_fail_with_helpful_messages(
    tmp_path, filename: str, content: bytes, message: str
) -> None:
    path = tmp_path / filename
    path.write_bytes(content)

    with pytest.raises(PdfValidationError, match=message):
        PdfIngestor().inspect(path)


def test_missing_and_oversized_files_are_rejected(tmp_path) -> None:
    with pytest.raises(PdfValidationError, match="does not exist"):
        PdfIngestor().inspect(tmp_path / "missing.pdf")

    path = tmp_path / "large.pdf"
    save_text_pdf(path, [[(50, 100, "This PDF is larger than the test limit")]])
    with pytest.raises(PdfValidationError, match="above the"):
        PdfIngestor(max_pdf_bytes=10).inspect(path)


def test_encrypted_pdf_requires_the_correct_password(tmp_path) -> None:
    plain = pymupdf.open()
    page = plain.new_page()
    page.insert_text((50, 100), "Protected evidence")
    path = tmp_path / "protected.pdf"
    plain.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="reader-password",
    )
    plain.close()

    ingestor = PdfIngestor()
    with pytest.raises(PdfValidationError, match="needs a password"):
        ingestor.inspect(path)
    with pytest.raises(PdfValidationError, match="incorrect"):
        ingestor.inspect(path, password="wrong-password")

    assert ingestor.inspect(path, password="reader-password").page_count == 1


def test_repeated_headers_and_footers_remain_identifiable(tmp_path) -> None:
    path = tmp_path / "report.pdf"
    save_text_pdf(
        path,
        [
            [(40, 35, "Quarterly Review"), (40, 150, "First page body fact"), (40, 780, "Page 1")],
            [(40, 35, "Quarterly Review"), (40, 150, "Second page body fact"), (40, 780, "Page 2")],
            [(40, 35, "Quarterly Review"), (40, 780, "Page 3")],
        ],
    )

    result = PdfIngestor().ingest(path)

    assert result.pages[0].passages[0].role is PassageRole.HEADER
    assert result.pages[0].passages[-1].role is PassageRole.FOOTER
    assert any(item.role is PassageRole.BODY for item in result.pages[0].passages)
    assert result.pages[2].warnings[0].code == "ocr_needed"
    assert result.document.status is DocumentStatus.PARTIAL
    assert result.document.warnings == [
        "PDF page 3: This page has little or no extractable text and may require OCR."
    ]


def test_passages_keep_page_offsets_bounding_boxes_and_stable_order(tmp_path) -> None:
    path = tmp_path / "columns.pdf"
    save_text_pdf(
        path,
        [[
            (50, 100, "Left column begins"),
            (50, 140, "Left column continues"),
            (340, 110, "Right column begins"),
            (340, 150, "Right column continues"),
        ]],
    )

    page = next(PdfIngestor().iter_pages(path))
    texts = [passage.text for passage in page.passages]

    assert texts.index("Left column begins") < texts.index("Left column continues")
    assert texts.index("Left column continues") < texts.index("Right column begins")
    assert [passage.reading_order for passage in page.passages] == list(range(len(page.passages)))
    assert all(passage.page_index == 0 for passage in page.passages)
    assert all(passage.bbox is not None for passage in page.passages)
    assert all(passage.char_end > passage.char_start for passage in page.passages)
    assert all(len(passage.text_hash) == 64 for passage in page.passages)


def test_long_blocks_split_without_losing_text_or_offsets(tmp_path) -> None:
    path = tmp_path / "long.pdf"
    text = " ".join(f"Sentence {index} contains a grounded value." for index in range(30))
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_textbox((40, 80, 560, 700), text, fontsize=9)
    document.save(path)
    document.close()

    passages = PdfIngestor(max_passage_chars=120).ingest(path).passages

    assert len(passages) > 1
    assert all(len(passage.text) <= 120 for passage in passages)
    assert [passage.reading_order for passage in passages] == list(range(len(passages)))
    assert all(
        current.char_end < following.char_start
        for current, following in zip(passages, passages[1:])
    )


def test_table_rows_and_cell_locations_are_preserved(tmp_path) -> None:
    path = tmp_path / "table.pdf"
    document = pymupdf.open()
    page = document.new_page(width=500, height=500)
    for y in (100, 150, 200):
        page.draw_line((50, y), (350, y), width=1)
    for x in (50, 200, 350):
        page.draw_line((x, 100), (x, 200), width=1)
    page.insert_text((65, 130), "Metric", fontsize=10)
    page.insert_text((215, 130), "FY24", fontsize=10)
    page.insert_text((65, 180), "Revenue", fontsize=10)
    page.insert_text((215, 180), "100", fontsize=10)
    document.save(path)
    document.close()

    result = PdfIngestor().ingest(path)

    assert len(result.tables) == 1
    assert result.tables[0].rows == (("Metric", "FY24"), ("Revenue", "100"))
    assert len(result.tables[0].cells) == 4
    assert all(cell.bbox is not None for cell in result.tables[0].cells)
    assert result.tables[0].bbox.x0 == pytest.approx(50)


def test_real_column_bleed_keeps_director_and_resignation_together() -> None:
    with pymupdf.open(DELHIVERY_REPORT) as document:
        naive_text = document[23].get_text("text", sort=True)
    subject = "Mr. Suvir Suren Sujan, Non-Executive Director"
    unrelated = "SVP - Business Development"
    identifier = "(DIN: 01173669)"

    assert naive_text.index(subject) < naive_text.index(unrelated) < naive_text.index(identifier)

    page = PdfIngestor().extract_page(DELHIVERY_REPORT, 23, include_tables=False)
    grounded_passages = [
        passage for passage in page.passages
        if subject in passage.text and identifier in passage.text
    ]

    assert len(grounded_passages) == 1
    assert "resigned from the Board with effect" in grounded_passages[0].text
    assert unrelated not in grounded_passages[0].text
    assert grounded_passages[0].bbox.x1 < 300


def test_page_index_outside_the_document_is_rejected(tmp_path) -> None:
    path = tmp_path / "one-page.pdf"
    save_text_pdf(path, [[(50, 100, "Only page")]])

    with pytest.raises(IndexError, match="outside"):
        PdfIngestor().extract_page(path, 1)


def test_a_page_without_extractable_text_is_flagged_for_ocr(tmp_path: Path) -> None:
    """An empty page is a page we could not read, not a page with nothing on it.

    Recording it as needing OCR keeps a scanned insert from silently becoming a gap in
    the knowledge layer.
    """
    document = pymupdf.open()
    document.new_page()
    readable = document.new_page()
    readable.insert_text((72, 72), "Revenue from operations stood at 100 million.")
    path = tmp_path / "one-blank-page.pdf"
    document.save(path)
    document.close()

    ingested = PdfIngestor().ingest(path)

    warnings = [warning.message for page in ingested.pages for warning in page.warnings]
    assert any("OCR" in message for message in warnings)
    # The readable page still produces passages, so one blank page costs only itself.
    assert any(page.passages for page in ingested.pages)
