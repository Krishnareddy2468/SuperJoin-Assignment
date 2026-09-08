"""Layout-aware PDF ingestion with page-level provenance.

PyMuPDF gives us fast text geometry, while pdfplumber supplies table structure.
The ingestor keeps those concerns separate and returns ordinary domain objects
that later phases can normalize, extract from, and store.
"""

from __future__ import annotations

import hashlib
import re
import statistics
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf

from factlayer.schema import BoundingBox, Document, DocumentStatus, Passage, PassageRole


DEFAULT_MAX_PDF_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_PASSAGE_CHARS = 1_500


class PdfValidationError(ValueError):
    """Raised when a file cannot be processed safely as a PDF."""


@dataclass(frozen=True)
class IngestionWarning:
    code: str
    message: str
    page_index: int | None = None


@dataclass(frozen=True)
class TableCell:
    row_index: int
    column_index: int
    text: str
    bbox: BoundingBox | None = None


@dataclass(frozen=True)
class ExtractedTable:
    page_index: int
    table_index: int
    bbox: BoundingBox
    rows: tuple[tuple[str, ...], ...]
    cells: tuple[TableCell, ...]


@dataclass(frozen=True)
class IngestedPage:
    page_index: int
    passages: tuple[Passage, ...]
    tables: tuple[ExtractedTable, ...]
    warnings: tuple[IngestionWarning, ...]
    elapsed_ms: float


@dataclass(frozen=True)
class IngestedPdf:
    document: Document
    pages: tuple[IngestedPage, ...]
    warnings: tuple[IngestionWarning, ...] = field(default_factory=tuple)
    elapsed_ms: float = 0.0

    @property
    def passages(self) -> tuple[Passage, ...]:
        return tuple(passage for page in self.pages for passage in page.passages)

    @property
    def tables(self) -> tuple[ExtractedTable, ...]:
        return tuple(table for page in self.pages for table in page.tables)


@dataclass
class _TextBlock:
    text: str
    bbox: BoundingBox
    role: PassageRole = PassageRole.BODY
    column: int = 0

    @property
    def width(self) -> float:
        return self.bbox.x1 - self.bbox.x0


class PdfIngestor:
    """Read PDFs page by page without losing their visual reading order."""

    def __init__(
        self,
        *,
        max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        minimum_text_characters: int = 20,
    ):
        if max_pdf_bytes <= 0:
            raise ValueError("max_pdf_bytes must be positive")
        if max_passage_chars < 100:
            raise ValueError("max_passage_chars must be at least 100")
        if minimum_text_characters < 0:
            raise ValueError("minimum_text_characters cannot be negative")
        self.max_pdf_bytes = max_pdf_bytes
        self.max_passage_chars = max_passage_chars
        self.minimum_text_characters = minimum_text_characters

    def inspect(self, pdf_path: Path | str, *, password: str | None = None) -> Document:
        """Validate a PDF and return the document record used by later stages."""
        path = self._validate_file(pdf_path)
        content_hash = self._content_hash(path)
        with self._open_pdf(path, password=password) as pdf:
            page_count = pdf.page_count
        return Document(
            id=f"doc_{content_hash[:24]}",
            content_hash=content_hash,
            original_filename=path.name,
            page_count=page_count,
            status=DocumentStatus.REGISTERED,
        )

    def ingest(self, pdf_path: Path | str, *, password: str | None = None) -> IngestedPdf:
        """Collect a complete result; use ``iter_pages`` for streaming consumers."""
        started = time.perf_counter()
        document = self.inspect(pdf_path, password=password)
        pages = tuple(self.iter_pages(pdf_path, document=document, password=password))
        warnings = tuple(warning for page in pages for warning in page.warnings)
        final_status = DocumentStatus.PARTIAL if warnings else DocumentStatus.COMPLETE
        warning_messages = [
            (
                f"PDF page {warning.page_index + 1}: {warning.message}"
                if warning.page_index is not None
                else warning.message
            )
            for warning in warnings
        ]
        document = document.model_copy(
            update={"status": final_status, "warnings": warning_messages}
        )
        return IngestedPdf(
            document=document,
            pages=pages,
            warnings=warnings,
            elapsed_ms=(time.perf_counter() - started) * 1_000,
        )

    def iter_pages(
        self,
        pdf_path: Path | str,
        *,
        document: Document | None = None,
        password: str | None = None,
    ) -> Iterator[IngestedPage]:
        """Yield one processed page at a time and close both PDF readers reliably."""
        path = self._validate_file(pdf_path)
        document = document or self.inspect(path, password=password)

        with self._open_pdf(path, password=password) as pdf:
            furniture = self._repeated_furniture(pdf)
            plumber_pdf = None
            table_open_warning: IngestionWarning | None = None
            try:
                plumber_pdf = pdfplumber.open(path, password=password)
            except Exception as error:
                table_open_warning = IngestionWarning(
                    code="tables_unavailable",
                    message=f"Table extraction is unavailable for this PDF: {error}",
                )

            try:
                for page_index in range(pdf.page_count):
                    plumber_page = plumber_pdf.pages[page_index] if plumber_pdf else None
                    try:
                        page = self._extract_page(
                            pdf[page_index],
                            document_id=document.id,
                            repeated_furniture=furniture,
                            plumber_page=plumber_page,
                        )
                    finally:
                        if plumber_page is not None:
                            plumber_page.close()
                    if table_open_warning and page_index == 0:
                        page = IngestedPage(
                            page_index=page.page_index,
                            passages=page.passages,
                            tables=page.tables,
                            warnings=(table_open_warning, *page.warnings),
                            elapsed_ms=page.elapsed_ms,
                        )
                    yield page
            finally:
                if plumber_pdf is not None:
                    plumber_pdf.close()

    def extract_page(
        self,
        pdf_path: Path | str,
        page_index: int,
        *,
        password: str | None = None,
        include_tables: bool = True,
    ) -> IngestedPage:
        """Process one page for diagnostics and layout regression tests."""
        path = self._validate_file(pdf_path)
        document = self.inspect(path, password=password)
        with self._open_pdf(path, password=password) as pdf:
            if page_index < 0 or page_index >= pdf.page_count:
                raise IndexError(
                    f"PDF page index {page_index} is outside 0..{pdf.page_count - 1}"
                )
            furniture = self._repeated_furniture(pdf)
            plumber_pdf = pdfplumber.open(path, password=password) if include_tables else None
            try:
                plumber_page = plumber_pdf.pages[page_index] if plumber_pdf else None
                return self._extract_page(
                    pdf[page_index],
                    document_id=document.id,
                    repeated_furniture=furniture,
                    plumber_page=plumber_page,
                )
            finally:
                if plumber_pdf:
                    plumber_pdf.close()

    def _extract_page(
        self,
        page: pymupdf.Page,
        *,
        document_id: str,
        repeated_furniture: set[str],
        plumber_page: Any | None,
    ) -> IngestedPage:
        started = time.perf_counter()
        warnings: list[IngestionWarning] = []
        blocks = self._text_blocks(page, repeated_furniture)
        ordered = self._reading_order(blocks, page.rect.width)
        merged = self._merge_adjacent_blocks(ordered)
        passages = self._passages(merged, document_id, page.number)

        visible_text = sum(len(item.text.strip()) for item in passages if item.role is PassageRole.BODY)
        if visible_text < self.minimum_text_characters:
            warnings.append(
                IngestionWarning(
                    code="ocr_needed",
                    message="This page has little or no extractable text and may require OCR.",
                    page_index=page.number,
                )
            )

        tables: tuple[ExtractedTable, ...] = ()
        if plumber_page is not None:
            try:
                tables = tuple(self._tables(plumber_page, page.number))
            except Exception as error:
                warnings.append(
                    IngestionWarning(
                        code="table_extraction_failed",
                        message=f"Tables could not be extracted from this page: {error}",
                        page_index=page.number,
                    )
                )

        return IngestedPage(
            page_index=page.number,
            passages=tuple(passages),
            tables=tables,
            warnings=tuple(warnings),
            elapsed_ms=(time.perf_counter() - started) * 1_000,
        )

    def _text_blocks(self, page: pymupdf.Page, repeated_furniture: set[str]) -> list[_TextBlock]:
        blocks: list[_TextBlock] = []
        page_height = page.rect.height
        for raw in page.get_text("blocks", sort=False):
            if len(raw) > 6 and raw[6] != 0:
                continue
            x0, y0, x1, y1, text = raw[:5]
            cleaned = self._clean_text(text)
            if not cleaned or x1 <= x0 or y1 <= y0:
                continue
            key = self._furniture_key(cleaned)
            role = PassageRole.BODY
            if key in repeated_furniture and y1 <= page_height * 0.12:
                role = PassageRole.HEADER
            elif key in repeated_furniture and y0 >= page_height * 0.88:
                role = PassageRole.FOOTER
            blocks.append(
                _TextBlock(
                    text=cleaned,
                    bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    role=role,
                )
            )
        return blocks

    def _reading_order(self, blocks: list[_TextBlock], page_width: float) -> list[_TextBlock]:
        if not blocks:
            return []
        headers = sorted((block for block in blocks if block.role is PassageRole.HEADER), key=self._y_key)
        footers = sorted((block for block in blocks if block.role is PassageRole.FOOTER), key=self._y_key)
        body = [block for block in blocks if block.role is PassageRole.BODY]
        wide = sorted((block for block in body if block.width >= page_width * 0.72), key=self._y_key)
        narrow = [block for block in body if block not in wide]
        self._assign_columns(narrow, page_width)

        ordered_body: list[_TextBlock] = []
        remaining = list(narrow)
        for separator in wide:
            before = [block for block in remaining if (block.bbox.y0 + block.bbox.y1) / 2 < separator.bbox.y0]
            ordered_body.extend(self._sort_columns(before))
            ordered_body.append(separator)
            remaining = [block for block in remaining if block not in before]
        ordered_body.extend(self._sort_columns(remaining))
        return [*headers, *ordered_body, *footers]

    @staticmethod
    def _assign_columns(blocks: list[_TextBlock], page_width: float) -> None:
        if not blocks:
            return
        typical_width = statistics.median(block.width for block in blocks)
        threshold = max(24.0, min(page_width * 0.18, typical_width * 0.55))
        clusters: list[list[_TextBlock]] = []
        for block in sorted(blocks, key=lambda item: item.bbox.x0):
            if not clusters:
                clusters.append([block])
                continue
            cluster_x = statistics.mean(item.bbox.x0 for item in clusters[-1])
            if block.bbox.x0 - cluster_x > threshold:
                clusters.append([block])
            else:
                clusters[-1].append(block)
        for column, cluster in enumerate(clusters):
            for block in cluster:
                block.column = column

    @staticmethod
    def _sort_columns(blocks: list[_TextBlock]) -> list[_TextBlock]:
        return sorted(blocks, key=lambda item: (item.column, item.bbox.y0, item.bbox.x0))

    def _merge_adjacent_blocks(self, blocks: list[_TextBlock]) -> list[_TextBlock]:
        merged: list[_TextBlock] = []
        for block in blocks:
            if not merged:
                merged.append(block)
                continue
            previous = merged[-1]
            vertical_gap = block.bbox.y0 - previous.bbox.y1
            same_flow = (
                block.role is previous.role
                and block.column == previous.column
                and -1 <= vertical_gap <= 9
                and abs(block.bbox.x0 - previous.bbox.x0) <= 35
                and len(previous.text) + len(block.text) + 1 <= self.max_passage_chars
            )
            if not same_flow:
                merged.append(block)
                continue
            merged[-1] = _TextBlock(
                text=f"{previous.text}\n{block.text}",
                bbox=BoundingBox(
                    x0=min(previous.bbox.x0, block.bbox.x0),
                    y0=min(previous.bbox.y0, block.bbox.y0),
                    x1=max(previous.bbox.x1, block.bbox.x1),
                    y1=max(previous.bbox.y1, block.bbox.y1),
                ),
                role=previous.role,
                column=previous.column,
            )
        return merged

    def _passages(
        self, blocks: list[_TextBlock], document_id: str, page_index: int
    ) -> list[Passage]:
        passages: list[Passage] = []
        cursor = 0
        reading_order = 0
        for block in blocks:
            for chunk in self._split_text(block.text):
                passage_id = hashlib.sha256(
                    f"{document_id}\0{page_index}\0{reading_order}\0{chunk}".encode()
                ).hexdigest()[:24]
                passages.append(
                    Passage(
                        id=f"passage_{passage_id}",
                        document_id=document_id,
                        page_index=page_index,
                        reading_order=reading_order,
                        role=block.role,
                        text=chunk,
                        char_start=cursor,
                        char_end=cursor + len(chunk),
                        bbox=block.bbox,
                        text_hash=hashlib.sha256(chunk.encode()).hexdigest(),
                    )
                )
                cursor += len(chunk) + 2
                reading_order += 1
        return passages

    def _split_text(self, text: str) -> list[str]:
        remaining = text.strip()
        chunks: list[str] = []
        while len(remaining) > self.max_passage_chars:
            limit = self.max_passage_chars
            floor = int(limit * 0.6)
            candidates = [
                remaining.rfind("\n\n", floor, limit),
                remaining.rfind(". ", floor, limit),
                remaining.rfind("\n", floor, limit),
                remaining.rfind(" ", floor, limit),
            ]
            split_at = max(candidates)
            if split_at < floor:
                split_at = limit
            elif remaining[split_at : split_at + 2] == ". ":
                split_at += 1
            chunk = remaining[:split_at].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks

    def _repeated_furniture(self, pdf: pymupdf.Document) -> set[str]:
        pages_by_key: dict[str, set[int]] = defaultdict(set)
        for page in pdf:
            page_height = page.rect.height
            for raw in page.get_text("blocks", sort=False):
                if len(raw) > 6 and raw[6] != 0:
                    continue
                _, y0, _, y1, text = raw[:5]
                if y1 > page_height * 0.12 and y0 < page_height * 0.88:
                    continue
                key = self._furniture_key(self._clean_text(text))
                if key:
                    pages_by_key[key].add(page.number)
        return {key for key, pages in pages_by_key.items() if len(pages) >= 2}

    def _tables(self, page: Any, page_index: int) -> Iterator[ExtractedTable]:
        for table_index, table in enumerate(page.find_tables()):
            extracted_rows = table.extract() or []
            rows = tuple(
                tuple(self._clean_text(cell or "") for cell in row)
                for row in extracted_rows
            )
            table_bbox = self._safe_pdfplumber_bbox(table.bbox, page.width, page.height)
            nonempty_rows = sum(any(cell for cell in row) for row in rows)
            populated_columns = {
                column_index
                for row in rows
                for column_index, text in enumerate(row)
                if text
            }
            if table_bbox is None or nonempty_rows < 2 or len(populated_columns) < 2:
                continue
            raw_cells = list(table.cells or [])
            cells: list[TableCell] = []
            flat_index = 0
            for row_index, row in enumerate(rows):
                for column_index, text in enumerate(row):
                    bbox = None
                    if flat_index < len(raw_cells) and raw_cells[flat_index]:
                        bbox = self._safe_pdfplumber_bbox(
                            raw_cells[flat_index], page.width, page.height
                        )
                    cells.append(TableCell(row_index, column_index, text, bbox))
                    flat_index += 1
            yield ExtractedTable(
                page_index=page_index,
                table_index=table_index,
                bbox=table_bbox,
                rows=rows,
                cells=tuple(cells),
            )

    @staticmethod
    def _safe_pdfplumber_bbox(
        coordinates: tuple[float, float, float, float],
        page_width: float,
        page_height: float,
    ) -> BoundingBox | None:
        x0, y0, x1, y1 = coordinates
        tolerance = 0.5
        if (
            x0 < -tolerance
            or y0 < -tolerance
            or x1 > page_width + tolerance
            or y1 > page_height + tolerance
            or x1 <= x0
            or y1 <= y0
        ):
            return None
        return BoundingBox(
            x0=max(0, x0),
            y0=max(0, y0),
            x1=min(page_width, x1),
            y1=min(page_height, y1),
        )

    def _validate_file(self, pdf_path: Path | str) -> Path:
        path = Path(pdf_path)
        if not path.exists():
            raise PdfValidationError(f"PDF file does not exist: {path}")
        if not path.is_file():
            raise PdfValidationError(f"PDF path is not a regular file: {path}")
        size = path.stat().st_size
        if size == 0:
            raise PdfValidationError("The uploaded PDF is empty")
        if size > self.max_pdf_bytes:
            raise PdfValidationError(
                f"PDF is {size} bytes, above the {self.max_pdf_bytes}-byte limit"
            )
        with path.open("rb") as source:
            header = source.read(1_024)
        if b"%PDF-" not in header:
            raise PdfValidationError("The uploaded file does not contain a PDF header")
        return path

    @staticmethod
    def _open_pdf(path: Path, *, password: str | None) -> pymupdf.Document:
        try:
            pdf = pymupdf.open(path)
        except Exception as error:
            raise PdfValidationError(f"The PDF could not be opened: {error}") from error
        if not pdf.is_pdf:
            pdf.close()
            raise PdfValidationError("The uploaded file is not recognized as a PDF")
        if pdf.needs_pass:
            if not password:
                pdf.close()
                raise PdfValidationError("The PDF is encrypted and needs a password")
            if not pdf.authenticate(password):
                pdf.close()
                raise PdfValidationError("The supplied PDF password is incorrect")
        if pdf.page_count == 0:
            pdf.close()
            raise PdfValidationError("The PDF does not contain any pages")
        return pdf

    @staticmethod
    def _content_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", "").replace("\x07", "•")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _furniture_key(text: str) -> str:
        lowered = text.casefold()
        lowered = re.sub(r"\d+", "#", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered.strip(" ·|—-#")

    @staticmethod
    def _y_key(block: _TextBlock) -> tuple[float, float]:
        return block.bbox.y0, block.bbox.x0


__all__ = [
    "ExtractedTable",
    "IngestedPage",
    "IngestedPdf",
    "IngestionWarning",
    "PdfIngestor",
    "PdfValidationError",
    "TableCell",
]
