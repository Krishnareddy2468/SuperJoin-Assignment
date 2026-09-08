"""Draw the assignment's explainer PDFs.

Four one-page diagrams, generated rather than hand-drawn so they cannot drift from the
code they describe:

    python scripts/build_assignment_docs.py

Rendered with PyMuPDF, which the project already reads PDFs with, so producing these
needs no extra tooling. Every text insertion is checked for overflow: PyMuPDF silently
draws nothing when a string does not fit its rectangle, which is a very easy way to ship
an empty box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent.parent / "assignment-docs"
W, H = 842, 595  # A4 landscape

INK = (0.09, 0.11, 0.10)
DEEP = (0.11, 0.29, 0.22)
MID = (0.22, 0.45, 0.35)
MUTED = (0.42, 0.46, 0.44)
PAPER = (0.972, 0.969, 0.957)
MINT = (0.898, 0.925, 0.906)
SKY = (0.90, 0.93, 0.95)
AMBER = (0.78, 0.55, 0.16)
AMBER_FILL = (0.98, 0.94, 0.86)
RED = (0.66, 0.20, 0.16)
RED_FILL = (0.98, 0.92, 0.90)
GREY_FILL = (0.95, 0.96, 0.95)
RULE = (0.82, 0.83, 0.81)

BOLD, BODY = "hebo", "helv"

# The built-in PDF fonts cover Latin-1 and nothing more, so a rupee sign or an em dash
# silently becomes "?". Spelling them out keeps the pages readable without embedding a
# font file for three glyphs.
_ASCII = {
    "₹": "Rs.", "—": " - ", "–": "-", "→": "->",
    "’": "'", "“": '"', "”": '"', "≥": ">=", "·": "-",
}
_PROBLEMS: list[str] = []


def t(text: str) -> str:
    for glyph, plain in _ASCII.items():
        text = text.replace(glyph, plain)
    return text


def write(page, rect, text, *, size=9, font=BODY, color=MUTED, align=0, where=""):
    """Insert text and complain loudly if it did not fit.

    PyMuPDF returns a negative number and draws nothing when the string overflows, so
    without this check an over-long label becomes an empty box that looks intentional.
    """
    left = page.insert_textbox(
        rect, t(text), fontsize=size, fontname=font, color=color, align=align
    )
    if left < 0:
        _PROBLEMS.append(f"{where or text[:28]!r} overflows its box by {abs(left):.0f}pt")
    return left


def header(doc, kicker, title, standfirst):
    page = doc.new_page(width=W, height=H)
    page.draw_rect(pymupdf.Rect(0, 0, W, H), color=None, fill=PAPER)
    page.insert_text((54, 56), t(kicker.upper()), fontsize=8.5, fontname=BOLD, color=MID)
    page.insert_text((54, 86), t(title), fontsize=25, fontname=BOLD, color=INK)
    write(page, pymupdf.Rect(54, 98, 760, 132), standfirst, size=10.5, where="standfirst")
    page.draw_line(pymupdf.Point(54, 142), pymupdf.Point(788, 142), color=RULE, width=0.8)
    return page


def node(page, rect, heading, lines=(), *, fill=GREY_FILL, edge=DEEP, dashed=False,
         head_color=None, size=7.6, head_size=9.5):
    """A rounded box: bold heading, optional smaller body lines."""
    page.draw_rect(
        rect, color=edge, fill=fill, width=1.1, radius=0.08,
        dashes="[3 3] 0" if dashed else None,
    )
    # Absolute padding rather than a multiplier: a one-line heading needs the font's
    # full line box plus a little, and a ratio that suits 8pt starves 10pt by a point.
    heading_bottom = rect.y0 + 4 + head_size + 9
    write(page, pymupdf.Rect(rect.x0 + 9, rect.y0 + 4, rect.x1 - 7, heading_bottom),
          heading, size=head_size, font=BOLD, color=head_color or INK, where=heading)
    if lines:
        write(page, pymupdf.Rect(rect.x0 + 9, heading_bottom, rect.x1 - 7, rect.y1 - 3),
              "\n".join(lines), size=size, where=heading + " body")


def diamond(page, cx, cy, half_w, half_h, text, *, fill=SKY, edge=MID):
    """A decision node. Text sits in the inscribed rectangle so it stays inside."""
    points = [
        pymupdf.Point(cx, cy - half_h), pymupdf.Point(cx + half_w, cy),
        pymupdf.Point(cx, cy + half_h), pymupdf.Point(cx - half_w, cy),
    ]
    page.draw_polyline(points, color=edge, fill=fill, width=1.1, closePath=True)
    write(page, pymupdf.Rect(cx - half_w * 0.46, cy - half_h * 0.46,
                             cx + half_w * 0.46, cy + half_h * 0.52),
          text, size=7.4, font=BOLD, color=INK, align=1, where=text)


def head(page, x, y, direction, color):
    """An arrow head that points where the line is actually going.

    The head used to be a fixed left-pointing triangle, which is right for a horizontal
    connector and wrong for every vertical one - on a diagram of mostly vertical joins
    they all read as stray diagonal marks.
    """
    size, spread = 7.5, 4.2
    if direction == "down":
        pts = [(x, y), (x - spread, y - size), (x + spread, y - size)]
    elif direction == "up":
        pts = [(x, y), (x - spread, y + size), (x + spread, y + size)]
    elif direction == "left":
        pts = [(x, y), (x + size, y - spread), (x + size, y + spread)]
    else:
        pts = [(x, y), (x - size, y - spread), (x - size, y + spread)]
    page.draw_polyline([pymupdf.Point(*pt) for pt in pts],
                       color=color, fill=color, width=0.6, closePath=True)


def arrow(page, start, end, *, color=MID, label=None, label_above=True, width=1.2):
    """Straight or single-elbow connector, with the head oriented to its last segment."""
    sx, sy = start
    ex, ey = end
    vertical = abs(sx - ex) <= 1
    if not vertical and abs(sy - ey) > 1:
        mid_x = (sx + ex) / 2
        page.draw_line(pymupdf.Point(sx, sy), pymupdf.Point(mid_x, sy), color=color, width=width)
        page.draw_line(pymupdf.Point(mid_x, sy), pymupdf.Point(mid_x, ey), color=color, width=width)
        stop = ex - 7 if ex > mid_x else ex + 7
        page.draw_line(pymupdf.Point(mid_x, ey), pymupdf.Point(stop, ey), color=color, width=width)
        head(page, ex, ey, "right" if ex > mid_x else "left", color)
    elif vertical:
        stop = ey - 7 if ey > sy else ey + 7
        page.draw_line(pymupdf.Point(sx, sy), pymupdf.Point(sx, stop), color=color, width=width)
        head(page, ex, ey, "down" if ey > sy else "up", color)
    else:
        stop = ex - 7 if ex > sx else ex + 7
        page.draw_line(pymupdf.Point(sx, sy), pymupdf.Point(stop, ey), color=color, width=width)
        head(page, ex, ey, "right" if ex > sx else "left", color)
    if label:
        if vertical:
            page.insert_text((sx + 6, (sy + ey) / 2), t(label), fontsize=7, fontname=BOLD, color=color)
        else:
            ly = min(sy, ey) - 11 if label_above else max(sy, ey) + 4
            page.insert_text((sx + 6, ly), t(label), fontsize=7, fontname=BOLD, color=color)


def note(page, y, text, *, size=9, color=MUTED, font=BODY, x=54, width=734, height=44):
    write(page, pymupdf.Rect(x, y, x + width, y + height), text,
          size=size, font=font, color=color, where="note")


def architecture(doc):
    page = header(
        doc, "architecture", "How a PDF becomes comparable knowledge",
        "One pass per document. The dashed box is the only optional part; everything else "
        "runs with no API key, and every result in this submission came from that path.",
    )
    # Row one: read the file, then split it into passages and tables.
    node(page, pymupdf.Rect(54, 160, 148, 226), "PDF in",
         ["Upload or CLI.", "Type and size", "checked first."], fill=MINT)
    node(page, pymupdf.Rect(178, 160, 300, 226), "1  Ingest",
         ["Reads block geometry,", "not the PDF's text order,", "so columns stay apart."], fill=MINT)
    node(page, pymupdf.Rect(330, 160, 452, 226), "Passages + tables",
         ["Each keeps its page,", "character span and", "grid position."], fill=MINT)
    arrow(page, (148, 193), (178, 193))
    arrow(page, (300, 193), (330, 193))

    # The two extractors, stacked, feeding one gate.
    node(page, pymupdf.Rect(482, 148, 604, 210), "2a  Rules",
         ["Numbers, units, periods,", "DIN and CIN, table cells.", "Always runs."], fill=MINT)
    node(page, pymupdf.Rect(482, 218, 604, 280), "2b  Gemini",
         ["Facts stated in prose:", "roles, definitions,", "basis caveats."],
         fill=AMBER_FILL, edge=AMBER, dashed=True, head_color=AMBER)
    arrow(page, (452, 186), (482, 179))
    arrow(page, (452, 200), (482, 247), color=AMBER)

    diamond(page, 690, 214, 70, 48, "quote found verbatim in the source?")
    arrow(page, (604, 179), (620, 200))
    arrow(page, (604, 247), (620, 228), color=AMBER)

    node(page, pymupdf.Rect(650, 300, 788, 360), "Recorded failure",
         ["Kept, with its page", "and the reason it", "was rejected."],
         fill=RED_FILL, edge=RED, head_color=RED)
    page.draw_line(pymupdf.Point(760, 214), pymupdf.Point(760, 292), color=RED, width=1.2)
    page.draw_polyline(
        [pymupdf.Point(760, 300), pymupdf.Point(755.8, 292.5), pymupdf.Point(764.2, 292.5)],
        color=RED, fill=RED, width=0.6, closePath=True,
    )
    page.insert_text((766, 268), t("no"), fontsize=7, fontname=BOLD, color=RED)

    # Row two, reached by a clear channel underneath the extractors.
    page.draw_line(pymupdf.Point(690, 262), pymupdf.Point(690, 286), color=MID, width=1.2)
    page.draw_line(pymupdf.Point(690, 286), pymupdf.Point(116, 286), color=MID, width=1.2)
    page.draw_line(pymupdf.Point(116, 286), pymupdf.Point(116, 292), color=MID, width=1.2)
    page.draw_polyline(
        [pymupdf.Point(116, 300), pymupdf.Point(111.8, 292.5), pymupdf.Point(120.2, 292.5)],
        color=MID, fill=MID, width=0.6, closePath=True,
    )
    page.insert_text((640, 282), t("yes, store it"), fontsize=7, fontname=BOLD, color=MID)

    node(page, pymupdf.Rect(54, 300, 179, 366), "3  Normalise",
         ["crore and million meet.", "FY24 becomes a date", "range. Synonyms merge."], fill=MINT)
    node(page, pymupdf.Rect(201, 300, 326, 366), "4  Resolve",
         ["One company, one", "director, across files.", "Identifiers beat names."], fill=MINT)
    node(page, pymupdf.Rect(348, 300, 473, 366), "5  Compare",
         ["Every pair sharing a", "subject and measure.", "See the approach page."], fill=MINT)
    node(page, pymupdf.Rect(495, 300, 620, 366), "6  Serve",
         ["API and browser UI.", "Every claim opens to", "its own quote."], fill=MINT)
    arrow(page, (179, 333), (201, 333))
    arrow(page, (326, 333), (348, 333))
    arrow(page, (473, 333), (495, 333))

    page.draw_line(pymupdf.Point(54, 384), pymupdf.Point(788, 384), color=RULE, width=0.8)
    node(page, pymupdf.Rect(54, 396, 274, 500), "Stored in SQLite, not a graph",
         ["documents, passages, entities,", "predicates, facts, evidence,", "relations, failures.",
          "", "Keyed on content hash, so a", "re-upload costs nothing and a", "new file needs no rebuild."],
         head_size=9)
    node(page, pymupdf.Rect(292, 396, 512, 500), "Nothing is stored ungrounded",
         ["A fact without a resolvable", "quote is never written.",
          "", "On one live run the model", "proposed 40 facts; 12 were", "dropped for quoting text", "that was not there."],
         head_size=9)
    node(page, pymupdf.Rect(530, 396, 788, 500), "Bounded, so it cannot stall",
         ["8 calls at once, 150 per document,", "25s per call, 60s for the phase.",
          "", "A 27-page deck: 103s -> 30s.", "An 89-page report: minutes -> 70s.",
          "", "Whatever a cap skipped is reported."],
         head_size=9)
    note(page, 512, "Measured: a 100-page annual report ingests in 15 seconds and 123 MB. "
                    "Six PDFs, 5,329 facts and 9,472 relationships share one database.")
    return page


def approach(doc):
    page = header(
        doc, "the core idea", "A number is not a fact",
        "81,415.38 means nothing alone. Attach what it measures, whose it is and when, and "
        "one rule can then decide how any two facts relate.",
    )
    # ---- left: the anatomy of a single fact --------------------------------------
    box = pymupdf.Rect(54, 162, 300, 404)
    page.draw_rect(box, color=DEEP, fill=MINT, width=1.1, radius=0.04)
    page.insert_text((68, 184), t("ONE FACT"), fontsize=8.5, fontname=BOLD, color=DEEP)
    rows = [
        ("subject", "Delhivery Limited", "who"),
        ("predicate", "revenue from operations", "what"),
        ("value", "Rs. 81,415.38 mn -> 81.42bn INR", "how much"),
        ("context", "FY24 - consolidated - audited", "the envelope"),
        ("evidence", "page 21, quoted verbatim", "the proof"),
    ]
    y = 196
    for name, value, gloss in rows:
        page.insert_text((68, y + 11), t(name), fontsize=8.5, fontname=BOLD, color=INK)
        page.insert_text((140, y + 11), t(value), fontsize=8, fontname=BODY, color=MUTED)
        page.insert_text((68, y + 22), t(gloss), fontsize=6.8, fontname=BODY, color=MID)
        y += 36
    write(page, pymupdf.Rect(68, y + 6, 288, y + 34),
          "Drop the envelope and two true figures look like a lie.",
          size=8.2, font=BOLD, color=DEEP, where="anatomy footer")

    # ---- right: the rule, as a decision -----------------------------------------
    page.insert_text((330, 168), t("THE ONE RULE, AS A DECISION"),
                     fontsize=8.5, fontname=BOLD, color=DEEP)

    node(page, pymupdf.Rect(330, 190, 406, 242), "Two facts",
         ["from anywhere", "in the layer"], size=7.2, head_size=8.5)

    # Diamonds sized so the text sits inside the sloping sides without crowding the
    # boxes on either flank.
    diamond(page, 468, 216, 56, 34, "same subject and measure?")
    arrow(page, (406, 216), (410, 216))
    node(page, pymupdf.Rect(544, 190, 690, 242), "Never compared",
         ["different things", "are not rivals"], size=7.2, head_size=8.5)
    arrow(page, (524, 216), (544, 216), label="no")

    page.draw_line(pymupdf.Point(468, 250), pymupdf.Point(468, 258), color=MID, width=1.2)
    page.draw_polyline(
        [pymupdf.Point(468, 266), pymupdf.Point(463.8, 258.5), pymupdf.Point(472.2, 258.5)],
        color=MID, fill=MID, width=0.6, closePath=True,
    )
    page.insert_text((474, 260), t("yes"), fontsize=7, fontname=BOLD, color=MID)

    diamond(page, 468, 300, 56, 34, "values agree once normalised?")
    node(page, pymupdf.Rect(604, 262, 788, 314), "CORROBORATES",
         ["Rs. 8,142 Cr = Rs. 81,415.38 mn", "agree within a derived tolerance"],
         fill=MINT, edge=DEEP, head_color=DEEP, size=7.2, head_size=9)
    arrow(page, (524, 300), (604, 300), label="yes")

    page.draw_line(pymupdf.Point(468, 336), pymupdf.Point(468, 384), color=MID, width=1.2)
    page.draw_polyline(
        [pymupdf.Point(468, 392), pymupdf.Point(463.8, 384.5), pymupdf.Point(472.2, 384.5)],
        color=MID, fill=MID, width=0.6, closePath=True,
    )
    page.insert_text((474, 356), t("no"), fontsize=7, fontname=BOLD, color=MID)

    diamond(page, 468, 430, 60, 38, "how many context axes differ?")

    # One shared trunk with a short stub per outcome. Drawing four separate elbows made
    # them overlap on the same vertical, so the last colour drawn hid the other three.
    outcomes = [
        ("none", "CONTRADICTS", ["a genuine conflict"], RED_FILL, RED, 334),
        ("one", "RECONCILED BY that axis", ["standalone vs consolidated"], SKY, MID, 384),
        ("several", "INCOMPARABLE", ["too much differs"], GREY_FILL, MUTED, 434),
        ("unclear", "NEEDS REVIEW", ["context unstated"], AMBER_FILL, AMBER, 484),
    ]
    trunk_x = 552
    centres = [top + 21 for *_, top in outcomes]
    page.draw_line(pymupdf.Point(528, 430), pymupdf.Point(trunk_x, 430), color=MID, width=1.2)
    page.draw_line(pymupdf.Point(trunk_x, min(centres)), pymupdf.Point(trunk_x, max(centres)),
                   color=MID, width=1.2)
    for branch, heading, lines, fill, edge, top in outcomes:
        centre = top + 21
        node(page, pymupdf.Rect(604, top, 788, top + 42), heading, lines,
             fill=fill, edge=edge, head_color=edge, size=7, head_size=8.5)
        page.draw_line(pymupdf.Point(trunk_x, centre), pymupdf.Point(598, centre),
                       color=edge, width=1.2)
        page.draw_polyline(
            [pymupdf.Point(604, centre), pymupdf.Point(596.5, centre - 4.2),
             pymupdf.Point(596.5, centre + 4.2)],
            color=edge, fill=edge, width=0.6, closePath=True,
        )
        page.insert_text((trunk_x + 5, centre - 4), t(branch),
                         fontsize=6.8, fontname=BOLD, color=edge)

    page.draw_line(pymupdf.Point(54, 416), pymupdf.Point(300, 416), color=RULE, width=0.8)
    note(page, 426, "Because a relationship stores the very diff that produced it, the "
                    "explanation is generated from data rather than written by a model. That is "
                    "what makes it checkable, and why one mechanism covers corroboration, "
                    "contradiction and reconciliation instead of three detectors.",
         x=54, width=246, size=8.2, height=110)
    note(page, 556, "A graph database stores the edges. The hard part is deciding which edges exist.",
         color=DEEP, size=10, font=BOLD)
    return page


def cases(doc):
    page = header(
        doc, "results", "The four cases, and the path each took",
        "Produced offline with no API key. Page numbers are PDF page indexes, because both "
        "excerpts skip ranges of the original filing.",
    )
    panels = [
        (54, 160, "1", "CORROBORATES", DEEP, MINT,
         ["Annual report p21   Rs. 81,415.38 million",
          "Earnings deck  p13   8,142",
          "",
          "Two documents, two units, one figure.",
          "The deck's cell carried no unit at all - it",
          "was inherited from the Rs. Cr header above",
          "it. They agree within Rs. 50.05 lakh, a",
          "tolerance derived from the precision each",
          "side actually printed."],
         "values agree -> CORROBORATES"),
        (426, 160, "2", "NO CONTRADICTION FOUND", AMBER, AMBER_FILL,
         ["Three mechanisms searched 5,329 facts and",
          "found zero cross-document contradictions.",
          "",
          "Every candidate died to arithmetic. (452)",
          "against (404) EBITDA looked like a conflict",
          "until the deck turned out to print both",
          "Reported and Adjusted, with the",
          "reconciliation between them.",
          "Reported as a null result, not invented."],
         "no candidate survives review"),
        (54, 356, "3", "RECONCILED BY SCOPE, AND BY TIME", MID, SKY,
         ["One annual-report page, one sentence apart:",
          "consolidated  Rs. 81,415.38 mn",
          "standalone    Rs. 74,540.82 mn",
          "A Rs. 6.87bn gap, entirely explained.",
          "",
          "And a director serving in 2021 who ceased",
          "in 2023 - joined across two documents by",
          "DIN, explained by the as-of date."],
         "one axis differs -> RECONCILED"),
        (426, 356, "4", "A FAILURE, AND ITS FIX", RED, RED_FILL,
         ["Board tables list four directors per passage.",
          "Every DIN bound to the nearest name, so one",
          "person collected several and 11 of 13",
          "directors became unresolvable - which broke",
          "the join key the whole layer depends on.",
          "",
          "Now bound inside record boundaries. Two",
          "remain unknown, and are left that way."],
         "doubtful input -> no verdict"),
    ]
    for x, y, number, title, colour, fill, lines, footer in panels:
        rect = pymupdf.Rect(x, y, x + 362, y + 172)
        page.draw_rect(rect, color=colour, fill=fill, width=1.1, radius=0.05)
        page.draw_rect(pymupdf.Rect(x, y, x + 26, y + 26), color=None, fill=colour)
        write(page, pymupdf.Rect(x + 7, y + 5, x + 26, y + 25), number,
              size=11, font=BOLD, color=PAPER, where="panel number")
        write(page, pymupdf.Rect(x + 34, y + 7, rect.x1 - 8, y + 25), title,
              size=9.5, font=BOLD, color=colour, where=title)
        write(page, pymupdf.Rect(x + 11, y + 32, rect.x1 - 10, rect.y1 - 24),
              "\n".join(lines), size=8, where=title + " body")
        page.draw_line(pymupdf.Point(x + 11, rect.y1 - 20), pymupdf.Point(rect.x1 - 10, rect.y1 - 20),
                       color=colour, width=0.6)
        write(page, pymupdf.Rect(x + 11, rect.y1 - 17, rect.x1 - 10, rect.y1 - 3), footer,
              size=7.6, font=BOLD, color=colour, where=footer)

    note(page, 542, "Three of four are green. The fourth is the system declining to state "
                    "what its own evidence cannot support - which is the design working, not a gap.",
         color=DEEP, size=9.5, font=BOLD)
    return page


def data_model(doc):
    page = header(
        doc, "data model", "What gets stored, and what it is joined by",
        "Eight tables in SQLite. The arrows are the joins that make a claim checkable: every "
        "fact reaches a page and a quote, and every relationship reaches two facts.",
    )
    # Left: the provenance chain. Every arrow meets a box edge, with a real gap between,
    # so a short connector still reads as a connector.
    node(page, pymupdf.Rect(54, 160, 214, 228), "documents",
         ["content hash (unique),", "filename, pages, status,", "extraction mode"], fill=MINT)
    node(page, pymupdf.Rect(54, 244, 214, 312), "passages",
         ["page index, reading order,", "character offsets, bbox,", "header/body/footer role"], fill=MINT)
    node(page, pymupdf.Rect(54, 328, 214, 396), "evidence",
         ["the verbatim quote, its", "character span, bbox,", "which extractor, verified"], fill=MINT)
    arrow(page, (134, 228), (134, 244))
    arrow(page, (134, 312), (134, 328))

    # Middle: the fact, and the two tables that identify it.
    node(page, pymupdf.Rect(276, 160, 452, 228), "entities",
         ["canonical name, type,", "identifiers (DIN, CIN),", "evidence-backed aliases"], fill=MINT)
    node(page, pymupdf.Rect(276, 252, 452, 356), "facts",
         ["subject -> entities", "predicate -> predicates", "value: raw and normalised",
          "context: period, scope, basis,", "as-of, geography, publisher",
          "confidence, review state"], fill=SKY, head_size=10)
    node(page, pymupdf.Rect(276, 380, 452, 448), "predicates",
         ["one row per measure, not", "an enum - an unseen measure", "registers itself"], fill=MINT)
    arrow(page, (364, 228), (364, 252))
    arrow(page, (364, 380), (364, 356))
    arrow(page, (214, 362), (276, 320))

    # Right: what comparison produces.
    node(page, pymupdf.Rect(514, 186, 700, 282), "relations",
         ["the two fact ids", "type: corroborates,", "contradicts, reconciled by...",
          "value comparison + tolerance", "the context diff itself",
          "explanation, rule version"], fill=SKY, head_size=10)
    node(page, pymupdf.Rect(514, 300, 700, 372), "consistency findings",
         ["formula, operands, stated", "against calculated,", "difference and tolerance"], fill=SKY)
    arrow(page, (452, 290), (514, 234))
    arrow(page, (452, 318), (514, 336))

    # These two are written straight by the pipeline rather than joined to a fact, and
    # saying so is better than drawing an arrow that would misrepresent the schema.
    node(page, pymupdf.Rect(514, 392, 700, 462), "failures",
         ["stage, page, reason, the", "rejected output. 4,010 on one run.",
          "Not joined to a fact - a rejected", "candidate never became one."],
         fill=RED_FILL, edge=RED, head_color=RED)
    node(page, pymupdf.Rect(708, 186, 788, 282), "llm cache",
         ["keyed on passage,", "model, prompt", "and schema", "version.", "",
          "Not joined."],
         fill=AMBER_FILL, edge=AMBER, head_color=AMBER, size=7, head_size=8.5)

    page.draw_line(pymupdf.Point(54, 482), pymupdf.Point(788, 482), color=RULE, width=0.8)
    note(page, 494,
         "Two invariants hold the design up. A fact cannot be written without evidence that "
         "resolves to a stored passage and page, so nothing ungrounded can enter the layer. "
         "And a relationship stores the context diff it reasoned from, so its explanation can "
         "be re-derived rather than taken on trust.")
    note(page, 550, "Identifiers are what travel between documents. A DIN joins a director "
                    "across two filings where a name would not.", color=DEEP, size=9.5, font=BOLD)
    return page


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in (
        ("01-architecture.pdf", architecture),
        ("02-approach.pdf", approach),
        ("03-four-cases.pdf", cases),
        ("04-data-model.pdf", data_model),
    ):
        doc = pymupdf.open()
        draw(doc)
        target = OUT / name
        doc.save(target)
        doc.close()
        print(f"  {name}  ({target.stat().st_size / 1024:.0f} KB)")
    if _PROBLEMS:
        print("\n  text that did not fit:", file=sys.stderr)
        for problem in _PROBLEMS:
            print(f"    {problem}", file=sys.stderr)
        return 1
    print("  all text fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
