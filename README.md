# Engineering Intern Hiring Assignment

### Welcome!

This assignment is intentionally open-ended. We want to see how you explore an unfamiliar problem and turn your ideas into something that works.

- Use any language, framework, database, LLM, coding agent, or library.
- We care more about your approach and creativity than production-level polish.
- Be honest about what works, what does not, and what you would improve.

## The Challenge: Build a Fact Knowledge Layer

Important facts are often scattered across documents, stated in different ways, supported by other evidence, or contradicted elsewhere.

You will receive **three PDFs as a starter dataset**. Build a system that:

- extracts meaningful numerical or semantic facts;
- links every fact to evidence in its source document; and
- identifies when facts corroborate, contradict, or can be reconciled through context.

Provide a simple **API or UI** through which we can upload PDFs and inspect the results. We may test your solution with additional PDFs, so it should not rely on hard-coded facts, filenames, schemas, or document-specific rules.

The documents should guide what counts as a fact and how it is represented. Your schema, storage, interface, and output format are entirely up to you.

> **A graph database or visualization alone is not the solution.** The interesting part is how facts are discovered, grounded, compared, and explained.

### Show Us These Four Cases

Your submission should include at least one example of each:

1. A fact corroborated across documents, even if expressed differently.
2. A genuine or likely contradiction.
3. An apparent contradiction explained by context, such as time, scope, or units.
4. An extraction or reasoning failure you found and how you handled—or would improve—it.

Show the source evidence and your system's reasoning for the first three.

For inspiration, two revenue figures may differ because they cover different periods; a director may appear active in one document and resigned in a later one; or differently written addresses may refer to the same place. These are examples, not a required data model or checklist of facts.

## What We Are Looking For

- A thoughtful and creative approach.
- Useful facts that are grounded in the PDFs.
- Sensible handling of ambiguity, context, and uncertainty.
- A solution that can generalize beyond the starter documents.
- Clear engineering decisions and trade-offs.

We do not expect perfect extraction or a production-ready system. A smaller, understandable prototype is better than a large system whose behavior is unclear.

## Brownie Points

If the core experience works, try extending it to handle:

- large PDFs without significant performance issues;
- many PDFs in the same knowledge layer;
- a schema that evolves dynamically as new kinds of facts appear; or
- new documents incrementally, without rebuilding all existing knowledge.

These are suggestions, not additional requirements. Feel free to explore another extension that meaningfully improves the core system.

## Submission

Use git meaningfully and complete the Developer's Section below with:

- setup and run instructions;
- your approach, important decisions, and AI tools used;
- known limitations and possible next steps; and
- a demo video of **3 minutes or less** showing a PDF being processed and the required cases above.

Keep credentials out of the repository. If the project requires a paid service, include enough sample output and video footage for us to evaluate it without needing your account.

### Before You Submit

- [x] The project runs from my instructions and accepts new PDFs through an API or UI.
- [x] Results contain facts, source evidence, and cross-document relationships.
- [x] I demonstrate the four required cases — see [`docs/CASES.md`](docs/CASES.md). Case 2 is
      reported as a null result with the candidates and the arithmetic that ruled each one out,
      rather than a manufactured example.
- [x] I have documented my approach and included a demo video, linked under
      [Video Demo](#video-demo).

Most importantly, have fun tinkering. We are excited to see how you think.

## Start here

Four one-page diagrams in [`assignment-docs/`](assignment-docs/) explain the whole system
faster than any prose here can:

- **[Architecture](assignment-docs/01-architecture.pdf)** - the pipeline as a flowchart, including the grounding gate every fact must pass
- **[Approach](assignment-docs/02-approach.pdf)** - why a number is not a fact, with the comparison rule drawn as a decision tree
- **[The four cases](assignment-docs/03-four-cases.pdf)** - what the system reports, including the case it refuses to answer
- **[Data model](assignment-docs/04-data-model.pdf)** - the eight tables, and the joins that make a claim checkable
- **[Issues and solutions](assignment-docs/05-issues-and-solutions.md)** - the real bugs found while building this and how each was fixed, and why the system defaults to offline mode

[`assignment-docs/screenshots/`](assignment-docs/screenshots/) shows the same thing running in
the browser, and [`docs/CASES.md`](docs/CASES.md) has the full evidence behind each case.

## Developer's Section

### Setup and Run Instructions

Python 3.11. Create a virtual environment and install the exact versions I developed against:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

Check it works without configuring anything:

```bash
python -m factlayer.cli --help
python -m factlayer.cli llm-status
python -m pytest
```

`llm-status` will tell you the LLM is off and why. That is fine — the deterministic extractor
is the primary path and needs no credentials. If you want the optional Gemini extractor, copy
`.env.example` to `.env` and add a `GEMINI_API_KEY`. Passing `--no-llm` (or `--offline`)
guarantees no network call either way.

Run the pipeline over a PDF, several PDFs, or a whole directory:

```bash
python -m factlayer.cli --offline --database data/demo.db ingest starter-datasets/delhivery
python -m factlayer.cli --offline --database data/demo.db link
python -m factlayer.cli --offline --database data/demo.db status
python -m factlayer.cli --offline --database data/demo.db report --output data/report.json
```

`ingest` reads and extracts, `link` compares everything and runs the consistency checks,
`status` prints the counts, `report` dumps the lot as JSON. Uploading the same bytes again is
reported as a duplicate and reuses what is already stored; add `--reprocess` when you actually
want to replace a document's results.

For the browser interface:

```bash
uvicorn factlayer.api:app --reload
```

`http://127.0.0.1:8000` is the UI and `/docs` is the interactive API. The UI and the CLI go
through the same service, so they cannot disagree. Uploads are validated, stored under a
content-hash name in the ignored `data/` directory, and deduplicated on re-upload.

Two smaller commands worth knowing: `models` prints the configured model without contacting it,
and `models --check` asks the API which models your key can actually reach. That second one is
worth running once — model names move, and it will tell you if the configured chain is stale.

To regenerate the committed sample output:

```bash
python -m factlayer.cli --offline --database data/demo.db report --output data/report.json
python scripts/build_evidence_pack.py data/report.json reports/delhivery-evidence-pack.json
```

The full report runs to tens of megabytes and stays in the ignored `data/` directory. What is
committed under `reports/` is a condensed pack you can actually open and read.

### Video Demo

**[Watch the demo (Google Drive)](https://drive.google.com/file/d/1K7k-EFeO6UOnD0CFv7fCdgA_6XAlRZl2/view?usp=sharing)**

The walkthrough covers why a number on its own is not a fact, the architecture, why offline is
the default and when the LLM earns its place, and each of the four required cases against the
running system.

### Approach

The whole thing rests on one idea: **a number is not a fact until it carries its context.**

`81,415.38` on its own is meaningless. It only becomes comparable once you also know what it
measures, whose it is, over what period, on what reporting scope and basis, and in what units.
So a stored fact is a value plus that envelope plus its evidence:

```
Fact
  subject     canonical entity      (Delhivery Limited | DIN 01173669)
  predicate   canonical measure     (revenue_from_operations | board_membership_status)
  value       raw text + normalized number, unit, currency, scale
  context     period, scope, basis, as-of date, geography, publisher
  evidence    document, PDF page, verbatim quote, character span, bbox
```

Once facts look like that, one rule produces every relationship. Corroboration, contradiction
and reconciliation are not three features — they are three outcomes of the same comparison.
Establish comparability, compare the normalized values, then diff the two context envelopes:

```
values agree                              -> CORROBORATES
values differ, contexts equivalent        -> CONTRADICTS
values differ, exactly one axis differs   -> RECONCILED_BY_<PERIOD|SCOPE|BASIS|AS_OF|UNIT>
several axes differ, or context missing   -> INCOMPARABLE / NEEDS_REVIEW
```

The relationship stores the diff that produced it, which means the explanation is generated
from data rather than written by a model. That is the part I care about most: you can check the
reasoning instead of trusting it. It is also why one code path covers all four required cases.

Grounding is a precondition rather than a reporting field. Nothing is stored without evidence,
and a fact proposed by the model has to quote a span that really exists in its source passage.
The check runs in code and drops what fails it — on one live run over the earnings deck, 12 of
roughly 40 proposed facts were rejected for quoting text that was not there.

There are two extractors behind one schema. The deterministic one — regex, table geometry, unit
and period normalisation, DIN and CIN anchors — always runs and needs no key. The optional
Gemini one picks up facts stated in prose that rules cannot reach: role appointments, metric
definitions, caveats like *"FY22 numbers are on pro forma basis"*. Both emit the same `Fact` and
both pass the same grounding and normalisation, so a result does not depend on which found it.
Everything demonstrated in this submission runs with no API key.

The diagrams in [`assignment-docs/`](assignment-docs/) show the pipeline and this rule as
flowcharts, which is faster than reading the description above.

#### Decisions worth defending

**SQLite, not a graph database.** The brief is right that a graph is not the answer. The hard
part is deciding *whether* two facts relate, not storing the edge once you know. SQLite keeps
the prototype light and makes content-hash dedupe and incremental ingest almost free.

**Predicates are rows, not an enum.** An unseen measure registers itself. That is what lets the
macroeconomy set produce facts without a migration.

**Facts inside one document are compared too.** Standalone and consolidated revenue sit on the
same page, and restricting comparison to across documents would have hidden the clearest
reconciliation in the corpus. That needed a guard first: cells sharing a passage are siblings of
one table row rather than rival claims, and comparing them naively produced over 1,700
contradictions no document ever made.

**Silence is not disagreement.** A field only one source states cannot explain away a
difference, but it should not veto an agreement either. Missing context blocks a contradiction,
costs confidence, and gets named in the explanation.

**Abstention is a real answer.** The reporting-entity detector returns nothing when no company
clearly leads, so an economic report never gets filed under a data vendor credited in one of its
tables. Several parts of the system would rather say "cannot tell" than guess.

#### AI tools used

I used Claude (Anthropic) throughout, as a pair programmer rather than a code generator:
designing the context-envelope model, writing the pipeline, and — most usefully — auditing its
own output against the actual PDFs. Several real defects came out of that rather than from
tests: a linking rule that manufactured 1,700 false contradictions out of table cells, an
entity-resolution bug where one weak extraction pushed a whole document into review, and a
Gemini request schema that had been failing every single call silently while the pipeline
quietly fell back to deterministic results.

Google Gemini (`gemini-2.5-flash`) is the optional extraction model, behind a provider interface
with a fallback chain.

### Limitations and Next Steps

Worst first.

**I did not find a genuine contradiction, so Case 2 is a null result.** Three mechanisms went
looking across 5,329 facts and 9,472 relationships and turned up zero cross-document
contradictions. Each candidate was ruled out by arithmetic rather than hand-waving — the working
is in [`docs/CASES.md`](docs/CASES.md). Manufacturing one would have been easy and dishonest.
The checker does work: it recomputed a stated 12.68% growth as 12.69% using operands from a
different document.

**Predicate labels are the weakest part of the system, and they cause most of the remaining
noise.** 1,986 of 2,691 Delhivery relationships sit in `needs_review`, and all 13 labelled
`contradicts` are artefacts of table rows whose labels — `additions`, `balance_as_at_march`,
`% margin` — do not name a measure on their own. The fix is to carry the section or column
header into the predicate so a row label is qualified by what it sits under. That single change
would also fix the macro problem below, which is why it is the first thing I would do next.

**The macroeconomy set is where it shows.** Facts extract fine — 3,125 of them, with no
document-specific code — but the layer barely functions on them, for three compounding reasons.
Only 219 carry a reporting period, because institutional reports put periods in table headers
and chart axes rather than in prose. 1,014 of 1,375 predicates are used exactly once, and a
predicate used once can never match anything. And nothing gives the three reports a shared
subject, so all 6,781 relationships sit inside a single document and an IMF figure never meets
an RBI one. Company filings escape this only because a registered name on the cover supplies a
subject. Giving non-corporate documents one — the economy, the country, the programme a report
is about — is the most interesting piece of unfinished work here.

**Two directors out of thirteen are still unidentified.** Board tables list several directors per
extracted passage, each record ending in that director's DIN. Binding identity inside record
boundaries fixed the join key the whole layer depends on and turned the time-based reconciliation
green. But where a record straddles a page break — name at the foot of one page, DIN at the head
of the next — the name still cannot be recovered, so those facts stay in review and their
relationships are withheld rather than guessed. Records need stitching across page boundaries
during ingestion.

**Table-cell evidence offsets can point at the wrong cell.** Grounding finds a cell value with a
plain text search, so it lands on the first occurrence of, say, `4.8%` in the passage. The quote
is genuinely verbatim and the guard passes, but the character span may belong to a sibling cell.
Cell spans should come from the bounding box instead of a search.

**Component-total checks raise candidates and never a verdict.** Both consistency checks find
their own operands now — growth from predicate shape, totals from grid position — and totals
raise 15 same-context candidates on the Delhivery set while promoting none. That is deliberate:
a statement prints subtotals among its line items, and a row we failed to parse is invisible to
the check, so a naive sum either double counts or comes up short. Both look exactly like a
document contradicting itself. Detecting subtotals is what would let a real mismatch be
asserted.

**Scale is untested beyond this corpus.** Six PDFs and roughly 600 pages are comfortable — a
100-page annual report ingests in 15 seconds and 123 MB, and the LLM phase is bounded so a large
upload cannot stall. Nothing here has met a 1,000-page filing or concurrent uploads.

Two smaller things I would also do: persist the grounding rejection count in the run summary so
it can be quoted without re-reading failures, and give a reviewer some way to accept a
`needs_review` relationship, because the review queue currently has no exit.

### Additional Notes

**It runs with no API key.** Every case in [`docs/CASES.md`](docs/CASES.md) comes from the
offline path, so you can evaluate this without credentials or billing. `/health` and the CLI both
report which mode produced a result, so offline output can never be mistaken for model output.

**Where to look first.** [`assignment-docs/`](assignment-docs/) has four one-page diagrams and
the dashboard screenshots. [`docs/CASES.md`](docs/CASES.md) has the four required cases with
quotes, PDF page indexes and the system's own reasoning. [`reports/`](reports/) holds committed
evidence packs for both datasets.

**Hybrid mode.** With a key, model calls run concurrently, one document spends at most 150 of
them, no single call waits longer than 25 seconds and the phase stops at 60. That took a 27-page
deck from 103 seconds to 30, and an 89-page report from an estimated 8–15 minutes to 70. Whatever
a cap skips is reported rather than hidden.

**Every number in this README** comes from the offline run reproduced by the commands above.
