# Assignment docs

Four one-page diagrams. Read them in order and you have the whole system in about
three minutes.

| | | |
|---|---|---|
| **[01 · Architecture](01-architecture.pdf)** | How a PDF becomes comparable knowledge | The six stages, what is optional, and what it costs to run |
| **[02 · Approach](02-approach.pdf)** | A number is not a fact | The context envelope, and the single rule behind every relationship |
| **[03 · The four cases](03-four-cases.pdf)** | What the system actually reports | Real verdicts and quotes, including the one case with no answer |
| **[04 · Data model](04-data-model.pdf)** | What gets stored, and what joins it | The eight tables, and the two invariants that make a claim checkable |

**[05 · Issues and solutions](05-issues-and-solutions.md)** is the honest build log: the real
bugs found while building this, in the order they showed up, and why the system defaults to
offline mode. Worth reading if you want to see how this was actually built rather than how it
looks once finished.

**[Dashboard screenshots](screenshots/)** show the same ideas running in the browser: seven
captures from upload through to a relationship with its explanation, and the failures the
system chooses to show rather than hide. The strongest single image is
[`06-relationship-explained.png`](screenshots/06-relationship-explained.png) - two figures,
the context axis separating them, and the evidence for each.

Longer write-ups live elsewhere: [`../docs/CASES.md`](../docs/CASES.md) has the full evidence
for each case, and the main [`../README.md`](../README.md) covers setup, decisions and limitations.

**01** and **02** are flowcharts: the pipeline with its grounding gate, and the comparison
rule drawn as the decision tree it actually is. **04** is the schema with its joins.

These are generated, not hand-drawn, so they cannot drift from the code. The build also
fails if any label overflows its box, because PyMuPDF silently draws nothing in that case
and an empty box looks deliberate:

```bash
python scripts/build_assignment_docs.py
```
