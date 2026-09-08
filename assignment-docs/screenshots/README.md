# Dashboard screenshots

Seven captures, in the order a reviewer clicks through the app.

| Image | What it shows |
|---|---|
| [`01-overview.png`](01-overview.png) | Landing page: the drop zone and the four stat cards |
| [`02-documents.png`](02-documents.png) | Documents tab, with page counts, fact counts and extraction mode |
| [`03-facts.png`](03-facts.png) | Facts list: subject, predicate, value, confidence, review state |
| [`04-fact-evidence.png`](04-fact-evidence.png) | One fact opened - raw and normalised value, context envelope, and the verbatim quote with its page |
| [`05-relationships.png`](05-relationships.png) | Relationships list, showing the verdict badges side by side |
| [`06-relationship-explained.png`](06-relationship-explained.png) | One relationship opened - compared values, the context difference, and both quotes |
| [`07-failures.png`](07-failures.png) | Failures tab: uncertainty recorded with a page and a reason |

## The two that carry the most weight

`06-relationship-explained.png` is the single most important image in the submission. It
shows the thing the assignment actually asks for: two figures, the context axis that
separates them, and the evidence for each - none of it narrated by a model.

`07-failures.png` matters for the opposite reason. Showing what the system could not read
is a deliberate decision, and it only reads as deliberate when you can see it.

## Known gaps in this set

Worth stating rather than hiding, since the captures are part of the submission.

- **`04-fact-evidence.png` shows a content-hash filename in the Publisher field.** That was
  a real bug: an upload is stored under a generated name, and re-ingesting that path stamped
  the hash onto every fact's context. It is fixed, but only for documents ingested after the
  fix - these facts predate it. Re-shoot after a fresh ingest to show the real filename.
- **Both documents here are macroeconomy reports**, so the relationship predicates read
  `per cent` against itself. The Delhivery set gives stronger screens: standalone versus
  consolidated revenue reconciled by scope, and Rs. 8,142 Cr matched against
  Rs. 81,415.38 million.
