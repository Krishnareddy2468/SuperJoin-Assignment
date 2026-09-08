# The four required cases

Everything below was produced by the offline pipeline with **no API key**, and can be
reproduced with the commands in [`../README.md`](../README.md). Page numbers are **PDF page
indexes**, not the numbers printed on the page: both curated excerpts skip ranges of the
original filing, so printed numbers would send a reviewer to the wrong sheet.

Nothing here is special-cased. The same comparison rule produced every relationship, and the
same rule running over the India macroeconomy set finds relationships there too without a
line of document-specific code.

Supporting output: [`../reports/delhivery-evidence-pack.json`](../reports/delhivery-evidence-pack.json).

---

## Case 1 — corroborated across documents, expressed differently

**Verdict: `CORROBORATES`, confidence 0.81**

| | Annual report FY24 | Q4 FY24 earnings deck |
|---|---|---|
| PDF page | 21 | 13 |
| Quote | "the revenue from operations on consolidated basis for FY24 stood at **₹ 81,415.38 million**" | "**8,142**" (cell under the table's `₹ Cr` header) |
| Normalized | 81,415,380,000 INR | 81,420,000,000 INR |
| Period | FY 2023-24 | FY 2023-24 |

> total revenue from contracts with customers: 81420000000 INR and 81415380000.00 INR agree
> within a tolerance of 5005000.00; material context is compatible.

**Why this is not a string match.** Three separate things had to happen for these two to meet:

1. **Scale.** One figure is in millions, the other in crore. Both normalize to base rupees.
2. **A unit stated once, elsewhere.** The deck cell is literally the characters `8,142`. Its
   unit lives in the table's corner header (`₹ Cr`) and is inherited by every cell beneath it.
   Without that step the fact is a bare number and can never be compared to anything.
3. **Wording.** The report writes *revenue from operations*, the deck *revenue from customers*.
   These are the same Ind AS line item under two house styles, so a small vocabulary table maps
   them to one predicate. `revenue from services` is deliberately **not** in that table — decks
   define it as excluding traded goods, so it is a narrower measure even when the numbers land
   close together.

**The tolerance is derived, not guessed.** ₹5,005,000 is the sum of each side's rounding
precision: a figure printed as whole crore carries ±₹0.5 crore, one printed to two decimal
places of a million carries ±₹5,000. The two differ by ₹4,620,000, which is inside that. Had we
compared them with a fixed percentage tolerance we would have got the right answer for the wrong
reason.

---

## Case 2 — a likely contradiction candidate

**Verdict: `LIKELY_CONTRADICTION` candidate, confidence 0.45; requires review.**

The Q4 FY23 earnings table gives **total income of ₹1,934 Cr** (PDF page 16). Re-adding the
grounded rows that the extractor found gives **₹3,795 Cr**:

| Grounded row | Value |
|---|---:|
| Revenue from operations | ₹1,860 Cr |
| Revenue for services | ₹1,860 Cr |
| Other income | ₹75 Cr |
| Revenue from traded goods | ₹0 Cr |
| **Recomputed total** | **₹3,795 Cr** |

The stated total and recomputed total refer to the same table, quarter, unit, and source passage,
so the ₹1,861 Cr gap is a **likely inconsistency** worth showing to a reviewer. The consistency
checker keeps it in `NEEDS_REVIEW` rather than asserting a definitive contradiction because
`revenue from services` is a subtotal of `revenue from operations`, and the extracted component
list is not proven complete. That uncertainty is the point of the candidate: it shows the system
can surface a grounded contradiction-shaped result without pretending that an extraction error is
a fact about the filing.

The source evidence is directly inspectable in the running UI and the source PDF: each number has
a verified quote, PDF page, table coordinates, and the arithmetic explanation.

**What searched for it.** Three independent mechanisms:

- **The comparison rule.** A confirmed contradiction needs the same subject and measure, values that
  disagree beyond their rounding precision, and context envelopes that are genuinely equivalent.
  The refreshed run now gives the three macro reports a shared `India` subject and surfaces
  cross-document candidates, but many remain in review because their period or basis is incomplete.
  The consistency layer still surfaces the grounded candidate above rather than treating every
  noisy candidate as a filing error.
- **Arithmetic self-checks.** Growth claims are recomputed from the operands the documents
  themselves publish. On the Delhivery set this fired once and *confirmed* the filing:

  ```
  stated 12.68%   recomputed 12.69204...%   tolerance 0.0197pp   -> CORROBORATES
  operands: 8,142 (FY 2023-24) and 7,225 (FY 2022-23)
  ```

  Note this check crosses documents: the stated growth comes from the annual report and the two
  operands from the earnings deck.

- **Component totals.** Every stated total in a statement column is re-added from the line items
  printed above it, using each cell's grid position rather than its wording, because row labels
  repeat down a statement. This raises **15 candidates** on the Delhivery set and promotes none of
  them, which is the correct outcome and worth explaining.

  An earlier version did promote ten of them to `LIKELY_CONTRADICTION`, and every one was our own
  fault rather than the document's:

  ```
  total_income = 8,594   components: revenue_from_services 8,142
                                     revenue_from_customers 8,142   <- the same money, twice
                                     other_income 453
  ```

  An income statement prints subtotals among its line items - "revenue from customers" already
  contains "revenue from services" - so a naive sum double counts. Elsewhere the components were
  simply incomplete, because a row we failed to parse is invisible to the check: one expense total
  was tested against five of its line items with freight, the largest, missing. Both faults look
  exactly like a statement disagreeing with itself.

  Completeness is therefore never inferred from the rows we happened to read. The check reports the
  arithmetic and leaves the verdict at `NEEDS_REVIEW`, and promoting a candidate stays a judgement
  a person makes after reading the statement. Counting parsed rows is not evidence about unparsed
  ones.

**The candidates, and why each was rejected.** All three survived a first glance and none
survived arithmetic:

| Candidate | Looks like | Actually |
|---|---|---|
| EBITDA "increased by Rs. 578 Cr to Rs. 127 Cr from Rs. (452 Cr)" | 127 − (−452) = 579, not 578 | All three figures are whole crore, so each carries ±0.5 Cr. The ±1.5 Cr combined tolerance covers a 1 Cr gap. |
| Female workforce growth stated as both "60%" and "59%" | a document disagreeing with itself | Operands 5,594 and 3,519 give 58.97%. "59%" is correct to the nearest point; "60%" is a coarser rounding of the same number, not a rival claim. |
| FY23 EBITDA appears as both `(452)` and `(404)` | two values for one metric | `(452)` is **Reported** EBITDA and `(404)` is **Adjusted**. The deck prints the reconciliation between them. Our own weak label `cr` collapsed two measures — an extraction fault, not a document conflict. See Case 4. |

**The 13 relationships the system does label `contradicts`** in the Delhivery layer are all the
third kind: table rows whose labels (`additions`, `balance_as_at_march`) repeat down a statement
and get flattened into one predicate. They are our defects, and we say so rather than dressing
them up as findings.

**What would trigger a confirmed one.** A stated total that its own components do not sum to, a
growth figure the underlying values contradict, or the same metric at identical period, scope and
basis carrying different values in two filings. The machinery for the first two exists and is
tested; the Delhivery candidates above remain explainable after source review, while the macro run
now keeps its noisier cross-document candidates in the review queue.

**The honest conclusion is itself the finding.** The candidate is useful precisely because it is
not silently promoted: the evidence and arithmetic are visible, while the subtotal and incomplete
row coverage remain explicit reasons for human review. Nearly every other apparent conflict is a
context difference, which is why the context envelope, not a contradiction detector alone, is the
centre of this design.

---

## Case 3 — apparent contradictions explained by context

### 3a — explained by scope · `RECONCILED_BY_SCOPE`, confidence 0.85

Both figures sit on **PDF page 21 of the same annual report**, one sentence apart:

| | Quote | Normalized | Scope |
|---|---|---|---|
| A | "revenue from operations on **consolidated** basis for FY24 stood at ₹ 81,415.38 million" | 81,415,380,000 INR | consolidated |
| B | "revenue from operations on **standalone** basis for FY24 stood at ₹ 74,540.82 million" | 74,540,820,000 INR | standalone |

> ...refer to different scope contexts (consolidated versus standalone), which explains why they
> should not be treated as the same observation.

A ₹6.87 billion gap on the same metric, in the same period, in one document. Same subject, same
predicate, values disagree far beyond tolerance — a contradiction detector with no notion of
scope reports a serious conflict here. The envelope differs on exactly one axis, so the system
names that axis instead.

This case is also why same-document comparison is enabled by default. Restricting comparison to
different documents would have hidden it completely.

### 3b — explained by time · `RECONCILED_BY_AS_OF`, confidence 0.60

| | Annual report FY24 (p23) | Prospectus 2022 (p84) |
|---|---|---|
| State | `ceased` | `serving` |
| As of | 2023-09-27 | 2021-12-24 |
| Quote | "Donald Francis Colleran, Non-Executive Director (DIN: 09431299), was liable to..." | "Donald Francis Colleran ... Period of Directorship: Since December 24, 2021 ... DIN: 09431299" |

> Board membership status: ceased and serving refer to different as of contexts
> (2023-09-27 versus 2021-12-24), which explains why they should not be treated as the same
> observation.

The two documents describe the same person's board seat and say different things. They are joined
across documents by **DIN**, not by name — a stable identifier survives spelling and formatting
differences that a name does not — and `as_of` is identified as the single axis that explains the
difference. A director listed as serving in 2021 and ceased in 2023 is a state change, not a
conflict.

**Getting here required fixing how identity is bound.** The prospectus lists up to four directors
per extracted passage, and each record is a run of fields ending in that director's DIN. The
original rule attached every DIN to the nearest preceding titled name, so one person collected
several DINs and entity resolution — correctly — refused to merge them, leaving nearly every
director as an unresolvable review entity. Binding now respects record boundaries: the DIN before
this one marks where this record began.

**Two directors are still not resolved, and are left that way.** Suvir Suren Sujan's record starts
on page 84 and ends with his DIN on page 85; Sandeep Kumar Barasia's splits the same way. Their
facts stay `needs_review` with *"A DIN was found, but the person's name could not be resolved
nearby"*, and their relationships stay unstated. The information really is split across a page
boundary, and inventing the link would be worse than admitting it.

---

## Case 4 — an extraction failure, and what we did about it

**The failure: one DIN, bound to the wrong person, in a table listing four directors at once.**

The prospectus prints each director as a run of fields that ends with their DIN, several records to
an extracted passage. The original rule looked backwards from a DIN for the nearest titled name
anywhere on the page. In a passage holding four records that meant **every DIN in the passage
resolved to the same name**, so one person accumulated several DINs.

Entity resolution then did the right thing for the wrong reason: it saw conflicting identifiers for
one name, refused to merge, and produced review entities. Eleven of thirteen directors ended up
unresolvable, and because DIN is the join key the whole knowledge layer depends on, every
cross-document director relationship was blocked. **Case 3b was collateral damage from this bug,
not a separate problem.**

**A false trail worth recording.** The first diagnosis was wrong. The interleaved text

```
Period of Directorship: Since September 9, 2014
DIN: 01173669
Kalpana Jaisingh Morparia
```

reads exactly like a mis-binding — a DIN stranded between another director's date and that
director's name — and it was written up as one. Reading the underlying PDF blocks disproved it:
records run *name → fields → DIN*, so `Since September 9, 2014` and `DIN: 01173669` both belong to
Suvir Suren Sujan, whose name sits at the foot of the previous page, and Morparia's record simply
begins on the next line. The extraction was right and the explanation was wrong. It is recorded
here because a plausible story about a defect is not evidence of one.

**How it is handled now.**

1. **Identity is bound within record boundaries.** The previous DIN marks where this record began,
   so a DIN can only take a name from its own record. Every resolved director now carries the
   correct DIN: Bharati → 02227607, Colleran → 09431299, Barua → 05131571, and so on.
2. **Names are found by structure, not vocabulary.** A record opens with the name, directly above
   the first `Label: value` line. Company names bleeding in from the neighbouring "Other
   Directorships" column are not followed by a label, which excludes them without maintaining a
   blocklist of words like *Properties* or *Council*. An intermediate version used a word blocklist
   and confidently mislabelled Sujan as **"East Group Properties"** — a reminder that a heuristic
   which fails loudly is safer than one that fails plausibly.
3. **Doubt travels and blocks conclusions.** Where a name genuinely cannot be recovered — a record
   split across a page break — the fact stays `needs_review`, the warning rides along with the
   status fact as well as the identifier fact, and any relationship touching it is withheld.

**Residual, and the next fix.** Two of thirteen directors remain unresolved because their records
straddle a page boundary. Carrying the previous page's tail into the search helps for records that
end near the top of a page but does not cover every layout. The next step is to stitch records
across page boundaries during ingestion rather than during extraction.

**A second, quieter failure worth naming.** 1,986 of 2,691 Delhivery relationships sit in
`needs_review`, and all 13 labelled `contradicts` are artefacts of table rows whose labels
(`additions`, `balance_as_at_march`, `% margin`) do not name a measure on their own. The system
prefers an honest "cannot tell" to a confident guess, but the volume shows how much of the
remaining work is label quality rather than comparison logic.
