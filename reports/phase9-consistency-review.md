# Phase 9 consistency review

This note records the manual review behind Phase 9. It deliberately separates an apparent
inconsistency from a defensible contradiction. Page numbers below are PDF page positions in the
provided excerpts, so another reviewer can find the same evidence without relying on printed page
labels.

## Strongest candidate: female workforce growth

The FY24 annual-report excerpt uses two nearby descriptions on PDF page 8:

- “increased 60% year-on-year” in the narrative; and
- “increased by 59% year-on-year” in the summary bullet.

PDF page 17 supplies the operands: 5,594 female employees in FY24 and 3,519 in FY23. The recomputed
growth is:

```text
(5,594 - 3,519) / 3,519 × 100 = 58.9656%
```

The 59% statement is the nearest whole-percent result. The 60% narrative is a coarser presentation,
not strong enough to call a genuine contradiction. The system should preserve the two statements
for review, but it must not present the difference as definitive evidence that the underlying fact
is wrong.

## Second candidate: stated EBITDA increase

The FY24 earnings presentation says on PDF page 5 that EBITDA “increased by Rs. 578 Cr to Rs. 127 Cr
from Rs. (452 Cr).” Subtracting the displayed endpoints gives ₹579 crore. However, all three figures
are displayed as whole crores. Each can therefore carry up to ₹0.5 crore of visible rounding error,
giving the check a combined tolerance of ₹1.5 crore. The ₹1 crore difference remains inside that
tolerance and is not a contradiction.

## Confirmed control case

The same presentation reports FY24 revenue of ₹8,142 crore, FY23 revenue of ₹7,225 crore, and 12.7%
year-on-year growth on PDF page 17. Recalculation gives about 12.69%, which supports the stated 12.7%
after rounding. This is retained as a regression test so normal financial-statement rounding does
not produce a false alert.

## Review outcome

The strongest qualified candidate is the Q4 FY23 total-income check on PDF page 16. The filing
states ₹1,934 Cr, while the grounded rows currently recompute to ₹3,795 Cr. The checker leaves it
at `NEEDS_REVIEW` because the extracted component list is incomplete and includes both “revenue
from operations” and its “revenue for services” subtotal. It is therefore presented as a likely
contradiction candidate, with the arithmetic and extraction caveat visible, rather than as a
confirmed filing error.

The checker can detect a clear mismatch when grounded, equivalent-context operands exist;
uncertain mismatches are labelled `LIKELY_CONTRADICTION`, and incomplete or incompatible inputs
are labelled `NEEDS_REVIEW`. This keeps the required likely case honest while preserving the
review queue for the extraction fixes still needed.
