# Issues I ran into, and how I fixed them

This is the honest version of the build. Not a polished changelog, but the actual problems
found while building this, in the order they showed up, with the reasoning behind each fix.
Every number here can be checked against the committed evidence packs and the test suite.

## Why the system defaults to offline mode

This is the question that started this document, so it goes first.

The pipeline has two extractors. A deterministic one reads regex, table geometry, and unit
rules, and it needs no API key at all. An optional one calls Gemini for facts that rules
genuinely cannot read, like a role appointment or a caveat buried in a sentence.

I default to the deterministic path for one reason: anyone evaluating this should be able to
reproduce every result without paying for anything or trusting my account. The brief itself
says that if a project needs a paid service, I should include enough output for someone to
judge it without needing my credentials. Offline mode is how I satisfy that honestly rather
than by just attaching screenshots.

It is not a workaround for something broken. Every one of the four required cases in this
submission, and every number quoted in the README, comes from the offline path. The optional
model adds semantic facts on top of that when a key is present, but it is never the thing being
demonstrated as working.

## The entity that quietly poisoned a whole document

The first real bug. An entity's "review state" was being copied from whichever fact happened to
mention it. So if one shaky extraction touched a company or a person, that entity got flagged
for review, and every later fact resolved against that same entity inherited the flag. One weak
line in a document was enough to push the whole document's relationships into needs review,
which is why an early run showed almost nothing corroborating or reconciling even though the
underlying facts were fine.

The fix separated two different questions that had been treated as one: is this identity
settled, versus was this particular mention confident. Review state now only tracks the first
one. A test pins this down so it cannot regress quietly again.

## Same document comparisons were either missing or too aggressive

Standalone and consolidated revenue for the same company sit on the same page of an annual
report. Comparing facts only across different documents meant that reconciliation was
impossible to find, because both numbers lived in one file.

Turning on same-document comparison without a second check was worse. Facts sitting in the same
table row, like a margin percentage repeated across eight quarterly columns, started getting
compared against each other and produced over seventeen hundred contradictions that no document
had actually stated. A percentage column is not eight rival answers to one question, it is eight
different quarters.

The fix requires two facts to come from separate passages before they count as separate claims,
on top of sharing a subject and a predicate. That single rule cut a large batch of manufactured
contradictions down to thirteen, and every one of those thirteen is now a genuine table
extraction artifact I document openly rather than a false positive I hid.

## A Gemini schema that was failing every single call

Once a key was added, the model extractor looked like it was falling back to offline silently on
every document. It was not falling back gracefully, it was failing every request outright and
swallowing the error into a generic offline message. The actual API response was rejecting the
request schema, because a strict validation model renders as `additionalProperties: false` and
the Gemini API does not accept that field.

The fix was to hand write the schema sent to the model, separate from the schema used to
validate what comes back. Those are two different jobs and only one of them has to satisfy a
remote API's particular dialect. Once that was fixed, a live run kept over thirty facts from the
model and rejected roughly a third of the rest for quoting text that was not actually in the
source, which is the grounding check doing its job rather than the request failing before it
ever reached the model.

## Hybrid mode used to make a document look like it had hung

Before it was bounded, a call to the model extractor ran one passage at a time. A twenty seven
page deck took over a minute and a half. A ninety page report would have taken somewhere between
eight and fifteen minutes, which is functionally indistinguishable from the process being stuck.

The fix bounds three things: how many model calls run at once, how many calls a single document
may spend in total, and how long the whole model phase is allowed to take before the run moves
on with whatever it already has. The same deck now finishes in under thirty seconds, and if a
cap is hit partway through, the run says so explicitly instead of pretending nothing was left
out.

## Board tables that quietly mixed up directors

A prospectus lists several directors one after another in the same block of extracted text, each
one ending in their own identification number. The first version of the binding logic looked
backward from a number for the closest name on the page, which meant that in a passage holding
four directors, every single one of those numbers bound to whichever name happened to be nearest,
regardless of whose record it actually belonged to.

The result was that eleven of thirteen directors became impossible to identify consistently,
because the same identification number kept attaching to different names depending on where it
sat in the text. That number is the key that joins a person across two different documents, so
this one bug quietly broke every cross document relationship about a director.

The fix binds each number within its own record, using the previous number in the passage as the
boundary of where the current record begins. Two directors are still not identifiable, because
their record happens to be split across a page break, and I left that as a stated limitation
rather than guessing at a name that is not actually there to find.

## Names that could be confidently wrong

While fixing the binding above, an early attempt used a list of words to reject things that
looked like a person's name but were not, like a company name bleeding in from a neighboring
column. That approach genuinely mislabeled one director as the name of an unrelated company,
confidently and without any warning attached.

I replaced the word list with a rule based on the shape of the text instead: a name is the line
sitting directly above a field like "Designation" or "Address", not just any capitalized phrase
nearby. A wrong guess that fails loudly, by leaving a fact unresolved and flagged, is worth far
more than a wrong guess that fails quietly and looks like a correct answer.

## Every uploaded fact carried the wrong publisher

An uploaded file is stored on disk under a generated name based on its content, so that the same
file uploaded twice is recognized as a duplicate. The processing step was re-reading the document
from that generated name and using it as the publisher recorded on every fact, so a fact from a
file called "quarterly-update.pdf" ended up tagged with a publisher that read like a long string
of letters and numbers instead.

The fix carries the document's real registered name into extraction instead of re-deriving it
from the stored path. A test now uploads a real file and checks that every fact it produces
credits the actual filename.

## The browser badge that read like a contradiction

The little status indicator in the corner of the page used to read "Online, offline extraction
mode" at the same time, which looks like it contradicts itself even though it does not. Those two
words were describing two unrelated things: whether the browser had reached the server at all,
and whether the model extractor happened to be turned on. It also only ever mentioned the mode
when the model was off, so a session running with a key never told you that out loud.

I separated the two ideas and made the wording symmetric either way, so it now reads "system
ready, offline mode" or "system ready, hybrid mode" depending on what is actually configured.

## A fix that looked like it had not applied

Right after that, a screenshot still showed the old wording even though the file on disk and the
running server were both already correct. The cause was not the fix. The static file server was
sending caching information but no explicit caching policy, which let the browser reuse an old
cached copy of the page's script on a plain reload without even checking back with the server.

The fix sets an explicit no caching instruction on every static file the server sends, so a
plain reload is guaranteed to fetch whatever is actually running, not whatever the browser
remembers from before.

## One bad fact threw away a hundred pages of good ones

The store deliberately refuses to let a single predicate mean two different kinds of value.
If a measure has been recorded as a number, it cannot later be stored as a category, because
comparing those two would be meaningless. That guard is correct and worth keeping.

The problem was what happened when it fired. Nothing caught the refusal, so it travelled all
the way up and aborted the entire ingest. On the hundred page annual report, a measure normally
reported as a number turned up once as a category somewhere in the document, and the whole run
came out marked as failed. Every fact extracted before that point was discarded even though
there was nothing wrong with any of them. In the browser the document simply read "Failed" with
a single warning line, which gives no hint that over a thousand perfectly good facts had just
been thrown away because of one mismatch.

The fix catches the refusal around the single fact that caused it, records it as a normalization
failure with its page and the offending predicate, and carries on with the rest of the document.
One mismatched fact now costs exactly that one fact. Rerunning the same annual report afterwards
took it from failed with everything lost, to partial with 1,204 facts kept and the single
conflict listed openly in the failures tab.

The test for this deliberately checks the old behaviour too. Reverting the fix and rerunning the
test makes it fail with the document marked as failed, which is how I know the test is actually
guarding something rather than just passing.

## A live server that could not survive its own database being wiped

The schema for the store was only built once, right when the process started. If the
underlying database file got deleted and recreated afterward, for example by rerunning a
setup script that clears the data directory while a server was still pointed at it, the next
request opened a fresh connection to a brand new, empty file with no tables in it at all.
Every request after that failed with a database error, which looks identical to the API being
down, and the only fix was to notice and restart the process.

The fix checks the schema on every connection instead of only at startup. Every step of that
check was already written to be safe to run more than once, so this costs a handful of cheap
lookups per request and makes a live server recover from a wiped database file on its own,
without needing to be restarted. A test deletes the file out from under a running store and
checks that the same instance keeps working afterward.

## Predicate labels are still the weakest part of the system

Not fixed, and worth stating plainly rather than glossing over. A very large share of
relationships, close to two thousand out of twenty seven hundred on the primary dataset, sit in a
needs review state because a label taken from a table row does not name a measure clearly enough
on its own. A column heading like "percentage margin" repeated under different unlabeled
sections is the most common shape of this problem.

On the second, previously unseen dataset the same weakness compounds. Only a small fraction of
facts carry an identifiable reporting period, and a large majority of the predicates discovered
there are used exactly once, which means they can never be matched against anything else. Fixing
predicate labels to carry the section or column heading they sit under is the single change most
likely to improve both problems at once, and it is the next thing I would build.

## What this list is meant to show

None of these were found by getting the design right on the first attempt. Nearly all of them
were found by running the actual pipeline against the actual PDFs and reading the actual output,
not by reasoning about the code in the abstract. The corroboration, contradiction, and
reconciliation cases in this submission survived because each of these bugs was caught before it
could quietly shape what the demo showed.
