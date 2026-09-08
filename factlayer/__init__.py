"""factlayer — a fact knowledge layer over PDF documents.

Extracts facts from PDFs, grounds every fact to a verbatim span in its source,
and explains how facts relate: corroborating, contradicting, or reconcilable
through context (period, scope, basis, as-of date).

The design turns on one idea: a fact is a value *plus the context envelope that
makes it comparable*. Relation discovery is then a single rule — establish
comparability, then diff the context envelopes — rather than three separate
special-cased detectors. See `factlayer.link`.
"""

__version__ = "0.1.0"
