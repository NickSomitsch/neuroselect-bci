# ADR 0031: Verifiable manuscript drafting and assembly

- Status: Accepted
- Date: 2026-07-29

## Context

The publication display provides exact tables, figures, captions, evidence roles, and checksums, but
a journal manuscript adds another opportunity for drift. Values can be transcribed incorrectly,
citations can be omitted, exploratory findings can be promoted to primary evidence, and a polished
document can be mistaken for submission approval.

## Decision

Maintain the journal-neutral prose in `paper/manuscript.md`, the bibliography in
`paper/references.yaml`, every principal quantitative statement in
`paper/manuscript-claim-ledger.yaml`, and synchronized journal-editable LaTeX/BibTeX forms under
`paper/latex/`. Assemble any final manuscript format only after verifying:

1. the exact clean publication-display manifest and display configuration;
2. one occurrence of every locked table and figure marker;
3. every citation key and first-citation numbering;
4. every registered source value and its required human-readable phrase;
5. the current Git revision and source-tree cleanliness; and
6. the four external submission gates from the publication protocol.

The Word renderer follows the `narrative_proposal` document preset with a named
`journal-manuscript` override: US Letter, one-inch side margins, Calibri 11-point justified body,
blue restrained headings, compact captioned tables, full-width publication figures, a running
header, and page numbers. The cover uses the restrained `editorial_cover` pattern without a
decorative hero image.

The LaTeX renderer uses a journal-neutral `article` layout with one-inch US Letter margins,
numbered sections, native citations, BibTeX, vector PDF figures, booktabs tables, and landscape
pages for wide tables. Tectonic compilation is mandatory. The assembler fails if the committed
LaTeX/BibTeX files differ from the verified Markdown, reference registry, or publication display.

The artifact inventory records `assembly_ready` and `submission_ready` separately. A clean verified
document may be assembly-ready while submission remains blocked by open-access confirmation,
institutional secondary-use wording, independent domain review, and final author metadata.

## Consequences

The paper now has reproducible DOCX, Markdown, LaTeX, BibTeX, and compiled PDF forms sourced from
the same evidence. The self-contained LaTeX bundle includes all five vector figures and all ten
tables, including negative, heterogeneous, secondary-comparator, and exploratory findings. The
quantitative claim ledger prevents the principal result language from silently diverging from the
CSV or tracked configuration.

Document rendering does not confer scientific validity, ethics approval, authorship agreement, or
journal acceptance. After committing the assembly implementation, the final bundle must be rebuilt
from the clean commit. The next plan step is independent scientific review and journal-specific
submission preparation, not another evidence-generating experiment.
