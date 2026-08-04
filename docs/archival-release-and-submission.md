# Archival release and journal submission

This workflow enforces a maximum mandatory publication fee of €0. The selected route is RBET only
after the University of Innsbruck confirms both truthful affiliation and complete APC coverage in
writing. Otherwise it is a Neuroinformatics Original Article using subscription publication, not
optional open access. No paid fallback is configured.

Authoritative references: [UIBK Taylor & Francis agreement](https://www.uibk.ac.at/en/open-access/funding/publishing-agreements/taylor-and-francis/),
[Neuroinformatics publishing options](https://link.springer.com/journal/12021/how-to-publish-with-us),
[Neuroinformatics submission guidelines](https://link.springer.com/journal/12021/submission-guidelines),
and [Zenodo DOI reservation](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/).

## 1. Complete the external gates

Register and validate an ORCID. Contact `open-access@uibk.ac.at` and the appropriate degree or
departmental contact using the truthful description that the work was conducted independently
while enrolled. Ask about affiliation, RBET coverage, corresponding-author and institutional-email
requirements, the 2026 allocation, and approved secondary-use wording for public, deidentified
Study P data. Do not commit correspondence or personal contact details.

Obtain independent BCI review of EEG terminology, statistics, calibration, replay boundaries,
candidate-generation claims, and limitations. A reviewer remains a reviewer unless they meet all
authorship criteria and accept accountability. Record only the status, date, and a non-private
note in `configs/publication/submission_v1.yaml`.

Fill the ORCID, affiliation, email, funding, competing-interests, ethics, CRediT, and gate fields in
that config. Use `Independent Researcher` if UIBK affiliation is not affirmatively approved. Never
infer declarations from silence.

## 2. Reserve the archive DOI

Create a Zenodo Software draft, reserve its version-specific DOI, and place only that DOI in
`configs/publication/release_v1.yaml`. Do not publish the record yet. Add the same DOI to
`CITATION.cff` only after reservation. Remove or set `date-released` only when the release is
actually published; the current pre-release citation metadata does not invent a publication date.
The final release builder rejects a missing or mismatched DOI, a dirty
worktree, a missing `v0.1.0` tag, inconsistent versions, altered evidence manifests, and restricted
payloads.

During review of the mechanics, a clearly blocked preview can be generated with:

```bash
make publication-release PUBLICATION_RELEASE_ARGS="--allow-pending --overwrite"
make publication-release-check PUBLICATION_RELEASE_CHECK_ARGS="--allow-pending"
```

The preview is not publishable. The final commands omit `--allow-pending`.

## 3. Build the selected journal package

The automatic route is available in the Python interface; explicit Make commands are:

```bash
make journal-submission JOURNAL=neuroinformatics JOURNAL_SUBMISSION_ARGS="--overwrite"
make journal-submission-check JOURNAL=neuroinformatics
```

or, only after both UIBK gates are satisfied:

```bash
make journal-submission JOURNAL=rbet JOURNAL_SUBMISSION_ARGS="--overwrite"
make journal-submission-check JOURNAL=rbet
```

Use `--allow-pending` only for a watermarked-by-metadata development preview. Neuroinformatics is
assembled as an Original Article using the pinned official Springer Nature LaTeX template,
author-year references, a 150–250-word abstract, 4–6 keywords, declarations, individual displays,
and a checksum inventory. RBET is assembled as Original Research with full and anonymous files,
title-page and cover-letter material, a private reviewer archive, and direct-identifier scans.

## 4. Final release order

After external review and final corrections, run the clean repository, manuscript, claim-ledger,
release, and journal checks. Rebuild the release twice and compare archive hashes. Commit the DOI
and final metadata, create annotated tag `v0.1.0`, build once more from that tag, publish the GitHub
release, upload the identical files to the reserved Zenodo record, publish Zenodo, download it, and
verify `SHA256SUMS`. The GitHub tag, Zenodo inventory, and manuscript citation must name the same
40-character commit.

Actual ORCID, email, GitHub release, Zenodo publication, and journal-portal actions remain under the
author's control. If a journal requires an institutional ethics determination that cannot be
obtained, submission stops; the repository never invents an exemption.
