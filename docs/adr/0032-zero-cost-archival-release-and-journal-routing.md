# ADR 0032: Zero-cost archival release and journal routing

- Status: Accepted
- Date: 2026-07-31

## Context

The verified offline manuscript needs an exact archival software/evidence release and a venue
package, but no experiment may be selected or replaced according to outcome. Publication must not
require a fee, institutional identity must not be implied, and double-anonymous review must not be
misrepresented after a public repository already exists.

## Decision

Use RBET only when UIBK confirms truthful student affiliation and full APC coverage in writing.
Otherwise use the Neuroinformatics subscription route. There is no paid fallback. The release is
`v0.1.0`, with an exact tagged source archive and a separate manifest-verified public-evidence
archive deposited identically on GitHub and Zenodo.

Typed builders fail on pending author declarations, missing ORCID or DOI, route gates, altered
manifests, dirty/tag/version mismatches, restricted artifacts, journal-format rules, and direct
identifiers in RBET reviewer files. Development previews remain explicitly non-submittable.

## Consequences

The accepted offline evidence protocol remains frozen. This decision changes packaging and venue
routing, not experiments or claims. Account actions and institutional decisions remain external.
All weak, null, and unfavorable findings remain in the manuscript and distributable evidence.

