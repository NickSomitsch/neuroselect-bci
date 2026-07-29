# ADR 0030: Publication tables and figures

- Status: Accepted
- Date: 2026-07-29

## Context

The offline paper evidence now includes a frozen primary analysis, a secondary EEGNet comparator,
an exploratory test-exposed candidate-generation comparison, locked Step 4 ablations, and a
locked hierarchical-opening experiment. Copying values into figures by hand would risk
transcription errors, stale estimates, silent omission of negative results, and accidental
pooling across evidence tiers.

## Decision

Use one checksum-pinned display recipe that reads only verified, clean source artifacts. It emits:

1. an evidence-hierarchy table;
2. primary language, original-task EEG, and counterfactual tables with paired intervals;
3. explicitly exploratory candidate-generation and opening-generalization tables;
4. five colorblind-safe figure families, each confined to one evidence component;
5. SVG and PDF vector files plus 300-dpi PNG files;
6. a shared caption sheet, inventory, and SHA-256 manifest.

The renderer performs no model execution, resampling, outcome selection, or pooled scoring.
Primary, secondary-comparator, and exploratory labels are part of every table and figure's
inventory metadata. Zero results, including complete-message failure and unseen-family opening
failure, cannot be omitted from the locked display.

## Consequences

The manuscript can cite one auditable display bundle while preserving the evidence hierarchy.
Readers can recover exact table values from CSV and edit vector figures without raster loss.
Negative and heterogeneous results remain visible: personalization helps overall conditional
ranking but harms concise top-1 ranking, EEGNet has no established selection-ranking advantage
and worse calibration, full fusion does not improve completion over BCI-only replay, hierarchy
improves observed-component composition, and all tested methods fail on unseen paraphrase
families.

A bundle generated from uncommitted code is marked non-publication-ready. Implementation-time
rendering may use the explicit dirty-development flag for visual inspection, but the final bundle
must be regenerated from a clean commit. These files support an offline computational manuscript;
they do not create live-use, participant-benefit, or clinical evidence.
