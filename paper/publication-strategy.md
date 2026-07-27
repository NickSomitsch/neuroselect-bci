# NeuroSelect offline paper: publication strategy

This document is the human-readable companion to
`configs/publication/offline_methods_v1.yaml`. The YAML protocol is canonical when a source run,
estimand, or submission gate is in question.

## Planned article

**Working title:** *NeuroSelect: A Reproducible Offline Framework for Personalized Language
Ranking and P300 Candidate Fusion*

The manuscript is an offline computational Original Research article. It evaluates four evidence
tiers separately:

1. controlled simulation for engineering behavior;
2. held-out synthetic language for candidate availability and ranking;
3. held-out original-task Study P EEG for P300 decoding; and
4. counterfactual replay for candidate-system behavior over remapped recorded probabilities.

It is not a participant study of NeuroSelect, a clinical evaluation, or evidence that generated
text was selected by the source EEG participants.

## Venue order

1. **Research in Biomedical Engineering and Technology** — primary Original Research route.
2. **Frontiers in Neuroinformatics** — funded Methods or Technology and Code fallback.
3. **Journal of Open Source Software** — possible later, separate software paper after the project
   has a sustained public development and reuse history.

The University of Innsbruck agreement states that eligible student corresponding authors can
receive Taylor & Francis open-access coverage for eligible Original Research articles, subject to
a limited allocation. Eligibility must be confirmed with `open-access@uibk.ac.at`; it is not
assumed by the protocol.

## Evidence and analysis status

The four source artifacts and the Step 13 research report are pinned by run ID, Git revision, and
manifest digest. Their interpretation is retrospective because they existed before this document.
All publication analyses added after the protocol freeze are prospective relative to this
decision.

The primary paper will report weak or null results as prominently as favorable ones:

- low candidate availability and zero complete-message availability;
- average conditional ranking improvement with profile-level heterogeneity;
- the original-task EEG decoder result, without calling it word-decoding accuracy; and
- no meaningful complete-system advantage over BCI-only replay in the frozen counterfactual
  sample.

Any candidate-generation v2 experiment is an exploratory supplement and is reported whether it
improves or worsens the result.

## Submission blockers

Protocol validation is automated with:

```bash
make publication-protocol-check
```

Submission additionally requires:

- written University of Innsbruck/Taylor & Francis open-access confirmation;
- institutionally approved wording for secondary use of the public Study P data;
- review by a supervisor or independent BCI researcher; and
- confirmed affiliation, institutional email, ORCID, funding, conflict, CRediT, and authorship
  metadata.

Nick Somitsch remains the sole author by default. A supervisor is added as an author only after a
qualifying intellectual contribution and acceptance of responsibility for the paper; otherwise
the contribution is acknowledged.
