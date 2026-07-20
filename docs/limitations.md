# Limitations

NeuroSelect is a research prototype for reducing explicit selections in a candidate-based
communication workflow. It does not decode unrestricted thoughts, imagined open-vocabulary
speech, identity, diagnosis, or private intent from scalp EEG.

## Evidence limitations

- Controlled simulation keeps the intended span visible and uses synthetic neural probabilities.
  It tests software mechanics, not free-form candidate recall or participant performance.
- Study P models solve the source target/non-target flash task. Event AUROC and stimulus-code
  accuracy are not word accuracy, message accuracy, or clinical communication speed.
- Counterfactual replay maps a recorded target position to a candidate tile. The source participant
  did not select or endorse the displayed word.
- The repository contains no complete all-subject Study P result, real held-out language-model
  LoRA, claim-eligible A–F fusion result, live EEG result, or participant usability study.
- Modeled interaction time is not wall-clock latency or measured user speed. Hierarchical
  bootstrap intervals are descriptive and do not establish efficacy or non-inferiority.

## Model and data limitations

P300 responses vary across people, sessions, fatigue, attention, equipment, and preprocessing.
Study P condition order may confound chronological session comparisons. Dataset labels and source
metadata do not justify broader clinical-status inferences. Small subject-specific adapters can be
unstable, and hardware/runtime differences can affect neural training despite fixed seeds.

The fixture language backend is deterministic and intentionally limited. Personal retrieval is
lexical rather than semantic, and lexical injection screening is not a complete content-safety or
privacy classifier. Candidate risk tags cover a narrow tracked vocabulary and cannot identify all
medical, financial, legal, consent, coercion, or emergency content.

## System limitations

The API has no authentication and is safe only on loopback in a trusted single-user environment.
Sessions are in memory, concurrent multi-process operation is unsupported, and the UI is not a
validated assistive technology. Explicit confirmation reduces incorrect attribution but cannot
prove comprehension, capacity, voluntariness, authorship, or the absence of interface error.

The project is not a medical device, diagnostic tool, treatment, emergency service, or substitute
for an established augmentative and alternative communication system. Do not make consequential
decisions from its outputs.
