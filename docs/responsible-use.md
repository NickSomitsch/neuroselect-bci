# Responsible use

## Intended use

Use NeuroSelect for reproducible software research, offline evaluation of candidate-ranking
mechanics, accessibility prototyping with synthetic profiles, and study-design review. Every
candidate is provisional, every selected sensitive candidate requires the configured confirmation,
and every final message requires an exact explicit confirmation challenge.

## Prohibited or unsupported use

Do not use NeuroSelect to:

- claim mind reading, infer unexpressed beliefs, or attribute generated text to a person without
  their explicit confirmation;
- diagnose, treat, triage, determine capacity or consent, or make medical, legal, financial,
  employment, insurance, policing, or eligibility decisions;
- communicate emergencies or control safety-critical equipment;
- monitor a person covertly, upload their EEG by default, or train personalization from text they
  did not knowingly approve;
- expose the current unauthenticated API to a network or operate it as a multi-user service; or
- report simulation, original-task decoding, and counterfactual replay as a pooled performance
  result.

## Researcher responsibilities

Obtain dataset licenses and consent independently, minimize data, document retention and deletion,
separate development from held-out subjects and sessions, publish negative results and dependency
gates, and retain provenance for every model and table. Review retrieved records and candidate
explanations with users. Provide an immediate non-BCI cancel path and a conventional communication
fallback.

Future human-subject or live-hardware work requires ethics review, accessibility testing with the
intended population, adverse-event handling, authentication, secure storage, and a protocol that
distinguishes device errors from user confirmation.
