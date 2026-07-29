# Limitations

NeuroSelect is a research prototype for reducing explicit selections in a candidate-based
communication workflow. It does not decode unrestricted thoughts, imagined open-vocabulary
speech, identity, diagnosis, or private intent from scalp EEG.

## Evidence limitations

- Controlled simulation keeps the intended span visible and uses synthetic neural probabilities.
  It tests software mechanics, not free-form candidate recall or participant performance.
- Study P models solve the source target/non-target flash task. Event AUROC and occurrence-level
  target-event ranking metrics are not word accuracy, message accuracy, or clinical communication
  speed. Exact target-event-set recovery is especially stringent and must not be labeled ordinary
  symbol accuracy.
- Counterfactual replay maps a recorded target position to a candidate tile. The source participant
  did not select or endorse the displayed word.
- The repository has complete fixed-split xDAWN and held-out language artifacts plus a balanced
  A–F counterfactual replay. It still contains no live EEG result, participant-authored
  NeuroSelect message, or participant usability study. Local model artifacts remain ignored and
  must be verified through their manifests.
- Modeled interaction time is not wall-clock latency or measured user speed. Hierarchical
  bootstrap intervals are descriptive and do not establish efficacy or non-inferiority.

## Model and data limitations

P300 responses vary across people, sessions, fatigue, attention, equipment, and preprocessing.
Study P condition order may confound chronological session comparisons. Dataset labels and source
metadata do not justify broader clinical-status inferences. Small subject-specific adapters can be
unstable, and hardware/runtime differences can affect neural training despite fixed seeds.

The fixture language backend and controlled style proxy are deterministic and intentionally
limited. The optional Qwen backend is Apple-silicon-specific, its relative phrase support is not a
calibrated intent probability, and fixed decoding may still vary across runtime/hardware versions.
Personal retrieval is lexical rather than semantic, and lexical injection screening is not a
complete content-safety or privacy classifier. Candidate risk tags cover a narrow tracked
vocabulary and cannot identify all medical, financial, legal, consent, coercion, or emergency
content.

## System limitations

The API has no authentication and is safe only on loopback in a trusted single-user environment.
Sessions are in memory, concurrent multi-process operation is unsupported, and the UI is not a
validated assistive technology. Explicit confirmation reduces incorrect attribution but cannot
prove comprehension, capacity, voluntariness, authorship, or the absence of interface error.

The project is not a medical device, diagnostic tool, treatment, emergency service, or substitute
for an established augmentative and alternative communication system. Do not make consequential
decisions from its outputs.

The frozen language evaluation makes an intended span available in only about 29% of rounds and
never makes every span of a held-out message available. Conditional ranking improvements therefore
do not establish end-to-end message completion. Profile-level effects are heterogeneous, including
a negative top-1 effect for the concise synthetic profile.

The exploratory candidate-generation v2 comparison raises exact-span availability to 52.5% and
complete-message availability to 2.1%, but it is profile-conditioned, grammar-routed, and
test-exposed. It diagnoses a candidate-set construction alternative on the same synthetic
benchmark and cannot replace the frozen primary result.

The Step 4 robustness benchmark deliberately holds out complete opening combinations while keeping
their stems and actions observable during fitting. Two-stage composition reaches 100% opening
coverage on that constrained benchmark, but this is not evidence of unrestricted natural-language
coverage: only three stems and nine actions are present, the source is synthetic and
developer-authored, and the offline replay teacher-forces the correct first-stage selection. The
extra stage also adds one BCI selection to every opening. On the original test-exposed benchmark,
opening availability is 18.8% rather than 100%.

The harder hierarchical experiment removes the small-menu advantage by using 24 fitted stems and
48 content words. Intent/stem/content hierarchy covers 66.7% of held-out combinations, but requires
three selections and exposes 19 candidates across its menus. All tested retrieval hierarchies have
0% exact coverage for completely unseen paraphrase-family stems. Hierarchy therefore improves
composition of known parts but does not solve open-vocabulary generation. The source remains
synthetic and developer-authored.
