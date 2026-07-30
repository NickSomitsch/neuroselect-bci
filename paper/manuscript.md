# Abstract

**Background:** Language prediction can reduce the number of explicit selections required by a
brain-computer interface (BCI), but a language model can rank an intended continuation only when
that continuation is present in the visible candidate set. Studies that report ranking
conditionally on candidate availability can therefore obscure a more basic coverage bottleneck.

**Objective:** We developed NeuroSelect, a reproducible offline framework that keeps candidate
generation, language ranking, P300 decoding, retrieval, fusion, and confirmation as separately
auditable components. We asked whether profile-specific language adapters improve ranking on
held-out synthetic messages, how a classical and a neural decoder perform on held-out public P300
data, whether fused evidence improves a counterfactual replay, and whether compositional candidate
interfaces mitigate observed opening failures.

**Methods:** Four fixed synthetic profiles contributed **3,990 held-out next-span trials from
1,000 messages**. A pinned four-bit Qwen3-4B model generated target-blind candidates, and four
independently trained low-rank adapters rescored fixed candidate sets. Public bigP3BCI Study P data
were split by participant into training, validation, and test partitions. A calibrated xDAWN-LDA
decoder and a secondary EEGNet comparator were evaluated on the original target/non-target flash
task. Recorded held-out P300 probabilities were then remapped to synthetic candidate menus in a
balanced offline counterfactual replay. Exploratory experiments tested grammar/profile ablations
and one-, two-, and three-stage opening composition.

**Results:** The frozen language generator made the intended span available in **28.7% of trials
(95% CI 27.3%-30.1%)**, and complete-message availability was **0.0%**. Conditional on
availability, personalization improved top-1 recall by **9.6 percentage points (95% CI
6.4-12.9)** overall, but reduced it by **9.1 points (95% CI 2.9-15.2 reduction)** for the concise
profile. On **21,491 held-out labeled epochs**, xDAWN-LDA achieved **AUROC 0.800** and **Brier
score 0.062**. EEGNet had no clearly established selection-ranking advantage and a **0.089 higher
Brier score (95% CI 0.084-0.095)**. In counterfactual replay, the complete system changed
top-1 recall by **+0.7 percentage points (95% CI 0.0-2.8)** and completion by **0.0 points**
relative to BCI-only replay. Exploratory target-blind generation increased span availability to
**52.5%**, while three-stage composition covered **66.7%** of unseen combinations of observed
components but **0.0%** of openings from unseen paraphrase families.

**Conclusions:** NeuroSelect provides a checksum-addressed way to study language-assisted P300
candidate selection without merging unlike evidence tiers. The experiments identify candidate
coverage, especially message openings and unseen surface families, as the largest observed
bottleneck in the evaluated pipeline. They do not demonstrate live communication benefit,
clinical efficacy, or open-vocabulary thought decoding.

**Keywords:** brain-computer interface; P300; language prediction; personalization; candidate
generation; counterfactual replay; reproducible research.

# 1. Introduction

Brain-computer interfaces create a non-muscular control channel from measured neural activity to an
external device [@wolpaw2002]. The P300 matrix speller is a canonical non-invasive example: a user
attends to a desired item while rows, columns, or individual stimuli flash, and target flashes
elicit event-related activity that can be distinguished from non-target flashes [@farwell1988].
Many variants have since changed stimulus design, classification, stopping rules, and the displayed
symbol set [@rezeika2018; @pan2022]. Despite these advances, spelling remains a sequence of costly
explicit selections. Language information is consequently attractive because it can raise the
prior probability of plausible continuations or expose multiword candidates. Earlier offline P300
work showed that integrating natural-language probabilities with dynamic classification can
improve character-level accuracy and bit rate [@speier2012].

Predictive-language integration has also been evaluated in online P300 spelling. A predictive-word
interface reduced task-completion time but also reduced accuracy [@ryan2011], while a trigram
language model incorporated into online dynamic stopping reduced the required data collection and
increased communication rate under its tested protocol [@mainsah2014]. More recently, offline
simulations have compared GPT-2, BERT, and BART language components for P300 spelling and reported
model-dependent typing-speed estimates [@parthasarathy2024]. Those studies primarily addressed
character or word prediction and communication-rate simulation. NeuroSelect instead isolates exact
phrase-candidate coverage, conditional ranking, original-task EEG decoding, and counterfactual
phrase-menu replay so that performance at one layer is not treated as evidence at another.

Moving from character priors to generated phrase menus creates a different failure mode. Let an
intended next span be \(y\), a target-blind generator produce a finite candidate set \(C(x)\) from
context \(x\), and a ranker order the members of that set. Ranking quality is relevant only when
\(y \in C(x)\). A strong conditional ranker cannot recover an intended continuation that is absent
from the interface, and averaging only over available targets can overstate end-to-end usefulness.
This distinction is especially important when a compact menu is required for a P300 interface.

Personalization adds a second distinction. A language model adapted to one person's preferred
style may alter ranking without improving candidate generation. The effect can also vary by style
profile rather than appearing as a uniform gain. Low-rank adaptation (LoRA) offers an efficient way
to train separate parameter updates while freezing a common base model [@hu2022], but the resulting
adapters should be evaluated on held-out messages and the exact visible candidate set. Retrieval
context should likewise remain distinguishable from parametric model adaptation; the present work
uses transparent local lexical retrieval rather than treating all contextual augmentation as an
undifferentiated model score [@lewis2020].

Evaluating the full interaction poses an additional evidence problem. Public P300 datasets provide
recorded neural responses to their original stimuli, not neural responses to newly generated
phrase tiles. NeuroSelect therefore separates original-task EEG decoding from counterfactual
candidate replay. In the original task, a decoder estimates target/non-target flash probabilities.
In the replay, those recorded probabilities are remapped to candidate positions under explicit
rules. The remapped words were never selected or endorsed by the source participant. This
separation avoids relabeling event discrimination as word-decoding accuracy.

The paper makes three contributions. First, it introduces an offline framework and evidence
contract that preserve candidate, language, EEG, retrieval, fusion, and confirmation provenance.
Second, it reports frozen component and replay results, including unfavorable coverage and
profile-level findings. Third, it uses the failure pattern to motivate explicitly exploratory
candidate-generation and hierarchical-opening experiments. The primary research questions concern
target availability, conditional personalization, held-out original-task P300 decoding, and
prespecified fusion contrasts. The exploratory experiments ask whether known linguistic components
can be composed within a fixed candidate budget and where that strategy fails.

The evidence hierarchy is central to interpretation. Synthetic language does not become EEG
evidence; original-task EEG does not become message-level communication; counterfactual replay does
not become participant use; and developer-authored exploratory benchmarks do not replace the
frozen primary result.

{{table:table-1-evidence-hierarchy}}

# 2. Materials and methods

## 2.1 Study design and reproducibility contract

NeuroSelect was evaluated as an offline computational study using synthetic language and secondary
analysis of a public P300 dataset. No participant was recruited and no participant used the
NeuroSelect interface. The tracked publication protocol fixed five questions, source-run
identities, allowed and prohibited claims, and a requirement that weak or null results remain
reportable. The primary language, xDAWN-LDA, and counterfactual artifacts existed before that
publication protocol and are therefore retrospective evidence. The later candidate-generation and
opening-generalization designs are labeled exploratory; their locked or test-exposed status is
reported separately.

Every experiment writes a typed result and a checksum-addressed manifest containing the Git
revision, configuration digest, dependency versions, hardware summary, input digests, output
digests, and random seeds. Manuscript tables and figures are generated only from verified clean
manifests. The manuscript assembler verifies the display manifest, all embedded files, references,
and a quantitative claim ledger before producing synchronized LaTeX, PDF, Word, and Markdown
documents. The LaTeX source and BibTeX bibliography are tracked for journal submission, while the
compiled bundle includes the exact publication figures. A dirty source tree can be used for
development rendering only and is explicitly marked non-ready. Source code, tests, protocol,
analysis recipes, and manuscript sources are maintained in a version-controlled NeuroSelect
repository and will be made public with the submission release. Non-restricted result manifests,
tables, figures, and publication-analysis outputs will be archived with that release.

## 2.2 Synthetic messages and leakage boundaries

The language benchmark contains four fixed synthetic communication profiles: casual, concise,
formal, and reflective. Each profile contributes 250 test messages; across profiles, the frozen
test set contains **1,000 messages**. Each message is segmented into target next spans, producing
**3,990 language trials**. Training, validation, and test messages are disjoint. Profile adapters
were optimized only on the corresponding training corpus. Validation loss was monitored during
the fixed training schedule, but it was not used for early stopping or best-checkpoint selection.
Test loss was evaluated after training, and test messages were then used for the reported ranking
benchmark; no test message contributed a training gradient. Because the adapter artifacts existed
before the publication protocol was written, these results are retrospective rather than a
prospectively sealed evaluation.

Synthetic profiles are controlled style conditions, not people. They contain no private user data
and do not establish population generalization. An intended target is used only after generation
to compute availability and rank. Generator APIs do not accept the intended target, and artifact
checks record this constraint.

Primary next-span contexts were constructed by teacher forcing: after each target span, the next
trial context included the preceding reference spans rather than the system's earlier generated or
selected output. This isolates per-span candidate generation and ranking from cascading interface
errors. Complete-message availability nevertheless required every target span in a message to be
available.

## 2.3 Candidate generation and language scoring

The frozen generator used Qwen/Qwen3-4B-MLX-4bit at revision
52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495. Qwen3 is a family of dense and
mixture-of-experts language models with support for reasoning and non-reasoning modes [@qwen32025].
NeuroSelect disabled thinking, set temperature to zero, allowed at most 512 generated tokens, and
limited each visible menu to nine language candidates; three separately scored control actions
were handled outside that language-candidate budget. Structured-output validation rejected invalid
candidates without inserting the intended answer.

The generic language score is the candidate's mean token log likelihood under the base model. Four
profile-specific LoRA adapters were trained independently and verified by checksums before
evaluation. The personalized condition rescored the same fixed candidate set; it did not generate a
new personalized set. This makes personalization a paired ranking comparison rather than a
combined generation-and-ranking intervention. Each adapter used 16 adapted transformer layers, a
batch size of one, a learning rate of \(10^{-5}\), 600 update iterations, a maximum sequence length
of 512 tokens, gradient checkpointing, and seed 20260723. Validation loss was evaluated every 100
iterations; the final scheduled adapter, rather than a validation-selected checkpoint, was used.
Training and inference used MLX on Apple silicon [@hannun2023].

For each trial, target availability equals one when the exact intended span occurs in the
candidate set. Unconditional top-k recall counts unavailable targets as failures. Conditional
top-k recall and reciprocal rank use only available targets. Complete-message availability equals
one only when every target span in the message is available. Exact message accuracy additionally
requires every target span to be available and ranked first.

## 2.4 Public EEG source and preprocessing

Study P was drawn from bigP3BCI version 1.0.0 on PhysioNet, licensed CC BY 4.0
[@mainsah2025]. The release contains **19 participants and 228 EDF+ spelling-block recordings**
from two sessions, acquired with **32 EEG channels at 256 Hz**. The source task compared
predictive and non-predictive 9-by-8 P300 spelling [@ryan2011]. NeuroSelect verified source files against the
official SHA-256 inventory and retained subject, session, run, condition, flash, target label, and
selection-trial provenance.

The model split held out complete participants: **13 training, 3 validation, and 3 test
participants**. Source folders named Train and Test describe task blocks and label availability;
they are not model partitions. Only labeled blocks were used for supervised training and
evaluation. Test blocks with zero-only label streams were retained as unknown for replay
engineering but excluded from supervised denominators.

Preprocessing used MNE-Python [@gramfort2013]. The fixed recipe applied a 60-Hz notch filter,
0.5-20-Hz band-pass filter, average reference, epochs from -0.1 to 0.8 seconds, baseline correction
from -0.1 to 0 seconds, peak-to-peak artifact rejection, and resampling to 128 Hz. Rejected epochs
remained represented in the preprocessing report.

## 2.5 Original-task P300 decoders

The prespecified primary decoder used two xDAWN spatial components with regularization 0.1,
followed by linear discriminant analysis with automatic covariance shrinkage. xDAWN was designed
to enhance evoked responses in BCI data [@rivet2009]. A logistic calibration model with
regularization parameter \(C=1\) was fitted to the LDA decision scores from the three held-out
validation participants. The locked calibrated model was then evaluated on the three held-out test
participants; seed 20260719 fixed the recipe.

EEGNet served as a secondary comparator. It is a compact convolutional architecture that uses
depthwise and separable convolutions across several EEG paradigms [@lawhern2018]. It used the same
participant split, preprocessing artifacts, labeled test epochs, and selection trials as the
xDAWN-LDA analysis. Its purpose was to test whether a neural comparator changed the original-task
conclusion, not to select the more favorable model after observing test performance. The locked
recipe used eight temporal filters, depth multiplier two, 16 pointwise filters, temporal kernels
of 31 and 15 samples, pooling factors of four and four, dropout 0.25, batch size 64, AdamW with
learning rate \(10^{-3}\) and weight decay \(10^{-4}\), and seed 20260720. Training ran for at most
100 epochs with validation-loss early stopping after 12 non-improving epochs; the best validation
state was restored. Held-validation logits were temperature-scaled before test evaluation.

Epoch metrics were area under the receiver operating characteristic curve (AUROC), balanced
accuracy, Brier score, and expected calibration error (ECE). Calibration was reported because a
fusion system consumes probabilities rather than labels alone [@guo2017]. Selection-trial metrics
ranked occurrence-level target events: exact target-event-set recovery, target-event recall at the
known number of targets, target-event average precision, and whether the highest-probability event
was a target. These are not symbol or word accuracies.

## 2.6 Counterfactual candidate replay and fusion

The counterfactual builder sampled held-out language trials and held-out original-task EEG
selection trials under a fixed balanced design. Each of six prespecified conditions contained
**144 counterfactual trials from 3 held-out EEG participants**. For a sampled source selection,
recorded target/non-target probabilities were remapped to positions in a synthetic candidate menu.
The mapping preserved provenance to the source subject, session, recording, selection, event, and
candidate. It did not assert that the participant saw or intended the remapped phrase.

The six conditions were BCI only, generic language only, neural plus generic language, neural plus
personalized language, neural plus personalized language and retrieval, and the complete system
with safety policy. Here, “neural” denotes EEG-derived probabilities and does not imply that the
primary decoder was a neural network. Fusion weights were 0.65 for EEG evidence, 0.15 for generic
language, 0.08 for personalization, and 0.12 for retrieval, with a 0.35 risk penalty and a maximum
0.08 diversity adjustment. Local lexical retrieval added profile-relevant context without changing
the underlying EEG. The safety policy requested a repeat for low EEG support, a small EEG margin,
or EEG-language conflict; it abstained when EEG evidence was missing or the fused score or margin
was below threshold. Risk tags added a penalty and enhanced-confirmation flag rather than directly
forcing a repeat. Primary outcomes were top-1 recall, selection completion, and repeat requests.

## 2.7 Exploratory candidate-generation experiments

The first exploratory comparison replaced the frozen generator with a target-blind,
profile-conditioned, grammar-routed candidate bank fitted on training and validation messages. The
existing test benchmark and its primary outcome had already been inspected, so this v2 comparison
is explicitly test-exposed. All methods retained a nine-candidate budget. Subsequent locked
ablations removed profile conditioning, removed grammar routing, or used frequency alone.

A two-stage opening interface first displayed opening stems and then displayed continuations
conditional on a simulated observed stem selection. On the existing test-exposed benchmark it was
compared with one-stage generation. A separate developer-authored robustness benchmark held out
exact stem-action combinations while allowing every component to appear in training or validation.
The target action was not passed to the second-stage generator. This benchmark tests composition
of known parts, not natural-language generalization.

The final exploratory experiment increased the component inventory to 24 fitted stems and 48
content words across request, preference, clarification, and status intents. It compared one-stage
complete-phrase retrieval, two-stage stem/content composition, and three-stage
intent/stem/content composition under the same maximum menu size. Two challenges were fixed:
unseen exact combinations of observed components and paraphrase-family stems absent from fitting.
The latter is the explicit closed-vocabulary stress test.

## 2.8 Statistical analysis

The analysis reports descriptive effects for fixed offline samples. For language outcomes,
complete messages were resampled within fixed profile strata. For EEG selection metrics, held-out
participants were resampled first and selection trials second. Counterfactual paired contrasts
were resampled at the held-out-participant and complete-message levels. Exploratory language
contrasts resampled complete messages within profile strata. Every reported interval used
**10,000 bootstrap resamples**, percentile limits, and the protocol-fixed seeds. No multiplicity
adjustment was applied. The intervals condition on the fitted checkpoints, fixed candidate
generation realization, and evaluated samples; they do not include uncertainty from repeated
adapter or EEGNet training or stochastic generation. Intervals are descriptive and do not
establish clinical efficacy, population-level superiority, or non-inferiority.

No synthetic-language, EEG, counterfactual, or exploratory observation was pooled into a single
system score. Scikit-learn supplied the classical machine-learning estimators and metrics
[@pedregosa2011].

# 3. Results

## 3.1 Frozen candidate availability and personalization

Across the frozen 3,990-span benchmark, exact target availability was **0.287**, with a 95%
message-clustered interval of **[0.273, 0.301]**. Availability varied modestly by profile, from
0.256 for reflective to 0.308 for formal. No held-out message had all target spans available:
complete-message availability was **0.000**. The unconditional generic and personalized top-1
recalls were 0.041 and 0.068, respectively, because unavailable targets remained failures.

Conditional on availability, personalized top-1 recall exceeded generic top-1 recall by
**+0.096 [95% CI +0.064, +0.129]** overall. The paired MRR difference was **+0.087 [95% CI
+0.063, +0.112]**. Effects were heterogeneous. Casual, formal, and reflective profiles had
positive conditional top-1 differences, whereas concise had a **-0.091 [95% CI -0.152, -0.029]**
difference. Thus the overall gain does not support a claim that every profile benefited.

{{figure:figure-1-language-bottleneck}}

{{table:table-2-language-primary}}

## 3.2 Held-out original-task P300 decoding

Both decoders were evaluated on **21,491 labeled test epochs** from three held-out participants.
xDAWN-LDA achieved **AUROC 0.800**, balanced accuracy 0.586, **Brier 0.062**, and ECE 0.017.
At the selection-trial level, its target-event recall at the known target count was **0.355 [95%
CI 0.291, 0.423]**, target-event average precision was 0.396 [0.322, 0.459], and the top event was
a target in 0.560 [0.421, 0.694] of selections. Exact recovery of the complete occurrence-level
target-event set was 0.009 [0.000, 0.032], illustrating the stringency of that outcome.

EEGNet produced a similar AUROC of 0.802 and higher balanced accuracy of 0.731, but its Brier score
was 0.152 and ECE was 0.218. Paired selection contrasts for target-event recall, average
precision, and top-event hit all had intervals spanning zero. The EEGNet-minus-xDAWN Brier
difference was **+0.089 [95% CI +0.084, +0.095]**, where a positive value is worse. The comparator
therefore did not establish a selection-ranking advantage and was substantially less well
calibrated under the locked recipes.

{{figure:figure-2-p300-comparison}}

{{table:table-3-p300-original-task}}

{{table:table-3b-p300-paired-contrasts}}

## 3.3 Counterfactual fusion

Target availability in the balanced replay was **0.271 in every condition**, making the coverage
ceiling visible rather than excluding unavailable trials. BCI-only replay achieved top-1 recall
and completion of 0.250. Generic language alone achieved 0.042 for both outcomes. Neural plus
language and neural plus personalized language each achieved 0.250. Adding retrieval raised
top-1 recall and completion to 0.257.

The complete system achieved top-1 recall 0.257, completion 0.250, and repeat-request rate 0.042.
Relative to BCI-only replay, its paired top-1 difference was **+0.007 [95% CI +0.000, +0.028]**,
while its completion difference was **+0.000 [95% CI +0.000, +0.000]**. The safety policy did not
change top-1 recall, reduced completion by 0.007, and increased repeat requests by 0.042. These
results do not demonstrate an end-to-end advantage for the complete configuration.

{{figure:figure-3-counterfactual-fusion}}

{{table:table-4-counterfactual-conditions}}

{{table:table-4b-counterfactual-contrasts}}

## 3.4 Exploratory target-blind candidate generation

On the existing test-exposed benchmark, full v2 generation increased exact-span availability from
0.287 to **0.525**, a paired difference of **+0.238 [95% CI +0.219, +0.257]**. Complete-message
availability rose from 0.000 to 0.021. Removing grammar routing reduced span availability to 0.129;
removing profile conditioning reduced it to 0.473. Frequency-only retrieval reached 0.471. These
comparisons indicate that the grammar route made the largest contribution on the exposed
benchmark, but they are exploratory and not a replacement for the frozen result.

The two-stage opening interface reached span availability 0.560, opening availability 0.188, and
complete-message availability 0.091 on the existing benchmark. On the locked held-out-combination
benchmark, one-stage full v2 had **0.000 opening availability**, while two-stage composition had
**1.000 opening availability** and 0.323 complete-message availability. The gain required a mean
of 1.250 reached stages rather than one stage. Because the stems and actions were individually
observed and the benchmark was developer-authored, complete opening coverage is evidence of
constrained composition only.

{{figure:figure-4-candidate-generation}}

{{table:table-5-candidate-generation}}

{{table:table-5b-candidate-contrasts}}

## 3.5 Exploratory hierarchical openings

On 288 held-out stem/content combinations, one-stage phrase retrieval had **0.000 availability**,
two-stage stem/content composition reached **0.250**, and three-stage intent/stem/content
composition reached **0.667**. Coverage per required selection was 0.000, 0.125, and 0.222,
respectively. The three-stage method reached three menus and exposed a mean of 19 candidates, so
its coverage gain carried explicit interaction cost.

On 384 openings whose paraphrase-family stems were absent from fitting, **all three methods had
0.000 availability**. Two-stage composition failed at the unseen stem; three-stage composition
could select an intent but still failed to supply the required stem. The hierarchy therefore
combined observed components but did not provide open-vocabulary generation.

{{figure:figure-5-opening-generalization}}

{{table:table-6-opening-generalization}}

{{table:table-6b-opening-contrasts}}

# 4. Discussion

## 4.1 Principal findings

The main result is a bottleneck decomposition rather than a claim of complete-system efficacy.
First, the frozen target-blind generator made intended spans available in fewer than one third of
rounds and never covered an entire held-out message. Second, profile adapters improved average
conditional ranking, but the effect was not uniform and the concise profile worsened on conditional
top-1 recall. Third, the primary xDAWN-LDA decoder discriminated original-task target events with
an AUROC of 0.800 and lower Brier score and ECE than EEGNet, whereas EEGNet did not establish a
paired selection-ranking advantage. Fourth, counterfactual fusion
did not improve completion over BCI-only replay. Finally, exploratory hierarchy improved
composition when every component had been observed, yet every tested method failed on unseen
paraphrase families.

Taken together, these findings show why component separation matters. Reporting only conditional
personalized ranking would emphasize a positive average effect while hiding the 28.7% candidate
ceiling and 0% complete-message coverage. Reporting only EEG AUROC would hide the much lower exact
target-event-set recovery. Reporting only the best exploratory hierarchy would hide its additional
selections, developer-authored vocabulary, and complete failure on unseen families.

## 4.2 Candidate availability precedes ranking

Phrase ranking and candidate construction are sequential constraints. The ranker can affect which
visible item appears first but cannot recover an absent target. In NeuroSelect, personalization was
applied only after a fixed set was created, so its average conditional benefit did not change
availability. This design exposes the distinction cleanly and suggests that future work should
prioritize candidate recall under a realistic menu and interaction budget before investing in more
complex rankers.

The exploratory v2 result demonstrates that availability is changeable: grammar routing and
profile-aware retrieval approximately doubled span coverage on the existing benchmark. It also
shows why this is not yet a solved problem. Complete-message coverage remained low, the benchmark
was exposed during development, and the robustness sources were synthetic and developer-authored.
The hierarchy then clarified what compositional menus can and cannot do. They reduce the
combinatorial burden for known stems and contents, but they do not invent an unseen surface family.

An open-vocabulary system would require a mechanism that can generate or spell a missing component
rather than only retrieve it from a fitted bank. Plausible next designs include a hybrid menu with
an explicit character-level fallback, constrained subword generation, semantic retrieval over a
larger user-controlled lexicon, or a generative opening stage with conservative confirmation.
Those alternatives should be evaluated on an independently authored benchmark with lexical-family
holdout and measured selection cost. Intended targets must remain unavailable to the generator at
inference.

## 4.3 Personalization is heterogeneous

The overall conditional ranking improvement supports the narrower statement that profile-specific
adapters changed ordering on the frozen synthetic benchmark. It does not imply a universal
personalization benefit. The concise profile's negative conditional top-1 effect is especially
important because a system optimized only for the pooled mean could reduce performance in a
controlled style condition.

Several mechanisms could produce this pattern. Concise messages provide shorter contexts and may
leave fewer style cues for an adapter. The adapter may increase likelihood for generally concise
candidates that differ from the exact held-out span. Mean-token log likelihood can also favor
different phrase lengths or lexical choices. Because profiles are synthetic, the result is a
controlled model-behavior finding rather than a claim about user groups. A future participant-free
study could test adapter stability across seeds, corpus sizes, and independently authored profiles,
and could report semantic acceptability alongside exact-span recall.

## 4.4 Neural evidence and probability calibration

The original-task analysis supports using the xDAWN-LDA probabilities as a transparent offline
neural evidence source. It does not support calling 0.800 AUROC a word or message accuracy.
Selection trials contain multiple target occurrences, and exact event-set recovery was deliberately
strict. The large gap between epoch AUROC and exact set recovery illustrates how apparently strong
binary discrimination can translate into a much harder structured selection problem.

EEGNet's balanced accuracy was higher, but its paired selection-ranking intervals crossed zero and
its probabilities were much less calibrated. A fusion system needs meaningful relative evidence,
not merely a thresholded class decision. xDAWN-LDA had been prespecified as the primary decoder;
EEGNet was added later as a locked comparator rather than used for post-test model selection. Its
calibration result therefore informs interpretation and future model choice, not the choice of the
reported primary model. Broader conclusions would require more held-out participants, repeated
training seeds, and possibly nested cross-validation. The current test set contains three
participants, so participant-level uncertainty remains substantial.

## 4.5 What counterfactual replay establishes

Counterfactual replay is useful because it exercises the implemented candidate and fusion path with
recorded neural probabilities and explicit provenance. It can reveal policy behavior, unavailable
targets, and contrasts between scoring conditions. Here it showed that adding personalization did
not change the selected item in the balanced sample, retrieval produced only a small top-1 change,
and the safety layer converted some selections into repeat requests without improving completion.

Replay cannot establish that a person could attend to the remapped tiles, that the neural response
would remain unchanged under a different display, or that the resulting phrase expresses the
participant's intent. The source participants selected characters in the Study P task. They did not
author the synthetic messages or use NeuroSelect. A live or clinical claim would require a new
protocol, ethics review, consent, real interface exposure, usability outcomes, and comparison with
established augmentative and alternative communication options.

## 4.6 Reproducibility contribution

The framework's methodological contribution is its fail-closed evidence chain. Data selection,
participant splits, preprocessing, model recipes, language adapters, replay mappings, statistical
resampling, tables, figures, manuscript claims, and document assembly all carry machine-readable
identity. The pipeline refuses unverified source replacement and records whether the source tree
was clean. This does not make the scientific conclusions stronger than their samples, but it makes
the limits and transformations inspectable.

The manuscript itself is assembled from the same display bundle used for exact CSV tables and
300-dpi/vector figures. A claim ledger verifies the principal quantitative phrases against source
cells or tracked configuration values. External submission gates remain separate: a clean build
does not substitute for institutional secondary-use wording, open-access confirmation, scientific
review, or final author metadata.

## 4.7 Limitations

The study has eight main limitations. First, language profiles and messages are synthetic; exact-span
metrics can penalize semantically acceptable alternatives, while controlled language may
underrepresent natural communication. Second, the primary language protocol is retrospective and
the v2 benchmark was exposed during development. The later robustness experiments were locked
before execution but remain developer-authored.

Third, Study P measures an original predictive/non-predictive spelling task. Predictive and
non-predictive condition order varies across the source design, so session differences cannot be
attributed solely to time or electrode drift. Only three participants were held out for test, and
the EEG results do not represent NeuroSelect phrase selection. Fourth, counterfactual replay
preserves recorded evidence but changes the displayed interpretation; it is not a substitute for
live data.

Fifth, lexical retrieval and tracked safety tags are intentionally narrow. They do not constitute a
semantic memory system, a general content-safety classifier, or a clinical safeguard. Sixth, the
hierarchical opening experiments teacher-force simulated correct intermediate selections and
report planned selection cost rather than measured interaction time, fatigue, visual load, or
error correction. Seventh, each LoRA adapter and EEGNet were trained with one fixed seed, and
generation was evaluated as one frozen realization. The bootstrap intervals therefore do not
represent training- or generation-induced variability. Eighth, primary next-span evaluation
teacher-forced the reference history, so it does not measure cascading errors during autonomous
message construction.

# 5. Conclusions

NeuroSelect is a reproducible offline framework for studying profile-conditioned language ranking
and counterfactual P300 candidate fusion while preserving evidence boundaries. In the frozen
evaluation, low target availability and zero complete-message availability constrained the
language path. Personalization improved conditional ranking on average but reduced one
profile-level top-1 outcome. The prespecified calibrated xDAWN-LDA decoder achieved an original-task
AUROC of 0.800. The secondary EEGNet comparator did not establish a selection-ranking advantage,
and counterfactual full fusion did not establish a completion advantage over BCI-only replay.

Exploratory compositional menus improved coverage for new combinations of observed parts and made
message openings a tractable object of study. Their failure on every unseen paraphrase family
defines the next technical question: how to introduce a verified generative or spelling fallback
without hiding added selections, unsafe attribution, or target leakage. The present evidence
supports an offline computational methods paper, not a claim of live communication or clinical
benefit.

# Declarations

## Data and code availability

Source code, tracked protocols, synthetic benchmark sources, tests, and manuscript sources will be
released at https://github.com/NickSomitsch/neuroselect-bci under the MIT License before
submission. Study P is available separately from PhysioNet under CC BY 4.0 [@mainsah2025]. Raw EEG,
cached base-model weights, trained adapters, and other license-controlled local artifacts will not
be redistributed in the source repository. Each reported run records source and output digests
needed to verify local artifacts; non-restricted result manifests, tables, figures, and
publication-analysis outputs will accompany the public release.

## Ethics statement

This work recruited no participants and performed no live NeuroSelect study. It used public,
deidentified Study P data and synthetic language. Final journal submission remains contingent on
institutionally approved wording for the secondary use of Study P; this manuscript does not
independently claim an ethics exemption.

## Author contributions

Nick Somitsch conceived and implemented the software, designed the offline experiments, performed
the analyses, and drafted the manuscript. Final CRediT terminology and any additional qualifying
authorship will be confirmed before submission.

## Competing interests and funding

Competing-interest and funding declarations are external submission metadata and must be confirmed
by the author before journal submission; no statement is inferred from the repository.

# References

{{references}}
