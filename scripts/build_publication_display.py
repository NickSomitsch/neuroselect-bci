"""Build publication-ready tables and figures from frozen NeuroSelect evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from neuroselect.evaluation.candidate_generation_step4 import (
    CandidateGenerationDataset,
    CandidateGenerationMethod,
    CandidateGenerationStep4Metric,
    CandidateGenerationStep4Result,
)
from neuroselect.evaluation.candidate_generation_step4_artifacts import (
    read_candidate_generation_step4_artifacts,
)
from neuroselect.evaluation.candidate_generation_v2 import (
    CandidateGenerationV2Interval,
    CandidateGenerationV2Metrics,
    CandidateGenerationV2Result,
)
from neuroselect.evaluation.candidate_generation_v2_artifacts import (
    read_candidate_generation_v2_artifacts,
)
from neuroselect.evaluation.opening_generalization import (
    OpeningChallenge,
    OpeningGeneralizationMetric,
    OpeningGeneralizationResult,
    OpeningMethod,
)
from neuroselect.evaluation.opening_generalization_artifacts import (
    read_opening_generalization_artifacts,
)
from neuroselect.provenance import RunManifest
from neuroselect.publication import (
    DEFAULT_PUBLICATION_DISPLAY_CONFIG,
    PublicationAnalysisResult,
    PublicationDisplaySpec,
    PublicationTable,
    RenderedPublicationFigure,
    load_publication_display_spec,
    load_publication_protocol,
    read_publication_analysis,
    read_publication_display,
    write_publication_display,
)
from neuroselect.publication.analysis import (
    PublicationEstimate,
    PublicationInterval,
)

plt.switch_backend("Agg")

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
BLACK = "#222222"
GRAY = "#6F6F6F"
LIGHT_GRAY = "#D9D9D9"

PROFILE_LABELS = {
    "overall": "Overall",
    "synthetic-casual": "Casual",
    "synthetic-concise": "Concise",
    "synthetic-formal": "Formal",
    "synthetic-reflective": "Reflective",
}
METHOD_LABELS = {
    CandidateGenerationMethod.FULL_V2: "Full v2",
    CandidateGenerationMethod.NO_PROFILE_CONDITIONING: "No profile",
    CandidateGenerationMethod.NO_GRAMMAR_ROUTING: "No grammar",
    CandidateGenerationMethod.FREQUENCY_ONLY: "Frequency only",
    CandidateGenerationMethod.TWO_STAGE_OPENING: "Two-stage opening",
}
OPENING_METHOD_LABELS = {
    OpeningMethod.ONE_STAGE_PHRASE: "One-stage phrase",
    OpeningMethod.TWO_STAGE_STEM_CONTENT: "Two-stage stem/content",
    OpeningMethod.THREE_STAGE_INTENT_STEM_CONTENT: "Three-stage intent/stem/content",
}
CONDITION_LABELS = {
    "a_bci_only": "BCI only",
    "b_generic_language_only": "Generic language only",
    "c_neural_language": "Neural + language",
    "d_neural_personalized": "Neural + personalized",
    "e_neural_personalized_rag": "Neural + personalized + RAG",
    "f_complete_system": "Complete system",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PUBLICATION_DISPLAY_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/publication/paper-display-v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit development rendering; the manifest will remain non-publication-ready.",
    )
    return parser.parse_args()


def git_state() -> tuple[str, str | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not status:
        return revision, None
    digest = hashlib.sha256(
        subprocess.run(["git", "diff", "--binary", "HEAD"], check=True, capture_output=True).stdout
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw_path in sorted(path for path in untracked if path):
        path = Path(raw_path.decode())
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return revision, digest.hexdigest()


def _source(spec: PublicationDisplaySpec, source_id: str) -> Any:
    return next(item for item in spec.sources if item.source_id == source_id)


def _verify_manifest(
    manifest: RunManifest,
    *,
    expected_sha256: str,
    source_id: str,
) -> None:
    if manifest.digest() != expected_sha256:
        raise ValueError(f"{source_id} manifest differs from the publication display pin")
    if manifest.metadata.get("working_tree_dirty") is not False:
        raise ValueError(f"{source_id} must have clean source provenance")


def load_sources(
    spec: PublicationDisplaySpec,
) -> tuple[
    PublicationAnalysisResult,
    CandidateGenerationV2Result,
    CandidateGenerationStep4Result,
    OpeningGeneralizationResult,
]:
    protocol = load_publication_protocol(spec.publication_protocol)
    if protocol.digest() != spec.expected_protocol_sha256:
        raise ValueError("publication protocol differs from the display pin")

    primary_pin = _source(spec, "primary-analysis")
    primary, primary_manifest = read_publication_analysis(primary_pin.path)
    _verify_manifest(
        primary_manifest,
        expected_sha256=primary_pin.expected_manifest_sha256,
        source_id=primary_pin.source_id,
    )
    step3_pin = _source(spec, "candidate-generation-v2")
    step3, _, step3_manifest = read_candidate_generation_v2_artifacts(step3_pin.path)
    _verify_manifest(
        step3_manifest,
        expected_sha256=step3_pin.expected_manifest_sha256,
        source_id=step3_pin.source_id,
    )
    step4_pin = _source(spec, "candidate-generation-step4")
    step4, _, _, step4_manifest = read_candidate_generation_step4_artifacts(step4_pin.path)
    _verify_manifest(
        step4_manifest,
        expected_sha256=step4_pin.expected_manifest_sha256,
        source_id=step4_pin.source_id,
    )
    opening_pin = _source(spec, "opening-generalization")
    opening, _, opening_manifest = read_opening_generalization_artifacts(opening_pin.path)
    _verify_manifest(
        opening_manifest,
        expected_sha256=opening_pin.expected_manifest_sha256,
        source_id=opening_pin.source_id,
    )

    results = (primary, step3, step4, opening)
    if any(result.protocol_sha256 != spec.expected_protocol_sha256 for result in results):
        raise ValueError("a display source uses a different publication protocol")
    if step4.step3_manifest_sha256 != step3_manifest.digest():
        raise ValueError("Step 4 does not reference the pinned Step 3 source")
    if opening.step4_manifest_sha256 != step4_manifest.digest():
        raise ValueError("opening generalization does not reference the pinned Step 4 source")
    if (
        step3.intended_target_exposed_to_generator
        or step4.intended_target_exposed_to_generators
        or opening.intended_opening_exposed_to_generators
    ):
        raise ValueError("publication display sources must remain target-blind")
    return results


def _estimate_index(
    analysis: PublicationAnalysisResult,
) -> dict[tuple[str, str, str, str], PublicationEstimate]:
    return {
        (item.component, item.scope, item.variant, item.metric): item for item in analysis.estimates
    }


def _interval_index(
    analysis: PublicationAnalysisResult,
) -> dict[tuple[str, str, str, str], PublicationInterval]:
    return {
        (item.component, item.scope, item.contrast, item.metric): item
        for item in analysis.intervals
    }


def _estimate(
    index: dict[tuple[str, str, str, str], PublicationEstimate],
    component: str,
    scope: str,
    variant: str,
    metric: str,
) -> PublicationEstimate:
    return index[(component, scope, variant, metric)]


def _interval(
    index: dict[tuple[str, str, str, str], PublicationInterval],
    component: str,
    scope: str,
    contrast: str,
    metric: str,
) -> PublicationInterval:
    return index[(component, scope, contrast, metric)]


def _rate(value: float) -> str:
    return f"{value:.3f}"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _ci(lower: float, upper: float, *, signed: bool = False) -> str:
    formatter = _signed if signed else _rate
    return f"[{formatter(lower)}, {formatter(upper)}]"


def _language_table(analysis: PublicationAnalysisResult) -> PublicationTable:
    estimates = _estimate_index(analysis)
    intervals = _interval_index(analysis)
    rows: list[tuple[str, ...]] = []
    for scope, label in PROFILE_LABELS.items():
        availability = _estimate(
            estimates, "language", scope, "observed", "target_availability_rate"
        )
        availability_ci = _interval(
            intervals, "language", scope, "rate", "target_availability_rate"
        )
        top1_delta = _interval(
            intervals,
            "language",
            scope,
            "personalized-minus-generic",
            "top1_delta_given_available",
        )
        mrr_delta = _interval(
            intervals,
            "language",
            scope,
            "personalized-minus-generic",
            "mrr_delta_given_available",
        )
        rows.append(
            (
                label,
                str(availability.sample_count),
                _rate(availability.estimate),
                _ci(availability_ci.lower_bound, availability_ci.upper_bound),
                _rate(
                    _estimate(
                        estimates, "language", scope, "observed", "generic_top1_unconditional"
                    ).estimate
                ),
                _rate(
                    _estimate(
                        estimates,
                        "language",
                        scope,
                        "observed",
                        "personalized_top1_unconditional",
                    ).estimate
                ),
                f"{_signed(top1_delta.estimate)} "
                f"{_ci(top1_delta.lower_bound, top1_delta.upper_bound, signed=True)}",
                _rate(
                    _estimate(
                        estimates,
                        "language",
                        scope,
                        "observed",
                        "generic_mrr_given_available",
                    ).estimate
                ),
                _rate(
                    _estimate(
                        estimates,
                        "language",
                        scope,
                        "observed",
                        "personalized_mrr_given_available",
                    ).estimate
                ),
                f"{_signed(mrr_delta.estimate)} "
                f"{_ci(mrr_delta.lower_bound, mrr_delta.upper_bound, signed=True)}",
            )
        )
    return PublicationTable(
        item_id="table-2-language-primary",
        title="Frozen held-out language availability and personalization",
        evidence_roles=("primary",),
        source_ids=("primary-analysis",),
        columns=(
            "Profile",
            "Spans",
            "Target available",
            "Availability 95% CI",
            "Generic top-1 (all)",
            "Personalized top-1 (all)",
            "Top-1 delta given available (95% CI)",
            "Generic MRR given available",
            "Personalized MRR given available",
            "MRR delta given available (95% CI)",
        ),
        rows=tuple(rows),
        caption=(
            "Primary synthetic-language evidence. Top-1 rates marked “all” include unavailable "
            "targets as failures. Paired deltas and MRR are conditional on target availability; "
            "95% intervals resample complete messages within fixed profile strata."
        ),
    )


def _p300_estimate_ci(
    intervals: dict[tuple[str, str, str, str], PublicationInterval],
    variant: str,
    metric: str,
) -> str:
    item = _interval(intervals, "p300", "overall", variant, metric)
    return f"{_rate(item.estimate)} {_ci(item.lower_bound, item.upper_bound)}"


def _p300_tables(
    analysis: PublicationAnalysisResult,
) -> tuple[PublicationTable, PublicationTable]:
    estimates = _estimate_index(analysis)
    intervals = _interval_index(analysis)
    rows: list[tuple[str, ...]] = []
    for variant, label, role in (
        ("xdawn", "xDAWN-LDA", "Primary decoder"),
        ("eegnet", "EEGNet", "Secondary comparator"),
    ):
        reference = _estimate(estimates, "p300", "overall", variant, "auroc")

        rows.append(
            (
                label,
                role,
                str(reference.sample_count),
                _rate(_estimate(estimates, "p300", "overall", variant, "auroc").estimate),
                _rate(
                    _estimate(estimates, "p300", "overall", variant, "balanced_accuracy").estimate
                ),
                _rate(_estimate(estimates, "p300", "overall", variant, "brier_score").estimate),
                _rate(
                    _estimate(
                        estimates,
                        "p300",
                        "overall",
                        variant,
                        "expected_calibration_error",
                    ).estimate
                ),
                _p300_estimate_ci(intervals, variant, "exact_target_event_set_accuracy"),
                _p300_estimate_ci(intervals, variant, "target_event_recall_at_k"),
                _p300_estimate_ci(intervals, variant, "target_event_average_precision"),
                _p300_estimate_ci(intervals, variant, "top_event_hit_rate"),
            )
        )
    estimates_table = PublicationTable(
        item_id="table-3-p300-original-task",
        title="Held-out original-task Study P decoder performance",
        evidence_roles=("primary-with-secondary-comparator",),
        source_ids=("primary-analysis",),
        columns=(
            "Model",
            "Role",
            "Labeled epochs",
            "AUROC",
            "Balanced accuracy",
            "Brier",
            "ECE",
            "Exact event set (95% CI)",
            "Target-event recall@k (95% CI)",
            "Target-event AP (95% CI)",
            "Top-event hit (95% CI)",
        ),
        rows=tuple(rows),
        caption=(
            "Original-task public-EEG evidence over 21,491 held-out labeled epochs from three "
            "subjects. Selection intervals use a held-out-subject then selection-trial bootstrap. "
            "Selection metrics concern occurrence-level target events, not NeuroSelect symbols."
        ),
    )
    contrast_rows: list[tuple[str, ...]] = []
    metric_labels = (
        ("exact_target_event_set_accuracy", "Exact target-event set"),
        ("target_event_recall_at_k", "Target-event recall@k"),
        ("target_event_average_precision", "Target-event average precision"),
        ("top_event_hit_rate", "Top-event hit"),
        ("brier_score", "Brier (positive is worse)"),
    )
    for metric, label in metric_labels:
        item = _interval(intervals, "p300", "overall", "eegnet-minus-xdawn", metric)
        contrast_rows.append(
            (
                label,
                _signed(item.estimate),
                _ci(item.lower_bound, item.upper_bound, signed=True),
                item.sampling_unit,
            )
        )
    contrast_table = PublicationTable(
        item_id="table-3b-p300-paired-contrasts",
        title="Paired EEGNet-minus-xDAWN selection contrasts",
        evidence_roles=("primary-with-secondary-comparator",),
        source_ids=("primary-analysis",),
        columns=("Metric", "EEGNet - xDAWN", "95% CI", "Sampling unit"),
        rows=tuple(contrast_rows),
        caption=(
            "Secondary model comparison on identical held-out subjects and selection trials. "
            "Intervals crossing zero do not establish a selection-ranking difference; the "
            "strictly positive Brier contrast indicates worse EEGNet calibration."
        ),
    )
    return estimates_table, contrast_table


def _counterfactual_tables(
    analysis: PublicationAnalysisResult,
) -> tuple[PublicationTable, PublicationTable]:
    estimates = _estimate_index(analysis)
    rows: list[tuple[str, ...]] = []
    for variant, label in CONDITION_LABELS.items():
        reference = _estimate(
            estimates, "counterfactual", "overall", variant, "target_availability_rate"
        )
        rows.append(
            (
                label,
                str(reference.sample_count),
                _rate(reference.estimate),
                _rate(
                    _estimate(
                        estimates,
                        "counterfactual",
                        "overall",
                        variant,
                        "top_1_candidate_recall",
                    ).estimate
                ),
                _rate(
                    _estimate(
                        estimates,
                        "counterfactual",
                        "overall",
                        variant,
                        "selection_completion_rate",
                    ).estimate
                ),
                _rate(
                    _estimate(
                        estimates,
                        "counterfactual",
                        "overall",
                        variant,
                        "repeat_request_rate",
                    ).estimate
                ),
            )
        )
    conditions_table = PublicationTable(
        item_id="table-4-counterfactual-conditions",
        title="Offline counterfactual fusion conditions",
        evidence_roles=("primary",),
        source_ids=("primary-analysis",),
        columns=(
            "Condition",
            "Condition trials",
            "Target available",
            "Top-1 recall",
            "Selection completion",
            "Repeat request",
        ),
        rows=tuple(rows),
        caption=(
            "Primary counterfactual-replay evidence. Each condition contains 144 balanced trials "
            "from three held-out EEG subjects. Recorded P300 evidence was remapped to synthetic "
            "candidate menus; these are not live communication outcomes."
        ),
    )
    contrast_labels = {
        "f_complete_system-minus-a_bci_only": "Complete system - BCI only",
        "d_neural_personalized-minus-c_neural_language": ("Neural personalized - neural language"),
        "e_neural_personalized_rag-minus-d_neural_personalized": ("Add retrieval context"),
        "f_complete_system-minus-e_neural_personalized_rag": "Add safety policy",
    }
    metric_labels = {
        "top_1_candidate_recall": "Top-1 recall",
        "selection_completion_rate": "Selection completion",
        "repeat_request_rate": "Repeat request",
    }
    contrast_rows = tuple(
        (
            contrast_labels[item.contrast],
            metric_labels[item.metric],
            _signed(item.estimate),
            _ci(item.lower_bound, item.upper_bound, signed=True),
            item.sampling_unit,
        )
        for item in analysis.intervals
        if item.component == "counterfactual"
    )
    contrasts_table = PublicationTable(
        item_id="table-4b-counterfactual-contrasts",
        title="Prespecified paired counterfactual contrasts",
        evidence_roles=("primary",),
        source_ids=("primary-analysis",),
        columns=("Contrast", "Metric", "Difference", "95% CI", "Sampling unit"),
        rows=contrast_rows,
        caption=(
            "Paired differences resample complete messages within held-out EEG subjects. "
            "The complete system did not improve completion over BCI-only replay."
        ),
    )
    return conditions_table, contrasts_table


def _step3_metric(result: CandidateGenerationV2Result, scope: str) -> CandidateGenerationV2Metrics:
    return next(item for item in result.metrics if item.scope == scope)


def _step3_interval(
    result: CandidateGenerationV2Result,
    scope: str,
    metric: str,
) -> CandidateGenerationV2Interval:
    return next(item for item in result.intervals if item.scope == scope and item.metric == metric)


def _step4_metric(
    result: CandidateGenerationStep4Result,
    dataset: CandidateGenerationDataset,
    method: CandidateGenerationMethod,
    scope: str,
) -> CandidateGenerationStep4Metric:
    return next(
        item
        for item in result.metrics
        if item.dataset_id is dataset and item.method is method and item.scope == scope
    )


def _candidate_tables(
    step3: CandidateGenerationV2Result,
    step4: CandidateGenerationStep4Result,
) -> tuple[PublicationTable, PublicationTable]:
    original = _step3_metric(step3, "overall")
    rows: list[tuple[str, ...]] = [
        (
            "Existing test-exposed",
            "Frozen v1 reference",
            _rate(original.baseline_target_availability_rate),
            _rate(_step3_metric(step3, "span-0").baseline_target_availability_rate),
            _rate(cast(float, original.baseline_message_availability_rate)),
            "1.000",
        )
    ]
    for dataset, dataset_label in (
        (CandidateGenerationDataset.EXISTING_EXPOSED, "Existing test-exposed"),
        (CandidateGenerationDataset.ROBUSTNESS_HOLDOUT, "Held-out combinations"),
    ):
        for method in CandidateGenerationMethod:
            overall = _step4_metric(step4, dataset, method, "overall")
            rows.append(
                (
                    dataset_label,
                    METHOD_LABELS[method],
                    _rate(overall.availability_rate),
                    _rate(_step4_metric(step4, dataset, method, "opening").availability_rate),
                    _rate(
                        _step4_metric(step4, dataset, method, "complete_messages").availability_rate
                    ),
                    _rate(cast(float, overall.mean_selection_stages)),
                )
            )
    estimates_table = PublicationTable(
        item_id="table-5-candidate-generation",
        title="Exploratory target-blind candidate availability",
        evidence_roles=("exploratory",),
        source_ids=("candidate-generation-v2", "candidate-generation-step4"),
        columns=(
            "Dataset",
            "Method",
            "Span availability",
            "Opening availability",
            "Complete-message availability",
            "Mean planned/reached stages",
        ),
        rows=tuple(rows),
        caption=(
            "Exploratory synthetic evidence at a nine-candidate menu budget. The existing "
            "benchmark was inspected during v2 development; the held-out-combination benchmark "
            "was locked before Step 4 execution. Intended targets were used only after generation "
            "for scoring."
        ),
    )
    contrasts: list[tuple[str, ...]] = []
    for metric, label in (
        ("availability_delta", "v2 - frozen v1, spans"),
        ("message_availability_delta", "v2 - frozen v1, complete messages"),
    ):
        step3_interval = _step3_interval(step3, "overall", metric)
        contrasts.append(
            (
                "Existing test-exposed",
                label,
                _signed(step3_interval.estimate),
                _ci(
                    step3_interval.lower_bound,
                    step3_interval.upper_bound,
                    signed=True,
                ),
            )
        )
    selected_contrasts = {
        "full_v2-minus-no_profile_conditioning-overall",
        "full_v2-minus-no_grammar_routing-overall",
        "two_stage_opening-minus-full_v2-overall",
        "two_stage_opening-minus-full_v2-opening",
        "two_stage_opening-minus-full_v2-complete_messages",
    }
    for step4_contrast in step4.contrasts:
        if step4_contrast.contrast_id not in selected_contrasts:
            continue
        contrasts.append(
            (
                (
                    "Existing test-exposed"
                    if step4_contrast.dataset_id is CandidateGenerationDataset.EXISTING_EXPOSED
                    else "Held-out combinations"
                ),
                step4_contrast.contrast_id.replace("_", " "),
                _signed(step4_contrast.estimate),
                _ci(
                    step4_contrast.lower_bound,
                    step4_contrast.upper_bound,
                    signed=True,
                ),
            )
        )
    contrast_table = PublicationTable(
        item_id="table-5b-candidate-contrasts",
        title="Selected paired candidate-generation contrasts",
        evidence_roles=("exploratory",),
        source_ids=("candidate-generation-v2", "candidate-generation-step4"),
        columns=("Dataset", "Contrast", "Difference", "95% CI"),
        rows=tuple(contrasts),
        caption=(
            "Paired 10,000-resample intervals over complete messages within fixed synthetic "
            "profile strata. Step 3 v2 comparisons are test-exposed; Step 4 comparisons were "
            "locked before execution."
        ),
    )
    return estimates_table, contrast_table


def _opening_metric(
    result: OpeningGeneralizationResult,
    challenge: OpeningChallenge,
    method: OpeningMethod,
) -> OpeningGeneralizationMetric:
    return next(
        item
        for item in result.metrics
        if item.challenge == challenge.value and item.method is method and item.scope == "overall"
    )


def _opening_tables(
    opening: OpeningGeneralizationResult,
) -> tuple[PublicationTable, PublicationTable]:
    challenge_labels = {
        OpeningChallenge.HELDOUT_COMBINATION: "Held-out stem/content combinations",
        OpeningChallenge.HELDOUT_PARAPHRASE_FAMILY: "Unseen paraphrase family",
    }
    rows: list[tuple[str, ...]] = []
    for challenge, challenge_label in challenge_labels.items():
        for method in OpeningMethod:
            item = _opening_metric(opening, challenge, method)
            rows.append(
                (
                    challenge_label,
                    OPENING_METHOD_LABELS[method],
                    str(item.trial_count),
                    _rate(item.availability_rate),
                    str(item.planned_selections),
                    _rate(item.coverage_per_required_selection),
                    _rate(item.mean_menus_reached),
                    _rate(item.mean_candidate_exposures),
                )
            )
    estimates_table = PublicationTable(
        item_id="table-6-opening-generalization",
        title="Locked hierarchical opening-generalization results",
        evidence_roles=("exploratory",),
        source_ids=("opening-generalization",),
        columns=(
            "Challenge",
            "Method",
            "Openings",
            "Availability",
            "Planned selections",
            "Coverage per required selection",
            "Mean menus reached",
            "Mean candidate exposures",
        ),
        rows=tuple(rows),
        caption=(
            "Exploratory target-blind opening evidence under a maximum menu size of nine. "
            "Combination tests contain unseen exact pairs of observed components; paraphrase-"
            "family tests contain stems absent from fitting. Downstream menus use only simulated "
            "observed selections."
        ),
    )
    contrast_rows = tuple(
        (
            (
                "Held-out combinations"
                if item.challenge == OpeningChallenge.HELDOUT_COMBINATION.value
                else "Unseen paraphrase family"
            ),
            item.metric.replace("_", " "),
            (
                f"{OPENING_METHOD_LABELS[item.reference_method]} - "
                f"{OPENING_METHOD_LABELS[item.comparator_method]}"
            ),
            _signed(item.estimate),
            _ci(item.lower_bound, item.upper_bound, signed=True),
        )
        for item in opening.contrasts
    )
    contrasts_table = PublicationTable(
        item_id="table-6b-opening-contrasts",
        title="Paired hierarchical opening contrasts",
        evidence_roles=("exploratory",),
        source_ids=("opening-generalization",),
        columns=("Challenge", "Metric", "Contrast", "Difference", "95% CI"),
        rows=contrast_rows,
        caption=(
            "Paired 10,000-resample intervals over openings within fixed synthetic-profile "
            "strata. Zero availability for every unseen-family method is retained rather than "
            "omitted."
        ),
    )
    return estimates_table, contrasts_table


def _evidence_map_table(
    analysis: PublicationAnalysisResult,
    step3: CandidateGenerationV2Result,
    step4: CandidateGenerationStep4Result,
    opening: OpeningGeneralizationResult,
) -> PublicationTable:
    estimates = _estimate_index(analysis)
    language_n = _estimate(
        estimates, "language", "overall", "observed", "target_availability_rate"
    ).sample_count
    p300_n = _estimate(estimates, "p300", "overall", "xdawn", "auroc").sample_count
    counterfactual_n = _estimate(
        estimates,
        "counterfactual",
        "overall",
        "a_bci_only",
        "target_availability_rate",
    ).sample_count
    step3_n = _step3_metric(step3, "overall").trial_count
    robust_n = _step4_metric(
        step4,
        CandidateGenerationDataset.ROBUSTNESS_HOLDOUT,
        CandidateGenerationMethod.FULL_V2,
        "overall",
    ).denominator
    rows = (
        (
            "Primary",
            "Held-out language",
            f"{language_n} spans / 1000 messages / 4 fixed profiles",
            "Availability and paired ranking",
            "Synthetic teacher-forced language only",
        ),
        (
            "Primary",
            "xDAWN original-task EEG",
            f"{p300_n} labeled epochs / 3 held-out subjects",
            "Binary decoding and target-event ranking",
            "Occurrence-level Study P events, not symbols",
        ),
        (
            "Secondary comparator",
            "EEGNet original-task EEG",
            f"{p300_n} identical held-out epochs",
            "Paired model comparison",
            "No clear selection-ranking gain; worse calibration",
        ),
        (
            "Primary",
            "Counterfactual replay",
            f"{counterfactual_n} trials per condition / 3 held-out subjects",
            "Paired fusion and completion contrasts",
            "Offline remapping, not participant use",
        ),
        (
            "Exploratory test-exposed",
            "Candidate generation v2",
            f"{step3_n} existing benchmark spans",
            "Target-blind phrase availability",
            "Developer inspected the test benchmark",
        ),
        (
            "Exploratory locked",
            "Step 4 candidate ablations",
            f"{step3_n} exposed + {robust_n} combination-holdout spans",
            "Ablations and two-stage interface",
            "Synthetic developer-authored benchmark",
        ),
        (
            "Exploratory locked",
            "Opening generalization",
            (
                f"{opening.holdout_counts['combination_test_count']} combinations + "
                f"{opening.holdout_counts['family_test_count']} unseen-family openings"
            ),
            "Hierarchical composition and selection cost",
            "Closed-vocabulary interface evidence",
        ),
    )
    return PublicationTable(
        item_id="table-1-evidence-hierarchy",
        title="Evidence hierarchy and non-pooling boundaries",
        evidence_roles=(
            "evidence-map",
            "primary",
            "primary-with-secondary-comparator",
            "exploratory",
        ),
        source_ids=(
            "primary-analysis",
            "candidate-generation-v2",
            "candidate-generation-step4",
            "opening-generalization",
        ),
        columns=("Evidence tier", "Component", "Analysis units", "Role", "Interpretation limit"),
        rows=rows,
        caption=(
            "Every evidence tier is reported separately. No row is a live NeuroSelect user study, "
            "and no aggregate “system score” is computed across synthetic language, original-task "
            "EEG, counterfactual replay, and exploratory interface experiments."
        ),
    )


def build_tables(
    analysis: PublicationAnalysisResult,
    step3: CandidateGenerationV2Result,
    step4: CandidateGenerationStep4Result,
    opening: OpeningGeneralizationResult,
) -> tuple[PublicationTable, ...]:
    return (
        _evidence_map_table(analysis, step3, step4, opening),
        _language_table(analysis),
        *_p300_tables(analysis),
        *_counterfactual_tables(analysis),
        *_candidate_tables(step3, step4),
        *_opening_tables(opening),
    )


def _publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": BLACK,
            "axes.labelcolor": BLACK,
            "text.color": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "neuroselect-paper-display-v1",
        }
    )


def _render_figure(figure: Figure, spec: PublicationDisplaySpec) -> dict[Any, bytes]:
    outputs: dict[Any, bytes] = {}
    for suffix in spec.figure_formats:
        buffer = io.BytesIO()
        metadata: dict[str, Any]
        if suffix == "png":
            metadata = {"Software": "NeuroSelect paper-display-v1"}
        elif suffix == "svg":
            metadata = {"Creator": "NeuroSelect", "Date": None}
        else:
            metadata = {
                "Creator": "NeuroSelect",
                "CreationDate": None,
                "ModDate": None,
            }
        figure.savefig(
            buffer,
            format=suffix,
            dpi=spec.raster_dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        outputs[suffix] = buffer.getvalue()
    plt.close(figure)
    return outputs


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def _language_figure(
    analysis: PublicationAnalysisResult,
    spec: PublicationDisplaySpec,
) -> RenderedPublicationFigure:
    estimates = _estimate_index(analysis)
    intervals = _interval_index(analysis)
    scopes = tuple(PROFILE_LABELS)
    labels = [PROFILE_LABELS[scope] for scope in scopes]
    y = np.arange(len(scopes))[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    availability = [
        _estimate(estimates, "language", scope, "observed", "target_availability_rate").estimate
        for scope in scopes
    ]
    availability_intervals = [
        _interval(intervals, "language", scope, "rate", "target_availability_rate")
        for scope in scopes
    ]
    colors = [BLACK, BLUE, VERMILLION, GREEN, PURPLE]
    for position, point, interval, color in zip(
        y, availability, availability_intervals, colors, strict=True
    ):
        axes[0].errorbar(
            point,
            position,
            xerr=[[point - interval.lower_bound], [interval.upper_bound - point]],
            fmt="o",
            color=color,
            capsize=2.5,
            markersize=5,
            linewidth=1.2,
        )
    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(0, 0.36)
    axes[0].set_xlabel("Target availability (proportion)")
    axes[0].set_title("Candidate availability")
    axes[0].grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    axes[0].text(
        0.01,
        -0.24,
        "Complete-message availability: 0.000 (all profiles)",
        transform=axes[0].transAxes,
        color=GRAY,
        fontsize=7.2,
    )
    _panel_label(axes[0], "A")

    effects = [
        _interval(
            intervals,
            "language",
            scope,
            "personalized-minus-generic",
            "top1_delta_given_available",
        )
        for scope in scopes
    ]
    axes[1].axvline(0, color=GRAY, linewidth=0.8, linestyle="--")
    for position, item, color in zip(y, effects, colors, strict=True):
        axes[1].errorbar(
            item.estimate,
            position,
            xerr=[
                [item.estimate - item.lower_bound],
                [item.upper_bound - item.estimate],
            ],
            fmt="D",
            color=color,
            capsize=2.5,
            markersize=4.5,
            linewidth=1.2,
        )
    axes[1].set_yticks(y, labels)
    axes[1].set_xlim(-0.18, 0.34)
    axes[1].set_xlabel("Personalized - generic top-1")
    axes[1].set_title("Effect given target availability")
    axes[1].grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    _panel_label(axes[1], "B")
    figure.subplots_adjust(wspace=0.48, bottom=0.22)
    return RenderedPublicationFigure(
        item_id="figure-1-language-bottleneck",
        title="Frozen language availability and conditional personalization",
        evidence_roles=("primary",),
        source_ids=("primary-analysis",),
        caption=(
            "Primary synthetic-language evidence. (A) Exact target availability with 95% "
            "message-clustered bootstrap intervals. (B) Paired personalized-minus-generic top-1 "
            "difference conditional on availability. The concise profile is the only profile "
            "with a clearly negative conditional top-1 difference. Complete-message availability "
            "was zero."
        ),
        files=cast(Any, _render_figure(figure, spec)),
    )


def _p300_figure(
    analysis: PublicationAnalysisResult,
    spec: PublicationDisplaySpec,
) -> RenderedPublicationFigure:
    estimates = _estimate_index(analysis)
    intervals = _interval_index(analysis)
    selection_metrics = (
        ("exact_target_event_set_accuracy", "Exact event set"),
        ("target_event_recall_at_k", "Recall@k"),
        ("target_event_average_precision", "Average precision"),
        ("top_event_hit_rate", "Top-event hit"),
    )
    y = np.arange(len(selection_metrics))[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.35, 1]})
    for offset, variant, label, color, marker in (
        (-0.10, "xdawn", "xDAWN-LDA (primary)", BLUE, "o"),
        (0.10, "eegnet", "EEGNet (secondary)", ORANGE, "s"),
    ):
        for position, (metric, _) in zip(y, selection_metrics, strict=True):
            item = _interval(intervals, "p300", "overall", variant, metric)
            axes[0].errorbar(
                item.estimate,
                position + offset,
                xerr=[
                    [item.estimate - item.lower_bound],
                    [item.upper_bound - item.estimate],
                ],
                fmt=marker,
                color=color,
                capsize=2.5,
                markersize=4.5,
                linewidth=1.1,
            )
        axes[0].plot([], [], marker=marker, color=color, linestyle="none", label=label)
    axes[0].set_yticks(y, [label for _, label in selection_metrics])
    axes[0].set_xlim(0, 0.82)
    axes[0].set_xlabel("Selection metric (proportion)")
    axes[0].set_title("Occurrence-level target-event ranking")
    axes[0].grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    axes[0].legend(loc="upper right", frameon=False)
    _panel_label(axes[0], "A")

    metrics = ("brier_score", "expected_calibration_error")
    metric_labels = ("Brier", "ECE")
    positions = np.arange(len(metrics))
    width = 0.34
    for offset, variant, label, color, hatch in (
        (-width / 2, "xdawn", "xDAWN-LDA", BLUE, ""),
        (width / 2, "eegnet", "EEGNet", ORANGE, "//"),
    ):
        values = [
            _estimate(estimates, "p300", "overall", variant, metric).estimate for metric in metrics
        ]
        axes[1].bar(
            positions + offset,
            values,
            width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor=BLACK,
            linewidth=0.5,
        )
    axes[1].set_xticks(positions, metric_labels)
    axes[1].set_ylim(0, 0.25)
    axes[1].set_ylabel("Calibration error (lower is better)")
    axes[1].set_title("Epoch-level calibration")
    axes[1].grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    axes[1].legend(frameon=False)
    _panel_label(axes[1], "B")
    figure.subplots_adjust(wspace=0.43, bottom=0.18)
    return RenderedPublicationFigure(
        item_id="figure-2-p300-comparison",
        title="Original-task P300 decoder evidence",
        evidence_roles=("primary-with-secondary-comparator",),
        source_ids=("primary-analysis",),
        caption=(
            "Original-task public-EEG evidence. (A) xDAWN-LDA and the secondary EEGNet comparator "
            "with held-out-subject then selection-trial 95% intervals. Paired intervals for the "
            "three ranking differences included zero. (B) Epoch-level Brier score and expected "
            "calibration error; lower is better. EEGNet had substantially worse calibration."
        ),
        files=cast(Any, _render_figure(figure, spec)),
    )


def _counterfactual_figure(
    analysis: PublicationAnalysisResult,
    spec: PublicationDisplaySpec,
) -> RenderedPublicationFigure:
    estimates = _estimate_index(analysis)
    conditions = tuple(CONDITION_LABELS)
    labels = [CONDITION_LABELS[item] for item in conditions]
    y = np.arange(len(conditions))[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
    height = 0.27
    top1 = [
        _estimate(
            estimates, "counterfactual", "overall", condition, "top_1_candidate_recall"
        ).estimate
        for condition in conditions
    ]
    completion = [
        _estimate(
            estimates, "counterfactual", "overall", condition, "selection_completion_rate"
        ).estimate
        for condition in conditions
    ]
    repeat = [
        _estimate(estimates, "counterfactual", "overall", condition, "repeat_request_rate").estimate
        for condition in conditions
    ]
    axes[0].barh(y + height / 2, top1, height, color=BLUE, label="Top-1 recall")
    axes[0].barh(
        y - height / 2,
        completion,
        height,
        color=ORANGE,
        hatch="//",
        edgecolor=BLACK,
        linewidth=0.4,
        label="Selection completion",
    )
    axes[0].scatter(repeat, y, marker="x", color=VERMILLION, label="Repeat request", zorder=3)
    target_ceiling = _estimate(
        estimates,
        "counterfactual",
        "overall",
        "a_bci_only",
        "target_availability_rate",
    ).estimate
    axes[0].axvline(
        target_ceiling,
        color=GRAY,
        linestyle="--",
        linewidth=1,
        label="Target availability",
    )
    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(0, 0.31)
    axes[0].set_xlabel("Proportion")
    axes[0].set_title("Condition-level outcomes")
    axes[0].grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    condition_handles, condition_legend_labels = axes[0].get_legend_handles_labels()
    _panel_label(axes[0], "A")

    selected = [
        item
        for item in analysis.intervals
        if item.component == "counterfactual"
        and item.metric in {"top_1_candidate_recall", "selection_completion_rate"}
    ]
    contrast_order = (
        "f_complete_system-minus-a_bci_only",
        "d_neural_personalized-minus-c_neural_language",
        "e_neural_personalized_rag-minus-d_neural_personalized",
        "f_complete_system-minus-e_neural_personalized_rag",
    )
    contrast_short = ("Full - BCI", "Personalization", "Retrieval", "Safety")
    cy = np.arange(len(contrast_order))[::-1]
    axes[1].axvline(0, color=GRAY, linestyle="--", linewidth=0.8)
    for offset, metric, label, color, marker in (
        (-0.10, "top_1_candidate_recall", "Top-1", BLUE, "o"),
        (0.10, "selection_completion_rate", "Completion", ORANGE, "s"),
    ):
        for position, contrast in zip(cy, contrast_order, strict=True):
            item = next(
                row for row in selected if row.contrast == contrast and row.metric == metric
            )
            axes[1].errorbar(
                item.estimate,
                position + offset,
                xerr=[
                    [item.estimate - item.lower_bound],
                    [item.upper_bound - item.estimate],
                ],
                fmt=marker,
                color=color,
                capsize=2.5,
                markersize=4,
                linewidth=1.0,
            )
        axes[1].plot([], [], marker=marker, color=color, linestyle="none", label=label)
    axes[1].set_yticks(cy, contrast_short)
    axes[1].set_xlim(-0.04, 0.11)
    axes[1].set_xlabel("Paired difference")
    axes[1].set_title("Prespecified contrasts")
    axes[1].grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    axes[1].legend(frameon=False)
    _panel_label(axes[1], "B")
    figure.legend(
        condition_handles,
        condition_legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
    )
    figure.subplots_adjust(wspace=0.45, bottom=0.24)
    return RenderedPublicationFigure(
        item_id="figure-3-counterfactual-fusion",
        title="Offline counterfactual fusion outcomes",
        evidence_roles=("primary",),
        source_ids=("primary-analysis",),
        caption=(
            "Primary counterfactual-replay evidence. (A) Outcomes for the six prespecified "
            "conditions; the dashed line is target availability, not a performance target. "
            "(B) Paired 95% intervals from resampling complete messages within held-out subjects. "
            "The complete system did not improve completion over BCI-only replay."
        ),
        files=cast(Any, _render_figure(figure, spec)),
    )


def _candidate_figure(
    step4: CandidateGenerationStep4Result,
    spec: PublicationDisplaySpec,
) -> RenderedPublicationFigure:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharex=True)
    methods = tuple(CandidateGenerationMethod)
    labels = [METHOD_LABELS[item] for item in methods]
    y = np.arange(len(methods))[::-1]
    scopes = (
        ("overall", "Span", BLUE, ""),
        ("opening", "Opening", ORANGE, "//"),
        ("complete_messages", "Complete message", GREEN, ".."),
    )
    height = 0.22
    for axis, dataset, title in (
        (axes[0], CandidateGenerationDataset.EXISTING_EXPOSED, "Existing test-exposed"),
        (axes[1], CandidateGenerationDataset.ROBUSTNESS_HOLDOUT, "Held-out combinations"),
    ):
        for index, (scope, label, color, hatch) in enumerate(scopes):
            values = [
                _step4_metric(step4, dataset, method, scope).availability_rate for method in methods
            ]
            axis.barh(
                y + (index - 1) * height,
                values,
                height,
                color=color,
                hatch=hatch,
                edgecolor=BLACK,
                linewidth=0.35,
                label=label,
            )
        axis.set_yticks(y, labels)
        axis.set_xlim(0, 1.05)
        axis.set_xlabel("Availability (proportion)")
        axis.set_title(title)
        axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    candidate_handles, candidate_legend_labels = axes[1].get_legend_handles_labels()
    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B")
    figure.suptitle("Exploratory target-blind candidate generation", y=1.01, fontsize=10)
    figure.legend(
        candidate_handles,
        candidate_legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
    )
    figure.subplots_adjust(wspace=0.35, bottom=0.24)
    return RenderedPublicationFigure(
        item_id="figure-4-candidate-generation",
        title="Exploratory candidate-generation ablations and robustness",
        evidence_roles=("exploratory",),
        source_ids=("candidate-generation-step4",),
        caption=(
            "Exploratory synthetic evidence at a nine-candidate budget. (A) Existing benchmark "
            "that was exposed during development. (B) Exact stem-action combinations held out "
            "before Step 4 execution. The two-stage interface improves opening and complete-"
            "message availability but adds a selection stage."
        ),
        files=cast(Any, _render_figure(figure, spec)),
    )


def _opening_figure(
    opening: OpeningGeneralizationResult,
    spec: PublicationDisplaySpec,
) -> RenderedPublicationFigure:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True)
    methods = tuple(OpeningMethod)
    labels = ("One stage", "Two stages", "Three stages")
    y = np.arange(len(methods))[::-1]
    height = 0.32
    for axis, challenge, title in (
        (
            axes[0],
            OpeningChallenge.HELDOUT_COMBINATION,
            "Held-out component combinations",
        ),
        (
            axes[1],
            OpeningChallenge.HELDOUT_PARAPHRASE_FAMILY,
            "Unseen paraphrase family",
        ),
    ):
        availability = [
            _opening_metric(opening, challenge, method).availability_rate for method in methods
        ]
        efficiency = [
            _opening_metric(opening, challenge, method).coverage_per_required_selection
            for method in methods
        ]
        axis.barh(
            y + height / 2,
            availability,
            height,
            color=BLUE,
            label="Availability",
        )
        axis.barh(
            y - height / 2,
            efficiency,
            height,
            color=ORANGE,
            hatch="//",
            edgecolor=BLACK,
            linewidth=0.4,
            label="Coverage / required selection",
        )
        axis.set_yticks(y, labels)
        axis.set_xlim(0, 0.75)
        axis.set_xlabel("Proportion")
        axis.set_title(title)
        axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    opening_handles, opening_legend_labels = axes[1].get_legend_handles_labels()
    axes[1].text(
        0.5,
        0.13,
        "All methods: 0.000",
        transform=axes[1].transAxes,
        color=VERMILLION,
        fontsize=7.5,
        fontweight="bold",
        ha="center",
    )
    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B")
    figure.suptitle("Exploratory hierarchical opening generalization", y=1.01, fontsize=10)
    figure.legend(
        opening_handles,
        opening_legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
    )
    figure.subplots_adjust(wspace=0.38, bottom=0.26)
    return RenderedPublicationFigure(
        item_id="figure-5-opening-generalization",
        title="Hierarchical opening composition and closed-vocabulary boundary",
        evidence_roles=("exploratory",),
        source_ids=("opening-generalization",),
        caption=(
            "Locked exploratory synthetic evidence. Availability and availability per planned "
            "selection are shown for (A) unseen combinations of observed components and (B) "
            "paraphrase-family stems absent from fitting. Hierarchy improved composition of "
            "observed components but every method failed on unseen surface families."
        ),
        files=cast(Any, _render_figure(figure, spec)),
    )


def build_figures(
    analysis: PublicationAnalysisResult,
    step4: CandidateGenerationStep4Result,
    opening: OpeningGeneralizationResult,
    spec: PublicationDisplaySpec,
) -> tuple[RenderedPublicationFigure, ...]:
    _publication_style()
    return (
        _language_figure(analysis, spec),
        _p300_figure(analysis, spec),
        _counterfactual_figure(analysis, spec),
        _candidate_figure(step4, spec),
        _opening_figure(opening, spec),
    )


def main() -> None:
    args = parse_args()
    spec = load_publication_display_spec(args.config)
    analysis, step3, step4, opening = load_sources(spec)
    tables = build_tables(analysis, step3, step4, opening)
    figures = build_figures(analysis, step4, opening, spec)
    revision, source_tree_sha256 = git_state()
    if source_tree_sha256 is not None and not args.allow_dirty:
        raise ValueError(
            "publication display requires a clean source tree; commit the implementation or "
            "pass --allow-dirty only for development validation"
        )
    inventory, manifest = write_publication_display(
        spec,
        tables,
        figures,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    restored, restored_manifest = read_publication_display(
        args.output,
        require_publication_ready=not args.allow_dirty,
    )
    assert (restored, restored_manifest) == (inventory, manifest)
    print(f"Run: {manifest.run_id}")
    print(f"Tables: {sum(item.item_kind == 'table' for item in inventory.items)}")
    print(f"Figures: {sum(item.item_kind == 'figure' for item in inventory.items)}")
    print(f"Figure formats: {', '.join(spec.figure_formats)} at {spec.raster_dpi} dpi")
    print(f"Source manifests verified: {len(spec.sources)}")
    print(f"Publication ready: {inventory.publication_ready}")
    print(f"Manifest SHA-256: {manifest.digest()}")
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
