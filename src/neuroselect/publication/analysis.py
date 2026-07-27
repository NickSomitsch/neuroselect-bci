"""Deterministic publication statistics over frozen NeuroSelect evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from neuroselect.decoding import (
    BinaryDecoderMetrics,
    DecoderEvaluation,
    EpochPrediction,
    selection_metrics_from_predictions,
)
from neuroselect.eeg import P300Label, verify_sha256
from neuroselect.evaluation import capture_runtime_environment
from neuroselect.evaluation.counterfactual_artifacts import read_counterfactual_artifacts
from neuroselect.evaluation.language_artifacts import read_held_out_language_artifacts
from neuroselect.evaluation.language_benchmark import LanguageBenchmarkTrial
from neuroselect.evaluation.models import EvaluationCondition, TrialRecord
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus
from neuroselect.publication.protocol import (
    PublicationProtocolSpec,
    assess_publication_protocol,
    load_publication_protocol,
)
from neuroselect.ranking import RankingDisposition

DEFAULT_PUBLICATION_ANALYSIS_CONFIG = Path("configs/publication/primary_analysis_v1.yaml")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class PublicationAnalysisSpec(BaseModel):
    """Locked statistical recipe applied after the publication-protocol freeze."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    analysis_id: str = Field(min_length=1, max_length=160)
    analysis_revision: Literal["offline-primary-analysis-v1"]
    analyzed_at: datetime
    publication_protocol: Path
    eegnet_artifacts: Path
    eegnet_required: bool = False
    expected_eegnet_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    bootstrap_resamples: int = Field(default=10_000, ge=2_000, le=100_000)
    bootstrap_seed: int = Field(default=20260727, ge=0)

    @model_validator(mode="after")
    def validate_recipe(self) -> PublicationAnalysisSpec:
        if self.analyzed_at.tzinfo is None or self.analyzed_at.utcoffset() is None:
            raise ValueError("publication analysis time must include a timezone")
        if self.eegnet_required != (self.expected_eegnet_manifest_sha256 is not None):
            raise ValueError("required EEGNet analysis must pin its manifest SHA-256")
        return self

    def digest(self) -> str:
        return _sha256_text(_canonical_json(self.model_dump(mode="json")))


class PublicationEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component: Literal["language", "p300", "counterfactual"]
    scope: str
    variant: str
    metric: str
    sample_count: int = Field(ge=1)
    estimate: float


class PublicationInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component: Literal["language", "p300", "counterfactual"]
    scope: str
    contrast: str
    metric: str
    estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = Field(gt=0.5, lt=1.0)
    resamples: int = Field(ge=2_000)
    sampling_unit: str

    @model_validator(mode="after")
    def validate_bounds(self) -> PublicationInterval:
        if self.lower_bound > self.upper_bound:
            raise ValueError("publication interval bounds must be ordered")
        return self


class PublicationAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    analysis_id: str
    analyzed_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: dict[str, str]
    estimates: tuple[PublicationEstimate, ...] = Field(min_length=1)
    intervals: tuple[PublicationInterval, ...] = Field(min_length=1)
    eegnet_included: bool
    limitations: tuple[str, ...] = Field(min_length=1)

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


def load_publication_analysis_spec(
    path: str | Path = DEFAULT_PUBLICATION_ANALYSIS_CONFIG,
) -> PublicationAnalysisSpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("publication analysis config must contain a YAML mapping")
    return PublicationAnalysisSpec.model_validate(payload)


def _interval(
    values: Sequence[float],
    *,
    component: Literal["language", "p300", "counterfactual"],
    scope: str,
    contrast: str,
    metric: str,
    estimate: float,
    spec: PublicationAnalysisSpec,
    sampling_unit: str,
) -> PublicationInterval:
    alpha = (1.0 - spec.confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(values, dtype=np.float64), (alpha, 1.0 - alpha))
    return PublicationInterval(
        component=component,
        scope=scope,
        contrast=contrast,
        metric=metric,
        estimate=estimate,
        lower_bound=float(lower),
        upper_bound=float(upper),
        confidence_level=spec.confidence_level,
        resamples=spec.bootstrap_resamples,
        sampling_unit=sampling_unit,
    )


def _language_metrics(trials: Sequence[LanguageBenchmarkTrial]) -> dict[str, float]:
    if not trials:
        raise ValueError("language publication metrics require trials")
    available = [trial for trial in trials if trial.target_available]
    messages: dict[str, list[LanguageBenchmarkTrial]] = defaultdict(list)
    for trial in trials:
        messages[trial.message_id].append(trial)

    def _hit(trial: LanguageBenchmarkTrial, personalized: bool, rank: int) -> float:
        observed = trial.personalized_rank if personalized else trial.generic_rank
        return float(observed is not None and observed <= rank)

    result = {
        "target_availability_rate": len(available) / len(trials),
        "message_target_availability_rate": float(
            np.mean([all(item.target_available for item in rows) for rows in messages.values()])
        ),
        "generic_top1_unconditional": float(np.mean([_hit(item, False, 1) for item in trials])),
        "personalized_top1_unconditional": float(np.mean([_hit(item, True, 1) for item in trials])),
        "generic_top3_unconditional": float(np.mean([_hit(item, False, 3) for item in trials])),
        "personalized_top3_unconditional": float(np.mean([_hit(item, True, 3) for item in trials])),
    }
    if not available:
        return result
    generic_rr = [1.0 / item.generic_rank for item in available if item.generic_rank is not None]
    personalized_rr = [
        1.0 / item.personalized_rank for item in available if item.personalized_rank is not None
    ]
    result.update(
        {
            "generic_mrr_given_available": float(np.mean(generic_rr)),
            "personalized_mrr_given_available": float(np.mean(personalized_rr)),
            "mrr_delta_given_available": float(
                np.mean(np.asarray(personalized_rr) - np.asarray(generic_rr))
            ),
            "top1_delta_given_available": float(
                np.mean([_hit(item, True, 1) - _hit(item, False, 1) for item in available])
            ),
            "top3_delta_given_available": float(
                np.mean([_hit(item, True, 3) - _hit(item, False, 3) for item in available])
            ),
            "top1_delta_unconditional": (
                result["personalized_top1_unconditional"] - result["generic_top1_unconditional"]
            ),
            "top3_delta_unconditional": (
                result["personalized_top3_unconditional"] - result["generic_top3_unconditional"]
            ),
        }
    )
    return result


def _resample_language_trials(
    trials: Sequence[LanguageBenchmarkTrial],
    rng: np.random.Generator,
) -> list[LanguageBenchmarkTrial]:
    by_profile: dict[str, dict[str, list[LanguageBenchmarkTrial]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trial in trials:
        by_profile[trial.profile_id][trial.message_id].append(trial)
    sampled: list[LanguageBenchmarkTrial] = []
    for messages in by_profile.values():
        identifiers = tuple(sorted(messages))
        for index in rng.integers(0, len(identifiers), size=len(identifiers)):
            sampled.extend(messages[identifiers[int(index)]])
    return sampled


def _language_analysis(
    trials: Sequence[LanguageBenchmarkTrial],
    spec: PublicationAnalysisSpec,
) -> tuple[list[PublicationEstimate], list[PublicationInterval]]:
    estimates: list[PublicationEstimate] = []
    intervals: list[PublicationInterval] = []
    scopes: dict[str, list[LanguageBenchmarkTrial]] = {"overall": list(trials)}
    for profile in sorted({trial.profile_id for trial in trials}):
        scopes[profile] = [trial for trial in trials if trial.profile_id == profile]
    interval_metrics = (
        "target_availability_rate",
        "message_target_availability_rate",
        "mrr_delta_given_available",
        "top1_delta_given_available",
        "top3_delta_given_available",
        "top1_delta_unconditional",
        "top3_delta_unconditional",
    )
    for scope, rows in scopes.items():
        point = _language_metrics(rows)
        for metric, value in sorted(point.items()):
            estimates.append(
                PublicationEstimate(
                    component="language",
                    scope=scope,
                    variant="observed",
                    metric=metric,
                    sample_count=len(rows),
                    estimate=value,
                )
            )
        rng = np.random.default_rng(spec.bootstrap_seed + sum(scope.encode()))
        bootstrap: dict[str, list[float]] = {metric: [] for metric in interval_metrics}
        for _ in range(spec.bootstrap_resamples):
            sampled_metrics = _language_metrics(_resample_language_trials(rows, rng))
            for metric in interval_metrics:
                bootstrap[metric].append(sampled_metrics[metric])
        for metric in interval_metrics:
            contrast = "rate" if metric.endswith("_rate") else "personalized-minus-generic"
            intervals.append(
                _interval(
                    bootstrap[metric],
                    component="language",
                    scope=scope,
                    contrast=contrast,
                    metric=metric,
                    estimate=point[metric],
                    spec=spec,
                    sampling_unit="messages_within_fixed_profile_strata",
                )
            )

    for span_index in sorted({trial.span_index for trial in trials}):
        rows = [trial for trial in trials if trial.span_index == span_index]
        estimates.append(
            PublicationEstimate(
                component="language",
                scope=f"span-{span_index}",
                variant="observed",
                metric="target_availability_rate",
                sample_count=len(rows),
                estimate=float(np.mean([item.target_available for item in rows])),
            )
        )
    for word_count in sorted({len(trial.intended_text.split()) for trial in trials}):
        rows = [trial for trial in trials if len(trial.intended_text.split()) == word_count]
        estimates.append(
            PublicationEstimate(
                component="language",
                scope=f"target-words-{word_count}",
                variant="observed",
                metric="target_availability_rate",
                sample_count=len(rows),
                estimate=float(np.mean([item.target_available for item in rows])),
            )
        )
    profiles = sorted({trial.profile_id for trial in trials})
    for excluded in profiles:
        rows = [trial for trial in trials if trial.profile_id != excluded]
        estimates.append(
            PublicationEstimate(
                component="language",
                scope=f"leave-out-{excluded}",
                variant="sensitivity",
                metric="mrr_delta_given_available",
                sample_count=len(rows),
                estimate=_language_metrics(rows)["mrr_delta_given_available"],
            )
        )
    return estimates, intervals


def _expected_calibration_error(
    labels: np.ndarray[Any, np.dtype[np.int8]],
    probabilities: np.ndarray[Any, np.dtype[np.float64]],
    bins: int = 10,
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(labels[mask])) - float(np.mean(probabilities[mask]))
            )
    return result


def _binary_metrics(predictions: Sequence[EpochPrediction]) -> BinaryDecoderMetrics:
    labeled = [item for item in predictions if item.true_label is not P300Label.UNKNOWN]
    labels = np.asarray([item.true_label is P300Label.TARGET for item in labeled], dtype=np.int8)
    probabilities = np.asarray([item.target_probability for item in labeled], dtype=np.float64)
    predicted = probabilities >= 0.5
    return BinaryDecoderMetrics(
        auroc=float(roc_auc_score(labels, probabilities)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
        brier_score=float(brier_score_loss(labels, probabilities)),
        negative_log_likelihood=float(log_loss(labels, probabilities, labels=(0, 1))),
        expected_calibration_error=_expected_calibration_error(labels, probabilities),
    )


def _read_safe_decoder_evaluation(
    directory: Path,
) -> tuple[DecoderEvaluation, RunManifest]:
    manifest = RunManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    evaluation_path = directory / "evaluation.json"
    output = next(item for item in manifest.outputs if item.uri == "artifact://evaluation.json")
    verify_sha256(evaluation_path, output.sha256)
    evaluation = DecoderEvaluation.model_validate_json(evaluation_path.read_text(encoding="utf-8"))
    if manifest.run_kind is not RunKind.EEG_ORIGINAL_TASK or evaluation.dataset_sha256 != next(
        item.sha256 for item in manifest.datasets if item.uri == "dataset://study-p/model-test"
    ):
        raise ValueError("P300 evaluation does not agree with its manifest")
    return evaluation, manifest


def _selection_metric_values(predictions: Sequence[EpochPrediction]) -> dict[str, float]:
    _, selection = selection_metrics_from_predictions(predictions)
    if selection is None:
        raise ValueError("P300 selection metrics require scorable trials")
    return {
        "exact_target_event_set_accuracy": selection.exact_target_event_set_accuracy,
        "target_event_recall_at_k": selection.target_event_recall_at_k,
        "target_event_average_precision": selection.target_event_average_precision,
        "top_event_hit_rate": selection.top_event_hit_rate,
    }


def _p300_trial_rows(
    predictions: Sequence[EpochPrediction],
) -> dict[tuple[str, str], dict[str, float]]:
    groups: dict[tuple[str, str], list[EpochPrediction]] = defaultdict(list)
    for item in predictions:
        groups[(item.subject_id, item.selection_trial_id)].append(item)
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for key, trial in groups.items():
        labeled = [item for item in trial if item.true_label is not P300Label.UNKNOWN]
        values = _selection_metric_values(labeled)
        values["brier_score"] = float(
            np.mean(
                [
                    (item.target_probability - float(item.true_label is P300Label.TARGET)) ** 2
                    for item in labeled
                ]
            )
        )
        rows[key] = values
    return rows


def _hierarchical_trial_sample(
    keys: Sequence[tuple[str, str]],
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    by_subject: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in keys:
        by_subject[key[0]].append(key)
    subjects = tuple(sorted(by_subject))
    sampled: list[tuple[str, str]] = []
    for subject_index in rng.integers(0, len(subjects), size=len(subjects)):
        subject_keys = by_subject[subjects[int(subject_index)]]
        sampled.extend(
            subject_keys[int(index)]
            for index in rng.integers(0, len(subject_keys), size=len(subject_keys))
        )
    return sampled


def _mean_rows(
    rows: dict[tuple[str, str], dict[str, float]],
    keys: Iterable[tuple[str, str]],
) -> dict[str, float]:
    selected = [rows[key] for key in keys]
    return {metric: float(np.mean([row[metric] for row in selected])) for metric in selected[0]}


def _p300_analysis(
    evaluations: dict[str, DecoderEvaluation],
    spec: PublicationAnalysisSpec,
) -> tuple[list[PublicationEstimate], list[PublicationInterval]]:
    estimates: list[PublicationEstimate] = []
    intervals: list[PublicationInterval] = []
    rows_by_model: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for model, evaluation in evaluations.items():
        subjects = sorted({item.subject_id for item in evaluation.predictions})
        scopes = {"overall": list(evaluation.predictions)}
        scopes.update(
            {
                subject: [item for item in evaluation.predictions if item.subject_id == subject]
                for subject in subjects
            }
        )
        for scope, predictions in scopes.items():
            binary = _binary_metrics(predictions)
            selection = _selection_metric_values(predictions)
            for metric, value in {
                **binary.model_dump(),
                **selection,
            }.items():
                estimates.append(
                    PublicationEstimate(
                        component="p300",
                        scope=scope,
                        variant=model,
                        metric=metric,
                        sample_count=len(predictions),
                        estimate=float(value),
                    )
                )
        rows_by_model[model] = _p300_trial_rows(evaluation.predictions)

    keys = tuple(sorted(rows_by_model["xdawn"]))
    if any(set(rows) != set(keys) for rows in rows_by_model.values()):
        raise ValueError("P300 comparator evaluations must cover identical selection trials")
    rng = np.random.default_rng(spec.bootstrap_seed + 31)
    metrics = tuple(next(iter(rows_by_model["xdawn"].values())))
    distributions: dict[tuple[str, str], list[float]] = {
        (model, metric): [] for model in rows_by_model for metric in metrics
    }
    deltas: dict[str, list[float]] = {metric: [] for metric in metrics if "eegnet" in rows_by_model}
    for _ in range(spec.bootstrap_resamples):
        sampled_keys = _hierarchical_trial_sample(keys, rng)
        sampled = {model: _mean_rows(rows, sampled_keys) for model, rows in rows_by_model.items()}
        for model, values in sampled.items():
            for metric, value in values.items():
                distributions[(model, metric)].append(value)
        if "eegnet" in sampled:
            for metric in metrics:
                deltas[metric].append(sampled["eegnet"][metric] - sampled["xdawn"][metric])
    observed = {model: _mean_rows(rows, keys) for model, rows in rows_by_model.items()}
    for (model, metric), bootstrap_values in distributions.items():
        intervals.append(
            _interval(
                bootstrap_values,
                component="p300",
                scope="overall",
                contrast=model,
                metric=metric,
                estimate=observed[model][metric],
                spec=spec,
                sampling_unit="held_out_subject_then_selection_trial",
            )
        )
    for metric, bootstrap_values in deltas.items():
        intervals.append(
            _interval(
                bootstrap_values,
                component="p300",
                scope="overall",
                contrast="eegnet-minus-xdawn",
                metric=metric,
                estimate=observed["eegnet"][metric] - observed["xdawn"][metric],
                spec=spec,
                sampling_unit="paired_held_out_subject_then_selection_trial",
            )
        )
    return estimates, intervals


def _counterfactual_value(record: TrialRecord, metric: str) -> float:
    if metric == "top_1_candidate_recall":
        return float(record.top_1_correct)
    if metric == "selection_completion_rate":
        return float(record.explicit_selection_completed)
    if metric == "repeat_request_rate":
        return float(record.disposition is RankingDisposition.REQUEST_REPEAT)
    raise ValueError(f"unsupported counterfactual publication metric: {metric}")


def _counterfactual_sample(
    records: Sequence[TrialRecord],
    rng: np.random.Generator,
) -> list[TrialRecord]:
    by_subject: dict[str, dict[tuple[str, str], list[TrialRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if record.eeg_subject_id is None:
            raise ValueError("counterfactual publication records require EEG subject provenance")
        by_subject[record.eeg_subject_id][(record.profile_id, record.message_id)].append(record)
    subjects = tuple(sorted(by_subject))
    sampled: list[TrialRecord] = []
    for subject_index in rng.integers(0, len(subjects), size=len(subjects)):
        messages = by_subject[subjects[int(subject_index)]]
        message_ids = tuple(sorted(messages))
        for message_index in rng.integers(0, len(message_ids), size=len(message_ids)):
            sampled.extend(messages[message_ids[int(message_index)]])
    return sampled


def _condition_mean(
    records: Sequence[TrialRecord],
    condition: EvaluationCondition,
    metric: str,
) -> float:
    values = [
        _counterfactual_value(record, metric) for record in records if record.condition is condition
    ]
    if not values:
        raise ValueError(f"counterfactual condition is absent: {condition.value}")
    return float(np.mean(values))


def _counterfactual_analysis(
    records: Sequence[TrialRecord],
    spec: PublicationAnalysisSpec,
) -> tuple[list[PublicationEstimate], list[PublicationInterval]]:
    estimates: list[PublicationEstimate] = []
    intervals: list[PublicationInterval] = []
    conditions = tuple(
        sorted({record.condition for record in records}, key=lambda item: item.value)
    )
    metrics = (
        "top_1_candidate_recall",
        "selection_completion_rate",
        "repeat_request_rate",
    )
    for condition in conditions:
        rows = [record for record in records if record.condition is condition]
        for metric in metrics:
            estimates.append(
                PublicationEstimate(
                    component="counterfactual",
                    scope="overall",
                    variant=condition.value,
                    metric=metric,
                    sample_count=len(rows),
                    estimate=_condition_mean(rows, condition, metric),
                )
            )
        estimates.append(
            PublicationEstimate(
                component="counterfactual",
                scope="overall",
                variant=condition.value,
                metric="target_availability_rate",
                sample_count=len(rows),
                estimate=float(np.mean([record.target_available for record in rows])),
            )
        )
    contrasts = (
        (
            EvaluationCondition.F_COMPLETE_SYSTEM,
            EvaluationCondition.A_BCI_ONLY,
            ("top_1_candidate_recall", "selection_completion_rate"),
        ),
        (
            EvaluationCondition.D_NEURAL_PERSONALIZED,
            EvaluationCondition.C_NEURAL_LANGUAGE,
            ("top_1_candidate_recall", "selection_completion_rate"),
        ),
        (
            EvaluationCondition.E_NEURAL_PERSONALIZED_RAG,
            EvaluationCondition.D_NEURAL_PERSONALIZED,
            ("top_1_candidate_recall", "selection_completion_rate"),
        ),
        (
            EvaluationCondition.F_COMPLETE_SYSTEM,
            EvaluationCondition.E_NEURAL_PERSONALIZED_RAG,
            (
                "top_1_candidate_recall",
                "selection_completion_rate",
                "repeat_request_rate",
            ),
        ),
    )
    point: dict[tuple[EvaluationCondition, EvaluationCondition, str], float] = {}
    distributions: dict[tuple[EvaluationCondition, EvaluationCondition, str], list[float]] = {}
    for condition, reference, contrast_metrics in contrasts:
        for metric in contrast_metrics:
            key = condition, reference, metric
            point[key] = _condition_mean(records, condition, metric) - _condition_mean(
                records, reference, metric
            )
            distributions[key] = []
    rng = np.random.default_rng(spec.bootstrap_seed + 47)
    for _ in range(spec.bootstrap_resamples):
        sampled = _counterfactual_sample(records, rng)
        for key in distributions:
            condition, reference, metric = key
            distributions[key].append(
                _condition_mean(sampled, condition, metric)
                - _condition_mean(sampled, reference, metric)
            )
    for (condition, reference, metric), values in distributions.items():
        intervals.append(
            _interval(
                values,
                component="counterfactual",
                scope="overall",
                contrast=f"{condition.value}-minus-{reference.value}",
                metric=metric,
                estimate=point[(condition, reference, metric)],
                spec=spec,
                sampling_unit="held_out_subject_then_complete_message",
            )
        )
    return estimates, intervals


def _source_by_id(spec: PublicationProtocolSpec, source_id: str) -> Any:
    return next(source for source in spec.evidence_sources if source.source_id == source_id)


def build_publication_analysis(
    spec: PublicationAnalysisSpec,
) -> PublicationAnalysisResult:
    """Build source-pinned primary publication statistics without model execution."""

    protocol = load_publication_protocol(spec.publication_protocol)
    readiness = assess_publication_protocol(protocol)
    if not readiness.protocol_ready:
        raise ValueError("publication protocol or frozen evidence is not ready")
    if (
        spec.bootstrap_resamples != protocol.analysis_commitments.bootstrap_resamples
        or spec.bootstrap_seed != protocol.analysis_commitments.bootstrap_seed
    ):
        raise ValueError("publication analysis bootstrap settings differ from the protocol")

    language_source = _source_by_id(protocol, "held-out-language")
    language, language_manifest = read_held_out_language_artifacts(language_source.path)
    xdawn_source = _source_by_id(protocol, "xdawn-original-task")
    xdawn, xdawn_manifest = _read_safe_decoder_evaluation(xdawn_source.path)
    counterfactual_source = _source_by_id(protocol, "counterfactual-research")
    counterfactual, counterfactual_manifest = read_counterfactual_artifacts(
        counterfactual_source.path
    )
    source_manifests = {
        "held-out-language": language_manifest.digest(),
        "xdawn-original-task": xdawn_manifest.digest(),
        "counterfactual-research": counterfactual_manifest.digest(),
    }
    evaluations = {"xdawn": xdawn}
    if spec.eegnet_required:
        eegnet, eegnet_manifest = _read_safe_decoder_evaluation(spec.eegnet_artifacts)
        if eegnet_manifest.digest() != spec.expected_eegnet_manifest_sha256:
            raise ValueError("EEGNet manifest differs from the publication analysis pin")
        if eegnet.dataset_sha256 != xdawn.dataset_sha256:
            raise ValueError("xDAWN and EEGNet must use the identical held-out test dataset")
        evaluations["eegnet"] = eegnet
        source_manifests["eegnet-comparator"] = eegnet_manifest.digest()

    language_estimates, language_intervals = _language_analysis(language.trials, spec)
    p300_estimates, p300_intervals = _p300_analysis(evaluations, spec)
    counterfactual_estimates, counterfactual_intervals = _counterfactual_analysis(
        counterfactual.trial_records, spec
    )
    return PublicationAnalysisResult(
        analysis_id=spec.analysis_id,
        analyzed_at=spec.analyzed_at,
        config_sha256=spec.digest(),
        protocol_sha256=protocol.digest(),
        source_manifest_sha256=source_manifests,
        estimates=tuple(language_estimates + p300_estimates + counterfactual_estimates),
        intervals=tuple(language_intervals + p300_intervals + counterfactual_intervals),
        eegnet_included="eegnet" in evaluations,
        limitations=(
            "Intervals are descriptive for fixed synthetic profiles and three held-out EEG "
            "subjects.",
            "Language ranking is conditional on target-blind candidate availability.",
            "Selection metrics rank occurrence-level Study P target events, not NeuroSelect "
            "symbols.",
            "Counterfactual intervals resample subjects then complete messages and are not "
            "live-use evidence.",
            "The existing evidence interpretation is retrospective; only these added analyses "
            "were locked prospectively.",
        ),
    )


def _csv_content(rows: Sequence[BaseModel]) -> str:
    payload = [row.model_dump(mode="json") for row in rows]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=tuple(payload[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(payload)
    return output.getvalue()


def write_publication_analysis(
    result: PublicationAnalysisResult,
    spec: PublicationAnalysisSpec,
    output_dir: str | Path,
    *,
    git_sha: str,
    source_tree_sha256: str | None,
    overwrite: bool = False,
) -> RunManifest:
    destination = Path(output_dir)
    analysis_path = destination / "analysis.json"
    estimates_path = destination / "estimates.csv"
    intervals_path = destination / "intervals.csv"
    manifest_path = destination / "manifest.json"
    paths = (analysis_path, estimates_path, intervals_path, manifest_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite publication analysis: {existing}")
    destination.mkdir(parents=True, exist_ok=True)
    contents = {
        analysis_path: result.canonical_json() + "\n",
        estimates_path: _csv_content(result.estimates),
        intervals_path: _csv_content(result.intervals),
    }
    for path, content in contents.items():
        path.write_text(content, encoding="utf-8")
    package_versions, device = capture_runtime_environment()
    manifest = RunManifest(
        run_id=f"publication-analysis-{_sha256_text(result.canonical_json())[:20]}",
        run_kind=RunKind.PUBLICATION_ANALYSIS,
        status=RunStatus.COMPLETED,
        started_at=result.analyzed_at,
        completed_at=result.analyzed_at,
        git_sha=git_sha,
        config_sha256=spec.digest(),
        random_seeds={"hierarchical_bootstrap": spec.bootstrap_seed},
        package_versions=package_versions,
        device=device,
        datasets=tuple(
            ArtifactRef(
                artifact_id=source_id,
                uri=f"artifact://source-manifest/{source_id}",
                sha256=digest,
            )
            for source_id, digest in sorted(result.source_manifest_sha256.items())
        ),
        outputs=tuple(
            ArtifactRef(
                artifact_id=f"publication-{path.stem}",
                uri=f"artifact://{path.name}",
                sha256=_sha256_text(content),
                revision=spec.analysis_revision,
            )
            for path, content in contents.items()
        ),
        metadata={
            "protocol_sha256": result.protocol_sha256,
            "estimate_count": len(result.estimates),
            "interval_count": len(result.intervals),
            "eegnet_included": result.eegnet_included,
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest


def read_publication_analysis(
    directory: str | Path,
) -> tuple[PublicationAnalysisResult, RunManifest]:
    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    for name in ("analysis.json", "estimates.csv", "intervals.csv"):
        path = source / name
        expected = next(
            item.sha256 for item in manifest.outputs if item.uri == f"artifact://{name}"
        )
        if _sha256_text(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"publication analysis SHA-256 mismatch: {name}")
    result = PublicationAnalysisResult.model_validate_json(
        (source / "analysis.json").read_text(encoding="utf-8")
    )
    if (
        manifest.run_kind is not RunKind.PUBLICATION_ANALYSIS
        or manifest.config_sha256 != result.config_sha256
        or {item.artifact_id: item.sha256 for item in manifest.datasets}
        != result.source_manifest_sha256
    ):
        raise ValueError("publication analysis manifest does not agree with its result")
    return result, manifest
