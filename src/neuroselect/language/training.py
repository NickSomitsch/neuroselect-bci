"""Auditable MLX QLoRA command construction and adapter finalization."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from neuroselect.language.local_models import LocalModelConfig
from neuroselect.language.personalization import (
    PersonalizationAdapterManifest,
    sha256_file,
)
from neuroselect.language.personalization_data import PersonalizationCorpusManifest

DEFAULT_LORA_CONFIG = Path("configs/models/qwen3_4b_lora.yaml")


class LoraTrainingConfig(BaseModel):
    """Tracked resource-conscious MLX LoRA settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    trainer: Literal["mlx-lm"]
    trainer_revision: str = Field(min_length=1, max_length=128)
    fine_tune_type: Literal["lora"] = "lora"
    mask_prompt: Literal[True] = True
    num_layers: int = Field(default=16, ge=1, le=128)
    batch_size: int = Field(default=1, ge=1, le=32)
    iterations: int = Field(default=600, ge=1, le=100_000)
    validation_batches: int = Field(default=-1, ge=-1)
    test_batches: int = Field(default=-1, ge=-1)
    learning_rate: float = Field(default=1e-5, gt=0.0, le=1e-2)
    steps_per_report: int = Field(default=10, ge=1)
    steps_per_evaluation: int = Field(default=100, ge=1)
    save_every: int = Field(default=100, ge=1)
    maximum_sequence_length: int = Field(default=512, ge=64, le=32_768)
    gradient_checkpointing: bool = True
    seed: int = Field(default=20260723, ge=0)
    evaluate_test: bool = True

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def load_lora_training_config(
    path: str | Path = DEFAULT_LORA_CONFIG,
) -> LoraTrainingConfig:
    """Load strict training settings without importing MLX."""

    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("LoRA training configuration must contain a YAML mapping")
    return LoraTrainingConfig.model_validate(payload)


def build_mlx_lora_command(
    *,
    model_source: str | Path,
    corpus_dir: str | Path,
    adapter_dir: str | Path,
    config: LoraTrainingConfig,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Build the exact no-shell command used for MLX QLoRA training."""

    command = [
        python_executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(model_source),
        "--train",
        "--data",
        str(corpus_dir),
        "--fine-tune-type",
        config.fine_tune_type,
        "--num-layers",
        str(config.num_layers),
        "--batch-size",
        str(config.batch_size),
        "--iters",
        str(config.iterations),
        "--val-batches",
        str(config.validation_batches),
        "--learning-rate",
        str(config.learning_rate),
        "--steps-per-report",
        str(config.steps_per_report),
        "--steps-per-eval",
        str(config.steps_per_evaluation),
        "--adapter-path",
        str(adapter_dir),
        "--save-every",
        str(config.save_every),
        "--max-seq-length",
        str(config.maximum_sequence_length),
        "--seed",
        str(config.seed),
    ]
    if config.mask_prompt:
        command.append("--mask-prompt")
    if config.gradient_checkpointing:
        command.append("--grad-checkpoint")
    if config.evaluate_test:
        command.extend(("--test", "--test-batches", str(config.test_batches)))
    return tuple(command)


def finalize_personalization_adapter(
    *,
    adapter_dir: str | Path,
    corpus_manifest: PersonalizationCorpusManifest,
    model_config: LocalModelConfig,
    training_config: LoraTrainingConfig,
    mlx_lm_version: str,
    trained_at: datetime | None = None,
) -> PersonalizationAdapterManifest:
    """Checksum learned weights and write their immutable manifest."""

    destination = Path(adapter_dir)
    weights_path = destination / "adapters.safetensors"
    if not weights_path.is_file():
        raise ValueError(f"trained adapter weights not found: {weights_path}")
    corpus_digest = corpus_manifest.digest()
    training_digest = training_config.digest()
    identity_payload = "\0".join(
        (
            corpus_manifest.profile_id,
            model_config.model_id,
            model_config.model_revision,
            corpus_digest,
            training_digest,
        )
    )
    identity = hashlib.sha256(identity_payload.encode()).hexdigest()
    manifest = PersonalizationAdapterManifest(
        schema_version="1.0",
        adapter_id=f"lora-{corpus_manifest.profile_id}-{identity[:16]}",
        profile_id=corpus_manifest.profile_id,
        base_model_id=model_config.model_id,
        base_model_revision=model_config.model_revision,
        adapter_file=weights_path.name,
        adapter_sha256=sha256_file(weights_path),
        source_corpus_manifest_sha256=corpus_digest,
        training_config_sha256=training_digest,
        trainer_revision=training_config.trainer_revision,
        mlx_lm_version=mlx_lm_version,
        trained_at=trained_at or datetime.now(UTC),
        validation_evaluated=training_config.validation_batches != 0,
        test_evaluated=(training_config.evaluate_test and training_config.test_batches != 0),
    )
    content = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    (destination / "manifest.json").write_text(content + "\n", encoding="utf-8")
    return manifest
