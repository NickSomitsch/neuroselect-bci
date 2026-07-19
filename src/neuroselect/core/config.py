"""Typed application configuration loaded from versioned YAML files."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeConfig(BaseModel):
    """Local filesystem and reproducibility settings."""

    model_config = ConfigDict(extra="forbid")

    environment: Literal["development", "test", "research"] = "development"
    artifacts_dir: Path = Path("artifacts")
    results_dir: Path = Path("results")
    deterministic: bool = True


class ServiceConfig(BaseModel):
    """Loopback-only service binding for the local-first MVP."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)

    @model_validator(mode="after")
    def require_loopback(self) -> ServiceConfig:
        if self.host == "localhost":
            return self
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as error:
            raise ValueError("service host must be localhost or a loopback IP address") from error
        if not address.is_loopback:
            raise ValueError("service host must be loopback-only")
        return self


class SessionPolicyConfig(BaseModel):
    """Safety-sensitive defaults shared by the API and interface."""

    model_config = ConfigDict(extra="forbid")

    candidate_count: Literal[4, 6, 8, 12] = 8
    maximum_phrase_tokens: int = Field(default=4, ge=1, le=8)
    final_confirmation_required: Literal[True] = True
    finalization_confirmation_ttl_seconds: int = Field(default=300, ge=60, le=900)


class AppConfig(BaseModel):
    """Top-level configuration for a local NeuroSelect process."""

    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    session_policy: SessionPolicyConfig = Field(default_factory=SessionPolicyConfig)


def load_app_config(path: str | Path) -> AppConfig:
    """Load and validate an application configuration without implicit overrides."""

    with Path(path).open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("application configuration must contain a YAML mapping")
    return AppConfig.model_validate(payload)
