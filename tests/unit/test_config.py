from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.core.config import AppConfig, ServiceConfig, SessionPolicyConfig, load_app_config


def test_loads_versioned_demo_config() -> None:
    config = load_app_config("configs/demo/default.yaml")

    assert config.service.host == "127.0.0.1"
    assert config.session_policy.candidate_count == 8
    assert config.session_policy.final_confirmation_required is True


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_service_accepts_only_loopback_variants(host: str) -> None:
    assert ServiceConfig(host=host).host == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
def test_service_rejects_non_loopback_host(host: str) -> None:
    with pytest.raises(ValidationError, match="loopback"):
        ServiceConfig(host=host)


def test_policy_rejects_evidence_weight_overcommitment() -> None:
    with pytest.raises(ValidationError, match="cannot exceed one"):
        SessionPolicyConfig(minimum_neural_weight=0.8, maximum_non_neural_weight=0.3)


def test_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_app_config(config_path)


def test_app_config_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"unknown": True})
