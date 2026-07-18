from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from neuroselect.api import create_app
from neuroselect.bci import SeededNeuralSimulator, SimulationConfig
from neuroselect.orchestration import SessionOrchestrator
from neuroselect.retrieval import KnowledgeRecordInput, SQLiteKnowledgeStore
from neuroselect.synthetic import load_profiles

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def make_client() -> tuple[TestClient, SessionOrchestrator]:
    profiles = load_profiles()
    store = SQLiteKnowledgeStore(":memory:")
    for profile in profiles:
        for record in profile.knowledge:
            store.add(
                profile_id=profile.profile_id,
                record=KnowledgeRecordInput.model_validate(record.model_dump()),
                at_time=NOW,
            )
    service = SessionOrchestrator(
        profiles=profiles,
        knowledge_store=store,
        simulator=SeededNeuralSimulator(
            SimulationConfig(
                target_concentration=100.0,
                lapse_probability=0.0,
                ambiguous_probability=0.0,
            )
        ),
        clock=lambda: NOW,
        session_id_factory=lambda: "session-api-test",
        nonce_factory=lambda: "api-confirmation-nonce-12345",
    )
    return TestClient(create_app(service)), service


def test_manual_api_flow_requires_explicit_final_confirmation() -> None:
    client, service = make_client()
    try:
        assert client.get("/health").json() == {
            "status": "ok",
            "service": "neuroselect",
            "api_version": "v1",
        }
        created = client.post(
            "/api/v1/sessions",
            json={"profile_id": "synthetic-concise", "input_mode": "manual"},
        )
        assert created.status_code == 201
        session_id = created.json()["session"]["session_id"]

        round_response = client.post(
            f"/api/v1/sessions/{session_id}/rounds",
            json={"simulated_target_index": 0},
        )
        assert round_response.status_code == 200
        round_payload = round_response.json()
        assert round_payload["ranking"]["disposition"] == "abstain"
        candidate_id = round_payload["ranking"]["fused_top_candidate_id"]

        selected = client.post(
            f"/api/v1/sessions/{session_id}/actions",
            json={"action": "select", "candidate_id": candidate_id, "explicit": True},
        )
        assert selected.status_code == 200
        assert selected.json()["session"]["state"] == "draft"
        assert selected.json()["session"]["confirmed_spans"]

        challenge_response = client.post(f"/api/v1/sessions/{session_id}/finalization")
        assert challenge_response.status_code == 200
        challenge = challenge_response.json()
        finalized = client.post(
            f"/api/v1/sessions/{session_id}/finalization/confirm",
            json={
                "session_id": session_id,
                "text_sha256": challenge["text_sha256"],
                "confirmation_nonce": challenge["confirmation_nonce"],
                "explicit_confirmation": True,
                "high_risk_acknowledged": False,
            },
        )
        assert finalized.status_code == 200
        assert finalized.json()["session"]["state"] == "finalized"
    finally:
        service.close()


def test_api_maps_not_found_conflicts_and_validation_without_leaking_state() -> None:
    client, service = make_client()
    try:
        missing = client.get("/api/v1/sessions/missing")
        assert missing.status_code == 404
        assert "session not found" in missing.json()["detail"]

        unknown_profile = client.post(
            "/api/v1/sessions",
            json={"profile_id": "unknown", "input_mode": "simulation"},
        )
        assert unknown_profile.status_code == 422

        created = client.post(
            "/api/v1/sessions",
            json={"profile_id": "synthetic-concise", "input_mode": "simulation"},
        )
        session_id = created.json()["session"]["session_id"]
        empty_finalization = client.post(f"/api/v1/sessions/{session_id}/finalization")
        assert empty_finalization.status_code == 422

        started = client.post(
            f"/api/v1/sessions/{session_id}/rounds",
            json={"simulated_target_index": 0},
        )
        assert started.status_code == 200
        conflict = client.post(
            f"/api/v1/sessions/{session_id}/rounds",
            json={"simulated_target_index": 0},
        )
        assert conflict.status_code == 409

        invalid_action = client.post(
            f"/api/v1/sessions/{session_id}/actions",
            json={"action": "select", "explicit": True},
        )
        assert invalid_action.status_code == 422
        forbidden_finalize_action = client.post(
            f"/api/v1/sessions/{session_id}/actions",
            json={"action": "finalize", "explicit": True},
        )
        assert forbidden_finalize_action.status_code == 422
    finally:
        service.close()
