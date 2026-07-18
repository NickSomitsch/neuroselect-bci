"""Versioned local HTTP routes over the explicit-confirmation session service."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from neuroselect.core.models import FinalizationRequest
from neuroselect.orchestration import (
    CreateSessionRequest,
    FinalizationChallenge,
    ManualTextRequest,
    RoundRequest,
    SelectionConfirmationRequest,
    SessionActionRequest,
    SessionConflictError,
    SessionNotFoundError,
    SessionOrchestrator,
    SessionValidationError,
    SessionView,
    build_demo_orchestrator,
)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok"] = "ok"
    service: Literal["neuroselect"] = "neuroselect"
    api_version: Literal["v1"] = "v1"


def create_app(service: SessionOrchestrator | None = None) -> FastAPI:
    """Create the local API; callers can inject a deterministic service for tests."""

    orchestrator = service or build_demo_orchestrator()
    app = FastAPI(
        title="NeuroSelect local research API",
        version="0.1.0-dev",
        description=(
            "Local simulated/manual BCI communication research API. "
            "Generated candidates never constitute confirmed user text."
        ),
    )
    app.state.orchestrator = orchestrator

    @app.exception_handler(SessionNotFoundError)
    async def handle_not_found(_: Request, error: SessionNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(error)})

    @app.exception_handler(SessionConflictError)
    async def handle_conflict(_: Request, error: SessionConflictError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(error)})

    @app.exception_handler(SessionValidationError)
    async def handle_validation(_: Request, error: SessionValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(error)},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post(
        "/api/v1/sessions",
        response_model=SessionView,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(request: CreateSessionRequest) -> SessionView:
        return orchestrator.create_session(request)

    @app.get("/api/v1/sessions/{session_id}", response_model=SessionView)
    async def get_session(session_id: str) -> SessionView:
        return orchestrator.get_session(session_id)

    @app.post("/api/v1/sessions/{session_id}/rounds", response_model=SessionView)
    async def start_round(session_id: str, request: RoundRequest) -> SessionView:
        return orchestrator.start_round(session_id, request)

    @app.post("/api/v1/sessions/{session_id}/actions", response_model=SessionView)
    async def apply_action(session_id: str, request: SessionActionRequest) -> SessionView:
        return orchestrator.apply_action(session_id, request)

    @app.post(
        "/api/v1/sessions/{session_id}/selection-confirmation",
        response_model=SessionView,
    )
    async def resolve_selection(
        session_id: str, request: SelectionConfirmationRequest
    ) -> SessionView:
        return orchestrator.resolve_selection(session_id, request)

    @app.post("/api/v1/sessions/{session_id}/manual-text", response_model=SessionView)
    async def append_manual_text(session_id: str, request: ManualTextRequest) -> SessionView:
        return orchestrator.append_manual_text(session_id, request)

    @app.post(
        "/api/v1/sessions/{session_id}/finalization",
        response_model=FinalizationChallenge,
    )
    async def request_finalization(session_id: str) -> FinalizationChallenge:
        return orchestrator.request_finalization(session_id)

    @app.post(
        "/api/v1/sessions/{session_id}/finalization/confirm",
        response_model=SessionView,
    )
    async def confirm_finalization(session_id: str, request: FinalizationRequest) -> SessionView:
        return orchestrator.confirm_finalization(session_id, request)

    @app.post(
        "/api/v1/sessions/{session_id}/finalization/reject",
        response_model=SessionView,
    )
    async def reject_finalization(session_id: str) -> SessionView:
        return orchestrator.reject_finalization(session_id)

    return app
