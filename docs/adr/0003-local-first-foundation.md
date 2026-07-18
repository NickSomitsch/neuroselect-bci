# ADR 0003: Local-first Apple Silicon foundation

- Status: accepted
- Date: 2026-07-17

## Decision

- Use Python 3.12 with `uv` and a strict `src/` package layout.
- Use React, TypeScript, Vite, Node.js 22, and `pnpm` for the accessible web interface.
- Use typed Pydantic models as the source of truth for service and experiment contracts.
- Target localhost-only execution on an Apple M4 MacBook Air with 16 GB unified memory.
- Use MLX-LM for the default local quantized language model and LoRA path in later phases.
- Do not download models or datasets during repository setup or continuous integration.

## Consequences

FastAPI, MNE, MOABB, PyTorch, MLX-LM, and sentence-transformers are intentionally deferred until the phases that exercise them. This keeps the foundation fast to install and avoids platform-specific failures before relevant functionality exists.
