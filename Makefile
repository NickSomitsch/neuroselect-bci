.PHONY: setup sync ui-install synthetic-data synthetic-knowledge fusion-smoke simulated-evaluation study-p-data p300-baseline p300-replay api format format-check lint typecheck test build verify

setup: sync ui-install

sync:
	uv sync --all-groups --locked

ui-install:
	pnpm --dir ui install --frozen-lockfile

synthetic-data:
	uv run python scripts/generate_synthetic_benchmark.py

synthetic-knowledge:
	uv run python scripts/initialize_synthetic_knowledge.py --replace

fusion-smoke:
	uv run python scripts/run_fusion_smoke.py

simulated-evaluation:
	uv run python scripts/run_simulated_evaluation.py

study-p-data:
	uv run python scripts/prepare_study_p.py $(STUDY_P_ARGS)

p300-baseline:
	uv run python scripts/train_p300_baseline.py $(P300_BASELINE_ARGS)

p300-replay:
	uv run python scripts/replay_p300.py $(P300_REPLAY_ARGS)

api:
	uv run python scripts/run_api.py

format:
	uv run ruff format src tests scripts
	pnpm --dir ui format

format-check:
	uv run ruff format --check src tests scripts
	pnpm --dir ui format:check

lint:
	uv run ruff check src tests scripts
	pnpm --dir ui lint

typecheck:
	uv run mypy src tests scripts
	pnpm --dir ui typecheck

test:
	uv run pytest
	pnpm --dir ui test --run

build:
	pnpm --dir ui build

verify: format-check lint typecheck test build
