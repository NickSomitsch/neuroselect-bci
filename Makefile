.PHONY: setup sync local-language-sync local-language-cuda-sync language-cuda-preflight language-model-cache ui-install synthetic-data synthetic-knowledge language-personalization-data language-smoke language-lora language-evaluation language-research-adapter language-research-evaluation language-research-evaluation-cuda language-research-verify language-cloud-source-bundle language-cloud-bundle language-cloud-verify language-cloud-extract fusion-smoke simulated-evaluation counterfactual-input counterfactual-fusion counterfactual-evaluation counterfactual-research-input counterfactual-research-evaluation research-readiness development-report research-report publication-protocol-check publication-analysis release-check release study-p-data study-p-research-data p300-baseline p300-research-baseline p300-research-audit p300-eegnet p300-eegnet-pilot p300-eegnet-research p300-replay api format format-check lint typecheck test build package verify

STUDY_P_RESEARCH_SUBJECTS = P_01 P_02 P_03 P_04 P_05 P_06 P_07 P_08 P_09 P_10 P_11 P_12 P_13 P_14 P_15 P_16 P_17 P_18 P_19

setup: sync ui-install

sync:
	uv sync --all-groups --locked

local-language-sync:
	uv sync --all-groups --extra local-language --locked

local-language-cuda-sync:
	uv sync --extra local-language-cuda --no-dev --locked

language-cuda-preflight:
	uv run --extra local-language-cuda --no-dev python scripts/check_language_cuda.py

language-model-cache:
	uv run --extra local-language-cuda --no-dev python scripts/cache_language_model.py $(LANGUAGE_MODEL_CACHE_ARGS)

ui-install:
	pnpm --dir ui install --frozen-lockfile

synthetic-data:
	uv run python scripts/generate_synthetic_benchmark.py

synthetic-knowledge:
	uv run python scripts/initialize_synthetic_knowledge.py --replace

language-personalization-data:
	uv run python scripts/prepare_language_personalization.py

language-smoke:
	uv run python scripts/run_personalized_language.py

language-lora:
	uv run --extra local-language python scripts/train_language_lora.py $(LANGUAGE_LORA_ARGS)

language-evaluation:
	uv run --extra local-language python scripts/run_held_out_language_evaluation.py $(LANGUAGE_EVALUATION_ARGS)

language-research-adapter:
	@test -n "$(RESEARCH_PROFILE)" || (echo "RESEARCH_PROFILE is required" >&2; exit 2)
	uv run --extra local-language python scripts/train_language_lora.py \
		--corpus artifacts/language/personalization-v1/$(RESEARCH_PROFILE) \
		--output artifacts/models/language-lora/$(RESEARCH_PROFILE)-research-v1 \
		--training-config configs/models/qwen3_4b_lora.yaml \
		$(RESEARCH_ADAPTER_ARGS)

language-research-evaluation:
	uv run --extra local-language python scripts/run_held_out_language_evaluation.py \
		--config configs/experiments/held_out_language_personalization_research.yaml \
		--adapter-suffix=-research-v1 \
		--output artifacts/evaluation/held-out-language-personalization-research-v1 \
		$(LANGUAGE_RESEARCH_EVALUATION_ARGS)

language-research-evaluation-cuda:
	uv run --extra local-language-cuda --no-dev python scripts/run_held_out_language_evaluation.py \
		--config configs/experiments/held_out_language_personalization_research.yaml \
		--adapter-suffix=-research-v1 \
		--output artifacts/evaluation/held-out-language-personalization-research-v1 \
		$(LANGUAGE_RESEARCH_EVALUATION_ARGS)

language-research-verify:
	uv run python scripts/verify_language_research_evaluation.py \
		$(LANGUAGE_RESEARCH_VERIFY_ARGS)

language-cloud-source-bundle:
	uv run python scripts/create_step11_source_bundle.py \
		$(LANGUAGE_CLOUD_SOURCE_BUNDLE_ARGS)

language-cloud-bundle:
	uv run python scripts/manage_language_cloud_bundle.py create \
		$(LANGUAGE_CLOUD_BUNDLE_ARGS)

language-cloud-verify:
	@test -n "$(LANGUAGE_CLOUD_BUNDLE)" || (echo "LANGUAGE_CLOUD_BUNDLE is required" >&2; exit 2)
	uv run python scripts/manage_language_cloud_bundle.py verify \
		$(LANGUAGE_CLOUD_BUNDLE)

language-cloud-extract:
	@test -n "$(LANGUAGE_CLOUD_BUNDLE)" || (echo "LANGUAGE_CLOUD_BUNDLE is required" >&2; exit 2)
	uv run python scripts/manage_language_cloud_bundle.py extract \
		$(LANGUAGE_CLOUD_BUNDLE) \
		$(LANGUAGE_CLOUD_EXTRACT_ARGS)

fusion-smoke:
	uv run python scripts/run_fusion_smoke.py

simulated-evaluation:
	uv run python scripts/run_simulated_evaluation.py

counterfactual-input:
	uv run python scripts/prepare_counterfactual_input.py $(COUNTERFACTUAL_INPUT_ARGS)

counterfactual-fusion:
	uv run python scripts/run_counterfactual_fusion.py $(COUNTERFACTUAL_FUSION_ARGS)

counterfactual-evaluation:
	uv run python scripts/run_counterfactual_fusion.py \
		--input artifacts/evaluation/counterfactual-input-development-v1/input.json \
		--config configs/experiments/counterfactual_fusion_development.yaml \
		--output artifacts/evaluation/counterfactual-fusion-development-v1 \
		$(COUNTERFACTUAL_EVALUATION_ARGS)

counterfactual-research-input: research-readiness
	uv run python scripts/prepare_counterfactual_input.py \
		--language-artifacts artifacts/evaluation/held-out-language-personalization-research-v1 \
		--decoder-artifacts artifacts/models/p300-xdawn-lda-research-v1 \
		--preparation-config configs/experiments/counterfactual_input_research.yaml \
		--fusion-config configs/experiments/counterfactual_fusion_research.yaml \
		--output artifacts/evaluation/counterfactual-input-research-v1 \
		$(COUNTERFACTUAL_RESEARCH_INPUT_ARGS)

counterfactual-research-evaluation:
	uv run python scripts/run_counterfactual_fusion.py \
		--input artifacts/evaluation/counterfactual-input-research-v1/input.json \
		--config configs/experiments/counterfactual_fusion_research.yaml \
		--output artifacts/evaluation/counterfactual-fusion-research-v1 \
		$(COUNTERFACTUAL_RESEARCH_EVALUATION_ARGS)

research-readiness:
	uv run python scripts/check_research_readiness.py $(RESEARCH_READINESS_ARGS)

development-report:
	uv run python scripts/build_research_report.py \
		--config configs/release/development_evidence_report.yaml \
		--output artifacts/reports/neuroselect-development-evidence-v1 \
		$(DEVELOPMENT_REPORT_ARGS)

research-report:
	uv run python scripts/build_research_report.py --overwrite $(RESEARCH_REPORT_ARGS)

publication-protocol-check:
	uv run python scripts/check_publication_protocol.py $(PUBLICATION_PROTOCOL_ARGS)

publication-analysis:
	uv run python scripts/build_publication_analysis.py $(PUBLICATION_ANALYSIS_ARGS)

release-check:
	uv run python scripts/check_release.py $(RELEASE_CHECK_ARGS)

study-p-data:
	uv run python scripts/prepare_study_p.py $(STUDY_P_ARGS)

study-p-research-data:
	uv run python scripts/prepare_study_p.py \
		--download \
		--accept-license \
		--subjects $(STUDY_P_RESEARCH_SUBJECTS) \
		--source-partitions train \
		--download-workers 16 \
		--overwrite \
		$(STUDY_P_RESEARCH_DATA_ARGS)
	uv run python scripts/check_p300_research_evidence.py --data-only

p300-baseline:
	uv run python scripts/train_p300_baseline.py $(P300_BASELINE_ARGS)

p300-research-baseline:
	uv run python scripts/check_p300_research_evidence.py --data-only
	uv run python scripts/train_p300_baseline.py \
		--output artifacts/models/p300-xdawn-lda-research-v1 \
		$(P300_RESEARCH_BASELINE_ARGS)

p300-research-audit:
	uv run python scripts/check_p300_research_evidence.py $(P300_RESEARCH_AUDIT_ARGS)

p300-eegnet:
	uv run python scripts/train_eegnet.py $(P300_EEGNET_ARGS)

p300-eegnet-pilot:
	uv run python scripts/train_eegnet.py \
		--config configs/decoding/eegnet_pilot.yaml \
		--output artifacts/models/p300-eegnet-pilot-v1 \
		--batch-limit-per-partition 5 \
		--skip-drift \
		$(P300_EEGNET_PILOT_ARGS)

p300-eegnet-research:
	uv run python scripts/train_eegnet.py \
		--config configs/decoding/eegnet_research.yaml \
		--output artifacts/models/p300-eegnet-research-v1 \
		--skip-drift \
		$(P300_EEGNET_RESEARCH_ARGS)

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

package:
	uv build

verify: format-check lint typecheck test build

release: verify package research-report release-check
	uv run python scripts/check_release.py --report artifacts/reports/neuroselect-research-release-v1
