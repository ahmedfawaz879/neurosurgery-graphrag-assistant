.PHONY: install test lint run-harness report serve docker-build docker-run deploy-render deploy-fly

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests scripts

run-harness:
	python -m src.eval.run_harness --out results/run.csv

report:
	python scripts/generate_report.py \
		--run-csv results/run.csv \
		--summary-out results/summary.csv \
		--figure-out results/figures/gaprag_metrics_by_system.png

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t neurosurgery-graphrag-assistant .

docker-run:
	docker compose up --build

deploy-render:
	@echo "Render deploys automatically from the branch connected in the Render"
	@echo "dashboard once render.yaml is present (Blueprint deploy) -- push to that"
	@echo "branch, or run 'render deploy' with the Render CLI + RENDER_API_KEY set."
	@echo "See render.yaml and docs/ARCHITECTURE.md."

deploy-fly:
	fly deploy
