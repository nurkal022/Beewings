# BeeWings — convenience targets. Run `make help` for the list.
.DEFAULT_GOAL := help
PORT ?= 8000
IMAGE ?= beewings-api

.PHONY: help install models run api test docker-build docker-run docker-save compose-up compose-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with the API extra (use a venv first)
	pip install -e ".[api]"

models: ## Download model weights into checkpoints/ (from GitHub Release)
	bash scripts/download_models.sh

run: api ## Alias for `make api`

api: ## Start the REST API (http://localhost:$(PORT))
	BEEWINGS_API_PORT=$(PORT) beewings-api

test: ## Run the API test suite
	pytest tests/test_api.py -v

docker-build: ## Build the Docker image ($(IMAGE))
	docker build -t $(IMAGE) .

docker-run: ## Run the Docker image, exposing port $(PORT)
	docker run --rm -p $(PORT):8000 $(IMAGE)

docker-save: ## Export the image to beewings-api.tar.gz (offline transfer)
	docker save $(IMAGE):latest | gzip > beewings-api.tar.gz
	@echo "Wrote beewings-api.tar.gz — load with: docker load < beewings-api.tar.gz"

compose-up: ## Start via docker compose (mounts ./data_io -> /data)
	docker compose up --build

compose-down: ## Stop docker compose
	docker compose down
