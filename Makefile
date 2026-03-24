.PHONY: help dev test lint format typecheck build up down logs migrate seed clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development
dev: ## Start development environment
	docker compose up -d postgres redis minio
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-all: ## Start all services including monitoring
	docker compose up -d

frontend: ## Start frontend development server
	cd frontend && npm run dev

# Testing
test: ## Run all tests
	cd backend && python -m pytest tests/ -v

test-unit: ## Run unit tests only
	cd backend && python -m pytest tests/unit/ -v

test-integration: ## Run integration tests
	cd backend && python -m pytest tests/integration/ -v -m integration

test-ml: ## Run ML tests
	cd backend && python -m pytest tests/ml/ -v -m ml

test-cov: ## Run tests with coverage report
	cd backend && python -m pytest tests/ --cov=app --cov-report=html --cov-report=term-missing

# Code Quality
lint: ## Run linter
	cd backend && ruff check app/ tests/

format: ## Format code
	cd backend && ruff format app/ tests/

typecheck: ## Run type checker
	cd backend && mypy app/

quality: lint typecheck ## Run all code quality checks

# Database
migrate: ## Run database migrations
	cd backend && alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create MSG="description")
	cd backend && alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Rollback last migration
	cd backend && alembic downgrade -1

seed: ## Seed database with initial data
	cd backend && python -m app.db.seed

# Docker
build: ## Build Docker images
	docker compose build

up: ## Start all containers
	docker compose up -d

down: ## Stop all containers
	docker compose down

logs: ## View container logs
	docker compose logs -f

# Cleanup
clean: ## Remove generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Load Testing
loadtest: ## Run load tests with locust
	cd backend && locust -f tests/load/locustfile.py --host=http://localhost:8000
