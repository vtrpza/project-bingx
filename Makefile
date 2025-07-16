# Enterprise Trading Bot - Makefile
# ==================================

.PHONY: help install test test-unit test-integration test-coverage lint format clean run dev

# Default target
help:
	@echo "Enterprise Trading Bot - Available Commands:"
	@echo ""
	@echo "  install         Install all dependencies"
	@echo "  install-test    Install test dependencies"
	@echo "  test            Run all tests"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage   Run tests with coverage report"
	@echo "  test-parallel   Run tests in parallel"
	@echo "  lint            Run code linting"
	@echo "  format          Format code"
	@echo "  clean           Clean build artifacts"
	@echo "  run             Run the application"
	@echo "  dev             Run in development mode"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-test:
	pip install -r requirements-test.txt

install-all: install install-test

# Testing
test:
	pytest -v

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-coverage:
	pytest --cov=. --cov-report=html --cov-report=term-missing --cov-report=json

test-parallel:
	pytest -n auto --dist worksteal

test-fast:
	pytest -x --tb=short

test-slow:
	pytest -m slow -v

test-watch:
	pytest-watch

# Code Quality
lint:
	@echo "Running linting..."
	@python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@python -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "Formatting code..."
	@python -m black . --line-length=100
	@python -m isort . --profile=black

# Security
security-check:
	@echo "Running security check..."
	@python -m bandit -r . -ll

# Performance
benchmark:
	pytest tests/ -k "benchmark" --benchmark-only

# Development
run:
	python main.py

dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Database (if added later)
migrate:
	@echo "Database migrations not implemented yet"

# Docker (if added later)
docker-build:
	docker build -t trading-bot .

docker-run:
	docker run -p 8000:8000 trading-bot

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf coverage.json
	@rm -rf __pycache__/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete

# CI/CD
ci: lint test-coverage security-check
	@echo "CI pipeline completed"

# Documentation
docs:
	@echo "Documentation generation not implemented yet"

# Environment
env-check:
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Virtual environment: $(VIRTUAL_ENV)"

# Monitoring
health-check:
	@echo "Running health check..."
	@curl -f http://localhost:8000/health || echo "Service not running"

# Deployment
deploy-staging:
	@echo "Staging deployment not implemented yet"

deploy-prod:
	@echo "Production deployment not implemented yet"

# Logs
logs:
	@echo "Showing recent logs..."
	@tail -f logs/trading_bot.log 2>/dev/null || echo "No log file found"

# Version
version:
	@echo "Trading Bot Version: 1.0.0"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# All-in-one commands
setup: install-all
	@echo "Setup completed!"

check: lint test-unit
	@echo "Quick check completed!"

full-test: clean test-coverage security-check
	@echo "Full test suite completed!"