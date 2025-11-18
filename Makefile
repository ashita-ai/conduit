.PHONY: help install test test-cov type-check lint format clean all

help:
	@echo "Conduit Development Commands"
	@echo ""
	@echo "  make install     Install development dependencies"
	@echo "  make test        Run tests with coverage"
	@echo "  make test-cov    Run tests and show coverage report"
	@echo "  make type-check  Run mypy type checking"
	@echo "  make lint        Run ruff linting"
	@echo "  make format      Format code with black"
	@echo "  make clean       Remove build artifacts"
	@echo "  make all         Run format, lint, type-check, and test"

install:
	pip install -e ".[dev]"

test:
	pytest --cov=conduit --cov-report=term-missing

test-cov:
	pytest --cov=conduit --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

type-check:
	mypy conduit/

lint:
	ruff check conduit/

format:
	black conduit/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: format lint type-check test
