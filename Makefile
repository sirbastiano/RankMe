.PHONY: install install-dev test lint format type-check clean build docs help

# Default target
all: install-dev

# PDM installation commands
install:
	@echo "Installing rankme package..."
	pdm install --prod

install-dev:
	@echo "Installing rankme package with development dependencies..."
	pdm install

# Testing
test:
	@echo "Running tests..."
	pdm run pytest

test-cov:
	@echo "Running tests with coverage..."
	pdm run pytest --cov=rankme --cov-report=html --cov-report=term

test-verbose:
	@echo "Running tests in verbose mode..."
	pdm run pytest -v

# Code quality
lint:
	@echo "Running linting..."
	pdm run flake8 rankme tests
	pdm run mypy rankme

format:
	@echo "Formatting code..."
	pdm run black rankme tests
	pdm run isort rankme tests

format-check:
	@echo "Checking code formatting..."
	pdm run black --check rankme tests
	pdm run isort --check-only rankme tests

type-check:
	@echo "Running type checking..."
	pdm run mypy rankme

# Pre-commit hooks
pre-commit-install:
	@echo "Installing pre-commit hooks..."
	pdm run pre-commit install

pre-commit-run:
	@echo "Running pre-commit on all files..."
	pdm run pre-commit run --all-files

# Build and publish
build:
	@echo "Building package..."
	pdm build

publish-test:
	@echo "Publishing to test PyPI..."
	pdm publish --repository testpypi

publish:
	@echo "Publishing to PyPI..."
	pdm publish

# Documentation
docs:
	@echo "Building documentation..."
	pdm run sphinx-build -b html docs docs/_build/html

docs-serve:
	@echo "Serving documentation locally..."
	cd docs/_build/html && python -m http.server 8000

# CI Testing Commands
ci-docs:
	@echo "Testing documentation build like CI..."
	pdm install -G dev -G docs
	pdm run sphinx-build -b html docs docs/_build/html

ci-security:
	@echo "Testing security checks like CI..."
	pdm install -G dev
	pdm run bandit -r rankme/ -ll
	pdm run pip-audit --desc --format=json || echo "Security scan completed"

ci-build:
	@echo "Testing package build like CI..."
	pdm install
	pdm build
	@ls -la dist/

ci-all: ci-docs ci-security ci-build
	@echo "All CI checks completed successfully!"

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean
	@echo "Cleaning all artifacts including PDM environment..."
	pdm venv remove --yes in-project || true

# Development workflow
dev-setup: install-dev pre-commit-install
	@echo "Development environment setup complete!"

check: format-check lint type-check test
	@echo "All checks passed!"

# Quick development commands
quick-test:
	@echo "Running quick test subset..."
	pdm run pytest tests/ -x

fix: format lint
	@echo "Auto-fixing code issues..."

# CI/CD simulation
ci-test:
	@echo "Running CI-like test suite..."
	pdm run pytest tests/ -v --cov=rankme --cov-report=xml --cov-report=term-missing

ci-check: format-check lint type-check ci-test
	@echo "All CI checks passed!"

security-check:
	@echo "Running security checks..."
	pdm add -dG dev bandit[toml] safety
	pdm run bandit -r rankme/ -ll  # Use low-low severity threshold
	pdm run safety scan --json || echo "Safety scan completed with warnings"

benchmark:
	@echo "Running performance benchmarks..."
	pdm add -dG dev pytest-benchmark
	pdm run pytest tests/ -k benchmark --benchmark-only

# Help
help:
	@echo "Available commands:"
	@echo "  install          Install package only"
	@echo "  install-dev      Install package with dev dependencies"
	@echo "  test             Run tests"
	@echo "  test-cov         Run tests with coverage"
	@echo "  test-verbose     Run tests in verbose mode"
	@echo "  lint             Run linting (flake8, mypy)"
	@echo "  format           Format code (black, isort)"
	@echo "  format-check     Check code formatting"
	@echo "  type-check       Run type checking"
	@echo "  build            Build package"
	@echo "  publish-test     Publish to test PyPI"
	@echo "  publish          Publish to PyPI"
	@echo "  docs             Build documentation"
	@echo "  docs-serve       Serve docs locally on port 8000"
	@echo "  clean            Clean build artifacts"
	@echo "  clean-all        Clean all artifacts including PDM env"
	@echo "  dev-setup        Setup development environment"
	@echo "  check            Run all checks (format, lint, type, test)"
	@echo "  quick-test       Run quick test subset"
	@echo "  fix              Auto-fix code issues"
	@echo "  ci-test          Run CI-like test suite with coverage"
	@echo "  ci-check         Run all CI checks"
	@echo "  security-check   Run security analysis"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  help             Show this help message"