.PHONY: help install install-dev lint lint-fix format test test-scripts test-verbose clean pre-commit-install pre-commit-run all-check

# Default target
help:
	@echo "Available commands:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install all dependencies (including dev)"
	@echo "  make lint             - Run linter and auto-fix issues"
	@echo "  make format           - Format code with ruff"
	@echo "  make test             - Run inference tests"
	@echo "  make test-scripts     - Run repository script tests"
	@echo "  make test-verbose     - Run tests with verbose output"
	@echo "  make pre-commit-install - Install pre-commit hooks"
	@echo "  make pre-commit-run   - Run pre-commit on all files"
	@echo "  make all-check        - Run format, lint, and test"
	@echo "  make clean            - Clean up cache files and artifacts"

# Install production dependencies
install:
	uv sync

# Install all dependencies including dev
install-dev:
	uv sync --group dev

# Run linter and auto-fix issues
lint:
	cd src && uv run ruff check --fix . ../tests/scripts

# Format code
format:
	cd src && uv run ruff format . ../tests/scripts

# Run inference tests
test:
	cd src && uv run pytest

# Run repository script tests separately from the inference test suite
test-scripts:
	cd src && uv run pytest ../tests/scripts

# Run tests with verbose output
test-verbose:
	cd src && uv run pytest -v

# Install pre-commit hooks
pre-commit-install:
	uv run pre-commit install

# Run pre-commit on all files
pre-commit-run:
	uv run pre-commit run --all-files

# Run all checks (format, lint, test)
all-check: format lint test test-scripts
	@echo "All checks passed!"

# Clean up cache files and artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "Cleaned up cache files and artifacts"

clean-output:
	rm -rf data
	rm -rf phonemes

run:
	uv run python src/phoneme_modelling/generate_synthetic_audio.py

generate:
	uv run python src/phoneme_modelling/generate_synthetic_audio.py --mode generate

eval:
	uv run python src/phoneme_modelling/generate_synthetic_audio.py --mode eval
