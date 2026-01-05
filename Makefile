.PHONY: setup
setup:
	./devenv.sh

.PHONY: type-check
type-check:
	uv run python scripts/type-baseline/check-type-baseline.py

.PHONY: lint
lint:
	uv run ruff format && uv run ruff check --fix;

# ONLY use this in CI. Locally, make it always do the fix.
.PHONY: lint-check
lint-check:
	uv run ruff format --check && uv run ruff check;

test: type-check lint
	uv run python -m unittest;

test-all: type-check lint
	source .env.test; RUN_SLOW_TESTS=true uv run python -m unittest;

test-ci: type-check lint
	uv run python -m unittest;

# Example usage: make run-example
# This will run examples/example.py
run-%:
	source .env; uv run python examples/$*.py

# Start interactive REPL with Freeplay client initialized
# By default connects to production (app.freeplay.ai)
# Use 'make repl-local' for local development with SSL bypass
.PHONY: repl
repl:
	set -a; source .env 2>/dev/null || true; set +a; uv run python -i scripts/repl_setup.py

# Start REPL in local mode (connects to localhost, disables SSL verification)
.PHONY: repl-local
repl-local:
	set -a; source .env 2>/dev/null || true; set +a; uv run python -i scripts/repl_setup.py --local

# Alias for repl
.PHONY: console
console: repl
