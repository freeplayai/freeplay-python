.PHONY: type-checks
type-checks:
	poetry run mypy src tests;# examples;

test: type-checks
	poetry env use 3.8
	poetry run python -m unittest;

test-all: type-checks
	poetry env use 3.8
	source .env.test; RUN_SLOW_TESTS=true poetry run python -m unittest;

# Example usage: make run-example
# This will run examples/example.py
run-%:
	source .env; poetry run python examples/$*.py