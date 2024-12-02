.PHONY: type-checks
type-checks:
	poetry run mypy src tests;

test: type-checks
	poetry env use 3.8
	poetry run python -m unittest;

# Example usage: make run-example
# This will run examples/example.py
run-%:
	source .env; poetry run python examples/$*.py