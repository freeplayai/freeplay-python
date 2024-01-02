.PHONY: type-checks
type-checks:
	poetry run mypy src;
	poetry run mypy tests;

test: type-checks
	poetry run python -m unittest;
