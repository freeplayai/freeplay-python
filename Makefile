.PHONY: type-checks
type-checks:
	poetry run mypy src;
	poetry run mypy tests;

test: type-checks
	[[ -f .env ]] && source .env; FREEPLAY_LOG_LEVEL=CRITICAL poetry run python -m unittest;
