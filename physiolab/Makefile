.PHONY: setup dev test schema check build
setup:
	uv sync --directory backend --frozen
	cd web && npm ci

dev:
	python3 scripts/dev.py

schema:
	uv run --directory backend python ../scripts/export-schema.py
	cd web && npm run types

test:
	uv run --directory backend pytest

check:
	uv run --directory backend ruff check src tests
	uv run --directory backend ruff format --check src tests
	$(MAKE) test
	cd web && npm run build

build:
	cd web && npm run build
