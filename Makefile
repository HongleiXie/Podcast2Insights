.PHONY: ci

ci:
	uv sync --group dev
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -q
