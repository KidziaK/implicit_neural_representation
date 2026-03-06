pyre:
    uv run pyrefly check

lint:
    uv run ruff check --fix

test:
    uv run pytest tests
