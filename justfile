set windows-shell := ["powershell.exe", "-c"]

pyre:
    uv run pyrefly check

lint:
    uv run ruff check --fix

test:
    uv run pytest tests
