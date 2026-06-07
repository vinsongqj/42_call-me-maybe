.PHONY: install run debug lint lint-strict

install:
	uv sync
	uv pip install flake8 mypy pydantic numpy

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .mypy_cache .pytest_cache data/output

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict

test:
	python3 -m unittest discover -s tests -p "test_*.py"