.PHONY: install run debug lint lint-strict
GREEN := \033[0;32m
YELLOW := \033[1;33m
RESET := \033[0m

install:
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	@uv sync
	@uv pip install flake8 mypy pydantic numpy

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	@echo "$(YELLOW)Cleaning build files...$(RESET)"
	@rm -rf __pycache__ src/__pycache__ tests/__pycache__ llm_sdk/llm_sdk/llm_sdk/__pycache__ tests/.mypy_cache .mypy_cache .pytest_cache data/output

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict

test:
	@echo "$(GREEN)Running comprehensive unit testing...$(RESET)"
	python3 -m unittest discover -s tests -p "test_*.py"