*This project was created as part of the 42 curriculum by vgoh*

# Call Me Maybe - A Constrained Decoding Project

## Description

Call Me Maybe is a function calling system that translates natural language prompts into structured JSON function calls using a 0.6B parameter language model (Qwen/Qwen3-0.6B).

This project focuses on implementing constrained decoding to intercept the model's token selection at every generation step and masking the tokens that don't fit the target JSON schema, without relying on prompt engineering.

If the prompt is:
 
```
"What is the sum of 40 and 2?"
```
 
The system does not answer `42`. Instead it returns:
 
```json
{
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

The function is chosen by the model itself, constrained decoding only controls the structure of the output and not what output is chosen.
---

## Requirements

- Python 3.10+
- ``uv`` package manager
- The llm_sdk/ directory to be copied into the project root (provided separately)
---

## Instructions

A Makefile has been created for convenience. After cloning the repository, you may run the following commands:
```bash
make install       # Installs the dependencies required to run the program
make run           # Runs the program
make debug         # Debugs the program with python debugger
make lint          # Runs mypy and flake8 linting tests
make lint-strict   # Runs mypy with the --strict flag alongside flake8
make clean         # Removes all build files
```
If you would like to run the program manually, you may input the following commands in the terminal:
1. Install the dependencies using ``uv sync``
2. Run the program using ``uv run python -m src``, or with more explicit paths (all optional, paths shown are default):
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```
---
## Example Usage
 
Place your input files in `data/input/`:
 
**`data/input/functions_definition.json`**
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```
 
**`data/input/function_calling_tests.json`**
```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```
 
**`data/output/function_calling_results.json`** (generated)
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  }
]
```
 
---

## Resources
https://www.instagram.com/reels/DWyPEO3DwNe/
