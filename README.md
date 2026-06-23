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

## Algorithm explanation

A greedy constrained decoding loop is used with a grammar-based token mask. Here is the pipeline for each prompt:

1. **Schema parsing** -  `parse_function_schemas()` converts `functions_definition.json` into a compact `{function_name: [(param, type), ...]}` lookup used by the grammar engine.

2. **Prompt construction** - The function schemas and natural language prompt are combined into a single prompt, then tokenized into input IDs via `Small_LLM_Model.encode()`.

3. **Token-by-token generation loop**
    - `model.get_logits_from_input_ids(input_ids)` runs a forward pass and returns a raw score for every token in the vocabulary.
    - `TrieJSONRulebook.get_allowed_characters()` checks the text generated so far and ensures that only the characters that conform to the JSON schema are returned.
    - `CustomTokenizer.get_allowed_token_ids(rulebook)` walks the entire vocabulary and calls `rulebook_can_walk_token()` for each entry. Any token that violates the grammar is discarded.
    - A logit mask is created to mask all unallowed tokens to `-∞` while allowed tokens keep their original score.
    - `argmax` over the masked logits picks the highest scoring token.
    - The chosen token's character string is appended to the grammar state via `rulebook.advance()` and the token ID is appended to `input_ids` for the next step.
    - Generation stops early when the accumulated text ends with `}}`.

4. **Post processing** - The raw text is trimmed, parsed as JSON and validated through the `FunctionCallResult` Pydantic model before being written to the output JSON file.

---

## Design decisions

- **`TrieJSONRulebook` as a character-level state machine** — Rather than running a full JSON parser on every candidate token, the grammar tracks a single string `text_so_far` and exposes `get_allowed_characters()`, which returns the exact set of characters permitted at the current position. This makes validation O(text length) rather than O(vocab size × parse cost).
 
- **Vocabulary cache in `CustomTokenizer`** - Scanning all vocabulary entries per generation step is intensive, so results are memoised by `text_so_far` string, ensuring each unique grammar state is only evaluated once.

- **Greedy decoding (`argmax`) over sampling** - Since the grammar already constrains the output to a small allowed set, stochastic sampling would only add noise. Greedy selection picks the most likely valid token and converges faster.

- **Pydantic for output validation** - `FunctionCallResult` enforces the schema structure of the output JSON, making sure any irregularities are flagged as exceptions instead of silently being written to the output file.

- **Forbidden dependencies** - `dspy`, `transformers`, `outlines`, PyTorch and Huggingface libraries are all excluded, Only `numpy`, `json` and `pydantic` are used, keeping the dependency requirements mininal and constrained decoding logic fully transparent.

---

## Performance analysis

| Metric | Target | Notes |
|---|---|---|
| JSON validity | 100% | Guaranteed by grammar mask; every output is parseable |
| Function selection accuracy | ≥ 90% | Determined by the LLM; constrained decoding does not interfere with name selection |
| Processing time | ~ 30 seconds | Tested using ``time make run`` for the default `data/input/function_calling_tests.json` provided with the project |
| Cache hit rate | ~80 % | Tested by recalling the same prompt 5 consecutive times |

---

## Challenges faced

## Testing strategy

## Resources
https://www.instagram.com/reels/DWyPEO3DwNe/
