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

A greedy constrained decoding loop is used with a grammar-based token mask. The pipeline follows these steps:

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
| JSON validity | 100% | Guaranteed by grammar mask since every output is parseable |
| Function selection accuracy | ≥ 90% | Determined by the LLM, constrained decoding does not interfere with name selection |
| Processing time | ~ 30 seconds | Tested using ``time make run`` for the default `data/input/function_calling_tests.json` provided with the project |
| Cache hit rate | ~80 % | Tested by recalling the same prompt 5 consecutive times |

---

## Challenges faced

- Understanding the jargon necessary to tackle this project (weights, tokens, logits, masks, etc.)
- Installing the LLM dependencies (too large to install on campus provided iMacs).
- Mapping tokens to characters because the model's vocabulary uses byte-pair encoding, so certain special characters had to be normalized before use.
- Wrangling the output to fit the targeted schemas.

## Testing strategy

- **End-to-End Validation** - The pipeline is validated against the provided functions_definition.json and function_calling_tests.json, ensuring that for every natural language prompt, the output JSON structure conforms to the expected function schema.

- **Schema Integrity** - By enforcing constraints via the TrieJSONRulebook at the character level, the LLM is physically unable to generate malformed JSON, eliminating the need for post-generation structural validation tests.

- **Performance Profiling** - Telemetry is used to monitor the engine's behavior under different schemas, ensuring that latency and cache-hit rates remain within expected bounds during high-frequency token generation.

---

## Bonuses 

- **Visualization of generation process** - A visualizer was made to show the generation process of each JSON output along with the cache hit rate and a completion indicator.
- **Performance optimizations (caching)** - `_mask_cache` in `CustomTokenizer` memoizes unique tokens to prevent redundant walk operations.
- **Support for complex nested function arguments** - `TrieJSONRulebook` in `grammar.py` explicitly handles recursive/nested schema structures. The grammar state machine tracks index pointers (ptr) through parameters and handles closing braces/brackets for nested objects, ensuring that complex, multi-parameter function calls are correctly enforced.
- **Demonstration of how encoding and decoding integrate with constrained decoding** - the model's `raw_encoded` input IDs are fed into the generation loop, and at each step, `forced_logits` is calculated by applying a custom_mask derived from the grammar state.

## Resources

- [HackerRank - Data Structures: Tries](https://youtu.be/zIjfhVPRZCg?si=SXCiSeMgBoo13OBM)

- [A random Instagram reel](https://www.instagram.com/reels/DWyPEO3DwNe/)

### Disclosure of AI use

To be honest, I get overwhelmed by documentation, and there were very few reliable resources I could find on YouTube to learn about the subjects required for this project, so I had to rely mostly on AI as a teacher to guide me through this project.

Claude and Gemini were used to solidify my understanding about the topic, assisting in writing some sections of the readme, edge case detection and catching any issues I might have missed.
