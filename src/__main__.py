import argparse
import json
import sys
import os
import numpy as np
from llm_sdk import Small_LLM_Model

from src.grammar import TrieJSONRulebook
from src.tokenizer import CustomTokenizer
from src.validator import FunctionCallResult
from src.parser import parse_function_schemas


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Me Maybe")
    parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str, default="data/output/function_calling_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not os.path.exists(args.functions_definition) or not os.path.exists(args.input):
        return

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.functions_definition, "r", encoding="utf-8") as f:
            functions_data = json.load(f)
        with open(args.input, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)
    except json.JSONDecodeError as e:
        print("Error: One of the input JSON configuration files is malformed!")
        print(f"Details: {e}")
        sys.exit(1)

    schema_metadata = parse_function_schemas(functions_data)
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    results = []

    for idx, item in enumerate(prompts_data):
        prompt_text = item.get("prompt", "")
        print(f"\n--- Processing prompt {idx + 1}/{len(prompts_data)}: {prompt_text} ---", flush=True)

        rulebook = TrieJSONRulebook(schema_metadata)
        tokenizer = CustomTokenizer(vocab_path)

        system_prompt = (
            f"Functions schemas:\n{json.dumps(functions_data)}\n\n"
            f"Query: {prompt_text}\n\n"
            f"Correct JSON output call:"
        )

        raw_encoded = model.encode(system_prompt)
        input_ids = (
            raw_encoded.cpu().numpy().flatten().tolist()
            if hasattr(raw_encoded, "cpu")
            else np.array(raw_encoded).flatten().tolist()
        )

        for token_step in range(150):
            normalized_text = "".join(rulebook.text_so_far.split())
            if normalized_text.endswith("}}"):
                break

            logits = model.get_logits_from_input_ids(input_ids)
            logits = np.array(logits).flatten()
            custom_mask = np.full_like(logits, -float("inf"))

            allowed_token_ids = tokenizer.get_allowed_token_ids(rulebook)
            if allowed_token_ids:
                for token_id in allowed_token_ids:
                    if token_id < len(custom_mask):
                        custom_mask[token_id] = 0.0
            else:
                custom_mask.fill(0.0)

            total_vocab_size = len(tokenizer.vocab)
            masked_count = total_vocab_size - len(allowed_token_ids)

            forced_logits = logits + custom_mask
            chosen_token_id = int(np.argmax(forced_logits))

            chosen_char_str = tokenizer.decode_single_token(chosen_token_id)

            if chosen_char_str:
                rulebook.advance(chosen_char_str)
            input_ids.append(chosen_token_id)

            GREEN = "\033[92m"
            YELLOW = "\033[93m"
            DIM = "\033[2m"
            RESET = "\033[0m"
            print(
                f"   [Step {token_step:03d}] "
                f"Allowed: {GREEN}{len(allowed_token_ids):<5}{RESET} | "
                f"Masked: {DIM}{masked_count:<5}{RESET} | "
                f"State: {YELLOW}{repr(rulebook.text_so_far)}{RESET}"
            )
            
            normalized_text = "".join(rulebook.text_so_far.split())
            if normalized_text.endswith("}}"):
                print(f"   [Complete ] Final Structural Closure Confirmed: {repr(rulebook.text_so_far)}")
                tokenizer.print_telemetry()
                break

        try:
            clean_json_str = rulebook.text_so_far.strip()

            start_idx = clean_json_str.find("{")
            end_idx = clean_json_str.rfind("}")
            if start_idx != -1 and end_idx != -1:
                clean_json_str = clean_json_str[start_idx:end_idx + 1]

            parsed = json.loads(clean_json_str)
            raw_params = parsed.get("parameters", {})
            normalized_params = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in raw_params.items()
            }

            validated = FunctionCallResult(
                prompt=prompt_text,
                name=parsed.get("name", ""),
                parameters=normalized_params
            )
            results.append(validated.model_dump())
            print(f"✅ Success: Generated {validated.name}")

        except Exception as e:
            print(f"❌ Failed to parse for query: {prompt_text}")
            print(f"Raw text: {repr(rulebook.text_so_far)}")
            print(f"Error details: {e}\n")
            continue

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()