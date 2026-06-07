import json
from typing import List, Dict, Any


class CustomTokenizer:
    def __init__(self, vocab_json_path: str) -> None:
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        self.vocab: Dict[int, str] = {}
        for token_str, token_id in raw_vocab.items():
            self.vocab[int(token_id)] = token_str

        self._mask_cache: Dict[str, List[int]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def decode_single_token(self, token_id: int) -> str:
        raw_piece = self.vocab.get(token_id, "")
        return raw_piece.replace("Ġ", " ").replace("Ċ", "\n")

    def get_allowed_token_ids(self, rulebook: Any) -> List[int]:
        cache_key = rulebook.text_so_far

        if cache_key in self._mask_cache:
            self.cache_hits += 1
            return self._mask_cache[cache_key]

        self.cache_misses += 1

        allowed_ids: List[int] = []
        allowed_chars = rulebook.get_allowed_characters()

        if not allowed_chars:
            self._mask_cache[cache_key] = []
            return []

        for token_id, token_str in self.vocab.items():
            cleaned_str = token_str.replace("Ġ", " ").replace("Ċ", "\n")
            if rulebook.can_walk_token(cleaned_str):
                allowed_ids.append(token_id)

        self._mask_cache[cache_key] = allowed_ids
        return allowed_ids

    def print_telemetry(self) -> None:
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_rate = self.cache_hits / total * 100
        else:
            hit_rate = 0
        msg = (
            f"[Performance Profile] "
            f"Cache Hits: {self.cache_hits} | "
            f"Cache Misses: {self.cache_misses} | "
            f"Hit Rate: {hit_rate:.2f}%"
        )
        print(msg)


def get_allowed_token_ids(vocab_json_path: str, rulebook: Any) -> List[int]:
    tokenizer = CustomTokenizer(vocab_json_path)
    return tokenizer.get_allowed_token_ids(rulebook)
