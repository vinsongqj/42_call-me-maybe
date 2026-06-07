import os
import sys
import unittest
from typing import Dict

# Append project root to sys.path so we can import src modules natively
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constraints import TrieJSONRulebook
from src.vocab_utils import CustomTokenizer
from src.generator_utils import get_schema_metadata


class TestGuidedDecodingEngine(unittest.TestCase):
    def setUp(self) -> None:
        """
        Sets up an isolated functional validation schema using your exact 
        production data format shapes.
        """
        # Formatted EXACTLY like your real functions_definition.json file
        self.raw_functions_data = [
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers together and return their sum.",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "returns": {"type": "number"}
            }
        ]
        
        # Pass it through your real parsing utility to ensure absolute unpacking compliance
        self.schema_metadata = get_schema_metadata(self.raw_functions_data)
        self.rulebook = TrieJSONRulebook(self.schema_metadata)

        # 2. Mock vocabulary mapping for tokenizer verification
        self.vocab_data: Dict[str, int] = {
            "{": 0,
            '"name"': 1,
            '"fn_add_numbers"': 2,
            '"parameters"': 3,
            "1": 4,
            "2.5": 5,
            "invalid_text": 6
        }
        
        self.mock_vocab_path = "tests/mock_vocab.json"
        os.makedirs(os.path.dirname(self.mock_vocab_path), exist_ok=True)
        import json
        with open(self.mock_vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab_data, f)
            
        self.tokenizer = CustomTokenizer(self.mock_vocab_path)

    def tearDown(self) -> None:
        """Cleans up localized testing file artifacts."""
        if os.path.exists(self.mock_vocab_path):
            os.remove(self.mock_vocab_path)

    def test_initial_state_constraints(self) -> None:
        """Verifies that the state machine strictly expects valid structural entry points."""
        self.assertTrue(self.rulebook.can_walk_token("{"))
        self.assertFalse(self.rulebook.can_walk_token("invalid_text"))

    def test_schema_path_progression(self) -> None:
        """Confirms the rulebook accurately transitions states as tokens are appended step-by-step."""
        self.assertTrue(self.rulebook.can_walk_token("{"))
        self.rulebook.advance("{")
        
        self.assertTrue(self.rulebook.can_walk_token('"name"'))
        self.rulebook.advance('"name"')
        
        self.rulebook.advance(":")
        
        self.assertTrue(self.rulebook.can_walk_token('"fn_add_numbers"'))

    def test_numerical_type_enforcement(self) -> None:
        """Ensures that syntax paths block text mutations when expecting numbers."""
        setup_sequence = [
            "{", '"name"', ":", '"fn_add_numbers"', ",", '"parameters"', ":", "{", '"a"', ":"
        ]
        for token in setup_sequence:
            self.rulebook.advance(token)
            
        self.assertTrue(self.rulebook.can_walk_token("1"))
        self.assertTrue(self.rulebook.can_walk_token("2.5"))
        self.assertFalse(self.rulebook.can_walk_token('"hello"'))

    def test_tokenizer_performance_caching(self) -> None:
        """Programmatically asserts that duplicate lookups hit the memory cache layer."""
        self.tokenizer.cache_hits = 0
        self.tokenizer.cache_misses = 0

        first_run = self.tokenizer.get_allowed_token_ids(self.rulebook)
        self.assertEqual(self.tokenizer.cache_misses, 1)
        self.assertEqual(self.tokenizer.cache_hits, 0)

        second_run = self.tokenizer.get_allowed_token_ids(self.rulebook)
        self.assertEqual(self.tokenizer.cache_hits, 1)
        self.assertEqual(self.tokenizer.cache_misses, 1)

        self.assertEqual(first_run, second_run)


if __name__ == "__main__":
    unittest.main()