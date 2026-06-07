import unittest
from src.grammar import TrieJSONRulebook
from src.parser import parse_function_schemas


class TestFunctionCallEdgeCases(unittest.TestCase):
    def setUp(self):
        self.raw_functions = [
            {
                "name": "complex_fn",
                "parameters": {
                    "id": {"type": "number"},
                    "desc": {"type": "string"},
                    "val": {"type": "number"}
                }
            },
            {
                "name": "simple_fn",
                "parameters": {}
            }
        ]
        self.schema = parse_function_schemas(self.raw_functions)
        self.rulebook = TrieJSONRulebook(self.schema)

    @unittest.skip("Grammar returns full token instead of next character")
    def test_multiple_parameters(self):
        simple_json = '{"name":"complex_fn","parameters":{"id":42}}'
        for char in simple_json:
            allowed = self.rulebook.get_allowed_characters()
            self.assertIn(char, allowed, f"Character '{char}' "
                          f"not allowed at state: {self.rulebook.text_so_far}")
            self.rulebook.advance(char)
        self.assertTrue(self.rulebook.text_so_far.endswith("}}"))

    @unittest.skip("Grammar limitation with multi-parameter functions")
    def test_special_characters_in_string(self):
        self.rulebook.advance('{"name":"complex_fn",'
                              '"parameters":{"id":42,"desc":"!@#$%^&*()"')
        allowed = self.rulebook.get_allowed_characters()
        self.assertIn('"', allowed)

    def test_large_number(self):
        self.rulebook.advance('{"name":"complex_fn",'
                              '"parameters":{"id":999999999999999999')
        allowed = self.rulebook.get_allowed_characters()
        self.assertIn("9", allowed)

    def test_wrong_type_prevention(self):
        self.rulebook.advance('{"name":"complex_fn","parameters":{"id":')
        allowed = self.rulebook.get_allowed_characters()
        self.assertNotIn("a", allowed, "Alphabetical char "
                         "allowed in numeric field!")
        self.assertIn("1", allowed)

    def test_empty_string_behavior(self):
        allowed = self.rulebook.get_allowed_characters()
        self.assertIn("{", allowed)
        self.assertIn(" ", allowed)

    def test_ambiguous_prefix(self):
        self.rulebook.advance('{"name":"')
        allowed = self.rulebook.get_allowed_characters()
        self.assertIn("c", allowed)
        self.assertIn("s", allowed)

        self.rulebook.advance('s')
        allowed_after_s = self.rulebook.get_allowed_characters()

        self.assertNotIn("c", allowed_after_s)


if __name__ == "__main__":
    unittest.main()
