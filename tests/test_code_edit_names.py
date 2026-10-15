"""Offline tests for name analysis in the edit vertical.

Module-level bindings, star imports, and missing-helper inference with
its parameter list. No model, no network.
"""
import unittest

from threetoks.code.edit import names


class MissingHelperTest(unittest.TestCase):
    def _helper(self, call: str, params: str = "nums"):
        src = f"def average({params}):\n    return {call}\n"
        return names.find_missing_helper(src, 1, {"average"})

    def test_positional_args_get_numbered_names(self):
        self.assertEqual(self._helper("_safe_add(nums, 1)"),
                         ("_safe_add", "arg1, arg2"))
        nine = ", ".join(["nums"] * 9)
        self.assertEqual(self._helper(f"_mix({nine})"),
                         ("_mix", ", ".join(f"arg{i}" for i in range(1, 10))))

    def test_keyword_and_starred_calls_get_matching_params(self):
        self.assertEqual(self._helper("_scale(nums, factor=2)"),
                         ("_scale", "arg1, factor"))
        self.assertEqual(self._helper("_merge(*nums)"), ("_merge", "*args"))
        self.assertEqual(self._helper("_merge(1, *nums, k=1, **{})"),
                         ("_merge", "arg1, *args, k, **kwargs"))

    def test_clean_function_has_no_missing_helper(self):
        self.assertIsNone(self._helper("len(nums)"))

    def test_bound_names_are_not_missing_helpers(self):
        self.assertIsNone(self._helper("key(nums)", "nums, *, key"))
        self.assertIsNone(self._helper("list(map(lambda cb: cb(1), nums))"))
        src = ("def average(nums):\n    try:\n        return nums()\n"
               "    except TypeError as retry:\n        return retry(nums)\n")
        self.assertIsNone(names.find_missing_helper(src, 1, set()))

    def test_star_import_file_has_no_missing_helper(self):
        src = ("from math import *\n\n\n"
               "def hyp(a, b):\n    return sqrt(a * a + b * b)\n")
        self.assertTrue(names.has_star_import(src))
        self.assertIsNone(names.find_missing_helper(src, 4, {"hyp"}))
        named = src.replace("import *", "import pi")
        self.assertFalse(names.has_star_import(named))
        self.assertEqual(names.find_missing_helper(named, 4, {"hyp", "pi"}),
                         ("sqrt", "arg1"))

    def test_method_target_is_found_by_its_def_line(self):
        src = "class A:\n    def m(self, x):\n        return _norm(x, k=1)\n"
        self.assertEqual(names.find_missing_helper(src, 2, {"A"}),
                         ("_norm", "arg1, k"))
        self.assertIsNone(names.find_missing_helper(src, 3, {"A"}))

    def test_module_level_names_cover_blocks_and_tuple_targets(self):
        src = ("import os\nfrom m import *\na, (b, *c) = 1, (2, 3)\n"
               "if os:\n    import json as js\nelse:\n    js = None\n"
               "try:\n    from x import y\nexcept ImportError as err:\n"
               "    y = None\nfor i, j in []:\n    pass\n"
               "with open('f') as (fh, other):\n    pass\n"
               "def f():\n    local = 1\n")
        self.assertEqual(names.module_level_names(src),
                         {"os", "a", "b", "c", "js", "y", "err", "i", "j",
                          "fh", "other", "f"})


if __name__ == "__main__":
    unittest.main()
