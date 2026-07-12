"""Offline tests for the MethodStore render/dedupe contract."""
import ast
import unittest

from threetoks.code.store import (STATUS_TESTED, STUB_BODY, MethodStore)


class RenderModuleTest(unittest.TestCase):
    def test_empty_store_renders_just_the_goal_docstring(self):
        store = MethodStore("do a thing")
        module = store.render_module()
        self.assertEqual(ast.get_docstring(ast.parse(module)), "do a thing")

    def test_bodied_method_renders_its_body_and_parses(self):
        store = MethodStore("double")
        record = store.add("double", "n", "n times two")
        record.body = "def double(n):\n    return n * 2"
        module = store.render_module()
        self.assertIn("return n * 2", module)
        ast.parse(module)

    def test_bodyless_method_renders_notimplemented_stub_with_contract(self):
        store = MethodStore("read")
        store.add("read_text", "path", "read a file into a string")
        module = store.render_module()
        self.assertIn(STUB_BODY.strip(), module)
        self.assertIn("read a file into a string", module)  # contract kept
        stub = next(n for n in ast.parse(module).body
                    if isinstance(n, ast.FunctionDef))
        self.assertEqual(ast.get_docstring(stub), "read a file into a string")

    def test_mixed_bodied_and_stub_still_parses(self):
        store = MethodStore("mixed")
        store.add("stub_me", "x", "not written yet")
        done = store.add("bodied", "y", "written")
        done.body = "def bodied(y):\n    return y"
        module = store.render_module()
        self.assertIn("raise NotImplementedError", module)  # stub_me
        self.assertIn("return y", module)                   # bodied
        ast.parse(module)

    def test_triple_quote_in_goal_and_contract_still_parses(self):
        # Render invariant must hold for hostile uncontrolled text.
        store = MethodStore('goal with """ inside')
        store.add("f", "x", 'contract with """ inside')
        ast.parse(store.render_module())


class DedupeTest(unittest.TestCase):
    def test_duplicate_name_returns_existing_record_and_keeps_length(self):
        store = MethodStore("g")
        first = store.add("f", "x", "first contract")
        again = store.add("f", "x2", "second contract ignored")
        self.assertIs(again, first)
        self.assertEqual(len(store.methods), 1)
        self.assertEqual(store.methods[0].contract, "first contract")

    def test_names_are_in_declaration_order(self):
        store = MethodStore("g")
        store.add("a", "", "")
        store.add("b", "", "")
        store.add("a", "", "")  # dup
        self.assertEqual(store.names(), ["a", "b"])


class BodiedExceptTest(unittest.TestCase):
    def _store_with_three(self):
        store = MethodStore("g")
        for name in ("a", "b", "c"):
            store.add(name, name, f"contract {name}")
        return store

    def test_returns_every_bodied_method_but_not_the_index(self):
        store = self._store_with_three()
        store.methods[0].body = "def a(a):\n    return 1"   # earlier, bodied
        store.methods[1].body = None                        # no body
        store.methods[2].body = "def c(c):\n    return 3"   # later, bodied
        # a repaired 'b' (index 1) may call earlier AND later siblings
        self.assertEqual([m.name for m in store.bodied_except(1)], ["a", "c"])

    def test_excludes_the_method_at_the_index_itself(self):
        store = self._store_with_three()
        store.methods[0].body = "def a(a):\n    return 1"
        store.methods[1].body = "def b(b):\n    return 2"
        # index 1 must not include itself even though it has a body
        self.assertEqual([m.name for m in store.bodied_except(1)], ["a"])

    def test_render_script_appends_a_main_guard_that_parses(self):
        store = MethodStore("g")
        store.add("go", "", "runs it").body = "def go():\n    return 1"
        script = store.render_script("go")
        self.assertTrue(script.endswith(
            'if __name__ == "__main__":\n    go()\n'), script)
        ast.parse(script)

    def test_render_script_without_entry_is_the_plain_module(self):
        store = MethodStore("g")
        store.add("go", "", "runs it")
        self.assertEqual(store.render_script(None), store.render_module())

    def test_signature_lines_formats_each_record_on_its_own_line(self):
        store = self._store_with_three()
        lines = store.signature_lines(store.methods)
        self.assertEqual(lines.count("\n"), 2)
        self.assertIn("a(a): contract a", lines)

    def test_signature_lines_none_marker_when_empty(self):
        store = MethodStore("g")
        self.assertEqual(store.signature_lines([]), "(none)")


class StatusRoundTripTest(unittest.TestCase):
    def test_tested_status_does_not_change_the_render(self):
        store = MethodStore("g")
        record = store.add("f", "x", "c")
        record.body = "def f(x):\n    return x"
        record.status = STATUS_TESTED
        ast.parse(store.render_module())
        self.assertIn("return x", store.render_module())


if __name__ == "__main__":
    unittest.main()
