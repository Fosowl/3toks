"""Core modules must import without the optional web extras installed.

Regression guard for the bare-install crash: ``python3 -m threetoks``
died with ``ModuleNotFoundError: requests`` because an eager import
chain (tui -> cli -> web.fetch) violated the extras-are-lazy rule. The
test simulates a machine without the extras via a meta-path blocker.
"""
import importlib
import sys
import unittest

EXTRAS = ("requests", "bs4", "markdownify", "selenium")
CORE_MODULES = ("threetoks.cli", "threetoks.tui", "threetoks.research",
                "threetoks.agents", "threetoks.memory",
                "threetoks.web.vertical", "threetoks.web.target")


class _BlockExtras:
    """Meta-path hook that pretends the extras are not installed."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in EXTRAS:
            raise ImportError(f"{name} blocked: simulating a bare install")
        return None


class LazyExtrasTest(unittest.TestCase):
    def test_core_imports_without_web_extras(self):
        saved = {name: sys.modules.pop(name) for name in list(sys.modules)
                 if name.split(".")[0] in EXTRAS
                 or name.split(".")[0] == "threetoks"}
        blocker = _BlockExtras()
        sys.meta_path.insert(0, blocker)
        try:
            for module in CORE_MODULES:
                importlib.import_module(module)
        finally:
            sys.meta_path.remove(blocker)
            for name in [n for n in sys.modules
                         if n.split(".")[0] == "threetoks"]:
                del sys.modules[name]
            sys.modules.update(saved)

    def test_web_layer_still_needs_the_extras(self):
        # sanity: the blocker really blocks (guards against a silent no-op)
        blocker = _BlockExtras()
        saved = sys.modules.pop("requests", None)
        sys.meta_path.insert(0, blocker)
        try:
            with self.assertRaises(ImportError):
                importlib.import_module("requests")
        finally:
            sys.meta_path.remove(blocker)
            if saved is not None:
                sys.modules["requests"] = saved


if __name__ == "__main__":
    unittest.main()
