"""Edit vertical: menu-navigated, span-scoped edits to existing code.

The sibling of the greenfield code vertical: instead of writing a module
from scratch, it locates one def in an existing corpus (free symbol-index
lookup when unique, one menu when ambiguous, tree navigation otherwise)
and asks the model for only the delta — never the whole file.
"""
from threetoks.code.edit.index import Symbol, SymbolIndex
from threetoks.code.edit.vertical import EditVertical

__all__ = ["EditVertical", "Symbol", "SymbolIndex"]
