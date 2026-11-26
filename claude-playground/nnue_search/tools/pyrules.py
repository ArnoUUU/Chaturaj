"""Import the repo's pure-Python rules (chaturaji/) on Python 3.9 without editing them.

chaturaji/state.py has one runtime `str | None` (Python 3.10+). This loader reads the four rule
modules, swaps that line for a 3.9-compatible alias, and registers them as the `chaturaji`
package in memory. Nothing on disk changes.
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "chaturaji"


def load():
    if "chaturaji.game" in sys.modules:
        return
    pkg = types.ModuleType("chaturaji")
    pkg.__path__ = [str(PKG)]
    sys.modules["chaturaji"] = pkg
    for name in ("constants", "state", "moves", "game"):
        src = (PKG / f"{name}.py").read_text(encoding="utf-8")
        src = src.replace("Board = list[list[str | None]]", "Board = list")
        mod = types.ModuleType(f"chaturaji.{name}")
        mod.__file__ = str(PKG / f"{name}.py")
        mod.__package__ = "chaturaji"
        sys.modules[f"chaturaji.{name}"] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        setattr(pkg, name, mod)
