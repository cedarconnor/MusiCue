"""Tests for the NATTEN legacy-compat patcher.

The patcher mutates an installed natten package. To keep the tests
hermetic we stand up a fake `natten/` directory and point patch_natten
at it via monkeypatching `import natten` to a stub module whose
__file__ lives in the fake directory.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _stand_up_fake_natten(tmp_path: Path) -> Path:
    """Create a fake natten package layout in tmp_path. Returns the
    package dir (the parent of __init__.py)."""
    pkg = tmp_path / "natten"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# fake natten\n", encoding="utf-8")
    (pkg / "functional.py").write_text(
        "# fake functional\n"
        "def some_existing_fn():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    return pkg


def _patch_imports(monkeypatch, pkg_dir: Path) -> None:
    """Make `import natten` resolve to a stub whose __file__ lives in pkg_dir."""
    fake = types.ModuleType("natten")
    fake.__file__ = str(pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "natten", fake)


def _patch_source(tmp_path: Path) -> Path:
    src = tmp_path / "shim.py"
    src.write_text("# legacy compat shim\nprint('shim loaded')\n", encoding="utf-8")
    return src


def test_patch_natten_writes_compat_and_appends_import(tmp_path, monkeypatch):
    from scripts.patch_natten import patch_natten

    pkg = _stand_up_fake_natten(tmp_path)
    _patch_imports(monkeypatch, pkg)
    src = _patch_source(tmp_path)

    changed = patch_natten(patch_source=src)
    assert changed is True
    assert (pkg / "_legacy_compat.py").exists()
    func_contents = (pkg / "functional.py").read_text(encoding="utf-8")
    assert "Legacy NATTEN" in func_contents
    assert "from natten._legacy_compat import" in func_contents


def test_patch_natten_idempotent(tmp_path, monkeypatch):
    from scripts.patch_natten import patch_natten

    pkg = _stand_up_fake_natten(tmp_path)
    _patch_imports(monkeypatch, pkg)
    src = _patch_source(tmp_path)

    assert patch_natten(patch_source=src) is True
    # Second call: nothing should change (no duplicate imports).
    assert patch_natten(patch_source=src) is False
    func_contents = (pkg / "functional.py").read_text(encoding="utf-8")
    # Only one occurrence of the marker.
    assert func_contents.count("Legacy NATTEN") == 1


def test_patch_natten_noop_when_natten_missing(tmp_path, monkeypatch, capsys):
    from scripts.patch_natten import patch_natten

    monkeypatch.setitem(sys.modules, "natten", None)
    changed = patch_natten(patch_source=_patch_source(tmp_path))
    assert changed is False
    assert "not installed" in capsys.readouterr().out
