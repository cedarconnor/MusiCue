"""Apply the NATTEN legacy-compat shim to the active environment.

NATTEN >= 0.17 dropped the procedural functions allin1 still imports
(natten1dqkrpb, natten1dav, natten2dqkrpb, natten2dav). This script
copies a pure-PyTorch reimplementation into the installed natten
package and appends an import line to natten/functional.py so allin1's
``from natten.functional import natten1dav, ...`` keeps working.

Idempotent: re-running this script is a no-op once the patch is applied.

Run from the repo root after installing allin1:
    python scripts/patch_natten.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_MARKER = "# --- Legacy NATTEN <0.17 compatibility shim (added by MusiCue install) ---"
_IMPORT_BLOCK = f"""

{_MARKER}
from natten._legacy_compat import (
    natten1dqkrpb, natten1dav, natten2dqkrpb, natten2dav,
)
"""


def patch_natten(patch_source: Path | None = None) -> bool:
    """Apply the patch to the active env's natten. Returns True if any
    change was made, False if the patch was already in place."""
    try:
        import natten  # type: ignore[import-not-found]
    except ImportError:
        print("natten is not installed in this environment; nothing to patch.")
        return False

    natten_dir = Path(natten.__file__).resolve().parent
    func_py = natten_dir / "functional.py"
    if not func_py.exists():
        raise RuntimeError(f"unexpected natten layout: {func_py} not found")

    if patch_source is None:
        patch_source = Path(__file__).resolve().parent / "patches" / "natten_legacy_compat.py"
    if not patch_source.exists():
        raise RuntimeError(f"patch source missing: {patch_source}")

    changed = False

    target_compat = natten_dir / "_legacy_compat.py"
    if not target_compat.exists() or target_compat.read_bytes() != patch_source.read_bytes():
        shutil.copyfile(patch_source, target_compat)
        print(f"wrote {target_compat}")
        changed = True

    contents = func_py.read_text(encoding="utf-8")
    if _MARKER not in contents:
        # Strip any trailing whitespace before appending so the file stays clean.
        new_contents = contents.rstrip() + _IMPORT_BLOCK
        func_py.write_text(new_contents, encoding="utf-8")
        print(f"appended legacy-compat import to {func_py}")
        changed = True

    if not changed:
        print("natten patch already applied; nothing to do.")
    return changed


if __name__ == "__main__":
    sys.exit(0 if patch_natten() is not None else 1)
