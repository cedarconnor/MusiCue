from __future__ import annotations

from pathlib import Path


def write_if_missing(example: Path, target: Path) -> bool:
    if not example.exists():
        raise FileNotFoundError(f".env.example not found at {example}")
    if target.exists():
        return False
    target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    example = repo / ".env.example"
    target = repo / ".env"
    wrote = write_if_missing(example, target)
    if wrote:
        print(f"Wrote {target}")
    else:
        print(f"{target} already exists; left it alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
