from __future__ import annotations

from typing import Callable


def _fetch_demucs() -> None:
    from demucs.pretrained import get_model

    get_model("htdemucs_ft")


def _fetch_clap() -> None:
    import laion_clap  # type: ignore[import-not-found]

    m = laion_clap.CLAP_Module(enable_fusion=False)
    m.load_ckpt()


def _fetchers() -> list[tuple[str, Callable[[], None]]]:
    return [
        ("demucs", _fetch_demucs),
        ("clap", _fetch_clap),
    ]


def run_all() -> list[tuple[str, bool, str | None]]:
    out: list[tuple[str, bool, str | None]] = []
    for name, fn in _fetchers():
        try:
            fn()
            out.append((name, True, None))
        except Exception as e:
            out.append((name, False, str(e)[:300]))
    return out


def main() -> int:
    print("Prefetching model weights — this can take a while on first run.")
    for name, ok, err in run_all():
        if ok:
            print(f"  [OK]   {name}")
        else:
            print(f"  [WARN] {name}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
