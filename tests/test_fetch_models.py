import pytest

from scripts import fetch_models


def test_run_all_returns_status_per_model(monkeypatch):
    calls: list[str] = []

    def ok():
        calls.append("ok")

    def fail():
        calls.append("fail")
        raise RuntimeError("simulated download error")

    monkeypatch.setattr(
        fetch_models, "_fetchers", lambda: [("demucs", ok), ("clap", fail)]
    )
    results = fetch_models.run_all()
    assert results == [
        ("demucs", True, None),
        ("clap", False, "simulated download error"),
    ]
    assert calls == ["ok", "fail"]


def test_main_exits_zero_even_on_failure(monkeypatch):
    def fail():
        raise RuntimeError("nope")

    monkeypatch.setattr(fetch_models, "_fetchers", lambda: [("clap", fail)])
    assert fetch_models.main() == 0
