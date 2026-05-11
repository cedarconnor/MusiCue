from datetime import datetime

from musicue.health.models import ComponentState, ComponentStatus
from musicue.health.readiness import _rollup, collect_report


def _cs(name: str, state: ComponentState, required: bool) -> ComponentStatus:
    return ComponentStatus(name=name, state=state, required=required)


def test_rollup_green_when_all_ready():
    items = [
        _cs("a", ComponentState.READY, True),
        _cs("b", ComponentState.READY, False),
    ]
    assert _rollup(items) == "green"


def test_rollup_red_when_required_missing():
    items = [
        _cs("a", ComponentState.MISSING, True),
        _cs("b", ComponentState.READY, False),
    ]
    assert _rollup(items) == "red"


def test_rollup_red_when_required_errored():
    items = [
        _cs("a", ComponentState.ERROR, True),
        _cs("b", ComponentState.READY, False),
    ]
    assert _rollup(items) == "red"


def test_rollup_amber_when_optional_missing():
    items = [
        _cs("a", ComponentState.READY, True),
        _cs("b", ComponentState.MISSING, False),
    ]
    assert _rollup(items) == "amber"


def test_rollup_amber_when_required_degraded():
    items = [
        _cs("a", ComponentState.DEGRADED, True),
        _cs("b", ComponentState.READY, False),
    ]
    assert _rollup(items) == "amber"


def test_collect_report_returns_real_data():
    report = collect_report()
    assert isinstance(report.checked_at, datetime)
    names = {c.name for c in report.components}
    assert names == {
        "python_venv",
        "torch",
        "cuda",
        "ffmpeg",
        "demucs",
        "basic_pitch",
        "allin1",
        "clap",
    }
    assert report.overall in {"green", "amber", "red"}
