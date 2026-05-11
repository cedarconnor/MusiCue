from datetime import datetime, timezone

import pytest

from musicue.health.models import (
    ComponentState,
    ComponentStatus,
    ReadinessReport,
)


def test_component_state_values():
    assert ComponentState.READY.value == "ready"
    assert ComponentState.DEGRADED.value == "degraded"
    assert ComponentState.MISSING.value == "missing"
    assert ComponentState.ERROR.value == "error"


def test_component_status_minimal():
    s = ComponentStatus(name="demucs", state=ComponentState.READY, required=True)
    assert s.name == "demucs"
    assert s.state == ComponentState.READY
    assert s.required is True
    assert s.version is None
    assert s.detail is None
    assert s.cache_path is None
    assert s.remediation is None


def test_component_status_full():
    s = ComponentStatus(
        name="allin1",
        state=ComponentState.MISSING,
        required=False,
        version=None,
        detail="package not importable",
        cache_path=None,
        remediation="uv pip install allin1",
    )
    assert s.detail == "package not importable"
    assert s.remediation == "uv pip install allin1"


def test_readiness_report_rejects_invalid_overall():
    with pytest.raises(ValueError):
        ReadinessReport(
            components=[],
            overall="purple",  # type: ignore[arg-type]
            checked_at=datetime.now(timezone.utc),
        )


def test_readiness_report_accepts_valid_overall():
    r = ReadinessReport(
        components=[],
        overall="green",
        checked_at=datetime.now(timezone.utc),
    )
    assert r.overall == "green"
