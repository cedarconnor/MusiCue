from __future__ import annotations

from datetime import datetime, timezone

from musicue.health import probes
from musicue.health.models import ComponentState, ComponentStatus, ReadinessReport


def _rollup(items: list[ComponentStatus]) -> str:
    if any(
        c.required and c.state in (ComponentState.MISSING, ComponentState.ERROR)
        for c in items
    ):
        return "red"
    if any(
        c.state in (
            ComponentState.MISSING,
            ComponentState.ERROR,
            ComponentState.DEGRADED,
        )
        for c in items
    ):
        return "amber"
    return "green"


def collect_report() -> ReadinessReport:
    components = [
        probes.probe_python_venv(),
        probes.probe_torch(),
        probes.probe_cuda(),
        probes.probe_ffmpeg(),
        probes.probe_demucs(),
        probes.probe_basic_pitch(),
        probes.probe_allin1(),
        probes.probe_clap(),
    ]
    return ReadinessReport(
        components=components,
        overall=_rollup(components),
        checked_at=datetime.now(timezone.utc),
    )
