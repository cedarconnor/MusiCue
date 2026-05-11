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


def _format_table(report: ReadinessReport) -> str:
    icon_for = {
        ComponentState.READY: "OK ",
        ComponentState.DEGRADED: "WRN",
        ComponentState.MISSING: "MIS",
        ComponentState.ERROR: "ERR",
    }
    rows = ["", f"MusiCue readiness — overall: {report.overall.upper()}", ""]
    rows.append(f"  {'STATE':<5} {'NAME':<14} {'VERSION':<10} DETAIL")
    rows.append("  " + "-" * 70)
    for c in report.components:
        version = c.version or "-"
        detail = c.detail or ""
        rows.append(
            f"  {icon_for[c.state]:<5} {c.name:<14} {version:<10} {detail}"
        )
    rows.append("")
    return "\n".join(rows)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="musicue.health.readiness")
    parser.add_argument("--print-table", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = collect_report()
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(_format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
