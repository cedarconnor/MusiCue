from __future__ import annotations

from fastapi import APIRouter, Request

from musicue.health.models import ReadinessReport
from musicue.health.readiness import collect_report

router = APIRouter()


@router.get("/api/health/readiness", response_model=ReadinessReport)
def get_readiness(request: Request) -> ReadinessReport:
    report = getattr(request.app.state, "readiness_report", None)
    if report is None:
        report = collect_report()
        request.app.state.readiness_report = report
    return report


@router.post("/api/health/readiness/refresh", response_model=ReadinessReport)
def refresh_readiness(request: Request) -> ReadinessReport:
    report = collect_report()
    request.app.state.readiness_report = report
    return report
