from musicue.health.models import ReadinessReport
from musicue.ui.server import create_app


def test_create_app_primes_readiness_report(tmp_path):
    app = create_app(storage_root=tmp_path)
    report = app.state.readiness_report
    assert isinstance(report, ReadinessReport)
    assert len(report.components) == 8
