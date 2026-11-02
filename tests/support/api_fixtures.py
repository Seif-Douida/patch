"""The API's test doubles (Phase 5): a monotonic clock, and the `agent-data` branch as the API
fetches it, published with the nightly job's own writer."""

from __future__ import annotations

from datetime import UTC, datetime

from patchpulse.agent.bundle import BUNDLE_FILE, REPORTS_FILE, Bundle, dump_bundle
from patchpulse.agent.report import Report
from patchpulse.agent.store import StoredReport
from patchpulse.pipeline.reports import reports_file


class Seconds:
    """A monotonic clock the test moves."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def stored(report: Report) -> StoredReport:
    assert report.bundle_id is not None
    assert report.horizon is not None
    return StoredReport(
        report_id=report.report_id,
        bundle_id=report.bundle_id,
        horizon=report.horizon,
        run_id=report.data_version,
        scope=report.scope,
        data_version=report.data_version,
        fingerprint="0" * 64,
        prompt_version=report.prompt_version,
        model_ids=report.model_ids,
        basic=report.basic,
        report=report,
        public={},
        created_at=datetime(2026, 10, 11, 3, tzinfo=UTC),
    )


class Published:
    """The `agent-data` branch as the API fetches it: files by name, or down."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.down = False
        self.fetched: list[str] = []

    def publish(self, bundle: Bundle, *reports: Report, reports_version: int | None = None) -> None:
        rows = {(r.bundle_id or "", r.horizon or 0): stored(r) for r in reports}
        version = bundle.data_version if reports_version is None else reports_version
        self.files = {
            BUNDLE_FILE: dump_bundle(bundle),
            REPORTS_FILE: reports_file(bundle, rows, data_version=version),
        }

    def __call__(self, name: str) -> bytes:
        self.fetched.append(name)
        if self.down:
            raise ConnectionError("raw.githubusercontent.com didn't answer")
        return self.files[name]
