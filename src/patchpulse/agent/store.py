"""app.report, the agent's stored reports, and which reports are due (design D6 and D10, spec
§10.8).

One report per patch bundle and horizon: the 7-day one once its 7 days have ended, the 14-day one
when Phase 4 allows and completes it. A horizon counts once Phase 4 has stored its estimate rows
(the effects stage writes them only for allowed, complete horizons), so the report never runs
ahead of its estimate.

A report is redone only when its evidence changes, never to get new wording: its fingerprint
covers the core tool results for its window (not the game overview, whose totals move every
night), its estimate rows at its horizon, the settings the gate and the rules read, and the prompt
version. A basic report is due again the next night. Rows are append-only; readers take each
(bundle, horizon)'s latest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import Connection, func, insert, select

from patchpulse.agent.bundle import Bundle
from patchpulse.agent.evidence import HORIZONS, bundle_scope, core_evidence
from patchpulse.agent.gate import PLACEBO_P
from patchpulse.agent.prompts import PROMPT_VERSION
from patchpulse.agent.report import Report, public_copy
from patchpulse.agent.rules import (
    H1_NULL_BAND_PTS,
    H2_NEW_PLAYERS_PTS,
    H4_COMPOSITION_SHARE,
    H4_SMALL_PTS,
)
from patchpulse.agent.tools import Tools, Trace
from patchpulse.causal.estimate import fingerprint as evidence_fingerprint
from patchpulse.db import tables

_NOT_FINGERPRINTED = frozenset({"get_game_overview"})  # whole-game totals: they move every night
_THRESHOLDS = (H1_NULL_BAND_PTS, H2_NEW_PLAYERS_PTS, H4_COMPOSITION_SHARE, H4_SMALL_PTS, PLACEBO_P)

Key = tuple[str, int]


@dataclass(frozen=True)
class ReportInputs:
    """What a bundle report's content depends on, apart from the models' wording."""

    bundle_id: str
    horizon: int
    calls: tuple[tuple[str, str, str], ...]  # (tool, args, result) as canonical JSON
    estimates: tuple[str, ...]  # the served estimate rows' fingerprints at the horizon
    gate: tuple[bool, bool]  # the causal config's validated and spike_causal_ok
    thresholds: tuple[float, ...]
    prompt_version: str


@dataclass(frozen=True)
class StoredReport:
    """One row of app.report."""

    report_id: str
    bundle_id: str
    horizon: int
    run_id: int
    scope: str
    data_version: int
    fingerprint: str
    prompt_version: str
    model_ids: Mapping[str, str]
    basic: bool
    report: Report
    public: dict[str, Any]
    created_at: datetime


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def report_inputs(
    bundle: Bundle, bundle_id: str, horizon: int, *, prompt_version: str = PROMPT_VERSION
) -> ReportInputs:
    """The report's core tool calls run on `bundle`, as the graph runs them."""
    tools = Tools(bundle, Trace(run="fingerprint"))
    core_evidence(tools, bundle_scope(bundle, bundle_id, horizon))
    calls: list[tuple[str, str, str]] = []
    estimates: list[str] = []
    for call in tools.trace.calls:
        if call.tool in _NOT_FINGERPRINTED:
            continue
        result = call.result
        if call.tool == "estimate_patch_effect":
            # Only this report's horizon: a 14-day row arriving later leaves the 7-day one alone.
            if "horizons" in result:
                shown = {k: v for k, v in result["horizons"].items() if k == str(horizon)}
                result = {**result, "horizons": shown}
            estimates += [r.fingerprint for r in tools.estimate_rows(call) if r.horizon == horizon]
        calls.append((call.tool, _canonical(call.args), _canonical(result)))
    return ReportInputs(
        bundle_id=bundle_id,
        horizon=horizon,
        calls=tuple(calls),
        estimates=tuple(estimates),
        gate=(bundle.causal.validated, bundle.causal.spike_causal_ok),
        thresholds=_THRESHOLDS,
        prompt_version=prompt_version,
    )


def report_fingerprint(inputs: ReportInputs) -> str:
    """SHA-256 over the inputs (Phase 4's fingerprint, so EFFECTS_VERSION is in it too)."""
    return evidence_fingerprint(
        inputs.bundle_id,
        inputs.horizon,
        inputs.calls,
        inputs.estimates,
        inputs.gate,
        inputs.thresholds,
        inputs.prompt_version,
    )


def _candidates(bundle: Bundle) -> list[tuple[str, int, date]]:
    """(bundle id, horizon, the window's last day) for every horizon with estimate rows."""
    found: list[tuple[str, int, date]] = []
    for game in bundle.games:
        starts: dict[str, date] = {}
        for patch in game.patches:
            if patch.bundle_id is not None:
                known = starts.get(patch.bundle_id)
                starts[patch.bundle_id] = patch.day if known is None else min(known, patch.day)
        estimated = {(e.bundle_id, e.horizon) for e in game.estimates}
        for bundle_id, start in starts.items():
            for horizon in HORIZONS:
                if (bundle_id, horizon) in estimated:
                    found.append((bundle_id, horizon, start + timedelta(days=horizon - 1)))
    return found


def reports_due(
    bundle: Bundle, stored: Mapping[Key, StoredReport], *, prompt_version: str = PROMPT_VERSION
) -> list[Key]:
    """The complete horizons with no report, a basic one, or one whose fingerprint changed; the
    latest windows first (a bundle's 14-day report before its 7-day one)."""
    ordered = sorted(_candidates(bundle), key=lambda c: (-c[2].toordinal(), c[0], -c[1]))
    due: list[Key] = []
    for bundle_id, horizon, _ in ordered:
        old = stored.get((bundle_id, horizon))
        if old is None or old.basic:
            due.append((bundle_id, horizon))
            continue
        inputs = report_inputs(bundle, bundle_id, horizon, prompt_version=prompt_version)
        if old.fingerprint != report_fingerprint(inputs):
            due.append((bundle_id, horizon))
    return due


def shown_reports(stored: Mapping[Key, StoredReport]) -> dict[str, StoredReport]:
    """The report the site and the API show for each bundle: the 14-day one once it exists, the
    7-day one until then (both stay stored)."""
    shown: dict[str, StoredReport] = {}
    for (bundle_id, horizon), row in sorted(stored.items()):
        if bundle_id not in shown or horizon > shown[bundle_id].horizon:
            shown[bundle_id] = row
    return shown


def _latest(*where: Any) -> Any:
    table = tables.report
    rank = (
        func.row_number()
        .over(partition_by=[table.c.bundle_id, table.c.horizon], order_by=table.c.run_id.desc())
        .label("rank")
    )
    ranked = select(table, rank).where(*where).subquery()
    return select(ranked).where(ranked.c.rank == 1)


def _stored(row: Any) -> StoredReport:
    return StoredReport(
        report_id=row.report_id,
        bundle_id=row.bundle_id,
        horizon=row.horizon,
        run_id=row.run_id,
        scope=row.scope,
        data_version=row.data_version,
        fingerprint=row.fingerprint,
        prompt_version=row.prompt_version,
        model_ids=json.loads(row.model_ids),
        basic=bool(row.basic),
        report=Report.model_validate_json(row.report_json),
        public=json.loads(row.public_json),
        created_at=row.created_at.replace(tzinfo=UTC),
    )


def latest_reports(connection: Connection) -> dict[Key, StoredReport]:
    """Each (bundle, horizon)'s latest report."""
    return {(r.bundle_id, r.horizon): _stored(r) for r in connection.execute(_latest())}


def write_report(
    connection: Connection,
    report: Report,
    *,
    fingerprint: str,
    run_id: int,
    created_at: datetime | None = None,
) -> bool:
    """Store a bundle report unless its key's latest row has the same fingerprint and isn't
    basic, or both are basic (a basic report never replaces a model-written one on the same
    evidence). Returns whether a row was written. A visitor's question is never stored here."""
    if report.scope != "bundle" or report.bundle_id is None or report.horizon is None:
        raise ValueError("only bundle reports are stored; a question report lives in the API")
    table = tables.report
    latest = connection.execute(
        _latest(table.c.bundle_id == report.bundle_id, table.c.horizon == report.horizon)
    ).first()
    if (
        latest is not None
        and latest.fingerprint == fingerprint
        and (report.basic or not latest.basic)
    ):
        return False
    connection.execute(
        insert(table),
        {
            "bundle_id": report.bundle_id,
            "horizon": report.horizon,
            "run_id": run_id,
            "report_id": report.report_id,
            "scope": report.scope,
            "data_version": report.data_version,
            "fingerprint": fingerprint,
            "prompt_version": report.prompt_version,
            "model_ids": json.dumps(report.model_ids, sort_keys=True),
            "basic": report.basic,
            "report_json": report.model_dump_json(),
            "public_json": json.dumps(public_copy(report), ensure_ascii=False),
            "created_at": (created_at or datetime.now(UTC)).astimezone(UTC).replace(tzinfo=None),
        },
    )
    return True
