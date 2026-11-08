"""Seif's blind audit of the judge (Phase 6 design D5, plan Task 7).

The calibration sample is drawn once from the main arm's items and exported with its text to
data/local/evals/judge_audit.jsonl, which never leaves the machine: claims and reviews are text.
The labeling app shows one item at a time, the question, the claim and its evidence, and never the
judge's verdict, the arm or whether the claim shipped. Each answer is appended and synced to
evals/judge/labels_seif.csv the moment it's saved: the item id, its kind, the label (1 supported, 0
not), unsure and the rubric's version, and no text, because the file is committed. A correction is
a new row; the latest counts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from patchpulse.evals.judge import CalibrationSample, ItemKind, JudgeLabel
from patchpulse.evals.records import EVALS_DATA, require_local
from patchpulse.labeling.audit import append_synced

AUDIT_PATH = EVALS_DATA / "judge_audit.jsonl"  # local: holds the text
LABELS_PATH = Path("evals") / "judge" / "labels_seif.csv"  # committed: ids and labels only
LABEL_COLUMNS = ("item_id", "kind", "label", "unsure", "rubric_version")
LABELS = (0, 1)


@dataclass(frozen=True)
class AuditView:
    """What the audit shows for an item, and nothing else: no verdict, arm or shipped flag."""

    item_id: str  # never shown: it names the arm
    kind: ItemKind
    claim_text: str
    evidence: str  # a review snippet, or the figure as JSON


def read_audit(path: Path = AUDIT_PATH) -> list[AuditView]:
    views = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            views.append(AuditView(row["item_id"], row["kind"], row["claim_text"], row["evidence"]))
    return views


def export_audit(sample: CalibrationSample, path: Path = AUDIT_PATH) -> Path:
    """Write the sample, in its drawn order, with the text the app shows. Drawn once: the same
    draw again is a no-op, a different one is refused (labels may exist)."""
    require_local(path)
    ids = [item.item_id for item in sample.items]
    if path.is_file():
        if [view.item_id for view in read_audit(path)] == ids:
            return path
        raise ValueError(f"{path.name} is already drawn; delete it by hand to draw again")
    rows = [
        {"item_id": i.item_id, "kind": i.kind, "claim_text": i.claim_text, "evidence": i.evidence}
        for i in sample.items
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def read_labels(path: Path = LABELS_PATH) -> dict[str, JudgeLabel]:
    """Seif's latest label per item."""
    if not path.exists():
        return {}
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    latest = rows.drop_duplicates("item_id", keep="last")
    return {
        str(item_id): JudgeLabel(
            kind=kind,  # written by the session: citation or figure
            label=str(label) == "1",
            unsure=str(unsure).lower() == "true",
            rubric_version=str(version),
        )
        for item_id, kind, label, unsure, version in zip(
            latest["item_id"],
            latest["kind"],
            latest["label"],
            latest["unsure"],
            latest["rubric_version"],
            strict=True,
        )
    }


class JudgeAuditSession:
    def __init__(
        self, views: Sequence[AuditView], labels_path: Path = LABELS_PATH, *, rubric_version: str
    ) -> None:
        self.views = list(views)
        self.labels_path = labels_path
        self.rubric_version = rubric_version
        self._kinds = {view.item_id: view.kind for view in self.views}

    def labels(self) -> dict[str, JudgeLabel]:
        return read_labels(self.labels_path)

    def next_item(self) -> AuditView | None:
        done = self.labels()
        return next((view for view in self.views if view.item_id not in done), None)

    def progress(self) -> tuple[int, int]:
        done = self.labels()
        return sum(1 for view in self.views if view.item_id in done), len(self.views)

    def save(self, item_id: str, label: int, *, unsure: bool) -> None:
        if item_id not in self._kinds:
            raise ValueError(f"item {item_id} is not in this audit")
        if label not in LABELS:
            raise ValueError(f"label {label!r} isn't 0 or 1")
        row: list[object] = [item_id, self._kinds[item_id], label, unsure, self.rubric_version]
        append_synced(self.labels_path, list(LABEL_COLUMNS), row)
