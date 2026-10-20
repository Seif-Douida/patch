"""A review as a Qdrant point (plan Task 7, design D2): its id, both vectors, and exactly the
payload the design lists."""

from __future__ import annotations

import numpy as np
import pandas as pd
from qdrant_client import models

from patchpulse.retrieval.points import STORED_FIELDS, fields, to_point
from patchpulse.retrieval.qdrant import BM25_MODEL, DENSE, PAYLOAD_INDEXES, SPARSE, bm25_options

ROW = {
    "review_id": 2**63 + 5,  # Steam ids are unsigned 64-bit
    "appid": 553850,
    "day": 20261012,
    "created_ts": 1_791_800_000,
    "language": "french",
    "voted_up": np.bool_(False),
    "playtime_bucket": None,
    "first_seen_run_id": np.int64(17),
    "version": pd.Timestamp("2026-10-12 08:30:00"),
    "aspects": "performance|stability_bugs",
    "aspect_model_version": float("nan"),  # pandas' null
    "fingerprint": "ab" * 32,
    "rules_version": "r1",
}


def test_point_payload_has_every_design_field_and_no_more() -> None:
    point = to_point(ROW, "Le jeu plante", np.full(4, 0.5, dtype=np.float32))

    assert point.id == 2**63 + 5
    assert set(point.payload or {}) == set(PAYLOAD_INDEXES) | set(STORED_FIELDS)
    assert STORED_FIELDS == ("text", "version", "fingerprint", "rules_version")
    payload = point.payload or {}
    assert payload["aspects"] == ["performance", "stability_bugs"]
    assert payload["aspect_model_version"] is None
    assert payload["playtime_bucket"] is None
    assert payload["version"] == "2026-10-12T08:30:00"
    assert payload["text"] == "Le jeu plante"
    for field in ("appid", "day", "created_ts", "first_seen_run_id"):
        assert type(payload[field]) is int
    assert payload["voted_up"] is False
    assert isinstance(point.vector, dict)
    assert point.vector[DENSE] == [0.5] * 4
    assert point.vector[SPARSE] == models.Document(
        text="Le jeu plante", model=BM25_MODEL, options=bm25_options("french")
    )


def test_a_review_without_aspects_has_an_empty_list() -> None:
    point = to_point({**ROW, "aspects": ""}, "x", np.zeros(4, dtype=np.float32))

    assert (point.payload or {})["aspects"] == []


def test_a_field_update_sets_every_field_but_the_text() -> None:
    # A field update (design D5) changes no vector, so it carries the whole payload but the text.
    point = to_point(ROW, "Le jeu plante", np.zeros(4, dtype=np.float32))

    assert fields(ROW) == {k: v for k, v in (point.payload or {}).items() if k != "text"}
