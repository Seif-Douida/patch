"""A review as a Qdrant point (design D2): id = the Steam recommendation id; the `dense` vector from
the collection's encoder; the `bm25` vector, computed by Qdrant from the text with the review's
language options; the indexed filter fields; and, stored only, the cleaned text, the version, the
fingerprint of the fields and the index rules' version."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from qdrant_client import models

from patchpulse.retrieval.qdrant import BM25_MODEL, DENSE, SPARSE, bm25_options

STORED_FIELDS = ("text", "version", "fingerprint", "rules_version")


def _value(value: Any) -> Any:
    """A JSON value: NumPy scalars as Python ones, pandas' nulls as None."""
    if value is None or (not isinstance(value, str | bytes) and pd.isna(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def payload(row: Mapping[str, Any], text: str) -> dict[str, Any]:
    """The payload of `core.review_index`'s row (its columns, as the pull writes them)."""
    return {**fields(row), "text": text}


def fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Every payload field but the text: what a field update sets (design D5)."""
    aspects = _value(row["aspects"]) or ""
    return {
        "appid": int(row["appid"]),
        "day": int(row["day"]),
        "created_ts": int(row["created_ts"]),
        "language": str(row["language"]),
        "voted_up": bool(row["voted_up"]),
        "playtime_bucket": _value(row["playtime_bucket"]),
        "aspects": [code for code in str(aspects).split("|") if code],
        "aspect_model_version": _value(row["aspect_model_version"]),
        "first_seen_run_id": int(row["first_seen_run_id"]),
        "version": pd.Timestamp(row["version"]).isoformat(),
        "fingerprint": str(row["fingerprint"]),
        "rules_version": str(row["rules_version"]),
    }


def to_point(row: Mapping[str, Any], text: str, vector: NDArray[np.float32]) -> models.PointStruct:
    """`text` is the cleaned text, the one the vector was embedded from."""
    return models.PointStruct(
        id=int(row["review_id"]),
        vector={
            DENSE: [float(x) for x in vector],
            SPARSE: models.Document(
                text=text, model=BM25_MODEL, options=bm25_options(str(row["language"]))
            ),
        },
        payload=payload(row, text),
    )
