"""The drift check's e5 vectors (Phase 7 design D2), read from the review index by review id.

Points are keyed by review id with the vector `dense` (Phase 3). Reading them sends no text and
needs only the read key. Most ids of an old week aren't in the index (ADR-022): those give no
row. A dropped connection is retried twice before it fails the check.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from patchpulse.retrieval.qdrant import DENSE

_BATCH = 256
ATTEMPTS = 3
BACKOFF_SECONDS = 2.0  # doubled after each failed try


class QdrantVectors:
    """A `VectorSource` over a collection (or alias) of the review index."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.collection = collection
        self.sleep = sleep

    def _retrieve(self, ids: list[int]) -> list[Any]:
        for attempt in range(ATTEMPTS):
            try:
                return list(
                    self.client.retrieve(
                        self.collection, ids, with_payload=False, with_vectors=[DENSE]
                    )
                )
            except ResponseHandlingException:
                if attempt == ATTEMPTS - 1:
                    raise
                self.sleep(BACKOFF_SECONDS * 2**attempt)
        raise AssertionError("unreachable")

    def by_id(self, ids: Sequence[int]) -> dict[int, NDArray[np.float32]]:
        """The vectors the index holds, keyed by review id (ids it lacks are left out)."""
        found: dict[int, NDArray[np.float32]] = {}
        for start in range(0, len(ids), _BATCH):
            for point in self._retrieve([int(i) for i in ids[start : start + _BATCH]]):
                vector = point.vector
                if isinstance(vector, dict) and DENSE in vector:
                    found[int(point.id)] = np.asarray(vector[DENSE], dtype=np.float32)
        return found

    def vectors(self, ids: Sequence[int]) -> NDArray[np.float32]:
        return stack(self.by_id(ids), ids)


def stack(found: dict[int, NDArray[np.float32]], ids: Sequence[int]) -> NDArray[np.float32]:
    """The rows for `ids` that `found` holds, in order; none gives an empty matrix (a 0-row
    reshape can't infer its width)."""
    rows = [found[int(i)] for i in ids if int(i) in found]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
