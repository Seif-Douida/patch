"""Phase 7 Task 17 (found by the first calibration, 18 Nov): ids missing from the review index.

A reference sample id that the index doesn't hold must give no row, not crash the drift check;
the cache asks the index once per id only to tell which ids came back.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from qdrant_client.http.exceptions import ResponseHandlingException

from patchpulse.models.monitor_cli import CachedVectors
from patchpulse.monitoring.vectors import QdrantVectors
from patchpulse.retrieval.qdrant import DENSE


class FakeIndex:
    """A Qdrant client holding 4-dimensional vectors for some ids."""

    def __init__(self, held: Sequence[int]) -> None:
        self.held = set(held)
        self.calls = 0

    def retrieve(self, collection: str, ids: list[int], **kwargs: Any) -> list[Any]:
        self.calls += 1
        return [
            SimpleNamespace(id=i, vector={DENSE: [float(i), 0.0, 0.0, 1.0]})
            for i in ids
            if i in self.held
        ]


def test_ids_the_index_lacks_give_an_empty_matrix() -> None:
    vectors = QdrantVectors(FakeIndex([]), "reviews")  # type: ignore[arg-type]

    found = vectors.vectors([7, 8])

    assert found.shape[0] == 0
    assert found.dtype == np.float32


def test_the_cache_keeps_the_ids_found_and_skips_the_rest() -> None:
    index = FakeIndex([1, 3])
    cached = CachedVectors(QdrantVectors(index, "reviews"))  # type: ignore[arg-type]

    first: NDArray[np.float32] = cached.vectors([1, 2, 3])
    again = cached.vectors([3, 2])
    nothing = cached.vectors([2])

    assert first.shape == (2, 4)
    assert first[:, 0].tolist() == [1.0, 3.0]
    assert again[:, 0].tolist() == [3.0]
    assert nothing.shape[0] == 0


def test_the_cache_asks_the_index_once_for_ids_it_lacks() -> None:
    # The second calibration (18 Nov) refetched every missing id on its own: ~1,000 requests a
    # week for the weeks the index doesn't cover, until the cluster dropped a connection.
    index = FakeIndex([1, 3])
    cached = CachedVectors(QdrantVectors(index, "reviews"))  # type: ignore[arg-type]

    cached.vectors([1, 2, 3, 4, 5])
    cached.vectors([2, 4])

    assert index.calls == 1


class FlakyIndex(FakeIndex):
    """Drops the connection on the first `drops` requests."""

    def __init__(self, held: Sequence[int], drops: int) -> None:
        super().__init__(held)
        self.drops = drops

    def retrieve(self, collection: str, ids: list[int], **kwargs: Any) -> list[Any]:
        if self.drops:
            self.drops -= 1
            self.calls += 1
            raise ResponseHandlingException(ConnectionError("server disconnected"))
        return super().retrieve(collection, ids, **kwargs)


def test_a_dropped_connection_is_retried() -> None:
    index = FlakyIndex([1], drops=1)
    vectors = QdrantVectors(index, "reviews", sleep=lambda _: None)  # type: ignore[arg-type]

    found = vectors.vectors([1])

    assert found.shape == (1, 4)
    assert index.calls == 2


def test_an_outage_still_raises_after_the_retries() -> None:
    index = FlakyIndex([1], drops=10)
    vectors = QdrantVectors(index, "reviews", sleep=lambda _: None)  # type: ignore[arg-type]

    with pytest.raises(ResponseHandlingException):
        vectors.vectors([1])
    assert index.calls == 3
