"""Phase 7: the drift check's e5 vectors, read from the index by review id."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from patchpulse.monitoring.vectors import QdrantVectors
from patchpulse.retrieval.qdrant import DENSE, ensure_collection

pytestmark = pytest.mark.qdrant


def test_vectors_come_back_for_the_ids_found_in_their_order(qdrant: QdrantClient) -> None:
    name = f"test_vectors_{uuid.uuid4().hex[:8]}"
    ensure_collection(qdrant, name, 4)
    try:
        qdrant.upsert(
            name,
            [
                models.PointStruct(id=i, vector={DENSE: [float(i), 0.0, 0.0, 1.0]}, payload={})
                for i in (1, 2, 3)
            ],
            wait=True,
        )
        found = QdrantVectors(qdrant, name).vectors([3, 99, 1])

        assert found.shape == (2, 4)
        assert found.dtype == np.float32
        # Qdrant normalises cosine vectors; the order follows the ids asked for.
        assert found[0, 0] > found[1, 0]
    finally:
        qdrant.delete_collection(name)
