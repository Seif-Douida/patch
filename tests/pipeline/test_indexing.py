"""The index stage's encoder (plan Task 15; the first indexed run, 20 Oct)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tiny_encoder import write_tiny_embedder

from patchpulse.pipeline.indexing import INDEX_THREADS, index_embedder
from patchpulse.retrieval.embedders import CANDIDATES

TINY = replace(CANDIDATES["e5-small"], name="tiny-index", dims=8, max_tokens=32)


def test_the_stage_embeds_on_one_thread(tmp_path: Path) -> None:
    # pp-nightly has 1 vCPU, but ONNX Runtime's default starts a thread per core it sees (12 on
    # the laptop): in the jobs image under a 1-CPU quota that embedded 3.2x slower than one thread,
    # the setting the speed floor was measured with (reports/retrieval/embedder-speed.json).
    write_tiny_embedder(tmp_path / "encoder", TINY, dim=TINY.dims)

    embedder = index_embedder(tmp_path / "encoder")

    assert INDEX_THREADS == 1
    assert embedder.session_options().intra_op_num_threads == INDEX_THREADS
