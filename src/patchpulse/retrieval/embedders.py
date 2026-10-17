"""The retrieval embedders (design D3, ADR-004): two multilingual candidates, each run as an int8
ONNX graph on ONNX Runtime. One file embeds a collection's reviews (the backlog on Seif's machine,
each night's additions in `pp-nightly`) and its queries, so every vector in an index comes from
the same encoder.

A build is a directory: `model.onnx` (int8, or fp32 for the parity checks), `tokenizer.json`, and
`embedder.json`, which records the spec, how the graph was quantized, the text cleaning version and
each file's SHA-256. `OnnxEmbedder` refuses a directory whose files don't match it.

NumPy, ONNX Runtime and tokenizers only (the `scoring` group). `build_embedder` also needs
huggingface-hub and onnx, imported when called (the `gpu` group, on Seif's machine).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from patchpulse.models.encoder import OnnxTextModel, l2_normalise, mean_pool
from patchpulse.models.text import CLEANING_VERSION

MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"
MANIFEST_FILE = "embedder.json"
# ONNX Runtime's dynamic quantization with unsigned 8-bit weights and one scale per output channel.
# Signed weights saturate on x86 CPUs without VNNI (Seif's i7-9750H), so a vector could depend on
# the CPU that computed it; they also left MiniLM under the 0.98 parity floor (0.976). Unsigned per
# channel: e5-small 0.995, MiniLM 0.990 (plan Task 4).
INT8 = "int8-dynamic-u8-per-channel"
FP32 = "fp32"
_HIDDEN = "last_hidden_state"


class EmbedderIntegrityError(RuntimeError):
    """A build's files don't match its `embedder.json` (missing, altered or unknown)."""


@dataclass(frozen=True)
class EmbedderSpec:
    name: str
    model_id: str
    revision: str  # a commit of the Hugging Face repo
    onnx_file: str  # the repo's fp32 ONNX export
    tokenizer_file: str
    pooling: Literal["mean", "cls"]
    query_prefix: str
    passage_prefix: str
    max_tokens: int
    dims: int  # the output size, after Matryoshka truncation where it applies
    licence: str


# Checked on Hugging Face on 2026-10-16 (plan Task 4): each repo ships its own ONNX export.
CANDIDATES: dict[str, EmbedderSpec] = {
    spec.name: spec
    for spec in (
        # The spec's default, already used by S1a.
        EmbedderSpec(
            name="e5-small",
            model_id="intfloat/multilingual-e5-small",
            revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            onnx_file="onnx/model.onnx",
            tokenizer_file="onnx/tokenizer.json",
            pooling="mean",
            query_prefix="query: ",
            passage_prefix="passage: ",
            max_tokens=512,
            dims=384,
            licence="MIT",
        ),
        # The same size, trained for similarity; a 128-token window, so long reviews are cut.
        EmbedderSpec(
            name="minilm-l12",
            model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
            onnx_file="onnx/model.onnx",
            tokenizer_file="tokenizer.json",
            pooling="mean",
            query_prefix="",
            passage_prefix="",
            max_tokens=128,
            dims=384,
            licence="Apache-2.0",
        ),
    )
}
# Dropped (ADR-004): gte-multilingual-base (no ONNX export); snowflake-arctic-embed-m-v2.0 (every
# int8 build under the 0.98 parity floor, 0.94 at best); multilingual-e5-base (parity 0.985, but
# under the speed floor at 2.1 passages/s, so ineligible to serve; dropped on 17 Oct rather than
# spend ~9 h embedding its backlog for the eval alone).


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(model_dir: Path, spec: EmbedderSpec, *, quantization: str) -> None:
    """`embedder.json` for the model and tokenizer already in `model_dir`."""
    manifest = {
        "spec": asdict(spec),
        "quantization": quantization,
        "cleaning_version": CLEANING_VERSION,
        "files": {name: _sha256(model_dir / name) for name in (MODEL_FILE, TOKENIZER_FILE)},
    }
    (model_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_manifest(model_dir: Path) -> dict[str, Any]:
    """The build's manifest, after checking every file against it."""
    path = model_dir / MANIFEST_FILE
    if not path.is_file():
        raise EmbedderIntegrityError(f"no {MANIFEST_FILE} in {model_dir}")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    names = {f.name for f in fields(EmbedderSpec)}
    if (
        set(manifest.get("spec", {})) != names
        or set(manifest.get("files", {})) != {MODEL_FILE, TOKENIZER_FILE}
        or manifest.get("quantization") not in (INT8, FP32)
    ):
        raise EmbedderIntegrityError(f"{MANIFEST_FILE} in {model_dir} isn't an embedder manifest")
    for name, expected in manifest["files"].items():
        file = model_dir / name
        if not file.is_file() or _sha256(file) != expected:
            raise EmbedderIntegrityError(f"{name} in {model_dir} doesn't match {MANIFEST_FILE}")
    return manifest


class OnnxEmbedder(OnnxTextModel):
    """`embed_passages` and `embed_queries` -> (n, dims) unit vectors: each with its prefix, pooled
    by the spec, cut to `dims`, then normalised."""

    kind = "embedder"

    def __init__(self, model_dir: Path, *, threads: int | None = None) -> None:
        manifest = read_manifest(model_dir)
        self.spec = EmbedderSpec(**manifest["spec"])
        self.quantization: str = manifest["quantization"]
        self.model_sha256: str = manifest["files"][MODEL_FILE]
        self.cleaning_version: str = manifest["cleaning_version"]
        super().__init__(model_dir, MODEL_FILE, threads=threads)
        names = [o.name for o in self._session.get_outputs()]
        self._output = _HIDDEN if _HIDDEN in names else names[0]
        self.dim = self.spec.dims

    def _rows(self, output: NDArray[np.float32], mask: NDArray[np.int64]) -> NDArray[np.float32]:
        pooled = mean_pool(output, mask) if self.spec.pooling == "mean" else output[:, 0, :]
        return l2_normalise(np.ascontiguousarray(pooled[:, : self.spec.dims]))

    def embed_passages(self, texts: Sequence[str], *, batch: int = 32) -> NDArray[np.float32]:
        return self._run(
            texts, prefix=self.spec.passage_prefix, max_tokens=self.spec.max_tokens, batch=batch
        )

    def embed_queries(self, texts: Sequence[str], *, batch: int = 32) -> NDArray[np.float32]:
        return self._run(
            texts, prefix=self.spec.query_prefix, max_tokens=self.spec.max_tokens, batch=batch
        )


def build_embedder(spec: EmbedderSpec, out_dir: Path, *, quantize: bool = True) -> Path:
    """`spec`'s ONNX export and tokenizer in `out_dir`, quantized to int8 with ONNX Runtime's
    dynamic quantization (our own, even where the repo ships one) unless `quantize=False`.

    Downloads from Hugging Face at the pinned revision; an existing, intact build of the same spec
    and quantization is reused.
    """
    quantization = INT8 if quantize else FP32
    if (out_dir / MANIFEST_FILE).is_file():
        existing = read_manifest(out_dir)
        if existing["spec"] == asdict(spec) and existing["quantization"] == quantization:
            return out_dir
    from huggingface_hub import hf_hub_download  # the gpu group, on Seif's machine

    onnx_path = hf_hub_download(spec.model_id, spec.onnx_file, revision=spec.revision)
    tokenizer_path = hf_hub_download(spec.model_id, spec.tokenizer_file, revision=spec.revision)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}-", dir=out_dir.parent))
    try:
        if quantize:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                onnx_path, staging / MODEL_FILE, weight_type=QuantType.QUInt8, per_channel=True
            )
        else:
            shutil.copyfile(onnx_path, staging / MODEL_FILE)
        shutil.copyfile(tokenizer_path, staging / TOKENIZER_FILE)
        write_manifest(staging, spec, quantization=quantization)
        shutil.rmtree(out_dir, ignore_errors=True)  # a build interrupted before this point
        os.replace(staging, out_dir)  # whole builds only
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out_dir
