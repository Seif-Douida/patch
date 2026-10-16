"""The e5 encoder for S1a and S1b, on ONNX Runtime (plan Task 12, design D5).

`intfloat/multilingual-e5-small` ships its own ONNX export (`onnx/model.onnx`), so that's used
rather than exporting from PyTorch, at a pinned revision. `build_encoder` quantizes it to int8 with
ONNX Runtime's dynamic quantization, for CPU scoring in Azure.

As the model card asks: each text gets the "query: " prefix (classification features), token
vectors are mean-pooled over real tokens only (padding excluded), and the result is scaled to unit
length.

NumPy, ONNX Runtime and tokenizers only (the `scoring` group). `build_encoder` also needs
huggingface-hub and onnx, imported when called (the `gpu` group, on Seif's machine).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from tokenizers import Tokenizer

E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # the repo's main on 2026-10-12
QUERY_PREFIX = "query: "
MAX_TOKENS = 256
ENCODER_FILE = "encoder.onnx"
TOKENIZER_FILE = "tokenizer.json"
PAD_TOKEN = "<pad>"  # noqa: S105 (the tokenizer's padding token, not a secret)


def mean_pool(hidden: NDArray[np.float32], mask: NDArray[np.int64]) -> NDArray[np.float32]:
    """Average of the token vectors where the attention mask is 1."""
    weights = mask[..., None].astype(np.float32)
    counts = np.maximum(weights.sum(axis=1), 1e-9)
    return np.asarray((hidden * weights).sum(axis=1) / counts, dtype=np.float32)


def l2_normalise(rows: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.asarray(rows / np.where(norms > 0, norms, 1.0), dtype=np.float32)


class OnnxTextModel:
    """Tokenize with a prefix, run an ONNX graph in length-sorted batches, and return one row per
    text from `_rows`. `threads` caps ONNX Runtime's CPU threads. Shared by S1a and S1b and by
    Phase 3's retrieval embedders."""

    kind = "model"

    def __init__(self, model_dir: Path, model_file: str, *, threads: int | None = None) -> None:
        graph, tokenizer = model_dir / model_file, model_dir / TOKENIZER_FILE
        missing = [path.name for path in (graph, tokenizer) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"no {self.kind} in {model_dir}: {', '.join(missing)} missing")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self._session: Any = ort.InferenceSession(
            str(graph), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer))
        pad_id = self._tokenizer.token_to_id(PAD_TOKEN)
        self._tokenizer.enable_padding(
            pad_id=pad_id if pad_id is not None else 0, pad_token=PAD_TOKEN
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._output = self._session.get_outputs()[0].name
        self.dim = int(self._session.get_outputs()[0].shape[-1])

    def _rows(self, output: NDArray[np.float32], mask: NDArray[np.int64]) -> NDArray[np.float32]:
        return output

    def _batch(self, texts: list[str]) -> NDArray[np.float32]:
        encodings = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        wanted = {name: value for name, value in feeds.items() if name in self._inputs}
        (output,) = self._session.run([self._output], wanted)
        return self._rows(np.asarray(output, dtype=np.float32), mask)

    def _run(
        self, texts: Sequence[str], *, prefix: str, max_tokens: int, batch: int
    ) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if not texts:
            return out
        self._tokenizer.enable_truncation(max_length=max_tokens)
        prefixed = [f"{prefix}{text}" for text in texts]
        # Batches of similar length waste less time on padding; rows go back to their places.
        order = np.argsort([len(text) for text in prefixed], kind="stable")
        for start in range(0, len(order), batch):
            rows = order[start : start + batch]
            out[rows] = self._batch([prefixed[i] for i in rows])
        return out


class OnnxEncoder(OnnxTextModel):
    """`embed(texts) -> (n, dim)` unit vectors: mean-pooled over real tokens, then normalised."""

    kind = "encoder"

    def __init__(self, model_dir: Path, *, threads: int | None = None) -> None:
        super().__init__(model_dir, ENCODER_FILE, threads=threads)

    def _rows(self, output: NDArray[np.float32], mask: NDArray[np.int64]) -> NDArray[np.float32]:
        return l2_normalise(mean_pool(output, mask))

    def embed(
        self,
        texts: Sequence[str],
        *,
        prefix: str = QUERY_PREFIX,
        max_tokens: int = MAX_TOKENS,
        batch: int = 32,
    ) -> NDArray[np.float32]:
        return self._run(texts, prefix=prefix, max_tokens=max_tokens, batch=batch)


class OnnxClassifier(OnnxTextModel):
    """`logits(texts) -> (n, outputs)`: a graph that pools and classifies itself (S1b)."""

    kind = "classifier"

    def __init__(self, model_dir: Path, model_file: str, *, threads: int | None = None) -> None:
        super().__init__(model_dir, model_file, threads=threads)

    def logits(
        self,
        texts: Sequence[str],
        *,
        prefix: str = QUERY_PREFIX,
        max_tokens: int = MAX_TOKENS,
        batch: int = 32,
    ) -> NDArray[np.float32]:
        return self._run(texts, prefix=prefix, max_tokens=max_tokens, batch=batch)


def build_encoder(
    out_dir: Path,
    *,
    model_id: str = E5_MODEL_ID,
    revision: str = E5_REVISION,
    quantize: bool = True,
) -> Path:
    """The e5 ONNX encoder and its tokenizer in `out_dir` (int8 unless `quantize=False`).

    Downloads from Hugging Face at the pinned revision on first use; an existing build is reused.
    """
    if (out_dir / ENCODER_FILE).is_file() and (out_dir / TOKENIZER_FILE).is_file():
        return out_dir
    from huggingface_hub import hf_hub_download  # the gpu group, on Seif's machine

    onnx_path = hf_hub_download(model_id, "onnx/model.onnx", revision=revision)
    tokenizer_path = hf_hub_download(model_id, "onnx/tokenizer.json", revision=revision)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}-", dir=out_dir.parent))
    try:
        if quantize:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(onnx_path, staging / ENCODER_FILE, weight_type=QuantType.QInt8)
        else:
            shutil.copyfile(onnx_path, staging / ENCODER_FILE)
        shutil.copyfile(tokenizer_path, staging / TOKENIZER_FILE)
        shutil.rmtree(out_dir, ignore_errors=True)  # a build interrupted before this point
        os.replace(staging, out_dir)  # whole builds only
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out_dir
