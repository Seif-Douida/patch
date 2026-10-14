"""e5 in PyTorch, and the embedding cache (plan Task 12, design D5).

`embed_texts_gpu` runs e5 in PyTorch (16-bit on CUDA): it needs the local `gpu` group, imported
when called, so this module stays importable in CI. It pools and normalises like `OnnxEncoder`;
`test_onnx_embeddings_match_torch` checks the two agree. S1a's head is trained on the int8 ONNX
encoder's own embeddings instead (what Azure computes; 16-bit training embeddings cost ~0.075 dev
macro-F1 when served through int8); S1b fine-tunes in PyTorch (Task 13).

`cached_embeddings` keeps each matrix in `data/local/embeddings/` (it's derived from review text,
so it never leaves this machine). The file name holds the model and a hash of the ordered ids, the
cleaning version and the caller's variant (revision, prefix, length...), so a change to any of them
computes afresh.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from patchpulse.models.encoder import (
    E5_MODEL_ID,
    E5_REVISION,
    MAX_TOKENS,
    QUERY_PREFIX,
    l2_normalise,
    mean_pool,
)
from patchpulse.models.text import CLEANING_VERSION

EMBEDDINGS_DIR = "embeddings"

Embed = Callable[[list[str]], NDArray[np.float32]]


def embed_texts_gpu(
    texts: Sequence[str],
    *,
    model_id: str = E5_MODEL_ID,
    revision: str = E5_REVISION,
    prefix: str = QUERY_PREFIX,
    max_tokens: int = MAX_TOKENS,
    batch: int = 64,
    device: str | None = None,
    fp16: bool = True,
) -> NDArray[np.float32]:
    import torch  # the gpu group
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()
    prefixed = [f"{prefix}{text}" for text in texts]
    order = np.argsort([len(text) for text in prefixed], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    half = fp16 and device == "cuda"
    with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=half):
        for start in range(0, len(order), batch):
            rows = order[start : start + batch]
            encoded = tokenizer(
                [prefixed[i] for i in rows],
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state.float().cpu().numpy()
            mask = encoded["attention_mask"].cpu().numpy()
            out[rows] = l2_normalise(mean_pool(hidden, mask))
    return out


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model_id).strip("-")


def cached_embeddings(
    cache_dir: Path,
    ids: Sequence[int],
    texts: Sequence[str],
    *,
    model_id: str,
    embed: Embed,
    variant: str = "",
) -> NDArray[np.float32]:
    """`embed(texts)`, or the saved matrix for the same model, ids (in order) and variant."""
    if len(ids) != len(texts):
        raise ValueError("need one text per id")
    key = json.dumps(
        {"ids": [int(i) for i in ids], "cleaning": CLEANING_VERSION, "variant": variant}
    )
    digest = hashlib.sha256(key.encode("ascii")).hexdigest()[:16]
    path = cache_dir / f"{_slug(model_id)}-{digest}.npy"
    if path.is_file():
        cached = np.load(path, allow_pickle=False)
        if cached.shape[0] == len(ids):
            return np.asarray(cached, dtype=np.float32)
    embedded = np.asarray(embed(list(texts)), dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, embedded, allow_pickle=False)
    return embedded
