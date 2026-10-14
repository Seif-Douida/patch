"""S1b's fine-tuning and ONNX export (plan Task 13): PyTorch, the local `gpu` group only.

- **Network:** e5-small, mean pooling over real tokens, dropout, a linear head with one logit per
  aspect. The word-embedding table can stay frozen (`S1bParams.freeze_word_embeddings`).
- **Training:** BCE with `pos_weights`, 16-bit autocast on CUDA, AdamW with a linear warm-up and
  decay, batches of similar-length reviews. After each epoch the loss on the silver holdout decides
  whether to stop; the best epoch's weights are kept.
- **Checkpoint:** `s1b.pt` (a state dict, loaded with `weights_only=True`) and `s1b.json`
  (`write_training`), so a finished fine-tune can be exported again without retraining.
- **Export:** the whole network to ONNX (logits out), then int8 with ONNX Runtime's dynamic
  quantization, next to e5's own `tokenizer.json`.

Texts are cleaned here (`clean_texts`), as `S1bScorer` cleans them when scoring.
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn
from transformers import AutoModel, AutoTokenizer

from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.encoder import E5_MODEL_ID, E5_REVISION, QUERY_PREFIX, TOKENIZER_FILE
from patchpulse.models.s1b import (
    CHECKPOINT_FILE,
    S1B_FILE,
    S1bParams,
    S1bTraining,
    early_stop,
    length_grouped_batches,
    pos_weights,
    write_training,
)
from patchpulse.models.text import clean_texts

log = logging.getLogger("patchpulse.models.s1b")


class S1bNet(nn.Module):
    def __init__(self, encoder: Any, outputs: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, outputs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits: torch.Tensor = self.head(self.dropout(pooled))
        return logits


def _pretrained() -> tuple[Any, S1bNet]:
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_ID, revision=E5_REVISION)
    encoder = AutoModel.from_pretrained(
        E5_MODEL_ID, revision=E5_REVISION, attn_implementation="eager"
    )
    return tokenizer, S1bNet(encoder, len(ASPECTS))


def _device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _prefixed(texts: Sequence[str]) -> list[str]:
    return [f"{QUERY_PREFIX}{text}" for text in clean_texts(texts)]


def _logits(
    net: S1bNet, tokenizer: Any, texts: list[str], *, device: str, max_len: int, batch: int
) -> torch.Tensor:
    """Logits for already-prefixed texts, in their order, without gradients."""
    net.eval()
    order = np.argsort([len(text) for text in texts], kind="stable")
    out = torch.zeros((len(texts), len(ASPECTS)), dtype=torch.float32)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"),
    ):
        for start in range(0, len(order), batch):
            rows = order[start : start + batch]
            encoded = tokenizer(
                [texts[i] for i in rows],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            logits = net(encoded["input_ids"], encoded["attention_mask"])
            out[torch.as_tensor(rows)] = logits.float().cpu()
    return out


def train_s1b(
    train_texts: Sequence[str],
    train_labels: ArrayLike,
    holdout_texts: Sequence[str],
    holdout_labels: ArrayLike,
    out_dir: Path,
    *,
    params: S1bParams | None = None,
    device: str | None = None,
) -> S1bTraining:
    params = params or S1bParams()
    device = _device(device)
    torch.manual_seed(params.seed)
    started = time.monotonic()
    tokenizer, net = _pretrained()
    net.to(device)
    if params.freeze_word_embeddings:
        net.encoder.embeddings.word_embeddings.weight.requires_grad_(False)
    y_train = torch.as_tensor(np.asarray(train_labels, dtype=np.float32))
    y_holdout = torch.as_tensor(np.asarray(holdout_labels, dtype=np.float32))
    weights = torch.as_tensor(pos_weights(train_labels), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in net.encoder.parameters() if p.requires_grad], "lr": params.lr},
            {"params": list(net.head.parameters()), "lr": params.head_lr},
        ],
        weight_decay=params.weight_decay,
    )
    total = math.ceil(len(train_texts) / params.batch) * params.epochs
    warmup = max(1, int(total * params.warmup))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    train_in, holdout_in = _prefixed(train_texts), _prefixed(holdout_texts)
    lengths = [len(text) for text in train_in]
    losses: list[float] = []
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(params.epochs):
        net.train()
        batches = length_grouped_batches(lengths, batch=params.batch, seed=params.seed, epoch=epoch)
        for step, rows in enumerate(batches, start=1):
            encoded = tokenizer(
                [train_in[i] for i in rows],
                padding=True,
                truncation=True,
                max_length=params.max_len,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = net(encoded["input_ids"], encoded["attention_mask"])
            loss = loss_fn(logits.float(), y_train[torch.as_tensor(rows)].to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if step % 200 == 0:
                log.info("epoch %d: %d of %d batches", epoch + 1, step, len(batches))
        holdout = _logits(
            net, tokenizer, holdout_in, device=device, max_len=params.max_len, batch=64
        )
        losses.append(float(loss_fn(holdout.to(device), y_holdout.to(device)).item()))
        best, stop = early_stop(losses, patience=params.patience)
        log.info("epoch %d: holdout loss %.4f (best: epoch %d)", epoch + 1, losses[-1], best)
        if best == epoch + 1:
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if stop:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / CHECKPOINT_FILE)
    training = S1bTraining(
        checkpoint=out_dir,
        epochs_run=len(losses),
        best_epoch=early_stop(losses, patience=params.patience)[0],
        holdout_losses=tuple(losses),
        seconds=time.monotonic() - started,
        params=params,
    )
    write_training(training)
    return training


def _load(checkpoint: Path, device: str) -> tuple[Any, S1bNet]:
    tokenizer, net = _pretrained()
    net.load_state_dict(torch.load(checkpoint / CHECKPOINT_FILE, weights_only=True))
    return tokenizer, net.to(device).eval()


def torch_logits(
    checkpoint: Path, texts: Sequence[str], *, device: str | None = None, batch: int = 32
) -> NDArray[np.float32]:
    """The checkpoint's logits in PyTorch: 32-bit on the CPU, 16-bit on CUDA."""
    device = _device(device)
    tokenizer, net = _load(checkpoint, device)
    logits = _logits(
        net, tokenizer, _prefixed(texts), device=device, max_len=S1bParams().max_len, batch=batch
    )
    return np.asarray(logits.numpy(), dtype=np.float32)


def export_s1b(checkpoint: Path, out_dir: Path, *, quantize: bool = True) -> Path:
    """`s1b.onnx` (int8 unless `quantize=False`) and e5's `tokenizer.json` in `out_dir`."""
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tokenizer, net = _load(checkpoint, "cpu")
    example = tokenizer(
        [f"{QUERY_PREFIX}an example", f"{QUERY_PREFIX}two"], padding=True, return_tensors="pt"
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}-", dir=out_dir.parent))
    try:
        fp32 = staging / "fp32.onnx"
        torch.onnx.export(
            net,
            (example["input_ids"], example["attention_mask"]),
            str(fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_shapes={
                "input_ids": {0: "batch", 1: "tokens"},
                "attention_mask": {0: "batch", 1: "tokens"},
            },
            external_data=False,
            verbose=False,  # its progress lines carry emoji
        )
        # The exporter annotates every intermediate shape. ONNX Runtime's quantizer transposes the
        # head's Gemm weight and then re-checks shapes against those stale notes (384 vs 10), so
        # they're dropped: they're optional, and ONNX infers them again.
        graph = onnx.load(str(fp32))
        del graph.graph.value_info[:]
        onnx.save(graph, str(fp32))
        if quantize:
            quantize_dynamic(fp32, staging / S1B_FILE, weight_type=QuantType.QInt8)
            fp32.unlink()
        else:
            fp32.rename(staging / S1B_FILE)
        tokenizer_path = hf_hub_download(E5_MODEL_ID, "onnx/tokenizer.json", revision=E5_REVISION)
        shutil.copyfile(tokenizer_path, staging / TOKENIZER_FILE)
        shutil.rmtree(out_dir, ignore_errors=True)
        staging.rename(out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out_dir
