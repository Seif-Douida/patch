"""A tiny fake e5 for the students' tests: a word-level tokenizer and an ONNX graph that looks each
token up in a table, so pooling, truncation and padding can be checked by hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from patchpulse.models.encoder import ENCODER_FILE, TOKENIZER_FILE

WORDS = (
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "query",
    ":",
    "good",
    "bad",
    "game",
    "fps",
    "drops",
    "crashes",
    "great",
    "story",
    "server",
    "down",
)
DIM = 4


def tiny_table() -> NDArray[np.float32]:
    table = np.random.default_rng(0).normal(size=(len(WORDS), DIM)).astype(np.float32)
    table[0] = 100.0  # <pad>: if padding were pooled, every padded embedding would move far
    return table


def write_tiny_encoder(model_dir: Path) -> NDArray[np.float32]:
    """`encoder.onnx` and `tokenizer.json` in `model_dir`; returns the lookup table."""
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 2), ("</s>", 3)]
    )
    tokenizer.save(str(model_dir / TOKENIZER_FILE))
    table = tiny_table()
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gather", ["table", "input_ids"], ["last_hidden_state"]),
            helper.make_node("Identity", ["attention_mask"], ["mask_echo"]),
        ],
        name="tiny-e5",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"]),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "last_hidden_state", TensorProto.FLOAT, ["batch", "tokens", DIM]
            ),
            helper.make_tensor_value_info("mask_echo", TensorProto.INT64, ["batch", "tokens"]),
        ],
        initializer=[numpy_helper.from_array(table, "table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(model_dir / ENCODER_FILE))
    return table


def expected_embedding(
    text: str, table: NDArray[np.float32], *, max_tokens: int = 256
) -> NDArray[np.float32]:
    """By hand: "<s> query : words </s>", cut to max_tokens, mean of the rows, unit length."""
    index = {w: i for i, w in enumerate(WORDS)}
    words = ["query", ":", *text.split()][: max_tokens - 2]
    ids = [2, *(index.get(w, 1) for w in words), 3]
    mean = table[ids].mean(axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)
