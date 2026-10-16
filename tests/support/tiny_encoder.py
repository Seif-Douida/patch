"""A tiny fake e5 for the students' tests, and fake retrieval embedders: a word-level tokenizer and
an ONNX graph that looks each token up in a table, so pooling, truncation and padding can be
checked by hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from patchpulse.models.encoder import ENCODER_FILE, TOKENIZER_FILE
from patchpulse.retrieval.embedders import MODEL_FILE as EMBEDDER_MODEL
from patchpulse.retrieval.embedders import TOKENIZER_FILE as EMBEDDER_TOKENIZER
from patchpulse.retrieval.embedders import EmbedderSpec, write_manifest

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


def write_tiny_classifier(model_dir: Path, *, file_name: str) -> NDArray[np.float32]:
    """A fake S1b: `tokenizer.json` and an ONNX graph whose logits are the masked mean of each
    token's row in a (words x aspects) table. Returns the table."""
    model_dir.mkdir(parents=True, exist_ok=True)
    write_tiny_encoder(model_dir)  # for its tokenizer
    (model_dir / ENCODER_FILE).unlink()
    table = np.random.default_rng(1).normal(size=(len(WORDS), 10)).astype(np.float32)
    table[0] = 100.0  # <pad>
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axis_tokens")
    last = numpy_helper.from_array(np.array([2], dtype=np.int64), "axis_last")
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gather", ["table", "input_ids"], ["rows"]),
            helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["mask_f", "axis_last"], ["mask_3d"]),
            helper.make_node("Mul", ["rows", "mask_3d"], ["masked"]),
            helper.make_node("ReduceSum", ["masked", "axis_tokens"], ["total"], keepdims=0),
            helper.make_node("ReduceSum", ["mask_3d", "axis_tokens"], ["count"], keepdims=0),
            helper.make_node("Div", ["total", "count"], ["logits"]),
        ],
        name="tiny-s1b",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"]),
        ],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 10])],
        initializer=[numpy_helper.from_array(table, "table"), axes, last],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(model_dir / file_name))
    return table


def expected_logits(text: str, table: NDArray[np.float32]) -> NDArray[np.float32]:
    index = {w: i for i, w in enumerate(WORDS)}
    ids = [2, *(index.get(w, 1) for w in ["query", ":", *text.split()]), 3]
    return np.asarray(table[ids].mean(axis=0), dtype=np.float32)


# Phase 3: fake retrieval embedders (plan Task 4). The prefixes "query: " and "passage: " are words.
EMBEDDER_WORDS = (*WORDS, "passage")


def write_tiny_embedder(
    model_dir: Path,
    spec: EmbedderSpec,
    *,
    mix: bool = False,
    template: bool = True,
    dim: int = DIM,
) -> NDArray[np.float32]:
    """`model.onnx`, `tokenizer.json` and `embedder.json` for `spec`. The graph looks each token up
    in a (words x DIM) table; with `mix`, every position also gets the masked sum of the text's
    rows, so a CLS-pooled vector depends on the whole text and never on padding. Without
    `template`, no <s> and </s> are added. `dim` must match `spec.dims`. Returns the table."""
    model_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = {w: i for i, w in enumerate(EMBEDDER_WORDS)}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if template:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>", special_tokens=[("<s>", 2), ("</s>", 3)]
        )
    tokenizer.save(str(model_dir / EMBEDDER_TOKENIZER))
    table = np.random.default_rng(2).normal(size=(len(EMBEDDER_WORDS), dim)).astype(np.float32)
    table[0] = 100.0  # <pad>
    nodes = [helper.make_node("Gather", ["table", "input_ids"], ["rows"])]
    if mix:
        nodes += [
            helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["mask_f", "axis_last"], ["mask_3d"]),
            helper.make_node("Mul", ["rows", "mask_3d"], ["masked"]),
            helper.make_node("ReduceSum", ["masked", "axis_tokens"], ["total"], keepdims=1),
            helper.make_node("Add", ["rows", "total"], ["last_hidden_state"]),
        ]
    else:
        nodes.append(helper.make_node("Identity", ["rows"], ["last_hidden_state"]))
    graph = helper.make_graph(
        nodes=nodes,
        name="tiny-embedder",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"]),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "last_hidden_state", TensorProto.FLOAT, ["batch", "tokens", dim]
            )
        ],
        initializer=[
            numpy_helper.from_array(table, "table"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axis_tokens"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "axis_last"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(model_dir / EMBEDDER_MODEL))
    write_manifest(model_dir, spec, quantization="fp32")
    return table


def embedder_ids(text: str, *, prefix: str, max_tokens: int, template: bool = True) -> list[int]:
    """By hand: the tiny tokenizer's ids for `prefix + text`, cut to `max_tokens`."""
    index = {w: i for i, w in enumerate(EMBEDDER_WORDS)}
    words = f"{prefix}{text}".replace(":", " : ").split()
    if not template:
        return [index.get(w, 1) for w in words][:max_tokens]
    return [*[2, *(index.get(w, 1) for w in words)][: max_tokens - 1], 3]
