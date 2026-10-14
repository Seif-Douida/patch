"""Silver embeddings on the GPU and their cache (plan Task 12). The cache test runs anywhere; the
ONNX-against-PyTorch check needs the `gpu` group, CUDA and the e5 download (Seif's machine)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from patchpulse.models.embed_gpu import cached_embeddings, embed_texts_gpu
from patchpulse.models.encoder import OnnxEncoder, build_encoder


def test_embedding_cache_is_keyed_by_model_and_ids(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> NDArray[np.float32]:
        calls.append(texts)
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3) + len(calls)

    first = cached_embeddings(tmp_path, [1, 2], ["a", "b"], model_id="org/m1", embed=embed)
    again = cached_embeddings(tmp_path, [1, 2], ["a", "b"], model_id="org/m1", embed=embed)
    assert len(calls) == 1
    assert np.array_equal(first, again)

    cached_embeddings(tmp_path, [2, 1], ["b", "a"], model_id="org/m1", embed=embed)  # row order
    cached_embeddings(tmp_path, [1, 2], ["a", "b"], model_id="org/m2", embed=embed)
    cached_embeddings(tmp_path, [1, 2], ["a", "b"], model_id="org/m1", embed=embed, variant="x")
    assert len(calls) == 4
    names = sorted(path.name for path in tmp_path.glob("*.npy"))
    assert len(names) == 4
    assert all(name.startswith(("org-m1-", "org-m2-")) for name in names)


def test_ids_and_texts_must_line_up(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one text per id"):
        cached_embeddings(
            tmp_path, [1, 2], ["a"], model_id="m", embed=lambda t: np.zeros((1, 3), np.float32)
        )


# Invented sentences, English and French: no review text in the repo.
SENTENCES = [
    "The frame rate drops every time a boss appears.",
    "Servers were down all weekend and matchmaking never worked.",
    "A beautiful world with a story I will remember for years.",
    "The battle pass is overpriced for what it offers.",
    "Controls feel clunky with a gamepad.",
    "It crashes to the desktop after the latest patch.",
    "Too much grinding and not enough new content.",
    "The new weapons are completely unbalanced.",
    "My cat likes watching me play this.",
    "Refunded because the launcher would not start.",
    "Le jeu rame dès qu'il y a trop d'ennemis à l'écran.",
    "Les serveurs sont tombés pendant tout le week-end.",
    "Une histoire magnifique et des personnages attachants.",
    "Les microtransactions gâchent complètement l'expérience.",
    "Les commandes sont pénibles à la manette.",
    "Le jeu plante à chaque chargement de sauvegarde.",
    "Il manque du contenu après la fin de l'histoire.",
    "Les nouvelles armes sont beaucoup trop puissantes.",
    "Mon chat adore regarder l'écran pendant que je joue.",
    "Remboursé, le lanceur refuse de démarrer.",
]


@pytest.mark.gpu
def test_onnx_embeddings_match_torch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch", reason="needs the gpu group")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    reference = embed_texts_gpu(SENTENCES, device="cpu", fp16=False)
    on_gpu = embed_texts_gpu(SENTENCES)  # 16-bit on the GTX 1650, as for the silver set
    fp32 = OnnxEncoder(build_encoder(tmp_path / "fp32", quantize=False)).embed(SENTENCES)
    int8 = OnnxEncoder(build_encoder(tmp_path / "int8")).embed(SENTENCES)

    def cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        return float((a * b).sum(axis=1).min())  # rows are unit length

    assert reference.shape == (len(SENTENCES), 384)
    assert cosine(on_gpu, reference) >= 0.99
    assert cosine(fp32, reference) >= 0.99
    assert cosine(int8, reference) >= 0.98
