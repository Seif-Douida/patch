"""Each candidate's ONNX builds against its PyTorch original (plan Task 4, design D3): the model
card's pooling and prefixes on 20 English and French sentences (invented, not reviews). Needs the
`gpu` group and the downloads, so it runs on Seif's machine only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from patchpulse.retrieval.embedders import CANDIDATES, EmbedderSpec, OnnxEmbedder, build_embedder

BUILDS = Path(__file__).resolve().parents[2] / "build" / "encoders"
SENTENCES = [
    "The game crashes every time I load my save.",
    "Frame rate drops below thirty in the city.",
    "Matchmaking takes ten minutes and then fails.",
    "The new update added three maps and a raid.",
    "Way too expensive for what you get.",
    "The story is moving and the characters feel real.",
    "Controls on a gamepad are clumsy.",
    "Servers were down all weekend.",
    "Great fun with friends, highly recommended.",
    "The patch nerfed my favourite weapon into the ground.",
    "Le jeu plante a chaque chargement de sauvegarde.",
    "Les performances sont catastrophiques depuis la mise a jour.",
    "Trop cher pour le contenu propose.",
    "Les serveurs sont tombes tout le week-end.",
    "L'histoire est magnifique et la musique aussi.",
    "La nouvelle mise a jour ajoute des cartes.",
    "Impossible de trouver une partie en ligne.",
    "Les controles a la manette sont penibles.",
    "Un tres bon jeu entre amis.",
    "Le correctif a casse mon arme preferee.",
]


def reference(spec: EmbedderSpec, texts: list[str], prefix: str) -> NDArray[np.float32]:
    """The original model in PyTorch (fp32, CPU), pooled and cut as the card says."""
    torch = pytest.importorskip("torch", reason="needs the gpu group")
    transformers = pytest.importorskip("transformers", reason="needs the gpu group")
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    model = transformers.AutoModel.from_pretrained(spec.model_id, revision=spec.revision).eval()
    batch = tokenizer(
        [f"{prefix}{text}" for text in texts],
        padding=True,
        truncation=True,
        max_length=spec.max_tokens,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    if spec.pooling == "mean":
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    else:
        pooled = hidden[:, 0]
    rows = np.asarray(pooled[:, : spec.dims].numpy(), dtype=np.float32)
    return np.asarray(rows / np.linalg.norm(rows, axis=1, keepdims=True), dtype=np.float32)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_candidate_matches_its_reference(name: str) -> None:
    spec = CANDIDATES[name]
    want = {
        "queries": reference(spec, SENTENCES, spec.query_prefix),
        "passages": reference(spec, SENTENCES, spec.passage_prefix),
    }

    for quantize, floor in ((False, 0.999), (True, 0.98)):
        build = BUILDS / f"{name}-{'int8' if quantize else 'fp32'}"
        embedder = OnnxEmbedder(build_embedder(spec, build, quantize=quantize))
        got = {
            "queries": embedder.embed_queries(SENTENCES),
            "passages": embedder.embed_passages(SENTENCES),
        }
        for kind, rows in got.items():
            cosines = (rows * want[kind]).sum(axis=1)
            print(f"{name} {build.name} {kind}: min cosine {cosines.min():.4f}")
            assert cosines.min() >= floor, (name, build.name, kind, cosines.min())
