"""The API image (Phase 5 plan Task 17): FastAPI with the agent, the pinned query encoder for
search_reviews and the models config, without the jobs image's pipeline."""

from __future__ import annotations

import re
from pathlib import Path

from image_groups import hidden_modules, image_groups, run_with_only

ROOT = Path(__file__).resolve().parents[2]


def runtime_stage() -> str:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8").split("\nFROM ", 2)[2]


def test_the_api_imports_the_agent_with_only_its_image_groups() -> None:
    installed = image_groups("Dockerfile")
    assert {"api", "agent", "retrieval", "causal"} <= installed
    hidden = hidden_modules(installed)
    assert {"dbt", "alembic", "sklearn"} <= hidden  # the jobs image's pipeline and scoring

    result = run_with_only(
        installed,
        "import patchpulse.agent.graph, patchpulse.api.app\n"
        "from patchpulse.api.ask import LiveDeps, live_limits\n"
        "from patchpulse.retrieval.embedders import OnnxEmbedder\n"
        "import onnxruntime, tokenizers\n"
        "print('imported')\n",
    )

    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip() == "imported"


def test_api_image_carries_the_encoder_and_the_config() -> None:
    runtime = runtime_stage()
    assert re.search(r"^WORKDIR /app$", runtime, re.M)  # config/models.yaml is read from here
    assert re.search(r"^COPY config ./config$", runtime, re.M)
    assert re.search(r"^COPY build/encoder /app/encoder$", runtime, re.M)
    assert re.search(r"PP_ENCODER_DIR=/app/encoder\b", runtime)
    # The image refuses to build with an encoder its own packages can't load and run, or with
    # an agent that doesn't import.
    assert runtime.index("COPY build/encoder ") < runtime.index("embed_queries")
    assert "import patchpulse.agent.graph, patchpulse.api.app" in runtime
