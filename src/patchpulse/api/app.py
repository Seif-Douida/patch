"""FastAPI entry point (`uvicorn patchpulse.api.app:app`)."""

import os

from fastapi import FastAPI

# Interactive docs stay off until the real API exists (Phase 5); every request wakes a replica.
app = FastAPI(title="PatchPulse API", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by the platform probe, the deploy smoke test and uptime.yml.

    It never touches SQL: a paused free-offer database must not make the API look down.
    `APP_VERSION` is baked into the image at build time (the git SHA).
    """
    return {"status": "ok", "version": os.environ.get("APP_VERSION", "dev")}
