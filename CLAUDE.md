Read PATCHPULSE_SPEC.md (design intent) and PROJECT_STATE.md (current state) before any task.
Follow the working agreement in SPEC §0. Current phase: see PROJECT_STATE.md.
Never create paid resources or enable billing on the Gemini project. Ask if unsure.

## Repo conventions

- Seif runs every `git commit` and `git push`. Print the exact commands instead of running them.
- Implementation plans live in `docs/plans/` (gitignored) and contain no full code snippets.
- Python: uv with Python 3.12. Checks: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run pytest`.
- Infra: Bicep in `infra/`. `tests/infra/test_zero_cost_guards.py` must pass. Never loosen a $0 guard or add a resource type without an ADR (ADR-017).
