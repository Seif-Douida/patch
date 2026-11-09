"""The arms of a live run (Phase 6 design D7): the main run and the ablations, each with a
single-model chain so no arm mixes models. A1 (verifier off vs on) needs no arm: it is read from
the main run's drafts."""

from __future__ import annotations

from dataclasses import dataclass

from patchpulse.agent.search import SearchMode

FLASH_LITE_35 = "gemini-3.5-flash-lite"
FLASH_LITE_31 = "gemini-3.1-flash-lite"


@dataclass(frozen=True)
class Arm:
    name: str
    chain: tuple[str, ...]
    search_mode: SearchMode = "hybrid"
    causal_tool: bool = True
    injection: bool = False


ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm("main", (FLASH_LITE_35,)),
        Arm("injection", (FLASH_LITE_35,), injection=True),
        Arm("a2", (FLASH_LITE_35,), search_mode="dense"),  # dense vs hybrid search
        Arm("a3", (FLASH_LITE_35,), causal_tool=False),  # no estimate_patch_effect
        Arm("a4", (FLASH_LITE_31,)),  # the other Flash Lite model
    )
}
