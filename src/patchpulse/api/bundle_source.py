"""The data the API answers from (Phase 5, design D2 and D9): the night's bundle and reports, read
from the `agent-data` branch, never from SQL.

The data is loaded on first use, then the published version (reports.json's `data_version`) is
checked at most every RECHECK_S; the bundle is downloaded again only when it changed. A failed
load, or a bundle and reports.json from different nights (a cache serving one file late), keeps
what was loaded and is tried again after RETRY_S. Each load is a new frozen snapshot: a run that
holds one keeps it for every call, V2's re-execution included, whatever loads meanwhile.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx2

from patchpulse.agent.bundle import BUNDLE_FILE, REPORTS_FILE, Bundle, load_bundle

log = logging.getLogger("patchpulse.api")

RECHECK_S = 300.0
RETRY_S = 30.0
FETCH_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class AgentData:
    data_version: int
    bundle: Bundle
    reports: Mapping[str, Mapping[str, Any]]  # bundle id -> the public copy it shows
    patches: Mapping[str, str]  # patch id -> its bundle id


def http_fetch(base_url: str) -> Callable[[str], bytes]:
    """Each file from the branch's raw URL."""

    def fetch(name: str) -> bytes:
        response = httpx2.get(
            f"{base_url.rstrip('/')}/{name}", timeout=FETCH_TIMEOUT_S, follow_redirects=True
        )
        response.raise_for_status()
        return response.content

    return fetch


class BundleSource:
    def __init__(
        self,
        fetch: Callable[[str], bytes],
        *,
        clock: Callable[[], float] = time.monotonic,
        recheck_s: float = RECHECK_S,
        retry_s: float = RETRY_S,
    ) -> None:
        self._fetch = fetch
        self._clock = clock
        self.recheck_s = recheck_s
        self.retry_s = retry_s
        self._data: AgentData | None = None
        self._next_check: float | None = None
        self._lock = threading.Lock()

    def current(self) -> AgentData | None:
        """The loaded data, checked against the published version when it's time; None until a
        load has worked."""
        if not self._due():
            return self._data
        # The first load makes callers wait for it; later ones serve the old data meanwhile.
        if not self._lock.acquire(blocking=self._data is None):
            return self._data
        try:
            if self._due():
                self._refresh()
            return self._data
        finally:
            self._lock.release()

    def _due(self) -> bool:
        return self._next_check is None or self._clock() >= self._next_check

    def _refresh(self) -> None:
        try:
            index = json.loads(self._fetch(REPORTS_FILE))
            version = int(index["data_version"])
            if self._data is None or version != self._data.data_version:
                bundle = load_bundle(self._fetch(BUNDLE_FILE))
                if bundle.data_version != version:
                    raise ValueError(
                        f"the bundle is version {bundle.data_version}, reports.json {version}"
                    )
                self._data = AgentData(version, bundle, index["reports"], index["patches"])
                log.info("agent data version %d loaded", version)
        except Exception as error:  # keep what we have; try again soon
            log.warning("agent data not loaded (%s: %s)", type(error).__name__, error)
            self._next_check = self._clock() + self.retry_s
        else:
            self._next_check = self._clock() + self.recheck_s
