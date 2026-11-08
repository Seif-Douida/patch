"""The frozen data bundle (Phase 6 design D2): one night's published agent bundle, fetched once,
kept under data/local and pinned by its data version and SHA-256 in the committed
`evals/data.yaml`, so every eval run and the eval bundle read the same night."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from patchpulse.agent.bundle import BUNDLE_FILE, REPORTS_FILE, Bundle, load_bundle
from patchpulse.evals.records import EVALS_DATA, require_local

DATA_PIN = Path("evals") / "data.yaml"


@dataclass(frozen=True)
class DataPin:
    data_version: int
    sha256: str


def data_file(data_version: int, root: Path = EVALS_DATA) -> Path:
    return root / f"data-v{data_version}.json.gz"


def fetch_data(
    fetch: Callable[[str], bytes], *, root: Path = EVALS_DATA, pin_path: Path = DATA_PIN
) -> DataPin:
    """The published bundle and reports.json; refused, with nothing written, when they come from
    different nights (a cache serving one file late)."""
    raw = fetch(BUNDLE_FILE)
    bundle = load_bundle(raw)
    reports_version = int(json.loads(fetch(REPORTS_FILE))["data_version"])
    if bundle.data_version != reports_version:
        raise ValueError(
            f"the published files disagree: the bundle is version {bundle.data_version}, "
            f"reports.json {reports_version}; fetch again once both are the same night's"
        )
    path = data_file(bundle.data_version, root)
    require_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    pin = DataPin(bundle.data_version, hashlib.sha256(raw).hexdigest())
    document = {"data_version": pin.data_version, "sha256": pin.sha256}
    pin_path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8", newline="\n")
    return pin


def read_data_pin(pin_path: Path = DATA_PIN) -> DataPin:
    if not pin_path.is_file():
        raise ValueError(f"no data bundle pinned yet: run `patchpulse-evals data --fetch` first "
                         f"({pin_path} is missing)")  # fmt: skip
    document = yaml.safe_load(pin_path.read_text(encoding="utf-8"))
    return DataPin(int(document["data_version"]), str(document["sha256"]))


def load_pinned_bundle(pin: DataPin, *, root: Path = EVALS_DATA) -> Bundle:
    """The pinned night's bundle, refused when its file's hash differs from the pin."""
    raw = data_file(pin.data_version, root).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise ValueError(
            f"data-v{pin.data_version}.json.gz's hash differs from {DATA_PIN}: fetch it again"
        )
    return load_bundle(raw)
