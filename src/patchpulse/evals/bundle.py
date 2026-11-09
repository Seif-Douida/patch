"""The eval bundle (Phase 6 design D10, ADR-014 as amended): everything CI replays, in one file.

The bundle is a tar of:

- `manifest.json`: the versions, the commit, the creation time and each file's SHA-256;
- `data.json.gz`: the pinned data bundle;
- `answers.sqlite`: the eval cache (model, judge and tool answers);
- `runs/main/*.json` and `runs/injection/*.json`: each case's recorded run;
- `pool.parquet`, `query_vectors.json` and `gold_test.parquet`: the retrieval and classification
  suites' data.

Several hold review text, so the tar is encrypted with AES-256-GCM. The file is `PPEVAL1`, a
12-byte nonce, then the ciphertext, with the magic as associated data. The key is 32 random bytes
in base64, made by `patchpulse-evals keygen`: the GitHub secret EVAL_BUNDLE_KEY and
PP_EVAL_BUNDLE_KEY in .env. Seif runs keygen; Claude never sees the key.

Seif uploads each build as the release asset `eval-bundle-v<N>`, and `evals/thresholds.yaml` pins
N and the file's SHA-256. Opening refuses a file whose hash isn't the pin (before decrypting), a
wrong key (writing nothing), and files that differ from the manifest.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from patchpulse.agent.prompts import PROMPT_VERSION
from patchpulse.evals.judge import RUBRIC_PATH, rubric_version
from patchpulse.evals.scoring import SCORING_VERSION

MAGIC = b"PPEVAL1"
NONCE_BYTES = 12
KEY_BYTES = 32
BUNDLE_DIR = Path("build") / "evals"
THRESHOLDS_PATH = Path("evals") / "thresholds.yaml"
MANIFEST = "manifest.json"
_PIN_BLOCK = re.compile(r"^bundle:[^\n]*\n(?:[ \t]+[^\n]*\n)*", re.MULTILINE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def bundle_name(version: int) -> str:
    return f"eval-bundle-v{version}.bin"


def keygen() -> str:
    """A new key: 32 random bytes in base64."""
    return base64.b64encode(os.urandom(KEY_BYTES)).decode("ascii")


def _key(key: str) -> bytes:
    try:
        raw = base64.b64decode(key.strip(), validate=True)
    except (binascii.Error, ValueError):
        raw = b""
    if len(raw) != KEY_BYTES:
        raise ValueError(
            "the eval bundle key must be 32 bytes in base64 (`patchpulse-evals keygen` makes one)"
        )
    return raw


def encrypt(data: bytes, key: str) -> bytes:
    nonce = os.urandom(NONCE_BYTES)
    return MAGIC + nonce + AESGCM(_key(key)).encrypt(nonce, data, MAGIC)


def decrypt(blob: bytes, key: str) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("not an eval bundle (no PPEVAL1 header)")
    nonce = blob[len(MAGIC) : len(MAGIC) + NONCE_BYTES]
    try:
        return AESGCM(_key(key)).decrypt(nonce, blob[len(MAGIC) + NONCE_BYTES :], MAGIC)
    except InvalidTag:
        raise ValueError(
            "can't decrypt the eval bundle: wrong key (PP_EVAL_BUNDLE_KEY) or a damaged file"
        ) from None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parquet(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


@dataclass(frozen=True)
class BundleSources:
    """What goes in: local files and the suites' frames, gathered by the `bundle` command."""

    data_file: Path  # the pinned data bundle
    data_version: int
    cache_file: Path  # the eval cache
    runs: Mapping[str, Path]  # arm -> its folder of stored cases
    pool: pd.DataFrame
    query_vectors: Mapping[str, list[float]]
    gold_test: pd.DataFrame
    commit: str


@dataclass(frozen=True)
class BuiltBundle:
    path: Path
    sha256: str
    manifest: dict[str, Any]


def build_bundle(
    version: int,
    *,
    key: str,
    sources: BundleSources,
    out_dir: Path = BUNDLE_DIR,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> BuiltBundle:
    """Writes `<out_dir>/eval-bundle-v<version>.bin`."""
    files: dict[str, bytes] = {
        "data.json.gz": sources.data_file.read_bytes(),
        "answers.sqlite": sources.cache_file.read_bytes(),
    }
    counts: dict[str, int] = {}
    for arm, folder in sorted(sources.runs.items()):
        stored = sorted(folder.glob("*.json"))
        counts[arm] = len(stored)
        files.update({f"runs/{arm}/{path.name}": path.read_bytes() for path in stored})
    files["pool.parquet"] = _parquet(sources.pool)
    files["query_vectors.json"] = json.dumps(dict(sources.query_vectors), sort_keys=True).encode()
    files["gold_test.parquet"] = _parquet(sources.gold_test)
    counts |= {"pool": len(sources.pool), "gold_test": len(sources.gold_test)}
    created = clock()
    manifest: dict[str, Any] = {
        "bundle_version": version,
        "data_version": sources.data_version,
        "data_sha256": _sha256(files["data.json.gz"]),
        "scoring_version": SCORING_VERSION,
        "prompt_version": PROMPT_VERSION,
        "rubric_version": rubric_version() if RUBRIC_PATH.is_file() else None,
        "commit": sources.commit,
        "created": created.isoformat(timespec="seconds"),
        "counts": counts,
        "pool_skipped": sources.pool.attrs.get("skipped"),
        "files": {name: _sha256(data) for name, data in sorted(files.items())},
    }
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        entries = {MANIFEST: json.dumps(manifest, indent=2, sort_keys=True).encode(), **files}
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mtime = int(created.timestamp())
            tar.addfile(member, io.BytesIO(data))
    blob = encrypt(archive.getvalue(), key)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / bundle_name(version)
    path.write_bytes(blob)
    return BuiltBundle(path, _sha256(blob), manifest)


def _check_manifest(folder: Path) -> None:
    manifest_path = folder / MANIFEST
    if not manifest_path.is_file():
        raise ValueError("the eval bundle has no manifest")
    listed: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    found = {
        path.relative_to(folder).as_posix()
        for path in folder.rglob("*")
        if path.is_file() and path != manifest_path
    }
    wrong = sorted(set(listed) ^ found)
    wrong += sorted(
        name
        for name in set(listed) & found
        if _sha256((folder / name).read_bytes()) != listed[name]
    )
    if wrong:
        raise ValueError(f"the eval bundle's files don't match its manifest: {', '.join(wrong)}")


def open_bundle(path: Path, key: str, *, expected_sha256: str, into: Path | None = None) -> Path:
    """Decrypt and unpack the bundle into `into` (new or empty; a new temporary folder by
    default) and return the folder. Nothing is written unless the hash, the key and the
    manifest all check out."""
    blob = path.read_bytes()
    digest = _sha256(blob)
    if digest != expected_sha256:
        raise ValueError(
            f"{path.name}'s hash ({digest[:12]}) differs from the pin ({expected_sha256[:12]}) "
            f"in {THRESHOLDS_PATH.as_posix()}: download the pinned release"
        )
    archive = decrypt(blob, key)
    if into is not None and into.exists() and any(into.iterdir()):
        raise ValueError(f"{into} isn't empty")
    folder = into or Path(tempfile.mkdtemp(prefix="pp-eval-bundle-"))
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(folder, filter="data")
        _check_manifest(folder)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder


@dataclass(frozen=True)
class BundlePin:
    version: int
    sha256: str


def read_bundle_pin(path: Path = THRESHOLDS_PATH) -> BundlePin | None:
    """The pinned bundle, or None while `bundle: null`."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pinned = document.get("bundle")
    if pinned is None:
        return None
    sha256 = str(pinned.get("sha256", ""))
    if not _SHA256.match(sha256):
        raise ValueError(f"{path}: bundle.sha256 isn't a SHA-256")
    return BundlePin(int(pinned["version"]), sha256)


def write_bundle_pin(version: int, sha256: str, path: Path = THRESHOLDS_PATH) -> None:
    """Pin a bundle in place of the `bundle:` block, keeping the rest of the file as it is."""
    text = path.read_text(encoding="utf-8")
    block = f"bundle:\n  version: {version}\n  sha256: {sha256}\n"
    updated, found = _PIN_BLOCK.subn(block, text, count=1)
    if found != 1:
        raise ValueError(f"{path} has no `bundle:` line to pin")
    path.write_text(updated, encoding="utf-8", newline="\n")
