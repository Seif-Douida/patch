"""The eval bundle (Phase 6 plan Task 10, design D10): everything CI replays, in one AES-256-GCM
file whose hash `evals/thresholds.yaml` pins. Refused before anything runs when stale, tampered
or opened with the wrong key."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from patchpulse.agent.prompts import PROMPT_VERSION
from patchpulse.config import LabSettings
from patchpulse.evals.bundle import (
    BundleSources,
    build_bundle,
    encrypt,
    keygen,
    open_bundle,
    read_bundle_pin,
    write_bundle_pin,
)
from patchpulse.evals.scoring import SCORING_VERSION

SENTINEL = "SENTINEL-REVIEW-TEXT"
KEY = base64.b64encode(b"k" * 32).decode()  # a low-entropy test key
WRONG = base64.b64encode(b"w" * 32).decode()


def sources(tmp_path: Path) -> BundleSources:
    local = tmp_path / "data" / "local" / "evals"
    (local / "runs" / "main").mkdir(parents=True)
    (local / "runs" / "injection").mkdir(parents=True)
    data = local / "data-v60.json.gz"
    data.write_bytes(gzip.compress(b'{"data_version": 60}'))
    cache = local / "answers.sqlite"
    cache.write_bytes(b"SQLite format 3\x00" + SENTINEL.encode())
    (local / "runs" / "main" / "p0.json").write_text(f'{{"text": "{SENTINEL}"}}\n', "utf-8")
    (local / "runs" / "injection" / "inj-01.json").write_text('{"case": "inj-01"}\n', "utf-8")
    pool = pd.DataFrame(
        {"review_id": [1, 2], "text": [SENTINEL, "two"], "vector": [[0.5, 0.25], [0.0, 1.0]]}
    )
    gold = pd.DataFrame(
        {"review_id": [5], "text": [SENTINEL], "language": ["english"], "aspects": ["none"]}
    )
    return BundleSources(
        data_file=data,
        data_version=60,
        cache_file=cache,
        runs={"main": local / "runs" / "main", "injection": local / "runs" / "injection"},
        pool=pool,
        query_vectors={"fps drops": [0.5, 0.5]},
        gold_test=gold,
        commit="abc1234",
    )


def test_a_bundle_round_trips(tmp_path: Path) -> None:
    given = sources(tmp_path)
    built = build_bundle(1, key=KEY, sources=given, out_dir=tmp_path / "build")

    folder = open_bundle(built.path, KEY, expected_sha256=built.sha256, into=tmp_path / "open")

    assert built.path == tmp_path / "build" / "eval-bundle-v1.bin"
    assert (folder / "data.json.gz").read_bytes() == given.data_file.read_bytes()
    assert (folder / "answers.sqlite").read_bytes() == given.cache_file.read_bytes()
    assert (folder / "runs" / "main" / "p0.json").read_text("utf-8") == (
        given.runs["main"] / "p0.json"
    ).read_text("utf-8")
    assert (folder / "runs" / "injection" / "inj-01.json").is_file()
    pool = pd.read_parquet(folder / "pool.parquet")
    assert list(pool["review_id"]) == [1, 2]
    assert list(pool["text"]) == [SENTINEL, "two"]
    assert [list(v) for v in pool["vector"]] == [[0.5, 0.25], [0.0, 1.0]]
    pd.testing.assert_frame_equal(pd.read_parquet(folder / "gold_test.parquet"), given.gold_test)
    assert json.loads((folder / "query_vectors.json").read_text("utf-8")) == given.query_vectors


def test_the_manifest_records_versions_and_hashes(tmp_path: Path) -> None:
    built = build_bundle(1, key=KEY, sources=sources(tmp_path), out_dir=tmp_path / "build")
    folder = open_bundle(built.path, KEY, expected_sha256=built.sha256, into=tmp_path / "open")

    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))

    assert manifest == built.manifest
    assert manifest["bundle_version"] == 1
    assert manifest["data_version"] == 60
    assert manifest["scoring_version"] == SCORING_VERSION
    assert manifest["prompt_version"] == PROMPT_VERSION
    assert manifest["commit"] == "abc1234"
    assert manifest["counts"] == {"main": 1, "injection": 1, "pool": 2, "gold_test": 1}
    assert manifest["created"].endswith("+00:00")
    data = (folder / "data.json.gz").read_bytes()
    assert manifest["data_sha256"] == hashlib.sha256(data).hexdigest()
    assert set(manifest["files"]) == {
        "data.json.gz", "answers.sqlite", "runs/main/p0.json", "runs/injection/inj-01.json",
        "pool.parquet", "query_vectors.json", "gold_test.parquet",
    }  # fmt: skip
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest


def test_a_bundle_whose_hash_differs_is_refused(tmp_path: Path) -> None:
    built = build_bundle(1, key=KEY, sources=sources(tmp_path), out_dir=tmp_path / "build")

    with pytest.raises(ValueError, match="hash"):
        open_bundle(built.path, WRONG, expected_sha256="0" * 64, into=tmp_path / "open")

    assert not (tmp_path / "open").exists()


def test_a_wrong_key_is_refused_without_writing_files(tmp_path: Path) -> None:
    built = build_bundle(1, key=KEY, sources=sources(tmp_path), out_dir=tmp_path / "build")

    with pytest.raises(ValueError, match="wrong key"):
        open_bundle(built.path, WRONG, expected_sha256=built.sha256, into=tmp_path / "open")

    assert not (tmp_path / "open").exists()


def test_a_file_that_differs_from_its_manifest_is_refused(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in (
            ("manifest.json", json.dumps({"files": {"pool.parquet": "0" * 64}}).encode()),
            ("pool.parquet", b"changed"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    path = tmp_path / "tampered.bin"
    path.write_bytes(encrypt(buffer.getvalue(), KEY))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=r"pool\.parquet"):
        open_bundle(path, KEY, expected_sha256=digest, into=tmp_path / "open")

    assert not (tmp_path / "open").exists()


def test_the_encrypted_file_holds_no_plain_text(tmp_path: Path) -> None:
    built = build_bundle(1, key=KEY, sources=sources(tmp_path), out_dir=tmp_path / "build")

    blob = built.path.read_bytes()

    assert blob.startswith(b"PPEVAL1")
    assert SENTINEL.encode() not in blob
    assert b"manifest" not in blob
    assert b"data_version" not in blob


def test_keygen_gives_32_random_bytes_and_settings_read_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = keygen(), keygen()
    monkeypatch.setenv("PP_EVAL_BUNDLE_KEY", KEY)

    assert len(base64.b64decode(first, validate=True)) == 32
    assert first != second
    key = LabSettings().eval_bundle_key
    assert key is not None
    assert key.get_secret_value() == KEY
    with pytest.raises(ValueError, match="32 bytes"):
        encrypt(b"x", base64.b64encode(b"short").decode())


def test_the_pin_is_written_into_the_thresholds_file(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.yaml"
    path.write_text("# The gate's pins.\nbundle: null\n\n# Tolerances.\nmacro_f1: -0.02\n", "utf-8")

    assert read_bundle_pin(path) is None
    write_bundle_pin(1, "a" * 64, path)
    write_bundle_pin(2, "b" * 64, path)

    pin = read_bundle_pin(path)
    assert pin is not None
    assert (pin.version, pin.sha256) == (2, "b" * 64)
    text = path.read_text("utf-8")
    assert text.startswith("# The gate's pins.\nbundle:\n")
    assert text.endswith("# Tolerances.\nmacro_f1: -0.02\n")
