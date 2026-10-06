"""Publishing through the GitHub App (plan Task 13, ADR-018), against an in-process fake GitHub."""

from __future__ import annotations

import base64
import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx2
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr

from patchpulse.publish.github import (
    API_VERSION,
    FAILURE_TITLE,
    SITE_BRANCH,
    GitHubError,
    GitHubPublisher,
    close_issue,
    dispatch,
    installation_token,
    open_or_update_issue,
    replace_branch,
)

REPO = "Seif-Douida/patch"
APP_ID = "123456"
NOW = datetime(2026, 10, 6, 3, 30, tzinfo=UTC)


@pytest.fixture(scope="module")
def key_pair() -> tuple[str, rsa.RSAPublicKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()
    return pem, key.public_key()


class FakeGitHub:
    """Just enough of the REST API: the app installation, Git Data, dispatches and issues."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.refs: dict[str, str] = {}
        self.issues: list[dict[str, Any]] = []
        self.comments: dict[int, list[str]] = {}
        self.dispatches: list[dict[str, Any]] = []
        self.commits: dict[str, dict[str, Any]] = {}
        self.blobs: dict[str, bytes] = {}
        self.trees: dict[str, list[dict[str, Any]]] = {}
        self.fail_token = False

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self))

    def body(self, index: int = -1) -> Any:
        return json.loads(self.requests[index].content)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        method = request.method
        payload: Any = json.loads(request.content) if request.content else {}
        if path == f"/repos/{REPO}/installation":
            return httpx2.Response(200, json={"id": 42})
        if path == "/app/installations/42/access_tokens":
            if self.fail_token:
                return httpx2.Response(
                    401, json={"message": "A JSON web token could not be decoded"}
                )
            return httpx2.Response(201, json={"token": "ghs_installation", "expires_at": "x"})
        if path == f"/repos/{REPO}/git/blobs":
            sha = f"blob{len(self.blobs)}"
            self.blobs[sha] = base64.b64decode(payload["content"])
            return httpx2.Response(201, json={"sha": sha})
        if path == f"/repos/{REPO}/git/trees":
            sha = f"tree{len(self.trees)}"
            self.trees[sha] = payload["tree"]
            return httpx2.Response(201, json={"sha": sha})
        if path == f"/repos/{REPO}/git/commits":
            sha = f"commit{len(self.commits)}"
            self.commits[sha] = payload
            return httpx2.Response(201, json={"sha": sha})
        if path.startswith(f"/repos/{REPO}/git/refs/heads/") and method == "PATCH":
            branch = path.rsplit("/", 1)[1]
            if branch not in self.refs:
                return httpx2.Response(422, json={"message": "Reference does not exist"})
            self.refs[branch] = payload["sha"]
            return httpx2.Response(200, json={"object": {"sha": payload["sha"]}})
        if path == f"/repos/{REPO}/git/refs" and method == "POST":
            self.refs[payload["ref"].removeprefix("refs/heads/")] = payload["sha"]
            return httpx2.Response(201, json={"object": {"sha": payload["sha"]}})
        if path == f"/repos/{REPO}/dispatches":
            self.dispatches.append(payload)
            return httpx2.Response(204)
        if path == f"/repos/{REPO}/issues" and method == "GET":
            label = request.url.params["labels"]
            found = [i for i in self.issues if i["state"] == "open" and label in i["labels"]]
            return httpx2.Response(200, json=found)
        if path == f"/repos/{REPO}/issues" and method == "POST":
            issue = {"number": len(self.issues) + 1, "state": "open", **payload}
            self.issues.append(issue)
            return httpx2.Response(201, json=issue)
        if path.startswith(f"/repos/{REPO}/issues/") and path.endswith("/comments"):
            number = int(path.split("/")[-2])
            self.comments.setdefault(number, []).append(payload["body"])
            return httpx2.Response(201, json={})
        if path.startswith(f"/repos/{REPO}/issues/") and method == "PATCH":
            issue = self.issues[int(path.rsplit("/", 1)[1]) - 1]
            issue.update(payload)
            return httpx2.Response(200, json=issue)
        return httpx2.Response(404, json={"message": f"unexpected {method} {path}"})


@pytest.fixture
def github() -> FakeGitHub:
    return FakeGitHub()


def test_token_exchange_uses_a_short_lived_jwt(
    github: FakeGitHub, key_pair: tuple[str, rsa.RSAPublicKey]
) -> None:
    pem, public_key = key_pair

    token = installation_token(APP_ID, pem, REPO, github.client(), now=NOW)

    assert token == "ghs_installation"
    bearer = github.requests[0].headers["Authorization"].removeprefix("Bearer ")
    claims = jwt.decode(bearer, public_key, algorithms=["RS256"], options={"verify_exp": False})
    assert claims["iss"] == APP_ID
    assert claims["iat"] <= NOW.timestamp() - 30  # backdated for clock skew
    assert claims["exp"] - claims["iat"] <= 600  # GitHub allows at most 10 minutes
    # The installation token is narrowed to this repo and the two permissions publishing needs.
    assert github.body(1) == {
        "repositories": ["patch"],
        "permissions": {"contents": "write", "issues": "write"},
    }
    assert github.requests[1].headers["X-GitHub-Api-Version"] == API_VERSION


def test_replace_branch_creates_an_orphan_commit_and_force_updates(github: FakeGitHub) -> None:
    github.refs[SITE_BRANCH] = "old"
    files = {"index.json": b'{"a": 1}\n', "games/553850.json": b'{"b": 2}\n'}

    commit = replace_branch("ghs", REPO, SITE_BRANCH, files, "site data v7", github.client())

    assert github.refs[SITE_BRANCH] == commit
    assert github.commits[commit]["parents"] == []  # one commit, no history to bloat the branch
    assert github.commits[commit]["message"] == "site data v7"
    (tree,) = github.trees.values()
    stored = {entry["path"]: github.blobs[entry["sha"]] for entry in tree}
    assert stored == files
    patch = next(r for r in github.requests if r.method == "PATCH")
    assert json.loads(patch.content)["force"] is True


def test_replace_branch_creates_the_ref_when_missing(github: FakeGitHub) -> None:
    commit = replace_branch(
        "ghs", REPO, SITE_BRANCH, {"index.json": b"{}\n"}, "v1", github.client()
    )

    assert github.refs == {SITE_BRANCH: commit}


def test_dispatch_sends_the_event_type_and_payload(github: FakeGitHub) -> None:
    dispatch("ghs", REPO, "site-data-updated", {"data_version": 7}, github.client())

    assert github.dispatches == [
        {"event_type": "site-data-updated", "client_payload": {"data_version": 7}}
    ]


def test_failure_opens_one_issue_and_success_closes_it(github: FakeGitHub) -> None:
    client = github.client()

    first = open_or_update_issue(
        "ghs", REPO, title=FAILURE_TITLE, body="run 1", label="alert", http=client
    )
    again = open_or_update_issue(
        "ghs", REPO, title=FAILURE_TITLE, body="run 2", label="alert", http=client
    )
    closed = close_issue(
        "ghs", REPO, title=FAILURE_TITLE, label="alert", comment="run 3 ok", http=client
    )

    assert first == again == closed == 1
    assert len(github.issues) == 1
    assert github.issues[0]["labels"] == ["alert"]
    assert github.comments[1] == ["run 2", "run 3 ok"]
    assert github.issues[0]["state"] == "closed"


def test_closing_without_an_open_issue_does_nothing(github: FakeGitHub) -> None:
    closed = close_issue(
        "ghs", REPO, title=FAILURE_TITLE, label="alert", comment="ok", http=github.client()
    )

    assert closed is None
    assert all(r.method == "GET" for r in github.requests)


def test_api_errors_name_the_call_and_status(github: FakeGitHub) -> None:
    with pytest.raises(GitHubError, match=r"POST /repos/Seif-Douida/patch/nope.*HTTP 404"):
        dispatch("ghs", "Seif-Douida/patch/nope", "x", {}, github.client())


def test_private_key_is_never_logged(
    github: FakeGitHub,
    key_pair: tuple[str, rsa.RSAPublicKey],
    caplog: pytest.LogCaptureFixture,
) -> None:
    pem, _ = key_pair
    github.fail_token = True
    publisher = GitHubPublisher(
        app_id=APP_ID, private_key=SecretStr(pem), repo=REPO, http=github.client(), now=lambda: NOW
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(GitHubError) as raised:
        publisher.publish({"index.json": b"{}\n"}, data_version=1)

    jwt_sent = github.requests[0].headers["Authorization"].removeprefix("Bearer ")
    key_body = pem.splitlines()[1]
    for text in (str(raised.value), caplog.text, repr(publisher)):
        assert key_body not in text
        assert jwt_sent not in text
    assert "HTTP 401" in str(raised.value)


def test_publisher_publishes_then_dispatches_the_commit(
    github: FakeGitHub, key_pair: tuple[str, rsa.RSAPublicKey]
) -> None:
    pem, _ = key_pair
    publisher = GitHubPublisher(
        app_id=APP_ID, private_key=SecretStr(pem), repo=REPO, http=github.client(), now=lambda: NOW
    )

    publisher.publish({"index.json": b"{}\n"}, data_version=9)

    assert github.dispatches == [
        {
            "event_type": "site-data-updated",
            "client_payload": {"data_version": 9, "commit": github.refs[SITE_BRANCH]},
        }
    ]


def test_publisher_reports_failures_as_an_issue_and_closes_it_on_success(
    github: FakeGitHub, key_pair: tuple[str, rsa.RSAPublicKey]
) -> None:
    pem, _ = key_pair
    publisher = GitHubPublisher(
        app_id=APP_ID, private_key=SecretStr(pem), repo=REPO, http=github.client(), now=lambda: NOW
    )

    publisher.report(run_id=5, status="failed", error="dbt build failed: boom")
    publisher.report(run_id=6, status="succeeded", error=None)

    (issue,) = github.issues
    assert issue["title"] == FAILURE_TITLE
    assert "run 5" in issue["body"]
    assert "dbt build failed: boom" in issue["body"]
    assert issue["state"] == "closed"
    assert "run 6" in github.comments[1][-1]
