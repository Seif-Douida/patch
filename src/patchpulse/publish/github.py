"""Publishing through a GitHub App (ADR-018).

The nightly job writes the site JSON to the orphan `site-data` branch as a single commit, then
sends a `repository_dispatch` that starts publish-site.yml: a commit on a branch without workflow
files can't trigger a workflow by itself. A failed run opens (or comments on) one issue labelled
`alert`; the next successful run closes it.

The app is installed on this repo only, with Contents and Issues write, and a ruleset keeps it off
`main`. Each installation token is narrowed further to this repo and those two permissions. The
private key and the tokens only ever travel in request headers: errors carry the call, the HTTP
status and GitHub's message, never a header.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import jwt
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr

log = logging.getLogger("patchpulse.publish")

API = "https://api.github.com"
API_VERSION = "2026-03-10"
SITE_BRANCH = "site-data"
DISPATCH_EVENT = "site-data-updated"
FAILURE_TITLE = "pipeline failed: nightly"
ALERT_LABEL = "alert"
# GitHub accepts app JWTs valid for at most 10 minutes; backdate for clock skew.
_JWT_LIFETIME = timedelta(minutes=9)
_CLOCK_SKEW = timedelta(seconds=60)
_TIMEOUT_S = 30.0


class GitHubError(RuntimeError):
    """A GitHub API call failed; the message names the call and status, never a credential."""


def _call(
    http: httpx2.Client,
    method: str,
    path: str,
    *,
    bearer: str,
    body: Mapping[str, Any] | None = None,
    params: Mapping[str, str] | None = None,
) -> Any:
    response = http.request(
        method,
        API + path,
        json=body,
        params=params,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "PatchPulse",
        },
        timeout=_TIMEOUT_S,
    )
    if response.is_error:
        try:
            message = str(response.json().get("message", ""))
        except ValueError:
            message = ""
        raise GitHubError(f"{method} {path}: HTTP {response.status_code} {message}".strip())
    return response.json() if response.content else None


def usable_private_key(pem: str) -> bool:
    """Whether `pem` is the unencrypted RSA private key that RS256 signing needs.

    Checked before the nightly run starts: PyJWT only finds out at publish, half an hour later, and
    its message ("Could not parse the provided public key") points the wrong way.
    """
    try:
        key = serialization.load_pem_private_key(pem.encode(), password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm):  # TypeError: the key is encrypted
        return False
    return isinstance(key, rsa.RSAPrivateKey)


def app_jwt(app_id: str, private_key: str, *, now: datetime) -> str:
    """The short-lived token that authenticates as the app itself."""
    claims = {
        "iat": int((now - _CLOCK_SKEW).timestamp()),
        "exp": int((now + _JWT_LIFETIME).timestamp()),
        "iss": app_id,
    }
    return jwt.encode(claims, private_key, algorithm="RS256")


def installation_token(
    app_id: str, private_key: str, repo: str, http: httpx2.Client, *, now: datetime
) -> str:
    """An installation token for `repo`, limited to Contents and Issues write."""
    bearer = app_jwt(app_id, private_key, now=now)
    installation = _call(http, "GET", f"/repos/{repo}/installation", bearer=bearer)
    access = _call(
        http,
        "POST",
        f"/app/installations/{installation['id']}/access_tokens",
        bearer=bearer,
        body={
            "repositories": [repo.split("/", 1)[1]],
            "permissions": {"contents": "write", "issues": "write"},
        },
    )
    token: str = access["token"]
    return token


def replace_branch(
    token: str,
    repo: str,
    branch: str,
    files: Mapping[str, bytes],
    message: str,
    http: httpx2.Client,
) -> str:
    """Point `branch` at a new parentless commit holding exactly `files`; returns its SHA."""
    entries = []
    for path in sorted(files):
        blob = _call(
            http,
            "POST",
            f"/repos/{repo}/git/blobs",
            bearer=token,
            body={"content": base64.b64encode(files[path]).decode(), "encoding": "base64"},
        )
        entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    tree = _call(http, "POST", f"/repos/{repo}/git/trees", bearer=token, body={"tree": entries})
    commit = _call(
        http,
        "POST",
        f"/repos/{repo}/git/commits",
        bearer=token,
        body={"message": message, "tree": tree["sha"], "parents": []},
    )
    sha: str = commit["sha"]
    try:
        _call(
            http,
            "PATCH",
            f"/repos/{repo}/git/refs/heads/{branch}",
            bearer=token,
            body={"sha": sha, "force": True},
        )
    except GitHubError as error:
        if "HTTP 422" not in str(error) and "HTTP 404" not in str(error):
            raise
        # The first publish: the branch doesn't exist yet.
        _call(
            http,
            "POST",
            f"/repos/{repo}/git/refs",
            bearer=token,
            body={"ref": f"refs/heads/{branch}", "sha": sha},
        )
    return sha


def dispatch(
    token: str, repo: str, event_type: str, payload: Mapping[str, Any], http: httpx2.Client
) -> None:
    _call(
        http,
        "POST",
        f"/repos/{repo}/dispatches",
        bearer=token,
        body={"event_type": event_type, "client_payload": dict(payload)},
    )


def _open_issue(token: str, repo: str, title: str, label: str, http: httpx2.Client) -> int | None:
    issues = _call(
        http,
        "GET",
        f"/repos/{repo}/issues",
        bearer=token,
        params={"labels": label, "state": "open", "per_page": "100"},
    )
    number = next((issue["number"] for issue in issues if issue["title"] == title), None)
    return int(number) if number is not None else None


def open_or_update_issue(
    token: str, repo: str, *, title: str, body: str, label: str, http: httpx2.Client
) -> int:
    """One open issue per title: create it, or comment on the one already open."""
    number = _open_issue(token, repo, title, label, http)
    if number is None:
        created = _call(
            http,
            "POST",
            f"/repos/{repo}/issues",
            bearer=token,
            body={"title": title, "body": body, "labels": [label]},
        )
        return int(created["number"])
    _call(
        http, "POST", f"/repos/{repo}/issues/{number}/comments", bearer=token, body={"body": body}
    )
    return number


def close_issue(
    token: str, repo: str, *, title: str, label: str, comment: str, http: httpx2.Client
) -> int | None:
    """Comment on and close the open issue with this title, if there is one."""
    number = _open_issue(token, repo, title, label, http)
    if number is None:
        return None
    _call(
        http,
        "POST",
        f"/repos/{repo}/issues/{number}/comments",
        bearer=token,
        body={"body": comment},
    )
    _call(
        http,
        "PATCH",
        f"/repos/{repo}/issues/{number}",
        bearer=token,
        body={"state": "closed", "state_reason": "completed"},
    )
    return number


class GitHubPublisher:
    """The runner's publisher in Azure: site data to `site-data`, run outcomes to an issue."""

    def __init__(
        self,
        *,
        app_id: str,
        private_key: SecretStr,
        repo: str,
        http: httpx2.Client,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.app_id = app_id
        self.repo = repo
        self._private_key = private_key
        self._http = http
        self._now = now

    def __repr__(self) -> str:
        return f"GitHubPublisher(app_id={self.app_id!r}, repo={self.repo!r})"

    def _token(self) -> str:
        return installation_token(
            self.app_id,
            self._private_key.get_secret_value(),
            self.repo,
            self._http,
            now=self._now(),
        )

    def publish(self, files: Mapping[str, bytes], *, data_version: int) -> None:
        token = self._token()
        commit = replace_branch(
            token, self.repo, SITE_BRANCH, files, f"site data v{data_version}", self._http
        )
        dispatch(
            token,
            self.repo,
            DISPATCH_EVENT,
            {"data_version": data_version, "commit": commit},
            self._http,
        )
        log.info("published %d files to %s (%s) and dispatched", len(files), SITE_BRANCH, commit)

    def report(self, *, run_id: int, status: str, error: str | None) -> None:
        token = self._token()
        if status == "succeeded":
            closed = close_issue(
                token,
                self.repo,
                title=FAILURE_TITLE,
                label=ALERT_LABEL,
                comment=f"Recovered: run {run_id} succeeded.",
                http=self._http,
            )
            if closed is not None:
                log.info("closed issue #%d", closed)
            return
        body = (
            f"Nightly run {run_id} {status}.\n\n```\n{(error or 'no detail')[:3000]}\n```\n\n"
            "Details: `ops.pipeline_run` and the pp-nightly logs in Log Analytics "
            "(ContainerAppConsoleLogs_CL, ContainerJobName_s == 'pp-nightly'). "
            "The next successful run closes this issue."
        )
        number = open_or_update_issue(
            token, self.repo, title=FAILURE_TITLE, body=body, label=ALERT_LABEL, http=self._http
        )
        log.info("reported the failure on issue #%d", number)
