"""What every teacher provider shares: an answer, and the two ways a request can fail (ADR-020).

The teacher (`teacher.py`) depends only on these, so a provider is a module that answers with
`LLMResponse` and raises one of the two errors: `gemini.py` for Google's free tier and
`openai_compat.py` for a self-hosted, OpenAI-style server.
"""

from __future__ import annotations

from dataclasses import dataclass

# Every provider retries the same way: 6 attempts, 275 s of waiting in all, since overloads
# usually pass by then; what still fails raises TeacherUnavailableError.
MAX_ATTEMPTS = 6
BACKOFF_S = (5.0, 15.0, 45.0, 90.0, 120.0)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int  # the answer plus any thinking: both count against limits
    finish_reason: str


class TeacherRequestError(RuntimeError):
    """The provider refused this request (4xx) or blocked it; `status` is the HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


class TeacherUnavailableError(RuntimeError):
    """The provider kept failing (5xx, rate limits, network) through every retry: try again later.

    Not a `TeacherRequestError`, so an outage is never mistaken for a problem with the request.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message
