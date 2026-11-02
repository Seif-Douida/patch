"""`config/causal.yaml`: the default method the benchmark chose (ADR-007's rule), committed with
the report it came from, and whether a method met the rule (`validated`; with none the default is
interrupted time series and the site says its intervals weren't validated). Until the real
benchmark has run, the default is interrupted time series with no report hash, which reads as
"not yet benchmarked"."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CAUSAL_CONFIG_PATH = Path("config/causal.yaml")


@dataclass(frozen=True)
class CausalConfig:
    default_method: str
    effects_version: str
    report_sha256: str | None
    validated: bool  # a method met ADR-007's rule; False until the benchmark has run
    # Phase 5's V4: causal wording on volume-spike windows (the benchmark's spike verdict).
    spike_causal_ok: bool = False

    @property
    def benchmarked(self) -> bool:
        return self.report_sha256 is not None


def read_causal_config(path: Path = CAUSAL_CONFIG_PATH) -> CausalConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    method = str(data["default_method"])
    if method not in {"its", "synth"}:
        raise ValueError(f"{path}: default_method must be its or synth, not {method!r}")
    sha = data.get("report_sha256")
    return CausalConfig(
        method,
        str(data["effects_version"]),
        str(sha) if sha else None,
        validated=bool(data.get("validated", False)),
        spike_causal_ok=bool(data.get("spike_causal_ok", False)),
    )


def write_causal_config(config: CausalConfig, path: Path = CAUSAL_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# The causal default method (ADR-007), set by `patchpulse-ml causal benchmark`.\n"
        + yaml.safe_dump(
            {
                "default_method": config.default_method,
                "effects_version": config.effects_version,
                "report_sha256": config.report_sha256,
                "validated": config.validated,
                "spike_causal_ok": config.spike_causal_ok,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
