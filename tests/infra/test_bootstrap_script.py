"""Static checks for infra/bootstrap/bootstrap.ps1.

On Windows `az` is az.cmd, which forwards its arguments through cmd.exe inside an IF (...) block.
PowerShell passes an argument without spaces unquoted, so cmd.exe acts on any metacharacter in
it: `--query 'length(@)'` closed the IF block early and az failed with
"-o was unexpected at this time". Count results in PowerShell instead.
"""

import re
from pathlib import Path

BOOTSTRAP = Path(__file__).resolve().parents[2] / "infra" / "bootstrap" / "bootstrap.ps1"
CMD_METACHARACTERS = frozenset("()&|<>^")
_AZ_CALL = re.compile(r"Invoke-Az\s+[a-z]")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def az_calls(script: str) -> list[str]:
    """Each Invoke-Az call as one logical line (PowerShell backtick continuations joined)."""
    logical = script.replace("`\r\n", " ").replace("`\n", " ")
    return [line for line in logical.splitlines() if _AZ_CALL.search(line)]


def unsafe_literals(call: str) -> list[str]:
    """Quoted literals passed to az that contain a cmd.exe metacharacter."""
    match = _AZ_CALL.search(call)
    arguments = call[match.start() :] if match else ""
    return [literal for literal in _QUOTED.findall(arguments) if CMD_METACHARACTERS & set(literal)]


def test_scan_sees_the_scripts_az_calls() -> None:
    assert len(az_calls(BOOTSTRAP.read_text(encoding="utf-8"))) >= 15


def test_az_arguments_contain_no_cmd_metacharacters() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")
    offenders = {
        call.strip(): found for call in az_calls(script) if (found := unsafe_literals(call))
    }
    assert not offenders, f"cmd.exe would act on these az arguments: {offenders}"


def test_scan_catches_the_query_that_broke_bootstrap() -> None:
    (call,) = az_calls("$n = Invoke-Az sql db list-editions `\n    --query 'length(@)' -o tsv")
    assert unsafe_literals(call) == ["'length(@)'"]
