"""The cost-guard workflow must only count PatchPulse's own spending (ADR-017).

The kill switch can only switch off pp-api, so its cost check is scoped to the rg-patchpulse
resource group. Spending elsewhere in the subscription is the $1 budget's job; counting it here
disabled pp-api over another project's NAT gateway on 2026-09-28.
"""

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "cost-guard.yml"


def test_cost_query_is_scoped_to_the_patchpulse_resource_group() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    (url,) = re.findall(r"https://management\.azure\.com/\S*CostManagement/query\S*", text)
    assert "/resourceGroups/$RG/" in url, url
    assert re.search(r"^\s+RG: rg-patchpulse$", text, flags=re.MULTILINE)
