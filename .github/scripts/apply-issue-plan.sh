#!/usr/bin/env bash
# Apply an issue plan (Phase 7): the lines patchpulse.monitoring.alerts.IssuePlan writes to
# DIR/issues.tsv, one per action - action, label, title, body file - with gh and the workflow's
# GITHUB_TOKEN (GH_TOKEN and GH_REPO in the environment).
# - alert: open the issue unless one with that title and label is already open. An open one is
#   left as it is: cost-guard runs every 15 minutes and would otherwise comment each time.
# - resolve: comment on the open issue and close it; nothing if none is open.
set -euo pipefail

dir="$1"
plan="$dir/issues.tsv"
if [ ! -s "$plan" ]; then
  echo "No issue to open or close."
  exit 0
fi

while IFS=$'\t' read -r action label title body; do
  number=$(gh issue list --label "$label" --state open --limit 100 --json number,title \
    --jq "[.[] | select(.title == \"$title\")][0].number // empty")
  case "$action" in
    alert)
      if [ -n "$number" ]; then
        echo "\"$title\" is already open (#$number); leaving it."
        continue
      fi
      gh label create "$label" --color B60205 --description "Automated alert (ADR-013)" --force
      gh issue create --title "$title" --label "$label" --body-file "$dir/$body"
      ;;
    resolve)
      if [ -z "$number" ]; then
        continue
      fi
      gh issue comment "$number" --body-file "$dir/$body"
      gh issue close "$number"
      echo "closed #$number ($title)"
      ;;
    *)
      echo "::error::unknown issue action: $action"
      exit 1
      ;;
  esac
done < "$plan"
