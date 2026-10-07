"""Seif's labeling app (design D3): the blind audit, then the blind re-label two weeks later.

Run it with:

    uv run --group lab streamlit run src/patchpulse/labeling/app.py

One review at a time: keys 1-0 toggle the aspects, U marks the answer as unsure, Enter saves and
moves on. Every answer is on disk the moment it's saved (`audit.AuditSession`). The app shows game,
language, thumbs and text only: never the split, the patch proximity, or anyone else's labels.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from patchpulse.config import get_lab_settings
from patchpulse.labeling.audit import (
    NOTES_FILE,
    SEIF_LABELS_FILE,
    SEIF_RELABEL_FILE,
    AuditSession,
    as_plain_text,
    relabel_selection,
)
from patchpulse.labeling.gold import AUDIT_FILE
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.models.snapshot import load_snapshot

GUIDELINES = Path("docs/labeling-guidelines.md")
RELABEL_SEED = 20261024
MODES = ("Audit (150 reviews)", "Re-label (30 reviews, 14+ days later)")


@st.cache_data
def _reviews(data_dir: str) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    snapshot = load_snapshot(Path(data_dir))
    columns = ["review_id", "appid", "language", "voted_up", "text"]
    reviews = {
        int(review_id): {
            "appid": int(appid),
            "language": str(language),
            "voted_up": bool(up),
            "text": str(text),
        }
        for review_id, appid, language, up, text in snapshot.reviews[columns].itertuples(
            index=False
        )
    }
    names = {
        int(a): str(n) for a, n in zip(snapshot.games["appid"], snapshot.games["name"], strict=True)
    }
    return reviews, names


def _session(mode: str) -> AuditSession:
    settings = get_lab_settings()
    notes = settings.data_dir / NOTES_FILE
    if mode == MODES[0]:
        queue = [int(i) for i in pd.read_csv(settings.gold_dir / AUDIT_FILE)["review_id"]]
        return AuditSession(queue, settings.gold_dir / SEIF_LABELS_FILE, notes_path=notes)
    queue = relabel_selection(
        settings.gold_dir / SEIF_LABELS_FILE, today=datetime.now(UTC).date(), seed=RELABEL_SEED
    )
    return AuditSession(queue, settings.gold_dir / SEIF_RELABEL_FILE, notes_path=notes)


def main() -> None:
    st.set_page_config(page_title="PatchPulse labeling", layout="centered")
    mode = st.sidebar.radio("Session", MODES)
    try:
        session = _session(mode)
    except (ValueError, FileNotFoundError) as error:
        st.sidebar.warning(str(error))
        return
    done, total = session.progress()
    st.sidebar.progress(done / total if total else 1.0, text=f"{done} of {total} labeled")
    if GUIDELINES.exists():
        with st.sidebar.expander("Guidelines", expanded=False):
            st.markdown(GUIDELINES.read_text(encoding="utf-8"))

    review_id = session.next_item()
    if review_id is None:
        st.success("This session is complete. Thank you. Close the tab whenever you like.")
        return
    reviews, names = _reviews(str(get_lab_settings().data_dir))
    review = reviews[review_id]
    thumbs = "thumbs up" if review["voted_up"] else "thumbs down"
    st.subheader(names.get(int(review["appid"]), str(review["appid"])))
    st.caption(f"{review['language']}, {thumbs}, review {review_id}")
    # Normal-contrast text in a fixed-height box that scrolls, so the buttons stay on screen.
    st.container(height=230, border=True).markdown(as_plain_text(str(review["text"])))

    selected: set[str] = st.session_state.setdefault(f"selected-{review_id}", set())
    columns = st.columns(2)
    for position, code in enumerate(ASPECTS, start=1):
        key = str(position % 10)
        mark = "☑" if code in selected else "☐"
        if columns[(position - 1) % 2].button(
            f"{mark} {code}",
            key=f"{review_id}-{code}",
            shortcut=key,
            width="stretch",
            type="primary" if code in selected else "secondary",
        ):
            selected ^= {code}
            st.rerun()
    unsure_key = f"unsure-{review_id}"
    unsure = bool(st.session_state.get(unsure_key, False))
    if st.button(f"{'☑' if unsure else '☐'} unsure", key=f"{unsure_key}-button", shortcut="u"):
        st.session_state[unsure_key] = not unsure
        st.rerun()
    note = st.text_input("Note (stays on this machine)", key=f"note-{review_id}")
    st.caption("Nothing ticked saves as `none`.")
    if st.button("Save and next", type="primary", shortcut="enter"):
        session.save(review_id, frozenset(selected), unsure=unsure, note=note)
        st.rerun()


main()
