"""Seif's labeling app (design D3): the blind audit, then the blind re-label two weeks later; and
Phase 3's blind relevance audit (design D7).

Run it with:

    uv run --group lab streamlit run src/patchpulse/labeling/app.py

One review at a time: keys 1-0 toggle the aspects, U marks the answer as unsure, Enter saves and
moves on. Every answer is on disk the moment it's saved (`audit.AuditSession`). The app shows game,
language, thumbs and text only: never the split, the patch proximity, or anyone else's labels.

The relevance audit shows one query and one review at a time: keys 0, 1 and 2 pick the grade, U
marks it unsure, Enter saves and moves on (`retrieval.audit.RelevanceAuditSession`). It never shows
the judges' grade, the rank or the system that found the review.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
    review_lookup,
    scroll_hint,
)
from patchpulse.labeling.gold import AUDIT_FILE
from patchpulse.labeling.taxonomy import ASPECTS
from patchpulse.retrieval.audit import (
    AUDIT_PATH,
    GRADES_PATH,
    RelevanceAuditSession,
    pair_lookup,
    read_audit,
)

GUIDELINES = Path("docs/labeling-guidelines.md")
RELEVANCE_GUIDELINES = Path("docs/relevance-guidelines.md")
QUERIES = Path("data/retrieval/queries.csv")
RELABEL_SEED = 20261024
MODES = (
    "Audit (150 reviews)",
    "Re-label (30 reviews, 14+ days later)",
    "Relevance audit (300 pairs)",
)
GRADE_LABELS = {0: "0  not relevant", 1: "1  partly relevant", 2: "2  relevant"}


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


def _relevance_audit() -> None:
    try:
        queue = read_audit(AUDIT_PATH)
        views = pair_lookup(str(get_lab_settings().data_dir), str(QUERIES), str(AUDIT_PATH))
    except (ValueError, FileNotFoundError, KeyError) as error:
        st.sidebar.warning(f"The relevance audit isn't ready: {error}")
        return
    session = RelevanceAuditSession(queue, GRADES_PATH)
    done, total = session.progress()
    st.sidebar.progress(done / total if total else 1.0, text=f"{done} of {total} graded")
    if RELEVANCE_GUIDELINES.exists():
        with st.sidebar.expander("Relevance guidelines", expanded=False):
            st.markdown(RELEVANCE_GUIDELINES.read_text(encoding="utf-8"))

    pair = session.next_item()
    if pair is None:
        st.success("The relevance audit is complete. Thank you. Close the tab whenever you like.")
        return
    view = views[pair]
    st.subheader(view.query)
    edited = f", last edited {view.edited}" if view.edited else ""
    st.caption(f"{view.game}. Review in {view.language}, written {view.date}{edited}.")
    hint = scroll_hint(view.text)
    if hint:
        st.warning(hint)
    st.container(height=230, border=True).markdown(as_plain_text(view.text))

    grade_key, unsure_key = f"grade-{pair[0]}-{pair[1]}", f"unsure-{pair[0]}-{pair[1]}"
    chosen = st.session_state.get(grade_key)
    for column, (grade, label) in zip(st.columns(3), GRADE_LABELS.items(), strict=True):
        if column.button(
            label,
            key=f"{grade_key}-{grade}",
            shortcut=str(grade),
            width="stretch",
            type="primary" if chosen == grade else "secondary",
        ):
            st.session_state[grade_key] = grade
            st.rerun()
    unsure = bool(st.session_state.get(unsure_key, False))
    if st.button(f"{'☑' if unsure else '☐'} unsure", key=f"{unsure_key}-button", shortcut="u"):
        st.session_state[unsure_key] = not unsure
        st.rerun()
    saved = st.button("Save and next", type="primary", shortcut="enter", disabled=chosen is None)
    if saved and chosen is not None:
        session.save(pair, int(chosen), unsure=unsure)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="PatchPulse labeling", layout="centered")
    mode = st.sidebar.radio("Session", MODES)
    if mode == MODES[2]:
        _relevance_audit()
        return
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
    review = review_lookup(str(get_lab_settings().data_dir)).reviews[review_id]
    thumbs = "thumbs up" if review.voted_up else "thumbs down"
    st.subheader(review.game)
    st.caption(f"{review.language}, {thumbs}, review {review_id}")
    # Normal-contrast text in a fixed-height box that scrolls, so the buttons stay on screen.
    hint = scroll_hint(review.text)
    if hint:
        st.warning(hint)
    st.container(height=230, border=True).markdown(as_plain_text(review.text))

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
