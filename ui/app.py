"""Streamlit Web UI for Smart Lost & Found — bonus feature (+2 pts).

Provides a browser-based interface to every core operation:

* Register a lost item (image upload plus optional free-text description).
* Register a found item.
* Search for matches against a registered item.
* Browse every registered item.
* Generate the AI cost report.

The UI calls the same service-layer functions as the HTTP API and CLI,
so business logic is consistent across every entry point.

Run::

    streamlit run ui/app.py

The same ``.env`` file used by the API is loaded automatically via
``src.config.settings``.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.config import settings
from src.models import Item
from src.services.ai_service import describe_item_async, embed_async
from src.services.cost_meter import get_cost_meter
from src.storage import build_repo


def _run(coro):
    """Execute an async coroutine inside a fresh event loop.

    Streamlit's script runner is synchronous, so each user interaction
    drives a one-shot event loop through this helper.
    """
    return asyncio.run(coro)


st.set_page_config(
    page_title="Smart Lost & Found",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve the images directory relative to the project root so the app works
# regardless of which directory Streamlit was launched from.
_IMAGES_ROOT = (_ROOT / settings.IMAGES_DIR).resolve()


def _image_path(filename: str) -> Path:
    """Return the absolute path to a stored image file."""
    return _IMAGES_ROOT / filename


st.title("Smart Lost & Found")
st.caption("AI-powered lost and found item matching system")

page = st.sidebar.selectbox(
    "Navigate",
    [
        "Home",
        "Register Lost Item",
        "Register Found Item",
        "Search Matches",
        "Browse Items",
        "Cost Report",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Provider:** `{settings.LLM_PROVIDER}` / `{settings.LLM_MODEL}`\n\n"
    f"**Embedder:** `{settings.EMBEDDING_PROVIDER}` / `{settings.EMBEDDING_MODEL}`"
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Demo mode — the database ships pre-seeded with sample lost & found "
    "items. New uploads during this session are stored in an ephemeral "
    "SQLite file and may be wiped when the container restarts."
)


async def _register(
    image_bytes: bytes,
    original_name: str,
    status: str,
    user_text: str,
) -> Item:
    """Run the full registration pipeline for a single image upload.

    Saves the bytes to disk, describes the image with the VLM, embeds
    the description, and persists the resulting :class:`Item`.
    """
    repo = build_repo()
    await repo.initialize()
    try:
        filename = repo.save_image(image_bytes, original_name)
        stored_path = str(_image_path(filename))

        description = await describe_item_async(stored_path, user_text)
        embedding_vec = await embed_async(description.to_search_text())

        item = Item(
            status=status,
            filename=filename,
            original_name=original_name,
            user_text=user_text or None,
            vlm_description=description.to_dict(),
            embedding=embedding_vec.tolist(),
        )
        return await repo.insert_item(item)
    finally:
        await repo.close()


async def _list_items(status: str | None) -> list[Item]:
    """Return every registered item, optionally filtered by status."""
    repo = build_repo()
    await repo.initialize()
    try:
        return await repo.list_items(status)
    finally:
        await repo.close()


async def _search(item_id: int, k: int):
    """Fetch ``item_id`` and its top-k matches, with candidates hydrated.

    Returns:
        A tuple ``(query_item, hydrated_matches)`` where each hydrated
        match is ``(MatchRecord, Item | None)`` so the UI can display
        candidate details inline. Returns ``(None, [])`` when the query
        item does not exist.
    """
    from src.core.matcher import find_matches

    repo = build_repo()
    await repo.initialize()
    try:
        item = await repo.get_item(item_id)
        if item is None:
            return None, []
        matches = await find_matches(item, repo, k=k)
        hydrated = []
        for m in matches:
            candidate = await repo.get_item(m.candidate_id)
            hydrated.append((m, candidate))
        return item, hydrated
    finally:
        await repo.close()


def _registration_form(status: str) -> None:
    """Render the upload form for a lost or found registration page.

    Args:
        status: ``"lost"`` or ``"found"`` — controls the heading text,
            button label, and the status applied to the persisted item.
    """
    label = "lost" if status == "lost" else "found"
    st.header(f"Register a {label.capitalize()} Item")

    uploaded = st.file_uploader(
        "Upload item image (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"upload_{status}",
    )
    user_text = st.text_area(
        "Optional description (color, brand, notes…)",
        height=80,
        key=f"text_{status}",
    )

    if uploaded is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded, caption="Preview", use_column_width=True)

        image_bytes = uploaded.read()
        if len(image_bytes) > settings.MAX_IMAGE_BYTES:
            st.error(
                f"Image is too large ({len(image_bytes):,} bytes). "
                f"Maximum: {settings.MAX_IMAGE_BYTES:,} bytes."
            )
            return

        with col2:
            if st.button(f"Register as {label.capitalize()}", key=f"btn_{status}"):
                with st.spinner("Analysing image with AI…"):
                    try:
                        saved = _run(
                            _register(image_bytes, uploaded.name, status, user_text)
                        )
                        desc = saved.vlm_description or {}
                        st.success(f"Item registered with ID **{saved.id}**")
                        st.json(
                            {
                                "id": saved.id,
                                "status": saved.status,
                                "object_class": desc.get("object_class"),
                                "colors": desc.get("colors", []),
                                "brand": desc.get("brand"),
                                "confidence": desc.get("confidence"),
                            }
                        )
                    except Exception as exc:
                        st.error(f"Registration failed: {exc}")


if page == "Home":
    st.header("Welcome")
    st.markdown(
        """
        **Smart Lost & Found** is an AI-powered demo that matches images of
        *lost* items against images of *found* items using a vision-language
        model and semantic embeddings.

        ### How it works
        1. **Register** a lost or found item by uploading an image.
        2. The image is described by GPT-4o (vision) into structured fields:
           object class, colors, brand, distinguishing marks, confidence.
        3. The description is embedded into a vector and stored in SQLite.
        4. **Search** finds the top matches in the opposite pool by cosine
           similarity between embeddings.

        ### Try the demo
        - Open **Browse Items** to see the pre-seeded sample dataset
          (8 lost items, 7 found items).
        - Open **Search Matches** and try item ID `2` (a navy backpack);
          you should see the matching found backpack at the top.
        - Or upload your own image via **Register Lost Item** /
          **Register Found Item**.

        ### Tech stack
        OpenAI GPT-4o · OpenAI text-embedding-3-small · SQLite (aiosqlite) ·
        Streamlit · Pydantic · Tenacity (retries) · custom async pipeline
        with rate limiting and cost tracking.
        """
    )

    col1, col2, col3 = st.columns(3)
    try:
        all_items = _run(_list_items(None))
        lost_count = sum(1 for it in all_items if it.status == "lost")
        found_count = sum(1 for it in all_items if it.status == "found")
        with col1:
            st.metric("Total items", len(all_items))
        with col2:
            st.metric("Lost items", lost_count)
        with col3:
            st.metric("Found items", found_count)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load item counts: {exc}")

elif page == "Register Lost Item":
    _registration_form("lost")

elif page == "Register Found Item":
    _registration_form("found")

elif page == "Search Matches":
    st.header("Search for Matches")
    st.caption(
        "Enter the ID of a registered item to find the best matches in the opposite pool."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        item_id = st.number_input("Item ID", min_value=1, step=1, value=1)
    with col2:
        k = st.slider("Number of matches (k)", min_value=1, max_value=10, value=3)

    if st.button("Find Matches"):
        with st.spinner("Searching…"):
            try:
                item, matches = _run(_search(int(item_id), k))
                if item is None:
                    st.error(f"No item found with ID {item_id}.")
                elif not matches:
                    opposite = "found" if item.status == "lost" else "lost"
                    st.info(f"No {opposite} items in the database yet.")
                else:
                    desc = item.vlm_description or {}
                    query_img = _image_path(item.filename)
                    head_cols = st.columns([1, 3])
                    with head_cols[0]:
                        if query_img.exists():
                            st.image(str(query_img), caption=f"Query #{item.id}", use_column_width=True)
                    with head_cols[1]:
                        st.markdown(
                            f"**Query:** {item.status} item #{item.id} — "
                            f"`{desc.get('object_class', '?')}` "
                            f"({', '.join(desc.get('colors', []))})"
                        )
                        st.markdown(f"_{(item.user_text or '').strip() or '(no user text)'}_")
                    st.markdown("---")
                    for rank, (match, candidate) in enumerate(matches, start=1):
                        cand_desc = (candidate.vlm_description or {}) if candidate else {}
                        with st.expander(
                            f"#{rank} — ID {match.candidate_id} | "
                            f"Score: {match.score:.4f} | {cand_desc.get('object_class', '?')}",
                            expanded=(rank == 1),
                        ):
                            cols = st.columns([1, 1, 2])
                            with cols[0]:
                                if candidate is not None:
                                    cand_img = _image_path(candidate.filename)
                                    if cand_img.exists():
                                        st.image(str(cand_img), use_column_width=True)
                            with cols[1]:
                                st.markdown(f"**Score:** {match.score:.4f}")
                                st.markdown(f"**Reason:** {match.reason}")
                            with cols[2]:
                                st.json(cand_desc)
            except Exception as exc:
                st.error(f"Search failed: {exc}")

elif page == "Browse Items":
    st.header("Browse Registered Items")

    status_filter = st.radio(
        "Status filter",
        options=["all", "lost", "found"],
        horizontal=True,
    )
    filter_val = None if status_filter == "all" else status_filter

    if st.button("Refresh"):
        st.cache_data.clear()

    view = st.radio(
        "View",
        options=["Grid (with thumbnails)", "Table"],
        horizontal=True,
        index=0,
    )

    try:
        items = _run(_list_items(filter_val))
        st.markdown(f"**{len(items)} item(s)** found.")

        if items and view == "Table":
            rows = []
            for it in items:
                desc = it.vlm_description or {}
                rows.append(
                    {
                        "ID": it.id,
                        "Status": it.status,
                        "Class": desc.get("object_class", "—"),
                        "Colors": ", ".join(desc.get("colors", [])),
                        "Brand": desc.get("brand") or "—",
                        "Original Name": it.original_name or "—",
                        "Registered": str(it.created_at)[:19] if it.created_at else "—",
                    }
                )
            st.dataframe(rows, use_container_width=True)

        elif items and view == "Grid (with thumbnails)":
            cols_per_row = 4
            for row_start in range(0, len(items), cols_per_row):
                row_items = items[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, it in zip(cols, row_items):
                    desc = it.vlm_description or {}
                    img_path = _image_path(it.filename)
                    with col:
                        if img_path.exists():
                            st.image(str(img_path), use_column_width=True)
                        else:
                            st.info("(image missing)")
                        st.markdown(
                            f"**#{it.id}** · `{it.status}`  \n"
                            f"{desc.get('object_class', '?')}  \n"
                            f"_{', '.join(desc.get('colors', []) or ['—'])}_"
                        )
    except Exception as exc:
        st.error(f"Could not load items: {exc}")

elif page == "Cost Report":
    st.header("AI Cost Report")
    st.caption("Estimated spend based on token counts logged to the JSONL cost file.")

    since_hours = st.number_input(
        "Show last N hours (0 = all time)",
        min_value=0,
        step=1,
        value=0,
    )

    if st.button("Generate Report"):
        try:
            since = timedelta(hours=since_hours) if since_hours > 0 else None
            report = get_cost_meter().report(since=since)
            st.code(report, language="text")
        except Exception as exc:
            st.error(f"Could not generate report: {exc}")
