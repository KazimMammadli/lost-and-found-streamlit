"""Command-line interface for the Smart Lost & Found service.

Commands
--------
* ``register-lost IMAGE [--text TEXT]`` — register a lost item.
* ``register-found IMAGE [--text TEXT]`` — register a found item.
* ``search-matches --id ID [--k N]`` — find top-k matches for an item.
* ``list [--status lost|found|all]`` — list registered items.
* ``cost-report [--since HOURS]`` — print AI cost summary.

Design notes
------------
* Every command shares its business logic with the HTTP API via the
  ``src/services`` and ``src/storage`` modules, so behaviour is identical
  from both entry points.
* :func:`asyncio.run` bridges the sync CLI boundary into the async
  service layer.
* ``print`` is used exclusively for user-facing terminal output; all
  diagnostics use :mod:`logging`.
* File errors, missing items, and AI failures are reported as clean
  error messages, never as raw tracebacks.

Examples
--------
::

    python -m src.cli register-lost path/to/image.jpg --text "navy backpack"
    python -m src.cli register-found path/to/image.png
    python -m src.cli search-matches --id 1 --k 5
    python -m src.cli list --status lost
    python -m src.cli cost-report --since 24
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Literal

from src.config import settings
from src.core.matcher import find_matches
from src.models import Item
from src.services.ai_service import describe_item_async, embed_async
from src.storage import build_repo

log = logging.getLogger(__name__)


async def _do_register(
    image_path: Path, user_text: str | None, status: Literal["lost", "found"]
) -> None:
    """Run the full registration pipeline for a single image.

    Args:
        image_path: Local path to the image file.
        user_text: Optional free-text description from the user.
        status: ``"lost"`` or ``"found"``.

    Exits the process with status 1 on any validation or pipeline failure.
    """
    if not image_path.exists():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    suffix = image_path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        print(
            f"Error: unsupported file type '{suffix}'. "
            "Only .png and .jpeg are accepted.",
            file=sys.stderr,
        )
        sys.exit(1)

    data = image_path.read_bytes()
    if len(data) > settings.MAX_IMAGE_BYTES:
        print(
            f"Error: file too large ({len(data)} bytes). "
            f"Maximum is {settings.MAX_IMAGE_BYTES} bytes.",
            file=sys.stderr,
        )
        sys.exit(1)

    repo = build_repo()
    try:
        await repo.initialize()

        filename = repo.save_image(data, image_path.name)
        stored_path = str(Path(settings.IMAGES_DIR) / filename)

        print("Describing image with VLM…")
        description = await describe_item_async(stored_path, user_text or "")

        print("Generating embedding…")
        embedding_vec = await embed_async(description.to_search_text())

        item = Item(
            status=status,
            filename=filename,
            original_name=image_path.name,
            user_text=user_text,
            vlm_description=description.to_dict(),
            embedding=embedding_vec.tolist(),
        )
        saved = await repo.insert_item(item)

        print(f"\nRegistered {status} item:")
        print(f"  ID:       {saved.id}")
        print(f"  Class:    {description.object_class}")
        print(f"  Colors:   {', '.join(description.colors)}")
        if description.brand:
            print(f"  Brand:    {description.brand}")
        print(f"  Stored:   {filename}")

        log.info("cli.register.ok", extra={"id": saved.id, "status": status})

    except Exception as exc:
        log.error("cli.register.failed", extra={"error": str(exc)})
        print(f"Error during registration: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        await repo.close()


async def _do_search_matches(item_id: int, k: int) -> None:
    """Print the top-k matches for the given item ID.

    Args:
        item_id: Database ID of the query item.
        k: Number of matches to return.
    """
    repo = build_repo()
    try:
        await repo.initialize()

        item = await repo.get_item(item_id)
        if item is None:
            print(f"Error: no item with id={item_id}.", file=sys.stderr)
            sys.exit(1)

        if item.embedding is None:
            print(
                f"Error: item {item_id} has no embedding. Re-register the item.",
                file=sys.stderr,
            )
            sys.exit(1)

        records = await find_matches(item, repo, k=k)

        opposite = "found" if item.status == "lost" else "lost"
        print(f"\nTop-{k} {opposite} matches for {item.status} item {item_id}:")

        if not records:
            print("  (no matches found — register some items in the opposite pool)")
            return

        for i, rec in enumerate(records, start=1):
            print(f"  {i}. ID={rec.candidate_id}  score={rec.score:.4f}  {rec.reason}")

        log.info("cli.search.ok", extra={"query_id": item_id, "count": len(records)})

    except Exception as exc:
        log.error("cli.search.failed", extra={"error": str(exc)})
        print(f"Error during search: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        await repo.close()


def _do_cost_report(since_hours: int | None) -> None:
    """Print the cost summary from the JSONL cost log.

    Args:
        since_hours: When provided, restricts the report to the last
            ``since_hours`` hours.
    """
    from src.services.cost_meter import get_cost_meter

    since = timedelta(hours=since_hours) if since_hours is not None else None
    print(get_cost_meter().report(since=since))


async def _do_list(status: str | None) -> None:
    """Print a table of registered items.

    Args:
        status: Optional status filter (``lost`` / ``found`` / ``all``).
            ``None`` is equivalent to ``all``.
    """
    repo = build_repo()
    try:
        await repo.initialize()
        items = await repo.list_items(status)

        label = status or "all"
        print(f"\nRegistered items ({label}): {len(items)} total")

        if not items:
            print("  (none)")
            return

        print(f"  {'ID':<6} {'Status':<8} {'Class':<20} {'Original Name'}")
        print("  " + "-" * 60)
        for item in items:
            obj_class = (
                item.vlm_description.get("object_class", "—")
                if item.vlm_description
                else "—"
            )
            print(
                f"  {item.id!s:<6} {item.status:<8} {obj_class:<20} "
                f"{item.original_name or '—'}"
            )

    except Exception as exc:
        log.error("cli.list.failed", extra={"error": str(exc)})
        print(f"Error listing items: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        await repo.close()


def register_lost(args: argparse.Namespace) -> None:
    """``register-lost`` entry point: invoke :func:`_do_register` with ``lost``."""
    asyncio.run(_do_register(Path(args.image), args.text, "lost"))


def register_found(args: argparse.Namespace) -> None:
    """``register-found`` entry point: invoke :func:`_do_register` with ``found``."""
    asyncio.run(_do_register(Path(args.image), args.text, "found"))


def search_matches(args: argparse.Namespace) -> None:
    """``search-matches`` entry point."""
    asyncio.run(_do_search_matches(args.id, args.k))


def list_items(args: argparse.Namespace) -> None:
    """``list`` entry point."""
    asyncio.run(_do_list(args.status))


def cost_report(args: argparse.Namespace) -> None:
    """``cost-report`` entry point."""
    _do_cost_report(args.since)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with every supported subcommand."""
    parser = argparse.ArgumentParser(
        prog="lostfound",
        description="Smart Lost & Found CLI",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.required = True

    p_lost = sub.add_parser("register-lost", help="Register a lost item")
    p_lost.add_argument("image", help="Path to the image file (JPEG or PNG)")
    p_lost.add_argument("--text", default=None, help="Optional free-text description")

    p_found = sub.add_parser("register-found", help="Register a found item")
    p_found.add_argument("image", help="Path to the image file (JPEG or PNG)")
    p_found.add_argument("--text", default=None, help="Optional free-text description")

    p_search = sub.add_parser("search-matches", help="Find top-k matches for an item")
    p_search.add_argument("--id", type=int, required=True, help="Item ID to find matches for")
    p_search.add_argument(
        "--k", type=int, default=3, help="Number of matches to return (default: 3)"
    )

    p_list = sub.add_parser("list", help="List registered items")
    p_list.add_argument(
        "--status",
        choices=["lost", "found", "all"],
        default=None,
        help="Filter by status (default: all)",
    )

    p_cost = sub.add_parser("cost-report", help="Print AI cost summary from the JSONL log")
    p_cost.add_argument(
        "--since",
        type=int,
        default=None,
        metavar="HOURS",
        help="Limit report to last N hours (default: all time)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional explicit argv (used by tests). When ``None``,
            argparse reads from ``sys.argv``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "register-lost": register_lost,
        "register-found": register_found,
        "search-matches": search_matches,
        "list": list_items,
        "cost-report": cost_report,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
