"""Cost telemetry — bonus feature (+2 pts).

Every successful AI call appends a single JSON line to ``COST_LOG_PATH``
containing the provider, model, token counts, an estimated dollar cost,
and the UTC timestamp. The CLI's ``cost-report`` command reads that file
back and prints an aggregated summary.

Typical usage from the service layer::

    from src.services.cost_meter import get_cost_meter

    get_cost_meter().record(
        provider="openai",
        model="gpt-4o",
        prompt_tokens=320,
        completion_tokens=80,
    )

Typical usage from the CLI::

    python -m src.cli cost-report --since 24
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

PRICING_USD_PER_1K: dict[tuple[str, str], tuple[float, float]] = {
    ("anthropic", "claude-sonnet-4-6"):      (0.003,    0.015),
    ("anthropic", "claude-opus-4-7"):        (0.015,    0.075),
    ("openai",    "gpt-4o"):                 (0.005,    0.015),
    ("openai",    "gpt-4o-mini"):            (0.00015,  0.0006),
    ("openai",    "text-embedding-3-small"): (0.00002,  0.0),
    ("openai",    "text-embedding-3-large"): (0.00013,  0.0),
    ("gemini",    "gemini-2.0-flash"):       (0.000075, 0.0003),
    ("gemini",    "text-embedding-004"):     (0.000025, 0.0),
}
"""Published pricing as of May 2026, USD per 1 000 tokens.

Each entry maps ``(provider, model)`` (lower-cased) to a
``(prompt_per_1k, completion_per_1k)`` tuple. Embedding models charge
only for input tokens, so their completion rate is zero.
"""


def estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate the dollar cost of a single AI call.

    Args:
        provider: Provider name (``anthropic``, ``openai``, ``gemini``).
        model: Model identifier as used in the provider's API.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens (use 0 for embeddings).

    Returns:
        Estimated cost in USD. Returns ``0.0`` if the
        ``(provider, model)`` pair is not in :data:`PRICING_USD_PER_1K`.
    """
    key = (provider.lower(), model.lower())
    p_rate, c_rate = PRICING_USD_PER_1K.get(key, (0.0, 0.0))
    return (prompt_tokens / 1000) * p_rate + (completion_tokens / 1000) * c_rate


@dataclass
class CostEvent:
    """A single AI call recorded for cost telemetry.

    Attributes:
        provider: Provider name.
        model: Model identifier.
        prompt_tokens: Input token count.
        completion_tokens: Output token count (0 for embeddings).
        dollars: Estimated cost in USD.
        timestamp: ISO-8601 UTC timestamp at the moment of recording.
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    dollars: float
    timestamp: str


class CostMeter:
    """Append-only JSONL cost log with a human-readable report formatter.

    Safe for single-process use without explicit locking: each :meth:`record`
    call opens, appends, and closes the file, so the OS append guarantee
    keeps lines intact. Concurrent multi-process writers would need
    additional locking.
    """

    def __init__(self, log_path: str) -> None:
        """Bind the meter to ``log_path``.

        The parent directory is not created here — it is created on first
        write so an unused meter never touches the filesystem.
        """
        self._path = Path(log_path)

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Append one cost event to the JSONL log file.

        Args:
            provider: Provider name (will be passed through to
                :func:`estimate_cost`).
            model: Model identifier.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
        """
        dollars = estimate_cost(provider, model, prompt_tokens, completion_tokens)
        event = CostEvent(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            dollars=dollars,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")

        log.debug(
            "cost_meter.recorded",
            extra={"provider": provider, "model": model, "dollars": f"{dollars:.6f}"},
        )

    def _read_events(self, since: timedelta | None = None) -> Iterator[CostEvent]:
        """Stream cost events from the log file, optionally filtered by time.

        Args:
            since: When provided, only events whose timestamp is more
                recent than ``now - since`` are yielded.

        Yields:
            :class:`CostEvent` instances in file order. Lines that fail
            to parse are logged at ``WARNING`` and skipped.
        """
        if not self._path.exists():
            return
        cutoff = (
            (datetime.now(UTC) - since).isoformat() if since else None
        )
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if cutoff and d.get("timestamp", "") < cutoff:
                        continue
                    yield CostEvent(**d)
                except (json.JSONDecodeError, TypeError):
                    log.warning("cost_meter.bad_line", extra={"line": line[:80]})

    def report(self, since: timedelta | None = None) -> str:
        """Return a human-readable summary of recorded cost events.

        Args:
            since: When provided, only events from the last ``since``
                window contribute to the summary.

        Returns:
            A multi-line summary including total spend, token counts,
            per-provider breakdown, and the five most expensive calls.
            Returns a friendly fallback when no events have been recorded.
        """
        events = list(self._read_events(since))
        if not events:
            return "No cost events recorded yet."

        total_dollars = sum(e.dollars for e in events)
        total_prompt = sum(e.prompt_tokens for e in events)
        total_completion = sum(e.completion_tokens for e in events)

        by_provider: dict[str, float] = defaultdict(float)
        for e in events:
            by_provider[e.provider] += e.dollars

        lines = [
            "",
            f"Cost report ({len(events)} calls"
            + (f", last {since}" if since else "")
            + ")",
            f"  Total spend:      ${total_dollars:.4f}",
            f"  Prompt tokens:    {total_prompt:,}",
            f"  Completion tokens:{total_completion:,}",
            "",
            "  By provider:",
        ]
        for provider, cost in sorted(by_provider.items(), key=lambda x: -x[1]):
            lines.append(f"    {provider:<20} ${cost:.4f}")

        top5 = sorted(events, key=lambda e: -e.dollars)[:5]
        lines += ["", "  Top 5 most expensive calls:"]
        for e in top5:
            lines.append(
                f"    {e.provider}/{e.model} - ${e.dollars:.6f} "
                f"({e.prompt_tokens}+{e.completion_tokens} tokens) @ {e.timestamp[:19]}"
            )

        return "\n".join(lines) + "\n"


_meter: CostMeter | None = None
"""Lazily-constructed module-level singleton; access via :func:`get_cost_meter`."""


def _make_meter() -> CostMeter:
    """Build a :class:`CostMeter` bound to the configured ``COST_LOG_PATH``.

    Imports settings inside the function so this module can be imported
    safely even before configuration is fully resolved.
    """
    from src.config import settings

    return CostMeter(settings.COST_LOG_PATH)


def get_cost_meter() -> CostMeter:
    """Return the process-wide :class:`CostMeter`, constructing it on first use."""
    global _meter
    if _meter is None:
        _meter = _make_meter()
    return _meter
