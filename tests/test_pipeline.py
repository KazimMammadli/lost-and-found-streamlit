"""Tests for src/concurrency/pipeline.py.

Validates that ``run_with_semaphore``:
- runs all tasks and returns results in order
- returns exceptions as values rather than raising (graceful degradation)
- bounds actual concurrency to ``SEMAPHORE_LIMIT``
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from src.concurrency.pipeline import run_with_semaphore


async def test_runs_all_tasks():
    async def _coro(n):
        return n

    factories = [lambda n=n: _coro(n) for n in range(5)]
    results = await run_with_semaphore(factories)
    assert sorted(results) == [0, 1, 2, 3, 4]


async def test_returns_exceptions_as_values():
    """A failing task should not abort the entire gather."""

    async def _ok():
        return "success"

    async def _fail():
        raise ValueError("oops")

    factories = [lambda: _ok(), lambda: _fail(), lambda: _ok()]
    results = await run_with_semaphore(factories)

    assert results[0] == "success"
    assert isinstance(results[1], ValueError)
    assert results[2] == "success"


async def test_all_tasks_can_fail_without_raising():
    async def _fail():
        raise RuntimeError("boom")

    factories = [lambda: _fail() for _ in range(3)]
    results = await run_with_semaphore(factories)
    assert all(isinstance(r, RuntimeError) for r in results)


async def test_empty_input_returns_empty_list():
    results = await run_with_semaphore([])
    assert results == []


async def test_single_task():
    async def _one():
        return 42

    results = await run_with_semaphore([lambda: _one()])
    assert results == [42]


async def test_semaphore_limits_concurrency():
    """Verify that at most SEMAPHORE_LIMIT tasks run at the same time."""
    max_concurrent = 0
    current = 0

    async def _task():
        nonlocal max_concurrent, current
        current += 1
        max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)
        current -= 1
        return True

    with patch("src.concurrency.pipeline.settings") as mock_settings:
        mock_settings.SEMAPHORE_LIMIT = 3
        factories = [lambda: _task() for _ in range(20)]
        await run_with_semaphore(factories)

    assert max_concurrent <= 3


async def test_results_preserve_task_count():
    async def _coro(n):
        return n * 2

    n = 8
    factories = [lambda i=i: _coro(i) for i in range(n)]
    results = await run_with_semaphore(factories)
    assert len(results) == n


async def test_tasks_run_concurrently_not_sequentially():
    """Wall-clock time for N concurrent tasks should be < N × per-task time."""
    import time

    async def _slow():
        await asyncio.sleep(0.05)
        return 1

    n = 10
    with patch("src.concurrency.pipeline.settings") as mock_settings:
        mock_settings.SEMAPHORE_LIMIT = n
        start = time.perf_counter()
        results = await run_with_semaphore([lambda: _slow() for _ in range(n)])
        elapsed = time.perf_counter() - start

    assert all(r == 1 for r in results)
    assert elapsed < 0.3, f"Expected concurrent execution, but took {elapsed:.2f}s"
