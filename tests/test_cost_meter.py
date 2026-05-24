"""Tests for the cost telemetry module (src/services/cost_meter.py).

Validates pricing lookups, JSONL persistence, and the report formatter.
"""

from __future__ import annotations

import json
from datetime import timedelta

from src.services.cost_meter import PRICING_USD_PER_1K, CostMeter, estimate_cost


def test_estimate_cost_known_provider():
    cost = estimate_cost("anthropic", "claude-sonnet-4-6", 1000, 500)
    expected = (1000 / 1000) * 0.003 + (500 / 1000) * 0.015
    assert abs(cost - expected) < 1e-9


def test_estimate_cost_unknown_provider_returns_zero():
    cost = estimate_cost("unknown_provider", "unknown_model", 1000, 1000)
    assert cost == 0.0


def test_estimate_cost_zero_tokens():
    cost = estimate_cost("openai", "gpt-4o-mini", 0, 0)
    assert cost == 0.0


def test_estimate_cost_embedding_no_completion():
    cost = estimate_cost("openai", "text-embedding-3-small", 1000, 0)
    assert cost > 0.0
    cost_with_completion = estimate_cost("openai", "text-embedding-3-small", 1000, 500)
    assert cost == cost_with_completion


def test_estimate_cost_case_insensitive():
    cost_lower = estimate_cost("anthropic", "claude-sonnet-4-6", 100, 50)
    cost_upper = estimate_cost("ANTHROPIC", "CLAUDE-SONNET-4-6", 100, 50)
    assert abs(cost_lower - cost_upper) < 1e-9


def test_all_pricing_keys_are_tuples():
    for key, (p_rate, c_rate) in PRICING_USD_PER_1K.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert p_rate >= 0.0
        assert c_rate >= 0.0


def test_record_creates_file(tmp_path):
    meter = CostMeter(str(tmp_path / "cost.jsonl"))
    meter.record("anthropic", "claude-sonnet-4-6", 100, 50)
    assert (tmp_path / "cost.jsonl").exists()


def test_record_appends_valid_jsonl(tmp_path):
    meter = CostMeter(str(tmp_path / "cost.jsonl"))
    meter.record("openai", "gpt-4o-mini", 200, 100)
    meter.record("openai", "gpt-4o-mini", 300, 150)

    lines = (tmp_path / "cost.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "provider" in obj
        assert "model" in obj
        assert "prompt_tokens" in obj
        assert "completion_tokens" in obj
        assert "dollars" in obj
        assert "timestamp" in obj


def test_record_calculates_cost_correctly(tmp_path):
    meter = CostMeter(str(tmp_path / "cost.jsonl"))
    meter.record("anthropic", "claude-sonnet-4-6", 1000, 500)

    line = (tmp_path / "cost.jsonl").read_text().strip()
    obj = json.loads(line)
    expected = estimate_cost("anthropic", "claude-sonnet-4-6", 1000, 500)
    assert abs(obj["dollars"] - expected) < 1e-9


def test_record_creates_parent_directories(tmp_path):
    nested = tmp_path / "deep" / "nested" / "cost.jsonl"
    meter = CostMeter(str(nested))
    meter.record("gemini", "gemini-2.0-flash", 50, 25)
    assert nested.exists()


def test_report_empty_returns_message(tmp_path):
    meter = CostMeter(str(tmp_path / "cost.jsonl"))
    report = meter.report()
    assert "No cost events" in report


def test_report_shows_total_spend(tmp_path):
    meter = CostMeter(str(tmp_path / "cost.jsonl"))
    meter.record("anthropic", "claude-sonnet-4-6", 1000, 500)
    meter.record("openai", "gpt-4o-mini", 500, 250)

    report = meter.report()
    assert "Total spend" in report
    assert "$" in report


def test_report_shows_provider_breakdown(tmp_path):
    meter = CostMeter(str(tmp_path / "cost.jsonl"))
    meter.record("anthropic", "claude-sonnet-4-6", 1000, 500)
    meter.record("openai", "gpt-4o-mini", 500, 250)

    report = meter.report()
    assert "anthropic" in report
    assert "openai" in report


def test_report_shows_top5(tmp_path):
    meter = CostMeter(str(tmp_path / "cost.jsonl"))
    for i in range(7):
        meter.record("anthropic", "claude-sonnet-4-6", 1000 * (i + 1), 500)

    report = meter.report()
    assert "Top 5" in report


def test_report_since_filters_old_events(tmp_path):
    """Events with very old timestamps are excluded by the since filter."""
    log_path = tmp_path / "cost.jsonl"
    meter = CostMeter(str(log_path))

    meter.record("anthropic", "claude-sonnet-4-6", 100, 50)

    old_event = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "prompt_tokens": 999,
        "completion_tokens": 999,
        "dollars": 99.99,
        "timestamp": "2000-01-01T00:00:00+00:00",
    }
    with log_path.open("a") as f:
        f.write(json.dumps(old_event) + "\n")

    report = meter.report(since=timedelta(hours=1))
    assert "1 calls" in report


def test_report_token_counts(tmp_path):
    meter = CostMeter(str(tmp_path / "cost.jsonl"))
    meter.record("anthropic", "claude-sonnet-4-6", 300, 150)

    report = meter.report()
    assert "300" in report
    assert "150" in report
