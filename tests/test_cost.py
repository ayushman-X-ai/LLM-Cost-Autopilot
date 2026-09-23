import pytest
from autopilot.config import load_models
from autopilot.metrics import baseline_cost, savings

def row(inp, out, cost): return {"input_tokens": inp, "output_tokens": out, "cost_usd": cost}

def test_baseline_cost_uses_premium_pricing():
    models = load_models("config/models.yaml")
    base = baseline_cost([row(1000, 1000, 0.0)], models["openai_high"])
    assert base == pytest.approx(models["openai_high"].estimate_cost(1000, 1000))
    assert base > 0

def test_savings_numbers():
    models = load_models("config/models.yaml")
    rows = [row(1000, 1000, 0.001), row(2000, 1000, 0.002)]
    out = savings(rows, models["openai_high"])
    expected_base = models["openai_high"].estimate_cost(3000, 2000)
    assert out["actual_cost"] == pytest.approx(0.003)
    assert out["baseline_cost"] == pytest.approx(expected_base)
    assert out["savings"] == pytest.approx(expected_base - 0.003)
    assert out["savings_pct"] == pytest.approx((expected_base - 0.003) / expected_base * 100)

def test_savings_zero_when_free_local_baseline():
    models = load_models("config/models.yaml")
    out = savings([row(1000, 1000, 0.0)], models["ollama_llama"])
    assert out["baseline_cost"] == 0 and out["savings_pct"] == 0  # no division by zero

def test_savings_empty_rows():
    models = load_models("config/models.yaml")
    out = savings([], models["openai_high"])
    assert out == {"actual_cost": 0, "baseline_cost": 0, "savings": 0, "savings_pct": 0}

def test_local_routing_beats_premium_baseline():
    """The headline claim: local tier_1 routing costs less than the premium baseline."""
    models = load_models("config/models.yaml")
    rows = [row(500, 300, 0.0) for _ in range(10)]  # 10 free ollama requests
    out = savings(rows, models["openai_high"])
    assert out["savings"] > 0 and out["savings_pct"] == pytest.approx(100.0)
