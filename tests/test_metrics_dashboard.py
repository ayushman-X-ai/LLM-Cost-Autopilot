"""Guards for the dashboard hero metric: actual vs baseline vs saved."""
import pytest
from autopilot.config import load_models, load_routing
from autopilot.metrics import savings

def test_dashboard_baseline_model_resolves():
    models=load_models("config/models.yaml")
    routing=load_routing("config/routing.yaml")
    name=(routing.get("metrics",{}) or {}).get("baseline_model")
    assert name in models
    # baseline must be the most expensive configured model (maximises honest savings math)
    baseline=models[name]
    for m in models.values():
        per_token=baseline.input_cost_per_1k+baseline.output_cost_per_1k
        other=m.input_cost_per_1k+m.output_cost_per_1k
        assert per_token>=other, f"{name} is not the priciest model ({m.name} costs more)"

def test_hero_metrics_shape():
    models=load_models("config/models.yaml")
    rows=[{"input_tokens":100,"output_tokens":50,"cost_usd":0.0001}]*5
    out=savings(rows,models["openai_high"])
    assert set(out)=={"actual_cost","baseline_cost","savings","savings_pct"}
    assert out["baseline_cost"]>out["actual_cost"]>=0
    assert 0<out["savings_pct"]<=100
