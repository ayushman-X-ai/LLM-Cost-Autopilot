import pytest
from autopilot.config import load_models, load_routing, ModelConfig

def test_configs():
    models=load_models("config/models.yaml"); routing=load_routing("config/routing.yaml")
    assert models and "tiers" in routing
    assert routing["tiers"]["tier_1"]["model"] in models

def test_all_tiers_have_models_and_fallbacks():
    routing=load_routing("config/routing.yaml"); models=load_models("config/models.yaml")
    for tier,cfg in routing["tiers"].items():
        assert cfg["model"] in models, f"{tier} primary missing"
        assert cfg.get("fallbacks"), f"{tier} needs a fallback chain"
        for fb in cfg["fallbacks"]:
            assert fb in models, f"{tier} fallback {fb} missing"
        assert cfg["quality_threshold"] > 0

def test_cloud_models_have_real_pricing():
    models=load_models("config/models.yaml")
    for name,m in models.items():
        assert m.pricing_updated_at, f"{name} missing pricing_updated_at"
        if m.provider == "ollama":
            assert m.input_cost_per_1k == 0 and m.output_cost_per_1k == 0
        else:
            assert m.input_cost_per_1k > 0 and m.output_cost_per_1k > 0, f"{name} still free"

def test_premium_costs_more_than_medium():
    models=load_models("config/models.yaml")
    assert models["openai_high"].input_cost_per_1k > models["openai_mini"].input_cost_per_1k
    assert models["anthropic_sonnet"].output_cost_per_1k > models["anthropic_haiku"].output_cost_per_1k

def test_verification_config_shape():
    routing=load_routing("config/routing.yaml")
    v=routing["verification"]
    assert v["default_mode"] in ("fast","quality")
    weights=v["weights"]
    assert abs(sum(weights.values())-1.0) < 1e-9
    assert set(weights) == {"correctness","relevance","completeness","instruction_following"}
    assert v["escalation_models"], "need an escalation chain"

def test_baseline_model_configured():
    routing=load_routing("config/routing.yaml"); models=load_models("config/models.yaml")
    assert routing["metrics"]["baseline_model"] in models

def test_estimate_cost_math():
    m=ModelConfig(name="x",provider="openai",model_id="g",input_cost_per_1k=0.001,
                  output_cost_per_1k=0.002,average_latency_ms=1,quality_tier="medium")
    assert m.estimate_cost(1000,500) == pytest.approx(0.001+0.001)
    assert m.estimate_cost(0,0) == 0
