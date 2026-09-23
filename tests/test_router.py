import pytest
from autopilot.config import ModelConfig, load_routing
from autopilot.models import Message
from autopilot.router import CostRouter

def msgs(text): return [Message(role="user", content=text)]

def test_classify_and_route_uses_tier_config():
    r = CostRouter()
    decision = r.classify_and_route(msgs("Rewrite this sentence professionally: please send the report."))
    routing = load_routing("config/routing.yaml")
    assert decision.tier in routing["tiers"]
    assert decision.model_name == routing["tiers"][decision.tier]["model"]
    assert 0 <= decision.features["classifier_confidence"] <= 1
    assert decision.estimated_cost_usd >= 0

def test_candidate_chain_order():
    r = CostRouter()
    names = [m.name for m in r.candidates_for("tier_1")]
    assert names == ["ollama_llama", "openai_mini", "anthropic_haiku"]

def test_candidate_chain_skips_disabled_models():
    r = CostRouter()
    r.routing = {"tiers": {"tier_x": {"model": "openai_mini", "fallbacks": ["ghost_model", "openai_high"]}}}
    names = [m.name for m in r.candidates_for("tier_x")]
    assert names == ["openai_mini", "openai_high"]  # unknown ghost_model dropped

def test_candidate_chain_deduplicates():
    r = CostRouter()
    r.routing = {"tiers": {"tier_y": {"model": "openai_mini", "fallbacks": ["openai_mini", "anthropic_haiku"]}}}
    names = [m.name for m in r.candidates_for("tier_y")]
    assert names == ["openai_mini", "anthropic_haiku"]

def test_missing_tier_raises():
    r = CostRouter()
    with pytest.raises(ValueError):
        r.candidates_for("tier_9")
    with pytest.raises(ValueError):
        r.tier_config("tier_9")

def test_all_models_disabled_raises():
    r = CostRouter()
    r.models = {k: ModelConfig(**{**v.__dict__, "enabled": False}) for k, v in r.models.items()}
    with pytest.raises(ValueError):
        r.candidates_for("tier_1")

def test_disabled_primary_model_raises(monkeypatch):
    r = CostRouter()
    r.models["ollama_llama"] = ModelConfig(**{**r.models["ollama_llama"].__dict__, "enabled": False})
    monkeypatch.setattr(r.classifier, "predict", lambda text: ("tier_1", 0.9, {}))
    with pytest.raises(ValueError, match="disabled"):
        r.classify_and_route(msgs("anything"))

def test_confidence_reported_in_reason():
    r = CostRouter()
    decision = r.classify_and_route(msgs("Summarize this text in five bullet points: " + "word " * 80))
    assert "complexity (classifier confidence" in decision.reason
