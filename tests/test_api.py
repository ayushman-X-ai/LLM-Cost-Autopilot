import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from autopilot.db import get_request, recent
from autopilot.providers import ProviderError
from tests.conftest import make_response

SIMPLE = {"messages": [{"role": "user", "content": "Rewrite this sentence professionally: please send the report."}]}

def test_health(client):
    assert client.get("/v1/health").json() == {"status": "ok"}

def test_models_registry_has_real_pricing(client):
    models = client.get("/v1/models").json()["models"]
    assert models["openai_mini"]["input_cost_per_1k"] > 0
    assert models["openai_high"]["output_cost_per_1k"] > models["openai_mini"]["output_cost_per_1k"]
    assert all(m["pricing_updated_at"] for m in models.values())
    assert models["ollama_llama"]["input_cost_per_1k"] == 0.0

def test_stats_shape(client):
    s = client.get("/v1/stats").json()
    for key in ("requests", "cost_usd", "escalation_rate", "avg_latency_ms",
                "input_tokens", "output_tokens", "by_model", "by_tier"):
        assert key in s

def test_fast_mode_returns_immediately(client):
    r = client.post("/v1/completions", json=SIMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["verification_mode"] == "fast"
    assert body["verification_queued"] is True
    assert body["quality_score"] is None
    assert body["verified"] is None
    assert body["escalated"] is False
    assert body["routing_reason"]

def test_fast_mode_skips_queue_when_verification_disabled(client, monkeypatch):
    monkeypatch.setenv("VERIFICATION_ENABLED", "false")
    from autopilot.settings import get_settings
    get_settings.cache_clear()
    body = client.post("/v1/completions", json=SIMPLE).json()
    assert body["verification_queued"] is False

def test_quality_mode_pass(client, force_tier, monkeypatch):
    import autopilot.api as api
    from autopilot.verifier import VerificationResult
    force_tier("tier_2")
    async def fake_verify(*args, **kwargs):
        return VerificationResult(quality_score=4.6, agreement=0.9, dimensions={"correctness": 5},
                                  threshold=4.0, agreement_threshold=0.7, passed=True)
    monkeypatch.setattr(api, "verify", fake_verify)
    body = client.post("/v1/completions", json={**SIMPLE, "verification_mode": "quality"}).json()
    assert body["verification_mode"] == "quality"
    assert body["verification_queued"] is False
    assert body["verified"] is True
    assert body["quality_score"] == 4.6
    assert body["escalated"] is False
    row = get_request(body["id"])
    assert row["status"] == "verified" and row["quality_score"] == 4.6

def test_quality_mode_escalates_on_failure(client, force_tier, monkeypatch):
    import autopilot.api as api
    from autopilot.verifier import VerificationResult
    force_tier("tier_1")
    async def fake_verify(*args, **kwargs):
        return VerificationResult(quality_score=2.0, agreement=0.4, dimensions={"correctness": 2},
                                  threshold=3.8, agreement_threshold=0.7, passed=False, error="below_threshold")
    monkeypatch.setattr(api, "verify", fake_verify)
    body = client.post("/v1/completions", json={**SIMPLE, "verification_mode": "quality"}).json()
    assert body["verified"] is False
    assert body["escalated"] is True
    assert body["escalation_model"] == "openai_high"
    assert body["model"] == "openai_high"
    row = get_request(body["id"])
    assert row["escalated"] == 1 and row["status"] == "escalated"
    # escalated cost includes the original answer
    assert row["cost_usd"] >= 0

def test_invalid_verification_mode(client):
    r = client.post("/v1/completions", json={**SIMPLE, "verification_mode": "sometimes"})
    assert r.status_code == 422

def test_budget_protection_402(client, force_tier, monkeypatch):
    from autopilot.settings import get_settings
    force_tier("tier_3")
    monkeypatch.setenv("MAX_REQUEST_COST_USD", "0.000001")
    get_settings.cache_clear()
    r = client.post("/v1/completions", json=SIMPLE)
    assert r.status_code == 402

def test_cost_guard_routes_to_affordable_fallback(client, force_tier, monkeypatch):
    from autopilot.settings import get_settings
    force_tier("tier_3")  # primary openai_high, fallbacks anthropic_sonnet then anthropic_haiku
    monkeypatch.setenv("MAX_REQUEST_COST_USD", "0.004")  # affordable only for haiku
    get_settings.cache_clear()
    r = client.post("/v1/completions", json=SIMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["routing_event"] == "cost_guard"
    assert body["original_model"] == "openai_high"
    assert body["fallback_model"] == "anthropic_haiku"
    assert body["model"] == "anthropic_haiku"

def test_cloud_disabled_403_when_no_local_candidate(client, force_tier, monkeypatch):
    from autopilot.settings import get_settings
    force_tier("tier_3")  # no ollama in this chain
    monkeypatch.setenv("ALLOW_CLOUD", "false")
    get_settings.cache_clear()
    assert client.post("/v1/completions", json=SIMPLE).status_code == 403

def test_cloud_disabled_falls_back_to_ollama(client, force_tier, monkeypatch):
    from autopilot.settings import get_settings
    force_tier("tier_2")  # openai_mini -> anthropic_haiku -> ollama_llama
    monkeypatch.setenv("ALLOW_CLOUD", "false")
    get_settings.cache_clear()
    body = client.post("/v1/completions", json=SIMPLE).json()
    assert body["model"] == "ollama_llama"
    assert body["routing_event"] == "cloud_disabled"
    assert body["original_model"] == "openai_mini"
    assert body["fallback_model"] == "ollama_llama"

def test_provider_failure_fallback(client, force_tier, monkeypatch):
    import autopilot.api as api
    force_tier("tier_2")
    async def flaky(messages, model, temperature=0.2, max_tokens=512):
        if model.provider == "openai":
            raise ProviderError("simulated timeout", retryable=True, timeout=True)
        return make_response(model, text="fallback answer")
    monkeypatch.setattr(api, "complete_with_retry", flaky)
    body = client.post("/v1/completions", json=SIMPLE).json()
    assert body["routing_event"] == "provider_failure"
    assert body["original_model"] == "openai_mini"
    assert body["fallback_model"] == "anthropic_haiku"
    assert body["model"] == "anthropic_haiku"
    assert "simulated timeout" in body["reason"]
    row = get_request(body["id"])
    assert row["routing_event"] == "provider_failure" and row["original_model"] == "openai_mini"

def test_ollama_fallback_when_all_cloud_fails(client, force_tier, monkeypatch):
    import autopilot.api as api
    force_tier("tier_2")
    async def cloud_down(messages, model, temperature=0.2, max_tokens=512):
        if model.provider != "ollama":
            raise ProviderError("connection refused", retryable=True)
        return make_response(model, text="local answer")
    monkeypatch.setattr(api, "complete_with_retry", cloud_down)
    body = client.post("/v1/completions", json=SIMPLE).json()
    assert body["model"] == "ollama_llama"
    assert body["routing_event"] == "provider_failure"
    assert body["fallback_model"] == "ollama_llama"

def test_all_models_fail_502(client, force_tier, monkeypatch):
    import autopilot.api as api
    force_tier("tier_2")
    async def dead(messages, model, temperature=0.2, max_tokens=512):
        raise ProviderError("provider outage", retryable=True)
    monkeypatch.setattr(api, "complete_with_retry", dead)
    assert client.post("/v1/completions", json=SIMPLE).status_code == 502

def test_routing_config_get_defaults_to_yaml(client):
    body = client.get("/v1/routing-config").json()
    assert body["source"] == "yaml" and body["version"] == 0
    assert "tiers" in body["config"]

def test_routing_config_put_history_rollback(client):
    cfg = client.get("/v1/routing-config").json()["config"]
    cfg["tiers"]["tier_1"]["quality_threshold"] = 3.1
    put = client.put("/v1/routing-config?created_by=tester", json=cfg).json()
    assert put["version"] >= 1
    active = client.get("/v1/routing-config").json()
    assert active["source"] == "db" and active["config"]["tiers"]["tier_1"]["quality_threshold"] == 3.1
    history = client.get("/v1/routing-config/history").json()["versions"]
    assert len(history) >= 1 and history[0]["created_by"] == "tester"
    # change again, then roll back to the first persisted version
    cfg2 = json.loads(json.dumps(cfg)); cfg2["tiers"]["tier_1"]["quality_threshold"] = 3.9
    client.put("/v1/routing-config?created_by=tester2", json=cfg2)
    target = [v for v in history if v["created_by"] == "tester"][0]["version"]
    rb = client.post("/v1/routing-config/rollback", json={"version": target}).json()
    assert rb["restored_version"] == target
    assert client.get("/v1/routing-config").json()["config"]["tiers"]["tier_1"]["quality_threshold"] == 3.1

def test_routing_config_put_requires_tiers(client):
    assert client.put("/v1/routing-config", json={"nope": 1}).status_code == 422
    assert client.post("/v1/routing-config/rollback", json={}).status_code == 422
    assert client.post("/v1/routing-config/rollback", json={"version": 999}).status_code == 404

def test_concurrent_requests(client):
    def _send(_):
        return client.post("/v1/completions", json=SIMPLE).status_code
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(_send, range(16)))
    assert codes == [200] * 16
    assert len(recent()) >= 16
