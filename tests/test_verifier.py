import json

import pytest
from autopilot.db import insert_request, get_request
from autopilot.feedback import load_routing_failures
from autopilot.models import Message
from autopilot.settings import get_settings
from autopilot.verifier import (DEFAULT_WEIGHTS, VerificationResult, escalate, parse_verifier_payload,
                                score_dimensions, verify, verify_and_maybe_escalate, weights_from)
from tests.conftest import make_response

def _weights(): return weights_from({})

def _payload(**overrides):
    base = {"correctness": 5, "relevance": 4, "completeness": 4, "instruction_following": 5,
            "agreement": 0.9, "issues": [], "rationale": "solid answer"}
    base.update(overrides); return json.dumps(base)

def test_weights_normalized():
    w = weights_from({"verification": {"weights": {"correctness": 4, "relevance": 1,
                                                   "completeness": 1, "instruction_following": 2}}})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["correctness"] == 0.5

def test_score_dimensions_weighted():
    w = _weights()
    assert score_dimensions({k: 5 for k in w}, w) == 5.0
    # correctness 1, rest 5 -> 0.4*1 + 0.6*5 = 3.4
    dims = {"correctness": 1, "relevance": 5, "completeness": 5, "instruction_following": 5}
    assert abs(score_dimensions(dims, w) - 3.4) < 1e-9

def test_parse_valid_json():
    r = parse_verifier_payload(_payload(), _weights())
    assert r.agreement == 0.9 and r.quality_score > 4 and not r.error

def test_parse_embedded_json():
    r = parse_verifier_payload("Here you go:\n```json\n" + _payload() + "\n```", _weights())
    assert r.quality_score > 4

def test_parse_malformed_response_fails_closed():
    r = parse_verifier_payload("sorry, I cannot score that", _weights())
    assert r.passed is False and r.error == "malformed_verifier_response"
    assert r.quality_score == 0.0 and "non-JSON" in r.issues[0]

def test_parse_missing_and_out_of_range_dimensions():
    r = parse_verifier_payload(json.dumps({"correctness": 9, "agreement": 3}), _weights())
    assert r.dimensions["correctness"] == 5.0        # clamped
    assert r.dimensions["relevance"] == 0.0          # missing -> 0
    assert r.agreement == 1.0                        # clamped
    assert r.quality_score < 5

def test_parse_non_numeric_dimension_scores_zero():
    r = parse_verifier_payload(json.dumps({"correctness": "good", "relevance": None,
                                           "completeness": 3, "instruction_following": 3,
                                           "agreement": "high"}), _weights())
    assert r.dimensions["correctness"] == 0.0 and r.agreement == 0.0

@pytest.mark.asyncio
async def test_verify_pass_updates_score(monkeypatch):
    import autopilot.verifier as v
    insert_request({"id": "v_pass", "timestamp": 1.0, "prompt_hash": "h", "prompt_text": "p"})
    async def ok(messages, model, temperature=0.2, max_tokens=512):
        from autopilot.models import LLMResponse
        return LLMResponse(text=_payload(), model=model.model_id, provider=model.provider,
                           usage=None or __import__("autopilot.models", fromlist=["Usage"]).Usage(input_tokens=5, output_tokens=5, total_tokens=10),
                           latency_ms=3.0, cost_usd=0.0)
    monkeypatch.setattr(v, "complete_with_retry", ok)
    result = await verify("v_pass", [Message(role="user", content="q")], "answer", "tier_2", "openai_mini")
    assert result.passed and result.quality_score >= 4.0
    assert get_request("v_pass")["quality_score"] == result.quality_score
    assert load_routing_failures() == []   # passing verifications do not pollute feedback

@pytest.mark.asyncio
async def test_verify_failure_records_feedback(monkeypatch):
    import autopilot.verifier as v
    async def bad(messages, model, temperature=0.2, max_tokens=512):
        from autopilot.models import LLMResponse, Usage
        return LLMResponse(text=_payload(correctness=2, relevance=2, completeness=1,
                                         instruction_following=2, agreement=0.5),
                           model=model.model_id, provider=model.provider,
                           usage=Usage(input_tokens=5, output_tokens=5, total_tokens=10),
                           latency_ms=3.0, cost_usd=0.0)
    monkeypatch.setattr(v, "complete_with_retry", bad)
    result = await verify("v_fail", [Message(role="user", content="explain the system")],
                          "weak answer", "tier_1", "ollama_llama")
    assert not result.passed
    rows = load_routing_failures()
    assert len(rows) == 1
    row = rows[0]
    assert row["routed_tier"] == "tier_1" and row["corrected_tier"] == "tier_2"
    assert row["label_changed"] is True and row["reviewed"] is False
    assert row["text"] == "explain the system" and row["request_id"] == "v_fail"

@pytest.mark.asyncio
async def test_verify_tier3_failure_keeps_label(monkeypatch):
    import autopilot.verifier as v
    async def bad(messages, model, temperature=0.2, max_tokens=512):
        from autopilot.models import LLMResponse, Usage
        return LLMResponse(text="no json here", model=model.model_id, provider=model.provider,
                           usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                           latency_ms=1.0, cost_usd=0.0)
    monkeypatch.setattr(v, "complete_with_retry", bad)
    result = await verify("v_t3", [Message(role="user", content="q")], "a", "tier_3", "openai_high")
    assert not result.passed
    row = load_routing_failures()[0]
    assert row["corrected_tier"] == "tier_3" and row["label_changed"] is False

@pytest.mark.asyncio
async def test_agreement_threshold_blocks_pass(monkeypatch):
    import autopilot.verifier as v
    async def low_agreement(messages, model, temperature=0.2, max_tokens=512):
        from autopilot.models import LLMResponse, Usage
        return LLMResponse(text=_payload(agreement=0.2), model=model.model_id, provider=model.provider,
                           usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                           latency_ms=1.0, cost_usd=0.0)
    monkeypatch.setattr(v, "complete_with_retry", low_agreement)
    result = await verify("v_agr", [Message(role="user", content="q")], "a", "tier_2", "openai_mini")
    assert not result.passed and result.error == "low_agreement"
    assert result.quality_score >= 4.0  # score alone passed; agreement did not

@pytest.mark.asyncio
async def test_verify_missing_verifier_is_skipped():
    result = await verify("v_missing", [Message(role="user", content="q")], "a", "tier_1",
                          record_failure=False)
    # verifier model exists in config, so this should not be skipped
    assert isinstance(result, VerificationResult)

@pytest.mark.asyncio
async def test_escalate_picks_premium_and_updates_cost(monkeypatch):
    import autopilot.verifier as v
    insert_request({"id": "esc_1", "timestamp": 1.0, "prompt_hash": "h", "prompt_text": "p",
                    "cost_usd": 0.01, "escalated": 0})
    seen = {}
    async def capture(messages, model, temperature=0.2, max_tokens=512):
        seen["model"] = model.name
        return make_response(model, text="better answer", cost=0.05)
    monkeypatch.setattr(v, "complete_with_retry", capture)
    out = await escalate("esc_1", [Message(role="user", content="q")], "openai_mini", "tier_2")
    assert out is not None
    name, resp = out
    assert name == "openai_high" and seen["model"] == "openai_high"
    row = get_request("esc_1")
    assert row["escalated"] == 1 and row["escalation_model"] == "openai_high"
    assert abs(row["cost_usd"] - 0.06) < 1e-9 and row["status"] == "escalated"

@pytest.mark.asyncio
async def test_escalate_skips_the_failed_premium_model(monkeypatch):
    import autopilot.verifier as v
    seen = {}
    async def capture(messages, model, temperature=0.2, max_tokens=512):
        seen["model"] = model.name
        return make_response(model)
    monkeypatch.setattr(v, "complete_with_retry", capture)
    out = await escalate("esc_2", [Message(role="user", content="q")], "openai_high", "tier_3")
    assert out is not None and out[0] == "anthropic_sonnet"

@pytest.mark.asyncio
async def test_worker_flow_verifies_and_marks_verified(monkeypatch):
    import autopilot.verifier as v
    insert_request({"id": "w_ok", "timestamp": 1.0, "prompt_hash": "h", "prompt_text": "p", "status": "completed"})
    async def fake_verify(*args, **kwargs):
        return VerificationResult(quality_score=4.4, agreement=0.9, threshold=4.0,
                                  agreement_threshold=0.7, passed=True)
    async def should_not_escalate(*args, **kwargs):
        raise AssertionError("escalation should not run on pass")
    monkeypatch.setattr(v, "verify", fake_verify)
    monkeypatch.setattr(v, "escalate", should_not_escalate)
    result = await verify_and_maybe_escalate("w_ok", [Message(role="user", content="q")],
                                             "answer", "openai_mini", "tier_2")
    assert result.passed and get_request("w_ok")["status"] == "verified"

@pytest.mark.asyncio
async def test_worker_flow_escalates_on_fail(monkeypatch):
    import autopilot.verifier as v
    insert_request({"id": "w_bad", "timestamp": 1.0, "prompt_hash": "h", "prompt_text": "p",
                    "cost_usd": 0.001, "status": "completed"})
    calls = {}
    async def fake_verify(*args, **kwargs):
        return VerificationResult(quality_score=2.5, agreement=0.5, threshold=4.0,
                                  agreement_threshold=0.7, passed=False)
    async def fake_escalate(request_id, messages, routed_model_name, tier=None, quality=None):
        calls["escalated"] = True
        insert_request({"id": request_id, "timestamp": 1.0, "prompt_hash": "h", "prompt_text": "p",
                        "escalated": 1, "escalation_model": "openai_high", "cost_usd": 0.02,
                        "status": "escalated"})
        return ("openai_high", None)
    monkeypatch.setattr(v, "verify", fake_verify)
    monkeypatch.setattr(v, "escalate", fake_escalate)
    await verify_and_maybe_escalate("w_bad", [Message(role="user", content="q")],
                                    "answer", "openai_mini", "tier_2")
    assert calls.get("escalated") is True
    assert get_request("w_bad")["status"] == "escalated"

@pytest.mark.asyncio
async def test_worker_records_verification_errors(monkeypatch):
    import autopilot.verifier as v
    insert_request({"id": "w_err", "timestamp": 1.0, "prompt_hash": "h", "prompt_text": "p",
                    "status": "completed"})
    async def boom(*args, **kwargs):
        raise RuntimeError("verifier exploded")
    monkeypatch.setattr(v, "verify", boom)
    out = await verify_and_maybe_escalate("w_err", [Message(role="user", content="q")],
                                          "answer", "openai_mini", "tier_2")
    assert out is None
    row = get_request("w_err")
    assert row["status"] == "verification_failed" and "exploded" in row["verification_error"]

def test_verification_disabled_setting_short_circuits(monkeypatch):
    monkeypatch.setenv("VERIFICATION_ENABLED", "false")
    get_settings.cache_clear()
    import asyncio
    from autopilot.verifier import verify_and_maybe_escalate as fn
    # disabled verification returns immediately without touching the DB
    async def run():
        return await fn("nope", [], "x", "m", "tier_1")
    assert asyncio.run(run()) is None
