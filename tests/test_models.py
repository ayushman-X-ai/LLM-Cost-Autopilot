import pytest
from pydantic import ValidationError
from autopilot.models import CompletionRequest, CompletionResponse, Usage

def test_request():
    r=CompletionRequest(messages=[{"role":"user","content":"hello"}]); assert r.max_tokens==512
    assert r.verification_mode is None

def test_request_verification_mode_allowed():
    for mode in ("fast","quality"):
        r=CompletionResponse(id="x",response="y",model="m",provider="p",complexity_tier="tier_1",
                             routing_reason="r",usage=Usage(),cost_usd=0,latency_ms=1,
                             verification_queued=False,verification_mode=mode)
        assert r.verification_mode==mode

def test_request_rejects_unknown_verification_mode():
    with pytest.raises(ValidationError):
        CompletionRequest(messages=[{"role":"user","content":"hi"}],verification_mode="banana")

def test_request_rejects_unknown_role():
    with pytest.raises(ValidationError):
        CompletionRequest(messages=[{"role":"wizard","content":"hi"}])

def test_response_fallback_fields_default_none():
    r=CompletionResponse(id="x",response="y",model="m",provider="p",complexity_tier="tier_1",
                         routing_reason="r",usage=Usage(),cost_usd=0,latency_ms=1,
                         verification_queued=True)
    assert r.routing_event is None and r.original_model is None
    assert r.fallback_model is None and r.quality_score is None and r.verified is None
