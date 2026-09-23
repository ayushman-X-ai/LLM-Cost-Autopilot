import httpx
import pytest
from autopilot.config import ModelConfig
from autopilot.providers import (DryRunProvider, Provider, ProviderError, classify_exception,
                                 complete_with_retry, provider_for)

def make_model(**kw):
    base = dict(name="m", provider="ollama", model_id="llama3.2:b", input_cost_per_1k=0.0,
                output_cost_per_1k=0.0, average_latency_ms=100, quality_tier="low", enabled=True)
    base.update(kw)
    return ModelConfig(**base)

def test_timeout_is_retryable():
    e = classify_exception(httpx.ReadTimeout("timed out"))
    assert e.retryable and e.timeout

def test_connection_error_is_retryable():
    req = httpx.Request("POST", "http://x/api")
    e = classify_exception(httpx.ConnectError("refused", request=req))
    assert e.retryable and not e.timeout

def test_generic_timeout_class_name():
    class APITimeoutError(Exception): pass
    e = classify_exception(APITimeoutError("SDK timeout"))
    assert e.retryable and e.timeout

def test_rate_limit_and_server_errors_retryable():
    class RateLimitError(Exception):
        status_code = 429
    class InternalServerError(Exception):
        status_code = 503
    assert classify_exception(RateLimitError("429")).retryable
    assert classify_exception(InternalServerError("503")).retryable

def test_client_errors_not_retryable():
    class BadRequestError(Exception):
        status_code = 400
    e = classify_exception(BadRequestError("bad request"))
    assert not e.retryable and not e.timeout

class FlakyProvider(Provider):
    def __init__(self, fail_times, exc):
        self.fail_times = fail_times; self.exc = exc; self.calls = 0
    async def complete(self, messages, model, temperature=0.2, max_tokens=512):
        self.calls += 1
        if self.calls <= self.fail_times: raise self.exc
        from tests.conftest import make_response
        return make_response(model)

@pytest.mark.asyncio
async def test_retry_recovers_from_transient_failures(monkeypatch):
    flaky = FlakyProvider(2, ProviderError("timeout", retryable=True, timeout=True))
    monkeypatch.setattr("autopilot.providers.provider_for", lambda m: flaky)
    out = await complete_with_retry([], make_model(), attempts=3, backoff_seconds=0)
    assert out is not None and flaky.calls == 3

@pytest.mark.asyncio
async def test_retry_gives_up_after_attempts(monkeypatch):
    flaky = FlakyProvider(99, ProviderError("timeout", retryable=True, timeout=True))
    monkeypatch.setattr("autopilot.providers.provider_for", lambda m: flaky)
    with pytest.raises(ProviderError):
        await complete_with_retry([], make_model(), attempts=2, backoff_seconds=0)
    assert flaky.calls == 2

@pytest.mark.asyncio
async def test_non_retryable_fails_immediately(monkeypatch):
    flaky = FlakyProvider(99, ProviderError("invalid key", retryable=False))
    monkeypatch.setattr("autopilot.providers.provider_for", lambda m: flaky)
    with pytest.raises(ProviderError):
        await complete_with_retry([], make_model(), attempts=3, backoff_seconds=0)
    assert flaky.calls == 1

@pytest.mark.asyncio
async def test_retry_uses_settings_attempts(monkeypatch, ):
    flaky = FlakyProvider(99, ProviderError("timeout", retryable=True, timeout=True))
    monkeypatch.setattr("autopilot.providers.provider_for", lambda m: flaky)
    from autopilot.settings import get_settings
    attempts = get_settings().provider_max_attempts
    with pytest.raises(ProviderError):
        await complete_with_retry([], make_model(), backoff_seconds=0)
    assert flaky.calls == attempts

def test_dry_run_provider_selected():
    model = make_model(provider="openai")
    assert isinstance(provider_for(model), DryRunProvider)

@pytest.mark.asyncio
async def test_dry_run_uses_registry_pricing():
    model = make_model(name="openai_mini", provider="openai", input_cost_per_1k=0.00015,
                       output_cost_per_1k=0.0006)
    from autopilot.models import Message
    out = await DryRunProvider().complete([Message(role="user", content="one two three")], model)
    assert out.usage.input_tokens == 3 and out.usage.output_tokens == 12
    assert out.cost_usd == pytest.approx(model.estimate_cost(3, 12))
    assert out.cost_usd > 0  # cloud models are no longer free in dry runs

def test_unknown_provider_raises():
    with pytest.raises(ProviderError):
        provider_for(make_model(provider="mystery"))
