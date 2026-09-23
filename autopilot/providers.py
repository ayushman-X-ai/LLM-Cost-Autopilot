import asyncio, json, time
from abc import ABC, abstractmethod
import httpx
from .config import ModelConfig
from .models import Message, LLMResponse, Usage
from .settings import get_settings

class ProviderError(RuntimeError):
    """Raised when a provider call fails.

    retryable: safe to retry (timeout, connection error, 408/429, 5xx).
    timeout:   the failure was a timeout specifically.
    """
    def __init__(self, message: str, *, retryable: bool = False, timeout: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.timeout = timeout

def classify_exception(e: Exception) -> ProviderError:
    """Map an arbitrary SDK/httpx exception onto a ProviderError with retry hints."""
    name = type(e).__name__.lower()
    status = getattr(e, "status_code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(e, httpx.TimeoutException) or "timeout" in name:
        return ProviderError(f"timeout: {e}", retryable=True, timeout=True)
    if "connect" in name or isinstance(e, (httpx.ConnectError, httpx.NetworkError)):
        return ProviderError(f"connection error: {e}", retryable=True)
    if status in (408, 429) or (isinstance(status, int) and status >= 500):
        return ProviderError(f"provider status {status}: {e}", retryable=True)
    return ProviderError(str(e), retryable=False)

class Provider(ABC):
    @abstractmethod
    async def complete(self, messages, model: ModelConfig, temperature=0.2, max_tokens=512) -> LLMResponse: ...

class DryRunProvider(Provider):
    async def complete(self, messages, model, temperature=0.2, max_tokens=512):
        system_text=" ".join(m.content for m in messages if m.role=="system")
        if "strict quality evaluator" in system_text:
            # Simulated verifier verdict: a passing score so dry-run demos do not
            # escalate every request or pollute the routing-failure feedback loop.
            text=json.dumps({"correctness":5,"relevance":4,"completeness":4,
                             "instruction_following":5,"agreement":0.9,"issues":[],
                             "rationale":"Dry-run simulated verification."})
        else:
            text = "[DRY RUN] This is a simulated response from the routed model."
        # Simulate usage instead of a fixed token count so dry-run cost reports
        # differentiate model price tiers: output scales with input length,
        # capped by the requested max_tokens.
        inp=sum(len(m.content.split()) for m in messages)
        out=min(max_tokens, max(12, round(1.5*inp)))
        return LLMResponse(text=text, model=model.model_id, provider=model.provider, usage=Usage(input_tokens=inp, output_tokens=out, total_tokens=inp+out), latency_ms=5, cost_usd=model.estimate_cost(inp,out), finish_reason="stop")

class OpenAIProvider(Provider):
    async def complete(self, messages, model, temperature=0.2, max_tokens=512):
        s=get_settings()
        if not s.openai_api_key: raise ProviderError("OPENAI_API_KEY is not configured")
        from openai import AsyncOpenAI
        client=AsyncOpenAI(api_key=s.openai_api_key, timeout=s.provider_timeout_seconds, max_retries=0)
        t=time.perf_counter()
        try:
            r=await client.chat.completions.create(model=model.model_id, messages=[m.model_dump() for m in messages], temperature=temperature, max_tokens=max_tokens)
        except Exception as e: raise classify_exception(e) from e
        latency=(time.perf_counter()-t)*1000
        usage=r.usage
        inp=getattr(usage,"prompt_tokens",0) or 0; out=getattr(usage,"completion_tokens",0) or 0
        return LLMResponse(text=r.choices[0].message.content or "", model=model.model_id, provider="openai", usage=Usage(input_tokens=inp, output_tokens=out, total_tokens=inp+out), latency_ms=latency, cost_usd=model.estimate_cost(inp,out), finish_reason=r.choices[0].finish_reason, raw=r.model_dump())

class AnthropicProvider(Provider):
    async def complete(self, messages, model, temperature=0.2, max_tokens=512):
        s=get_settings()
        if not s.anthropic_api_key: raise ProviderError("ANTHROPIC_API_KEY is not configured")
        from anthropic import AsyncAnthropic
        client=AsyncAnthropic(api_key=s.anthropic_api_key, timeout=s.provider_timeout_seconds, max_retries=0)
        system="\n".join(m.content for m in messages if m.role=="system") or None
        msgs=[{"role":m.role if m.role in ("user","assistant") else "user","content":m.content} for m in messages if m.role!="system"]
        t=time.perf_counter()
        try:
            r=await client.messages.create(model=model.model_id, max_tokens=max_tokens, temperature=temperature, system=system, messages=msgs)
        except Exception as e: raise classify_exception(e) from e
        latency=(time.perf_counter()-t)*1000
        inp=getattr(r.usage,"input_tokens",0) or 0; out=getattr(r.usage,"output_tokens",0) or 0
        text="".join(getattr(x,"text","") for x in r.content)
        return LLMResponse(text=text, model=model.model_id, provider="anthropic", usage=Usage(input_tokens=inp, output_tokens=out, total_tokens=inp+out), latency_ms=latency, cost_usd=model.estimate_cost(inp,out), finish_reason=getattr(r,"stop_reason",None), raw=r.model_dump())

class OllamaProvider(Provider):
    async def complete(self, messages, model, temperature=0.2, max_tokens=512):
        s=get_settings(); t=time.perf_counter()
        payload={"model":model.model_id,"messages":[m.model_dump() for m in messages],"stream":False,"options":{"temperature":temperature,"num_predict":max_tokens}}
        try:
            async with httpx.AsyncClient(timeout=s.provider_timeout_seconds) as client:
                r=await client.post(f"{s.ollama_base_url.rstrip('/')}/api/chat", json=payload); r.raise_for_status(); data=r.json()
        except Exception as e: raise classify_exception(e) from e
        latency=(time.perf_counter()-t)*1000
        inp=int(data.get("prompt_eval_count",0) or 0); out=int(data.get("eval_count",0) or 0)
        return LLMResponse(text=data.get("message",{}).get("content", ""), model=model.model_id, provider="ollama", usage=Usage(input_tokens=inp, output_tokens=out, total_tokens=inp+out), latency_ms=latency, cost_usd=model.estimate_cost(inp,out), finish_reason=data.get("done_reason"), raw=data)

def provider_for(model: ModelConfig) -> Provider:
    s=get_settings()
    if model.provider not in ("openai","anthropic","ollama"):
        # fail fast on registry misconfiguration, even in dry-run mode
        raise ProviderError(f"Unknown provider: {model.provider}")
    if s.dry_run: return DryRunProvider()
    if model.provider=="openai": return OpenAIProvider()
    if model.provider=="anthropic": return AnthropicProvider()
    if model.provider=="ollama": return OllamaProvider()
    raise ProviderError(f"Unknown provider: {model.provider}")

async def complete_with_retry(messages, model: ModelConfig, temperature=0.2, max_tokens=512, *, attempts: int | None = None, backoff_seconds: float = 0.25) -> LLMResponse:
    """Call a provider, retrying retryable failures (timeout/connection/429/5xx) with exponential backoff.

    Non-retryable failures (bad request, auth, unknown provider) are raised immediately.
    """
    s=get_settings()
    tries=attempts if attempts is not None else max(1, s.provider_max_attempts)
    last: ProviderError | None = None
    for i in range(tries):
        try:
            return await provider_for(model).complete(messages, model, temperature, max_tokens)
        except ProviderError as e:
            last=e
            if not e.retryable or i == tries-1: raise
            await asyncio.sleep(backoff_seconds * (2 ** i))
    raise last  # pragma: no cover - defensive
