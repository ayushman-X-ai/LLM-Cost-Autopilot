"""Quality verification with weighted, multi-dimension scoring.

The verifier asks a model to score a candidate answer on four dimensions
(correctness, relevance, completeness, instruction following) plus an
agreement score. The weighted combination is the quality_score:

    correctness 0.4 | relevance 0.2 | completeness 0.2 | instruction 0.2

A verification fails when quality_score < the tier threshold OR agreement <
agreement_threshold. Failures are escalated to a premium model (see
`escalate`) and recorded in the feedback loop for retraining.
"""
import json, re
from dataclasses import dataclass, field
from .config import load_models, load_routing
from .db import update_request, get_request
from .feedback import record_routing_failure
from .logging_utils import get_logger, event
from .models import Message, LLMResponse, Usage
from .providers import complete_with_retry
from .settings import get_settings

logger=get_logger()

DEFAULT_WEIGHTS={"correctness":0.4,"relevance":0.2,"completeness":0.2,"instruction_following":0.2}
DIMENSIONS=tuple(DEFAULT_WEIGHTS)

VERIFY_PROMPT="""You are a strict quality evaluator. Compare the candidate answer to the original user request.

Score each dimension from 1 (unacceptable) to 5 (excellent):
- correctness: factually sound, no fabricated content
- relevance: directly addresses what was asked, no filler
- completeness: covers the required points for the requested scope
- instruction_following: obeys every explicit instruction/format constraint

Also return agreement (0-1): how confident you are that this answer satisfies the request.

Return JSON only, no markdown, with exactly these keys:
{"correctness": 1-5, "relevance": 1-5, "completeness": 1-5, "instruction_following": 1-5, "agreement": 0-1, "issues": ["short list of concrete problems"], "rationale": "one sentence"}"""

@dataclass
class VerificationResult:
    quality_score: float = 0.0
    agreement: float = 0.0
    dimensions: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    rationale: str = ""
    threshold: float = 4.0
    agreement_threshold: float = 0.70
    passed: bool = False
    skipped: bool = False
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "quality_score": self.quality_score, "agreement": self.agreement,
            "dimensions": self.dimensions, "issues": self.issues, "rationale": self.rationale,
            "threshold": self.threshold, "agreement_threshold": self.agreement_threshold,
            "passed": self.passed, "skipped": self.skipped, "error": self.error,
        }

def weights_from(routing: dict) -> dict[str, float]:
    raw=(routing.get("verification",{}) or {}).get("weights") or {}
    weights={k: float(raw.get(k, DEFAULT_WEIGHTS[k])) for k in DEFAULT_WEIGHTS}
    total=sum(weights.values())
    if total <= 0: return dict(DEFAULT_WEIGHTS)
    return {k: v/total for k, v in weights.items()}

def clamp(value, lo, hi, default=0.0):
    try: v=float(value)
    except (TypeError, ValueError): return default
    return max(lo, min(hi, v))

def score_dimensions(dimensions: dict[str, float], weights: dict[str, float]) -> float:
    return round(sum(weights[k]*float(dimensions.get(k,0.0)) for k in weights), 4)

def parse_verifier_payload(text: str, weights: dict[str, float]) -> VerificationResult:
    """Parse verifier JSON into a scored result; malformed output degrades to score 0 / fail."""
    data=None
    try:
        data=json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match=re.search(r"\{.*\}", text or "", re.S)
        if match:
            try: data=json.loads(match.group(0))
            except json.JSONDecodeError: data=None
    if not isinstance(data, dict):
        return VerificationResult(dimensions={k:0.0 for k in weights}, quality_score=0.0,
                                  issues=["Verifier returned non-JSON output"], rationale=(text or "")[:500],
                                  passed=False, error="malformed_verifier_response")
    dimensions={k: clamp(data.get(k), 1, 5, default=0.0) for k in weights}
    agreement=clamp(data.get("agreement"), 0, 1, default=0.0)
    issues=data.get("issues") if isinstance(data.get("issues"), list) else []
    rationale=str(data.get("rationale",""))[:1000]
    return VerificationResult(dimensions=dimensions, agreement=agreement,
                              quality_score=score_dimensions(dimensions, weights),
                              issues=[str(i) for i in issues], rationale=rationale)

async def verify(request_id: str, messages: list[Message], candidate_text: str, tier: str,
                 routed_model_name: str | None = None, *, record_failure: bool = True) -> VerificationResult:
    """Score a candidate answer, persist the quality score, and record failures for retraining."""
    s=get_settings()
    routing=load_routing(s.routing_config_path)
    models=load_models(s.models_config_path)
    vcfg=routing.get("verification",{}) or {}
    weights=weights_from(routing)
    threshold=float((routing.get("tiers",{}) or {}).get(tier,{}).get("quality_threshold", vcfg.get("quality_threshold", 4.0)))
    agreement_threshold=float(vcfg.get("agreement_threshold", 0.70))
    verifier=models.get(vcfg.get("verifier_model","anthropic_haiku"))
    if verifier is None:
        return VerificationResult(skipped=True, passed=True, threshold=threshold,
                                  agreement_threshold=agreement_threshold,
                                  error=f"verifier model '{vcfg.get('verifier_model')}' not found")
    prompt=[Message(role="system",content=VERIFY_PROMPT),
            Message(role="user",content=json.dumps({"request":[m.model_dump() for m in messages],"candidate":candidate_text}, ensure_ascii=False))]
    try:
        vr=await complete_with_retry(prompt, verifier, temperature=0, max_tokens=400)
    except Exception as e:
        return VerificationResult(skipped=True, passed=True, threshold=threshold,
                                  agreement_threshold=agreement_threshold, error=f"verifier call failed: {e}")
    result=parse_verifier_payload(vr.text, weights)
    result.threshold=threshold; result.agreement_threshold=agreement_threshold
    result.passed=bool(result.quality_score >= threshold and result.agreement >= agreement_threshold)
    if result.error is None and not result.passed:
        result.error = "below_threshold" if result.quality_score < threshold else "low_agreement"
    update_request(request_id, quality_score=result.quality_score)
    if not result.passed and record_failure:
        prompt_text="\n".join(m.content for m in messages)
        row=record_routing_failure(prompt_text, tier, routed_model_name or "unknown", result.quality_score,
                                   result.issues, request_id, threshold, result.agreement, result.rationale)
        event(logger,"verification_failed",request_id=request_id,tier=tier,model=routed_model_name,
              quality_score=result.quality_score,threshold=threshold,failure_id=row["id"])
    else:
        event(logger,"verification_passed",request_id=request_id,tier=tier,model=routed_model_name,
              quality_score=result.quality_score,threshold=threshold)
    return result

async def escalate(request_id: str, messages: list[Message], routed_model_name: str, tier: str | None = None,
                   quality: VerificationResult | None = None) -> tuple[str, LLMResponse] | None:
    """Re-run a failed answer on a higher-tier model. Returns (model_name, response) or None."""
    s=get_settings()
    routing=load_routing(s.routing_config_path)
    models=load_models(s.models_config_path)
    vcfg=routing.get("verification",{}) or {}
    names=vcfg.get("escalation_models") or [vcfg.get("escalation_model","openai_high")]
    def eligible(m) -> bool:
        if not m or not m.enabled: return False
        if not s.allow_cloud and m.provider!="ollama": return False
        return True
    chosen=None
    for prefer_different in (True, False):
        for name in names:
            m=models.get(name)
            if not eligible(m): continue
            if prefer_different and m.name==routed_model_name: continue
            chosen=m; break
        if chosen: break
    if chosen is None: return None
    er=await complete_with_retry(messages, chosen, temperature=0.2, max_tokens=512)
    orig=get_request(request_id) or {}
    total=float(orig.get("cost_usd") or 0)+er.cost_usd
    update_request(request_id, escalated=1, escalation_model=chosen.name, cost_usd=total, status="escalated")
    event(logger,"escalated",request_id=request_id,tier=tier,from_model=routed_model_name,
          escalation_model=chosen.name,extra_cost_usd=er.cost_usd)
    return chosen.name, er

async def verify_and_maybe_escalate(request_id: str, messages: list[Message], candidate_text: str,
                                    routed_model_name: str, tier: str) -> VerificationResult | None:
    """Fast-mode path: verify in the background, escalate when quality is insufficient."""
    s=get_settings()
    if not s.verification_enabled: return None
    try:
        result=await verify(request_id, messages, candidate_text, tier, routed_model_name)
        if not result.passed:
            await escalate(request_id, messages, routed_model_name, tier, result)
        else:
            update_request(request_id, status="verified")
        return result
    except Exception as e:
        update_request(request_id, verification_error=str(e), status="verification_failed")
        return None
