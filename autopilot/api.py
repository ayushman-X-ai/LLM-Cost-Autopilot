import hashlib, json, time, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .config import load_models, load_routing
from .db import insert_request, stats, insert_routing_version, latest_routing_version, list_routing_versions, get_routing_version
from .logging_utils import get_logger, event
from .models import CompletionRequest, CompletionResponse, Usage
from .providers import complete_with_retry, ProviderError
from .router import CostRouter
from .settings import get_settings
from .verifier import verify, escalate
from .worker import queue, VerificationJob

logger=get_logger()
router=None

@asynccontextmanager
async def lifespan(app):
    global router
    router=CostRouter()
    persisted=latest_routing_version()
    if persisted:
        # Configuration versioning: the last persisted config survives restarts.
        router.routing=json.loads(persisted["config"])
        event(logger,"routing_config_loaded",version=persisted["version"],source="db")
    await queue.start()
    try:
        yield
    finally:
        await queue.stop()

app=FastAPI(title="LLM Cost Autopilot",version="0.1.0",lifespan=lifespan)

@app.get("/v1/health")
async def health(): return {"status":"ok"}

@app.get("/v1/models")
async def models():
    s=get_settings(); return {"models":{k:v.__dict__ for k,v in load_models(s.models_config_path).items()}}

@app.get("/v1/stats")
async def get_stats(): return stats()

@app.get("/v1/routing-config")
async def get_routing_config():
    s=get_settings()
    persisted=latest_routing_version()
    if persisted:
        return {"config":json.loads(persisted["config"]),"version":persisted["version"],
                "created_at":persisted["created_at"],"created_by":persisted["created_by"],"source":"db"}
    return {"config":load_routing(s.routing_config_path),"version":0,"created_at":None,
            "created_by":None,"source":"yaml"}

@app.put("/v1/routing-config")
async def put_routing(payload:dict, created_by:str="api"):
    # Runtime update, persisted as a new version so it survives restarts.
    global router
    if router is None: raise HTTPException(503,"Router not initialized")
    if not isinstance(payload,dict) or "tiers" not in payload:
        raise HTTPException(422,"Routing config must contain a 'tiers' mapping")
    version=insert_routing_version(payload, created_by)
    router.routing=payload
    event(logger,"routing_config_updated",version=version,created_by=created_by)
    return {"status":"updated","version":version,"config":payload}

@app.get("/v1/routing-config/history")
async def routing_config_history(limit:int=50):
    rows=list_routing_versions(max(1,min(limit,500)))
    return {"versions":[{**{k:v for k,v in r.items() if k!="config"},"config":json.loads(r["config"])} for r in rows]}

@app.post("/v1/routing-config/rollback")
async def rollback(payload:dict):
    global router
    if router is None: raise HTTPException(503,"Router not initialized")
    target=payload.get("version")
    if target is None: raise HTTPException(422,"Body must contain 'version'")
    row=get_routing_version(int(target))
    if not row: raise HTTPException(404,f"Routing config version {target} not found")
    config=json.loads(row["config"])
    new_version=insert_routing_version(config, created_by=f"rollback:{row['version']}")
    router.routing=config
    event(logger,"routing_config_rolled_back",restored_version=row["version"],version=new_version)
    return {"status":"rolled_back","restored_version":row["version"],"version":new_version,"config":config}

def _sum_usage(a: Usage, b: Usage) -> Usage:
    return Usage(input_tokens=a.input_tokens+b.input_tokens, output_tokens=a.output_tokens+b.output_tokens,
                 total_tokens=a.total_tokens+b.total_tokens)

@app.post("/v1/completions",response_model=CompletionResponse)
async def completions(req:CompletionRequest):
    if not router: raise HTTPException(503,"Router not initialized")
    s=get_settings()
    decision=router.classify_and_route(req.messages)
    vcfg=(router.routing or {}).get("verification",{}) or {}
    mode=req.verification_mode or vcfg.get("default_mode","fast")
    if mode not in ("fast","quality"):
        raise HTTPException(422,"verification_mode must be 'fast' or 'quality'")
    try: candidates=router.candidates_for(decision.tier)
    except ValueError as e: raise HTTPException(422,str(e))

    rid="req_"+uuid.uuid4().hex
    started=time.perf_counter()
    prompt_text="\n".join(m.content for m in req.messages)
    prompt_hash=hashlib.sha256(prompt_text.encode()).hexdigest()
    max_tokens=req.max_tokens or 512

    # Candidate chain: primary first, then configured fallbacks.
    # Rules: cloud-disabled filters to local models; the per-request budget
    # filters out candidates whose estimated cost exceeds MAX_REQUEST_COST_USD.
    def est(m): return m.estimate_cost(max(1,len(prompt_text.split())),max_tokens)
    eligible=[m for m in candidates if s.allow_cloud or m.provider=="ollama"]
    if not eligible: raise HTTPException(403,"Cloud providers disabled")
    affordable=[m for m in eligible if est(m) <= s.max_request_cost_usd]
    if not affordable: raise HTTPException(402,"Estimated request cost exceeds MAX_REQUEST_COST_USD")
    intended=candidates[0]   # the model routing actually chose before cloud/budget filtering

    result=None; used=affordable[0]; failures:list[tuple[str,str]]=[]
    for m in affordable:
        try:
            result=await complete_with_retry(req.messages,m,req.temperature,max_tokens)
            used=m; break
        except ProviderError as e:
            failures.append((m.name,str(e))); continue
    if result is None:
        event(logger,"request_failed",request_id=rid,tier=decision.tier,errors=failures)
        raise HTTPException(502,f"All models failed for {decision.tier}: {failures}")

    routing_event=None; original_model=None; fallback_model=None; reason=None
    if used.name != intended.name:
        original_model=intended.name; fallback_model=used.name
        runtime_err=next((r for n,r in failures if n==intended.name), None)
        if runtime_err is not None:
            routing_event="provider_failure"; reason=runtime_err
        elif intended.provider!="ollama" and not s.allow_cloud:
            routing_event="cloud_disabled"; reason="Cloud providers disabled"
        else:
            routing_event="cost_guard"; reason="Estimated request cost exceeds MAX_REQUEST_COST_USD"

    verify_on=bool(s.verification_enabled and vcfg.get("enabled",True))
    final_text=result.text; final_usage=result.usage; final_cost=result.cost_usd
    final_model=used.name; final_provider=result.provider
    quality=None; verified=None; escalated=False; escalation_model=None; queued=False

    if verify_on and mode=="quality":
        # Quality mode: verify before returning, escalate to a premium model when it fails.
        vresult=await verify(rid,req.messages,result.text,decision.tier,used.name)
        if not vresult.skipped:
            quality=vresult.quality_score; verified=bool(vresult.passed)
            if not vresult.passed:
                esc=await escalate(rid,req.messages,used.name,decision.tier,vresult)
                if esc:
                    esc_name, er = esc
                    final_text=er.text; final_usage=_sum_usage(final_usage,er.usage)
                    final_cost+=er.cost_usd; final_model=esc_name
                    final_provider=er.provider; escalated=True; escalation_model=esc_name
    elif verify_on:
        # Fast mode: return immediately, verify in the background.
        queued=True

    status="completed"
    if verify_on and mode=="quality":
        if escalated: status="escalated"
        elif verified is True: status="verified"
        elif verified is False: status="verification_failed"

    latency_ms=(time.perf_counter()-started)*1000
    insert_request({"id":rid,"timestamp":time.time(),"prompt_hash":prompt_hash,"prompt_text":prompt_text[:4000],
                    "complexity_tier":decision.tier,"routed_model":used.name,"provider":final_provider,
                    "input_tokens":final_usage.input_tokens,"output_tokens":final_usage.output_tokens,
                    "cost_usd":final_cost,"latency_ms":latency_ms,"quality_score":quality,
                    "escalated":int(escalated),"escalation_model":escalation_model,"status":status,
                    "verification_mode":mode,"routing_event":routing_event,"original_model":original_model,
                    "fallback_model":used.name if routing_event else None,"fallback_reason":reason})
    if queued:
        await queue.enqueue(VerificationJob(rid,req.messages,result.text,used.name,decision.tier))
    event(logger,"request_completed",request_id=rid,tier=decision.tier,model=final_model,cost=final_cost,
          latency_ms=latency_ms,verification_mode=mode,verification_queued=queued,escalated=escalated,
          routing_event=routing_event)
    return CompletionResponse(id=rid,response=final_text,model=final_model,provider=final_provider,
                              complexity_tier=decision.tier,routing_reason=decision.reason,usage=final_usage,
                              cost_usd=final_cost,latency_ms=latency_ms,verification_queued=queued,
                              verification_mode=mode,quality_score=quality,verified=verified,
                              escalated=escalated,escalation_model=escalation_model,routing_event=routing_event,
                              original_model=original_model,fallback_model=fallback_model,reason=reason)
