"""Load-test the running API and produce a full experiment report.

Sends a mixed prompt load (all three routing tiers), then reports success
rates, latency percentiles, routed cost vs premium-model baseline,
escalation rate and quality pass rate.

Usage (API must be running):
    python scripts/load_test.py --requests 1000 --concurrency 20
    python scripts/load_test.py --requests 500 --verification-mode quality

Writes artifacts/load_test_report.txt and artifacts/load_test_report.json.
"""
import argparse, asyncio, json, statistics, sys, time
from pathlib import Path
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/load_test.py`
from autopilot.config import load_models, load_routing
from autopilot.settings import get_settings

PROMPTS=[
    "Rewrite this sentence professionally: please send me the report.",          # tier_1
    "Extract the order number from this message: Order 8841 was placed by sam@example.com.",  # tier_1
    "Translate this sentence into Spanish: the meeting moved to tuesday.",        # tier_1
    "Summarize the following text in five bullet points: Artificial intelligence is changing software engineering by making code generation and review faster, but teams still need evaluation and governance.",  # tier_2
    "Draft a polite follow-up email to a client about the quarterly roadmap that includes a clear call to action.",  # tier_2
    "Explain how rate limiting works to a non-technical product manager in two paragraphs.",  # tier_2
    "Design a multi-tenant AI gateway that routes requests across three providers, enforces per-tenant budgets, performs asynchronous quality verification, and stays observable during provider outages. Cover the data model, retry policy and deployment topology.",  # tier_3
    "Compare Raft, Paxos and EPaxos for a geo-distributed database and justify the choice for a payments platform with strict audit requirements.",  # tier_3
    "Analyze the tradeoffs between event sourcing and CRUD for order management under high write throughput, include failure-mode analysis, and recommend one.",  # tier_3
]

def percentile(values:list[float], p:float)->float:
    if not values: return 0.0
    ordered=sorted(values)
    idx=max(0,min(len(ordered)-1,int(round(p/100*(len(ordered)-1)))))
    return ordered[idx]

async def snapshot(client, base_url)->dict:
    try:
        r=await client.get(f"{base_url}/v1/stats")
        return r.json() if r.status_code==200 else {}
    except Exception:
        return {}

async def one(client, base_url, sem, payload):
    async with sem:
        t=time.perf_counter()
        try:
            r=await client.post(f"{base_url}/v1/completions",json=payload)
            return {"ok":r.status_code==200,"status":r.status_code,
                    "latency_ms":(time.perf_counter()-t)*1000,
                    "body":r.json() if r.status_code==200 else {"error":r.text[:300]}}
        except Exception as e:
            return {"ok":False,"status":0,"latency_ms":(time.perf_counter()-t)*1000,"body":{"error":str(e)}}

def db_quality_pass_rate(threshold:float)->float|None:
    """Share of verified requests at/above the quality threshold (read from the audit DB)."""
    try:
        from autopilot.db import connect
        with connect() as c:
            total=c.execute("SELECT COUNT(*) n FROM requests WHERE quality_score IS NOT NULL").fetchone()["n"]
            if not total: return None
            ok=c.execute("SELECT COUNT(*) n FROM requests WHERE quality_score >= ?",(threshold,)).fetchone()["n"]
            return round(ok/total*100,2)
    except Exception:
        return None

async def main(n:int, c:int, base_url:str, mode:str|None):
    payload_base={"messages":[{"role":"user","content":PROMPTS[0]}]}
    if mode: payload_base["verification_mode"]=mode
    sem=asyncio.Semaphore(c)
    async with httpx.AsyncClient(timeout=300) as client:
        before=await snapshot(client,base_url)
        start=time.perf_counter()
        tasks=[one(client,base_url,sem,{**payload_base,"messages":[{"role":"user","content":PROMPTS[i%len(PROMPTS)]}]}) for i in range(n)]
        results=await asyncio.gather(*tasks)
        wall=time.perf_counter()-start
        after=await snapshot(client,base_url)

    ok=[r for r in results if r["ok"]]
    failed=[r for r in results if not r["ok"]]
    lat=[r["latency_ms"] for r in ok]

    # cost comparison from the audit trail (before/after deltas)
    s=get_settings(); models=load_models(s.models_config_path); routing=load_routing(s.routing_config_path)
    baseline=models.get((routing.get("metrics",{}) or {}).get("baseline_model")) or max(models.values(),key=lambda m:m.input_cost_per_1k+m.output_cost_per_1k)
    def field(d,k): return float(d.get(k) or 0)
    run_cost=field(after,"cost_usd")-field(before,"cost_usd")
    run_in=int(field(after,"input_tokens")-field(before,"input_tokens"))
    run_out=int(field(after,"output_tokens")-field(before,"output_tokens"))
    baseline_cost=baseline.estimate_cost(run_in,run_out)
    savings=baseline_cost-run_cost
    savings_pct=(savings/baseline_cost*100) if baseline_cost else 0.0
    run_requests=max(1,int(field(after,"requests")-field(before,"requests")))
    escalations=field(after,"escalations")-field(before,"escalations")
    escalation_rate=escalations/run_requests*100
    threshold=float((routing.get("tiers",{}).get("tier_2",{}) or {}).get("quality_threshold",
                    (routing.get("verification",{}) or {}).get("quality_threshold",4.0)))
    quality_pass=db_quality_pass_rate(threshold)
    # Quality parity: share of verified requests that reached the premium
    # quality bar (tier threshold). This is the "did routing preserve quality?"
    # headline metric next to cost savings.

    tier_counts={}
    model_counts={}
    for r in ok:
        b=r["body"]; tier_counts[b.get("complexity_tier","?")]=tier_counts.get(b.get("complexity_tier","?"),0)+1
        model_counts[b.get("model","?")]=model_counts.get(b.get("model","?"),0)+1
    errors={}
    for r in failed:
        key=str(r["status"]); errors[key]=errors.get(key,0)+1

    report={
        "generated_at":time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "dry_run_simulation" if get_settings().dry_run else "live",
        "requests":n,"concurrency":c,"verification_mode":mode or "default",
        "successful":len(ok),"failed":len(failed),"error_status_counts":errors,
        "wall_clock_s":round(wall,2),"throughput_rps":round(len(ok)/wall,2) if wall else 0,
        "latency_ms":{"avg":round(statistics.fmean(lat),1) if lat else 0,
                      "p50":round(percentile(lat,50),1),"p95":round(percentile(lat,95),1),
                      "p99":round(percentile(lat,99),1),"max":round(max(lat),1) if lat else 0},
        "cost":{"baseline_model":baseline.name,"baseline_cost_usd":round(baseline_cost,6),
                "autopilot_cost_usd":round(run_cost,6),"saved_usd":round(savings,6),
                "cost_reduction_pct":round(savings_pct,2),
                "input_tokens":run_in,"output_tokens":run_out},
        "escalation_rate_pct":round(escalation_rate,2),
        "quality_threshold":threshold,
        "quality_parity_pct":quality_pass,
        "quality_pass_rate_pct":quality_pass,
        "tier_distribution":tier_counts,"model_distribution":model_counts,
    }
    lines=[
        "LOAD TEST REPORT",
        "="*60,
        f"Mode:                {'DRY RUN (simulated responses, registry prices - not real provider costs)' if get_settings().dry_run else 'live provider calls'}",
        f"Requests:            {n:>10}",
        f"Successful:          {len(ok):>10}",
        f"Failed:              {len(failed):>10}",
        f"Concurrency:         {c:>10}",
        f"Verification mode:   {(mode or 'default'):>10}",
        "",
        f"Average latency:     {report['latency_ms']['avg']:>10.1f} ms",
        f"P50:                 {report['latency_ms']['p50']:>10.1f} ms",
        f"P95:                 {report['latency_ms']['p95']:>10.1f} ms",
        f"P99:                 {report['latency_ms']['p99']:>10.1f} ms",
        "",
        f"Baseline cost:       ${baseline_cost:>9.4f}  (everything on {baseline.name})",
        f"Autopilot cost:      ${run_cost:>9.4f}",
        f"Savings:             ${savings:>9.4f}  ({savings_pct:.2f}%)",
        "",
        f"Escalation rate:     {escalation_rate:>9.2f}%",
        f"Quality parity:      " + ((f"{quality_pass:>8.1f}%  (verified >= threshold {threshold})") if quality_pass is not None else "      n/a"),
        "",
        f"Tier distribution:   {tier_counts}",
        f"Model distribution:  {model_counts}",
    ]
    text="\n".join(lines)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/load_test_report.txt").write_text(text+"\n",encoding="utf-8")
    Path("artifacts/load_test_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(text)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--requests",type=int,default=1000)
    p.add_argument("--concurrency",type=int,default=20)
    p.add_argument("--base-url",default="http://127.0.0.1:8000")
    p.add_argument("--verification-mode",choices=["fast","quality"],default=None)
    a=p.parse_args()
    asyncio.run(main(a.requests,a.concurrency,a.base_url,a.verification_mode))
