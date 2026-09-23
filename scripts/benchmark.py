"""Benchmark every configured model on identical prompts.

For each model x prompt the script records cost, latency and (optionally)
LLM-judged quality on a 1-5 scale, then writes:
    artifacts/benchmark.json          raw records
    artifacts/benchmark_report.md     cost / latency / quality comparison table

Usage:
    python scripts/benchmark.py --limit 5            # judge on by default
    python scripts/benchmark.py --limit 5 --no-judge # skip quality judging
"""
import argparse, asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python scripts/benchmark.py`
from autopilot.config import load_models, load_routing
from autopilot.models import Message
from autopilot.providers import complete_with_retry, ProviderError
from autopilot.settings import get_settings
from autopilot.verifier import VERIFY_PROMPT, parse_verifier_payload, weights_from

PROMPTS=[
"Rewrite this sentence professionally: please send the report.",
"Extract the customer's email address and order number from this message: Order 8841 was placed by sam@example.com.",
"Summarize the following text in five bullet points: Artificial intelligence is changing software engineering by making code generation and review faster, but teams still need evaluation and governance.",
"Compare event-driven architecture and request-response architecture for a globally distributed notification system, including failure modes, delivery semantics, operational tradeoffs and a recommendation for three workload patterns.",
"Design a multi-tenant AI gateway that routes requests across three model providers, enforces per-tenant budgets, performs asynchronous quality verification, supports streaming, and remains observable under provider outages. Explain the data model, consistency boundaries, retry policy and deployment topology.",
]

async def judge_one(judge_model, prompt: str, answer: str, weights: dict) -> float | None:
    payload=[Message(role="system",content=VERIFY_PROMPT),
             Message(role="user",content=json.dumps({"request":[{"role":"user","content":prompt}],"candidate":answer},ensure_ascii=False))]
    try:
        jr=await complete_with_retry(payload, judge_model, temperature=0, max_tokens=400)
    except ProviderError:
        return None
    parsed=parse_verifier_payload(jr.text, weights)
    if parsed.error: return None
    return parsed.quality_score

async def main(limit: int, judge: bool):
    s=get_settings()
    dry_run=s.dry_run
    models=load_models(s.models_config_path)
    routing=load_routing(s.routing_config_path)
    prompts=PROMPTS[:limit]
    judge_model=models.get((routing.get("verification",{}) or {}).get("verifier_model")) if judge else None
    judge_weights=weights_from(routing)   # judge with the configured quality weights, not hard-coded defaults

    records=[]
    for name,m in models.items():
        if not m.enabled: continue
        for p in prompts:
            try:
                r=await complete_with_retry([Message(role="user",content=p)],m,max_tokens=300)
                rec={"model":name,"provider":m.provider,"model_id":m.model_id,"prompt":p,
                     "latency_ms":r.latency_ms,"cost_usd":r.cost_usd,
                     "input_tokens":r.usage.input_tokens,"output_tokens":r.usage.output_tokens,
                     "text":r.text[:500]}
                if judge_model is not None:
                    rec["quality"]=await judge_one(judge_model, p, r.text, judge_weights)
                records.append(rec)
            except ProviderError as e:
                records.append({"model":name,"provider":m.provider,"prompt":p,"error":str(e)})

    # aggregate per model
    table=[]
    for name,m in models.items():
        ok=[r for r in records if r["model"]==name and "error" not in r]
        err=[r for r in records if r["model"]==name and "error" in r]
        if not ok and not err: continue
        qualities=[r["quality"] for r in ok if r.get("quality") is not None]
        table.append({
            "model":name,"provider":m.provider,"model_id":m.model_id,
            "requests":len(ok)+len(err),"errors":len(err),
            "avg_cost_usd":round(sum(r["cost_usd"] for r in ok)/len(ok),6) if ok else None,
            "avg_latency_ms":round(sum(r["latency_ms"] for r in ok)/len(ok),1) if ok else None,
            "avg_quality":round(sum(qualities)/len(qualities),2) if qualities else None,
            "pricing_updated_at":m.pricing_updated_at,
        })

    def fmt(v, spec="{:.2f}", na="-"):
        return na if v is None else spec.format(v)
    lines=[
        "# Model benchmark",
        "",
        f"Prompts per model: {len(prompts)} | generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Quality judged by: {judge_model.name if judge_model else 'disabled'}",
        ("Mode: DRY_RUN (simulated responses billed at registry prices - not real provider costs)" if dry_run else "Mode: live provider calls"),
        "",
        "| Model | Provider | Cost / request | Avg latency | Quality (1-5) | Errors |",
        "|---|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(f"| {row['model']} ({row['model_id']}) | {row['provider']} | "
                     f"${fmt(row['avg_cost_usd'],'{:.6f}')} | {fmt(row['avg_latency_ms'],'{:.0f}')} ms | "
                     f"{fmt(row['avg_quality'])} / 5 | {row['errors']} |")
    report_md="\n".join(lines)+"\n"

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/benchmark.json").write_text(json.dumps({"records":records,"table":table},indent=2),encoding="utf-8")
    Path("artifacts/benchmark_report.md").write_text(report_md,encoding="utf-8")
    print(report_md)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--limit",type=int,default=5)
    p.add_argument("--judge",action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args(); asyncio.run(main(a.limit,a.judge))
