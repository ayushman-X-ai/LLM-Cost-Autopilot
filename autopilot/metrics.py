from .db import recent

def baseline_cost(rows, baseline_model):
    total=0.0
    for r in rows:
        inp=r.get("input_tokens") or 0; out=r.get("output_tokens") or 0
        total += baseline_model.estimate_cost(inp,out)
    return total

def savings(rows, baseline_model):
    actual=sum((r.get("cost_usd") or 0) for r in rows)
    base=baseline_cost(rows,baseline_model)
    return {"actual_cost":actual,"baseline_cost":base,"savings":base-actual,"savings_pct":((base-actual)/base*100 if base else 0)}
