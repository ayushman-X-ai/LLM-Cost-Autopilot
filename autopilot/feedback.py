"""Routing-failure feedback loop.

Every failed verification becomes a candidate training example:

    routing failure -> routing_failures.jsonl -> retrain -> better routing

Records are written as JSONL under data/ so they stay inspectable and
versionable; `python -m autopilot.cli retrain` merges them into the next
training run.
"""
import json, time, uuid
from pathlib import Path
from .settings import get_settings

TIER_ORDER = ["tier_1", "tier_2", "tier_3"]

def next_tier(tier: str) -> str:
    """Tier one step up the ladder (capped at tier_3) - the assumed correct routing for a failure."""
    if tier not in TIER_ORDER: return tier
    return TIER_ORDER[min(TIER_ORDER.index(tier) + 1, len(TIER_ORDER) - 1)]

def append_jsonl(path: str | Path, row: dict) -> None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False)+"\n")

def load_jsonl(path: str | Path) -> list[dict]:
    p=Path(path)
    if not p.exists(): return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def record_routing_failure(text: str, routed_tier: str, routed_model: str, quality_score: float,
                           issues: list[str] | None = None, request_id: str | None = None,
                           threshold: float | None = None, agreement: float | None = None,
                           rationale: str | None = None) -> dict:
    """Persist a routing failure as a candidate training example for retraining."""
    corrected=next_tier(routed_tier)
    row={
        "id": "fail_"+uuid.uuid4().hex,
        "timestamp": time.time(),
        "request_id": request_id,
        "text": text,
        "tier": corrected,                 # proposed label used by retrain
        "routed_tier": routed_tier,
        "corrected_tier": corrected,
        "routed_model": routed_model,
        "quality_score": quality_score,
        "threshold": threshold,
        "agreement": agreement,
        "issues": issues or [],
        "rationale": rationale,
        "label_changed": corrected != routed_tier,
        "source": "routing_failure",
        "reviewed": False,
    }
    append_jsonl(get_settings().routing_failures_path, row)
    return row

def load_routing_failures(path: str | None = None) -> list[dict]:
    return load_jsonl(path or get_settings().routing_failures_path)
