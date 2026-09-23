import shutil
from pathlib import Path
import pytest
from autopilot.settings import get_settings

ARTIFACT_SRC = Path("artifacts/complexity_classifier.joblib")

@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets a clean temp database, JSONL paths, and dry-run providers."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "logs.jsonl"))
    monkeypatch.setenv("ROUTING_FAILURES_PATH", str(tmp_path / "routing_failures.jsonl"))
    monkeypatch.setenv("EVALUATION_RESULTS_PATH", str(tmp_path / "evaluation_results.jsonl"))
    monkeypatch.setenv("EVAL_DATASET_PATH", str(tmp_path / "routing_eval.jsonl"))
    monkeypatch.setenv("EVAL_REPORT_PATH", str(tmp_path / "routing_eval_report.json"))
    monkeypatch.setenv("VERIFICATION_ENABLED", "true")
    monkeypatch.setenv("ALLOW_CLOUD", "true")
    monkeypatch.setenv("MAX_REQUEST_COST_USD", "0.25")
    monkeypatch.setenv("PROVIDER_MAX_ATTEMPTS", "3")
    # Copy the trained classifier so tests never clobber the repo artifact.
    artifact = tmp_path / "classifier.joblib"
    if ARTIFACT_SRC.exists():
        shutil.copy(ARTIFACT_SRC, artifact)
    monkeypatch.setenv("CLASSIFIER_ARTIFACT", str(artifact))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from autopilot.api import app
    with TestClient(app) as c:
        yield c

@pytest.fixture
def force_tier(monkeypatch):
    """Force the router to a specific tier/model so API tests stay deterministic."""
    def _force(tier: str, model_name: str | None = None):
        import autopilot.api as api
        from autopilot.config import load_routing
        from autopilot.models import RoutingDecision
        assert api.router is not None, "declare the `client` fixture first"
        routing = load_routing(get_settings().routing_config_path)
        name = model_name or routing["tiers"][tier]["model"]
        def fake(messages):
            return RoutingDecision(tier=tier, model_name=name,
                                   reason=f"forced {tier} for test",
                                   estimated_cost_usd=0.0, features={"classifier_confidence": 0.99})
        monkeypatch.setattr(api.router, "classify_and_route", fake)
    return _force

def make_response(model, text="ok", cost=None, usage_tokens=(10, 20)):
    from autopilot.models import LLMResponse, Usage
    inp, out = usage_tokens
    usage = Usage(input_tokens=inp, output_tokens=out, total_tokens=inp + out)
    return LLMResponse(text=text, model=model.model_id, provider=model.provider, usage=usage,
                       latency_ms=5.0, cost_usd=cost if cost is not None else model.estimate_cost(inp, out),
                       finish_reason="stop")
