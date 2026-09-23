from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    database_path: str = "data/autopilot.db"
    models_config_path: str = "config/models.yaml"
    routing_config_path: str = "config/routing.yaml"
    classifier_artifact: str = "artifacts/complexity_classifier.joblib"
    log_path: str = "logs/autopilot.jsonl"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    dry_run: bool = False
    verification_enabled: bool = True
    allow_cloud: bool = True
    max_request_cost_usd: float = 0.25
    provider_timeout_seconds: float = 60.0
    provider_max_attempts: int = 3
    routing_failures_path: str = "data/routing_failures.jsonl"
    eval_dataset_path: str = "data/routing_eval.jsonl"
    eval_report_path: str = "artifacts/routing_eval_report.json"
    evaluation_results_path: str = "data/evaluation_results.jsonl"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for p in [s.database_path, s.models_config_path, s.routing_config_path, s.classifier_artifact, s.log_path]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    return s
