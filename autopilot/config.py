from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model_id: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    average_latency_ms: float
    quality_tier: str
    enabled: bool = True
    pricing_updated_at: str | None = None
    pricing_notes: str | None = None

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000) * self.input_cost_per_1k + (output_tokens / 1000) * self.output_cost_per_1k

def load_models(path: str) -> dict[str, ModelConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {name: ModelConfig(name=name, **cfg) for name, cfg in raw.get("models", {}).items()}

def load_routing(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
