from .classifier import ComplexityClassifier
from .config import ModelConfig, load_models, load_routing
from .models import Message, RoutingDecision
from .settings import get_settings

class CostRouter:
    def __init__(self):
        s=get_settings(); self.models=load_models(s.models_config_path); self.routing=load_routing(s.routing_config_path); self.classifier=ComplexityClassifier(s.classifier_artifact)

    def tier_config(self, tier: str) -> dict:
        cfg=self.routing.get("tiers",{}).get(tier)
        if not cfg: raise ValueError(f"No routing rule for {tier}")
        return cfg

    def candidates_for(self, tier: str) -> list[ModelConfig]:
        """Primary model first, then configured fallbacks (enabled models only, no duplicates)."""
        cfg=self.tier_config(tier)
        names=[cfg.get("model"), *cfg.get("fallbacks",[])]
        out: list[ModelConfig] = []
        for name in names:
            if not name: continue
            model=self.models.get(name)
            if model and model.enabled and all(model.name != m.name for m in out):
                out.append(model)
        if not out: raise ValueError(f"No enabled models configured for {tier}")
        return out

    def classify_and_route(self,messages:list[Message]):
        text="\n".join(f"{m.role}: {m.content}" for m in messages)
        tier,confidence,features=self.classifier.predict(text)
        cfg=self.tier_config(tier)
        model=self.models[cfg["model"]]
        if not model.enabled: raise ValueError(f"Model {model.name} disabled")
        estimated=model.estimate_cost(max(1,len(text.split())),512)
        reason=f"{tier} complexity (classifier confidence {confidence:.2f}); routed to configured {model.quality_tier}-quality model"
        return RoutingDecision(tier=tier,model_name=model.name,reason=reason,estimated_cost_usd=estimated,features={**features,"classifier_confidence":confidence})
