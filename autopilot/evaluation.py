"""Routing evaluation: measure classifier/routing quality on a held-out labeled set.

Produces accuracy, per-tier precision/recall/F1, a confusion matrix, per-tier
accuracy and the routing error rate. Each run is appended to
data/evaluation_results.jsonl and a full report is written to
artifacts/routing_eval_report.json.
"""
import json, time
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from .classifier import ComplexityClassifier
from .feedback import append_jsonl, load_jsonl
from .settings import get_settings

def _text(row: dict) -> str:
    return row.get("text") or row.get("prompt") or ""

def _expected(row: dict) -> str:
    return row.get("tier") or row.get("expected_tier") or ""

def evaluate(dataset_path: str | None = None, artifact_path: str | None = None,
             classifier: ComplexityClassifier | None = None, write_report: bool = True) -> dict:
    s=get_settings()
    path=dataset_path or s.eval_dataset_path
    rows=load_jsonl(path)
    if not rows: raise ValueError(f"No evaluation rows found in {path}")
    clf=classifier or ComplexityClassifier(artifact_path or s.classifier_artifact)
    y_true=[_expected(r) for r in rows]
    y_pred=[clf.predict(_text(r))[0] for r in rows]
    labels=sorted(set(y_true))
    accuracy=float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_tier={label: {"precision": float(precision[i]), "recall": float(recall[i]),
                      "f1": float(f1[i]), "support": int(support[i])}
              for i, label in enumerate(labels)}
    # Per-tier accuracy = share of that tier's prompts routed correctly (i.e. recall).
    tier_accuracy={label: per_tier[label]["recall"] for label in labels}
    report={
        "timestamp": time.time(),
        "dataset": path,
        "n": len(rows),
        "labels": labels,
        "accuracy": accuracy,
        "routing_error_rate": round(1.0 - accuracy, 6),
        "per_tier": per_tier,
        "tier_accuracy": tier_accuracy,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
        "errors": [{"id": rows[i].get("id"), "text": _text(rows[i])[:200],
                    "expected": y_true[i], "predicted": y_pred[i]}
                   for i in range(len(rows)) if y_true[i] != y_pred[i]],
    }
    if write_report:
        out=Path(s.eval_report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary={k: report[k] for k in ("timestamp","dataset","n","labels","accuracy",
                                        "routing_error_rate","per_tier","tier_accuracy","confusion_matrix")}
        summary["id"]="eval_"+str(int(report["timestamp"]))
        append_jsonl(s.evaluation_results_path, summary)
    return report
