"""Prompt-complexity classifier (tier_1 simple / tier_2 moderate / tier_3 hard).

Pipeline: TF-IDF (word 1-2 grams + char 3-5 grams, sublinear) + engineered
numeric features -> Logistic Regression (class_weight="balanced").

The training metrics report stratified 5-fold cross-validation, not fit-set
accuracy: fit-set accuracy on template-heavy data reads ~100% while held-out
accuracy was 76.6%. The CV number is the honest headline metric.
"""
from pathlib import Path
import json, re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .settings import get_settings

FEATURE_NAMES = [
    "char_length", "word_count", "avg_word_length", "question_count",
    "imperative_count", "constraint_count", "constraint_clause_count",
    "numeric_tokens", "has_context", "has_code", "reasoning", "comparison",
    "structured_output", "planning_depth", "reasoning_words", "task_words",
]

# Words that signal "produce an artifact / explain / plan" rather than transform.
_TASK_WORDS = [
    "design", "architect", "plan", "strategy", "evaluate", "compare", "analyze",
    "synthes", "optimiz", "optimis", "diagnos", "debug", "refactor", "migrat",
    "threat model", "tradeoff", "trade-off", "prove", "derive", "justify",
    "recommend", "review", "rewrite", "draft", "summariz", "summaris", "explain",
    "convert", "translate", "extract", "classify", "categoriz",
]

_REASONING_WORDS = [
    "step by step", "reason", "derive", "prove", "justify", "why", "implication",
    "consequence", "tradeoff", "trade-off", "analyze", "evaluate", "in-depth",
    "rigorous", "failure mode", "edge case", "invariant", "consistency",
]

# Words that signal multi-part, long-horizon work (tier_3 flavor).
_PLANNING_WORDS = [
    "architecture", "multi-tenant", "distributed", "scalability", "resilien",
    "zero-downtime", "rollout", "migration", "exactly-once", "idempoten",
    "threat model", "threat-model", "stride", "slo", "error budget", "backfill",
    "cutover", "failover", "partition", "throughput", "audit require",
    "data residency", "compliance", "chaos", "geo-distributed", "consensus",
]

# Coarse one-word actions typical of tier_1 ("fix grammar", "translate this").
_IMPERATIVES = [
    "rewrite", "extract", "convert", "translate", "fix", "sort", "list", "format",
    "count", "spellcheck", "reformat", "lowercase", "uppercase", "capitalize",
    "truncate", "title-case", "titlecase", "reverse", "trim", "categorize",
    "categorise", "label",
]

_CONSTRAINT_TERMS = [
    "must ", "exactly ", "only ", "at least ", "no more than ", "at most ",
    "include ", "exclude ", "under ", "within ", "keep ", "cover ",
    "each ", "every ", "per ", "while preserving", "without changing",
]

_STRUCTURED_TERMS = [
    "json", "table", "schema", "columns", "bullet points", "format as",
    "markdown", "csv", "yaml", "acceptance criteria", "action item",
]


def _count_terms(low: str, terms: list[str]) -> int:
    return sum(low.count(t) for t in terms)


def extract_features(text: str) -> dict:
    """Engineered numeric features describing shape, task type and constraints.

    All string matching is case-insensitive; these features complement the
    TF-IDF signal and are exposed through the API for auditability.
    """
    low = text.lower()
    words = text.split()
    n_words = len(words)
    return {
        "char_length": len(text),
        "word_count": n_words,
        "avg_word_length": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "question_count": text.count("?"),
        "imperative_count": _count_terms(low, _IMPERATIVES),
        "constraint_count": _count_terms(low, _CONSTRAINT_TERMS),
        "constraint_clause_count": len(re.findall(r"[,;]", text)) + low.count(" and "),
        "numeric_tokens": len(re.findall(r"\d", text)),
        "has_context": int(any(x in low for x in ["context:", "document:", "passage:", "here is the text", "following text", "below"])),
        "has_code": int(bool(re.search(r"```|def |class |select |function |import |update |join ", text, re.I))),
        "reasoning": int(any(x in low for x in _REASONING_WORDS)),
        "comparison": int(any(x in low for x in ["compare ", "versus", " vs ", "difference between", "trade-offs", "tradeoffs", "pros and cons", "weigh"])),
        "structured_output": int(any(x in low for x in _STRUCTURED_TERMS)),
        "planning_depth": _count_terms(low, _PLANNING_WORDS),
        "reasoning_words": _count_terms(low, _REASONING_WORDS),
        "task_words": _count_terms(low, _TASK_WORDS),
    }


def _frame(texts: list[str]) -> pd.DataFrame:
    """Wrap raw texts plus engineered features in a DataFrame for the pipeline."""
    feats = [extract_features(t) for t in texts]
    df = pd.DataFrame({"text": texts})
    for name in FEATURE_NAMES:
        df[name] = [f[name] for f in feats]
    return df


def _build_pipeline() -> Pipeline:
    vectorizer = ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                 min_df=1, max_features=20000), "text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 sublinear_tf=True, min_df=2, max_features=20000), "text"),
        ("num", StandardScaler(with_mean=False), FEATURE_NAMES),
    ])
    clf = LogisticRegression(max_iter=5000, C=2.0, class_weight="balanced")
    return Pipeline([("features", vectorizer), ("model", clf)])


def train(dataset_path="data/complexity_dataset.jsonl", artifact_path=None, extra_rows=None):
    """Train the classifier and persist it with honest CV metrics.

    Metrics are computed with stratified 5-fold cross-validation over the full
    labeled set (training rows + any feedback rows), which estimates held-out
    performance instead of memorization accuracy.
    """
    artifact_path = artifact_path or get_settings().classifier_artifact
    rows = [json.loads(x) for x in Path(dataset_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    n_base = len(rows)
    # Feedback-loop rows (routing failures) are merged in, deduplicated by text.
    if extra_rows:
        seen = {r["text"] for r in rows}
        for r in extra_rows:
            if r.get("text") and r.get("tier") and r["text"] not in seen:
                rows.append(r); seen.add(r["text"])
    texts = [r["text"] for r in rows]
    y = np.array([r["tier"] for r in rows])

    pipe = _build_pipeline()
    labels = sorted(set(y))
    n_min = int(min((y == lab).sum() for lab in labels))
    # Honest estimate: out-of-fold predictions across stratified folds.
    if n_min >= 5 and len(labels) > 1:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = _frame(texts)
        oof = cross_val_predict(pipe, X, y, cv=folds)
        cv_acc = float(accuracy_score(y, oof))
        cv_report = classification_report(y, oof, output_dict=True, zero_division=0)
        cv_cm = confusion_matrix(y, oof, labels=labels).tolist()
    else:
        cv_acc, cv_report, cv_cm = None, None, None  # too few rows for 5-fold CV

    # Final fit on all rows.
    pipe.fit(_frame(texts), y)
    pred = pipe.predict(_frame(texts))
    metrics = {
        "accuracy_training": float(accuracy_score(y, pred)),
        "accuracy_cv": cv_acc,
        "cv_folds": 5 if cv_acc is not None else None,
        "n_training": len(rows), "n_base": n_base,
        "n_feedback_rows": len(rows) - n_base, "labels": labels,
        "confusion_matrix_cv": cv_cm,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "report": classification_report(y, pred, output_dict=True, zero_division=0),
        "report_cv": cv_report,
    }
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": pipe, "classifier": pipe, "features": FEATURE_NAMES,
                 "metrics": metrics, "feature_extractor": extract_features}, artifact_path)
    Path(artifact_path).with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


class ComplexityClassifier:
    def __init__(self, artifact_path=None):
        self.path = artifact_path or get_settings().classifier_artifact
        if not Path(self.path).exists():
            train(artifact_path=self.path)
        self.bundle = joblib.load(self.path)

    def predict(self, text: str):
        """Return (tier, confidence, features) for a prompt.

        Uses the trained pipeline's class probabilities; confidence is the
        max class probability.
        """
        proba = self.bundle["classifier"].predict_proba(_frame([text]))[0]
        classes = self.bundle["classifier"].classes_
        idx = int(np.argmax(proba))
        return classes[idx], float(proba[idx]), extract_features(text)
