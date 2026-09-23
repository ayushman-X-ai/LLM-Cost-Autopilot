from autopilot.classifier import extract_features, train, ComplexityClassifier

def test_features():
    f=extract_features("Compare A and B and explain the tradeoffs in a JSON table.")
    assert f["comparison"]==1 and f["structured_output"]==1

def test_feature_flags():
    f=extract_features("Reason step by step about this code: ```def x(): pass``` why does it fail? Must include tests.")
    assert f["has_code"]==1 and f["reasoning"]==1 and f["constraint_count"]>=1

def test_training(tmp_path):
    out=tmp_path/"model.joblib"; m=train("data/complexity_dataset.jsonl",str(out)); assert out.exists(); assert m["accuracy_training"]>0.5

def test_training_reports_cv_not_just_fit_accuracy(tmp_path):
    """Fit-set accuracy is ~100% on template data; CV is the honest metric."""
    out=tmp_path/"model.joblib"; m=train("data/complexity_dataset.jsonl",str(out))
    assert m["accuracy_cv"] is not None
    assert 0.5 < m["accuracy_cv"] < 1.0, "CV should be a realistic held-out estimate"
    assert m["confusion_matrix_cv"] is not None

def test_prediction_confidence_is_calibrated_probability(tmp_path):
    out=tmp_path/"model.joblib"; train("data/complexity_dataset.jsonl",str(out))
    c=ComplexityClassifier(str(out))
    tier,conf,_=c.predict("Rewrite this sentence professionally.")
    assert tier in {"tier_1","tier_2","tier_3"} and 0.34 <= conf <= 1.0

def test_training_with_feedback_rows(tmp_path):
    out=tmp_path/"model.joblib"
    extra=[{"id":"f1","text":"A failing prompt that should route higher now.","tier":"tier_3",
            "source":"routing_failure","reviewed":False}]
    m=train("data/complexity_dataset.jsonl",str(out),extra_rows=extra)
    assert m["n_feedback_rows"]==1
    assert m["n_training"]==m["n_base"]+1

def test_training_dedupes_feedback_rows(tmp_path):
    out=tmp_path/"model.joblib"
    base="Rewrite this sentence in a professional tone."
    extra=[{"id":"dup","text":base,"tier":"tier_1"}]
    m=train("data/complexity_dataset.jsonl",str(out),extra_rows=extra)
    assert m["n_feedback_rows"]==0

def test_prediction():
    c=ComplexityClassifier("artifacts/complexity_classifier.joblib")
    tier,conf,features=c.predict("Rewrite this sentence professionally.")
    assert tier in {"tier_1","tier_2","tier_3"} and 0<=conf<=1

def test_prediction_complex_prompt_uses_features():
    c=ComplexityClassifier()
    tier,conf,features=c.predict(
        "Design a distributed scheduler with exactly-once semantics across regions; "
        "compare approaches, prove correctness step by step, and output a JSON schema.")
    assert features["reasoning"]==1 and features["comparison"]==1
    assert conf>0
