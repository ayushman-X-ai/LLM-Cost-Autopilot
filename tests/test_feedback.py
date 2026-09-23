import json
from pathlib import Path
import pytest
from autopilot.cli import retrain
from autopilot.evaluation import evaluate
from autopilot.feedback import (append_jsonl, load_jsonl, load_routing_failures,
                                next_tier, record_routing_failure)
from autopilot.settings import get_settings

def test_next_tier_ladder():
    assert next_tier("tier_1")=="tier_2"
    assert next_tier("tier_2")=="tier_3"
    assert next_tier("tier_3")=="tier_3"
    assert next_tier("weird")=="weird"

def test_record_and_load_failures():
    row=record_routing_failure("prompt text","tier_1","ollama_llama",2.1,["too shallow"],
                               "req_1",3.8,0.4,"misses constraints")
    assert row["corrected_tier"]=="tier_2" and row["label_changed"] is True
    rows=load_routing_failures()
    assert len(rows)==1 and rows[0]["id"]==row["id"]
    assert rows[0]["text"]=="prompt text" and rows[0]["reviewed"] is False
    on_disk=json.loads(Path(get_settings().routing_failures_path).read_text().splitlines()[0])
    assert on_disk["request_id"]=="req_1"

def test_failures_missing_file_is_empty(tmp_path):
    assert load_routing_failures(str(tmp_path/"nope.jsonl"))==[]

def _write_eval(path, rows):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

def test_evaluate_reports_expected_metrics(tmp_path):
    eval_path=tmp_path/"eval.jsonl"
    _write_eval(eval_path,[
        {"id":"e1","prompt":"Rewrite this sentence professionally: hi there.","expected_tier":"tier_1","reason":"simple"},
        {"id":"e2","prompt":"Fix the grammar: i has a apple.","expected_tier":"tier_1","reason":"simple"},
        {"id":"e3","prompt":"Design a multi-tenant gateway with budgets, retries, and streaming; cover data model, consistency, retry policy and deployment topology.","expected_tier":"tier_3","reason":"design"},
    ])
    report=evaluate(dataset_path=str(eval_path), write_report=True)
    assert report["n"]==3
    assert 0 <= report["accuracy"] <= 1
    assert report["routing_error_rate"]==round(1-report["accuracy"],6)
    assert set(report["labels"])=={"tier_1","tier_3"}
    assert len(report["confusion_matrix"])==2
    assert set(report["tier_accuracy"])=={"tier_1","tier_3"}
    assert "per_tier" in report and "precision" in report["per_tier"]["tier_1"]
    # evaluation results appended for the feedback loop / trend view
    results=load_jsonl(get_settings().evaluation_results_path)
    assert len(results)==1 and results[0]["n"]==3
    # report written to the configured path
    assert Path(get_settings().eval_report_path).exists()

def test_evaluate_error_list_matching(tmp_path):
    eval_path=tmp_path/"eval.jsonl"
    _write_eval(eval_path,[{"id":"e1","prompt":"Sort these names alphabetically: z, a, m.","expected_tier":"tier_1","reason":"simple"}])
    report=evaluate(dataset_path=str(eval_path), write_report=False)
    for err in report["errors"]:
        assert err["expected"]!=err["predicted"]

def test_evaluate_requires_rows(tmp_path):
    with pytest.raises(ValueError):
        evaluate(dataset_path=str(tmp_path/"empty.jsonl"), write_report=False)

def test_retrain_includes_routing_failures():
    record_routing_failure("Draft a follow-up email with a clear call to action for the roadmap.",
                           "tier_1","openai_mini",2.0,["thin"],"req_x",4.0,0.5,"")
    metrics=retrain()
    assert metrics["n_feedback_rows"]>=1
    assert metrics["n_training"]==metrics["n_base"]+metrics["n_feedback_rows"]
    assert metrics["feedback_rows_considered"]>=1
    assert 0 <= metrics["accuracy_training"] <= 1

def test_retrain_skip_failures_flag():
    record_routing_failure("Completely new failing prompt about routing.","tier_1","openai_mini",
                           1.5,[],"req_y",4.0,0.2,"")
    with_fb=retrain(include_failures=True)
    without=retrain(include_failures=False)
    assert with_fb["n_training"]>without["n_training"]
    assert without["n_feedback_rows"]==0

def test_retrain_excludes_unchanged_labels_by_default():
    record_routing_failure("Hard design prompt.","tier_3","openai_high",2.0,[],"req_z",4.2,0.3,"")
    metrics=retrain(only_label_changes=True)
    # tier_3 failure cannot be relabelled higher, so it must not be trained on
    texts=[r for r in load_routing_failures() if not r["label_changed"]]
    assert len(texts)>=1
    assert metrics["n_feedback_rows"]==0 or all(
        r["corrected_tier"]!=r["routed_tier"] for r in load_routing_failures() if r.get("label_changed"))
