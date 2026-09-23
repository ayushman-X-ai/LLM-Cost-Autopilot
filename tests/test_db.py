import sqlite3, uuid
from autopilot.db import (connect, insert_request, recent, get_request, update_request, stats,
                          insert_routing_version, latest_routing_version, list_routing_versions,
                          get_routing_version)

def test_db_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH",str(tmp_path/"test.db"))
    from autopilot.settings import get_settings; get_settings.cache_clear()
    rid="t_"+uuid.uuid4().hex
    insert_request({"id":rid,"timestamp":1.0,"prompt_hash":"x","prompt_text":"hello","complexity_tier":"tier_1","routed_model":"ollama_llama","provider":"ollama","input_tokens":1,"output_tokens":2,"cost_usd":0,"latency_ms":1})
    assert any(x["id"]==rid for x in recent())

def test_update_and_get_request():
    rid="u_"+uuid.uuid4().hex
    insert_request({"id":rid,"timestamp":1.0,"prompt_hash":"x","prompt_text":"p",
                    "complexity_tier":"tier_2","routed_model":"openai_mini","provider":"openai",
                    "input_tokens":10,"output_tokens":20,"cost_usd":0.001,"latency_ms":10,"status":"completed"})
    update_request(rid, quality_score=4.5, status="verified", escalated=1, escalation_model="openai_high")
    row=get_request(rid)
    assert row["quality_score"]==4.5 and row["status"]=="verified" and row["escalated"]==1

def test_stats_include_tokens_and_distributions():
    rid="s_"+uuid.uuid4().hex
    insert_request({"id":rid,"timestamp":1.0,"prompt_hash":"x","prompt_text":"p",
                    "complexity_tier":"tier_3","routed_model":"openai_high","provider":"openai",
                    "input_tokens":100,"output_tokens":200,"cost_usd":0.01,"latency_ms":100,
                    "quality_score":4.2,"escalated":1,"status":"escalated",
                    "verification_mode":"quality"})
    s=stats()
    assert s["requests"]>=1 and s["cost_usd"]>=0.01
    assert s["input_tokens"]>=100 and s["output_tokens"]>=200
    assert s["escalation_rate"]>=0 and s["avg_quality"] is not None
    assert any(m["model"]=="openai_high" for m in s["by_model"])
    assert any(t["tier"]=="tier_3" for t in s["by_tier"])

def test_audit_columns_for_fallback_and_mode():
    rid="f_"+uuid.uuid4().hex
    insert_request({"id":rid,"timestamp":1.0,"prompt_hash":"x","prompt_text":"p",
                    "complexity_tier":"tier_2","routed_model":"anthropic_haiku","provider":"anthropic",
                    "input_tokens":1,"output_tokens":1,"cost_usd":0.0,"latency_ms":5,
                    "verification_mode":"fast","routing_event":"provider_failure",
                    "original_model":"openai_mini","fallback_model":"anthropic_haiku",
                    "fallback_reason":"timeout"})
    row=get_request(rid)
    assert row["routing_event"]=="provider_failure" and row["original_model"]=="openai_mini"
    assert row["fallback_model"]=="anthropic_haiku" and row["verification_mode"]=="fast"

def test_routing_version_crud():
    v1=insert_routing_version({"tiers":{"tier_1":{"model":"ollama_llama"}}},"tester")
    v2=insert_routing_version({"tiers":{"tier_1":{"model":"openai_mini"}}},"tester2")
    assert v2>v1
    assert latest_routing_version()["version"]==v2
    assert get_routing_version(v1)["config"].startswith("{")
    assert get_routing_version(99999) is None
    versions=list_routing_versions()
    assert [x["version"] for x in versions][:2]==[v2,v1]
    assert versions[0]["created_by"]=="tester2"

def test_schema_migrates_old_databases(tmp_path, monkeypatch):
    """Pre-feature databases (no audit/version columns) are upgraded on connect."""
    db=tmp_path/"old.db"
    conn=sqlite3.connect(db)
    conn.execute("""CREATE TABLE requests (
        id TEXT PRIMARY KEY, timestamp REAL NOT NULL, prompt_hash TEXT NOT NULL, prompt_text TEXT,
        complexity_tier TEXT, routed_model TEXT, provider TEXT, input_tokens INTEGER,
        output_tokens INTEGER, cost_usd REAL, latency_ms REAL, quality_score REAL,
        escalated INTEGER DEFAULT 0, escalation_model TEXT, status TEXT DEFAULT 'completed',
        verification_error TEXT)""")
    conn.commit(); conn.close()
    monkeypatch.setenv("DATABASE_PATH",str(db))
    from autopilot.settings import get_settings; get_settings.cache_clear()
    rid="m_"+uuid.uuid4().hex
    insert_request({"id":rid,"timestamp":1.0,"prompt_hash":"x","prompt_text":"p",
                    "routing_event":"provider_failure","verification_mode":"fast",
                    "original_model":"openai_mini","fallback_model":"ollama_llama",
                    "fallback_reason":"timeout"})
    row=get_request(rid)
    assert row["routing_event"]=="provider_failure"
    # routing_versions table exists after migration
    assert insert_routing_version({"tiers":{}},"migrate")>=1
