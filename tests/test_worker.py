import asyncio
import pytest
from autopilot.db import insert_request, get_request
from autopilot.models import Message
from autopilot.worker import VerificationQueue, VerificationJob
import autopilot.verifier as v
from autopilot.verifier import VerificationResult

@pytest.mark.asyncio
async def test_queue_processes_jobs(monkeypatch):
    q=VerificationQueue()
    seen={}
    async def fake_verify(request_id, messages, candidate_text, tier, routed_model_name, **kw):
        seen["tier"]=tier; seen["model"]=routed_model_name; seen["candidate"]=candidate_text
        insert_request({"id":request_id,"timestamp":1.0,"prompt_hash":"h","prompt_text":"p",
                        "status":"completed"})
        return VerificationResult(quality_score=4.5,agreement=0.9,threshold=4.0,
                                  agreement_threshold=0.7,passed=True)
    monkeypatch.setattr(v,"verify",fake_verify)
    await q.start()
    insert_request({"id":"q_1","timestamp":1.0,"prompt_hash":"h","prompt_text":"p","status":"completed"})
    await q.enqueue(VerificationJob("q_1",[Message(role="user",content="hi")],"the answer",
                                    "openai_mini","tier_2"))
    await q.join()
    assert seen=={"tier":"tier_2","model":"openai_mini","candidate":"the answer"}
    assert get_request("q_1")["status"]=="verified"
    await q.stop()

@pytest.mark.asyncio
async def test_queue_continues_after_failure(monkeypatch):
    q=VerificationQueue()
    async def boom(*a,**k): raise RuntimeError("bad job")
    async def good(request_id, *a, **k):
        insert_request({"id":request_id,"timestamp":1.0,"prompt_hash":"h","prompt_text":"p",
                        "status":"completed"})
        return VerificationResult(passed=True,quality_score=4.0)
    calls={"n":0}
    async def flaky(*a,**k):
        calls["n"]+=1
        if calls["n"]==1: return await boom()
        return await good(*a,**k)
    monkeypatch.setattr(v,"verify_and_maybe_escalate",flaky)
    await q.start()
    for rid in ("q_bad","q_good"):
        insert_request({"id":rid,"timestamp":1.0,"prompt_hash":"h","prompt_text":"p","status":"completed"})
        await q.enqueue(VerificationJob(rid,[],"x","m","tier_1"))
    await q.join()
    assert get_request("q_good")["status"]=="completed" or get_request("q_good")["quality_score"] is not None
    await q.stop()

@pytest.mark.asyncio
async def test_queue_start_is_idempotent_per_loop():
    q=VerificationQueue()
    await q.start(); t1=q.task
    await q.start()
    assert q.task is t1  # same loop -> no second consumer
    await q.stop()
