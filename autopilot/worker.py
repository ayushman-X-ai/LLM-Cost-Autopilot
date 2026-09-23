import asyncio
from dataclasses import dataclass
from .verifier import verify_and_maybe_escalate
from .models import Message

@dataclass
class VerificationJob:
    request_id:str
    messages:list[Message]
    candidate_text:str
    routed_model_name:str
    tier:str = "tier_2"

class VerificationQueue:
    def __init__(self): self.q=asyncio.Queue(); self.task=None; self._loop=None
    async def start(self):
        loop=asyncio.get_running_loop()
        if self.task is None or self._loop is not loop:
            # Fresh task (and queue) whenever the event loop changes - the app
            # may be restarted in tests or by a process manager.
            self.q=asyncio.Queue(); self._loop=loop
            self.task=asyncio.create_task(self._run())
    async def enqueue(self,job):
        await self.q.put(job)
    async def join(self):
        """Wait for all queued verifications to finish (used by tests and graceful shutdown)."""
        await self.q.join()
    async def stop(self):
        """Cancel the consumer task; safe to call when it never started."""
        if self.task is not None:
            self.task.cancel()
            try: await self.task
            except asyncio.CancelledError: pass
            self.task=None; self._loop=None
    async def _run(self):
        while True:
            job=await self.q.get()
            try: await verify_and_maybe_escalate(job.request_id,job.messages,job.candidate_text,job.routed_model_name,job.tier)
            finally: self.q.task_done()

queue=VerificationQueue()
