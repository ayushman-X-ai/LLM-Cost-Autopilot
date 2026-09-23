# Demo Script

### 0:00 Problem
“Sending every request to the strongest model is simple but potentially expensive. This service adds a routing and verification layer.”

### 0:30 Simple request
Send a formatting/extraction request. Show the low-cost/local route.

### 1:00 Moderate request
Send a summarization/classification task. Show Tier 2 routing.

### 1:30 Complex request
Send a multi-constraint architecture question. Show Tier 3 routing.

### 2:00 Verification (fast vs quality)
Send the same request twice: `verification_mode: fast` returns immediately with `verification_queued: true`; `quality` returns with `quality_score` and `verified`. Explain the two paths.

### 2:30 Escalation
Force a failing verification (mocked verifier response below threshold). Show the quality-mode response returning the escalated premium answer, with `escalated`, `escalation_model` and the audit row status.

### 3:00 Fallback routing
Block a provider (remove the key / stop the local model) and send a request. Show `routing_event: provider_failure`, `original_model`, `fallback_model`, `reason` — the request still succeeds.

### 3:30 Dashboard hero
Show ACTUAL COST vs BASELINE COST, SAVED and COST REDUCTION % as the headline, plus model distribution, latency and escalation rate underneath.

### 4:00 Measured, not assumed
Run `python -m autopilot.cli evaluate` — accuracy, per-tier metrics, confusion matrix. Then `python -m autopilot.cli retrain` picking up accumulated `routing_failures.jsonl`.

### 4:30 Config versioning
`PUT /v1/routing-config`, restart the service, show the change is still active, then roll back from `/history`.

### 5:00 Engineering takeaway
Explain that routing quality is measured, not assumed; failures feed back into classifier training; pricing is stamped and auditable; and provider failures degrade through a fallback chain instead of failing the request.
