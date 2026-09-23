# Interview Talking Points

## Problem
LLM applications often overuse expensive models. The system attempts to preserve quality while reducing average inference cost.

## Core engineering decisions

- Provider abstraction prevents vendor-specific logic from leaking from routing; adapters add timeouts and classified retryable errors (timeout/connection/429/5xx) with exponential backoff.
- Complexity classification is separated from routing policy; policy (primary, fallback chain, thresholds, weights) is YAML.
- Two verification modes: `fast` keeps the user-facing path async; `quality` blocks, verifies with a weighted 4-dimension score, and returns a premium-corrected answer on failure.
- Quality is a weighted score (correctness 40 / relevance 20 / completeness 20 / instruction 20) plus an agreement threshold — configurable because thresholds should be use-case specific.
- Provider failures degrade through a per-tier fallback chain and are reported explicitly (`routing_event`), not swallowed.
- Escalations and failed verifications are logged so routing failures become training/evaluation data (`routing_failures.jsonl` -> `retrain`).
- Routing accuracy is measured on a held-out eval set (confusion matrix, per-tier precision/recall, routing error rate), not asserted.
- Routing configuration is versioned in SQLite with history and rollback; runtime changes survive restarts.
- Model prices are configuration with an explicit `pricing_updated_at`, not constants — stale pricing is visible.
- SQLite in WAL mode gives an inspectable audit trail for a zero-infrastructure demo without write serialization.

## Metrics to report only after measurement

- Routed cost vs premium baseline, cost reduction %
- Quality parity / verifier score and quality pass rate
- Escalation rate and provider fallback rate
- P50/P95/P99 latency (from `scripts/load_test.py`, 500–1,000 requests)
- Classifier accuracy, per-tier metrics and confusion matrix (from `autopilot.cli evaluate`)

Do not invent these values. Run the evaluation, benchmark and load test, and report the measured numbers (they are written to `artifacts/`).

## Likely follow-up questions

- **What happens when the cheap answer is bad?** In `quality` mode the user never sees it: verification runs before returning and the premium model's answer is returned instead. In `fast` mode the failure is audited, escalated asynchronously and fed back into retraining.
- **How do you avoid rewarding verbosity?** The verifier prompt scores correctness/relevance/completeness/instruction-following separately with explicit instructions not to reward verbosity, and agreement is capped separately from the score.
- **How do you know routing works?** Held-out evaluation with a confusion matrix; current starter numbers show tier_2 under-routing, which is exactly what the feedback loop is for.
- **What is the failure mode of the router itself?** Fallback chains plus explicit `routing_event` fields, so downstream systems can observe degraded routing.
