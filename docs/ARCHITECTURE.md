# Architecture Notes

## Request path

1. FastAPI validates the request (including `verification_mode: fast | quality`).
2. `CostRouter` converts messages into classifier input.
3. The classifier emits a complexity tier and confidence.
4. YAML maps the tier to a primary model plus an ordered fallback chain.
5. Eligibility filters apply: cloud-disabled removes non-Ollama candidates, the per-request budget removes candidates over `MAX_REQUEST_COST_USD`.
6. The provider adapter executes the request with timeout + retry/backoff; if the chosen model still fails, the next fallback is tried.
7. The result is normalized and persisted (with `routing_event` / `original_model` / `fallback_model` when a fallback was used).
8. Verification runs according to the requested mode:
   - **fast**: a background job verifies after the response was returned.
   - **quality**: verification happens before returning; a failed score triggers premium escalation and the corrected answer is returned.
9. Escalations and audit rows update the SQLite trail; failed verifications are written to the feedback loop.

## Verification modes

```text
fast:    Request -> Router -> Model -> Return -> async verify -> (fail) -> escalate (audited)
quality: Request -> Router -> Model -> verify -> PASS -> Return
                                             -> FAIL -> premium model -> Return corrected answer
```

`fast` optimises latency and treats verification as monitoring; `quality` optimises answer quality and accepts the extra latency. The mode is a per-request field so the same deployment can serve both.

## Quality scoring

The verifier scores four dimensions 1–5 plus an agreement value 0–1:

| Dimension | Weight |
|---|---|
| correctness | 40% |
| relevance | 20% |
| completeness | 20% |
| instruction following | 20% |

`quality_score = Σ weight × dimension` (weights normalised from `config/routing.yaml`). A verification passes when `quality_score >= tier threshold` **and** `agreement >= agreement_threshold`. Weights and thresholds are configuration because the guide calls for use-case-specific quality thresholds. Malformed verifier output degrades to score 0 and a failed verification (fail closed), never a crash.

## Fallback routing

Each tier declares an ordered fallback chain. A request walks the chain when the current candidate:

- exhausts retries on a retryable error (timeout, connection, 408/429/5xx), or
- is filtered out because `ALLOW_CLOUD=false`, or
- is filtered out because its estimated cost exceeds `MAX_REQUEST_COST_USD`.

The response reports what happened: `routing_event` (`provider_failure`, `cloud_disabled`, `cost_guard`), `original_model`, `fallback_model`, `reason`. This is what makes the layer behave like real infrastructure between an application and flaky LLM providers.

## Feedback loop

```text
routing failure -> routing_failures.jsonl -> python -m autopilot.cli retrain -> better routing
```

Failed verifications are recorded with the routed tier, a proposed corrected tier (one step up, capped at tier_3), score, issues and rationale. `retrain` merges label-changing failures into the training set (deduplicated by text), retrains, and reports how many feedback rows were absorbed. Tier_3 failures cannot be relabelled higher, so they are excluded by default (`--all-failures` overrides).

## Evaluation

`data/routing_eval.jsonl` is a held-out, deterministically generated set (500 prompts, labels assigned by construction). `python -m autopilot.cli evaluate` reports accuracy, per-tier precision/recall/F1, confusion matrix, per-tier accuracy and routing error rate; each run appends a summary to `data/evaluation_results.jsonl` and writes `artifacts/routing_eval_report.json`. Routing quality is measured, not assumed.

## Configuration versioning

`PUT /v1/routing-config` persists a full config snapshot to the `routing_versions` table and applies it in memory; on startup the latest version is re-applied, so runtime changes survive restarts. `/history` lists versions and `/rollback` re-publishes an old config as a new version (append-only, auditable).

## Why not route only on token count?

Token count is useful but insufficient. A short prompt can require difficult reasoning, while a long prompt can be a simple extraction task. The classifier therefore combines lexical TF-IDF signals (word 1-2 grams + char 3-5 grams, sublinear) with engineered behavioural features (task verbs, constraint clauses, planning depth, reasoning lexicon). Training metrics report stratified 5-fold cross-validation, not fit-set accuracy: on template-heavy data the fit accuracy reads ~100% while held-out accuracy was originally 76.6%. After dataset deduplication/expansion and feature work, held-out accuracy is 93.6% on the 500-prompt set and 81.1% on a deliberately adversarial boundary/look-alike set (`data/routing_eval_boundary.jsonl`) — the gap between those two numbers is itself part of the honest reporting. `tests/test_dataset.py` enforces dataset integrity: unique texts, tier balance, and train/eval disjointness.

## Quality verification caveat

The verifier uses an LLM-as-judge. It is useful for a portfolio prototype but is not ground truth. A production system should combine deterministic task-specific checks, human feedback, golden datasets and statistical evaluation. The feedback loop here is the seam where human-reviewed labels would enter.

## Cost accounting

`ModelConfig.estimate_cost()` calculates estimated provider cost from configured input/output token prices. Prices are configuration — each model carries `pricing_updated_at` so stale pricing is visible — because provider prices change. Baseline/savings math (`autopilot.metrics`) re-prices the same token counts at the configured premium baseline (`metrics.baseline_model`).

## Routing policy

The routing map is externalized in YAML. This makes it easy to test different policies:

- Tier 1 -> local/cheap model (+ fallbacks)
- Tier 2 -> mid-tier model (+ fallbacks)
- Tier 3 -> high-quality model (+ fallbacks)

The policy should be evaluated against a representative workload (`autopilot.cli evaluate`) before claiming any specific savings percentage.

## Persistence

SQLite runs in WAL mode with a busy timeout so concurrent request writes, verification updates and config versioning do not serialise on fsync. `connect()` migrates older databases by adding new audit columns (`verification_mode`, `routing_event`, `original_model`, `fallback_model`, `fallback_reason`) via `ALTER TABLE`.
