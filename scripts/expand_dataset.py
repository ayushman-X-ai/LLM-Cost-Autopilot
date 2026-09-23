"""Expand data/complexity_dataset.jsonl with a curated, human-reviewable set.

The original starter file repeats 72 unique texts across 210 rows (each text
~3x), so the model memorizes templates instead of learning complexity. This
script:

1. keeps the original rows but drops exact duplicate texts,
2. appends ~140 hand-written prompts, including deliberate boundary cases
   (tier_1 vs tier_2 and tier_2 vs tier_3 look-alikes),
3. asserts integrity: valid tiers, balance, no duplicates, and no text overlap
   with the held-out evaluation set data/routing_eval.jsonl,
4. writes the merged dataset back, interleaved by tier for readability.

Labels follow the same rubric as the evaluation set:
  tier_1: single mechanical transformation/extraction/recall, <= 1 simple constraint
  tier_2: multi-part drafting, structured rewriting, comparisons with a
          recommendation, audience explanations, data interpretation,
          constraint satisfaction, tutorials
  tier_3: architecture/design with explicit sub-deliverables (data model,
          consistency, retry, topology), formal proofs/derivations,
          distributed correctness, threat modeling, migration with rollback,
          multi-objective optimization, evaluation frameworks

All expansion rows carry source="curated_expansion" and reviewed=false: the
labels are drafted, but a human should review them before presenting the set
as hand-curated. Run `python -m autopilot.cli train` afterwards.

Usage:
    python scripts/expand_dataset.py [--dataset data/complexity_dataset.jsonl]
"""
import argparse, json
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Curated expansion: (tier, prompt). Hand-written, no template generation.
# ---------------------------------------------------------------------------
CURATED = [
    # ------------------------------------------------------------- tier_1
    ("tier_1", "Translate this product description into German: Our lamp uses 40 percent less energy than standard bulbs and comes with a two-year warranty."),
    ("tier_1", "Extract every URL from this email and return them as a list: You can review the invoice at https://pay.example.com/inv/2291 or read the docs at https://docs.example.com/billing."),
    ("tier_1", "Fix the grammar in this review: me and my friend was very happy with the hotel, the rooms were clean and the staff was friendly."),
    ("tier_1", "Convert 37 degrees Celsius to Fahrenheit."),
    ("tier_1", "Sort these due dates chronologically: 2026-05-11, 2025-12-01, 2026-01-09, 2025-07-30."),
    ("tier_1", "Rewrite this sentence in passive voice: The team shipped the release on Friday."),
    ("tier_1", "Return the order total printed in this receipt: subtotal $18.38, tax $1.47, total $19.85."),
    ("tier_1", "Make this headline title case: ten things every developer should know about caching."),
    ("tier_1", "Change this sentence to future tense: She finished the report last night."),
    ("tier_1", "Count how many times the word 'delay' appears in this paragraph: The delay on Tuesday caused another delay on Wednesday, and the second delay pushed the whole week back."),
    ("tier_1", "Extract the tracking number from this shipping notice: Your order 8841 shipped via FedEx, tracking 1Z999AA10123456784, and arrives Tuesday."),
    ("tier_1", "Rewrite this sentence so it is under ten words: We regret to inform you that the feature you requested has been postponed until further notice."),
    ("tier_1", "Convert this bulleted list into a numbered list: apples, pears, plums."),
    ("tier_1", "Is 97 a prime number? Answer yes or no."),
    ("tier_1", "What currency does Japan use?"),
    ("tier_1", "Normalize these phone numbers to the format (XXX) XXX-XXXX: 555-0143, 5550187, 555 0199."),
    ("tier_1", "Correct the capitalization in this sentence: the ceo of our partner company will visit on monday."),
    ("tier_1", "Join these sentences into one sentence using 'because': The launch slipped. The vendor missed the deadline."),
    ("tier_1", "Extract the meeting time from this note: Sync with Priya and Dan, Thursday 3pm, room B, bring the vendor quote."),
    ("tier_1", "Remove duplicate items from this list: red, blue, red, green, blue, yellow."),
    ("tier_1", "Give a one-sentence definition of 'load balancing'."),
    ("tier_1", "Which of these is not a programming language: Java, HTML, Python, Rust?"),
    ("tier_1", "Reverse the order of words in this sentence: deploy the service after the migration completes."),
    ("tier_1", "Round 3.14159 to two decimal places."),
    ("tier_1", "Rewrite this warning label politely: You did not include the required signature."),
    ("tier_1", "Extract all dollar amounts from this message: You were charged $12.99 on March 3 and refunded $6.50 on March 9; the remaining balance is $6.49."),
    ("tier_1", "Answer with a single word: what color do you get by mixing blue and yellow?"),
    ("tier_1", "Convert this date to ISO format: December 1, 2025."),
    ("tier_1", "Split this sentence at the semicolon into two sentences: The staging deploy passed; the production deploy is scheduled for tonight."),
    ("tier_1", "Rewrite this sentence without the filler words: So basically, I just wanted to kind of ask if we could maybe move the meeting."),
    ("tier_1", "Add a greeting to the start of this email: Could you resend the signed contract when you get a chance?"),
    ("tier_1", "Tell me the number of characters in this string: autonomy."),
    ("tier_1", "Return the last four digits of this card number: 4111 1111 1111 1234."),
    ("tier_1", "Which is larger, 3/4 or 5/8?"),
    ("tier_1", "Rewrite this sentence as a question: The invoice is due next Friday."),
    ("tier_1", "Extract the invoice number from this email: Invoice INV-2291 for $4,120 is attached; PO number 778 is referenced on page one."),
    ("tier_1", "Fix the double negative in this sentence: I don't know nothing about that."),
    ("tier_1", "Lowercase this product title: WIRELESS NOISE-CANCELLING HEADPHONES."),
    ("tier_1", "Pick the correctly spelled word: accomodate / accommodate / acommodate."),
    ("tier_1", "Return just the domain name from this URL: https://status.example.com/incidents/4412."),
    ("tier_1", "Summarize this sentence in five words or fewer: The committee approved the budget after a long debate about staffing."),
    ("tier_1", "Does this sentence contain a date? Answer yes or no: The release moved to next quarter because legal review is pending."),
    ("tier_1", "List the names of the people mentioned in this message: Sam will review the spec and Priya waits on the vendor quote."),
    ("tier_1", "Rewrite this instruction in the imperative: It would be great if you could send the confirmation."),
    ("tier_1", "Which word is a synonym of 'quick': rapid, narrow, heavy, late?"),
    ("tier_1", "Split this CSV line into its four fields and print them one per line: name,role,start_date,city."),
    ("tier_1", "Return the second item in this list: alpha, beta, gamma, delta."),
    ("tier_1", "Classify this customer message as urgent or routine: My order arrived with a cracked lid, please advise."),
    ("tier_1", "Classify this email as spam or not spam: Congratulations, you have been selected for a free gift card."),
    ("tier_1", "Label this ticket as hardware or software: Keyboard model K-22 registers double keystrokes on all keys."),
    ("tier_1", "Rewrite this address as a single line: 100 Main St, Apt 4, Springfield, IL."),

    # ------------------------------------------------------------- tier_2
    ("tier_2", "Summarize the article below in five bullet points, then list any metrics the author cites: Artificial intelligence is changing software engineering by making code generation and review faster, but teams still need evaluation and governance to catch regressions before customers do."),
    ("tier_2", "Rewrite this paragraph in active voice and cut its length by half without losing any facts: The quarterly report was reviewed by the leadership team, and it was decided by them that the hiring plan would be paused by the finance group until the new revenue targets are met by the sales org."),
    ("tier_2", "Draft a polite but firm reminder email to a client whose invoice is 30 days overdue; include a clear payment deadline and the exact amount owed."),
    ("tier_2", "Explain how database indexing works to a non-technical product manager, using a warehouse or library analogy, in two short paragraphs."),
    ("tier_2", "Turn these meeting notes into action items with an owner and due date for each: Sam will review the spec by Wednesday; Priya waits on the vendor quote; launch still targeted for the 14th; staffing risk flagged."),
    ("tier_2", "Compare REST and GraphQL for our new mobile app and recommend one, with two concrete reasons."),
    ("tier_2", "Review this SQL query for readability and index usage without changing its results: SELECT u.id, COUNT(o.id) FROM users u JOIN orders o ON o.user_id=u.id GROUP BY u.id;"),
    ("tier_2", "Given the weekly signups below, explain what changed and suggest one next step: week 1: 120, week 2: 96, week 3: 141, week 4: 133; paid conversion slipped from 2.1% to 1.4%."),
    ("tier_2", "Convert this requirement into acceptance criteria a QA team can test: users must be able to export all their data as CSV within 60 seconds."),
    ("tier_2", "Write three versions of a launch announcement for our new export feature: one playful, one formal, one ultra-short."),
    ("tier_2", "Create a two-day onboarding agenda for new engineers with sessions, breaks, and one hands-on lab."),
    ("tier_2", "Rewrite this customer complaint as a structured bug report with steps to reproduce and a severity: the app crashed twice today and I lost my draft, totally unacceptable."),
    ("tier_2", "Group the feedback below into themes and give each theme a short label: The search is fast but filters reset too often. Love the dark mode. Price feels high for small teams."),
    ("tier_2", "Explain the difference between authentication and authorization with one concrete example of each."),
    ("tier_2", "Draft a release announcement for our new export feature covering the benefits, how to enable it, and the known limitations."),
    ("tier_2", "Summarize this support thread and answer the customer's final question at the end: Customer cannot log in after the password reset; no email arrives; they ask whether their account was deleted."),
    ("tier_2", "Weigh the pros and cons of moving our scheduled reports from cron to a job queue, and recommend one option for a team of four."),
    ("tier_2", "Break this goal into a prioritized task list with dependencies: migrate billing to the new provider before the August renewal."),
    ("tier_2", "Explain what the drop in trial-to-paid conversion implies for onboarding and what to try next, given: conversion 2.1% -> 1.4% after the price change; cancellations cite cost."),
    ("tier_2", "Write a short tutorial that walks a new user through connecting a custom domain, including two common pitfalls."),
    ("tier_2", "Compare TCP and UDP for our realtime telemetry and recommend one with justification."),
    ("tier_2", "Rewrite the description below so it works both on the homepage and in the technical changelog, keeping the facts identical: Battery costs fell again this year, accelerating electric vehicle adoption in markets with strong charging infrastructure."),
    ("tier_2", "Analyze this customer review and separate product issues from support issues: Mobile app crashes when uploading photos. Desktop works fine. Support replied within an hour but the fix is still pending."),
    ("tier_2", "Turn this rough outline into a complete project brief with goals, scope, and non-goals: idea - let teams save and share filtered dashboard views."),
    ("tier_2", "Draft three interview questions for a senior backend candidate, and for each one describe what a strong answer covers."),
    ("tier_2", "Given these two vendor quotes, compare total cost over 12 months and recommend one: vendor A $900/month flat; vendor B $0.0004 per request with an estimated 2.5M requests/month."),
    ("tier_2", "Rewrite this error message for end users, then write a second version for our status page: Exception: null pointer in billing adapter at line 442."),
    ("tier_2", "Summarize the differences between the two plans below in a three-row comparison table: Starter $9/user/month with email support; Team $19/user/month with SSO, audit logs, and priority support."),
    ("tier_2", "Explain idempotency to a junior developer using an example from payments, then show what goes wrong without it."),
    ("tier_2", "From the log excerpt, identify the two most frequent failure types and propose one fix for each: timeout /v1/export x41, connection refused warehouse-db x17, validation error x12, timeout /v1/export x9."),
    ("tier_2", "Create a checklist for our first production deploy, ordered from code freeze to rollback readiness."),
    ("tier_2", "Draft a friendly nudge message for teammates who have not filled in the survey, plus a firmer follow-up for next week."),
    ("tier_2", "Convert this paragraph into five flashcards, each with a question and an answer: Event-driven architecture decouples producers from consumers through a broker. Messages are typically delivered at-least-once, so consumers must deduplicate. Dead-letter queues hold poison messages for inspection."),
    ("tier_2", "Compare the two onboarding flows described below and recommend which to A/B test first: Flow A is a five-step wizard with a progress bar; Flow B skips setup and drops users into a sample project."),
    ("tier_2", "Rewrite this policy summary at an eighth-grade reading level while keeping every requirement: Access is granted on a least-privilege basis; quarterly access reviews are mandatory; shared accounts are prohibited and violations are reported to security."),
    ("tier_2", "Given the survey responses below, summarize sentiment in three bullets and recommend one action: Setup took hours and docs skipped auth. Support was friendly. Would like SSO on the cheap plan."),
    ("tier_2", "Turn this single-paragraph idea into a one-page proposal with risks and milestones: Let customers schedule reports to arrive as PDFs in their inbox every Monday."),
    ("tier_2", "Write an escalation email to a vendor about repeated SLA misses; include the specific incidents from this month and the remedy we want."),
    ("tier_2", "Review this regex and explain what it matches in plain language, then suggest a clearer alternative: ^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+)@(?:[a-z0-9-]+\\.)+[a-z]{2,}$"),
    ("tier_2", "Compare monthly and annual billing for a 40-person company and recommend one, given a tight cash position this year."),
    ("tier_2", "Rewrite these release notes so each bullet starts with a verb and mentions the affected area: Fixed a bug in exports; new dashboard filters; billing webhooks are faster now."),
    ("tier_2", "Given this quarter's latency data, write three findings and one recommendation for the capacity plan: eu 210ms, us 180ms, apac 460ms; error rate 0.7% in apac only."),
    ("tier_2", "Design a simple feedback form for our beta users with exactly six questions, mixing rating scales and free text."),
    ("tier_2", "Write a job description summary for a part-time data analyst with responsibilities and must-have skills."),
    ("tier_2", "Summarize this changelog for end users, ignoring internal refactors, in five bullets: Migrated storage layer; added CSV export; fixed dark-mode contrast; internal queue rewrite; new keyboard shortcuts."),
    ("tier_2", "Rewrite the FAQ answer below to be 30 percent shorter while keeping the link and the warranty terms: You can return any purchase within 30 days; see our full policy at https://example.com/returns, and the two-year warranty still applies to all lamps."),
    ("tier_2", "Draft a weekly status update for stakeholders covering progress, risks, and asks, using these notes: import feature done; themes cut from scope; legal approval still blocked; Dan owns the plan."),
    ("tier_2", "Explain the tradeoffs between a monolith and microservices for our 12-person team in plain language, then state which you would pick."),

    # ------------------------------------------------------------- tier_3
    ("tier_3", "Design a multi-tenant billing service with per-tenant budgets, idempotent webhook delivery, and a complete audit trail; cover the data model, consistency boundaries, and retry policy."),
    ("tier_3", "Prove that the halting problem is undecidable, explaining every step of the diagonalization argument."),
    ("tier_3", "Compare Raft and Paxos for a payments ledger that must survive network partitions; justify the choice and describe the operational runbook."),
    ("tier_3", "Architect a zero-downtime migration of our monolithic order service to microservices, including dual-write reconciliation, backfill, and a rollback plan."),
    ("tier_3", "Threat model our document e-signature flow using STRIDE, enumerate mitigations, and state the residual risk we accept."),
    ("tier_3", "Derive the time and space complexity of the algorithm below, then propose and justify an optimization:\ndef solve(items):\n    best = 0\n    for i in range(len(items)):\n        for j in range(i+1, len(items)):\n            best = max(best, items[i] + items[j])\n    return best"),
    ("tier_3", "Design an evaluation framework for our routing classifier: dataset construction, statistical significance testing, and regression detection after each retrain."),
    ("tier_3", "Design a rate limiter that stays correct across 200 nodes with clock skew; compare token bucket and sliding window approaches and justify the choice."),
    ("tier_3", "Plan the staged rollout of a new search ranking model to 50M users with cohort gating, metric thresholds per stage, and automated rollback."),
    ("tier_3", "Design a reliability strategy for our webhook delivery service: SLOs, error budgets, chaos experiments, and the on-call escalation policy."),
    ("tier_3", "Evaluate event sourcing versus a CRUD data model for tracking warehouse inventory levels under sustained write load, analyze the failure modes of each, and recommend one."),
    ("tier_3", "We must cut p95 checkout latency below 300ms while keeping the public API unchanged and staying within the current budget. Compare three strategies, quantify the tradeoffs, and recommend one."),
    ("tier_3", "Design the data migration and dual-write reconciliation plan for moving order history to a new storage engine with zero data loss."),
    ("tier_3", "Specify the idempotency semantics of our payments API formally: state the invariants, enumerate the concurrent interleavings, and describe how the implementation enforces them."),
    ("tier_3", "Compare leader election approaches for our scheduler under network partitions, including split-brain prevention and the operator runbook."),
    ("tier_3", "Review the security architecture of our customer data platform used by a regulated healthcare startup: enumerate attack paths, detection signals, and a prioritized remediation plan."),
    ("tier_3", "Design an observability strategy for a 200-node distributed job scheduler: metrics, traces, and the specific alerts that page a human."),
    ("tier_3", "Derive a routing policy that minimizes cost while holding quality above 4.0 and p95 latency under 300ms; specify the classifier target, the escalation rules, and the evaluation plan."),
    ("tier_3", "Architect a multi-region deployment for our session store that must survive a zone failure, keep 99.95% availability, and satisfy GDPR data residency."),
    ("tier_3", "Write an implementation plan for a distributed job scheduler with exactly-once semantics: idempotency keys, reconciliation, backpressure, and observability."),
    ("tier_3", "Synthesize the arguments for and against feature flags at scale, identify where the evidence conflicts, and state the experiment that would resolve it."),
    ("tier_3", "Design a capacity plan for our vector search service to grow 10x: storage, shard strategy, and the cost model, with the validation steps before launch."),
    ("tier_3", "Diagnose and redesign our ingestion pipeline where throughput collapses nightly during the vendor sync: form hypotheses, specify measurements, and propose structural fixes rather than tuning."),
    ("tier_3", "Compare consensus-based replication, primary-secondary, and CRDTs for our collaborative editor, and justify the choice with conflict and latency analysis."),
    ("tier_3", "Design the migration of our billing system from monthly subscriptions to usage-based pricing with proration, refunds, and no customer-visible downtime."),
    ("tier_3", "Formally specify the retry and deduplication contract between our order service and the warehouse system, including poison-message handling."),
    ("tier_3", "Design an experiment to determine whether our new cache layer changes error rates, including power analysis and stopping rules."),
    ("tier_3", "Architect the 12-week SOC 2 readiness program for our platform: control mapping, evidence collection, and the sequencing that minimizes engineering disruption."),
    ("tier_3", "Evaluate three storage engines for our time-series telemetry under strict write-amplification and retention constraints; include a failure-mode analysis and a recommendation."),
    ("tier_3", "Design a canary analysis system that automatically promotes or rolls back deploys based on metric gates; specify the statistical method and its assumptions."),
    ("tier_3", "Plan and justify a decommissioning strategy for our legacy auth service with zero customer disruption; include traffic shadowing and cutover criteria."),
    ("tier_3", "Design the consistency model for a shopping cart that must work offline across devices and reconcile conflicts deterministically."),
    ("tier_3", "Write a design doc for horizontally scaling our notification fan-out from 1k to 1M deliveries per minute, including the data model and failure modes."),
    ("tier_3", "Compare token bucket, leaky bucket, and sliding window log rate limiters for a public API with bursty enterprise traffic, and justify the operational choice."),
    ("tier_3", "Architect a data pipeline that joins streaming events with nightly reference data exactly once, covering late arrivals and reconciliation."),
    ("tier_3", "Design a schema-versioning and backfill strategy for a 4TB Postgres cluster with a 30-minute maintenance window per quarter."),
    ("tier_3", "Review our disaster recovery plan for the payments database: RTO/RPO targets, restore drills, corruption detection, and the gaps you find."),
    ("tier_3", "Design the multi-tenant isolation strategy for our report engine: noisy neighbors, quotas, and the admission-control policy."),
    ("tier_3", "Derive the consistency requirements for our inventory counter across checkout, warehouse, and analytics, then design the reconciliation process."),
    ("tier_3", "Compare three approaches to zero-downtime schema changes on a sharded cluster and recommend one with a cutover runbook."),
    ("tier_3", "Design a model-registry and rollback system for our routing classifier: artifact versioning, shadow evaluation, and promotion criteria."),
    ("tier_3", "Plan the consolidation of five internal tools into one platform over two quarters, including the migration sequencing and the risk register."),
    ("tier_3", "Design a multi-region active-active deployment for our API gateway, covering conflict resolution, health checks, and traffic failover."),
    ("tier_3", "Write a migration plan to move our customer exports from synchronous API calls to an async object-store workflow, including backpressure and retry semantics."),
    ("tier_3", "Design an audit-log architecture that is tamper-evident, queryable for seven years, and survives region loss; include storage lifecycle and access controls."),
    ("tier_3", "Compare three caching strategies for our pricing service under strict staleness limits; include an invalidation analysis and a recommendation."),
    ("tier_3", "Design the upgrade strategy for a fleet of 500 edge nodes across 12 countries with no downtime and no version skew above one minor release."),
    ("tier_3", "Prove that the sum of two even integers is even, explaining each step of the argument."),
    ("tier_3", "Prove that the square root of 3 is irrational, and explain each step."),
    ("tier_3", "Derive the closed-form solution of the recurrence T(n) = 2T(n/2) + n and prove it by induction."),
    ("tier_3", "Diagnose why our p99 latency doubles every Friday and redesign the affected path: form hypotheses, specify measurements, and propose structural fixes."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/complexity_dataset.jsonl")
    ap.add_argument("--eval-set", default="data/routing_eval.jsonl")
    args = ap.parse_args()

    path = Path(args.dataset)
    original = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 1. Deduplicate the original rows by text (starter file repeats each text ~3x).
    seen: set[str] = set()
    base: list[dict] = []
    for r in original:
        t = r["text"].strip()
        if t not in seen:
            seen.add(t)
            base.append(r)

    # 2. Append curated rows not yet present (script is idempotent: re-running
    # after a previous expansion skips rows that were already merged).
    tier_counts = Counter(r["tier"] for r in base)
    curated_by_tier: dict[str, list[dict]] = {"tier_1": [], "tier_2": [], "tier_3": []}
    for tier, text in CURATED:
        text = text.strip()
        if text in seen:
            continue
        seen.add(text)
        n = len(curated_by_tier[tier]) + 1
        curated_by_tier[tier].append({
            "id": f"cur_{tier.replace('tier_', 't')}_{n:03d}",
            "text": text, "tier": tier,
            "source": "curated_expansion", "reviewed": False,
        })

    # 3. Interleave deterministically: curated rows are round-robined across
    # tiers so the file is not tier-blocked; previously merged rows follow.
    merged: list[dict]
    n_curated = sum(len(v) for v in curated_by_tier.values())
    if n_curated:
        merged = []
        queues = [curated_by_tier[t] for t in ("tier_1", "tier_2", "tier_3")]
        qi = 0
        while any(queues):
            q = queues[qi % 3]
            if q:
                merged.append(q.pop(0))
            qi += 1
            # weave one pre-existing row after every two new curated rows
            if len(merged) % 2 == 0 and base:
                merged.append(base.pop(0))
        merged.extend(base)
    else:
        merged = base

    # 4. Integrity checks before writing.
    counts = Counter(r["tier"] for r in merged)
    texts = [r["text"] for r in merged]
    assert len(texts) == len(set(texts)), "duplicate texts in merged dataset"
    assert all(r["tier"] in ("tier_1", "tier_2", "tier_3") for r in merged)
    assert all(r.get("text") and r.get("id") for r in merged)
    assert min(counts.values()) / max(counts.values()) >= 0.75, f"unbalanced: {counts}"
    eval_path = Path(args.eval_set)
    if eval_path.exists():
        eval_texts = {json.loads(l).get("prompt", "").strip()
                      for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        overlap = eval_texts & set(texts)
        assert not overlap, f"train/eval overlap: {sorted(overlap)[:3]}"

    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8")
    print(f"wrote {len(merged)} unique rows to {path} ({dict(counts)})")
    print(f"curated additions this run: {n_curated} (source=curated_expansion, reviewed=false)")


if __name__ == "__main__":
    main()
