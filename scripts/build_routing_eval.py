"""Build data/routing_eval.jsonl: a held-out, labeled routing evaluation set.

Deterministic and reproducible (no randomness, no LLM calls). Labels are
assigned by construction: each template is a known complexity tier, so every
generated prompt carries a defensible expected_tier and a short reason.

Usage:
    python scripts/build_routing_eval.py [--out data/routing_eval.jsonl]
"""
import argparse, json
from pathlib import Path

TOPICS = [
    "the quarterly roadmap", "a customer onboarding flow", "our new pricing page",
    "the mobile checkout redesign", "the data retention policy", "the incident review process",
    "a partner integration", "the search ranking change", "API rate limits",
    "the email notification system", "the analytics pipeline", "the support escalation path",
    "the release checklist", "the vendor selection process", "the on-call rotation",
    "the feature flag rollout",
]
SENTENCES = [
    "please send me the report",
    "i will be late to the meeting",
    "we need this done by friday",
    "the server is down again",
    "can you take a look at this",
    "thanks for helping with the bug",
    "the meeting moved to tuesday",
    "i did not get the invoice",
    "lets ship it next week",
    "the customer is unhappy about the delay",
    "the deploy failed during the night",
    "we should talk before the launch",
]
MESSAGES = [
    "Order 8841 was placed by sam@example.com on March 3",
    "Hi, this is Dana from Acme Corp, reach me at 555-0143",
    "Invoice INV-2291 for $4,120 is attached, PO number 778",
    "Ticket 4412: customer jamie@example.com cannot log in",
    "Shipment tracking 1Z999AA10123456784 arrives Tuesday",
    "Please wire payment to account 55-1290, ref NW-204",
    "Meeting invite: luis@example.com, Thursday 3pm, room B",
    "Refund request for order 5521 from kate@example.com",
]
VALUES = [
    '["red", "blue", "green"]', '{"name": "Ada", "age": 36}',
    '["10", "20", "30"]', '{"city": "Lisbon", "zip": "1000"}',
    '["a=1", "b=2"]', '{"enabled": true, "retries": 3}',
    '["2026-01-05", "2026-02-10"]', '{"qty": 7, "sku": "X-99"}',
]
NAMES = ["Zara, Omar, Lena, Kai", "Miguel, Anna, Petro, Bella", "Nina, Oscar, Farah, Liam",
         "Yuki, Pavel, Ines, Marco", "Tara, Hugo, Sofia, Deniz", "Ravi, Elena, Jonah, Mei"]
WORDS_API = ["API", "cache", "latency", "queue", "token", "index", "webhook", "schema"]
ARTICLES = [
    "Artificial intelligence is changing software engineering by making code generation faster, but teams still need evaluation and governance.",
    "Remote work has shifted commuting patterns; cities now face lower transit ridership and different retail demand downtown.",
    "The new regulation requires companies to document data retention periods and report breaches within 72 hours.",
    "Battery costs fell again this year, accelerating electric vehicle adoption in markets with strong charging infrastructure.",
    "Open-source maintainers report burnout as usage grows without matching funding or contributor support.",
    "Cloud spend keeps rising as teams duplicate environments and rarely reclaim unused resources.",
]
COMPLAINTS = [
    "the app crashed twice today and i lost my draft, totally unacceptable",
    "charged twice for the same order and support never replied",
    "the delivery was late and the box was damaged, again",
    "i cannot log in after the password reset, no email arrives",
    "your driver left the package in the rain",
    "the discount code did not apply at checkout even though it was valid",
]
NOTES = [
    "Sam will review the spec by Wednesday; Priya waits on the vendor quote; launch still targeted for the 14th; risk flagged around staffing.",
    "We agreed to cut scope, keep the import feature, drop the themes work, and revisit in Q3; Dan owns the plan.",
    "Budget approved for two hires; office move delayed until lease signed; security review booked for Friday.",
    "Customer said onboarding is confusing; they want a checklist and sample data; renew in June if improved.",
    "Blocked on legal approval; API design mostly done; testing starts once staging is stable.",
]
FEEDBACK = [
    "The search is fast but filters reset too often. Love the dark mode. Price feels high for small teams.",
    "Setup took hours and docs skipped auth. Support was friendly. Would like SSO on the cheap plan.",
    "Mobile app crashes when uploading photos. Desktop works fine. Notifications arrive late.",
    "Reports are useful but export is slow and CSV columns are mislabeled.",
    "Onboarding emails were helpful; in-app tooltips were confusing and covered the buttons.",
]
GOALS = [
    "cut p95 checkout latency below 400ms", "migrate billing to the new provider",
    "reduce cloud spend by 30%", "improve trial-to-paid conversion",
    "prepare the service for a SOC 2 audit", "launch the public API beta",
]
QUERIES = [
    "SELECT u.id, COUNT(o.id) FROM users u JOIN orders o ON o.user_id=u.id GROUP BY u.id;",
    "SELECT * FROM events WHERE created_at > NOW() - INTERVAL '7 days' ORDER BY created_at DESC;",
    "SELECT p.name, SUM(l.qty) FROM products p JOIN line_items l ON l.product_id=p.id GROUP BY p.name;",
    "UPDATE accounts SET balance = balance - 100 WHERE id = 42;",
    "SELECT DISTINCT session_id FROM page_views WHERE url LIKE '/pricing%';",
]
DATAS = [
    "week 1: 120 signups, week 2: 96, week 3: 141, week 4: 133, paid conversion 2.1% -> 1.4%",
    "latency by region: eu 210ms, us 180ms, apac 460ms; error rate 0.7% in apac only",
    "churn 4.2% -> 5.1% after the price change; cancellations cite cost",
    "searches without results rose from 8% to 17% after the index swap",
]
REQUIREMENTS = [
    "users must be able to export all data as CSV within 60 seconds",
    "admins can revoke a session and see who revoked it",
    "the report must run on a schedule and email a PDF",
    "guests can checkout without an account but may later claim the order",
]
CONSTRAINTS = [
    "budget under $5k/month, p99 under 300ms, and no single region dependency",
    "must survive a zone failure, keep 99.95% availability, and stay under 200ms p95",
    "team of four, twelve-week deadline, and no changes to the public API",
    "GDPR data residency, audit logging, and under 500ms for 10k concurrent users",
]
THEOREMS = [
    "the sum of the first n odd numbers equals n squared",
    "there are infinitely many prime numbers",
    "the square root of 2 is irrational",
    "every forest has at least one tree of odd degree",
    "the composition of two bijections is a bijection",
]
ALGOS = [
    "def solve(items):\n    best = 0\n    for i in range(len(items)):\n        for j in range(i+1, len(items)):\n            best = max(best, items[i] + items[j])\n    return best",
    "def find(xs, target):\n    for i, x in enumerate(xs):\n        if x == target:\n            return i\n        for y in xs[i+1:]:\n            if x + y == target:\n                return (i, xs.index(y))\n    return -1",
    "def dedupe(xs):\n    out = []\n    for x in xs:\n        if xs.count(x) == 1:\n            out.append(x)\n    return out",
    "def path(g, a, b):\n    seen = set()\n    def dfs(n):\n        seen.add(n)\n        for m in g[n]:\n            if m == b: return True\n            if m not in seen and dfs(m): return True\n        return False\n    return dfs(a)",
]
SCENARIOS = [
    "a payments platform with strict audit requirements",
    "a consumer chat product with 50M daily messages",
    "an internal analytics service for a 200-person company",
    "a healthcare startup handling regulated records",
    "an e-commerce marketplace during flash sales",
]
DOMAINS = [
    "order management", "clinical scheduling", "freight dispatch",
    "digital payments", "content moderation", "inventory forecasting",
]
SYSTEMS = [
    "an LLM prompt playground", "a webhook delivery service",
    "a multi-tenant billing service", "a real-time collaboration editor",
    "a document e-signature flow", "an ad bidding engine",
]
FEATURES = ["team workspaces", "the new search API", "usage-based billing", "offline mode", "two-factor authentication", "shared dashboards"]
TASKS = ["connect a domain, configure DNS, and verify the certificate", "invite teammates and assign roles", "set up a billing alert before launch", "export a report and schedule it to email weekly", "recover a deleted project from the backup list"]
METRICS = ["weekly active teams", "checkout conversion", "support first-response time", "trial-to-paid rate", "infrastructure cost per request"]
CONCEPTS = ["rate limiting", "database indexing", "event-driven architecture", "idempotency", "caching", "load shedding", "feature flags", "observability"]
CONCEPT_PAIRS = [("SQL", "NoSQL"), ("REST", "GraphQL"), ("queueing", "batching"), ("authentication", "authorization"), ("horizontal", "vertical scaling"), ("TCP", "UDP")]
COUNTRIES = ["France", "Japan", "Brazil", "Kenya", "Norway", "Canada"]
DATES = ["March 3, 2026", "12/01/2025", "the 5th of June 2024", "Aug 17 2023"]
WORDBANK = ["bright", "quick", "happy", "solid", "clear", "narrow"]
SYNONYMS = [("happy", "glad sad tired"), ("quick", "fast slow early"), ("narrow", "wide thin broad"), ("bright", "dark shiny dull"), ("solid", "liquid firm hollow"), ("calm", "quiet angry rough")]
ADJECTIVES = ["polished", "urgent", "friendly", "neutral", "confident", "apologetic"]
FEATURES_BENEFIT = ["faster imports", "shared dashboards", "audit logs", "custom fields", "bulk editing", "saved views"]

TEMPLATES = [
    # ---------------------------------------------------------------- tier_1
    ("tier_1", "Single-step transformation or extraction with no analysis required.",
     "Rewrite this sentence in a professional tone: {sentence}.", {"sentence": SENTENCES}),
    ("tier_1", "Single field extraction from a short message.",
     "Extract the order number from this message: {message}.", {"message": MESSAGES}),
    ("tier_1", "Binary text classification, one label.",
     "Classify this support ticket as billing or technical: {message}", {"message": MESSAGES}),
    ("tier_1", "Mechanical grammar correction only.",
     "Fix the grammar in this sentence: i think {sentence}", {"sentence": SENTENCES}),
    ("tier_1", "Direct translation, single sentence.",
     "Translate this sentence into Spanish: {sentence}.", {"sentence": SENTENCES}),
    ("tier_1", "Format conversion to JSON with no reasoning.",
     "Convert these values to JSON: {values}", {"values": VALUES}),
    ("tier_1", "Deterministic sorting of a short list.",
     "Sort these names alphabetically: {names}", {"names": NAMES}),
    ("tier_1", "One-line definition lookup.",
     "Give me a one-line definition of {word}.", {"word": WORDS_API}),
    ("tier_1", "Table formatting of given values.",
     "Turn this list into a markdown table: {names}", {"names": NAMES}),
    ("tier_1", "Compression to one sentence, no analysis.",
     "Summarize this note in one sentence: {note}", {"note": NOTES}),
    ("tier_1", "Extract a single contact detail.",
     "Return the email address found here: {message}", {"message": MESSAGES}),
    ("tier_1", "Title truncation/rewrite only.",
     "Rewrite this title to be under ten words: {topic}.", {"topic": TOPICS}),
    ("tier_1", "Tense change only.",
     "Change this sentence to past tense: {sentence}.", {"sentence": SENTENCES}),
    ("tier_1", "Simple enumeration of named items.",
     "List the colors mentioned: the report is red, the chart is {word}, the flag is green.", {"word": WORDS_API}),
    ("tier_1", "Unit conversion, no context needed.",
     "Convert {n} kilometers to miles.", {"n": ["3", "12", "45", "100", "7", "250", "18"]}),
    ("tier_1", "Case normalization only.",
     "Make this bullet point lowercase: Added support for {topic}.", {"topic": TOPICS}),
    ("tier_1", "Fill-in-the-blank recall question.",
     "Fill in the blank: The capital of {country} is ____.", {"country": COUNTRIES}),
    ("tier_1", "Odd-one-out over a tiny list.",
     "Which word does not belong: {a} {b} {c} {d}?", {"a": WORDBANK, "b": WORDBANK, "c": WORDBANK, "d": WORDBANK}),
    ("tier_1", "Short creative line, single constraint.",
     "Write a subject line for an email about {topic}.", {"topic": TOPICS}),
    ("tier_1", "Sentence splitting only.",
     "Split this sentence into two sentences: {sentence} and it matters.", {"sentence": SENTENCES}),
    ("tier_1", "Literal word counting.",
     "How many words are in this sentence: {sentence}?", {"sentence": SENTENCES}),
    ("tier_1", "Name extraction from structured text.",
     "Extract the person's name from: {message}", {"message": MESSAGES}),
    ("tier_1", "Date reformatting only.",
     "Reformat this date as YYYY-MM-DD: {date}.", {"date": DATES}),
    ("tier_1", "Synonym selection from a fixed list.",
     "Pick the synonym of '{word}' from: {options}", {"pairs": SYNONYMS}),
    ("tier_1", "Truncation to a word budget.",
     "Truncate this text to five words: {sentence}.", {"sentence": SENTENCES}),
    # ---------------------------------------------------------------- tier_2
    ("tier_2", "Multi-point summarization with a structure constraint.",
     "Summarize the following text in five bullet points: {article}", {"article": ARTICLES}),
    ("tier_2", "Comparison plus per-option pros/cons.",
     "Compare {a} and {b} for a small team and list three pros and cons of each.", {"a": CONCEPTS, "b": CONCEPTS}),
    ("tier_2", "Drafting a message with explicit requirements.",
     "Draft a polite follow-up email to a client about {topic} that includes a clear call to action.", {"topic": TOPICS}),
    ("tier_2", "Audience adaptation across two paragraphs.",
     "Explain how {concept} works to a non-technical product manager in two paragraphs.", {"concept": CONCEPTS}),
    ("tier_2", "Restructuring unstructured feedback into a bug report.",
     "Rewrite this customer complaint as a concise bug report with steps to reproduce: {complaint}", {"complaint": COMPLAINTS}),
    ("tier_2", "Thematic grouping over several items.",
     "Analyze the feedback below and group it into themes: {feedback}", {"feedback": FEEDBACK}),
    ("tier_2", "Notes to structured action items with owners.",
     "Turn these meeting notes into an action item list with owners and due dates: {note}", {"note": NOTES}),
    ("tier_2", "Comparison with a recommendation for a scenario.",
     "Compare the options below and recommend one for {scenario}: {a} vs {b}", {"scenario": SCENARIOS, "a": CONCEPTS, "b": CONCEPTS}),
    ("tier_2", "Summary plus risk extraction.",
     "Summarize this article and extract three risks the author mentions: {article}", {"article": ARTICLES}),
    ("tier_2", "Announcement drafting with three required elements.",
     "Write a release announcement for {feature} covering benefits, migration steps, and a rollback plan.", {"feature": FEATURES}),
    ("tier_2", "Concept differentiation with examples.",
     "Explain the difference between {a} and {b} with one concrete example each.", {"pairs2": CONCEPT_PAIRS}),
    ("tier_2", "Constraint satisfaction over a set of items.",
     "Create a one-week dinner plan under 600 calories per meal using: {items}", {"items": VALUES}),
    ("tier_2", "Code review suggestions without changing behavior.",
     "Review this SQL query and suggest improvements while preserving its output: {query}", {"query": QUERIES}),
    ("tier_2", "Multiple alternatives with different tones.",
     "Draft three alternative headlines about {topic}, each in a different tone.", {"topic": TOPICS}),
    ("tier_2", "Task decomposition with dependencies.",
     "Break this goal into a prioritized task list with dependencies: {goal}", {"goal": GOALS}),
    ("tier_2", "Data interpretation to explain a change.",
     "Explain why the metric moved given this data: {data}", {"data": DATAS}),
    ("tier_2", "Requirements to acceptance criteria conversion.",
     "Convert this requirement into acceptance criteria: {req}", {"req": REQUIREMENTS}),
    ("tier_2", "Thread summarization plus answering a question.",
     "Summarize this conversation thread and answer the final question: {note}", {"note": NOTES}),
    ("tier_2", "Pros/cons analysis of a migration decision.",
     "Weigh the pros and cons of migrating {a} to {b} for a mid-size team.", {"a": CONCEPTS, "b": CONCEPTS}),
    ("tier_2", "Step-by-step tutorial with pitfalls.",
     "Write a short tutorial that walks a new user through {task}, including likely pitfalls.", {"task": TASKS}),
    ("tier_2", "Structured recap of a decision with rationale.",
     "Summarize this decision log entry, including the rationale and open questions: {note}", {"note": NOTES}),
    ("tier_2", "Comparative recommendation grounded in data.",
     "Given this week's numbers, which of {a} or {b} should the team prioritize next: {data}", {"a": CONCEPTS, "b": CONCEPTS, "data": DATAS}),
    ("tier_2", "Multi-constraint planning.",
     "Draft a two-day conference agenda about {topic} with four sessions and two breaks.", {"topic": TOPICS}),
    ("tier_2", "Explanation plus implications for a stakeholder.",
     "Explain what {metric} trend implies for {scenario} and what to do next.", {"metric": METRICS, "scenario": SCENARIOS}),
    ("tier_2", "Rewrite with multiple style and content constraints.",
     "Rewrite the paragraph below in active voice, cut 30% of the words, and keep all numbers: {article}", {"article": ARTICLES}),
    # ---------------------------------------------------------------- tier_3
    ("tier_3", "System design with data model, consistency, retry and topology.",
     "Design a multi-tenant {system} that routes work across three providers, enforces per-tenant budgets, supports streaming, and stays observable during outages. Cover the data model, consistency boundaries, retry policy, and deployment topology.", {"system": SYSTEMS}),
    ("tier_3", "Formal proof with step-by-step derivation.",
     "Prove that {theorem}, and explain each step of the argument.", {"theorem": THEOREMS}),
    ("tier_3", "Distributed correctness under clock skew with protocol comparison.",
     "Design a rate limiter that stays correct across 200 distributed nodes with clock skew; compare token bucket and sliding window approaches and justify your choice for {scenario}.", {"scenario": SCENARIOS}),
    ("tier_3", "Zero-downtime migration plan with rollback strategy.",
     "Architect a zero-downtime migration of {domain} from a monolith to services, including data consistency, backfill, rollback, and cutover strategy.", {"domain": DOMAINS}),
    ("tier_3", "Storage-paradigm tradeoff analysis with failure modes.",
     "Analyze the tradeoffs between event sourcing and CRUD for {domain} under high write throughput, include failure-mode analysis, and recommend one.", {"domain": DOMAINS}),
    ("tier_3", "Complexity derivation plus algorithmic optimization.",
     "Derive the time and space complexity of this algorithm, then propose and justify an optimization:\n{algo}", {"algo": ALGOS}),
    ("tier_3", "Threat modeling across STRIDE with residual risk.",
     "Threat-model {system} for a regulated fintech deployment: enumerate STRIDE categories, mitigations, and residual risk.", {"system": SYSTEMS}),
    ("tier_3", "Distributed scheduler design with exactly-once semantics.",
     "Write an implementation plan for a distributed job scheduler with exactly-once semantics for {scenario}, covering idempotency keys, reconciliation, backpressure, and observability.", {"scenario": SCENARIOS}),
    ("tier_3", "Consensus protocol comparison for a geo-distributed database.",
     "Compare Raft, Paxos, and EPaxos for a geo-distributed database and justify which to use for {scenario}, including partition behavior and operational cost.", {"scenario": SCENARIOS}),
    ("tier_3", "Evaluation framework design with statistics and regression detection.",
     "Design an evaluation framework for LLM routing quality across {scenario}: metrics, dataset construction, statistical significance, and how to detect regressions after model updates.", {"scenario": SCENARIOS}),
    ("tier_3", "Capacity/cost optimization under multiple hard constraints.",
     "Given the constraints ({constraints}), propose an architecture for {domain} and show how you would validate it before launch.", {"constraints": CONSTRAINTS, "domain": DOMAINS}),
    ("tier_3", "End-to-end incident and reliability strategy.",
     "Design a reliability strategy for {system} handling {constraints}: SLOs, error budgets, chaos experiments, and escalation policy.", {"system": SYSTEMS, "constraints": CONSTRAINTS}),
    ("tier_3", "Multi-objective optimization with explicit tradeoffs.",
     "You must ship {goal} under {constraints}. Compare at least three strategies, quantify the tradeoffs, and recommend one with a justification.", {"goal": GOALS, "constraints": CONSTRAINTS}),
    ("tier_3", "Security architecture review with adversarial analysis.",
     "Review the security architecture of {system} used by {scenario}: enumerate attack paths, detection signals, and a prioritized remediation plan.", {"system": SYSTEMS, "scenario": SCENARIOS}),
    ("tier_3", "Migration/consistency design for a specific domain.",
     "Design the data migration and dual-write reconciliation plan for moving {domain} to a new storage layer with zero data loss.", {"domain": DOMAINS}),
    ("tier_3", "Research-grade synthesis with competing evidence.",
     "Synthesize the arguments for and against {a} in {scenario}, identify where the evidence conflicts, and state what experiment would resolve it.", {"a": CONCEPTS, "scenario": SCENARIOS}),
    ("tier_3", "Cost/latency/quality routing policy derivation.",
     "Derive a routing policy that minimizes cost while keeping p95 latency under 300ms and quality above a 4.0 threshold for {scenario}. Specify the classifier target, escalation rules, and evaluation plan.", {"scenario": SCENARIOS}),
    ("tier_3", "Formal specification with invariants and edge cases.",
     "Specify {system} formally: state the invariants, enumerate edge cases and concurrent interleavings, and describe how the implementation enforces them.", {"system": SYSTEMS}),
    ("tier_3", "Regulatory compliance architecture with conflicting requirements.",
     "Design an architecture for {domain} that satisfies {constraints}: address data residency, retention, auditability, and the conflicts between them.", {"domain": DOMAINS, "constraints": CONSTRAINTS}),
    ("tier_3", "Large-scale rollout plan with sequencing and risk.",
     "Plan the staged rollout of {feature} to 50M users with automated rollback, cohort analysis, and no downtime; include the metric gate for each stage.", {"feature": FEATURES}),
    ("tier_3", "Cross-cutting performance diagnosis and redesign.",
     "Diagnose and redesign {system} where {metric} degrades under load: form hypotheses, specify measurements, and propose structural fixes rather than tuning.", {"system": SYSTEMS, "metric": METRICS}),
    ("tier_3", "Protocol-level comparison under adversarial conditions.",
     "Compare approaches to leader election and failover in {scenario} under network partitions, including split-brain prevention and the operator runbook.", {"scenario": SCENARIOS}),
]

def build(max_rows: int | None = None) -> list[dict]:
    rows = []
    for tier, reason, template, pools in TEMPLATES:
        keys = list(pools)
        count = max(len(v) for v in pools.values())
        for i in range(count):
            # offset each slot by its position so {a}/{b} never render the same item
            fill = {k: pools[k][(i + pos) % len(pools[k])] for pos, k in enumerate(keys)}
            # tuple-slot templates unpack positionally for readability
            if "pairs2" in fill:
                a, b = fill.pop("pairs2"); fill["a"], fill["b"] = a, b
            if "pairs" in fill:
                word, options = fill.pop("pairs"); fill["word"], fill["options"] = word, options
            rows.append({"prompt": template.format(**fill), "expected_tier": tier, "reason": reason})
    # interleave tiers deterministically so the file is not tier-blocked
    by_tier: dict[str, list[dict]] = {}
    for r in rows:
        by_tier.setdefault(r["expected_tier"], []).append(r)
    ordered, i = [], 0
    while any(by_tier.values()):
        tier = ["tier_1", "tier_2", "tier_3"][i % 3]
        if by_tier.get(tier):
            ordered.append(by_tier[tier].pop(0))
        i += 1
    out = []
    for n, r in enumerate(ordered, start=1):
        out.append({"id": f"route_{n:03d}", **r})
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/routing_eval.jsonl")
    a = p.parse_args()
    rows = build()
    seen, unique = set(), []
    for r in rows:
        if r["prompt"] in seen: continue
        seen.add(r["prompt"]); unique.append(r)
    out = Path(a.out); out.parent.mkdir(exist_ok=True)
    cap = 500  # keep the eval set inside the recommended 300-500 range
    unique = unique[:cap]
    with out.open("w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {t: sum(1 for r in unique if r["expected_tier"] == t) for t in ("tier_1", "tier_2", "tier_3")}
    print(f"wrote {len(unique)} rows to {out} ({counts})")

if __name__ == "__main__":
    main()
