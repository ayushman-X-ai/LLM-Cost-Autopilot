"""Build data/routing_eval_boundary.jsonl: a hand-written held-out stress set.

Unlike data/routing_eval.jsonl (template-generated, deterministic), every row
here is written by hand and targets the decision boundaries that dominate the
error matrix: tier_1 vs tier_2 and tier_2 vs tier_3 look-alikes, plus ambiguous
shading that a human reviewer can relabel. Rows carry "notes" explaining the
trap so disagreement between the classifier and the label is easy to arbitrate.

Labels follow the same rubric as the main eval set:
  tier_1: single mechanical transformation/extraction/recall, no analysis
  tier_2: multi-part drafting/restructuring, comparisons with a recommendation,
          audience explanations, data interpretation, constraint satisfaction
  tier_3: architecture/design with explicit sub-deliverables, formal
          proofs/derivations, distributed correctness, threat modeling,
          migration with rollback, multi-objective optimization

Rebuild after editing:  python scripts/build_boundary_eval.py
Then evaluate with:     python -m autopilot.cli evaluate --dataset data/routing_eval_boundary.jsonl
"""
import argparse, json
from collections import Counter
from pathlib import Path

ROWS = [
    # ------------------------------------------------- tier_1 (mechanical)
    ("tier_1", "Rewrite this sentence in a professional tone: hey, send me the numbers asap.",
     "Short single-step transformation."),
    ("tier_1", "Extract the total amount from this receipt: subtotal $18.38, tax $1.47, total $19.85.",
     "Single-field extraction."),
    ("tier_1", "Convert 42 degrees Fahrenheit to Celsius.",
     "Deterministic conversion."),
    ("tier_1", "Sort these version tags in ascending order: v2.10, v2.2, v1.9, v10.1.",
     "Mechanical ordering, no analysis."),
    ("tier_1", "Return only the phone number contained in this message: Hi, this is Dana from Acme Corp, call me at 555-0143.",
     "Regex-like extraction."),
    ("tier_1", "Change this sentence to past tense: The committee will announce the results tomorrow.",
     "Tense transformation only."),
    ("tier_1", "How many words are in this sentence: the quick brown fox jumps over the lazy dog?",
     "Literal counting."),
    ("tier_1", "Make this bullet lowercase: Added support for the new search API.",
     "Case normalization."),
    ("tier_1", "Classify this customer message as urgent or routine: My card was charged twice this morning.",
     "Binary classification, one label, no analysis required."),
    ("tier_1", "Fix the grammar in this sentence: me and him was late to the meeting again.",
     "Mechanical correction."),
    ("tier_1", "Which of these is a valid IPv4 address: 256.1.1.1, 192.168.1.1, 300.0.0.4, 1.2.3?",
     "Single-fact recall/verification."),
    ("tier_1", "Extract every URL from this email: review the invoice at https://pay.example.com/inv/2291 or read the docs at https://docs.example.com/billing.",
     "Pattern extraction from short text."),

    # ------------------------------------------------- tier_1/trier_2 look-alikes
    ("tier_1", "Rewrite this paragraph in a professional tone: hey team, the deploy broke stuff again and customers are mad, lets figure it out tomorrow.",
     "LOOK-ALIKE: longer input, but still one single transformation."),
    ("tier_1", "Translate this customer review into German and nothing else: The battery life is great but the app keeps crashing.",
     "LOOK-ALIKE: 'review' in text but single mechanical translation."),
    ("tier_1", "Summarize this in exactly five words: The board approved the budget after a lengthy debate about staffing levels.",
     "LOOK-ALIKE: summarization keyword with tight mechanical constraint; still one trivial step."),
    ("tier_1", "List every dollar amount that appears in this message: You were charged $12.99 on March 3 and refunded $6.50 on March 9.",
     "LOOK-ALIKE: multiple extractions but same mechanical operation repeated."),

    # ------------------------------------------------- tier_2
    ("tier_2", "Summarize this article in five bullet points and flag any metrics mentioned: Cloud spend keeps rising as teams duplicate environments and rarely reclaim unused resources.",
     "Multi-point summarization with structure constraint."),
    ("tier_2", "Compare RabbitMQ and Kafka for our notification service and recommend one for a team of six.",
     "Comparison with recommendation."),
    ("tier_2", "Turn these meeting notes into action items with owners and due dates: Sam reviews the spec Wednesday; Priya waits on the vendor quote; launch targeted for the 14th.",
     "Restructuring notes into a deliverable."),
    ("tier_2", "Explain how OAuth works to a non-technical product manager in two paragraphs.",
     "Audience adaptation."),
    ("tier_2", "Rewrite this paragraph in active voice and cut 30 percent of the words while keeping every number: The report was reviewed by the team and it was decided that hiring would be paused by finance until targets are met.",
     "Rewrite with multiple constraints."),
    ("tier_2", "Given weekly signups 120, 96, 141, 133 with conversion slipping from 2.1 to 1.4 percent, explain what changed and suggest a next step.",
     "Data interpretation."),
    ("tier_2", "Draft a polite follow-up email to a client about the delayed shipment that includes a clear call to action.",
     "Drafting with requirements."),
    ("tier_2", "Convert this requirement into acceptance criteria: users must be able to export all data as CSV within 60 seconds.",
     "Requirements transformation."),
    ("tier_2", "Group the feedback below into themes with labels: search is fast but filters reset; love dark mode; price feels high.",
     "Thematic grouping."),
    ("tier_2", "Write three headlines about the pricing change, each in a different tone.",
     "Multiple alternatives with tone constraints."),
    ("tier_2", "Review this SQL for readability without changing results: SELECT u.id, COUNT(o.id) FROM users u JOIN orders o ON o.user_id=u.id GROUP BY u.id.",
     "Code review, behavior-preserving."),
    ("tier_2", "Create a one-week dinner plan under 600 calories per meal using chicken, rice, and seasonal vegetables.",
     "Constraint satisfaction."),

    # ------------------------------------------------- tier_2/tier_3 look-alikes
    ("tier_2", "Explain the difference between SQL and NoSQL databases with one concrete example of each.",
     "LOOK-ALIKE: technical concepts, but explanation only - no design deliverable."),
    ("tier_2", "Describe what a rate limiter does and why APIs need one, in plain language.",
     "LOOK-ALIKE: distributed-systems vocabulary, but descriptive not design."),
    ("tier_2", "List the pros and cons of microservices for our 12-person team.",
     "LOOK-ALIKE: architecture topic but a plain enumeration, not a justified design."),
    ("tier_2", "Summarize the key ideas of the CAP theorem and give one example of each tradeoff.",
     "LOOK-ALIKE: deep topic, but recall-and-explain, not derivation or design."),
    ("tier_2", "Draft a two-day conference agenda about developer tooling with four sessions and two breaks.",
     "LOOK-ALIKE: multi-part planning but low technical depth."),
    ("tier_2", "Compare token bucket and sliding window rate limiting and explain when each is a better fit.",
     "LOOK-ALIKE: protocol comparison without a deployment justification or runbook."),

    # ------------------------------------------------- tier_3
    ("tier_3", "Design a multi-tenant billing service with per-tenant budgets, idempotent webhook delivery, and an audit trail; cover the data model, consistency boundaries, and retry policy.",
     "Design with explicit sub-deliverables."),
    ("tier_3", "Prove that the square root of 2 is irrational, explaining each step of the argument.",
     "Formal proof."),
    ("tier_3", "Architect a zero-downtime migration from a monolith to services including dual-write reconciliation, backfill, and rollback.",
     "Migration design with rollback."),
    ("tier_3", "Threat model our e-signature flow with STRIDE, enumerate mitigations, and state residual risk.",
     "Threat modeling."),
    ("tier_3", "Design a rate limiter that stays correct across 200 nodes with clock skew; compare approaches and justify the choice.",
     "Distributed correctness under skew."),
    ("tier_3", "Derive the time and space complexity of this algorithm and propose a justified optimization: def solve(items): ... nested pair loop ...",
     "Derivation plus optimization."),
    ("tier_3", "Plan the staged rollout of a new ranking model to 50M users with cohort gating and automated rollback.",
     "Large-scale rollout design."),
    ("tier_3", "Design a reliability strategy for our webhook service: SLOs, error budgets, chaos experiments, and escalation policy.",
     "Cross-cutting reliability design."),
    ("tier_3", "Compare Raft, Paxos, and EPaxos for a geo-distributed database and justify the choice including partition behavior.",
     "Protocol comparison with operational justification."),
    ("tier_3", "Specify the retry and deduplication contract between two services formally, including poison-message handling.",
     "Formal specification with invariants."),
    ("tier_3", "We must cut p95 latency below 300ms with no public API changes and no budget increase; compare three strategies, quantify tradeoffs, and recommend one.",
     "Multi-objective optimization."),
    ("tier_3", "Design an evaluation framework for our routing classifier with dataset construction, significance testing, and regression detection.",
     "Framework design with statistics."),

    # ------------------------------------------------- tier_3 look-alikes (long but lower tier)
    ("tier_2", "Write a detailed blog post explaining how we reduced p99 latency by 40 percent, covering the symptoms, the investigation, and the fix, in a friendly tone.",
     "LOOK-ALIKE: long and technical, but it is narrative writing, not design or derivation."),
    ("tier_2", "Create a study plan for a engineer preparing for a staff-level systems design interview over four weeks, with weekly goals and reading list.",
     "LOOK-ALIKE: planning with constraints, but generic planning rather than system design."),
    ("tier_2", "Rewrite this incident postmortem so it is suitable for the public status page, keeping all facts: at 14:02 the database failed over; requests failed for 9 minutes; no data was lost.",
     "LOOK-ALIKE: reliability vocabulary but a rewriting task with constraints."),

    # ------------------------------------------------- ambiguous (human review)
    ("tier_2", "Summarize the attached 20-page research paper in one page.",
     "AMBIGUOUS: heavy comprehension but single-step output; placed in tier_2 - relabel if your annotation guide disagrees."),
    ("tier_2", "Turn this customer interview transcript into a prioritized feature list with rationale.",
     "AMBIGUOUS: extraction plus judgment; placed in tier_2."),
    ("tier_3", "Design a data model for multi-region user profiles with offline edits and deterministic conflict resolution.",
     "AMBIGUOUS: design deliverable but narrower scope than full architecture; placed in tier_3."),
    ("tier_1", "Extract the three most important action items from this email: ...",
     "AMBIGUOUS: 'most important' adds judgment; placed in tier_1 mechanically, flag for review."),
]

def build() -> list[dict]:
    rows = []
    for n, (tier, prompt, note) in enumerate(ROWS, start=1):
        rows.append({"id": f"bnd_{n:03d}", "prompt": prompt, "expected_tier": tier, "reason": note})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/routing_eval_boundary.jsonl")
    args = ap.parse_args()
    rows = build()
    texts = [r["prompt"] for r in rows]
    assert len(texts) == len(set(texts)), "duplicate prompts"
    counts = Counter(r["expected_tier"] for r in rows)
    # Train-set disjointness check.
    train_path = Path("data/complexity_dataset.jsonl")
    if train_path.exists():
        train_texts = {json.loads(l)["text"].strip()
                       for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        overlap = train_texts & set(texts)
        assert not overlap, f"boundary eval overlaps train set: {sorted(overlap)[:3]}"
    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out} ({dict(counts)})")

if __name__ == "__main__":
    main()
