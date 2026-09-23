"""Dataset integrity guards: balance, uniqueness, and train/eval disjointness.

These tests prevent the two failure modes that shipped originally: a training
file padded with duplicate rows (72 unique texts across 210 rows) and
train/eval overlap that inflates reported accuracy.
"""
import json
from collections import Counter
from pathlib import Path

TRAIN = Path("data/complexity_dataset.jsonl")
EVAL = Path("data/routing_eval.jsonl")
BOUNDARY = Path("data/routing_eval_boundary.jsonl")


def _rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_training_dataset_is_deduplicated():
    rows = _rows(TRAIN)
    texts = [r["text"].strip() for r in rows]
    dupes = {t for t, c in Counter(texts).items() if c > 1}
    assert not dupes, f"duplicate training texts: {sorted(dupes)[:3]}"


def test_training_dataset_is_balanced():
    rows = _rows(TRAIN)
    counts = Counter(r["tier"] for r in rows)
    assert set(counts) == {"tier_1", "tier_2", "tier_3"}
    assert min(counts.values()) / max(counts.values()) >= 0.75, counts


def test_training_rows_are_wellformed():
    rows = _rows(TRAIN)
    for r in rows:
        assert r.get("id"), f"row missing id: {r}"
        assert len(r["text"].strip()) >= 10, f"row text too short: {r['id']}"
        assert r["tier"] in {"tier_1", "tier_2", "tier_3"}


def test_train_and_eval_sets_are_disjoint():
    train_texts = {r["text"].strip() for r in _rows(TRAIN)}
    eval_texts = {(r.get("prompt") or r.get("text")).strip() for r in _rows(EVAL)}
    overlap = train_texts & eval_texts
    assert not overlap, f"train/eval overlap inflates accuracy: {sorted(overlap)[:3]}"


def test_boundary_eval_is_disjoint_and_annotated():
    rows = _rows(BOUNDARY)
    assert len(rows) >= 40, "boundary stress set shrank unexpectedly"
    train_texts = {r["text"].strip() for r in _rows(TRAIN)}
    for r in rows:
        assert (r.get("prompt") or "").strip() not in train_texts
        assert r.get("reason"), f"boundary row missing annotation note: {r['id']}"
    counts = Counter(r["expected_tier"] for r in rows)
    assert set(counts) == {"tier_1", "tier_2", "tier_3"}


def test_eval_set_has_expected_shape():
    rows = _rows(EVAL)
    assert len(rows) >= 300
    counts = Counter(r.get("expected_tier") for r in rows)
    assert set(counts) == {"tier_1", "tier_2", "tier_3"}
    assert min(counts.values()) >= 100, counts
