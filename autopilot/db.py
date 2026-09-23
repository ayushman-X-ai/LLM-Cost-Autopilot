import json, sqlite3, time
from pathlib import Path
from typing import Any
from .settings import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
 id TEXT PRIMARY KEY,
 timestamp REAL NOT NULL,
 prompt_hash TEXT NOT NULL,
 prompt_text TEXT,
 complexity_tier TEXT,
 routed_model TEXT,
 provider TEXT,
 input_tokens INTEGER,
 output_tokens INTEGER,
 cost_usd REAL,
 latency_ms REAL,
 quality_score REAL,
 escalated INTEGER DEFAULT 0,
 escalation_model TEXT,
 status TEXT DEFAULT 'completed',
 verification_error TEXT,
 verification_mode TEXT,
 routing_event TEXT,
 original_model TEXT,
 fallback_model TEXT,
 fallback_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(routed_model);
CREATE TABLE IF NOT EXISTS routing_versions (
 version INTEGER PRIMARY KEY AUTOINCREMENT,
 created_at REAL NOT NULL,
 config TEXT NOT NULL,
 created_by TEXT
);
"""

# Columns added after the first release; connect() migrates older databases.
EXTRA_COLUMNS = {
    "verification_mode": "TEXT",
    "routing_event": "TEXT",
    "original_model": "TEXT",
    "fallback_model": "TEXT",
    "fallback_reason": "TEXT",
}

def connect():
    s=get_settings(); Path(s.database_path).parent.mkdir(parents=True, exist_ok=True)
    c=sqlite3.connect(s.database_path, timeout=30); c.row_factory=sqlite3.Row
    # WAL + NORMAL keep concurrent request/verification writes from serialising on fsync.
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA busy_timeout=10000")
    c.executescript(SCHEMA)
    existing={r["name"] for r in c.execute("PRAGMA table_info(requests)")}
    for col, decl in EXTRA_COLUMNS.items():
        if col not in existing:
            c.execute(f"ALTER TABLE requests ADD COLUMN {col} {decl}")
    return c

def insert_request(row: dict[str, Any]):
    cols=list(row.keys()); vals=[row[c] for c in cols]
    with connect() as c:
        c.execute(f"INSERT OR REPLACE INTO requests ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})", vals)

def get_request(request_id: str):
    with connect() as c:
        r=c.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
        return dict(r) if r else None

def update_request(request_id: str, **fields):
    if not fields: return
    with connect() as c:
        c.execute(f"UPDATE requests SET {','.join(k+'=?' for k in fields)} WHERE id=?", [*fields.values(), request_id])

def recent(limit=1000):
    with connect() as c: return [dict(r) for r in c.execute("SELECT * FROM requests ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()]

def stats():
    with connect() as c:
        total=c.execute("SELECT COUNT(*) n FROM requests").fetchone()["n"]
        cost=c.execute("SELECT COALESCE(SUM(cost_usd),0) n FROM requests").fetchone()["n"]
        escalated=c.execute("SELECT COALESCE(SUM(escalated),0) n FROM requests").fetchone()["n"]
        avg_latency=c.execute("SELECT COALESCE(AVG(latency_ms),0) n FROM requests").fetchone()["n"]
        avg_quality=c.execute("SELECT AVG(quality_score) n FROM requests WHERE quality_score IS NOT NULL").fetchone()["n"]
        input_tokens=c.execute("SELECT COALESCE(SUM(input_tokens),0) n FROM requests").fetchone()["n"]
        output_tokens=c.execute("SELECT COALESCE(SUM(output_tokens),0) n FROM requests").fetchone()["n"]
        verified=c.execute("SELECT COUNT(*) n FROM requests WHERE quality_score IS NOT NULL").fetchone()["n"]
        by_model=[dict(r) for r in c.execute("SELECT routed_model model, COUNT(*) count, COALESCE(SUM(cost_usd),0) cost FROM requests GROUP BY routed_model ORDER BY count DESC").fetchall()]
        by_tier=[dict(r) for r in c.execute("SELECT complexity_tier tier, COUNT(*) count FROM requests GROUP BY complexity_tier ORDER BY complexity_tier").fetchall()]
        fallbacks=c.execute("SELECT COUNT(*) n FROM requests WHERE routing_event='provider_failure'").fetchone()["n"]
        return {"requests":total,"cost_usd":cost,"escalations":escalated,"escalation_rate":(escalated/total if total else 0),
                "avg_latency_ms":avg_latency,"avg_quality":avg_quality,
                "input_tokens":input_tokens,"output_tokens":output_tokens,
                "verified_requests":verified,"fallbacks":fallbacks,
                "by_model":by_model,"by_tier":by_tier}

def insert_routing_version(config: dict, created_by: str = "api") -> int:
    with connect() as c:
        cur=c.execute("INSERT INTO routing_versions (created_at, config, created_by) VALUES (?,?,?)",
                      (time.time(), json.dumps(config), created_by))
        return int(cur.lastrowid)

def latest_routing_version():
    with connect() as c:
        r=c.execute("SELECT * FROM routing_versions ORDER BY version DESC LIMIT 1").fetchone()
        return dict(r) if r else None

def list_routing_versions(limit: int = 50):
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM routing_versions ORDER BY version DESC LIMIT ?", (limit,)).fetchall()]

def get_routing_version(version: int):
    with connect() as c:
        r=c.execute("SELECT * FROM routing_versions WHERE version=?", (version,)).fetchone()
        return dict(r) if r else None
