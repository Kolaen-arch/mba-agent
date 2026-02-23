"""
Database: SQLite persistence for chat sessions, messages, and document versions.
Uses thread-local connections for safe concurrent access.
"""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional


DB_PATH = "./data/mba_agent.db"

_local = threading.local()


def get_db(db_path: str | None = None) -> sqlite3.Connection:
    """Get or create a thread-local database connection."""
    db_path = db_path or DB_PATH
    conn = getattr(_local, 'conn', None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.ProgrammingError:
            # Connection was closed, create new one
            pass

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _local.conn = conn
    return conn


@contextmanager
def get_db_ctx(db_path: str | None = None):
    """Context manager for one-off database operations with a fresh connection."""
    db_path = db_path or DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with get_db_ctx(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'chat',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            context_sources TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS doc_versions (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            change_summary TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS session_costs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_create_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_usd REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_doc_versions_file ON doc_versions(filename, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_session_costs ON session_costs(session_id, created_at);
    """)


# --- Sessions ---

def create_session(title: str, mode: str = "chat") -> dict:
    conn = get_db()
    sid = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO sessions (id, title, mode, created_at, updated_at) VALUES (?,?,?,?,?)",
        (sid, title, mode, now, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    return dict(row)


def list_sessions(limit: int = 50) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT s.*, COUNT(m.id) as message_count "
        "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
        "GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def update_session_title(session_id: str, title: str) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
        (title, datetime.utcnow().isoformat(), session_id),
    )
    conn.commit()


def delete_session(session_id: str) -> None:
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()


# --- Messages ---

def add_message(
    session_id: str,
    role: str,
    content: str,
    context_sources: list[str] | None = None,
) -> dict:
    conn = get_db()
    mid = str(uuid.uuid4())[:12]
    now = datetime.utcnow().isoformat()
    sources_json = json.dumps(context_sources) if context_sources else None
    conn.execute(
        "INSERT INTO messages (id, session_id, role, content, context_sources, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (mid, session_id, role, content, sources_json, now),
    )
    conn.execute(
        "UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone()
    return dict(row)


def get_messages(session_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d["context_sources"]:
            d["context_sources"] = json.loads(d["context_sources"])
        result.append(d)
    return result


# --- Doc Versions ---

def save_doc_version(
    filename: str,
    content: str,
    change_summary: str = "",
    session_id: str | None = None,
) -> dict:
    conn = get_db()
    vid = str(uuid.uuid4())[:12]
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO doc_versions (id, session_id, filename, content, change_summary, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (vid, session_id, filename, content, change_summary, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM doc_versions WHERE id=?", (vid,)).fetchone()
    return dict(row)


def get_doc_versions(filename: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM doc_versions WHERE filename=? ORDER BY created_at DESC",
        (filename,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_doc_version(version_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM doc_versions WHERE id=?", (version_id,)).fetchone()
    return dict(row) if row else None


# --- Cost Tracking ---

# Pricing per 1M tokens (as of 2025)
PRICING = {
    "claude-opus-4-6":              {"input": 15.0,  "output": 75.0, "cache_read": 1.5,  "cache_create": 18.75},
    "claude-sonnet-4-5-20250929":   {"input": 3.0,   "output": 15.0, "cache_read": 0.3,  "cache_create": 3.75},
}

# Fallback for unknown models
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_create": 3.75}


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  cache_read_tokens: int = 0, cache_create_tokens: int = 0) -> float:
    """Estimate USD cost for an API call."""
    p = PRICING.get(model, _DEFAULT_PRICING)
    cost = (
        (input_tokens / 1_000_000) * p["input"]
        + (output_tokens / 1_000_000) * p["output"]
        + (cache_read_tokens / 1_000_000) * p["cache_read"]
        + (cache_create_tokens / 1_000_000) * p["cache_create"]
    )
    return round(cost, 6)


def add_cost(
    session_id: str,
    mode: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_create_tokens: int = 0,
) -> dict:
    """Record a cost entry for a session."""
    conn = get_db()
    cid = str(uuid.uuid4())[:12]
    now = datetime.utcnow().isoformat()
    usd = estimate_cost(model, input_tokens, output_tokens, cache_read_tokens, cache_create_tokens)
    conn.execute(
        "INSERT INTO session_costs (id, session_id, mode, model, input_tokens, output_tokens, "
        "cache_read_tokens, cache_create_tokens, estimated_usd, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (cid, session_id, mode, model, input_tokens, output_tokens,
         cache_read_tokens, cache_create_tokens, usd, now),
    )
    conn.commit()
    return {"id": cid, "estimated_usd": usd}


def get_session_cost(session_id: str) -> dict:
    """Get total cost for a session."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM session_costs WHERE session_id=? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()

    entries = [dict(r) for r in rows]
    total_usd = sum(e["estimated_usd"] for e in entries)
    total_input = sum(e["input_tokens"] for e in entries)
    total_output = sum(e["output_tokens"] for e in entries)

    return {
        "session_id": session_id,
        "total_usd": round(total_usd, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "call_count": len(entries),
        "entries": entries,
    }
