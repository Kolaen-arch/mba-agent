"""
Database: SQLite persistence for chat sessions, messages, and document versions.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


DB_PATH = "./data/mba_agent.db"


def get_db(db_path: str | None = None) -> sqlite3.Connection:
    db_path = db_path or DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: str | None = None) -> None:
    conn = get_db(db_path)
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

        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_doc_versions_file ON doc_versions(filename, created_at DESC);
    """)
    conn.commit()
    conn.close()


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
    conn.close()
    return dict(row)


def list_sessions(limit: int = 50) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT s.*, COUNT(m.id) as message_count "
        "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
        "GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_session_title(session_id: str, title: str) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
        (title, datetime.utcnow().isoformat(), session_id),
    )
    conn.commit()
    conn.close()


def delete_session(session_id: str) -> None:
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()


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
    conn.close()
    return dict(row)


def get_messages(session_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    conn.close()
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
    conn.close()
    return dict(row)


def get_doc_versions(filename: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM doc_versions WHERE filename=? ORDER BY created_at DESC",
        (filename,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_doc_version(version_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM doc_versions WHERE id=?", (version_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
