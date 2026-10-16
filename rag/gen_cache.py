"""
rag/gen_cache.py
----------------
Generation cache for evaluation reuse.
Stores and retrieves LLM completions keyed by (model, prompt_variant, incident_id/query_hash).
Only used when use_cache=True is explicitly passed (default False).
"""

import os
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DB_PATH = CACHE_DIR / "gen_cache.sqlite3"


def _get_connection() -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH), timeout=10.0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gen_cache (
            cache_key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            prompt_variant TEXT NOT NULL,
            query_id TEXT NOT NULL,
            root_cause TEXT NOT NULL,
            remediation_steps TEXT NOT NULL,
            raw_output TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def make_cache_key(model: str, prompt_variant: str, query_text_or_id: str) -> str:
    """Generate a deterministic cache key from model, prompt_variant, and query."""
    normalized_id = query_text_or_id.strip()
    if len(normalized_id) > 64:
        query_hash = hashlib.sha256(normalized_id.encode("utf-8")).hexdigest()[:16]
    else:
        query_hash = normalized_id
    return f"{model}:{prompt_variant}:{query_hash}"


def get_cached_generation(
    model: str,
    prompt_variant: str,
    query_text_or_id: str
) -> Optional[Dict[str, Any]]:
    """Retrieve cached generation if present."""
    cache_key = make_cache_key(model, prompt_variant, query_text_or_id)
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT root_cause, remediation_steps, raw_output FROM gen_cache WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "root_cause": row[0],
                "remediation_steps": row[1],
                "raw_output": row[2] or "",
                "cache_hit": True
            }
    except Exception as e:
        print(f"[gen_cache] Cache read error: {e}")
    return None


def set_cached_generation(
    model: str,
    prompt_variant: str,
    query_text_or_id: str,
    root_cause: str,
    remediation_steps: str,
    raw_output: str = ""
) -> None:
    """Save a generation into cache."""
    cache_key = make_cache_key(model, prompt_variant, query_text_or_id)
    try:
        conn = _get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO gen_cache
            (cache_key, model, prompt_variant, query_id, root_cause, remediation_steps, raw_output)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (cache_key, model, prompt_variant, query_text_or_id[:64], root_cause, remediation_steps, raw_output))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[gen_cache] Cache write error: {e}")
