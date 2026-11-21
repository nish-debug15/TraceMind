"""
TraceMind Random Precedent Control Generation (Block 3 - E1)
-----------------------------------------------------------
Picks one random precedent (seed 42) uniformly from quality:high train incidents
excluding each query's real top-3 retrieved IDs. Computes true cosine similarity.
Generates completions under both baseline and gate-v2 prompt formats using
openai/gpt-oss-120b (temp 0, 20 RPM throttle, SQLite cache).

Outputs: eval/results/random_precedent_generations.json
"""

import os
import sys
import re
import json
import time
import random
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from clustering.cluster import embed
from rag.prompts import PROMPT_REGISTRY
from rag.generate import RateLimiter, LLM_MAX_RPM, GROQ_API_URL

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.json"
JUDGMENTS_FILE = PROJECT_ROOT / "eval" / "results" / "relevance_judgments.json"
CACHE_DB_PATH = PROJECT_ROOT / "eval" / "cache" / "e1_cache.sqlite"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "random_precedent_generations.json"

MODEL_ID = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM)


def clean_rc(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*[*_#]+\s*", "", cleaned)
    cleaned = re.sub(r"\s*[*_#]+\s*$", "", cleaned)
    return cleaned.strip()


def parse_rca(raw_text: str) -> Tuple[str, str, str]:
    rc_match = re.search(r"ROOT CAUSE:\s*(.*?)(?=REMEDIATION STEPS:|$)", raw_text, re.DOTALL | re.IGNORECASE)
    rem_match = re.search(r"REMEDIATION STEPS:\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)
    rc = rc_match.group(1).strip() if rc_match else raw_text.strip()
    rem = rem_match.group(1).strip() if rem_match else ""
    rc = clean_rc(rc)

    has_rc = bool(rc)
    has_rem = bool(rem)
    if has_rc and has_rem:
        status = "both_parsed"
    elif has_rc:
        status = "root_cause_only"
    elif has_rem:
        status = "remediation_only"
    else:
        status = "failed"
    return rc, rem, status


def init_cache_db() -> sqlite3.Connection:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS e1_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT,
                prompt_variant TEXT,
                prompt_hash TEXT,
                query_id TEXT,
                precedent_id TEXT,
                similarity REAL,
                raw_output TEXT,
                root_cause TEXT,
                remediation_steps TEXT,
                parse_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return conn


def get_cached(conn: sqlite3.Connection, key: str):
    cur = conn.cursor()
    cur.execute("SELECT raw_output, root_cause, remediation_steps, parse_status FROM e1_cache WHERE cache_key = ?", (key,))
    row = cur.fetchone()
    if row:
        return {
            "raw_output": row[0],
            "root_cause": row[1],
            "remediation_steps": row[2],
            "parse_status": row[3]
        }
    return None


def set_cached(conn: sqlite3.Connection, key: str, model: str, variant: str, p_hash: str, qid: str, pid: str, sim: float, raw: str, rc: str, rem: str, status: str):
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO e1_cache
               (cache_key, model, prompt_variant, prompt_hash, query_id, precedent_id, similarity, raw_output, root_cause, remediation_steps, parse_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (key, model, variant, p_hash, qid, pid, sim, raw, rc, rem, status)
        )


def call_llm(messages: List[Dict[str, str]], api_key: str, max_tokens: int = 800, max_retries: int = 30) -> str:
    delay = 2.0
    for attempt in range(max_retries):
        _rate_limiter.acquire()
        time.sleep(3.0)
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_ID,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                },
                timeout=60.0
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or msg.get("reasoning") or "").strip()
                if not content:
                    time.sleep(2.0)
                    continue
                return content
            elif resp.status_code == 429:
                err_text = ""
                try:
                    err_text = resp.json().get("error", {}).get("message", "")
                except Exception:
                    pass
                wait_sec = None
                m = re.search(r"Please try again in (?:(\d+)m)?([\d\.]+)s", err_text)
                if m:
                    mins = float(m.group(1)) if m.group(1) else 0.0
                    secs = float(m.group(2))
                    wait_sec = mins * 60.0 + secs + 2.0
                if wait_sec is None:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait_sec = float(retry_after) if retry_after else delay
                    except Exception:
                        wait_sec = delay
                print(f"    [Rate limit 429] Waiting {wait_sec:.1f}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait_sec)
                delay = min(delay * 2.0, 30.0)
            elif resp.status_code >= 500:
                print(f"    [Server Error {resp.status_code}] Waiting {delay:.1f}s...", flush=True)
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
            else:
                raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"    [Request Exception] {e}. Retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)
    raise RuntimeError("Exceeded maximum retries calling Groq API.")


def run_e1_generation():
    print("=" * 60)
    print("TraceMind E1: Random-Precedent Control Generation")
    print("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required.")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    print(f"Loaded {len(test_records)} test queries.")

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    train_high = [r for r in train_records if r.get("quality") == "high"]
    train_high_map = {r["incident_id"]: r for r in train_high}
    print(f"Loaded {len(train_high)} quality:high train precedents.")

    with open(JUDGMENTS_FILE, "r", encoding="utf-8") as f:
        judg_data = json.load(f)
    real_top3_map = {
        q["query_incident_id"]: {h["retrieved_incident_id"] for h in q["hits"]}
        for q in judg_data.get("judgments", [])
    }

    # Pre-embed queries and precedents to compute exact cosine similarity
    print("Pre-embedding queries and precedents for cosine similarity...")
    q_embeddings = {}
    for r in test_records:
        q_embeddings[r["incident_id"]] = embed(r.get("raw_log_excerpt", ""))

    p_embeddings = {}
    for r in train_high:
        text = f"{r.get('raw_log_excerpt', '')[:1000]} [SEP] {r.get('root_cause', '')[:1000]}"
        p_embeddings[r["incident_id"]] = embed(text)

    # Sample random precedent per query (seed 42)
    rng = random.Random(42)
    query_precedent_pairs = []
    for r in test_records:
        qid = r["incident_id"]
        real_top3 = real_top3_map.get(qid, set())
        candidates = [p for p in train_high if p["incident_id"] not in real_top3]
        candidates.sort(key=lambda x: x["incident_id"])
        chosen_p = rng.choice(candidates)
        pid = chosen_p["incident_id"]

        sim = float(np.dot(q_embeddings[qid], p_embeddings[pid]))
        query_precedent_pairs.append({
            "query": r,
            "precedent": chosen_p,
            "similarity": round(sim, 4)
        })

    print(f"Sampled 93 random query-precedent pairs (mean sim: {np.mean([p['similarity'] for p in query_precedent_pairs]):.4f}).")

    cache_conn = init_cache_db()
    num_new_calls = 0
    num_cached_calls = 0

    results = []
    start_time = time.time()

    formats = [
        ("baseline", "baseline", 800),
        ("gate_v2", "relevance_gate_v2", 300)
    ]

    for idx, pair in enumerate(query_precedent_pairs):
        q = pair["query"]
        p = pair["precedent"]
        sim = pair["similarity"]

        qid = q["incident_id"]
        pid = p["incident_id"]
        q_log = q.get("raw_log_excerpt", "")
        p_rc = p.get("root_cause", "")
        p_rem = p.get("remediation_steps", "")

        pair_result = {
            "query_incident_id": qid,
            "query_company": q.get("company", "unknown"),
            "random_precedent_id": pid,
            "random_precedent_company": p.get("company", "unknown"),
            "similarity_score": sim,
        }

        for arm_name, variant_key, max_tok in formats:
            prompt_def = PROMPT_REGISTRY[variant_key]
            sys_msg = prompt_def["system"]
            if "{similarity" in prompt_def["user_template"]:
                user_msg = prompt_def["user_template"].format(
                    ref_root_cause=p_rc,
                    ref_remediation_steps=p_rem,
                    log_text=q_log,
                    similarity=sim
                )
            else:
                user_msg = prompt_def["user_template"].format(
                    ref_root_cause=p_rc,
                    ref_remediation_steps=p_rem,
                    log_text=q_log
                )

            p_hash = hashlib.sha256(f"{sys_msg}\n{user_msg}".encode("utf-8")).hexdigest()
            cache_key = hashlib.sha256(f"{MODEL_ID}:{variant_key}:{qid}:{pid}".encode("utf-8")).hexdigest()

            cached = get_cached(cache_conn, cache_key)
            if cached is not None:
                raw_out = cached["raw_output"]
                rc = cached["root_cause"]
                rem = cached["remediation_steps"]
                status = cached["parse_status"]
                num_cached_calls += 1
            else:
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg}
                ]
                raw_out = call_llm(messages, api_key, max_tokens=max_tok)
                rc, rem, status = parse_rca(raw_out)
                set_cached(cache_conn, cache_key, MODEL_ID, variant_key, p_hash, qid, pid, sim, raw_out, rc, rem, status)
                num_new_calls += 1

            arm_res = {
                "root_cause": rc,
                "remediation_steps": rem,
                "raw_output": raw_out,
                "parse_status": status
            }

            if arm_name == "gate_v2":
                v_match = re.search(r"Precedent:\s*(relevant|not relevant)", rc, re.IGNORECASE)
                arm_res["precedent_verdict"] = v_match.group(1).lower() if v_match else "missing"

            pair_result[arm_name] = arm_res

        results.append(pair_result)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(query_precedent_pairs):
            print(f"  Processed {idx + 1}/{len(query_precedent_pairs)} queries ({num_new_calls} new calls, {num_cached_calls} cached)...", flush=True)

    output_data = {
        "metadata": {
            "num_queries": len(results),
            "seed": 42,
            "model": MODEL_ID,
            "temperature": 0.0,
            "num_new_llm_calls": num_new_calls,
            "num_cached_calls": num_cached_calls,
            "total_calls": num_new_calls + num_cached_calls,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "generations": results
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Artifact] Saved E1 random precedent generations to {OUTPUT_FILE}")
    print(f"Total queries: {len(results)}, New LLM calls: {num_new_calls}, Cached: {num_cached_calls}")
    print("=" * 60)


if __name__ == "__main__":
    run_e1_generation()
