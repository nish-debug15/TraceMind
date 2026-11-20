"""
TraceMind Judged Retrieval Relevance Study (Block 4)
---------------------------------------------------
Retrieves top-3 precedents from rag/chroma_db_train for all 93 test queries
using deployed 2-pass retrieval logic. Masks vendor names in query and precedents
using the clustering stoplist. Evaluates technical relevance with an LLM judge
using a fixed 3-point rubric, SQLite caching, and a 20 RPM throttle.
"""

import os
import sys
import re
import json
import time
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Tuple
import requests
from dotenv import load_dotenv

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

import chromadb
from clustering.cluster import embed

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
TRAIN_FILE = PROJECT_ROOT / "data" / "train.json"
CHROMA_TRAIN_DIR = PROJECT_ROOT / "rag" / "chroma_db_train"
COLLECTION_NAME = "postmortems_train"

RUBRIC_FILE = PROJECT_ROOT / "eval" / "rubrics" / "relevance_v1.md"
CACHE_DB_PATH = PROJECT_ROOT / "eval" / "cache" / "judge_cache.sqlite"
RESULTS_FILE = PROJECT_ROOT / "eval" / "results" / "relevance_judgments.json"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "qwen/qwen3.8-27b"
RPM_LIMIT = 20
SLEEP_BETWEEN_CALLS = 60.0 / RPM_LIMIT  # 3.0 seconds

# Build vendor stoplist
BASE_STOPLIST = {
    "github", "sentry", "cloudflare", "twilio", "discord", "shopify", "reddit",
    "atlassian", "datadog", "digitalocean", "circleci", "netlify", "openai",
    "dropbox", "squarespace", "launchdarkly", "google", "aws", "anthropic",
    "slack", "jira", "confluence", "bitbucket", "acuity", "copilot", "chatgpt",
    "travis", "travisci"
}

def load_stoplist() -> re.Pattern:
    stoplist = set(BASE_STOPLIST)
    if TRAIN_FILE.exists():
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            for r in json.load(f):
                c = r.get("company", "").strip().lower()
                if c:
                    stoplist.add(c)
    if TEST_FILE.exists():
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            for r in json.load(f):
                c = r.get("company", "").strip().lower()
                if c:
                    stoplist.add(c)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(c) for c in sorted(stoplist, key=len, reverse=True)) + r")\b",
        re.IGNORECASE
    )
    return pattern

COMPANY_PATTERN = load_stoplist()

def mask_vendor(text: str) -> str:
    if not text:
        return ""
    masked = COMPANY_PATTERN.sub("[VENDOR]", text)
    masked = re.sub(r"(\[VENDOR\]\s*)+", "[VENDOR] ", masked)
    return masked.strip()

def init_cache_db() -> sqlite3.Connection:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS judge_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT,
                score INTEGER,
                rationale TEXT,
                raw_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return conn

def get_cached_judgment(conn: sqlite3.Connection, key: str) -> Tuple[int, str] | None:
    cur = conn.cursor()
    cur.execute("SELECT score, rationale FROM judge_cache WHERE cache_key = ?", (key,))
    row = cur.fetchone()
    if row:
        return int(row[0]), str(row[1])
    return None

def set_cached_judgment(conn: sqlite3.Connection, key: str, model: str, score: int, rationale: str, raw_resp: str):
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO judge_cache (cache_key, model, score, rationale, raw_response) VALUES (?, ?, ?, ?, ?)",
            (key, model, score, rationale, raw_resp)
        )

def retrieve_top3_two_pass(collection, query_log: str) -> List[Dict[str, Any]]:
    query_emb = embed(query_log)
    
    # Pass 1: quality: high
    res_high = collection.query(
        query_embeddings=[query_emb],
        n_results=3,
        where={"quality": "high"}
    )
    
    hits = []
    if res_high.get("metadatas") and res_high["metadatas"][0]:
        for meta, dist in zip(res_high["metadatas"][0], res_high["distances"][0]):
            sim = 1.0 - (dist / 2.0)
            if sim >= 0.40:
                hits.append({
                    "metadata": meta,
                    "similarity": round(sim, 4),
                    "retrieval_pass": 1
                })
    
    # Pass 2: fallback across full corpus if < 3
    if len(hits) < 3:
        res_all = collection.query(
            query_embeddings=[query_emb],
            n_results=3
        )
        if res_all.get("metadatas") and res_all["metadatas"][0]:
            seen_ids = {h["metadata"]["incident_id"] for h in hits}
            for meta, dist in zip(res_all["metadatas"][0], res_all["distances"][0]):
                if meta["incident_id"] not in seen_ids:
                    sim = 1.0 - (dist / 2.0)
                    hits.append({
                        "metadata": meta,
                        "similarity": round(sim, 4),
                        "retrieval_pass": 2
                    })
                if len(hits) >= 3:
                    break
                    
    # Format hits with rank 1..3
    formatted_hits = []
    for rank, h in enumerate(hits[:3], start=1):
        formatted_hits.append({
            "rank": rank,
            "retrieved_incident_id": h["metadata"]["incident_id"],
            "company": h["metadata"].get("company", "unknown"),
            "similarity_score": h["similarity"],
            "retrieval_pass": h["retrieval_pass"],
            "raw_log_excerpt": h["metadata"].get("raw_log_excerpt", ""),
            "root_cause": h["metadata"].get("root_cause", "")
        })
    return formatted_hits

def call_groq_judge(prompt: str, api_key: str, max_retries: int = 4) -> Tuple[int, str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are an expert SRE evaluation judge. Evaluate technical incident relevance based on the rubric. Respond strictly with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    delay = 3.0
    for attempt in range(max_retries):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=45.0)
            if resp.status_code == 200:
                raw_json = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(raw_json)
                score = int(parsed["score"])
                if score not in (0, 1, 2):
                    raise ValueError(f"Invalid score: {score}")
                rationale = str(parsed.get("rationale", "")).strip()
                # Enforce <= 25 words
                words = rationale.split()
                if len(words) > 25:
                    rationale = " ".join(words[:25])
                return score, rationale, raw_json
            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else delay
                print(f"    [Rate limit 429] Sleeping {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                delay *= 2.0
            else:
                print(f"    [API Error {resp.status_code}]: {resp.text}")
                time.sleep(delay)
                delay *= 2.0
        except Exception as e:
            print(f"    [Exception attempt {attempt+1}]: {e}")
            time.sleep(delay)
            delay *= 2.0
            
    raise RuntimeError("Failed to obtain judgment after retries.")

def run_evaluation():
    print("=" * 60)
    print("TraceMind Judged Retrieval Relevance Study (Block 4)")
    print("=" * 60)
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required.")
        
    if not RUBRIC_FILE.exists():
        raise FileNotFoundError(f"Rubric file not found at {RUBRIC_FILE}")
    rubric_bytes = RUBRIC_FILE.read_bytes()
    rubric_sha256 = hashlib.sha256(rubric_bytes).hexdigest()
    rubric_text = rubric_bytes.decode("utf-8").strip()
    print(f"Rubric SHA-256: {rubric_sha256}")
    
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_incidents = json.load(f)
    print(f"Loaded {len(test_incidents)} test incidents.")
    
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_TRAIN_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"Connected to ChromaDB '{COLLECTION_NAME}' ({collection.count()} items).")
    
    cache_conn = init_cache_db()
    
    total_pairs = len(test_incidents) * 3
    num_new_llm_calls = 0
    num_cached_calls = 0
    
    judged_records = []
    
    print("\nRetrieving top-3 precedents and obtaining judgments...")
    start_time = time.time()
    
    for idx, inc in enumerate(test_incidents):
        q_id = inc["incident_id"]
        q_company = inc.get("company", "unknown")
        q_log = inc.get("raw_log_excerpt", "")
        q_rc = inc.get("root_cause", "")
        
        # Mask query
        q_log_masked = mask_vendor(q_log)
        q_rc_masked = mask_vendor(q_rc)
        
        # 2-pass top-3 retrieval
        hits = retrieve_top3_two_pass(collection, q_log)
        
        judged_hits = []
        for hit in hits:
            p_log_masked = mask_vendor(hit["raw_log_excerpt"])
            p_rc_masked = mask_vendor(hit["root_cause"])
            
            # Key for cache
            content_for_key = f"{rubric_sha256}\n---QUERY---\n{q_log_masked}\n{q_rc_masked}\n---PRECEDENT---\n{p_log_masked}\n{p_rc_masked}"
            cache_key = hashlib.sha256(content_for_key.encode("utf-8")).hexdigest()
            
            cached = get_cached_judgment(cache_conn, cache_key)
            if cached is not None:
                score, rationale = cached
                num_cached_calls += 1
            else:
                prompt = f"""{rubric_text}

---
Incoming Query Incident:
- Log Excerpt:
{q_log_masked}
- Ground-Truth Root Cause:
{q_rc_masked}

Retrieved Historical Precedent:
- Precedent Log Excerpt:
{p_log_masked}
- Precedent Root Cause:
{p_rc_masked}

Assign integer score (0, 1, or 2) and concise rationale (<= 25 words).
Output JSON: {{"score": int, "rationale": str}}"""

                score, rationale, raw_resp = call_groq_judge(prompt, api_key)
                set_cached_judgment(cache_conn, cache_key, MODEL_ID, score, rationale, raw_resp)
                num_new_llm_calls += 1
                time.sleep(SLEEP_BETWEEN_CALLS)
                
            judged_hits.append({
                "rank": hit["rank"],
                "retrieved_incident_id": hit["retrieved_incident_id"],
                "company": hit["company"],
                "similarity_score": hit["similarity_score"],
                "retrieval_pass": hit["retrieval_pass"],
                "score": score,
                "rationale": rationale
            })
            
        judged_records.append({
            "query_incident_id": q_id,
            "query_company": q_company,
            "hits": judged_hits
        })
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_incidents):
            print(f"  Processed {idx + 1}/{len(test_incidents)} queries ({num_new_llm_calls} new calls, {num_cached_calls} cached)...")

    # Compute distributions
    rank1_scores = [rec["hits"][0]["score"] for rec in judged_records]
    best_of_3_scores = [max(h["score"] for h in rec["hits"]) for rec in judged_records]
    
    n = len(test_incidents)
    rank1_dist = {
        "score_0": {"count": rank1_scores.count(0), "pct": round(rank1_scores.count(0) / n * 100, 2)},
        "score_1": {"count": rank1_scores.count(1), "pct": round(rank1_scores.count(1) / n * 100, 2)},
        "score_2": {"count": rank1_scores.count(2), "pct": round(rank1_scores.count(2) / n * 100, 2)},
    }
    best3_dist = {
        "score_0": {"count": best_of_3_scores.count(0), "pct": round(best_of_3_scores.count(0) / n * 100, 2)},
        "score_1": {"count": best_of_3_scores.count(1), "pct": round(best_of_3_scores.count(1) / n * 100, 2)},
        "score_2": {"count": best_of_3_scores.count(2), "pct": round(best_of_3_scores.count(2) / n * 100, 2)},
    }
    
    output_data = {
        "metadata": {
            "model": MODEL_ID,
            "judge_same_family_as_generator": False,
            "rubric_path": "eval/rubrics/relevance_v1.md",
            "rubric_sha256": rubric_sha256,
            "num_queries": n,
            "total_pairs": total_pairs,
            "num_new_llm_calls": num_new_llm_calls,
            "num_cached_calls": num_cached_calls,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "summary": {
            "rank_1_distribution": rank1_dist,
            "best_of_3_distribution": best3_dist
        },
        "judgments": judged_records
    }
    
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Artifact] Saved judgments to {RESULTS_FILE}")
    
    print("\n" + "=" * 60)
    print("JUDGED RETRIEVAL RELEVANCE RESULTS (N = 93)")
    print("=" * 60)
    print("Rank-1 Distribution:")
    for s in [0, 1, 2]:
        d = rank1_dist[f"score_{s}"]
        print(f"  Score {s}: {d['count']} ({d['pct']}%)")
    print("\nBest-of-Top-3 Distribution (Max score across 3 hits):")
    for s in [0, 1, 2]:
        d = best3_dist[f"score_{s}"]
        print(f"  Score {s}: {d['count']} ({d['pct']}%)")
    print(f"\nNew LLM Calls: {num_new_llm_calls}, Cached: {num_cached_calls}, Total pairs: {total_pairs}")
    print(f"Elapsed time: {time.time() - start_time:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
