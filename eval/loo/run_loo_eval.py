"""
eval/loo/run_loo_eval.py
------------------------
Executes Experiment 2 (E2: Leave-One-Out Same-Company Regime):
1. Loads 93 sampled incidents from eval/loo/sample.json.
2. Loads retrieval results from eval/loo/retrieval.json (conditions a, b, c).
3. Generates completions for 6 cells across all 93 queries:
   - zero-shot baseline
   - zero-shot gate_v2
   - condition (a) RAG baseline (top-1)
   - condition (a) RAG gate_v2 (top-1)
   - condition (b) RAG baseline (top-1)
   - condition (b) RAG gate_v2 (top-1)
   SQLite cache: eval/cache/e2_cache.sqlite
4. Judges technical relevance of top-1 precedents for condition (a), (b), and (c) against queries:
   Model: qwen/qwen3.8-27b, Rubric: eval/rubrics/relevance_v1.md, vendor masked.
   SQLite cache: eval/cache/judge_cache.sqlite
5. Scores full metric suite:
   - ROUGE-1/2/L F1, BERTScore F1, Cosine, Word counts, Length-matched ROUGE-L (median GT length)
   - Remediation on both-parsed subsets
   - Paired deltas with company-clustered 95% CIs and Bonferroni 99% CIs (family m=5, alpha=0.01):
     * RAG(a) - zero-shot
     * RAG(b) - zero-shot
     * RAG(a) - RAG(b) (same-company premium)
6. Relevance@1 distributions for conditions (a), (b), (c).
7. Stratifies RAG(a) - zero-shot delta by judged score (where n_companies >= 5, else low_n: True).
8. Automated verdict per eval/PREREG.md.

Outputs: eval/results/loo_regime_results.json
"""

import os
import sys
import re
import json
import time
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple
import requests
import numpy as np
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from bert_score import score as calc_bertscore

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from clustering.cluster import embed
from rag.prompts import PROMPT_REGISTRY
from rag.generate import RateLimiter, LLM_MAX_RPM, GROQ_API_URL
from eval.analyze_e1 import FastClusterBootstrap, clean_rc

SAMPLE_FILE = PROJECT_ROOT / "eval" / "loo" / "sample.json"
RETRIEVAL_FILE = PROJECT_ROOT / "eval" / "loo" / "retrieval.json"
RUBRIC_FILE = PROJECT_ROOT / "eval" / "rubrics" / "relevance_v1.md"
E2_CACHE_PATH = PROJECT_ROOT / "eval" / "cache" / "e2_cache.sqlite"
JUDGE_CACHE_PATH = PROJECT_ROOT / "eval" / "cache" / "judge_cache.sqlite"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "loo_regime_results.json"

GEN_MODEL_ID = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL_ID = "qwen/qwen3.8-27b"

_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM)

COMPANY_PATTERN = re.compile(
    r"\b(reddit|slack|github|gitlab|circleci|travisci|atlassian|twilio|pagerduty|datadog|newrelic|dynatrace|sentry|cloudflare|fastly|akamai|aws|amazon web services|gcp|google cloud|azure|microsoft azure|mongodb|redis|elastic|elasticsearch|docker|kubernetes)\b",
    re.IGNORECASE,
)


def mask_vendor(text: str) -> str:
    masked = COMPANY_PATTERN.sub("[VENDOR]", text)
    masked = re.sub(r"(\[VENDOR\]\s*)+", "[VENDOR] ", masked)
    return masked.strip()


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


def init_e2_cache_db() -> sqlite3.Connection:
    E2_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(E2_CACHE_PATH))
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS e2_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT,
                cell_name TEXT,
                prompt_hash TEXT,
                query_id TEXT,
                precedent_id TEXT,
                raw_output TEXT,
                root_cause TEXT,
                remediation_steps TEXT,
                parse_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return conn


def get_e2_cached(conn: sqlite3.Connection, key: str) -> Dict[str, Any] | None:
    cur = conn.cursor()
    cur.execute("SELECT raw_output, root_cause, remediation_steps, parse_status FROM e2_cache WHERE cache_key = ?", (key,))
    row = cur.fetchone()
    if row:
        return {
            "raw_output": row[0],
            "root_cause": row[1],
            "remediation_steps": row[2],
            "parse_status": row[3],
        }
    return None


def set_e2_cached(conn: sqlite3.Connection, key: str, model: str, cell_name: str, p_hash: str, qid: str, pid: str, raw: str, rc: str, rem: str, status: str):
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO e2_cache
               (cache_key, model, cell_name, prompt_hash, query_id, precedent_id, raw_output, root_cause, remediation_steps, parse_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (key, model, cell_name, p_hash, qid, pid, raw, rc, rem, status)
        )


def init_judge_cache_db() -> sqlite3.Connection:
    JUDGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(JUDGE_CACHE_PATH))
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


def get_cached_judge(conn: sqlite3.Connection, key: str) -> Tuple[int, str] | None:
    cur = conn.cursor()
    cur.execute("SELECT score, rationale FROM judge_cache WHERE cache_key = ?", (key,))
    row = cur.fetchone()
    if row:
        return int(row[0]), str(row[1])
    return None


def set_cached_judge(conn: sqlite3.Connection, key: str, model: str, score: int, rationale: str, raw_resp: str):
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO judge_cache (cache_key, model, score, rationale, raw_response) VALUES (?, ?, ?, ?, ?)",
            (key, model, score, rationale, raw_resp)
        )


def call_llm_gen(messages: List[Dict[str, str]], api_key: str, max_tokens: int = 800, max_retries: int = 30) -> str:
    delay = 2.0
    for attempt in range(max_retries):
        _rate_limiter.acquire()
        time.sleep(3.0)
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": GEN_MODEL_ID, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens},
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
            delay = min(delay * 2.0, 30.0)
    raise RuntimeError("Exceeded maximum retries calling Groq API.")


def call_llm_judge(prompt: str, api_key: str, max_retries: int = 30) -> Tuple[int, str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": JUDGE_MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are an expert SRE evaluation judge. Evaluate technical incident relevance based on the rubric. Respond strictly with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    delay = 2.0
    for attempt in range(max_retries):
        _rate_limiter.acquire()
        time.sleep(3.0)
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60.0)
            if resp.status_code == 200:
                raw_json = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(raw_json)
                score = int(parsed["score"])
                if score not in (0, 1, 2):
                    raise ValueError(f"Invalid score: {score}")
                rationale = str(parsed.get("rationale", "")).strip()
                words = rationale.split()
                if len(words) > 25:
                    rationale = " ".join(words[:25])
                return score, rationale, raw_json
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
                print(f"    [Judge 429] Waiting {wait_sec:.1f}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait_sec)
                delay = min(delay * 2.0, 30.0)
            elif resp.status_code >= 500:
                print(f"    [Judge Server Error {resp.status_code}] Waiting {delay:.1f}s...", flush=True)
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
            else:
                raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"    [Judge Request Exception] {e}. Retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)
    raise RuntimeError("Exceeded maximum retries calling Groq Judge API.")


def compute_arm_metrics(
    test_ids: List[str],
    gt_rc_map: Dict[str, str],
    pred_rc_map: Dict[str, str],
    gt_embeddings: np.ndarray,
    scorer: rouge_scorer.RougeScorer,
    median_gt_limit: int,
    bootstrapper: FastClusterBootstrap,
) -> Dict[str, Any]:
    preds = [clean_rc(pred_rc_map[i]) for i in test_ids]
    refs = [clean_rc(gt_rc_map[i]) for i in test_ids]

    r1_list, r2_list, rl_list = [], [], []
    for ref, pred in zip(refs, preds):
        s = scorer.score(ref, pred)
        r1_list.append(s["rouge1"].fmeasure)
        r2_list.append(s["rouge2"].fmeasure)
        rl_list.append(s["rougeL"].fmeasure)

    lm_preds = [" ".join(p.split()[:median_gt_limit]) for p in preds]
    lm_rl_list = [scorer.score(ref, lmp)["rougeL"].fmeasure for ref, lmp in zip(refs, lm_preds)]

    safe_preds = [p if p.strip() else "none" for p in preds]
    safe_refs = [r if r.strip() else "none" for r in refs]
    _, _, bs = calc_bertscore(safe_preds, safe_refs, model_type="bert-base-uncased", verbose=False)
    bs_list = bs.numpy().tolist()

    pred_embeddings = embed(safe_preds)
    pred_norm = pred_embeddings / np.maximum(np.linalg.norm(pred_embeddings, axis=1, keepdims=True), 1e-12)
    gt_norm = gt_embeddings / np.maximum(np.linalg.norm(gt_embeddings, axis=1, keepdims=True), 1e-12)
    cos_list = np.sum(pred_norm * gt_norm, axis=1).tolist()

    word_counts = [len(p.split()) for p in preds]

    return {
        "rouge1_f1": bootstrapper.compute_multi_ci(r1_list),
        "rouge2_f1": bootstrapper.compute_multi_ci(r2_list),
        "rougeL_f1": bootstrapper.compute_multi_ci(rl_list),
        "length_matched_rougeL": bootstrapper.compute_multi_ci(lm_rl_list),
        "bert_score_f1": bootstrapper.compute_multi_ci(bs_list),
        "cosine": bootstrapper.compute_multi_ci(cos_list),
        "word_count": {
            "mean": round(float(np.mean(word_counts)), 2),
            "median": round(float(np.median(word_counts)), 2),
        },
        "_raw": {
            "rouge1_f1": r1_list,
            "rouge2_f1": r2_list,
            "rougeL_f1": rl_list,
            "length_matched_rougeL": lm_rl_list,
            "bert_score_f1": bs_list,
            "cosine": cos_list,
        }
    }


def compute_paired_deltas(
    arm_a_raw: Dict[str, List[float]],
    arm_b_raw: Dict[str, List[float]],
    bootstrapper: FastClusterBootstrap,
) -> Dict[str, Any]:
    deltas = {}
    metrics = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "length_matched_rougeL", "bert_score_f1", "cosine"]
    for m in metrics:
        diff = [a - b for a, b in zip(arm_a_raw[m], arm_b_raw[m])]
        deltas[m] = bootstrapper.compute_multi_ci(diff)
    return deltas


def evaluate_remediation_pairwise(
    test_ids: List[str],
    id_to_company: Dict[str, str],
    gt_rem_map: Dict[str, str],
    preds_a: Dict[str, str],
    preds_b: Dict[str, str],
    scorer: rouge_scorer.RougeScorer,
) -> Dict[str, Any]:
    both_ids = [
        i for i in test_ids
        if preds_a.get(i, "").strip() and preds_b.get(i, "").strip() and len(gt_rem_map.get(i, "").split()) >= 5
    ]
    if not both_ids:
        return {"n_both_parsed": 0, "status": "no incidents parsed in both arms"}

    sub_bootstrapper = FastClusterBootstrap(both_ids, id_to_company, n_resamples=10000, seed=42)

    a_texts = [preds_a[i].strip() for i in both_ids]
    b_texts = [preds_b[i].strip() for i in both_ids]
    gts = [gt_rem_map[i].strip() for i in both_ids]

    rl_a = [scorer.score(g, p)["rougeL"].fmeasure for g, p in zip(gts, a_texts)]
    rl_b = [scorer.score(g, p)["rougeL"].fmeasure for g, p in zip(gts, b_texts)]
    rl_diff = [a - b for a, b in zip(rl_a, rl_b)]

    _, _, bs_a_t = calc_bertscore(a_texts, gts, model_type="bert-base-uncased", verbose=False)
    _, _, bs_b_t = calc_bertscore(b_texts, gts, model_type="bert-base-uncased", verbose=False)
    bs_a = bs_a_t.numpy().tolist()
    bs_b = bs_b_t.numpy().tolist()
    bs_diff = [a - b for a, b in zip(bs_a, bs_b)]

    return {
        "n_both_parsed": len(both_ids),
        "arm_a_rougeL": sub_bootstrapper.compute_multi_ci(rl_a),
        "arm_b_rougeL": sub_bootstrapper.compute_multi_ci(rl_b),
        "arm_a_bert_score": sub_bootstrapper.compute_multi_ci(bs_a),
        "arm_b_bert_score": sub_bootstrapper.compute_multi_ci(bs_b),
        "delta_rougeL": sub_bootstrapper.compute_multi_ci(rl_diff),
        "delta_bert_score": sub_bootstrapper.compute_multi_ci(bs_diff),
    }


def run_e2_evaluation():
    print("=" * 60)
    print("TraceMind E2: Leave-One-Out (LOO) Generation, Judging & Scoring")
    print("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from environment.")

    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        sample_data = json.load(f)
    sample_records = sample_data["sample"]
    test_ids = [r["incident_id"] for r in sample_records]
    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in sample_records}
    gt_rc_map = {r["incident_id"]: r["root_cause"] for r in sample_records}
    gt_rem_map = {r["incident_id"]: r.get("remediation_steps", "") for r in sample_records}

    gt_word_counts = [len(gt_rc_map[i].split()) for i in test_ids]
    median_gt_limit = int(round(float(np.median(gt_word_counts))))
    print(f"Loaded {len(test_ids)} LOO incidents across {len(set(id_to_company.values()))} companies.")
    print(f"Ground truth root cause median length: {median_gt_limit} words.")

    with open(RETRIEVAL_FILE, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)
    retrieval_by_query = {item["query_incident_id"]: item for item in retrieval_data["retrieval"]}

    rubric_bytes = RUBRIC_FILE.read_bytes()
    rubric_sha256 = hashlib.sha256(rubric_bytes).hexdigest()
    rubric_text = rubric_bytes.decode("utf-8").strip()

    e2_cache_conn = init_e2_cache_db()
    judge_cache_conn = init_judge_cache_db()

    new_gen_calls = 0
    cached_gen_calls = 0
    new_judge_calls = 0
    cached_judge_calls = 0

    # 1. Run LLM Relevance Judge on Top-1 for Conditions (a), (b), (c)
    print("\n--- Phase 1: LLM Relevance Judging on Top-1 Precedents ---")
    judgments_by_condition: Dict[str, List[Dict[str, Any]]] = {
        "condition_a": [],
        "condition_b": [],
        "condition_c": [],
    }

    cond_keys = [
        ("condition_a", "condition_a_same_company_filtered"),
        ("condition_b", "condition_b_cross_company_filtered"),
        ("condition_c", "condition_c_same_company_unfiltered"),
    ]

    for q_idx, inc in enumerate(sample_records):
        qid = inc["incident_id"]
        q_log = inc.get("raw_log_excerpt", "")
        q_rc = inc.get("root_cause", "")
        q_log_masked = mask_vendor(q_log)
        q_rc_masked = mask_vendor(q_rc)

        ret_item = retrieval_by_query[qid]

        for short_name, ret_key in cond_keys:
            top1_hit = ret_item[ret_key]["hits"][0]
            p_log_masked = mask_vendor(top1_hit.get("raw_log_excerpt", ""))
            p_rc_masked = mask_vendor(top1_hit.get("root_cause", ""))

            content_for_key = f"{rubric_sha256}\n---QUERY---\n{q_log_masked}\n{q_rc_masked}\n---PRECEDENT---\n{p_log_masked}\n{p_rc_masked}"
            cache_key = hashlib.sha256(content_for_key.encode("utf-8")).hexdigest()

            cached = get_cached_judge(judge_cache_conn, cache_key)
            if cached is not None:
                score, rationale = cached
                cached_judge_calls += 1
            else:
                prompt = (
                    f"{rubric_text}\n\n---\nIncoming Query Incident:\n- Log Excerpt:\n{q_log_masked}\n"
                    f"- Ground-Truth Root Cause:\n{q_rc_masked}\n\nRetrieved Historical Precedent:\n"
                    f"- Precedent Log Excerpt:\n{p_log_masked}\n- Precedent Root Cause:\n{p_rc_masked}\n\n"
                    f"Rate technical relevance on the 3-point scale (0, 1, or 2). Respond strictly in JSON:\n"
                    f'{{\n  "score": <0, 1, or 2>,\n  "rationale": "<1-2 sentence explanation citing specific technical mechanisms>"\n}}'
                )
                score, rationale, raw_resp = call_llm_judge(prompt, api_key)
                set_cached_judge(judge_cache_conn, cache_key, JUDGE_MODEL_ID, score, rationale, raw_resp)
                new_judge_calls += 1

            judgments_by_condition[short_name].append({
                "query_incident_id": qid,
                "precedent_incident_id": top1_hit["incident_id"],
                "similarity_score": top1_hit["similarity_score"],
                "score": score,
                "rationale": rationale,
            })

        if (q_idx + 1) % 10 == 0 or (q_idx + 1) == len(sample_records):
            print(f"  Judged {q_idx + 1}/{len(sample_records)} queries (new: {new_judge_calls}, cached: {cached_judge_calls})...", flush=True)

    # 2. Run Generations Across 6 Cells
    print("\n--- Phase 2: Generating LOO Completions Across 6 Cells ---")
    cells_generations: Dict[str, Dict[str, Dict[str, str]]] = {
        "baseline_zeroshot": {},
        "gate_v2_zeroshot": {},
        "baseline_rag_a": {},
        "gate_v2_rag_a": {},
        "baseline_rag_b": {},
        "gate_v2_rag_b": {},
    }

    cell_configs = [
        ("baseline_zeroshot", "baseline", "none", None),
        ("gate_v2_zeroshot", "zero_shot_gate_format", "none", None),
        ("baseline_rag_a", "baseline", "a", "condition_a_same_company_filtered"),
        ("gate_v2_rag_a", "relevance_gate_v2", "a", "condition_a_same_company_filtered"),
        ("baseline_rag_b", "baseline", "b", "condition_b_cross_company_filtered"),
        ("gate_v2_rag_b", "relevance_gate_v2", "b", "condition_b_cross_company_filtered"),
    ]

    for q_idx, inc in enumerate(sample_records):
        qid = inc["incident_id"]
        q_log = inc.get("raw_log_excerpt", "")
        ret_item = retrieval_by_query[qid]

        for cell_name, prompt_variant, cond_id, ret_key in cell_configs:
            prompt_def = PROMPT_REGISTRY[prompt_variant]
            sys_msg = prompt_def["system"]

            if ret_key is None:
                # Zero-shot
                pid = "none"
                if cell_name == "baseline_zeroshot":
                    user_msg = f"New Incident Log Excerpt:\n{q_log}\n\nProvide your response in the following format:\nROOT CAUSE:\n(your analysis)\n\nREMEDIATION STEPS:\n(your steps)"
                else:
                    user_msg = prompt_def["user_template"].format(log_text=q_log)
                max_tok = 800 if prompt_variant == "baseline" else 300
            else:
                top1_hit = ret_item[ret_key]["hits"][0]
                pid = top1_hit["incident_id"]
                p_rc = top1_hit.get("root_cause", "")
                p_rem = top1_hit.get("remediation_steps", "")
                sim = top1_hit["similarity_score"]

                if "{similarity" in prompt_def["user_template"]:
                    user_msg = prompt_def["user_template"].format(
                        ref_root_cause=p_rc, ref_remediation_steps=p_rem, log_text=q_log, similarity=sim
                    )
                else:
                    user_msg = prompt_def["user_template"].format(
                        ref_root_cause=p_rc, ref_remediation_steps=p_rem, log_text=q_log
                    )
                max_tok = 800 if prompt_variant == "baseline" else 300

            p_hash = hashlib.sha256(f"{sys_msg}\n{user_msg}".encode("utf-8")).hexdigest()
            cache_key = hashlib.sha256(f"{GEN_MODEL_ID}:{p_hash}:{qid}:{pid}".encode("utf-8")).hexdigest()

            cached = get_e2_cached(e2_cache_conn, cache_key)
            if cached is not None:
                raw_out = cached["raw_output"]
                rc = cached["root_cause"]
                rem = cached["remediation_steps"]
                status = cached["parse_status"]
                cached_gen_calls += 1
            else:
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg}
                ]
                raw_out = call_llm_gen(messages, api_key, max_tokens=max_tok)
                rc, rem, status = parse_rca(raw_out)
                set_e2_cached(e2_cache_conn, cache_key, GEN_MODEL_ID, cell_name, p_hash, qid, pid, raw_out, rc, rem, status)
                new_gen_calls += 1

            cells_generations[cell_name][qid] = {
                "root_cause": rc,
                "remediation_steps": rem,
                "raw_output": raw_out,
                "parse_status": status,
            }

        if (q_idx + 1) % 10 == 0 or (q_idx + 1) == len(sample_records):
            print(f"  Generated {q_idx + 1}/{len(sample_records)} queries (new calls: {new_gen_calls}, cached: {cached_gen_calls})...", flush=True)

    # 3. Compute Relevance Distributions
    print("\n--- Phase 3: Computing Relevance Distributions ---")
    relevance_distributions = {}
    for short_name in ["condition_a", "condition_b", "condition_c"]:
        scores = [j["score"] for j in judgments_by_condition[short_name]]
        n_total = len(scores)
        c0 = scores.count(0)
        c1 = scores.count(1)
        c2 = scores.count(2)
        mean_s = float(np.mean(scores))
        relevance_distributions[short_name] = {
            "num_judged": n_total,
            "score_0_unrelated": {"count": c0, "pct": round(float(c0 / n_total * 100), 2)},
            "score_1_partially_relevant": {"count": c1, "pct": round(float(c1 / n_total * 100), 2)},
            "score_2_highly_relevant": {"count": c2, "pct": round(float(c2 / n_total * 100), 2)},
            "mean_relevance_score": round(mean_s, 4),
            "judged_relevant_pct": round(float((c1 + c2) / n_total * 100), 2),
        }

    # 4. Score Full Metric Suite
    print("\n--- Phase 4: Scoring Full Metric Suite & Deltas ---")
    bootstrapper = FastClusterBootstrap(test_ids, id_to_company, n_resamples=10000, seed=42)
    clean_gts = [clean_rc(gt_rc_map[i]) for i in test_ids]
    gt_embeddings = embed(clean_gts)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    format_results = {}
    for fmt in ["baseline", "gate_v2"]:
        print(f"\nEvaluating format: {fmt}...")
        zs_name = f"{fmt}_zeroshot"
        rag_a_name = f"{fmt}_rag_a"
        rag_b_name = f"{fmt}_rag_b"

        zs_rc = {i: cells_generations[zs_name][i]["root_cause"] for i in test_ids}
        zs_rem = {i: cells_generations[zs_name][i]["remediation_steps"] for i in test_ids}

        rag_a_rc = {i: cells_generations[rag_a_name][i]["root_cause"] for i in test_ids}
        rag_a_rem = {i: cells_generations[rag_a_name][i]["remediation_steps"] for i in test_ids}

        rag_b_rc = {i: cells_generations[rag_b_name][i]["root_cause"] for i in test_ids}
        rag_b_rem = {i: cells_generations[rag_b_name][i]["remediation_steps"] for i in test_ids}

        print(f"  Scoring arm: zero_shot ({fmt})...")
        arm_zs = compute_arm_metrics(test_ids, gt_rc_map, zs_rc, gt_embeddings, scorer, median_gt_limit, bootstrapper)

        print(f"  Scoring arm: rag_a ({fmt})...")
        arm_rag_a = compute_arm_metrics(test_ids, gt_rc_map, rag_a_rc, gt_embeddings, scorer, median_gt_limit, bootstrapper)

        print(f"  Scoring arm: rag_b ({fmt})...")
        arm_rag_b = compute_arm_metrics(test_ids, gt_rc_map, rag_b_rc, gt_embeddings, scorer, median_gt_limit, bootstrapper)

        print(f"  Computing paired deltas ({fmt})...")
        delta_rag_a_vs_zs = compute_paired_deltas(arm_rag_a["_raw"], arm_zs["_raw"], bootstrapper)
        delta_rag_b_vs_zs = compute_paired_deltas(arm_rag_b["_raw"], arm_zs["_raw"], bootstrapper)
        delta_rag_a_vs_rag_b = compute_paired_deltas(arm_rag_a["_raw"], arm_rag_b["_raw"], bootstrapper)

        print(f"  Scoring remediation on both-parsed subsets ({fmt})...")
        rem_a_vs_zs = evaluate_remediation_pairwise(test_ids, id_to_company, gt_rem_map, rag_a_rem, zs_rem, scorer)
        rem_b_vs_zs = evaluate_remediation_pairwise(test_ids, id_to_company, gt_rem_map, rag_b_rem, zs_rem, scorer)
        rem_a_vs_b = evaluate_remediation_pairwise(test_ids, id_to_company, gt_rem_map, rag_a_rem, rag_b_rem, scorer)

        # Clean _raw
        del arm_zs["_raw"]
        del arm_rag_a["_raw"]
        del arm_rag_b["_raw"]

        format_results[fmt] = {
            "arms": {
                "zero_shot": arm_zs,
                "rag_condition_a_same_company": arm_rag_a,
                "rag_condition_b_cross_company": arm_rag_b,
            },
            "deltas": {
                "rag_a_minus_zeroshot": delta_rag_a_vs_zs,
                "rag_b_minus_zeroshot": delta_rag_b_vs_zs,
                "rag_a_minus_rag_b_same_company_premium": delta_rag_a_vs_rag_b,
            },
            "remediation_both_parsed": {
                "rag_a_minus_zeroshot": rem_a_vs_zs,
                "rag_b_minus_zeroshot": rem_b_vs_zs,
                "rag_a_minus_rag_b": rem_a_vs_b,
            }
        }

    # 5. Stratify RAG(a) - zero-shot delta by Judged Relevance Score (Gate-v2)
    print("\n--- Phase 5: Stratifying RAG(a) - Zero-Shot Delta by Judged Relevance ---")
    judgments_a = {j["query_incident_id"]: j["score"] for j in judgments_by_condition["condition_a"]}
    stratified_deltas = {}

    g2_a_rc = [clean_rc(cells_generations["gate_v2_rag_a"][i]["root_cause"]) for i in test_ids]
    g2_zs_rc = [clean_rc(cells_generations["gate_v2_zeroshot"][i]["root_cause"]) for i in test_ids]
    g2_diff_rl = [
        scorer.score(clean_rc(gt_rc_map[i]), ra)["rougeL"].fmeasure - scorer.score(clean_rc(gt_rc_map[i]), zs)["rougeL"].fmeasure
        for i, ra, zs in zip(test_ids, g2_a_rc, g2_zs_rc)
    ]

    for score_val in [0, 1, 2]:
        stratum_ids = [i for i in test_ids if judgments_a[i] == score_val]
        stratum_diffs = [d for i, d in zip(test_ids, g2_diff_rl) if judgments_a[i] == score_val]
        stratum_companies = set(id_to_company[i] for i in stratum_ids)
        n_comps = len(stratum_companies)
        n_items = len(stratum_ids)

        if n_comps < 5 or n_items < 5:
            stratified_deltas[f"relevance_score_{score_val}"] = {
                "score_level": score_val,
                "num_incidents": n_items,
                "num_companies": n_comps,
                "mean_delta_rougeL": round(float(np.mean(stratum_diffs)), 4) if stratum_diffs else 0.0,
                "low_n": True,
                "status": f"insufficient company count (n_companies={n_comps} < 5) for reliable clustered bootstrap"
            }
        else:
            strat_bootstrapper = FastClusterBootstrap(stratum_ids, id_to_company, n_resamples=10000, seed=42)
            ci_res = strat_bootstrapper.compute_multi_ci(stratum_diffs)
            stratified_deltas[f"relevance_score_{score_val}"] = {
                "score_level": score_val,
                "num_incidents": n_items,
                "num_companies": n_comps,
                "mean_delta_rougeL": ci_res["mean"],
                "unadjusted_ci_95": ci_res["unadjusted_ci_95"],
                "adjusted_ci": ci_res["adjusted_ci"],
                "status": ci_res["status"],
                "low_n": False,
            }

    # 6. Mechanical Verdict per PREREG.md
    # Rule for gate_v2:
    # 'RAG helps when same-company history exists' IF:
    # 1. RAG(a) - zero-shot Bonferroni CI excludes zero
    # 2. Remains excluding zero under near-duplicate filter (Condition a is filtered)
    # 3. Remains excluding zero under length-matching
    # ELSE: 'no detected effect'
    g2_delta_a = format_results["gate_v2"]["deltas"]["rag_a_minus_zeroshot"]
    rag_a_zs_adj_excludes = g2_delta_a["rougeL_f1"]["excludes_zero_after_adjustment"]
    rag_a_zs_lm_adj_excludes = g2_delta_a["length_matched_rougeL"]["excludes_zero_after_adjustment"]

    if rag_a_zs_adj_excludes and rag_a_zs_lm_adj_excludes:
        verdict = "RAG helps when same-company history exists"
    else:
        verdict = "no detected effect"

    print(f"\n[E2 Verdict]: {verdict}")
    print(f"  rag_a_minus_zeroshot adj excludes zero: {rag_a_zs_adj_excludes}")
    print(f"  rag_a_minus_zeroshot length-matched adj excludes zero: {rag_a_zs_lm_adj_excludes}")

    summary_text = (
        f"Experiment 2 (Leave-One-Out Same-Company Regime, N=93 across 17 companies): "
        f"Relevance@1: Condition (a) same-company filtered judged {relevance_distributions['condition_a']['judged_relevant_pct']}% relevant (mean score {relevance_distributions['condition_a']['mean_relevance_score']:.2f}), "
        f"Condition (b) cross-company judged {relevance_distributions['condition_b']['judged_relevant_pct']}% relevant (mean score {relevance_distributions['condition_b']['mean_relevance_score']:.2f}), "
        f"Condition (c) unfiltered judged {relevance_distributions['condition_c']['judged_relevant_pct']}% relevant (mean score {relevance_distributions['condition_c']['mean_relevance_score']:.2f}). "
        f"Gate-v2 format ROUGE-L F1: Zero-shot={format_results['gate_v2']['arms']['zero_shot']['rougeL_f1']['mean']:.4f}, "
        f"RAG(a)={format_results['gate_v2']['arms']['rag_condition_a_same_company']['rougeL_f1']['mean']:.4f}, "
        f"RAG(b)={format_results['gate_v2']['arms']['rag_condition_b_cross_company']['rougeL_f1']['mean']:.4f}. "
        f"Delta RAG(a)-ZS={format_results['gate_v2']['deltas']['rag_a_minus_zeroshot']['rougeL_f1']['mean']:+.4f} "
        f"(99% Bonferroni CI {format_results['gate_v2']['deltas']['rag_a_minus_zeroshot']['rougeL_f1']['adjusted_ci']}, status: {format_results['gate_v2']['deltas']['rag_a_minus_zeroshot']['rougeL_f1']['status']}). "
        f"Delta RAG(b)-ZS={format_results['gate_v2']['deltas']['rag_b_minus_zeroshot']['rougeL_f1']['mean']:+.4f} "
        f"(99% Bonferroni CI {format_results['gate_v2']['deltas']['rag_b_minus_zeroshot']['rougeL_f1']['adjusted_ci']}, status: {format_results['gate_v2']['deltas']['rag_b_minus_zeroshot']['rougeL_f1']['status']}). "
        f"Delta RAG(a)-RAG(b) (same-company premium)={format_results['gate_v2']['deltas']['rag_a_minus_rag_b_same_company_premium']['rougeL_f1']['mean']:+.4f}. "
        f"Per PREREG.md criteria, the automated verdict is '{verdict}'."
    )

    # Compute company concentration for queries and top-1 precedents
    query_comp_counts = {}
    for r in sample_records:
        c = r.get("company", "Unknown")
        query_comp_counts[c] = query_comp_counts.get(c, 0) + 1
    query_comp_shares = {
        c: {"count": cnt, "pct": round(cnt / len(sample_records) * 100, 2)}
        for c, cnt in sorted(query_comp_counts.items(), key=lambda x: -x[1])
    }

    top1_comp_shares = {}
    for short_name, ret_key in cond_keys:
        comp_cnts = {}
        for r in sample_records:
            top1_c = retrieval_by_query[r["incident_id"]][ret_key]["hits"][0]["company"]
            comp_cnts[top1_c] = comp_cnts.get(top1_c, 0) + 1
        top1_comp_shares[short_name] = {
            c: {"count": cnt, "pct": round(cnt / len(sample_records) * 100, 2)}
            for c, cnt in sorted(comp_cnts.items(), key=lambda x: -x[1])
        }

    company_concentration = {
        "num_queries": len(sample_records),
        "num_query_companies": len(query_comp_counts),
        "query_company_shares": query_comp_shares,
        "top1_precedent_company_shares": top1_comp_shares,
    }

    results_obj = {
        "regime": "leave-one-out on train companies; NOT comparable to company-disjoint test numbers",
        "metadata": {
            "num_incidents": len(test_ids),
            "num_companies": len(set(id_to_company.values())),
            "generation_model": GEN_MODEL_ID,
            "judge_model": JUDGE_MODEL_ID,
            "judge_rubric": "eval/rubrics/relevance_v1.md",
            "bootstrap_resamples": 10000,
            "seed": 42,
            "family_size_m": 5,
            "adjusted_alpha": 0.01,
            "unadjusted_alpha": 0.05,
            "gt_median_word_count": median_gt_limit,
            "num_new_gen_calls": new_gen_calls,
            "num_cached_gen_calls": cached_gen_calls,
            "num_new_judge_calls": new_judge_calls,
            "num_cached_judge_calls": cached_judge_calls,
        },
        "company_concentration": company_concentration,
        "relevance_at_1": relevance_distributions,
        "formats": format_results,
        "stratified_deltas_by_relevance_gate_v2": stratified_deltas,
        "verdict": verdict,
        "summary": summary_text,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_obj, f, indent=2)

    print(f"\n[Artifact] Successfully wrote LOO regime results to {OUTPUT_FILE}")
    print(f"Overall verdict: {verdict}")
    print("=" * 60)
    return results_obj


if __name__ == "__main__":
    run_e2_evaluation()
