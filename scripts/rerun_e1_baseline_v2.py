"""
TraceMind Rerun E1 Baseline Random Arm (Block 3)
-----------------------------------------------
Executes 93 baseline-format random-precedent generations with the fixed prompt builder
aligned with control_2x2 (eval/evaluate.py).
Generations are cached in eval/cache/e1_v2_cache.sqlite.
Outputs:
- eval/results/random_precedent_generations_v2.json
- eval/results/random_precedent_control_v2.json
Keeps existing random_precedent_control.json untouched.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from eval.scorer import score_cell, word_count, compute_bert_scores, _ROUGE_SCORER
from eval.analyze_e1 import FastClusterBootstrap, determine_verdict
from rag.generate import RateLimiter, LLM_MAX_RPM, GROQ_API_URL

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

ORIG_GEN_FILE = RESULTS_DIR / "random_precedent_generations.json"
GEN_V2_FILE = RESULTS_DIR / "random_precedent_generations_v2.json"
CONTROL_V2_FILE = RESULTS_DIR / "random_precedent_control_v2.json"
CACHE_DB_PATH = CACHE_DIR / "e1_v2_cache.sqlite"

MODEL_ID = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM)

BASELINE_SYSTEM_PROMPT = (
    "You are an expert SRE assistant analyzing infrastructure incidents. "
    "Structure your output into ROOT CAUSE: and REMEDIATION STEPS: sections."
)


def build_baseline_user_prompt(ref_root_cause: str, ref_remediation_steps: str, log_text: str) -> str:
    return (
        f"Historical Incident Reference:\n"
        f"- Root Cause: {ref_root_cause}\n"
        f"- Remediation: {ref_remediation_steps}\n\n"
        f"New Incident Log Excerpt:\n"
        f"{log_text}\n\n"
        f"Provide your response in the following format:\n"
        f"ROOT CAUSE:\n"
        f"(your analysis)\n\n"
        f"REMEDIATION STEPS:\n"
        f"(your steps)"
    )


def parse_rca(generated_text: str) -> Tuple[str, str, str]:
    norm = generated_text
    for marker in ["### ROOT CAUSE", "## ROOT CAUSE", "ROOT CAUSE:", "Root Cause:"]:
        if marker in norm:
            norm = norm.replace(marker, "ROOT CAUSE:")
            break
    for marker in ["### REMEDIATION STEPS", "## REMEDIATION STEPS", "REMEDIATION STEPS:", "Remediation Steps:"]:
        if marker in norm:
            norm = norm.replace(marker, "REMEDIATION STEPS:")
            break
    root_cause = norm
    remediation_steps = ""
    if "REMEDIATION STEPS:" in norm:
        parts = norm.split("REMEDIATION STEPS:")
        root_cause = parts[0].replace("ROOT CAUSE:", "").strip()
        remediation_steps = parts[1].strip()
    elif "ROOT CAUSE:" in norm:
        root_cause = norm.replace("ROOT CAUSE:", "").strip()

    status = "both_parsed" if (root_cause and remediation_steps) else ("root_cause_only" if root_cause else "failed")
    return root_cause, remediation_steps, status


def init_cache_db() -> sqlite3.Connection:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS e1_v2_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT,
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
                m_ms = re.search(r"Please try again in (\d+(?:\.\d+)?)ms", err_text)
                if m_ms:
                    wait_sec = float(m_ms.group(1)) / 1000.0 + 1.0
                else:
                    m = re.search(r"Please try again in (?:(\d+)m(?!s))?([\d\.]+)s", err_text)
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


def run_rerun_and_rescore():
    print("=" * 80)
    print("BLOCK 3: Rerunning E1 Baseline Random Arm with Fixed Prompt Builder")
    print("=" * 80)

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is required.")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    test_ids = [r["incident_id"] for r in test_records]
    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in test_records}
    gt_rc_map = {r["incident_id"]: r["root_cause"] for r in test_records}
    gt_rem_map = {r["incident_id"]: r.get("remediation_steps", "") for r in test_records}

    gt_word_counts = [word_count(gt_rc_map[i]) for i in test_ids]
    median_gt_limit = int(round(float(np.median(gt_word_counts))))

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_map = {r["incident_id"]: r for r in json.load(f)}

    with open(ORIG_GEN_FILE, "r", encoding="utf-8") as f:
        orig_generations = json.load(f)["generations"]

    cache_conn = init_cache_db()
    cur = cache_conn.cursor()

    num_new_calls = 0
    num_cached_calls = 0

    new_baseline_gens = {}
    v2_generation_records = []

    for idx, gen in enumerate(orig_generations):
        qid = gen["query_incident_id"]
        pid = gen["random_precedent_id"]
        sim = gen["similarity_score"]
        q_rec = [r for r in test_records if r["incident_id"] == qid][0]
        q_log = q_rec["raw_log_excerpt"]
        p_rec = train_map[pid]
        p_rc = p_rec.get("root_cause", "")
        p_rem = p_rec.get("remediation_steps", "")

        user_msg = build_baseline_user_prompt(p_rc, p_rem, q_log)
        p_hash = hashlib.sha256(f"{BASELINE_SYSTEM_PROMPT}\n{user_msg}".encode("utf-8")).hexdigest()
        cache_key = hashlib.sha256(f"{MODEL_ID}:{p_hash}:{qid}:{pid}".encode("utf-8")).hexdigest()

        cur.execute("SELECT raw_output, root_cause, remediation_steps, parse_status FROM e1_v2_cache WHERE cache_key = ?", (cache_key,))
        row = cur.fetchone()

        if row:
            raw_out, rc, rem, status = row
            num_cached_calls += 1
        else:
            messages = [
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]
            raw_out = call_llm(messages, api_key, max_tokens=800)
            rc, rem, status = parse_rca(raw_out)
            with cache_conn:
                cur.execute(
                    """INSERT OR REPLACE INTO e1_v2_cache
                       (cache_key, model, prompt_hash, query_id, precedent_id, raw_output, root_cause, remediation_steps, parse_status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (cache_key, MODEL_ID, p_hash, qid, pid, raw_out, rc, rem, status)
                )
            num_new_calls += 1
            print(f"  [{idx+1:02d}/93] Generated baseline random for {qid} ({len(rc.split())} words, {status})", flush=True)

        new_baseline_gens[qid] = {
            "root_cause": rc,
            "remediation_steps": rem,
            "raw_output": raw_out,
            "parse_status": status,
        }

        v2_record = dict(gen)
        v2_record["baseline"] = {
            "root_cause": rc,
            "remediation_steps": rem,
            "raw_output": raw_out,
            "parse_status": status,
            "prompt_hash": p_hash,
        }
        v2_generation_records.append(v2_record)

    print(f"\nBaseline arm rerun complete: {num_new_calls} new LLM calls, {num_cached_calls} cached calls.")

    # Save random_precedent_generations_v2.json
    gen_v2_out = {
        "metadata": {
            "num_queries": len(test_ids),
            "seed": 42,
            "model": MODEL_ID,
            "temperature": 0.0,
            "num_new_llm_calls": num_new_calls,
            "num_cached_calls": num_cached_calls,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "Baseline format random arm regenerated with fixed prompt builder matching control_2x2/eval/evaluate.py."
        },
        "generations": v2_generation_records
    }
    with open(GEN_V2_FILE, "w", encoding="utf-8") as f:
        json.dump(gen_v2_out, f, indent=2)
    print(f"Saved generations v2 to {GEN_V2_FILE}")

    # Now Rescore E1 v2
    print("\n--- Rescoring E1 into random_precedent_control_v2.json ---")
    bootstrapper = FastClusterBootstrap(test_ids, id_to_company, n_resamples=10000, seed=42)

    with open(CACHE_DIR / "zeroshot_generations.json", "r", encoding="utf-8") as f:
        cell_base_zs = json.load(f)
    with open(CACHE_DIR / "rag_generations.json", "r", encoding="utf-8") as f:
        cell_base_rag = json.load(f)
    with open(CACHE_DIR / "ablation_zero_shot_gate_format.json", "r", encoding="utf-8") as f:
        cell_gate_zs = json.load(f)
    with open(CACHE_DIR / "ablation_relevance_gate_v2.json", "r", encoding="utf-8") as f:
        cell_gate_rag = json.load(f)

    def rescore_arm(preds_map, rems_map):
        preds = [preds_map[i] for i in test_ids]
        refs = [gt_rc_map[i] for i in test_ids]
        scored = score_cell(preds, refs)
        per_inc = scored["per_incident"]

        lm_preds = [" ".join(p.split()[:median_gt_limit]) for p in preds]
        lm_rl = [_ROUGE_SCORER.score(ref or "none", lmp or "none")["rougeL"].fmeasure for ref, lmp in zip(refs, lm_preds)]
        wc_list = per_inc["word_count"]

        return {
            "rouge1_f1": bootstrapper.compute_multi_ci(per_inc["rouge1_f1"]),
            "rouge2_f1": bootstrapper.compute_multi_ci(per_inc["rouge2_f1"]),
            "rougeL_f1": bootstrapper.compute_multi_ci(per_inc["rougeL_f1"]),
            "length_matched_rougeL": bootstrapper.compute_multi_ci(lm_rl),
            "bert_score_f1": bootstrapper.compute_multi_ci(per_inc["bert_score_f1"]),
            "cosine": bootstrapper.compute_multi_ci(per_inc["cosine"]),
            "word_count": {
                "mean": round(float(np.mean(wc_list)), 2),
                "median": round(float(np.median(wc_list)), 2),
            },
            "_raw": {
                "rouge1_f1": per_inc["rouge1_f1"],
                "rouge2_f1": per_inc["rouge2_f1"],
                "rougeL_f1": per_inc["rougeL_f1"],
                "length_matched_rougeL": lm_rl,
                "bert_score_f1": per_inc["bert_score_f1"],
                "cosine": per_inc["cosine"],
            }
        }

    def compute_paired_deltas(arm_a_raw, arm_b_raw):
        deltas = {}
        for m in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "length_matched_rougeL", "bert_score_f1", "cosine"]:
            diff = [a - b for a, b in zip(arm_a_raw[m], arm_b_raw[m])]
            deltas[m] = bootstrapper.compute_multi_ci(diff)
        return deltas

    def evaluate_remediation_pairwise(preds_a, preds_b):
        both_ids = [
            i for i in test_ids
            if preds_a.get(i, "").strip() and preds_b.get(i, "").strip() and len(gt_rem_map.get(i, "").split()) >= 5
        ]
        if not both_ids:
            return {"n_both_parsed": 0, "status": "no incidents parsed in both arms"}
        sub_boot = FastClusterBootstrap(both_ids, id_to_company, n_resamples=10000, seed=42)
        a_texts = [preds_a[i].strip() for i in both_ids]
        b_texts = [preds_b[i].strip() for i in both_ids]
        gts = [gt_rem_map[i].strip() for i in both_ids]
        rl_a = [_ROUGE_SCORER.score(g, p)["rougeL"].fmeasure for g, p in zip(gts, a_texts)]
        rl_b = [_ROUGE_SCORER.score(g, p)["rougeL"].fmeasure for g, p in zip(gts, b_texts)]
        rl_diff = [a - b for a, b in zip(rl_a, rl_b)]
        bs_a = compute_bert_scores(a_texts, gts)
        bs_b = compute_bert_scores(b_texts, gts)
        bs_diff = [a - b for a, b in zip(bs_a, bs_b)]
        return {
            "n_both_parsed": len(both_ids),
            "arm_a_rougeL": sub_boot.compute_multi_ci(rl_a),
            "arm_b_rougeL": sub_boot.compute_multi_ci(rl_b),
            "arm_a_bert_score": sub_boot.compute_multi_ci(bs_a),
            "arm_b_bert_score": sub_boot.compute_multi_ci(bs_b),
            "delta_rougeL": sub_boot.compute_multi_ci(rl_diff),
            "delta_bert_score": sub_boot.compute_multi_ci(bs_diff),
        }

    # 1. Baseline format
    b_zs_rc = {i: cell_base_zs[i]["root_cause"] for i in test_ids}
    b_zs_rem = {i: cell_base_zs[i].get("remediation_steps", "") for i in test_ids}
    b_rag_rc = {i: cell_base_rag[i]["root_cause"] for i in test_ids}
    b_rag_rem = {i: cell_base_rag[i].get("remediation_steps", "") for i in test_ids}
    b_rand_rc = {i: new_baseline_gens[i]["root_cause"] for i in test_ids}
    b_rand_rem = {i: new_baseline_gens[i]["remediation_steps"] for i in test_ids}

    b_zs_arm = rescore_arm(b_zs_rc, b_zs_rem)
    b_rag_arm = rescore_arm(b_rag_rc, b_rag_rem)
    b_rand_arm = rescore_arm(b_rand_rc, b_rand_rem)

    b_deltas = {
        "real_minus_random": compute_paired_deltas(b_rag_arm["_raw"], b_rand_arm["_raw"]),
        "random_minus_zeroshot": compute_paired_deltas(b_rand_arm["_raw"], b_zs_arm["_raw"]),
        "real_minus_zeroshot": compute_paired_deltas(b_rag_arm["_raw"], b_zs_arm["_raw"]),
    }
    b_rem = {
        "real_minus_random": evaluate_remediation_pairwise(b_rag_rem, b_rand_rem),
        "random_minus_zeroshot": evaluate_remediation_pairwise(b_rand_rem, b_zs_rem),
        "real_minus_zeroshot": evaluate_remediation_pairwise(b_rag_rem, b_zs_rem),
    }
    b_verdict = determine_verdict(
        random_minus_zs_excludes=b_deltas["random_minus_zeroshot"]["rougeL_f1"]["excludes_zero_after_adjustment"],
        real_minus_random_excludes=b_deltas["real_minus_random"]["rougeL_f1"]["excludes_zero_after_adjustment"]
    )

    # 2. Gate-v2 format
    g_zs_rc = {i: cell_gate_zs[i]["root_cause"] for i in test_ids}
    g_zs_rem = {i: cell_gate_zs[i].get("remediation_steps", "") for i in test_ids}
    g_rag_rc = {i: cell_gate_rag[i]["root_cause"] for i in test_ids}
    g_rag_rem = {i: cell_gate_rag[i].get("remediation_steps", "") for i in test_ids}
    orig_gate_rand = {g["query_incident_id"]: g["gate_v2"] for g in orig_generations}
    g_rand_rc = {i: orig_gate_rand[i]["root_cause"] for i in test_ids}
    g_rand_rem = {i: orig_gate_rand[i].get("remediation_steps", "") for i in test_ids}

    g_zs_arm = rescore_arm(g_zs_rc, g_zs_rem)
    g_rag_arm = rescore_arm(g_rag_rc, g_rag_rem)
    g_rand_arm = rescore_arm(g_rand_rc, g_rand_rem)

    g_deltas = {
        "real_minus_random": compute_paired_deltas(g_rag_arm["_raw"], g_rand_arm["_raw"]),
        "random_minus_zeroshot": compute_paired_deltas(g_rand_arm["_raw"], g_zs_arm["_raw"]),
        "real_minus_zeroshot": compute_paired_deltas(g_rag_arm["_raw"], g_zs_arm["_raw"]),
    }
    g_rem = {
        "real_minus_random": evaluate_remediation_pairwise(g_rag_rem, g_rand_rem),
        "random_minus_zeroshot": evaluate_remediation_pairwise(g_rand_rem, g_zs_rem),
        "real_minus_zeroshot": evaluate_remediation_pairwise(g_rag_rem, g_zs_rem),
    }
    g_verdict = determine_verdict(
        random_minus_zs_excludes=g_deltas["random_minus_zeroshot"]["rougeL_f1"]["excludes_zero_after_adjustment"],
        real_minus_random_excludes=g_deltas["real_minus_random"]["rougeL_f1"]["excludes_zero_after_adjustment"]
    )

    control_v2 = {
        "metadata": {
            "num_incidents": len(test_ids),
            "num_companies": len(set(id_to_company.values())),
            "bootstrap_resamples": 10000,
            "seed": 42,
            "family_size_m": 5,
            "adjusted_alpha": 0.01,
            "unadjusted_alpha": 0.05,
            "gt_median_word_count": median_gt_limit,
            "version": "v2",
            "scorer": "eval/scorer.py canonical module",
            "baseline_random_status": "rerun with aligned prompt builder",
            "num_new_llm_calls": num_new_calls,
        },
        "formats": {
            "baseline": {
                "arms": {
                    "zero_shot": b_zs_arm,
                    "random_precedent": b_rand_arm,
                    "real_rag": b_rag_arm,
                },
                "deltas": b_deltas,
                "remediation_both_parsed": b_rem,
                "verdict": b_verdict,
            },
            "gate_v2": {
                "arms": {
                    "zero_shot": g_zs_arm,
                    "random_precedent": g_rand_arm,
                    "real_rag": g_rag_arm,
                },
                "deltas": g_deltas,
                "remediation_both_parsed": g_rem,
                "verdict": g_verdict,
            }
        },
        "verdict": g_verdict,
        "summary": f"E1 Control v2: Baseline random arm rerun with aligned prompt builder. Gate-v2 verdict: {g_verdict}. Baseline verdict: {b_verdict}."
    }

    with open(CONTROL_V2_FILE, "w", encoding="utf-8") as f:
        json.dump(control_v2, f, indent=2)
    print(f"Saved E1 Control v2 to {CONTROL_V2_FILE}")

    # Print summary comparison
    print("\n" + "=" * 80)
    print("E1 V2 RESULTS SUMMARY")
    print("=" * 80)
    print(f"Baseline Random Arm Word Count: Mean = {b_rand_arm['word_count']['mean']}, Median = {b_rand_arm['word_count']['median']}")
    print(f"Baseline Real RAG Word Count:   Mean = {b_rag_arm['word_count']['mean']}, Median = {b_rag_arm['word_count']['median']}")
    print(f"Baseline Zero-Shot Word Count:  Mean = {b_zs_arm['word_count']['mean']}, Median = {b_zs_arm['word_count']['median']}")
    print(f"\nBaseline Deltas:")
    print(f"  Random - Zero-Shot ROUGE-L: {b_deltas['random_minus_zeroshot']['rougeL_f1']['mean']:.4f} (99% CI {b_deltas['random_minus_zeroshot']['rougeL_f1']['adjusted_ci']}) [{b_deltas['random_minus_zeroshot']['rougeL_f1']['status']}]")
    print(f"  Real - Random ROUGE-L:     {b_deltas['real_minus_random']['rougeL_f1']['mean']:.4f} (99% CI {b_deltas['real_minus_random']['rougeL_f1']['adjusted_ci']}) [{b_deltas['real_minus_random']['rougeL_f1']['status']}]")
    print(f"  Baseline Verdict: {b_verdict}")
    print(f"\nGate-v2 Deltas:")
    print(f"  Random - Zero-Shot ROUGE-L: {g_deltas['random_minus_zeroshot']['rougeL_f1']['mean']:.4f} (99% CI {g_deltas['random_minus_zeroshot']['rougeL_f1']['adjusted_ci']}) [{g_deltas['random_minus_zeroshot']['rougeL_f1']['status']}]")
    print(f"  Real - Random ROUGE-L:     {g_deltas['real_minus_random']['rougeL_f1']['mean']:.4f} (99% CI {g_deltas['real_minus_random']['rougeL_f1']['adjusted_ci']}) [{g_deltas['real_minus_random']['rougeL_f1']['status']}]")
    print(f"  Gate-v2 Verdict: {g_verdict}")
    print("=" * 80)


if __name__ == "__main__":
    run_rerun_and_rescore()
