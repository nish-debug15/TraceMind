"""
TraceMind Prompt Robustness Ablation Suite
-----------------------------------------
Evaluates output sensitivity across the 3 core RAG prompt formulations:
1. baseline: Standard SRE persona and reference format.
2. relevance_gate: Precedent relevance checking guardrail (soft check).
3. relevance_gate_v2: Strict machine-parseable 'Precedent: [relevant|not relevant]' header.

Requirements:
- Evaluates strictly on held-out postmortems (data/splits/test.json, 93 incidents).
- Zero data leakage: retrieves strictly against rag/chroma_db_train (postmortems_train).
- Caches all generations in SQLite and eval/cache/ to ensure reproducibility.
- Saves results to eval/results/prompt_ablation.json.
"""

import os
import sys
import re
import json
import time
import argparse
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import requests
from dotenv import load_dotenv

# Force UTF-8 encoding for stdout on Windows
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from rouge_score import rouge_scorer
from bert_score import score as calc_bertscore
from clustering.cluster import embed
from rag.prompts import PROMPT_REGISTRY, PROMPT_VARIANTS
from rag.generate import RateLimiter, LLM_MAX_RPM, GROQ_API_URL
from rag.gen_cache import get_cached_generation, set_cached_generation

DATA_DIR = PROJECT_ROOT / "data"
TEST_FILE = DATA_DIR / "splits" / "test.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

RAG_VS_ZS_FILE = RESULTS_DIR / "rag_vs_zero_shot.json"
RAG_CACHE_FILE = CACHE_DIR / "rag_generations.json"
OUTPUT_FILE = RESULTS_DIR / "prompt_ablation.json"

MODEL_ID = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM)


def parse_rca_output(raw_text: str) -> tuple:
    """Parse ROOT CAUSE and REMEDIATION STEPS from LLM completion."""
    rc_match = re.search(r"ROOT CAUSE:\s*(.*?)(?=REMEDIATION STEPS:|$)", raw_text, re.DOTALL | re.IGNORECASE)
    rem_match = re.search(r"REMEDIATION STEPS:\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)
    rc = rc_match.group(1).strip() if rc_match else raw_text.strip()
    rem = rem_match.group(1).strip() if rem_match else ""
    return rc, rem


def call_llm_with_retry(messages: List[Dict[str, str]], api_key: str, max_retries: int = 4) -> str:
    """Call Groq chat completion with rate limiting and retry."""
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
                    "temperature": 0.2,
                    "max_tokens": 800,
                },
                timeout=60.0
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_sec = float(retry_after) if retry_after else delay
                print(f"  [Rate Limit / 429] Waiting {sleep_sec:.1f}s...")
                time.sleep(sleep_sec)
                delay *= 2.0
            elif resp.status_code >= 500:
                time.sleep(delay)
                delay *= 2.0
            else:
                raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
    raise RuntimeError("Exceeded maximum retries calling Groq API.")


def run_prompt_ablation(limit: int = None, force_refresh: bool = False):
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing {TEST_FILE}.")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    if limit:
        test_records = test_records[:limit]

    n_incidents = len(test_records)
    print(f"[Prompt Ablation] Evaluating {n_incidents} test incidents across 3 prompts...")

    # Load pre-retrieved train matches from rag_vs_zero_shot.json to guarantee zero leakage and consistency
    retrieval_map = {}
    if RAG_VS_ZS_FILE.exists():
        with open(RAG_VS_ZS_FILE, "r", encoding="utf-8") as f:
            rvz_data = json.load(f)
            for item in rvz_data.get("per_incident_scores", []):
                retrieval_map[item["incident_id"]] = item["retrieval"]

    # Also check baseline generations from rag_generations.json
    baseline_cache = {}
    if RAG_CACHE_FILE.exists():
        with open(RAG_CACHE_FILE, "r", encoding="utf-8") as f:
            baseline_cache = json.load(f)

    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()

    prompt_names = ["baseline", "relevance_gate", "relevance_gate_v2"]
    generations_by_variant = {p: {} for p in prompt_names}

    for p in prompt_names:
        cache_path = CACHE_DIR / f"ablation_{p}.json"
        if cache_path.exists() and not force_refresh:
            with open(cache_path, "r", encoding="utf-8") as f:
                generations_by_variant[p] = json.load(f)

    # Process all test incidents
    for idx, record in enumerate(test_records):
        inc_id = record["incident_id"]
        log_text = record["raw_log_excerpt"]
        retrieval_info = retrieval_map.get(inc_id)

        if not retrieval_info:
            print(f"Warning: No pre-retrieved context for {inc_id}. Skipping.")
            continue

        ret_meta = retrieval_info.get("metadata", {})
        similarity = retrieval_info.get("similarity_score", 0.0)

        for p_name in prompt_names:
            # If already in variant cache
            if inc_id in generations_by_variant[p_name] and not force_refresh:
                continue

            # Special case for baseline: check baseline_cache
            if p_name == "baseline" and inc_id in baseline_cache and not force_refresh:
                b_item = baseline_cache[inc_id]
                generations_by_variant[p_name][inc_id] = {
                    "root_cause": b_item["root_cause"],
                    "remediation_steps": b_item["remediation_steps"],
                    "raw_output": b_item.get("raw_output", ""),
                    "source": "rag_cache"
                }
                continue

            # Check SQLite gen cache
            sqlite_cached = get_cached_generation(
                model=MODEL_ID,
                prompt_variant=p_name,
                query_text_or_id=log_text,
                incident_id=inc_id
            )
            if sqlite_cached and not force_refresh:
                generations_by_variant[p_name][inc_id] = {
                    "root_cause": sqlite_cached["root_cause"],
                    "remediation_steps": sqlite_cached["remediation_steps"],
                    "raw_output": sqlite_cached.get("raw_output", ""),
                    "source": "sqlite_cache"
                }
                continue

            # Format messages
            variant_def = PROMPT_REGISTRY[p_name]
            sys_msg = variant_def["system"]
            user_msg = variant_def["user_template"].format(
                ref_root_cause=ret_meta.get("root_cause", ""),
                ref_remediation_steps=ret_meta.get("remediation_steps", ""),
                log_text=log_text,
                similarity=similarity
            )
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ]

            print(f"  [{idx+1}/{n_incidents}] Generating {p_name} for {inc_id}...")
            raw_out = call_llm_with_retry(messages, groq_api_key)
            rc, rem = parse_rca_output(raw_out)

            generations_by_variant[p_name][inc_id] = {
                "root_cause": rc,
                "remediation_steps": rem,
                "raw_output": raw_out,
                "source": "live_groq"
            }

            # Update SQLite cache
            set_cached_generation(
                model=MODEL_ID,
                prompt_variant=p_name,
                query_text_or_id=log_text,
                root_cause=rc,
                remediation_steps=rem,
                raw_output=raw_out,
                incident_id=inc_id
            )

            # Persist local cache file
            with open(CACHE_DIR / f"ablation_{p_name}.json", "w", encoding="utf-8") as f:
                json.dump(generations_by_variant[p_name], f, indent=2)

    # Save all caches
    for p_name in prompt_names:
        with open(CACHE_DIR / f"ablation_{p_name}.json", "w", encoding="utf-8") as f:
            json.dump(generations_by_variant[p_name], f, indent=2)

    # Now score all variants against ground truth
    print("\n[Prompt Ablation] Scoring all variants against ground truth...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    gt_by_id = {r["incident_id"]: r["root_cause"] for r in test_records}
    valid_ids = [r["incident_id"] for r in test_records if r["incident_id"] in retrieval_map]
    ground_truth = [gt_by_id[i] for i in valid_ids]

    results_summary = {}

    for p_name in prompt_names:
        p_preds = [generations_by_variant[p_name][i]["root_cause"] for i in valid_ids]

        # ROUGE-L
        rl_scores = [scorer.score(gt, pred)["rougeL"].fmeasure for gt, pred in zip(ground_truth, p_preds)]

        # BERTScore F1
        print(f"  Scoring BERTScore for {p_name}...")
        _, _, bs_f1 = calc_bertscore(p_preds, ground_truth, model_type="bert-base-uncased", verbose=False)
        bs_vals = bs_f1.numpy().tolist()

        # Cosine Similarity
        cos_sims = []
        for gt, pred in zip(ground_truth, p_preds):
            v_pred = np.array(embed(pred[:1000]))
            v_ref = np.array(embed(gt[:1000]))
            cos_sims.append(float(v_pred @ v_ref))

        results_summary[p_name] = {
            "rougeL_f1": round(float(np.mean(rl_scores)), 4),
            "rougeL_f1_std": round(float(np.std(rl_scores)), 4),
            "bert_score_f1": round(float(np.mean(bs_vals)), 4),
            "bert_score_f1_std": round(float(np.std(bs_vals)), 4),
            "cosine_sim": round(float(np.mean(cos_sims)), 4),
            "cosine_sim_std": round(float(np.std(cos_sims)), 4),
        }

    # Compute spread / variance across prompts
    rl_means = [results_summary[p]["rougeL_f1"] for p in prompt_names]
    bs_means = [results_summary[p]["bert_score_f1"] for p in prompt_names]
    cos_means = [results_summary[p]["cosine_sim"] for p in prompt_names]

    spread = {
        "rougeL_f1_spread": round(float(max(rl_means) - min(rl_means)), 4),
        "rougeL_f1_variance": round(float(np.var(rl_means)), 6),
        "bert_score_f1_spread": round(float(max(bs_means) - min(bs_means)), 4),
        "bert_score_f1_variance": round(float(np.var(bs_means)), 6),
        "cosine_sim_spread": round(float(max(cos_means) - min(cos_means)), 4),
        "cosine_sim_variance": round(float(np.var(cos_means)), 6),
    }

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(PROJECT_ROOT)).strip()
    except Exception:
        git_sha = "unknown"

    output_payload = {
        "config": {
            "model": MODEL_ID,
            "test_dataset": "data/splits/test.json",
            "num_samples": len(valid_ids),
            "prompt_variants": prompt_names,
            "bert_score_model": "bert-base-uncased",
            "git_sha": git_sha,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "variants": results_summary,
        "spread": spread,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("PROMPT ROBUSTNESS ABLATION RESULTS (N = 93)")
    print("=" * 80)
    print(f"{'Prompt Variant':<22} | {'ROUGE-L F1':<14} | {'BERTScore F1':<14} | {'Cosine Sim':<14}")
    print("-" * 72)
    for p_name in prompt_names:
        r = results_summary[p_name]
        print(f"{p_name:<22} | {r['rougeL_f1']:<14.4f} | {r['bert_score_f1']:<14.4f} | {r['cosine_sim']:<14.4f}")
    print("-" * 72)
    print(f"{'Spread (Max - Min)':<22} | {spread['rougeL_f1_spread']:<14.4f} | {spread['bert_score_f1_spread']:<14.4f} | {spread['cosine_sim_spread']:<14.4f}")
    print(f"{'Variance':<22} | {spread['rougeL_f1_variance']:<14.6f} | {spread['bert_score_f1_variance']:<14.6f} | {spread['cosine_sim_variance']:<14.6f}")
    print(f"\n[Artifact] Saved prompt ablation results to {OUTPUT_FILE}")

    return output_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate prompt robustness ablation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test records to evaluate")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and regenerate completions")
    args = parser.parse_args()

    run_prompt_ablation(limit=args.limit, force_refresh=args.force_refresh)


if __name__ == "__main__":
    main()
