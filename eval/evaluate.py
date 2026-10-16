"""
TraceMind Evaluation Suite
--------------------------
Evaluates Root Cause Analysis (RCA) on held-out test postmortems (zero leakage).
Compares RAG-augmented generation vs. Zero-Shot (No-RAG) baseline.

Requirements:
- GROQ_API_KEY must be set in environment or .env. Mock responses cause immediate hard failure.
- Generations are cached to disk (rag/gen_cache.py and eval/cache/) to prevent API exhaustion.
- Results are broken down by Full Set, Quality:High subset, and Quality:Low subset.

Usage:
    python eval/evaluate.py [--limit N] [--force-refresh] [--use-bertscore]
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import argparse
from typing import Dict, List, Any
import numpy as np
import requests
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

from rouge_score import rouge_scorer
import chromadb

from clustering.cluster import embed
from rag.generate import retrieve, COLLECTION_NAME, CHROMA_DB_PATH, RateLimiter, LLM_MAX_RPM, GROQ_API_URL
from rag.gen_cache import get_cached_generation, set_cached_generation

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR.parent / "data"
TEST_FILE = DATA_DIR / "test.json"
CACHE_DIR = EVAL_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

RAG_CACHE_FILE = CACHE_DIR / "rag_generations.json"
ZEROSHOT_CACHE_FILE = CACHE_DIR / "zeroshot_generations.json"
RESULTS_FILE = EVAL_DIR / "evaluation_report.json"

MODEL_ID = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM)


def check_no_leakage(test_records: List[Dict[str, Any]]):
    """Assert that zero test incidents exist in the ChromaDB retrieval index."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    all_chroma_ids = set(collection.get()["ids"])
    test_ids = {r["incident_id"] for r in test_records}
    overlap = test_ids.intersection(all_chroma_ids)
    if overlap:
        raise RuntimeError(
            f"CRITICAL LEAKAGE ERROR: {len(overlap)} test incidents found inside the retrieval index! "
            f"Overlap sample: {list(overlap)[:5]}. Rebuild index from train.json only."
        )
    print(f"[Verification] Index contains {collection.count()} train records. Test leakage: 0 records.")


def call_llm_with_retry(prompt: str, api_key: str, max_retries: int = 4) -> str:
    """Execute LLM chat completion via Groq with rate limiting and exponential backoff."""
    delay = 2.0
    for attempt in range(max_retries):
        _rate_limiter.acquire()
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_ID,
                    "messages": [
                        {"role": "system", "content": "You are an expert SRE assistant analyzing infrastructure incidents. Structure your output into ROOT CAUSE: and REMEDIATION STEPS: sections."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500,
                },
                timeout=60.0
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if not content or "[MOCK]" in content:
                    raise RuntimeError("Received mock or empty response from Groq API.")
                return content
            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_sec = float(retry_after) if retry_after else delay
                print(f"  [Rate Limit / 429] Waiting {sleep_sec:.1f}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_sec)
                delay *= 2.0
            elif resp.status_code >= 500:
                print(f"  [Server Error {resp.status_code}] Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [Network Error: {e}] Retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError("Exceeded maximum retries calling Groq API.")


def parse_rca(generated_text: str) -> Dict[str, str]:
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
    return {
        "root_cause": root_cause,
        "remediation_steps": remediation_steps
    }


def load_cache(cache_file: Path) -> Dict[str, Any]:
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Any], cache_file: Path):
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def generate_rag(record: Dict[str, Any], api_key: str, use_cache: bool = True) -> Dict[str, Any]:
    log_text = record["raw_log_excerpt"]
    match_meta, similarity = retrieve(log_text, k=1)
    if not match_meta:
        match_meta = {"root_cause": "N/A", "remediation_steps": "N/A"}

    if use_cache:
        cached = get_cached_generation(model=MODEL_ID, prompt_variant="rag", query_text_or_id=log_text)
        if cached:
            return {
                "root_cause": cached["root_cause"],
                "remediation_steps": cached["remediation_steps"],
                "retrieved_incident_id": match_meta.get("incident_id"),
                "similarity_score": similarity,
                "prompt": "(cached)",
                "is_mock": False
            }

    prompt = f"""Historical Incident Reference:
- Root Cause: {match_meta['root_cause']}
- Remediation: {match_meta['remediation_steps']}

New Incident Log Excerpt:
{log_text}

Provide your response in the following format:
ROOT CAUSE:
(your analysis)

REMEDIATION STEPS:
(your steps)"""
    raw_gen = call_llm_with_retry(prompt, api_key)
    parsed = parse_rca(raw_gen)
    if use_cache and not parsed["root_cause"].startswith("[MOCK]"):
        set_cached_generation(
            model=MODEL_ID,
            prompt_variant="rag",
            query_text_or_id=log_text,
            root_cause=parsed["root_cause"],
            remediation_steps=parsed["remediation_steps"],
            raw_output=raw_gen
        )
    return {
        "root_cause": parsed["root_cause"],
        "remediation_steps": parsed["remediation_steps"],
        "retrieved_incident_id": match_meta.get("incident_id"),
        "similarity_score": similarity,
        "prompt": prompt,
        "is_mock": False
    }


def generate_zeroshot(record: Dict[str, Any], api_key: str, use_cache: bool = True) -> Dict[str, Any]:
    log_text = record["raw_log_excerpt"]
    if use_cache:
        cached = get_cached_generation(model=MODEL_ID, prompt_variant="zeroshot", query_text_or_id=log_text)
        if cached:
            return {
                "root_cause": cached["root_cause"],
                "remediation_steps": cached["remediation_steps"],
                "prompt": "(cached)",
                "is_mock": False
            }

    prompt = f"""New Incident Log Excerpt:
{log_text}

Provide your response in the following format:
ROOT CAUSE:
(your analysis)

REMEDIATION STEPS:
(your steps)"""
    raw_gen = call_llm_with_retry(prompt, api_key)
    parsed = parse_rca(raw_gen)
    if use_cache and not parsed["root_cause"].startswith("[MOCK]"):
        set_cached_generation(
            model=MODEL_ID,
            prompt_variant="zeroshot",
            query_text_or_id=log_text,
            root_cause=parsed["root_cause"],
            remediation_steps=parsed["remediation_steps"],
            raw_output=raw_gen
        )
    return {
        "root_cause": parsed["root_cause"],
        "remediation_steps": parsed["remediation_steps"],
        "prompt": prompt,
        "is_mock": False
    }


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    cosine_sims = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

        v_pred = np.array(embed(pred))
        v_ref = np.array(embed(ref))
        sim = float(v_pred @ v_ref)
        cosine_sims.append(sim)

    return {
        "rouge1_f1": float(np.mean(r1)),
        "rouge2_f1": float(np.mean(r2)),
        "rougeL_f1": float(np.mean(rl)),
        "semantic_sim": float(np.mean(cosine_sims))
    }


def run_evaluation(limit: int = None, force_refresh: bool = False, use_bertscore: bool = False):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("\n" + "!" * 70)
        print("CRITICAL ERROR: GROQ_API_KEY is NOT set!")
        print(f"Evaluation requires real inference on {MODEL_ID}. MOCK output is strictly forbidden.")
        print("Set your Groq API key before evaluating: export GROQ_API_KEY='your_key'")
        print("!" * 70 + "\n")
        sys.exit(1)

    print(f"Connecting to Groq API ({MODEL_ID})...")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    check_no_leakage(test_records)

    if limit:
        test_records = test_records[:limit]
        print(f"Limiting evaluation to first {limit} test samples.")

    rag_cache = {} if force_refresh else load_cache(RAG_CACHE_FILE)
    zeroshot_cache = {} if force_refresh else load_cache(ZEROSHOT_CACHE_FILE)

    print(f"\n--- Generating Responses ({len(test_records)} items) ---")
    for i, r in enumerate(test_records):
        inc_id = r["incident_id"]
        print(f"[{i+1}/{len(test_records)}] Incident: {inc_id} (Quality: {r.get('quality', 'high')})")

        # 1. RAG Generation
        if inc_id not in rag_cache:
            print("  Fetching RAG generation...")
            res = generate_rag(r, groq_api_key, use_cache=not force_refresh)
            assert res["is_mock"] is False, "Mock generation detected in RAG!"
            rag_cache[inc_id] = res
            save_cache(rag_cache, RAG_CACHE_FILE)
        else:
            print("  [RAG] Loaded from cache")

        # 2. Zero-Shot Generation
        if inc_id not in zeroshot_cache:
            print("  Fetching Zero-Shot generation...")
            res = generate_zeroshot(r, groq_api_key, use_cache=not force_refresh)
            assert res["is_mock"] is False, "Mock generation detected in Zero-Shot!"
            zeroshot_cache[inc_id] = res
            save_cache(zeroshot_cache, ZEROSHOT_CACHE_FILE)
        else:
            print("  [Zero-Shot] Loaded from cache")

    # Evaluate subsets: Full, Quality:High, Quality:Low
    subsets = {
        "Full Set": test_records,
        "Quality: High Only": [r for r in test_records if r.get("quality") == "high"],
        "Quality: Low Only": [r for r in test_records if r.get("quality") == "low"],
    }

    results = {}
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (RAG vs. ZERO-SHOT ABLATION)")
    print("=" * 80)

    for subset_name, subset_records in subsets.items():
        if not subset_records:
            continue
        print(f"\n### {subset_name} (N = {len(subset_records)})")

        ref_rc = [r["root_cause"] for r in subset_records]
        rag_pred_rc = [rag_cache[r["incident_id"]]["root_cause"] for r in subset_records]
        zs_pred_rc = [zeroshot_cache[r["incident_id"]]["root_cause"] for r in subset_records]

        rag_metrics = compute_metrics(rag_pred_rc, ref_rc)
        zs_metrics = compute_metrics(zs_pred_rc, ref_rc)

        results[subset_name] = {
            "sample_size": len(subset_records),
            "RAG": rag_metrics,
            "ZeroShot": zs_metrics,
        }

        # Format comparison table
        print(f"{'Metric':<20} | {'Zero-Shot (No RAG)':<20} | {'RAG (TraceMind)':<20} | {'Delta':<10}")
        print("-" * 75)
        for metric in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "semantic_sim"]:
            zs_val = zs_metrics[metric]
            rag_val = rag_metrics[metric]
            diff = rag_val - zs_val
            pct = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            print(f"{metric:<20} | {zs_val:<20.4f} | {rag_val:<20.4f} | {pct:<10}")

    # Optional BERTScore
    if use_bertscore:
        from bert_score import score as calc_bertscore
        print("\nComputing BERTScore (roberta-large)...")
        all_refs = [r["root_cause"] for r in test_records]
        all_rag = [rag_cache[r["incident_id"]]["root_cause"] for r in test_records]
        all_zs = [zeroshot_cache[r["incident_id"]]["root_cause"] for r in test_records]
        _, _, f1_rag = calc_bertscore(all_rag, all_refs, lang="en", verbose=False)
        _, _, f1_zs = calc_bertscore(all_zs, all_refs, lang="en", verbose=False)
        print(f"BERTScore F1 -> Zero-Shot: {f1_zs.mean().item():.4f} | RAG: {f1_rag.mean().item():.4f}")
        results["bertscore"] = {
            "zero_shot_f1": float(f1_zs.mean().item()),
            "rag_f1": float(f1_rag.mean().item())
        }

    # Similarity floor analysis on test set
    sim_scores = [rag_cache[r["incident_id"]]["similarity_score"] for r in test_records]
    above_floor = sum(1 for s in sim_scores if s >= 0.40)
    print(f"\n--- Retrieval Similarity Distribution (N = {len(test_records)}) ---")
    print(f"Mean Cosine Similarity: {np.mean(sim_scores):.4f}")
    print(f"Median Cosine Similarity: {np.median(sim_scores):.4f}")
    print(f"Hits above 0.40 floor: {above_floor}/{len(test_records)} ({above_floor/len(test_records)*100:.1f}%)")

    results["retrieval_analysis"] = {
        "mean_similarity": float(np.mean(sim_scores)),
        "median_similarity": float(np.median(sim_scores)),
        "above_0.40_floor_count": above_floor,
        "above_0.40_floor_pct": round(above_floor / len(test_records) * 100, 1),
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull evaluation report saved to {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TraceMind RAG vs Zero-Shot")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test records to evaluate")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and regenerate all completions")
    parser.add_argument("--use-bertscore", action="store_true", help="Calculate BERTScore in addition to ROUGE")
    args = parser.parse_args()

    run_evaluation(limit=args.limit, force_refresh=args.force_refresh, use_bertscore=args.use_bertscore)
