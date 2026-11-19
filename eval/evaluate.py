"""
TraceMind Evaluation Suite
--------------------------
Evaluates Root Cause Analysis (RCA) on held-out test postmortems (zero leakage).
Compares RAG-augmented generation vs. Zero-Shot (No-RAG) baseline against the TRAIN-ONLY index.

Requirements:
- NO synthetic data. Evaluates strictly on data/splits/test.json (93 incidents).
- Evaluates against TRAIN-ONLY ChromaDB index (rag/chroma_db_train, collection postmortems_train).
- Zero data leakage verified before execution.
- Generations cached to disk (rag/gen_cache.py and eval/cache/).
- Results saved to eval/results/rag_vs_zero_shot.json.

Usage:
    python eval/evaluate.py [--limit N] [--force-refresh] [--use-bertscore]
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
from typing import Dict, List, Any
import numpy as np
import requests
from dotenv import load_dotenv

# Force UTF-8 encoding for stdout on Windows
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load local environment variables if available
load_dotenv()

from rouge_score import rouge_scorer
import chromadb

from clustering.cluster import embed
from rag.generate import retrieve, RateLimiter, LLM_MAX_RPM, GROQ_API_URL
from rag.gen_cache import get_cached_generation, set_cached_generation

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_FILE = DATA_DIR / "splits" / "test.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CACHE_DIR = EVAL_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
RAG_CACHE_FILE = CACHE_DIR / "rag_generations.json"
ZEROSHOT_CACHE_FILE = CACHE_DIR / "zeroshot_generations.json"
RESULTS_FILE = EVAL_DIR / "evaluation_report.json"
RAG_VS_ZEROSHOT_FILE = RESULTS_DIR / "rag_vs_zero_shot.json"

TRAIN_CHROMA_PATH = PROJECT_ROOT / "rag" / "chroma_db_train"
TRAIN_COLLECTION_NAME = "postmortems_train"

MODEL_ID = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM)


def check_no_leakage(test_records: List[Dict[str, Any]]):
    """Assert that zero test incidents exist in the ChromaDB train retrieval index."""
    if not TRAIN_CHROMA_PATH.exists():
        raise RuntimeError(f"Train index not found at {TRAIN_CHROMA_PATH}. Run scripts/verify_no_leakage.py first.")
    
    client = chromadb.PersistentClient(path=str(TRAIN_CHROMA_PATH))
    collection = client.get_collection(name=TRAIN_COLLECTION_NAME)
    all_chroma_ids = set(collection.get()["ids"])
    test_ids = {r["incident_id"] for r in test_records}
    overlap = test_ids.intersection(all_chroma_ids)
    if overlap:
        raise RuntimeError(
            f"CRITICAL LEAKAGE ERROR: {len(overlap)} test incidents found inside the retrieval index! "
            f"Overlap sample: {list(overlap)[:5]}."
        )
    print(f"[Verification] Train index ({TRAIN_COLLECTION_NAME}) contains {collection.count()} records. Test leakage: 0 records.")


def call_llm_with_retry(prompt: str, api_key: str, max_retries: int = 4) -> str:
    """Execute LLM chat completion via Groq with rate limiting and exponential backoff."""
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
                    "messages": [
                        {"role": "system", "content": "You are an expert SRE assistant analyzing infrastructure incidents. Structure your output into ROOT CAUSE: and REMEDIATION STEPS: sections."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 800,
                },
                timeout=60.0
            )
            if resp.status_code == 200:
                data = resp.json()
                raw_msg = data["choices"][0]["message"].get("content") or ""
                cleaned = re.sub(r"<think>.*?</think>", "", raw_msg, flags=re.DOTALL).strip()
                if not cleaned:
                    cleaned = raw_msg.strip()
                if not cleaned or "[MOCK]" in cleaned:
                    if attempt < max_retries - 1:
                        time.sleep(3.0)
                        continue
                    raise RuntimeError("Received mock or empty response from Groq API.")
                return cleaned
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


def generate_rag(record: Dict[str, Any], api_key: str, use_cache: bool = True) -> tuple[Dict[str, Any], bool]:
    """Returns (result_dict, is_cache_hit)."""
    log_text = record["raw_log_excerpt"]
    match_meta, similarity = retrieve(
        log_text,
        k=1,
        persist_directory=str(TRAIN_CHROMA_PATH),
        collection_name=TRAIN_COLLECTION_NAME,
    )
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
            }, True

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
    }, False


def generate_zeroshot(record: Dict[str, Any], api_key: str, use_cache: bool = True) -> tuple[Dict[str, Any], bool]:
    """Returns (result_dict, is_cache_hit)."""
    log_text = record["raw_log_excerpt"]
    if use_cache:
        cached = get_cached_generation(model=MODEL_ID, prompt_variant="zeroshot", query_text_or_id=log_text)
        if cached:
            return {
                "root_cause": cached["root_cause"],
                "remediation_steps": cached["remediation_steps"],
                "prompt": "(cached)",
                "is_mock": False
            }, True

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
    }, False


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, median, and sample standard deviation."""
    arr = np.array(values, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4)
    }


def run_evaluation(limit: int = None, force_refresh: bool = False, use_bertscore: bool = False):
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
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

    # Hard leakage check against train index
    check_no_leakage(test_records)

    if limit:
        test_records = test_records[:limit]
        print(f"Limiting evaluation to first {limit} test samples.")

    rag_cache = {} if force_refresh else load_cache(RAG_CACHE_FILE)
    zeroshot_cache = {} if force_refresh else load_cache(ZEROSHOT_CACHE_FILE)

    rag_cache_hits = 0
    rag_live_calls = 0
    zs_cache_hits = 0
    zs_live_calls = 0

    print(f"\n--- Generating Responses ({len(test_records)} items) ---")
    for i, r in enumerate(test_records):
        inc_id = r["incident_id"]

        # 1. RAG Generation
        if inc_id not in rag_cache:
            res, hit = generate_rag(r, groq_api_key, use_cache=not force_refresh)
            assert res["is_mock"] is False, "Mock generation detected in RAG!"
            rag_cache[inc_id] = res
            save_cache(rag_cache, RAG_CACHE_FILE)
            if hit:
                rag_cache_hits += 1
            else:
                rag_live_calls += 1
        else:
            rag_cache_hits += 1

        # 2. Zero-Shot Generation
        if inc_id not in zeroshot_cache:
            res, hit = generate_zeroshot(r, groq_api_key, use_cache=not force_refresh)
            assert res["is_mock"] is False, "Mock generation detected in Zero-Shot!"
            zeroshot_cache[inc_id] = res
            save_cache(zeroshot_cache, ZEROSHOT_CACHE_FILE)
            if hit:
                zs_cache_hits += 1
            else:
                zs_live_calls += 1
        else:
            zs_cache_hits += 1

    print(f"Cache report -> RAG: {rag_cache_hits} hits / {rag_live_calls} live | ZeroShot: {zs_cache_hits} hits / {zs_live_calls} live")

    # Compute per-incident scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per_incident_scores = []

    print("\nComputing per-incident metrics (ROUGE-1, ROUGE-2, ROUGE-L, embedding cosine)...")
    for r in test_records:
        inc_id = r["incident_id"]
        ref_rc = r["root_cause"]
        rag_item = rag_cache[inc_id]
        zs_item = zeroshot_cache[inc_id]

        # ROUGE
        s_rag = scorer.score(ref_rc, rag_item["root_cause"])
        s_zs = scorer.score(ref_rc, zs_item["root_cause"])

        # Embeddings
        v_ref = np.array(embed(ref_rc))
        v_rag = np.array(embed(rag_item["root_cause"]))
        v_zs = np.array(embed(zs_item["root_cause"]))
        sim_rag = float(v_rag @ v_ref)
        sim_zs = float(v_zs @ v_ref)

        r1_rag = float(s_rag["rouge1"].fmeasure)
        r2_rag = float(s_rag["rouge2"].fmeasure)
        rl_rag = float(s_rag["rougeL"].fmeasure)

        r1_zs = float(s_zs["rouge1"].fmeasure)
        r2_zs = float(s_zs["rouge2"].fmeasure)
        rl_zs = float(s_zs["rougeL"].fmeasure)

        entry = {
            "incident_id": inc_id,
            "company": r.get("company", "unknown"),
            "quality": r.get("quality", "high"),
            "retrieval": {
                "matched_incident_id": rag_item.get("retrieved_incident_id"),
                "similarity_score": round(float(rag_item.get("similarity_score", 0.0)), 4),
            },
            "rag": {
                "root_cause": rag_item["root_cause"],
                "remediation_steps": rag_item["remediation_steps"],
                "rouge1_f1": round(r1_rag, 4),
                "rouge2_f1": round(r2_rag, 4),
                "rougeL_f1": round(rl_rag, 4),
                "semantic_sim": round(sim_rag, 4),
            },
            "zeroshot": {
                "root_cause": zs_item["root_cause"],
                "remediation_steps": zs_item["remediation_steps"],
                "rouge1_f1": round(r1_zs, 4),
                "rouge2_f1": round(r2_zs, 4),
                "rougeL_f1": round(rl_zs, 4),
                "semantic_sim": round(sim_zs, 4),
            },
            "delta": {
                "rouge1_f1": round(r1_rag - r1_zs, 4),
                "rouge2_f1": round(r2_rag - r2_zs, 4),
                "rougeL_f1": round(rl_rag - rl_zs, 4),
                "semantic_sim": round(sim_rag - sim_zs, 4),
            }
        }
        per_incident_scores.append(entry)

    # Evaluate subsets: Full Set, Quality: High Only, Quality: Low Only
    subsets = {
        "Full Set": per_incident_scores,
        "Quality: High Only": [entry for entry in per_incident_scores if entry.get("quality") == "high"],
        "Quality: Low Only": [entry for entry in per_incident_scores if entry.get("quality") == "low"],
    }

    aggregates = {}
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (RAG vs. ZERO-SHOT ABLATION)")
    print("=" * 80)

    for subset_name, subset_entries in subsets.items():
        if not subset_entries:
            continue
        print(f"\n### {subset_name} (N = {len(subset_entries)})")

        metric_keys = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "semantic_sim"]
        subset_stats = {
            "sample_size": len(subset_entries),
            "RAG": {},
            "ZeroShot": {},
            "Delta": {}
        }

        for k in metric_keys:
            rag_vals = [e["rag"][k] for e in subset_entries]
            zs_vals = [e["zeroshot"][k] for e in subset_entries]
            delta_vals = [e["delta"][k] for e in subset_entries]

            subset_stats["RAG"][k] = compute_distribution_stats(rag_vals)
            subset_stats["ZeroShot"][k] = compute_distribution_stats(zs_vals)
            subset_stats["Delta"][k] = compute_distribution_stats(delta_vals)

        aggregates[subset_name] = subset_stats

        # Print comparison table
        print(f"{'Metric':<18} | {'Zero-Shot (mean±std)':<22} | {'RAG (mean±std)':<22} | {'Delta':<10}")
        print("-" * 78)
        for k in metric_keys:
            zs_s = subset_stats["ZeroShot"][k]
            rag_s = subset_stats["RAG"][k]
            d_s = subset_stats["Delta"][k]
            diff = rag_s["mean"] - zs_s["mean"]
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            zs_str = f"{zs_s['mean']:.4f} ± {zs_s['std']:.4f}"
            rag_str = f"{rag_s['mean']:.4f} ± {rag_s['std']:.4f}"
            print(f"{k:<18} | {zs_str:<22} | {rag_str:<22} | {diff_str:<10}")

    # Optional BERTScore
    bertscore_results = None
    if use_bertscore:
        from bert_score import score as calc_bertscore
        print("\nComputing BERTScore (roberta-large)...")
        all_refs = [r["root_cause"] for r in test_records]
        all_rag = [e["rag"]["root_cause"] for e in per_incident_scores]
        all_zs = [e["zeroshot"]["root_cause"] for e in per_incident_scores]
        _, _, f1_rag = calc_bertscore(all_rag, all_refs, lang="en", verbose=False)
        _, _, f1_zs = calc_bertscore(all_zs, all_refs, lang="en", verbose=False)
        f1_rag_arr = f1_rag.numpy()
        f1_zs_arr = f1_zs.numpy()
        print(f"BERTScore F1 -> Zero-Shot: {f1_zs_arr.mean():.4f} | RAG: {f1_rag_arr.mean():.4f}")
        bertscore_results = {
            "zero_shot": compute_distribution_stats(f1_zs_arr.tolist()),
            "rag": compute_distribution_stats(f1_rag_arr.tolist()),
            "delta": compute_distribution_stats((f1_rag_arr - f1_zs_arr).tolist())
        }

    # Similarity floor analysis on test set
    sim_scores = [e["retrieval"]["similarity_score"] for e in per_incident_scores]
    above_floor = sum(1 for s in sim_scores if s >= 0.40)
    print(f"\n--- Retrieval Similarity Distribution (N = {len(test_records)}) ---")
    print(f"Mean Cosine Similarity: {np.mean(sim_scores):.4f}")
    print(f"Median Cosine Similarity: {np.median(sim_scores):.4f}")
    print(f"Hits above 0.40 floor: {above_floor}/{len(test_records)} ({above_floor/len(test_records)*100:.1f}%)")

    # Get git SHA
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(PROJECT_ROOT)).strip()
    except Exception:
        git_sha = "unknown"

    # Save frozen RAG vs Zero-Shot evaluation JSON
    frozen_output = {
        "config": {
            "model": MODEL_ID,
            "prompt_version": "baseline",
            "train_index_path": str(TRAIN_CHROMA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "train_collection_name": TRAIN_COLLECTION_NAME,
            "test_dataset": "data/splits/test.json",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_sha": git_sha,
            "seed": 42,
            "n": len(test_records),
            "cache_hits": {"rag": rag_cache_hits, "zeroshot": zs_cache_hits},
            "live_calls": {"rag": rag_live_calls, "zeroshot": zs_live_calls},
        },
        "aggregates": aggregates,
        "retrieval_analysis": {
            "mean_similarity": float(np.mean(sim_scores)),
            "median_similarity": float(np.median(sim_scores)),
            "above_0.40_floor_count": above_floor,
            "above_0.40_floor_pct": round(above_floor / len(test_records) * 100, 1),
        },
        "per_incident_scores": per_incident_scores,
        "verification": {
            "total_test_incidents": len(test_records),
            "total_evaluated": len(per_incident_scores),
            "zero_leakage_verified": True
        }
    }
    if bertscore_results:
        frozen_output["bertscore"] = bertscore_results

    with open(RAG_VS_ZEROSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(frozen_output, f, indent=2)
    print(f"\n[Artifact] Frozen RAG vs Zero-Shot results saved to {RAG_VS_ZEROSHOT_FILE}")

    # Legacy report format for backward compatibility
    legacy_results = {
        subset: {
            "sample_size": data["sample_size"],
            "RAG": {k: data["RAG"][k]["mean"] for k in data["RAG"]},
            "ZeroShot": {k: data["ZeroShot"][k]["mean"] for k in data["ZeroShot"]},
        }
        for subset, data in aggregates.items()
    }
    legacy_results["retrieval_analysis"] = frozen_output["retrieval_analysis"]
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(legacy_results, f, indent=2)
    print(f"[Artifact] Legacy evaluation report saved to {RESULTS_FILE}")

    return frozen_output


def main():
    parser = argparse.ArgumentParser(description="Evaluate TraceMind RAG vs Zero-Shot")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test records to evaluate")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and regenerate all completions")
    parser.add_argument("--use-bertscore", action="store_true", help="Calculate BERTScore in addition to ROUGE")
    args = parser.parse_args()

    run_evaluation(limit=args.limit, force_refresh=args.force_refresh, use_bertscore=args.use_bertscore)


if __name__ == "__main__":
    main()
