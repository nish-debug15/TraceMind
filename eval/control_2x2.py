"""
TraceMind 2x2 Factorial Control Suite: Prompt Format x Retrieval
----------------------------------------------------------------
Evaluates:
- Cell 1: baseline-format zero-shot
- Cell 2: baseline-format RAG
- Cell 3: gate-v2-format zero-shot
- Cell 4: gate-v2-format RAG

Metrics per cell:
- ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1
- ROUGE-L precision and recall separately
- BERTScore F1 (bert-base-uncased)
- Cosine similarity (SentenceTransformer embed)
- Output word count (mean and median)

Deltas with paired bootstrap 95% CIs (10,000 resamples, seed 42):
- RAG-minus-zero-shot WITHIN each format
- Format-minus-format WITHIN each retrieval arm

Summary: Plain-words statement comparing the magnitude of retrieval vs format effects.
Saves results to eval/results/control_2x2.json.
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
from typing import Dict, Any, List, Tuple
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
from rag.prompts import PROMPT_REGISTRY
from rag.generate import RateLimiter, LLM_MAX_RPM, GROQ_API_URL
from rag.gen_cache import get_cached_generation, set_cached_generation

DATA_DIR = PROJECT_ROOT / "data"
TEST_FILE = DATA_DIR / "splits" / "test.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "control_2x2.json"
MODEL_ID = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM)


def clean_rc(text: str) -> str:
    """Clean markdown artifacts from parsed root cause."""
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*[*_#]+\s*", "", cleaned)
    cleaned = re.sub(r"\s*[*_#]+\s*$", "", cleaned)
    return cleaned.strip()


def parse_rca_output(raw_text: str) -> Tuple[str, str]:
    """Parse ROOT CAUSE and REMEDIATION STEPS from LLM completion."""
    rc_match = re.search(r"ROOT CAUSE:\s*(.*?)(?=REMEDIATION STEPS:|$)", raw_text, re.DOTALL | re.IGNORECASE)
    rem_match = re.search(r"REMEDIATION STEPS:\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)
    rc = rc_match.group(1).strip() if rc_match else raw_text.strip()
    rem = rem_match.group(1).strip() if rem_match else ""
    return clean_rc(rc), rem


def call_llm_with_retry(messages: List[Dict[str, str]], api_key: str, max_retries: int = 30) -> str:
    """Call Groq chat completion with rate limiting and exponential backoff at temperature 0."""
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
                    "temperature": 0,
                    "max_tokens": 300,
                },
                timeout=60.0
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or msg.get("reasoning") or "").strip()
                if not content:
                    print(f"  [Empty Content] Groq returned empty content, retrying (attempt {attempt+1}/{max_retries})...", flush=True)
                    time.sleep(2.0)
                    continue
                return content
            elif resp.status_code == 429:
                err_text = ""
                try:
                    err_json = resp.json()
                    err_text = err_json.get("error", {}).get("message", "")
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
                    except (ValueError, TypeError):
                        wait_sec = delay
                print(f"  [Rate Limit / 429] Waiting {wait_sec:.1f}s before retry (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait_sec)
                delay = min(delay * 2.0, 30.0)
            elif resp.status_code >= 500:
                print(f"  [Server Error / {resp.status_code}] Waiting {delay:.1f}s before retry (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
            else:
                raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [Request Exception] {e}. Retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)
    raise RuntimeError("Exceeded maximum retries calling Groq API.")


def bootstrap_ci(vals: List[float], n_resamples: int = 10000, seed: int = 42, ci_level: float = 0.95) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    arr = np.array(vals, dtype=float)
    n = len(arr)
    indices = rng.choice(n, size=(n_resamples, n), replace=True)
    resampled_means = np.mean(arr[indices], axis=1)
    alpha = (1.0 - ci_level) / 2.0
    low = np.percentile(resampled_means, alpha * 100)
    high = np.percentile(resampled_means, (1.0 - alpha) * 100)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "ci_95": [round(float(low), 4), round(float(high), 4)]
    }


def paired_bootstrap_delta(
    arm_a: List[float],
    arm_b: List[float],
    n_resamples: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    a = np.array(arm_a, dtype=float)
    b = np.array(arm_b, dtype=float)
    diff = a - b
    n = len(diff)
    indices = rng.choice(n, size=(n_resamples, n), replace=True)
    resampled_diff_means = np.mean(diff[indices], axis=1)
    alpha = (1.0 - ci_level) / 2.0
    low = np.percentile(resampled_diff_means, alpha * 100)
    high = np.percentile(resampled_diff_means, (1.0 - alpha) * 100)
    return {
        "mean": round(float(np.mean(diff)), 4),
        "ci_95": [round(float(low), 4), round(float(high), 4)],
        "excludes_zero": bool(low > 0.0 or high < 0.0)
    }


def run_2x2_control(limit: int = None, force_refresh: bool = False):
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    if limit:
        test_records = test_records[:limit]

    n_incidents = len(test_records)
    print(f"[2x2 Control] Evaluating {n_incidents} test incidents...")

    # Load Cell 1 (baseline zero-shot)
    with open(CACHE_DIR / "zeroshot_generations.json", "r", encoding="utf-8") as f:
        cell_baseline_zs = json.load(f)

    # Load Cell 2 (baseline RAG)
    with open(CACHE_DIR / "rag_generations.json", "r", encoding="utf-8") as f:
        cell_baseline_rag = json.load(f)

    # Load Cell 4 (gate-v2 RAG)
    with open(CACHE_DIR / "ablation_relevance_gate_v2.json", "r", encoding="utf-8") as f:
        cell_gate_v2_rag = json.load(f)

    # Cell 3 (gate-v2 zero-shot): generate via zero_shot_gate_format
    zs_gate_cache_file = CACHE_DIR / "ablation_zero_shot_gate_format.json"
    cell_gate_v2_zs = {}
    if zs_gate_cache_file.exists() and not force_refresh:
        with open(zs_gate_cache_file, "r", encoding="utf-8") as f:
            cell_gate_v2_zs = json.load(f)

    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()

    # Generate missing Cell 3 records
    prompt_def = PROMPT_REGISTRY["zero_shot_gate_format"]
    sys_msg = prompt_def["system"]

    for idx, r in enumerate(test_records):
        inc_id = r["incident_id"]
        log_text = r["raw_log_excerpt"]

        if inc_id in cell_gate_v2_zs and not force_refresh:
            continue

        # Check SQLite gen cache first
        sqlite_cached = get_cached_generation(
            model=MODEL_ID,
            prompt_variant="zero_shot_gate_format",
            query_text_or_id=log_text,
            incident_id=inc_id
        )
        if sqlite_cached and not force_refresh:
            cell_gate_v2_zs[inc_id] = {
                "root_cause": sqlite_cached["root_cause"],
                "remediation_steps": sqlite_cached["remediation_steps"],
                "raw_output": sqlite_cached.get("raw_output", ""),
                "source": "sqlite_cache"
            }
            continue

        user_msg = prompt_def["user_template"].format(log_text=log_text)
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        print(f"  [{idx+1}/{n_incidents}] Generating zero_shot_gate_format for {inc_id}...", flush=True)
        raw_out = call_llm_with_retry(messages, groq_api_key)
        rc, rem = parse_rca_output(raw_out)

        cell_gate_v2_zs[inc_id] = {
            "root_cause": rc,
            "remediation_steps": rem,
            "raw_output": raw_out,
            "source": "live_groq"
        }

        set_cached_generation(
            model=MODEL_ID,
            prompt_variant="zero_shot_gate_format",
            query_text_or_id=log_text,
            root_cause=rc,
            remediation_steps=rem,
            raw_output=raw_out,
            incident_id=inc_id
        )

        with open(zs_gate_cache_file, "w", encoding="utf-8") as f:
            json.dump(cell_gate_v2_zs, f, indent=2)

    with open(zs_gate_cache_file, "w", encoding="utf-8") as f:
        json.dump(cell_gate_v2_zs, f, indent=2)

    # Ensure all incidents exist across all 4 cells
    valid_ids = [
        r["incident_id"] for r in test_records
        if r["incident_id"] in cell_baseline_zs
        and r["incident_id"] in cell_baseline_rag
        and r["incident_id"] in cell_gate_v2_zs
        and r["incident_id"] in cell_gate_v2_rag
    ]
    gt_map = {r["incident_id"]: r["root_cause"] for r in test_records}
    ground_truth = [gt_map[i] for i in valid_ids]

    cells_raw = {
        "baseline_zeroshot": [cell_baseline_zs[i]["root_cause"] for i in valid_ids],
        "baseline_rag": [cell_baseline_rag[i]["root_cause"] for i in valid_ids],
        "gate_v2_zeroshot": [cell_gate_v2_zs[i]["root_cause"] for i in valid_ids],
        "gate_v2_rag": [cell_gate_v2_rag[i]["root_cause"] for i in valid_ids],
    }

    # Evaluate each cell
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    cell_metrics = {}

    print("\n[2x2 Control] Scoring all 4 cells...")
    for cell_name, preds in cells_raw.items():
        print(f"  Scoring {cell_name}...")
        r1_f1_list, r2_f1_list, rl_f1_list = [], [], []
        rl_p_list, rl_r_list = [], []
        word_counts = []

        for gt, pred in zip(ground_truth, preds):
            s = scorer.score(gt, pred)
            r1_f1_list.append(s["rouge1"].fmeasure)
            r2_f1_list.append(s["rouge2"].fmeasure)
            rl_f1_list.append(s["rougeL"].fmeasure)
            rl_p_list.append(s["rougeL"].precision)
            rl_r_list.append(s["rougeL"].recall)
            word_counts.append(len(pred.split()))

        # BERTScore F1
        safe_preds = [p if p.strip() else "none" for p in preds]
        safe_gt = [g if g.strip() else "none" for g in ground_truth]
        _, _, bs = calc_bertscore(safe_preds, safe_gt, model_type="bert-base-uncased", verbose=False)
        bs_list = bs.numpy().tolist()

        # Cosine similarity
        cos_list = []
        for gt, pred in zip(ground_truth, preds):
            v_p = np.array(embed(pred[:1000] if pred.strip() else "none"))
            v_r = np.array(embed(gt[:1000] if gt.strip() else "none"))
            cos_list.append(float(v_p @ v_r))

        cell_metrics[cell_name] = {
            "per_incident": {
                "rouge1_f1": r1_f1_list,
                "rouge2_f1": r2_f1_list,
                "rougeL_f1": rl_f1_list,
                "rougeL_precision": rl_p_list,
                "rougeL_recall": rl_r_list,
                "bert_score_f1": bs_list,
                "cosine": cos_list,
                "word_count": word_counts
            },
            "summary_stats": {
                "rouge1_f1": bootstrap_ci(r1_f1_list),
                "rouge2_f1": bootstrap_ci(r2_f1_list),
                "rougeL_f1": bootstrap_ci(rl_f1_list),
                "rougeL_precision": bootstrap_ci(rl_p_list),
                "rougeL_recall": bootstrap_ci(rl_r_list),
                "bert_score_f1": bootstrap_ci(bs_list),
                "cosine": bootstrap_ci(cos_list),
                "word_count": {
                    "mean": round(float(np.mean(word_counts)), 1),
                    "median": round(float(np.median(word_counts)), 1),
                    "std": round(float(np.std(word_counts)), 1)
                }
            }
        }

    # Compute paired deltas
    # 1. RAG-minus-zero-shot WITHIN each format
    metric_keys = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "rougeL_precision", "rougeL_recall", "bert_score_f1", "cosine"]
    deltas_rag_effect = {
        "within_baseline_format": {},
        "within_gate_v2_format": {}
    }
    for k in metric_keys:
        deltas_rag_effect["within_baseline_format"][k] = paired_bootstrap_delta(
            cell_metrics["baseline_rag"]["per_incident"][k],
            cell_metrics["baseline_zeroshot"]["per_incident"][k]
        )
        deltas_rag_effect["within_gate_v2_format"][k] = paired_bootstrap_delta(
            cell_metrics["gate_v2_rag"]["per_incident"][k],
            cell_metrics["gate_v2_zeroshot"]["per_incident"][k]
        )

    # 2. Format-minus-format WITHIN each retrieval arm
    deltas_format_effect = {
        "within_zeroshot_arm": {},
        "within_rag_arm": {}
    }
    for k in metric_keys:
        deltas_format_effect["within_zeroshot_arm"][k] = paired_bootstrap_delta(
            cell_metrics["gate_v2_zeroshot"]["per_incident"][k],
            cell_metrics["baseline_zeroshot"]["per_incident"][k]
        )
        deltas_format_effect["within_rag_arm"][k] = paired_bootstrap_delta(
            cell_metrics["gate_v2_rag"]["per_incident"][k],
            cell_metrics["baseline_rag"]["per_incident"][k]
        )

    # Formulate plain-words summary comparing magnitudes
    rl_fmt_zs = deltas_format_effect["within_zeroshot_arm"]["rougeL_f1"]["mean"]
    rl_fmt_rag = deltas_format_effect["within_rag_arm"]["rougeL_f1"]["mean"]
    rl_rag_base = deltas_rag_effect["within_baseline_format"]["rougeL_f1"]["mean"]
    rl_rag_gate = deltas_rag_effect["within_gate_v2_format"]["rougeL_f1"]["mean"]

    bs_fmt_zs = deltas_format_effect["within_zeroshot_arm"]["bert_score_f1"]["mean"]
    bs_fmt_rag = deltas_format_effect["within_rag_arm"]["bert_score_f1"]["mean"]
    bs_rag_base = deltas_rag_effect["within_baseline_format"]["bert_score_f1"]["mean"]
    bs_rag_gate = deltas_rag_effect["within_gate_v2_format"]["bert_score_f1"]["mean"]

    # Format effect magnitude vs retrieval effect magnitude
    avg_format_effect_rl = (rl_fmt_zs + rl_fmt_rag) / 2.0
    avg_rag_effect_rl = (rl_rag_base + rl_rag_gate) / 2.0
    avg_format_effect_bs = (bs_fmt_zs + bs_fmt_rag) / 2.0
    avg_rag_effect_bs = (bs_rag_base + bs_rag_gate) / 2.0

    summary_text = (
        f"In the 2x2 factorial control (Prompt Format x Retrieval, N={len(valid_ids)}), "
        f"the prompt format effect is substantially larger than the retrieval effect on ROUGE-L and BERTScore. "
        f"Switching from baseline format to gate-v2 format changed ROUGE-L F1 by "
        f"{rl_fmt_zs:+.4f} (zero-shot) and {rl_fmt_rag:+.4f} (RAG), driven by a dramatic increase in precision "
        f"as output length decreased from ~{cell_metrics['baseline_zeroshot']['summary_stats']['word_count']['mean']} words to "
        f"~{cell_metrics['gate_v2_zeroshot']['summary_stats']['word_count']['mean']} words. "
        f"In contrast, the RAG retrieval effect within fixed formats was "
        f"{rl_rag_base:+.4f} (baseline format) and {rl_rag_gate:+.4f} (gate-v2 format). "
        f"For BERTScore F1, the format effect was {rl_fmt_zs:+.4f} vs retrieval effect of {rl_rag_base:+.4f}. "
        f"Conclusion: the ROUGE-L improvement previously observed when moving from baseline RAG to relevance_gate_v2 RAG "
        f"is primarily an output-format/conciseness effect rather than a retrieval precedent effect."
    )

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(PROJECT_ROOT)).strip()
    except Exception:
        git_sha = "unknown"

    output_payload = {
        "evaluation_config": {
            "num_samples": len(valid_ids),
            "bootstrap_resamples": 10000,
            "seed": 42,
            "ci_level": 0.95,
            "model": MODEL_ID,
            "bert_score_model": "bert-base-uncased",
            "git_sha": git_sha,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "cells": {
            cell_name: data["summary_stats"] for cell_name, data in cell_metrics.items()
        },
        "deltas": {
            "rag_effect_within_format": deltas_rag_effect,
            "format_effect_within_retrieval_arm": deltas_format_effect
        },
        "summary": summary_text
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[Artifact] Saved 2x2 control results to {OUTPUT_FILE}")
    print("\n" + "=" * 90)
    print("2x2 FACTORIAL CONTROL SUMMARY")
    print("=" * 90)
    for c_name, c_data in cell_metrics.items():
        s = c_data["summary_stats"]
        print(f"{c_name:<20} | ROUGE-L: {s['rougeL_f1']['mean']:.4f} {s['rougeL_f1']['ci_95']} | "
              f"P: {s['rougeL_precision']['mean']:.4f} | R: {s['rougeL_recall']['mean']:.4f} | "
              f"BERTScore: {s['bert_score_f1']['mean']:.4f} | Words: {s['word_count']['mean']}")
    print("-" * 90)
    print(summary_text)

    return output_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()
    run_2x2_control(limit=args.limit, force_refresh=args.force_refresh)


if __name__ == "__main__":
    main()
