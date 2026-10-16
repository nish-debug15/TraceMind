"""
Prompt Robustness Ablation Suite
--------------------------------
Tests output sensitivity of the generation model across 3 distinct prompt formulations:
1. Standard SRE Prompt (default in TraceMind)
2. Minimal / Direct Context Prompt (no role persona)
3. Step-by-Step Diagnostic Prompt (Chain-of-Thought style)

Usage:
    python eval/prompt_ablation.py [--limit N]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import requests
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rouge_score import rouge_scorer
from clustering.cluster import embed
from rag.generate import retrieve, RateLimiter, LLM_MAX_RPM, GROQ_API_URL
from rag.gen_cache import get_cached_generation, set_cached_generation

TEST_FILE = Path(__file__).parent.parent / "data" / "test.json"
CACHE_FILE = Path(__file__).parent / "cache" / "prompt_ablation_cache.json"
MODEL_ID = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM)

PROMPT_VARIANTS = {
    "variant_1_standard": (
        "You are an SRE assistant analyzing infrastructure incidents. Use the following historical incident to understand "
        "the context and format, then analyze the New Log Input and provide a precise Root Cause Analysis and actionable Remediation Steps.\n\n"
        "Historical Incident:\n- Root Cause: {root_cause}\n- Remediation: {remediation_steps}\n\n"
        "New Log Input:\n{log_text}\n\n"
        "Provide your response in the following format:\nROOT CAUSE:\n(your analysis)\n\nREMEDIATION STEPS:\n(your steps)"
    ),
    "variant_2_minimal": (
        "Context postmortem:\nCause: {root_cause}\nFix: {remediation_steps}\n\n"
        "Given this error log:\n{log_text}\n\n"
        "Output:\nROOT CAUSE:\n(analysis)\n\nREMEDIATION STEPS:\n(steps)"
    ),
    "variant_3_diagnostic_cot": (
        "You are a Principal Reliability Architect. Review the historical incident:\n"
        "- Prior Root Cause: {root_cause}\n"
        "- Prior Remediation: {remediation_steps}\n\n"
        "Analyze the following incoming log systematically:\n{log_text}\n\n"
        "Step 1: Identify the failing component.\n"
        "Step 2: Synthesize findings into the final response:\n"
        "ROOT CAUSE:\n(your analysis)\n\nREMEDIATION STEPS:\n(your steps)"
    ),
}


def call_groq_llm(prompt: str, api_key: str, max_retries: int = 4) -> str:
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
                        {"role": "system", "content": "You are an expert SRE assistant analyzing infrastructure incidents."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500,
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


def run_prompt_ablation(limit: int = 20):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("CRITICAL: GROQ_API_KEY is required for prompt ablation evaluation.")
        sys.exit(1)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)[:limit]

    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    variant_scores = {name: {"r1": [], "rL": [], "sim": []} for name in PROMPT_VARIANTS}

    print(f"Running Prompt Robustness Ablation across {len(records)} test samples using {MODEL_ID}...")
    for i, r in enumerate(records):
        inc_id = r["incident_id"]
        log_text = r["raw_log_excerpt"]
        match_meta, _ = retrieve(log_text, k=1)
        if not match_meta:
            continue

        cache.setdefault(inc_id, {})

        for name, template in PROMPT_VARIANTS.items():
            prompt = template.format(
                root_cause=match_meta["root_cause"],
                remediation_steps=match_meta["remediation_steps"],
                log_text=log_text
            )

            # Check SQLite gen_cache first
            cached_item = get_cached_generation(model=MODEL_ID, prompt_variant=name, query_text_or_id=log_text)
            if cached_item:
                resp = cached_item["raw_output"] or cached_item["root_cause"]
                cache[inc_id][name] = resp
            elif name in cache[inc_id]:
                resp = cache[inc_id][name]
            else:
                print(f"  [{i+1}/{len(records)}] Generating {name} for {inc_id}...")
                resp = call_groq_llm(prompt, groq_api_key)
                cache[inc_id][name] = resp
                set_cached_generation(
                    model=MODEL_ID,
                    prompt_variant=name,
                    query_text_or_id=log_text,
                    root_cause=resp,
                    remediation_steps="",
                    raw_output=resp
                )
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)

            # Score against ground truth
            scores = scorer.score(r["root_cause"], resp)
            v_pred = np.array(embed(resp[:1000]))
            v_ref = np.array(embed(r["root_cause"][:1000]))

            variant_scores[name]["r1"].append(scores["rouge1"].fmeasure)
            variant_scores[name]["rL"].append(scores["rougeL"].fmeasure)
            variant_scores[name]["sim"].append(float(v_pred @ v_ref))

    print("\n--- Prompt Robustness Results ---")
    print(f"{'Variant':<28} | {'ROUGE-1 F1':<12} | {'ROUGE-L F1':<12} | {'Cosine Sim':<12}")
    print("-" * 72)
    for name, s in variant_scores.items():
        print(f"{name:<28} | {np.mean(s['r1']):<12.4f} | {np.mean(s['rL']):<12.4f} | {np.mean(s['sim']):<12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    run_prompt_ablation(limit=args.limit)
