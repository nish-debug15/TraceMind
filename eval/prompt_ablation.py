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
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import InferenceClient
from rouge_score import rouge_scorer
import numpy as np

from clustering.cluster import embed
from rag.generate import retrieve

TEST_FILE = Path(__file__).parent.parent / "data" / "test.json"
CACHE_FILE = Path(__file__).parent / "cache" / "prompt_ablation_cache.json"
MODEL_ID = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

PROMPT_VARIANTS = {
    "variant_1_standard": (
        "You are an SRE assistant analyzing infrastructure incidents. Use the following historical incident to understand "
        "the context and format, then analyze the New Log Input and provide a precise Root Cause Analysis and actionable Remediation Steps.\n\n"
        "Historical Incident:\n- Root Cause: {root_cause}\n- Remediation: {remediation_steps}\n\n"
        "New Log Input:\n{log_text}\n\n"
        "Provide your response in the following format:\nRoot Cause: (your analysis)\nRemediation Steps: (your steps)"
    ),
    "variant_2_minimal": (
        "Context postmortem:\nCause: {root_cause}\nFix: {remediation_steps}\n\n"
        "Given this error log:\n{log_text}\n\n"
        "Output:\nRoot Cause: (analysis)\nRemediation Steps: (steps)"
    ),
    "variant_3_diagnostic_cot": (
        "You are a Principal Reliability Architect. Review the historical incident:\n"
        "- Prior Root Cause: {root_cause}\n"
        "- Prior Remediation: {remediation_steps}\n\n"
        "Analyze the following incoming log systematically:\n{log_text}\n\n"
        "Step 1: Identify the failing component.\n"
        "Step 2: Synthesize findings into the final response:\n"
        "Root Cause: (your analysis)\nRemediation Steps: (your steps)"
    ),
}

def run_prompt_ablation(limit: int = 20):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("CRITICAL: HF_TOKEN is required for prompt ablation evaluation.")
        sys.exit(1)

    client = InferenceClient(token=hf_token)
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)[:limit]

    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    variant_scores = {name: {"r1": [], "rL": [], "sim": []} for name in PROMPT_VARIANTS}

    print(f"Running Prompt Robustness Ablation across {len(records)} test samples...")
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
            if name not in cache[inc_id]:
                print(f"  [{i+1}/{len(records)}] Generating {name} for {inc_id}...")
                resp = client.text_generation(prompt, model=MODEL_ID, max_new_tokens=256, temperature=0.2)
                cache[inc_id][name] = resp
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            else:
                resp = cache[inc_id][name]

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
