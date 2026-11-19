"""
eval/remediation_eval.py
------------------------
Evaluates the remediation_steps section across the 4 cells of the 2x2 factorial control:
1. baseline_zeroshot
2. baseline_rag
3. gate_v2_zeroshot
4. gate_v2_rag

Requirements:
- Parses generated remediation section using existing parser.
- Reports parse-failure count per cell.
- Excludes incidents whose ground-truth remediation is empty or under 5 words, reporting that count.
- Scores ROUGE-L F1 and BERTScore F1 (bert-base-uncased) per cell.
- Computes within-format RAG-minus-zero-shot deltas with company-clustered bootstrap 95% CIs.
- Verifies zero-shot gate-format contains remediation section.
- Saves results to eval/results/remediation_eval.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as calc_bertscore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.cluster_bootstrap import two_stage_cluster_bootstrap_ci

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "remediation_eval.json"

CELL_FILES = {
    "baseline_zeroshot": CACHE_DIR / "zeroshot_generations.json",
    "baseline_rag": CACHE_DIR / "rag_generations.json",
    "gate_v2_zeroshot": CACHE_DIR / "ablation_zero_shot_gate_format.json",
    "gate_v2_rag": CACHE_DIR / "ablation_relevance_gate_v2.json",
}


def run_remediation_eval() -> Dict[str, Any]:
    print("[Remediation Eval] Loading test records and ground truth remediation...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    # Check ground-truth remediation word counts and exclusions
    all_incidents = [r["incident_id"] for r in test_records]
    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in test_records}
    gt_rem_map = {r["incident_id"]: (r.get("remediation_steps") or "").strip() for r in test_records}

    excluded_ids = []
    eligible_ids = []
    for inc_id in all_incidents:
        gt_text = gt_rem_map[inc_id]
        wc = len(gt_text.split())
        if wc < 5:
            excluded_ids.append(inc_id)
        else:
            eligible_ids.append(inc_id)

    print(f"Total test incidents: {len(all_incidents)}")
    print(f"Excluded due to GT remediation < 5 words: {len(excluded_ids)}")
    print(f"Eligible incidents for scoring: {len(eligible_ids)}")

    # Load 4 cell cached generations
    cell_data = {}
    for cname, cpath in CELL_FILES.items():
        with open(cpath, "r", encoding="utf-8") as f:
            cell_data[cname] = json.load(f)

    # Parse and check parse failures
    parse_failures = {}
    cell_preds = {}
    for cname, cdict in cell_data.items():
        fails = []
        preds = []
        for inc_id in eligible_ids:
            item = cdict.get(inc_id, {})
            rem = item.get("remediation_steps", "").strip()
            if not rem:
                fails.append(inc_id)
                preds.append("")
            else:
                preds.append(rem)
        parse_failures[cname] = {
            "failure_count": len(fails),
            "total_eligible": len(eligible_ids),
            "coverage_rate": round(float((len(eligible_ids) - len(fails)) / len(eligible_ids)), 4),
            "failed_incident_ids": fails,
        }
        cell_preds[cname] = preds

    # Check stopping condition for gate-v2 zero-shot
    gz_fails = parse_failures["gate_v2_zeroshot"]["failure_count"]
    if gz_fails == len(eligible_ids):
        raise RuntimeError(
            "CRITICAL: Zero-shot gate-format outputs do not contain a remediation section at all (100% failure). STOPPING."
        )

    # Score ROUGE-L F1 and BERTScore F1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    ground_truth = [gt_rem_map[i] for i in eligible_ids]

    cell_scores = {}
    print("\n[Remediation Eval] Scoring all 4 cells on eligible test incidents (n=93)...")
    for cname, preds in cell_preds.items():
        print(f"  Scoring {cname}...")
        rl_list = []
        for gt, pred in zip(ground_truth, preds):
            if not pred.strip():
                rl_list.append(0.0)
            else:
                s = scorer.score(gt, pred)["rougeL"].fmeasure
                rl_list.append(s)

        # Sanitize for BERTScore: empty completions evaluated as 'none'
        safe_preds = [p if p.strip() else "none" for p in preds]
        safe_gt = [g if g.strip() else "none" for g in ground_truth]

        _, _, bs = calc_bertscore(safe_preds, safe_gt, model_type="bert-base-uncased", verbose=False)
        bs_list = bs.numpy().tolist()

        rl_ci = two_stage_cluster_bootstrap_ci(eligible_ids, rl_list, id_to_company, n_resamples=10000, seed=42)
        bs_ci = two_stage_cluster_bootstrap_ci(eligible_ids, bs_list, id_to_company, n_resamples=10000, seed=42)

        cell_scores[cname] = {
            "rougeL_f1": rl_ci,
            "bert_score_f1": bs_ci,
            "_raw_rl": rl_list,
            "_raw_bs": bs_list,
        }

    # Compute within-format RAG-minus-zero-shot deltas
    base_rl_diff = [
        r - z for r, z in zip(cell_scores["baseline_rag"]["_raw_rl"], cell_scores["baseline_zeroshot"]["_raw_rl"])
    ]
    base_bs_diff = [
        r - z for r, z in zip(cell_scores["baseline_rag"]["_raw_bs"], cell_scores["baseline_zeroshot"]["_raw_bs"])
    ]

    gate_rl_diff = [
        r - z for r, z in zip(cell_scores["gate_v2_rag"]["_raw_rl"], cell_scores["gate_v2_zeroshot"]["_raw_rl"])
    ]
    gate_bs_diff = [
        r - z for r, z in zip(cell_scores["gate_v2_rag"]["_raw_bs"], cell_scores["gate_v2_zeroshot"]["_raw_bs"])
    ]

    deltas = {
        "baseline_format_rag_effect": {
            "rougeL_delta": two_stage_cluster_bootstrap_ci(eligible_ids, base_rl_diff, id_to_company, n_resamples=10000, seed=42),
            "bert_score_delta": two_stage_cluster_bootstrap_ci(eligible_ids, base_bs_diff, id_to_company, n_resamples=10000, seed=42),
        },
        "gate_v2_format_rag_effect": {
            "rougeL_delta": two_stage_cluster_bootstrap_ci(eligible_ids, gate_rl_diff, id_to_company, n_resamples=10000, seed=42),
            "bert_score_delta": two_stage_cluster_bootstrap_ci(eligible_ids, gate_bs_diff, id_to_company, n_resamples=10000, seed=42),
        }
    }

    # Clean raw lists from output dict
    cleaned_cells = {}
    for cname, cdata in cell_scores.items():
        cleaned_cells[cname] = {
            "rougeL_f1": cdata["rougeL_f1"],
            "bert_score_f1": cdata["bert_score_f1"],
            "coverage_rate": parse_failures[cname]["coverage_rate"],
            "parse_failure_count": parse_failures[cname]["failure_count"],
        }

    summary_text = (
        f"Remediation section evaluation across 4 cells (N={len(eligible_ids)}, excluded GT < 5 words = {len(excluded_ids)}):\n"
        f"1. Parse Failures / Missing Section:\n"
        f"   - baseline_zeroshot: {parse_failures['baseline_zeroshot']['failure_count']}/93 ({parse_failures['baseline_zeroshot']['coverage_rate']*100:.1f}% coverage)\n"
        f"   - baseline_rag:      {parse_failures['baseline_rag']['failure_count']}/93 ({parse_failures['baseline_rag']['coverage_rate']*100:.1f}% coverage)\n"
        f"   - gate_v2_zeroshot:  {parse_failures['gate_v2_zeroshot']['failure_count']}/93 ({parse_failures['gate_v2_zeroshot']['coverage_rate']*100:.1f}% coverage)\n"
        f"   - gate_v2_rag:       {parse_failures['gate_v2_rag']['failure_count']}/93 ({parse_failures['gate_v2_rag']['coverage_rate']*100:.1f}% coverage)\n"
        f"2. Cell Scores (All {len(eligible_ids)} test incidents, clustered 95% CIs):\n"
        f"   - baseline_zeroshot: ROUGE-L={cleaned_cells['baseline_zeroshot']['rougeL_f1']['mean']} {cleaned_cells['baseline_zeroshot']['rougeL_f1']['ci_95']}, BERTScore={cleaned_cells['baseline_zeroshot']['bert_score_f1']['mean']} {cleaned_cells['baseline_zeroshot']['bert_score_f1']['ci_95']}\n"
        f"   - baseline_rag:      ROUGE-L={cleaned_cells['baseline_rag']['rougeL_f1']['mean']} {cleaned_cells['baseline_rag']['rougeL_f1']['ci_95']}, BERTScore={cleaned_cells['baseline_rag']['bert_score_f1']['mean']} {cleaned_cells['baseline_rag']['bert_score_f1']['ci_95']}\n"
        f"   - gate_v2_zeroshot:  ROUGE-L={cleaned_cells['gate_v2_zeroshot']['rougeL_f1']['mean']} {cleaned_cells['gate_v2_zeroshot']['rougeL_f1']['ci_95']}, BERTScore={cleaned_cells['gate_v2_zeroshot']['bert_score_f1']['mean']} {cleaned_cells['gate_v2_zeroshot']['bert_score_f1']['ci_95']}\n"
        f"   - gate_v2_rag:       ROUGE-L={cleaned_cells['gate_v2_rag']['rougeL_f1']['mean']} {cleaned_cells['gate_v2_rag']['rougeL_f1']['ci_95']}, BERTScore={cleaned_cells['gate_v2_rag']['bert_score_f1']['mean']} {cleaned_cells['gate_v2_rag']['bert_score_f1']['ci_95']}\n"
        f"3. Within-Format RAG-minus-Zero-Shot Deltas (Clustered 95% CIs):\n"
        f"   - Baseline Format: ROUGE-L delta = {deltas['baseline_format_rag_effect']['rougeL_delta']['mean']:+.4f} {deltas['baseline_format_rag_effect']['rougeL_delta']['ci_95']} (significant: {deltas['baseline_format_rag_effect']['rougeL_delta']['significant_under_cluster_bootstrap']}), "
        f"BERTScore delta = {deltas['baseline_format_rag_effect']['bert_score_delta']['mean']:+.4f} {deltas['baseline_format_rag_effect']['bert_score_delta']['ci_95']} (significant: {deltas['baseline_format_rag_effect']['bert_score_delta']['significant_under_cluster_bootstrap']})\n"
        f"   - Gate-v2 Format:  ROUGE-L delta = {deltas['gate_v2_format_rag_effect']['rougeL_delta']['mean']:+.4f} {deltas['gate_v2_format_rag_effect']['rougeL_delta']['ci_95']} (significant: {deltas['gate_v2_format_rag_effect']['rougeL_delta']['significant_under_cluster_bootstrap']}), "
        f"BERTScore delta = {deltas['gate_v2_format_rag_effect']['bert_score_delta']['mean']:+.4f} {deltas['gate_v2_format_rag_effect']['bert_score_delta']['ci_95']} (significant: {deltas['gate_v2_format_rag_effect']['bert_score_delta']['significant_under_cluster_bootstrap']})"
    )

    results_payload = {
        "metadata": {
            "total_test_incidents": len(all_incidents),
            "excluded_gt_under_5_words": len(excluded_ids),
            "eligible_incidents": len(eligible_ids),
            "num_unique_companies": len(set(id_to_company.values())),
            "bootstrap_resamples": 10000,
            "seed": 42,
            "ci_level": 0.95,
        },
        "parse_failures": parse_failures,
        "cells": cleaned_cells,
        "deltas": deltas,
        "summary": summary_text,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nSaved remediation evaluation to {OUTPUT_FILE}")
    print("\n" + summary_text)
    return results_payload


if __name__ == "__main__":
    run_remediation_eval()
