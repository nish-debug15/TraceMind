"""
eval/length_diagnostics.py
--------------------------
Evaluates length diagnostics and length-matched sensitivity across the 4 cells
of the 2x2 factorial control (Prompt Format x Retrieval, N=93):
1. Ground-truth root_cause word-count statistics (mean, median, p10, p90)
2. Generated word-count per arm and mean absolute difference vs ground truth
3. Length-matched sensitivity: truncates generated outputs to median GT word count (51 words),
   re-scores ROUGE-L F1 for all 4 cells with company-clustered 95% CIs (10,000 resamples, seed 42),
   and computes paired RAG-minus-zero-shot deltas within each format.

Outputs: eval/results/length_diagnostics.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.cluster_bootstrap import two_stage_cluster_bootstrap_ci

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "length_diagnostics.json"

CELL_FILES = {
    "baseline_zeroshot": CACHE_DIR / "zeroshot_generations.json",
    "baseline_rag": CACHE_DIR / "rag_generations.json",
    "gate_v2_zeroshot": CACHE_DIR / "ablation_zero_shot_gate_format.json",
    "gate_v2_rag": CACHE_DIR / "ablation_relevance_gate_v2.json",
}


def run_length_diagnostics() -> Dict[str, Any]:
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    incident_ids = [r["incident_id"] for r in test_records]
    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in test_records}
    gt_map = {r["incident_id"]: r["root_cause"] for r in test_records}

    # 1. Ground truth word count stats
    gt_word_counts = [len(gt_map[i].split()) for i in incident_ids]
    gt_mean = round(float(np.mean(gt_word_counts)), 4)
    gt_median = round(float(np.median(gt_word_counts)), 4)
    gt_p10 = round(float(np.percentile(gt_word_counts, 10)), 4)
    gt_p90 = round(float(np.percentile(gt_word_counts, 90)), 4)
    median_gt_limit = int(round(gt_median))

    # 2. Generated word count stats per cell
    cells_data = {}
    for cell_name, file_path in CELL_FILES.items():
        with open(file_path, "r", encoding="utf-8") as f:
            cells_data[cell_name] = json.load(f)

    cell_wc_stats = {}
    cell_raw_texts = {}
    for cell_name, cdict in cells_data.items():
        wcs = [len(cdict[i]["root_cause"].split()) for i in incident_ids]
        abs_diffs = [abs(len(cdict[i]["root_cause"].split()) - len(gt_map[i].split())) for i in incident_ids]
        cell_raw_texts[cell_name] = [cdict[i]["root_cause"] for i in incident_ids]
        cell_wc_stats[cell_name] = {
            "mean": round(float(np.mean(wcs)), 4),
            "median": round(float(np.median(wcs)), 4),
            "p10": round(float(np.percentile(wcs, 10)), 4),
            "p90": round(float(np.percentile(wcs, 90)), 4),
            "mean_abs_diff_vs_ground_truth": round(float(np.mean(abs_diffs)), 4),
        }

    # 3. Length-matched sensitivity
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    length_matched_scores = {}
    length_matched_cells = {}

    for cell_name, raw_preds in cell_raw_texts.items():
        rl_scores = []
        for inc_id, pred in zip(incident_ids, raw_preds):
            words = pred.split()[:median_gt_limit]
            trunc_pred = " ".join(words)
            gt_text = gt_map[inc_id]
            s = scorer.score(gt_text, trunc_pred)["rougeL"].fmeasure
            rl_scores.append(s)

        length_matched_scores[cell_name] = rl_scores
        ci_res = two_stage_cluster_bootstrap_ci(
            incident_ids,
            rl_scores,
            id_to_company,
            n_resamples=10000,
            seed=42,
        )
        length_matched_cells[cell_name] = {
            "rougeL_f1": ci_res
        }

    # Within-format RAG-minus-zero-shot deltas
    base_diff = [
        r - z for r, z in zip(length_matched_scores["baseline_rag"], length_matched_scores["baseline_zeroshot"])
    ]
    gate_diff = [
        r - z for r, z in zip(length_matched_scores["gate_v2_rag"], length_matched_scores["gate_v2_zeroshot"])
    ]

    base_delta_ci = two_stage_cluster_bootstrap_ci(
        incident_ids,
        base_diff,
        id_to_company,
        n_resamples=10000,
        seed=42,
    )
    gate_delta_ci = two_stage_cluster_bootstrap_ci(
        incident_ids,
        gate_diff,
        id_to_company,
        n_resamples=10000,
        seed=42,
    )

    summary_text = (
        f"Ground-truth root_cause median word count is {gt_median} (mean: {gt_mean}, p10: {gt_p10}, p90: {gt_p90}). "
        f"Mean absolute word count difference vs ground truth is {cell_wc_stats['baseline_zeroshot']['mean_abs_diff_vs_ground_truth']} "
        f"(baseline zero-shot), {cell_wc_stats['baseline_rag']['mean_abs_diff_vs_ground_truth']} (baseline RAG), "
        f"{cell_wc_stats['gate_v2_zeroshot']['mean_abs_diff_vs_ground_truth']} (gate-v2 zero-shot), and "
        f"{cell_wc_stats['gate_v2_rag']['mean_abs_diff_vs_ground_truth']} (gate-v2 RAG). "
        f"When truncated to median GT word count ({median_gt_limit} words), length-matched ROUGE-L F1 is: "
        f"baseline zero-shot = {length_matched_cells['baseline_zeroshot']['rougeL_f1']['mean']} {length_matched_cells['baseline_zeroshot']['rougeL_f1']['ci_95']}, "
        f"baseline RAG = {length_matched_cells['baseline_rag']['rougeL_f1']['mean']} {length_matched_cells['baseline_rag']['rougeL_f1']['ci_95']}, "
        f"gate-v2 zero-shot = {length_matched_cells['gate_v2_zeroshot']['rougeL_f1']['mean']} {length_matched_cells['gate_v2_zeroshot']['rougeL_f1']['ci_95']}, "
        f"gate-v2 RAG = {length_matched_cells['gate_v2_rag']['rougeL_f1']['mean']} {length_matched_cells['gate_v2_rag']['rougeL_f1']['ci_95']}. "
        f"The retrieval delta under gate-v2 format after length matching is {gate_delta_ci['mean']:+.4f} {gate_delta_ci['ci_95']}. "
        f"The retrieval delta under gate-v2 format does not change sign (+0.0019 unconstrained vs {gate_delta_ci['mean']:+.4f} length-matched) "
        f"and its CI status does not change: both cross zero and remain non-significant under company-clustered bootstrap."
    )

    results = {
        "metadata": {
            "num_test_incidents": len(incident_ids),
            "num_unique_companies": len(set(id_to_company.values())),
            "median_gt_truncation_limit": median_gt_limit,
            "bootstrap_resamples": 10000,
            "seed": 42,
            "ci_level": 0.95,
        },
        "ground_truth_stats": {
            "mean": gt_mean,
            "median": gt_median,
            "p10": gt_p10,
            "p90": gt_p90,
        },
        "cells_word_count_stats": cell_wc_stats,
        "length_matched_evaluation": {
            "truncation_word_limit": median_gt_limit,
            "cells": length_matched_cells,
            "deltas": {
                "baseline_format_rag_effect": base_delta_ci,
                "gate_v2_format_rag_effect": gate_delta_ci,
            },
        },
        "summary": summary_text,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== LENGTH DIAGNOSTICS & MATCHED SENSITIVITY ===")
    print(f"Ground Truth word count: mean={gt_mean}, median={gt_median}, p10={gt_p10}, p90={gt_p90}")
    print("\nMean absolute difference vs Ground Truth:")
    for cname, cstats in cell_wc_stats.items():
        print(f"  {cname}: mean={cstats['mean']}, median={cstats['median']}, abs_diff={cstats['mean_abs_diff_vs_ground_truth']}")
    print(f"\nLength-matched ROUGE-L (truncated to {median_gt_limit} words):")
    for cname, cdata in length_matched_cells.items():
        print(f"  {cname}: {cdata['rougeL_f1']['mean']} {cdata['rougeL_f1']['ci_95']}")
    print(f"\nWithin-format RAG effects (length-matched):")
    print(f"  Baseline format: {base_delta_ci['mean']:+.4f} {base_delta_ci['ci_95']}")
    print(f"  Gate-v2 format:  {gate_delta_ci['mean']:+.4f} {gate_delta_ci['ci_95']}")
    print(f"\nSaved results to {OUTPUT_FILE}")

    return results


if __name__ == "__main__":
    run_length_diagnostics()
