"""
eval/abstention_split.py
------------------------
Splits held-out test incidents (N=93) by the retrieval similarity floor (<0.40):
- Answered group (similarity >= 0.40, N=86)
- Abstained group (similarity < 0.40, N=7, flagged low_n: true)

Reports for both RAG arms (baseline RAG and relevance_gate_v2 RAG):
- Sample size (n)
- Mean top-1 cosine retrieval similarity
- Metrics: ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1, BERTScore F1, Cosine similarity
- Bootstrap CIs for Answered group
- Raw summary statistics (no CI) for Abstained group due to low n (7)

Saves results to eval/results/abstention_split.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rouge_score import rouge_scorer
from bert_score import score as calc_bertscore
from clustering.cluster import embed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
RAG_VS_ZS_FILE = RESULTS_DIR / "rag_vs_zero_shot.json"
OUTPUT_FILE = RESULTS_DIR / "abstention_split.json"


def compute_group_metrics(preds: List[str], ground_truth: List[str]) -> Dict[str, Any]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for gt, p in zip(ground_truth, preds):
        s = scorer.score(gt, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    _, _, bs = calc_bertscore(preds, ground_truth, model_type="bert-base-uncased", verbose=False)
    bs_vals = bs.numpy().tolist()

    cos_vals = []
    for gt, p in zip(ground_truth, preds):
        vp = np.array(embed(p[:1000]))
        vg = np.array(embed(gt[:1000]))
        cos_vals.append(float(vp @ vg))

    return {
        "rouge1_f1": r1,
        "rouge2_f1": r2,
        "rougeL_f1": rl,
        "bert_score_f1": bs_vals,
        "cosine": cos_vals
    }


def bootstrap_ci(vals: List[float], n_resamples: int = 10000, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    arr = np.array(vals, dtype=float)
    n = len(arr)
    idx = rng.choice(n, size=(n_resamples, n), replace=True)
    means = np.mean(arr[idx], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": round(float(np.mean(arr)), 4),
        "ci_95": [round(float(low), 4), round(float(high), 4)],
        "std": round(float(np.std(arr, ddof=1)), 4)
    }


def run_abstention_split():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    with open(RAG_VS_ZS_FILE, "r", encoding="utf-8") as f:
        rvz_data = json.load(f)

    with open(CACHE_DIR / "rag_generations.json", "r", encoding="utf-8") as f:
        baseline_rag = json.load(f)

    with open(CACHE_DIR / "ablation_relevance_gate_v2.json", "r", encoding="utf-8") as f:
        gate_v2_rag = json.load(f)

    sim_map = {}
    for item in rvz_data.get("per_incident_scores", []):
        sim_map[item["incident_id"]] = item["retrieval"]["similarity_score"]

    answered_ids = [r["incident_id"] for r in test_records if sim_map.get(r["incident_id"], 0.0) >= 0.40]
    abstained_ids = [r["incident_id"] for r in test_records if sim_map.get(r["incident_id"], 0.0) < 0.40]

    gt_map = {r["incident_id"]: r["root_cause"] for r in test_records}

    print(f"Total test incidents: {len(test_records)}")
    print(f"Answered incidents (sim >= 0.40): {len(answered_ids)}")
    print(f"Abstained incidents (sim < 0.40): {len(abstained_ids)}")

    groups = {
        "answered": {
            "n": len(answered_ids),
            "low_n": False,
            "mean_top1_similarity": round(float(np.mean([sim_map[i] for i in answered_ids])), 4),
            "median_top1_similarity": round(float(np.median([sim_map[i] for i in answered_ids])), 4),
            "ids": answered_ids,
        },
        "abstained": {
            "n": len(abstained_ids),
            "low_n": True,
            "mean_top1_similarity": round(float(np.mean([sim_map[i] for i in abstained_ids])), 4),
            "median_top1_similarity": round(float(np.median([sim_map[i] for i in abstained_ids])), 4),
            "ids": abstained_ids,
        }
    }

    for g_name, g_info in groups.items():
        g_ids = g_info["ids"]
        g_gt = [gt_map[i] for i in g_ids]

        # Baseline RAG
        b_preds = [baseline_rag[i]["root_cause"] for i in g_ids]
        b_raw = compute_group_metrics(b_preds, g_gt)

        # Gate v2 RAG
        g_preds = [gate_v2_rag[i]["root_cause"] for i in g_ids]
        g_raw = compute_group_metrics(g_preds, g_gt)

        if not g_info["low_n"]:
            # Compute CIs
            g_info["baseline_rag"] = {
                k: bootstrap_ci(vals) for k, vals in b_raw.items()
            }
            g_info["gate_v2_rag"] = {
                k: bootstrap_ci(vals) for k, vals in g_raw.items()
            }
        else:
            # Raw values only, NO CIs computed due to low n
            g_info["ci_withheld_reason"] = "sample size too small for reliable inference"
            g_info["baseline_rag"] = {
                k: {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4)
                }
                for k, vals in b_raw.items()
            }
            g_info["gate_v2_rag"] = {
                k: {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4)
                }
                for k, vals in g_raw.items()
            }

        # Remove raw id list from final serialization
        del g_info["ids"]

    output_payload = {
        "similarity_floor_threshold": 0.40,
        "total_test_incidents": len(test_records),
        "groups": groups,
        "note": "Abstained group has n=7 (low_n: true). In accordance with statistical validity guidelines, confidence intervals are withheld and raw descriptive statistics are reported only."
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[Artifact] Saved abstention split to {OUTPUT_FILE}")
    print("\n--- Answered (N=86, mean sim = {groups['answered']['mean_top1_similarity']:.4f}) ---")
    print(f"  Baseline RAG: ROUGE-L = {groups['answered']['baseline_rag']['rougeL_f1']['mean']:.4f} {groups['answered']['baseline_rag']['rougeL_f1']['ci_95']}, BERTScore = {groups['answered']['baseline_rag']['bert_score_f1']['mean']:.4f}")
    print(f"  Gate v2 RAG:  ROUGE-L = {groups['answered']['gate_v2_rag']['rougeL_f1']['mean']:.4f} {groups['answered']['gate_v2_rag']['rougeL_f1']['ci_95']}, BERTScore = {groups['answered']['gate_v2_rag']['bert_score_f1']['mean']:.4f}")

    print(f"\n--- Abstained (N=7, low_n: true, mean sim = {groups['abstained']['mean_top1_similarity']:.4f}) ---")
    print(f"  Baseline RAG: ROUGE-L = {groups['abstained']['baseline_rag']['rougeL_f1']['mean']:.4f} (raw mean), BERTScore = {groups['abstained']['baseline_rag']['bert_score_f1']['mean']:.4f}")
    print(f"  Gate v2 RAG:  ROUGE-L = {groups['abstained']['gate_v2_rag']['rougeL_f1']['mean']:.4f} (raw mean), BERTScore = {groups['abstained']['gate_v2_rag']['bert_score_f1']['mean']:.4f}")

    return output_payload


if __name__ == "__main__":
    run_abstention_split()
