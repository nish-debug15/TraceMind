"""
TraceMind Bootstrap Confidence Intervals & BERTScore Computation
----------------------------------------------------------------
Computes 95% bootstrap confidence intervals (10,000 resamples, fixed seed 42)
for ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, and semantic similarity.
Calculates abstention rate where top-1 retrieval similarity < 0.40.
Saves results to eval/results/metrics_with_ci.json.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Force UTF-8 encoding for stdout on Windows
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
RAG_VS_ZS_FILE = RESULTS_DIR / "rag_vs_zero_shot.json"
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
OUTPUT_FILE = RESULTS_DIR / "metrics_with_ci.json"


def paired_bootstrap(
    rag_vals: List[float],
    zs_vals: List[float],
    n_resamples: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    """Compute 95% bootstrap confidence intervals with paired differences."""
    rng = np.random.RandomState(seed)
    r = np.array(rag_vals, dtype=float)
    z = np.array(zs_vals, dtype=float)
    diff = r - z
    n = len(r)

    # Generate resampled index matrix (n_resamples, n)
    indices = rng.choice(n, size=(n_resamples, n), replace=True)

    r_resampled_means = np.mean(r[indices], axis=1)
    z_resampled_means = np.mean(z[indices], axis=1)
    diff_resampled_means = np.mean(diff[indices], axis=1)

    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100

    r_low, r_high = np.percentile(r_resampled_means, [lower_pct, upper_pct])
    z_low, z_high = np.percentile(z_resampled_means, [lower_pct, upper_pct])
    d_low, d_high = np.percentile(diff_resampled_means, [lower_pct, upper_pct])

    return {
        "rag": {
            "mean": round(float(np.mean(r)), 4),
            "ci_95": [round(float(r_low), 4), round(float(r_high), 4)],
        },
        "zeroshot": {
            "mean": round(float(np.mean(z)), 4),
            "ci_95": [round(float(z_low), 4), round(float(z_high), 4)],
        },
        "paired_delta": {
            "mean": round(float(np.mean(diff)), 4),
            "ci_95": [round(float(d_low), 4), round(float(d_high), 4)],
            "strictly_positive": bool(d_low > 0.0),
        }
    }


def compute_bertscore(rag_preds: List[str], zs_preds: List[str], ground_truth: List[str]) -> Dict[str, List[float]]:
    """Compute BERTScore F1 for RAG and ZeroShot predictions."""
    from bert_score import score
    print("[BERTScore] Scoring predictions with bert-base-uncased...")
    _, _, f1_rag = score(rag_preds, ground_truth, model_type="bert-base-uncased", verbose=False)
    _, _, f1_zs = score(zs_preds, ground_truth, model_type="bert-base-uncased", verbose=False)
    return {
        "rag": f1_rag.numpy().tolist(),
        "zeroshot": f1_zs.numpy().tolist()
    }


def main():
    if not RAG_VS_ZS_FILE.exists():
        raise FileNotFoundError(f"Missing {RAG_VS_ZS_FILE}. Run eval/evaluate.py first.")

    with open(RAG_VS_ZS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    gt_by_id = {r["incident_id"]: r["root_cause"] for r in test_records}

    per_incident = data["per_incident_scores"]
    n_incidents = len(per_incident)

    ground_truth = [gt_by_id[e["incident_id"]] for e in per_incident]
    rag_preds = [e["rag"]["root_cause"] for e in per_incident]
    zs_preds = [e["zeroshot"]["root_cause"] for e in per_incident]

    # Compute BERTScore F1
    bs_results = compute_bertscore(rag_preds, zs_preds, ground_truth)

    # Attach BERTScore into per-incident records
    for i, e in enumerate(per_incident):
        e["rag"]["bert_score_f1"] = round(float(bs_results["rag"][i]), 4)
        e["zeroshot"]["bert_score_f1"] = round(float(bs_results["zeroshot"][i]), 4)
        e["delta"]["bert_score_f1"] = round(float(bs_results["rag"][i] - bs_results["zeroshot"][i]), 4)

    # Compute Abstention Rate (< 0.40 top-1 retrieval similarity)
    sim_scores = [e["retrieval"]["similarity_score"] for e in per_incident]
    abstention_threshold = 0.40
    abstained = [s for s in sim_scores if s < abstention_threshold]
    abstention_count = len(abstained)
    abstention_rate = round(abstention_count / n_incidents, 4)

    # Evaluate metrics
    metric_keys = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "semantic_sim", "bert_score_f1"]
    metric_cis = {}

    print("\n" + "=" * 90)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples, 95% CI, seed=42, N={n_incidents})")
    print("=" * 90)
    print(f"{'Metric':<16} | {'Zero-Shot [95% CI]':<26} | {'RAG [95% CI]':<26} | {'Paired Delta [95% CI]':<24} | {'Signif?':<7}")
    print("-" * 105)

    strictly_positive_metrics = []

    for k in metric_keys:
        if k == "bert_score_f1":
            r_vals = bs_results["rag"]
            z_vals = bs_results["zeroshot"]
        else:
            r_vals = [e["rag"][k] for e in per_incident]
            z_vals = [e["zeroshot"][k] for e in per_incident]

        boot = paired_bootstrap(r_vals, z_vals, n_resamples=10000, seed=42)
        metric_cis[k] = boot

        zs_str = f"{boot['zeroshot']['mean']:.4f} [{boot['zeroshot']['ci_95'][0]:.4f}, {boot['zeroshot']['ci_95'][1]:.4f}]"
        rag_str = f"{boot['rag']['mean']:.4f} [{boot['rag']['ci_95'][0]:.4f}, {boot['rag']['ci_95'][1]:.4f}]"
        d_mean = boot['paired_delta']['mean']
        d_sign = "+" if d_mean >= 0 else ""
        delta_str = f"{d_sign}{d_mean:.4f} [{boot['paired_delta']['ci_95'][0]:.4f}, {boot['paired_delta']['ci_95'][1]:.4f}]"
        is_pos = boot['paired_delta']['strictly_positive']
        signif_str = "YES (*)" if is_pos else "NO"
        if is_pos:
            strictly_positive_metrics.append(k)

        print(f"{k:<16} | {zs_str:<26} | {rag_str:<26} | {delta_str:<24} | {signif_str:<7}")

    print("\n" + "-" * 105)
    print(f"Abstention Analysis (threshold < {abstention_threshold}):")
    print(f"  Total test incidents: {n_incidents}")
    print(f"  Abstained incidents (top-1 similarity < {abstention_threshold}): {abstention_count}")
    print(f"  Abstention rate: {abstention_rate:.2%}")
    print(f"Strictly positive paired delta metrics: {strictly_positive_metrics}")

    # Output artifact
    output_data = {
        "evaluation_config": {
            "num_samples": n_incidents,
            "bootstrap_resamples": 10000,
            "seed": 42,
            "ci_level": 0.95,
            "bert_score_model": "bert-base-uncased",
            "abstention_threshold": abstention_threshold,
            "git_sha": data.get("config", {}).get("git_sha", "unknown")
        },
        "abstention": {
            "threshold": abstention_threshold,
            "abstained_count": abstention_count,
            "total_count": n_incidents,
            "abstention_rate": abstention_rate
        },
        "metrics": metric_cis,
        "strictly_positive_metrics": strictly_positive_metrics,
        "per_incident_bert_scores": [
            {
                "incident_id": e["incident_id"],
                "rag_bert_score_f1": e["rag"]["bert_score_f1"],
                "zeroshot_bert_score_f1": e["zeroshot"]["bert_score_f1"],
                "delta_bert_score_f1": e["delta"]["bert_score_f1"]
            }
            for e in per_incident
        ]
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Artifact] Saved metrics with CI to {OUTPUT_FILE}")

    # Update rag_vs_zero_shot.json to include bert_score in per_incident_scores and config
    data["bertscore"] = {
        "model": "bert-base-uncased",
        "zero_shot_mean": metric_cis["bert_score_f1"]["zeroshot"]["mean"],
        "rag_mean": metric_cis["bert_score_f1"]["rag"]["mean"],
        "delta_mean": metric_cis["bert_score_f1"]["paired_delta"]["mean"]
    }
    with open(RAG_VS_ZS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Artifact] Updated {RAG_VS_ZS_FILE} with BERTScore fields.")

    # Assertion required by Block 5
    assert len(strictly_positive_metrics) > 0, "Assertion failed: No metric achieved strictly positive 95% CI!"
    print(f"[Assertion Passed] Strictly positive CI verified for: {', '.join(strictly_positive_metrics)}")


if __name__ == "__main__":
    main()
