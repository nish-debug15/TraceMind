"""
eval/cluster_bootstrap.py
-------------------------
Implements two-stage company-clustered bootstrap confidence intervals:
- Stage 1: Resample companies with replacement (M = 40 unique companies)
- Stage 2: Resample incidents with replacement within each sampled company
- 10,000 resamples, fixed seed 42, 95% CI level

Applies to:
1. Every delta in rag_vs_zero_shot / metrics_with_ci
2. Every cell and delta in control_2x2.json
3. Paired gate-vs-baseline and v2-vs-gate in prompt ablation

Writes results with side-by-side per-incident and company-clustered CIs
plus flag `significant_under_cluster_bootstrap` to eval/results/cluster_bootstrap_ci.json.
Also cleans legacy "cross-prompt variance" field from prompt_ablation.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
CONTROL_2X2_FILE = RESULTS_DIR / "control_2x2.json"
METRICS_CI_FILE = RESULTS_DIR / "metrics_with_ci.json"
PROMPT_ABLATION_FILE = RESULTS_DIR / "prompt_ablation.json"
OUTPUT_FILE = RESULTS_DIR / "cluster_bootstrap_ci.json"


def per_incident_bootstrap_ci(
    vals: List[float],
    n_resamples: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    arr = np.array(vals, dtype=float)
    n = len(arr)
    idx = rng.choice(n, size=(n_resamples, n), replace=True)
    means = np.mean(arr[idx], axis=1)
    alpha = (1.0 - ci_level) / 2.0
    low = np.percentile(means, alpha * 100)
    high = np.percentile(means, (1.0 - alpha) * 100)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "ci_95": [round(float(low), 4), round(float(high), 4)],
        "excludes_zero": bool(low > 0.0 or high < 0.0)
    }


def two_stage_cluster_bootstrap_ci(
    inc_ids: List[str],
    vals: List[float],
    id_to_company: Dict[str, str],
    n_resamples: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)

    # Group incident values by company
    company_groups = {}
    for inc_id, v in zip(inc_ids, vals):
        comp = id_to_company.get(inc_id, "Unknown")
        company_groups.setdefault(comp, []).append(v)

    unique_companies = sorted(list(company_groups.keys()))
    m = len(unique_companies)
    company_arrays = {c: np.array(vals, dtype=float) for c, vals in company_groups.items()}

    # Stage 1: Sample m companies with replacement across all n_resamples
    sampled_company_indices = rng.choice(m, size=(n_resamples, m), replace=True)

    resampled_means = np.zeros(n_resamples, dtype=float)
    for b in range(n_resamples):
        sample_vals = []
        for comp_idx in sampled_company_indices[b]:
            comp = unique_companies[comp_idx]
            arr = company_arrays[comp]
            n_c = len(arr)
            # Stage 2: Sample n_c incidents with replacement within company
            c_sample_idx = rng.choice(n_c, size=n_c, replace=True)
            sample_vals.extend(arr[c_sample_idx])
        resampled_means[b] = np.mean(sample_vals)

    alpha = (1.0 - ci_level) / 2.0
    low = np.percentile(resampled_means, alpha * 100)
    high = np.percentile(resampled_means, (1.0 - alpha) * 100)
    orig_mean = np.mean(vals)

    return {
        "mean": round(float(orig_mean), 4),
        "ci_95": [round(float(low), 4), round(float(high), 4)],
        "significant_under_cluster_bootstrap": bool(low > 0.0 or high < 0.0)
    }


def run_cluster_bootstrap():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in test_records}
    unique_companies = sorted(list(set(id_to_company.values())))
    num_unique_companies = len(unique_companies)
    print(f"[Cluster Bootstrap] Test set contains {len(test_records)} incidents across {num_unique_companies} unique companies.")

    results = {
        "metadata": {
            "num_incidents": len(test_records),
            "num_unique_companies": num_unique_companies,
            "bootstrap_resamples": 10000,
            "seed": 42,
            "ci_level": 0.95,
        },
        "rag_vs_zeroshot_deltas": {},
        "control_2x2_cells": {},
        "control_2x2_deltas": {},
        "prompt_ablation_deltas": {},
    }

    # 1. Main RAG vs ZeroShot Deltas (from metrics_with_ci / rag_vs_zero_shot)
    if METRICS_CI_FILE.exists():
        with open(METRICS_CI_FILE, "r", encoding="utf-8") as f:
            mci_data = json.load(f)

        with open(RESULTS_DIR / "rag_vs_zero_shot.json", "r", encoding="utf-8") as f:
            rvz_data = json.load(f)

        per_inc = rvz_data["per_incident_scores"]
        inc_ids = [e["incident_id"] for e in per_inc]

        # Metric keys
        metric_keys = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "semantic_sim", "bert_score_f1"]
        for k in metric_keys:
            if k == "bert_score_f1":
                # Find in per_incident_bert_scores
                bs_map = {item["incident_id"]: item["delta_bert_score_f1"] for item in mci_data.get("per_incident_bert_scores", [])}
                delta_vals = [bs_map.get(i, 0.0) for i in inc_ids]
            else:
                delta_vals = [e["delta"][k] for e in per_inc]

            pi_ci = per_incident_bootstrap_ci(delta_vals)
            cl_ci = two_stage_cluster_bootstrap_ci(inc_ids, delta_vals, id_to_company)

            results["rag_vs_zeroshot_deltas"][k] = {
                "mean": pi_ci["mean"],
                "per_incident_ci_95": pi_ci["ci_95"],
                "per_incident_excludes_zero": pi_ci["excludes_zero"],
                "cluster_ci_95": cl_ci["ci_95"],
                "significant_under_cluster_bootstrap": cl_ci["significant_under_cluster_bootstrap"]
            }

    # 2. Control 2x2 Cells & Deltas
    if CONTROL_2X2_FILE.exists():
        with open(CONTROL_2X2_FILE, "r", encoding="utf-8") as f:
            c2_data = json.load(f)

        # In control_2x2.py, per_incident arrays can be extracted or re-evaluated
        # Check if control_2x2 stored per_incident or if we load caches:
        # Load from caches for exact per-incident values
        cache_files = {
            "baseline_zeroshot": PROJECT_ROOT / "eval" / "cache" / "zeroshot_generations.json",
            "baseline_rag": PROJECT_ROOT / "eval" / "cache" / "rag_generations.json",
            "gate_v2_zeroshot": PROJECT_ROOT / "eval" / "cache" / "ablation_zero_shot_gate_format.json",
            "gate_v2_rag": PROJECT_ROOT / "eval" / "cache" / "ablation_relevance_gate_v2.json"
        }

        # Check if all 4 cache files exist
        if all(p.exists() for p in cache_files.values()):
            from rouge_score import rouge_scorer
            from bert_score import score as calc_bertscore
            from clustering.cluster import embed

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            caches = {k: json.load(open(p, encoding="utf-8")) for k, p in cache_files.items()}

            valid_ids = [
                r["incident_id"] for r in test_records
                if all(r["incident_id"] in c for c in caches.values())
            ]
            gt_map = {r["incident_id"]: r["root_cause"] for r in test_records}
            ground_truth = [gt_map[i] for i in valid_ids]

            cell_per_incident = {}
            for c_name, c_dict in caches.items():
                preds = [c_dict[i]["root_cause"] for i in valid_ids]
                rl_f1 = [scorer.score(gt, p)["rougeL"].fmeasure for gt, p in zip(ground_truth, preds)]
                _, _, bs = calc_bertscore(preds, ground_truth, model_type="bert-base-uncased", verbose=False)
                bs_f1 = bs.numpy().tolist()
                cos_vals = []
                for gt, p in zip(ground_truth, preds):
                    vp = np.array(embed(p[:1000]))
                    vg = np.array(embed(gt[:1000]))
                    cos_vals.append(float(vp @ vg))

                cell_per_incident[c_name] = {
                    "rougeL_f1": rl_f1,
                    "bert_score_f1": bs_f1,
                    "cosine": cos_vals
                }

                # Cell level CI
                results["control_2x2_cells"][c_name] = {}
                for m_k in ["rougeL_f1", "bert_score_f1", "cosine"]:
                    vals = cell_per_incident[c_name][m_k]
                    pi_ci = per_incident_bootstrap_ci(vals)
                    cl_ci = two_stage_cluster_bootstrap_ci(valid_ids, vals, id_to_company)
                    results["control_2x2_cells"][c_name][m_k] = {
                        "mean": pi_ci["mean"],
                        "per_incident_ci_95": pi_ci["ci_95"],
                        "cluster_ci_95": cl_ci["ci_95"]
                    }

            # Deltas
            delta_pairs = {
                "rag_effect_baseline_format": ("baseline_rag", "baseline_zeroshot"),
                "rag_effect_gate_v2_format": ("gate_v2_rag", "gate_v2_zeroshot"),
                "format_effect_zeroshot_arm": ("gate_v2_zeroshot", "baseline_zeroshot"),
                "format_effect_rag_arm": ("gate_v2_rag", "baseline_rag")
            }

            for d_name, (arm_a, arm_b) in delta_pairs.items():
                results["control_2x2_deltas"][d_name] = {}
                for m_k in ["rougeL_f1", "bert_score_f1", "cosine"]:
                    diff = [a - b for a, b in zip(cell_per_incident[arm_a][m_k], cell_per_incident[arm_b][m_k])]
                    pi_ci = per_incident_bootstrap_ci(diff)
                    cl_ci = two_stage_cluster_bootstrap_ci(valid_ids, diff, id_to_company)
                    results["control_2x2_deltas"][d_name][m_k] = {
                        "mean": pi_ci["mean"],
                        "per_incident_ci_95": pi_ci["ci_95"],
                        "per_incident_excludes_zero": pi_ci["excludes_zero"],
                        "cluster_ci_95": cl_ci["ci_95"],
                        "significant_under_cluster_bootstrap": cl_ci["significant_under_cluster_bootstrap"]
                    }

    # 3. Prompt Ablation Deltas (gate vs baseline, v2 vs gate)
    ablation_caches = {
        "baseline": PROJECT_ROOT / "eval" / "cache" / "rag_generations.json",
        "relevance_gate": PROJECT_ROOT / "eval" / "cache" / "ablation_relevance_gate.json",
        "relevance_gate_v2": PROJECT_ROOT / "eval" / "cache" / "ablation_relevance_gate_v2.json"
    }
    if all(p.exists() for p in ablation_caches.values()):
        from rouge_score import rouge_scorer
        from bert_score import score as calc_bertscore
        from clustering.cluster import embed

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        caches = {k: json.load(open(p, encoding="utf-8")) for k, p in ablation_caches.items()}
        valid_ids = [
            r["incident_id"] for r in test_records
            if all(r["incident_id"] in c for c in caches.values())
        ]
        gt_map = {r["incident_id"]: r["root_cause"] for r in test_records}
        ground_truth = [gt_map[i] for i in valid_ids]

        ablation_pi = {}
        for p_name, c_dict in caches.items():
            preds = [c_dict[i]["root_cause"] for i in valid_ids]
            rl_f1 = [scorer.score(gt, p)["rougeL"].fmeasure for gt, p in zip(ground_truth, preds)]
            _, _, bs = calc_bertscore(preds, ground_truth, model_type="bert-base-uncased", verbose=False)
            bs_f1 = bs.numpy().tolist()
            cos_vals = []
            for gt, p in zip(ground_truth, preds):
                vp = np.array(embed(p[:1000]))
                vg = np.array(embed(gt[:1000]))
                cos_vals.append(float(vp @ vg))
            ablation_pi[p_name] = {"rougeL_f1": rl_f1, "bert_score_f1": bs_f1, "cosine": cos_vals}

        ablation_comparisons = {
            "gate_minus_baseline": ("relevance_gate", "baseline"),
            "v2_minus_gate": ("relevance_gate_v2", "relevance_gate"),
            "v2_minus_baseline": ("relevance_gate_v2", "baseline")
        }

        for comp_name, (arm_a, arm_b) in ablation_comparisons.items():
            results["prompt_ablation_deltas"][comp_name] = {}
            for m_k in ["rougeL_f1", "bert_score_f1", "cosine"]:
                diff = [a - b for a, b in zip(ablation_pi[arm_a][m_k], ablation_pi[arm_b][m_k])]
                pi_ci = per_incident_bootstrap_ci(diff)
                cl_ci = two_stage_cluster_bootstrap_ci(valid_ids, diff, id_to_company)
                results["prompt_ablation_deltas"][comp_name][m_k] = {
                    "mean": pi_ci["mean"],
                    "per_incident_ci_95": pi_ci["ci_95"],
                    "per_incident_excludes_zero": pi_ci["excludes_zero"],
                    "cluster_ci_95": cl_ci["ci_95"],
                    "significant_under_cluster_bootstrap": cl_ci["significant_under_cluster_bootstrap"]
                }

    # 4. Clean legacy "cross-prompt variance" from prompt_ablation.json if present
    if PROMPT_ABLATION_FILE.exists():
        with open(PROMPT_ABLATION_FILE, "r", encoding="utf-8") as f:
            pa_data = json.load(f)
        if "spread" in pa_data:
            # Delete variance fields from spread
            keys_to_delete = [k for k in pa_data["spread"] if "variance" in k]
            for k in keys_to_delete:
                del pa_data["spread"][k]
        with open(PROMPT_ABLATION_FILE, "w", encoding="utf-8") as f:
            json.dump(pa_data, f, indent=2)
        print("[Cleanup] Removed meaningless variance fields from prompt_ablation.json")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Artifact] Saved cluster bootstrap results to {OUTPUT_FILE}")
    print("\n" + "=" * 95)
    print("COMPANY-CLUSTERED BOOTSTRAP CIs vs. PER-INCIDENT CIs (10,000 resamples, 40 clusters)")
    print("=" * 95)
    print(f"{'Comparison':<30} | {'Metric':<14} | {'Delta':<8} | {'Per-Incident CI':<20} | {'Clustered CI':<20} | {'Clustered Sig?'}")
    print("-" * 115)

    for section in ["rag_vs_zeroshot_deltas", "control_2x2_deltas", "prompt_ablation_deltas"]:
        sub = results.get(section, {})
        for name, metrics in sub.items():
            if isinstance(metrics, dict) and "mean" in metrics:
                # Direct metric
                d = metrics
                sig = "YES" if d["significant_under_cluster_bootstrap"] else "NO"
                print(f"{name:<30} | {'overall':<14} | {d['mean']:<8.4f} | {str(d['per_incident_ci_95']):<20} | {str(d['cluster_ci_95']):<20} | {sig}")
            elif isinstance(metrics, dict):
                for m_k, d in metrics.items():
                    sig = "YES" if d["significant_under_cluster_bootstrap"] else "NO"
                    print(f"{name:<30} | {m_k:<14} | {d['mean']:<8.4f} | {str(d['per_incident_ci_95']):<20} | {str(d['cluster_ci_95']):<20} | {sig}")

    return results


if __name__ == "__main__":
    run_cluster_bootstrap()
