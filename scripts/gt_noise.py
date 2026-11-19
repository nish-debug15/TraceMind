"""
scripts/gt_noise.py
-------------------
Rule-based detector for status-page boilerplate ground-truth in postmortems.
Flags incidents where ground-truth root_cause contains customer-facing status
boilerplate rather than architectural root-cause analysis.

Explicit phrase list + max length threshold, completely deterministic, no LLM.

Outputs:
1. eval/results/gt_noise_flags.json: All flagged incident IDs + root_cause texts for manual review.
2. eval/results/gt_noise_sensitivity.json: Full-set (N=93) vs unflagged subset (N=86) metrics
   for baseline RAG, gate-v2 RAG, and zero-shot with company-clustered bootstrap CIs.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
FLAGS_FILE = PROJECT_ROOT / "eval" / "results" / "gt_noise_flags.json"
SENSITIVITY_FILE = PROJECT_ROOT / "eval" / "results" / "gt_noise_sensitivity.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"

# Explicit phrase list indicative of status-page operational notices
BOILERPLATE_PHRASES = [
    "resolved",
    "recovered",
    "fix has been deployed",
    "fix has been implemented",
    "monitoring all systems",
    "monitoring the results",
    "monitoring the issue",
    "monitoring systems",
    "back to normal",
    "mitigating the issue",
    "working towards mitigating",
    "investigating the issue",
    "we believe the",
    "should be functional",
    "fully functional",
    "all systems operational",
    "working as intended",
    "no longer seeing errors",
    "service has been restored",
    "incident has been resolved",
    "traffic is back",
    "issues have been resolved",
]

# Ground-truth root causes shorter than this that contain boilerplate are flagged
MAX_BOILERPLATE_LENGTH = 150


def is_ground_truth_noisy(root_cause: str) -> Tuple[bool, str]:
    """Check if root_cause text is status-page boilerplate."""
    rc = root_cause.strip().lower()
    if len(rc) <= MAX_BOILERPLATE_LENGTH:
        for phrase in BOILERPLATE_PHRASES:
            if phrase in rc:
                return True, f"Length {len(rc)} <= {MAX_BOILERPLATE_LENGTH} and contains '{phrase}'"
    return False, ""


def two_stage_cluster_bootstrap_ci(
    inc_ids: List[str],
    vals: List[float],
    id_to_company: Dict[str, str],
    n_resamples: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    company_groups = {}
    for inc_id, v in zip(inc_ids, vals):
        comp = id_to_company.get(inc_id, "Unknown")
        company_groups.setdefault(comp, []).append(v)

    unique_companies = sorted(list(company_groups.keys()))
    m = len(unique_companies)
    company_arrays = {c: np.array(v, dtype=float) for c, v in company_groups.items()}
    sampled_company_indices = rng.choice(m, size=(n_resamples, m), replace=True)

    resampled_means = np.zeros(n_resamples, dtype=float)
    for b in range(n_resamples):
        sample_vals = []
        for comp_idx in sampled_company_indices[b]:
            comp = unique_companies[comp_idx]
            arr = company_arrays[comp]
            n_c = len(arr)
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


def compute_metrics_for_subset(
    inc_ids: List[str],
    gt_map: Dict[str, str],
    pred_map: Dict[str, str],
    id_to_company: Dict[str, str]
) -> Dict[str, Any]:
    from rouge_score import rouge_scorer
    from bert_score import score as calc_bertscore
    from clustering.cluster import embed

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    gts = [gt_map[i] for i in inc_ids]
    preds = [pred_map[i] for i in inc_ids]

    r1, r2, rl = [], [], []
    for g, p in zip(gts, preds):
        s = scorer.score(g, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    _, _, bs = calc_bertscore(preds, gts, model_type="bert-base-uncased", verbose=False)
    bs_vals = bs.numpy().tolist()

    cos_vals = []
    for g, p in zip(gts, preds):
        vp = np.array(embed(p[:1000]))
        vg = np.array(embed(g[:1000]))
        cos_vals.append(float(vp @ vg))

    metric_arrays = {
        "rouge1_f1": r1,
        "rouge2_f1": r2,
        "rougeL_f1": rl,
        "bert_score_f1": bs_vals,
        "cosine": cos_vals
    }

    stats = {}
    for k, vals in metric_arrays.items():
        stats[k] = two_stage_cluster_bootstrap_ci(inc_ids, vals, id_to_company)
    return stats, metric_arrays


def run_gt_noise_analysis():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in records}
    gt_map = {r["incident_id"]: r["root_cause"] for r in records}

    flagged = []
    clean_records = []

    for r in records:
        inc_id = r["incident_id"]
        company = id_to_company[inc_id]
        rc = r.get("root_cause", "")
        noisy, reason = is_ground_truth_noisy(rc)

        if noisy:
            flagged.append({
                "incident_id": inc_id,
                "company": company,
                "length": len(rc.strip()),
                "reason": reason,
                "root_cause": rc.strip()
            })
        else:
            clean_records.append(r)

    clean_ids = [r["incident_id"] for r in clean_records]
    all_ids = [r["incident_id"] for r in records]

    print(f"Total test incidents: {len(records)}")
    print(f"Flagged as noisy ground-truth: {len(flagged)} ({len(flagged)/len(records)*100:.1f}%)")
    print(f"Clean incidents: {len(clean_ids)} ({len(clean_ids)/len(records)*100:.1f}%)")

    # 1. Save gt_noise_flags.json
    FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FLAGS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total_incidents": len(records),
            "flagged_count": len(flagged),
            "clean_count": len(clean_ids),
            "noise_rate": round(len(flagged) / len(records), 4),
            "detector_criteria": {
                "max_length_threshold": MAX_BOILERPLATE_LENGTH,
                "phrases": BOILERPLATE_PHRASES
            },
            "flagged_incidents": flagged
        }, f, indent=2)
    print(f"[Artifact] Saved flagged incidents to {FLAGS_FILE}")

    # 2. Re-score arms on full-set vs unflagged subset
    with open(CACHE_DIR / "zeroshot_generations.json", "r", encoding="utf-8") as f:
        zs_gens = {k: v["root_cause"] for k, v in json.load(f).items()}
    with open(CACHE_DIR / "rag_generations.json", "r", encoding="utf-8") as f:
        base_rag_gens = {k: v["root_cause"] for k, v in json.load(f).items()}
    with open(CACHE_DIR / "ablation_relevance_gate_v2.json", "r", encoding="utf-8") as f:
        gate_v2_gens = {k: v["root_cause"] for k, v in json.load(f).items()}

    arms = {
        "zero_shot": zs_gens,
        "baseline_rag": base_rag_gens,
        "gate_v2_rag": gate_v2_gens
    }

    full_results = {}
    clean_results = {}
    full_arrays = {}
    clean_arrays = {}

    print("\nScoring full set (N=93)...")
    for arm_name, pred_map in arms.items():
        stats, arrs = compute_metrics_for_subset(all_ids, gt_map, pred_map, id_to_company)
        full_results[arm_name] = stats
        full_arrays[arm_name] = arrs

    print("Scoring clean subset (N=86)...")
    for arm_name, pred_map in arms.items():
        stats, arrs = compute_metrics_for_subset(clean_ids, gt_map, pred_map, id_to_company)
        clean_results[arm_name] = stats
        clean_arrays[arm_name] = arrs

    # Compute paired deltas (RAG - ZeroShot) on both full and clean
    paired_deltas = {
        "full_set": {},
        "clean_subset": {}
    }
    for m_k in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bert_score_f1", "cosine"]:
        # Baseline RAG vs Zero-Shot
        diff_full = [r - z for r, z in zip(full_arrays["baseline_rag"][m_k], full_arrays["zero_shot"][m_k])]
        diff_clean = [r - z for r, z in zip(clean_arrays["baseline_rag"][m_k], clean_arrays["zero_shot"][m_k])]
        paired_deltas["full_set"][f"baseline_rag_minus_zs_{m_k}"] = two_stage_cluster_bootstrap_ci(
            all_ids, diff_full, id_to_company
        )
        paired_deltas["clean_subset"][f"baseline_rag_minus_zs_{m_k}"] = two_stage_cluster_bootstrap_ci(
            clean_ids, diff_clean, id_to_company
        )

        # Gate v2 RAG vs Zero-Shot
        diff_full_v2 = [r - z for r, z in zip(full_arrays["gate_v2_rag"][m_k], full_arrays["zero_shot"][m_k])]
        diff_clean_v2 = [r - z for r, z in zip(clean_arrays["gate_v2_rag"][m_k], clean_arrays["zero_shot"][m_k])]
        paired_deltas["full_set"][f"gate_v2_rag_minus_zs_{m_k}"] = two_stage_cluster_bootstrap_ci(
            all_ids, diff_full_v2, id_to_company
        )
        paired_deltas["clean_subset"][f"gate_v2_rag_minus_zs_{m_k}"] = two_stage_cluster_bootstrap_ci(
            clean_ids, diff_clean_v2, id_to_company
        )

    sensitivity_payload = {
        "metadata": {
            "full_sample_size": len(all_ids),
            "unflagged_sample_size": len(clean_ids),
            "flagged_count": len(flagged),
            "noise_rate": round(len(flagged) / len(all_ids), 4),
            "bootstrap_resamples": 10000,
            "seed": 42
        },
        "full_set_metrics": full_results,
        "clean_subset_metrics": clean_results,
        "paired_deltas": paired_deltas
    }

    with open(SENSITIVITY_FILE, "w", encoding="utf-8") as f:
        json.dump(sensitivity_payload, f, indent=2)

    print(f"\n[Artifact] Saved GT noise sensitivity analysis to {SENSITIVITY_FILE}")
    print("\n" + "=" * 90)
    print("GROUND-TRUTH NOISE SENSITIVITY COMPARISON (Full N=93 vs. Clean N=86)")
    print("=" * 90)
    print(f"{'Arm':<16} | {'Metric':<14} | {'Full Set (N=93)':<24} | {'Clean Subset (N=86)':<24} | {'Change':<8}")
    print("-" * 90)
    for arm_name in ["zero_shot", "baseline_rag", "gate_v2_rag"]:
        for m_k in ["rougeL_f1", "bert_score_f1", "cosine"]:
            f_stat = full_results[arm_name][m_k]
            c_stat = clean_results[arm_name][m_k]
            diff = c_stat["mean"] - f_stat["mean"]
            print(f"{arm_name:<16} | {m_k:<14} | {f_stat['mean']:.4f} {f_stat['ci_95']} | {c_stat['mean']:.4f} {c_stat['ci_95']} | {diff:+.4f}")

    return sensitivity_payload


if __name__ == "__main__":
    run_gt_noise_analysis()
