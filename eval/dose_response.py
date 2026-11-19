"""
eval/dose_response.py
---------------------
Exploratory dose-response analysis relating top-1 ChromaDB retrieval cosine similarity
to downstream RAG generation gains (RAG minus zero-shot delta) for:
1. Baseline prompt format (ROUGE-L F1 delta, BERTScore F1 delta)
2. Gate-v2 prompt format (ROUGE-L F1 delta, BERTScore F1 delta)

Computes:
- Spearman rank correlation rho with two-stage company-clustered bootstrap 95% CI (10,000 resamples, seed 42)
- Terciles of top-1 retrieval similarity (Low, Mid, High; n=31 each) with mean delta and clustered 95% CIs
- Split by retrieval pass (Pass 1 quality:high vs Pass 2 fallback)
- Saves eval/figures/dose_response.png
- Saves eval/results/dose_response.json with field `exploratory: true`
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rouge_score import rouge_scorer
from bert_score import score as calc_bertscore
from eval.cluster_bootstrap import two_stage_cluster_bootstrap_ci

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.json"
RVZ_FILE = PROJECT_ROOT / "eval" / "results" / "rag_vs_zero_shot.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
FIGURES_DIR = PROJECT_ROOT / "eval" / "figures"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
OUTPUT_JSON = RESULTS_DIR / "dose_response.json"
OUTPUT_PNG = FIGURES_DIR / "dose_response.png"


def cluster_bootstrap_spearman(
    inc_ids: List[str],
    x_vals: List[float],
    y_vals: List[float],
    id_to_company: Dict[str, str],
    n_resamples: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    """Two-stage company-clustered bootstrap CI for Spearman rho."""
    rng = np.random.RandomState(seed)

    company_data = {}
    for inc_id, x, y in zip(inc_ids, x_vals, y_vals):
        comp = id_to_company.get(inc_id, "Unknown")
        company_data.setdefault(comp, []).append((x, y))

    unique_companies = sorted(list(company_data.keys()))
    m = len(unique_companies)
    comp_arrays = {c: np.array(pairs, dtype=float) for c, pairs in company_data.items()}

    point_rho = float(spearmanr(x_vals, y_vals).statistic)

    sampled_comp_idx = rng.choice(m, size=(n_resamples, m), replace=True)
    boot_rhos = np.zeros(n_resamples, dtype=float)

    for b in range(n_resamples):
        sample_x = []
        sample_y = []
        for c_idx in sampled_comp_idx[b]:
            comp = unique_companies[c_idx]
            arr = comp_arrays[comp]
            n_c = len(arr)
            sub_idx = rng.choice(n_c, size=n_c, replace=True)
            sample_x.extend(arr[sub_idx, 0])
            sample_y.extend(arr[sub_idx, 1])

        # Handle potential zero-variance edge cases in resamples
        if len(set(sample_x)) < 2 or len(set(sample_y)) < 2:
            boot_rhos[b] = 0.0
        else:
            r = spearmanr(sample_x, sample_y).statistic
            boot_rhos[b] = 0.0 if np.isnan(r) else float(r)

    alpha = (1.0 - ci_level) / 2.0
    low = np.percentile(boot_rhos, alpha * 100)
    high = np.percentile(boot_rhos, (1.0 - alpha) * 100)

    return {
        "spearman_rho": round(point_rho, 4),
        "ci_95": [round(float(low), 4), round(float(high), 4)],
        "excludes_zero": bool(low > 0.0 or high < 0.0)
    }


def run_dose_response() -> Dict[str, Any]:
    print("[Dose-Response] Loading test records and retrieval data...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    incident_ids = [r["incident_id"] for r in test_records]
    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in test_records}
    gt_map = {r["incident_id"]: r["root_cause"] for r in test_records}

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    train_quality_map = {r["incident_id"]: r.get("quality", "unknown") for r in train_records}

    # Load baseline format scores from rag_vs_zero_shot.json
    with open(RVZ_FILE, "r", encoding="utf-8") as f:
        rvz_data = json.load(f)
    rvz_map = {item["incident_id"]: item for item in rvz_data["per_incident_scores"]}

    # Top-1 retrieval similarities
    sims = [rvz_map[i]["retrieval"]["similarity_score"] for i in incident_ids]

    # Baseline format deltas
    base_rl_deltas = [rvz_map[i]["delta"]["rougeL_f1"] for i in incident_ids]
    base_bs_deltas = [rvz_map[i]["delta"]["bert_score_f1"] for i in incident_ids]

    # Gate-v2 format generations and scoring
    with open(CACHE_DIR / "ablation_relevance_gate_v2.json", "r", encoding="utf-8") as f:
        gate_rag = json.load(f)
    with open(CACHE_DIR / "ablation_zero_shot_gate_format.json", "r", encoding="utf-8") as f:
        gate_zs = json.load(f)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    gate_rl_rag = [scorer.score(gt_map[i], gate_rag[i]["root_cause"])["rougeL"].fmeasure for i in incident_ids]
    gate_rl_zs = [scorer.score(gt_map[i], gate_zs[i]["root_cause"])["rougeL"].fmeasure for i in incident_ids]
    gate_rl_deltas = [r - z for r, z in zip(gate_rl_rag, gate_rl_zs)]

    safe_rag = [gate_rag[i]["root_cause"] if gate_rag[i]["root_cause"].strip() else "none" for i in incident_ids]
    safe_zs = [gate_zs[i]["root_cause"] if gate_zs[i]["root_cause"].strip() else "none" for i in incident_ids]
    safe_gt = [gt_map[i] if gt_map[i].strip() else "none" for i in incident_ids]

    print("[Dose-Response] Computing BERTScore for Gate-v2 arm...")
    _, _, bs_rag = calc_bertscore(safe_rag, safe_gt, model_type="bert-base-uncased", verbose=False)
    _, _, bs_zs = calc_bertscore(safe_zs, safe_gt, model_type="bert-base-uncased", verbose=False)
    gate_bs_deltas = (bs_rag - bs_zs).numpy().tolist()

    # Pass classification: Pass 1 (high quality & sim >= 0.40) vs Pass 2 fallback
    passes = []
    for i, s in zip(incident_ids, sims):
        mid = rvz_map[i]["retrieval"]["matched_incident_id"]
        q = train_quality_map.get(mid, "low")
        p_num = 1 if (q == "high" and s >= 0.40) else 2
        passes.append(p_num)

    # Terciles of similarity
    sorted_indices = np.argsort(sims)
    tercile_assignments = np.zeros(len(incident_ids), dtype=int)
    tercile_assignments[sorted_indices[:31]] = 1
    tercile_assignments[sorted_indices[31:62]] = 2
    tercile_assignments[sorted_indices[62:]] = 3

    tercile_ranges = {
        "low": [float(min(np.array(sims)[tercile_assignments == 1])), float(max(np.array(sims)[tercile_assignments == 1]))],
        "mid": [float(min(np.array(sims)[tercile_assignments == 2])), float(max(np.array(sims)[tercile_assignments == 2]))],
        "high": [float(min(np.array(sims)[tercile_assignments == 3])), float(max(np.array(sims)[tercile_assignments == 3]))],
    }

    # Correlation analyses
    print("[Dose-Response] Computing Spearman rho with clustered bootstrap CIs...")
    base_rl_rho = cluster_bootstrap_spearman(incident_ids, sims, base_rl_deltas, id_to_company)
    base_bs_rho = cluster_bootstrap_spearman(incident_ids, sims, base_bs_deltas, id_to_company)
    gate_rl_rho = cluster_bootstrap_spearman(incident_ids, sims, gate_rl_deltas, id_to_company)
    gate_bs_rho = cluster_bootstrap_spearman(incident_ids, sims, gate_bs_deltas, id_to_company)

    # Function to compute bin stats
    def compute_subgroup_stats(mask: np.ndarray) -> Dict[str, Any]:
        sub_ids = [incident_ids[idx] for idx in range(len(incident_ids)) if mask[idx]]
        sub_sims = [sims[idx] for idx in range(len(incident_ids)) if mask[idx]]
        sub_base_rl = [base_rl_deltas[idx] for idx in range(len(incident_ids)) if mask[idx]]
        sub_base_bs = [base_bs_deltas[idx] for idx in range(len(incident_ids)) if mask[idx]]
        sub_gate_rl = [gate_rl_deltas[idx] for idx in range(len(incident_ids)) if mask[idx]]
        sub_gate_bs = [gate_bs_deltas[idx] for idx in range(len(incident_ids)) if mask[idx]]

        return {
            "n": int(len(sub_ids)),
            "mean_sim": round(float(np.mean(sub_sims)), 4),
            "sim_range": [round(float(min(sub_sims)), 4), round(float(max(sub_sims)), 4)],
            "baseline": {
                "rougeL_delta": two_stage_cluster_bootstrap_ci(sub_ids, sub_base_rl, id_to_company),
                "bert_score_delta": two_stage_cluster_bootstrap_ci(sub_ids, sub_base_bs, id_to_company),
            },
            "gate_v2": {
                "rougeL_delta": two_stage_cluster_bootstrap_ci(sub_ids, sub_gate_rl, id_to_company),
                "bert_score_delta": two_stage_cluster_bootstrap_ci(sub_ids, sub_gate_bs, id_to_company),
            }
        }

    tercile_stats = {
        "low_tercile": compute_subgroup_stats(tercile_assignments == 1),
        "mid_tercile": compute_subgroup_stats(tercile_assignments == 2),
        "high_tercile": compute_subgroup_stats(tercile_assignments == 3),
    }

    pass_stats = {
        "pass1_high_quality": compute_subgroup_stats(np.array(passes) == 1),
        "pass2_fallback": compute_subgroup_stats(np.array(passes) == 2),
        "pass_recorded_in_cache": False,
        "pass_determination_rule": "Pass 1: matched incident in train.json has quality == 'high' and similarity >= 0.40; Pass 2: fallback to any quality",
    }

    summary_text = (
        f"Exploratory dose-response analysis between top-1 retrieval similarity and downstream generation delta (N=93):\n"
        f"1. Baseline Format:\n"
        f"   - ROUGE-L delta vs sim: Spearman rho = {base_rl_rho['spearman_rho']:+.4f} (95% clustered CI {base_rl_rho['ci_95']}, excludes zero: {base_rl_rho['excludes_zero']}). "
        f"Tercile mean deltas: Low={tercile_stats['low_tercile']['baseline']['rougeL_delta']['mean']:+.4f}, "
        f"Mid={tercile_stats['mid_tercile']['baseline']['rougeL_delta']['mean']:+.4f}, "
        f"High={tercile_stats['high_tercile']['baseline']['rougeL_delta']['mean']:+.4f}.\n"
        f"   - BERTScore delta vs sim: Spearman rho = {base_bs_rho['spearman_rho']:+.4f} (95% clustered CI {base_bs_rho['ci_95']}, excludes zero: {base_bs_rho['excludes_zero']}). "
        f"Tercile mean deltas: Low={tercile_stats['low_tercile']['baseline']['bert_score_delta']['mean']:+.4f}, "
        f"Mid={tercile_stats['mid_tercile']['baseline']['bert_score_delta']['mean']:+.4f}, "
        f"High={tercile_stats['high_tercile']['baseline']['bert_score_delta']['mean']:+.4f}.\n"
        f"2. Gate-v2 Format:\n"
        f"   - ROUGE-L delta vs sim: Spearman rho = {gate_rl_rho['spearman_rho']:+.4f} (95% clustered CI {gate_rl_rho['ci_95']}, excludes zero: {gate_rl_rho['excludes_zero']}). "
        f"Tercile mean deltas: Low={tercile_stats['low_tercile']['gate_v2']['rougeL_delta']['mean']:+.4f}, "
        f"Mid={tercile_stats['mid_tercile']['gate_v2']['rougeL_delta']['mean']:+.4f}, "
        f"High={tercile_stats['high_tercile']['gate_v2']['rougeL_delta']['mean']:+.4f}.\n"
        f"   - BERTScore delta vs sim: Spearman rho = {gate_bs_rho['spearman_rho']:+.4f} (95% clustered CI {gate_bs_rho['ci_95']}, excludes zero: {gate_bs_rho['excludes_zero']}). "
        f"Tercile mean deltas: Low={tercile_stats['low_tercile']['gate_v2']['bert_score_delta']['mean']:+.4f}, "
        f"Mid={tercile_stats['mid_tercile']['gate_v2']['bert_score_delta']['mean']:+.4f}, "
        f"High={tercile_stats['high_tercile']['gate_v2']['bert_score_delta']['mean']:+.4f}."
    )

    results_payload = {
        "exploratory": True,
        "metadata": {
            "num_test_incidents": len(incident_ids),
            "num_unique_companies": len(set(id_to_company.values())),
            "bootstrap_resamples": 10000,
            "seed": 42,
            "ci_level": 0.95,
        },
        "spearman_correlations": {
            "baseline_format": {
                "rougeL_delta": base_rl_rho,
                "bert_score_delta": base_bs_rho,
            },
            "gate_v2_format": {
                "rougeL_delta": gate_rl_rho,
                "bert_score_delta": gate_bs_rho,
            }
        },
        "terciles": tercile_stats,
        "retrieval_pass_split": pass_stats,
        "summary": summary_text,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[Dose-Response] Saved results to {OUTPUT_JSON}")

    # Generate Figure
    print("[Dose-Response] Generating figure eval/figures/dose_response.png...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fig.suptitle("Exploratory Retrieval Similarity Dose-Response (N = 93)", fontsize=15, fontweight="bold", y=0.98)

    sub_configs = [
        (0, 0, base_rl_deltas, "Baseline Format — ROUGE-L F1 Delta", base_rl_rho, "tab:blue"),
        (0, 1, base_bs_deltas, "Baseline Format — BERTScore F1 Delta", base_bs_rho, "tab:cyan"),
        (1, 0, gate_rl_deltas, "Gate-v2 Format — ROUGE-L F1 Delta", gate_rl_rho, "tab:purple"),
        (1, 1, gate_bs_deltas, "Gate-v2 Format — BERTScore F1 Delta", gate_bs_rho, "tab:olive"),
    ]

    tercile_names = ["low_tercile", "mid_tercile", "high_tercile"]

    for row, col, deltas, title_text, rho_dict, pt_color in sub_configs:
        ax = axes[row, col]
        ax.scatter(sims, deltas, alpha=0.55, color=pt_color, edgecolors="none", s=35, label="Test Incident")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

        # Plot tercile means with CIs
        t_means_x = []
        t_means_y = []
        t_errs_low = []
        t_errs_high = []

        is_gate = "Gate-v2" in title_text
        metric_key = "rougeL_delta" if "ROUGE-L" in title_text else "bert_score_delta"
        arm_key = "gate_v2" if is_gate else "baseline"

        for tname in tercile_names:
            t_data = tercile_stats[tname]
            tx = t_data["mean_sim"]
            ty = t_data[arm_key][metric_key]["mean"]
            ci = t_data[arm_key][metric_key]["ci_95"]
            t_means_x.append(tx)
            t_means_y.append(ty)
            t_errs_low.append(ty - ci[0])
            t_errs_high.append(ci[1] - ty)

        ax.errorbar(
            t_means_x,
            t_means_y,
            yerr=[t_errs_low, t_errs_high],
            fmt="D--",
            color="black",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            markersize=7,
            label="Tercile Mean [95% CI]"
        )

        # Mark 0.40 threshold
        ax.axvline(0.40, color="red", linestyle=":", linewidth=1.0, alpha=0.6, label="0.40 Floor")

        rho_str = f"Spearman $\\rho$: {rho_dict['spearman_rho']:+.3f}\n95% CI: {rho_dict['ci_95']}"
        ax.text(
            0.05, 0.92,
            rho_str,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray")
        )

        ax.set_title(title_text, fontsize=11.5, fontweight="semibold")
        ax.set_ylabel("RAG Delta (RAG - Zero-Shot)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        if row == 1:
            ax.set_xlabel("Top-1 Retrieval Cosine Similarity", fontsize=10)
        if row == 0 and col == 0:
            ax.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close()
    print(f"[Dose-Response] Saved figure to {OUTPUT_PNG}")

    print("\n" + summary_text)
    return results_payload


if __name__ == "__main__":
    run_dose_response()
