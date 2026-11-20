"""
eval/multiplicity_adjusted.py
------------------------------
Applies Bonferroni multiplicity adjustments within two pre-specified families:
- Family 1: Root-cause deltas (m = 5: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, cosine).
  Family-wise alpha = 0.05 / 5 = 0.01 -> 99% CI.
- Family 2: Remediation deltas (m = 2: ROUGE-L, BERTScore).
  Family-wise alpha = 0.05 / 2 = 0.025 -> 97.5% CI.

Computes two-stage company-clustered bootstrap intervals (10,000 resamples, seed 42)
for both unadjusted (95%) and Bonferroni-adjusted levels simultaneously from the
same bootstrap distribution.

Saves results to eval/results/multiplicity_adjusted.json with:
- unadjusted_ci_95: [low, high]
- adjusted_ci: [low, high]
- excludes_zero_unadjusted: bool
- excludes_zero_after_adjustment: bool
- status: 'excludes zero' | 'unadjusted only' | 'crosses zero'
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
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
OUTPUT_FILE = RESULTS_DIR / "multiplicity_adjusted.json"


def two_stage_cluster_bootstrap_multi_ci(
    inc_ids: List[str],
    vals: List[float],
    id_to_company: Dict[str, str],
    n_resamples: int = 10000,
    seed: int = 42,
    unadjusted_alpha: float = 0.05,
    adjusted_alpha: float = 0.01,
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

    orig_mean = float(np.mean(vals))

    # Unadjusted CI
    unadj_low_p = (unadjusted_alpha / 2.0) * 100
    unadj_high_p = (1.0 - unadjusted_alpha / 2.0) * 100
    unadj_low = float(np.percentile(resampled_means, unadj_low_p))
    unadj_high = float(np.percentile(resampled_means, unadj_high_p))
    unadj_excludes_zero = bool(unadj_low > 0.0 or unadj_high < 0.0)

    # Adjusted CI
    adj_low_p = (adjusted_alpha / 2.0) * 100
    adj_high_p = (1.0 - adjusted_alpha / 2.0) * 100
    adj_low = float(np.percentile(resampled_means, adj_low_p))
    adj_high = float(np.percentile(resampled_means, adj_high_p))
    adj_excludes_zero = bool(adj_low > 0.0 or adj_high < 0.0)

    if adj_excludes_zero:
        status = "excludes zero"
    elif unadj_excludes_zero and not adj_excludes_zero:
        status = "unadjusted only"
    else:
        status = "crosses zero"

    return {
        "mean": round(orig_mean, 4),
        "unadjusted_ci_95": [round(unadj_low, 4), round(unadj_high, 4)],
        "excludes_zero_unadjusted": unadj_excludes_zero,
        "adjusted_alpha": adjusted_alpha,
        "adjusted_ci_level": round(1.0 - adjusted_alpha, 4),
        "adjusted_ci": [round(adj_low, 4), round(adj_high, 4)],
        "excludes_zero_after_adjustment": adj_excludes_zero,
        "status": status,
    }


def compute_multiplicity_adjusted() -> Dict[str, Any]:
    print("[Multiplicity Adjustment] Loading test records and evaluation results...", flush=True)
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in test_records}
    test_ids = [r["incident_id"] for r in test_records]

    results = {
        "metadata": {
            "num_incidents": len(test_records),
            "num_unique_companies": len(set(id_to_company.values())),
            "bootstrap_resamples": 10000,
            "seed": 42,
            "method": "two-stage company-clustered bootstrap with Bonferroni correction",
        },
        "family_1_root_cause": {
            "description": "Root-cause paired deltas (RAG - Zero-Shot, Baseline Format). m = 5 tests. Alpha = 0.05 / 5 = 0.01 (99% CI).",
            "family_size_m": 5,
            "family_alpha": 0.01,
            "ci_level": 0.99,
            "tests": {},
        },
        "family_2_remediation": {
            "description": "Remediation paired deltas (RAG - Zero-Shot). m = 2 tests per comparison (ROUGE-L, BERTScore). Alpha = 0.05 / 2 = 0.025 (97.5% CI).",
            "family_size_m": 2,
            "family_alpha": 0.025,
            "ci_level": 0.975,
            "baseline_format_unconditional": {},
            "baseline_format_conditional": {},
            "gate_v2_format_unconditional": {},
            "gate_v2_format_conditional": {},
        },
        "summary": {},
    }

    # =========================================================================
    # Family 1: Root-Cause Deltas (m = 5: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, cosine)
    # =========================================================================
    rvz_file = RESULTS_DIR / "rag_vs_zero_shot.json"
    mci_file = RESULTS_DIR / "metrics_with_ci.json"

    with open(rvz_file, "r", encoding="utf-8") as f:
        rvz_data = json.load(f)
    with open(mci_file, "r", encoding="utf-8") as f:
        mci_data = json.load(f)

    per_inc = rvz_data["per_incident_scores"]
    inc_ids = [e["incident_id"] for e in per_inc]
    bs_map = {item["incident_id"]: item["delta_bert_score_f1"] for item in mci_data.get("per_incident_bert_scores", [])}

    family1_metrics = [
        ("rouge1_f1", [e["delta"]["rouge1_f1"] for e in per_inc]),
        ("rouge2_f1", [e["delta"]["rouge2_f1"] for e in per_inc]),
        ("rougeL_f1", [e["delta"]["rougeL_f1"] for e in per_inc]),
        ("bert_score_f1", [bs_map.get(i, 0.0) for i in inc_ids]),
        ("semantic_sim", [e["delta"]["semantic_sim"] for e in per_inc]),
    ]

    print("\nComputing Family 1 (Root Cause Deltas, m=5, 99% CI)...", flush=True)
    for metric_name, diff_vals in family1_metrics:
        ci_res = two_stage_cluster_bootstrap_multi_ci(
            inc_ids=inc_ids,
            vals=diff_vals,
            id_to_company=id_to_company,
            n_resamples=10000,
            seed=42,
            unadjusted_alpha=0.05,
            adjusted_alpha=0.01,
        )
        results["family_1_root_cause"]["tests"][metric_name] = ci_res
        print(f"  {metric_name:<15}: delta={ci_res['mean']:+.4f}, unadj 95%={ci_res['unadjusted_ci_95']}, adj 99%={ci_res['adjusted_ci']}, status='{ci_res['status']}'", flush=True)

    # =========================================================================
    # Family 2: Remediation Deltas (m = 2: ROUGE-L, BERTScore)
    # =========================================================================
    print("\nLoading remediation predictions and scoring...", flush=True)
    from rouge_score import rouge_scorer
    from bert_score import score as calc_bertscore

    cache_base_z = json.load(open(CACHE_DIR / "zeroshot_generations.json", encoding="utf-8"))
    cache_base_r = json.load(open(CACHE_DIR / "rag_generations.json", encoding="utf-8"))
    cache_gate_z = json.load(open(CACHE_DIR / "ablation_zero_shot_gate_format.json", encoding="utf-8"))
    cache_gate_r = json.load(open(CACHE_DIR / "ablation_relevance_gate_v2.json", encoding="utf-8"))

    gt_rem_map = {r["incident_id"]: (r.get("remediation_steps") or "").strip() for r in test_records}
    eligible_ids = [i for i in test_ids if len(gt_rem_map[i].split()) >= 5]
    gt_rem_texts = [gt_rem_map[i] for i in eligible_ids]

    rem_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def extract_preds(cache_dict):
        return [cache_dict.get(i, {}).get("remediation_steps", "").strip() for i in eligible_ids]

    preds_bz = extract_preds(cache_base_z)
    preds_br = extract_preds(cache_base_r)
    preds_gz = extract_preds(cache_gate_z)
    preds_gr = extract_preds(cache_gate_r)

    # Unconditional ROUGE-L
    bz_rl_u = [rem_scorer.score(gt, p)["rougeL"].fmeasure if p else 0.0 for gt, p in zip(gt_rem_texts, preds_bz)]
    br_rl_u = [rem_scorer.score(gt, p)["rougeL"].fmeasure if p else 0.0 for gt, p in zip(gt_rem_texts, preds_br)]
    gz_rl_u = [rem_scorer.score(gt, p)["rougeL"].fmeasure if p else 0.0 for gt, p in zip(gt_rem_texts, preds_gz)]
    gr_rl_u = [rem_scorer.score(gt, p)["rougeL"].fmeasure if p else 0.0 for gt, p in zip(gt_rem_texts, preds_gr)]

    # Unconditional BERTScore (computed once per cell)
    print("  Calculating BERTScore for remediation cells...", flush=True)
    safe_gt = [g if g else "none" for g in gt_rem_texts]
    _, _, bs_bz_t = calc_bertscore([p if p else "none" for p in preds_bz], safe_gt, model_type="bert-base-uncased", verbose=False)
    _, _, bs_br_t = calc_bertscore([p if p else "none" for p in preds_br], safe_gt, model_type="bert-base-uncased", verbose=False)
    _, _, bs_gz_t = calc_bertscore([p if p else "none" for p in preds_gz], safe_gt, model_type="bert-base-uncased", verbose=False)
    _, _, bs_gr_t = calc_bertscore([p if p else "none" for p in preds_gr], safe_gt, model_type="bert-base-uncased", verbose=False)

    bz_bs_u = bs_bz_t.numpy().tolist()
    br_bs_u = bs_br_t.numpy().tolist()
    gz_bs_u = bs_gz_t.numpy().tolist()
    gr_bs_u = bs_gr_t.numpy().tolist()

    # Unconditional diffs
    base_u_rl_diff = [r - z for r, z in zip(br_rl_u, bz_rl_u)]
    base_u_bs_diff = [r - z for r, z in zip(br_bs_u, bz_bs_u)]
    gate_u_rl_diff = [r - z for r, z in zip(gr_rl_u, gz_rl_u)]
    gate_u_bs_diff = [r - z for r, z in zip(gr_bs_u, gz_bs_u)]

    print("\nComputing Family 2 (Remediation Unconditional, m=2, 97.5% CI)...", flush=True)
    results["family_2_remediation"]["baseline_format_unconditional"]["rougeL_delta"] = two_stage_cluster_bootstrap_multi_ci(
        eligible_ids, base_u_rl_diff, id_to_company, n_resamples=10000, seed=42, unadjusted_alpha=0.05, adjusted_alpha=0.025
    )
    results["family_2_remediation"]["baseline_format_unconditional"]["bert_score_delta"] = two_stage_cluster_bootstrap_multi_ci(
        eligible_ids, base_u_bs_diff, id_to_company, n_resamples=10000, seed=42, unadjusted_alpha=0.05, adjusted_alpha=0.025
    )
    results["family_2_remediation"]["gate_v2_format_unconditional"]["rougeL_delta"] = two_stage_cluster_bootstrap_multi_ci(
        eligible_ids, gate_u_rl_diff, id_to_company, n_resamples=10000, seed=42, unadjusted_alpha=0.05, adjusted_alpha=0.025
    )
    results["family_2_remediation"]["gate_v2_format_unconditional"]["bert_score_delta"] = two_stage_cluster_bootstrap_multi_ci(
        eligible_ids, gate_u_bs_diff, id_to_company, n_resamples=10000, seed=42, unadjusted_alpha=0.05, adjusted_alpha=0.025
    )

    # Conditional on both parsed
    both_base_ids = [
        i for i in eligible_ids
        if cache_base_z.get(i, {}).get("remediation_steps", "").strip() and cache_base_r.get(i, {}).get("remediation_steps", "").strip()
    ]
    both_gate_ids = [
        i for i in eligible_ids
        if cache_gate_z.get(i, {}).get("remediation_steps", "").strip() and cache_gate_r.get(i, {}).get("remediation_steps", "").strip()
    ]

    base_c_rl_diff = [
        rem_scorer.score(gt_rem_map[i], cache_base_r[i]["remediation_steps"].strip())["rougeL"].fmeasure -
        rem_scorer.score(gt_rem_map[i], cache_base_z[i]["remediation_steps"].strip())["rougeL"].fmeasure
        for i in both_base_ids
    ]
    gt_base_c = [gt_rem_map[i] for i in both_base_ids]
    _, _, bs_bz_c = calc_bertscore([cache_base_z[i]["remediation_steps"].strip() for i in both_base_ids], gt_base_c, model_type="bert-base-uncased", verbose=False)
    _, _, bs_br_c = calc_bertscore([cache_base_r[i]["remediation_steps"].strip() for i in both_base_ids], gt_base_c, model_type="bert-base-uncased", verbose=False)
    base_c_bs_diff = (bs_br_c - bs_bz_c).numpy().tolist()

    gate_c_rl_diff = [
        rem_scorer.score(gt_rem_map[i], cache_gate_r[i]["remediation_steps"].strip())["rougeL"].fmeasure -
        rem_scorer.score(gt_rem_map[i], cache_gate_z[i]["remediation_steps"].strip())["rougeL"].fmeasure
        for i in both_gate_ids
    ]
    gt_gate_c = [gt_rem_map[i] for i in both_gate_ids]
    _, _, bs_gz_c = calc_bertscore([cache_gate_z[i]["remediation_steps"].strip() for i in both_gate_ids], gt_gate_c, model_type="bert-base-uncased", verbose=False)
    _, _, bs_gr_c = calc_bertscore([cache_gate_r[i]["remediation_steps"].strip() for i in both_gate_ids], gt_gate_c, model_type="bert-base-uncased", verbose=False)
    gate_c_bs_diff = (bs_gr_c - bs_gz_c).numpy().tolist()

    print("\nComputing Family 2 (Remediation Conditional on Both Parsed, m=2, 97.5% CI)...", flush=True)
    results["family_2_remediation"]["baseline_format_conditional"]["rougeL_delta"] = two_stage_cluster_bootstrap_multi_ci(
        both_base_ids, base_c_rl_diff, id_to_company, n_resamples=10000, seed=42, unadjusted_alpha=0.05, adjusted_alpha=0.025
    )
    results["family_2_remediation"]["baseline_format_conditional"]["bert_score_delta"] = two_stage_cluster_bootstrap_multi_ci(
        both_base_ids, base_c_bs_diff, id_to_company, n_resamples=10000, seed=42, unadjusted_alpha=0.05, adjusted_alpha=0.025
    )
    results["family_2_remediation"]["gate_v2_format_conditional"]["rougeL_delta"] = two_stage_cluster_bootstrap_multi_ci(
        both_gate_ids, gate_c_rl_diff, id_to_company, n_resamples=10000, seed=42, unadjusted_alpha=0.05, adjusted_alpha=0.025
    )
    results["family_2_remediation"]["gate_v2_format_conditional"]["bert_score_delta"] = two_stage_cluster_bootstrap_multi_ci(
        both_gate_ids, gate_c_bs_diff, id_to_company, n_resamples=10000, seed=42, unadjusted_alpha=0.05, adjusted_alpha=0.025
    )

    # Summary
    f1 = results["family_1_root_cause"]["tests"]
    f2_u = results["family_2_remediation"]["baseline_format_unconditional"]
    f2_c = results["family_2_remediation"]["baseline_format_conditional"]

    summary_text = (
        f"Multiplicity-adjusted inference across 2 pre-specified families (10,000 company-clustered resamples, seed 42):\n"
        f"1. Family 1 (Root-cause paired deltas, baseline format, m=5, Bonferroni alpha=0.01 / 99% CI):\n"
        f"   - ROUGE-1 F1: delta = {f1['rouge1_f1']['mean']:+.4f}, unadj 95% {f1['rouge1_f1']['unadjusted_ci_95']}, adj 99% {f1['rouge1_f1']['adjusted_ci']} -> status: '{f1['rouge1_f1']['status']}'\n"
        f"   - ROUGE-2 F1: delta = {f1['rouge2_f1']['mean']:+.4f}, unadj 95% {f1['rouge2_f1']['unadjusted_ci_95']}, adj 99% {f1['rouge2_f1']['adjusted_ci']} -> status: '{f1['rouge2_f1']['status']}'\n"
        f"   - ROUGE-L F1: delta = {f1['rougeL_f1']['mean']:+.4f}, unadj 95% {f1['rougeL_f1']['unadjusted_ci_95']}, adj 99% {f1['rougeL_f1']['adjusted_ci']} -> status: '{f1['rougeL_f1']['status']}'\n"
        f"   - BERTScore F1: delta = {f1['bert_score_f1']['mean']:+.4f}, unadj 95% {f1['bert_score_f1']['unadjusted_ci_95']}, adj 99% {f1['bert_score_f1']['adjusted_ci']} -> status: '{f1['bert_score_f1']['status']}'\n"
        f"   - Cosine Sim: delta = {f1['semantic_sim']['mean']:+.4f}, unadj 95% {f1['semantic_sim']['unadjusted_ci_95']}, adj 99% {f1['semantic_sim']['adjusted_ci']} -> status: '{f1['semantic_sim']['status']}'\n"
        f"2. Family 2 (Remediation paired deltas, m=2, Bonferroni alpha=0.025 / 97.5% CI):\n"
        f"   - Baseline unconditional ROUGE-L: delta = {f2_u['rougeL_delta']['mean']:+.4f}, unadj 95% {f2_u['rougeL_delta']['unadjusted_ci_95']}, adj 97.5% {f2_u['rougeL_delta']['adjusted_ci']} -> status: '{f2_u['rougeL_delta']['status']}'\n"
        f"   - Baseline unconditional BERTScore: delta = {f2_u['bert_score_delta']['mean']:+.4f}, unadj 95% {f2_u['bert_score_delta']['unadjusted_ci_95']}, adj 97.5% {f2_u['bert_score_delta']['adjusted_ci']} -> status: '{f2_u['bert_score_delta']['status']}'\n"
        f"   - Baseline conditional ROUGE-L (n={len(both_base_ids)}): delta = {f2_c['rougeL_delta']['mean']:+.4f}, unadj 95% {f2_c['rougeL_delta']['unadjusted_ci_95']}, adj 97.5% {f2_c['rougeL_delta']['adjusted_ci']} -> status: '{f2_c['rougeL_delta']['status']}'\n"
        f"   - Baseline conditional BERTScore (n={len(both_base_ids)}): delta = {f2_c['bert_score_delta']['mean']:+.4f}, unadj 95% {f2_c['bert_score_delta']['unadjusted_ci_95']}, adj 97.5% {f2_c['bert_score_delta']['adjusted_ci']} -> status: '{f2_c['bert_score_delta']['status']}'\n"
    )
    results["summary"] = summary_text

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Artifact] Saved multiplicity adjusted results to {OUTPUT_FILE}", flush=True)
    print("\n" + summary_text, flush=True)
    return results


if __name__ == "__main__":
    compute_multiplicity_adjusted()
