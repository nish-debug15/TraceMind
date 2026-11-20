"""
TraceMind Judged Retrieval Relevance Analysis (Block 6)
-------------------------------------------------------
Analyzes 279 judged relevance pairs across 93 test incidents.
Outputs eval/results/relevance_analysis.json with:
a) Relevance@1 and best-of-top-3 (score=2, score>=1) with company-clustered 95% CIs
b) Chroma similarity predicting relevance: Spearman rho with clustered 95% CI, AUROC (score >= 1)
c) RAG gain stratified by judged relevance (scores 0, 1, 2) for ROUGE-L, BERTScore,
   and remediation ROUGE-L (parse-conditional), with clustered CIs if n_companies >= 5
d) Guardrail check: Rank-1 judged relevance distribution for Abstained (N=7) vs Answered (N=86)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JUDGMENTS_FILE = PROJECT_ROOT / "eval" / "results" / "relevance_judgments.json"
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
RAG_EVAL_FILE = PROJECT_ROOT / "eval" / "results" / "rag_vs_zero_shot.json"
REMEDIATION_FILE = PROJECT_ROOT / "eval" / "results" / "remediation_eval.json"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "relevance_analysis.json"


def cluster_bootstrap_ci(company_to_vals, stat_fn, n_resamples=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    companies = list(company_to_vals.keys())
    n_companies = len(companies)
    
    boot_stats = []
    for _ in range(n_resamples):
        sampled_comps = rng.choice(companies, size=n_companies, replace=True)
        sample = []
        for c in sampled_comps:
            vals = company_to_vals[c]
            if len(vals) == 1:
                sample.extend(vals)
            else:
                idx = rng.choice(len(vals), size=len(vals), replace=True)
                sample.extend([vals[i] for i in idx])
        boot_stats.append(stat_fn(sample))
        
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1.0 - alpha) * 100))
    return [round(lower, 4), round(upper, 4)]


def analyze_relevance():
    print("=" * 60)
    print("TraceMind Relevance Analysis (Block 6 - EXPLORATORY)")
    print("=" * 60)

    with open(JUDGMENTS_FILE, "r", encoding="utf-8") as f:
        judgements_data = json.load(f)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = {r["incident_id"]: r for r in json.load(f)}

    with open(RAG_EVAL_FILE, "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    with open(REMEDIATION_FILE, "r", encoding="utf-8") as f:
        rem_data = json.load(f)

    # Index RAG vs Zero-Shot root cause scores by incident_id
    rc_deltas = {}
    for item in rag_data.get("per_incident_scores", []):
        iid = item["incident_id"]
        d = item.get("delta", {})
        rc_deltas[iid] = {
            "rougeL_delta": d.get("rougeL_f1", 0.0),
            "bert_score_delta": d.get("bert_score_f1", 0.0),
            "similarity_score": item.get("retrieval", {}).get("similarity_score", 0.0)
        }

    # Index remediation parse-conditional deltas
    cond_parsed = rem_data.get("conditional_on_both_parsed", {}).get("baseline_format", {})
    cond_parsed_ids = set(cond_parsed.get("incident_ids", []))
    
    # Load cached remediation generations for exact per-incident ROUGE-L delta
    rag_gens_file = PROJECT_ROOT / "eval" / "cache" / "rag_generations.json"
    zero_gens_file = PROJECT_ROOT / "eval" / "cache" / "zeroshot_generations.json"
    
    with open(rag_gens_file, "r", encoding="utf-8") as f:
        rag_gens = json.load(f)
    with open(zero_gens_file, "r", encoding="utf-8") as f:
        zero_gens = json.load(f)

    from rouge_score import rouge_scorer
    rem_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rem_per_incident = {}
    for iid in cond_parsed_ids:
        gt_rem = test_records[iid].get("remediation_steps", "")
        r_rem = rag_gens[iid].get("remediation_steps", "")
        z_rem = zero_gens[iid].get("remediation_steps", "")
        r_rl = rem_scorer.score(gt_rem, r_rem)["rougeL"].fmeasure
        z_rl = rem_scorer.score(gt_rem, z_rem)["rougeL"].fmeasure
        rem_per_incident[iid] = {
            "rem_rougeL_delta": r_rl - z_rl
        }

    # Extract query-level data
    queries_data = []
    for q in judgements_data["judgments"]:
        q_id = q["query_incident_id"]
        q_comp = q["query_company"]
        hits = q["hits"]
        
        r1_hit = hits[0]
        r1_score = r1_hit["score"]
        r1_sim = r1_hit["similarity_score"]
        best_score = max(h["score"] for h in hits)
        
        delta_info = rc_deltas.get(q_id, {})
        rem_info = rem_per_incident.get(q_id, {})
        is_cond_parsed = q_id in cond_parsed_ids
        
        queries_data.append({
            "incident_id": q_id,
            "company": q_comp,
            "r1_score": r1_score,
            "r1_sim": r1_sim,
            "best_score": best_score,
            "r1_is_score_2": 1 if r1_score == 2 else 0,
            "r1_is_score_ge_1": 1 if r1_score >= 1 else 0,
            "best_is_score_2": 1 if best_score == 2 else 0,
            "best_is_score_ge_1": 1 if best_score >= 1 else 0,
            "rougeL_delta": delta_info.get("rougeL_delta", 0.0),
            "bert_score_delta": delta_info.get("bert_score_delta", 0.0),
            "is_cond_parsed": is_cond_parsed,
            "rem_rougeL_delta": rem_info.get("rem_rougeL_delta", 0.0) if is_cond_parsed else None
        })

    n_total = len(queries_data)
    print(f"Total test queries analyzed: {n_total}")

    # -------------------------------------------------------------
    # (a) Relevance@1 and Best-of-Top-3 with company-clustered 95% CIs
    # -------------------------------------------------------------
    metrics_a = [
        ("relevance_at_1_score_2", "r1_is_score_2"),
        ("relevance_at_1_score_ge_1", "r1_is_score_ge_1"),
        ("best_of_3_score_2", "best_is_score_2"),
        ("best_of_3_score_ge_1", "best_is_score_ge_1")
    ]

    relevance_rates = {}
    for m_key, f_key in metrics_a:
        comp_map = defaultdict(list)
        all_vals = []
        for q in queries_data:
            val = q[f_key]
            comp_map[q["company"]].append(val)
            all_vals.append(val)
            
        mean_val = float(np.mean(all_vals))
        ci_95 = cluster_bootstrap_ci(comp_map, lambda s: float(np.mean(s)))
        relevance_rates[m_key] = {
            "fraction": round(mean_val, 4),
            "percentage": round(mean_val * 100.0, 2),
            "count": int(np.sum(all_vals)),
            "total": n_total,
            "company_clustered_ci_95": ci_95
        }
        print(f"  {m_key}: {mean_val:.4f} ({mean_val*100:.1f}%) [95% CI: {ci_95}]")

    # -------------------------------------------------------------
    # (b) Does Chroma similarity predict relevance?
    # -------------------------------------------------------------
    top1_sims = [q["r1_sim"] for q in queries_data]
    top1_scores = [q["r1_score"] for q in queries_data]
    top1_binary = [q["r1_is_score_ge_1"] for q in queries_data]

    rho_stat = float(spearmanr(top1_sims, top1_scores).statistic)
    
    # Clustered bootstrap for Spearman rho
    comp_rho_map = defaultdict(list)
    for q in queries_data:
        comp_rho_map[q["company"]].append((q["r1_sim"], q["r1_score"]))

    def calc_spearman(pairs):
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        res = spearmanr(xs, ys).statistic
        return 0.0 if np.isnan(res) else float(res)

    rho_ci = cluster_bootstrap_ci(comp_rho_map, calc_spearman)
    
    # AUROC of top-1 sim predicting judged relevance >= 1
    auroc = float(roc_auc_score(top1_binary, top1_sims))
    
    similarity_prediction = {
        "spearman_rho": round(rho_stat, 4),
        "spearman_rho_clustered_ci_95": rho_ci,
        "rho_excludes_zero": bool(rho_ci[0] > 0 or rho_ci[1] < 0),
        "auroc_relevance_ge_1": round(auroc, 4)
    }
    print(f"\nSimilarity prediction:")
    print(f"  Spearman rho: {rho_stat:.4f} [95% CI: {rho_ci}], AUROC: {auroc:.4f}")

    # -------------------------------------------------------------
    # (c) RAG gain stratified by judged relevance (scores 0, 1, 2)
    # -------------------------------------------------------------
    stratified_gain = {}
    for score_level in [0, 1, 2]:
        slice_queries = [q for q in queries_data if q["r1_score"] == score_level]
        n_slice = len(slice_queries)
        comps_in_slice = set(q["company"] for q in slice_queries)
        n_comps = len(comps_in_slice)
        is_low_n = n_comps < 5

        slice_res = {
            "n_incidents": n_slice,
            "n_companies": n_comps,
            "low_n": is_low_n
        }

        # ROUGE-L delta
        rl_vals = [q["rougeL_delta"] for q in slice_queries]
        rl_mean = float(np.mean(rl_vals)) if rl_vals else 0.0
        slice_res["rougeL_delta"] = {"mean": round(rl_mean, 4)}
        if not is_low_n:
            comp_rl = defaultdict(list)
            for q in slice_queries:
                comp_rl[q["company"]].append(q["rougeL_delta"])
            slice_res["rougeL_delta"]["ci_95"] = cluster_bootstrap_ci(comp_rl, lambda s: float(np.mean(s)))
        else:
            slice_res["rougeL_delta"]["ci_95"] = None

        # BERTScore delta
        bs_vals = [q["bert_score_delta"] for q in slice_queries]
        bs_mean = float(np.mean(bs_vals)) if bs_vals else 0.0
        slice_res["bert_score_delta"] = {"mean": round(bs_mean, 4)}
        if not is_low_n:
            comp_bs = defaultdict(list)
            for q in slice_queries:
                comp_bs[q["company"]].append(q["bert_score_delta"])
            slice_res["bert_score_delta"]["ci_95"] = cluster_bootstrap_ci(comp_bs, lambda s: float(np.mean(s)))
        else:
            slice_res["bert_score_delta"]["ci_95"] = None

        # Remediation ROUGE-L delta (conditional on both parsed)
        rem_parsed_slice = [q for q in slice_queries if q["is_cond_parsed"]]
        n_rem_parsed = len(rem_parsed_slice)
        n_rem_comps = len(set(q["company"] for q in rem_parsed_slice))
        rem_low_n = n_rem_comps < 5
        rem_vals = [q["rem_rougeL_delta"] for q in rem_parsed_slice]
        rem_mean = float(np.mean(rem_vals)) if rem_vals else 0.0

        slice_res["remediation_conditional_rougeL_delta"] = {
            "n_parsed": n_rem_parsed,
            "n_companies": n_rem_comps,
            "low_n": rem_low_n,
            "mean": round(rem_mean, 4)
        }
        if not rem_low_n:
            comp_rem = defaultdict(list)
            for q in rem_parsed_slice:
                comp_rem[q["company"]].append(q["rem_rougeL_delta"])
            slice_res["remediation_conditional_rougeL_delta"]["ci_95"] = cluster_bootstrap_ci(comp_rem, lambda s: float(np.mean(s)))
        else:
            slice_res["remediation_conditional_rougeL_delta"]["ci_95"] = None

        stratified_gain[f"score_{score_level}"] = slice_res
        print(f"\nStratified Score {score_level} (N={n_slice}, comps={n_comps}, low_n={is_low_n}):")
        print(f"  ROUGE-L Delta: {rl_mean:+.4f} (CI: {slice_res['rougeL_delta']['ci_95']})")
        print(f"  BERTScore Delta: {bs_mean:+.4f} (CI: {slice_res['bert_score_delta']['ci_95']})")
        print(f"  Remediation Delta (Parsed N={n_rem_parsed}): {rem_mean:+.4f}")

    # -------------------------------------------------------------
    # (d) Guardrail check: Abstained (sim < 0.40) vs Answered (sim >= 0.40)
    # -------------------------------------------------------------
    abstained_queries = [q for q in queries_data if q["r1_sim"] < 0.40]
    answered_queries = [q for q in queries_data if q["r1_sim"] >= 0.40]

    def get_score_dist(q_list):
        scores = [q["r1_score"] for q in q_list]
        n_q = len(q_list)
        return {
            "n": n_q,
            "score_0": {"count": scores.count(0), "pct": round(scores.count(0) / n_q * 100, 2) if n_q else 0.0},
            "score_1": {"count": scores.count(1), "pct": round(scores.count(1) / n_q * 100, 2) if n_q else 0.0},
            "score_2": {"count": scores.count(2), "pct": round(scores.count(2) / n_q * 100, 2) if n_q else 0.0},
            "relevance_ge_1_pct": round(sum(1 for s in scores if s >= 1) / n_q * 100, 2) if n_q else 0.0
        }

    guardrail_check = {
        "abstained_n7": get_score_dist(abstained_queries),
        "answered_n86": get_score_dist(answered_queries),
        "difference_relevance_ge_1_pct": round(
            get_score_dist(answered_queries)["relevance_ge_1_pct"] - get_score_dist(abstained_queries)["relevance_ge_1_pct"], 2
        )
    }

    print(f"\nGuardrail Check:")
    print(f"  Abstained (N={len(abstained_queries)}): {guardrail_check['abstained_n7']}")
    print(f"  Answered  (N={len(answered_queries)}): {guardrail_check['answered_n86']}")

    output_data = {
        "exploratory": True,
        "metadata": {
            "num_queries": n_total,
            "num_companies": len(set(q["company"] for q in queries_data)),
            "bootstrap_resamples": 10000,
            "seed": 42
        },
        "relevance_rates": relevance_rates,
        "similarity_prediction": similarity_prediction,
        "stratified_gain_by_relevance": stratified_gain,
        "guardrail_check": guardrail_check
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Artifact] Saved relevance analysis to {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    analyze_relevance()
