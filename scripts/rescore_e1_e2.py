"""
TraceMind Rescoring Script (Block 2)
------------------------------------
Applies eval/scorer.py single canonical scorer across:
- E1: eval/results/random_precedent_control.json -> eval/results/random_precedent_control_rescored.json
- E2: eval/results/loo_regime_results.json -> eval/results/loo_regime_results_rescored.json

Prints a report of every cell and metric whose value changed vs previously committed JSONs.
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

from eval.scorer import score_cell, word_count, compute_bert_scores, _ROUGE_SCORER
from eval.analyze_e1 import FastClusterBootstrap, determine_verdict

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

ORIG_E1_FILE = RESULTS_DIR / "random_precedent_control.json"
RESC_E1_FILE = RESULTS_DIR / "random_precedent_control_rescored.json"

ORIG_E2_FILE = RESULTS_DIR / "loo_regime_results.json"
RESC_E2_FILE = RESULTS_DIR / "loo_regime_results_rescored.json"


def rescore_arm(
    test_ids: List[str],
    gt_rc_map: Dict[str, str],
    pred_rc_map: Dict[str, str],
    median_gt_limit: int,
    bootstrapper: FastClusterBootstrap,
) -> Dict[str, Any]:
    preds = [pred_rc_map[i] for i in test_ids]
    refs = [gt_rc_map[i] for i in test_ids]

    scored = score_cell(preds, refs)
    per_inc = scored["per_incident"]

    # Length-matched ROUGE-L
    lm_preds = [" ".join(p.split()[:median_gt_limit]) for p in preds]
    lm_rl = [_ROUGE_SCORER.score(ref or "none", lmp or "none")["rougeL"].fmeasure for ref, lmp in zip(refs, lm_preds)]

    wc_list = per_inc["word_count"]

    return {
        "rouge1_f1": bootstrapper.compute_multi_ci(per_inc["rouge1_f1"]),
        "rouge2_f1": bootstrapper.compute_multi_ci(per_inc["rouge2_f1"]),
        "rougeL_f1": bootstrapper.compute_multi_ci(per_inc["rougeL_f1"]),
        "length_matched_rougeL": bootstrapper.compute_multi_ci(lm_rl),
        "bert_score_f1": bootstrapper.compute_multi_ci(per_inc["bert_score_f1"]),
        "cosine": bootstrapper.compute_multi_ci(per_inc["cosine"]),
        "word_count": {
            "mean": round(float(np.mean(wc_list)), 2),
            "median": round(float(np.median(wc_list)), 2),
        },
        "_raw": {
            "rouge1_f1": per_inc["rouge1_f1"],
            "rouge2_f1": per_inc["rouge2_f1"],
            "rougeL_f1": per_inc["rougeL_f1"],
            "length_matched_rougeL": lm_rl,
            "bert_score_f1": per_inc["bert_score_f1"],
            "cosine": per_inc["cosine"],
        }
    }


def compute_paired_deltas(
    arm_a_raw: Dict[str, List[float]],
    arm_b_raw: Dict[str, List[float]],
    bootstrapper: FastClusterBootstrap,
) -> Dict[str, Any]:
    deltas = {}
    metrics = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "length_matched_rougeL", "bert_score_f1", "cosine"]
    for m in metrics:
        diff = [a - b for a, b in zip(arm_a_raw[m], arm_b_raw[m])]
        deltas[m] = bootstrapper.compute_multi_ci(diff)
    return deltas


def evaluate_remediation_pairwise_rescored(
    test_ids: List[str],
    id_to_company: Dict[str, str],
    gt_rem_map: Dict[str, str],
    preds_a: Dict[str, str],
    preds_b: Dict[str, str],
) -> Dict[str, Any]:
    both_ids = [
        i for i in test_ids
        if preds_a.get(i, "").strip() and preds_b.get(i, "").strip() and len(gt_rem_map.get(i, "").split()) >= 5
    ]
    if not both_ids:
        return {"n_both_parsed": 0, "status": "no incidents parsed in both arms"}

    sub_bootstrapper = FastClusterBootstrap(both_ids, id_to_company, n_resamples=10000, seed=42)

    a_texts = [preds_a[i].strip() for i in both_ids]
    b_texts = [preds_b[i].strip() for i in both_ids]
    gts = [gt_rem_map[i].strip() for i in both_ids]

    rl_a = [_ROUGE_SCORER.score(g, p)["rougeL"].fmeasure for g, p in zip(gts, a_texts)]
    rl_b = [_ROUGE_SCORER.score(g, p)["rougeL"].fmeasure for g, p in zip(gts, b_texts)]
    rl_diff = [a - b for a, b in zip(rl_a, rl_b)]

    bs_a = compute_bert_scores(a_texts, gts)
    bs_b = compute_bert_scores(b_texts, gts)
    bs_diff = [a - b for a, b in zip(bs_a, bs_b)]

    return {
        "n_both_parsed": len(both_ids),
        "arm_a_rougeL": sub_bootstrapper.compute_multi_ci(rl_a),
        "arm_b_rougeL": sub_bootstrapper.compute_multi_ci(rl_b),
        "arm_a_bert_score": sub_bootstrapper.compute_multi_ci(bs_a),
        "arm_b_bert_score": sub_bootstrapper.compute_multi_ci(bs_b),
        "delta_rougeL": sub_bootstrapper.compute_multi_ci(rl_diff),
        "delta_bert_score": sub_bootstrapper.compute_multi_ci(bs_diff),
    }


def run_rescoring():
    print("=" * 80)
    print("RE-SCORING E1 AND E2 WITH CANONICAL SCORER (BLOCK 2)")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Rescore E1
    # -------------------------------------------------------------
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    test_ids = [r["incident_id"] for r in test_records]
    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in test_records}
    gt_rc_map = {r["incident_id"]: r["root_cause"] for r in test_records}
    gt_rem_map = {r["incident_id"]: r.get("remediation_steps", "") for r in test_records}

    gt_word_counts = [word_count(gt_rc_map[i]) for i in test_ids]
    median_gt_limit = int(round(float(np.median(gt_word_counts))))

    bootstrapper = FastClusterBootstrap(test_ids, id_to_company, n_resamples=10000, seed=42)

    with open(CACHE_DIR / "zeroshot_generations.json", "r", encoding="utf-8") as f:
        cell_base_zs = json.load(f)
    with open(CACHE_DIR / "rag_generations.json", "r", encoding="utf-8") as f:
        cell_base_rag = json.load(f)
    with open(CACHE_DIR / "ablation_zero_shot_gate_format.json", "r", encoding="utf-8") as f:
        cell_gate_zs = json.load(f)
    with open(CACHE_DIR / "ablation_relevance_gate_v2.json", "r", encoding="utf-8") as f:
        cell_gate_rag = json.load(f)
    with open(RESULTS_DIR / "random_precedent_generations.json", "r", encoding="utf-8") as f:
        rand_gens = {g["query_incident_id"]: g for g in json.load(f)["generations"]}

    loaded_gens = {
        "baseline_zeroshot": cell_base_zs,
        "baseline_rag": cell_base_rag,
        "gate_v2_zeroshot": cell_gate_zs,
        "gate_v2_rag": cell_gate_rag,
    }

    e1_formats = {}
    for fmt in ["baseline", "gate_v2"]:
        zs_rc = {i: loaded_gens[f"{fmt}_zeroshot"][i]["root_cause"] for i in test_ids}
        zs_rem = {i: loaded_gens[f"{fmt}_zeroshot"][i].get("remediation_steps", "") for i in test_ids}

        rand_rc = {i: rand_gens[i][fmt]["root_cause"] for i in test_ids}
        rand_rem = {i: rand_gens[i][fmt].get("remediation_steps", "") for i in test_ids}

        rag_rc = {i: loaded_gens[f"{fmt}_rag"][i]["root_cause"] for i in test_ids}
        rag_rem = {i: loaded_gens[f"{fmt}_rag"][i].get("remediation_steps", "") for i in test_ids}

        zs_arm = rescore_arm(test_ids, gt_rc_map, zs_rc, median_gt_limit, bootstrapper)
        rand_arm = rescore_arm(test_ids, gt_rc_map, rand_rc, median_gt_limit, bootstrapper)
        rag_arm = rescore_arm(test_ids, gt_rc_map, rag_rc, median_gt_limit, bootstrapper)

        deltas = {
            "real_minus_random": compute_paired_deltas(rag_arm["_raw"], rand_arm["_raw"], bootstrapper),
            "random_minus_zeroshot": compute_paired_deltas(rand_arm["_raw"], zs_arm["_raw"], bootstrapper),
            "real_minus_zeroshot": compute_paired_deltas(rag_arm["_raw"], zs_arm["_raw"], bootstrapper),
        }

        rem_both = {
            "real_minus_random": evaluate_remediation_pairwise_rescored(test_ids, id_to_company, gt_rem_map, rag_rem, rand_rem),
            "random_minus_zeroshot": evaluate_remediation_pairwise_rescored(test_ids, id_to_company, gt_rem_map, rand_rem, zs_rem),
            "real_minus_zeroshot": evaluate_remediation_pairwise_rescored(test_ids, id_to_company, gt_rem_map, rag_rem, zs_rem),
        }

        v = determine_verdict(
            random_minus_zs_excludes=deltas["random_minus_zeroshot"]["rougeL_f1"]["excludes_zero_after_adjustment"],
            real_minus_random_excludes=deltas["real_minus_random"]["rougeL_f1"]["excludes_zero_after_adjustment"]
        )

        e1_formats[fmt] = {
            "arms": {
                "zero_shot": zs_arm,
                "random_precedent": rand_arm,
                "real_rag": rag_arm,
            },
            "deltas": deltas,
            "remediation_both_parsed": rem_both,
            "verdict": v,
        }

    e1_verdict = e1_formats["gate_v2"]["verdict"]
    rescored_e1 = {
        "metadata": {
            "num_incidents": len(test_ids),
            "num_companies": len(set(id_to_company.values())),
            "bootstrap_resamples": 10000,
            "seed": 42,
            "family_size_m": 5,
            "adjusted_alpha": 0.01,
            "unadjusted_alpha": 0.05,
            "gt_median_word_count": median_gt_limit,
            "rescore_scorer": "eval/scorer.py canonical module",
        },
        "formats": e1_formats,
        "verdict": e1_verdict,
        "summary": f"Rescored E1 with canonical eval/scorer.py. Verdict: {e1_verdict}."
    }

    with open(RESC_E1_FILE, "w", encoding="utf-8") as f:
        json.dump(rescored_e1, f, indent=2)
    print(f"Saved rescored E1 to {RESC_E1_FILE}")

    # -------------------------------------------------------------
    # 2. Rescore E2
    # -------------------------------------------------------------
    import sqlite3
    e2_cache_conn = sqlite3.connect(str(CACHE_DIR / "e2_cache.sqlite"))
    e2_cur = e2_cache_conn.cursor()

    with open(PROJECT_ROOT / "eval" / "loo" / "sample.json", "r", encoding="utf-8") as f:
        loo_sample = json.load(f)["sample"]

    loo_ids = [q["incident_id"] for q in loo_sample]
    loo_company_map = {q["incident_id"]: q["company"] for q in loo_sample}
    loo_gt_rc_map = {q["incident_id"]: q["root_cause"] for q in loo_sample}
    loo_gt_rem_map = {q["incident_id"]: q.get("remediation_steps", "") for q in loo_sample}

    loo_gt_words = [word_count(loo_gt_rc_map[i]) for i in loo_ids]
    loo_gt_median = int(round(float(np.median(loo_gt_words))))

    loo_bootstrapper = FastClusterBootstrap(loo_ids, loo_company_map, n_resamples=10000, seed=42)

    from rag.prompts import PROMPT_REGISTRY
    import hashlib

    with open(PROJECT_ROOT / "eval" / "loo" / "retrieval.json", "r", encoding="utf-8") as f:
        loo_retrieval_map = {q["query_incident_id"]: q for q in json.load(f)["retrieval"]}

    def load_loo_cell(cell_name, prompt_variant, ret_key):
        pdef = PROMPT_REGISTRY[prompt_variant]
        sys_msg = pdef["system"]
        res = {}
        for inc in loo_sample:
            qid = inc["incident_id"]
            q_log = inc["raw_log_excerpt"]
            if ret_key is None:
                pid = "none"
                if cell_name == "baseline_zeroshot":
                    user_msg = f"New Incident Log Excerpt:\n{q_log}\n\nProvide your response in the following format:\nROOT CAUSE:\n(your analysis)\n\nREMEDIATION STEPS:\n(your steps)"
                else:
                    user_msg = pdef["user_template"].format(log_text=q_log)
            else:
                top1 = loo_retrieval_map[qid][ret_key]["hits"][0]
                pid = top1["incident_id"]
                p_rc = top1["root_cause"]
                p_rem = top1["remediation_steps"]
                sim = top1["similarity_score"]
                if "{similarity" in pdef["user_template"]:
                    user_msg = pdef["user_template"].format(ref_root_cause=p_rc, ref_remediation_steps=p_rem, log_text=q_log, similarity=sim)
                else:
                    user_msg = pdef["user_template"].format(ref_root_cause=p_rc, ref_remediation_steps=p_rem, log_text=q_log)
            p_hash = hashlib.sha256(f"{sys_msg}\n{user_msg}".encode("utf-8")).hexdigest()
            cache_key = hashlib.sha256(f"openai/gpt-oss-120b:{p_hash}:{qid}:{pid}".encode("utf-8")).hexdigest()
            e2_cur.execute("SELECT root_cause, remediation_steps FROM e2_cache WHERE cache_key = ?", (cache_key,))
            row = e2_cur.fetchone()
            if not row:
                raise KeyError(f"Missing generation for {qid} in {cell_name}")
            res[qid] = {"root_cause": row[0], "remediation_steps": row[1]}
        return res

    e2_cell_data = {
        "gate_v2_zeroshot": load_loo_cell("gate_v2_zeroshot", "zero_shot_gate_format", None),
        "gate_v2_rag_a": load_loo_cell("gate_v2_rag_a", "relevance_gate_v2", "condition_a_same_company_filtered"),
        "gate_v2_rag_b": load_loo_cell("gate_v2_rag_b", "relevance_gate_v2", "condition_b_cross_company_filtered"),
        "baseline_zeroshot": load_loo_cell("baseline_zeroshot", "baseline", None),
        "baseline_rag_a": load_loo_cell("baseline_rag_a", "baseline", "condition_a_same_company_filtered"),
        "baseline_rag_b": load_loo_cell("baseline_rag_b", "baseline", "condition_b_cross_company_filtered"),
    }

    with open(ORIG_E2_FILE, "r", encoding="utf-8") as f:
        orig_e2 = json.load(f)

    rescored_e2_formats = {}
    for fmt, prefix in [("baseline", "baseline"), ("gate_v2", "gate_v2")]:
        zs_rc = {i: e2_cell_data[f"{prefix}_zeroshot"][i]["root_cause"] for i in loo_ids}
        zs_rem = {i: e2_cell_data[f"{prefix}_zeroshot"][i]["remediation_steps"] for i in loo_ids}

        rag_a_rc = {i: e2_cell_data[f"{prefix}_rag_a"][i]["root_cause"] for i in loo_ids}
        rag_a_rem = {i: e2_cell_data[f"{prefix}_rag_a"][i]["remediation_steps"] for i in loo_ids}

        rag_b_rc = {i: e2_cell_data[f"{prefix}_rag_b"][i]["root_cause"] for i in loo_ids}
        rag_b_rem = {i: e2_cell_data[f"{prefix}_rag_b"][i]["remediation_steps"] for i in loo_ids}

        zs_arm = rescore_arm(loo_ids, loo_gt_rc_map, zs_rc, loo_gt_median, loo_bootstrapper)
        rag_a_arm = rescore_arm(loo_ids, loo_gt_rc_map, rag_a_rc, loo_gt_median, loo_bootstrapper)
        rag_b_arm = rescore_arm(loo_ids, loo_gt_rc_map, rag_b_rc, loo_gt_median, loo_bootstrapper)

        deltas = {
            "rag_a_minus_zeroshot": compute_paired_deltas(rag_a_arm["_raw"], zs_arm["_raw"], loo_bootstrapper),
            "rag_b_minus_zeroshot": compute_paired_deltas(rag_b_arm["_raw"], zs_arm["_raw"], loo_bootstrapper),
            "rag_a_minus_rag_b_same_company_premium": compute_paired_deltas(rag_a_arm["_raw"], rag_b_arm["_raw"], loo_bootstrapper),
        }

        rem_both = {
            "rag_a_minus_zeroshot": evaluate_remediation_pairwise_rescored(loo_ids, loo_company_map, loo_gt_rem_map, rag_a_rem, zs_rem),
            "rag_b_minus_zeroshot": evaluate_remediation_pairwise_rescored(loo_ids, loo_company_map, loo_gt_rem_map, rag_b_rem, zs_rem),
            "rag_a_minus_rag_b_same_company_premium": evaluate_remediation_pairwise_rescored(loo_ids, loo_company_map, loo_gt_rem_map, rag_a_rem, rag_b_rem),
        }

        rescored_e2_formats[fmt] = {
            "arms": {
                "zero_shot": zs_arm,
                "rag_condition_a_same_company": rag_a_arm,
                "rag_condition_b_cross_company": rag_b_arm,
            },
            "deltas": deltas,
            "remediation_both_parsed": rem_both,
        }

    # Stratified deltas
    stratified_deltas = {}
    with open(PROJECT_ROOT / "eval" / "results" / "relevance_judgments.json", "r", encoding="utf-8") as f:
        # Load from orig_e2 judgments
        cond_a_judgments = orig_e2.get("per_incident_judgments", {}).get("condition_a", {})

    judge_scores = {}
    judge_cur = sqlite3.connect(str(CACHE_DIR / "judge_cache.sqlite")).cursor()
    with open(PROJECT_ROOT / "eval" / "loo" / "retrieval.json", "r", encoding="utf-8") as f:
        loo_ret = json.load(f)["retrieval"]

    for q in loo_ret:
        qid = q["query_incident_id"]
        top1_a = q["condition_a_same_company_filtered"]["hits"][0]["incident_id"]
        k = f"qwen/qwen3.8-27b:{qid}:{top1_a}"
        import hashlib
        h = hashlib.sha256(k.encode("utf-8")).hexdigest()
        judge_cur.execute("SELECT score FROM judge_cache WHERE cache_key = ?", (h,))
        row = judge_cur.fetchone()
        judge_scores[qid] = row[0] if row else 0

    zs_raw_g = rescored_e2_formats["gate_v2"]["arms"]["zero_shot"]["_raw"]["rougeL_f1"]
    ra_raw_g = rescored_e2_formats["gate_v2"]["arms"]["rag_condition_a_same_company"]["_raw"]["rougeL_f1"]
    id_to_idx = {qid: idx for idx, qid in enumerate(loo_ids)}

    for s_level in [0, 1, 2]:
        s_ids = [qid for qid in loo_ids if judge_scores.get(qid) == s_level]
        s_comps = {loo_company_map[i] for i in s_ids}
        if len(s_ids) >= 5:
            s_diffs = [ra_raw_g[id_to_idx[i]] - zs_raw_g[id_to_idx[i]] for i in s_ids]
            s_boot = FastClusterBootstrap(s_ids, loo_company_map, n_resamples=10000, seed=42)
            stratified_deltas[f"relevance_score_{s_level}"] = {
                "score_level": s_level,
                "num_incidents": len(s_ids),
                "num_companies": len(s_comps),
                "mean_delta_rougeL": round(float(np.mean(s_diffs)), 4),
                "unadjusted_ci_95": s_boot.compute_multi_ci(s_diffs)["unadjusted_ci_95"],
                "adjusted_ci": s_boot.compute_multi_ci(s_diffs)["adjusted_ci"],
                "status": s_boot.compute_multi_ci(s_diffs)["status"],
                "low_n": len(s_comps) < 5,
            }

    # E2 verdict
    e2_gate_delta = rescored_e2_formats["gate_v2"]["deltas"]["rag_a_minus_zeroshot"]["rougeL_f1"]
    if e2_gate_delta["excludes_zero_after_adjustment"]:
        e2_verdict = "supported"
    else:
        e2_verdict = "no detected effect"

    rescored_e2 = dict(orig_e2)
    rescored_e2["formats"] = rescored_e2_formats
    rescored_e2["stratified_deltas_by_relevance_gate_v2"] = stratified_deltas
    rescored_e2["verdict"] = e2_verdict
    rescored_e2["metadata"]["rescore_scorer"] = "eval/scorer.py canonical module"

    with open(RESC_E2_FILE, "w", encoding="utf-8") as f:
        json.dump(rescored_e2, f, indent=2)
    print(f"Saved rescored E2 to {RESC_E2_FILE}")

    # -------------------------------------------------------------
    # 3. Print Detailed Comparison vs Original Committed JSONs
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CELL-BY-CELL COMPARISON: RESCORED VS ORIGINAL COMMITTED")
    print("=" * 80)

    with open(ORIG_E1_FILE, "r", encoding="utf-8") as f:
        orig_e1 = json.load(f)

    print("\n--- EXPERIMENT 1 (E1) COMPARISON ---")
    for fmt in ["baseline", "gate_v2"]:
        print(f"\nFormat: {fmt}")
        for arm_name in ["zero_shot", "random_precedent", "real_rag"]:
            orig_arm = orig_e1["formats"][fmt]["arms"][arm_name]
            resc_arm = rescored_e1["formats"][fmt]["arms"][arm_name]
            print(f"  Arm: {arm_name}")
            for m in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bert_score_f1", "cosine"]:
                o_val = orig_arm[m]["mean"]
                r_val = resc_arm[m]["mean"]
                diff = r_val - o_val
                mark = f"CHANGE: {diff:+.4f}" if abs(diff) > 1e-5 else "NO CHANGE"
                print(f"    {m:14s}: orig={o_val:.4f}, rescored={r_val:.4f} -> {mark}")
            ow_m = orig_arm["word_count"]["mean"]
            rw_m = resc_arm["word_count"]["mean"]
            print(f"    {'word_count_mean':14s}: orig={ow_m:.1f}, rescored={rw_m:.1f} -> {'CHANGE: ' + f'{rw_m-ow_m:+.1f}' if abs(rw_m-ow_m)>0.01 else 'NO CHANGE'}")

        print(f"\n  Deltas for {fmt}:")
        for delta_name in ["random_minus_zeroshot", "real_minus_random", "real_minus_zeroshot"]:
            orig_d = orig_e1["formats"][fmt]["deltas"][delta_name]
            resc_d = rescored_e1["formats"][fmt]["deltas"][delta_name]
            for m in ["rougeL_f1", "bert_score_f1", "cosine"]:
                o_val = orig_d[m]["mean"]
                r_val = resc_d[m]["mean"]
                diff = r_val - o_val
                mark = f"CHANGE: {diff:+.4f}" if abs(diff) > 1e-5 else "NO CHANGE"
                print(f"    {delta_name} {m}: orig={o_val:.4f}, rescored={r_val:.4f} -> {mark}")

    print("\n--- EXPERIMENT 2 (E2) COMPARISON ---")
    for fmt in ["baseline", "gate_v2"]:
        print(f"\nFormat: {fmt}")
        for arm_name in ["zero_shot", "rag_condition_a_same_company", "rag_condition_b_cross_company"]:
            orig_arm = orig_e2["formats"][fmt]["arms"][arm_name]
            resc_arm = rescored_e2["formats"][fmt]["arms"][arm_name]
            print(f"  Arm: {arm_name}")
            for m in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bert_score_f1", "cosine"]:
                o_val = orig_arm[m]["mean"]
                r_val = resc_arm[m]["mean"]
                diff = r_val - o_val
                mark = f"CHANGE: {diff:+.4f}" if abs(diff) > 1e-5 else "NO CHANGE"
                print(f"    {m:14s}: orig={o_val:.4f}, rescored={r_val:.4f} -> {mark}")
            ow_m = orig_arm["word_count"]["mean"]
            rw_m = resc_arm["word_count"]["mean"]
            print(f"    {'word_count_mean':14s}: orig={ow_m:.1f}, rescored={rw_m:.1f} -> {'CHANGE: ' + f'{rw_m-ow_m:+.1f}' if abs(rw_m-ow_m)>0.01 else 'NO CHANGE'}")

        print(f"\n  Deltas for {fmt}:")
        for delta_name in ["rag_a_minus_zeroshot", "rag_b_minus_zeroshot", "rag_a_minus_rag_b_same_company_premium"]:
            orig_d = orig_e2["formats"][fmt]["deltas"][delta_name]
            resc_d = rescored_e2["formats"][fmt]["deltas"][delta_name]
            for m in ["rougeL_f1", "bert_score_f1", "cosine"]:
                o_val = orig_d[m]["mean"]
                r_val = resc_d[m]["mean"]
                diff = r_val - o_val
                mark = f"CHANGE: {diff:+.4f}" if abs(diff) > 1e-5 else "NO CHANGE"
                print(f"    {delta_name} {m}: orig={o_val:.4f}, rescored={r_val:.4f} -> {mark}")


if __name__ == "__main__":
    run_rescoring()
