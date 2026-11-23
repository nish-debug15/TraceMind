"""
TraceMind LOO Stratified Deltas by Judged Relevance (Block 6)
------------------------------------------------------------
Computes RAG(a)-minus-zero-shot deltas:
- ROUGE-L F1 delta
- BERTScore F1 delta
- Remediation ROUGE-L on both-parsed incidents
Stratified by judged top-1 relevance score (0: unrelated, 1: partially relevant, 2: highly relevant).
Reports n, n_companies, low_n (true where n_companies < 5), and word counts per cell.

Outputs: eval/results/loo_stratified.json
"""

import sys
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.scorer import score_cell, word_count, compute_bert_scores, _ROUGE_SCORER
from eval.loo.run_loo_eval import mask_vendor

SAMPLE_FILE = PROJECT_ROOT / "eval" / "loo" / "sample.json"
RETRIEVAL_FILE = PROJECT_ROOT / "eval" / "loo" / "retrieval.json"
RUBRIC_FILE = PROJECT_ROOT / "eval" / "rubrics" / "relevance_v1.md"
JUDGE_CACHE = PROJECT_ROOT / "eval" / "cache" / "judge_cache.sqlite"
E2_CACHE = PROJECT_ROOT / "eval" / "cache" / "e2_cache.sqlite"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "loo_stratified.json"


def generate_stratified_results():
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        sample = json.load(f)["sample"]
    sample_ids = [q["incident_id"] for q in sample]
    id_to_comp = {q["incident_id"]: q["company"] for q in sample}
    gt_rc_map = {q["incident_id"]: q["root_cause"] for q in sample}
    gt_rem_map = {q["incident_id"]: q.get("remediation_steps", "") for q in sample}

    with open(RETRIEVAL_FILE, "r", encoding="utf-8") as f:
        ret_map = {q["query_incident_id"]: q for q in json.load(f)["retrieval"]}

    with open(RUBRIC_FILE, "r", encoding="utf-8") as f:
        rubric_text = f.read()
    rubric_sha256 = hashlib.sha256(rubric_text.encode("utf-8")).hexdigest()

    # 1. Extract judged scores for Condition (a)
    j_conn = sqlite3.connect(str(JUDGE_CACHE))
    j_cur = j_conn.cursor()
    judge_scores = {}
    for inc in sample:
        qid = inc["incident_id"]
        q_comp = inc["company"]
        q_log_m = mask_vendor(inc.get("raw_log_excerpt", ""))
        q_rc_m = mask_vendor(inc.get("root_cause", ""))

        top1 = ret_map[qid]["condition_a_same_company_filtered"]["hits"][0]
        p_comp = top1["company"]
        p_log_m = mask_vendor(top1.get("raw_log_excerpt", ""))
        p_rc_m = mask_vendor(top1.get("root_cause", ""))

        content = f"{rubric_sha256}\n---QUERY---\n{q_log_m}\n{q_rc_m}\n---PRECEDENT---\n{p_log_m}\n{p_rc_m}"
        ckey = hashlib.sha256(content.encode("utf-8")).hexdigest()
        j_cur.execute("SELECT score FROM judge_cache WHERE cache_key = ?", (ckey,))
        row = j_cur.fetchone()
        assert row is not None, f"Missing judge cache for {qid}"
        judge_scores[qid] = row[0]

    # 2. Extract gate_v2 zero_shot and rag_a generations from e2_cache
    from rag.prompts import PROMPT_REGISTRY
    e2_conn = sqlite3.connect(str(E2_CACHE))
    e2_cur = e2_conn.cursor()

    zs_pdef = PROMPT_REGISTRY["zero_shot_gate_format"]
    raga_pdef = PROMPT_REGISTRY["relevance_gate_v2"]

    zs_gens = {}
    raga_gens = {}

    for inc in sample:
        qid = inc["incident_id"]
        q_log = inc["raw_log_excerpt"]

        # Zero-shot
        zs_user = zs_pdef["user_template"].format(log_text=q_log)
        zs_hash = hashlib.sha256(f"{zs_pdef['system']}\n{zs_user}".encode("utf-8")).hexdigest()
        zs_key = hashlib.sha256(f"openai/gpt-oss-120b:{zs_hash}:{qid}:none".encode("utf-8")).hexdigest()
        e2_cur.execute("SELECT root_cause, remediation_steps FROM e2_cache WHERE cache_key = ?", (zs_key,))
        row = e2_cur.fetchone()
        zs_gens[qid] = {"root_cause": row[0], "remediation_steps": row[1]}

        # RAG(a)
        top1 = ret_map[qid]["condition_a_same_company_filtered"]["hits"][0]
        pid = top1["incident_id"]
        sim = top1["similarity_score"]
        ra_user = raga_pdef["user_template"].format(
            ref_root_cause=top1["root_cause"],
            ref_remediation_steps=top1["remediation_steps"],
            log_text=q_log,
            similarity=sim
        )
        ra_hash = hashlib.sha256(f"{raga_pdef['system']}\n{ra_user}".encode("utf-8")).hexdigest()
        ra_key = hashlib.sha256(f"openai/gpt-oss-120b:{ra_hash}:{qid}:{pid}".encode("utf-8")).hexdigest()
        e2_cur.execute("SELECT root_cause, remediation_steps FROM e2_cache WHERE cache_key = ?", (ra_key,))
        row_ra = e2_cur.fetchone()
        raga_gens[qid] = {"root_cause": row_ra[0], "remediation_steps": row_ra[1]}

    # 3. Stratified metrics computation
    strata = {}
    for s in [0, 1, 2]:
        s_ids = [i for i in sample_ids if judge_scores[i] == s]
        s_comps = sorted(list(set(id_to_comp[i] for i in s_ids)))
        n_items = len(s_ids)
        n_comps = len(s_comps)
        is_low_n = bool(n_comps < 5)

        s_refs_rc = [gt_rc_map[i] for i in s_ids]
        s_zs_rc = [zs_gens[i]["root_cause"] for i in s_ids]
        s_ra_rc = [raga_gens[i]["root_cause"] for i in s_ids]

        # Word counts
        zs_wc = [word_count(p) for p in s_zs_rc]
        ra_wc = [word_count(p) for p in s_ra_rc]

        # Root cause scoring
        zs_scored = score_cell(s_zs_rc, s_refs_rc)
        ra_scored = score_cell(s_ra_rc, s_refs_rc)

        delta_rl_vals = [a - b for a, b in zip(ra_scored["per_incident"]["rougeL_f1"], zs_scored["per_incident"]["rougeL_f1"])]
        delta_bs_vals = [a - b for a, b in zip(ra_scored["per_incident"]["bert_score_f1"], zs_scored["per_incident"]["bert_score_f1"])]

        # Remediation on both-parsed
        both_ids = [
            i for i in s_ids
            if raga_gens[i]["remediation_steps"].strip() and zs_gens[i]["remediation_steps"].strip() and len(gt_rem_map[i].split()) >= 5
        ]
        both_comps = sorted(list(set(id_to_comp[i] for i in both_ids)))

        if both_ids:
            b_refs_rem = [gt_rem_map[i] for i in both_ids]
            b_ra_rem = [raga_gens[i]["remediation_steps"] for i in both_ids]
            b_zs_rem = [zs_gens[i]["remediation_steps"] for i in both_ids]

            rl_ra_rem = [_ROUGE_SCORER.score(g, p)["rougeL"].fmeasure for g, p in zip(b_refs_rem, b_ra_rem)]
            rl_zs_rem = [_ROUGE_SCORER.score(g, p)["rougeL"].fmeasure for g, p in zip(b_refs_rem, b_zs_rem)]
            delta_rem_rl_vals = [a - b for a, b in zip(rl_ra_rem, rl_zs_rem)]
            rem_mean_delta = round(float(np.mean(delta_rem_rl_vals)), 4)
        else:
            rem_mean_delta = None

        strata[f"score_{s}"] = {
            "score_level": s,
            "score_label": "unrelated" if s == 0 else ("partially_relevant" if s == 1 else "highly_relevant"),
            "n": n_items,
            "n_companies": n_comps,
            "low_n": is_low_n,
            "deltas_rag_a_minus_zeroshot": {
                "rougeL_f1": {
                    "mean": round(float(np.mean(delta_rl_vals)), 4),
                },
                "bert_score_f1": {
                    "mean": round(float(np.mean(delta_bs_vals)), 4),
                },
                "remediation_both_parsed": {
                    "n_both_parsed": len(both_ids),
                    "n_companies_both_parsed": len(both_comps),
                    "delta_remediation_rougeL": {
                        "mean": rem_mean_delta,
                    }
                }
            },
            "word_counts": {
                "zero_shot": {
                    "mean": round(float(np.mean(zs_wc)), 1),
                    "median": round(float(np.median(zs_wc)), 1),
                },
                "rag_a": {
                    "mean": round(float(np.mean(ra_wc)), 1),
                    "median": round(float(np.median(ra_wc)), 1),
                }
            }
        }

    out_data = {
        "metadata": {
            "regime": "E2 Leave-One-Out (Gate-v2 format)",
            "total_queries": len(sample_ids),
            "scoring_module": "eval/scorer.py canonical module",
        },
        "strata": strata
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved stratified results to {OUTPUT_FILE}")
    print(json.dumps(out_data, indent=2))
    return out_data


if __name__ == "__main__":
    generate_stratified_results()
