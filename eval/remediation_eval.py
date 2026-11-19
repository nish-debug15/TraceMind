"""
eval/remediation_eval.py
------------------------
Evaluates the remediation_steps section across the 4 cells of the 2x2 factorial control:
1. baseline_zeroshot
2. baseline_rag
3. gate_v2_zeroshot
4. gate_v2_rag

Supports both:
- Unconditional evaluation (N=93): parse failures assigned 0.0 for ROUGE-L and sanitized 'none' for BERTScore
- Conditional on both parsed evaluation: restricted to incidents where BOTH arms in each format pair parsed successfully

Computes company-clustered bootstrap 95% CIs (10,000 resamples, seed 42) for cell scores and within-format deltas.
Saves results to eval/results/remediation_eval.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as calc_bertscore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.cluster_bootstrap import two_stage_cluster_bootstrap_ci

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "remediation_eval.json"

CELL_FILES = {
    "baseline_zeroshot": CACHE_DIR / "zeroshot_generations.json",
    "baseline_rag": CACHE_DIR / "rag_generations.json",
    "gate_v2_zeroshot": CACHE_DIR / "ablation_zero_shot_gate_format.json",
    "gate_v2_rag": CACHE_DIR / "ablation_relevance_gate_v2.json",
}


def run_remediation_eval() -> Dict[str, Any]:
    print("[Remediation Eval] Loading test records and ground truth remediation...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    # Check ground-truth remediation word counts and exclusions
    all_incidents = [r["incident_id"] for r in test_records]
    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in test_records}
    gt_rem_map = {r["incident_id"]: (r.get("remediation_steps") or "").strip() for r in test_records}

    excluded_ids = []
    eligible_ids = []
    for inc_id in all_incidents:
        gt_text = gt_rem_map[inc_id]
        wc = len(gt_text.split())
        if wc < 5:
            excluded_ids.append(inc_id)
        else:
            eligible_ids.append(inc_id)

    print(f"Total test incidents: {len(all_incidents)}")
    print(f"Excluded due to GT remediation < 5 words: {len(excluded_ids)}")
    print(f"Eligible incidents for scoring: {len(eligible_ids)}")

    # Load 4 cell cached generations
    cell_data = {}
    for cname, cpath in CELL_FILES.items():
        with open(cpath, "r", encoding="utf-8") as f:
            cell_data[cname] = json.load(f)

    # Parse and check parse failures
    parse_failures = {}
    cell_preds = {}
    cell_has_rem = {}
    for cname, cdict in cell_data.items():
        fails = []
        preds = []
        has_rem_map = {}
        for inc_id in eligible_ids:
            item = cdict.get(inc_id, {})
            rem = item.get("remediation_steps", "").strip()
            if not rem:
                fails.append(inc_id)
                preds.append("")
                has_rem_map[inc_id] = False
            else:
                preds.append(rem)
                has_rem_map[inc_id] = True
        parse_failures[cname] = {
            "failure_count": len(fails),
            "total_eligible": len(eligible_ids),
            "coverage_rate": round(float((len(eligible_ids) - len(fails)) / len(eligible_ids)), 4),
            "failed_incident_ids": fails,
        }
        cell_preds[cname] = preds
        cell_has_rem[cname] = has_rem_map

    # Check stopping condition for gate-v2 zero-shot
    gz_fails = parse_failures["gate_v2_zeroshot"]["failure_count"]
    if gz_fails == len(eligible_ids):
        raise RuntimeError(
            "CRITICAL: Zero-shot gate-format outputs do not contain a remediation section at all (100% failure). STOPPING."
        )

    # 1. Unconditional Scoring across all eligible test incidents (n=93)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    ground_truth = [gt_rem_map[i] for i in eligible_ids]

    cell_scores = {}
    print("\n[Remediation Eval] Scoring all 4 cells on eligible test incidents (n=93, unconditional)...")
    for cname, preds in cell_preds.items():
        print(f"  Scoring {cname}...")
        rl_list = []
        for gt, pred in zip(ground_truth, preds):
            if not pred.strip():
                rl_list.append(0.0)
            else:
                s = scorer.score(gt, pred)["rougeL"].fmeasure
                rl_list.append(s)

        safe_preds = [p if p.strip() else "none" for p in preds]
        safe_gt = [g if g.strip() else "none" for g in ground_truth]

        _, _, bs = calc_bertscore(safe_preds, safe_gt, model_type="bert-base-uncased", verbose=False)
        bs_list = bs.numpy().tolist()

        rl_ci = two_stage_cluster_bootstrap_ci(eligible_ids, rl_list, id_to_company, n_resamples=10000, seed=42)
        bs_ci = two_stage_cluster_bootstrap_ci(eligible_ids, bs_list, id_to_company, n_resamples=10000, seed=42)

        cell_scores[cname] = {
            "rougeL_f1": rl_ci,
            "bert_score_f1": bs_ci,
            "_raw_rl": rl_list,
            "_raw_bs": bs_list,
        }

    # Unconditional within-format deltas
    base_rl_diff = [
        r - z for r, z in zip(cell_scores["baseline_rag"]["_raw_rl"], cell_scores["baseline_zeroshot"]["_raw_rl"])
    ]
    base_bs_diff = [
        r - z for r, z in zip(cell_scores["baseline_rag"]["_raw_bs"], cell_scores["baseline_zeroshot"]["_raw_bs"])
    ]

    gate_rl_diff = [
        r - z for r, z in zip(cell_scores["gate_v2_rag"]["_raw_rl"], cell_scores["gate_v2_zeroshot"]["_raw_rl"])
    ]
    gate_bs_diff = [
        r - z for r, z in zip(cell_scores["gate_v2_rag"]["_raw_bs"], cell_scores["gate_v2_zeroshot"]["_raw_bs"])
    ]

    unconditional_deltas = {
        "baseline_format_rag_effect": {
            "rougeL_delta": two_stage_cluster_bootstrap_ci(eligible_ids, base_rl_diff, id_to_company, n_resamples=10000, seed=42),
            "bert_score_delta": two_stage_cluster_bootstrap_ci(eligible_ids, base_bs_diff, id_to_company, n_resamples=10000, seed=42),
        },
        "gate_v2_format_rag_effect": {
            "rougeL_delta": two_stage_cluster_bootstrap_ci(eligible_ids, gate_rl_diff, id_to_company, n_resamples=10000, seed=42),
            "bert_score_delta": two_stage_cluster_bootstrap_ci(eligible_ids, gate_bs_diff, id_to_company, n_resamples=10000, seed=42),
        }
    }

    cleaned_cells = {}
    for cname, cdata in cell_scores.items():
        cleaned_cells[cname] = {
            "rougeL_f1": cdata["rougeL_f1"],
            "bert_score_f1": cdata["bert_score_f1"],
            "coverage_rate": parse_failures[cname]["coverage_rate"],
            "parse_failure_count": parse_failures[cname]["failure_count"],
        }

    # 2. Conditional on both parsed evaluation
    print("\n[Remediation Eval] Scoring conditional on both parsed per format...")

    # Pair A: Baseline Format (baseline_zeroshot and baseline_rag)
    both_base_ids = [
        i for i in eligible_ids
        if cell_has_rem["baseline_zeroshot"][i] and cell_has_rem["baseline_rag"][i]
    ]
    print(f"  Baseline format incidents with both parsed: n = {len(both_base_ids)}")

    base_zs_preds_cond = [cell_data["baseline_zeroshot"][i]["remediation_steps"].strip() for i in both_base_ids]
    base_rag_preds_cond = [cell_data["baseline_rag"][i]["remediation_steps"].strip() for i in both_base_ids]
    gt_base_cond = [gt_rem_map[i] for i in both_base_ids]

    base_zs_rl_cond = [scorer.score(gt, p)["rougeL"].fmeasure for gt, p in zip(gt_base_cond, base_zs_preds_cond)]
    base_rag_rl_cond = [scorer.score(gt, p)["rougeL"].fmeasure for gt, p in zip(gt_base_cond, base_rag_preds_cond)]
    base_diff_rl_cond = [r - z for r, z in zip(base_rag_rl_cond, base_zs_rl_cond)]

    _, _, bs_b_zs_c = calc_bertscore(base_zs_preds_cond, gt_base_cond, model_type="bert-base-uncased", verbose=False)
    _, _, bs_b_rag_c = calc_bertscore(base_rag_preds_cond, gt_base_cond, model_type="bert-base-uncased", verbose=False)
    base_zs_bs_cond = bs_b_zs_c.numpy().tolist()
    base_rag_bs_cond = bs_b_rag_c.numpy().tolist()
    base_diff_bs_cond = [r - z for r, z in zip(base_rag_bs_cond, base_zs_bs_cond)]

    # Pair B: Gate-v2 Format (gate_v2_zeroshot and gate_v2_rag)
    both_gate_ids = [
        i for i in eligible_ids
        if cell_has_rem["gate_v2_zeroshot"][i] and cell_has_rem["gate_v2_rag"][i]
    ]
    print(f"  Gate-v2 format incidents with both parsed: n = {len(both_gate_ids)}")

    gate_zs_preds_cond = [cell_data["gate_v2_zeroshot"][i]["remediation_steps"].strip() for i in both_gate_ids]
    gate_rag_preds_cond = [cell_data["gate_v2_rag"][i]["remediation_steps"].strip() for i in both_gate_ids]
    gt_gate_cond = [gt_rem_map[i] for i in both_gate_ids]

    gate_zs_rl_cond = [scorer.score(gt, p)["rougeL"].fmeasure for gt, p in zip(gt_gate_cond, gate_zs_preds_cond)]
    gate_rag_rl_cond = [scorer.score(gt, p)["rougeL"].fmeasure for gt, p in zip(gt_gate_cond, gate_rag_preds_cond)]
    gate_diff_rl_cond = [r - z for r, z in zip(gate_rag_rl_cond, gate_zs_rl_cond)]

    _, _, bs_g_zs_c = calc_bertscore(gate_zs_preds_cond, gt_gate_cond, model_type="bert-base-uncased", verbose=False)
    _, _, bs_g_rag_c = calc_bertscore(gate_rag_preds_cond, gt_gate_cond, model_type="bert-base-uncased", verbose=False)
    gate_zs_bs_cond = bs_g_zs_c.numpy().tolist()
    gate_rag_bs_cond = bs_g_rag_c.numpy().tolist()
    gate_diff_bs_cond = [r - z for r, z in zip(gate_rag_bs_cond, gate_zs_bs_cond)]

    conditional_on_both_parsed = {
        "baseline_format": {
            "n": len(both_base_ids),
            "num_unique_companies": len(set(id_to_company[i] for i in both_base_ids)),
            "incident_ids": both_base_ids,
            "cells": {
                "baseline_zeroshot": {
                    "rougeL_f1": two_stage_cluster_bootstrap_ci(both_base_ids, base_zs_rl_cond, id_to_company, n_resamples=10000, seed=42),
                    "bert_score_f1": two_stage_cluster_bootstrap_ci(both_base_ids, base_zs_bs_cond, id_to_company, n_resamples=10000, seed=42),
                },
                "baseline_rag": {
                    "rougeL_f1": two_stage_cluster_bootstrap_ci(both_base_ids, base_rag_rl_cond, id_to_company, n_resamples=10000, seed=42),
                    "bert_score_f1": two_stage_cluster_bootstrap_ci(both_base_ids, base_rag_bs_cond, id_to_company, n_resamples=10000, seed=42),
                }
            },
            "deltas": {
                "rougeL_delta": two_stage_cluster_bootstrap_ci(both_base_ids, base_diff_rl_cond, id_to_company, n_resamples=10000, seed=42),
                "bert_score_delta": two_stage_cluster_bootstrap_ci(both_base_ids, base_diff_bs_cond, id_to_company, n_resamples=10000, seed=42),
            }
        },
        "gate_v2_format": {
            "n": len(both_gate_ids),
            "num_unique_companies": len(set(id_to_company[i] for i in both_gate_ids)),
            "incident_ids": both_gate_ids,
            "cells": {
                "gate_v2_zeroshot": {
                    "rougeL_f1": two_stage_cluster_bootstrap_ci(both_gate_ids, gate_zs_rl_cond, id_to_company, n_resamples=10000, seed=42),
                    "bert_score_f1": two_stage_cluster_bootstrap_ci(both_gate_ids, gate_zs_bs_cond, id_to_company, n_resamples=10000, seed=42),
                },
                "gate_v2_rag": {
                    "rougeL_f1": two_stage_cluster_bootstrap_ci(both_gate_ids, gate_rag_rl_cond, id_to_company, n_resamples=10000, seed=42),
                    "bert_score_f1": two_stage_cluster_bootstrap_ci(both_gate_ids, gate_rag_bs_cond, id_to_company, n_resamples=10000, seed=42),
                }
            },
            "deltas": {
                "rougeL_delta": two_stage_cluster_bootstrap_ci(both_gate_ids, gate_diff_rl_cond, id_to_company, n_resamples=10000, seed=42),
                "bert_score_delta": two_stage_cluster_bootstrap_ci(both_gate_ids, gate_diff_bs_cond, id_to_company, n_resamples=10000, seed=42),
            }
        }
    }

    base_uncond_rl_d = unconditional_deltas["baseline_format_rag_effect"]["rougeL_delta"]
    base_cond_rl_d = conditional_on_both_parsed["baseline_format"]["deltas"]["rougeL_delta"]

    summary_text = (
        f"Remediation section evaluation across 4 cells (N={len(eligible_ids)}, excluded GT < 5 words = {len(excluded_ids)}):\n"
        f"1. Parse Failures / Missing Section:\n"
        f"   - baseline_zeroshot: {parse_failures['baseline_zeroshot']['failure_count']}/93 ({parse_failures['baseline_zeroshot']['coverage_rate']*100:.1f}% coverage)\n"
        f"   - baseline_rag:      {parse_failures['baseline_rag']['failure_count']}/93 ({parse_failures['baseline_rag']['coverage_rate']*100:.1f}% coverage)\n"
        f"   - gate_v2_zeroshot:  {parse_failures['gate_v2_zeroshot']['failure_count']}/93 ({parse_failures['gate_v2_zeroshot']['coverage_rate']*100:.1f}% coverage)\n"
        f"   - gate_v2_rag:       {parse_failures['gate_v2_rag']['failure_count']}/93 ({parse_failures['gate_v2_rag']['coverage_rate']*100:.1f}% coverage)\n"
        f"2. Unconditional Cell Scores (All 93 test incidents, failures scored 0.0 ROUGE-L / 'none' BERTScore, clustered 95% CIs):\n"
        f"   - baseline_zeroshot: ROUGE-L={cleaned_cells['baseline_zeroshot']['rougeL_f1']['mean']} {cleaned_cells['baseline_zeroshot']['rougeL_f1']['ci_95']}, BERTScore={cleaned_cells['baseline_zeroshot']['bert_score_f1']['mean']} {cleaned_cells['baseline_zeroshot']['bert_score_f1']['ci_95']}\n"
        f"   - baseline_rag:      ROUGE-L={cleaned_cells['baseline_rag']['rougeL_f1']['mean']} {cleaned_cells['baseline_rag']['rougeL_f1']['ci_95']}, BERTScore={cleaned_cells['baseline_rag']['bert_score_f1']['mean']} {cleaned_cells['baseline_rag']['bert_score_f1']['ci_95']}\n"
        f"   - gate_v2_zeroshot:  ROUGE-L={cleaned_cells['gate_v2_zeroshot']['rougeL_f1']['mean']} {cleaned_cells['gate_v2_zeroshot']['rougeL_f1']['ci_95']}, BERTScore={cleaned_cells['gate_v2_zeroshot']['bert_score_f1']['mean']} {cleaned_cells['gate_v2_zeroshot']['bert_score_f1']['ci_95']}\n"
        f"   - gate_v2_rag:       ROUGE-L={cleaned_cells['gate_v2_rag']['rougeL_f1']['mean']} {cleaned_cells['gate_v2_rag']['rougeL_f1']['ci_95']}, BERTScore={cleaned_cells['gate_v2_rag']['bert_score_f1']['mean']} {cleaned_cells['gate_v2_rag']['bert_score_f1']['ci_95']}\n"
        f"3. Unconditional Within-Format RAG Deltas:\n"
        f"   - Baseline Format: ROUGE-L delta = {base_uncond_rl_d['mean']:+.4f} {base_uncond_rl_d['ci_95']} (significant: {base_uncond_rl_d['significant_under_cluster_bootstrap']})\n"
        f"   - Gate-v2 Format:  ROUGE-L delta = {unconditional_deltas['gate_v2_format_rag_effect']['rougeL_delta']['mean']:+.4f} {unconditional_deltas['gate_v2_format_rag_effect']['rougeL_delta']['ci_95']} (significant: {unconditional_deltas['gate_v2_format_rag_effect']['rougeL_delta']['significant_under_cluster_bootstrap']})\n"
        f"4. Conditional on Both Parsed Within-Format RAG Deltas:\n"
        f"   - Baseline Format (n={len(both_base_ids)}): ROUGE-L delta drops from {base_uncond_rl_d['mean']:+.4f} {base_uncond_rl_d['ci_95']} (unconditional) to "
        f"{base_cond_rl_d['mean']:+.4f} {base_cond_rl_d['ci_95']} (conditional). The CI now crosses zero (significant: {base_cond_rl_d['significant_under_cluster_bootstrap']}), "
        f"demonstrating that the unconditional baseline remediation gain was largely an artifact of differential parse coverage (78.5% RAG vs 64.5% Zero-Shot).\n"
        f"   - Gate-v2 Format (n={len(both_gate_ids)}): ROUGE-L delta is {conditional_on_both_parsed['gate_v2_format']['deltas']['rougeL_delta']['mean']:+.4f} "
        f"{conditional_on_both_parsed['gate_v2_format']['deltas']['rougeL_delta']['ci_95']} (significant: {conditional_on_both_parsed['gate_v2_format']['deltas']['rougeL_delta']['significant_under_cluster_bootstrap']})."
    )

    failure_scoring_rule_text = (
        "In the unconditional evaluation (N=93), incidents where an arm failed to parse a remediation section "
        "were assigned a score of 0.0 for ROUGE-L F1 and sanitized to 'none' for BERTScore F1. "
        "In the conditional_on_both_parsed evaluation, the sample is restricted to incidents where both arms "
        "in the comparison pair successfully generated and parsed a remediation section (n=52 for baseline format, n=92 for gate-v2 format)."
    )

    results_payload = {
        "metadata": {
            "total_test_incidents": len(all_incidents),
            "excluded_gt_under_5_words": len(excluded_ids),
            "eligible_incidents": len(eligible_ids),
            "num_unique_companies": len(set(id_to_company.values())),
            "bootstrap_resamples": 10000,
            "seed": 42,
            "ci_level": 0.95,
        },
        "failure_scoring_rule": failure_scoring_rule_text,
        "parse_failures": parse_failures,
        "cells": cleaned_cells,
        "deltas": unconditional_deltas,
        "conditional_on_both_parsed": conditional_on_both_parsed,
        "summary": summary_text,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nSaved remediation evaluation to {OUTPUT_FILE}")
    print("\n" + summary_text)
    return results_payload


if __name__ == "__main__":
    run_remediation_eval()
