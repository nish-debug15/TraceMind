"""
eval/analyze_e1.py
------------------
Evaluates Experiment 1 (E1: Random-Precedent Control) on data/splits/test.json (N=93).

Compares 3 arms:
1. Zero-shot
2. Random-precedent (seed 42)
3. Real top-1 RAG

Separately evaluated under both prompt formats:
- Baseline format
- Gate-v2 format

Metrics computed per arm:
- ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1
- BERTScore F1 (bert-base-uncased)
- Cosine similarity (SentenceTransformer embed)
- Word count statistics (mean, median)
- Length-matched ROUGE-L (truncated to median ground-truth length)
- Remediation metrics on both-parsed subset (ROUGE-L F1, BERTScore F1)

Paired deltas computed with two-stage company-clustered bootstrap (10,000 resamples, seed 42):
- real_minus_random
- random_minus_zeroshot
- real_minus_zeroshot
Reporting both unadjusted 95% CIs and Bonferroni-adjusted 99% CIs (family m=5, alpha=0.01).

Automated verdict per format and overall mechanically derived from eval/PREREG.md:
- 'style_priming' if random_minus_zeroshot excludes 0 AND real_minus_random crosses 0
- 'precedent_content' if real_minus_random excludes 0
- 'inconclusive' otherwise

Outputs: eval/results/random_precedent_control.json
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as calc_bertscore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clustering.cluster import embed

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
RANDOM_GEN_FILE = RESULTS_DIR / "random_precedent_generations.json"
OUTPUT_FILE = RESULTS_DIR / "random_precedent_control.json"

CACHE_FILES = {
    "baseline_zeroshot": CACHE_DIR / "zeroshot_generations.json",
    "baseline_rag": CACHE_DIR / "rag_generations.json",
    "gate_v2_zeroshot": CACHE_DIR / "ablation_zero_shot_gate_format.json",
    "gate_v2_rag": CACHE_DIR / "ablation_relevance_gate_v2.json",
}


def clean_rc(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*[*_#]+\s*", "", cleaned)
    cleaned = re.sub(r"\s*[*_#]+\s*$", "", cleaned)
    return cleaned.strip()


class FastClusterBootstrap:
    """
    Optimized two-stage company-clustered bootstrap sampler.
    Pre-samples company and incident draw indices to evaluate arbitrary metric vectors
    in milliseconds while maintaining exact mathematical equivalence to the two-stage process.
    """
    def __init__(self, inc_ids: List[str], id_to_company: Dict[str, str], n_resamples: int = 10000, seed: int = 42):
        self.inc_ids = list(inc_ids)
        self.id_to_company = id_to_company
        self.n_resamples = n_resamples
        self.seed = seed

        rng = np.random.RandomState(seed)
        comp_to_indices: Dict[str, List[int]] = {}
        for idx, inc_id in enumerate(self.inc_ids):
            c = self.id_to_company.get(inc_id, "Unknown")
            comp_to_indices.setdefault(c, []).append(idx)

        self.unique_companies = sorted(list(comp_to_indices.keys()))
        m = len(self.unique_companies)
        sampled_comp_idx = rng.choice(m, size=(n_resamples, m), replace=True)

        self.resample_indices: List[np.ndarray] = []
        for b in range(n_resamples):
            sample_idx = []
            for c_idx in sampled_comp_idx[b]:
                comp = self.unique_companies[c_idx]
                c_idxs = comp_to_indices[comp]
                n_c = len(c_idxs)
                chosen = rng.choice(n_c, size=n_c, replace=True)
                for ch in chosen:
                    sample_idx.append(c_idxs[ch])
            self.resample_indices.append(np.array(sample_idx, dtype=np.int32))

    def compute_multi_ci(
        self,
        vals: List[float],
        unadjusted_alpha: float = 0.05,
        adjusted_alpha: float = 0.01,
    ) -> Dict[str, Any]:
        arr = np.array(vals, dtype=float)
        orig_mean = float(np.mean(arr))

        resampled_means = np.array([np.mean(arr[s]) for s in self.resample_indices], dtype=float)

        # Unadjusted CI (95%)
        unadj_low = float(np.percentile(resampled_means, (unadjusted_alpha / 2.0) * 100))
        unadj_high = float(np.percentile(resampled_means, (1.0 - unadjusted_alpha / 2.0) * 100))
        unadj_excludes_zero = bool(unadj_low > 0.0 or unadj_high < 0.0)

        # Adjusted CI (99% for family m=5)
        adj_low = float(np.percentile(resampled_means, (adjusted_alpha / 2.0) * 100))
        adj_high = float(np.percentile(resampled_means, (1.0 - adjusted_alpha / 2.0) * 100))
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


def compute_arm_scores(
    test_ids: List[str],
    gt_rc_map: Dict[str, str],
    pred_rc_map: Dict[str, str],
    gt_embeddings: np.ndarray,
    scorer: rouge_scorer.RougeScorer,
    median_gt_limit: int,
    bootstrapper: FastClusterBootstrap,
) -> Dict[str, Any]:
    preds = [clean_rc(pred_rc_map[i]) for i in test_ids]
    refs = [clean_rc(gt_rc_map[i]) for i in test_ids]

    r1_list, r2_list, rl_list = [], [], []
    for ref, pred in zip(refs, preds):
        s = scorer.score(ref, pred)
        r1_list.append(s["rouge1"].fmeasure)
        r2_list.append(s["rouge2"].fmeasure)
        rl_list.append(s["rougeL"].fmeasure)

    # Length-matched ROUGE-L
    lm_preds = [" ".join(p.split()[:median_gt_limit]) for p in preds]
    lm_rl_list = [scorer.score(ref, lmp)["rougeL"].fmeasure for ref, lmp in zip(refs, lm_preds)]

    # BERTScore
    safe_preds = [p if p.strip() else "none" for p in preds]
    safe_refs = [r if r.strip() else "none" for r in refs]
    _, _, bs = calc_bertscore(safe_preds, safe_refs, model_type="bert-base-uncased", verbose=False)
    bs_list = bs.numpy().tolist()

    # Embedding cosine similarity
    pred_embeddings = embed(safe_preds)
    pred_norm = pred_embeddings / np.maximum(np.linalg.norm(pred_embeddings, axis=1, keepdims=True), 1e-12)
    gt_norm = gt_embeddings / np.maximum(np.linalg.norm(gt_embeddings, axis=1, keepdims=True), 1e-12)
    cos_list = np.sum(pred_norm * gt_norm, axis=1).tolist()

    # Word counts
    word_counts = [len(p.split()) for p in preds]

    return {
        "rouge1_f1": bootstrapper.compute_multi_ci(r1_list),
        "rouge2_f1": bootstrapper.compute_multi_ci(r2_list),
        "rougeL_f1": bootstrapper.compute_multi_ci(rl_list),
        "length_matched_rougeL": bootstrapper.compute_multi_ci(lm_rl_list),
        "bert_score_f1": bootstrapper.compute_multi_ci(bs_list),
        "cosine": bootstrapper.compute_multi_ci(cos_list),
        "word_count": {
            "mean": round(float(np.mean(word_counts)), 2),
            "median": round(float(np.median(word_counts)), 2),
        },
        "_raw": {
            "rouge1_f1": r1_list,
            "rouge2_f1": r2_list,
            "rougeL_f1": rl_list,
            "length_matched_rougeL": lm_rl_list,
            "bert_score_f1": bs_list,
            "cosine": cos_list,
        }
    }


def compute_paired_deltas(
    arm_a_raw: Dict[str, List[float]],
    arm_b_raw: Dict[str, List[float]],
    bootstrapper: FastClusterBootstrap,
) -> Dict[str, Any]:
    """Computes arm_a minus arm_b paired deltas with company-clustered and Bonferroni CIs."""
    deltas = {}
    metrics = ["rouge1_f1", "rouge2_f1", "rougeL_f1", "length_matched_rougeL", "bert_score_f1", "cosine"]
    for m in metrics:
        diff = [a - b for a, b in zip(arm_a_raw[m], arm_b_raw[m])]
        deltas[m] = bootstrapper.compute_multi_ci(diff)
    return deltas


def evaluate_remediation_pairwise(
    test_ids: List[str],
    id_to_company: Dict[str, str],
    gt_rem_map: Dict[str, str],
    preds_a: Dict[str, str],
    preds_b: Dict[str, str],
    scorer: rouge_scorer.RougeScorer,
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

    rl_a = [scorer.score(g, p)["rougeL"].fmeasure for g, p in zip(gts, a_texts)]
    rl_b = [scorer.score(g, p)["rougeL"].fmeasure for g, p in zip(gts, b_texts)]
    rl_diff = [a - b for a, b in zip(rl_a, rl_b)]

    _, _, bs_a_t = calc_bertscore(a_texts, gts, model_type="bert-base-uncased", verbose=False)
    _, _, bs_b_t = calc_bertscore(b_texts, gts, model_type="bert-base-uncased", verbose=False)
    bs_a = bs_a_t.numpy().tolist()
    bs_b = bs_b_t.numpy().tolist()
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


def determine_verdict(random_minus_zs_excludes: bool, real_minus_random_excludes: bool) -> str:
    """Pre-registered automated decision rule from eval/PREREG.md."""
    if random_minus_zs_excludes and not real_minus_random_excludes:
        return "style_priming"
    elif real_minus_random_excludes:
        return "precedent_content"
    else:
        return "inconclusive"


def run_e1_analysis():
    print("=" * 60)
    print("TraceMind E1: Random-Precedent Control Analysis")
    print("=" * 60)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    test_ids = [r["incident_id"] for r in test_records]
    id_to_company = {r["incident_id"]: r.get("company", "Unknown") for r in test_records}
    gt_rc_map = {r["incident_id"]: r["root_cause"] for r in test_records}
    gt_rem_map = {r["incident_id"]: r.get("remediation_steps", "") for r in test_records}

    # Ground truth root cause stats
    gt_word_counts = [len(gt_rc_map[i].split()) for i in test_ids]
    median_gt_limit = int(round(float(np.median(gt_word_counts))))
    print(f"Loaded {len(test_ids)} test incidents across {len(set(id_to_company.values()))} companies.")
    print(f"Ground truth root cause median length: {median_gt_limit} words.")

    print("\nPre-computing two-stage cluster bootstrap indices (10,000 resamples, seed 42)...")
    bootstrapper = FastClusterBootstrap(test_ids, id_to_company, n_resamples=10000, seed=42)

    print("\nPre-embedding ground truth root causes...")
    clean_gts = [clean_rc(gt_rc_map[i]) for i in test_ids]
    gt_embeddings = embed(clean_gts)

    print("\nLoading cached generations...")
    loaded_cache = {}
    for name, path in CACHE_FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            loaded_cache[name] = json.load(f)

    with open(RANDOM_GEN_FILE, "r", encoding="utf-8") as f:
        random_gen_data = json.load(f)

    random_generations = {item["query_incident_id"]: item for item in random_gen_data["generations"]}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    formats_results = {}

    for fmt in ["baseline", "gate_v2"]:
        print(f"\nEvaluating format: {fmt}...")

        # 1. Zero-shot
        zs_key = f"{fmt}_zeroshot"
        zs_preds_rc = {i: loaded_cache[zs_key][i]["root_cause"] for i in test_ids}
        zs_preds_rem = {i: loaded_cache[zs_key][i].get("remediation_steps", "") for i in test_ids}

        # 2. Random-precedent
        rand_preds_rc = {i: random_generations[i][fmt]["root_cause"] for i in test_ids}
        rand_preds_rem = {i: random_generations[i][fmt].get("remediation_steps", "") for i in test_ids}

        # 3. Real RAG
        rag_key = f"{fmt}_rag"
        rag_preds_rc = {i: loaded_cache[rag_key][i]["root_cause"] for i in test_ids}
        rag_preds_rem = {i: loaded_cache[rag_key][i].get("remediation_steps", "") for i in test_ids}

        print(f"  Scoring arm: zero_shot ({fmt})...")
        arm_zs = compute_arm_scores(test_ids, gt_rc_map, zs_preds_rc, gt_embeddings, scorer, median_gt_limit, bootstrapper)

        print(f"  Scoring arm: random_precedent ({fmt})...")
        arm_rand = compute_arm_scores(test_ids, gt_rc_map, rand_preds_rc, gt_embeddings, scorer, median_gt_limit, bootstrapper)

        print(f"  Scoring arm: real_rag ({fmt})...")
        arm_real = compute_arm_scores(test_ids, gt_rc_map, rag_preds_rc, gt_embeddings, scorer, median_gt_limit, bootstrapper)

        print(f"  Computing paired deltas ({fmt})...")
        delta_real_vs_rand = compute_paired_deltas(arm_real["_raw"], arm_rand["_raw"], bootstrapper)
        delta_rand_vs_zs = compute_paired_deltas(arm_rand["_raw"], arm_zs["_raw"], bootstrapper)
        delta_real_vs_zs = compute_paired_deltas(arm_real["_raw"], arm_zs["_raw"], bootstrapper)

        print(f"  Scoring remediation on both-parsed subsets ({fmt})...")
        rem_real_vs_rand = evaluate_remediation_pairwise(test_ids, id_to_company, gt_rem_map, rag_preds_rem, rand_preds_rem, scorer)
        rem_rand_vs_zs = evaluate_remediation_pairwise(test_ids, id_to_company, gt_rem_map, rand_preds_rem, zs_preds_rem, scorer)
        rem_real_vs_zs = evaluate_remediation_pairwise(test_ids, id_to_company, gt_rem_map, rag_preds_rem, zs_preds_rem, scorer)

        # Check PREREG decision rule on ROUGE-L F1
        rl_rand_zs_excludes = delta_rand_vs_zs["rougeL_f1"]["excludes_zero_after_adjustment"]
        rl_real_rand_excludes = delta_real_vs_rand["rougeL_f1"]["excludes_zero_after_adjustment"]
        fmt_verdict = determine_verdict(rl_rand_zs_excludes, rl_real_rand_excludes)

        print(f"  [Verdict for {fmt}]: {fmt_verdict} (random-zs adj excludes zero: {rl_rand_zs_excludes}, real-random adj excludes zero: {rl_real_rand_excludes})")

        # Strip _raw for clean output JSON
        del arm_zs["_raw"]
        del arm_rand["_raw"]
        del arm_real["_raw"]

        formats_results[fmt] = {
            "arms": {
                "zero_shot": arm_zs,
                "random_precedent": arm_rand,
                "real_rag": arm_real,
            },
            "deltas": {
                "real_minus_random": delta_real_vs_rand,
                "random_minus_zeroshot": delta_rand_vs_zs,
                "real_minus_zeroshot": delta_real_vs_zs,
            },
            "remediation_both_parsed": {
                "real_minus_random": rem_real_vs_rand,
                "random_minus_zeroshot": rem_rand_vs_zs,
                "real_minus_zeroshot": rem_real_vs_zs,
            },
            "verdict": fmt_verdict,
        }

    overall_verdict = formats_results["gate_v2"]["verdict"]

    summary_text = (
        f"Experiment 1 (Random-Precedent Control, N=93): Evaluated whether retrieval gains reflect "
        f"domain-specific precedent content or general exemplar/style priming. Under gate-v2 format, "
        f"zero-shot ROUGE-L F1 is {formats_results['gate_v2']['arms']['zero_shot']['rougeL_f1']['mean']:.4f}, "
        f"random-precedent ROUGE-L F1 is {formats_results['gate_v2']['arms']['random_precedent']['rougeL_f1']['mean']:.4f}, "
        f"and real top-1 RAG ROUGE-L F1 is {formats_results['gate_v2']['arms']['real_rag']['rougeL_f1']['mean']:.4f}. "
        f"Delta random_minus_zeroshot is {formats_results['gate_v2']['deltas']['random_minus_zeroshot']['rougeL_f1']['mean']:+.4f} "
        f"(99% Bonferroni CI {formats_results['gate_v2']['deltas']['random_minus_zeroshot']['rougeL_f1']['adjusted_ci']}, status: {formats_results['gate_v2']['deltas']['random_minus_zeroshot']['rougeL_f1']['status']}). "
        f"Delta real_minus_random is {formats_results['gate_v2']['deltas']['real_minus_random']['rougeL_f1']['mean']:+.4f} "
        f"(99% Bonferroni CI {formats_results['gate_v2']['deltas']['real_minus_random']['rougeL_f1']['adjusted_ci']}, status: {formats_results['gate_v2']['deltas']['real_minus_random']['rougeL_f1']['status']}). "
        f"Per PREREG.md criteria, the automated verdict is '{overall_verdict}'."
    )

    results_obj = {
        "metadata": {
            "num_incidents": len(test_ids),
            "num_companies": len(set(id_to_company.values())),
            "bootstrap_resamples": 10000,
            "seed": 42,
            "family_size_m": 5,
            "adjusted_alpha": 0.01,
            "unadjusted_alpha": 0.05,
            "gt_median_word_count": median_gt_limit,
        },
        "formats": formats_results,
        "verdict": overall_verdict,
        "summary": summary_text,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_obj, f, indent=2)

    print(f"\n[Artifact] Successfully wrote E1 control results to {OUTPUT_FILE}")
    print(f"Overall verdict: {overall_verdict}")
    print("=" * 60)


if __name__ == "__main__":
    run_e1_analysis()
