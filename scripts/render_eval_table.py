"""
Render TraceMind Evaluation Tables for README and Reports
---------------------------------------------------------
Reads evaluated JSON results from eval/results/ and prints dynamically
formatted Markdown tables representing:
1. 2x2 Factorial Control: Prompt Format x Retrieval (with per-incident & clustered CIs, output word counts)
2. Benchmark Evaluation: RAG vs. Zero-Shot Baseline (with side-by-side per-incident and company-clustered CIs)
3. Retrieval Distribution & Guardrail Abstention Split (Answered N=86 vs Abstained N=7)
4. Ground-Truth Noise Sensitivity Analysis (Full Set N=93 vs Clean Subset N=86)
5. Prompt Robustness Ablation (baseline, relevance_gate, relevance_gate_v2)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
RAG_VS_ZS_FILE = RESULTS_DIR / "rag_vs_zero_shot.json"
METRICS_CI_FILE = RESULTS_DIR / "metrics_with_ci.json"
PROMPT_ABLATION_FILE = RESULTS_DIR / "prompt_ablation.json"
CONTROL_2X2_FILE = RESULTS_DIR / "control_2x2.json"
CLUSTER_CI_FILE = RESULTS_DIR / "cluster_bootstrap_ci.json"
ABSTENTION_FILE = RESULTS_DIR / "abstention_split.json"
GT_NOISE_FILE = RESULTS_DIR / "gt_noise_sensitivity.json"


def render_markdown_tables() -> str:
    lines = []

    # 1. 2x2 Factorial Control Table
    if CONTROL_2X2_FILE.exists():
        with open(CONTROL_2X2_FILE, "r", encoding="utf-8") as f:
            c2_data = json.load(f)

        cells = c2_data.get("cells", {})
        deltas = c2_data.get("deltas", {})
        n = c2_data.get("evaluation_config", {}).get("num_samples", 93)

        lines.append(f"### 2x2 Factorial Control: Prompt Format × Retrieval (Held-Out Test Set, N = {n})")
        lines.append("")
        lines.append("> Evaluates whether generation gains stem from historical precedent retrieval or output conciseness/formatting instructions. 95% bootstrap confidence intervals computed across 10,000 resamples (seed 42).")
        lines.append("")
        lines.append("| Experimental Cell | ROUGE-L F1 [95% CI] | ROUGE-L Precision | ROUGE-L Recall | BERTScore F1 | Mean Word Count |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")

        cell_labels = {
            "baseline_zeroshot": "Baseline Format — Zero-Shot",
            "baseline_rag": "Baseline Format — RAG",
            "gate_v2_zeroshot": "Gate-v2 Format — Zero-Shot",
            "gate_v2_rag": "Gate-v2 Format — RAG",
        }

        for c_key in ["baseline_zeroshot", "baseline_rag", "gate_v2_zeroshot", "gate_v2_rag"]:
            if c_key not in cells:
                continue
            c = cells[c_key]
            lbl = cell_labels.get(c_key, c_key)
            rl_str = f"{c['rougeL_f1']['mean']:.4f} `[{c['rougeL_f1']['ci_95'][0]:.4f}, {c['rougeL_f1']['ci_95'][1]:.4f}]`"
            p_str = f"{c['rougeL_precision']['mean']:.4f}"
            r_str = f"{c['rougeL_recall']['mean']:.4f}"
            bs_str = f"{c['bert_score_f1']['mean']:.4f}"
            wc_str = f"{c['word_count']['mean']:.1f} (med: {c['word_count']['median']:.1f})"
            lines.append(f"| **{lbl}** | {rl_str} | {p_str} | {r_str} | {bs_str} | {wc_str} |")

        lines.append("")
        lines.append("#### Factorial Paired Deltas")
        lines.append("")
        lines.append("| Comparison | ROUGE-L Delta [95% CI] | BERTScore Delta [95% CI] | CI Excludes Zero? | Primary Driver |")
        lines.append("|:---|:---:|:---:|:---:|:---|")

        rag_eff = deltas.get("rag_effect_within_format", {})
        fmt_eff = deltas.get("format_effect_within_retrieval_arm", {})

        if "within_baseline_format" in rag_eff:
            d = rag_eff["within_baseline_format"]
            rl_d = f"**{d['rougeL_f1']['mean']:+.4f}** `[{d['rougeL_f1']['ci_95'][0]:.4f}, {d['rougeL_f1']['ci_95'][1]:.4f}]`"
            bs_d = f"**{d['bert_score_f1']['mean']:+.4f}** `[{d['bert_score_f1']['ci_95'][0]:.4f}, {d['bert_score_f1']['ci_95'][1]:.4f}]`"
            sig = "Yes" if d['rougeL_f1']['excludes_zero'] else "No"
            lines.append(f"| **RAG Effect** (within Baseline format) | {rl_d} | {bs_d} | {sig} | Precedent Retrieval |")

        if "within_gate_v2_format" in rag_eff:
            d = rag_eff["within_gate_v2_format"]
            rl_d = f"**{d['rougeL_f1']['mean']:+.4f}** `[{d['rougeL_f1']['ci_95'][0]:.4f}, {d['rougeL_f1']['ci_95'][1]:.4f}]`"
            bs_d = f"**{d['bert_score_f1']['mean']:+.4f}** `[{d['bert_score_f1']['ci_95'][0]:.4f}, {d['bert_score_f1']['ci_95'][1]:.4f}]`"
            sig = "Yes" if d['rougeL_f1']['excludes_zero'] else "No"
            lines.append(f"| **RAG Effect** (within Gate-v2 format) | {rl_d} | {bs_d} | {sig} | Precedent Retrieval |")

        if "within_zeroshot_arm" in fmt_eff:
            d = fmt_eff["within_zeroshot_arm"]
            rl_d = f"**{d['rougeL_f1']['mean']:+.4f}** `[{d['rougeL_f1']['ci_95'][0]:.4f}, {d['rougeL_f1']['ci_95'][1]:.4f}]`"
            bs_d = f"**{d['bert_score_f1']['mean']:+.4f}** `[{d['bert_score_f1']['ci_95'][0]:.4f}, {d['bert_score_f1']['ci_95'][1]:.4f}]`"
            sig = "Yes" if d['rougeL_f1']['excludes_zero'] else "No"
            lines.append(f"| **Format Effect** (within Zero-Shot arm) | {rl_d} | {bs_d} | {sig} | Conciseness / Precision |")

        if "within_rag_arm" in fmt_eff:
            d = fmt_eff["within_rag_arm"]
            rl_d = f"**{d['rougeL_f1']['mean']:+.4f}** `[{d['rougeL_f1']['ci_95'][0]:.4f}, {d['rougeL_f1']['ci_95'][1]:.4f}]`"
            bs_d = f"**{d['bert_score_f1']['mean']:+.4f}** `[{d['bert_score_f1']['ci_95'][0]:.4f}, {d['bert_score_f1']['ci_95'][1]:.4f}]`"
            sig = "Yes" if d['rougeL_f1']['excludes_zero'] else "No"
            lines.append(f"| **Format Effect** (within RAG arm) | {rl_d} | {bs_d} | {sig} | Conciseness / Precision |")

        lines.append("")
        if "summary" in c2_data:
            lines.append(f"> **2x2 Control Finding:** {c2_data['summary']}")
            lines.append("")

    # 2. Main Benchmark Table with Side-by-Side Per-Incident and Company-Clustered CIs
    if CLUSTER_CI_FILE.exists():
        with open(CLUSTER_CI_FILE, "r", encoding="utf-8") as f:
            cl_data = json.load(f)

        n = cl_data.get("metadata", {}).get("num_incidents", 93)
        m = cl_data.get("metadata", {}).get("num_unique_companies", 40)
        deltas_main = cl_data.get("rag_vs_zeroshot_deltas", {})

        lines.append(f"### Benchmark Evaluation: Baseline RAG vs. Zero-Shot Baseline (Held-Out Test Set, N = {n})")
        lines.append("")
        lines.append(f"> Evaluated strictly on held-out postmortems (`data/splits/test.json`) across {m} unseen companies against the TRAIN-ONLY retrieval index (`rag/chroma_db_train`, 662 incidents). Zero data leakage verified. Confidence intervals computed via 10,000 resamples (seed 42) comparing per-incident i.i.d. bootstrap vs. two-stage company-clustered bootstrap.")
        lines.append("")
        lines.append("| Metric | Paired Delta Mean | Per-Incident 95% CI | Company-Clustered 95% CI | CI Excludes Zero (Clustered) |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")

        labels = {
            "rouge1_f1": "ROUGE-1 F1",
            "rouge2_f1": "ROUGE-2 F1",
            "rougeL_f1": "ROUGE-L F1",
            "bert_score_f1": "BERTScore F1",
            "semantic_sim": "Embedding Cosine Sim",
        }

        for k in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bert_score_f1", "semantic_sim"]:
            if k not in deltas_main:
                continue
            d = deltas_main[k]
            lbl = labels.get(k, k)
            d_mean = f"**{d['mean']:+.4f}**"
            pi_str = f"`[{d['per_incident_ci_95'][0]:.4f}, {d['per_incident_ci_95'][1]:.4f}]`"
            cl_str = f"`[{d['cluster_ci_95'][0]:.4f}, {d['cluster_ci_95'][1]:.4f}]`"
            sig_str = "CI excludes zero" if d["significant_under_cluster_bootstrap"] else "Crosses zero"
            lines.append(f"| **{lbl}** | {d_mean} | {pi_str} | {cl_str} | {sig_str} |")

        lines.append("")

    # 3. Retrieval Distribution & Guardrail Abstention Split
    if ABSTENTION_FILE.exists():
        with open(ABSTENTION_FILE, "r", encoding="utf-8") as f:
            ab_data = json.load(f)

        groups = ab_data.get("groups", {})
        ans = groups.get("answered", {})
        abs_g = groups.get("abstained", {})

        lines.append("#### Retrieval Distribution & Guardrail Abstention Split")
        lines.append("")
        lines.append(f"- **Answered Partition (Sim ≥ 0.40, N = {ans.get('n', 86)}):** Mean top-1 cosine similarity = `{ans.get('mean_top1_similarity', 0.0):.4f}`.")
        lines.append(f"  - Baseline RAG: ROUGE-L = `{ans.get('baseline_rag', {}).get('rougeL_f1', {}).get('mean', 0.0):.4f}` `[{ans.get('baseline_rag', {}).get('rougeL_f1', {}).get('ci_95', [0,0])[0]:.4f}, {ans.get('baseline_rag', {}).get('rougeL_f1', {}).get('ci_95', [0,0])[1]:.4f}]`, BERTScore = `{ans.get('baseline_rag', {}).get('bert_score_f1', {}).get('mean', 0.0):.4f}`")
        lines.append(f"  - Gate-v2 RAG:  ROUGE-L = `{ans.get('gate_v2_rag', {}).get('rougeL_f1', {}).get('mean', 0.0):.4f}` `[{ans.get('gate_v2_rag', {}).get('rougeL_f1', {}).get('ci_95', [0,0])[0]:.4f}, {ans.get('gate_v2_rag', {}).get('rougeL_f1', {}).get('ci_95', [0,0])[1]:.4f}]`, BERTScore = `{ans.get('gate_v2_rag', {}).get('bert_score_f1', {}).get('mean', 0.0):.4f}`")
        lines.append(f"- **Abstained Partition (Sim < 0.40, N = {abs_g.get('n', 7)}, low_n: true):** Mean top-1 cosine similarity = `{abs_g.get('mean_top1_similarity', 0.0):.4f}`.")
        lines.append(f"  - Baseline RAG: ROUGE-L raw mean = `{abs_g.get('baseline_rag', {}).get('rougeL_f1', {}).get('mean', 0.0):.4f}`, BERTScore raw mean = `{abs_g.get('baseline_rag', {}).get('bert_score_f1', {}).get('mean', 0.0):.4f}`")
        lines.append(f"  - Gate-v2 RAG:  ROUGE-L raw mean = `{abs_g.get('gate_v2_rag', {}).get('rougeL_f1', {}).get('mean', 0.0):.4f}`, BERTScore raw mean = `{abs_g.get('gate_v2_rag', {}).get('bert_score_f1', {}).get('mean', 0.0):.4f}`")
        lines.append("  - *Note: Due to small sample size (n=7), confidence intervals are withheld and raw descriptive statistics are reported.*")
        lines.append("")

    # 4. Ground-Truth Noise Sensitivity Table
    if GT_NOISE_FILE.exists():
        with open(GT_NOISE_FILE, "r", encoding="utf-8") as f:
            gtn_data = json.load(f)

        meta = gtn_data.get("metadata", {})
        f_metrics = gtn_data.get("full_set_metrics", {})
        c_metrics = gtn_data.get("clean_subset_metrics", {})

        lines.append(f"#### Ground-Truth Noise Sensitivity Analysis (Full N = {meta.get('full_sample_size', 93)} vs. Clean N = {meta.get('unflagged_sample_size', 86)})")
        lines.append("")
        lines.append(f"> Rule-based detector flagged {meta.get('flagged_count', 7)} incidents ({meta.get('noise_rate', 0.0753):.1%}) containing status-page resolution boilerplate rather than technical root-cause mechanics. Evaluated with company-clustered 95% CIs.")
        lines.append("")
        lines.append("| Experimental Arm | Metric | Full Set (N=93) Clustered [95% CI] | Clean Subset (N=86) Clustered [95% CI] | Change |")
        lines.append("|:---|:---|:---:|:---:|:---:|")

        arm_names = {
            "zero_shot": "Zero-Shot Baseline",
            "baseline_rag": "Baseline RAG",
            "gate_v2_rag": "Gate-v2 RAG"
        }

        for a_key in ["zero_shot", "baseline_rag", "gate_v2_rag"]:
            a_lbl = arm_names.get(a_key, a_key)
            for m_k, m_lbl in [("rougeL_f1", "ROUGE-L F1"), ("bert_score_f1", "BERTScore F1"), ("cosine", "Cosine Sim")]:
                f_s = f_metrics.get(a_key, {}).get(m_k, {})
                c_s = c_metrics.get(a_key, {}).get(m_k, {})
                f_str = f"{f_s.get('mean', 0.0):.4f} `[{f_s.get('ci_95', [0,0])[0]:.4f}, {f_s.get('ci_95', [0,0])[1]:.4f}]`"
                c_str = f"{c_s.get('mean', 0.0):.4f} `[{c_s.get('ci_95', [0,0])[0]:.4f}, {c_s.get('ci_95', [0,0])[1]:.4f}]`"
                diff = c_s.get("mean", 0.0) - f_s.get("mean", 0.0)
                diff_str = f"{diff:+.4f}"
                lines.append(f"| {a_lbl} | {m_lbl} | {f_str} | {c_str} | **{diff_str}** |")

        lines.append("")

    # 5. Prompt Robustness Ablation Table
    if PROMPT_ABLATION_FILE.exists():
        with open(PROMPT_ABLATION_FILE, "r", encoding="utf-8") as f:
            p_data = json.load(f)

        lines.append("#### Prompt Robustness Ablation (Test Set, N = 93)")
        lines.append("")
        lines.append("| Prompt Variant | ROUGE-L F1 | BERTScore F1 | Cosine Similarity | Guardrail Structure |")
        lines.append("|:---|:---:|:---:|:---:|:---|")

        variants = p_data.get("variants", {})
        for name, stats in variants.items():
            rl = stats.get("rougeL_f1", 0.0)
            bs = stats.get("bert_score_f1", 0.0)
            cos = stats.get("cosine_sim", 0.0)
            guard = "Strict machine-parseable `Precedent:` verdict header" if "v2" in name else ("Soft instruction check" if "gate" in name else "None")
            lines.append(f"| `{name}` | {rl:.4f} | {bs:.4f} | {cos:.4f} | {guard} |")

        if "spread" in p_data:
            sp = p_data["spread"]
            lines.append("")
            lines.append(f"> **Prompt Spread across variants**: ROUGE-L spread: `{sp.get('rougeL_f1_spread', 0.0):.4f}`, BERTScore spread: `{sp.get('bert_score_f1_spread', 0.0):.4f}`, Cosine spread: `{sp.get('cosine_sim_spread', 0.0):.4f}`.")
        lines.append("")

    return "\n".join(lines)


def main():
    table_md = render_markdown_tables()
    print(table_md)


if __name__ == "__main__":
    main()
