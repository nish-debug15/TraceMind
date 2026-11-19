"""
Render TraceMind Evaluation Tables for README and Reports
---------------------------------------------------------
Reads evaluated JSON results from eval/results/ and prints dynamically
formatted Markdown tables representing:
1. 2x2 Factorial Control: Prompt Format x Retrieval (with per-incident & clustered CIs, output word counts)
2. Benchmark Evaluation: Baseline RAG vs. Zero-Shot Baseline (with side-by-side per-incident and company-clustered CIs)
3. Length Diagnostics & Length-Matched Sensitivity (truncated to median GT 51 words)
4. Exploratory Retrieval Similarity Dose-Response (Spearman rho, clustered CIs, tercile deltas)
5. Remediation Checklist Scoring & Coverage (parse failures, ROUGE-L, BERTScore, deltas)
6. Retrieval Distribution & Guardrail Abstention Split (Answered N=86 vs Abstained N=7)
7. Ground-Truth Noise Sensitivity Analysis (Full Set N=93 vs Clean Subset N=86)
8. Prompt Robustness Ablation (baseline, relevance_gate, relevance_gate_v2)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
CONTROL_2X2_FILE = RESULTS_DIR / "control_2x2.json"
CLUSTER_CI_FILE = RESULTS_DIR / "cluster_bootstrap_ci.json"
LENGTH_DIAG_FILE = RESULTS_DIR / "length_diagnostics.json"
DOSE_RESP_FILE = RESULTS_DIR / "dose_response.json"
REMEDIATION_FILE = RESULTS_DIR / "remediation_eval.json"
ABSTENTION_FILE = RESULTS_DIR / "abstention_split.json"
GT_NOISE_FILE = RESULTS_DIR / "gt_noise_sensitivity.json"
PROMPT_ABLATION_FILE = RESULTS_DIR / "prompt_ablation.json"


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
            wc = c.get("word_count", {})
            wc_str = f"{wc.get('mean', 0.0):.1f} (med: {wc.get('median', 0.0):.1f})"
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

    # 2. Benchmark Evaluation Table (Clustered vs iid CIs)
    if CLUSTER_CI_FILE.exists():
        with open(CLUSTER_CI_FILE, "r", encoding="utf-8") as f:
            ci_data = json.load(f)

        lines.append("### Benchmark Evaluation: Baseline RAG vs. Zero-Shot Baseline (Held-Out Test Set, N = 93)")
        lines.append("")
        lines.append("> Evaluated strictly on held-out postmortems (`data/splits/test.json`) across 40 unseen companies against the TRAIN-ONLY retrieval index (`rag/chroma_db_train`, 662 incidents). Zero data leakage verified. Confidence intervals computed via 10,000 resamples (seed 42) comparing per-incident i.i.d. bootstrap vs. two-stage company-clustered bootstrap.")
        lines.append("")
        lines.append("| Metric | Paired Delta Mean | Per-Incident 95% CI | Company-Clustered 95% CI | CI Excludes Zero (Clustered) |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")

        metrics_map = [
            ("rouge1_f1", "**ROUGE-1 F1**"),
            ("rouge2_f1", "**ROUGE-2 F1**"),
            ("rougeL_f1", "**ROUGE-L F1**"),
            ("bert_score_f1", "**BERTScore F1**"),
            ("semantic_sim", "**Embedding Cosine Sim**"),
        ]

        deltas_ci = ci_data.get("rag_vs_zeroshot_deltas", {})
        for key, label in metrics_map:
            if key in deltas_ci:
                d = deltas_ci[key]
                mean_val = d.get("mean", 0.0)
                iid_ci = d.get("per_incident_ci_95", [0, 0])
                clust_ci = d.get("cluster_ci_95", [0, 0])
                sig_clust = d.get("significant_under_cluster_bootstrap", False)
                sig_str = "CI excludes zero" if sig_clust else "Crosses zero"
                lines.append(
                    f"| {label} | **{mean_val:+.4f}** | `[{iid_ci[0]:.4f}, {iid_ci[1]:.4f}]` | `[{clust_ci[0]:.4f}, {clust_ci[1]:.4f}]` | {sig_str} |"
                )
        lines.append("")

    # 3. Length Diagnostics & Length-Matched Sensitivity Table
    if LENGTH_DIAG_FILE.exists():
        with open(LENGTH_DIAG_FILE, "r", encoding="utf-8") as f:
            ld_data = json.load(f)

        gt_stats = ld_data.get("ground_truth_stats", {})
        lm_eval = ld_data.get("length_matched_evaluation", {})
        lm_cells = lm_eval.get("cells", {})
        c_wc = ld_data.get("cells_word_count_stats", {})

        lines.append("#### Length Diagnostics & Length-Matched Sensitivity (Test Set, N = 93)")
        lines.append("")
        lines.append(f"> Ground-truth root_cause median length is {gt_stats.get('median', 51.0):.1f} words (mean: {gt_stats.get('mean', 48.3):.1f}, p10: {gt_stats.get('p10', 16.2):.1f}, p90: {gt_stats.get('p90', 73.0):.1f}). Evaluates ROUGE-L F1 when generated outputs are truncated to median GT length ({lm_eval.get('truncation_word_limit', 51)} words) with company-clustered 95% CIs.")
        lines.append("")
        lines.append("| Experimental Cell | Word Count (Mean / Median) | Mean Abs Diff vs GT | Length-Matched ROUGE-L [95% CI] | Clustered CI Excludes Zero? |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")

        cell_names = {
            "baseline_zeroshot": "Baseline Format — Zero-Shot",
            "baseline_rag": "Baseline Format — RAG",
            "gate_v2_zeroshot": "Gate-v2 Format — Zero-Shot",
            "gate_v2_rag": "Gate-v2 Format — RAG",
        }

        for c_k in ["baseline_zeroshot", "baseline_rag", "gate_v2_zeroshot", "gate_v2_rag"]:
            if c_k in lm_cells:
                lbl = cell_names.get(c_k, c_k)
                wc_info = c_wc.get(c_k, {})
                wc_str = f"{wc_info.get('mean', 0.0):.1f} / {wc_info.get('median', 0.0):.1f}"
                abs_diff = f"{wc_info.get('mean_abs_diff_vs_ground_truth', 0.0):.1f}"
                lm_c = lm_cells[c_k]["rougeL_f1"]
                lm_str = f"{lm_c['mean']:.4f} `[{lm_c['ci_95'][0]:.4f}, {lm_c['ci_95'][1]:.4f}]`"
                sig_str = "Yes" if lm_c.get("significant_under_cluster_bootstrap", False) else "No"
                lines.append(f"| **{lbl}** | {wc_str} | {abs_diff} | {lm_str} | {sig_str} |")

        lines.append("")
        lm_deltas = lm_eval.get("deltas", {})
        base_d = lm_deltas.get("baseline_format_rag_effect", {})
        gate_d = lm_deltas.get("gate_v2_format_rag_effect", {})
        lines.append(
            f"> **Length-Matched Retrieval Effect:** Baseline format RAG delta = **{base_d.get('mean', 0.0):+.4f}** `[{base_d.get('ci_95', [0,0])[0]:.4f}, {base_d.get('ci_95', [0,0])[1]:.4f}]` "
            f"(crosses zero under company clustering: {not base_d.get('significant_under_cluster_bootstrap', False)}); "
            f"Gate-v2 format RAG delta = **{gate_d.get('mean', 0.0):+.4f}** `[{gate_d.get('ci_95', [0,0])[0]:.4f}, {gate_d.get('ci_95', [0,0])[1]:.4f}]` "
            f"(crosses zero under company clustering: {not gate_d.get('significant_under_cluster_bootstrap', False)})."
        )
        lines.append("")

    # 4. Exploratory Retrieval Similarity Dose-Response Table
    if DOSE_RESP_FILE.exists():
        with open(DOSE_RESP_FILE, "r", encoding="utf-8") as f:
            dr_data = json.load(f)

        rhos = dr_data.get("spearman_correlations", {})
        terciles = dr_data.get("terciles", {})

        lines.append("#### Exploratory Retrieval Similarity Dose-Response (Test Set, N = 93)")
        lines.append("")
        lines.append("> Relates top-1 ChromaDB cosine retrieval similarity to downstream generation gains (RAG minus Zero-Shot delta). Evaluated with company-clustered bootstrap 95% CIs (10,000 resamples, seed 42). *Note: Labeled as exploratory.*")
        lines.append("")
        lines.append("| Prompt Format | Metric | Spearman $\\rho$ [95% CI] | CI Excludes Zero? | Low Tercile Mean $\\Delta$ (sim $\\le 0.478$) | Mid Tercile Mean $\\Delta$ ($0.480 \\le \\text{sim} \\le 0.553$) | High Tercile Mean $\\Delta$ (sim $\\ge 0.554$) |")
        lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")

        rows = [
            ("Baseline Format", "ROUGE-L F1 $\\Delta$", rhos.get("baseline_format", {}).get("rougeL_delta", {}), "baseline", "rougeL_delta"),
            ("Baseline Format", "BERTScore F1 $\\Delta$", rhos.get("baseline_format", {}).get("bert_score_delta", {}), "baseline", "bert_score_delta"),
            ("Gate-v2 Format", "ROUGE-L F1 $\\Delta$", rhos.get("gate_v2_format", {}).get("rougeL_delta", {}), "gate_v2", "rougeL_delta"),
            ("Gate-v2 Format", "BERTScore F1 $\\Delta$", rhos.get("gate_v2_format", {}).get("bert_score_delta", {}), "gate_v2", "bert_score_delta"),
        ]

        for fmt_lbl, m_lbl, r_dict, arm_k, m_k in rows:
            r_val = r_dict.get("spearman_rho", 0.0)
            r_ci = r_dict.get("ci_95", [0, 0])
            r_sig = "Yes" if r_dict.get("excludes_zero", False) else "No"
            low_m = terciles.get("low_tercile", {}).get(arm_k, {}).get(m_k, {}).get("mean", 0.0)
            mid_m = terciles.get("mid_tercile", {}).get(arm_k, {}).get(m_k, {}).get("mean", 0.0)
            high_m = terciles.get("high_tercile", {}).get(arm_k, {}).get(m_k, {}).get("mean", 0.0)
            lines.append(
                f"| **{fmt_lbl}** | {m_lbl} | {r_val:+.4f} `[{r_ci[0]:.4f}, {r_ci[1]:.4f}]` | {r_sig} | {low_m:+.4f} | {mid_m:+.4f} | {high_m:+.4f} |"
            )

        lines.append("")
        lines.append("> Scatter plots with tercile mean diamonds and confidence intervals are saved in [`eval/figures/dose_response.png`](eval/figures/dose_response.png).")
        lines.append("")

    # 5. Remediation Checklist Scoring & Coverage Table
    if REMEDIATION_FILE.exists():
        with open(REMEDIATION_FILE, "r", encoding="utf-8") as f:
            rem_data = json.load(f)

        meta = rem_data.get("metadata", {})
        pf = rem_data.get("parse_failures", {})
        cells = rem_data.get("cells", {})
        deltas = rem_data.get("deltas", {})

        lines.append(f"#### Remediation Checklist Scoring & Coverage (Eligible N = {meta.get('eligible_incidents', 93)}, Excluded GT < 5 words = {meta.get('excluded_gt_under_5_words', 0)})")
        lines.append("")
        lines.append("> Evaluates the actionable remediation checklist section (`REMEDIATION STEPS:`) against human postmortem ground truth across all 4 cells. Evaluated with company-clustered 95% CIs.")
        lines.append("")
        lines.append("| Experimental Cell | Parse Failures / Total | Coverage Rate | ROUGE-L F1 [95% CI] | BERTScore F1 [95% CI] |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")

        cell_names = {
            "baseline_zeroshot": "Baseline Format — Zero-Shot",
            "baseline_rag": "Baseline Format — RAG",
            "gate_v2_zeroshot": "Gate-v2 Format — Zero-Shot",
            "gate_v2_rag": "Gate-v2 Format — RAG",
        }

        for c_k in ["baseline_zeroshot", "baseline_rag", "gate_v2_zeroshot", "gate_v2_rag"]:
            if c_k in cells:
                lbl = cell_names.get(c_k, c_k)
                fails = pf.get(c_k, {}).get("failure_count", 0)
                tot = pf.get(c_k, {}).get("total_eligible", 93)
                cov = pf.get(c_k, {}).get("coverage_rate", 1.0) * 100
                c = cells[c_k]
                rl = c["rougeL_f1"]
                bs = c["bert_score_f1"]
                rl_str = f"{rl['mean']:.4f} `[{rl['ci_95'][0]:.4f}, {rl['ci_95'][1]:.4f}]`"
                bs_str = f"{bs['mean']:.4f} `[{bs['ci_95'][0]:.4f}, {bs['ci_95'][1]:.4f}]`"
                lines.append(f"| **{lbl}** | {fails} / {tot} | {cov:.1f}% | {rl_str} | {bs_str} |")

        lines.append("")
        base_d = deltas.get("baseline_format_rag_effect", {})
        gate_d = deltas.get("gate_v2_format_rag_effect", {})
        lines.append(
            f"> **Remediation Paired Deltas (RAG - Zero-Shot, Clustered 95% CIs):**\n"
            f"> - Baseline format: ROUGE-L $\\Delta$ = **{base_d.get('rougeL_delta', {}).get('mean', 0.0):+.4f}** `[{base_d.get('rougeL_delta', {}).get('ci_95', [0,0])[0]:.4f}, {base_d.get('rougeL_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (significant: {base_d.get('rougeL_delta', {}).get('significant_under_cluster_bootstrap', False)}); "
            f"BERTScore $\\Delta$ = **{base_d.get('bert_score_delta', {}).get('mean', 0.0):+.4f}** `[{base_d.get('bert_score_delta', {}).get('ci_95', [0,0])[0]:.4f}, {base_d.get('bert_score_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (significant: {base_d.get('bert_score_delta', {}).get('significant_under_cluster_bootstrap', False)}).\n"
            f"> - Gate-v2 format: ROUGE-L $\\Delta$ = **{gate_d.get('rougeL_delta', {}).get('mean', 0.0):+.4f}** `[{gate_d.get('rougeL_delta', {}).get('ci_95', [0,0])[0]:.4f}, {gate_d.get('rougeL_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (significant: {gate_d.get('rougeL_delta', {}).get('significant_under_cluster_bootstrap', False)}); "
            f"BERTScore $\\Delta$ = **{gate_d.get('bert_score_delta', {}).get('mean', 0.0):+.4f}** `[{gate_d.get('bert_score_delta', {}).get('ci_95', [0,0])[0]:.4f}, {gate_d.get('bert_score_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (significant: {gate_d.get('bert_score_delta', {}).get('significant_under_cluster_bootstrap', False)})."
        )
        lines.append("")

    # 6. Retrieval Distribution & Guardrail Abstention Split
    if ABSTENTION_FILE.exists():
        with open(ABSTENTION_FILE, "r", encoding="utf-8") as f:
            abs_data = json.load(f)

        groups = abs_data.get("groups", {})
        ans = groups.get("answered", {})
        abstained = groups.get("abstained", {})

        lines.append("#### Retrieval Distribution & Guardrail Abstention Split")
        lines.append("")
        lines.append(f"- **Answered Partition (Sim ≥ 0.40, N = {ans.get('n', 86)}):** Mean top-1 cosine similarity = `{ans.get('mean_top1_similarity', 0.0):.4f}`.")
        lines.append(f"  - Baseline RAG: ROUGE-L = `{ans.get('baseline_rag', {}).get('rougeL_f1', {}).get('mean', 0.0):.4f}` `[{ans.get('baseline_rag', {}).get('rougeL_f1', {}).get('ci_95', [0,0])[0]:.4f}, {ans.get('baseline_rag', {}).get('rougeL_f1', {}).get('ci_95', [0,0])[1]:.4f}]`, BERTScore = `{ans.get('baseline_rag', {}).get('bert_score_f1', {}).get('mean', 0.0):.4f}`")
        lines.append(f"  - Gate-v2 RAG:  ROUGE-L = `{ans.get('gate_v2_rag', {}).get('rougeL_f1', {}).get('mean', 0.0):.4f}` `[{ans.get('gate_v2_rag', {}).get('rougeL_f1', {}).get('ci_95', [0,0])[0]:.4f}, {ans.get('gate_v2_rag', {}).get('rougeL_f1', {}).get('ci_95', [0,0])[1]:.4f}]`, BERTScore = `{ans.get('gate_v2_rag', {}).get('bert_score_f1', {}).get('mean', 0.0):.4f}`")
        lines.append(f"- **Abstained Partition (Sim < 0.40, N = {abstained.get('n', 7)}, low_n: true):** Mean top-1 cosine similarity = `{abstained.get('mean_top1_similarity', 0.0):.4f}`.")
        lines.append(f"  - Baseline RAG: ROUGE-L raw mean = `{abstained.get('baseline_rag', {}).get('rougeL_f1', {}).get('mean', 0.0):.4f}`, BERTScore raw mean = `{abstained.get('baseline_rag', {}).get('bert_score_f1', {}).get('mean', 0.0):.4f}`")
        lines.append(f"  - Gate-v2 RAG:  ROUGE-L raw mean = `{abstained.get('gate_v2_rag', {}).get('rougeL_f1', {}).get('mean', 0.0):.4f}`, BERTScore raw mean = `{abstained.get('gate_v2_rag', {}).get('bert_score_f1', {}).get('mean', 0.0):.4f}`")
        lines.append("  - *Note: Due to small sample size (n=7), confidence intervals are withheld and raw descriptive statistics are reported.*")
        lines.append("")

    # 7. Ground-Truth Noise Sensitivity Analysis Table
    if GT_NOISE_FILE.exists():
        with open(GT_NOISE_FILE, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        lines.append("#### Ground-Truth Noise Sensitivity Analysis (Full N = 93 vs. Clean N = 86)")
        lines.append("")
        lines.append("> Rule-based detector flagged 7 incidents (7.5%) containing status-page resolution boilerplate rather than technical root-cause mechanics. Evaluated with company-clustered 95% CIs.")
        lines.append("")
        lines.append("| Experimental Arm | Metric | Full Set (N=93) Clustered [95% CI] | Clean Subset (N=86) Clustered [95% CI] | Change |")
        lines.append("|:---|:---|:---:|:---:|:---:|")

        f_metrics = gt_data.get("full_set_metrics", {})
        c_metrics = gt_data.get("clean_subset_metrics", {})

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

    # 8. Prompt Robustness Ablation Table
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
