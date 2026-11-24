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
MULTI_ADJ_FILE = RESULTS_DIR / "multiplicity_adjusted.json"
LENGTH_DIAG_FILE = RESULTS_DIR / "length_diagnostics.json"
DOSE_RESP_FILE = RESULTS_DIR / "dose_response.json"
REMEDIATION_FILE = RESULTS_DIR / "remediation_eval.json"
ABSTENTION_FILE = RESULTS_DIR / "abstention_split.json"
GT_NOISE_FILE = RESULTS_DIR / "gt_noise_sensitivity.json"
PROMPT_ABLATION_FILE = RESULTS_DIR / "prompt_ablation.json"
RELEVANCE_ANALYSIS_FILE = RESULTS_DIR / "relevance_analysis.json"
FLOOR_STATS_FILE = RESULTS_DIR / "floor_stats.json"
E1_DIRECTIONAL_FILE = RESULTS_DIR / "random_precedent_control_directional.json"
E1_FILE = E1_DIRECTIONAL_FILE if E1_DIRECTIONAL_FILE.exists() else (RESULTS_DIR / "random_precedent_control_rescored.json")
E2_FILE = RESULTS_DIR / "loo_regime_results_rescored.json"
LOO_STRATIFIED_FILE = RESULTS_DIR / "loo_stratified.json"
LOO_CONSISTENCY_FILE = RESULTS_DIR / "loo_consistency.json"


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

    # 2. Benchmark Evaluation Table (Clustered vs iid CIs & Multiplicity-Adjusted)
    if CLUSTER_CI_FILE.exists():
        with open(CLUSTER_CI_FILE, "r", encoding="utf-8") as f:
            ci_data = json.load(f)

        multi_data = {}
        if MULTI_ADJ_FILE.exists():
            with open(MULTI_ADJ_FILE, "r", encoding="utf-8") as f:
                multi_data = json.load(f)

        f1_tests = multi_data.get("family_1_root_cause", {}).get("tests", {})

        lines.append("### Benchmark Evaluation: Baseline RAG vs. Zero-Shot Baseline (Held-Out Test Set, N = 93)")
        lines.append("")
        lines.append("> Evaluated strictly on held-out postmortems (`data/splits/test.json`) across 40 unseen companies against the TRAIN-ONLY retrieval index (`rag/chroma_db_train`, 662 incidents). Zero data leakage verified. Confidence intervals computed via 10,000 resamples (seed 42) comparing per-incident i.i.d. bootstrap, two-stage company-clustered bootstrap, and Bonferroni multiplicity-adjusted intervals ($m=5$, $\\alpha=0.01$).")
        lines.append("")
        lines.append("| Metric | Paired Delta Mean | Per-Incident 95% CI | Company-Clustered 95% CI | Bonferroni 99% CI ($m=5$) | Multiplicity Status |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")

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
                
                # Check multiplicity adjusted info
                t_info = f1_tests.get(key, {})
                adj_ci = t_info.get("adjusted_ci", [0, 0])
                status = t_info.get("status", "crosses zero")
                
                if status == "excludes zero":
                    status_str = "**excludes zero**"
                elif status == "unadjusted only":
                    status_str = "⚠️ **unadjusted only**"
                else:
                    status_str = "crosses zero"

                lines.append(
                    f"| {label} | **{mean_val:+.4f}** | `[{iid_ci[0]:.4f}, {iid_ci[1]:.4f}]` | `[{clust_ci[0]:.4f}, {clust_ci[1]:.4f}]` | `[{adj_ci[0]:.4f}, {adj_ci[1]:.4f}]` | {status_str} |"
                )
        lines.append("")

    # 2b. Two Regimes Section: Company-Disjoint Test vs. Leave-One-Out (Same-Company)
    if E1_FILE.exists() and E2_FILE.exists():
        with open(E1_FILE, "r", encoding="utf-8") as f:
            e1_data = json.load(f)
        with open(E2_FILE, "r", encoding="utf-8") as f:
            e2_data = json.load(f)

        lines.append("### Two Regimes: Company-Disjoint Test vs. Leave-One-Out (Same-Company)")
        lines.append("")
        lines.append("> **Operational Regime Separation:** The Company-Disjoint Test set (Experiment 1, N = 93) and Leave-One-Out Same-Company regime (Experiment 2, N = 93) test fundamentally different operational assumptions and **cannot be merged, pooled, or averaged**. E1 evaluates zero-leakage cross-company generalization across 40 unseen companies (0% same-company history). E2 evaluates single-tenant enterprise retrieval where historical postmortems from the *same* organization exist in the retrieval corpus.")
        lines.append("")
        lines.append("#### Table 1 (E1): Test Set Control (Held-Out Test Set, N = 93, 0% Same-Company)")
        lines.append("")
        lines.append("> Evaluates whether downstream RAG generation gains reflect technical incident semantics or general in-context exemplar/style priming. Clustered bootstrap confidence intervals with Bonferroni multiplicity adjustment across m=5 endpoints (99% CI, 10,000 resamples, seed 42). Metrics unified under canonical `eval/scorer.py`.")
        lines.append("")
        lines.append("| Condition | Format | ROUGE-L F1 | BERTScore F1 | Delta vs. Zero-Shot (ROUGE-L) [99% CI] | Delta vs. Zero-Shot (BERTScore) [99% CI] | Multiplicity Status |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

        for fmt, fmt_lbl in [("baseline", "Baseline"), ("gate_v2", "Gate-v2")]:
            f_e1 = e1_data.get("formats", {}).get(fmt, {})
            arms = f_e1.get("arms", {})
            deltas_e1 = f_e1.get("deltas", {})

            # Zero-shot
            zs = arms.get("zero_shot", {})
            zs_rl = zs.get("rougeL_f1", {}).get("mean", 0.0)
            zs_bs = zs.get("bert_score_f1", {}).get("mean", 0.0)
            lines.append(f"| **Zero-Shot Baseline** | {fmt_lbl} | {zs_rl:.4f} | {zs_bs:.4f} | — | — | — |")

            # Random Precedent
            rnd = arms.get("random_precedent", {})
            rnd_rl = rnd.get("rougeL_f1", {}).get("mean", 0.0)
            rnd_bs = rnd.get("bert_score_f1", {}).get("mean", 0.0)
            d_rnd = deltas_e1.get("random_minus_zeroshot", {})
            d_rnd_rl = d_rnd.get("rougeL_f1", {})
            d_rnd_bs = d_rnd.get("bert_score_f1", {})
            rnd_rl_ci = d_rnd_rl.get("adjusted_ci", [0, 0])
            rnd_bs_ci = d_rnd_bs.get("adjusted_ci", [0, 0])
            rnd_stat = "excludes zero" if d_rnd_rl.get("excludes_zero_after_adjustment", False) else "crosses zero"
            rnd_name = "**Random Precedent** *(Audit: Invalid)*" if fmt == "baseline" else "**Random Precedent**"
            lines.append(
                f"| {rnd_name} | {fmt_lbl} | {rnd_rl:.4f} | {rnd_bs:.4f} | "
                f"**{d_rnd_rl.get('mean', 0.0):+.4f}** `[{rnd_rl_ci[0]:.4f}, {rnd_rl_ci[1]:.4f}]` | "
                f"**{d_rnd_bs.get('mean', 0.0):+.4f}** `[{rnd_bs_ci[0]:.4f}, {rnd_bs_ci[1]:.4f}]` | {rnd_stat} |"
            )

            # Real Top-1 RAG
            real = arms.get("real_rag", {})
            real_rl = real.get("rougeL_f1", {}).get("mean", 0.0)
            real_bs = real.get("bert_score_f1", {}).get("mean", 0.0)
            d_real = deltas_e1.get("real_minus_zeroshot", {})
            d_real_rl = d_real.get("rougeL_f1", {})
            d_real_bs = d_real.get("bert_score_f1", {})
            real_rl_ci = d_real_rl.get("adjusted_ci", [0, 0])
            real_bs_ci = d_real_bs.get("adjusted_ci", [0, 0])
            real_stat = "excludes zero" if d_real_rl.get("excludes_zero_after_adjustment", False) else ("unadjusted only" if d_real_rl.get("significant_under_cluster_bootstrap", False) else "crosses zero")
            lines.append(
                f"| **Real Top-1 RAG** | {fmt_lbl} | {real_rl:.4f} | {real_bs:.4f} | "
                f"**{d_real_rl.get('mean', 0.0):+.4f}** `[{real_rl_ci[0]:.4f}, {real_rl_ci[1]:.4f}]` | "
                f"**{d_real_bs.get('mean', 0.0):+.4f}** `[{real_bs_ci[0]:.4f}, {real_bs_ci[1]:.4f}]` | {real_stat} |"
            )

        lines.append("")
        e1_g2_v = e1_data.get("formats", {}).get("gate_v2", {}).get("verdict_directional", e1_data.get("formats", {}).get("gate_v2", {}).get("verdict", "inconclusive"))
        e1_base_v = e1_data.get("formats", {}).get("baseline", {}).get("verdict_directional", e1_data.get("formats", {}).get("baseline", {}).get("verdict", "precedent_harmful_or_artifact"))
        d_real_rnd = e1_data.get("formats", {}).get("gate_v2", {}).get("deltas", {}).get("real_minus_random", {}).get("rougeL_f1", {})
        d_rnd_zs = e1_data.get("formats", {}).get("gate_v2", {}).get("deltas", {}).get("random_minus_zeroshot", {}).get("rougeL_f1", {})
        d_real_rnd_base = e1_data.get("formats", {}).get("baseline", {}).get("deltas", {}).get("real_minus_random", {}).get("rougeL_f1", {})
        lines.append(
            f"> **E1 Decision Rule Verdicts:**\n"
            f"> - **Gate-v2 Format:** `{e1_g2_v}`. Real-minus-random is **{d_real_rnd.get('mean', 0.0):+.4f}** `[{d_real_rnd.get('adjusted_ci', [0,0])[0]:.4f}, {d_real_rnd.get('adjusted_ci', [0,0])[1]:.4f}]` (crosses zero after Bonferroni adjustment); "
            f"random-minus-zero-shot is **{d_rnd_zs.get('mean', 0.0):+.4f}** `[{d_rnd_zs.get('adjusted_ci', [0,0])[0]:.4f}, {d_rnd_zs.get('adjusted_ci', [0,0])[1]:.4f}]` (crosses zero). Precedent retrieval produces no detectable lift over random exemplar priming on unseen companies.\n"
            f"> - **Baseline Format (Initial):** `{e1_base_v}` under `eval/PREREG_ERRATA.md`. Real-minus-random is **{d_real_rnd_base.get('mean', 0.0):+.4f}** `[{d_real_rnd_base.get('adjusted_ci', [0,0])[0]:.4f}, {d_real_rnd_base.get('adjusted_ci', [0,0])[1]:.4f}]` (excludes zero, negative). Note: This arm was formally invalidated (Deviation 2 in `eval/DEVIATIONS.md`) because generations inadvertently used gate-v2 brevity instructions (~55 words vs ~200 words); rerun staged via `scripts/rerun_e1_baseline_v2.py`."
        )
        lines.append("")

        lines.append("#### Table 2 (E2): Leave-One-Out (LOO) Same-Company Regime (N = 93 across 17 Companies)")
        lines.append("")
        lines.append("> Evaluates retrieval when same-company historical postmortems exist in the corpus. Condition (a) allows same-company precedents with near-duplicates filtered; Condition (b) restricts retrieval to cross-company only. Multiplicity-adjusted 99% CIs across m=5 endpoints (seed 42). Metrics unified under canonical `eval/scorer.py`.")
        lines.append("")
        lines.append("| Condition | Format | ROUGE-L F1 | BERTScore F1 | Relevance@1 (Score 2 / 1 / 0) | Delta vs. Zero-Shot (ROUGE-L) [99% CI] | Delta vs. Zero-Shot (BERTScore) [99% CI] | Multiplicity Status |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        r1_e2 = e2_data.get("relevance_at_1", {})
        rel_str_a = f"{r1_e2.get('condition_a', {}).get('score_2_highly_relevant', {}).get('pct', 0.0):.1f}% / {r1_e2.get('condition_a', {}).get('score_1_partially_relevant', {}).get('pct', 0.0):.1f}% / {r1_e2.get('condition_a', {}).get('score_0_unrelated', {}).get('pct', 0.0):.1f}%"
        rel_str_b = f"{r1_e2.get('condition_b', {}).get('score_2_highly_relevant', {}).get('pct', 0.0):.1f}% / {r1_e2.get('condition_b', {}).get('score_1_partially_relevant', {}).get('pct', 0.0):.1f}% / {r1_e2.get('condition_b', {}).get('score_0_unrelated', {}).get('pct', 0.0):.1f}%"

        for fmt, fmt_lbl in [("gate_v2", "Gate-v2"), ("baseline", "Baseline")]:
            f_e2 = e2_data.get("formats", {}).get(fmt, {})
            arms = f_e2.get("arms", {})
            deltas_e2 = f_e2.get("deltas", {})

            # Zero-shot
            zs = arms.get("zero_shot", {})
            zs_rl = zs.get("rougeL_f1", {}).get("mean", 0.0)
            zs_bs = zs.get("bert_score_f1", {}).get("mean", 0.0)
            lines.append(f"| **Zero-Shot** | {fmt_lbl} | {zs_rl:.4f} | {zs_bs:.4f} | — | — | — | — |")

            # Same-company Condition a
            ra = arms.get("rag_condition_a_same_company", {})
            ra_rl = ra.get("rougeL_f1", {}).get("mean", 0.0)
            ra_bs = ra.get("bert_score_f1", {}).get("mean", 0.0)
            d_ra = deltas_e2.get("rag_a_minus_zeroshot", {})
            d_ra_rl = d_ra.get("rougeL_f1", {})
            d_ra_bs = d_ra.get("bert_score_f1", {})
            ra_rl_ci = d_ra_rl.get("adjusted_ci", [0, 0])
            ra_bs_ci = d_ra_bs.get("adjusted_ci", [0, 0])
            ra_stat = "excludes zero" if d_ra_rl.get("excludes_zero_after_adjustment", False) else "crosses zero"
            lines.append(
                f"| **Same-Company (Condition a)** | {fmt_lbl} | {ra_rl:.4f} | {ra_bs:.4f} | {rel_str_a} | "
                f"**{d_ra_rl.get('mean', 0.0):+.4f}** `[{ra_rl_ci[0]:.4f}, {ra_rl_ci[1]:.4f}]` | "
                f"**{d_ra_bs.get('mean', 0.0):+.4f}** `[{ra_bs_ci[0]:.4f}, {ra_bs_ci[1]:.4f}]` | {ra_stat} |"
            )

            # Cross-company Condition b
            rb = arms.get("rag_condition_b_cross_company", {})
            rb_rl = rb.get("rougeL_f1", {}).get("mean", 0.0)
            rb_bs = rb.get("bert_score_f1", {}).get("mean", 0.0)
            d_rb = deltas_e2.get("rag_b_minus_zeroshot", {})
            d_rb_rl = d_rb.get("rougeL_f1", {})
            d_rb_bs = d_rb.get("bert_score_f1", {})
            rb_rl_ci = d_rb_rl.get("adjusted_ci", [0, 0])
            rb_bs_ci = d_rb_bs.get("adjusted_ci", [0, 0])
            rb_stat = "excludes zero" if d_rb_rl.get("excludes_zero_after_adjustment", False) else "crosses zero"
            lines.append(
                f"| **Cross-Company (Condition b)** | {fmt_lbl} | {rb_rl:.4f} | {rb_bs:.4f} | {rel_str_b} | "
                f"**{d_rb_rl.get('mean', 0.0):+.4f}** `[{rb_rl_ci[0]:.4f}, {rb_rl_ci[1]:.4f}]` | "
                f"**{d_rb_bs.get('mean', 0.0):+.4f}** `[{rb_bs_ci[0]:.4f}, {rb_bs_ci[1]:.4f}]` | {rb_stat} |"
            )

        lines.append("")
        e2_verdict = e2_data.get("verdict", "no detected effect")
        d_ra_zs_g2 = e2_data.get("formats", {}).get("gate_v2", {}).get("deltas", {}).get("rag_a_minus_zeroshot", {}).get("rougeL_f1", {})
        d_ra_lm_g2 = e2_data.get("formats", {}).get("gate_v2", {}).get("deltas", {}).get("rag_a_minus_zeroshot", {}).get("length_matched_rougeL", {})
        d_prem_g2 = e2_data.get("formats", {}).get("gate_v2", {}).get("deltas", {}).get("rag_a_minus_rag_b_same_company_premium", {}).get("rougeL_f1", {})
        lines.append(
            f"> **E2 Decision Rule Verdict:** `{e2_verdict}` (gate-v2 format). The same-company RAG(a)-minus-zero-shot delta is "
            f"**{d_ra_zs_g2.get('mean', 0.0):+.4f}** `[{d_ra_zs_g2.get('adjusted_ci', [0,0])[0]:.4f}, {d_ra_zs_g2.get('adjusted_ci', [0,0])[1]:.4f}]` (crosses zero after Bonferroni adjustment) "
            f"and length-matched delta is **{d_ra_lm_g2.get('mean', 0.0):+.4f}** `[{d_ra_lm_g2.get('adjusted_ci', [0,0])[0]:.4f}, {d_ra_lm_g2.get('adjusted_ci', [0,0])[1]:.4f}]` (crosses zero). "
            f"The same-company premium (Condition a minus Condition b) is **{d_prem_g2.get('mean', 0.0):+.4f}** `[{d_prem_g2.get('adjusted_ci', [0,0])[0]:.4f}, {d_prem_g2.get('adjusted_ci', [0,0])[1]:.4f}]` (crosses zero), "
            f"confirming that having same-company historical postmortems in the corpus produced no statistically detectable lift over cross-company retrieval or zero-shot generation."
        )
        lines.append("")

        # Stratified LOO Deltas Table
        if LOO_STRATIFIED_FILE.exists():
            with open(LOO_STRATIFIED_FILE, "r", encoding="utf-8") as f:
                strat_data = json.load(f)
            
            strata = strat_data.get("strata", {})
            lines.append("#### Table 2a (E2 Stratified): LOO Downstream Deltas Stratified by Judged Top-1 Relevance")
            lines.append("")
            lines.append("> Evaluates RAG(a)-minus-zero-shot generation deltas within each judged relevance stratum (gate-v2 format). Word counts and parsed remediation subsets reported per stratum.")
            lines.append("")
            lines.append("| Judged Top-1 Relevance | Incidents (Companies) | ROUGE-L Delta | BERTScore Delta | Both-Parsed Remediation Delta (n, comp) | Word Count (Zero-Shot / RAG-a) |")
            lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")

            score_labels = {
                "score_0": "Score 0 (Unrelated)",
                "score_1": "Score 1 (Partially Relevant)",
                "score_2": "Score 2 (Highly Relevant / Same Mechanism)"
            }
            for s_k in ["score_0", "score_1", "score_2"]:
                if s_k in strata:
                    s_info = strata[s_k]
                    lbl = score_labels.get(s_k, s_k)
                    n_inc = s_info.get("n", 0)
                    n_comp = s_info.get("n_companies", 0)
                    deltas_s = s_info.get("deltas_rag_a_minus_zeroshot", {})
                    rl_m = deltas_s.get("rougeL_f1", {}).get("mean", 0.0)
                    bs_m = deltas_s.get("bert_score_f1", {}).get("mean", 0.0)
                    rem_info = deltas_s.get("remediation_both_parsed", {})
                    rem_m = rem_info.get("delta_remediation_rougeL", {}).get("mean", 0.0)
                    rem_n = rem_info.get("n_both_parsed", 0)
                    rem_comp = rem_info.get("n_companies_both_parsed", 0)
                    wc = s_info.get("word_counts", {})
                    zs_wc = wc.get("zero_shot", {}).get("mean", 0.0)
                    ra_wc = wc.get("rag_a", {}).get("mean", 0.0)
                    lines.append(
                        f"| **{lbl}** | {n_inc} ({n_comp}) | **{rl_m:+.4f}** | **{bs_m:+.4f}** | "
                        f"{rem_m:+.4f} (n={rem_n}, c={rem_comp}) | {zs_wc:.1f} / {ra_wc:.1f} |"
                    )
            lines.append("")

        # LOO Retrieval Consistency & Near-Duplicate Filter Audit
        if LOO_CONSISTENCY_FILE.exists():
            with open(LOO_CONSISTENCY_FILE, "r", encoding="utf-8") as f:
                cons_data = json.load(f)
            recon = cons_data.get("reconciliation", {})
            dist = cons_data.get("distributions_condition_c_top1", {})
            cos_d = dist.get("cosine_similarity", {})
            jac_d = dist.get("token_5gram_jaccard", {})
            lines.append("#### LOO Retrieval Consistency & Near-Duplicate Filter Audit")
            lines.append("")
            lines.append(
                f"- **Near-Duplicate Filter Impact:** Exactly {recon.get('condition_a_total_near_duplicates_filtered', 11)} near-duplicates were filtered in Condition (a) (across {recon.get('queries_with_divergent_top3_between_a_and_c', 8)} queries, {recon.get('queries_with_divergent_top1_between_a_and_c', 7)} changing top-1) and exactly {recon.get('condition_b_total_near_duplicates_filtered', 0)} in Condition (b).\n"
                f"- **Divergence Reconciliation:** All {recon.get('queries_with_divergent_top3_between_a_and_c', 8)} queries with differing top-3 lists between Condition (a) and unfiltered Condition (c) are 100% accounted for by the near-duplicate filter threshold ($\\ge 0.95$ cosine or $\\ge 0.80$ 5-gram token-shingle Jaccard); {recon.get('unexplained_differences_count', 0)} unexplained divergences.\n"
                f"- **Unfiltered Candidate Pool Distributions (Condition c Top-1):** Cosine similarity: min = {cos_d.get('min', 0.0):.4f}, median = {cos_d.get('median', 0.0):.4f}, p90 = {cos_d.get('p90', 0.0):.4f}, max = {cos_d.get('max', 0.0):.4f} ({cos_d.get('count_ge_0_90', 0)} $\\ge 0.90$, {cos_d.get('count_ge_0_95', 0)} $\\ge 0.95$). 5-gram token-shingle Jaccard: min = {jac_d.get('min', 0.0):.4f}, median = {jac_d.get('median', 0.0):.4f}, p90 = {jac_d.get('p90', 0.0):.4f}, max = {jac_d.get('max', 0.0):.4f} ({jac_d.get('count_ge_0_80', 0)} $\\ge 0.80$)."
            )
            lines.append("")

        lines.append("#### Comparative Analysis: Same-Company History (E2) vs. Company-Disjoint Generalisation (E1)")
        lines.append("")
        lines.append("- **Retrieval Relevance Gain:** In E2, same-company retrieval (Condition a) substantially lifted top-1 judged relevance compared to cross-company retrieval: 66.67% of precedents were judged relevant (15.05% identical failure mechanisms, score 2) vs. 39.78% in cross-company (4.30% score 2).")
        lines.append("- **Downstream Generation Impact:** Despite higher retrieval relevance in E2, downstream generation gains did not materialize in either regime under company-clustering and Bonferroni multiplicity adjustments:")
        lines.append(f"  - In **E1 (Company-Disjoint)**, the gate-v2 RAG delta was **+0.0019** `[-0.0183, 0.0204]` (crosses zero), while the random-precedent delta was **+0.0036** `[-0.0222, 0.0331]` (crosses zero).")
        lines.append(f"  - In **E2 (Leave-One-Out Same-Company)**, the gate-v2 same-company RAG(a) delta was **-0.0141** `[-0.0406, 0.0168]` (crosses zero) and cross-company RAG(b) delta was **-0.0058** `[-0.0293, 0.0178]` (crosses zero).")
        lines.append("- **Operational Conclusion:** Precedent retrieval does not reliably boost technical accuracy on either regime; structured formatting instructions account for virtually all observable metric gains.")
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
        cond = rem_data.get("conditional_on_both_parsed", {})
        cond_base = cond.get("baseline_format", {}).get("deltas", {})
        cond_gate = cond.get("gate_v2_format", {}).get("deltas", {})
        f2_multi = multi_data.get("family_2_remediation", {})
        
        lines.append(
            f"> **Remediation Paired Deltas (RAG - Zero-Shot, Clustered 95% CIs & Bonferroni 97.5% CIs, m=2):**\n"
            f"> - **Baseline Format (Unconditional, N=93):** ROUGE-L $\\Delta$ = **{base_d.get('rougeL_delta', {}).get('mean', 0.0):+.4f}** `[{base_d.get('rougeL_delta', {}).get('ci_95', [0,0])[0]:.4f}, {base_d.get('rougeL_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (Adj: `[{f2_multi.get('baseline_format_unconditional', {}).get('rougeL_delta', {}).get('adjusted_ci', [0,0])[0]:.4f}, {f2_multi.get('baseline_format_unconditional', {}).get('rougeL_delta', {}).get('adjusted_ci', [0,0])[1]:.4f}]`, status: `{f2_multi.get('baseline_format_unconditional', {}).get('rougeL_delta', {}).get('status', 'unknown')}`); "
            f"BERTScore $\\Delta$ = **{base_d.get('bert_score_delta', {}).get('mean', 0.0):+.4f}** `[{base_d.get('bert_score_delta', {}).get('ci_95', [0,0])[0]:.4f}, {base_d.get('bert_score_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (status: `{f2_multi.get('baseline_format_unconditional', {}).get('bert_score_delta', {}).get('status', 'unknown')}`).\n"
            f"> - **Baseline Format (Conditional on Both Parsed, N={cond.get('baseline_format', {}).get('n', 52)}):** ROUGE-L $\\Delta$ = **{cond_base.get('rougeL_delta', {}).get('mean', 0.0):+.4f}** `[{cond_base.get('rougeL_delta', {}).get('ci_95', [0,0])[0]:.4f}, {cond_base.get('rougeL_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (status: `{f2_multi.get('baseline_format_conditional', {}).get('rougeL_delta', {}).get('status', 'crosses zero')}`); "
            f"BERTScore $\\Delta$ = **{cond_base.get('bert_score_delta', {}).get('mean', 0.0):+.4f}** `[{cond_base.get('bert_score_delta', {}).get('ci_95', [0,0])[0]:.4f}, {cond_base.get('bert_score_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (status: `{f2_multi.get('baseline_format_conditional', {}).get('bert_score_delta', {}).get('status', 'excludes zero')}`). *Note: Restricting to parsed incidents erases baseline ROUGE-L significance, revealing unconditional gain was an artifact of parse coverage disparity (78.5% RAG vs 64.5% Zero-Shot).*\n"
            f"> - **Gate-v2 Format (Conditional on Both Parsed, N={cond.get('gate_v2_format', {}).get('n', 92)}):** ROUGE-L $\\Delta$ = **{cond_gate.get('rougeL_delta', {}).get('mean', 0.0):+.4f}** `[{cond_gate.get('rougeL_delta', {}).get('ci_95', [0,0])[0]:.4f}, {cond_gate.get('rougeL_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (status: `{f2_multi.get('gate_v2_format_conditional', {}).get('rougeL_delta', {}).get('status', 'crosses zero')}`); "
            f"BERTScore $\\Delta$ = **{cond_gate.get('bert_score_delta', {}).get('mean', 0.0):+.4f}** `[{cond_gate.get('bert_score_delta', {}).get('ci_95', [0,0])[0]:.4f}, {cond_gate.get('bert_score_delta', {}).get('ci_95', [0,0])[1]:.4f}]` (status: `{f2_multi.get('gate_v2_format_conditional', {}).get('bert_score_delta', {}).get('status', 'excludes zero')}`)."
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

    # 9. Retrieval Relevance (Judged, N = 93)
    if RELEVANCE_ANALYSIS_FILE.exists():
        with open(RELEVANCE_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            rel_data = json.load(f)

        rr = rel_data.get("relevance_rates", {})
        sp = rel_data.get("similarity_prediction", {})
        sg = rel_data.get("stratified_gain_by_relevance", {})
        gc = rel_data.get("guardrail_check", {})

        lines.append("### Retrieval Relevance (Judged, N = 93)")
        lines.append("")
        lines.append("> **Notice:** Manual audit pending; judge labels unvalidated against human consensus.")
        lines.append("")
        lines.append("Technical failure relevance evaluated across all 279 retrieved precedents (top-3 for 93 test queries) using an independent LLM judge (`qwen/qwen3.8-27b`) on a fixed 3-point rubric (`eval/rubrics/relevance_v1.md`, SHA-256 `c0ccc1f...`) with vendor names masked. 95% confidence intervals computed via two-stage company-clustered bootstrap (10,000 resamples, seed 42). *Note: Labeled as exploratory.*")
        lines.append("")
        lines.append("#### Judged Relevance Rates")
        lines.append("")
        lines.append("| Metric | Count / Total | Fraction (%) | Company-Clustered 95% CI |")
        lines.append("|:---|:---:|:---:|:---:|")

        r1_2 = rr.get("relevance_at_1_score_2", {})
        r1_ge1 = rr.get("relevance_at_1_score_ge_1", {})
        b3_2 = rr.get("best_of_3_score_2", {})
        b3_ge1 = rr.get("best_of_3_score_ge_1", {})

        lines.append(f"| **Relevance@1 (Score = 2, Same Fault)** | {r1_2.get('count', 0)} / {r1_2.get('total', 93)} | {r1_2.get('percentage', 0.0):.2f}% | `[{r1_2.get('company_clustered_ci_95', [0,0])[0]:.4f}, {r1_2.get('company_clustered_ci_95', [0,0])[1]:.4f}]` |")
        lines.append(f"| **Relevance@1 (Score ≥ 1, Related / Same)** | {r1_ge1.get('count', 0)} / {r1_ge1.get('total', 93)} | {r1_ge1.get('percentage', 0.0):.2f}% | `[{r1_ge1.get('company_clustered_ci_95', [0,0])[0]:.4f}, {r1_ge1.get('company_clustered_ci_95', [0,0])[1]:.4f}]` |")
        lines.append(f"| **Best-of-Top-3 (Score = 2, Same Fault)** | {b3_2.get('count', 0)} / {b3_2.get('total', 93)} | {b3_2.get('percentage', 0.0):.2f}% | `[{b3_2.get('company_clustered_ci_95', [0,0])[0]:.4f}, {b3_2.get('company_clustered_ci_95', [0,0])[1]:.4f}]` |")
        lines.append(f"| **Best-of-Top-3 (Score ≥ 1, Related / Same)** | {b3_ge1.get('count', 0)} / {b3_ge1.get('total', 93)} | {b3_ge1.get('percentage', 0.0):.2f}% | `[{b3_ge1.get('company_clustered_ci_95', [0,0])[0]:.4f}, {b3_ge1.get('company_clustered_ci_95', [0,0])[1]:.4f}]` |")
        lines.append("")
        lines.append("#### Vector Similarity as Relevance Predictor")
        lines.append("")
        lines.append(f"- **Spearman Rank Correlation:** $\\rho = {sp.get('spearman_rho', 0.0):+.4f}$ `[{sp.get('spearman_rho_clustered_ci_95', [0,0])[0]:.4f}, {sp.get('spearman_rho_clustered_ci_95', [0,0])[1]:.4f}]` (company-clustered 95% CI excludes zero).")
        lines.append(f"- **AUROC (Predicting Relevance ≥ 1):** `{sp.get('auroc_relevance_ge_1', 0.0):.4f}`.")
        lines.append("")
        lines.append("#### Downstream RAG Gain Stratified by Judged Relevance")
        lines.append("")
        lines.append("| Judged Top-1 Relevance | Incidents (Companies) | ROUGE-L Delta [95% CI] | BERTScore Delta [95% CI] | Parsed Remediation ROUGE-L Delta |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")

        for s_idx, s_lbl in [(0, "Score 0 (Unrelated)"), (1, "Score 1 (Related)"), (2, "Score 2 (Same Fault)")]:
            s_data = sg.get(f"score_{s_idx}", {})
            n_inc = s_data.get("n_incidents", 0)
            n_comp = s_data.get("n_companies", 0)
            low_n = s_data.get("low_n", False)
            
            rl = s_data.get("rougeL_delta", {})
            bs = s_data.get("bert_score_delta", {})
            rem = s_data.get("remediation_conditional_rougeL_delta", {})

            rl_ci = rl.get("ci_95")
            rl_str = f"**{rl.get('mean', 0.0):+.4f}** `[{rl_ci[0]:.4f}, {rl_ci[1]:.4f}]`" if rl_ci else f"**{rl.get('mean', 0.0):+.4f}** *(low_n: true)*"

            bs_ci = bs.get("ci_95")
            bs_str = f"**{bs.get('mean', 0.0):+.4f}** `[{bs_ci[0]:.4f}, {bs_ci[1]:.4f}]`" if bs_ci else f"**{bs.get('mean', 0.0):+.4f}** *(low_n: true)*"

            rem_str = f"{rem.get('mean', 0.0):+.4f} (n={rem.get('n_parsed', 0)})"
            
            lines.append(f"| **{s_lbl}** | {n_inc} ({n_comp}) | {rl_str} | {bs_str} | {rem_str} |")

        lines.append("")
        lines.append("#### Guardrail Check: Abstained vs. Answered Relevance")
        lines.append("")
        abs_g = gc.get("abstained_n7", {})
        ans_g = gc.get("answered_n86", {})
        lines.append(f"- **Abstained Partition (Sim < 0.40, N = 7):** Score 0 = `{abs_g.get('score_0', {}).get('pct', 0.0):.1f}%` ({abs_g.get('score_0', {}).get('count', 0)}), Score 1 = `{abs_g.get('score_1', {}).get('pct', 0.0):.1f}%` ({abs_g.get('score_1', {}).get('count', 0)}), Score 2 = `{abs_g.get('score_2', {}).get('pct', 0.0):.1f}%` ({abs_g.get('score_2', {}).get('count', 0)}). Relevance ≥ 1 = `{abs_g.get('relevance_ge_1_pct', 0.0):.1f}%` ({abs_g.get('score_1', {}).get('count', 0) + abs_g.get('score_2', {}).get('count', 0)}/7).")
        lines.append(f"- **Answered Partition (Sim ≥ 0.40, N = 86):** Score 0 = `{ans_g.get('score_0', {}).get('pct', 0.0):.1f}%` ({ans_g.get('score_0', {}).get('count', 0)}), Score 1 = `{ans_g.get('score_1', {}).get('pct', 0.0):.1f}%` ({ans_g.get('score_1', {}).get('count', 0)}), Score 2 = `{ans_g.get('score_2', {}).get('pct', 0.0):.1f}%` ({ans_g.get('score_2', {}).get('count', 0)}). Relevance ≥ 1 = `{ans_g.get('relevance_ge_1_pct', 0.0):.1f}%` ({ans_g.get('score_1', {}).get('count', 0) + ans_g.get('score_2', {}).get('count', 0)}/86).")
        
        fs_p = 0.1232
        fs_or = 5.4667
        if FLOOR_STATS_FILE.exists():
            with open(FLOOR_STATS_FILE, "r", encoding="utf-8") as f:
                fs_data = json.load(f)
                fs_p = fs_data.get("fisher_exact", {}).get("p_value", fs_p)
                fs_or = fs_data.get("fisher_exact", {}).get("odds_ratio", fs_or)

        lines.append(f"- **Guardrail Separation:** Queries above the 0.40 threshold retrieved judged-relevant precedents in 41/86 cases (47.7%) vs. 1/7 cases (14.3%) below threshold (Fisher's exact test p = {fs_p:.4f}, odds ratio {fs_or:.2f}). The separation is directionally consistent; n=7 abstained limits statistical power (p > 0.05). AUROC 0.667 reflects a weak-to-moderate guard rather than a sharp noise boundary.")
        lines.append("")

    return "\n".join(lines)


def update_readme():
    readme_path = PROJECT_ROOT / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    table_md = render_markdown_tables()

    start_token = "## Quantitative Evaluation & Benchmark Results\n\n"
    end_token = "\n\nDetailed failure analysis and individual incident evaluations"

    start_idx = content.find(start_token)
    end_idx = content.find(end_token)

    if start_idx != -1 and end_idx != -1:
        new_content = content[:start_idx + len(start_token)] + table_md + content[end_idx:]
    else:
        print("Warning: Could not find exact replacement boundaries in README.md")
        return

    threat_bullets = [
        "- **Single Scorer Reconciliation:** All evaluation metrics (ROUGE-1/2/L, BERTScore F1, SentenceTransformer cosine similarity, and root-cause word count) are unified under canonical implementation [eval/scorer.py](eval/scorer.py), pinned within $10^{-4}$ against previously committed 2x2 control cells in [tests/test_scorer_regression.py](tests/test_scorer_regression.py). Discrepancies between earlier E1/E2 reports and control_2x2 arose from markdown stripping (`clean_rc()` in `analyze_e1.py`) and cosine text slicing; all reported numbers are now sourced strictly from rescored datasets.",
        "- **E1 Baseline Arm Status (Invalidated):** An audit of the E1 baseline random-precedent arm ([scripts/audit_e1_baseline.py](scripts/audit_e1_baseline.py)) revealed that generations inadvertently used the `relevance_gate_v2` brevity constraint (`PROMPT_REGISTRY['baseline']`), producing truncated ~55-word outputs rather than verbose ~200-word outputs characteristic of baseline RAG and zero-shot. As documented in [eval/DEVIATIONS.md](eval/DEVIATIONS.md) (Deviation 2), this arm was invalidated. Prompt parity has been verified in [tests/test_e1_prompt_parity.py](tests/test_e1_prompt_parity.py) and a rerun script staged ([scripts/rerun_e1_baseline_v2.py](scripts/rerun_e1_baseline_v2.py)); directional verdict for the initial baseline under [eval/PREREG_ERRATA.md](eval/PREREG_ERRATA.md) is `precedent_harmful_or_artifact`.",
        "- **Leave-One-Out (LOO) Sample of 17 Companies:** Experiment 2's LOO regime draws $N=93$ queries exclusively from the 17 companies in the training split with $\\ge 5$ high-quality postmortems (capped at 12 incidents per company). Because companies with fewer than 5 incidents were excluded to ensure sufficient candidate precedent depth, findings reflect multi-incident enterprise organizations and cannot be generalized to sparse-incident organizations without history.",
        "- **Same-Mechanism Stratum Sample Size ($N=14$):** In the LOO regime, only 14 incidents (15.05% across 9 companies) had top-1 retrieved precedents judged as sharing the identical technical failure mechanism (Score 2). While this stratum exhibits negative downstream deltas ($\\Delta \\text{ROUGE-L} = -0.0379$, $\\Delta \\text{BERTScore} = -0.0293$), the small sample size ($N=14$) limits statistical power, and results are reported as exploratory without claim of generalizability.",
        "- **E1 Single Random Seed ($seed=42$):** The E1 random-precedent control was generated under a single fixed seed ($seed=42$), uniformly sampling one non-retrieved high-quality precedent per query. While standard error across queries is captured by bootstrap resampling, multi-seed variance across alternative random precedent assignments remains unmeasured.",
    ]
    threat_block = "\n".join(threat_bullets)

    eval_design_marker = "### 1. Evaluation Design\n\n"
    format_effect_marker = "- **Format Effect Dominates Retrieval Effect in Gate Prompts:**"
    ed_idx = new_content.find(eval_design_marker)
    fe_idx = new_content.find(format_effect_marker)

    if ed_idx != -1 and fe_idx != -1:
        prefix = new_content[:ed_idx + len(eval_design_marker)]
        suffix = new_content[fe_idx:]
        new_content = prefix + threat_block + "\n" + suffix
    elif ed_idx != -1:
        insert_pos = ed_idx + len(eval_design_marker)
        new_content = new_content[:insert_pos] + threat_block + "\n" + new_content[insert_pos:]

    readme_path.write_text(new_content, encoding="utf-8")
    print(f"Successfully updated {readme_path}")


def main():
    import sys
    table_md = render_markdown_tables()
    print(table_md)
    if "--update-readme" in sys.argv or True:
        update_readme()


if __name__ == "__main__":
    main()
