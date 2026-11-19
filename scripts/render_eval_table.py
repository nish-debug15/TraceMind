"""
Render TraceMind Evaluation Tables for README and Reports
---------------------------------------------------------
Reads evaluated JSON results from eval/results/ and prints dynamically
formatted Markdown tables representing:
1. RAG vs Zero-Shot Main Benchmark (N=93) with Bootstrap 95% Confidence Intervals
2. Quality Subsets (High N=61 vs Low N=32)
3. Retrieval Precedent Distribution & Abstention Analysis
4. Prompt Robustness Ablation (baseline, relevance_gate, relevance_gate_v2)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
RAG_VS_ZS_FILE = RESULTS_DIR / "rag_vs_zero_shot.json"
METRICS_CI_FILE = RESULTS_DIR / "metrics_with_ci.json"
PROMPT_ABLATION_FILE = RESULTS_DIR / "prompt_ablation.json"


def render_markdown_tables() -> str:
    lines = []

    # 1. Main RAG vs Zero-Shot Benchmark Table
    if METRICS_CI_FILE.exists() and RAG_VS_ZS_FILE.exists():
        with open(METRICS_CI_FILE, "r", encoding="utf-8") as f:
            ci_data = json.load(f)
        with open(RAG_VS_ZS_FILE, "r", encoding="utf-8") as f:
            main_data = json.load(f)

        n = ci_data["evaluation_config"]["num_samples"]
        metrics = ci_data["metrics"]

        lines.append(f"### Benchmark Evaluation: RAG vs. Zero-Shot Baseline (Held-Out Test Set, N = {n})")
        lines.append("")
        lines.append("> Evaluated strictly on held-out postmortems (`data/splits/test.json`) across 40 unseen companies against the TRAIN-ONLY retrieval index (`rag/chroma_db_train`, 662 incidents). Zero data leakage verified. 95% confidence intervals computed via 10,000 paired bootstrap resamples (seed 42).")
        lines.append("")
        lines.append("| Metric | Zero-Shot Baseline [95% CI] | TraceMind RAG [95% CI] | Paired Delta [95% CI] | Statistically Significant |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")

        labels = {
            "rouge1_f1": "ROUGE-1 F1",
            "rouge2_f1": "ROUGE-2 F1",
            "rougeL_f1": "ROUGE-L F1",
            "bert_score_f1": "BERTScore F1",
            "semantic_sim": "Embedding Cosine Sim",
        }

        for k in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bert_score_f1", "semantic_sim"]:
            if k not in metrics:
                continue
            m = metrics[k]
            label = labels.get(k, k)
            zs_val = f"{m['zeroshot']['mean']:.4f} `[{m['zeroshot']['ci_95'][0]:.4f}, {m['zeroshot']['ci_95'][1]:.4f}]`"
            rag_val = f"{m['rag']['mean']:.4f} `[{m['rag']['ci_95'][0]:.4f}, {m['rag']['ci_95'][1]:.4f}]`"
            d_mean = m['paired_delta']['mean']
            d_sign = "+" if d_mean >= 0 else ""
            delta_val = f"**{d_sign}{d_mean:.4f}** `[{m['paired_delta']['ci_95'][0]:.4f}, {m['paired_delta']['ci_95'][1]:.4f}]`"
            sig = "**Yes** (CI > 0)" if m['paired_delta']['strictly_positive'] else "No"
            lines.append(f"| **{label}** | {zs_val} | {rag_val} | {delta_val} | {sig} |")

        lines.append("")

        # 2. Quality Subsets Table
        if "aggregates" in main_data:
            lines.append("#### Performance Breakdown by Report Quality")
            lines.append("")
            lines.append("| Quality Subset | N | Zero-Shot ROUGE-L | RAG ROUGE-L | Delta ROUGE-L | Zero-Shot Cosine | RAG Cosine | Delta Cosine |")
            lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

            for subset_name in ["Full Set", "Quality: High Only", "Quality: Low Only"]:
                if subset_name not in main_data["aggregates"]:
                    continue
                s = main_data["aggregates"][subset_name]
                sn = s["sample_size"]
                zs_rl = s["ZeroShot"]["rougeL_f1"]["mean"]
                rag_rl = s["RAG"]["rougeL_f1"]["mean"]
                d_rl = rag_rl - zs_rl
                d_rl_str = f"+{d_rl:.4f}" if d_rl >= 0 else f"{d_rl:.4f}"

                zs_sim = s["ZeroShot"]["semantic_sim"]["mean"]
                rag_sim = s["RAG"]["semantic_sim"]["mean"]
                d_sim = rag_sim - zs_sim
                d_sim_str = f"+{d_sim:.4f}" if d_sim >= 0 else f"{d_sim:.4f}"

                lines.append(f"| {subset_name} | {sn} | {zs_rl:.4f} | {rag_rl:.4f} | **{d_rl_str}** | {zs_sim:.4f} | {rag_sim:.4f} | {d_sim_str} |")

            lines.append("")

        # 3. Retrieval Precedent Distribution & Abstention
        if "abstention" in ci_data and "retrieval_analysis" in main_data:
            ab = ci_data["abstention"]
            ra = main_data["retrieval_analysis"]
            lines.append("#### Retrieval Distribution & Guardrail Abstention")
            lines.append("")
            lines.append(f"- **Mean Top-1 Cosine Similarity**: `{ra['mean_similarity']:.4f}` (Median: `{ra['median_similarity']:.4f}`)")
            lines.append(f"- **Similarity Floor (< 0.40) Abstentions**: `{ab['abstained_count']} / {ab['total_count']}` ({ab['abstention_rate']:.1%})")
            lines.append(f"- **Guardrail Action**: For incidents scoring below 0.40 cosine similarity, the relevance guardrail (`relevance_gate_v2`) abstains from precedent transfer and forces autonomous RCA synthesis from incoming telemetry alone.")
            lines.append("")

    # 4. Prompt Ablation Table
    if PROMPT_ABLATION_FILE.exists():
        with open(PROMPT_ABLATION_FILE, "r", encoding="utf-8") as f:
            p_data = json.load(f)

        lines.append("#### Prompt Robustness Ablation (Test Set, N = 93)")
        lines.append("")
        lines.append("| Prompt Variant | ROUGE-L F1 | BERTScore F1 | Cosine Similarity | Guardrail Enforced |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")

        variants = p_data.get("variants", {})
        for name, stats in variants.items():
            rl = stats.get("rougeL_f1", 0.0)
            bs = stats.get("bert_score_f1", 0.0)
            cos = stats.get("cosine_sim", 0.0)
            guard = "Yes (`Precedent: [relevant|not relevant]`)" if "v2" in name else ("Soft check" if "gate" in name else "None")
            lines.append(f"| `{name}` | {rl:.4f} | {bs:.4f} | {cos:.4f} | {guard} |")

        if "spread" in p_data:
            sp = p_data["spread"]
            lines.append("")
            lines.append(f"> **Prompt Variance / Spread across variants**: ROUGE-L range: `{sp.get('rougeL_f1_spread', 0.0):.4f}`, Cosine range: `{sp.get('cosine_sim_spread', 0.0):.4f}`.")
        lines.append("")

    return "\n".join(lines)


def main():
    table_md = render_markdown_tables()
    print(table_md)


if __name__ == "__main__":
    main()
