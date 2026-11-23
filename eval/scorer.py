"""
TraceMind Unified Evaluation Scorer Module
------------------------------------------
Single canonical scoring implementation for control_2x2, E1, and E2:
- ROUGE-1/2/L (rouge-score with stemmer=True)
- BERTScore (bert-base-uncased, F1)
- Cosine similarity (SentenceTransformer embed, normalized dot product on first 1000 chars)
- Word count: defined as len(text.split()) on the whole generated root-cause section
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as calc_bertscore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clustering.cluster import embed

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def word_count(text: str) -> int:
    """Canonical word count function: whitespace split on whole section."""
    if not text:
        return 0
    return len(text.split())


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        s = _ROUGE_SCORER.score(ref or "none", pred or "none")
        r1.append(float(s["rouge1"].fmeasure))
        r2.append(float(s["rouge2"].fmeasure))
        rl.append(float(s["rougeL"].fmeasure))
    return {
        "rouge1_f1": r1,
        "rouge2_f1": r2,
        "rougeL_f1": rl,
    }


def compute_bert_scores(predictions: List[str], references: List[str]) -> List[float]:
    """Compute BERTScore F1 with fixed model, idf, and batching."""
    safe_preds = [p if p.strip() else "none" for p in predictions]
    safe_refs = [r if r.strip() else "none" for r in references]
    _, _, bs = calc_bertscore(safe_preds, safe_refs, model_type="bert-base-uncased", verbose=False)
    return bs.numpy().tolist()


def compute_cosine_scores(predictions: List[str], references: List[str]) -> List[float]:
    """Compute SentenceTransformer cosine similarity on first 1000 characters."""
    cos_scores = []
    for pred, ref in zip(predictions, references):
        p_text = pred[:1000] if pred.strip() else "none"
        r_text = ref[:1000] if ref.strip() else "none"
        v_p = np.array(embed(p_text))
        v_r = np.array(embed(r_text))
        cos_scores.append(float(v_p @ v_r))
    return cos_scores


def score_cell(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """
    Score a single cell of predictions against ground truth references.
    Returns per-incident metric arrays and summary statistics.
    """
    n = len(predictions)
    if n != len(references):
        raise ValueError(f"Length mismatch: {n} predictions vs {len(references)} references.")

    # 1. ROUGE
    rouge_res = compute_rouge_scores(predictions, references)

    # 2. BERTScore
    bs_list = compute_bert_scores(predictions, references)

    # 3. Cosine
    cos_list = compute_cosine_scores(predictions, references)

    # 4. Word count
    wc_list = [word_count(p) for p in predictions]

    return {
        "per_incident": {
            "rouge1_f1": rouge_res["rouge1_f1"],
            "rouge2_f1": rouge_res["rouge2_f1"],
            "rougeL_f1": rouge_res["rougeL_f1"],
            "bert_score_f1": bs_list,
            "cosine": cos_list,
            "word_count": wc_list,
        },
        "summary": {
            "rouge1_f1": round(float(np.mean(rouge_res["rouge1_f1"])), 4),
            "rouge2_f1": round(float(np.mean(rouge_res["rouge2_f1"])), 4),
            "rougeL_f1": round(float(np.mean(rouge_res["rougeL_f1"])), 4),
            "bert_score_f1": round(float(np.mean(bs_list)), 4),
            "cosine": round(float(np.mean(cos_list)), 4),
            "word_count": {
                "mean": round(float(np.mean(wc_list)), 1),
                "median": round(float(np.median(wc_list)), 1),
                "std": round(float(np.std(wc_list)), 1),
            },
        },
    }
