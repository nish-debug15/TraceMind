"""
TraceMind Failure Analysis Script
---------------------------------
Finds the 10 test incidents with lowest RAG ROUGE-L in eval/results/rag_vs_zero_shot.json.
Extracts:
- incident_id
- company
- top-1 retrieval similarity score
- retrieved incident_id and retrieved company (looked up from data/splits/train.json)
- ground truth root cause (first 120 chars)
- generated root cause (first 120 chars)
- failure category / diagnosis

Saves results to eval/results/failure_analysis.json.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_VS_ZS_FILE = PROJECT_ROOT / "eval" / "results" / "rag_vs_zero_shot.json"
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.json"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "failure_analysis.json"


def analyze_failures(n_bottom: int = 10):
    if not RAG_VS_ZS_FILE.exists():
        raise FileNotFoundError(f"Missing {RAG_VS_ZS_FILE}")

    with open(RAG_VS_ZS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    gt_map = {r["incident_id"]: r for r in test_records}

    train_map = {}
    if TRAIN_FILE.exists():
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            train_records = json.load(f)
            train_map = {r["incident_id"]: r for r in train_records}

    per_incident = data["per_incident_scores"]

    # Sort ascending by RAG ROUGE-L F1
    sorted_by_rl = sorted(per_incident, key=lambda x: x["rag"]["rougeL_f1"])
    bottom_incidents = sorted_by_rl[:n_bottom]

    failure_records = []

    print(f"\n{'#' * 85}")
    print(f"BOTTOM {n_bottom} INCIDENTS BY RAG ROUGE-L")
    print(f"{'#' * 85}")

    for idx, inc in enumerate(bottom_incidents, 1):
        inc_id = inc["incident_id"]
        test_meta = gt_map.get(inc_id, {})
        company = inc.get("company") or test_meta.get("company", "Unknown")

        retrieval = inc.get("retrieval", {})
        sim = round(float(retrieval.get("similarity_score", 0.0)), 4)
        matched_id = retrieval.get("matched_incident_id") or retrieval.get("incident_id", "")
        train_incident = train_map.get(matched_id, {})
        ret_company = train_incident.get("company") or retrieval.get("metadata", {}).get("company", "Unknown")

        gt_rc = test_meta.get("root_cause", "").strip()
        gen_rc = inc["rag"]["root_cause"].strip()
        rl_score = inc["rag"]["rougeL_f1"]
        log_text = test_meta.get("raw_log_excerpt", "")
        log_len = len(log_text)

        # Failure mode categorization
        # 1. Similarity floor breach (< 0.40): Low retrieval confidence
        # 2. Telemetry underspecified / Status-page brevity: Log excerpt < 150 chars, lacking stack trace or error code
        # 3. Status-page generic summary: Ground truth is non-technical status update (e.g. "We believe the 503s have been resolved")
        # 4. Cross-company domain gap: Different companies where architectural vocabulary diverted LLM
        # 5. Spurious correlation / Red herring: Precedent matched on incidental phrasing
        is_generic_status = any(phrase in gt_rc.lower() for phrase in [
            "resolved", "recovered", "monitoring all systems", "working as intended", "mitigating the issue", "fix has been deployed"
        ])

        if sim < 0.40:
            category = "Low Retrieval Confidence (Floor Breach < 0.40)"
        elif is_generic_status and len(gt_rc) < 120:
            category = "Status-Page Non-Technical Ground Truth (Boilerplate Reference)"
        elif log_len < 150:
            category = "Telemetry Underspecified (Log Excerpt < 150 chars)"
        elif company != ret_company:
            category = "Cross-Company Domain Gap (Divergent Architecture)"
        else:
            category = "Spurious Precedent Correlation"

        record = {
            "rank": idx,
            "incident_id": inc_id,
            "company": company,
            "top1_similarity": sim,
            "retrieved_incident_id": matched_id,
            "retrieved_company": ret_company,
            "rougeL_f1": rl_score,
            "failure_category": category,
            "ground_truth_root_cause_120": gt_rc[:120],
            "generated_root_cause_120": gen_rc[:120],
            "full_ground_truth_root_cause": gt_rc,
            "full_generated_root_cause": gen_rc,
            "raw_log_excerpt_120": log_text[:120]
        }
        failure_records.append(record)

        print(f"\n[{idx}] Incident: {inc_id} | Company: {company} | ROUGE-L: {rl_score:.4f}")
        print(f"    Retrieval: Sim={sim:.4f} from {ret_company} ({matched_id})")
        print(f"    Category: {category}")
        print(f"    GT Root Cause (120c):  {gt_rc[:120]}...")
        print(f"    Gen Root Cause (120c): {gen_rc[:120]}...")

    # Tally failure modes
    categories = {}
    for r in failure_records:
        cat = r["failure_category"]
        categories[cat] = categories.get(cat, 0) + 1

    output_payload = {
        "analysis_config": {
            "num_failures_analyzed": n_bottom,
            "source_benchmark": "eval/results/rag_vs_zero_shot.json",
            "test_dataset": "data/splits/test.json",
            "ranking_metric": "rag.rougeL_f1 (ascending)"
        },
        "failure_mode_counts": categories,
        "bottom_incidents": failure_records
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[Artifact] Saved failure analysis to {OUTPUT_FILE}")
    print("\nFailure Mode Counts:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count}")

    return output_payload


if __name__ == "__main__":
    analyze_failures()
