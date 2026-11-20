"""
TraceMind Manual Relevance Audit Agreement Scorer (Block 5)
-----------------------------------------------------------
Reads eval/audit/relevance_audit_sheet.csv and eval/audit/relevance_audit_key.json.
Checks that all 30 human_label cells are populated with integer values in {0, 1, 2}.
Computes:
- Exact agreement (%)
- Quadratic-weighted Cohen's kappa

If human_label column is still blank, exits cleanly with message:
'Audit pending: 30 rows unjudged'
"""

import sys
import csv
import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "eval" / "audit"
SHEET_FILE = AUDIT_DIR / "relevance_audit_sheet.csv"
KEY_FILE = AUDIT_DIR / "relevance_audit_key.json"


def score_audit():
    if not SHEET_FILE.exists():
        print(f"Error: Audit sheet {SHEET_FILE} does not exist.")
        sys.exit(1)

    if not KEY_FILE.exists():
        print(f"Error: Audit key {KEY_FILE} does not exist.")
        sys.exit(1)

    with open(KEY_FILE, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    with open(SHEET_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 30:
        print(f"Error: Expected 30 rows in audit sheet, found {len(rows)}.")
        sys.exit(1)

    unjudged = [r for r in rows if not str(r.get("human_label", "")).strip()]
    if unjudged:
        print(f"Audit pending: {len(unjudged)} rows unjudged")
        sys.exit(0)

    # Validate human labels
    human_labels = []
    judge_labels = []
    
    for r in rows:
        pid = r["pair_id"]
        val_str = str(r["human_label"]).strip()
        try:
            val = int(val_str)
            if val not in (0, 1, 2):
                raise ValueError()
        except ValueError:
            print(f"Error: Invalid human_label '{val_str}' for {pid}. Must be 0, 1, or 2.")
            sys.exit(1)

        human_labels.append(val)
        judge_val = int(key_data[pid]["judge_score"])
        judge_labels.append(judge_val)

    # Compute exact agreement
    exact_matches = sum(1 for h, j in zip(human_labels, judge_labels) if h == j)
    exact_agreement_pct = (exact_matches / len(rows)) * 100.0

    # Compute quadratic-weighted Cohen's kappa
    kappa = cohen_kappa_score(judge_labels, human_labels, labels=[0, 1, 2], weights="quadratic")

    print("=" * 60)
    print("MANUAL RELEVANCE AUDIT AGREEMENT RESULTS (N = 30)")
    print("=" * 60)
    print(f"Exact Agreement:              {exact_matches}/30 ({exact_agreement_pct:.2f}%)")
    print(f"Quadratic-Weighted Cohen's κ: {kappa:.4f}")
    print("=" * 60)

    return exact_agreement_pct, kappa


if __name__ == "__main__":
    score_audit()
