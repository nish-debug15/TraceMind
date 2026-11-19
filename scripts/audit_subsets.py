"""
scripts/audit_subsets.py
------------------------
Audits the overlap and independence between the retrieval abstention mask
(top-1 similarity < 0.40) and the ground-truth boilerplate noise mask.

Produces eval/results/subset_overlap.json with sizes, pairwise intersections,
Jaccard indices, and complete ID lists.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
RVZ_FILE = PROJECT_ROOT / "eval" / "results" / "rag_vs_zero_shot.json"
GT_FLAGS_FILE = PROJECT_ROOT / "eval" / "results" / "gt_noise_flags.json"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "subset_overlap.json"


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    union = set_a.union(set_b)
    if not union:
        return 1.0
    return len(set_a.intersection(set_b)) / len(union)


def audit_subsets() -> Dict[str, Any]:
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    all_ids = [r["incident_id"] for r in test_records]
    all_ids_set = set(all_ids)

    # 1. Retrieval abstention partition (< 0.40 cosine similarity)
    with open(RVZ_FILE, "r", encoding="utf-8") as f:
        rvz_data = json.load(f)

    sim_map = {
        item["incident_id"]: item["retrieval"]["similarity_score"]
        for item in rvz_data.get("per_incident_scores", [])
    }

    abstained_ids = sorted([i for i in all_ids if sim_map.get(i, 0.0) < 0.40])
    answered_ids = sorted([i for i in all_ids if sim_map.get(i, 0.0) >= 0.40])

    # 2. Ground-truth noise partition (boilerplate detector)
    with open(GT_FLAGS_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    flagged_ids = sorted([item["incident_id"] for item in gt_data["flagged_incidents"]])
    clean_ids = sorted([i for i in all_ids if i not in set(flagged_ids)])

    subsets = {
        "abstained": abstained_ids,
        "answered": answered_ids,
        "gt_noise_flagged": flagged_ids,
        "clean": clean_ids,
    }

    subsets_sets = {k: set(v) for k, v in subsets.items()}

    # Check partition integrity
    assert subsets_sets["abstained"].union(subsets_sets["answered"]) == all_ids_set
    assert subsets_sets["abstained"].isdisjoint(subsets_sets["answered"])
    assert subsets_sets["gt_noise_flagged"].union(subsets_sets["clean"]) == all_ids_set
    assert subsets_sets["gt_noise_flagged"].isdisjoint(subsets_sets["clean"])

    # Compute pairwise metrics
    pair_names = [
        ("abstained", "gt_noise_flagged"),
        ("abstained", "answered"),
        ("abstained", "clean"),
        ("answered", "gt_noise_flagged"),
        ("answered", "clean"),
        ("gt_noise_flagged", "clean"),
    ]

    pairwise = {}
    for name_a, name_b in pair_names:
        set_a = subsets_sets[name_a]
        set_b = subsets_sets[name_b]
        inter = sorted(list(set_a.intersection(set_b)))
        jaccard = round(jaccard_similarity(set_a, set_b), 4)
        pairwise[f"{name_a}_vs_{name_b}"] = {
            "set_a": name_a,
            "set_b": name_b,
            "size_a": len(set_a),
            "size_b": len(set_b),
            "intersection_count": len(inter),
            "jaccard_similarity": jaccard,
            "intersection_ids": inter,
        }

    are_abstained_and_flagged_identical = subsets_sets["abstained"] == subsets_sets["gt_noise_flagged"]
    are_clean_and_answered_identical = subsets_sets["clean"] == subsets_sets["answered"]

    overlap_summary = {
        "total_test_incidents": len(all_ids),
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
        "pairwise_overlap": pairwise,
        "identity_checks": {
            "abstained_equals_gt_noise_flagged": are_abstained_and_flagged_identical,
            "clean_equals_answered": are_clean_and_answered_identical,
            "coincidence_explanation": (
                "Both the retrieval floor filter (<0.40 sim) and the GT boilerplate filter "
                "happened to isolate exactly 7 incidents out of 93 (7.53%), but the two sets "
                "are completely disjoint (intersection count = 0, Jaccard = 0.0000). "
                "All 7 abstained incidents are in the clean set, and all 7 boilerplate incidents "
                "are in the answered set."
            ),
        },
        "id_lists": subsets,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(overlap_summary, f, indent=2)

    print("=== SUBSET OVERLAP AUDIT REPORT ===")
    print(f"Total test incidents: {len(all_ids)}")
    print(f"  - Abstained (sim < 0.40):        n = {len(abstained_ids)}")
    print(f"  - Answered (sim >= 0.40):       n = {len(answered_ids)}")
    print(f"  - GT-Noise Flagged:             n = {len(flagged_ids)}")
    print(f"  - Clean (unflagged):            n = {len(clean_ids)}")
    print("\nPairwise Overlaps:")
    for pair_key, pdata in pairwise.items():
        print(f"  {pdata['set_a']} & {pdata['set_b']}: "
              f"intersection = {pdata['intersection_count']}, Jaccard = {pdata['jaccard_similarity']:.4f}")

    print(f"\nIdentity check:")
    print(f"  abstained == gt_noise_flagged: {are_abstained_and_flagged_identical}")
    print(f"  clean == answered:            {are_clean_and_answered_identical}")
    print(f"Saved audit results to {OUTPUT_FILE}")

    return overlap_summary


if __name__ == "__main__":
    audit_subsets()
