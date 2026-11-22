"""
eval/loo/sample.py
------------------
Selects an evaluation sample for Experiment 2 (E2: Leave-One-Out Same-Company Regime).
Rules (fixed a priori in eval/PREREG.md):
- Filter companies with >= 5 quality: high incidents in train.json.
- Cap at max 12 incidents per company so one company doesn't dominate.
- Sample exactly 93 incidents using seed 42.
- Save to eval/loo/sample.json.
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.json"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "loo" / "sample.json"


def select_loo_sample(seed: int = 42, target_size: int = 93, max_per_company: int = 12, min_company_high: int = 5) -> Dict[str, Any]:
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_records = json.load(f)

    # 1. Filter quality: high incidents
    high_quality_records = [r for r in train_records if r.get("quality") == "high"]

    # 2. Group by company
    company_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in high_quality_records:
        comp = r.get("company", "Unknown")
        company_groups.setdefault(comp, []).append(r)

    # 3. Filter companies with >= min_company_high incidents
    eligible_companies = {
        comp: records for comp, records in company_groups.items()
        if len(records) >= min_company_high
    }

    # 4. Cap at max_per_company per eligible company
    rng = random.Random(seed)
    capped_pool: List[Dict[str, Any]] = []
    company_pool_sizes: Dict[str, int] = {}

    for comp in sorted(eligible_companies.keys()):
        items = list(eligible_companies[comp])
        # Sort by incident_id first for deterministic ordering before shuffle
        items.sort(key=lambda x: x["incident_id"])
        rng.shuffle(items)
        selected = items[:max_per_company]
        capped_pool.extend(selected)
        company_pool_sizes[comp] = len(selected)

    # 5. Shuffle capped pool and select target_size incidents
    # Sort capped pool deterministically first
    capped_pool.sort(key=lambda x: x["incident_id"])
    rng.shuffle(capped_pool)
    sample_records = capped_pool[:target_size]

    # Compute company distribution in final sample
    sample_distribution: Dict[str, int] = {}
    for r in sample_records:
        c = r.get("company", "Unknown")
        sample_distribution[c] = sample_distribution.get(c, 0) + 1

    sample_distribution = dict(sorted(sample_distribution.items(), key=lambda x: (-x[1], x[0])))

    results = {
        "metadata": {
            "source_corpus": "data/splits/train.json",
            "total_train_records": len(train_records),
            "total_high_quality_train": len(high_quality_records),
            "min_company_high_threshold": min_company_high,
            "max_per_company_cap": max_per_company,
            "num_eligible_companies": len(eligible_companies),
            "eligible_companies": sorted(list(eligible_companies.keys())),
            "capped_pool_size": len(capped_pool),
            "sample_size": len(sample_records),
            "num_sampled_companies": len(sample_distribution),
            "seed": seed,
            "company_distribution": sample_distribution,
        },
        "sample": sample_records,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[LOO Sampler] Successfully sampled {len(sample_records)} incidents across {len(sample_distribution)} companies.")
    print(f"[LOO Sampler] Written to {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    select_loo_sample()
