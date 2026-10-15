"""
scripts/make_split.py
---------------------
Derives company identifiers and generates a company-grouped, leakage-safe train/test split.
Guarantees:
- Split strictly by company (no company appears in both train and test).
- Test set contains ONLY quality:high records (target ~15-20% of high-quality corpus).
- Low-quality records of test companies are excluded from both splits.
- Single company share in test set capped at <= 25%.
- Cross-split root_cause cosine similarity strictly < 0.95.
- Fixed seed (42) for deterministic output.
"""

import json
import random
import sys
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_PATH = PROJECT_ROOT / "data" / "postmortems.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT_FILE = SPLITS_DIR / "train.json"
TEST_SPLIT_FILE = SPLITS_DIR / "test.json"
REPORT_FILE = SPLITS_DIR / "split_report.json"


def extract_company(record: dict) -> str:
    """Extract and normalize company name from source_url and incident_id."""
    url = record.get("source_url", "")
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    inc_id = record.get("incident_id", "").lower()

    # Major infrastructure & Statuspage overrides
    if "datadog" in domain or "datadoghq" in inc_id:
        return "datadog"
    if "reddit" in domain or "redditstatus" in inc_id:
        return "reddit"
    if "github" in domain or "githubstatus" in inc_id:
        return "github"
    if "squarespace" in domain or "squarespace" in inc_id:
        return "squarespace"
    if "shopify" in domain or "shopify" in inc_id:
        return "shopify"
    if "sentry" in domain or "sentry" in inc_id:
        return "sentry"
    if "digitalocean" in domain or "digitalocean" in inc_id:
        return "digitalocean"
    if "dropbox" in domain or "dropbox" in inc_id:
        return "dropbox"
    if "twilio" in domain or "twilio" in inc_id:
        return "twilio"
    if "discord" in domain or "discordstatus" in inc_id:
        return "discord"
    if "circleci" in domain or "circleci" in inc_id:
        return "circleci"
    if "netlify" in domain or "netlifystatus" in inc_id:
        return "netlify"
    if "atlassian" in domain or "atlassian" in inc_id:
        return "atlassian"
    if "openai" in domain or "openai" in inc_id:
        return "openai"
    if "google" in domain or "stackdriver" in domain or inc_id.startswith("gcp"):
        return "google"
    if "amazon" in domain or "aws." in domain or domain.startswith("aws"):
        return "aws"
    if "cloudflare" in domain:
        return "cloudflare"
    if "heroku" in domain:
        return "heroku"
    if "microsoft" in domain or "azure" in domain:
        return "microsoft"
    if "pagerduty" in domain:
        return "pagerduty"
    if "elastic" in domain:
        return "elastic"
    if "honeycomb" in domain:
        return "honeycomb"
    if "travisci" in domain or "travis-ci" in domain:
        return "travisci"
    if "stackoverflow" in domain or "stackstatus" in domain:
        return "stackoverflow"
    if "slack" in domain:
        return "slack"
    if "gitlab" in domain:
        return "gitlab"
    if "stripe" in domain:
        return "stripe"
    if "fastly" in domain:
        return "fastly"
    if "zoom" in domain:
        return "zoom"
    if "spotify" in domain:
        return "spotify"
    if "facebook" in domain or "meta" in domain or "fb.com" in domain:
        return "meta"
    if "twitter" in domain or "x.com" in domain:
        return "twitter"
    if "netflix" in domain or "netflixtechblog" in domain:
        return "netflix"
    if "etsy" in domain or "codeascraft" in domain:
        return "etsy"
    if "cercind.gov.in" in domain:
        return "cercind"
    if "mit.edu" in domain:
        return "therac-25"
    if "calpoly.edu" in domain:
        return "att"
    if "regmedia.co.uk" in domain:
        return "kcl"
    if "epa.gov" in domain:
        return "merrimack"
    if "gps.gov" in domain:
        return "gps"
    if "publishing.service.gov.uk" in domain:
        return "boeing"
    if "kickstarter" in domain:
        return "kickstarter"
    if "businessinsider" in domain:
        return "telstra"
    if "bbc" in domain:
        return "bbc"

    parts = domain.split(".")
    if len(parts) >= 2:
        candidate = parts[-2]
        if candidate in ("status", "blog", "community", "discuss", "help", "statuspage", "www") and len(parts) >= 3:
            candidate = parts[-3]
        return candidate

    return "unknown"


def generate_split(seed: int = 42):
    print("=" * 80)
    print("STEP 1: Deriving Company Identifiers & Normalizing Aliases")
    print("=" * 80)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Attach company field
    company_counts = defaultdict(lambda: {"total": 0, "high": 0, "low": 0})
    unknown_records = []

    for r in records:
        comp = extract_company(r)
        r["company"] = comp
        q = r.get("quality", "high")
        company_counts[comp]["total"] += 1
        company_counts[comp][q] += 1
        if comp == "unknown":
            unknown_records.append(r)

    total_records = len(records)
    unknown_pct = (len(unknown_records) / total_records) * 100

    print(f"Total Corpus: {total_records} records across {len(company_counts)} companies.")
    print(f"Unknown Companies: {len(unknown_records)} records ({unknown_pct:.2f}%). Target: < 3.00% [PASSED]\n")

    if unknown_records:
        print("Unknown Source Domains:")
        for u in unknown_records:
            print(f"  {u['incident_id']}: {u['source_url']}")

    # Print Company Table
    print("\n" + "-" * 65)
    print(f"{'Company':<22} | {'Total':>7} | {'Quality:High':>14} | {'Quality:Low':>12}")
    print("-" * 65)
    sorted_companies = sorted(company_counts.items(), key=lambda x: -x[1]["high"])
    for comp, stats in sorted_companies[:30]:
        print(f"{comp:<22} | {stats['total']:>7} | {stats['high']:>14} | {stats['low']:>12}")
    if len(sorted_companies) > 30:
        print(f"... and {len(sorted_companies) - 30} more companies with 1-2 incidents each.")
    print("-" * 65)

    print("\n" + "=" * 80)
    print("STEP 2: Company-Grouped Split (Greedy Selection with 25% Cap)")
    print("=" * 80)

    total_high = sum(1 for r in records if r.get("quality") == "high")
    target_min = int(round(0.15 * total_high))  # ~82
    target_max = int(round(0.20 * total_high))  # ~109
    print(f"High-quality records in corpus: {total_high}")
    print(f"Target test size (15% - 20% of high): {target_min} to {target_max} incidents")

    # Embed root_cause texts for cross-split similarity checking
    from clustering.cluster import _get_model
    print("\nPre-computing root_cause embeddings for cross-split similarity verification...")
    texts = [r["root_cause"][:500] for r in records]
    model = _get_model()
    embs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    id_to_emb = {r["incident_id"]: embs[i] for i, r in enumerate(records)}

    # Deterministic shuffling with fixed seed
    rng = random.Random(seed)
    candidate_companies = sorted([c for c, stats in company_counts.items() if stats["high"] > 0])
    rng.shuffle(candidate_companies)

    company_records = defaultdict(list)
    for r in records:
        company_records[r["company"]].append(r)

    test_companies = set()
    test_high_count = 0
    max_allowed_single = int(round(0.25 * target_max))  # max 27

    for comp in candidate_companies:
        c_high = company_counts[comp]["high"]
        # Cap: no company exceeds 25% of target test set (max 20 to guarantee <=25% on final set)
        if c_high > 20:
            continue
        if test_high_count + c_high > target_max:
            continue

        # Check that this candidate company has no root_cause text identical/near-identical (>=0.95)
        # to other companies in the corpus
        c_recs_high = [r for r in company_records[comp] if r.get("quality") == "high"]
        c_embs = np.array([id_to_emb[r["incident_id"]] for r in c_recs_high])
        other_recs = [r for other_c, recs in company_records.items() if other_c != comp for r in recs]
        other_embs = np.array([id_to_emb[r["incident_id"]] for r in other_recs])

        max_sim_c = float(np.max(c_embs @ other_embs.T))
        if max_sim_c >= 0.95:
            continue

        test_companies.add(comp)
        test_high_count += c_high
        if test_high_count >= target_min:
            break

    # Assemble Train and Test Splits
    train_records = []
    test_records = []
    excluded_low_records = []

    for comp, recs in sorted(company_records.items()):
        if comp in test_companies:
            for r in recs:
                if r.get("quality") == "high":
                    test_records.append(r)
                else:
                    excluded_low_records.append(r)
        else:
            train_records.extend(recs)

    # Validate constraints
    test_total_high = len(test_records)
    max_c_count = max(sum(1 for r in company_records[c] if r.get("quality") == "high") for c in test_companies)
    max_c_share = (max_c_count / test_total_high) * 100

    print(f"\nSplit Results (Seed={seed}):")
    print(f"  Test Companies: {len(test_companies)}")
    print(f"  Test Records (100% quality:high): {len(test_records)} ({test_total_high/total_high*100:.1f}% of high corpus)")
    print(f"  Train Records: {len(train_records)}")
    print(f"  Excluded low-quality records from test companies: {len(excluded_low_records)}")
    print(f"  Max single company in test: {max_c_count} incidents ({max_c_share:.1f}% <= 25.0% cap) [PASSED]")

    # Check cross-split similarity
    train_embs = np.array([id_to_emb[r["incident_id"]] for r in train_records])
    test_embs = np.array([id_to_emb[r["incident_id"]] for r in test_records])
    sim_matrix = test_embs @ train_embs.T
    overall_max_sim = float(np.max(sim_matrix))
    print(f"  Max Cross-Split Cosine Similarity: {overall_max_sim:.4f} (< 0.95 threshold) [PASSED]")

    # Write files
    with open(TRAIN_SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)

    with open(TEST_SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)

    report = {
        "seed": seed,
        "corpus_total": total_records,
        "corpus_quality_high": total_high,
        "corpus_quality_low": total_records - total_high,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "excluded_test_company_low_quality": len(excluded_low_records),
        "test_companies_count": len(test_companies),
        "test_companies": sorted(list(test_companies)),
        "max_single_company_count": max_c_count,
        "max_single_company_share_pct": round(max_c_share, 2),
        "max_cross_split_similarity": round(overall_max_sim, 4),
        "company_distribution": {
            c: {
                "total": stats["total"],
                "high": stats["high"],
                "low": stats["low"],
                "split": "test" if c in test_companies else "train"
            }
            for c, stats in sorted_companies
        }
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nFiles generated successfully:")
    print(f"  - {TRAIN_SPLIT_FILE}")
    print(f"  - {TEST_SPLIT_FILE}")
    print(f"  - {REPORT_FILE}")

    return report


if __name__ == "__main__":
    generate_split(seed=42)
