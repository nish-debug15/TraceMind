"""
Merge and deduplicate all scraper outputs into the final data/postmortems.json.
Loads all *_raw.json files from output/, deduplicates by source_url,
validates schema, and writes the final dataset.
"""
import json
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path(__file__).parent / "output"
FINAL_OUT = Path(__file__).parent.parent.parent / "data" / "postmortems.json"

REQUIRED_FIELDS = {"incident_id", "raw_log_excerpt", "root_cause", "remediation_steps", "source_url"}


def main():
    print("=" * 60)
    print("Merge & Dedup: Combining all scraper outputs")
    print("=" * 60)

    all_records = []
    source_counts = Counter()

    # Load all raw files
    for raw_file in sorted(OUTPUT_DIR.glob("*_raw.json")):
        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            source_name = raw_file.stem.replace("_raw", "")
            print(f"  Loaded {len(data):>4d} records from {raw_file.name}")
            source_counts[source_name] = len(data)
            all_records.extend(data)
        except Exception as e:
            print(f"  Error loading {raw_file.name}: {e}")

    print(f"\nTotal raw records: {len(all_records)}")

    # Validate and filter
    valid_records = []
    invalid_count = 0
    invalid_reasons = Counter()

    for r in all_records:
        keys = set(r.keys())
        missing = REQUIRED_FIELDS - keys
        extra = keys - REQUIRED_FIELDS

        if missing:
            invalid_count += 1
            invalid_reasons[f"missing fields: {missing}"] += 1
            continue
        if extra:
            # Remove extra fields silently — some scrapers might add metadata
            for k in extra:
                del r[k]

        # Check non-empty
        empty_fields = [k for k in REQUIRED_FIELDS if not r.get(k, "").strip()]
        if empty_fields:
            invalid_count += 1
            invalid_reasons[f"empty fields: {empty_fields}"] += 1
            continue

        # Check source_url is a real URL
        if not r["source_url"].startswith("http"):
            invalid_count += 1
            invalid_reasons["invalid source_url"] += 1
            continue

        valid_records.append(r)

    print(f"Valid records: {len(valid_records)}")
    if invalid_count:
        print(f"Invalid records dropped: {invalid_count}")
        for reason, count in invalid_reasons.most_common():
            print(f"  - {reason}: {count}")

    # Deduplicate by source_url (keep the record with longest root_cause on conflict)
    deduped = {}
    duplicates = 0

    for r in valid_records:
        url = r["source_url"].rstrip("/").lower()
        if url in deduped:
            duplicates += 1
            # Keep the one with more content
            existing = deduped[url]
            existing_len = len(existing.get("root_cause", "")) + len(existing.get("remediation_steps", ""))
            new_len = len(r.get("root_cause", "")) + len(r.get("remediation_steps", ""))
            if new_len > existing_len:
                deduped[url] = r
        else:
            deduped[url] = r

    final_list = list(deduped.values())

    print(f"Duplicates removed: {duplicates}")
    print(f"Final record count: {len(final_list)}")

    # Per-source breakdown of final records
    final_source_counts = Counter()
    for r in final_list:
        prefix = r["incident_id"].split("-")[0]
        final_source_counts[prefix] += 1

    print(f"\nFinal breakdown by source:")
    for src, count in final_source_counts.most_common():
        print(f"  {src}: {count}")

    # Write output
    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_OUT, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {FINAL_OUT}")


if __name__ == "__main__":
    main()
