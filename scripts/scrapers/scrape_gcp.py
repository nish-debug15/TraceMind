"""
Scraper for GCP Incident History.
Single JSON file download from status.cloud.google.com/incidents.json
"""
import requests
import json
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "output" / "gcp_raw.json"
GCP_URL = "https://status.cloud.google.com/incidents.json"
HEADERS = {"User-Agent": "TraceMind-Scraper/1.0"}


def main():
    print("=" * 60)
    print("Source 3: GCP Incident History")
    print("=" * 60)

    print(f"Fetching {GCP_URL}...")
    try:
        r = requests.get(GCP_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"FATAL: Failed to fetch GCP incidents: {e}")
        return

    print(f"Got {len(data)} total incidents from GCP")

    results = []
    skipped_no_updates = 0
    skipped_too_short = 0
    skipped_no_desc = 0

    for inc in data:
        # GCP schema: each incident has 'number', 'begin', 'end', 'external_desc',
        # 'updates' (array of {text, when, status, ...}), 'service_name', 'severity', etc.
        # Also may have 'uri' field.

        updates = inc.get("updates", [])
        if len(updates) < 2:
            skipped_no_updates += 1
            continue

        desc = inc.get("external_desc", "").strip()
        if not desc:
            skipped_no_desc += 1
            continue

        # Find the most detailed update for root cause
        # Updates are typically in reverse chronological order (newest first)
        # Look for the longest substantive update as root cause
        all_update_texts = [u.get("text", "").strip() for u in updates if u.get("text", "").strip()]

        if not all_update_texts:
            skipped_too_short += 1
            continue

        # Final update (first in the list = newest = resolution)
        final_update_text = all_update_texts[0]
        if len(final_update_text) <= 50:
            skipped_too_short += 1
            continue

        # Root cause: find the most detailed update text
        # Often the "identified" or "update" messages contain the root cause
        root_cause_text = max(all_update_texts, key=len)

        # Remediation: the final resolution update
        remediation_text = final_update_text

        number = inc.get("number", inc.get("id", "unknown"))
        source_url = inc.get("uri", "")
        if source_url and not source_url.startswith("http"):
            source_url = f"https://status.cloud.google.com/{source_url.lstrip('/')}"
        elif not source_url:
            source_url = f"https://status.cloud.google.com/incidents/{number}"

        results.append({
            "incident_id": f"gcp-{number}",
            "raw_log_excerpt": desc,
            "root_cause": root_cause_text,
            "remediation_steps": remediation_text,
            "source_url": source_url
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults: {len(results)} records saved")
    print(f"Skipped: {skipped_no_updates} (< 2 updates), {skipped_too_short} (update too short), {skipped_no_desc} (no description)")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
