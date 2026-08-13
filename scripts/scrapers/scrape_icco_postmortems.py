"""
Scraper for Nat Welch's icco/postmortems — pre-parsed database of danluu/post-mortems.
Downloads the repo as a zip (single request) to avoid GitHub API rate limits,
then parses YAML frontmatter + Markdown body from each data/*.md file.
"""
import requests
import json
import re
import io
import zipfile
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "output" / "icco_raw.json"

REPO_ZIP_URL = "https://github.com/icco/postmortems/archive/refs/heads/main.zip"
HEADERS = {"User-Agent": "TraceMind-Scraper/1.0"}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text. Returns (metadata_dict, body)."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    # Parse YAML manually (avoid heavy dep — fields are simple key: value)
    metadata = {}
    for line in parts[1].strip().split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value:
                metadata[key] = value
    body = parts[2].strip()
    return metadata, body


def extract_code_blocks(body: str) -> list[str]:
    """Extract fenced code blocks (potential log excerpts)."""
    pattern = r"```[\w]*\n(.*?)```"
    matches = re.findall(pattern, body, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def extract_section(body: str, keywords: list[str]) -> str:
    """Extract text under a markdown heading matching any of the keywords."""
    lines = body.split("\n")
    capture = False
    captured = []
    for line in lines:
        if line.startswith("#"):
            if capture:
                break  # hit next heading, stop
            heading_text = line.lstrip("#").strip().lower()
            if any(kw in heading_text for kw in keywords):
                capture = True
                continue
        elif capture:
            captured.append(line)
    return "\n".join(captured).strip()


def extract_fields(body: str) -> tuple[str, str, str]:
    """Extract raw_log_excerpt, root_cause, remediation_steps from body."""

    # 1. raw_log_excerpt: prefer code blocks, else first paragraph
    code_blocks = extract_code_blocks(body)
    if code_blocks:
        raw_log = code_blocks[0][:500]
    else:
        # First substantive paragraph (skip very short lines)
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip() and len(p.strip()) > 30]
        raw_log = paragraphs[0][:500] if paragraphs else body[:300]

    # 2. root_cause: look for section headings
    root_cause = extract_section(body, ["root cause", "cause", "what happened", "what went wrong", "analysis", "summary"])
    if not root_cause:
        # Fallback: search for paragraphs mentioning causal language
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        for p in paragraphs:
            p_lower = p.lower()
            if any(w in p_lower for w in ["caused by", "root cause", "due to", "because", "resulted from", "failure"]):
                root_cause = p[:500]
                break
    if not root_cause:
        # Last resort: use first substantive paragraph
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip() and len(p.strip()) > 30]
        root_cause = paragraphs[0][:500] if paragraphs else body[:300]

    # 3. remediation: look for section headings
    remediation = extract_section(body, ["remediation", "fix", "resolution", "action item", "what we did",
                                         "corrective", "mitigation", "recovery", "lesson", "prevention"])
    if not remediation:
        # Fallback: search for actionable language
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        for p in paragraphs:
            p_lower = p.lower()
            if any(w in p_lower for w in ["fixed", "resolved", "rolled back", "deployed", "patched",
                                           "mitigated", "restored", "implemented", "updated"]):
                remediation = p[:500]
                break
    if not remediation:
        # Last resort: use last paragraph
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip() and len(p.strip()) > 30]
        remediation = paragraphs[-1][:500] if paragraphs else body[-300:]

    return raw_log, root_cause, remediation


def main():
    print("=" * 60)
    print("Source 1: icco/postmortems (Nat Welch's parsed database)")
    print("=" * 60)

    print(f"Downloading repo zip from {REPO_ZIP_URL}...")
    try:
        r = requests.get(REPO_ZIP_URL, headers=HEADERS, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"FATAL: Failed to download repo zip: {e}")
        return

    print(f"Downloaded {len(r.content) / 1024:.0f} KB. Extracting...")

    z = zipfile.ZipFile(io.BytesIO(r.content))
    md_files = [f for f in z.namelist() if "/data/" in f and f.endswith(".md")]
    print(f"Found {len(md_files)} markdown files in data/")

    results = []
    skipped = 0

    for filepath in sorted(md_files):
        filename = Path(filepath).stem
        try:
            content = z.read(filepath).decode("utf-8", errors="replace")
            metadata, body = parse_frontmatter(content)

            source_url = metadata.get("url") or metadata.get("archive_url")
            if not source_url:
                skipped += 1
                continue

            if not body or len(body.strip()) < 50:
                skipped += 1
                continue

            raw_log, root_cause, remediation = extract_fields(body)

            # Skip if we couldn't extract meaningful content
            if not raw_log or not root_cause:
                skipped += 1
                continue

            results.append({
                "incident_id": f"icco-{filename}",
                "raw_log_excerpt": raw_log.strip(),
                "root_cause": root_cause.strip(),
                "remediation_steps": remediation.strip(),
                "source_url": source_url
            })
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            skipped += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults: {len(results)} records saved, {skipped} skipped")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
