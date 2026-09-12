import json
import glob
from pathlib import Path

def is_boilerplate(text):
    text = text.lower()
    boilerplates = [
        "this incident has been resolved",
        "no workaround at this time",
        "none",
        "n/a",
        "we are currently investigating",
        "engineers are investigating",
        "we will provide an update",
        "resolved"
    ]
    if len(text.strip()) < 20:
        return True
    for b in boilerplates:
        if text.strip() == b:
            return True
        # If the entire text is very close to boilerplate
        if len(text.strip()) < len(b) + 15 and b in text:
            return True
    return False

def tag_quality():
    raw_files = glob.glob("n:/gitt/TraceMind/scripts/scrapers/output/*_raw.json")
    for filepath in raw_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for record in data:
            rc = record.get("root_cause", "")
            rem = record.get("remediation_steps", "")
            if is_boilerplate(rc) or is_boilerplate(rem):
                record["quality"] = "low"
            else:
                record["quality"] = "high"
                
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print("Quality tagging completed on raw files.")

if __name__ == "__main__":
    tag_quality()
