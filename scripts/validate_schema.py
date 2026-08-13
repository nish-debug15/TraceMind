import json
import sys

REQUIRED_FIELDS = {"incident_id", "raw_log_excerpt", "root_cause", "remediation_steps", "source_url"}

def validate():
    try:
        with open("data/postmortems.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("data/postmortems.json not found.")
        sys.exit(1)

    if not isinstance(data, list):
        print("Root should be a list of records.")
        sys.exit(1)
        
    valid = 0
    invalid = 0
    for record in data:
        keys = set(record.keys())
        if keys != REQUIRED_FIELDS:
            invalid += 1
            continue
        if not record.get("incident_id") or not record.get("source_url"):
            invalid += 1
            continue
        valid += 1
        
    print(f"Total valid: {valid}, Total invalid: {invalid}")

if __name__ == "__main__":
    validate()
