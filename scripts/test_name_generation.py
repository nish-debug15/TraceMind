import pickle
import json
import os
import re
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

load_dotenv(".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

INDEX_PATH = "clustering/cluster_index.pkl"
TRAIN_PATH = "data/splits/train.json"
OLD_PROFILES_PATH = "clustering/cluster_profiles.json"

with open(INDEX_PATH, "rb") as f:
    idx = pickle.load(f)
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train = json.load(f)
with open(OLD_PROFILES_PATH, "r", encoding="utf-8") as f:
    old_profiles = json.load(f)

labels = idx["labels"]
n_clusters = idx["n_clusters"]

# Group by cluster
clusters = {}
for i, l in enumerate(labels):
    if l != -1:
        clusters.setdefault(l, []).append(train[i])

# Function to get content words
def get_content_words(name: str):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", name.lower())
    filler = {"and", "for", "the", "with", "due", "after", "via", "from"}
    return [w for w in words if w not in ENGLISH_STOP_WORDS and w not in filler]

def word_in_text(word: str, text: str) -> bool:
    base = word.rstrip('s')
    if len(base) > 4 and base.endswith('ing'):
        base = base[:-3]
    elif len(base) > 4 and base.endswith('ed'):
        base = base[:-2]
    pattern = r"\b" + re.escape(base) + r"[a-z]*\b"
    return bool(re.search(pattern, text, re.IGNORECASE))

def validate_name(name: str, members: list) -> tuple[bool, list]:
    content_words = get_content_words(name)
    half_n = len(members) / 2.0
    qualifying = []
    for w in content_words:
        count = sum(1 for m in members if word_in_text(w, m.get("root_cause", "")))
        if count >= half_n:
            qualifying.append(f"{w}({count}/{len(members)})")
    return len(qualifying) >= 2, qualifying

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

results = {}

for cid in range(n_clusters):
    members = clusters[cid]
    size = len(members)
    old_name = old_profiles.get(str(cid), {}).get("name", "Unknown")
    
    # Cap to shortest 6 members to stay compact
    rc_texts = []
    for idx_m, m in enumerate(members[:6], 1):
        co = m.get("company", "Unknown")
        rc = (m.get("root_cause") or "").strip()[:500]
        rc_texts.append(f"Incident {idx_m} (Company: {co}):\n{rc}")
    
    rc_block = "\n\n".join(rc_texts)
    
    prompt = (
        "Name what these incidents are about in 2-5 words. Use only the incidents below. "
        "If all members are from one company you may use the product name. Do not add causes that are not stated in the members.\n\n"
        f"INCIDENTS ROOT CAUSES:\n{rc_block}\n\n"
        "Provide your response in JSON format with two keys:\n"
        '- "name": 2 to 5 words in plain English naming what these incidents are about.\n'
        '- "description": exactly one sentence describing what these incidents are about.\n\n'
        "Respond with valid JSON only."
    )
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an SRE assistant. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    # Rate limit pacing
    time.sleep(3.2)
    
    for attempt in range(5):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                content = json.loads(data["choices"][0]["message"]["content"])
                new_name = content.get("name", "").strip()
                desc = content.get("description", "").strip()
                break
            elif resp.status_code == 429:
                wait_sec = float(resp.headers.get("retry-after", 10.0))
                print(f"  [429] Waiting {wait_sec}s for cluster {cid}...")
                time.sleep(wait_sec)
            else:
                time.sleep(2.0)
        except Exception as e:
            time.sleep(2.0)
    else:
        new_name = old_name
        desc = ""

    is_valid, qual = validate_name(new_name, members)
    results[cid] = {
        "id": cid,
        "size": size,
        "old_name": old_name,
        "new_name": new_name,
        "description": desc,
        "valid": is_valid,
        "qualifying": qual,
        "members_sample": [
            (m.get("company", ""), (m.get("root_cause") or "").replace("\n", " ").strip()[:100])
            for m in members[:2]
        ]
    }
    status_str = "PASS" if is_valid else "FAIL"
    print(f"Cluster {cid:>2} ({size:>2} members): [{status_str}] '{new_name}' (Qual: {qual})")

with open("scripts/name_generation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nDone! Results saved to scripts/name_generation_results.json")
