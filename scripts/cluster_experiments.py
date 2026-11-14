import json
import re
import time
import numpy as np
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer
import hdbscan

print("=" * 80)
print("CLUSTER EXPERIMENTS (3 Variants x 8 HDBSCAN Configs = 24 Runs)")
print("=" * 80)

train_path = "data/splits/train.json"
with open(train_path, "r", encoding="utf-8") as f:
    train = json.load(f)

# 1. Company and boilerplate definitions for Variant C
COMPANIES_STOPLIST = {
    "github", "sentry", "cloudflare", "twilio", "discord", "shopify", "reddit",
    "atlassian", "datadog", "digitalocean", "circleci", "netlify", "openai",
    "dropbox", "squarespace", "launchdarkly", "google", "aws", "anthropic",
    "slack", "jira", "confluence", "bitbucket", "acuity", "copilot", "chatgpt",
    "travis", "travisci"
}
for r in train:
    c = r.get("company", "").strip().lower()
    if c:
        COMPANIES_STOPLIST.add(c)

company_pattern = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(COMPANIES_STOPLIST, key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)

boilerplate_patterns = [
    r"this incident has been resolved\.?",
    r"the issue has been identified and a fix is being implemented\.?",
    r"our engineering team continues to work on implementing and rolling out a fix\.?",
    r"our engineering team has confirmed that the issue.*?has been (fully )?resolved\.?",
    r"our engineering team has resolved the issue\.?",
    r"our engineering team is investigating.*?\.?",
    r"our team has identified the cause.*?\.?",
    r"we are currently experiencing a service disruption\.?",
    r"we have identified an issue.*?\.?",
    r"we have identified the root cause.*?\.?",
    r"we are actively investigating.*?\.?",
    r"a fix has been applied.*?\.?",
    r"a fix is being implemented.*?\.?",
    r"all systems are now fully operational.*?\.?",
    r"services are currently processing the backlog.*?\.?"
]
boilerplate_regex = re.compile("|".join(boilerplate_patterns), re.IGNORECASE)

def clean_variant_c(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept_sentences = [s for s in sentences if not boilerplate_regex.search(s.strip())]
        cleaned_line = " ".join(kept_sentences).strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    text_clean = " ".join(cleaned_lines)
    text_clean = company_pattern.sub("", text_clean)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()
    return text_clean

# 2. Build datasets
# Variant A: All train records
records_a = train
texts_a = [f"{r.get('raw_log_excerpt', '')[:1000]} [SEP] {r.get('root_cause', '')[:1000]}" for r in records_a]

# Variant B: High quality only
records_b = [r for r in train if r.get("quality") == "high"]
texts_b = [r.get("root_cause", "")[:1000] for r in records_b]

# Variant C: High quality only, cleaned
records_c = records_b
texts_c = [clean_variant_c(r.get("root_cause", "")[:1000]) for r in records_c]

print(f"Variant A records: {len(records_a)}")
print(f"Variant B records: {len(records_b)}")
print(f"Variant C records: {len(records_c)}")

# 3. Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
print("\nGenerating embeddings for Variant A...")
emb_a = model.encode(texts_a, show_progress_bar=False, normalize_embeddings=True, batch_size=64)

print("Generating embeddings for Variant B...")
emb_b = model.encode(texts_b, show_progress_bar=False, normalize_embeddings=True, batch_size=64)

print("Generating embeddings for Variant C...")
emb_c = model.encode(texts_c, show_progress_bar=False, normalize_embeddings=True, batch_size=64)

variants = [
    ("a) Baseline (all train, log [SEP] root_cause)", records_a, emb_a),
    ("b) High-quality only (root_cause only)", records_b, emb_b),
    ("c) High-quality, company & boilerplate stripped", records_c, emb_c)
]

min_cluster_sizes = [4, 6, 8, 10]
min_samples_list = [2, 3]

results = []

for var_name, var_records, var_emb in variants:
    print(f"\n--- Running Grid for Variant: {var_name} ---")
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
                cluster_selection_method="eom"
            )
            labels = clusterer.fit_predict(var_emb)
            n_clusters = len(set(labels) - {-1})
            n_noise = int(np.sum(labels == -1))
            total_pts = len(labels)
            noise_pct = (n_noise / total_pts) * 100.0
            
            if n_clusters == 0:
                results.append({
                    "variant": var_name,
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "n_clusters": 0,
                    "noise_pct": round(noise_pct, 1),
                    "median_company_count": 0,
                    "pct_clusters_ge_3_cos": 0.0,
                    "pct_clusters_ge_80_top": 0.0,
                    "passes_acceptance": False,
                    "labels": labels.tolist()
                })
                continue
                
            co_counts_per_cluster = []
            ge_80_flags = []
            ge_3_cos_flags = []
            
            for cid in set(labels) - {-1}:
                members = [var_records[i] for i, l in enumerate(labels) if l == cid]
                size = len(members)
                co_counter = Counter(m.get("company", "unknown") for m in members)
                n_cos = len(co_counter)
                co_counts_per_cluster.append(n_cos)
                ge_3_cos_flags.append(n_cos >= 3)
                
                top_co, top_cnt = co_counter.most_common(1)[0]
                ge_80_flags.append((top_cnt / size) >= 0.80)
                
            median_co = float(np.median(co_counts_per_cluster))
            pct_ge_3_cos = (sum(ge_3_cos_flags) / n_clusters) * 100.0
            pct_ge_80_top = (sum(ge_80_flags) / n_clusters) * 100.0
            
            passes = (pct_ge_3_cos >= 70.0) and (noise_pct <= 45.0)
            
            res_entry = {
                "variant": var_name,
                "min_cluster_size": mcs,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "noise_pct": round(noise_pct, 1),
                "median_company_count": median_co,
                "pct_clusters_ge_3_cos": round(pct_ge_3_cos, 1),
                "pct_clusters_ge_80_top": round(pct_ge_80_top, 1),
                "passes_acceptance": passes,
                "labels": labels.tolist()
            }
            results.append(res_entry)
            print(f"mcs={mcs:<2} ms={ms} | Clusters: {n_clusters:>2} | Noise: {noise_pct:>5.1f}% | Med Cos: {median_co} | % Co>=3: {pct_ge_3_cos:>5.1f}% | % Top>=80%: {pct_ge_80_top:>5.1f}% | Passes: {passes}")

# Save results
clean_results = []
for r in results:
    r_copy = dict(r)
    r_copy.pop("labels", None)
    clean_results.append(r_copy)

with open("scripts/cluster_experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(clean_results, f, indent=2)

print("\nFinished all 24 grid configurations.")
