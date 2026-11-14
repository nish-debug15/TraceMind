import json
import pickle
from collections import Counter

# Load index and train data
with open("clustering/cluster_index.pkl", "rb") as f:
    idx = pickle.load(f)

with open("data/splits/train.json", "r", encoding="utf-8") as f:
    train = json.load(f)

labels = idx["labels"]
n_clusters = idx["n_clusters"]

print(f"{'CID':<4} | {'Size':<5} | {'Cos':<4} | {'Top Company':<15} | {'Top Share':<10} | {'>=80%?'}")
print("-" * 55)

ge_80_count = 0
for cid in range(n_clusters):
    members = [train[i] for i, l in enumerate(labels) if l == cid]
    size = len(members)
    co_counts = Counter(m.get("company", "unknown") for m in members)
    top_company, top_cnt = co_counts.most_common(1)[0]
    top_share = top_cnt / size
    ge_80 = top_share >= 0.80
    if ge_80:
        ge_80_count += 1
    print(f"{cid:<4} | {size:<5} | {len(co_counts):<4} | {top_company:<15} | {top_share * 100:>6.1f}%   | {'YES' if ge_80 else 'NO'}")

fraction_ge_80 = ge_80_count / n_clusters
print("-" * 55)
print(f"Total clusters: {n_clusters}")
print(f"Clusters with top company >= 80%: {ge_80_count} / {n_clusters} ({fraction_ge_80 * 100:.1f}%)")
