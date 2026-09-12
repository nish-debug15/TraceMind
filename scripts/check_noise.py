import json
import pickle
from pathlib import Path

def check_correlation():
    DATA_PATH = Path("n:/gitt/TraceMind/data/postmortems.json")
    INDEX_PATH = Path("n:/gitt/TraceMind/clustering/cluster_index.pkl")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
        
    labels = index["labels"]
    
    noise_quality = {"low": 0, "high": 0}
    cluster_quality = {"low": 0, "high": 0}
    
    for r, label in zip(records, labels):
        q = r.get("quality", "unknown")
        if label == -1:
            noise_quality[q] = noise_quality.get(q, 0) + 1
        else:
            cluster_quality[q] = cluster_quality.get(q, 0) + 1
            
    total_noise = sum(noise_quality.values())
    total_clustered = sum(cluster_quality.values())
    
    print("--- Noise Correlation ---")
    if total_noise > 0:
        print(f"Noise points: {total_noise}")
        print(f"  Quality Low:  {noise_quality['low']} ({(noise_quality['low']/total_noise)*100:.1f}%)")
        print(f"  Quality High: {noise_quality['high']} ({(noise_quality['high']/total_noise)*100:.1f}%)")
    
    if total_clustered > 0:
        print(f"\nClustered points: {total_clustered}")
        print(f"  Quality Low:  {cluster_quality['low']} ({(cluster_quality['low']/total_clustered)*100:.1f}%)")
        print(f"  Quality High: {cluster_quality['high']} ({(cluster_quality['high']/total_clustered)*100:.1f}%)")

if __name__ == "__main__":
    check_correlation()
