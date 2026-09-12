import numpy as np
from clustering.cluster import embed

def test_threshold():
    similar_pairs = [
        ("database connection pool exhausted, 50/50 connections in use", "DB connection pool full, all connections allocated"),
        ("network latency spike detected on eth0", "high latency observed on network interface eth0"),
        ("OOMKilled container web-api-7f8d9b", "container killed due to out of memory (OOM)"),
        ("SSL certificate for api.example.com expires in 3 days", "TLS cert expiring soon for api.example.com")
    ]
    unrelated_pairs = [
        ("database connection pool exhausted, 50/50 connections in use", "SSL certificate for api.example.com expires in 3 days"),
        ("OOMKilled container web-api-7f8d9b", "network latency spike detected on eth0"),
        ("DNS resolution failed for internal-service.cluster.local after 5 retries", "Redis sentinel failover triggered - master unreachable for 30s"),
        ("ALERT: CPU utilization at 98% on worker-node-12 for past 15 minutes", "ERROR: HTTP 503 from upstream load balancer - all backends unhealthy")
    ]

    print("--- Similar Pairs ---")
    for t1, t2 in similar_pairs:
        v1, v2 = np.array(embed(t1)), np.array(embed(t2))
        sim = float(v1 @ v2)
        print(f"Sim: {sim:.4f} | {t1[:30]}... <-> {t2[:30]}...")

    print("\n--- Unrelated Pairs ---")
    for t1, t2 in unrelated_pairs:
        v1, v2 = np.array(embed(t1)), np.array(embed(t2))
        sim = float(v1 @ v2)
        print(f"Sim: {sim:.4f} | {t1[:30]}... <-> {t2[:30]}...")

if __name__ == "__main__":
    test_threshold()
