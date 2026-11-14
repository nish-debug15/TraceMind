import sys
import json
from pathlib import Path

sys.path.insert(0, ".")
from clustering.cluster import assign_cluster
from rag.generate import retrieve

with open("clustering/cluster_profiles.json", "r", encoding="utf-8") as f:
    profiles = json.load(f)

demo_logs = [
    ("Redis failover", "ERROR: Redis sentinel failover triggered - master 10.0.4.12:6379 unreachable for 30s. Failover elected replica 10.0.4.13:6379 as new master. Write latency spiked to 2400ms."),
    ("DB pool exhaustion", "[2026-10-15 08:23:41] [ERROR] [db-proxy-pool] Connection timeout acquiring connection from pool.\nFATAL: remaining connection slots are reserved for non-replication superuser connections.\nPoolExhaustedError: database connection pool exhausted, 50/50 connections in use\n[2026-10-15 08:23:42] [WARN] [health-check] Liveness probe failed for pod db-proxy-6b8f7d9"),
    ("OOMKilled", "CRITICAL 2026-10-14T14:12:05Z kubelet worker-node-04: OOMKilled process 18421 (payments-api) total-vm:8421048kB, anon-rss:4194304kB. Memory limit 4Gi exceeded."),
    ("Expired TLS", "WARNING: SSL certificate verification failed for api.stripe.com. OpenSSL Error: certificate has expired on 2026-10-14 23:59:59 GMT. TLS handshake terminated."),
    ("Kafka lag", "CRITICAL: Kafka consumer lag exceeding 100000 messages on topic payments-events across partitions [0, 1, 2]. Offset commit timeout."),
    ("DNS failures", "ERROR: DNS resolution failed for internal-service.cluster.local after 5 retries. SERVFAIL returned by CoreDNS upstream 10.96.0.10:53"),
    ("Disk full", "WARNING: Disk usage at 95% on /var/log/application.log - log rotation failed: No space left on device. Write operations blocked."),
    ("Bad config push", "FATAL: Application startup failed due to syntax error in config.yaml line 42: invalid boolean value 'enable_auth'. Process exited with code 1."),
    ("Duplicate burst", "ERROR: Duplicate transaction burst detected. Transaction ID txn_9841289 submitted 450 times in 3 seconds. Rate limit triggered on payment gateway."),
    ("Printer/HVAC out-of-domain", "ALERT [Building-HVAC-Unit-4]: Chiller compressor head pressure exceeded 450 PSI. Ambient temperature 104F. Emergency refrigerant purge valve opened. Office printer Paper Jam Tray 2."),
]

results = []

for scenario, log in demo_logs:
    # 1. Cluster assignment
    c_res = assign_cluster(log)
    cid = c_res["cluster_id"]
    is_noise = c_res["is_noise"]
    sim = c_res["confidence"]
    
    prof = profiles.get(str(cid), {})
    c_name = prof.get("name", "Unknown")
    c_conf = prof.get("confidence", "ok")
    
    # 2. RAG retrieval
    match_meta, rag_sim = retrieve(log, k=1)
    ret_co = match_meta.get("company", "Unknown") if match_meta else "None"
    ret_rc = (match_meta.get("root_cause") or "").replace("\n", " ").strip()[:100] if match_meta else "None"
    
    results.append({
        "scenario": scenario,
        "cluster_id": cid,
        "name": c_name,
        "confidence": c_conf,
        "centroid_sim": sim,
        "is_noise": "Yes" if is_noise else "No",
        "retrieved_company": ret_co,
        "retrieved_root_cause": ret_rc,
        "retrieved_sim": rag_sim
    })

with open("scripts/demo_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"{'Scenario':<26} | {'CID':>3} | {'Name':<35} | {'Conf':<4} | {'Sim':>6} | {'Noise':<5} | {'Retrieved Co':<12} | {'Retrieved RC'}")
print("-" * 135)
for r in results:
    print(f"{r['scenario']:<26} | {r['cluster_id']:>3} | {r['name'][:35]:<35} | {r['confidence']:<4} | {r['centroid_sim']:>6.4f} | {r['is_noise']:<5} | {r['retrieved_company']:<12} | {r['retrieved_root_cause'][:50]}")
