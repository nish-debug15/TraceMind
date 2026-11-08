import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.generate import retrieve

test_path = Path("data/test.json")
with open(test_path, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Total test records: {len(records)}")

# Define categorization rule based on root cause keywords / patterns
def categorize(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["traffic spike", "pool exhausted", "capacity", "connection pool", "rate limit", "saturation", "overload", "high load", "concurrency", "thread pool", "quota"]):
        return "Capacity / Traffic Spike"
    elif any(k in t for k in ["oom", "out of memory", "memory leak", "heap", "memory limit", "killed process"]):
        return "Memory / OOM / Resource Leak"
    elif any(k in t for k in ["dns", "bgp", "network", "packet", "connection timeout", "tls", "ssl", "handshake", "unreachable", "routing", "latency", "socket"]):
        return "Network / Connectivity"
    elif any(k in t for k in ["config", "misconfigur", "deployment", "rollout", "migration", "flag", "yaml", "parameter", "typo", "bad push", "release"]):
        return "Configuration / Bad Deployment"
    elif any(k in t for k in ["disk", "storage", "filesystem", "corrupt", "iops", "database", "volume", "postgres", "mysql", "replication"]):
        return "Storage / Database / Disk"
    elif any(k in t for k in ["hardware", "power", "datacenter", "kernel panic", "node crash", "reboot", "cpu lockup"]):
        return "Hardware / Infrastructure"
    elif any(k in t for k in ["credential", "certificate expired", "auth", "token", "security", "compromised", "permission", "unauthorized"]):
        return "Security / Auth / Credentials"
    elif any(k in t for k in ["dependency", "third-party", "provider", "upstream", "aws down", "cloudflare", "stripe"]):
        return "Third-Party / Dependency Outage"
    else:
        return "Software Bug / Logic Error"

hits = 0
results = []
for i, r in enumerate(records):
    inc_id = r["incident_id"]
    query_cat = categorize(r["root_cause"] + " " + r.get("raw_log_excerpt", ""))
    match_meta, sim = retrieve(r["raw_log_excerpt"], k=1)
    ret_cat = categorize(match_meta["root_cause"] + " " + match_meta.get("raw_log_excerpt", ""))
    is_hit = (query_cat == ret_cat)
    if is_hit:
        hits += 1
    results.append({
        "incident_id": inc_id,
        "company": r["company"],
        "query_category": query_cat,
        "retrieved_id": match_meta["incident_id"],
        "retrieved_company": match_meta.get("company", "unknown"),
        "retrieved_category": ret_cat,
        "similarity": round(sim, 4),
        "hit": is_hit
    })

print(f"Top-1 Retrieval Category Hit Rate: {hits}/{len(records)} ({hits/len(records)*100:.2f}%)")

cat_counts = {}
for res in results:
    cat_counts[res["query_category"]] = cat_counts.get(res["query_category"], 0) + 1

print("\nCategory breakdown in test set:")
for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
    cat_hits = sum(1 for res in results if res["query_category"] == cat and res["hit"])
    print(f"  - {cat:<32}: {count:>2} incidents (hit rate: {cat_hits}/{count} = {cat_hits/count*100:.1f}%)")

with open("eval/category_benchmark_report.json", "w", encoding="utf-8") as f:
    json.dump({
        "sample_size": len(records),
        "total_hits": hits,
        "hit_rate_pct": round(hits/len(records)*100, 2),
        "breakdown": cat_counts,
        "details": results
    }, f, indent=2)
