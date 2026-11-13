"""
scripts/build_cluster_profiles.py
---------------------------------
One-time offline script to build cluster profiles for TraceMind.
Builds and uses HDBSCAN index over the 662 training incidents (data/splits/train.json).
Outputs clustering/cluster_profiles.json.
"""

import os
import re
import json
import time
import pickle
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INDEX_PATH = PROJECT_ROOT / "clustering" / "cluster_index.pkl"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.json"
OUTPUT_PATH = PROJECT_ROOT / "clustering" / "cluster_profiles.json"

CLUSTER_METADATA = {
    0: {
        "name": "CI build runner provisioning failures",
        "specific_desc": "Virtual machine cleanup and build runner initialization failures in CI infrastructure.",
        "broad_summary": "build runner initialization and virtual machine cleanup failures",
        "fixes": ["Restart worker supervisor daemon", "Clean up orphaned VM instances", "Roll back runner version"]
    },
    1: {
        "name": "Droplet regional service disruption",
        "specific_desc": "Regional compute instance action failures and block storage attachment degradation.",
        "broad_summary": "Droplet action failures and block storage volume attachment degradation across regions",
        "fixes": ["Restore regional storage controllers", "Rebalance hypervisor capacity", "Restart compute control plane"]
    },
    2: {
        "name": "DNS and certificate provisioning failures",
        "specific_desc": "DNS API service unavailability and automated TLS certificate issuance failures.",
        "broad_summary": "DNS API outages and automated TLS certificate provisioning degradation",
        "fixes": ["Failover to secondary DNS API", "Renew certificates manually", "Add retry backoff for certificate authority"]
    },
    3: {
        "name": "Mac job start failures",
        "specific_desc": "Network routing instability and container initialization failures for macOS build runners.",
        "broad_summary": "macOS job startup errors and network routing failures in hosted Mac runner pools",
        "fixes": ["Reroute runner network traffic", "Drain unhealthy Mac hosts", "Increase job startup timeout"]
    },
    4: {
        "name": "Search and chat connector outages",
        "specific_desc": "Integration failures in enterprise search indexes and third-party cloud drive connectors.",
        "broad_summary": "third-party connector timeouts and conversational search indexing disruptions",
        "fixes": ["Re-authenticate third-party OAuth tokens", "Purge invalid connector cache", "Restart search indexing worker"]
    },
    5: {
        "name": "AI agent runner execution failures",
        "specific_desc": "Autonomous agent execution errors and attachment upload failures during runtime processing.",
        "broad_summary": "AI agent runner crashes and attachment upload failures during model execution",
        "fixes": ["Scale runner container pool", "Increase upload payload size limits", "Retry transient agent tasks"]
    },
    6: {
        "name": "CI workflow insights data delays",
        "specific_desc": "Lag in reporting pipeline telemetry and workflow metrics aggregation.",
        "broad_summary": "telemetry aggregation lag and delayed workflow analytics reporting",
        "fixes": ["Increase pipeline consumer concurrency", "Backfill missing analytics records", "Scale reporting datastore"]
    },
    7: {
        "name": "Usage API data delay",
        "specific_desc": "Delayed metric ingestion causing lag in customer usage and billing API responses.",
        "broad_summary": "delays in customer usage calculations and delayed billing API data delivery",
        "fixes": ["Flush usage calculation queues", "Scale event processing workers", "Add query caching for billing queries"]
    },
    8: {
        "name": "Appointment scheduling service disruption",
        "specific_desc": "Online appointment scheduling service downtime and booking database query degradation.",
        "broad_summary": "online scheduling unavailability and calendar sync processing disruptions",
        "fixes": ["Failover calendar database", "Restart scheduling API pods", "Clear cached availability slots"]
    },
    9: {
        "name": "CI/CD build queue datastore degradation",
        "specific_desc": "Build queue backlog caused by slow queries and replication lag in backing replica sets.",
        "broad_summary": "build queue delays caused by datastore query contention and replica degradation",
        "fixes": ["Add indexes on build queue collections", "Failover to healthy replica member", "Throttle concurrent job enqueueing"]
    },
    10: {
        "name": "API timeouts and performance degradation",
        "specific_desc": "Elevated HTTP API error rates and request latency under high ingestion traffic.",
        "broad_summary": "elevated HTTP 5xx responses and API request timeouts during traffic spikes",
        "fixes": ["Scale API gateway instances", "Increase upstream connection timeouts", "Enable rate limiting on abusive endpoints"]
    },
    11: {
        "name": "Serverless inference service errors",
        "specific_desc": "Model execution timeouts and invocation failures in managed serverless AI pipelines.",
        "broad_summary": "serverless AI model inference timeouts and upstream model provider errors",
        "fixes": ["Reroute requests to secondary model region", "Increase inference execution timeout", "Apply exponential retry backoff"]
    },
    12: {
        "name": "Datacenter power and generator failure",
        "specific_desc": "Utility power loss and backup generator failover failure triggering cascading storage and compute outages.",
        "broad_summary": "utility power loss and backup generator depletion causing cascading infrastructure outages",
        "fixes": ["Transfer load to alternate availability zone", "Restart storage cluster nodes", "Perform file system consistency checks"]
    },
    13: {
        "name": "Datacenter power loss and network partition",
        "specific_desc": "Cooling plant failure and power disruption causing datacenter network isolation and fiber link drops.",
        "broad_summary": "cooling system and power disruptions leading to fiber link isolation and network partitioning",
        "fixes": ["Drain affected compute zones", "Reroute WAN fiber traffic", "Restore auxiliary cooling systems"]
    },
    14: {
        "name": "Authoritative DNS and traffic routing outage",
        "specific_desc": "Widespread edge network routing failures and authoritative DNS resolution drops.",
        "broad_summary": "authoritative DNS resolution degradation and edge traffic routing failures",
        "fixes": ["Withdraw poisoned BGP routes", "Roll back edge configuration update", "Reroute traffic around impacted data centers"]
    },
    15: {
        "name": "Managed database connection failures",
        "specific_desc": "Intermittent network timeouts and connection drops to hosted PostgreSQL and MongoDB clusters.",
        "broad_summary": "intermittent connection drops and DNS timeouts to managed database clusters",
        "fixes": ["Increase client connection timeout", "Restart proxy connection poolers", "Failover to standby database replica"]
    },
    16: {
        "name": "Cloud identity and access management downtime",
        "specific_desc": "Single sign-on authentication failures and user management directory outages.",
        "broad_summary": "authentication service disruptions and directory synchronization failures",
        "fixes": ["Bypass degraded SSO provider", "Purge corrupted token caches", "Scale authentication backend pods"]
    },
    17: {
        "name": "Cloud file synchronization and sharing outage",
        "specific_desc": "File sharing link generation errors and block storage synchronization delays.",
        "broad_summary": "file sharing link degradation and client file synchronization service disruptions",
        "fixes": ["Restart metadata sync workers", "Scale block storage endpoints", "Invalidate cached sharing permissions"]
    },
    18: {
        "name": "Multi-product telemetry latency",
        "specific_desc": "End-to-end telemetry ingestion delays affecting real-user monitoring, APM, and log search.",
        "broad_summary": "cross-product telemetry delays affecting RUM, APM traces, and log processing pipelines",
        "fixes": ["Scale streaming ingestion buffers", "Prioritize real-time alert pipelines", "Backfill degraded dashboard views"]
    },
    19: {
        "name": "Edge Function and CDN network errors",
        "specific_desc": "Serverless edge compute invocation failures and CDN origin error spikes.",
        "broad_summary": "serverless edge function runtime errors and CDN origin gateway timeouts",
        "fixes": ["Roll back edge function deployment", "Bypass caching edge layer", "Increase edge runtime memory limit"]
    },
    20: {
        "name": "Carrier SMS delivery degradation",
        "specific_desc": "SMS message transmission delays and routing failures across global telecommunication carrier networks.",
        "broad_summary": "SMS delivery delays and carrier network routing failures across long codes",
        "fixes": ["Reroute traffic through alternate carrier partner", "Queue messages with exponential retry", "Throttle traffic to congested routes"]
    },
    21: {
        "name": "Carrier identity and authentication failure",
        "specific_desc": "Mobile subscriber identity authentication failures and missing SNA network redirects in regional carrier gateways.",
        "broad_summary": "mobile subscriber identity verification drops and gateway API failures in the UK",
        "fixes": ["Failover to SMS OTP verification fallback", "Update carrier gateway API endpoints", "Reset SNA redirect handlers"]
    },
    22: {
        "name": "Site build and deployment latency",
        "specific_desc": "Build system queue backlog causing elevated latency from commit to edge propagation.",
        "broad_summary": "static site deployment pipeline latency and build container creation delays",
        "fixes": ["Scale build scheduling daemon", "Clear hung build containers", "Prioritize production branch deployments"]
    },
    23: {
        "name": "Monitor evaluation and metrics backlog",
        "specific_desc": "Delayed metric ingestion causing evaluation lag for automated alert monitors and synthetic tests.",
        "broad_summary": "alert monitor evaluation delays and metric ingestion backlog during peak load",
        "fixes": ["Increase Kafka partition consumers", "Scale metric aggregation cluster", "Pause non-critical scheduled monitors"]
    },
    24: {
        "name": "Telemetry spans and metrics ingestion failure",
        "specific_desc": "Distributed trace spans and metric data dropped due to ingestion pipeline buffer saturation.",
        "broad_summary": "distributed tracing span processing failures and regional metric ingestion lag",
        "fixes": ["Scale trace ingestion cluster", "Increase buffer memory on pipeline relays", "Drop low-priority sample traces"]
    },
    25: {
        "name": "Event ingestion latency and backlog",
        "specific_desc": "Backlog in asynchronous event ingestion pipeline causing delayed alert notifications.",
        "broad_summary": "event ingestion latency and queue backlog processing across regional clusters",
        "fixes": ["Add consumer partitions to event queue", "Restart stalled event workers", "Optimize indexing throughput"]
    },
    26: {
        "name": "Alerting notification and ingestion delay",
        "specific_desc": "Delays between threshold breach detection and outbound notification webhook delivery.",
        "broad_summary": "notification webhook delays and alert state transition backlog",
        "fixes": ["Scale notification dispatcher workers", "Retry failed webhook payloads", "Drain notification queue backlog"]
    },
    27: {
        "name": "Static site build failures",
        "specific_desc": "Build container execution timeouts and failure to compile static web assets.",
        "broad_summary": "build container failure spikes and asset compilation timeouts",
        "fixes": ["Roll back build image update", "Increase container memory limits", "Clear corrupted build cache"]
    },
    28: {
        "name": "Web performance degradation and gateway timeouts",
        "specific_desc": "Elevated HTTP 504 gateway timeout rates and slow client response times under traffic burst.",
        "broad_summary": "elevated HTTP 504 gateway timeouts and frontend web performance degradation",
        "fixes": ["Scale upstream web application pods", "Increase reverse proxy keep-alive timeouts", "Enable static asset caching at CDN"]
    },
    29: {
        "name": "Web UI and dashboard loading failures",
        "specific_desc": "Client dashboard frontend bundle load failures and single-page application rendering timeouts.",
        "broad_summary": "web console single-page application loading errors and asset delivery timeouts",
        "fixes": ["Invalidate cached JavaScript bundles", "Roll back frontend console deployment", "Scale static asset storage origins"]
    },
    30: {
        "name": "Comment processing and vote counting lag",
        "specific_desc": "Asynchronous database write queue backlog causing delayed comment rendering and vote aggregation.",
        "broad_summary": "delayed comment submission processing and vote tally indexing lag",
        "fixes": ["Scale async write queue workers", "Throttle non-essential batch recalculations", "Partition comment datastore"]
    },
    31: {
        "name": "Elevated web and API error rates",
        "specific_desc": "Elevated HTTP 5xx server error rate and sporadic request drops across public web and API endpoints.",
        "broad_summary": "elevated HTTP 5xx error rates and sporadic request failures across public web endpoints",
        "fixes": ["Restart degraded backend application servers", "Apply emergency rate limiting", "Roll back offending API release"]
    },
    32: {
        "name": "Website connectivity and 503 errors",
        "specific_desc": "Hosted website unavailability with HTTP 503 Service Unavailable errors returned by edge proxies.",
        "broad_summary": "HTTP 503 service unavailable errors and edge proxy connectivity drops for hosted sites",
        "fixes": ["Restart edge proxy instances", "Verify upstream origin DNS routing", "Scale origin web server clusters"]
    },
    33: {
        "name": "Site loading and domain timeouts",
        "specific_desc": "DNS lookup delays and connection timeouts when resolving custom domain names.",
        "broad_summary": "custom domain resolution timeouts and slow initial site loading times",
        "fixes": ["Purge stale DNS resolver cache", "Re-sync custom domain routing table", "Increase origin connection timeout"]
    },
    34: {
        "name": "AI model provider and Copilot disruptions",
        "specific_desc": "Third-party LLM inference provider outages causing inline code completion failures.",
        "broad_summary": "third-party AI model provider timeouts and degraded code completion availability",
        "fixes": ["Failover to backup model provider", "Implement client-side model timeout fallback", "Scale model proxy gateway"]
    },
    35: {
        "name": "Conversational AI generation timeouts",
        "specific_desc": "Elevated request drop rates and generation timeouts in conversational AI assistant sessions.",
        "broad_summary": "conversational chat generation timeouts and response streaming interruptions",
        "fixes": ["Scale inference model replica pods", "Throttle background context recalculations", "Apply graceful degradation on model context window"]
    },
    36: {
        "name": "AI model API elevated error rates",
        "specific_desc": "Public model completion API returning elevated HTTP 500 and 503 error rates during load spikes.",
        "broad_summary": "elevated HTTP 5xx responses and rate-limit drops on public model completion endpoints",
        "fixes": ["Scale inference cluster capacity", "Shed non-interactive batch inference requests", "Restart failed GPU worker nodes"]
    },
    37: {
        "name": "Database connection pool exhaustion and failover",
        "specific_desc": "Primary relational database CPU saturation causing connection pool exhaustion and unplanned replica failover.",
        "broad_summary": "database CPU spikes, replication lag, and connection pool exhaustion",
        "fixes": ["Terminate long-running blocking transactions", "Scale connection pooler max connections", "Promote read replica to primary"]
    },
    38: {
        "name": "CI workflow execution and runner queue delays",
        "specific_desc": "Workflow job queuing delays and webhook dispatch failures in cloud runner infrastructure.",
        "broad_summary": "CI workflow queue delays and hosted runner execution timeouts",
        "fixes": ["Scale hosted runner pool", "Retry hung workflow jobs", "Increase webhook delivery queue workers"]
    },
}

def extract_root_cause_line(text: str, max_len: int = 150) -> str:
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    line = lines[0]
    line = re.sub(r"^(\*+|\#+|-|\d+\.)\s*", "", line).strip()
    line = line.replace("**", "").replace("*", "").replace("`", "")
    if re.match(r"^root cause:?$", line, re.IGNORECASE) and len(lines) > 1:
        line = re.sub(r"^(\*+|\#+|-|\d+\.)\s*", "", lines[1]).strip()
        line = line.replace("**", "").replace("*", "").replace("`", "")
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) > max_len:
        line = line[:max_len - 3].rstrip() + "..."
    return line


def build():
    with open(INDEX_PATH, "rb") as f:
        idx = pickle.load(f)
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train = json.load(f)

    labels = idx["labels"]
    centroids = idx["centroids"]
    cluster_ids = idx["cluster_ids"]
    n_clusters = idx["n_clusters"]

    companies_in_data = set(r.get("company", "").lower().strip() for r in train if r.get("company"))
    extra_companies = {
        "github", "sentry", "cloudflare", "twilio", "discord", "shopify", "reddit",
        "atlassian", "datadog", "digitalocean", "circleci", "netlify", "openai",
        "dropbox", "squarespace", "launchdarkly", "google", "aws", "anthropic",
        "slack", "jira", "confluence", "bitbucket", "acuity", "copilot", "chatgpt"
    }

    boilerplate = {
        "inconvenience", "apologize", "apologies", "possible", "implement",
        "implementing", "implemented", "solution", "updated", "accessible",
        "result", "note", "getting", "properly", "looking", "level", "working",
        "expected", "quickly", "shortly", "investigating", "actively", "began",
        "leading", "multiple", "users", "elevated", "mitigation", "remediating",
        "number", "ability", "new", "fix", "com", "platform",
        "team", "engineering team", "engineering", "service", "currently", "issues",
        "issue", "experienced", "problem", "incident", "incidents", "status", "page",
        "resolved", "pdt", "pst", "utc", "gmt", "est", "edt", "cst", "cdt",
        "october", "november", "december", "january", "february", "march",
        "april", "may", "june", "july", "august", "september"
    }

    stop_list = list(set(ENGLISH_STOP_WORDS) | companies_in_data | extra_companies | boilerplate)

    clusters = {}
    noise_members = []
    for i, l in enumerate(labels):
        rec = train[i]
        if l == -1:
            noise_members.append(rec)
        else:
            clusters.setdefault(l, []).append(rec)

    cluster_corpus = []
    for cid in range(n_clusters):
        members = clusters[cid]
        text = " ".join(
            (m.get("root_cause") or "") + " " + (m.get("raw_log_excerpt") or "")
            for m in members
        )
        cluster_corpus.append(text)

    vec = TfidfVectorizer(
        stop_words=stop_list,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        ngram_range=(1, 2),
        max_df=0.80,
        min_df=1,
    )
    tfidf = vec.fit_transform(cluster_corpus)
    feature_names = vec.get_feature_names_out()

    cluster_keywords = {}
    for cid in range(n_clusters):
        row = tfidf.getrow(cid).toarray()[0]
        top_indices = row.argsort()[::-1][:5]
        top_terms = [feature_names[i] for i in top_indices if row[i] > 0]
        cluster_keywords[cid] = top_terms

    model = SentenceTransformer("all-MiniLM-L6-v2")
    cluster_examples = {}
    for cid in range(n_clusters):
        members = clusters[cid]
        c_idx = cluster_ids.index(cid)
        centroid = centroids[c_idx]

        member_texts = [
            f"{(m.get('raw_log_excerpt') or '')[:1000]} [SEP] {(m.get('root_cause') or '')[:1000]}"
            for m in members
        ]
        embs = model.encode(member_texts, normalize_embeddings=True, show_progress_bar=False)
        sims = embs @ centroid
        top_indices = np.argsort(sims)[::-1][:3]
        examples = []
        for idx_pos in top_indices:
            m = members[idx_pos]
            examples.append({
                "company": m.get("company", "Unknown"),
                "root_cause_line": extract_root_cause_line(m.get("root_cause") or ""),
                "source_url": m.get("source_url") or "",
            })
        cluster_examples[cid] = examples

    profiles = {}
    for cid in range(n_clusters):
        members = clusters[cid]
        size = len(members)
        comps = Counter(m.get("company", "Unknown") for m in members)
        top_co, count = comps.most_common(1)[0]
        pct = (count / size) * 100
        top_comps = [c for c, _ in comps.most_common(3)]

        is_broad = size >= 25 or pct > 60.0
        kind = "broad" if is_broad else "specific"
        conf = "low" if size < 5 else "ok"

        meta = CLUSTER_METADATA[cid]
        name = meta["name"]
        fixes = meta["fixes"]

        if kind == "broad":
            desc = f"A broad group of status incidents mostly from {top_co} involving {meta['broad_summary']}."
        else:
            desc = meta["specific_desc"]

        profiles[str(cid)] = {
            "id": int(cid),
            "name": name,
            "description": desc,
            "kind": kind,
            "confidence": conf,
            "size": int(size),
            "company_count": int(len(comps)),
            "top_companies": top_comps,
            "top_company_share": round(pct, 1),
            "keywords": cluster_keywords[cid],
            "typical_fixes": fixes,
            "examples": cluster_examples[cid],
        }

    noise_comps = Counter(m.get("company", "Unknown") for m in noise_members)
    top_noise_co, noise_count = noise_comps.most_common(1)[0]
    noise_share = round((noise_count / len(noise_members)) * 100, 1)

    profiles["-1"] = {
        "id": -1,
        "name": "Novel pattern",
        "description": "No recurring failure signature in the historical corpus. Treat the match as a hypothesis.",
        "kind": "specific",
        "confidence": "ok",
        "size": int(len(noise_members)),
        "company_count": int(len(noise_comps)),
        "top_companies": [c for c, _ in noise_comps.most_common(3)],
        "top_company_share": noise_share,
        "keywords": [],
        "typical_fixes": [],
        "examples": [],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print(f"Saved {len(profiles)} cluster profiles to {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
