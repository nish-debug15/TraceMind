import json
import pickle
import re
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

INDEX_PATH = "clustering/cluster_index.pkl"
TRAIN_PATH = "data/splits/train.json"
RESULTS_PATH = "scripts/name_generation_results.json"
OUTPUT_PATH = "clustering/cluster_profiles.json"

with open(INDEX_PATH, "rb") as f:
    idx = pickle.load(f)

with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train = json.load(f)

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    name_results = json.load(f)

labels = idx["labels"]
centroids = idx["centroids"]
n_clusters = idx["n_clusters"]

from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("all-MiniLM-L6-v2")
texts = []
for r in train:
    log = r.get("raw_log_excerpt", "")[:1000]
    rc = r.get("root_cause", "")[:1000]
    texts.append(f"{log} [SEP] {rc}")
embeddings = st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

# 1. Stopwords definition
COMPANIES_STOPLIST = {
    "github", "sentry", "cloudflare", "twilio", "discord", "shopify", "reddit",
    "atlassian", "datadog", "digitalocean", "circleci", "netlify", "openai",
    "dropbox", "squarespace", "launchdarkly", "google", "aws", "anthropic",
    "slack", "jira", "confluence", "bitbucket", "acuity", "copilot", "chatgpt",
    "travis", "travisci"
}
for item in train:
    c = item.get("company", "").lower()
    if c:
        COMPANIES_STOPLIST.add(c)

BOILERPLATE_STOPLIST = {
    "inconvenience", "apologize", "apologies", "possible", "implement", "implementing",
    "implemented", "solution", "updated", "accessible", "result", "note", "getting",
    "properly", "looking", "level", "working", "expected", "quickly", "shortly",
    "investigating", "actively", "began", "leading", "multiple", "users", "elevated",
    "mitigation", "remediating", "number", "ability", "new", "fix", "com", "platform"
}

SECTION_D_STOPLIST = {
    "identified", "experiencing", "impacted", "update", "soon", "affected", "provide",
    "details", "share", "trouble", "deploying", "create", "led", "days", "hours",
    "kingdom", "pacific", "errors", "services", "performance", "degraded", "delayed"
}

COMBINED_STOPLIST = set(ENGLISH_STOP_WORDS) | COMPANIES_STOPLIST | BOILERPLATE_STOPLIST | SECTION_D_STOPLIST

# 2. Group members by cluster
clusters = {}
cluster_member_indices = {}
for i, l in enumerate(labels):
    if l != -1:
        clusters.setdefault(l, []).append(train[i])
        cluster_member_indices.setdefault(l, []).append(i)

# 3. TF-IDF Keywords
cluster_docs = []
for cid in range(n_clusters):
    m_texts = []
    for m in clusters[cid]:
        rc = m.get("root_cause") or ""
        log = m.get("raw_log_excerpt") or ""
        m_texts.append(f"{rc} {log}")
    cluster_docs.append(" ".join(m_texts))

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words=list(COMBINED_STOPLIST),
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    min_df=1,
    max_df=0.85
)
tfidf_matrix = vectorizer.fit_transform(cluster_docs)
feature_names = vectorizer.get_feature_names_out()

cluster_keywords = {}
for cid in range(n_clusters):
    members = clusters[cid]
    row = tfidf_matrix[cid].toarray()[0]
    sorted_indices = row.argsort()[::-1]
    
    surviving = []
    for idx_term in sorted_indices:
        score = row[idx_term]
        if score <= 0:
            break
        term = feature_names[idx_term]
        tokens = term.split()
        
        if any(t in COMBINED_STOPLIST for t in tokens):
            continue
        if len(tokens) == 2 and tokens[0] == tokens[1]:
            continue
            
        member_count = 0
        pattern = r"\b" + re.escape(tokens[0]) + r"\b" if len(tokens) == 1 else r"\b" + re.escape(tokens[0]) + r"\s+" + re.escape(tokens[1]) + r"\b"
        for m in members:
            text = f"{m.get('root_cause', '')} {m.get('raw_log_excerpt', '')}"
            if re.search(pattern, text, re.IGNORECASE):
                member_count += 1
                
        if member_count >= 2:
            surviving.append(term)
            if len(surviving) == 4:
                break
                
    if len(surviving) < 3:
        cluster_keywords[cid] = []
    else:
        cluster_keywords[cid] = surviving

# 4. Assemble final profiles
profiles = {}

# Noise profile (-1)
noise_count = sum(1 for l in labels if l == -1)
profiles["-1"] = {
    "id": -1,
    "name": "Novel pattern",
    "description": "No recurring failure signature in the historical corpus. Treat the match as a hypothesis.",
    "size": noise_count,
    "confidence": "ok",
    "source": {
        "top_company": "various",
        "top_share": 0.0,
        "company_count": 0
    },
    "keywords": [],
    "typical_fixes": [],
    "examples": []
}

for cid in range(n_clusters):
    members = clusters[cid]
    indices = cluster_member_indices[cid]
    size = len(members)
    
    # Source & confidence
    co_counts = Counter(m.get("company", "unknown") for m in members)
    top_co, top_cnt = co_counts.most_common(1)[0]
    top_share = round(top_cnt / size, 2)
    company_count = len(co_counts)
    confidence = "low" if size < 5 else "ok"
    source = {
        "top_company": top_co,
        "top_share": top_share,
        "company_count": company_count
    }
    
    # Generated name & description
    gen_info = name_results.get(str(cid), {})
    name = gen_info.get("new_name", f"Cluster {cid}")
    desc = gen_info.get("description", "")
    
    # Representative examples: 3 closest to centroid
    centroid = centroids[cid]
    member_sims = []
    for idx_in_train, m in zip(indices, members):
        emb = embeddings[idx_in_train]
        sim = float(np.dot(centroid, emb))
        member_sims.append((sim, m))
    member_sims.sort(key=lambda x: x[0], reverse=True)
    
    examples = []
    for _, m in member_sims[:3]:
        rc_line = (m.get("root_cause") or "").replace("\n", " ").strip()
        if len(rc_line) > 140:
            rc_line = rc_line[:137] + "..."
        examples.append({
            "company": m.get("company", ""),
            "root_cause": rc_line,
            "source_url": m.get("source_url", "")
        })
        
    # Typical fixes: extract recurring phrases or actions from members
    fixes = []
    for m in members:
        rem = m.get("remediation_steps", [])
        if isinstance(rem, list):
            for step in rem:
                step_str = step.strip().rstrip(".")
                if step_str and len(step_str) < 90 and step_str not in fixes:
                    fixes.append(step_str)
                    if len(fixes) == 3:
                        break
        if len(fixes) == 3:
            break
            
    profiles[str(cid)] = {
        "id": cid,
        "name": name,
        "description": desc,
        "size": size,
        "confidence": confidence,
        "source": source,
        "keywords": cluster_keywords[cid],
        "typical_fixes": fixes,
        "examples": examples
    }

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=2)

print(f"Successfully generated {len(profiles)} cluster profiles (including noise -1) to {OUTPUT_PATH}.")
