---
title: Tracemind Backend
emoji: 🧠
colorFrom: red
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# TraceMind — RAG-Powered Root Cause Analysis for SRE Logs

An automated NLP pipeline that clusters log anomalies and uses Retrieval-Augmented Generation (RAG) to match current infrastructure failures against historical postmortems — generating plain-English root cause summaries and remediation steps, with full pipeline transparency so the reasoning can be audited, not just trusted.

## Problem Statement

When critical infrastructure fails, Site Reliability Engineers (SREs) lose valuable time manually correlating unstructured server logs with past incident reports. This project automates that correlation: given a raw log dump, the system detects the anomaly, retrieves the most similar past incident, and generates an actionable root cause analysis.

## Architecture

```
Raw Log Input
     │
     ▼
Embedding Layer (sentence-transformers)
     │
     ▼
Anomaly Clustering (HDBSCAN)  ──►  groups/dedupes incoming log chunks
     │                              before retrieval — reduces redundant
     │                              queries and groups near-duplicate
     │                              incidents rather than acting as the
     │                              standalone headline result
     ▼
Vector Retrieval (ChromaDB) ──► Historical Postmortem Corpus
     │
     ▼
RAG-Augmented Prompt Construction
     │
     ▼
LLM Generation (Groq / Llama 3.3 70B Versatile, prompted — no fine-tuning)
     │
     ▼
Root Cause Summary + Remediation Steps
     │
     ▼
FastAPI Backend Layer
     │
     ▼
Next.js Frontend (Vercel)
```

**Why clustering *and* retrieval:** clustering is a preprocessing/dedup step — it groups near-duplicate log anomalies so the retrieval stage isn't re-querying against the same failure signature repeatedly. Retrieval against the postmortem corpus is the actual matching mechanism. This is stated explicitly to avoid the two stages reading as redundant.

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | sentence-transformers |
| Clustering | HDBSCAN |
| Vector Store | ChromaDB |
| Generation | Groq / Llama-3.3-70b-versatile (prompted, not fine-tuned) |
| Backend API | FastAPI (hosted on Render/Railway/HF Spaces — needs real compute, not serverless) |
| Frontend | Next.js, deployed on Vercel |
| Evaluation | ROUGE, BERTScore, ablation, human "Actionability" rating |

**Note on hosting split:** the ML pipeline (embeddings, clustering, vector search, LLM inference) runs on a compute-backed host, exposed via FastAPI. Vercel serves the frontend only — its serverless functions aren't suited to long-running inference workloads. Frontend calls the backend over REST.

## Dataset

The corpus consists of **828 curated incident postmortems** standardized into a unified schema, sourced aggregator-first across:

- **danluu/post-mortems** — master curated list spanning Google, AWS, Cloudflare, GitLab, Slack, Heroku, Stripe, and others
- **Nat Welch's parsed postmortems** — pre-structured subset evaluated for direct reuse
- **awesome-tech-postmortems** — supplementary curated list with non-overlapping entries
- **SRE Weekly archive** — years of back-issues and outage retrospectives
- **Statuspage.io-hosted incident histories** — uniform scraper covering 30–40+ domains
- Cloudflare Blog, GCP incident history, AWS post-event summaries

### Train / Test Split (Zero-Leakage Guarantee)
To guarantee valid quantitative evaluation and eliminate data leakage, the corpus is split 80/20 into stratified splits based on postmortem quality:
- **Train split (`data/train.json`):** 662 records (435 `quality:high`, 227 `quality:low`). This forms the exclusive basis for the ChromaDB retrieval index and HDBSCAN clustering centroids.
- **Held-out test split (`data/test.json`):** 166 records (109 `quality:high`, 57 `quality:low`). Test incidents are strictly excluded from the vector database.

### Schema
```json
{
  "incident_id": "string",
  "raw_log_excerpt": "string",
  "root_cause": "string",
  "remediation_steps": "string",
  "source_url": "string",
  "quality": "high | low"
}
```

## Clustering & Retrieval Specifics

- **Embeddings:** `all-MiniLM-L6-v2` (384-dimensional normalized embeddings).
- **Anomaly Clustering (HDBSCAN):** Fitted on the train corpus with `min_cluster_size=3` and `min_samples=2`, producing 40 distinct failure clusters (55.4% noise rate, consistent with the corpus baseline quality distribution).
- **2-Pass Vector Retrieval:** Queries ChromaDB for `quality:high` incidents first with a `0.40` cosine similarity floor. If no match exceeds this floor, it falls back to all records.
- **LLM Generation:** `llama-3.3-70b-versatile` via Groq Cloud API (OpenAI-compatible chat completions). Transparent mock fallback (`is_mock: true`) is provided for offline development when `GROQ_API_KEY` is unset.
- **Hugging Face Space Secret:** In your Space settings under *Variables and secrets*, add `GROQ_API_KEY` as a secret.

## Pipeline Trace (Key Feature)

The editorial Next.js frontend includes a "Show Pipeline Trace" toggle exposing:
- The matched historical incident with a clickable `source_url`
- The computed cosine similarity score
- The exact prompt payload delivered to the LLM
- An unmistakable warning badge whenever mock generation is active

## Evaluation Suite (`eval/evaluate.py`)

- **Leakage Prevention:** Programmatically asserts 0 overlapping incident IDs between `data/test.json` and ChromaDB.
- **Ablation Study (Primary Quantitative Metric):** Direct comparison of **RAG (TraceMind)** against a **Zero-Shot (No-RAG)** baseline across the held-out test split.
- **Quality-Stratified Reporting:** Metrics are disaggregated for the Full Set ($N=166$), High-Quality incidents ($N=109$), and Low-Quality incidents ($N=57$).
- **Metrics:** ROUGE-1, ROUGE-2, ROUGE-L, and Semantic Embedding Similarity (with optional `--use-bertscore`).
- **Disk Caching:** All model completions are cached to `rag/gen_cache.py` (and `eval/cache/`) to survive API quotas and network interruptions.
- **Mock Prohibition:** The evaluation script fails hard if `GROQ_API_KEY` is missing or if any mock completion is returned.

**Note on Test Coverage:** Automated test coverage exists for the clustering module (`tests/test_clustering.py`) and backend API (`tests/test_backend.py`) (19/19 passing). Data scrapers currently do not have automated unit tests.

## 12-Week Roadmap Status

| Weeks | Milestone | Status |
|---|---|---|
| 1–2 | Aggregator-first data sourcing + schema standardization (828 records) | Complete |
| 3–4 | Embedding + HDBSCAN clustering (40 clusters, noise correlation validated) | Complete |
| 5–6 | RAG layer: ChromaDB vector storage + 2-pass quality retrieval | Complete |
| 7–8 | LLM integration (Groq Llama 3.3 70B via API + mock transparency) | Complete |
| 9–10 | Pipeline integration + FastAPI backend + Next.js editorial UI | Complete |
| 11–12 | Evaluation suite, ablation benchmark, cloud deployment | In Progress |

## Setup

### Backend
```bash
git clone https://github.com/nish-debug15/TraceMind.git
cd TraceMind
pip install -r requirements.txt
# Copy sample env and add GROQ_API_KEY
cp .env.example .env
# Run from repository root so sibling packages resolve cleanly:
uvicorn backend.main:app --reload
```

### Frontend
```bash
cd TraceMind/frontend
npm install
npm run dev
```

### Running Evaluation
```bash
export GROQ_API_KEY="your_groq_api_key"
python eval/evaluate.py --limit 20   # Run smoke test on 20 held-out samples
python eval/evaluate.py              # Full ablation on all 166 test incidents
```

## License

TBD
