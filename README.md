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

An automated NLP pipeline that clusters log anomalies and uses Retrieval-Augmented Generation (RAG) to match incoming infrastructure failures against historical incident postmortems. TraceMind synthesizes root cause analyses and prioritized remediation checklists with full pipeline transparency, enabling Site Reliability Engineers (SREs) to audit the reasoning rather than trusting a black box.

---

## Status

**Backend verified on Hugging Face Spaces (`https://nishhhx0-tracemind-backend.hf.space`), frontend pending Vercel deploy.**

- **ML Inference & Vector Retrieval:** Active on Hugging Face Spaces via FastAPI mounted on Gradio.
- **LLM Provider:** `openai/gpt-oss-120b` via Groq Cloud API with client-side rate limiting and automatic caching.
- **Frontend Dashboard:** Dark/light Bento console built with Next.js 16, Tailwind CSS v4, and shadcn/ui.
- **Verification Status:** 30/30 unit and integration tests passing (`python -m pytest`).

---

## Architecture

```
Raw Server Log Input
         │
         ▼
Embedding Layer (sentence-transformers / all-MiniLM-L6-v2, 384-d)
         │
         ▼
Anomaly Clustering (HDBSCAN)  ──►  Groups similar incident reports; the group label is shown in the Pipeline Trace and does not change retrieval (39 clusters)
         │
         ▼
2-Pass Vector Retrieval (ChromaDB)
  ├── Pass 1: Filter quality:high postmortems (0.40 cosine similarity floor)
  └── Pass 2: Fallback across full postmortem corpus if Pass 1 misses
         │
         ▼
RAG-Augmented Prompt Construction (relevance_gate_v2)
  └── Enforces strict precedent relevance gating ("Precedent: relevant | not relevant")
         │
         ▼
LLM Generation (openai/gpt-oss-120b via Groq Cloud API)
         │
         ▼
Structured RCA Output
  ├── Precedent Relevance Badge ("Precedent applied" vs. "No relevant precedent")
  ├── Root Cause Synthesis (Markdown)
  ├── 5-Step Actionable Remediation Checklist
  └── Pipeline Trace (Telemetry, Cosine Similarity Gauge, Cluster ID, Raw Prompt)
```

---

## Tech Stack

| Component | Technology | Role / Specification |
|---|---|---|
| **Embeddings** | `sentence-transformers` | `all-MiniLM-L6-v2` (384-dimensional normalized dense vectors) |
| **Clustering** | `HDBSCAN` | 39 clusters + 338 noise on train corpus (`min_cluster_size=3`, `min_samples=2`), observational only |
| **Vector Store** | `ChromaDB` | Persistent vector index with 2-pass quality-biased retrieval and 0.40 cosine floor |
| **LLM Generation** | Groq Cloud API | `openai/gpt-oss-120b` (prompted inference, zero fine-tuning, `relevance_gate_v2`) |
| **Backend API** | `FastAPI` / `Uvicorn` | REST API (`/analyze`, `/health`) with per-IP rate limiting and proxy routing (Hosted on Hugging Face Spaces) |
| **Frontend UI** | `Next.js 16` (Turbopack) | Dark/light Bento dashboard, Tailwind CSS v4, shadcn/ui (Hosted on Vercel) |
| **Evaluation** | Python / Pytest | ROUGE (1, 2, L), BERTScore (F1), Embedding Cosine Similarity, Bootstrap 95% CIs |

---

## Dataset & Zero-Leakage Guarantee

The postmortem corpus contains **828 curated incident reports** standardized into a unified schema, sourced aggregator-first across:

- **`icco/postmortems`**: Curated tech industry postmortem collection.
- **Statuspage.io incident histories**: Structured scrape across 20 SaaS and infrastructure providers (Datadog, Reddit, Discord, Twilio, Sentry, CircleCI, DigitalOcean, GitHub, Dropbox, Squarespace, and others).
- **Google Cloud Platform (GCP)**: Official incident retrospectives and outage disclosures.
- **Cloudflare Blog**: Detailed technical postmortems.

### Postmortem Schema

```json
{
  "incident_id": "string",
  "company": "string",
  "raw_log_excerpt": "string",
  "root_cause": "string",
  "remediation_steps": "string",
  "source_url": "string",
  "quality": "high | low"
}
```

*Corpus Quality Breakdown:* **544 High Quality** / **284 Low Quality** (828 total).

### Company-Grouped Split (Zero Data Leakage)

To eliminate data leakage and validate real-world out-of-distribution generalization, the corpus is split **strictly by company** using fixed seed 42 (`scripts/make_split.py`):

- **Train Split (`data/train.json`):** 662 incidents across 18 companies. Forms the exclusive corpus for HDBSCAN clustering centroids and the ChromaDB retrieval index (`rag/chroma_db`).
- **Held-Out Test Split (`data/test.json`):** 93 incidents across 40 companies. **100% of test records are strictly `quality: high`**.
- **Excluded Contamination Filter:** 73 low-quality incidents from the 40 test companies are excluded from both splits to prevent quality and company leakage.
- **Zero Company Overlap:** No company in the test set exists in the training set (`set(train_companies) & set(test_companies) == set()`).
- **Maximum Cross-Split Similarity:** **0.8438** between test and train `root_cause` embeddings (strictly below the 0.95 near-duplicate threshold).

---

## Retrieval & Pipeline Trace

TraceMind is designed for auditability. Rather than producing opaque predictions, the console surfaces an interactive **Pipeline Trace**:

- **Matched Historical Precedent:** The nearest past incident with title, company, and direct hyperlink to the official public postmortem.
- **Similarity Gauge:** Semicircular SVG gauge displaying the cosine similarity score with an explicit **0.40 threshold marker** (scores ≥ 0.60 labeled *Strong match*, 0.40–0.59 labeled *Moderate match*, < 0.40 labeled *Weak match, treat as hypothesis*).
- **Relevance Gate Badge:** Real-time badge indicating whether the historical match was applied (`Precedent applied`) or rejected as irrelevant (`No relevant precedent`).
- **Raw Prompt Inspection:** Tabbed view displaying the exact system prompt, user prompt, and model parameters delivered to the LLM.

![Pipeline Trace Demo](docs/pipeline_trace_demo.png)

---

## Quantitative Evaluation & Benchmark Results

Comprehensive evaluation results, prompt robustness ablation, bootstrap confidence intervals, and failure analyses are generated directly from committed benchmark outputs: see `eval/results/`.

---

## Known Limitations

1. **Vendor Style Concentration in Clusters**: 89.7% of clusters (35 of 39) have ≥80% of members belonging to a single company. Clustering primarily captures vendor postmortem writing style and domain terminology rather than platform-agnostic failure mechanisms.
2. **Re-Clustering Grid Invariance**: An extensive 24-run HDBSCAN hyperparameter search across 3 text variants (raw concatenation, quality:high root cause only, and company/boilerplate stripped) found no configuration capable of achieving both ≥70% multi-company representation (≥3 companies per cluster) and acceptable noise (≤45%).
3. **Index vs. Runtime Text Asymmetry**: Centroids in `clustering/cluster_index.pkl` were generated from concatenated incident logs and retrospective root-cause analyses (`raw_log_excerpt [SEP] root_cause`), whereas runtime queries (`/analyze`) embed only incoming raw log text.
4. **Observational Clustering Role**: HDBSCAN cluster assignment is purely observational in the Pipeline Trace for SRE visibility. It does not gate, filter, or alter ChromaDB vector retrieval, LLM prompts, or synthesis.

---

## Security & Production Hardening

- **Locked CORS Configuration:** Cross-Origin Resource Sharing is strictly constrained to `http://localhost:3000`, `http://localhost:3001`, and the designated production domain (`https://tracemind.vercel.app`). Wildcard matching (`*.vercel.app`) is disabled to prevent unauthorized API key consumption.
- **Client IP Rate Limiting:** In-memory sliding window rate limiter enforcing a strict maximum of **15 analysis requests per minute per IP** (`IP_RATE_LIMIT_RPM`). Requests exceeding this limit receive `HTTP 429 Too Many Requests`.
- **Pre-Warmed Generation Cache:** All frontend example presets are pre-warmed in SQLite (`scripts/prewarm_cache.py`). Preset clicks return cached completions instantaneously without consuming Groq API tokens or triggering 429/502 errors.
- **Explicit API Base Path:** The frontend uses `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_API_BASE_PATH` explicitly, decoupling API routing from URL string heuristics.

---

## Setup & Local Development

### Prerequisites
- Python 3.11+
- Node.js 20+ and npm
- A Groq Cloud API Key ([console.groq.com](https://console.groq.com))

### 1. Environment Configuration

Create a `.env` file in the repository root:

```bash
# LLM Provider Configuration
GROQ_API_KEY="gsk_your_actual_key_here"
LLM_MODEL="openai/gpt-oss-120b"
LLM_MAX_RPM=20

# Backend Security & Networking
IP_RATE_LIMIT_RPM=15
ALLOWED_ORIGINS="http://localhost:3000,https://tracemind.vercel.app"

# Frontend API Target
NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
NEXT_PUBLIC_API_BASE_PATH=""
```

> **Security Note:** Never commit `.env` or expose API keys. The `.gitignore` file enforces exclusion of all environment files.

### 2. Backend Setup & Ingestion

```bash
# Install Python dependencies
pip install -r requirements.txt

# Ingest the 662 training postmortems into ChromaDB
python rag/generate.py --force

# Pre-warm SQLite generation cache for frontend presets
python scripts/prewarm_cache.py

# Verify zero data leakage against held-out test split
python scripts/verify_no_leakage.py

# Start FastAPI backend (http://127.0.0.1:8000)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Frontend Setup

```bash
cd frontend

# Install Node dependencies
npm install

# Run Next.js in development mode (http://localhost:3000)
npm run dev

# Or build for production
npm run build
npm run start
```

### 4. Running the Test Suite & Evaluation

```bash
# Run backend, clustering, split, and smoke verification tests
python -m pytest

# Run live deployment smoke test (defaults to http://127.0.0.1:8000 or custom BACKEND_URL)
BACKEND_URL=https://nishhhx0-tracemind-backend.hf.space python scripts/smoke_live.py

# Run full RAG vs. Zero-Shot evaluation on 93 test incidents
python eval/evaluate.py

# Run failure category retrieval benchmark
python eval/category_benchmark.py
```

---

## License

This project is licensed under the [MIT License](LICENSE).
