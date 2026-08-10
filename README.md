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
LLM Generation (Mistral-7B / Llama3-8B, prompted — no fine-tuning)
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
| Generation | Mistral-7B / Llama3-8B (prompted, not fine-tuned) |
| Backend API | FastAPI (hosted on Render/Railway/HF Spaces — needs real compute, not serverless) |
| Frontend | Next.js, deployed on Vercel |
| Evaluation | ROUGE, BERTScore, ablation, human "Actionability" rating |

**Note on hosting split:** the ML pipeline (embeddings, clustering, vector search, LLM inference) runs on a compute-backed host, exposed via FastAPI. Vercel serves the frontend only — its serverless functions aren't suited to long-running inference workloads. Frontend calls the backend over REST.

## Dataset

Target: 200–300 curated incident postmortems, standardized into a common JSON schema. Sourced aggregator-first rather than site-by-site, to reach a volume where clustering is meaningful:

- **danluu/post-mortems** — master curated list spanning Google, AWS, Cloudflare, GitLab, Slack, Heroku, Stripe, and others
- **Nat Welch's parsed postmortems** — pre-structured subset, evaluated for direct reuse
- **awesome-tech-postmortems** — supplementary curated list, non-overlapping entries
- **SRE Weekly archive** — years of back-issues, "Outages" section per issue
- **Statuspage.io-hosted incident histories** — shared HTML structure across many SaaS companies; one generic scraper covers 30–40+ domains with consistent output, the highest-ROI source for volume
- GitHub Status, Cloudflare Blog, GCP incident history, AWS post-event summaries

Schema:
```json
{
  "incident_id": "",
  "raw_log_excerpt": "",
  "root_cause": "",
  "remediation_steps": "",
  "source_url": ""
}
```

If the target volume isn't reached, this is stated plainly in the final report rather than overstated — clustering results are framed as illustrative/preprocessing rather than a standalone claim.

## Pipeline Trace (Key Feature)

The frontend includes a "Show Pipeline Trace" toggle that exposes:
- The raw log anomaly detected
- The retrieved historical incident, with similarity score
- The exact prompt sent to the LLM

This makes the RAG mechanism auditable rather than a black box — the core differentiator for the jury demo.

## Evaluation

- **ROUGE / BERTScore:** generated RCA summary vs. ground-truth postmortem text
- **Ablation (primary quantitative story):** pipeline performance with vs. without RAG retrieval
- **Prompt robustness ablation:** 2–3 prompt variants tested for output sensitivity
- **Actionability score (1–5):** peer-rated usefulness of generated remediation steps

Sample size is stated explicitly alongside all reported metrics — this is a proof-of-concept evaluation, not a claim of statistical significance at scale.

## 12-Week Roadmap

| Weeks | Milestone |
|---|---|
| 1–2 | Aggregator-first data sourcing (danluu list, Statuspage.io scraper, SRE Weekly) + schema standardization |
| 3–4 | Embedding + HDBSCAN clustering (core NLP module) |
| 5–6 | RAG layer: ChromaDB vector storage + retrieval |
| 7–8 | LLM integration via prompting (no LoRA) + prompt robustness ablation |
| 9–10 | Pipeline integration + Next.js frontend + FastAPI backend wiring |
| 11–12 | Evaluation, report, buffer |

## Explicit Scope Boundaries

- No LoRA fine-tuning — base model + retrieval-augmented prompting only, to keep the project deliverable within a single semester
- Dataset target is 200–300 samples, not production scale — this is a proof-of-concept pipeline
- Clustering is a preprocessing/dedup aid to retrieval, not evaluated as a standalone result

## Setup

### Backend
```bash
git clone https://github.com/nish-debug15/TraceMind.git
cd TraceMind/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd TraceMind/frontend
npm install
npm run dev
```

Deploy frontend to Vercel; deploy backend to a compute-backed host (Render/Railway/HF Spaces).

## License

TBD
