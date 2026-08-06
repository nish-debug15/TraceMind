# RAG-Powered Root Cause Analysis (RCA) Pipeline for Unstructured SRE Logs

An automated NLP pipeline that clusters log anomalies and uses Retrieval-Augmented Generation (RAG) to match current infrastructure failures against historical postmortems — generating plain-English root cause summaries and remediation steps.

## Problem Statement

When critical infrastructure fails, Site Reliability Engineers (SREs) lose valuable time manually correlating unstructured server logs with past incident reports. This project automates that correlation: given a raw log dump, the system detects the anomaly, retrieves the most similar past incident, and generates an actionable root cause analysis — with full pipeline transparency so the reasoning can be audited, not just trusted.

## Architecture

```
Raw Log Input
     │
     ▼
Embedding Layer (sentence-transformers)
     │
     ▼
Anomaly Clustering (HDBSCAN)
     │
     ▼
Vector Retrieval (ChromaDB) ──> Historical Postmortem Corpus
     │
     ▼
RAG-Augmented Prompt Construction
     │
     ▼
LLM Generation (Mistral-7B / Llama3-8B, no fine-tuning)
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

## Tech Stack

| Component        | Tool                              |
|-------------------|------------------------------------|
| Embeddings        | sentence-transformers             |
| Clustering        | HDBSCAN                           |
| Vector Store      | ChromaDB                          |
| Generation        | Mistral-7B / Llama3-8B (prompted, not fine-tuned) |
| Backend API       | FastAPI (hosted on Render/Railway/HF Spaces — needs real compute, not serverless) |
| Frontend          | Next.js, deployed on Vercel |
| Evaluation        | ROUGE, BERTScore, human "Actionability" rating |

**Note on hosting split**: the ML pipeline (embeddings, clustering, vector search, LLM inference) runs on a compute-backed host, exposed via FastAPI. Vercel serves the frontend only — its serverless functions aren't suited to long-running inference workloads. Frontend calls the backend over REST.

## Dataset

50–100 curated incident postmortems scraped from public sources (GitHub Status, Cloudflare blog, Atlassian Statuspage), standardized into a common JSON schema:

```json
{
  "incident_id": "",
  "raw_log_excerpt": "",
  "root_cause": "",
  "remediation_steps": "",
  "source_url": ""
}
```

## Pipeline Trace (Key Feature)

The frontend includes a "Show Pipeline Trace" toggle that exposes:
1. The raw log anomaly detected.
2. The retrieved historical incident, with similarity score.
3. The exact prompt sent to the LLM.

This makes the RAG mechanism auditable rather than a black box — the core differentiator for the jury demo.

## Evaluation

- **ROUGE / BERTScore**: generated RCA summary vs. ground-truth postmortem text.
- **Ablation**: pipeline performance with vs. without RAG retrieval.
- **Actionability score (1–5)**: peer-rated usefulness of the generated remediation steps.

## 12-Week Roadmap

| Weeks | Milestone |
|-------|-----------|
| 1–2   | Data scraping + schema standardization |
| 3–4   | Embedding + HDBSCAN clustering (core NLP module) |
| 5–6   | RAG layer: ChromaDB vector storage + retrieval |
| 7–8   | LLM integration via prompting (no LoRA) |
| 9–10  | Pipeline integration + Next.js frontend + FastAPI backend wiring |
| 11–12 | Evaluation, report, buffer |

## Explicit Scope Boundaries

- **No LoRA fine-tuning** — base model + retrieval-augmented prompting only, to keep the project deliverable within a single semester.
- Dataset is intentionally small (50–100 samples); this is a proof-of-concept pipeline, not a production-scale system.

## Setup

**Backend**
```bash
git clone https://github.com/nish-debug15/TraceMind.git
cd rca-pipeline/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend**
```bash
cd rca-pipeline/frontend
npm install
npm run dev
```

Deploy frontend to Vercel; deploy backend to a compute-backed host (Render/Railway/HF Spaces).

## License

MIT
