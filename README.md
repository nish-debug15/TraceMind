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
```

## Tech Stack

| Component        | Tool                              |
|-------------------|------------------------------------|
| Embeddings        | sentence-transformers             |
| Clustering        | HDBSCAN                           |
| Vector Store      | ChromaDB                          |
| Generation        | Mistral-7B / Llama3-8B (prompted, not fine-tuned) |
| UI                | Streamlit                         |
| Evaluation        | ROUGE, BERTScore, human "Actionability" rating |

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

The Streamlit demo includes a "Show Pipeline Trace" toggle that exposes:
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
| 9–10  | Pipeline integration + Streamlit UI |
| 11–12 | Evaluation, report, buffer |

## Explicit Scope Boundaries

- **No LoRA fine-tuning** — base model + retrieval-augmented prompting only, to keep the project deliverable within a single semester.
- Dataset is intentionally small (50–100 samples); this is a proof-of-concept pipeline, not a production-scale system.

## Setup

```bash
git clone <repo-url>
cd rca-pipeline
pip install -r requirements.txt
streamlit run app.py
```

## License

MIT
