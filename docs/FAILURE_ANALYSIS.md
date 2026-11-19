# TraceMind RCA Failure Analysis

This document provides a root-cause breakdown of the **10 lowest-scoring test incidents** (ranked by RAG ROUGE-L F1) on the held-out test suite (`data/splits/test.json`, N = 93 incidents across 40 unseen companies). Evaluated strictly against the train-only retrieval index (`rag/chroma_db_train`, 662 incidents).

All statistics and excerpts are drawn directly from [`eval/results/failure_analysis.json`](file:///n:/gitt/TraceMind/eval/results/failure_analysis.json) and [`eval/results/rag_vs_zero_shot.json`](file:///n:/gitt/TraceMind/eval/results/rag_vs_zero_shot.json).

---

## 1. Failure Mode Taxonomy & Distribution

Across the bottom 10 incidents by lexical ROUGE-L overlap, failure modes partition into three primary systemic categories:

| Failure Category | Incident Count | % of Bottom 10 | Primary Vulnerability |
|:---|:---:|:---:|:---|
| **Status-Page Non-Technical Ground Truth** | 4 | 40% | Reference postmortem contains customer-facing recovery boilerplate rather than technical causal mechanics |
| **Cross-Company Domain Gap** | 4 | 40% | Retrieved precedent from an unseen company introduces conflicting architectural assumptions |
| **Telemetry Underspecified (< 150 chars)** | 2 | 20% | Log excerpt lacks stack traces, HTTP error codes, or telemetry context |

```
Distribution of Bottom 10 Failures:
[====================] 40% Status-Page Boilerplate Ground Truth (4)
[====================] 40% Cross-Company Domain Gap (4)
[==========          ] 20% Telemetry Underspecified (2)
```

---

## 2. Quantitative Summary of the Bottom 10 Incidents

| Rank | Incident ID | Target Company | Retrieved Company | Top-1 Similarity | ROUGE-L F1 | Failure Category |
|:---:|:---|:---|:---|:---:|:---:|:---|
| 1 | `discordstatus-9bm86bnmfhfr` | Discord | CircleCI | 0.4526 | **0.0000** | Status-Page Boilerplate |
| 2 | `discordstatus-dsv750sj44fn` | Discord | GitHub | 0.5602 | **0.0197** | Status-Page Boilerplate |
| 3 | `shopify-tq1zmqvw6cjz` | Shopify | OpenAI | 0.4894 | **0.0278** | Status-Page Boilerplate |
| 4 | `discordstatus-k42p6rqypf7v` | Discord | OpenAI | 0.5708 | **0.0299** | Status-Page Boilerplate |
| 5 | `icco-f9f4ba4c-686c-46a4-982c-8dc21bd8a448` | incident.io | Google Cloud | 0.5718 | **0.0320** | Cross-Company Domain Gap |
| 6 | `icco-7d5ab17c-e367-4135-9c88-53f0f2df0073` | Bloomberg | LinkedIn | 0.5016 | **0.0365** | Telemetry Underspecified |
| 7 | `discordstatus-pcg3nsfq6l2y` | Discord | OpenAI | 0.5435 | **0.0388** | Cross-Company Domain Gap |
| 8 | `discordstatus-5sqkt5xhj2pc` | Discord | GitHub | 0.6532 | **0.0389** | Telemetry Underspecified |
| 9 | `shopify-26hrlv896m7j` | Shopify | Reddit | 0.5120 | **0.0424** | Cross-Company Domain Gap |
| 10 | `icco-848f9263-fe81-4bbc-b0d2-cc1671bb88a5` | Salesforce | GitHub | 0.5532 | **0.0466** | Cross-Company Domain Gap |

---

## 3. In-Depth Representative Case Studies

### Case 1: Status-Page Non-Technical Ground Truth (Boilerplate Reference)
- **Incident ID**: `discordstatus-9bm86bnmfhfr`
- **Target Company**: Discord
- **Retrieved Incident**: `circleci-njbxb52z1g1y` (CircleCI, Cosine Sim: `0.4526`)
- **ROUGE-L Score**: `0.0000`

#### Ground Truth Excerpt
> *"Voice calls have fully recovered."*

#### Generated Root Cause Excerpt
> *"The 'guilds unavailable to some members' symptom points to a partial-service failure rather than a complete outage. In a large distributed messaging/voice platform, member guild states are maintained in sharded in-memory session caches backed by persistent datastores..."*

#### Mechanism of Failure
The public Statuspage dataset periodically stores short customer-facing resolution notices as the final postmortem text rather than detailed engineering retrospectives. Here, the ground truth is simply *"Voice calls have fully recovered."* The LLM generated an elaborate and plausible architectural hypothesis explaining guild availability partitions, but because the reference text had 0 causal tokens, lexical overlap evaluated to exactly `0.0000`. This illustrates an evaluation artifact common in web-scraped postmortem datasets: low ROUGE does not always signify hallucination; it frequently reflects an uninformative reference target.

---

### Case 2: Cross-Company Domain Gap (Divergent Architecture)
- **Incident ID**: `icco-f9f4ba4c-686c-46a4-982c-8dc21bd8a448`
- **Target Company**: incident.io
- **Retrieved Incident**: `gcp-6876619551109882402` (Google Cloud, Cosine Sim: `0.5718`)
- **ROUGE-L Score**: `0.0320`

#### Ground Truth Excerpt
> *"On Wednesday, April 9, 2025, between 14:16 and 14:27 UTC, incident.io experienced intermittent availability issues, culminating in a 1-minute total outage..."*

#### Generated Root Cause Excerpt
> *"The outage was triggered by a configuration change to the safety-filtering service that underpins all Vertex AI endpoints across multiple regions. The rollout pushed an invalid policy rule that caused the filter proxy to drop valid requests..."*

#### Mechanism of Failure
Because the incoming incident log described generalized availability degradation and API 500 spikes, vector similarity matched a Google Cloud incident involving Vertex AI safety filtering (similarity `0.5718`). Under the baseline un-gated prompt, the LLM over-indexed on the historical precedent, hallucinating that incident.io was afflicted by a Vertex AI policy filter rollout. 

This error directly demonstrates the necessity of the **Relevance Gate Guardrail (`relevance_gate_v2`)**: when similarity is moderate (below `0.60`), the system must explicitly force the model to verify architectural relevance before adopting the precedent's root-cause entities.

---

### Case 3: Telemetry Underspecified (Log Excerpt < 150 Chars)
- **Incident ID**: `icco-7d5ab17c-e367-4135-9c88-53f0f2df0073`
- **Target Company**: Bloomberg
- **Retrieved Incident**: `icco-f89b8bcd-6742-4fd9-b8ab-c1c6e21691ec` (LinkedIn, Cosine Sim: `0.5016`)
- **ROUGE-L Score**: `0.0365`

#### Raw Log Telemetry (79 characters)
> *"Healthcare.gov rollout failure due to severe performance degradation under load."*

#### Ground Truth Excerpt
> *"A large organizational failure to build a website for United States healthcare."*

#### Generated Root Cause Excerpt
> *"The failure to deliver the United States healthcare website stemmed from a combination of systemic and tactical software engineering shortcomings: massive uncoordinated multi-contractor code integration, un-scalable relational database bottlenecks..."*

#### Mechanism of Failure
The input telemetry was a brief 79-character summary sentence rather than rich server logs or stack traces. Without operational metrics, error traces, or memory profiling data, the retrieval system matched generic scaling failures in the LinkedIn corpus. While the generated output accurately captured the broader engineering failure of Healthcare.gov, the model's vocabulary diverged from the idiosyncratic phrasing of the reference text (*"A large organizational failure..."*), yielding a ROUGE-L of `0.0365`.

---

## 4. Architectural Mitigations Implemented

To address these empirical failure patterns, TraceMind incorporates three structural guardrails:

1. **Cosine Similarity Floor (0.40)**:
   Any retrieval match falling below 0.40 cosine similarity is flagged as weak precedent. 7 out of 93 test incidents (7.53%) trigger this abstention floor, instructing the pipeline to discount historical analogies.
2. **Relevance Gate (`relevance_gate_v2`)**:
   Forces the generation model to emit a strict `Precedent: relevant` or `Precedent: not relevant` header on line one. If similarity is ungrounded or cross-company semantics conflict, the model discards the precedent and derives the root cause purely from the incoming telemetry.
3. **Observational Clustering Separation**:
   Clustering signatures are kept strictly observational in the Pipeline Trace and never alter vector retrieval or prompt construction, preventing spurious cluster associations from polluting generation context.
