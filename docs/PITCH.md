# TraceMind: 10-Minute Team Presentation Script

**Project:** TraceMind — RAG-Powered Root Cause Analysis for SRE Logs  
**Format:** 10:00 Total Run Time (Target: ~140 words per minute)  
**Slide Count:** 11 Slides Total  

---

## Presentation Overview & Timing Table

| Speaker | Section | Slide(s) | Target Time | Word Count |
|---|---|---|---|---|
| **1. Nishit** | Introduction & Team Overview | Slide 1–2 | 0:00 – 1:00 (1:00) | ~140 words |
| **2. Pragun** | Problem Statement & Objectives | Slide 3–4 | 1:00 – 2:45 (1:45) | ~245 words |
| **3. Pranav** | Dataset, Quality Filter & Clustering | Slide 5–6 | 2:45 – 4:30 (1:45) | ~245 words |
| **4. Sameera** | Database Layer & 2-Pass RAG Pipeline | Slide 7–8 | 4:30 – 6:15 (1:45) | ~245 words |
| **5. Nishit** | System Architecture & Live Demo | Slide 9 | 6:15 – 8:15 (2:00) | ~280 words |
| **6. Unique** | Evaluation, Limitations & Roadmap | Slide 10–11 | 8:15 – 10:00 (1:45) | ~245 words |

---

## 1. Nishit — Introduction & Team Roles (1:00)

### Slide 1: Title Slide
- **Title:** TraceMind: Auditable Root Cause Analysis for SRE Logs
- **Subtitle:** Retrieval-Augmented Postmortem Correlation with Zero Black-Box Generation
- **Team:** Nishit (System Architecture & Frontend), Pragun (Domain Analysis & Pipeline Design), Pranav (Data Engineering & Clustering), Sameera (Vector Storage & Retrieval), Unique (Benchmarking & Evaluation)

### Slide 2: Agenda & Core Premise
- **What We Built:** An automated log diagnostic pipeline that matches live server crashes against verified postmortems.
- **The Core Differentiator:** Full pipeline auditability (Pipeline Trace) showing raw retrieval similarity, matched precedent, and exact prompts.
- **Roadmap:** Problem -> Corpus -> RAG Pipeline -> Live Demo -> Quantitative Evaluation.

### Spoken Script (Nishit)
"Good morning, faculty and jury members. I am Nishit, presenting TraceMind alongside Pragun, Pranav, Sameera, and Unique.

TraceMind is an auditable incident analysis system for Site Reliability Engineers. When production infrastructure fails, engineers are flooded with thousands of cryptic logs and must rapidly determine what broke and how to recover. TraceMind ingests that raw log excerpt, matches it against verified historical postmortems, and synthesizes a clear root cause analysis with an actionable five-step remediation checklist.

Unlike generic AI assistants, TraceMind is never a black box. Every recommendation exposes an interactive Pipeline Trace showing the exact historical incident retrieved, its mathematical similarity score, and the raw prompt delivered to the model.

Today, Pragun covers the operational problem, Pranav details our curated postmortem corpus, Sameera explains our vector retrieval and relevance gate, I will demonstrate the live console, and Unique will present our quantitative benchmarks."

**Handoff:** *"I will now hand over to Pragun to explain why manual incident correlation during production outages remains so painful."*

---

## 2. Pragun — Problem Statement & Objectives (1:45)

### Slide 3: The Outage Reality
- **The Operational Bottleneck:** During high-severity outages, Mean Time to Resolution (MTTR) is dominated by diagnosis, not execution.
- **Information Asymmetry:** Raw stack traces tell *what* crashed, but rarely *why* or *how* past teams mitigated it.
- **Failure of Naive Search:** Regex and keyword searches miss synonymy (e.g., 'pool exhausted' vs. 'no connection slots available').

### Slide 4: Why Generic LLMs Fall Short
- **Hallucination Risk:** Off-the-shelf LLMs invent plausible-sounding shell commands, invalid configuration flags, and phantom architecture.
- **No Grounding:** Generic models lack company-specific context and precedent without a retrieval mechanism.
- **TraceMind Objective:** Provide grounded, verifiable incident diagnosis with strict precedent attribution within seconds.

### Spoken Script (Pragun)
"When a production service goes down at 2:00 AM, every minute of downtime costs thousands of dollars. But if you observe an SRE during an incident, they spend eighty percent of their time reading logs, searching internal wikis, and querying Slack channels trying to answer one question: *have we seen this failure before?*

Searching through thousands of unindexed incident postmortems during an active outage is like an emergency room doctor hunting through a basement of paper charts while a patient is in critical condition. Keyword searches fail completely because two engineers describe the same underlying bug with different vocabulary: one writes 'database pool exhausted,' while another logs 'connection acquisition timeout.' 

When teams turn to generic language models to diagnose logs, they hit a dangerous wall: hallucinations. A standard LLM will happily invent imaginary database configuration parameters or advise restarting a cluster that is already in a split-brain state. SREs cannot trust ungrounded suggestions during an active incident.

Our objective with TraceMind was clear: build a system that grounds every diagnosis in verified, real-world postmortems using Retrieval-Augmented Generation—which simply means fetching relevant factual reference documents before the model writes a single word. Most importantly, if no historical precedent matches the failure, the system must explicitly state that rather than guess.

I will now pass the floor to Pranav to explain how we collected and structured our incident corpus."

**Handoff:** *"Pranav will now explain how we collected our corpus of 828 real postmortems and structured our leakage-free dataset."*

---

## 3. Pranav — Data Engineering, Quality Filtering & Clustering (1:45)

### Slide 5: The Postmortem Corpus & Zero-Leakage Split
- **Corpus Size:** 828 standardized incident reports across 58 companies.
- **Sources:** `icco/postmortems`, Statuspage archives across 20 SaaS platforms, GCP incident disclosures, and Cloudflare.
- **Quality Filter:** 544 High-Quality postmortems (concrete root causes and step-by-step remediations) vs. 284 Low-Quality summaries.
- **Zero-Leakage Split:** 662 train incidents (18 companies) vs. 93 test incidents (40 companies). 100% company-disjoint split.

### Slide 6: HDBSCAN Clustering as Preprocessing
- **Clustering Role:** Preprocessing and deduplication, NOT the core search engine.
- **Algorithm:** HDBSCAN on 384-dimensional dense vectors (`all-MiniLM-L6-v2`).
- **Cluster Count:** 48 clusters identified on the training corpus (`min_cluster_size=3`).
- **Function:** Detects recurring failure signatures and isolates novel or noisy log patterns.

### Spoken Script (Pranav)
"A retrieval system is only as good as the ground truth behind it. We constructed a standardized dataset of 828 real postmortems sourced aggregator-first from curated repositories like icco, official Statuspage histories across 20 major platforms like Datadog, GitHub, and Twilio, alongside official retrospectives from Google Cloud and Cloudflare.

Early on, we realized that raw quantity was harmful. Two hundred and eighty-four of those incidents were brief status tweets like 'we resolved a DNS issue.' We built a quality filter that separated our corpus into 544 high-quality records—containing detailed root causes and multi-step remediation procedures—and 284 low-quality summaries.

To evaluate generalization honestly, we enforced a strict company-grouped split using fixed seed 42. Our training set has 662 incidents across 18 companies. Our held-out test set contains 93 incidents across 40 completely different companies, with 100% of test records rated high quality. Zero companies overlap between train and test, ensuring the model never cheats by remembering company-specific hostnames or error patterns.

Before retrieval, we run HDBSCAN—a density-based clustering algorithm that groups similar items without forcing every point into a cluster. Think of HDBSCAN as a postal mail sorter: it groups near-duplicate logs into bundles so repetitive noise doesn't distort our index. It found 48 distinct anomaly clusters in our training data. It is important to emphasize that clustering is our deduplication and anomaly-detection filter; the actual matching is performed by our vector retrieval layer.

Sameera will now explain how our database and retrieval pipeline work."

**Handoff:** *"Sameera will now walk through our vector database layer and two-pass retrieval mechanism."*

---

## 4. Sameera — Database Layer & 2-Pass RAG Pipeline (1:45)

### Slide 7: The Vector Database Layer (ChromaDB)
- **Vector Database:** Converts text into numerical vectors capturing semantic meaning; searches by concept, not exact keywords.
- **ChromaDB Architecture:** Embedded, in-process vector store running directly inside our FastAPI backend with zero external server overhead.
- **Stored Incident Schema:** 384-dimensional embedding, raw log excerpt, root cause, remediation steps, source URL, and quality label.
- **Initialization:** Built once from the 662 training incidents and loaded into memory on backend startup.

### Slide 8: 2-Pass Retrieval & The Relevance Gate
- **Pass 1:** Queries `where={"quality": "high"}` with a mandatory **0.40 cosine similarity floor**.
- **Pass 2 (Fallback):** Expands search across the full 662-incident corpus if Pass 1 misses.
- **Relevance Gate:** Prompt instructs the model to inspect the match; outputs `'Precedent: not relevant'` on line 1 if the match is unrelated.
- **Zero Hallucination:** Tested on 7 out-of-distribution incidents below 0.40 similarity: 7/7 (100%) correctly rejected false precedents.

### Spoken Script (Sameera)
"To match an incoming error log against past incidents, we use a vector database. A vector database converts text into a list of numbers—an embedding—that captures its conceptual meaning, allowing us to search by semantic similarity rather than exact keyword matches.

Think of a traditional database like a book index matching exact words, while a vector database is a library catalog: 'connection acquisition timeout' and 'database pool exhausted' sit on the exact same shelf because their vectors point in the same direction.

We chose ChromaDB because it is lightweight and runs directly in-process inside our backend with no separate database server. Each record stores the 384-dimensional embedding alongside rich metadata: the raw log excerpt, the root cause, remediation steps, the source URL, and the quality label. The collection is built once from our postmortem corpus and loaded directly into memory when the backend starts.

When an engineer inputs a log, we execute a two-pass retrieval. Pass One filters strictly for high-quality incidents with a 0.40 cosine similarity floor. If that threshold is not met, Pass Two falls back across all corpus records.

Finally, we prevent hallucination with our Relevance Gate. If the retrieved incident is unrelated, the model is instructed to output 'Precedent: not relevant' on line one and diagnose the log independently. On the seven test incidents below our 0.40 floor, the gate scored 100%—all seven correctly rejected false precedents.

Nishit will now demonstrate this live in our console."

**Handoff:** *"Nishit will now demonstrate the system live and show our Pipeline Trace in action."*

---

## 5. Nishit — System Architecture & Live Demo (2:00)

### Slide 9: System Architecture & Live Demonstration
- **Architecture:** Next.js 16 (Bento UI) -> FastAPI on Hugging Face Spaces -> Groq (`openai/gpt-oss-120b`).
- **Live Flow:** Log Input -> Soft-Wrap Editor -> 2-Pass ChromaDB Retrieval -> Prompt Synthesis -> Groq Inference.
- **Interactive Trace:** Matched URL, Cosine Similarity Gauge with 0.40 tick, Relevance Badge, Raw Prompt Inspector.
- **Resilience:** Client-side rate limiting (15 RPM), SQLite pre-warmed cache for zero-latency instant preset demonstration.

### Spoken Script (Nishit)
"Let us examine how these components connect in practice. Our architecture pairs a Next.js 16 frontend styled as a light-first Bento console with a FastAPI backend hosted on Hugging Face Spaces. The backend computes embeddings via sentence-transformers, queries ChromaDB in-process, and requests generation from Groq running the 120-billion parameter GPT-OSS model.

*(Action: Transition to browser window showing TraceMind console)*

Here is the TraceMind console. In the Incident Log tile, I will paste an excerpt from a production outage: a database proxy connection pool exhaustion error. Notice our log editor soft-wraps long stack traces while keeping logical gutter line numbers strictly aligned. 

*(Action: Click 'Analyze incident' or press Cmd+Enter)*

In under two seconds, the pipeline processes the log. On the right, the stepper tracks our completed pipeline stages: embed, cluster assignment, ChromaDB retrieval, and generation.

TraceMind presents two primary outputs: first, a synthesized Root Cause analysis in markdown explaining that worker connections saturated the pool limit. Second, an actionable five-step Remediation Checklist with interactive checkboxes—from terminating idle transactions to raising pool limits.

Now, look at our core differentiator: the Pipeline Trace tile. The Pipeline Trace is like an aircraft flight recorder: instead of asking engineers to trust an AI prediction blindly, it exposes the exact telemetry behind the result. 

On the right is our similarity gauge showing 0.51, safely above our 0.40 threshold tick. It displays the matched precedent from GitHub's public archive with a direct hyperlink to the official document. Above the analysis, our relevance badge confirms: 'Precedent applied.' In the Prompt tab, you can inspect the exact prompt delivered to Groq.

If network connectivity ever drops during an outage, our system includes a pre-warmed SQLite generation cache for all preset scenarios, ensuring instant offline continuity."

**Handoff:** *"Unique will now review our quantitative evaluation and summarize our findings."*

---

## 6. Unique — Evaluation, Limitations & Future Work (1:45)

### Slide 10: Quantitative Benchmarks (Held-Out Test Set, N = 93)
- **Ablation Study (N = 93):**
  - ROUGE-1 F1: 0.1809 (Zero-Shot) -> **0.1941** (RAG) [**+0.0132**]
  - ROUGE-2 F1: 0.0384 (Zero-Shot) -> **0.0439** (RAG) [**+0.0055**]
  - ROUGE-L F1: 0.1061 (Zero-Shot) -> **0.1210** (RAG) [**+0.0149**]
- **Retrieval Quality (N = 93):**
  - Mean Top-1 Similarity: **0.5273** (Median: 0.5333)
  - Above 0.40 Threshold: **86 / 93 (92.5%)**
  - Top-1 Category Hit Rate: **33.3% (31/93)** across 9 failure categories.
- **Relevance Gate Precision:** **7 / 7 (100%)** correct rejections on weak precedent (< 0.40).

### Slide 11: Honest Limitations & Roadmap
- **Proof-of-Concept Sample Size:** Evaluated on 93 held-out incidents; directionally positive, not a claim of statistical significance.
- **Infrastructure Constraints:** Groq free-tier rate limit (20 RPM) requires client-side throttling.
- **Future Work:** Continuous log streaming (Kafka/Fluentd), local fine-tuned SLM (8B), and automated Jira/PagerDuty ticket sync.
- **Closing Thought:** Reliability is built on transparency, not blind faith.

### Spoken Script (Unique)
"To validate TraceMind objectively, we evaluated our held-out test split of 93 high-quality incidents from 40 companies that the system had never seen during indexing.

Comparing TraceMind’s RAG pipeline against an unaugmented zero-shot baseline across all 93 test records, RAG consistently outperforms raw generation: ROUGE-1 F1 increases from 0.1809 to 0.1941, ROUGE-2 increases from 0.0384 to 0.0439, and ROUGE-L increases from 0.1061 to 0.1210. In our category benchmark, the retrieval system achieved a top-one failure category hit rate of 33.3 percent across nine infrastructure failure modes, with software bugs reaching 54.5 percent.

Our similarity audit revealed a mean top-one similarity of 0.5273, with 92.5 percent of test incidents scoring above our 0.40 confidence floor. On the remaining 7.5 percent where similarity dropped below 0.40, our Relevance Gate achieved 100 percent accuracy in preventing false historical analogies.

We want to be entirely transparent about our limitations. This is an academic proof of concept evaluated on 93 test incidents, so we do not claim statistical significance. Furthermore, operating on Groq’s free-tier rate limit of 20 requests per minute required strict client-side throttling. 

For future work, we plan to connect live streaming collectors like Kafka, explore fine-tuning smaller 8-billion parameter models on sanitized postmortems, and close the loop with automatic PagerDuty incident annotations.

TraceMind proves that during a critical outage, the most valuable AI is not the one that sounds most confident, but the one that shows its work. 

Thank you. We are now ready for your questions."

---

## 8 Likely Jury Questions & Truthful Answers

### Q1: Why not fine-tune an open-source model (like Llama-3-8B) on all 828 postmortems instead of using RAG?
**Answer:** Fine-tuning bakes historical facts into static model weights, creating two severe problems: catastrophic forgetting and hallucinations that cannot be cited. With RAG, updating the system requires only adding a new JSON postmortem to ChromaDB without hours of GPU re-training. Crucially, fine-tuned models cannot provide a clickable hyperlink to the source postmortem or an audit trail of the retrieved document.

### Q2: If ChromaDB already performs vector retrieval, why is HDBSCAN clustering needed?
**Answer:** Clustering serves as a preprocessing and deduplication filter, not the matching engine. In real-world telemetry, 500 identical log lines often flood the system during a single network hiccup. HDBSCAN groups near-duplicate logs into 48 representative centroids so that ChromaDB is not overwhelmed by redundant noise.

### Q3: How do you prevent the LLM from hallucinating when retrieval returns an irrelevant incident?
**Answer:** We implemented a two-part defense. First, a mathematical threshold: if Pass One retrieval similarity falls below 0.40, it is flagged as weak precedent. Second, our `relevance_gate_v2` prompt explicitly instructs the LLM: if the historical case does not clearly match the failure, it must write 'Precedent: not relevant' on line one and synthesize the cause from the log alone. On our test set, this gate scored 100% accuracy (7/7) on low-similarity incidents.

### Q4: Why use Groq and a massive 120-billion parameter model instead of a local model?
**Answer:** We prioritized inference quality and diagnostic reasoning for complex root causes while keeping the backend lightweight enough to host on a free Hugging Face Space CPU tier. Groq provides LPU hardware delivering over 200 tokens per second with zero local GPU requirements. To protect against free-tier rate limits (20 RPM), we built an in-memory client-side rate limiter and a pre-warmed SQLite cache.

### Q5: Isn't a 93-record test set too small to prove the system works?
**Answer:** Yes, 93 records is a proof-of-concept sample size, which is why we explicitly state this limitation and do not claim formal statistical significance. However, our 93 test records were rigorously selected: they represent 40 completely unseen companies with zero data leakage, and 100% of them are verified high-quality postmortems. It is a rigorous directional benchmark.

### Q6: What happens if an incoming log has never happened anywhere in the tech industry before?
**Answer:** The vector retrieval score will reflect low similarity (< 0.40), the similarity gauge will label it 'Weak match, treat as hypothesis,' and the Relevance Gate will strip the precedent, causing the model to output 'No relevant precedent' and analyze the stack trace on its own merits without fabricating an analogy.

### Q7: Why are the ROUGE score improvements modest (+0.013 ROUGE-1)?
**Answer:** ROUGE measures exact n-gram word overlap against human postmortem write-ups. Two engineers describing the exact same Kubernetes OOMKill event may use completely different wording ('pod crashed due to cgroup memory cap' vs. 'payments container killed by Linux kernel OOM'). That is why we evaluate semantic embedding similarity and failure category hit rate alongside ROUGE.

### Q8: How does your quality label improve retrieval over vanilla semantic search?
**Answer:** Standard vector search only measures text closeness, meaning a 10-word vague status tweet with high keyword overlap could outrank a 500-word comprehensive engineering report. Our two-pass retrieval forces the search to look at `quality: high` records first, ensuring the LLM is primed with actionable remediation steps rather than brief status notices.

---

## Visuals & Screenshots Guide for Presentation Slides

1. **Slide 1 (Title):** TraceMind geometric logo and team roster.
2. **Slide 3 (Outage Reality):** Visual illustration of the 'Log Firehose' vs. human MTTR timeline.
3. **Slide 5 (Dataset Split):** Sankey diagram showing 828 corpus -> 544 high / 284 low -> 662 train (18 companies) / 93 test (40 companies).
4. **Slide 6 (Clustering):** 2D UMAP projection visualization of the 48 HDBSCAN clusters.
5. **Slide 7 (ChromaDB Architecture):** Diagram illustrating vector embedding mapping and metadata schema.
6. **Slide 8 (RAG Flow):** Flowchart of 2-Pass retrieval with the 0.40 cosine floor and Relevance Gate decision branch.
7. **Slide 9 (Live Demo):** Full-screen screenshot of the TraceMind console (`docs/pipeline_trace_demo.png`) highlighting the Pipeline Trace and Similarity Gauge.
8. **Slide 10 (Evaluation):** Benchmark table displaying ROUGE gains and Category Hit Rates across the 9 failure classes.
