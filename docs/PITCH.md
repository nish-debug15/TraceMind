# TraceMind: 10-Minute Team Presentation Script

**Project:** TraceMind — RAG-Powered Root Cause Analysis for SRE Logs  
**Format:** 10:00 Total Run Time (Target: ~140 words per minute)  
**Slide Count:** 11 Core Slides + 1 Backup Slide  

---

## Pre-Flight Checklist (Demo Day)

- [x] **Source Verification:** Datadog and Twilio confirmed in Statuspage source list (`scripts/make_split.py`).
- [x] **Frontend Live:** Vercel deployment accessible at `https://tracemind.vercel.app`.
- [x] **HF Space Awake:** FastAPI backend on Hugging Face Spaces responding at `https://nishhhx0-tracemind-backend.hf.space/health`.
- [x] **Demo Log Chosen:** DB proxy connection pool exhaustion preset ready in Incident Log editor.
- [x] **Cache Pre-Warmed:** SQLite generation cache pre-warmed for all frontend example presets (`scripts/prewarm_cache.py`).

---

## Presentation Overview & Timing Table

| Speaker | Section | Slide(s) | Target Time | Word Count |
|---|---|---|---|---|
| **1. Nishit** | Introduction & Team Overview | Slide 1–2 | 0:00 – 1:00 (1:00) | 146 words |
| **2. Pragun** | Problem Statement & Objectives | Slide 3–4 | 1:00 – 2:45 (1:45) | 185 words |
| **3. Pranav** | Dataset, Quality Filter & Clustering | Slide 5–6 | 2:45 – 4:30 (1:45) | 216 words |
| **4. Sameera** | Database Layer & 2-Pass RAG Pipeline | Slide 7–8 | 4:30 – 6:15 (1:45) | 238 words |
| **5. Nishit** | System Architecture & Live Demo | Slide 9 | 6:15 – 8:15 (2:00) | 192 words + demo |
| **6. Unique** | Evaluation, Limitations & Roadmap | Slide 10–11 | 8:15 – 10:00 (1:45) | 233 words |

---

## 1. Nishit — Introduction & Team Roles (1:00)

### Slide 1: Title Slide
- **Title:** TraceMind: Auditable Root Cause Analysis for SRE Logs
- **Subtitle:** Retrieval-Augmented Postmortem Correlation with Fully Inspectable Retrieval and Prompt
- **Team:** Nishit (System Architecture & Frontend), Pragun (Domain Analysis & Pipeline Design), Pranav (Data Engineering & Clustering), Sameera (Vector Storage & Retrieval), Unique (Benchmarking & Evaluation)

### Slide 2: Agenda & Core Premise
- **What We Built:** An automated log diagnostic pipeline that matches live server crashes against verified postmortems.
- **The Core Differentiator:** Retrieval and prompt are fully inspectable via our interactive Pipeline Trace (similarity gauge, matched precedent URL, raw prompt).
- **Roadmap:** Problem -> Corpus -> RAG Pipeline -> Live Demo -> Quantitative Evaluation.

### Spoken Script (Nishit)
"Good morning, faculty and jury members. I am Nishit, presenting TraceMind alongside Pragun, Pranav, Sameera, and Unique.

TraceMind is an auditable incident analysis system for Site Reliability Engineers. When production infrastructure fails, engineers are flooded with thousands of cryptic logs and must rapidly determine what broke and how to recover. TraceMind ingests that raw log excerpt, matches it against verified historical postmortems, and synthesizes a clear root cause analysis with an actionable five-step remediation checklist.

Unlike opaque AI assistants, TraceMind ensures the retrieval and prompt are fully inspectable. Every recommendation exposes an interactive Pipeline Trace showing the exact historical incident retrieved, its mathematical similarity score, and the raw prompt delivered to the model.

Today, Pragun covers the operational problem, Pranav details our curated postmortem corpus, Sameera explains our vector retrieval and relevance gate, I will demonstrate the live console, and Unique will present our quantitative benchmarks."

**Handoff:** *"I will now hand over to Pragun to explain why manual incident correlation during production outages remains so painful."*

---

## 2. Pragun — Problem Statement & Objectives (1:45)

### Slide 3: The Outage Reality
- **The Operational Bottleneck:** During high-severity outages, Mean Time to Resolution (MTTR) is dominated by diagnosis, not execution.
- **Information Asymmetry:** Raw stack traces tell *what* crashed, but rarely *why* or *how* past teams mitigated it.
- **Failure of Naive Search:** Keyword queries fail across synonymy (e.g., 'pool exhausted' vs. 'no connection slots available').

### Slide 4: Why Generic LLMs Fall Short
- **Hallucination Risk:** Ungrounded LLMs invent invalid configuration flags, phantom parameters, and risky terminal commands.
- **No Grounding:** Generic models lack infrastructure context and historical precedent without a retrieval mechanism.
- **TraceMind Objective:** Provide grounded incident diagnosis with strict precedent attribution within seconds.

### Spoken Script (Pragun)
"When a production service goes down at 2:00 AM, engineers face intense pressure to restore availability quickly. But during an active incident, engineers spend most of their triage time reading logs, searching internal wikis, and querying chat channels trying to answer one question: *have we seen this failure before?*

Searching through unindexed historical postmortems during an outage is like an emergency room doctor hunting through a basement of paper charts while a patient is in critical condition. Keyword searches fail because different teams describe the exact same bug with different vocabulary: one writes 'database pool exhausted,' while another logs 'connection acquisition timeout.'

When teams turn to generic language models to diagnose logs, they encounter hallucinations. A standard model without grounding may invent non-existent configuration parameters or suggest dangerous shell commands that exacerbate downtime. SREs cannot trust unverified suggestions during an active incident.

Our objective with TraceMind is to ground every diagnosis in verified, real-world postmortems using Retrieval-Augmented Generation—fetching relevant factual references before the model generates its response. Most importantly, if no historical precedent matches the failure, the system must explicitly say so rather than guess."

**Handoff:** *"Pranav will now explain how we collected our corpus of 828 real postmortems and structured our company-disjoint split."*

---

## 3. Pranav — Data Engineering, Quality Filtering & Clustering (1:45)

### Slide 5: The Postmortem Corpus & Company-Disjoint Split
- **Corpus Size:** 828 standardized incident reports across 58 companies.
- **Sources:** `icco/postmortems`, Statuspage archives across 20 SaaS platforms (including Datadog and Twilio), GCP incident disclosures, and Cloudflare.
- **Quality Filter:** 544 High-Quality postmortems (concrete root causes and multi-step remediations) vs. 284 Low-Quality summaries.
- **Company-Disjoint Split:** 662 train (18 companies) + 93 test (40 companies). The 73 low-quality incidents from test companies are discarded to prevent company leakage.

### Slide 6: HDBSCAN Clustering as Preprocessing
- **Clustering Role:** Preprocessing and deduplication, NOT the core search engine.
- **Algorithm:** HDBSCAN on 384-dimensional dense vectors (`all-MiniLM-L6-v2`).
- **Cluster Count:** 48 clusters identified on the training corpus (`min_cluster_size=3`).
- **Function:** Detects recurring failure signatures and isolates novel or noisy log patterns.

### Spoken Script (Pranav)
"A retrieval system is only as good as the ground truth behind it. We constructed a dataset of 828 real postmortems sourced aggregator-first from curated repositories like icco, official Statuspage histories across 20 platforms like Datadog and Twilio, and public retrospectives from Google Cloud and Cloudflare.

Raw quantity alone was noisy. Two hundred and eighty-four incidents were brief status updates. We built a quality filter separating our corpus into 544 high-quality records—with concrete root causes and multi-step remediations—and 284 low-quality summaries.

To evaluate out-of-distribution generalization honestly, we enforced a strict company-disjoint split using fixed seed 42. Our training set has 662 incidents across 18 companies. Our test set contains 93 incidents across 40 completely different companies, with 100% rated high quality. You will notice 662 plus 93 equals 755: the remaining 73 low-quality incidents from those 40 test companies were completely discarded to maintain a strict company-disjoint split while keeping the test set purely high-quality.

Before retrieval, we run HDBSCAN—a density-based clustering algorithm that groups similar data without forcing every outlier into a cluster. Think of HDBSCAN as a mail sorter: it groups near-duplicate logs into bundles so repetitive errors don't flood our index. It identified 48 distinct clusters in our training data. Crucially, clustering is only a deduplication filter; matching is handled by vector retrieval."

**Handoff:** *"Sameera will now walk through our vector database layer and two-pass retrieval mechanism."*

---

## 4. Sameera — Database Layer & 2-Pass RAG Pipeline (1:45)

### Slide 7: The Vector Database Layer (ChromaDB)
- **Vector Database:** Converts text into numerical vectors capturing semantic meaning; searches by concept, not exact keywords.
- **ChromaDB Architecture:** Embedded, in-process vector store running directly inside our FastAPI backend with zero external server overhead.
- **Stored Incident Schema:** 384-dimensional embedding, raw log excerpt, root cause, remediation steps, source URL, and quality label.
- **Initialization:** Built once from the 662 training incidents only and loaded into memory on backend startup; test incidents are never indexed.

### Slide 8: 2-Pass Retrieval & The Relevance Gate
- **Pass 1:** Queries `where={"quality": "high"}` with a mandatory **0.40 cosine similarity floor**.
- **Pass 2 (Fallback):** Expands search across the full 662 training records if Pass 1 misses or scores below 0.40.
- **Relevance Gate:** Prompt instructs the model to inspect the match; outputs `'Precedent: not relevant'` on line 1 if the match is unrelated.
- **Gate Evaluation:** On the 7 test incidents scoring below the 0.40 similarity floor, all 7 triggered the gate (line 1 = 'Precedent: not relevant'); note we did not manually verify whether all 7 were truly irrelevant.

### Spoken Script (Sameera)
"To match an incoming error log against past incidents, we use a vector database. A vector database converts text into a list of numbers—an embedding—capturing its conceptual meaning, allowing semantic similarity search rather than exact keyword matching.

Think of a traditional database like an index matching literal words, while a vector database is a library catalog: 'connection acquisition timeout' and 'database pool exhausted' sit on the exact same shelf because their vectors point in the same direction.

We chose ChromaDB because it is lightweight and runs directly in-process inside our FastAPI backend with no separate database server. Each record stores the 384-dimensional embedding alongside metadata: the raw log excerpt, root cause, remediation steps, source URL, and quality label. This collection is built once from the 662 training incidents only—test incidents are never indexed—and loaded directly into memory when the backend starts.

When an engineer inputs a log, we execute a two-pass retrieval. Pass One queries high-quality incidents with a 0.40 cosine similarity floor. If that threshold is missed, Pass Two falls back across all corpus records.

Finally, our Relevance Gate prevents false analogies. The prompt instructs the LLM to output 'Precedent: not relevant' on line one if the match is unrelated. In evaluation, across the seven test incidents scoring below the 0.40 similarity floor, all seven triggered the gate with 'Precedent: not relevant' on line one, though we did not manually verify if each precedent was truly irrelevant."

**Handoff:** *"Nishit will now demonstrate the system live and show our Pipeline Trace in action."*

---

## 5. Nishit — System Architecture & Live Demo (2:00)

### Slide 9: System Architecture & Live Demonstration
- **Architecture:** Next.js 16 (Bento UI) -> FastAPI on Hugging Face Spaces -> Groq (`openai/gpt-oss-120b`).
- **Live Flow:** Log Input -> Soft-Wrap Editor -> 2-Pass ChromaDB Retrieval -> Prompt Synthesis -> Groq Inference.
- **Interactive Trace:** Matched URL, Cosine Similarity Gauge with 0.40 tick, Relevance Badge, Raw Prompt Inspector.
- **Resilience:** Client-side rate limiting (15 RPM) and SQLite pre-warmed cache protecting strictly against Groq rate limits or provider failures.

### Spoken Script (Nishit)
"*(Action: Transition to console)*  
Here is the TraceMind console. In the Incident Log card, I paste a production error log showing a database proxy pool timeout. The editor soft-wraps long lines while keeping line numbers aligned to logical lines.

*(Action: Click 'Analyze incident')*  
I click Analyze. On the right, the stepper confirms each stage: embedding the log vector, cluster assignment, ChromaDB vector retrieval, and LLM generation via Groq.

*(Action: Point to Root Cause and Remediation)*  
TraceMind synthesizes a root cause explaining that active worker connections saturated the pool, alongside an actionable five-step remediation checklist.

*(Action: Highlight Pipeline Trace and Gauge)*  
Our core differentiator is the Pipeline Trace: retrieval and prompt are fully inspectable. The gauge displays a similarity score of [FILL FROM LIVE RUN: similarity score, e.g. 0.51], safely above our 0.40 threshold, and links to the matched public postmortem. The badge confirms: '[FILL FROM LIVE RUN: e.g. Precedent applied]'. In the Prompt tab, you can inspect the exact prompt delivered to Groq.

If Groq hits its free-tier rate limit or returns a network error, our pre-warmed SQLite cache automatically serves cached completions for example presets, shielding the demo from API failures."

**Handoff:** *"Unique will now review our quantitative evaluation and summarize our findings."*

---

## 6. Unique — Evaluation, Limitations & Future Work (1:45)

### Slide 10: Quantitative Benchmarks (Held-Out Test Set, N = 93)
- **Ablation Study (N = 93):**
  - ROUGE-1 F1: 0.1809 (Zero-Shot) -> **0.1941** (RAG) [**+0.0132**]
  - ROUGE-2 F1: 0.0384 (Zero-Shot) -> **0.0439** (RAG) [**+0.0055**]
  - ROUGE-L F1: 0.1061 (Zero-Shot) -> **0.1210** (RAG) [**+0.0149**]
  - Semantic Cosine Similarity: 0.5743 (Zero-Shot) -> **0.5764** (RAG) [**+0.0021**]
- **Retrieval Distribution (N = 93):**
  - Mean Top-1 Similarity: **0.5273** (Median: 0.5333)
  - Above 0.40 Threshold: **86 / 93 (92.5%)**
- **Relevance Gate Result:** All 7 test incidents scoring below the 0.40 floor triggered the gate (line 1 = 'Precedent: not relevant'); note we did not manually verify whether all 7 were truly irrelevant.

### Slide 11: Honest Limitations & Roadmap
- **Proof-of-Concept Sample Size:** Evaluated on 93 held-out incidents; directionally positive, not a claim of statistical significance.
- **High-Quality Test Set Bias:** Evaluated on 93 high-quality incidents only, making evaluation results optimistic relative to real-world noisy logs.
- **Infrastructure Constraints:** Groq free-tier rate limit (20 RPM) requires client-side throttling.
- **Future Work:** Continuous log streaming (Kafka/Fluentd), local fine-tuned SLM (8B), and automated Jira/PagerDuty ticket sync.
- **Closing Thought:** Reliability is built on transparency, not blind faith.
- **Live Deployments:** Frontend Console: https://tracemind.vercel.app | Backend API: https://nishhhx0-tracemind-backend.hf.space

### Backup Slide 12: Retrieval Category Benchmark (N = 93)
- **Top-1 Category Hit Rate:** 33.3% (31/93) across 9 infrastructure failure classes.
- **Baselines:** Random Chance: 11.1% (1/9) | Majority Class (*Software Bug*): 35.5% (33/93).
- **Category Breakdown:** Strongest on Software Bug (54.5%, 18/33) and Security (50.0%, 3/6); weakest on Hardware (0/3), Third-Party (0/2), and Memory (0/1).
- **Takeaway:** Coarse categorical matching is a weak evaluation signal; true operational utility stems from grounded root causes and concrete remediation steps.

### Spoken Script (Unique)
"To validate TraceMind objectively, we evaluated our held-out test split of 93 high-quality incidents from 40 companies that the system never saw during training.

Comparing TraceMind against an unaugmented zero-shot baseline on these 93 incidents, RAG achieved slightly higher overlap: ROUGE-1 F1 rose from 0.1809 to 0.1941, an increase of 0.0132, while ROUGE-2 and ROUGE-L showed similar modest gains as shown on the slide. Semantic embedding similarity remained essentially flat at 0.576.

In our failure category benchmark, retrieval hit the exact category in 33.3 percent of test incidents. That is slightly below the 35.5 percent majority-class baseline—meaning always guessing software bugs—because category matching is a weak, coarse signal. Our value is not labeling the bucket, but delivering the specific, grounded root cause and verified remediation checklist.

Crucially, on the seven test cases below our 0.40 similarity floor, all seven triggered the gate with 'Precedent: not relevant' on line one, though we did not manually verify irrelevance.

To be clear about limitations: this is an academic proof of concept on 93 test incidents, so we do not claim statistical significance. Furthermore, because our test set contains exclusively high-quality postmortems, results are optimistic compared to messy production logs. Groq's 20-RPM free tier also required rate limiting.

TraceMind proves that during an outage, the most useful AI is not the one that sounds confident, but the one where retrieval and prompt are fully inspectable. Thank you."

---

## 9 Likely Jury Questions & Truthful Answers

### Q1: Why not fine-tune an open-source model (like Llama-3-8B) on all 828 postmortems instead of using RAG?
**Answer:** Fine-tuning was out of scope for a one-semester project, and an 828-record corpus is too small to fine-tune effectively without overfitting. Additionally, fine-tuning bakes historical facts into static model weights, causing catastrophic forgetting and hallucinations that cannot be cited. With RAG, updating the system requires only adding a new JSON postmortem to ChromaDB without GPU re-training, and every prediction provides a direct hyperlink to the retrieved document.

### Q2: If ChromaDB already performs vector retrieval, why is HDBSCAN clustering needed?
**Answer:** Clustering serves as a preprocessing and deduplication filter, not the matching engine. Note that clustering is not included in our quantitative ablation; its practical contribution is providing the cluster ID in the Pipeline Trace and performing burst deduplication so that repetitive error spikes during a service failure do not overwhelm the retrieval pipeline.

### Q3: How do you prevent the LLM from hallucinating when retrieval returns an irrelevant incident?
**Answer:** We implemented a two-part defense. First, a mathematical threshold: if Pass One retrieval similarity falls below 0.40, it is flagged as weak precedent. Second, our `relevance_gate_v2` prompt explicitly instructs the LLM: if the historical case does not clearly match the failure, it must write 'Precedent: not relevant' on line one and synthesize the cause from the log alone. On our test set, all seven incidents scoring below the 0.40 floor triggered the gate with that exact header, though we did not manually verify whether all seven were truly irrelevant.

### Q4: Why use Groq and a massive 120-billion parameter model instead of a local model?
**Answer:** We prioritized inference quality and diagnostic reasoning for complex root causes while keeping the backend lightweight enough to host on a free Hugging Face Space CPU tier. Groq provides dedicated LPU hardware with zero local GPU requirements. To protect against free-tier rate limits (20 RPM), we built an in-memory client-side rate limiter and a pre-warmed SQLite cache.

### Q5: Isn't a 93-record test set too small to prove the system works?
**Answer:** Yes, 93 records is a proof-of-concept sample size, which is why we explicitly state this limitation and do not claim formal statistical significance. Furthermore, because our test set contains exclusively high-quality postmortems, the benchmark results are optimistic compared to real-world operational environments where logs are frequently messy or incomplete. However, the 93 test records represent 40 unseen companies under a strict company-disjoint split.

### Q6: What happens if an incoming log has never happened anywhere in the tech industry before?
**Answer:** The vector retrieval score will reflect low similarity (< 0.40), the UI similarity gauge will label it with its exact frontend string 'Weak match, treat as hypothesis,' and the Relevance Gate will strip the precedent, causing the UI to display the 'No relevant precedent' badge while the model analyzes the stack trace on its own merits without fabricating a false analogy.

### Q7: Why are the ROUGE score improvements modest (+0.013 ROUGE-1)?
**Answer:** ROUGE measures exact n-gram word overlap against human postmortem write-ups. Two engineers describing the exact same Kubernetes OOMKill event may use completely different wording ('pod crashed due to cgroup memory cap' vs. 'payments container killed by Linux kernel OOM'). That is why semantic cosine similarity (0.576 vs 0.574) and ROUGE both show modest quantitative shifts; the primary qualitative value is delivering actionable, grounded remediation steps rather than matching verbatim prose.

### Q8: How does your quality label improve retrieval over vanilla semantic search?
**Answer:** Standard vector search only measures text closeness, meaning a 10-word vague status tweet with high keyword overlap could outrank a 500-word comprehensive engineering report. Our two-pass retrieval forces the search to look at `quality: high` records first, ensuring the LLM is primed with actionable remediation steps rather than brief status notices. Note that the quality filter itself was not isolated in an ablation study; it was designed into the data pipeline based on qualitative inspection of low-quality scraped records.

### Q9: Your category hit rate is below the majority baseline. So does retrieval work?
**Answer:** Category hit rate is a coarse, imperfect proxy for incident diagnosis. In operations, predicting that an outage belongs to the broad category of 'Software Bug' provides zero actionable value to an on-call engineer. The true value of retrieval is delivering the specific historical root cause, architecture component, and five concrete remediation steps from a verified incident, alongside a hyperlink to the postmortem. Evaluating retrieval on coarse categorical tags obscures this utility; our planned future work focuses on human-rated actionability and per-record relevance scoring.

---

## Visuals & Screenshots Guide for Presentation Slides

1. **Slide 1 (Title):** TraceMind geometric logo and team roster.
2. **Slide 3 (Outage Reality):** Visual illustration of the 'Log Firehose' vs. human MTTR timeline.
3. **Slide 5 (Dataset Split):** Sankey diagram showing 828 corpus -> 544 high / 284 low -> 662 train (18 companies) / 93 test (40 companies), with 73 low-quality test-company records discarded.
4. **Slide 6 (Clustering):** 2D UMAP projection visualization of the 48 HDBSCAN clusters.
5. **Slide 7 (ChromaDB Architecture):** Diagram illustrating vector embedding mapping and metadata schema.
6. **Slide 8 (RAG Flow):** Flowchart of 2-Pass retrieval with the 0.40 cosine floor and Relevance Gate decision branch.
7. **Slide 9 (Live Demo):** Full-screen screenshot of the TraceMind console (`docs/pipeline_trace_demo.png`) highlighting the Pipeline Trace and Similarity Gauge.
8. **Slide 10 (Evaluation):** Benchmark table displaying ROUGE gains, Semantic Similarity, and 0.40 threshold pass rates.
9. **Backup Slide 12 (Category Hit Rate):** Category benchmark breakdown table comparing category hit rate (33.3%) against Random Chance (11.1%) and Majority Baseline (35.5%).
