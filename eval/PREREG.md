# Pre-Registered Decision Rules: Experiment 1 (E1) & Experiment 2 (E2)

This document pre-registers the statistical criteria, decision boundaries, sampling constraints, and automated verdict rules for two controlled evaluation experiments prior to executing any model generations.

---

## Experiment 1 (E1): Random-Precedent Control

### Purpose
To isolate whether downstream RAG generation gains stem from actual precedent technical content (domain-specific incident semantics) versus general in-context exemplar/style priming (providing a structured postmortem example regardless of technical relevance).

### Decision Rules
Evaluated separately within each prompt format (`baseline` and `gate_v2`) using two-stage company-clustered bootstrap confidence intervals (10,000 resamples, seed 42) with Bonferroni multiplicity adjustment across evaluated metric endpoints ($m=5$, $\alpha=0.01$, 99% CI):

1. **Claim: `"retrieval gain is exemplar/style priming"`**
   - **Condition:** $\text{random} - \text{zero-shot}$ paired delta CI excludes zero after company-clustered Bonferroni adjustment **AND** $\text{real} - \text{random}$ paired delta CI does **not** exclude zero (crosses zero).
2. **Claim: `"retrieval gain reflects precedent content"`**
   - **Condition:** $\text{real} - \text{random}$ paired delta CI excludes zero after company-clustered Bonferroni adjustment.
3. **Otherwise report: `"inconclusive"`**
   - **Condition:** Neither of the above conditions is met (e.g., both cross zero, or neither satisfies the directional criteria).

---

## Experiment 2 (E2): Leave-One-Out (LOO) Same-Company Regime

### Purpose
To test whether RAG provides actionable retrieval benefits when historical incidents from the *same* organization are available in the index (simulating an enterprise deployment where an engineering team queries their own historical postmortem repository), as opposed to the company-disjoint zero-leakage test set.

### Separation of Regimes
The Leave-One-Out (LOO) regime is an entirely separate evaluation regime from the company-disjoint test set (`data/splits/test.json`, $N=93$). LOO results will **never** be merged, averaged, or pooled with company-disjoint test benchmarks.

### Sampler Constraints & Caps (Fixed A Priori)
- **Source:** Train corpus (`data/splits/train.json` / `data/train.json`).
- **Eligibility:** Restricted to companies with $\ge 5$ `quality: high` incidents.
- **Per-Company Cap:** Maximum 12 incidents per company.
- **Total Sample Size:** Exactly 93 incidents selected using fixed seed 42.

### Near-Duplicate Filter Thresholds (Fixed A Priori)
To prevent trivial copy-paste leakage between near-identical incident reports, retrieved precedents are filtered out if either condition is met:
- **Embedding Cosine Similarity:** $\ge 0.95$
- **5-gram Token-Shingle Jaccard Similarity:** $\ge 0.80$
- **Self Exclusion:** Query incident ID and exact matching `source_url` are strictly excluded from all retrieval conditions.

### Decision Rules
In `gate_v2` format:

1. **Claim: `"RAG helps when same-company history exists"`**
   - **Condition:** $\text{RAG(a)} - \text{zero-shot}$ paired delta CI excludes zero after company-clustered Bonferroni adjustment **AND** remains excluding zero under the near-duplicate filter **AND** remains excluding zero under length-matching (truncating outputs to median ground-truth length).
2. **Otherwise report: `"no detected effect"`**
   - **Condition:** If any of the three criteria fail (crosses zero after Bonferroni adjustment, collapses under near-duplicate removal, or collapses under length-matching).

---

## Immutability Guarantee
All sample selections, thresholds, prompt structures, and decision rules defined herein are permanently locked prior to generation. No thresholds or rules will be modified post hoc.
