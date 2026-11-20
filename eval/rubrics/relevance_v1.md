# Incident Retrieval Relevance Rubric (v1)

Given an incoming infrastructure incident query and a retrieved historical incident precedent, assign an integer relevance score based on technical failure similarity:

- **2 = same failure mechanism (same fault class and component, e.g. thread pool exhaustion, BGP route leak, split-brain)**
- **1 = related (shares symptom or subsystem, e.g. both are database failovers, but the fault differs)**
- **0 = unrelated (different symptom, different subsystem, or uninformative)**

Output must be strict JSON: {"score": int, "rationale": str} with rationale <= 25 words.
