# Evaluation Protocol Deviations & Adjustments

## Deviation 1: LOO Retrieval Candidate Pool Size Expansion (30 -> 100)

- **Date:** 2026-11-22
- **Context:** Experiment 2 (Leave-One-Out Same-Company Evaluation, N=93).
- **Incident Description:** An `IndexError` occurred during Block 6 execution (`eval/loo/run_loo_eval.py`) when attempting to retrieve top-1 precedents under Condition (b) (`cross_company_only`).
- **Root Cause:** In `eval/loo/retrieval.py`, candidate retrieval from ChromaDB was initially executed with `n_results=30`. In `data/splits/train.json`, company `twilio` has 44 incidents. For 6 specific Twilio queries (indices 23, 33, 37, 50, 62, 81), all 30 nearest neighbors in the index were other Twilio incidents. Filtering for Condition (b) (`c_comp != q_comp`) consequently exhausted the entire 30-item candidate pool, leaving zero cross-company hits.
- **Timing of Discovery:** Discovered during generation runtime before any E2 outcome metric, similarity distribution, generation score, or delta was analyzed.
- **Remediation:** Expanded the candidate retrieval pool uniformly across all conditions (a), (b), and (c) to `n_results=100` for both the `quality: high` pass and the all-corpus fallback pass. Added hard assertions ensuring $\ge 3$ valid hits in each condition after filtering and confirming zero leak-guard violations.
- **Impact & Audit:** Documented per-query differences in `eval/loo/retrieval_pool_diff.json`. Condition (a) and Condition (c) top-3 hits remained 100% unchanged (0/93 changed). Condition (b) changed in 7/93 queries (successfully recovering all 6 Twilio queries from 0 hits to $\ge 3$ hits).


## Deviation 2: Invalidation of E1 Baseline Random-Precedent Arm Due to Prompt-Builder Disparity

- **Date:** 2026-11-23
- **Context:** Experiment 1 (Random-Precedent Control on Test Set, N=93).
- **Finding:** In `eval/run_random_precedent_gen.py` (lines 248, 272-286), the baseline-format random arm was constructed using `PROMPT_REGISTRY['baseline']` from `rag/prompts.py`, whose system prompt includes brevity constraints (`<concise, precise analysis...>`, `Do not output internal chain-of-thought...`) and omitted user format instructions. In contrast, `control_2x2` (via `eval/evaluate.py:108, 208-220`) generated `baseline_rag` using a permissive system prompt (`Structure your output into ROOT CAUSE: and REMEDIATION STEPS: sections.`) and explicit format instructions at the end of the user prompt.
- **Status:** The baseline-format random-precedent arm from the initial E1 run is marked **INVALID**.
- **Evidence:**
  1. Prompt hashes differ in 93/93 queries (100%).
  2. Word counts of baseline random arm (mean 54.7, median 55.0) mirror gate-v2 format (mean 40.7, median 36.0) rather than true baseline RAG (mean 196.0, median 187.0) or baseline zero-shot (mean 215.5, median 217.0).
  3. Remediation parse rate was artificially 100% (93/93) compared to 78.5% (73/93) in baseline RAG and 64.5% (60/93) in baseline zero-shot.
- **Offending Code Path:** `eval/run_random_precedent_gen.py`, lines 248, 272-286.
- **Remediation Plan:** Deferred to Block 3 where the prompt builder will be aligned with `eval/evaluate.py`, unit tested, and rerun into a versioned output (`random_precedent_control_v2.json`) without modifying existing committed files.
