# Evaluation Protocol Deviations & Adjustments

## Deviation 1: LOO Retrieval Candidate Pool Size Expansion (30 -> 100)

- **Date:** 2026-11-22
- **Context:** Experiment 2 (Leave-One-Out Same-Company Evaluation, N=93).
- **Incident Description:** An `IndexError` occurred during Block 6 execution (`eval/loo/run_loo_eval.py`) when attempting to retrieve top-1 precedents under Condition (b) (`cross_company_only`).
- **Root Cause:** In `eval/loo/retrieval.py`, candidate retrieval from ChromaDB was initially executed with `n_results=30`. In `data/splits/train.json`, company `twilio` has 44 incidents. For 6 specific Twilio queries (indices 23, 33, 37, 50, 62, 81), all 30 nearest neighbors in the index were other Twilio incidents. Filtering for Condition (b) (`c_comp != q_comp`) consequently exhausted the entire 30-item candidate pool, leaving zero cross-company hits.
- **Timing of Discovery:** Discovered during generation runtime before any E2 outcome metric, similarity distribution, generation score, or delta was analyzed.
- **Remediation:** Expanded the candidate retrieval pool uniformly across all conditions (a), (b), and (c) to `n_results=100` for both the `quality: high` pass and the all-corpus fallback pass. Added hard assertions ensuring $\ge 3$ valid hits in each condition after filtering and confirming zero leak-guard violations.
- **Impact & Audit:** Documented per-query differences in `eval/loo/retrieval_pool_diff.json`. Condition (a) and Condition (c) top-3 hits remained 100% unchanged (0/93 changed). Condition (b) changed in 7/93 queries (successfully recovering all 6 Twilio queries from 0 hits to $\ge 3$ hits).
