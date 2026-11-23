# Pre-Registration Errata: Directional Sign Requirement for E1 Decisions

**Date:** 2026-11-23  
**Target:** [`eval/PREREG.md`](file:///n:/gitt/TraceMind/eval/PREREG.md) (remains untouched as an immutable pre-registration record).

---

## 1. Context and Problem Statement

In the original decision rules defined in `eval/PREREG.md`, the decision rule for Experiment 1 (Random-Precedent Control) stated:
> "Claim *'retrieval gain reflects precedent content'* ONLY IF `real-minus-random` excludes zero after adjustment. Otherwise report *'inconclusive'*."

This rule omitted an explicit directional sign constraint ($\Delta > 0$). If real-retrieval performs statistically significantly *worse* than random-precedent retrieval (i.e. $\Delta < 0$ and CI excludes zero), the original rule would have mechanically triggered a claim of "precedent content", which is mathematically and conceptually incorrect. A negative delta indicates that the real precedent harmed performance relative to a random precedent, or that an artifact/format confound exists.

---

## 2. Updated Directional Decision Rules

For any analysis referencing this errata, the decision rules for Experiment 1 are amended as follows:

1. **"precedent_content"**:
   - Requires: `real_minus_random` > 0 **AND** company-clustered Bonferroni-adjusted CI excludes zero.
2. **"style_priming"**:
   - Requires: `random_minus_zeroshot` excludes zero after Bonferroni adjustment **AND** `real_minus_random` does not exclude zero.
3. **"precedent_harmful_or_artifact"**:
   - Triggered when: `real_minus_random` < 0 **AND** company-clustered Bonferroni-adjusted CI excludes zero.
4. **"inconclusive"**:
   - All other conditions where neither delta excludes zero after adjustment.

---

## 3. Directional Verdicts on E1 Datasets

Applying these directional rules to the evaluated E1 formats:

* **Gate-v2 Format:**
  * $\text{Random} - \text{Zero-Shot}$: ROUGE-L $\Delta = +0.0036$ (99% CI `[-0.0222, 0.0331]`, *crosses zero*).
  * $\text{Real} - \text{Random}$: ROUGE-L $\Delta = -0.0017$ (99% CI `[-0.0247, 0.0172]`, *crosses zero*).
  * **Original Verdict:** `"inconclusive"`
  * **Directional Verdict (`verdict_directional`):** **`"inconclusive"`**

* **Baseline Format (Initial Run):**
  * $\text{Random} - \text{Zero-Shot}$: ROUGE-L $\Delta = +0.0799$ (99% CI `[0.0528, 0.1153]`, *excludes zero*).
  * $\text{Real} - \text{Random}$: ROUGE-L $\Delta = -0.0657$ (99% CI `[-0.0973, -0.0397]`, *excludes zero*).
  * **Original Verdict:** `"precedent_content"` (flawed due to lack of sign check on negative delta).
  * **Directional Verdict (`verdict_directional`):** **`"precedent_harmful_or_artifact"`** (reflects that random baseline outputs were artificially concise due to prompt mismatch, outscoring real baseline RAG).
