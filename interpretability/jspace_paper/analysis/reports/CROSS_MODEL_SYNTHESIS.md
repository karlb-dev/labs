# CROSS_MODEL_SYNTHESIS.md — A6 output

Source: `tables/cross_model_evidence_matrix.csv` (`scripts/build_cross_model_matrix.py`); every cell carries status, tier, annotation, and evidence ids. Figure: `figures/cross_model_evidence_matrix.{png,pdf}`. There is no scalar workspace score; the matrix exists to make the multidimensionality visible.

## What the matrix shows

1. **Exactly one cell in the campaign is confirmatory + replicated:** Qwen span-safe causal specificity (with its replication column). Qwen additionally holds the only confirmatory mechanism cell (bridge rescue, unreplicated by protocol).
2. **The OLMo story is development-tier organization, not a weaker copy of Qwen's:** the 3.1 pair carries prespecified estimates whose HP3 side replicates while the Think-vs-Instruct interaction does not; Base -> Think -> Instruct carries the training-associated trajectory (C2) with the stage wedge capability-gated.
3. **The transport column is orthogonal to the causal column:** Gemma's only populated cell is a closed methods result; OLMo fails H6 in-band while carrying its development causal cells; Qwen — the causal anchor — was **never transport-gated at all**. Papers must not read column 8 as bounding column 3 (see `A5_H6_PHASE3_RECONCILIATION.md`).
4. **The gated column is uniform:** externalization is gated/blocked everywhere it is defined — the campaign's largest open question is a capability/power boundary, not a null.
5. **Untested cells stay visibly untested** (Qwen transport, OLMo fit ladders, Base/3.0-Think replication): absence of a verdict is a data state, never an implied negative.
