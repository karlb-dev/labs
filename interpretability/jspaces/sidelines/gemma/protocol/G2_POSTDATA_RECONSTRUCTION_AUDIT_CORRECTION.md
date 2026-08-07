# G2.1 target-blind reconstruction-audit correction

Status: `FROZEN_AFTER_G2_1_ROWS_BEFORE_THRESHOLD`, 2026-08-03. Tier: methods/development. Raw producer commit: `165e072b942ad06ba325b91811533e01a647be3f`.

The complete G2.1 backend-only grid existed, but the first `freeze` invocation stopped before creating a Parquet table, summary, figure, threshold, or registry event. No Stage-1 target artifact or outcome had been opened. The stop was caused solely by an analysis audit that compared two numerically different cosine accumulation definitions with an unjustified generic tolerance:

- producer summary: PyTorch float32 `cosine_similarity` over the concatenated all-slot tensor;
- reconstruction: float64 sum of the saved per-slot dot products and squared norms.

Every relative-error, maximum-difference, selected-slot, dtype-quantum, primal, deterministic-replay, row-count, and row-order check passed. The largest all-slot cosine residual between the two accumulation definitions was `9.226683939211888e-06` over 40,960 elements. This discrepancy is reduction-order rounding, not row drift; it is also hundreds of times smaller than the frozen path-ambiguity cosine margin (`1 - 0.995`).

The reconstruction audit is corrected to use the dimension-derived bound

```text
8 * float32_epsilon * ceil(log2(number_of_elements))
```

for this float32-versus-float64 cosine comparison only. All other reconstruction tolerances remain unchanged. The raw rows, stored summaries, config, scientific grid, ceiling formula, bootstrap, router, predictions, and firewall remain byte-for-byte unchanged. The eventual threshold and event must record both this analysis commit and the original raw producer commit.
