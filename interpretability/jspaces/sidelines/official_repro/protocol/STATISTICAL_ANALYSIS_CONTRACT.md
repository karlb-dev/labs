# Statistical analysis contract (frozen pre-data)

Study-1 tier: development/methods. No confirmatory family, no Holm
transfer, no retrospective power. Addendum §3.5/§3.7 folded in.

## 1. Five-state evidence code (plan §13.1)

`DIRECTION-REPRODUCED` (point estimate in the paper-predicted direction
and the prespecified interval excludes the null in that direction) ·
`DIRECTION-AMBIGUOUS` · `OPPOSITE` · `GATED` (no zero emitted) ·
`NOT-IDENTIFIED`. Claude magnitudes are descriptive context only; no
open-model result is `PASS`ed by exceeding a fraction of a Claude
effect.

Per-row predicted directions are frozen in the preregistration; where
the paper states no direction, the default is the Claude-observed sign
recorded as `direction_source: claude_observed` (addendum §3.5).

## 2. Independent units (plan §13.2)

flexible generalization → category (4; function/argument cells repeated
within) · verbal report → category (14) · probe swap → released category
and item, `multihop` reported as heterogeneous sensitivity ·
selectivity-language → passage/language (8) · selectivity-linecount →
passage (11) · introspection → concept (101; prefill/strength repeated) ·
directed modulation → target/problem or passage (phrasing repeated) ·
dual-task → released pair key (21) · capacity → trial seed × block
family · ignition → concept pair (carrier/α repeated) · top-down
summoning → item (7).

## 3. Intervals and estimates (plan §13.3)

Binary rates: Wilson 95% + cluster bootstrap over units when ≥ 8 units.
Paired binary changes: exact sign-flip test / paired cluster bootstrap.
Continuous rank/log-prob effects: paired cluster bootstrap (B = 10,000,
seed 20260808), equal-family primary weighting; trial-weighted as named
secondary. Small fixed families (FG categories, top-down items): exact
raw counts, no asymptotic theater. Cross-model: model-by-condition
interaction estimated on exact shared token-valid/capable rows, never
compared significance labels. Fit sensitivity: half A, half B, merged,
and half×intervention interaction in every OLMo causal conclusion.

FG honesty (addendum §3.7): with four categories, category-level lane
contrasts are descriptive by construction; the item-level interaction on
exact shared rows carries the quantitative cross-model claim; every FG
figure shows per-category exact counts.

## 4. Three-population reporting (plan §6.7, §9.3)

Every causal cell reports (1) full released-population accounting +
attrition waterfall (gated rows never zero), (2) all-executable
diagnostic estimate over token-valid + geometry-valid rows (never called
intent-to-treat), (3) the capability-conditioned primary where the
released protocol requires a correct baseline. Capability rules frozen
before interventions: verbal-report primary needs no baseline-correctness
condition (the answer is whatever the model produces); FG primary
conditions on source- and target-baseline correctness; probe-swap
primary conditions on baseline == released `answer`.

## 5. Multiplicity

Estimates grouped into named families for transparency; adjusted
secondary p-values may appear as sensitivity only and upgrade nothing.

## 6. Regeneration rule

Every aggregate, figure, table, and prose number regenerates from
committed schemas + registered Parquet/JSON without loading a model
(plan §12).
