# ModelPrint Preregistration

Status: frozen before loading any scientific target model. The addendum in `docs/SPEC_ADDENDUM.md` governs conflicts with the original specification.

## Pre-target amendment 1 — disabled `top_k`

The first non-target `qwen-smoke` port gate on 2026-08-22 observed that pinned vLLM 0.27.1 accepts wire value `top_k=-1` but records the effective `SamplingParams` value as `top_k=0`. Because the protocol requires requested and effective values to agree, the campaign now sends the version's explicit disabled value `0`. The original MP-2 freeze is superseded; no scientific target had been loaded and no target output existed when this adjustment was made. The replacement freeze hashes the request builder, decode constructor, port gate, and generation driver in addition to data/config inputs.

## Scope and unit of identity

The primary task is four-class, closed-set attribution among the exact served profiles listed in `config/models.json`. The label includes repository revision, tokenizer/chat template, parser policy, digest-pinned vLLM image, engine arguments, single-user-turn carrier policy, and explicit decode configuration. Claims do not generalize automatically to a model family, another quantization, another revision, hosted APIs, human text, or unseen models.

The independent split unit is `prompt_group_id`. Relation template aliases are a larger leakage unit. OASST1 is held out at the source level. Test prompt groups, test source rows, and evaluation-only variants cannot enter fitting.

## Primary hypotheses and falsifiers

| ID | Frozen hypothesis | Falsifier / adverse disposition |
|---|---|---|
| H1 | Raw semantic neighbors are dominated by prompt/topic. | Same-model/different-prompt retrieval wins consistently without learned projection. |
| H2 | Deterministic style adds held-out attribution signal. | Style performance does not exceed its permutation null or collapses on carrier/family holdout. |
| H3 | Prompt residualization improves content-controlled attribution. | Residual variants are unstable or fail to improve held-out suites. |
| H4 | Attribution improves with reference-token length. | No monotone length effect, or only a post-selected subset improves. |
| H5 | High-variance decoding weakens the frozen classifier. | High-variance performance is equal or stronger within uncertainty. |
| H6 | Chunk voting helps long and mixed-source outputs. | Whole-output evidence matches or exceeds chunk evidence. |
| H7 | A calibrated text-only hybrid supports useful selective attribution. | Selective gates fail; user-facing attribution is disabled. |
| H8 | SQL ANN preserves exact decisions at useful scale. | Recall or decision-agreement gates fail; exact remains default. |
| H9 | Supervised probes recover signal that semantic kNN misses. | Probes remain at their permutation null. |
| H10 | Cross-likelihood exceeds text-only attribution, especially for short/shifted rows. | Text-only matches or exceeds likelihood across suites. |
| H11 | A learned 64-dimensional fingerprint improves model-aligned retrieval. | Fingerprint kNN fails the permutation/LOFO gates or learns nuisance labels. |
| H12 | Persona instructions obscure text-only model style more than likelihood evidence. | Both channels degrade equally, or neither degrades. |
| H13 | Qwen-smoke paraphrase laundering damages phrase/style more than likelihood. | All channels collapse equally or text-style survives equally well. |

A clean null is a successful experimental outcome. Results cannot be rescued by selecting prompt subsets, metrics, transformations, or thresholds after opening test labels.

## Frozen data roles

- `train`: fit probes, projections, phrase statistics, standardization, medoids, and hybrid parameters.
- `calibration`: fit probability temperature, conformal threshold, novelty threshold, and abstention thresholds.
- `test_id`: primary in-distribution evaluation only.
- `test_source_holdout`: OASST1 distribution-shift evaluation only.
- evaluation-only renderings: persona and RAG-grounded robustness suites; never ordinary training rows.
- human controls: topic-matched Dolly/OASST responses sharing their prompt-group split; never target-class rows.

Nested tier manifests permit one largest-tier generation rotation. Smaller-tier analyses filter the same frozen rows, avoiding repeat model loads.

## Generation contract

Each primary request contains exactly one user message whose bytes are identical across profiles. No system message is used. Each request explicitly supplies temperature, `top_p`, `top_k`, `min_p`, repetition/presence/frequency penalties, seed, maximum tokens, `n=1`, and an empty stop list. The server starts with `--generation-config vllm`.

The four cells are deterministic (`det`), two natural samples (`nat-0`, `nat-1`), and high variance (`hv`). Maximum output tokens are fixed by carrier. The raw HTTP response is appended before derived storage. Final answer, reasoning, truncation, token counts, reference-token count, latency, residue, and self-name flags are retained. Failed attempts remain rows.

## Primary methods and metrics

Frozen methods are class prior, verbosity-only, exact SQL kNN vote, medoid, phrase log-odds, scalar/embedding linear probes, calibrated text-only hybrid, and Bayes likelihood. Frozen representations and detailed data roles are in `config/evaluation-plan.json`.

Headline metrics are accuracy, macro-F1, balanced accuracy, top-2 accuracy, grouped-bootstrap 95% interval, permutation-null mean/95th percentile, adaptive-bin ECE, NLL, Brier score, selective accuracy/coverage, conformal coverage/set size, and per-class confusion. Pairwise, clustering, OOD, retrieval, and ANN metrics follow the governing specification.

A closed-set representation earns a positive claim only if the grouped-bootstrap lower bound exceeds the 95th percentile of 200 within-prompt-group permutations on `test_id` and the leave-one-family-out rotation, while shuffled controls remain null and same-prompt neighbors are excluded. Calibration and coverage gates are frozen in `config/thresholds.json`.

## Text leakage controls

The two headline views are `raw-final-v1` and idempotent `name-masked-v1`. Template residue and explicit model/family/vendor naming fail the target port gate. Exact duplicates are stored once as text artifacts and mapped many-to-many; ambiguous cross-model collisions cannot support attribution. Truncated rows are retained but excluded from headline claims unless the preregistered stratified check shows no effect.

## SQL and ANN policy

SQL exact cosine distance is the state of record for reported test predictions and ANN ground truth. Bulk train-time NumPy neighbors are allowed only after at least 500 exact-query equivalence checks match top-20 ordering and float32 distances within `1e-4`.

Search tables are populated and frozen before index creation. Legacy and version-3 syntax are selected from the runtime capability artifact. ANN metrics are excluded unless query-plan evidence proves index use. Stale-vector-index behavior is forbidden. If exact p95 latency meets the frozen application budget at the largest achieved corpus, the result may be `ANN_UNNEEDED_AT_SCALE`.

## Compute adjustments and drop order

The requested full matrix is 2,500 prompts × four profiles × four decode cells = 40,000 outputs. The immutable job table is created at freeze. If runtime or disk prevents completion, completed cells remain valid and omissions receive `STOP_BUDGET`; no rows are fabricated or imputed. The binding drop order is addendum §P-5. Leakage controls, raw retention, exact ground truth, calibration, permutation nulls, and masked-view comparison cannot be dropped.
