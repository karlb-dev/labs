# Released-materials coverage matrix — Study 1

Built from the pinned bytes at `anthropics/jacobian-lens` @ `581d3986`
(vendored under `external/`, hash-pinned in
`external_record_manifest.json`) **before any GPU result**. The pinned
JSON files and READMEs win over the plan's prose; divergences from the
plan's §3.2 initial table are noted inline and logged in
`SPEC_DIVERGENCE_LOG.md`. Machine-readable twin:
`RELEASED_MATERIALS_COVERAGE_MATRIX.json`.

Fidelity classes: **R0** exact-released · **R1** reconstructed-from-paper
· **R2** deterministic-adaptation · **R3** not-identified-from-release.

## Six lens evaluations

| Set | Items | Prompt form | Readout position (pinned README) | Class | Disposition |
|---|---:|---|---|---|---|
| `lens-eval-multihop` | 93 | raw string | token immediately preceding `target` | **R1** | both lanes, paper-grid primary |
| `lens-eval-multilingual` | 107 | raw string | token immediately preceding `target` | **R1** | both lanes |
| `lens-eval-poetry` | 98 | raw string | last newline token (end of couplet line 1) | **R1** | both lanes |
| `lens-eval-typo` | 96 | raw string | final prompt token (last fragment of misspelling) | **R1** | both lanes |
| `lens-eval-order-ops` | 55 | raw string | token immediately preceding `target`; synonym sets (digit+word / symbol+word), rank = min over single-token synonyms | **R1** | both lanes |
| `lens-eval-association` | 102 | raw string | final prompt token (closing period) | **R1** | both lanes |

Class note: prompts and intermediates are exact released bytes and the
scoring rule (pass@k, min-over-layers rank at one position) is fully
specified by the pinned README — but position *location* requires
tokenizer-dependent reconstruction (finding the token preceding a target
substring under each lane's tokenizer), so R1 rather than R0. No
tokenizer-independent cell exists.

## Eleven experiment sets

| Set | Released material (verified counts) | Class | Study-1 disposition |
|---|---|---|---|
| `verbal-report` | 14 categories × 14 candidate words; prompt text + candidate rule in pinned README | **R1** | mandatory causal core. Release-literal candidate rule primary (first ten listed, skip answer); paper-text top-10-output exclusion as named sensitivity. "Final colon" position mapped per `RENDER_AND_POSITION_CONTRACT` §4 |
| `flexible-generalization` | 4 categories × 4 args × 4 funcs (= 192 ordered off-diagonal swaps); all templates + answers released | **R1** | mandatory causal core; α=1 primary, α=2 full sensitivity (paper-sourced) |
| `verbal-introspection` | 4-turn `intro_prompt` (final assistant turn empty), 2 prefills, **101** concepts (paper text says n=100 — divergence D3) | **R1** | mandatory report-channel cell; steering rule fully in pinned README; strength ladder reconstructed (contract §6) |
| `selectivity-language` | 8 passages (fr/de/es/it ×2), task templates, per-language intermediates + author sets | **R0/R1** | released readout contrast only (explicit − automatic label-hit); no invented continuation-equivalence primary |
| `selectivity-linecount` | 11 passages + widths; 3 prefill conditions + `continue`; `textwrap.fill` construction named in README | **R1** | run with frozen model-specific two-digit + number-word canon |
| `ignition` | 12 countries (66 pairs) + 16 alt_words + 12 idiom + 12 scrambled pairs; 40+20 carriers; α-sweep semantics in README | **R1** | after core; embedding interpolation at `{W}`; descriptive nonlinearity result |
| `capacity` | 4 block families with oversized pools (505/603/149/256), `targets_per_family`, proto labels; construction in README (canon is model-dependent **by released design**) | **R1** | after core; frozen trial RNG; distinct from campaign sparse-occupancy — never merged |
| `dual-task` | carrier + 21 pairs + 10 concept-pairs + 4+4 conditions | **R1** | after core; reachability rank ≤5 over response span |
| `top-down-summoning` | 7 items: stimulus, q1/q2, expected/foil sets, released swap pairs | **R1** | after core; Q2−Q1 readout contrast + released label↔foil swaps (generation per README "samples the continuation" → greedy, contract §7) |
| `directed-modulation` math + topic | 24 phrasings (4 groups → focus/suppress/control), 20 carriers, 24 math problems, 22 topic categories | **R1** | after core |
| `directed-modulation` line-break | underlying prose **not released**; README explicitly authorizes "any prose corpus filtered to alpha-heavy ASCII text" | **R2** | one pinned alpha-heavy ASCII corpus (WikiText rows after the fit population, frozen in the preregistration); never called exact |
| `probe-swap` prompt set | 90 items (multihop 29/90, 35 other categories); prompts/intermediates/answers/swap targets exact | prompts **R0**; official probe arm **R3**; raw token-vector arm **R2** (representation-adapted) | primary Study-1 arm = prompt-exact raw J-lens token-vector swap (`prompt_exact_representation_adapted_raw_jlens`; the paper's own §3.3 headline arm). Official linear-probe arm NOT-IDENTIFIED: probes are mean-difference probes over auxiliary prompt sets absent from the release. Optional study-authored probe arm: **declined for Study 1** (first drop under compute pressure; not specified before GPU work) |
| paper broad top-10 ablation | no released experiment JSON | **R3** | campaign cross-over (OR1.6) is explicitly non-official |

## Corrections to the plan's §3.2 initial table

1. `selectivity-language` initial class "R0/R1" retained but the executed
   metric is R1 (template fill + tokenizer-dependent label tracking).
2. `probe-swap` prompt bytes are R0 as stimuli; the plan's "R2/R3" row
   conflated arms — this matrix splits them (raw arm R2, probe arm R3).
3. `verbal-introspection` released concept count is 101, not the paper's
   100 (logged D3; all 101 run; no row dropped to match the paper).

## Population accounting rule

Every released row receives explicit `source_present`,
`tokenization_valid`, `geometry_valid`, `baseline_capable`, `executed`
states. Gated rows are **missing, never zero** (plan §6.7, §13.1).
