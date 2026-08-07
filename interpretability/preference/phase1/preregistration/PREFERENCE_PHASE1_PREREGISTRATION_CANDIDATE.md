# PREFERENCE_PHASE1_PREREGISTRATION_CANDIDATE.md

**Status: CANDIDATE — NOT FROZEN. PI review and approval pending (the
single human gate, addendum §I). Nothing below binds until the freeze
record exists and the tag `preference-phase1-freeze-v1` is laid.**

Everything in this document was fixed before any frozen-partition model
outcome was seen. Development-pilot outcomes (labeled `development`
throughout the repo) informed instrument repairs only, all recorded in
`DEVIATIONS.md`.

## 1. Hypotheses and claim ceiling

Hypotheses P1-H0 … P1-H6b verbatim from plan §2.2. Claim ceiling verbatim
from plan §2.3; the forbidden-phrase list is enforced mechanically by the
language-wall linter on every prose artifact. The preferred outcome is a
clean instrument, including an honest null.

## 2. Frozen inputs

| object | value |
|---|---|
| bank_version | `lab38_v2_phase1` |
| bank content hash | `8d5039af581204a5…` (full value in `data/lab38_preference_bank.meta.json`) |
| bank rows | 2,320 = 960 AR + 480 PC + 160 NC + 720 RO |
| codebook | `cb_final_41eec2d774` — AR `KP4`/`PK7` (neutral-prior gap 0.0031 nats), RO `VM2`/`GS2` (0.0275 nats), leading-space `none` |
| equality review | `agent_dual_code_provisional` x2 passes (hash in registry `pref1-equality-provisional-v1`); **PI ratings land with freeze approval and supersede for the frozen run's license** |
| primary model | `allenai/Olmo-3-7B-Instruct` @ `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, bf16 |
| smoke model | `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f7…` (never scientific) |
| replication model (drop-order gated) | `allenai/Olmo-3.1-32B-Instruct` @ `ac0587e4…` |
| chat-template hash | recorded per run in `diagnostics/model_manifest.json`; string-render == direct-render parity enforced per batch |
| generation | greedy, `do_sample=False`, `num_beams=1`, choice `max_new_tokens=8`, microtask budgets per BindingSpec, explicit `GenerationConfig` (never the shipped one) |
| scoring | full-target-sequence summed logprob, **single-row** (exact; bf16 batched kernels differ by up to 0.25 nats — recorded finding); margins mapped through response codes to content before analysis |
| parser | `strict_exact_code_v1` primary (surrounding whitespace only, exact code, never guesses); permissive parser sensitivity-only |
| invalid policy | invalid = missing (primary); worst-case bounds + permissive sensitivity reported |
| binding | enacted+valid rows execute exactly the parsed branch; hypothetical/RO/invalid never execute (`binding_executed=false` with reason); wrong-branch count must be 0 |
| splits | 5 incidentals per scenario: 3 train / 1 validation / 1 holdout; dev subset = train x both orders x letter labels (DEVIATIONS D7 v2); holdout outcomes stay unopened by any direction-selection code |

## 3. Behavioral decision rules (frozen numbers)

PC gate (plan §6.3): strict parse ≥ 0.98; binding execution among valid
enacted = 1.00; expected-content aggregate ≥ 0.85; every PC scenario
≥ 0.75; expected content wins in both order strata and both label
families; |first-position effect| < 0.10; wrong branches = 0. PC-SAFETY
reported separately; at least one PC-QUALITY scenario must pass the full
pipeline. If PC fails → `PREFERENCE_PHASE1_STOP_PC_FAILED.md`, stop.

Scenario graduation = the ten conjunctive criteria of addendum G2 with
SESOI 0.10, 90% hierarchical percentile bootstrap (10,000 replicates;
incidentals then cells), NC p95 floor, LOIO ≥ 0.05 with sign, nuisance
max < min(0.10, |effect|), margin-sign agreement, invalid-rate diff
< 0.05, train margin variance ≥ 0.10 nats over ≥ 24 finite train cells.
Frame-specific-only effects are reported narrowly and never graduate.
NC scenarios can never graduate; an NC passing criteria 2–7 is a
stop-and-ask instrument alarm (addendum M3).

Aggregate reporting per addendum E1 (no global signed test across
unrelated pole anchors): within-construct signed means for axes with ≥ 2
scenarios (`naming_convention` x3, `execution_mode` x2), |effect|
distribution against the NC floor (exploratory rank comparison), and the
graduated count. Stated/revealed comparison on exact `pair_key` matches;
RO compared separately to enacted-AR and hypothetical-AR. The
consequence-frame factor is reported as an in-context framing effect,
never as real-stakes sensitivity. First-position bias magnitude is a
first-class instrument finding. Exploratory `|RO| − |AR|` contrast
carries an exploratory label only.

## 4. Conditional mechanism (runs only per plan §9.3 stop rules)

Entry: ≥ 2 graduated AR scenarios (1 → case study). Object: scenario-
specific nuisance-residualized margin-covariance direction (plan §10.2;
ridge + top-vs-bottom mass-mean as sensitivity only, selected blind to
holdout). Decision position: final rendered prompt token (E5), captured
at relative depths {0.25, 0.40, 0.55, 0.70, 0.85} (block k writes stream
k+1). Fit on train incidentals; layer + dose selected on validation only;
holdout opened once. Doses: removal α=1.0 primary ({0.5,1.5}
sensitivity); addition β ∈ {1,2}·s (s = train-cell projection SD). Dose
guardrail: mean per-token KL < 0.15 nats over 32 continuation tokens on
16 frozen unrelated prompts. Controls per plan §10.5 with H1's re-signed
nuisance-direction construction; `d_code` IS the direct-output-readout
control. Primary causal endpoint per E14: paired holdout margin shift
(exact sign-flip, 2^16); strict-output flips descriptive. Holm across the
three predeclared mechanism primaries per scenario (AR removal, AR
addition monotonicity, AR→RO transfer). Mechanistic PC (a PC-QUALITY
scenario) must pass before any AR causal claim. No universal preference
vector under any outcome.

## 5. Lineage (optional) and DG (secondary)

Lineage: matched released OLMo endpoints only, capability-gated per plan
§11; runs only if time remains after the primary result; endpoint-
association language only. DG: after the primary report is banked; forced
CONTINUE/STOP/CHANGE is the DV; DG-SAFE stays forward-only (no sampled
generations, no refusal ablation, Lab-7 canonical set); an OLMo free-form
null is a recorded result; DG cannot block closeout.

## 6. Stop and drop rules

Stop-and-ask: addendum §M verbatim (model-pin failure, PC parse < 0.98
after two repairs, NC alarm, any wrong-branch execution, replay failure,
T4-only hardware at a frozen stage, the freeze gate itself, any step that
appears to require DG-SAFE generation / refusal ablation / pooled
directions, PC failure on the frozen run). Drop order: plan §15.4
(proprietary DG PC → DG battery → lineage → 32B → temperature
robustness); never dropped: bank audit, equality audit, PC gate,
counterbalance analysis, strict parser, binding audit, preregistration,
per-item records, held-out causal controls.

## 7. Multiplicity, power, and honesty notes

Per-scenario Holm is infeasible at 5 clusters (E13); family-wise error is
controlled by the conjunctive gates + the NC empirical floor. Power note
(G4): a true p = 0.60 is edge-detectable; p ≥ 0.65 comfortable; the bank
is never enlarged mid-run to chase significance. All pre-freeze outputs
are development evidence. Every headline number must trace to immutable
per-item rows.
