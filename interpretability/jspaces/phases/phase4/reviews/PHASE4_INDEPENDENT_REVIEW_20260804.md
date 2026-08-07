# Phase 4 independent review — 2026-08-04

**NARRATIVE-BLIND PROTOCOL REVIEW — NUMBERS RECONSTRUCTED FROM REGISTRY,
FROZEN CODE, CONFIGS, AND REGISTERED OUTPUTS BEFORE ANY PACKET COMPARISON**

## Required header fields

```text
reviewer identity or session identifier: Claude general-purpose review
  subagent, session 2026-08-04, narrative-blind per protocol
review date: 2026-08-04
source commit reviewed: 8762a9a57cd67044b6520ee9b7a17c57aec50792
  (branch interp_jspace_phase4_5; clean tree at review start)
materials supplied:
  - interpretability/jspaces/phases/phase4/reports/evidence_events.jsonl (82 rows,
    resolved via `python -m jspace_phase4 registry-list` and raw parsing)
  - frozen producer/analysis code under jspace_phase4/ and configs under
    configs/ (notably p4_qwen_branch_router.py,
    p4_qwen_canonical_lens_decision.py, p4_qwen_multilens_functional_gate.py
    ql_branch_from_gates, and the frozen threshold blocks in
    p4_qwen_multilens_functional_gate_a500_a1000_dev.yaml and
    p4_qwen_lens_influence_prompt323_dev.yaml)
  - registered outputs at their registered Drive/repo paths, hash-verified
    with sha256 before use
  - tests/ (full suite executed), `python -m jspace_phase4 verify` (executed)
  - durability/inventory artifacts: reports/PHASE4_PART5_DURABILITY_FRESH.json,
    manifests/phase4_pre_freeze_inventory_v4_5.json,
    reports/PHASE4_PART5_FOUNDATION.json,
    protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json,
    protocol/PRE_FREEZE_INVENTORY_POLICY_PHASE4.json,
    reports/PHASE4_PART4_DURABILITY_PASS1.json / PASS2.json
  - registered methods records: reports/PHASE4_RUNTIME_IDENTITY_SYNTHESIS.md,
    reports/PHASE4_PART4_PROMPT323_RUNTIME_BLOCK.md,
    reports/PHASE4_PART4_FUNCTIONAL_INSTRUMENTATION_INCIDENT.md,
    preregistration/PROMPT323_RUNTIME_CONTRACT_AMENDMENT.md,
    preregistration/SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md,
    preregistration/FREEZE_GATE_LEDGER_PHASE4.md,
    reports/TERMINAL_B_PRECOMMIT.md
  - side registries and release files: jspace_gemma/{reports,release},
    jspace_olmo_lineage/{reports,release}, claim ledgers
    gemma_transport_claim_ledger_v2.md and OLMO_LINEAGE_CLAIMS_TABLE_V2.md
  - review packet (read only AFTER independent reconstruction):
    PHASE4_FREEZE_REVIEW_INDEX.md, QWEN_A1000_CANONICAL_REVIEW_PACKET.md,
    P4_P2_PRODUCER_REVIEW_PACKET.md, BANK_B_ORTHOGONAL_REVIEW_PACKET.md,
    PHASE4_PERMANENT_DEFICIT_REVIEW_PACKET.md,
    PHASE4_SIDELINES_STUDY2_ADMISSION_REVIEW.md,
    READY_FOR_PHASE4_FREEZE_REVIEW.md, PHASE4_PI_DISPOSITION_TEMPLATE.md,
    A120_A250_STATE_ARCHIVAL_DISPOSITION.md,
    A120_STATE_EXACT_COPY_SEARCH_20260802.md
narrative-blind declaration: I did NOT open
  interpretability/jspaces/phases/phase4/paper/ (any file),
  reports/PHASE4_DEVELOPMENT_REPORT.md, reports/INPROGRESS_*.md,
  reports/RESUME_*.md, reports/PHASE4_PART4_EXECUTION_RECORD.md,
  reviews/jspace_lab_nextsteps_*.md, or any plan/addendum/inprogress/
  resume/paper_analysis document under /content/drive/MyDrive/interpret/
  (their names were unavoidably visible in directory listings; none was
  opened). Every decisive number below was first reconstructed from
  registry + frozen code + configs + hash-verified registered outputs and
  only then compared against the packet tables.
```

## Check results

### 1. Q-L4 is the unique output of the frozen branch table — **PASS**

Frozen table (`ql_branch_from_gates`, p4_qwen_multilens_functional_gate.py
lines 330–348): if any of {occupancy, centered_excess, span_safe_specific,
tail_rate, g4, bridge_rescue, bridge_preference} fails → Q-L4; else if
structural unstable → Q-L5; else if normalized_selected_span_overlap fails →
Q-L3; else if selected_id_jaccard fails → Q-L2; else Q-L1.

Registered gate inputs (functional_gate_result.json, SHA-256
`7625ae1f…af71a` verified at its registered Drive path): occupancy T,
centered_excess T, span_safe_specific T, tail_rate T, g4 T,
bridge_preference T, **bridge_rescue F**, normalized_selected_span_overlap F,
selected_id_jaccard F; structural gate verified-live with
all_structural_gates_pass = true. I re-executed the frozen
`ql_branch_from_gates` on these registered booleans: **Q-L4**, equal to the
registered `branch_candidate`. Uniqueness: bridge_rescue fails for two
independent reasons (|−0.294028| > 0.25 and a rescue sign flip: A500 mean
−0.129414 vs A1000 mean +0.164614), so rule 1 fires regardless of the
structural bit (structural_stable=False also yields Q-L4); the counterfactual
bridge_rescue=True yields Q-L3 (sparse gates fail), so no admissible reading
of these inputs reaches Q-L1/Q-L2. The registered decision
(canonical_lens_decision.json, SHA-256 `549a3f42…082b` verified) records
branch Q-L4, canonical_lens null, p4_p2_status
blocked-by-decision-sensitive-causal-endpoints, QL2 amendment
discarded_unused — exactly the frozen `actions[Q-L4]` row of
p4_qwen_canonical_lens_decision_a1000_dev.yaml. The earlier A250–A500
continuation was also mechanical: I ran the frozen read-only router
(`p4_qwen_branch_router`, ROUTES A/B/C frozen) live against the registered
A250–A500 gate; it hash-verified the result and emitted branch B →
`draw_a_n1000`. All five source hashes in the decision's input manifest
match the registered events (a1000 lens `6e48c773…f6bd6`; fit checkpoint
`fd5a4ae6…bf20` verified inside the hash-verified lens_fit_result.json).

### 2. Structural pass and sparse/causal failures use registered metrics and thresholds — **PASS**

From convergence_result.json (SHA-256 `eaf8a63e…193d` verified),
comparison `a500_vs_a1000`, band `assay_L20_L44`, reconstructed conservative
values = min over the three task strata of the layer-median statistic:

- task q50 = **0.9987020492553711** (min of 0.9989815 / 0.9987276 /
  0.9987020) ≥ frozen 0.95 → pass
- task q05 = **0.9981223940849304** (min of 0.9982159 / 0.9981382 /
  0.9981224) ≥ frozen 0.90 → pass

From the functional pair `a500_vs_a1000` against the thresholds frozen in
p4_qwen_multilens_functional_gate_a500_a1000_dev.yaml (frozen before A1000
existed; sole permitted edit was binding the A1000 SHA-256):

- selected-ID Jaccard median = **0.5384615384615384** (= 7/13) < 0.75 → fail
- normalized projector overlap median = **0.7098178863525391** < 0.85 → fail
- bridge-rescue difference = **−0.2940282352268696 nat**, |·| > 0.25 → fail
  (additionally sign-discordant: −0.129414 vs +0.164614)
- aggregates all inside tolerance: occupancy diff 0.0 ≤ 1.0 (L24/32/40);
  centered excess ≤ 0.0234 pp ≤ 1.0 pp; span-safe-specific mean diff
  0.0722098 ≤ 0.15 nat; tail-rate diff −0.0333333, |·| ≤ 0.05; g4 flip
  diff 0.0 ≤ 0.10 with both per-lens g4 passes; bridge-preference diff
  0.0158048 ≤ 0.25 with concordant signs.

I recomputed every gate boolean from these registered metrics using the
frozen `_functional_gates` arithmetic; all nine match the registered
booleans. Envelope payload_sha256 self-check passes.

### 3. Selection-margin strata do not rescue the canonical instrument — **PASS**

From selection_margin_result.json (SHA-256 `8a85a715…46e3` verified):
n_positions **17,381**, all retained; strata **near_tie 15,536 /
stable_core 1,845 / rank_deficient 0** (sum = 17,381). Even the most stable
stratum fails the frozen floors: stable_core Jaccard median **0.6666667**
< 0.75 and projector overlap median **0.8005522** < 0.85. The stable-core
threshold curve medians (0.5385–0.5833 across thresholds 0.0001–0.05) never
reach 0.75. All seven contract verdicts equal the frozen
`selection_margin_required_verdicts` (including behavioral_columns_used =
false and all_strata_retained_in_functional_gate = true);
functional_branch_candidate = Q-L4 matches; 52,261 manual lexical review
rows are recorded as behavior-blind and non-decisional.

### 4. Prompt-323 influence negligible under frozen materiality, correctly scoped — **PASS**

From influence_result.json (SHA-256 `a5359b02…efb15` verified), bound to
the registered lens hashes (a500 `84404956…`, a1000 `6e48c773…`):

- decision **negligible** ∈ frozen allowed set; prompt retained
  unconditionally (no trim/refit path exists in the frozen contract).
- materiality vs frozen thresholds (config
  p4_qwen_lens_influence_prompt323_dev.yaml = payload thresholds):
  a1000 median disagreement 1.252e-06 vs 0.02; q05 2.444e-06 vs 0.05;
  identity-adjusted matrix 1.609e-06 vs 0.03; a500 5.186e-06 / 9.775e-06 /
  6.020e-06 vs the same. Worst margin = 0.02 / 5.186e-06 ≈ **3,857-fold**
  below threshold (packet's ">3,800-fold" confirmed).
- current-runtime repeatability: primary max ‖J‖/√d = **181.77661786564192**,
  discarded-diagnostic repeat = **181.77742309031646**, worst per-layer
  normalized repeat difference **0.00457203840613829** ≤ frozen 0.5.
- historical runtime honestly failed and is non-gating: logged 173.345 vs
  current 181.7766, |diff| = **8.4316** > 0.5, role "reported-non-gating";
  `historical_runtime_reproduction_claimed=false`; scope
  `current-runtime-sensitivity-shape-only`. The prospective amendment
  PROMPT323_RUNTIME_CONTRACT_AMENDMENT.md hashes to
  `bac298d5…dac2b` = the payload's precommit pin, identical at commit
  92831d31e37e… (2026-08-03 16:00:49 Z), which predates the influence event
  (16:08:53 Z) and the decision (16:11:00 Z). The two blocked null attempts
  are preserved unpromoted: both analysis_state.json files are 1,146 bytes,
  SHA-256 `4eef3124ae3259bf…`, contribution null, zero completed layers —
  matching the registered block record. Equal-weight contract passes (tiny
  direct refit max abs err 8.94e-08 ≤ 1e-06; adjacent-checkpoint assertion
  n=195→198 max rel Frobenius err 0.0016 from the registered prompt-112
  source). The decision could not have been influence-sensitive: the frozen
  decision code accepts all three influence labels and retains the prompt
  under each.

### 5. P4-P1, P4-P2, P4-P3 dispositions follow mechanically — **PASS**

- **P4-P1 estimation-only:** registered bank_b_power_dev_v1.json — power
  **0.038** at the 0.25-nat joint SESOI with the current 10-family
  confirmatory side (target 0.8; conservative SD 6.0 nats basis);
  bank_b_design_feasibility_dev_v1.json — optimistic known-SD minimum
  **3,562 families**, **124** even at the consumed Phase 3 bridge-specific
  mean 1.342 nats; no reallocation of 40 families can repair it. The one
  permitted orthogonal feasibility shot was hard-conditioned on a Q-L1/Q-L2
  canonical instrument, so under Q-L4 it is not applicable and no geometry or
  outcome event exists (verified: no such event in the registry, no output
  directory).
- **P4-P2 removed under Q-L4:** the frozen action table maps Q-L4 →
  `blocked-by-decision-sensitive-causal-endpoints`, canonical_lens null. The
  pilot GPU config still contains `BIND_REGISTERED_CANONICAL_DECISION_SHA256`
  / `BIND_REGISTERED_A1000_SHA256` placeholders and
  `permitted_branches: [Q-L1, Q-L2]`; its output directory
  `metrics/qwen36-27b-mode-variance-pilot/` exists as an empty scaffold with
  **zero files**. No pilot, power execution, or intervention outcome exists.
- **P4-P3 blocked at 16/20:** reconstructed from the three hash-verified
  bank_w_capability_result.json files: Qwen capable families **20**/24
  (accuracy 0.83333 at both loads, paired diff 0.00000, CI90
  [−0.020833, +0.020833] ⊂ [−0.08, +0.08]); OLMo Think **17**/24; OLMo
  Instruct **17**/24; exact three-way intersection of the registered
  capable_family_ids = **16** (list reproduced), < frozen floor **20**. The
  floor itself is mechanical: bank_w_power_dev_v1 minimum conservative power
  at SESOI 0.15849625 nat = **0.703 / 0.806 / 0.858** at 16/20/24 common
  families (student-t ruler); the Holm-2 audit found the planned 24-family
  side at **0.7996 < 0.80** (first passing count **28** at 0.863; type-I
  0.0234–0.0264 inside [0.0125, 0.0375]) and an append-only correction event
  explicitly un-rounded 0.800 → 0.7996 to keep the failure visible; v3
  therefore superseded v2 without opening an outcome. The joint imported
  replay event records n_joint_common_capable_families **16**, minimum
  **20**, baseline_capability_ready false. No eligible model was dropped to
  manufacture support.

### 6. No confirmatory or replication intervention outcome anywhere — **PASS**

See `reviews/PHASE4_UNTOUCHED_DATA_AUDIT.md` (this session) for commands.
Summary: zero events in any of the three registries carry tier
phase4-confirmatory / phase4-replication (or any confirmatory/replication
tier natively; the sole "confirmatory" string is the Phase 3 boundary import
tier `phase3-confirmatory-import`); the raw phase4 registry contains 0
occurrences of either forbidden tier string; the full Drive run-root metrics
sweep (296 files) leaves no unaccounted intervention outcome; the P4-P2
producer directory is empty; no A2000/n2000 artifact exists anywhere in
registry, configs, code, or run root; the canonical output directory
contains exactly the three registered decision artifacts; the untouched
Bank-B and Bank-W confirmatory/replication partitions appear only as sealed
partition hashes. Full phase4 test suite: **302 passed, 0 failed**.

### 7. Study-2 imports preserve source tiers and forbidden meanings — **PASS**

Reconstructed and verified byte-for-byte:

- Gemma: bundle `9ef48b8a…f5b0` and markdown `547da552…b315` hash-match the
  release files; payload_sha256 `1751d22a…c731a` recomputes canonically;
  frozen prefix `2a144bcf…d30a` hash-matches and is an exact byte-prefix of
  the live gemma registry (36,279 of 43,090 bytes; single appended event =
  the source bundle-creation event `gm2-sidelines2-import-bundle-v1`).
- OLMo: bundle `c213dc74…b6c`, markdown `9e1d756c…e989`, payload
  `b5cfdf92…5be3`, frozen prefix `0a8973e0…fb58` all verified; byte-prefix
  of the live olmo registry (86,627 of 93,246 bytes; single appended event =
  `ol2-sidelines2-import-bundle-v1`).
- Native tiers: all 8 admitted Gemma events tier=methods; all 12 admitted
  OLMo Study-2 events tier ∈ {methods, development}; the 5 earlier admitted
  bank-W events likewise; none superseded or withdrawn inside its frozen
  prefix.
- No native `gm-*`, `gm2-*`, `ol-*`, `ol2-*` evidence_id exists as an event
  in the phase4 registry (0 hits); such IDs appear only inside
  source_evidence_id(s)/inputs metadata of the import events, as required.
- Both Study-1 imports (`p4-import-gemma-transport-v1`,
  `p4-import-olmo-lineage-final-v1`) remain live with exactly one event
  each; the Study-2 events declare them as dependencies.
- Imported meanings sit inside the side ledgers: the Gemma meaning maps onto
  GM2-C04/C05/C06/C09 with the GM2-C10–C15 prohibitions carried as
  forbidden uses; the OLMo meaning maps onto OL2-C02/C03/C04/C06/C07/C08/C10
  with the prohibited formulations carried. Numbers in the meanings
  (ceiling 0.07870368901355948; all-slot error 0.0024581113830208778; L56
  ε 0.10; 0.7788 / 16 / 18) all reconstruct from the hash-verified source
  outputs (below).

### 8. Gemma V2 relicensing does not erase the historical blocker — **PASS**

`gm-jvp-gemma-backend-parity-v1` (2026-08-02, tier methods,
`backend_parity_pass: false`, `stage1_mismatch_reproduced: true`) has exactly
one event in the gemma registry — never superseded, corrected, or withdrawn —
and lies inside the frozen prefix whose bytes I verified are an exact prefix
of the live registry, so the blocker's bytes are unchanged since the freeze.
The relicense is a separate later event `gm2-stage1-relicense-v1`
(`selected_branch: branch_1_relicense_without_recompute`,
`model_compute_performed: false`, `historical_rows_preserved: true`,
applicable ceiling 0.07870368901355948 from the target-blind calibration
event with `ceiling_frozen_before_registry_read: true`). Ledger row GM2-C09
states the Study-1 stop remains historically correct; the phase4 import
meaning repeats this. Nothing retroactively passes the 1e-5 gate.

### 9. OLMo wedge effects are missing, not zero — **PASS**

From the hash-verified stage results: Think-SFT 972 rows, capable rate
**0.006172839506** (Bank-S 0.008333, Bank-S direct+composed fully-capable
facts **0**); Think-DPO 972 rows, capable rate **0.003086419753** (Bank-S
0.002778, fully-capable facts **0**). The joint analysis `effects` block is
`estimable: false` with every effect field **null** and the explicit reason
"intervention effects and adjacent effect contrasts are missing, not zero";
router route `null_or_unresolved` with
`capability_gated_effects_are_missing_not_zero: true`. The phase4 import
carries `stage_effects_missing_not_zero: true` and forbids "treating
capability-gated intervention effects as zero". No wedge cohort effect
number exists anywhere to be misread as a zero.

### 10. H6 failure is not restated as invalidation of paired ablation evidence — **PASS**

From the hash-verified transport_joint_result.json: in-band (L24/L32/L40)
passing layer-epsilon cells = **0** on both checkpoints
(`in_band_pass_on_frozen_ladder: false`); Think passes only **L56 at
ε 0.10 (12/12 rows)**; Base's best late cell is L56/0.10 at **9/12 = 0.75 <
0.90** (no pass); route
`h6_fail_in_band_with_checkpoint_specific_late_anchor`; dose coverage
`null` with the licensed wording "missing coverage is not zero coverage".
The joint claim_boundary states it "neither invalidates paired ablations nor
identifies a training-objective effect"; the ledger's required distinctions
and the import's forbidden uses both prohibit "transport failure invalidates
the paired ablation result". No document in the freeze packet restates H6 as
invalidation.

### 11. The Bank-W pair result is a power/design closure, not an externalization null — **PASS**

From the hash-verified ol2_bank_w_olmo_pair_power_v1.json: shared capable
families **16** (Think 17 ∩ Instruct 17, reconstructed from the registered
capable_family_ids lists), power at the frozen SESOI **0.7788** vs target
0.80, minimum powered count **18**, route
**not-powered-at-current-support**, outcome-blind, all four type-I scenarios
pass. Ledger OL2-C10 licenses exactly this planning wording; "The OLMo pair
rules out externalization" is an explicitly prohibited formulation, and the
import forbids "treating the pair power near-miss as an intervention result
or scientific null". The freeze ledger presents it as "planning closure,
not a null" — within license.

### 12. The fresh durability proof contains only the known permanent deficit — **PASS**

reports/PHASE4_PART5_DURABILITY_FRESH.json (pass
`phase4-part5-fresh-materialization`, 2026-08-04 17:49–17:52 Z):
**521 references, 520 verified, exactly 1 failure**, and that failure is
byte-identical to the declared known deficit: the A120–A250
`state.json` with expected SHA-256
`361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8`,
`known_deficit: true`, `n_unexpected_failures: 0`,
`path_pin_conflicts: []` (zero). Resolution modes: 515 literal-path, **3
repository-materialization, 2 append-only-registry-prefix**. I spot-verified
beyond requirement: all 3 repository-materialization rows resolve to
hash-exact bytes in the current repo (joint bank-W JSON `ffe1ca8b…`, p4f30
png `70c56f90…`, p4f30 pdf `cb403e2e…`), and both registry-prefix rows
resolve exactly — `git show` at the pinned commits (gemma `fddab685…`, olmo
`6ac62d2c…`) reproduces SHA-256 `acd78247…` / `7cb7ece4…` and each blob is a
byte-prefix of its live registry. The one failure's path is genuinely absent
on the mount and its expected hash equals the original event's registered
output hash and the KNOWN_DURABILITY_DEFICITS entry. Explicit limitation
noted (non-blocking): the durability snapshot binds registry state at 81
rows (`20124c4c…`, which I reproduced as the SHA-256 of the current file's
first 81 rows); one later methods event (`p4-runtime-identity-synthesis-v1`,
single repo-markdown output) postdates it — I hash-verified that output
myself (`68740722…`), and my own full `python -m jspace_phase4 verify` run
at HEAD covered all 82 rows / 522 output references with **exit code 1 and
exactly the one known failure**. The earlier same-mounted Part-4 passes
(418/419, one known, zero unexpected) also reconstruct from their JSONs.

### 13. Registered figures/tables in the freeze packet match their registered hashes — **PASS**

The strongest available check was executed: `python -m jspace_phase4 verify`
at HEAD re-hashed every registered output of every live event (522
references, including all p4f figures, tables, parquet row stores, and the
review-packet-cited artifacts) with the single known state.json miss as the
only failure. In addition I independently re-hashed ~40 specific artifacts
cited in the packets, including: the six canonical-chain result files, the
A1000 lens-fit result and checkpoint pin `fd5a4ae6…bf20`, all 16 surviving
A120–A250 outputs (16 verified / 1 missing / 0 mismatched — matching the
deficit packet's table), the A120-A250 and A250-A500 trend results
(0.67479/0.53846; 0.70302/0.53846 with bridge diff 0.558909 fail and
span-safe-specific −0.024323 non-equivalent-TOST — matching the packet's
informational trend table), both Study-2 bundles and prefixes, and the two
Drive p4f14 figure copies (byte-identical to the registered repo files).
No hash mismatch was found anywhere.

### 14. The proposed terminal prose does not exceed any claim ledger — **PASS WITH EXPLICIT LIMITATION**

The terminal prose in READY_FOR_PHASE4_FREEZE_REVIEW.md,
FREEZE_GATE_LEDGER_PHASE4.md, and the preregistration candidate stays inside
the reconstructed evidence: Q-L4/no-instrument wording matches the frozen
branch semantics; "divergence between strong averaged-operator convergence
and unstable sparse/causal invariance" is exactly what q50/q05 = 0.9987/
0.9981 vs Jaccard 0.5385 / overlap 0.7098 / bridge −0.2940 support; the
prompt-323 result is everywhere scoped current-runtime-only; Study-2
sentences match the V2 ledgers including the mandatory "missing, not zero",
"late anchor is not an in-band pass", "unavailable is not zero coverage",
and "planning closure, not a null" qualifiers; no workspace/mechanism/
externalization claim appears; zero-Holm-family and no-alpha-transfer
statements match the registry (zero opened tests). Limitation — three
documentation-level errata, none affecting a number, hash-binding, or
verdict (see Discrepancies).

## Discrepancies found vs the packet (all documentation-level)

1. **QWEN_A1000_CANONICAL_REVIEW_PACKET.md "Event / commit" column is
   misattributed in four rows.** Registered code_commits are: functional
   `dbd54a36…` (packet shows `603bbcec…`, which is the margin event's
   commit); margin `603bbcec…` (packet shows `0ce3519b…`, the sealed-queue
   commit of the *withdrawn* functional attempt); influence `92831d31…`
   (packet shows `01236a3…`, the decision's commit); decision `01236a3f…`
   (packet shows `28e25fe…`, the import_code_commit of the next event,
   `p4-import-olmo-bank-w-capability-v1`). The bound RESULT hashes in the
   same rows are all correct and are what the decision code actually
   enforces, so this cannot change any outcome; it should still be corrected
   before freeze.
2. **READY_FOR_PHASE4_FREEZE_REVIEW.md cites the stale inventory payload**
   `0fbd2d4d…` (v4_4) although the current inventory is v4_5 with payload
   `71ae6031…` (recomputes canonically; same gate outcomes,
   NOT_REVIEW_READY solely from the known deficit). Its "Fresh independent
   rematerialization: BLOCKED" row also predates the Part-5
   fresh-materialization pass; the freeze ledger's corresponding row says
   PENDING. Treat the ledger as governing.
3. **Test-count prose is stale relative to HEAD:** the preregistration
   candidate says 300/300 and the Part-5 foundation snapshot recorded
   287; the suite at the reviewed commit passes **302/302**. Counts grew
   append-only with later test additions; no test fails.

## Required verdict fields

```text
permanent-deficit decision recommendation: ACCEPT WITH EXPLICIT LIMITATION
  Basis: the missing object is a single operational timing/peak-allocation
  state file, not a scientific table; its expected hash is declared in an
  append-only methods event and in the deficit register; the exhaustive
  negative search (Drive live/trash/revisions, DriveFS metadata, mirrors,
  repos, local staging) is recorded; no synthesis or surrogate was created
  (state_json_reconstructed=false); the companion capacity artifact was
  restored to exact registered bytes (6b0399df… hash match); the event's
  decision role is superseded by two fully durable later live gates that the
  Q-L4 chain actually reads; every durability output keeps the gate red
  rather than averaging the deficit away. Required limitation to carry into
  the release record: the A120–A250 functional event is citable only as
  partially durable, and the deficit line must remain in the release
  manifest and known-limitations file verbatim.

Terminal C recommendation: ACCEPT
  Basis: Q-L4 reconstructs mechanically and uniquely from prospectively
  frozen gates; no canonical sparse instrument is licensed; P4-P1
  (0.038 power), P4-P2 (Q-L4 removal before pilot), and P4-P3 (16/20)
  dispositions are forced by registered pre-outcome rules; the realized Holm
  family is empty with no alpha transfer; the untouched confirmatory/
  replication partitions are verifiably untouched; the Terminal-B/C
  indifference was precommitted (6f23a298, 2026-08-03 02:19:57 Z, ancestor
  of HEAD, predating the functional outcome); side admissions are
  methods-only within their ledgers. Conditions attached: fix discrepancy
  (1) above before the freeze commit; the prompt-323 current-runtime-only
  limitation and the permanent-deficit limitation must appear in the frozen
  release record; PI signature and the project's own genuinely external
  rematerialization requirement remain outstanding per the ledger (this
  review's verification ran from this session's mount and repo clone and
  cannot itself discharge that row).

per-gate table:
  1  Q-L4 unique from frozen table ............................ PASS
  2  structural pass + sparse/causal fails (registered) ....... PASS
  3  margin strata do not rescue .............................. PASS
  4  prompt-323 negligible, current-runtime scoped ............ PASS
  5  P4-P1/P4-P2/P4-P3 mechanical ............................. PASS
  6  no confirmatory/replication outcome anywhere ............. PASS
  7  Study-2 imports preserve tiers/forbidden meanings ........ PASS
  8  Gemma blocker preserved under relicense .................. PASS
  9  OLMo wedge effects missing-not-zero ...................... PASS
  10 H6 fail not restated as ablation invalidation ............ PASS
  11 Bank-W pair = power/design closure ....................... PASS
  12 fresh durability: only the known deficit ................. PASS
  13 packet figures/tables match registered hashes ............ PASS
  14 terminal prose within claim ledgers ...................... PASS WITH
     EXPLICIT LIMITATION (three documentation errata listed above)

signature or cryptographic/session provenance:
  Reviewer: Claude general-purpose review subagent, session 2026-08-04,
    narrative-blind per protocol (no human identity; this is a model-run
    review and is labeled as such — it can satisfy the narrative-blind
    reconstruction gate but the project's PI-signature gate remains a human
    action).
  Commit reviewed: 8762a9a57cd67044b6520ee9b7a17c57aec50792
    (interp_jspace_phase4_5, clean tree).
  Live phase4 registry at review: 82 rows; first-81-row SHA-256
    20124c4c95c3c179472db86f87144bdeee4aaf252a4302408e96f01266c2a00c
    (equals the durability snapshot pin).
  Executed: python -m pytest interpretability/jspaces/phases/phase4/tests -q
    → 302 passed; python -m jspace_phase4 verify → exit 1, sole failure =
    the known A120–A250 state.json (expected 361bda08e9ffbe1d…d26f45e8).
  Companion audit: reviews/PHASE4_UNTOUCHED_DATA_AUDIT.md (same session).
```
