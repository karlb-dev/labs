# jspace_lab_sidelines_2.md

## Consolidated side-track block 2: close the Gemma transport instrument and localize the OLMo training-stage transition on one Colab VM

**Status:** operative development and methods plan for one approximately 24-hour Colab VM that services two scientifically isolated side studies. This plan consolidates hardware scheduling, not evidence, registries, branches, or claims.

**Source branch:** `interp_jspace_part2`, at the exact remote HEAD observed at launch.

**Gemma working branch:** `interp_jspace_gemma_transport_2`

**OLMo working branch:** `interp_jspace_olmo_lineage_2`

**Gemma namespace:** `interpretability/jspaces/sidelines/gemma/`

**OLMo namespace:** `interpretability/jspaces/sidelines/olmo/`

**Gemma evidence prefix:** `gm2-`

**OLMo evidence prefix:** `ol2-`

**Gemma Drive root:** `/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_2_<date>/`

**OLMo Drive root:** `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_<date>/`

**Governs with:**

Gemma:

1. `jspace_lab_gemma_1.md`
2. `jspace_lab_gemma_1_addendum.md`
3. `interpretability/jspaces/sidelines/gemma/release/GEMMA_TRANSPORT_STATE_OF_RECORD.md`
4. `interpretability/jspaces/sidelines/gemma/release/gemma_transport_claim_ledger.md`
5. `jspace_lab_gemma_2.md`

OLMo:

1. `jspace_lab_olmo_lineage_1.md`
2. `jspace_lab_olmo_lineage_1_addendum.md`
3. `interpretability/jspaces/sidelines/olmo/reports/OLMO_LINEAGE_STATE_OF_RECORD.md`
4. `interpretability/jspaces/sidelines/olmo/reports/OLMO_LINEAGE_CLAIMS_TABLE.md`
5. `jspace_lab_olmo_lineage_2.md`

Cross-track:

1. `interpretability/jspaces/sidelines/gemma/release/TRANSPORT_GATE_PROTOCOL.md`
2. `interpretability/jspaces/phases/phase4/paper/PAPER_CONCLUSION_SKELETON.md`
3. `interpretability/jspaces/phases/paper_analysis/olmo_lineage.tex`
4. `interpretability/jspaces/phases/paper_analysis/gemma4_nonlinear_jacobian_handout.tex`
5. `interpretability/jspaces/phases/paper_analysis/Mapping_Neural_Curvature.pdf`

This consolidated plan supersedes the two draft study-2 files only in **GPU order, one-VM scope, drop rules, and branch integration**. Their precommitted scientific definitions, forbidden meanings, study-1 records, and threshold-before-target rules remain binding.

> ## Paste-line for the coding and research agent
>
> Read the governing Gemma and OLMo study-1 plans, addenda, states of record, claim ledgers, and the two study-2 proposals before changing code. Pull `origin/interp_jspace_part2` with `--ff-only`, require a clean source tree, record the exact parent SHA, and create two path-isolated worktrees from that same commit: `interp_jspace_gemma_transport_2` and `interp_jspace_olmo_lineage_2`. One VM does not mean one scientific branch. The packages, registries, evidence prefixes, Drive roots, models, and release bundles remain separate. GPU residency is serialized: never keep Gemma and OLMo weights resident together. CPU preparation may proceed in the inactive worktree while the other model owns the GPU. First run Gemma G2.1 backend-disagreement calibration and freeze its ceiling without reading Stage-1 outcome metrics; immediately apply the precommitted G2.2 relicensing router. This calibration is the dependency OLMo H6 needs. Then unload Gemma and run the OLMo SFT/DPO Tier-1 stage wedge, which is the highest-value paper question in the side block. If time remains, run OLMo in-band transport validation using the registered Gemma study-2 ceiling. Only after those decisive stages may the VM spend residual time on a narrowly selected Gemma mechanism slice, a conditional OLMo own-lens refit, or a receiver demonstration. Do not run a broad G2-G8 Gemma safari, full Bank-W intervention, O5 crossed grid, or unplanned model. Bank every stage to its own registry and root. Each branch must independently produce a state of record, claim ledger update, exact release bundle, tests, and handoff. Push both branches, then merge them back to `interp_jspace_part2` separately with ancestry preserved and Gemma before OLMo. No native side event enters the Phase 4 registry, and no result receives a stronger evidence tier merely because the two studies shared a GPU.

---

# 0. Executive decision

## 0.1 The VM has two main scientific jobs

### Job A: close the Gemma methods blocker

Study 1 measured a strong finite-scale tangent mismatch across all five tested Gemma layers, but the backend-parity gate failed its own very strict ceiling:

```text
observed all-slot backend relative error: 0.002458
frozen study-1 ceiling:                   0.000010
```

At the same time:

- selected-slot backend replay was bit-identical;
- all-slot cosine was approximately 0.99999958;
- maximum absolute disagreement was one bf16 quantum;
- the scientific tangent-versus-response mismatch was orders of magnitude larger.

The highest-value Gemma question is therefore not another architecture theory. It is:

> **Is 0.002458 a measured bf16 scheduling/batch floor that permits Stage-1 relicensing, or a genuine path ambiguity that invalidates "the exact JVP" on this architecture?**

This VM must answer that first.

### Job B: locate the OLMo transition inside the official Think recipe

Study 1 established a striking development-tier pattern:

- Base has near-zero Bank-S span-safe J-specific dependence.
- 3.0 Think has a negative direct effect.
- 3.1 Think preserves and strengthens that pattern.
- 3.1 Instruct is near zero.
- measured sparse capacity is broadly conserved;
- selected-span and J-mapped token geometry reorganize strongly from Base to 3.0 Think;
- 3.0 Think to 3.1 Think geometry is nearly frozen.

Official `Think-SFT` and `Think-DPO` checkpoints are available and ancestry-qualified. The highest-value OLMo question is:

> **Does the causal recruitment and dictionary reorganization appear at SFT, at DPO, or diffusely across the two-stage wedge?**

This is more informative than another endpoint, dose, or model family.

## 0.2 The cross-track dependency determines the order

OLMo study 2 also owes a finite-dose transport-validity check at L24/L32/L40. The transport protocol it plans to use is currently license-blocked by Gemma study 1's backend-parity ceiling.

Therefore:

```text
Gemma backend calibration
-> Gemma Stage-1 license decision
-> OLMo SFT/DPO stage wedge
-> OLMo in-band transport validation
-> optional mechanism follow-ups
```

CPU-only OLMo preparation may occur while Gemma owns the GPU, but no OLMo transport verdict may use an unregistered or outcome-shaped backend ceiling.

## 0.3 The block intentionally drops breadth

This 24-hour VM does **not** attempt every item in the two draft study-2 plans.

Not primary in this block:

- full Gemma G2 through G8 ladder;
- nonlinear lens construction;
- full Gemma late-band Phase-5 doorway;
- OLMo Bank-W intervention;
- targeted Bank-W family reauthoring;
- O5 full crossed transport/readout grid;
- broad receiver mapping;
- multiple new lens fits;
- additional model families;
- confirmatory or replication claims.

The block spends compute on discriminating experiments that can move paper sentences.

## 0.4 Four acceptable terminal combinations

### S2-A: Gemma benign floor, OLMo stage localized

Strongest outcome:

- calibrated backend ceiling licenses Stage 1;
- Gemma finite-scale transport mismatch becomes a closed methods result;
- OLMo effect appears cleanly at SFT or DPO;
- optional own-lens geometry co-localizes with causal onset.

### S2-B: Gemma benign floor, OLMo transition distributed or capability-gated

Still valuable:

- Gemma methods claim closes;
- OLMo narrows the change to "distributed across SFT/DPO" or identifies capability onset as the limiting stage;
- no false single-objective story.

### S2-C: Gemma path ambiguity, OLMo stage wedge succeeds

- Gemma remains an autodiff/instrument boundary case;
- no shared ceiling licenses OLMo H6;
- the OLMo stage wedge still runs because its paired causal grid does not require tangent prediction;
- OLMo transport verdict remains held.

### S2-D: both decisive stages are bounded nulls or gates

- Gemma backend behavior is characterized but not relicensed;
- SFT/DPO cells fail capability or do not localize onset;
- this bounds the current paper's claims and prevents speculation;
- release the negative cleanly.

---

# 1. State of record at launch

## 1.1 Gemma study 1

Expected live state:

| Object | State |
|---|---|
| Foundation and instrument goldens | complete |
| OLMo calibration | complete |
| Gemma Stage-1 grid | complete, 40 cells / 1,120 rows |
| Frozen Stage-1 classifier | `local_tangent_mismatch` at L22/L30/L37/L44/L52 |
| Backend parity | failed only the all-slot 1e-5 ceiling |
| G2-G8 | cancelled under hard-stop rule |
| Evidence tier | methods/development diagnostic |
| Stronger claims | blocked |
| Terminal release | `COMPLETE_METHODS_BLOCKER` |

Key registered values to reverify, not merely copy:

```text
backend selected slot:      exact
backend all-slot cosine:    approximately 0.99999958
backend all-slot rel. err:  0.002458
backend max abs:            0.0390625
```

The Stage-1 layer pattern is large enough that backend disagreement is unlikely to explain it, but likelihood is not a license.

Key registered artifact hashes for the `gm2-foundation-v1` imports (verify bytes, never trust filenames):

```text
frozen study-1 threshold file:  3cb1e68c548bce1dc350c8b60a52e5bc6594a4fadb7abec1c4e00f931d855630
Stage-1 summary:                0f28372591bc1ece4472b103d74d645416b1ddba59a08ae0688c19fccb56e384
Stage-1 row table:              3b74f1e983f47c1f917fd8c407a6ea1f8abf42854adc0b9d6c3d3cf18d921550
backend-parity artifact:        22c327764034f77496971f6c555af0ec6f8e99a0ceb80cea1677db24ca404b7c
backend-parity raw:             ac5ba50dbba6d3ed149cf5b7b6951b80bee5502d11ac90e4f93dc45d515c9e89
terminal import-envelope payload: 694c62db534953accadb4f2223109fabf689f02a5117333497f716c17bed0320
```

One study-1 instrument lesson binds the G2.1 design: the **activation-radial
direction family produced zero delivery/SNR-evaluable rows in the entire
1,120-row Stage-1 grid** (bf16 rounding degrades radial perturbations of the
activation itself). The calibration's direction draws must therefore come
from the three families that actually delivered (rademacher, gaussian,
sphere-tangent); if radial is wanted at all, its delivery repair is a
separate, explicitly optional cell — never a silent absence from the
calibration distribution.

## 1.2 OLMo lineage study 1

Expected live state:

| Objective | State |
|---|---|
| Bank-W capability | both OLMo endpoints pass independently; exact three-model support 16/20 |
| Bank-W intervention | gated out |
| Symmetric capacity | broadly conserved, occupancy 2 |
| Geometry | dictionary-formation pattern concentrated at Base-to-3.0 Think |
| 3.0-to-3.1 Think geometry | nearly stable |
| Instruct geometry | third span, not Base reversion |
| Official intermediates | Think-SFT and Think-DPO eligible |
| O5 | not identifiable with current cells |
| Independent reconstruction | complete |
| Evidence tier | development/methods |

Causal trajectory under the existing Bank-S assay:

| Checkpoint | Direct own-frame | Composed-minus-direct own-frame |
|---|---:|---:|
| Base | approximately 0 | approximately 0 |
| 3.0 Think | approximately -0.128 | approximately +0.072 |
| 3.1 Think | approximately -0.167 | approximately +0.118 |
| 3.1 Instruct | approximately -0.022 | approximately +0.005 |

This is the trajectory the SFT/DPO wedge must bracket.

## 1.3 Cross-track transport debt

The Gemma study-1 OLMo control produced exact-JVP measurements at the OLMo assay band:

| OLMo layer | Median tangent cosine | Median relative error at epsilon 0.10 |
|---|---:|---:|
| L24 | 0.932 | 0.362 |
| L32 | 0.962 | 0.273 |
| L40 | 0.974 | 0.225 |
| L56 | 0.992 | 0.124 |

The late anchor passes the frozen error gate; the causal assay band does not at epsilon 0.10. This does not invalidate the registered paired projection-ablation findings. It does block casual "usable transport coordinate" wording and any crossed-lens interpretation that assumes local finite-dose fidelity.

---

# 2. One VM, two branches, no scientific co-mingling

## 2.1 Create both branches from the same parent

From a clean source worktree:

```bash
set -euo pipefail

cd /content/labs
git fetch origin
git switch interp_jspace_part2
git pull --ff-only origin interp_jspace_part2
git status --short
test -z "$(git status --short)"

PARENT_SHA="$(git rev-parse HEAD)"
printf '%s\n' "$PARENT_SHA" > /content/jspace_sidelines_2_parent_sha.txt

git branch interp_jspace_gemma_transport_2 "$PARENT_SHA"
git branch interp_jspace_olmo_lineage_2 "$PARENT_SHA"

git worktree add /content/labs_gemma2 interp_jspace_gemma_transport_2
git worktree add /content/labs_olmo2 interp_jspace_olmo_lineage_2
```

If either branch already exists from a lawful resume, verify rather than recreate it.

Each branch foundation must record the same source parent SHA and its own:

- branch;
- namespace;
- evidence prefix;
- Drive root;
- registry prefix hash;
- imported study-1 artifacts;
- forbidden write paths;
- GPU model/cache plan;
- preregistered predictions;
- current stage.

## 2.2 Write isolation

Gemma branch may write only:

```text
interpretability/jspaces/sidelines/gemma/
interpretability/jspaces/phases/paper_analysis/gemma4_nonlinear_jacobian_handout.tex
interpretability/jspaces/phases/paper_analysis/scripts/gemma_transport_*
interpretability/jspaces/phases/paper_analysis/figures/gm*
```

OLMo branch may write only:

```text
interpretability/jspaces/sidelines/olmo/
interpretability/jspaces/phases/paper_analysis/olmo_lineage.tex
interpretability/jspaces/phases/paper_analysis/figures/olmo*
interpretability/jspaces/phases/paper_analysis/figures/olf*
```

Neither branch may edit:

```text
interpretability/jspaces/phases/phase4/reports/evidence_events.jsonl
interpretability/jspaces/phases/phase4/preregistration/
interpretability/jspaces/phases/phase4/reports/PHASE4_DEVELOPMENT_REPORT.md
```

A shared helper needed by both studies must be implemented locally in one side package with conformance tests, then imported by exact commit/hash. Do not hot-edit a mainline package from both branches.

## 2.3 Registry isolation

Gemma:

```text
interpretability/jspaces/sidelines/gemma/reports/evidence_events.jsonl
prefix: gm2-
```

OLMo:

```text
interpretability/jspaces/sidelines/olmo/reports/evidence_events.jsonl
prefix: ol2-
```

Study-1 events remain immutable. Study-2 imports them read-only by exact hash.

No event may be copied from one side registry into the other. Cross-track dependency is represented as a strict import envelope that points to a source event and verifies its output hash.

## 2.4 Drive isolation

Create roots before the first write. Never temporarily fall back to the Phase 4 root.

Suggested:

```bash
export GEMMA2_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_2_$(date -u +%Y%m%d)
export OLMO2_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_$(date -u +%Y%m%d)

mkdir -p "$GEMMA2_RUN_ROOT" "$OLMO2_RUN_ROOT"
```

Each root gets:

```text
manifests/
raw/
derived/
figures/
logs/
checkpoints/
release/
backups/
```

## 2.5 GPU serialization

Hard rule:

```text
one 31B/32B model resident at a time
```

Before switching models:

1. bank the current stage;
2. remove hooks;
3. delete model references;
4. run garbage collection;
5. empty CUDA cache once;
6. verify no process retains model VRAM;
7. preserve exact cache manifest;
8. only then stage the next model.

Do not run a Gemma model process and an OLMo model process concurrently, even if free VRAM appears sufficient.

## 2.6 CPU overlap is allowed

While Gemma owns the GPU, the OLMo worktree may:

- run unit tests;
- write foundation/preregistration;
- verify checkpoint inventory and tokenizer semantics;
- materialize manifests;
- compile configs;
- prepare analysis code;
- run Bank-W two-model power planning;
- regenerate existing figures;
- prepare release templates.

While OLMo owns the GPU, the Gemma worktree may:

- analyze registered vectors/tensors;
- finalize backend-calibration thresholds;
- update state of record;
- prepare mechanism plots;
- compile the paper;
- verify registry and release bundle.

CPU overlap must never read quarantined target outcomes before a threshold or branch decision is frozen.

---

# 3. Shared governance and contamination firewall

## 3.1 Thresholds before targets

### Gemma

G2.1 may compute only backend-versus-backend disagreement. It must not read, join, display, or condition on:

- Stage-1 tangent-versus-response error;
- the 0.002458 target value while deriving the new ceiling;
- which ceiling would relicense Stage 1;
- layer-specific Stage-1 classifier outcomes.

Freeze and register the calibrated ceiling first. Only G2.2 compares it to the historical target.

### OLMo

Freeze into `ol2-foundation-v1` before loading SFT or DPO:

- H5-SFT prediction;
- H5-DPO prediction;
- distributed-stage route;
- capability floor;
- common-frame direct-effect thresholds;
- Tier-2 trigger;
- H6 transport thresholds and epsilon ladder;
- allowed stage exclusions;
- no-post-hoc prompt repair rule.

## 3.2 Cross-track unopened-outcome boundary

Neither study may inspect the other study's not-yet-registered target outcome.

Allowed cross-track imports:

- exact model/control artifacts already public in a study-1 release;
- Gemma study-2 backend-calibration ceiling after its event is registered;
- transport protocol source/config/tests;
- methods-only manifests.

Forbidden:

- opening the OLMo stage-wedge result to decide Gemma layer selection;
- using Gemma mechanism outcomes to choose OLMo epsilon;
- using OLMo H6 results to recalibrate Gemma's ceiling;
- pooling rows across registries for a stronger evidence tier.

## 3.3 Contradiction heuristic

Standing lab law:

> When a new measurement contradicts a registered result, suspect the instrument, unit system, cache, hook, or analysis identity before inventing a biological story.

Required response to a contradiction:

1. stop downstream execution;
2. identify the exact invariant that should link the two results;
3. reproduce both on a sentinel cell;
4. inspect tokenization, batch composition, dtype, hooks, cache, normalization, and RNG;
5. register the incident and repair;
6. supersede rather than overwrite.

## 3.4 Claims remain tiered

Sharing a VM does not turn development evidence into confirmation.

Gemma study 2 remains methods/development unless separately preregistered otherwise.

OLMo study 2 remains development/natural-experiment evidence. Even a clean SFT/DPO wedge identifies an interval in an official recipe, not a randomized causal training objective.

---

# 4. Stage S0: dual foundation and preparation

**Compute:** CPU and storage. Model downloads may start.

**Elapsed ceiling:** 60 to 90 minutes.

## 4.1 Gemma foundation

Create or update:

```text
interpretability/jspaces/sidelines/gemma/preregistration/G2_STUDY2_FROZEN_DESIGN.md
interpretability/jspaces/sidelines/gemma/configs/gm2_backend_parity_calibration.yaml
interpretability/jspaces/sidelines/gemma/configs/gm2_stage1_relicense.yaml
interpretability/jspaces/sidelines/gemma/reports/INPROGRESS_GEMMA_TRANSPORT_2.md
```

Register:

```text
gm2-foundation-v1
```

The foundation must hash-import:

- study-1 registry prefix;
- state of record;
- claim ledger;
- JVP goldens;
- OLMo calibration;
- frozen study-1 thresholds;
- Gemma Stage-1 rows and curvature fits;
- backend-parity artifact;
- exact Gemma and OLMo model manifests.

The foundation event must not reveal a new G2.1 backend measurement.

## 4.2 OLMo foundation

Create or update:

```text
interpretability/jspaces/sidelines/olmo/preregistration/OLMO_LINEAGE_STUDY2_PREREGISTRATION.md
interpretability/jspaces/sidelines/olmo/configs/ol2_stage_wedge.yaml
interpretability/jspaces/sidelines/olmo/configs/ol2_transport_validation.yaml
interpretability/jspaces/sidelines/olmo/reports/INPROGRESS_OLMO_LINEAGE_2.md
```

Register:

```text
ol2-foundation-v1
```

The foundation must hash-import:

- study-1 registry prefix;
- state of record;
- claims table;
- checkpoint inventory v2;
- Base, 3.0 Think, 3.1 Think, and 3.1 Instruct lineage summaries;
- frozen 972-row F+S bank;
- frozen base-lens manifest;
- Bank-S producer and scientific seed namespace;
- study-1 capacity/geometry artifacts;
- current transport-gate protocol as license-pending.

Embed the SFT/DPO predictions verbatim before either intermediate model loads.

## 4.3 Tests

Run both suites from their own environments:

```bash
cd /content/labs_gemma2/interpretability/jspaces/sidelines/gemma
python -m pytest -q

cd /content/labs_olmo2/interpretability/jspaces/sidelines/olmo
python -m pytest -q
```

Add smoke tests for:

- distinct run roots;
- distinct registries;
- evidence-prefix rejection;
- forbidden mainline writes;
- cross-track import hash validation;
- model-unload sentinel;
- clean branch provenance.

## 4.4 Background staging

Start downloads or local materialization in the order needed:

1. Gemma exact snapshot;
2. OLMo control snapshot needed for G2.1, only if the calibration design requires it resident in the Gemma stage;
3. Think-SFT;
4. Think-DPO;
5. Base/3.1 Think for H6 only after the wedge.

Preserve at most two large snapshots locally if disk is constrained. Delete only after a release, hash verification, recovery mirror, and next-model manifest are complete.

---

# 5. GPU lane A: Gemma backend-disagreement calibration

## G2.1 - unconditional calibration gate

**Scientific priority:** first.

**Target event:**

```text
gm2-backend-parity-calibration-v1
```

## 5.1 Design

Run approximately 200 to 400 backend pairs over:

```text
models:       Gemma plus OLMo control
depth:        shallow, mid, late
prompts:      4 per model
batch sizes:  1, 4, 8
directions:   3 independent draws
backends:     torch.func.jvp and torch.autograd.functional.jvp
```

The exact cell count must come from the frozen config.

Per pair, record:

- model revision;
- layer;
- prompt hash;
- batch size;
- slot;
- direction seed/family;
- dtype;
- primal parity;
- tangent cosine;
- tangent relative error;
- maximum absolute difference;
- difference in dtype quanta;
- per-slot and all-slot summaries;
- deterministic replay;
- attention-only and MLP-only suffix screen on the frozen subset;
- kernel/backend warnings;
- exception/nonfinite state.

## 5.2 Calibration ceiling

Derive from G2.1's own distribution only.

Precommitted rule:

```text
ceiling = max(
    3 * q99(disagreement distribution),
    relative equivalent of 10 dtype quanta
)
```

If OLMo and Gemma distributions differ materially beyond dtype accounting, freeze per-model ceilings and register that as an architecture-dependent numerical floor.

The threshold file must include:

- source event ID;
- exact row-table hash;
- q50/q90/q95/q99/max;
- bootstrap uncertainty where appropriate;
- batch and slot sensitivity;
- per-model route;
- formula;
- no-target-read assertion;
- code/config commit.

Freeze the file before G2.2 reads the historical 0.002458 target.

## 5.3 G2.1 router

### G2.1-A: benign scheduling floor

Pattern:

- disagreement concentrated within a few dtype quanta;
- very high cosine;
- no scientifically large op-specific divergence;
- primal parity exact or within frozen tolerance;
- batch/slot variation consistent with kernel scheduling.

Action:

- freeze calibrated ceiling;
- proceed to G2.2 branch 1 or 2.

### G2.1-B: batch-composition nuisance

Pattern:

- batch size or slot materially changes disagreement;
- batch-1 is substantially tighter;
- no large op-specific path divergence.

Action:

- freeze batch-1 ceiling;
- require batch-1 replay/recompute in G2.2;
- record batch composition as a transport-protocol requirement.

### G2.1-C: path ambiguity

Pattern:

- disagreement grows beyond dtype accounting;
- localizes reproducibly to an op family;
- changes scientific direction materially;
- persists at batch 1;
- cannot be bounded by a stable numerical floor.

Action:

- hard stop Gemma scientific ladder;
- register the instrument blocker;
- do not relicense Stage 1;
- do not supply a ceiling to OLMo H6;
- proceed to the OLMo stage wedge, whose paired causal grid does not depend on tangent validity.

### G2.1-D: architecture-dependent floor

Action:

- freeze separate ceilings;
- use only the OLMo-specific ceiling for OLMo H6;
- preserve the architecture dependence as a methods result.

## 5.4 Acceptance and reconstruction

Before registration:

- rerun at least one pair in a fresh process;
- verify row-order independence;
- verify selected-slot and all-slot calculations;
- verify dtype-quantum conversion;
- reconstruct aggregate quantiles from raw rows;
- ensure no Stage-1 outcome field entered the calibration process;
- visually inspect any figure for misleading logarithmic floors.

---

# 6. Gemma G2.2: Stage-1 license decision

**Target event:** one of:

```text
gm2-stage1-relicense-v1
gm2-stage1-batch1-v1
gm2-stage1-remains-blocked-v1
```

The decision table was fixed before G2.1 and may not be rewritten after seeing the calibration.

## 6.1 Branch 1: relicensing without recompute

Requirements:

- calibrated applicable ceiling is at least 0.002458;
- frozen selected replay slot remains bit-identical under both backends;
- G2.1 did not identify path ambiguity;
- study-1 artifact hashes match exactly;
- no row selection or recomputation was used to improve Stage 1.

Action:

- relicense `gm-jvp-gemma-stage1-v1` as registered;
- promote the all-five-layer `local_tangent_mismatch` classification from operational diagnostic to closed methods result;
- preserve every study-1 row;
- update claim ledger and state of record;
- license only the finite-scale, tested-map sentence.

## 6.2 Branch 2: batch-1 declared-dose recompute

Requirements:

- G2.1 supports a benign batch-1 object;
- historical full-batch 0.002458 remains above the calibrated ceiling;
- no path ambiguity.

Run only the approximately 80 declared-dose rows:

```text
5 layers x 4 prompts x 4 directions
batch size 1
both exact backends
unchanged study-1 classifier
```

Do not rerun the full epsilon ladder or select favorable directions.

Possible outcomes:

- both backends reproduce the study-1 classifier at all layers;
- some layers change classification;
- delivery/SNR fails at batch 1.

Register the new rows beside, not over, study 1.

## 6.3 Branch 3: remains blocked

If G2.1 identifies path ambiguity:

- no relicensing;
- no mechanism attribution;
- retain methods blocker;
- export the backend ambiguity as the result;
- stop Gemma model compute unless a tiny reduced-width localization was prospectively included in the G2.1 design.

## 6.4 Licensed sentence ceiling

Even after successful relicensing, the strongest automatic sentence is:

> At the tested prompts, layers, directions, and intervention-relevant finite scales, the prompt-specific first-order tangent of the chosen source-to-target residual map predicts Gemma's finite response substantially less accurately than the same estimator predicts the OLMo control; the mismatch changes character with depth.

Still forbidden:

- Gemma is nondifferentiable;
- no Jacobian can model Gemma;
- Gemma has no workspace;
- nonlinear transport causes any behavioral difference;
- late-band J-space is absent.

---

# 7. Gemma residual-time mechanism slice, conditional and narrow

Run this section only after G2.2 closes successfully and after preserving enough time for the OLMo wedge and release.

## 7.1 Priority G2: layer and sublayer localization

First analyze existing Stage-1 vectors and curvature fits without new model compute.

Questions:

- Does the slope jump between L30 and L37 align with a global-attention boundary?
- Is the mismatch concentrated in attention, MLP, norm, or composition?
- Does the intercept-to-slope transition reflect dtype floor early and curvature late?
- Does single-position versus uniform-valid perturbation change the pattern?

New model compute, if needed, is limited to a frozen subset:

```text
layers:      L30, L37, L44
prompts:     2
directions:  fixed small set
suffixes:    attention-only, MLP-only, norm/composition
epsilon:     declared dose plus one smaller measurable dose
```

Register a localization event only if the cell identity was frozen before execution.

## 7.2 Priority G5: prompt/context heterogeneity

If the localization slice is complete and time remains, compare:

1. exact prompt-specific JVP;
2. prompt-averaged, position-specific estimate;
3. campaign averaged-J estimate.

This decomposes:

- prompt-averaging loss;
- position-averaging loss;
- finite-scale curvature.

Do not fit a nonlinear lens in this block.

## 7.3 Optional late-band micro-ladder

Only after OLMo Tier 1 is banked or if the OLMo intermediate models are unavailable.

Use:

```text
layers:      L44, L52
epsilon:     0.0005 to 0.01
injection:   fp32 residual injection
prompts:     frozen small subset
directions:  frozen small subset
```

The goal is to determine whether a measurable small-epsilon tangent regime exists, not to rescue a workspace claim.

Drop this before any OLMo wedge cell.

---

# 8. Model transition: Gemma to OLMo

Before staging Think-SFT:

1. finalize and register the current Gemma event;
2. update `INPROGRESS_GEMMA_TRANSPORT_2.md`;
3. push `interp_jspace_gemma_transport_2`;
4. verify Drive and local backups;
5. remove all Gemma hooks;
6. delete model and optimizer/tangent references;
7. run `gc.collect()`;
8. run `torch.cuda.empty_cache()`;
9. verify no other PID holds GPU memory;
10. preserve the exact Gemma snapshot manifest and recovery path;
11. stage the OLMo intermediate snapshot.

The OLMo branch must import only the registered Gemma G2.1 threshold event, not unregistered working files.

---

# 9. GPU lane B: OLMo SFT/DPO stage wedge

## OL2.1 - the spine of the OLMo study

**Target events:**

```text
ol2-stage-wedge-think-sft-tier1-v1
ol2-stage-wedge-think-dpo-tier1-v1
ol2-stage-wedge-joint-analysis-v1
```

## 9.1 Checkpoints

Pinned official intermediate checkpoints:

```text
allenai/Olmo-3-32B-Think-SFT
allenai/Olmo-3-32B-Think-DPO
```

Pin the **exact commit revisions** recorded in `ol-checkpoint-inventory-v2`
into `ol2_stage_wedge.yaml` at foundation time and load only by revision —
never `main`. The inventory event carries per-file blob IDs and SHA-256
values for both repositories; a hub-side update between inventory and load
must surface as a hash mismatch, not as silent drift. (Note the inventory
also found the 3.1-Instruct SFT/DPO repositories exist but fail the
ancestry-qualification rule — they are not wedge-eligible and must not be
substituted if a Think-SFT/DPO download stalls.)

Before model load, re-run the inventory-v2 semantic compatibility checks:

- exact model revision;
- architecture/config;
- tokenizer ID map;
- normalized merges;
- frozen prompt encodings;
- BOS behavior;
- chat-template differences;
- expected residual layers and d_model;
- common base ancestry qualification.

A serialization-only tokenizer difference is not a failure if the frozen semantic audit passes. A changed token ID or prompt encoding is a hard stop until accounted for.

## 9.2 Tier 1, no refits

For each intermediate:

1. run G5 capability on the frozen 972-row Bank F + Bank S battery;
2. freeze capable Bank-S cohort before interventions;
3. run the seven-condition Bank-S lineage grid in the frozen base-lens frame;
4. use the study-1 scientific seed namespace;
5. preserve exact item and condition order;
6. compute the same family-weighted direct, composed, and composition quantities;
7. run conformance audit.

Why base-lens frame:

- the study-1 common frame reproduced the endpoint trajectory;
- Tier 1 asks when causal recruitment appears, not when each recipient's own dictionary is best fit;
- avoiding two immediate 120-prompt fits buys a decisive stage wedge within one block.

## 9.3 Required seven conditions

Use the study-1 Bank-S contract:

- baseline;
- span-safe J;
- exact instantaneous rank/energy matched control;
- label-protected J;
- protected-energy control;
- mechanics-random control;
- logit-space label protection.

Every row records:

- model/checkpoint revision;
- item/family/fact IDs;
- direct/composed condition;
- aliases and token IDs;
- selected and protected directions;
- effective rank;
- matched-control rank and energy;
- baseline and intervention full-answer log probabilities;
- condition order;
- scientific seed namespace;
- cache replay drift;
- protection overlap;
- manifest hashes.

## 9.4 Capability gate

If an intermediate lacks enough capable Bank-S support:

- report it as capability-gated;
- do not edit prompts;
- do not lower the capability threshold;
- do not substitute a different bank after seeing intervention outcomes;
- record capability onset as a lineage result.

A gated checkpoint can still contribute descriptive geometry later, but not a comparable causal effect.

## 9.5 Frozen stage predictions

### H5-SFT

Prediction:

- SFT is already 3.0-Think-like;
- common-frame Bank-S direct specificity is at or below approximately `-0.08`;
- DPO adds no resolved increment.

Route:

> The tested J-space causal recruitment is installed by the SFT-stage boundary of the official Think recipe.

Use "by the SFT boundary," not "SFT objective caused it."

### H5-DPO

Prediction:

- SFT remains Base-like, approximately within `|direct| <= 0.04` with interval containing zero;
- DPO introduces the negative direct effect.

Route:

> The tested causal recruitment appears across the SFT-to-DPO interval.

### Distributed across stages

Pattern:

- SFT partially moves from Base;
- DPO adds a resolved additional change;
- neither simple threshold route captures the trajectory.

Route:

> Recruitment is distributed across the tested SFT/DPO wedge.

### Capability onset

Pattern:

- an earlier checkpoint is incapable and a later checkpoint becomes capable.

Route:

> The available assay localizes capability onset, not an intervention-specific causal transition.

### Null wedge

Pattern:

- SFT and DPO do not reproduce the first-release effect;
- 3.0 Think remains the first measured positive endpoint.

Route:

> The available official SFT/DPO checkpoints do not localize the first-release transition under the frozen assay; later recipe stages, checkpoint differences, or measurement variation remain open.

## 9.6 Joint analysis

Do not compare "significant here" versus "not significant there."

Compute:

- Base-to-SFT;
- SFT-to-DPO;
- DPO-to-3.0 Think;
- common-support adjacent contrasts;
- direct and composition components;
- exact same-family or shared-fact effects where available;
- family bootstrap or sign-flip under the frozen study-2 analysis;
- leave-one-family-out influence;
- baseline capability adjustment as sensitivity;
- no strict monotone story unless the actual intervals support it.

The primary stage router must be emitted mechanically from frozen thresholds.

---

# 10. Conditional OLMo Tier 2: one own-lens refit

Tier 2 is conditional. It is not an automatic reward for completing Tier 1.

## 10.1 Trigger

Fit exactly one intermediate's own lens only if:

- Tier 1 cleanly localizes the causal onset to one adjacent boundary;
- at least five GPU hours remain after preserving release time;
- the checkpoint passes capability;
- the 120-prompt recipe and corpus are hash-identical to study 1;
- the fit can complete and register within the remaining block;
- no Gemma or OLMo H6 never-drop stage is displaced.

Examples:

- SFT first becomes Think-like, fit SFT;
- SFT null and DPO first becomes Think-like, fit DPO;
- distributed outcome, do not fit both in this block.

## 10.2 Tier-2 question

> Does same-corpus dictionary formation co-localize with the causal onset?

Compare the new own lens with:

- Base;
- 3.0 Think;
- adjacent wedge checkpoint in the common frame.

Metrics:

- raw operator cosine;
- mapped-row cosine;
- unembedding-row cosine;
- selected-ID Jaccard;
- projector overlap;
- principal angles;
- readout transfer;
- identity fraction;
- capacity only if the estimator is already available cheaply.

## 10.3 Interpretation

Allowed:

- dictionary reorganization co-localizes with the tested causal onset;
- causal onset appears without a comparably large own-lens geometry transition;
- the transition is distributed or not identified.

Forbidden:

- the identified training objective creates the workspace;
- dictionary rotation mediates the behavioral effect without a causal transport test.

---

# 11. OLMo H6: finite-dose transport validation in the assay band

## OL2.2 - validity debt

**Priority:** after both Tier-1 wedge cells, before broad receiver or O5 work.

**Target event family:**

```text
ol2-transport-validation-base-v1
ol2-transport-validation-olmo31-think-v1
ol2-transport-validation-joint-v1
```

## 11.1 Dependency gate

Import the registered Gemma study-2 calibration event by exact hash.

Possible states:

### Ceiling available and applicable

Run measurements and issue licensed verdicts.

### Gemma path ambiguity

OLMo may still run measurements, but mark:

```text
license_status: held_pending_transport_backend_resolution
```

Do not claim pass/fail under an unlicensed protocol.

### OLMo-specific ceiling

Use the OLMo ceiling only. Do not use a pooled Gemma value for convenience.

## 11.2 Models and layers

Minimum:

```text
models: Base, OLMo-3.1 Think
layers: L24, L32, L40, L56
```

Add SFT or DPO only if:

- Tier 2 already loaded that checkpoint;
- the marginal compute is small;
- all primary Base/Think cells are complete.

## 11.3 Epsilon ladder

The prior epsilon 0.10 check is too coarse for the actual projection-ablation dose distribution.

Use a frozen ladder extending to:

```text
0.005, plus smaller measurable values if fp32 injection and delivery gates permit
```

Include epsilon 0.10 for continuity, not as the sole decision point.

For each cell:

- exact JVP;
- central secant;
- delivered perturbation fidelity;
- SNR;
- tangent cosine;
- forward and central relative error;
- gain;
- homogeneity;
- odd symmetry;
- additivity where feasible;
- intercept-plus-slope fit;
- dtype/backend parity under the imported ceiling.

## 11.4 Dose matching to the causal assay

Map registered span-safe removed-energy distributions to an effective relative epsilon by layer and position.

Implementation note: do not re-instrument the causal assay to obtain this
distribution — it already exists in registered outputs. The study-1 lineage
grids and the Phase 3/4 span-safe producers log per-position achieved rank
and removed-energy fraction in their position records (the v2 ablator's
`PositionRecord` stream; the per-item overlap JSON columns carry the same
accounting in the banked parquets). The mapping job is a CPU join of those
logged fractions against per-position residual norms, then a conversion to
relative-epsilon units — an afternoon of pandas, not a GPU pass.

Report:

- causal-assay epsilon distribution;
- percentage of intervention sites inside each transport-valid regime;
- whether the prior epsilon 0.10 failure is above the typical causal dose;
- whether a pass exists at intervention-relevant doses.

This is more informative than asking whether the model is "linear" in the abstract.

## 11.5 H6 router

### H6-pass in band at relevant doses

Allowed:

> The tested finite-dose transport approximation is usable at the OLMo assay band over the dose range covering the registered intervention distribution.

This can unlock a later O5 crossed-lens pilot.

### H6-late-only

Allowed:

> Transport fidelity passes only in the later anchor band under the tested protocol; in-band transport language and crossed-lens interpretation remain bounded.

### H6-scale-limited

Allowed:

> A small-epsilon local regime exists, but the causal intervention often exceeds it.

### H6-fail

Allowed:

> The average or prompt-specific first-order transport approximation does not meet the frozen finite-dose gate at the assay band.

This does not invalidate paired ablation effects. It narrows transport-based interpretations.

### H6-license-held

Measurements exist, but no pass/fail sentence until backend parity is resolved.

## 11.6 Do not open O5 automatically

Even an H6 pass only makes O5 interpretable. It does not create time or authorization for a full crossed grid. O5 remains optional and lower priority in this one-block plan.

---

# 12. CPU lane: paper-facing closures while the GPU runs

## 12.1 Bank-W OLMo-pair power calibration

Study 1's failure was exact three-model common support, not individual OLMo capability. Run a CPU-only planning analysis for an OLMo-pair primary:

- Think and Instruct only;
- shared capable families;
- same frozen load/derivation/redundancy design;
- shared-family sign flips;
- two-model max-T;
- type-I calibration;
- power at the existing SESOI;
- required family count.

This does not authorize an intervention. It answers whether the cheaper OLMo-pair redesign is worth a future study.

Inputs already exist in registered study-1 outputs: the per-model,
per-family capability rows behind `ol-bank-w-capability-{olmo31-think,
olmo31-instruct,joint}-dev-v1` (each endpoint passed 17/24 individually;
the block was the *three-model* intersection at 16 against the 20 floor).
Compute the OLMo-pair shared-capable family set from those rows — expected
in the high teens to low twenties — and run the two-model max-T power at
that support. Anchor the variance assumption the same way the mainline
ruler did (largest common-family Bank-S development SD, rounded up), and
state it; do not borrow any Qwen outcome.

Register as a planning/methods event, for example:

```text
ol2-bank-w-olmo-pair-power-v1
```

Do not use Qwen outcome information to tune the OLMo-pair rule.

## 12.2 Small OLMo closures

Run when CPU-only:

1. re-evaluate the two unresolved Base-common L40 capacity sensitivities with a larger frozen bootstrap;
2. register the identity-fraction profile for all study-1 checkpoints;
3. regenerate the lineage paper figure set from live evidence;
4. update the claim table with explicit study-2 slots, not filled outcomes.

## 12.3 Gemma paper preparation

While OLMo owns the GPU:

- update the backend-parity methods section;
- prepare branches for relicensed versus blocked wording;
- regenerate the tangent-ladder figure only from registered rows;
- add a backend-disagreement distribution figure;
- separate local derivative, finite response, and averaged map carefully;
- keep the stronger "no Jacobian" wording forbidden.

## 12.4 Release templates

Prepare before the final two hours:

Gemma:

```text
release/GEMMA_TRANSPORT_STATE_OF_RECORD_V2.md
release/gemma_transport_claim_ledger_v2.md
release/TRANSPORT_GATE_PROTOCOL_V2.md
release/IMPORT_BUNDLE_SIDELINES2.json
release/IMPORT_BUNDLE_SIDELINES2.md
```

OLMo:

```text
reports/OLMO_LINEAGE_STATE_OF_RECORD_V2.md
reports/OLMO_LINEAGE_CLAIMS_TABLE_V2.md
release/IMPORT_BUNDLE_SIDELINES2.json
release/IMPORT_BUNDLE_SIDELINES2.md
```

Use the package's existing release naming if it has a stricter convention. Do not overwrite study-1 state files unless the registry and schema explicitly define versioned supersession.

---

# 13. Residual-time router

After Gemma G2.1/G2.2, both OLMo Tier-1 cells, and the mandatory release debt are banked, choose exactly one residual path.

## R1: decisive OLMo wedge, enough time for one own-lens fit

Run Tier 2. This has the greatest chance of linking causal onset with dictionary formation.

## R2: OLMo wedge ambiguous, H6 not yet run

Run H6 Base plus 3.1 Think. Validity is more valuable than another ambiguous stage lens.

## R3: H6 complete, Gemma mechanism cause still broad

Run the narrow Gemma attention/MLP localization slice.

## R4: strong Think effect and enough time for receiver demonstration

Run one lesion-to-clean-activation patch rescue on the strongest consumed Bank-S families, with wrong-layer and unrelated-patch controls. Keep the half-day cap from study 1, but in this VM cap it to the remaining window.

## R5: no safe model block remains

Do CPU analysis, release, paper, tests, and merge. An idle GPU is preferable to an unregistered partial model experiment.

## Never select

- full Bank-W intervention;
- O5 full grid;
- nonlinear Gemma lens;
- two own-lens refits;
- new architecture;
- post-hoc task bank;
- target-informed threshold revision.

---

# 14. Twenty-four-hour schedule

This schedule assumes large snapshots are already partly cached. Bank early if staging is slower.

## T+0:00 to T+1:30 - dual foundation

GPU idle or downloading.

- create both worktrees;
- record exact shared parent SHA;
- create separate Drive roots;
- run both test suites;
- freeze G2.1 config and OLMo stage predictions;
- register `gm2-foundation-v1` and `ol2-foundation-v1`;
- begin Gemma staging;
- prepare OLMo configs and manifests on CPU.

## T+1:30 to T+6:00 - Gemma G2.1

Gemma owns GPU.

- backend calibration grid;
- row and aggregate reconstruction;
- batch/slot/op screen;
- freeze calibrated ceiling;
- register and push Gemma branch.

OLMo CPU lane:

- verify SFT/DPO inventory;
- prepare G5 and Bank-S configs;
- run OLMo-pair Bank-W power code;
- prebuild analysis templates.

## T+6:00 to T+7:30 - Gemma G2.2

- apply relicensing router;
- branch-1 release, branch-2 batch-1 declared-dose replay, or branch-3 blocker;
- update state/claims;
- bank exact event;
- unload Gemma.

If branch 2 requires more time, cap it to declared-dose rows only.

## T+7:30 to T+8:30 - model transition

- finalize Gemma backups and handoff;
- free GPU;
- stage Think-SFT;
- verify exact snapshot and tokenizer semantics.

## T+8:30 to T+12:00 - OLMo Think-SFT Tier 1

- G5 capability;
- freeze cohort;
- seven-condition Bank-S grid;
- conformance audit;
- register and push.

## T+12:00 to T+15:30 - OLMo Think-DPO Tier 1

- same sequence;
- no threshold changes;
- register and push.

## T+15:30 to T+17:00 - joint stage analysis

CPU plus optional model unload.

- adjacent contrasts;
- stage router;
- leave-one-family-out;
- capability adjustment;
- sentence-2 wording;
- decide Tier-2 trigger prospectively from the frozen rule.

## T+17:00 to T+20:30 - OLMo H6 or conditional Tier 2

Default: H6 Base plus 3.1 Think, because it closes a validity debt shared by several future designs.

Use Tier 2 instead only when the wedge is decisive, at least five hours remain at the actual measured fit rate, and H6 already has sufficient registered evidence or can be completed cheaply.

## T+20:30 to T+22:00 - one residual slice

Choose R1 through R5. Do not start a stage that cannot bank before release time.

## T+22:00 to T+24:00 - release and integration

Never drop:

- both state-of-record updates;
- claim ledgers;
- raw/derived manifests;
- registry verification;
- tests;
- paper update;
- release bundles;
- branch pushes;
- exact handoffs;
- serialized merge back or a fully specified merge-ready state if another branch holds the integration lock.

---

# 15. Stop rules and drop order

## 15.1 Gemma hard stops

Stop Gemma downstream science if:

- G2.1 ceiling code reads Stage-1 target metrics before freeze;
- path ambiguity branch triggers;
- model or snapshot hash mismatches;
- backend primal parity fails materially;
- nonfinite JVP appears without a precommitted classification;
- batch/slot reconstruction cannot reproduce raw rows;
- a threshold changes after target unblinding.

## 15.2 OLMo hard stops

Stop a checkpoint cell if:

- tokenizer semantic audit fails;
- frozen prompt encodings differ;
- G5 baseline replay fails;
- capability support misses the frozen gate;
- rank/energy/protection conformance fails;
- scientific seed namespace differs;
- a branch attempts to repair a weak result by changing tasks;
- own-lens Tier 2 trigger is not satisfied.

## 15.3 Cross-track hard stops

- no OLMo H6 verdict from an unregistered Gemma ceiling;
- no Gemma ceiling recalibration from OLMo H6;
- no simultaneous model residency;
- no shared registry;
- no native side event in Phase 4;
- no direct edit to `interp_jspace_part2` during compute;
- no merge race.

## 15.4 Drop order

```text
Gemma late-band micro-ladder
-> OLMo receiver demonstration
-> Gemma mechanism localization new forwards
-> OLMo Tier-2 own-lens refit
-> OLMo H6 extra checkpoints beyond Base and 3.1 Think
-> optional figures
```

Never drop:

1. Gemma G2.1;
2. Gemma G2.2 license decision;
3. both OLMo Tier-1 wedge cells, unless a hard capability gate makes one impossible;
4. joint OLMo stage router;
5. registry and release integrity;
6. branch pushes and handoffs.

## 15.5 VM reclaim

At the last complete atomic boundary:

- register or explicitly mark the current stage partial and unregistered;
- save recovery state;
- hash local and Drive copies;
- update the correct side handoff;
- push the active branch;
- do not create a pseudo-complete event;
- reserve CPU release work even if GPU compute ends early.

---

# 16. Branch-specific deliverables

## 16.1 Gemma required deliverables

```text
interpretability/jspaces/sidelines/gemma/preregistration/G2_STUDY2_FROZEN_DESIGN.md
interpretability/jspaces/sidelines/gemma/configs/gm2_backend_parity_calibration.yaml
interpretability/jspaces/sidelines/gemma/configs/gm2_stage1_relicense.yaml
interpretability/jspaces/sidelines/gemma/reports/INPROGRESS_GEMMA_TRANSPORT_2.md
interpretability/jspaces/sidelines/gemma/reports/GEMMA_TRANSPORT_STUDY2_REPORT.md
interpretability/jspaces/sidelines/gemma/release/GEMMA_TRANSPORT_STATE_OF_RECORD_V2.md
interpretability/jspaces/sidelines/gemma/release/gemma_transport_claim_ledger_v2.md
interpretability/jspaces/sidelines/gemma/release/TRANSPORT_GATE_PROTOCOL_V2.md
interpretability/jspaces/sidelines/gemma/release/IMPORT_BUNDLE_SIDELINES2.json
interpretability/jspaces/sidelines/gemma/release/IMPORT_BUNDLE_SIDELINES2.md
```

Required evidence:

```text
gm2-foundation-v1
gm2-backend-parity-calibration-v1
one G2.2 terminal event
```

Conditional:

```text
gm2-layer-sublayer-localization-v1
gm2-context-heterogeneity-v1
gm2-lateband-micro-ladder-v1
```

## 16.2 OLMo required deliverables

```text
interpretability/jspaces/sidelines/olmo/preregistration/OLMO_LINEAGE_STUDY2_PREREGISTRATION.md
interpretability/jspaces/sidelines/olmo/configs/ol2_stage_wedge.yaml
interpretability/jspaces/sidelines/olmo/configs/ol2_transport_validation.yaml
interpretability/jspaces/sidelines/olmo/reports/INPROGRESS_OLMO_LINEAGE_2.md
interpretability/jspaces/sidelines/olmo/reports/OLMO_LINEAGE_STUDY2_REPORT.md
interpretability/jspaces/sidelines/olmo/reports/OLMO_LINEAGE_STATE_OF_RECORD_V2.md
interpretability/jspaces/sidelines/olmo/reports/OLMO_LINEAGE_CLAIMS_TABLE_V2.md
interpretability/jspaces/sidelines/olmo/release/IMPORT_BUNDLE_SIDELINES2.json
interpretability/jspaces/sidelines/olmo/release/IMPORT_BUNDLE_SIDELINES2.md
```

Required evidence:

```text
ol2-foundation-v1
ol2-stage-wedge-think-sft-tier1-v1
ol2-stage-wedge-think-dpo-tier1-v1
ol2-stage-wedge-joint-analysis-v1
ol2-bank-w-olmo-pair-power-v1
```

Conditional:

```text
ol2-stage-wedge-<checkpoint>-own-lens-v1
ol2-stage-wedge-geometry-joint-v1
ol2-transport-validation-base-v1
ol2-transport-validation-olmo31-think-v1
ol2-transport-validation-joint-v1
ol2-receiver-demonstration-v1
```

## 16.3 Paper outputs

Gemma paper must distinguish:

- backend numerical floor;
- exact local tangent;
- finite-scale response;
- averaged map;
- tested architecture path;
- licensed claim ceiling.

OLMo paper must distinguish:

- capacity;
- geometry;
- causal recruitment;
- capability;
- training-stage interval;
- natural experiment versus randomized attribution;
- transport validity.

No shared-VM synthesis should merge these into one universal model ranking.

---

# 17. Release and merge-back protocol

## 17.1 Finalize each branch independently

### Gemma

```bash
cd /content/labs_gemma2
git status --short
python -m pytest -q interpretability/jspaces/sidelines/gemma/tests
git diff --check
git push -u origin interp_jspace_gemma_transport_2
```

Verify:

- registry event count and prefix;
- study-1 prefix unchanged;
- every live output hash;
- release-bundle hashes;
- paper regeneration;
- clean source commit.

### OLMo

```bash
cd /content/labs_olmo2
git status --short
python -m pytest -q interpretability/jspaces/sidelines/olmo/tests
git diff --check
git push -u origin interp_jspace_olmo_lineage_2
```

Verify the analogous OLMo properties.

## 17.2 Acquire the serialized integration lock

No side branch may merge while another VM owns the source integration lock.

If `interp_jspace_phase4_4` is still unmerged, push both side branches and leave an exact merge-ready handoff. Do not race or force the source. Once the mainline branch is integrated, merge sides in this order:

```text
Gemma study 2
-> OLMo study 2
```

Gemma first preserves the registered protocol dependency that OLMo H6 cites.

## 17.3 Merge Gemma with ancestry preserved

```bash
cd /content/labs
git fetch origin
git switch interp_jspace_part2
git pull --ff-only origin interp_jspace_part2
git status --short
test -z "$(git status --short)"

git merge --no-ff origin/interp_jspace_gemma_transport_2 \
  -m "Gemma transport study 2: calibrate backend parity and close the Stage-1 license"
```

Run:

- Gemma tests;
- Phase 4 import-envelope tests;
- namespace rejection tests;
- paper build;
- registry verification.

Do not create a Phase 4 import event for study 2 during this merge.

## 17.4 Merge OLMo with ancestry preserved

```bash
git merge --no-ff origin/interp_jspace_olmo_lineage_2 \
  -m "OLMo lineage study 2: run the SFT/DPO stage wedge and transport validation"
```

Run:

- OLMo tests;
- Gemma tests;
- Phase 4 tests;
- cross-track import-hash tests;
- paper builds;
- registry verification;
- no native `gm2-*` or `ol2-*` event in Phase 4.

Then:

```bash
git push origin interp_jspace_part2
git fetch origin
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/interp_jspace_part2)"
```

## 17.5 Integration record

Create a source-branch integration record containing:

- shared parent SHA;
- Gemma branch terminal SHA and merge commit;
- OLMo branch terminal SHA and merge commit;
- source registry hashes before and after;
- test counts;
- paper outputs;
- unresolved blockers;
- explicit statement that no study-2 side result was promoted into Phase 4 evidence.

Suggested path:

```text
interpretability/reviews/JSPACE_SIDELINES_2_INTEGRATION_RECORD.md
```

Do not squash away the side histories.

---

# 18. Completion criteria

## Branch and isolation

- [ ] Both branches were created from the same recorded `interp_jspace_part2` parent.
- [ ] Separate worktrees, registries, prefixes, and Drive roots were used.
- [ ] No model-scale processes overlapped on the GPU.
- [ ] No native side event entered Phase 4.
- [ ] Study-1 outputs remained immutable.

## Gemma

- [ ] G2.1 calibration ran without target leakage.
- [ ] Calibrated ceiling was frozen before G2.2.
- [ ] Batch, slot, model, and op structure were measured.
- [ ] One precommitted G2.2 branch was applied.
- [ ] Stage-1 license state is explicit.
- [ ] Stronger forbidden claims remain blocked.
- [ ] State of record, claim ledger, protocol, release bundle, and tests are complete.

## OLMo

- [ ] SFT and DPO revisions/tokenizers were semantically verified.
- [ ] Frozen predictions were registered before model load.
- [ ] G5 capability ran before each intervention grid.
- [ ] Tier-1 SFT and DPO cells are complete or honestly capability-gated.
- [ ] Joint stage router uses adjacent contrasts, not significance comparison.
- [ ] Any Tier-2 own-lens fit met its trigger.
- [ ] H6 uses only a registered applicable backend ceiling or holds its license.
- [ ] Bank-W intervention did not run.
- [ ] State of record, claims table, release bundle, and tests are complete.

## Scientific narrowing

- [ ] Gemma is classified as benign numerical floor, batch nuisance, architecture-dependent floor, or path ambiguity.
- [ ] OLMo transition is classified as SFT, DPO, distributed, capability onset, or unresolved.
- [ ] Transport validity is pass, late-only, scale-limited, fail, or license-held.
- [ ] No unplanned model or broad grid was added.
- [ ] The papers contain exact allowed wording and falsifiers.

## Repository

- [ ] Both side branches are clean and pushed.
- [ ] Integration is serialized.
- [ ] Gemma merges before OLMo.
- [ ] Ancestry is preserved.
- [ ] Union tests and paper builds pass.
- [ ] Remote `interp_jspace_part2` matches local.
- [ ] Integration record and final handoffs exist.

---

# 19. Final instruction to the agent

Treat the shared GPU as a railway junction, not a blender.

Gemma gets the first train because its backend ceiling is a protocol dependency. OLMo gets the longest scientific run because the SFT/DPO wedge can finally tell us where the observed recruitment appears. Everything else waits behind those two questions.

Close the instrument. Localize the stage. Validate transport if the protocol earns a license. Then merge two clean histories back into the campaign, with every boundary still visible.
