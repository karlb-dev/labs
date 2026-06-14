# Lab 12: Relation Geometry and Method Validation

**Advanced course, Group I: bridge and instruments**
**Evidence levels targeted:** `DECODE → CAUSAL`, plus `OBS` for profile and cosine structure
**Prerequisites:** Intro Labs 1, 2, 4, and 5. Optional: Lab 8 for feature-label extensions.

## Core question

The intro course localized factual recall mostly through capital-city examples. That is useful, but it can also become a velvet trap: a claim about “fact recall” may only be a claim about one relation with one template. Lab 12 scales the same instruments to a controlled relation dataset and asks:

> Do relation families share measurable internal geometry, or does each relation run its own small trick that happens to fit under one probe?

This is a **method-validation lab**, not a discovery trophy hunt. A clean negative result earns the lab. The advanced half will soon point probes, directions, and patches at fuzzier phenomena such as emotion, sycophancy, belief, persona, and self-report. Before doing that, this lab makes the instruments survive an easier but larger controlled setting.

## What you will build

You will build a relation probe-and-patch atlas:

```text
runs/lab12_relation_geometry-*/
  run_summary.md
  method_validation_card.md              # pass/fail posture before reading the pretty plots
  operationalization_audit.md            # the deflationary twin
  metrics.json
  results.csv                            # alias of tables/probe_report.csv
  ledger_suggestions.md

  diagnostics/
    frozen_data_manifest.json            # CSV hash, manifest check, filters, source
    tokenization_audit.csv               # single-token and role-position validation
    split_audit.csv                      # subject-grouped split, one row per item
    split_balance.csv                    # train/eval counts by family
    activation_norms_by_role_depth.csv   # norm audit before row normalization
    relation_depth_selection.json        # selected stream depths by role
    patch_pair_gate.csv                  # behavioral gate for every candidate pair
    patch_noop_check.json                # bench proof that self-patching is identity

  tables/
    relation_family_manifest.csv
    margin_report.csv
    margin_by_family.csv
    probe_report.csv
    probe_selectivity_by_depth.csv
    relation_cosine_matrix.csv
    patch_report.csv
    relation_transfer_matrix.csv
    localization_profile_similarity.csv
    relation_evidence_matrix.csv             # per-family evidence ledger
    swap_group_summary.csv                   # swap-group DECODE/CAUSAL rollup
    patch_specificity_by_family.csv          # matched-vs-control gaps by family
    role_handoff_summary.csv                 # peak recovery role/depth handoff
    geometry_block_summary.csv               # same-group vs cross-group cosine summary
    plot_reading_guide.csv                   # what each plot protects

  plots/
    relation_geometry_dashboard.png          # start here: gates, controls, transfer, geometry
    controlled_probe_atlas.png               # role x depth x swap-group selectivity
    relation_evidence_matrix.png             # family-level claim readiness
    relation_probe_by_layer.png
    relation_probe_selectivity.png
    relation_patch_heatmap.png
    role_handoff_summary.png                 # subject→final recovery timing
    relation_swap_recovery.png
    patch_specificity_by_family.png          # per-family matched/control gap
    relation_transfer_matrix.png
    patch_control_gaps.png
    profile_similarity_matrix.png            # localization-profile correlations
    relation_direction_cosines.png

  state/
    relation_directions.pt
    relation_directions_metadata.json
```

The headline artifacts are now `method_validation_card.md`, `plots/relation_geometry_dashboard.png`, `plots/controlled_probe_atlas.png`, `plots/relation_evidence_matrix.png`, `plots/patch_specificity_by_family.png`, `tables/relation_evidence_matrix.csv`, and `operationalization_audit.md`. The older microscope plots remain, but the new audit-board plots are the first artifacts to read.

## The dataset is the experiment

The frozen dataset is:

```text
data/advanced_relation_geometry.csv
```

It is generated deterministically by `data/make_advanced_relation_sets.py` and should be hashed in `data/MANIFEST.json`. The lab records the observed hash in `diagnostics/frozen_data_manifest.json` and, when the manifest has an expected hash entry, records whether it matches.

The central design is the **relation-swap group**. Inside a group, different relation families share the same subjects and the same template skeleton. For one subject, the prompts differ only at the relation-word token:

```text
The capital of France is   → Paris
The language of France is  → French
The continent of France is → Europe
```

That construction kills two cheap explanations by design: entity class and template syntax. It does **not** kill relation-word token echo, and the lab refuses to pretend otherwise. That remaining goblin is the point of the patching phase.

![Three prompts in the country_sem swap group share the subject (France) and the template skeleton, differing only at the relation-word token. Sharing the subject kills the entity-class explanation; sharing the template kills the syntax explanation. The relation word still varies, so relation-word token echo survives by construction — that is what the patching phase attacks.](../figures/lab12_swap_group_kills_two_confounds_leaves_echo.png)

Framed another way, the swap group is a *subtractive* control rather than an additive one: it removes the entity-class and template-syntax explanations by construction, so you never have to build a model of those confounds to rule them out. Only the relation-word token echo is left standing — on purpose — for the patching phase to attack. No amount of probe accuracy retires it.

The expected swap groups are:

| swap group | relation families | controlled confounds |
|---|---|---|
| `country_sem` | `capital_of`, `language_of`, `continent_of` | same countries, same template skeleton |
| `adj_morph` | `opposite_of`, `comparative_of` | same adjectives, same template skeleton |
| `month_seq` | `month_after`, `month_before` | same months, same template skeleton |

Other relation families can be included for breadth and margin diagnostics. They do not carry the controlled headline unless they have token-aligned swap partners.

## The three token roles

Every item is measured at three positions:

| role | meaning | how to interpret it |
|---|---|---|
| `relword` | the relation cue token, such as `capital` or `language` | Depth 0 is expected to decode relation identity because it is the word itself. That is a calibration trap, not a finding. |
| `subject` | the entity token, such as `France` | The interesting role. Relation information at this position after depth 0 must have been moved there by the network. |
| `final` | the last prompt token before answer prediction | The answer readout position. Separability here can reflect relation identity, answer class, and downstream computation all tangled together. |

The bench convention still applies: `streams[k]` is the **pre-norm residual stream after `k` blocks**. Depth 0 is embeddings. Depth `L` is the final-norm input.

## Vocabulary, operationalized

The load-bearing words in this lab have precise, measured meanings. Keep this table next to the plots.

| Term | What it actually means here | Why it matters |
|---|---|---|
| **selectivity** | `real-label probe accuracy − shuffled-label probe accuracy` at a fixed role/depth. | A probe can hit high raw accuracy by memorizing or by exploiting capacity. Subtracting the shuffled-label control measures how much *the representation* helped, not the probe. (The control-task idea, after Hewitt & Liang.) |
| **swap group** | A set of families sharing subject and template, differing only at the relation word. | A subtractive control that kills entity-class and template confounds by construction — no confound model required. |
| **relation-word token echo** | The probe/patch may be reading the relation *token*, not a relation *representation*. | The one confound the swap group cannot kill; the relation-swap patch is built to test it. |
| **subject-swap patch** | Lab-5 interchange: clean `France→Paris` into corrupt `Germany→Berlin`. | Localizes where subject-specific information for a relation is causally usable. |
| **relation-swap patch** | Patch the relation-word residual: clean `capital…` into corrupt `language…`, scored as `logit(Paris) − logit(French)`. | The direct causal pressure test on token echo. |
| **mismatched-vector control** | Patch with a vector from an unrelated family. | Measures recovery that comes from *destroying corrupt evidence*, not restoring clean content. A nonzero value is the margin-destruction floor, not relation use. |
| **handle vs mechanism** | A handle is something readable or causally usable; a mechanism is a named circuit that computes it. | The lab's hard ceiling: every result here is a handle. Geometry blocks and profile correlations are handles, never mechanisms. |

## Experiment steps

### 1. Tokenization and behavioral gates

The lab re-validates everything against the runtime tokenizer. The subject, relation word, target answer, hard distractor, and easy distractor must each be one token. The subject and relation-word positions must occur exactly once in the encoded prompt. Dropped rows are written to `diagnostics/tokenization_audit.csv`.

Next, the lab checks whether the model knows each item behaviorally. For patching pairs, the clean prompt must prefer the clean answer and the corrupt prompt must prefer the corrupt answer by a margin. Dropped pairs are not nuisance rows; they are evidence about the model and the dataset.

### 2. Probe phase: `DECODE`

The lab caches residual streams at the three roles, row-normalizes activations, and runs relation-identity probes across depth.

It evaluates three scopes:

1. `all`: all relation families. This is useful but not entity/template controlled.
2. `country_sem`, `adj_morph`, `month_seq`: swap-group scopes where subjects and templates are controlled.
3. Any swap group that loses family balance after tokenization or splitting is marked as incomplete rather than silently promoted.

Probe methods:

| method | where | claim posture |
|---|---|---|
| nearest-centroid multiclass | all roles, all depths | primary depth-by-depth relation-identity curve |
| one-vs-rest mass mean | all roles, all depths | direction geometry and saved handles |
| logistic one-vs-rest | final role only | upper-bound check, not the saved direction |
| shuffled labels | matched probe runs | capacity/noise control |
| random directions | matched role/depth | direction control |

The primary number is **selectivity**:

```text
selectivity = real centroid accuracy − shuffled-label centroid accuracy
```

The revised code writes `tables/probe_selectivity_by_depth.csv` and plots it in `plots/relation_probe_selectivity.png`. Direction depths are selected separately by role, using swap-group macro selectivity. Subject and final roles exclude depth 0 when possible, because depth 0 is a token-identity sanity row, not a representation claim.

### 3. Direction geometry: `DECODE` artifact, not use

The lab saves one-vs-rest mass-mean relation directions at the selected subject and final depths:

```text
state/relation_directions.pt
state/relation_directions_metadata.json
```

The cosine atlas asks whether relation directions form blocks, especially inside swap groups. A block in the cosine matrix is a **handle**. It is not evidence of shared heads, shared MLPs, or a shared algorithm. A direction can look geometrically neat because the relation-word token echo is geometrically neat.

### 4. Patching phase: scoped `CAUSAL`

The lab runs Lab 5-style interchange patching in two designs.

**Subject-swap, within relation:**

```text
clean:   The capital of France is   → Paris
corrupt: The capital of Germany is  → Berlin
patch:   replace one residual vector in the corrupt run with the clean vector
metric:  recovery of logit(Paris) − logit(Berlin)
```

This produces a causal localization profile by family. It answers where subject-specific information for a relation is causally usable.

**Relation-swap, within swap group:**

```text
clean:   The capital of France is   → Paris
corrupt: The language of France is  → French
patch:   replace the relation-word residual vector
metric:  recovery of logit(Paris) − logit(French)
```

This is the pressure test for relation-word token echo. If relation-token patching moves the answer margin and controls stay low, the safe causal claim is:

> The residual stream at the relation-word token carries relation identity that this answer pathway uses.

That still does not identify the mechanism that computed or moved the relation information.

### 5. Patching controls

| control | purpose |
|---|---|
| wrong-position patch | Same clean vector source pair, but patched at a shared template token. Should recover near zero. |
| mismatched-vector patch | Patch with a vector from an unrelated family. Measures margin movement from destroying corrupt evidence rather than restoring clean content. |
| behavioral gate | Prevents patch recovery denominators from being meaningless. |
| depth-band convention | Excludes depth 0 and depth `L` from causal summaries. |

Depth 0 patching is token substitution. Depth `L` for non-final positions cannot reach the readout. Both depths remain in raw tables as sanity rows but are excluded from band means and persistence depths.

## Running it

```bash
# Tier A smoke. Uses gpt2 and a small per-family cap.
python interp_bench.py --lab lab12 --tier a

# Tier B science on the course base model with all relation items.
python interp_bench.py --lab lab12 --tier b --relation-set full

# Restrict to one swap group.
python interp_bench.py --lab lab12 --tier b --relation-set full \
  --relations capital_of,language_of,continent_of

# Thin the patch grid when debugging runtime.
python interp_bench.py --lab lab12 --tier b --relation-set medium \
  --patch-grid subject,relation
```

`--relation-set small|medium|full` caps items per family. `--max-examples N` overrides that cap, still per family. `--relations` filters families. Keep whole swap groups together when you want controlled claims.

The code includes a tiny built-in Tier A fallback only for plumbing if the frozen CSV is absent. It is labeled `builtin_tier_a_smoke_fallback` in `metrics.json` and the method card. Do not ledger science claims from fallback data.

## What success looks like

A positive run has the following shape:

1. `method_validation_card.md` says the probe and patching gates passed.
2. Swap-group selectivity rises above shuffled controls at the subject or final role. The `relword` role at depth 0 can be high, but that is not a discovery.
3. Subject-swap patching shows family-specific recovery bands, usually with a handoff from subject role toward final role.
4. Relation-swap patching at the relation token beats wrong-position and mismatched-vector controls.
5. The transfer matrix has diagonal subject-swap cells and within-swap-group relation-swap cells. Cross-group cells remain empty when token-aligned pairs do not exist.
6. Cosine and profile-similarity artifacts are discussed as handles, not mechanisms.

A negative success is equally valuable:

- `all` accuracy is high, but within-swap-group selectivity is near shuffled. That says the probe may be reading entity class or template class, not relation identity.
- Matched patching moves margins, but mismatched vectors move them almost as much. That says the intervention is damaging corrupt evidence, not restoring clean content.
- The model does not know a relation family, especially on Tier A. That is a behavioral limitation, not a hook bug.

## Artifact reading path

1. Read `method_validation_card.md`. It tells you whether the lab earned positive language.
2. Open `plots/relation_geometry_dashboard.png`. This is the new cockpit: controlled selectivity, matched-vs-control patching, transfer, and geometry all in one frame.
3. Check `diagnostics/frozen_data_manifest.json`, `diagnostics/tokenization_audit.csv`, `diagnostics/split_balance.csv`, and `diagnostics/patch_pair_gate.csv`.
4. Check `tables/margin_by_family.csv`. Do not patch a relation the model does not know.
5. Read `plots/controlled_probe_atlas.png` and `plots/relation_probe_selectivity.png`, not just `relation_probe_by_layer.png`.
6. Read `plots/relation_evidence_matrix.png` beside `tables/relation_evidence_matrix.csv`. This is the family-by-family rent ledger.
7. Read `plots/patch_specificity_by_family.png`, `plots/patch_control_gaps.png`, and `plots/role_handoff_summary.png` before using causal language.
8. Read `tables/relation_transfer_matrix.csv` beside `plots/relation_transfer_matrix.png`.
9. Read `plots/profile_similarity_matrix.png` and `plots/relation_direction_cosines.png`, then immediately read `operationalization_audit.md`.
10. Only then edit `ledger_suggestions.md` into claims you would defend.

## Plot guide

### `relation_geometry_dashboard.png`

This is the first plot to read. It asks four questions in one place: did controlled selectivity clear shuffled labels, did matched patching beat controls, where does transfer live, and are the geometry/profile summaries only handles?

### `controlled_probe_atlas.png`

A role-by-depth atlas of swap-group selectivity. It makes the depth-0 relation-word trap visible while highlighting subject/final depths that might be more than token echo.

### `relation_evidence_matrix.png`

Rows are relation families. Columns line up behavior margins, controlled selectivity, matched patch recovery, specificity gap, and cosine/profile handles. It is the “every family pays rent” plot.

**Caveat: colors are scaled within each column, not globally.** The matrix normalizes each column on its own range, so a deep red in `hard margin` and a deep red in `final patch` are not the same magnitude — each is "high relative to this column." Use color to rank families *within* a metric and to spot empty cells (a family with no token-aligned swap partner, or one the behavioral gate dropped). Do not read a row's colors as a single comparable profile across columns; for cross-metric comparison go to the raw numbers in `tables/relation_evidence_matrix.csv`. The plot is a rent ledger — every family should pay *something* in each applicable column — not a heatmap of absolute strength.

### `relation_probe_by_layer.png`

The gray all-relation curve is allowed to be impressive and confounded at the same time. The controlled evidence lives in the swap-group curves. Dotted curves are shuffled-label controls.

### `relation_probe_selectivity.png`

This is the cleaner probe curve. It subtracts the shuffled control and shows which role/depth earned the saved relation direction. A flat line near zero means the controlled probe failed, even if raw accuracy looked high.

### `relation_patch_heatmap.png`

Rows are relation families, columns are stream depths, and color is mean recovery. Depth 0 is a sanity row. A useful profile has non-trivial recovery in the interior depth band.

### `role_handoff_summary.png`

This is the timing plot: which role peaks early, which role peaks late, and whether recovery looks like subject information being carried toward the final readout.

### `relation_swap_recovery.png`

This is the causal token-echo pressure test. Matched relation-token curves should beat the wrong-position control. If every curve tracks the control, the pretty story dissolves into steam.

### `patch_specificity_by_family.png`

A family-level causal firewall. It shows matched recovery, the stronger control floor, and the gap. Broad claims need multiple families clearing the gap, not one heroic bar.

### `patch_control_gaps.png`

This is the overclaim alarm. Matched patch recovery has to beat both wrong-position and mismatched-vector recovery before the run earns causal relation-use language.

**Reading the CAUSAL gate quantitatively.** Matched recovery alone is not the evidence — the *gap above the stronger control* is. The mismatched-vector bar is not a null; it is the recovery you get for free by destroying the corrupt run's evidence, so it sets the real floor. Subtract it:

```
content-specific recovery ≈ matched recovery − mismatched-vector recovery
```

In the run6 validation sweep this arithmetic was the whole lesson of the lab. On gpt2 (Tier A) the subject-swap bar (~0.84) clears everything decisively, but the relation-swap bar (~0.52) sits only ~0.16 above the mismatched-vector control (~0.36), while wrong-position is a clean ~0.00. That ~0.16, not the ~0.52, is the honest size of the *content-specific* relation-token effect. Now move to the 7B course model (Olmo-3-1025-7B): relation-swap matched recovery is ~0.25 against a mismatched-vector floor of ~0.26 — a gap of about **−0.01**. The probe still decodes relation identity inside the swap group with near-perfect selectivity, yet the relation-swap causal claim does not survive its controls at all (a second model, gemma-4-E4B, also fails the gate).

Two things to carry from that. First, **decodable is not causally used**: every model passes the probe and fails the relation-swap causal test. Second, gpt2's "pass" is a sliver about +0.01 over the 0.15 threshold that does *not* replicate on a real model — exactly the trap the lab warns against, and why its honest headline is "negative success," not a discovery. The one causal result that *does* travel is the **subject-swap** gap, large on both gpt2 (+0.48) and the 7B (+0.42): subject localization is robust, relation-word token echo is not. Report the gap, name the mismatched-vector floor, and never quote matched recovery as if the controls were zero.

### `relation_transfer_matrix.png`

Diagonal cells are within-relation subject swaps; within-swap-group off-diagonals are relation swaps. Cross-group cells are empty by design when token-aligned pairs do not exist.

### `profile_similarity_matrix.png`

This asks whether subject-swap localization profiles cluster by relation group. Similarity here is an `OBS` handle, not proof of shared heads or shared MLPs.

### `relation_direction_cosines.png`

Block structure is a clue. It is not a mechanism. Use it to choose the next microscope, not to write the final paragraph.

## The evidence ladder

![Three evidence levels of increasing strength. OBS geometry (cosine blocks, profile similarity) is descriptive and gated by nothing. DECODE probing must clear shuffled-label selectivity inside the swap group. CAUSAL patching must beat wrong-position and mismatched-vector controls. None of the three reaches a shared-mechanism claim, which requires head routing or feature methods.](../figures/lab12_obs_decode_causal_evidence_ladder.png)

Read the lab as a climb, not a checklist. Each rung licenses a stronger verb and demands a stronger control. OBS geometry lets you say *looks structured* — a handle for choosing the next microscope, gated by nothing, provable by nothing. DECODE lets you say *is readable here*, but only once selectivity (real minus shuffled) clears the gate inside a swap group, so entity and template are already controlled. CAUSAL lets you say *is causally used by this answer pathway*, but only once matched patching beats both the wrong-position and the mismatched-vector controls. The ceiling sits above all three: even a clean CAUSAL result buys a usable handle on the relation-word residual, not a demonstration that families share one mechanism. That last step — from handle to mechanism — is explicitly out of scope and handed to Lab 3 (which heads move the relation information) and Lab 8 (which features carry it).

## The operationalization audit

The favorite interpretation is:

> The model has shared relation geometry that it uses.

The audit attacks it with four cheap explanations:

| cheap explanation | control | allowed conclusion if control passes |
|---|---|---|
| entity-class direction | same-subject swap groups | not merely entity class |
| template/syntax direction | same-template swap groups | not merely template syntax |
| relation-word token echo | relation-swap patching | causally usable relation-token handle, not necessarily reusable geometry |
| patch machinery artifact | wrong-position and mismatched-vector controls | content-specific patching, if matched recovery beats controls |

The audit is deliberately stricter than the plots. Believe the audit more than your favorite heatmap.

## Writeup questions

1. Why is high `relword` probe accuracy at depth 0 not evidence of relation geometry?
2. If subject-role selectivity rises after depth 0, what information must have moved, and which intro method would you use next to find the mover?
3. When `all` accuracy is high but swap-group selectivity is low, what exactly did the uncontrolled probe probably learn?
4. Explain a nonzero mismatched-vector recovery without using the word “relation.”
5. Why are cross-group cells empty in the transfer matrix, and why is leaving them empty better than inventing unaligned pairs?
6. Write one positive claim and one deliberately overclaimed version of the same result. Name the artifact that separates them.
7. If the direction cosine atlas has strong same-group blocks, what follow-up would test whether those blocks correspond to shared components?

## Symptom-first debugging

| Symptom | Likely cause | Start here |
|---|---|---|
| Most items dropped at tokenization | tokenizer changed, targets are multi-token, relation/subject appears twice | `diagnostics/tokenization_audit.csv` |
| Split has a family with no eval rows | too small a cap or aggressive relation filter | `diagnostics/split_balance.csv`; raise `--relation-set` |
| High all-relation accuracy, flat swap-group selectivity | entity or template confound | `plots/controlled_probe_atlas.png`, `plots/relation_probe_selectivity.png` |
| Zero gated patch pairs | model does not know that relation under the margin gate | `tables/margin_by_family.csv`, `diagnostics/patch_pair_gate.csv` |
| Patch no-op fails | stream convention or hook site broken for this model | `diagnostics/patch_noop_check.json` |
| Relation-swap recovery high at depth 0 only | token substitution sanity check, not causal localization | `metrics.json` band convention |
| Mismatched control rivals matched patch | margin destruction, not content restoration | `plots/patch_specificity_by_family.png`, `plots/patch_control_gaps.png` |
| Transfer matrix mostly empty | relation filter removed swap partners or no token-aligned pairs exist | `tables/relation_family_manifest.csv` |

## Claim ledger guidance

Good `DECODE` claim:

```text
L12-C1 DECODE: On MODEL and frozen DATASET, relation identity is decodable at final-role stream depth D inside the country_sem swap group with accuracy A versus shuffled S, with subjects and templates controlled. This does not rule out relation-word token echo.
```

Good `CAUSAL` claim:

```text
L12-C2 CAUSAL: On gated relation-swap pairs in SWAP_GROUP, replacing the corrupt relation-word residual with the clean residual at depths 1..L−1 recovers mean margin R, beating wrong-position and mismatched-vector controls by G. This shows a causally usable relation-token handle for these prompts, not a shared mechanism.
```

Good negative claim:

```text
L12-N1 DECODE: The uncontrolled all-relation probe reached A, but swap-group selectivity stayed near shuffled. In this run, the probe did not validate relation geometry beyond entity/template confounds.
```

Bad claim:

```text
The model represents relations in a shared subspace.
```

It hides the scope, skips controls, and promotes a handle to a mechanism in one silky sentence.

## Extensions

- Add paraphrase templates within the country swap group and test whether saved directions transfer across templates. This is the direct attack on relation-word token echo.
- Use Lab 3 attention routing on high subject-role emergence depths. Which heads move relation information from `relword` to `subject`?
- Use Lab 2 DLA at the final token for one high-margin family and compare top writers across relation families.
- Use Lab 8 SAE dictionaries to label features aligned with relation directions. Report killed labels, not just tempting labels.
- Extend to multi-token subjects by mean-pooling subject spans, then document exactly what breaks relative to the single-token frame.

## Interpretation and ethics

The ethical lesson is about idealization. “Relation” is a useful scientific simplification, not a natural-kind stamp. This lab asks whether the simplification survives controlled stress. If it does, you have a handle for later social-concept labs. If it does not, you have a warning label to carry into them.

Reading prompt: did you find relation geometry, or twelve little tricks wearing one probe-shaped coat?

## References and context

The question "do relations share linear geometry, or is each one a separate trick" is an active research thread, and this lab is a controlled, deflationary take on it.

- Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" (2019) — the origin of selectivity (probe accuracy minus a control-task baseline); the `real − shuffled` metric here is the same idea.
- Hernandez et al., "Linearity of Relation Decoding in Transformer Language Models" (2024) — many relations are approximated by a linear relational embedding (a learned affine map subject → object). This lab asks whether such structure survives entity/template controls.
- Todd et al., "Function Vectors in Large Language Models" (2024) — a single vector, added at a position, can trigger a relation/task. The relation-swap patch is a controlled cousin of this.
- Hendel et al., "In-Context Learning Creates Task Vectors" (2023) — relations/tasks compressed into a transportable activation.
- Merullo et al., "Language Models Implement Simple Word2Vec-style Vector Arithmetic" (2024) — evidence for relation arithmetic in the residual stream; a "handle" worth distinguishing from a mechanism.

The contrast worth teaching: those papers mostly report that relation handles *exist and transport*. Lab 12's job is the harder, less glamorous half — showing how much of an apparent relation handle is entity class, template syntax, or token echo before any of it counts.
