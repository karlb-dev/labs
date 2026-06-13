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

  plots/
    relation_probe_by_layer.png
    relation_probe_selectivity.png
    relation_patch_heatmap.png
    relation_swap_recovery.png
    relation_transfer_matrix.png
    patch_control_gaps.png
    relation_direction_cosines.png

  state/
    relation_directions.pt
    relation_directions_metadata.json
```

The headline artifacts are `method_validation_card.md`, `plots/relation_probe_selectivity.png`, `plots/relation_swap_recovery.png`, `plots/patch_control_gaps.png`, `tables/relation_transfer_matrix.csv`, and `operationalization_audit.md`.

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
2. Check `diagnostics/frozen_data_manifest.json`, `diagnostics/tokenization_audit.csv`, and `diagnostics/split_balance.csv`.
3. Check `tables/margin_by_family.csv`. Do not patch a relation the model does not know.
4. Read `plots/relation_probe_selectivity.png`, not just `relation_probe_by_layer.png`.
5. Read `diagnostics/patch_pair_gate.csv`, then `plots/patch_control_gaps.png`.
6. Read `tables/relation_transfer_matrix.csv` beside `plots/relation_transfer_matrix.png`.
7. Read `plots/relation_direction_cosines.png`, then immediately read `operationalization_audit.md`.
8. Only then edit `ledger_suggestions.md` into claims you would defend.

## Plot guide

### `relation_probe_by_layer.png`

The gray all-relation curve is allowed to be impressive and confounded at the same time. The controlled evidence lives in the swap-group curves. Dotted curves are shuffled-label controls.

### `relation_probe_selectivity.png`

This is the cleaner probe plot. It subtracts the shuffled control and shows which role/depth earned the saved relation direction. A flat line near zero means the controlled probe failed, even if raw accuracy looked high.

### `relation_patch_heatmap.png`

Rows are relation families, columns are stream depths, and color is mean recovery. Depth 0 is a sanity row. A useful profile has non-trivial recovery in the interior depth band.

### `relation_swap_recovery.png`

This is the causal token-echo pressure test. Matched relation-token curves should beat the wrong-position control. If every curve tracks the control, the pretty story dissolves into steam.

### `patch_control_gaps.png`

This is the overclaim alarm. Matched patch recovery has to beat both wrong-position and mismatched-vector recovery before the run earns causal relation-use language.

### `relation_direction_cosines.png`

Block structure is a clue. It is not a mechanism. Use it to choose the next microscope, not to write the final paragraph.

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
| High all-relation accuracy, flat swap-group selectivity | entity or template confound | `plots/relation_probe_selectivity.png` |
| Zero gated patch pairs | model does not know that relation under the margin gate | `tables/margin_by_family.csv`, `diagnostics/patch_pair_gate.csv` |
| Patch no-op fails | stream convention or hook site broken for this model | `diagnostics/patch_noop_check.json` |
| Relation-swap recovery high at depth 0 only | token substitution sanity check, not causal localization | `metrics.json` band convention |
| Mismatched control rivals matched patch | margin destruction, not content restoration | `plots/patch_control_gaps.png` |
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
