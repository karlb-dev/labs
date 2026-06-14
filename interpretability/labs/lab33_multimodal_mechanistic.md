# Lab 33: Multimodal Mechanistic Interpretability

```text
Time estimate: 10-20 minutes Tier A synthetic smoke; Tier B currently runs the same synthetic audit unless a real-VLM extension is added
Compute tier: offline synthetic connector plus standard bench microscope checks; no real VLM hooks in the default lab
Dependencies: Labs 1, 4, 5, 8, 13, 27, and 31
Minimum passing artifacts: method_card.md, operationalization_audit.md, diagnostics/self_check_status.json, diagnostics/alignment_validation.json, tables/multimodal_evidence_matrix.csv, tables/multimodal_patch_report.csv, plots/multimodal_evidence_dashboard.png
Main plot: plots/multimodal_evidence_dashboard.png
Main table: tables/multimodal_evidence_matrix.csv
Evidence rung: OBS + DECODE + CAUSAL + AUDIT, scoped to synthetic connector mode only
Forbidden claim: The VLM has a human-like visual concept of color.
One-sentence allowed claim: The synthetic connector smoke path rendered controlled images, exercised modality readout and patching semantics, and identified the gates a real VLM run must clear before visual-mechanism claims.
Human-label requirement: none for default synthetic labels; any real-VLM extension must hand-check ambiguous images, OCR, background, and caption leakage rows
```

## Lab thesis

Multimodal interpretability has a trapdoor under every pretty image. A patch that looks visual can be a caption shortcut, an OCR shortcut, a background shortcut, or a region-alignment bug wearing a lab coat.

Lab 33 therefore starts with a synthetic connector. The connector is not pretending to be a vision-language model. It is a calibration rig for the artifact contract:

```text
image spec -> rendered image -> known visual/text/shortcut channels -> readout -> patch -> leak audit -> real-VLM readiness verdict
```

The default run should end with `science_ready=false`. That flag is not a failure. It is the lab being honest.

## What question this lab asks

The lab asks a methods question first:

```text
Can we build a multimodal audit package that separates visual evidence, text evidence, shortcut evidence, patch semantics, and alignment assumptions before touching a real VLM?
```

Only after a future real-VLM extension supplies real visual tokens, connector states, and language states can the science question be asked:

```text
Does an image-derived feature at a connector site causally affect the answer more than OCR, background, caption, random, and wrong-region controls?
```

The default synthetic connector can test the measurement choreography. It cannot validate a real model mechanism.

## Why this matters in the course progression

Text-only labs often assume that a token position is a stable object. Multimodal models make that assumption slippery. Image patches can correspond to variable numbers of image tokens. Rendered text can leak labels. Backgrounds can correlate with answers. Captions and user text can carry the answer before the model looks at the image.

Labs 26-31 taught formal hypotheses, path proxies, training dynamics, feature lineage, and automated-label audits. Lab 33 applies that same discipline to modalities. The goal is not to ship a flashy VLM demo. The goal is to prevent a flashy VLM demo from quietly counterfeiting evidence.

## Data

The default frozen suite lives at:

```text
data/multimodal_concept_pairs.jsonl
```

It is generated deterministically by:

```text
data/make_multimodal_concept_pairs.py
```

Each JSONL row contains:

```json
{
  "item_id": "color_red_blue_001",
  "image_path": "generated/color_red_blue_001_clean.png",
  "image_control_path": "generated/color_red_blue_001_corrupt.png",
  "question": "What color is the main object?",
  "target": "red",
  "distractor": "blue",
  "concept_family": "color",
  "text_control_prompt": "Answer the visual question using only the rendered image.",
  "split": "train",
  "image_spec": {},
  "control_spec": {},
  "region": "object",
  "notes": "..."
}
```

The lab renders PNGs inside the run directory. The repository does not need tracked binary images.

## Synthetic connector states

The run builds a known feature basis. For every clean and corrupt item it creates these state families:

| state | what it contains | safe interpretation |
|---|---|---|
| `vision_visual` | object, count, spatial, or chart feature only | visual signal with OCR/background removed |
| `vision` | visual signal plus OCR/background shortcut channels | what a careless visual readout might measure |
| `connector` | mixture of visual, caption-like, and question state | synthetic connector patch semantics |
| `language` | downstream mixture after connector | synthetic downstream language state |
| `caption` | answer-bearing synthetic caption state | useful positive control, not user text |
| `text_query` | question/family state without the answer label | text-only leakage check |
| `ocr_channel` | rendered text shortcut channel | leak control |
| `background_channel` | background shortcut channel | leak control |

A real VLM extension should replace these with captured states from the model. The table names are kept because they are useful audit bins, not because this lab has already found those bins in a real network.

## What the experiment measures

The central score is a clean-vs-corrupt recovery margin:

```text
margin = projection(target value) - projection(distractor value)
recovery = (patched_margin - corrupt_margin) / (clean_margin - corrupt_margin)
```

This is synthetic. The value directions are deterministic hash-basis directions. The score tests whether the audit code and controls behave as intended.

The main patch comparison is:

| patch | purpose |
|---|---|
| `connector_clean_to_corrupt` | upper-bound synthetic connector interchange |
| `visual_region_patch` | visual-only patch, the closest proxy to image-derived evidence |
| `language_state_patch` | downstream synthetic language state patch |
| `caption_patch` | answer-bearing caption positive control |
| `ocr_channel_patch` | rendered-text shortcut control |
| `background_channel_patch` | background shortcut control |
| `text_query_patch` | question-only text control |
| `wrong_region_or_ocr_patch` | wrong channel / shortcut patch |
| `random_patch_control` | deterministic random-vector control |

The lab reports specificity as:

```text
specificity_gap = visual_region_patch_recovery - max(ocr, background, text_query, wrong_region, random controls)
```

Connector recovery can be high and still not justify a visual claim. The visual-only patch must beat shortcut controls.

## Controls and falsifiers

| Control | What it attacks |
|---|---|
| OCR traps | The model reads rendered text instead of the visual object. |
| Background traps | The model uses background color or texture. |
| Text-query audit | The text prompt itself contains the answer. |
| Caption positive control | A caption-like channel can carry the answer; this is not visual evidence. |
| Wrong-region/shortcut patch | Patching any region or channel moves the margin. |
| Random patch control | The patch destroys corrupt evidence rather than restoring clean evidence. |
| Alignment validation | Image-token or region mapping is not stable enough for patching. |
| Synthetic-mode science flag | No real VLM states were captured. |

A control that matches the visual patch is not a nuisance. It is the warning bell for an overbroad claim.

## How to run

From `interpretability/`:

```bash
python interp_bench.py --lab lab33 --tier a --no-plots
python interp_bench.py --lab lab33 --tier a
python interp_bench.py --lab lab33 --tier b --prompt-set full
```

To regenerate the frozen JSONL and the Lab 33 manifest:

```bash
python data/make_multimodal_concept_pairs.py
```

Tier A and Tier B currently run the same synthetic connector audit, with different row caps. A future real-VLM extension should add an explicit mode flag rather than silently changing the claim semantics.

## Artifact tree

```text
runs/lab33_multimodal_mechanistic-*/
  run_summary.md
  method_card.md
  operationalization_audit.md
  real_vlm_extension_checklist.md
  metrics.json
  results.csv
  ledger_suggestions.md

  diagnostics/
    data_manifest.json
    schema_audit.csv
    render_validation.csv
    state_vector_audit.csv
    alignment_validation.json
    self_check_status.json
    safety_status.json

  tables/
    multimodal_prompt_manifest.csv
    modality_probe_report.csv
    multimodal_patch_report.csv
    multimodal_specificity_controls.csv
    cross_modal_transfer.csv
    ocr_background_leak_audit.csv
    text_leak_audit.csv
    multimodal_evidence_matrix.csv
    multimodal_counterexamples.csv
    plot_reading_guide.csv

  plots/
    multimodal_evidence_dashboard.png
    modality_handoff_atlas.png
    image_text_probe_transfer.png
    patch_recovery_by_modality.png
    concept_specificity_matrix.png
    spatial_region_patch_map.png
    cross_modal_feature_bridge.png
    ocr_background_shortcut_panel.png

  state/
    rendered_images_manifest.csv
    images/*.png
    multimodal_directions.pt
    multimodal_state_metadata.json
```

## Reading order

Start with `method_card.md`. It states the non-claim: this run is synthetic smoke, not real VLM evidence.

Then read:

1. `diagnostics/schema_audit.csv` and `diagnostics/data_manifest.json` to check the frozen suite.
2. `state/rendered_images_manifest.csv` and `tables/multimodal_prompt_manifest.csv` to inspect image specs, questions, regions, and shortcuts.
3. `diagnostics/alignment_validation.json` to see why real VLM claims are blocked by default.
4. `tables/modality_probe_report.csv` to compare visual, connector, caption, and text-query readouts.
5. `tables/multimodal_patch_report.csv` and `tables/multimodal_specificity_controls.csv` to compare visual patch recovery with shortcut controls.
6. `tables/ocr_background_leak_audit.csv` and `tables/text_leak_audit.csv` before writing any visual claim.
7. `tables/multimodal_evidence_matrix.csv` and `tables/multimodal_counterexamples.csv` for the claim posture.
8. `operationalization_audit.md` before editing `ledger_suggestions.md`.

## Figure guide

### `multimodal_evidence_dashboard.png`

The cockpit plot. It shows mean readout AUC by state, visual patch versus controls, leak-trigger counts, and real-VLM readiness.

### `modality_handoff_atlas.png`

State-type AUC by concept family. This shows whether synthetic visual, connector, language, caption, and text-query states behave differently.

### `image_text_probe_transfer.png`

Cross-modal direction transfer. Read caption transfer as a positive control and text-query transfer as a leakage check.

### `patch_recovery_by_modality.png`

Patch recovery by intervention type. This is the anti-vibes plot: visual recovery must beat OCR/background/text/random controls before visual language is allowed.

### `concept_specificity_matrix.png`

Family-level specificity gaps. Bright cells are synthetic audit passes, not real concepts.

### `spatial_region_patch_map.png`

Spatial rows only. It checks whether the region story is specific or whether wrong-region controls keep up.

### `cross_modal_feature_bridge.png`

Direction cosine and transfer summary between visual and caption-like channels. Useful for debugging bridge semantics, not evidence of a real shared mechanism.

### `ocr_background_shortcut_panel.png`

Shortcut rows get their own court hearing. This plot should make OCR/background traps visible before a student can overclaim.

## Expected outcomes

A healthy synthetic run should have:

- deterministic images rendered for every clean and corrupt spec;
- synthetic readout AUC above chance for visual and connector states;
- text-query readout near chance unless the text prompt leaks labels;
- connector and visual patches beating random controls on normal rows;
- OCR/background trap rows flagged as blockers for real-VLM claims;
- `science_ready=false` in every headline artifact.

A negative synthetic run is useful when it points to a broken audit instrument. For example, if `random_patch_control` matches `visual_region_patch`, the patch recovery score is not specific enough. If `text_query` decodes labels, the prompt suite leaks the answer.

## What this lab can claim

Allowed in the default run:

```text
AUDIT + CONSTRUCTION: The synthetic connector suite produced a complete multimodal audit package, including rendered images, modality readouts, patch controls, shortcut audits, and a science-readiness block for real-VLM claims.
```

Allowed only after a real-VLM extension:

```text
DECODE + CAUSAL: On model M and data family F, captured connector state S carried image-derived signal above OCR/background/caption/text/random controls, and patching S changed the answer margin by X with specificity gap Y.
```

## What this lab cannot claim

Do not write:

```text
The VLM sees the red object.
The model has a human-like visual concept of color.
The connector is aligned because the plot looks right.
The synthetic feature basis transfers to real images.
The OCR trap is safe because average performance is high.
```

## Real-VLM extension checklist

A future real-VLM version should add these before changing `science_ready`:

1. A model class that accepts image inputs and exposes vision, connector, and language states.
2. A tokenizer and image-processor alignment report with image-token counts, patch layout, region mapping, and resize/crop metadata.
3. Clean/corrupt image pairs with matched size, crop, background, and OCR conditions.
4. Region patching that maps a rendered region to actual image-token indices.
5. OCR and background trap rows that fail the claim if they match the visual patch.
6. Text-only, caption-only, and image-only controls.
7. Hand review of ambiguous images and any rendered-text rows.
8. A new method card that states exactly which real model states were captured.

## Common failure modes

| Symptom | Likely cause | Inspect |
|---|---|---|
| no images appear | renderer dependency or path issue | `diagnostics/render_validation.csv` |
| text-query AUC is high | prompt leaked target or distractor | `tables/text_leak_audit.csv` |
| OCR/background controls beat visual patch | shortcut channels explain the effect | `tables/ocr_background_leak_audit.csv` |
| random patch matches visual patch | recovery measures disruption rather than content restoration | `tables/multimodal_specificity_controls.csv` |
| dashboard looks positive but method card blocks claims | synthetic mode cannot support real-VLM language | `method_card.md` |
| spatial plot is blank | Tier cap selected no spatial rows | run `--prompt-set full` or increase `--max-examples` |

## Suggested extensions

1. Add a small open VLM path behind `--mode real_vlm`, with explicit alignment diagnostics.
2. Add image-token region patching for object, background, OCR text, and chart bars.
3. Add paired captions that deliberately leak the answer and prove caption-only controls block claims.
4. Add human-review columns for ambiguous renders.
5. Replace hash-basis directions with real activation probes once captured VLM states exist.
6. Carry the same shortcut gates into Lab 35 as part of a reproducible multimodal mini-study.

## Writeup questions

1. Which concept families passed the synthetic visual specificity gate?
2. Which shortcut control was most dangerous: OCR, background, text-query, wrong-region, or random?
3. Did caption transfer behave like a positive control while text-query stayed low?
4. Which row would block a real-VLM claim first?
5. What exact alignment diagnostic would your real-VLM extension need before patching a region?

## Ledger templates

Synthetic default run:

```text
[L33-C1] AUDIT + CONSTRUCTION | The Lab 33 synthetic connector suite rendered N image pairs and produced modality readout, patch, shortcut, alignment, and evidence-matrix artifacts. The run is not real-VLM evidence because science_ready=false.
Artifact: runs/<run>/tables/multimodal_evidence_matrix.csv | Falsifier: schema, rendering, self-check, or shortcut audit fails.
```

Future real-VLM run, only after the checklist passes:

```text
[L33-R1] DECODE + CAUSAL | On <model>, visual-region connector state for <family> recovered <X> of the clean-vs-corrupt margin, beating OCR/background/caption/text/random controls by <Y> after alignment validation.
Artifact: runs/<run>/tables/multimodal_specificity_controls.csv | Falsifier: OCR/background/text controls match recovery, or region alignment fails under image resizing/cropping.
```

## Safety and scope

The default suite uses benign synthetic shapes, counts, spatial positions, and charts. It does not analyze people, faces, identity, surveillance, private images, medical images, or real-world sensitive imagery. The safety status file exists so that multimodal labs inherit the same explicit boundary-setting as the text labs.
