# Lab 33: Multimodal Mechanistic Interpretability

```text
Time estimate: 10-20 minutes Tier A synthetic smoke; Tier B currently runs the same synthetic audit unless a real-VLM extension is added
Compute tier: offline synthetic connector plus standard bench microscope checks; no real VLM hooks in the default lab
Dependencies: Labs 1, 4, 5, 8, 13, 27, and 31
Minimum passing artifacts: method_card.md, operationalization_audit.md, diagnostics/self_check_status.json, diagnostics/alignment_validation.json, diagnostics/run_config_snapshot.json, diagnostics/warning_summary.csv, tables/multimodal_evidence_matrix.csv, tables/multimodal_patch_report.csv, tables/patch_dose_response.csv, tables/failure_specimens.csv, plot_manifest.json, plots/multimodal_evidence_dashboard.png
Main plot: plots/multimodal_evidence_dashboard.png
Main table: tables/multimodal_evidence_matrix.csv
Evidence rung: OBS + DECODE + CAUSAL + AUDIT, scoped to synthetic connector mode only
Forbidden claim: The VLM has a human-like visual concept of color.
One-sentence allowed claim: The synthetic connector smoke path rendered controlled images, exercised modality readout and patching semantics, and identified the gates a real VLM run must clear before visual-mechanism claims.
Human-label requirement: none for default synthetic labels; any real-VLM extension must hand-check ambiguous images, OCR, background, crop/resize, region alignment, and caption leakage rows
```

## Lab thesis

Multimodal interpretability has a trapdoor under every pretty image. A patch that looks visual can be a caption shortcut, an OCR shortcut, a background shortcut, or a region-alignment bug wearing a lab coat.

Lab 33 therefore starts with a synthetic connector. The connector is not pretending to be a vision-language model. It is a calibration rig for the artifact contract:

```text
image spec -> rendered image -> known visual/text/shortcut channels -> readout -> patch -> leak audit -> real-VLM readiness verdict
```

The default run should end with `science_ready=false`. That flag is not a failure. It is the lab keeping synthetic connector evidence separate from real-model evidence.

## What question this lab asks

The lab asks a methods question first:

```text
Can we build a multimodal audit package that separates visual evidence, text evidence, shortcut evidence, patch semantics, and alignment assumptions before touching a real VLM?
```

Only after a future real-VLM extension supplies real visual tokens, connector states, language states, and region mappings can the science question be asked:

```text
Does an image-derived feature at a connector site causally affect the answer more than OCR, background, caption, text-query, random, and wrong-region controls?
```

The default synthetic connector can test the measurement choreography. It cannot validate a real model mechanism.

## Why this matters in the course progression

Text-only labs often assume that a token position is a stable object. Multimodal models make that assumption slippery. Image patches can correspond to variable numbers of image tokens. Rendered text can leak labels. Backgrounds can correlate with answers. Captions and user text can carry the answer before the model looks at the image.

Labs 26-31 taught formal hypotheses, path proxies, training dynamics, feature lineage, and automated-label audits. Lab 33 applies that same discipline to modalities. The goal is not to ship a flashy VLM demo. The goal is to prevent a flashy VLM demo from being mistaken for evidence.

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

Tier A now has a built-in smoke fallback if the frozen JSONL is absent. That fallback is marked as `data_source=builtin_tier_a_smoke_fallback`, `fallback_data=true`, and `science_ready=false` in the manifest and warnings. It exists to test artifact plumbing only. A Tier B science run should use the frozen JSONL or fail loudly.

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

A real VLM extension should replace these with captured states from the model. The table names are useful audit bins, not evidence that those bins have already been found in a real network.

## What the experiment measures

The central score is a clean-vs-corrupt recovery margin:

```text
margin = projection(target value) - projection(distractor value)
recovery = (patched_margin - corrupt_margin) / (clean_margin - corrupt_margin)
```

This is synthetic. The value directions are deterministic hash-basis directions. The score tests whether the audit code and controls behave as intended.

The patch battery is:

| patch | role | purpose |
|---|---|---|
| `connector_clean_to_corrupt` | target upper bound | full synthetic connector interchange |
| `visual_region_patch` | target | visual-only patch, the closest proxy to image-derived evidence |
| `language_state_patch` | downstream target | downstream synthetic language state patch |
| `caption_patch` | positive control | answer-bearing caption channel can carry the answer |
| `ocr_channel_patch` | shortcut control | rendered-text shortcut control |
| `background_channel_patch` | shortcut control | background shortcut control |
| `text_query_patch` | text control | question-only state should not carry answer labels |
| `wrong_region_or_ocr_patch` | wrong-site/shortcut control | wrong channel or shortcut patch |
| `random_patch_control` | random control | detects perturbation or damage masquerading as recovery |

The lab reports specificity as:

```text
specificity_gap = visual_region_patch_recovery - max(ocr, background, text_query, wrong_region, random controls)
```

Connector recovery can be high and still not justify a visual claim. The visual-only patch must beat shortcut controls.

## Second-pass visualization and data contract

This pass upgrades the lab from a plot bundle into an inspectable evidence package. The goal is not more pictures. The goal is better raw material, clear source tables, and plots that make uncertainty and negative evidence harder to lose.

New or strengthened data artifacts:

| Artifact | Why it exists |
|---|---|
| `diagnostics/run_config_snapshot.json` | Captures lab id, tier, prompt set, model loaded by the bench, seed, thresholds, synthetic/fallback status, and alignment scope. |
| `diagnostics/warning_summary.csv` and `.json` | One machine-readable place for fallback data, failed schema rows, missing real-VLM alignment, shortcut triggers, text leaks, low sample counts, and control-matched rows. |
| `tables/state_score_long.csv` | Per-item, per-variant, per-state target/distractor projections and margins. This is the raw readout table under the aggregate AUCs. |
| `tables/patch_dose_response.csv` | Per-item patch dose sweep for visual, language, caption, OCR, background, text-query, and random patches. |
| `tables/multimodal_dose_response.csv` | Backward-friendly alias of the dose table. |
| `tables/patch_recovery_summary.csv` | Mean, standard error, min, max, and row count per patch type. |
| `tables/patch_type_summary.csv` | Backward-friendly alias of patch summary. |
| `tables/concept_family_summary.csv` | Family-level visual recovery, control floor, specificity gap, shortcut rate, text leak rate, and claim posture. |
| `tables/failure_specimens.csv` and `.jsonl` | Rows where controls match or beat visual, shortcuts fire, text leaks, or family-level evidence is blocked. |
| `cards/failure_specimens.md` | Human-readable specimen card list for the rows students should inspect before writing claims. |
| `tables/figure_sources/*.csv` | The exact source table used by each figure, so plots are not transient in-memory magic. |
| `tables/plot_manifest.csv` and `plot_manifest.json` | Figure path, source table, row count, metric, controls, claim supported, and interpretation caution. |

Stable identifiers were added where the prior tables were mostly human-readable rows: `item_uid`, `probe_cell_id`, `patch_row_id`, `specificity_row_id`, `transfer_cell_id`, `leak_row_id`, `text_leak_id`, `evidence_row_id`, `counterexample_id`, `dose_response_id`, and `score_row_id`.

## Controls and falsifiers

| Control | What it attacks | What a failure means |
|---|---|---|
| OCR traps | The model reads rendered text instead of the visual object. | Visual language must be downgraded to shortcut language. |
| Background traps | The model uses background color or texture. | Visual patch recovery is not object-specific. |
| Text-query audit | The text prompt itself contains the answer. | The result may be prompt leakage, not image evidence. |
| Caption positive control | A caption-like channel can carry the answer. | Useful upper-bound control, not visual evidence. |
| Wrong-region/shortcut patch | Patching any region or channel moves the margin. | Region/site labels are not earning their keep. |
| Random patch control | The patch destroys corrupt evidence rather than restoring clean evidence. | Recovery is not content-specific. |
| Alignment validation | Image-token or region mapping is not stable enough for patching. | Real-VLM claims stay blocked. |
| Synthetic-mode science flag | No real VLM states were captured. | The run is an audit package, not a model mechanism result. |

A control that matches the visual patch is not a nuisance. It is the warning bell for an overbroad claim.

## How to run

From `interpretability/` after adding Lab 33 to the bench registry:

```bash
python interp_bench.py --lab lab33 --tier a --no-plots
python interp_bench.py --lab lab33 --tier a
python interp_bench.py --lab lab33 --tier b --prompt-set full
```

To regenerate the frozen JSONL and the Lab 33 manifest:

```bash
python data/make_multimodal_concept_pairs.py
```

Tier A proves table plumbing, rendering, self-checks, warnings, manifests, and plot generation. Tier B currently runs the same synthetic connector audit with a larger selected row set if the frozen data is present. A future real-VLM extension should add an explicit mode flag rather than silently changing the claim semantics.

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
  plot_manifest.json

  diagnostics/
    data_manifest.json
    schema_audit.csv
    render_validation.csv
    state_vector_audit.csv
    alignment_validation.json
    run_config_snapshot.json
    warning_summary.csv
    warning_summary.json
    self_check_status.json
    safety_status.json

  tables/
    multimodal_prompt_manifest.csv
    modality_probe_report.csv
    state_score_long.csv
    multimodal_patch_report.csv
    multimodal_specificity_controls.csv
    patch_dose_response.csv
    multimodal_dose_response.csv
    patch_recovery_summary.csv
    patch_type_summary.csv
    concept_family_summary.csv
    cross_modal_transfer.csv
    ocr_background_leak_audit.csv
    text_leak_audit.csv
    multimodal_evidence_matrix.csv
    multimodal_counterexamples.csv
    failure_specimens.csv
    failure_specimens.jsonl
    plot_reading_guide.csv
    plot_manifest.csv
    figure_sources/*.csv

  cards/
    failure_specimens.md

  plots/
    multimodal_evidence_dashboard.png
    target_vs_control.png
    dose_response.png
    modality_handoff_atlas.png
    image_text_probe_transfer.png
    patch_recovery_by_modality.png
    concept_specificity_matrix.png
    spatial_region_patch_map.png
    cross_modal_feature_bridge.png
    ocr_background_shortcut_panel.png
    paired_examples.png

  state/
    rendered_images_manifest.csv
    images/*.png
    multimodal_directions.pt
    multimodal_state_metadata.json
```

## Reading order

Start with `run_summary.md`. It states the non-claim: this run is synthetic smoke, not real VLM evidence.

Then read:

1. `diagnostics/warning_summary.csv`: stop here if fallback data, text leakage, shortcut triggers, or control-matched visual rows are marked as blockers.
2. `diagnostics/schema_audit.csv` and `diagnostics/data_manifest.json`: verify the frozen suite and whether fallback data was used.
3. `state/rendered_images_manifest.csv` and `tables/multimodal_prompt_manifest.csv`: inspect image specs, questions, regions, and shortcuts.
4. `diagnostics/alignment_validation.json`: confirm why real VLM claims are blocked by default.
5. `tables/state_score_long.csv`: inspect clean/corrupt margins before trusting aggregate probe AUCs.
6. `tables/modality_probe_report.csv`: compare visual, connector, caption, and text-query readouts.
7. `tables/multimodal_patch_report.csv`, `tables/patch_dose_response.csv`, and `tables/multimodal_specificity_controls.csv`: compare visual recovery with shortcut, text, wrong-region, and random controls.
8. `tables/ocr_background_leak_audit.csv` and `tables/text_leak_audit.csv`: read these before writing any visual claim.
9. `tables/failure_specimens.csv` and `cards/failure_specimens.md`: inspect the rows that narrow or defeat broad language.
10. `tables/multimodal_evidence_matrix.csv` and `operationalization_audit.md`: write only the smallest supported claim.

Only after this should you open the plots. The dashboard is a map, not the underlying evidence.

## How to read the figures

Each plot answers one question and has a matching source CSV under `tables/figure_sources/`. The manifest records the source table, row count, main metric, control comparison, and the claim boundary.

| Plot | Source table | Question | How to read it |
|---|---|---|---|
| `multimodal_evidence_dashboard.png` | `tables/figure_sources/multimodal_evidence_dashboard.csv` | Does the synthetic audit package show readout, patch, shortcut, and readiness status in one place? | Use it as an overview. If shortcut or readiness panels look bad, do not cite visual language. |
| `target_vs_control.png` | `tables/figure_sources/target_vs_control.csv` | Does visual patch recovery beat the strongest shortcut/text/random control item by item? | Raw rows matter. A mean gap with many near-ties is a fragile result. |
| `dose_response.png` | `tables/figure_sources/dose_response.csv` | Does recovery change smoothly with patch dose, and do controls stay lower across dose? | A single lucky dose is weaker than a broad separation curve. |
| `modality_handoff_atlas.png` | `tables/figure_sources/modality_handoff_atlas.csv` | Which synthetic state types carry answer information by concept family? | High caption or connector AUC is expected. High text-query AUC is a leak warning. |
| `image_text_probe_transfer.png` | `tables/figure_sources/image_text_probe_transfer.csv` | Do image, caption, and text-query directions transfer? | Caption transfer is a positive control. Text-query transfer is not visual evidence. |
| `patch_recovery_by_modality.png` | `tables/figure_sources/patch_recovery_by_modality.csv` | Which patch types recover the clean margin and how variable are raw rows? | Bars are summaries; dots are where overclaiming hides. |
| `concept_specificity_matrix.png` | `tables/figure_sources/concept_specificity_matrix.csv` | Which families clear visual recovery, control floor, gap, and leak checks? | A bright visual recovery cell with a bright control cell is not a pass. |
| `spatial_region_patch_map.png` | `tables/figure_sources/spatial_region_patch_map.csv` | Are spatial rows region-specific or do wrong-region controls keep up? | If wrong-region recovery matches visual recovery, region semantics are not supported. |
| `cross_modal_feature_bridge.png` | `tables/figure_sources/cross_modal_feature_bridge.csv` | How do synthetic visual, connector, and caption directions align? | This debugs bridge semantics. It is not evidence of a real shared VLM feature. |
| `ocr_background_shortcut_panel.png` | `tables/figure_sources/ocr_background_shortcut_panel.csv` | Which OCR/background rows trigger shortcut behavior? | Shortcut rows must be inspected before claims are drafted. |
| `paired_examples.png` | `tables/figure_sources/paired_examples.csv` | Which specimens most clearly narrow, block, or contradict the aggregate story? | Read these examples after the dashboard and before the ledger suggestion. |

## Expected Tier A smoke behavior

A healthy Tier A smoke run should have:

- deterministic images rendered for clean and corrupt specs;
- schema, state-vector, and render audits written;
- `science_ready=false` in every headline artifact;
- `warning_summary.csv` present even when only informational warnings exist;
- all plot source tables and `plot_manifest.json` written;
- figures generated even for small sample counts, with placeholder panels where a subset is absent;
- failure specimens present when shortcuts or control-matched rows exist.

If the frozen JSONL is absent, Tier A may use the built-in smoke fallback. That run is plumbing only and should not enter the claim ledger as science.

## Expected Tier B synthetic behavior

A healthy Tier B synthetic run should have:

- frozen JSONL loaded rather than fallback rows;
- more rows per concept family;
- visual/connector synthetic readouts above chance for normal rows;
- text-query readout near chance unless the prompt leaks labels;
- visual patches beating random and shortcut controls on normal rows;
- OCR/background trap rows flagged as blockers for real-VLM claims;
- enough rows for standard-error bars to be meaningful;
- `science_ready=false`, because real VLM states are still absent.

Tier B improves evidence about the audit instrument. It still does not become a VLM mechanism result.

## Honest negative outcomes

| Pattern | Interpretation |
|---|---|
| `random_patch_control` matches `visual_region_patch` | The recovery score may be measuring perturbation or damage rather than content-specific visual mediation. |
| OCR/background controls beat the visual patch | Shortcut channels explain the effect better than visual object evidence. |
| `text_query` decodes labels | The prompt suite leaks the answer. |
| Caption dominates visual recovery | The positive control works, but the result should be written as caption-like evidence, not visual evidence. |
| Wrong-region control matches visual patch | Region alignment or site specificity failed. |
| `science_ready=true` in default synthetic mode | The readiness gate is broken. |
| Failure specimens are empty in a shortcut-heavy suite | The specimen collector or warnings are probably too forgiving. |

A negative result is not a failed lab. It is the lab doing its job before a real VLM run can produce expensive but ambiguous artifacts.

## What this lab can claim

Allowed in the default run:

```text
AUDIT + CONSTRUCTION: The synthetic connector suite produced a complete multimodal audit package, including rendered images, modality readouts, dose-response patch controls, shortcut audits, warning summaries, failure specimens, and a science-readiness block for real-VLM claims.
```

Allowed only after a real-VLM extension:

```text
DECODE + CAUSAL: On model M and data family F, captured connector state S carried image-derived signal above OCR/background/caption/text/random controls, and patching S changed the answer margin by X with specificity gap Y under alignment report A.
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
2. A tokenizer and image-processor alignment report with image-token counts, patch layout, region mapping, resize/crop metadata, and crop failure cases.
3. Clean/corrupt image pairs with matched size, crop, background, OCR, compression, and prompt conditions.
4. Region patching that maps a rendered region to actual image-token indices.
5. OCR and background trap rows that fail the claim if they match the visual patch.
6. Text-only, caption-only, image-only, and masked-image controls.
7. Hand review of ambiguous images, rendered-text rows, and background traps.
8. A new method card stating exactly which real model states were captured.
9. A run config snapshot that distinguishes synthetic connector mode from real-VLM mode in the CLI, not just in prose.

## Common failure modes

| Symptom | Likely cause | Inspect |
|---|---|---|
| no images appear | renderer dependency or path issue | `diagnostics/render_validation.csv` |
| all rows use fallback data | frozen JSONL missing | `diagnostics/data_manifest.json`, `diagnostics/warning_summary.csv` |
| text-query AUC is high | prompt leaked target or distractor | `tables/text_leak_audit.csv` |
| OCR/background controls beat visual patch | shortcut channels explain the effect | `tables/ocr_background_leak_audit.csv`, `target_vs_control.png` |
| all patches recover equally | recovery metric is nonspecific | `tables/patch_dose_response.csv`, `patch_recovery_by_modality.png` |
| spatial plot is a placeholder | no spatial rows selected by cap | `tables/multimodal_prompt_manifest.csv` |
| plot manifest missing rows | figure source table generation failed | `plot_manifest.json`, `tables/figure_sources/` |
| real-VLM claim appears in ledger suggestion | claim boundary bug | `method_card.md`, `ledger_suggestions.md` |

## Questions for the writeup

1. Which control most threatens the visual patch story in this run?
2. Does visual recovery beat controls broadly, or is the pattern carried by one specimen?
3. Is caption transfer helping calibrate the suite, or is it being accidentally interpreted as visual evidence?
4. Which figure would you open first if someone claimed the model had a color concept?
5. What exactly would a real-VLM extension need to measure before the synthetic `science_ready=false` block could be lifted?

## Interpretation and ethics

Multimodal interpretability is especially vulnerable to pictorial persuasion. A figure of a red object and a rising recovery curve can feel like direct evidence that the model saw red. This lab is designed to slow that sentence down.

The ethical hazard is not only harmful content. It is false confidence about model perception. If a medical, robotics, or accessibility system is justified by a visualization that cannot separate image evidence from OCR, background, captions, or prompt leakage, the plot has become a weak safety case. Lab 33’s job is to make that weakness visible.
