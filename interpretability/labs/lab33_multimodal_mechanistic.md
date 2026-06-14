# Lab 33: Multimodal Mechanistic Interpretability

Time estimate: 60-90 minutes for the synthetic connector smoke path.  
Compute tier: Tier A renders deterministic synthetic images and audits a synthetic connector; Tier B should replace this with a small VLM whose internals can be captured.  
Dependencies: Labs 1, 4, 5, 8, 13, and 31.  
Minimum passing artifacts: `tables/multimodal_prompt_manifest.csv`, `tables/modality_probe_report.csv`, `tables/multimodal_patch_report.csv`, `tables/cross_modal_transfer.csv`, `tables/multimodal_evidence_matrix.csv`, and `plots/multimodal_evidence_dashboard.png`.  
Main plot: `plots/multimodal_evidence_dashboard.png`.  
Main table: `tables/multimodal_evidence_matrix.csv`.  
Evidence rung: `OBS + DECODE + CAUSAL` in synthetic smoke mode only.  
Forbidden claim: "The VLM has a human-like visual concept of color."  
One-sentence allowed claim: "The synthetic connector smoke path produced a complete multimodal audit package and identified which gates must pass before real VLM claims."  
Human-label requirement: not required for the synthetic labels, but any real VLM extension must hand-check ambiguous images and OCR/background controls.

## Why This Lab Exists

Multimodal interpretability is where many text-only habits break. Image tokens
may have variable alignment, rendered text can leak the answer, and a patch that
looks visual may actually patch a caption shortcut.

Lab 33 therefore starts with a strict smoke mode:

```text
synthetic image spec -> rendered image -> image/text/connector states -> readout -> patch -> leak gate
```

This validates the artifact contract. It does not validate a real VLM
mechanism.

## Data

Default data lives in `data/multimodal_concept_pairs.jsonl`.

Each row contains:

```json
{
  "item_id": "...",
  "image_path": "generated/...",
  "question": "...",
  "target": "...",
  "distractor": "...",
  "concept_family": "color|shape|count|spatial|chart|ocr_control|background_control",
  "text_control_prompt": "...",
  "image_spec": {},
  "control_spec": {}
}
```

The lab renders PNGs into the run directory from `image_spec` and
`control_spec`, so the repo does not need tracked binary images.

## Synthetic Mode

The default implementation builds four state types:

- `vision`: synthetic image feature state, including explicit OCR/background
  channels;
- `connector`: image/text mixture used for patch semantics;
- `language`: connector plus text-control state;
- `text`: answer-bearing text-control state.

The run writes `science_ready=false`. That flag is intentional.

## Required Gates Before Real VLM Claims

### OCR And Background Leak Gate

Rows such as `ocr_trap_red_blueword_001` render a red object with the printed
word `BLUE`. A real VLM result must show that answer recovery is visual, not a
text-in-image shortcut.

### Alignment Validation

`diagnostics/alignment_validation.json` records that the synthetic connector has
fixed token counts. A real VLM extension must validate image-token and region
alignment before patching.

### Patch Controls

The patch report compares:

- clean reference;
- corrupt no-patch;
- connector clean-to-corrupt patch;
- visual-region patch;
- wrong-region or OCR patch;
- random patch control.

## How To Run

```bash
cd interpretability
python interp_bench.py --lab lab33 --tier a
python interp_bench.py --lab lab33 --tier b --prompt-set full
```

For a fast table-only smoke:

```bash
python interp_bench.py --lab lab33 --tier a --no-plots
```

## Reading Order

1. `method_card.md`

   Confirms that the run is synthetic smoke mode.

2. `tables/multimodal_prompt_manifest.csv`

   Lists rendered clean/corrupt images, questions, targets, regions, OCR text,
   and background controls.

3. `tables/modality_probe_report.csv`

   Shows synthetic readout AUC by family and state type.

4. `tables/multimodal_patch_report.csv`

   Checks whether clean-to-corrupt patch semantics behave as intended.

5. `tables/ocr_background_leak_audit.csv`

   The first table to inspect before making any visual claim.

6. `diagnostics/alignment_validation.json`

   Explains the image-token alignment status.

## Common Failure Modes

### OCR Shortcut

The model reads text printed in the image instead of using visual features.

### Background Shortcut

The background color or texture carries the label.

### Region Misalignment

The patch swaps the wrong image tokens or compares sequences with incompatible
image-token layouts.

### Text-Control Leakage

The text-only prompt already contains the answer, so cross-modal transfer looks
better than it should.

## Claim Grammar

Allowed in the default implementation:

```text
OBS + AUDIT: The synthetic connector smoke path rendered all images, produced
readout/patch/leak artifacts, and marked real-VLM claims as not science-ready.
```

Allowed only after a real VLM extension:

```text
DECODE + CAUSAL: For synthetic color-shape items, connector-layer state S
carries image-derived color information, and patching S recovers the clean
answer more than OCR/background/random controls.
```

Forbidden:

```text
The VLM has a human-like visual concept of color.
```

Also forbidden:

```text
The connector is aligned because the plot looks right.
The model saw the object rather than the printed word.
The synthetic connector result transfers to real images.
```

## Deliverable

Write a short multimodal audit:

- Which concept families pass the synthetic readout?
- Which patch type recovers the clean state most specifically?
- Which OCR/background row blocks real VLM claims?
- What alignment diagnostic would a real VLM extension need to add?
