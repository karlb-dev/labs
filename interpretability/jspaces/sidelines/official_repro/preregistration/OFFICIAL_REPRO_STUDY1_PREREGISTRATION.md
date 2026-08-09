# Study-1 preregistration — official released-materials reproduction (or1-)

**Signed at scaffold, before any model-backed scientific result.**
Evidence tier: prospective development/methods replication — no
confirmatory or replication-holdout language anywhere in Study 1.
Governing plan `jspace_lab_official_repro_1.md` + addendum (§2 binding);
the pinned upstream bytes and paper text win over both. The five protocol
contracts and the coverage matrix are constitutive parts of this
preregistration; their sha256s are recorded in
`configs/preregistration_manifest.json` at signature.

## 1. Study question and outputs

OR-Q1 does the public Qwen lens function on the public evaluations under
the campaign-pinned Qwen checkpoint · OR-Q2 which fully reconstructable
functional properties reproduce on Qwen under paper-literal coordinate
swapping/steering · OR-Q3 with one prospectively fitted OLMo lens, which
properties reproduce/fail/gate · OR-Q4 why did campaign and
released-materials assays differ (bounded cross-over). Not Phase 5; no
frozen registry reopened; no Claude test; no global-workspace label from
any single passing property.

## 2. Source and model pins

- `anthropics/jacobian-lens` @ `581d398613e5602a5af361e1c34d3a92ea82ba8e`
  (external_record_manifest.json).
- Qwen lane: `Qwen/Qwen3.6-27B` @ `6a9e13bd…13e9`, 64 layers, d=5120;
  published lens `neuronpedia/jacobian-lens` @ `a4114d7…d3a1`, file
  `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`,
  sha256 `1718c8c5…11e1`, 3,303,032,772 bytes; post-load checks d=5120,
  n_prompts=1000, source_layers=[0..62], finite, [5120,5120] each. **No
  Qwen fit.**
- OLMo lane: `allenai/Olmo-3.1-32B-Instruct` @ `ac0587e4…40dae`, 64
  layers. Frozen campaign comparator lens sha256 `e0f8b972…2dfb`
  (read-only). Study-1 primary = merged half-A/half-B official-estimator
  fit per OLMO_FIT_CONTRACT.
- Fit population: WikiText-103-raw-v1 train, upstream criterion verbatim,
  first 1000 raw texts; halves even/odd; manifest hash recorded before
  fit prompt 0.

## 3. Frozen analysis objects

- Rendering & positions: RENDER_AND_POSITION_CONTRACT (Qwen
  direct/non-thinking; assistant prefills stay open; five sentinels per
  schema per lane).
- Grids & bands: LAYER_BAND_CONTRACT (paper grid primary; campaign band
  named sensitivity; per-layer curves always kept).
- Interventions: COORDINATE_INTERVENTION_CONTRACT (pinv 2-coordinate
  swap; α=1 primary; α=2 full FG sensitivity; steering ladder
  {0,1,2,4,8,16} reconstructed; g-folding audit; near-singular ⇒
  GEOMETRY_GATED).
- Target tokens: single-token exact after frozen whitespace
  normalization; released synonym sets only; model-dependent canons for
  capacity/linecount derived per README and stored per lane; multi-token
  ⇒ TOKENIZATION_GATED (adapted aliases only as named R2 sensitivity).
- Capability gates: computed before the corresponding intervention
  block; three-population reporting per STATISTICAL_ANALYSIS_CONTRACT §4;
  FG primary conditions on source+target baseline correctness;
  probe-swap primary on baseline == released answer; verbal-report
  needs no baseline condition; the all-executable diagnostic is never
  called intent-to-treat.
- Statistics: STATISTICAL_ANALYSIS_CONTRACT (five-state evidence code;
  units; Wilson + cluster bootstrap B=10,000 seed 20260808; no
  retrospective power; no Holm family).

## 4. Experiment order (fixed)

OR1.0 foundation → OR1.1 conformance → OR1.2 Qwen lane (admission → six
evals → verbal report → flexible generalization → probe-swap raw arm) →
OR1.3 OLMo fit (timing gate → halves → split-half audit → merge) → OR1.4
OLMo admission + core → OR1.5 battery groups A (selectivity), B
(report/modulation), C (structure/access), D (visual sanity) → OR1.6
cross-over → OR1.7 synthesis/release. Complete-group banking; in-session
drop order per plan §14; never thin a partially executed primary block.

## 5. Cross-over subset (frozen before core outcomes; hashes in
`configs/crossover_subset_manifest.json` at freeze)

30 token-valid baseline-capable probe-swap items stratified by released
category (selection rule: lowest item index per category round-robin
until 30) · all eligible FG countries+months cells · 20
`lens-eval-multihop` items (first 20 token-valid). Qwen conditions:
baseline / paper swap / campaign output-protected dynamic top-10 J
ablation / exact rank+energy matched control. OLMo: new-merged vs
frozen-campaign lens × paper swap vs protected ablation on the campaign
band (+ paper-band primary for the new lens). One GPU-session ceiling;
bank complete condition blocks.

## 6. Fit route (chosen from timing only)

Break-evens per OLMO_FIT_CONTRACT §3 (18 GPU-h ceiling): 2×500 ⇒ merge
n=1000 iff ≤64.8 s/prompt; 2×250 ⇒ n=500 iff ≤129.6; 2×125 ⇒ n=250 iff
≤259.2; else instrument blocked. dim_batch frozen after two agreeing
sentinel repeats; no post-outcome extension.

## 7. Wording router and prohibited analyses

Terminal sentences only from plan §16.2 routes A–G; forbidden wording per
§16.3 (no "has/lacks a workspace", no "confirmed/refuted", no "exact
official replication" for R2/R3, no "null" for gated cells, no
task-capacity/sparse-occupancy conflation). Prohibited: outcome-selected
refits, adaptive bands, paper-number-tuned strengths, retrospective
power, arbitrary PASS fractions of Claude effects, reopening Q-L4 or any
frozen registry, editing paper-route claim tables.

## 8. Drop/resume order

Per plan §14: study-authored probe sensitivity (already declined) →
adapted line-break family → extra slice visualizations → after-core
α/dose sensitivities → ignition → unstarted cross-over conditions. Never
drop release assembly, raw-record validation, or state-of-record updates.

## 9. Deviations

Any post-signature deviation goes to `preregistration/DEVIATIONS.md`
with date, reason, and blast radius before the affected outcome is used.
