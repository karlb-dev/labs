# Mechanism contract (plan Part VI-VII; addendum F/G/E11-E13; D3/D6/D7)

Direction: unit mean of matched A-favor minus B-favor state differences
(train only; pairs within incidental/order/cmap/menu-paraphrase/|s|/
codebook). Decoder: fixed projection (E11). Precheck per §36; site/depth
selection per §37 with 0.01 score quantization (D6), upstream-then-
shallow tie-break; sites per E6/D7. Dose: +/-{0.5,1,2} train-projection
SD, E13 guard set (24 prompts; mean KL < 0.05, max < 0.20, generic shift
< 0.05), frozen on validation. Assays: matched donor patch (E12),
+/-dose addition, alpha=1 removal (0.5/1.5 sens), propagation (>= 0.5x
injected effect at menu_end), final-token positive control last.
Controls: position/code/semantic-identity/wrong-scenario factor
directions, 8 norm-matched randoms, self-patch, wrong-site, reserved
codebook; randoms and non-Holm controls on the first 16 holdout
receivers (declared), Holm primaries on all. Coupling: RO margin
contrast primary (D3); router per §48. Holdout opens exactly once per
scenario. No pooled cross-scenario direction, ever.
