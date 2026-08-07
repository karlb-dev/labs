# Claude Code Prompt — special-lab-1 v2: Resolve the Open Instruments

Paste below the line into Claude Code from `interpretability/`. This is a **delta run**, not a rebuild.

---

## Mission

Continue special-lab-1 (J-space replication on `allenai/Olmo-3-32B-Think`). The v1 run (`/content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726/`) is complete and its REPORT.md §7 lists the open items. Your job is to resolve the ones that currently block the verdict, in priority order, reusing the existing lens, scripts, caches, and metrics wherever possible. Create a new run dir `<v1_run>/../2026-07-26_v2/` but read v1's lens and metrics freely; do not refit anything that can be reused. Read v1's `PLAN.md`, `LOG.md`, and `REPORT.md` **in full** before writing any code — especially §4.2, §6 (limitations 2, 4, 5), and §7. Keep the same conventions: incremental Drive checkpointing, resumability, seeds recorded, bootstrap CIs, no repo commits, no README/index updates.

## Priority 1 — Variance-matched non-J control (v1 §7.1, limitation 4)

The v1 causal story hinges on this. Build a control subspace of top non-J PCs **scaled/selected so the removed activation energy per layer matches the J-span's removed energy exactly** (report matched-energy verification per layer in the metrics). Rerun the full static ablation battery (s7, doses k∈{10,20,40}, band L20–44, same task battery, same n, 2000-resample bootstrap) with four conditions: none, static J-span, variance-matched random, variance-matched non-J PCs. Decision this settles: is the non-J damage about energy or content? If variance-matched non-J still selectively destroys SQL/twohop_lp while J-span ≈ random, the "dissociation does not replicate" verdict is confirmed and clean. If the matched control's damage vanishes, the v1 causal null is an energy artifact and must be re-stated.

## Priority 2 — Frozen prompt-selected top-k ablation (v1 §7.2, limitation 5)

The confound-free version of the paper's intervention: for each item, run one clean forward pass, record the top-10 J-directions active per band layer over the prompt (dynamic **selection**), freeze that set, then generate with those frozen directions projected out (static **application**). Compare against v1's static-span and dyn10 results on the same battery. This is the instrument that can actually detect position-specific workspace structure without deleting live computation. If this produces the paper's dissociation (multi-step collapses, fluency survives), the headline verdict flips; run it before writing anything.

## Priority 3 — Late-band fit + redo readouts (v1 §7.4, limitation 2)

The v1 suppressed-CoT profile shows the answer's rank still collapsing L44→L60 — into the band that was never fitted. Fit the lens on layers {46, 50, 54, 58, 62} (reuse v1 fit settings: 120 WikiText prompts, chunked fit + merge, ~same GPU budget; extend only if convergence hasn't saturated). Then redo: s5 descriptive readouts (does variance share keep rising or peak?), s8 answer-time loading and cot-lead with the late layers included, and add the late band to the Priority 1/2 ablation grid as a second band condition. This decides whether OLMo's workspace is genuinely late-shifted relative to Claude's 33–92% band or whether v1 just measured the front half of it.

## Priority 4 — Null calibration for the cot-lead result

The v1 headline "workspace leads CoT text by median 46 steps (91% of items)" has no false-positive floor: top-8 over multiple read layers across hundreds of think steps gives any string many chances to appear. For each traced item, run the identical detector for matched foil tokens: (a) the wrong answers from the same item family (other bridge entities / other join columns / nearby numbers), (b) 5 frequency-matched random content words. Report the foil lead-detection rate and foil median "lead" with the same statistics. The claim survives only if answer-token detection separates cleanly from the foil floor; report the ROC-style separation, not just the headline.

## Priority 5 — CoT-rescue test (v1 §7.3)

Paper claim: externalized reasoning rescues workspace ablation. Rerun the Priority 1 and 2 winning-instrument ablations under think-mode prompting (open `<think>`, ≤400 tokens) on the twohop/arithmetic/SQL battery, vs the v1 no-think ablations. v1's suppression asymmetry predicts rescue here; either result is reportable.

## Priority 6 — Robustness

- Second seed (seed=1) on the Priority 1 and 2 grids; report both seeds side by side.
- Raise n on the weakest cells: SQL and arithmetic from 30 → 60 items, reusing v1 generators.
- Threshold sensitivity for active-concepts is already partly done (θ sweep); add the paper's exact threshold convention if the methods section specifies one (re-fetch and check; pin what you find in PLAN.md).

## Optional if GPU time remains (in order)

7. Dolma-corpus lens (v1 §7.5): fit a second lens on a Dolma slice, same recipe, and diff the two dictionaries (top-token overlap per direction, readout agreement on the s5 prompt set). This is the OLMo-distinctive contribution.
8. Confound-broken eval-awareness (v1 §7.8): implied-but-unstated test cues, zero eval-lexicon overlap, checked with a string audit; only if a direction still emerges, ablate on a task where eval framing measurably shifts behavior.
9. `famously`-attractor forensics against Neuronpedia's published Qwen lenses (v1 §7.6).

## Deliverables

- `report/REPORT_v2.md`: same structure as v1, leading with a **revised verdict slug** and a table showing which v1 claims survived, flipped, or sharpened. Include the v1-vs-v2 causal grids side by side.
- `report/summary_v2.json` with the headline numbers.
- **Claim ledger templates** in the repo's lab style (see `labs/lab36_severance_report_channel.md` "Claim templates" for the exact format): draft `[SL1-C1..C5]` entries covering (1) descriptive geometry with the 10×-thinner capacity number, (2) the causal verdict under the v2 instruments, (3) broadcast specificity under variance matching, (4) cot-lead with its foil floor, (5) answer-time loading under suppression. Each with Artifact path and Falsifier line. Also draft the lab-header block (evidence rung, forbidden claim, allowed one-sentence claim) so promotion to an official lab is a copy-paste decision, not a writing task.
- Updated LOG.md throughout; every figure regenerable from metrics.

## Guardrails

Same as v1: no commits, no lab registration, everything checkpointed to Drive incrementally, hard honesty about anything downgraded. If Priority 1–4 can't all fit in the available GPU window, do them in order and say exactly where you stopped — do not thin all of them to fit.
