# Muse Glimmer 30B — J-space sideline state of record

**Generated:** 2026-08-11T10:35Z (fit wall ~6.3 h; battery <2 min)  
**Tier:** development / methods only — **not confirmatory**  
**Branch:** `jspace_muse`  
**Model:** `meta-models/Muse-Glimmer-30B` @ `97c77dff50b2797bcc558fa2d909761dbc575c59`  
**Package:** `interpretability/jspaces/sidelines/muse/`  
**Drive root:** `/content/drive/MyDrive/interpret/special-lab-1/jspace-muse/`

## Question

Does the Phase-2 Jacobian-lens instrument work on Muse Glimmer (geometry,
fit, readout), and is there any indication of a workspace-like structure
worth a larger preregistered chase?

## Executive answer

| question | answer |
|---|---|
| Geometry OK (unlike Gemma)? | **YES** — identity + post-fit readout parity exact |
| 120 WikiText J-lens fittable? | **YES** — dim_batch=16, ~189 s/prompt, n=120 merged |
| J-lens depth advantage? | **YES, strong** — band mean rank advantage ~4330 vs logit lens |
| Selectivity-language? | **YES, direction** — contrast 0.50 (explicit 0.75 − auto 0.25) |
| Paper-literal VR swaps? | **NO** — top5 hit rate 0.0 on n=10 |
| Strong causal workspace package? | **WEAK** — ignition never hits rank≤5; ablation only mildly selective (1/6); modulation improves rank but stays ~50k (near-noise) |

**Bottom line (methods tier):** Muse is a **green instrument** for jlens —
geometry works, the lens fits, and the J-vs-logit depth profile looks more
like **Qwen** than **OLMo** (near-identity) or **Gemma** (geometry fail).
The **readout-side** workspace-adjacent signals (depth recovery +
selectivity) justify a larger sideline if you care about open-model J-lens
geometry. The **intervention-side** workspace package (VR, ignition, strong
protected ablation) does **not** look promising on this compact battery —
same broad pattern as official_repro Study 1 on open models (swaps weak).

---

## Geometry admission (pre-fit)

| gate | result |
|---|---|
| shape 52 × 6656 | PASS |
| CUDA load (~59.6 GB BF16) | PASS |
| finite residual hooks | PASS |
| identity-J readout parity | PASS (max abs = 0) |
| smoke top-1 = Paris | PASS |
| local-linearity odd-symmetry | SOFT FAIL (ε=1e-2 and 1e-3) — canary only |

### Muse-specific instrument patch

Muse applies `output_multiplier ≈ 0.196` **before** logit softcap (20.0).
Upstream jlens only softcaps. Without the multiplier, pre-softcap logits
saturate and top-k collapses. Fix: `jspace_muse.adapters._patch_muse_unembed`
(also folds multiplier into intervention unembedding rows).

## Lens fit

| item | value |
|---|---|
| n_prompts | 120 (Phase-2 draw-A WikiText) |
| dim_batch | 16 (24/32 OOM; ~189 s/prompt) |
| source layers | 21: 2…50 (skip final) |
| target | 51 |
| peak VRAM | 88.1 GB |
| wall | ~6.3 h (4×30 slices) |
| merged sha256 | `d73c01111b74f5f56e9ae924e9bedb937b2611592879379c4f869f8e4205fdae` |
| path | `…/jspace-muse/lens/muse_glimmer_lens.pt` (1.8 GB) |

Evidence: `muse-fit-wikitext120-v1`.

## Post-fit admission

| check | result |
|---|---|
| readout parity (band layers) | PASS exact |
| g-fold min cosine | **−0.341** (MATERIAL — worse than Qwen 0.38; OLMo was ~0.99) |
| g-fold immaterial? | NO — use unfolded `v_t` as primary (D9-style) |

## Battery headline (compact, development)

| cell | value | reading |
|---|---|---|
| J-lens band advantage (mean ranks) | **+4330** | strong depth recovery vs logit |
| mean J rank band / logit band | 772 / 5102 | |
| mean J rank late / logit late | 5.6 / 2.8 | logit wins at final (expected) |
| Selectivity-language contrast | **0.50** | direction of paper |
| Modulation focus improves | True | but ranks ~51k → ~50k (tiny, near-noise) |
| Dual-task math interference | large | descriptive; token ranks noisy |
| Capacity mean active@≤5 | 1.4 | small working set |
| Ignition median α@≤5 | **None** | no pair reached rank 5 |
| VR top5 hit rate | **0.0** | no paper-literal transfer (n=10) |
| Ablation J / random damage | 116.5 / 98.5 | mild edge; selective only **1/6** |

## Reading (careful)

1. **Instrument:** Valid. Muse is not a Gemma-class geometry dud. The
   multiplier patch is load-bearing; document it on any future Muse cell.
2. **Readout geometry:** J-lens recovers mid-depth content that logit lens
   loses — the cleanest positive here, and the reason Muse is interesting
   for jspace methods work.
3. **g-folding is highly material** (min cos −0.34). Intervention designs
   that ignore final-norm gain will mis-aim; prefer unfolded vectors and
   record the audit.
4. **Workspace mechanism (paper-literal):** Not supported by this battery.
   VR null, ignition null, ablation barely selective. Do not claim a
   Claude-like workspace from these numbers.
5. **What to chase next (if anything):**
   - Scale depth-profile + selectivity to confirmatory n with proper stats
   - Protected J-ablation with **matched energy** controls on a real task
     bank (phase2-style), not the 6-item smoke
   - Do **not** prioritise paper-literal α=1 country swaps on Muse given
     the 0% VR pilot
   - Optionally: local-linearity / H7-style ceiling (soft canary already red)

## Artifacts

| what | where |
|---|---|
| lens | Drive `jspace-muse/lens/muse_glimmer_lens.pt` |
| metrics | Drive `jspace-muse/metrics/*.json` |
| figures PNG | Drive `jspace-muse/figures/` + repo `reports/` |
| handout PDF/TeX | Drive `jspace-muse/reports/muse_jspace_handout.{pdf,tex}` |
| live state | `/content/drive/MyDrive/interpret/inprogress-muse.md` |
| evidence events | package `reports/evidence_events.jsonl` |

## Non-claims

No confirmatory language. Frozen Phase 1–4 / gemma / olmo / or1 registries
were not written. Multimodal/vision path not exercised (text-only).

---

*Sideline complete for the planned pilot scope.*
