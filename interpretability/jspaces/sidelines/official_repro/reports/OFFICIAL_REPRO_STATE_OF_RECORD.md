# OFFICIAL_REPRO_STATE_OF_RECORD — Study 1 (or1-)

**Status at 2026-08-09 ~12:10Z:** OR1.0–OR1.4 and OR1.6 complete and
registered; OR1.5 complete except three OLMo battery cells rerunning
under the or1-002 fix (selectivity-language, ignition, top-down);
OR1.7 synthesis in progress. Tier: development/methods throughout.
Vocabulary per plan §3.3: this is a **released-materials
reproduction** — prompt-exact where released, method-reconstructed
where specified (R1), representation-adapted where required (R2),
not-identified where absent (R3). No cell is called "the exact official
experiment."

## Instruments (OR-Q1)

- Qwen 3.6 27B @6a9e13bd + published n=1000 lens (byte-verified):
  readout parity exact; α=0 no-op exact; 48 GDN blocks; boot sentinel
  reads Italy@L39→euro@L58. **The instrument functions.**
- OLMo 3.1 32B Instruct @ac0587e4 + prospectively fitted n=250 merged
  lens (2×125 route frozen from timing at 161.4 s/prompt, dim_batch 16):
  parity exact; halves agree (eval pass@20 gaps ≤0.015; median
  sym-rel-Frobenius 8.8%, 2–4% in-band) — **no FIT-SENSITIVE flag**;
  merged lens concordant with the frozen campaign lens on the 9-layer
  intersection.
- g-folding audit: **material on Qwen (0.382), immaterial on OLMo
  (0.9886)** — D9; unfolded paper-literal v_t primary on both lanes.

## Reproductions (paper direction present at development tier)

1. **J-lens readout advantage (Qwen only):** J-lens recovers latent
   intermediates in-band where the logit lens is at rank ~10⁴
   (multihop .641 vs .542 p@20). On OLMo the advantage largely
   disappears (logit .590 vs J .563) — in-band Jacobians near identity.
   The paper's methods-comparison claim is **model-dependent**.
2. **Selectivity-language:** explicit−automatic contrast = 1.00 on Qwen
   (perfect reproduction of the released readout contrast).
3. **Directed modulation:** focus .72 > suppress .26 > 0 with the
   white-bear residue (Qwen).
4. **Dual-task interference:** present, asymmetric (math 31% ≫ concept
   7%, Qwen).
5. **Ignition:** transition width narrows with depth; concept pairs
   (.035) ≫ idiom controls (.209) in sharpness (Qwen).

## Partial / ambiguous

6. **Verbal report:** concordant ~25–30% top-5 on BOTH lanes (Qwen v2
   30.2% after the or1-001 correction; OLMo 25.5%) vs paper 88%.
7. **Probe-swap raw arm:** 14.3% (Qwen) / 9.3% (OLMo) capable top-1 vs
   paper 60%; successes emphatic (rank 480→1), failures under-strength.
8. **Flexible generalization:** ~5% capable α=1 both lanes,
   countries-driven; **α=2 collapses to 0% on both lanes** — the
   paper's Sonnet α=2 gain (76→101/192) inverts on open models.

## Nulls / opposite

9. **Verbal introspection:** no dose-response on either lane under the
   reconstructed ladder (median RR ~0.002 flat).
10. **Top-down summoning:** Q2−Q1 = 0.0 (Qwen, 7 items).

## Cross-over (OR-Q4) — the study's center of gravity

On the frozen 30-item official probe-swap subset: campaign protected
top-10 J-ablation breaks the top-1 answer on **14/30 (Qwen)** and
**5–6/19 (OLMo)** items vs **≤1** for exact rank/energy matched
controls — the campaign's C1-style selectivity **extends to the
official prompt population on both lanes**. Paper-literal swaps stay
weak everywhere (10–27%). New-merged and frozen-campaign OLMo lenses
agree under both families. **Licensed sentence (Route E + F blend):
the campaign-vs-paper discrepancy is attributable to intervention
semantics and prompt/task population — not lens fit, not harness.**
The two interventions estimate different causal questions.

## Incidents

- **or1-001** (verbal-report token form at chat boundary): v1
  understated; v2 registered and supersedes v1; OLMo ran v2 from the
  start.
- **or1-002** (BOS text shifted span offsets on OLMo raw renders):
  caught by the span audit hard stop; fixed with decoded-domain
  offsets + regression test; one unregistered file quarantined; three
  cells rerunning. No registered output was affected by either
  incident's defect.

## Gaps (recorded, never zero)

- Official linear-probe arm: R3 NOT-IDENTIFIED (probe corpus absent
  from release).
- Line-break modulation family: R2 corpus-empty under the pinned
  filter (8-row sentinel pool; 0 qualifying rows).
- Paper broad top-10 ablation: no released JSON; campaign cross-over
  explicitly non-official.
- Claude comparisons: descriptive context only.

## Boundaries unchanged

No frozen registry touched; no Phase 5 authorization; no
workspace-existence or -absence claim for either model; forbidden
wordings per plan §16.3 audited in the report.
