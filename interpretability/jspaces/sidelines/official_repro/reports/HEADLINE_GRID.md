# §16.1 headline evidence grid (auto-generated)

## lens eval quality
- fidelity: R1
- gates: none
- evidence: qwen=DIRECTION-REPRODUCED (readout advantage) · olmo=OPPOSITE (readout advantage absent)
- paper reference: J-lens recovers content at depths logit lens does not
- max licensed sentence: The J-lens readout advantage is model-dependent: present on Qwen, largely absent on OLMo whose in-band Jacobians are near identity.

## verbal report
- fidelity: R1 (D1 boundary mapping; v2 scoring per or1-001)
- gates: none
- evidence: qwen=DIRECTION-AMBIGUOUS (top5 30.2% vs paper 88%) · olmo=DIRECTION-AMBIGUOUS (top5 25.5%)
- paper reference: 88% top-5 (Claude)
- max licensed sentence: Concordant partial effect (~25-30% top-5) on both lanes; far below paper magnitude.

## flexible generalization
- fidelity: R1
- gates: capability-conditioned primary; per-category counts in figures
- evidence: qwen=DIRECTION-AMBIGUOUS (5.8%, countries-driven) · olmo=DIRECTION-AMBIGUOUS (5.2%)
- paper reference: 76/192 -> 101/192 at alpha 2 (Sonnet 4.5)
- max licensed sentence: Weak concordant alpha-1 effect concentrated in one category; the paper's alpha-2 gain REVERSES to 0% on both open lanes.

## probe-swap raw-token arm
- fidelity: R2 (prompt_exact_representation_adapted_raw_jlens); official probe arm R3 NOT-IDENTIFIED
- gates: tokenization 7/90 (qwen), 6/90 (olmo)
- evidence: qwen=DIRECTION-AMBIGUOUS (14.3% vs paper 60%) · olmo=DIRECTION-AMBIGUOUS (9.3%)
- paper reference: 60% top-1 (Claude n=90)
- max licensed sentence: Bimodal: successes are emphatic (rank 480->1) but sparse; modal failure is under-strength.

## selectivity language
- fidelity: R0/R1
- gates: none
- evidence: qwen=DIRECTION-REPRODUCED (contrast 1.00) · olmo=DIRECTION-REPRODUCED (contrast 1.00)
- paper reference: explicit>automatic, controls largely unmoved
- max licensed sentence: The released readout contrast reproduces (perfectly on Qwen).

## selectivity linecount
- fidelity: R1 (D10 assembly)
- gates: none
- evidence: qwen=DIRECTION-AMBIGUOUS (9% direct/letter vs 0% none) · olmo=DIRECTION-AMBIGUOUS
- paper reference: task-condition contrast over 11 passages
- max licensed sentence: Number-canon presence is rare at rank 1; weak condition ordering in the paper direction.

## verbal introspection
- fidelity: R1 (reconstructed strength ladder, D11)
- gates: none
- evidence: qwen=OPPOSITE/NULL (no dose-response) · olmo=OPPOSITE/NULL (no dose-response)
- paper reference: majority-report; MRR rises with strength (n=100)
- max licensed sentence: Injected-thought reporting does not reproduce under the reconstructed ladder on either lane.

## directed modulation
- fidelity: R1 (D12 pairing); line-break family R2 corpus-empty
- gates: line-break: pinned corpus yielded 0 rows (recorded gap)
- evidence: qwen=DIRECTION-REPRODUCED (focus .72 > suppress .26 > 0; white-bear residue) · olmo=DIRECTION-REPRODUCED
- paper reference: focus > suppress, suppression not zero
- max licensed sentence: The voluntary-modulation contrast reproduces with the white-bear residue on Qwen.

## dual task
- fidelity: R1 (D13 combined templates)
- gates: none
- evidence: qwen=DIRECTION-REPRODUCED (math interference 31%, concept 7%) · olmo=DIRECTION-REPRODUCED
- paper reference: single - dual reachability > 0
- max licensed sentence: Dual-task interference present and asymmetric (math >> concept).

## capacity task
- fidelity: R1 (frozen RNG; model-dependent canon by released design)
- gates: task-capacity; never merged with campaign sparse occupancy
- evidence: qwen=DESCRIPTIVE (~3 words at rank<=5) · olmo=DESCRIPTIVE
- paper reference: band-active list words at k (descriptive)
- max licensed sentence: A small working set (~3 words rank<=5 / ~8 at rank<=20) persists at list end.

## ignition
- fidelity: R1 (D12 carriers)
- gates: descriptive nonlinearity result (plan wording)
- evidence: qwen=DIRECTION-REPRODUCED (width narrows with depth; concept pairs sharper than idioms) · olmo=PENDING
- paper reference: sharp switching from workspace onset
- max licensed sentence: Sharp, depth-dependent winner-take-most transitions for real concept pairs.

## top-down summoning
- fidelity: R1 (D16 render)
- gates: n=7 items
- evidence: qwen=OPPOSITE/NULL (Q2-Q1 = 0.0) · olmo=PENDING
- paper reference: Q2 > Q1 expected-label readout
- max licensed sentence: No question-driven summoning signal at rank<=5 on Qwen (7 items).

## instrument cross-over
- fidelity: non-official (campaign machinery on official prompts)
- gates: frozen 30-item subsets
- evidence: qwen=ROUTE E+F evidence · olmo=lens-concordant
- paper reference: n/a (OR-Q4 diagnostic)
- max licensed sentence: Broad protected J-ablation is selectively destructive on official prompts on BOTH lanes while matched controls are not; coordinate swaps stay weak; new-vs-frozen lenses agree. The campaign-vs-paper discrepancy is intervention-semantics plus population, not lens fit and not harness.
