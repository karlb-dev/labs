# jspace_lab_nextsteps_3_1_addendum.md — Verification, Amendments, and the Execution Schedule

**Reviews:** `jspace_lab_nextsteps_3_1.md`, `inprogress_vm9_final.md`, `REPORT_PART2.md`, the VM9 handout, and branch `interp_jspace_part2` @ `74effdcc` (verified against the reviewed claims).
**Role:** independent verification of the Phase 3 review's decisive claims, the PI's resolutions and amendments, and the block-by-block GPU execution schedule. Where this file and `nextsteps_3_1` conflict, §4 (amendments) and §5 (PI resolutions) here govern; everywhere else `nextsteps_3_1` is the operative Phase 3 spec.

> **Paste-line for the coding/research agent**
> Read `jspace_lab_nextsteps_3_1.md` in full, then this file, then `inprogress_vm9_final.md`'s NEXT BLOCK list. Execute the Phase 3 queues in nextsteps §19 order with the amendments in §4 below, on the schedule in §6. Phase 2 artifacts are immutable. The very first task — before any new GPU arm exists — is §4.1 of this file: mine the Phase 2 parquets and selected-ID sidecars for the protected-span overlap statistics that `matched_control.py` and the v2 ablator already log. Stop at the Phase 3 preregistration candidate for PI sign-off exactly as nextsteps §16/P3-3 specifies.

---

## 1. Verification (what was independently checked at `74effdcc`)

Four decisive code claims in nextsteps_3_1, four confirmations:
1. **Partition seed inactive** — `partition.py:92` is `rng.shuffle(order) if False else None`; assignment is deterministic by family size and name. As the review says: no contamination (no outcomes entered the split), but the report must not imply seeded randomization. Clarification event required; `family_split_v2` required for Phase 3.
2. **Matched-control protected basis uses raw QR** — `matched_control.py:140–142`, reduced QR on `[h; protected_rows]` and on the protected rows, no rank test. Coherent clean-top-k rows make this a live numerical hazard. SVD replacement required. Note: the module *does* log `max_protected_cos` per basis (line 70) — see §4.1.
3. **Prose guard skipped the primary control** — `confirmatory_protected_grid.py:289–290`, explicit `continue` for `matched_control` on prose. The Instruct +0.72 NLL result has never been compared against the exact dose control. Workstream C is mandatory before any selectivity claim.
4. **Primary p-values are percentile-bootstrap sign tests** — `confirmatory_analysis.py:113`. The Phase 2 preregistered analysis stands as declared; the Phase 3 randomization-test upgrade in nextsteps §14.2 is adopted.

Standing from prior cycles: the paper's protection clause, occupancy/excess definitions, and broadcast assay were verified against the paper text directly; the family-map, R2-estimand, and generation-path defects were verified and are now repaired in the frozen Phase 2 record.

## 2. Phase 2 verdict — what is now bankable

Phase 2 is complete and, as far as this review can determine, *legitimate*: pre-outcome freeze with a dedicated commit and tag; the §7 stop rule fired twice before outcomes were viewed and produced Amendment 1 rather than silent patching; per-model G4 positive controls with own lenses (0.76/0.76/0.78 vs random 0.18–0.24); locked analysis run once from raw parquets after the last cell banked; the untouched replication partition run; and the independent-lens clause fully discharged (Think r=0.988 / tail Jaccard 0.878; Instruct 0.990 / 0.883). Both Holm-family primaries reject; HP3 replicates on held-out families with a stable threshold curve; HP1 is confirmatory-thin (9 items / 2 families on the two-hop leg) and inconclusive on replication — disclosed prominently, which is exactly right. Two headline reversals were absorbed with correct discipline: the corrected capacity estimand puts **Qwen at the lower edge of the reported Claude band** (6.05/5.02/5.99%), withdrawing the pilot's "all open models an order of magnitude below" for Qwen; and at confirmatory tier on hard items **both OLMo 3.1 checkpoints are one-hop-dominant**, collapsing the pilot's four-rung ladder into a cleaner two-cluster picture (OLMo = accessibility-organized content-channel damage; Qwen = forward, paper-shaped dissociation + near-Claude capacity). The safe sentence and the blocked sentence in nextsteps §0.2 are both exactly calibrated; adopt them verbatim.

## 3. Endorsements

The two scalpels — the thick within-fact paired bank and the span-safe protected-output audit — are the correct center of Phase 3 and the correct order. Also endorsed without amendment: the outcome hierarchy (§0.4), the prohibited-claims list (§4.5), Bank F/S/W separation (§5.2), the "capability for inclusion, baseline logprob as covariate" resolution of the difficulty-window tension (§5.5), the paired-contrast-first statistics (§14.1) with family sign-flip randomization, the heavy-tail reporting requirements, the lineage stop rules, the N8 three-level clean-room design with blind protocol, the drop order, and the never-drop list. The §12.1 Gemma correction is also accepted — and for the record, the "softcap distorts the fitted Jacobian" mechanism it corrects was this reviewer's earlier suggestion; nextsteps is right that the fit target is residual-to-residual and softcap enters only at readout, so the legitimate variants are hook/target choices, not a "pre-cap Jacobian." The correction is owned and the autopsy design in §12 is the right replacement.

## 4. Amendments and additions

**4.1 (NEW, runs first, CPU-only): mine the existing overlap logs before building new arms.** The v2 ablator and `matched_control.py` already log selected-ID hashes, per-basis `max_protected_cos`, and removed-energy profiles for every Phase 2 confirmatory item. Before implementing span-safe arms, compute from the *existing* parquets and sidecars: the distribution of J-span↔protected-span principal-angle proxies, answer-direction survival under the Phase 2 projectors, and the correlation between protected-span overlap and per-item damage (tail vs non-tail). This costs an evening of CPU and calibrates the entire Workstream B threat model: if overlap is uniformly small and uncorrelated with damage, §2.3's leakage account is pre-bounded and P3-2 shrinks; if overlap predicts tail membership, span-safe becomes the paper's crown-jewel control and deserves its full §6 apparatus. Either result is a figure. Register as `phase3-development` evidence.

**4.2 Bank F fact hygiene, two additions.** (a) Fact-level dedup must extend *across phases*: no Phase 3 fact bundle may share its underlying (bridge, answer) triple with any Phase 2 item, not merely no shared item — the released two-hop facts are few and famous, and reviewer-visible leakage between "motivating study" and "independent replication" would be corrosive. (b) Source triples from a pinned knowledge-graph snapshot (dump date recorded), with an adversarial-ambiguity pass: every composed prompt checked for alternative valid bridges (the multihop literature's classic failure). LLM-proposed, source-verified, exactly as nextsteps specifies.

**4.3 Bank S gets a named confirmatory role.** Synthetic in-context composition is the only bank that tests *working memory* rather than parametric recall — the distinction on which the word "workspace" turns. Amend the multiplicity plan: the Bank-S composed-minus-direct contrast (same P3-P1 statistic, synthetic families) becomes the first member of Family B, prespecified. Decision value: if Qwen's forward contrast appears on Bank S, "workspace" is earned; if only on Bank F, the honest noun is "knowledge-access channel." This is cheap (Bank S rides the same grids) and it is the single sentence reviewers will look for.

**4.4 P3-P1 continuity clause.** Keep the primary as specified (Qwen minus OLMo-pair mean on the thick bank), but preregister the Think-vs-Instruct within-fact contrast as a named estimation target with CI — it is the Phase 2 HP1 quantity, and the paper must show the thick-bank version of it beside the thin confirmatory one, whatever it says.

**4.5 Fold the Part 1 temporal thread into the Qwen mode factorial.** The foil-calibrated cot-lead and answer-time-loading results from Part 1 never received a confirmatory home. The Qwen thinking-on cells of P3-10 generate the needed traces for free; preregister a small cot-lead replication (frozen detector, foil floor, complete-sample) as a Family-C secondary inside that workstream rather than a separate study. Closes the loose end at near-zero marginal GPU cost.

**4.6 N8 assignment mechanics.** Run N8 as a genuinely separate, narrative-blind session: hand a fresh agent only `N8_REPRO_PROTOCOL.md` (commands, schemas, tolerances — no expected values, no report sections). Level 1 starts *now* on CPU in parallel with bank authoring; Level 3's one full cell is budgeted in Block C below. The reproducer's report is sealed before comparison, per nextsteps §13.3.

**4.7 Publication timing.** Hold any public artifact until P3-4/P3-5 bank — the thin two-hop leg is precisely what a hostile reader finds first — but create the `paper/` tree in Block A and write sections 1–3 (problem, assay repair, Phase 2 matrix) immediately: they are frozen, and writing them now surfaces any remaining wording that overclaims. The campaign report stays as the audit log per nextsteps §2.9.

**4.8 Small correction to §2.6's framing.** The paired family-clustered bootstrap was the *preregistered declared fallback* when MixedLM could not represent the pairing — Phase 2 followed its own registered plan, and the report's transparency note says so. The Phase 3 randomization upgrade is adopted because it is better inference, not because Phase 2 broke its contract. The sensitivity appendix should say this explicitly.

## 5. PI resolutions (build the candidate around these; freeze still requires sign-off)

R1 — Adopt the P3-P1/P2/P3 primary family as specified, with §4.4's continuity clause. R2 — Bank targets as specified (≥72 two-hop families, ≥36/side, ≥25/side intersection), with §4.2 hygiene. R3 — Span-safe (`meanJ_span_safe`) enters the primary family via P3-P2; lost rank is reported, never refilled in the primary arm. R4 — The Instruct specificity map (Workstream C) is mandatory before any "selective" wording; until it lands, the paper says "nonspecific J-channel vulnerability cannot be excluded for Instruct." R5 — Lineage, load, Qwen modes, fit-size symmetry, and Gemma block 1 are the strong-set additions in that order, under the drop rules. R6 — Gemma hard cap: two blocks, exit criteria as written, methods-note framing only. R7 — Capacity moderator language stays descriptive until ≥6 assay-valid points exist; the within-model load interaction (§11.5) is the sanctioned stronger test.

## 6. Execution schedule (24 h VM blocks, 96 GB class; disk-rotate one 32B resident as VM9 did)

**Parallel CPU track (starts immediately, no GPU):** N8 Level 1 in a blind session; `paper/` tree with sections 1–3; §4.1 overlap mining; Phase 2 closeout tag + registry clarifications (P3-0); Bank F/S authoring and validators.

**Block A — development (≈10–14 GPU h):**
1. §4.1 results reviewed → sets the P3-2 dev-gate thresholds.
2. P3-2 control refits: SVD protected bases, span-safe J, overlap-matched + persistent prototypes; dev mechanism grid on OLMo 3.1-Think (2–4 h) + a Qwen spot-check (1–2 h).
3. Workstream C first pass: exact matched prose control on Think + Instruct with the broadened guard battery (2–4 h) — this both fixes the §2.5 hole and feeds arm selection.
4. G5 scoring of the candidate bank on all three primaries in assay BOS units (2–5 h).
5. Power/MDE simulation from dev families; `SCIENTIFIC_PREREGISTRATION_PHASE3_CANDIDATE.md`. **STOP for PI sign-off.**

**Block B — freeze + primary grid (≈14–20 GPU h; may spill):**
`family_split_v2` freeze commit → N6-style cells in frozen order (Think → Instruct → Qwen), G4 first per model, full cell banked before next load → locked analysis once, from raw parquets. If the block ends mid-grid, the per-item checkpointing makes any prefix resumable; never start Qwen with <6 h remaining.

**Block C — replication + mechanism start + N8 (≈12–18 GPU h):**
Replication side with the reduced preregistered arm set → bridge-mediation factorial on the mechanism subset (OLMo Think + Qwen, 4–8 h) → N8 Level 2 sentinels for all three models + one Level 3 full cell (Qwen preferred).

**Block D (strong set, optional, ≈14–24 GPU h):** OLMo lineage minimal assay until the transition localizes → load study (OLMo pair + Qwen) → Qwen mode factorial with §4.5's cot-lead secondary → symmetric fit-size sensitivity (nested Qwen 120/250/500 + one larger OLMo point) → Gemma autopsy block 1 last.

Minimal paper-hardening = Blocks A–C ≈ 36–50 GPU h (consistent with nextsteps' 20–35 h estimate plus the mandatory prose/control work it folds elsewhere). Strong set ≈ +14–24 h. Drop order and never-drop list per nextsteps §16, unchanged.

## 7. Risks

(i) Bank authoring is the schedule's long pole — start it on the CPU track today, and treat family count, not item count, as the progress metric. (ii) The span-safe audit could remove the tail (Outcome C); if §4.1's mining already shows overlap→damage correlation, pre-draft the methods-correction framing so the result lands as a finding, not a funeral. (iii) Qwen's thick-bank capability may prune families asymmetrically; the split must balance the *intersection* floor, not just totals. (iv) Block B is the one block that must not be interrupted mid-model — schedule it when a full window is certain. (v) Keep the contradiction heuristic and the tier watermarks; they have caught every instrument fault so far.

## 8. Bottom line

Phase 2 turned "an impressive pile of careful results" into a frozen, replicated, preregistered core with two honest reversals absorbed along the way. Phase 3 as specified — two scalpels first, mechanism second, moderators third, Gemma last, everything behind a new freeze — is the right shape, and with the amendments above it starts one evening of CPU from now. The conundrum is genuinely close to cracked: what remains is not discovering the answer but making the answer undismissable.
