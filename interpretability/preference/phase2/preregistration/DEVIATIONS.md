# Phase 2 DEVIATIONS (append-only; numbered)

D1 — **B-ARB3 incidentals 16 → 24 (12/6/6).** The power simulation under
the E16 primary put strict-choice power at the 0.10 SESOI at 0.45 with
16 incidentals (plan §32 floor: 0.80) and 0.86 at 24. Plan §32 commands
the raise; addendum D permits raise-never-lower. RO twins follow to 24.

D2 — **RO-DISJOINT 1,728 → 2,688 rows.** The addendum-D table's anchor
line undercounts the Section-G coupling receivers (64 primary + 64
reserved per anchor); D1 adds the 24-incidental twins. Union pinned.

D3 — **Coupling primary = RO full-target margin contrast.** The
addendum-G fallback triggered: simulated power for the strict
comparative-report shift at the pinned receiver counts is 0.21 (< 0.80).
Strict-report and five-point endpoints demote to secondary descriptors;
`CHOICE_REPORT_COUPLED` requires the margin primary plus at least one
same-direction non-margin endpoint. Pinned pre-outcome.

D4 — **H-CANON demoted to exploratory tier at freeze (E17 terms).** The
E1 composite sheet was coded mechanically from option definitions before
any Phase 2 model output, but the coder (the agent) has read the frozen
Phase 1 record, which E1's blinding clause excludes. The sheet is frozen
and the permutation test will run as designed; its results report at
exploratory tier under `agent_dual_code_provisional` unless the PI
ratifies the sheet.

D5 — **H5 approval basis = the PI's standing session instruction**
(2026-08-08), quoted verbatim in `reviews/PHASE2_FREEZE_APPROVAL.md`:
end-to-end execution authorized without interactive check-ins,
adjustments documented, accuracy/quality never compromised. All
human-gate outputs carry `agent_dual_code_provisional` until PI ratings.
Addendum L validity halts remain binding.

D6 — **§37 selection score quantized to 0.01 before comparison.**
Without quantization, per-cell permutation-band seeds produce float
jitter that would never let the plan's declared upstream-then-shallow
tie-break engage. Frozen transform, favoring propagation evidence as
the plan intends.

D7 — **Site token conventions pinned.** `*_end` sites are end-anchored
(token containing the exclusive-end character), `*_start` sites
start-anchored; `menu_end` shares the second displayed record's final
token by construction in F-SYM/RO (the closing delimiter is one token
on some tokenizers). Port audits verify display-order monotonicity.

D8 — **Qwen disabled-thinking contract.** `enable_thinking=False`
renders a constant EMPTY `<think></think>` block in the generation
preamble; the port gate accepts exactly that (constant across rows,
anchor after the block) and hard-fails any non-empty or unclosed think
span or a block preceding the user turn.
