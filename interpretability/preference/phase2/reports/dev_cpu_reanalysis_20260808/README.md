# Phase 2 pre-VM CPU re-analysis — 2026-08-08 (development tier)

Workstation-only re-analysis of the frozen Phase 1 record, run before any
Phase 2 model compute. Inputs are read-only frozen artifacts
(`phase1/reports/frozen_{7b,32b}/results.jsonl`, the frozen bank and
codebook, the frozen graduation tables — tables used ONLY to validate the
pipeline, never as number sources). Governing successor plan:
`preference/plans/preference_2_2.md` (§0.2 carries the numbers and their
interpretations; this directory is the artifact of record for them).

| file | what |
|---|---|
| `reanalyze_phase1.py` | F1–F5 pipeline (validation, censoring identity, folded margins, aliasing, mechanism retrodiction, RO/concordance) |
| `reanalysis.json` | full output of the above, both models |
| `lexical_probe.py` | G-LEX early probe: unconditional option-string logprobs under a reference LM vs the folded margins |
| `lexical_probe_gpt2.json` | cross-family reference (gpt2, cached weights) |
| `lexical_probe_olmo7b.json` | self-reference (`allenai/Olmo-3-7B-Instruct`, fp16 on Apple MPS — numerics differ from the frozen CUDA bf16 runs; margins themselves come from the frozen record and are exact) |
| `port_audit_tokenizers.py` | P-1 codebook-survival audit vs Qwen/Gemma tokenizers |
| `port_audit_tokenizers.json` | result: frozen codebook survives on `Qwen/Qwen3.6-27B` and `google/gemma-4-31B-it` (and reproduces on the OLMo control) |

Headline results (all definitions in the scripts; tier: development):

- **Pipeline validation:** reproduces the frozen `graduation_decisions.csv`
  `effect` and `position_effect` columns exactly, 20/20 scenarios on both
  models, before any new number is trusted.
- **F1 censoring identity:** `position + |content| = 0.500` exact in 14/14
  AR+NC scenarios at 32B; 10/14 exact at 7B, max deviation 0.048 (invalid
  `PK4` cells only).
- **F2 folded margins:** AR content terms −0.2 … −2.2 nats; sig at 90%
  (incidental-clustered, df=4) in 23/24 AR-enacted and 24/24 RO
  model×scenario cells; toward pole_0 in 23/24 (both channels). NC folded
  margins are exactly 0 (surface cancellation is exact on identical
  options — the fold's built-in falsifier).
- **F3 aliasing:** label-rank and reply-order aliased to display position
  in 2320/2320 bank rows (rate 1.0 both).
- **F4 retrodiction:** raw-margin `MIN_MARGIN_STD` gate binds on 0/38
  scenario-channels per model (the gate never gated). Residualized margin
  sd under the frozen 7-column design: the one empirically identifiable
  direction (32B `pc_quality_config`) sits at 2.76 nats; every 32B AR
  scenario ≤ 0.92 (graduated `ar_docsection_readme` ≈ 0.21–0.26); PC
  family spans 0.52–2.76, so residual sd separates identifiable-from-not
  empirically but not as a clean family split — precheck thresholds must
  be pinned from these anchors.
- **F5:** 7B RO emitted one constant code in 9/12 AR scenarios (rate
  0.500 arithmetically forced); 32B 1/12. Matched-pair agreement
  (pair_key, both valid): 7B pooled 0.667 vs AR-only 0.535; 32B 0.774 vs
  0.660; mechanical floor 0.500.
- **Cross-structure:** AR folded terms 7B vs 32B sign-match 11/12,
  magnitude r 0.42; AR vs RO within model r 0.96 (7B) / 0.82 (32B).
- **G-LEX early result:** unconditional string-prior explanations are
  REFUTED at development tier. Self-reference (7B on its own bank):
  R² 0.08–0.11 vs its own folded margins, ~0.00 vs the 32B's, slope ≈ 0,
  pole_0 lean intact at Δ_lex = 0 (intercepts −0.86 / −1.21 nats).
  Cross-family (gpt2): R² 0.23–0.39 with NEGATIVE sign (prefers the
  pole_1 phrasings on net). NC Δ_lex exactly 0 under both references.
- **P-1 port audit:** the frozen code pairs (`KP4/PK7`, `VM2/GS2`)
  survive equal-token-count + distinct-first-token on both target
  tokenizers, bare and space-led — no codebook regeneration needed for
  the Qwen or Gemma cells.

Rerun: `python3.13 -m venv .venv && pip install numpy torch transformers`
then run the three scripts from this directory's originals under
`interpretability/preference_runs/phase2_cpu_20260808/` (gitignored) or
directly from here with paths intact. Scoring probes need the reference
weights cached locally.
