# Lab 38 instrument validation — Phase 1 part 1

Date: 2026-08-07. VM: Colab, RTX PRO 6000 Blackwell 96 GB. Branch
`interp_preference_phase1`. Registry: `preference/phase1/reports/evidence_events.jsonl`.

## Commands executed

```bash
python interpretability/preference/data/make_lab38_preference_bank.py --with-model
python interpretability/preference/data/make_lab38_preference_bank.py --frozen-codebook   # byte-determinism proof
python -m pytest interpretability/preference/phase1/tests -q                              # 46 passed
python interp_bench.py --lab lab38 --tier a --mode smoke --no-plots                       # adapter path
python -m preference_phase1.cli smoke --model-tier a --batch-size 4                       # runner path
# + interrupted-vs-uninterrupted resume parity script (see below)
```

## Environment

Python 3.12.13 · torch 2.11.0+cu128 · transformers 5.13.1 · tokenizers
0.22.2 · numpy 2.0.2 · pandas 2.2.2 · scipy 1.16.3. Models pinned in
`preference_phase1/models.py`: SmolLM2-135M-Instruct `12fd25f7`,
Olmo-3-7B-Instruct `6e5971d9`, Olmo-3.1-32B-Instruct `ac0587e4`.

## Results

| check | result |
|---|---|
| Unit + synthetic tests | **46 passed** (plan §7.1+K instrument suite; §7.2 ten planted-effect analysis cases; NC-path identity; registry mechanics) |
| Bank counts | 960 AR + 480 PC + 160 NC + 720 RO = **2,320**; dev subset 348 |
| Bank audit | PASS — balance, code independence, disjoint AR/RO alphabets, pairing, splits, binding presence |
| Bank determinism | regeneration from frozen codebook **byte-identical** (jsonl sha `634aded4…`) |
| Codebook | `cb_final_41eec2d774` on Olmo-3-7B tokenizer: AR KP4/PK7 gap **0.0031 nats**, RO VM2/GS2 gap **0.0275 nats** (< 0.7 threshold); no prefix relations; 4 distinct first tokens; leading-space policy `none` |
| Parser adversarial matrix | 12/12 expected outcomes (exact code only; lowercase, punctuation, two-codes, label, option-name, lookalike, empty all rejected; no guessing) |
| Chat-template parity | string-render ids == direct template ids on OLMo + SmolLM2 samples; generation prompt preserves prefix; boundary = final prompt token |
| Target scoring | full-sequence summed logprobs finite on all smoke rows; batched == single within 1e-3 nats (checked again per behavioral run) |
| Binding | invalid parse never executes; hypothetical frame never executes (E10); RO never executes; wrong-branch count 0; forced microtask plumbing probe OK (continuation renders, generation runs, validator executes) |
| Resume | interrupted (6 rows) + same-command resume **byte-matches** uninterrupted run on all 15 smoke rows (generations, margins, parse) |
| Tier-A caveat | SmolLM2 strict-parse valid rate 0.0 — a 135M capability fact, expected; smoke gates on instrument checks only and is never preference evidence |

## Known limitations

- Deterministic-replay certification is GPU-class-scoped (this Blackwell
  class); the frozen run re-certifies in its own session (addendum C3).
- Equality review is `agent_dual_code_provisional` with compressed pass
  separation (DEVIATIONS.md#D3); PI ratings required at freeze.
- DG track and mechanism module intentionally deferred (DEVIATIONS D5/D6).

## Permission table

| stage | licensed now? | authority |
|---|---|---|
| bank_audit / smoke (tier a) | YES | P1-G1 + this document |
| behavioral_dev (7B, dev subset) | **YES — licensed by this document (P1-G3)** | gates G1+G2 green above |
| behavioral_frozen | NO — requires PI freeze approval + freeze record | addendum §I |
| mechanism | NO — requires frozen-run graduation manifest | plan §10 |
| lineage / dg_smoke | NO — post-primary, drop-order gated | plan §11/§12 |
