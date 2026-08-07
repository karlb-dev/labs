# PLAN_v3 — Final Stretch (one 3-hour GPU window, then ship)

Paste-line for Claude Code:
> Read `/content/drive/MyDrive/interpret/inprogress.md`, then `jspace/code/PLAN_v2.md` on branch `interp_jspace` of karlb-dev/labs, then THIS file. You have ~3.0 GPU-hours on this VM and it will not come back. Execute this plan top to bottom with its time gates. Bank everything to the Drive run dir `2026-07-26_v2/` as you go; `refresh_handout.sh` + `push_lab.sh` at every phase boundary. When the GPU dies or the clock runs out, the lab ships with whatever is banked — the reporting phase is mandatory and runs on CPU.

## Standing rules

- v1 run dir is frozen. All new work → `2026-07-26_v2/`. Resume recipe from inprogress.md verbatim (jlens pinned clone, code from the v2 mirror, SSH push verified).
- **Clock discipline:** wall-clock gates below are hard. At each gate, bank partial results with per-cell n recorded and move on. Never let a phase overrun into the Qwen window.
- Start `nvidia-smi`-stamped timer at VM up = **T+0:00**.

## T+0:00–0:10 — Restore + triage (GPU idle ok)

1. Run the resume recipe; `s20`-style bf16 sanity.
2. **Immediately launch background downloads** (no GPU): Qwen 3.6 27B weights to the Drive HF cache, and Nanda's published Qwen lens from Neuronpedia (URL/artifact per PLAN_v2 §Q pins). These must be downloading while s16 finishes.
3. Count s16's banked cells per condition×task. Decision: if the key contrasts (frozen_j10 vs frozen_rand10 under think-mode, twohop + onehop + arithmetic) already have ≥20 items/cell banked → mark s16 DONE-AT-N and skip to Qwen early. Otherwise:

## T+0:10–1:10 — s16 CoT-rescue, capped completion (≤60 min hard)

- Trim the remaining grid to the decision-relevant cells only: **frozen_j10 and frozen_rand10 under open-`<think>`** on twohop/onehop/arithmetic. Drop SQL (known flaky 3-schema cell) and drop all static-span rescue cells — P1 showed static does nothing, so there is nothing to rescue; testing its rescue is dead compute.
- The question this answers, in one line for the report: *can externalized reasoning recover a fact that frozen-J deletion removed?* If think-mode restores recall toward baseline (frozen-J think ≫ frozen-J no-think 0.23), the content channel is bypassable via the CoT — the paper's rescue prediction holds in content-channel form. If recall stays at 0.23, the deletion is upstream of anything reasoning can reconstruct. Either way it's SL1-C6.
- At T+1:10 sharp: bank, refresh handout (§P5), push `"P5: CoT-rescue (capped) — <one-line result>"`.

## T+1:10–2:35 — Phase Q: Qwen 3.6 27B under our instruments (the priority)

Minimal decisive slice, in order; each step banks before the next starts:

1. **Q0 (≤15 min): lens + harness gate.** Load Qwen bf16; load Neuronpedia lens; build the dictionary in our format; run the 10-probe sanity + multihop bridge pass@1 (n=30). Gate: if lens format adaptation exceeds ~20 min of debugging, STOP Qwen-with-their-lens; fallback = fit a micro-lens with Nanda's own recipe (25 Pile/WikiText prompts, penultimate target — his published setting) ONLY if ≤35 min projected on this GPU; otherwise abandon Q, jump to P6, and record `Q_ABANDONED_<reason>` in the ledger.
2. **Q1 (~35 min): the marquee instrument.** frozen_j10 vs frozen_rand10, no-think, twohop (n=30) + onehop (n=30) + arith (n=15 fluency proxy) + prose NLL (n=20). The single question: *does frozen per-item J-ablation delete facts on Qwen too, single- and multi-hop alike?* This is the model-vs-harness verdict for the content-channel claim.
3. **Q2 (~20 min): the null check.** Energy-matched static k=20 (J-span vs matched-random), twohop_lp only (n=30). Does the clean static null hold on the model where the paper's dissociation was said to partially replicate?
4. **Q3 (only if ≥15 min slack): cot-lead spot check.** Qwen think-mode, 15 twohop items, our detector + the frequency-matched foil floor. Even n=15 with the foil floor is reportable as preliminary.
- T+2:35: bank, refresh handout (§Q with the decision-tree verdict stated explicitly: model difference vs method artifact), push `"Q: Qwen under our instruments — <verdict>"`.

## T+2:35–3:00 — P6 micro-robustness (whatever GPU remains)

- seed=1 on the frozen-J grid ONLY (the marquee claim gets the second seed first), twohop n=30. If time: seed=1 energy-matched k=20 twohop_lp. Skip n→60 doubling entirely — cross-seed beats bigger-n for credibility per GPU-minute. Bank + push.

## Reporting (CPU, mandatory, runs even if GPU died at any point above)

1. `report/REPORT_v2.md` — final: v1 claims table (survived/flipped/sharpened), P1–P6 + Q sections, the synthesis line (static span = nothing; frozen per-item = fact deletion; live per-token = computation deletion; late band = no hidden workspace; lead = foil-calibrated, two-lens), and an honest §"what 3 more GPU-hours would have bought" listing anything gated out.
2. `report/summary_v2.json` final numbers incl. rescue + Qwen.
3. **Ledger:** update SL1-C2 (add rescue clause), add SL1-C6 (CoT-rescue) and SL1-C7 (Qwen cross-model verdict) in the Lab-36 template format with Artifact + Falsifier lines; update the lab header's one-sentence allowed claim to the final form.
4. `labs/lab37_jspace_workspace.md`: flip Status from draft-in-progress to **complete, proposed for promotion**; fill the promotion-criteria checklist with what passed/failed.
5. `jspace/README.md` + PR #9 body: final state, headline results, artifact map, "reproduce from Drive" pointer. Flip `interpret/inprogress.md` to `FINAL — see REPORT_v2.md`.
6. Last handout refresh; final push `"Lab 37 final: <verdict slug>"`.

## Priority inversion rule (if things go wrong)

GPU dies mid-window → skip remaining GPU phases in order of DROP: P6 first, then Q3, then Q2, then s16-completion; **never** drop Q1 if the Qwen gate passed — a banked half-n Q1 outranks a finished s16. The report ships regardless, stating exactly where the clock stopped.
