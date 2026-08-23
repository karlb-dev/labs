# LogWarden Tier 1 state of record

- Run: `logwarden-smoke-20260823T031714Z`
- Freeze: `e105cfdd5af5345464853406c8d707018232f3326c909ca476970fb7137cbcf6` (`1819a96b69cb90c73c450aa2da5376bf6ab565c3d20f9293192746e5fc4e60df`)
- Corpus: 600/600 capture gates passed; 600 packets; leakage findings: 0.
- Analysis: 8,430 primary predictions, 5,085 metric rows, 36 paired contrasts; receipt `7fcfb31517c560a7fdaad9b03b800dec10898415c831377e3407135fff41737d`.
- Telemetry: 11,332 closed traces, 65,321 closed spans, 9,986 paired model requests/responses; receipt `8c2de36fe09cdc0905cfa54977b7e32aeb49641360fdc4458e698ad0e7f3cb84`.
- Safety: `SAFETY_CLEAN`.
- Primary adjudication: `CLEAN_NULL`.

## Adjudication

1. The corpus passed 600/600 capture checks and a zero-finding leakage audit.
2. Muse, Gemma, and Qwen completed every governed Tier-1 cell; OLMo is explicitly `STOP_PORT` after two gates.
3. B1 was strong: action accuracy 0.9167; B3 is the evaluator-only information ceiling.
4. No model/arm beat B1 under the complete frozen inference gate.
5. Regime/family differences are descriptive only and remain in the scorecard/taxonomy exports.
6. Hybrid retrieval had the best descriptive recall among executable search modes, but no paired search-mode inference was materialized.
7. Retrieval changed several marginal outcomes, but no causal retrieval contrast survived multiplicity and permutation control.
8. Tool behavior varied materially; fragile cells are labeled from required, forbidden, argument, and snapshot evidence.
9. Tier-1 packet correlation is a harness property (`PACKET_CORRELATION_ONLY`).
10. Calibration is limited; nonconverged fits and selective coverage are explicit.
11. Model-contract/policy failures dominate the retained agent failure funnel; SQL deadlocks affected orchestration probes, not immutable outputs.
12. Replay latency/token/GPU/SQL telemetry is complete; live throughput Pareto analysis remains Tier 2.
13. Derived A-router reproduced B1 because B1 resolved the retained episodes; it adds zero inference cost and no quality lift.
14–18. Live parity, event-rate capacity, raw-event correlation, SQL feature ablations, and ANN are Tier 2 and are not claimed.
19. The independent Tier-1 permission/procedure audit passed; this is not production-safety evidence.
20. Row-only and restored-database reconstruction both passed; their stable receipts are recorded in `LOGWARDEN_REPRODUCIBILITY.md`.
21. The principal unexplained result is why the declared permutation scheme is degenerate for several binary contrasts; no positive claim depends on it.
22. The defensible carry-forward architecture is deterministic rules for the known head, explicitly gated retrieval/model assistance for residual cases, and SQL Server as the durable evidence/queue/evaluation plane—not an autonomous remediation agent.
