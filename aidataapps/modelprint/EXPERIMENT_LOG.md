# ModelPrint Experiment Log

Append-only operator record.

- 2026-08-22T22:34:23.229Z — MP-0 SQL capability doctor completed for modelprint-full-20260822T222334Z; mode=ann_legacy; hash=47cb8194bee29f2a95512ec2bc3846b948f80fd35935e130f687fe48c9086cc2.

- 2026-08-22T22:35:07.474Z — MP-0 SQL capability doctor completed for modelprint-full-20260822T222334Z; mode=ann_legacy; hash=25304c872071ee92bb05758e6bafabb017edaa0bc48dfe7c340af0fd0a490b84.

- 2026-08-22T22:35:23.850Z — MP-0 SQL capability doctor completed for modelprint-full-20260822T222334Z; mode=ann_v3; hash=a86d065655be785fc6bbaf4ea1055af54afa19f18a6fec0c942f820682cfb355.

- 2026-08-22T22:35:46.458Z — MP-0 SQL capability doctor completed for modelprint-full-20260822T222334Z; mode=ann_legacy; hash=473947c1d9a2fe72466ea3ba21ff3891bcfe34cbe8c5cfb56a4c02d6f0cca4dd.

- 2026-08-22T22:45:07.038Z — MP-2 campaign freeze completed for modelprint-full-20260822T222334Z; campaign=8a04067836e2878c0ed1a3d5b6815a58b364f6cf006093dba00660b292de3716; freeze=ed17a0764ce693b341c5f1e8a4b09b0896a1191088f509ed5f79019709d3b0d6; jobs=40000.

- 2026-08-22T22:53:16.729Z — MP-3 port gate PASS for qwen-smoke; artifact=runs/modelprint-full-20260822T222334Z/environment/port-gate-qwen-smoke.json.

- 2026-08-22T22:59:19.281Z — MP-3 port gate STOP_PORT for qwen-smoke; artifact=runs/modelprint-full-20260822T222334Z/environment/port-gate-qwen-smoke.json.

- 2026-08-22T23:00:32.124Z — MP-2 campaign freeze completed for modelprint-full-20260822T222334Z; campaign=4a2d5c24d84c6de519943cf84b505fa8315351c46be7ac02f8df419f44744e98; freeze=77ef3b03aaa0d34a98ee9807906cfc22e97a9c4b96e5036af2d5b29690d4bb55; jobs=40000.

- 2026-08-22T23:00:40.939Z — MP-3 port gate PASS for qwen-smoke; artifact=runs/modelprint-full-20260822T222334Z/environment/port-gate-qwen-smoke.json.

- 2026-08-22T23:05:22.567Z — MP-3 port gate PASS for qwen-3.8-27b; artifact=runs/modelprint-full-20260822T222334Z/environment/port-gate-qwen-3.8-27b.json.

- 2026-08-22T23:05:58.000Z — Four-row Qwen 3.8 persistence pilot for campaign 2 stopped after generation-row commit when reserved T-SQL alias `view` failed the text-artifact batch. Campaign 2 is excluded; no pilot row will enter analysis. Alias corrected and the fixed transaction passed exact-vector, dimension-rejection, and set-based artifact integration checks before replacement freeze.

- 2026-08-22T23:07:54.545Z — MP-0 SQL capability doctor completed for modelprint-full-20260822T230728Z; mode=ann_legacy; hash=fe4af6bd11469784e2d93db9ca0bbbcb4aa30dd30023786f69ec1a8632004d06.

- 2026-08-22T23:08:01.810Z — MP-2 campaign freeze completed for modelprint-full-20260822T230728Z; campaign=52113ce90ed5302c0f40f55e79d5962aa692925721cec0ce3c2684c6947673d9; freeze=5953272ae8b4edc3f091980d9e507aa8d911c56f8ca88eb91fe3d4a492366799; jobs=40000.

- 2026-08-22T23:08:40.884Z — MP-3 port gate PASS for qwen-3.8-27b; artifact=runs/modelprint-full-20260822T230728Z/environment/port-gate-qwen-3.8-27b.json.

- 2026-08-22T23:08:57.368Z — MP-4 generation invocation completed for qwen-3.8-27b; selected=4; completed=4; failed=0.

- 2026-08-22T23:31:24.000Z — MP-5/MP-11 implementation checkpoint: TypeScript build and 12 CPU tests passed; the report builder reconstructed an honest partial results pack from campaign 3 SQL rows; API health, insufficient-text abstention, and evaluation-list endpoints passed against the live database. Scientific evaluation remains gated on completion of at least two target profiles.

- 2026-08-22T23:37:31.746Z — MP-2 robustness companion freeze linked to primary campaign 3; campaign=6449a7c5a62e6d1749fa88799f67d5164a436217dcd5019eb4e36db7e8657d04; jobs=2004; carrier counts={"direct-v1":185,"persona-v1":256,"rag-grounded-v1":60}.

- 2026-08-22T23:38:14.950Z — MP-4 robustness generation completed for qwen-3.8-27b; selected=1; completed=1; failed=0.

- 2026-08-22T23:56:50.000Z — Database extension checkpoint: the migration runner found historical recorded-hash differences for 001/003 in this already-running database. Before accepting the explicit `--allow-historical-drift` baseline exception, the SQL integration suite passed exact vector ranking, dimension rejection, and set-based artifact persistence. Existing migrations were not replayed or rewritten; migration 004 alone was applied and recorded.

- 2026-08-22T23:59:54.000Z — MP-5 pre-registered implementation adjustment: SQL Server 2025 CU8 rejected `AI_GENERATE_CHUNKS OVERLAP=100` because the function accepts a 0–50 percentage. The SQL-native comparator was changed to 20 percent overlap (approximately the intended 100/600 character ratio). Reference-token updates and app-side segments from the stopped attempt are idempotent; no generation data changed.

- 2026-08-23T00:20:10.114Z — MP-4 generation invocation completed for qwen-3.8-27b; selected=9096; completed=9096; failed=0.

- 2026-08-23T00:22:02.548Z — MP-4 robustness generation completed for qwen-3.8-27b; selected=500; completed=500; failed=0.

- 2026-08-23T00:38:31.712Z — MP-4 likelihood checkpoint for qwen-3.8-27b: decoded-character assistant-span slicing replaced unsafe separately-tokenized prefix slicing after 267 junction-merge failures were observed. The retained selective rerun recovered every prompted score (8,001/8,001 det/nat target and robustness rows). Unprompted echo scoring is explicitly unavailable for 17 one-token outputs because vLLM returns a null first-token log probability; those rows remain missing rather than receiving an invented likelihood.
- 2026-08-23T04:13:00Z — MP-4 Muse cross-likelihood runtime correction: the first Muse scoring pass preserved valid unprompted channels but reported prompted-span failures where vLLM's per-token `decoded_token` rendered Unicode byte-fallback pieces as U+FFFD (and sometimes absorbed the token-internal leading space). This is a token-inspection artifact, not altered model text or an HTTP failure. `sliceAssistantLogprobs` now retains exact ASCII matching and permits replacement runs only at governed non-ASCII code points, with an optional immediately preceding space only inside that fallback alternative. Tests prove exact junction behavior, recovery of the observed `⚠️` form, and refusal of an unrelated ASCII mismatch. Five original failures were replayed against the unchanged resident Muse server and all five recovered with the `unicode-byte-fallback` alignment. The already-running old-code pass was not interrupted; its partial channels remain append-only and an idempotent selective rerun will fill only SQL-missing cells before residency rotation.
- 2026-08-23T04:21:00Z — MP-4 empty-final likelihood policy: the governed non-HV scoring set currently contains 64 Muse rows whose retained final answer is empty after `finish_reason=length` exhausted the budget in reasoning. Per addendum C-15 these remain truncated data-quality rows and are excluded from headline claims. A likelihood for an empty string is unavailable: the scorer now excludes `LEN(final_text)=0` from work selection and records `unavailableEmptyFinal` in every checkpoint/manifest rather than emitting infinity, calling it a persisted score, or selecting the same rows forever. No likelihood value is imputed.
- 2026-08-23T04:34:15Z — MP-4 Muse likelihood residency complete: the original append-only pass retained 15,941 fully paired jobs and 61 prompted Unicode-alignment gaps; the corrected selective pass selected and recovered all 61 with zero failures. A final no-op audit selected zero and recorded `eligibleNonEmpty=15,938`, `promptedPersisted=15,938`, `unpromptedPersisted=15,938`, both missing counts zero, and `unavailableEmptyFinal=64`. The scorer manifest now reports total SQL completeness even after a selective/no-op invocation and preserves the cumulative raw SHA-256 `9d59cb2fbc7e2fdde8168e6654b19c53407b29aaf59f8a63668cf6f6a1682193`.
- 2026-08-23T04:41:00Z — MP-2 Gemma start attempt 1 stopped before API readiness. Pinned `google/gemma-4-31B-it` revision `842da379…` downloaded completely and loaded 57.91 GiB of weights, but at the frozen 16,384 context vLLM required 13.76 GiB KV cache while configured GPU utilization 0.78 exposed 13.22 GiB (estimated maximum context 15,728). No gate or campaign request ran. A runtime-only `CHAT_GPU_MEMORY_UTILIZATION` override was added with validation, effective-value container labeling, and a separate runtime-profile hash; retry uses 0.80 while model, revision, image, context, prompts, and decoding remain unchanged.
- 2026-08-23T04:45:00Z — MP-3 Gemma gate diagnosis: an initial invocation used the default derived container name and stopped before canaries; SQL evidence event 8 retains it and the run now materializes `port-gate-gemma-4-31b-attempt1-container-lookup.json`. With the exact retry container bound, the unchanged gate reached canaries but returned `STOP_PORT` because sequential/batched deterministic hashes differed; evidence event 9 and `port-gate-gemma-4-31b-attempt2-batch-sensitive.json` preserve that disposition. No campaign request ran. Retry enables the image-supported `VLLM_BATCH_INVARIANT=1`, retains effective GPU utilization 0.80, and reruns the unchanged governed gate.

- 2026-08-23T01:03:40.803Z — MP-3 port gate STOP_PORT for muse-glimmer-30b; artifact=runs/modelprint-full-20260822T230728Z/environment/port-gate-muse-glimmer-30b.json.

- 2026-08-23T01:04:23.277Z — MP-3 Muse initial port gate STOP_PORT: exact sequential/batched greedy outputs differed. Moderate runtime adjustment: enable the pinned vLLM image batch-invariant mode and rerun the unchanged governed gate; first artifact retained as environment/port-gate-muse-glimmer-30b-attempt1.json.

- 2026-08-23T01:10:10.870Z — MP-3 port gate PASS for muse-glimmer-30b; artifact=runs/modelprint-full-20260822T230728Z/environment/port-gate-muse-glimmer-30b.json.

- 2026-08-23T03:59:58.533Z — MP-4 generation invocation completed for muse-glimmer-30b; selected=10000; completed=10000; failed=0.

- 2026-08-23T04:05:08.802Z — MP-4 robustness generation completed for muse-glimmer-30b; selected=501; completed=501; failed=0.

- 2026-08-23T04:44:00.002Z — MP-3 port gate STOP_PORT for gemma-4-31b; artifact=runs/modelprint-full-20260822T230728Z/environment/port-gate-gemma-4-31b.json.

- 2026-08-23T04:44:46.252Z — MP-3 port gate STOP_PORT for gemma-4-31b; artifact=runs/modelprint-full-20260822T230728Z/environment/port-gate-gemma-4-31b.json.

- 2026-08-23T04:50:44.257Z — MP-3 port gate PASS for gemma-4-31b; artifact=runs/modelprint-full-20260822T230728Z/environment/port-gate-gemma-4-31b.json.

- 2026-08-23T06:47:20.694Z — MP-4 generation invocation completed for gemma-4-31b; selected=10000; completed=10000; failed=0.

- 2026-08-23T06:48:54.764Z — MP-4 robustness generation completed for gemma-4-31b; selected=501; completed=501; failed=0.

- 2026-08-23T07:37:03Z — MP-4 Gemma likelihood residency complete: all 23,939 eligible non-empty Qwen/Muse/Gemma target and robustness rows have prompted likelihood; 23,890 also have unprompted likelihood. The retained 49 prompted-only records were audited exhaustively: every record contains one valid prompted score over exactly one Gemma output token and only `No unprompted logprobs`, because vLLM has no previous-token distribution for the first token of an unprompted echo. Counts are Qwen=20, Muse=1, Gemma=28. No value was imputed; 64 empty-final Muse rows remain separately unavailable. Raw SHA-256 is `775df32e203cb9774846696634adf33a680320ddc4e3ec7a8b9a065130d1371b`. A pre-eviction SQL backup and Drive copy matched SHA-256 `5d3c7383be0a482173770abc1e260e583eafe99c585f39a8b8687c16c989b7bb`; Gemma then stopped cleanly and exact `hf cache rm` eviction removed one repository/revision and freed 62.6 GB.

- 2026-08-23T07:48:58.147Z — MP-3 port gate PASS for olmo-3.1-32b-instruct; artifact=runs/modelprint-full-20260822T230728Z/environment/port-gate-olmo-3.1-32b-instruct.json.
