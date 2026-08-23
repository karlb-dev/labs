# Agent and inference performance report

The reconciled telemetry set contains 11,332 closed traces, 65,321 closed spans, 9,986 paired model requests/responses, 4,161,680 metric samples, and 16,501 raw snapshots.

## Primary agent cells

| profile | arm | episodes | successful_episodes | agent_p50_ms | agent_p90_ms | agent_p95_ms | agent_p99_ms | model_client_p50_ms | model_client_p90_ms | model_client_p95_ms | model_client_p99_ms | model_requests | model_responses | prompt_tokens | completion_tokens | tokens_per_episode | model_seconds_per_success | tool_calls_per_success | first_pass_valid_rate | repaired_response_rate | rejected_response_rate | model_errors | length_finishes | snapshot_misses | tool_errors | validation_failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | A-direct | 480 | 454 | 12829.3560 | 14484.0292 | 15632.2459 | 18169.3292 | 12698.9990 | 14343.8272 | 15502.0854 | 18031.3301 | 480 | 480 | 505698 | 100325 | 1262.5479 | 13.5434 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| gemma-4-31b | A-rag | 410 | 404 | 18351.1870 | 21716.3045 | 24947.3999 | 27655.3759 | 17901.1990 | 21329.8758 | 24442.7747 | 26945.5803 | 820 | 820 | 1410108 | 97289 | 3676.5780 | 18.6513 | 1.0149 | 0.4756 | 0.5244 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| gemma-4-31b | A-tools | 480 | 460 | 11247.5760 | 17822.8548 | 18571.1352 | 21185.5237 | 11110.8280 | 17509.7762 | 18258.8684 | 20855.0892 | 638 | 638 | 1355577 | 84583 | 3000.3333 | 12.7518 | 0.3435 | 0.8699 | 0.1301 | 0.0000 | 0 | 0 | 77 | 0 | 0 |
| muse-glimmer-30b | A-direct | 480 | 428 | 26350.8945 | 34045.1566 | 36158.8196 | 42182.9594 | 26222.3365 | 33917.1631 | 36023.5663 | 42044.1901 | 480 | 480 | 439901 | 157382 | 1244.3396 | 29.7289 | 0.0000 | 0.9917 | 0.0000 | 0.0083 | 4 | 0 | 0 | 0 | 4 |
| muse-glimmer-30b | A-rag | 410 | 400 | 49814.3190 | 59108.3153 | 63729.9059 | 68472.7741 | 49431.0415 | 58737.6649 | 63343.2947 | 68105.6916 | 820 | 820 | 1230049 | 243377 | 3593.7220 | 51.4815 | 1.0250 | 0.9988 | 0.0000 | 0.0012 | 1 | 1 | 0 | 0 | 1 |
| muse-glimmer-30b | A-tools | 480 | 421 | 50113.7725 | 72626.7822 | 77301.8753 | 87831.8822 | 49559.0515 | 71899.6258 | 76500.5302 | 87056.2283 | 1263 | 1263 | 2999002 | 273178 | 6817.0417 | 55.5392 | 1.8717 | 0.9960 | 0.0000 | 0.0040 | 5 | 0 | 148 | 0 | 5 |
| qwen-3.8-27b | A-direct | 480 | 459 | 9723.6030 | 11081.3636 | 11517.3176 | 12264.4111 | 9593.7895 | 10948.8326 | 11362.6600 | 12125.0121 | 480 | 480 | 485175 | 85693 | 1189.3083 | 10.1237 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| qwen-3.8-27b | A-rag | 410 | 404 | 18732.4660 | 23153.7419 | 23927.0266 | 26380.0057 | 18347.6950 | 22758.6256 | 23522.2913 | 25992.9378 | 820 | 820 | 1371657 | 113051 | 3621.2390 | 19.1390 | 1.0149 | 1.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 |
| qwen-3.8-27b | A-tools | 480 | 429 | 18316.3040 | 26552.7593 | 27669.6027 | 30241.4295 | 17959.3070 | 25924.5297 | 26990.8974 | 29647.0164 | 1070 | 1070 | 2384091 | 90813 | 5156.0500 | 20.9267 | 1.3753 | 0.9907 | 0.0000 | 0.0093 | 10 | 0 | 82 | 0 | 10 |

## Persisted pipeline span distributions

| agent_arm_id | max_ms | mean_ms | model_profile_id | non_success_count | p50_ms | p90_ms | p95_ms | p99_ms | run_kind | sample_count | span_name |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A-direct | 18718.9310 | 12921.0545 | gemma-4-31b | 26 | 12848.5165 | 14339.8667 | 15134.3805 | 18122.6182 | agent_replay | 540 | agent.loop |
| A-direct | 39.6070 | 5.4018 | gemma-4-31b | 0 | 4.8050 | 6.9800 | 8.1484 | 13.7321 | agent_replay | 514 | decision.persist |
| A-direct | 1.4820 | 0.1375 | gemma-4-31b | 26 | 0.1230 | 0.1590 | 0.1952 | 0.4118 | agent_replay | 540 | decision.validate |
| A-direct | 18576.6170 | 12774.7145 | gemma-4-31b | 0 | 12708.2075 | 14193.4761 | 15001.1477 | 17979.4566 | agent_replay | 540 | model.operation |
| A-direct | 18576.0270 | 12773.9927 | gemma-4-31b | 0 | 12707.6135 | 14192.8862 | 15000.6987 | 17978.9296 | agent_replay | 540 | model.request |
| A-direct | 1.6900 | 0.1483 | gemma-4-31b | 0 | 0.1150 | 0.1772 | 0.3533 | 0.7115 | agent_replay | 540 | prompt.assemble |
| A-rag | 29777.0910 | 18663.1718 | gemma-4-31b | 6 | 18151.1510 | 21376.4588 | 24528.1708 | 27455.9166 | agent_replay | 470 | agent.loop |
| A-rag | 17.5130 | 5.6424 | gemma-4-31b | 0 | 5.0735 | 7.3571 | 8.5599 | 11.9808 | agent_replay | 464 | decision.persist |
| A-rag | 3.1320 | 0.1491 | gemma-4-31b | 6 | 0.1240 | 0.1760 | 0.2322 | 0.3553 | agent_replay | 470 | decision.validate |
| A-rag | 16.8730 | 9.6261 | gemma-4-31b | 0 | 9.6320 | 11.0468 | 11.4213 | 12.8866 | agent_replay | 470 | embedding.request |
| A-rag | 23624.4400 | 9118.8827 | gemma-4-31b | 0 | 9429.1310 | 15367.2740 | 16418.4878 | 20840.9351 | agent_replay | 940 | model.operation |
| A-rag | 23623.9370 | 9118.2894 | gemma-4-31b | 0 | 9428.6165 | 15366.7726 | 16417.9404 | 20840.3711 | agent_replay | 940 | model.request |
| A-rag | 1.6320 | 0.1424 | gemma-4-31b | 0 | 0.1300 | 0.1650 | 0.2030 | 0.4267 | agent_replay | 940 | prompt.assemble |
| A-rag | 165.8300 | 96.0812 | gemma-4-31b | 0 | 93.6480 | 105.3702 | 111.2338 | 139.6426 | agent_replay | 470 | retrieval.operation |
| A-rag | 167.0570 | 96.6903 | gemma-4-31b | 0 | 94.2585 | 106.0332 | 111.7953 | 140.7467 | agent_replay | 470 | tool.operation |
| A-tools | 22228.3080 | 12211.3748 | gemma-4-31b | 20 | 10956.7805 | 17699.2622 | 18473.4487 | 21083.3110 | agent_replay | 540 | agent.loop |
| A-tools | 15.3200 | 5.0970 | gemma-4-31b | 0 | 4.6945 | 6.3552 | 7.1683 | 10.6287 | agent_replay | 520 | decision.persist |
| A-tools | 2.1870 | 0.1398 | gemma-4-31b | 20 | 0.1230 | 0.1590 | 0.2380 | 0.4541 | agent_replay | 540 | decision.validate |
| A-tools | 8.6830 | 8.6830 | gemma-4-31b | 0 | 8.6830 | 8.6830 | 8.6830 | 8.6830 | agent_replay | 1 | embedding.request |
| A-tools | 17886.1750 | 9166.0636 | gemma-4-31b | 0 | 9887.1980 | 13793.4875 | 15227.4565 | 17053.7764 | agent_replay | 708 | model.operation |
| A-tools | 17885.6200 | 9165.4488 | gemma-4-31b | 0 | 9886.6750 | 13792.9046 | 15226.8510 | 17053.1561 | agent_replay | 708 | model.request |
| A-tools | 0.4680 | 0.1391 | gemma-4-31b | 0 | 0.1280 | 0.1640 | 0.2017 | 0.3549 | agent_replay | 708 | prompt.assemble |
| A-tools | 135.2930 | 135.2930 | gemma-4-31b | 0 | 135.2930 | 135.2930 | 135.2930 | 135.2930 | agent_replay | 1 | retrieval.operation |
| A-tools | 135.9200 | 2.8906 | gemma-4-31b | 0 | 1.4720 | 2.2839 | 2.7373 | 43.8014 | agent_replay | 168 | tool.operation |
| A-tools | 13957.5130 | 8363.7605 | gemma-4-31b | 0 | 7821.6550 | 10069.5255 | 11951.8472 | 13745.8631 | control_batching_sequential | 48 | agent.loop |
| A-tools | 9.0890 | 6.1057 | gemma-4-31b | 0 | 5.8815 | 7.2084 | 7.6909 | 8.8784 | control_batching_sequential | 48 | decision.persist |
| A-tools | 0.5150 | 0.1778 | gemma-4-31b | 0 | 0.1540 | 0.2651 | 0.4243 | 0.4840 | control_batching_sequential | 48 | decision.validate |
| A-tools | 9306.3870 | 5924.5668 | gemma-4-31b | 0 | 7010.3810 | 8402.0035 | 8606.9568 | 9170.0638 | control_batching_sequential | 66 | model.operation |
| A-tools | 9305.8860 | 5923.7319 | gemma-4-31b | 0 | 7009.3780 | 8401.0830 | 8606.1205 | 9169.5114 | control_batching_sequential | 66 | model.request |
| A-tools | 0.6020 | 0.1706 | gemma-4-31b | 0 | 0.1585 | 0.2010 | 0.2538 | 0.4219 | control_batching_sequential | 66 | prompt.assemble |
| A-tools | 44.7660 | 13.5909 | gemma-4-31b | 0 | 2.3945 | 43.4316 | 43.6448 | 44.5418 | control_batching_sequential | 18 | tool.operation |
| A-tools | 19245.4940 | 13351.1074 | gemma-4-31b | 0 | 12640.2925 | 16944.2335 | 17763.3122 | 18707.2316 | control_error_number_mask | 96 | agent.loop |
| A-tools | 16.4310 | 6.5886 | gemma-4-31b | 0 | 5.8235 | 9.1845 | 10.3815 | 16.1498 | control_error_number_mask | 96 | decision.persist |
| A-tools | 0.6250 | 0.1638 | gemma-4-31b | 0 | 0.1360 | 0.2265 | 0.3838 | 0.5936 | control_error_number_mask | 96 | decision.validate |
| A-tools | 10.9360 | 9.7931 | gemma-4-31b | 0 | 9.8910 | 10.7326 | 10.8343 | 10.9157 | control_error_number_mask | 10 | embedding.request |
| A-tools | 15847.1840 | 9056.2550 | gemma-4-31b | 0 | 10708.3040 | 13924.6566 | 14335.3035 | 15244.8220 | control_error_number_mask | 139 | model.operation |
| A-tools | 15846.5970 | 9054.6328 | gemma-4-31b | 0 | 10707.2140 | 13923.8800 | 14334.6933 | 15244.0907 | control_error_number_mask | 139 | model.request |
| A-tools | 3.2510 | 0.2800 | gemma-4-31b | 0 | 0.1580 | 0.4616 | 0.9356 | 1.3582 | control_error_number_mask | 139 | prompt.assemble |
| A-tools | 119.5050 | 91.1140 | gemma-4-31b | 0 | 88.5010 | 93.1647 | 106.3348 | 116.8710 | control_error_number_mask | 10 | retrieval.operation |
| A-tools | 123.3670 | 27.9557 | gemma-4-31b | 0 | 4.0080 | 90.3426 | 91.6695 | 110.9791 | control_error_number_mask | 43 | tool.operation |
| A-tools | 22395.8770 | 12577.9771 | gemma-4-31b | 2 | 11662.5555 | 17319.4055 | 18088.1490 | 22391.9772 | control_shuffled_runbooks | 96 | agent.loop |
| A-tools | 46.1640 | 5.7427 | gemma-4-31b | 0 | 4.9250 | 6.6632 | 7.7993 | 13.4698 | control_shuffled_runbooks | 94 | decision.persist |
| A-tools | 1.1290 | 0.1529 | gemma-4-31b | 2 | 0.1290 | 0.1700 | 0.2105 | 1.0539 | control_shuffled_runbooks | 96 | decision.validate |
| A-tools | 13956.1690 | 8737.6888 | gemma-4-31b | 0 | 10377.4100 | 12633.9630 | 12925.4760 | 13615.1621 | control_shuffled_runbooks | 131 | model.operation |
| A-tools | 13955.5620 | 8736.7912 | gemma-4-31b | 0 | 10376.2790 | 12632.6260 | 12924.8850 | 13614.6730 | control_shuffled_runbooks | 131 | model.request |
| A-tools | 1.3320 | 0.2192 | gemma-4-31b | 0 | 0.1400 | 0.3530 | 0.7415 | 1.2702 | control_shuffled_runbooks | 131 | prompt.assemble |
| A-tools | 47.8520 | 9.6632 | gemma-4-31b | 0 | 1.8560 | 47.7806 | 47.7813 | 47.8282 | control_shuffled_runbooks | 35 | tool.operation |
| A-direct | 54627.5010 | 26691.0254 | muse-glimmer-30b | 57 | 26357.9550 | 34579.6478 | 37270.6002 | 43119.0962 | agent_replay | 540 | agent.loop |
| A-direct | 99.5380 | 5.2993 | muse-glimmer-30b | 0 | 4.8140 | 6.5052 | 6.9153 | 8.2232 | agent_replay | 483 | decision.persist |
| A-direct | 0.5080 | 0.1294 | muse-glimmer-30b | 53 | 0.1220 | 0.1555 | 0.1710 | 0.3410 | agent_replay | 536 | decision.validate |
| A-direct | 54469.6260 | 26550.0455 | muse-glimmer-30b | 4 | 26223.4405 | 34429.3476 | 37138.6565 | 42962.1139 | agent_replay | 540 | model.operation |
| A-direct | 54468.9560 | 26549.3291 | muse-glimmer-30b | 4 | 26222.8205 | 34427.4262 | 37138.1850 | 42959.1446 | agent_replay | 540 | model.request |
| A-direct | 1.5650 | 0.1335 | muse-glimmer-30b | 0 | 0.1110 | 0.1470 | 0.1732 | 0.9974 | agent_replay | 540 | prompt.assemble |
| A-rag | 89880.5660 | 50563.3733 | muse-glimmer-30b | 12 | 49775.8505 | 59115.6859 | 63182.2379 | 68398.7435 | agent_replay | 470 | agent.loop |
| A-rag | 16.4990 | 5.6551 | muse-glimmer-30b | 0 | 5.3350 | 7.1314 | 7.6767 | 9.7715 | agent_replay | 458 | decision.persist |
| A-rag | 3.2240 | 0.1366 | muse-glimmer-30b | 10 | 0.1220 | 0.1513 | 0.1637 | 0.4272 | agent_replay | 468 | decision.validate |
| A-rag | 19769.2480 | 72.6757 | muse-glimmer-30b | 0 | 9.7105 | 10.7717 | 11.2259 | 12.5681 | agent_replay | 470 | embedding.request |
| A-rag | 75505.8330 | 25053.7538 | muse-glimmer-30b | 2 | 21565.6720 | 40983.8758 | 44258.0700 | 52643.3057 | agent_replay | 940 | model.operation |
| A-rag | 75505.2070 | 25053.1740 | muse-glimmer-30b | 2 | 21565.0190 | 40983.2783 | 44257.5636 | 52642.7672 | agent_replay | 940 | model.request |
| A-rag | 1.5560 | 0.1359 | muse-glimmer-30b | 0 | 0.1270 | 0.1601 | 0.1780 | 0.3738 | agent_replay | 940 | prompt.assemble |
| A-rag | 19857.0150 | 158.6946 | muse-glimmer-30b | 0 | 93.1815 | 101.9569 | 108.9772 | 146.3023 | agent_replay | 470 | retrieval.operation |
| A-rag | 19857.6470 | 159.2710 | muse-glimmer-30b | 0 | 93.7655 | 102.4435 | 109.5672 | 148.1348 | agent_replay | 470 | tool.operation |
| A-tools | 99495.0080 | 48275.7606 | muse-glimmer-30b | 66 | 49546.5630 | 72354.3785 | 77025.9325 | 86917.0996 | agent_replay | 540 | agent.loop |
| A-tools | 51.6630 | 5.5638 | muse-glimmer-30b | 0 | 5.1360 | 6.8238 | 7.4940 | 10.5003 | agent_replay | 474 | decision.persist |
| A-tools | 2.4740 | 0.1398 | muse-glimmer-30b | 55 | 0.1200 | 0.1522 | 0.1866 | 0.5164 | agent_replay | 529 | decision.validate |
| A-tools | 13.0760 | 9.8048 | muse-glimmer-30b | 0 | 9.7160 | 11.0088 | 11.2672 | 12.7335 | agent_replay | 325 | embedding.request |
| A-tools | 57920.1800 | 18420.5598 | muse-glimmer-30b | 6 | 15255.6015 | 37133.8669 | 41889.8914 | 47336.0162 | agent_replay | 1402 | model.operation |
| A-tools | 57919.6620 | 18419.9432 | muse-glimmer-30b | 6 | 15255.0930 | 37133.3439 | 41889.2753 | 47335.4870 | agent_replay | 1402 | model.request |
| A-tools | 3.4850 | 0.1560 | muse-glimmer-30b | 0 | 0.1400 | 0.1800 | 0.2030 | 0.4540 | agent_replay | 1402 | prompt.assemble |
| A-tools | 136.7690 | 94.9503 | muse-glimmer-30b | 0 | 93.4210 | 100.5116 | 104.6552 | 133.9630 | agent_replay | 325 | retrieval.operation |
| A-tools | 137.2100 | 36.6816 | muse-glimmer-30b | 0 | 1.9470 | 96.4748 | 99.6744 | 114.5346 | agent_replay | 867 | tool.operation |
| A-tools | 79797.5440 | 49149.4156 | muse-glimmer-30b | 4 | 50289.9675 | 69046.3799 | 73082.1248 | 77412.1295 | control_batching_sequential | 48 | agent.loop |
| A-tools | 8.1290 | 6.1771 | muse-glimmer-30b | 0 | 6.0560 | 7.1043 | 7.3882 | 7.8671 | control_batching_sequential | 44 | decision.persist |
| A-tools | 0.5190 | 0.1784 | muse-glimmer-30b | 3 | 0.1440 | 0.2684 | 0.4559 | 0.4983 | control_batching_sequential | 47 | decision.validate |
| A-tools | 10.8030 | 9.4055 | muse-glimmer-30b | 0 | 9.4110 | 9.8868 | 9.9852 | 10.5529 | control_batching_sequential | 34 | embedding.request |
| A-tools | 47627.2350 | 18088.6994 | muse-glimmer-30b | 1 | 14634.1390 | 37313.4620 | 40000.6312 | 45332.0352 | control_batching_sequential | 129 | model.operation |
| A-tools | 47626.6760 | 18087.9327 | muse-glimmer-30b | 1 | 14633.2850 | 37312.7874 | 39999.8182 | 45331.2644 | control_batching_sequential | 129 | model.request |
| A-tools | 0.6390 | 0.2000 | muse-glimmer-30b | 0 | 0.1750 | 0.2418 | 0.4642 | 0.6106 | control_batching_sequential | 129 | prompt.assemble |
| A-tools | 148.4810 | 114.8491 | muse-glimmer-30b | 0 | 105.6310 | 139.3561 | 144.9088 | 147.7972 | control_batching_sequential | 34 | retrieval.operation |
| A-tools | 149.0490 | 59.4257 | muse-glimmer-30b | 0 | 43.9400 | 137.3120 | 139.5490 | 147.3906 | control_batching_sequential | 81 | tool.operation |
| A-tools | 95990.1260 | 51269.8485 | muse-glimmer-30b | 25 | 49708.0890 | 75281.4495 | 80918.1957 | 84829.6542 | control_error_number_mask | 96 | agent.loop |
| A-tools | 10.7400 | 5.6073 | muse-glimmer-30b | 0 | 5.2150 | 7.2090 | 7.7465 | 9.8405 | control_error_number_mask | 71 | decision.persist |
| A-tools | 0.3220 | 0.1286 | muse-glimmer-30b | 17 | 0.1205 | 0.1578 | 0.1770 | 0.2872 | control_error_number_mask | 88 | decision.validate |
| A-tools | 13.6450 | 9.7464 | muse-glimmer-30b | 0 | 9.6380 | 10.9635 | 11.1288 | 12.3796 | control_error_number_mask | 58 | embedding.request |
| A-tools | 56053.7300 | 17599.2742 | muse-glimmer-30b | 4 | 14318.9640 | 38457.0624 | 44419.0882 | 50858.2492 | control_error_number_mask | 277 | model.operation |
| A-tools | 56053.1490 | 17598.3283 | muse-glimmer-30b | 4 | 14318.5430 | 38456.3340 | 44418.4326 | 50857.7142 | control_error_number_mask | 277 | model.request |
| A-tools | 4.7280 | 0.2224 | muse-glimmer-30b | 0 | 0.1500 | 0.1920 | 0.4548 | 2.0974 | control_error_number_mask | 277 | prompt.assemble |
| A-tools | 140.4700 | 95.5783 | muse-glimmer-30b | 0 | 92.9795 | 99.8503 | 108.1879 | 134.6554 | control_error_number_mask | 58 | retrieval.operation |
| A-tools | 142.3530 | 32.1846 | muse-glimmer-30b | 0 | 2.4380 | 96.9074 | 99.8322 | 130.6200 | control_error_number_mask | 185 | tool.operation |
| A-tools | 93351.6060 | 52100.4193 | muse-glimmer-30b | 16 | 49242.5325 | 80069.6645 | 83155.6995 | 93214.1733 | control_shuffled_runbooks | 96 | agent.loop |
| A-tools | 11.8430 | 5.3197 | muse-glimmer-30b | 0 | 4.9890 | 7.1300 | 7.3038 | 8.9002 | control_shuffled_runbooks | 80 | decision.persist |
| A-tools | 0.3230 | 0.1357 | muse-glimmer-30b | 9 | 0.1260 | 0.1694 | 0.2052 | 0.2746 | control_shuffled_runbooks | 89 | decision.validate |
| A-tools | 52699.6000 | 17030.6778 | muse-glimmer-30b | 3 | 13665.8480 | 36523.6080 | 40567.9070 | 49911.6776 | control_shuffled_runbooks | 291 | model.operation |
| A-tools | 52699.0820 | 17029.6519 | muse-glimmer-30b | 3 | 13665.4140 | 36523.0990 | 40567.2450 | 49911.0311 | control_shuffled_runbooks | 291 | model.request |
| A-tools | 2.5300 | 0.2179 | muse-glimmer-30b | 0 | 0.1540 | 0.2000 | 0.5330 | 1.7858 | control_shuffled_runbooks | 291 | prompt.assemble |
| A-tools | 44.4350 | 3.7665 | muse-glimmer-30b | 3 | 2.1090 | 6.6862 | 7.6114 | 43.7484 | control_shuffled_runbooks | 199 | tool.operation |
| A-direct | 20900.5890 | 9887.5528 | qwen-3.8-27b | 21 | 9645.3535 | 11231.8289 | 11808.3728 | 20753.8077 | agent_replay | 540 | agent.loop |
| A-direct | 19.6290 | 5.5956 | qwen-3.8-27b | 0 | 4.9440 | 7.1788 | 9.2499 | 16.3736 | agent_replay | 519 | decision.persist |
| A-direct | 1.4390 | 0.1423 | qwen-3.8-27b | 21 | 0.1200 | 0.1611 | 0.2629 | 0.7920 | agent_replay | 540 | decision.validate |
| A-direct | 13479.2550 | 9506.8430 | qwen-3.8-27b | 0 | 9447.4240 | 10919.2209 | 11315.7769 | 12087.3411 | agent_replay | 540 | model.operation |
| A-direct | 13478.6870 | 9506.1346 | qwen-3.8-27b | 0 | 9446.8885 | 10918.8154 | 11315.0155 | 12086.7816 | agent_replay | 540 | model.request |
| A-direct | 2.4400 | 0.1628 | qwen-3.8-27b | 0 | 0.1125 | 0.1741 | 0.3761 | 1.6280 | agent_replay | 540 | prompt.assemble |
| A-rag | 28236.5140 | 18526.9052 | qwen-3.8-27b | 6 | 18530.2810 | 22894.2717 | 23792.8170 | 26387.7896 | agent_replay | 470 | agent.loop |
| A-rag | 16.5860 | 5.6808 | qwen-3.8-27b | 0 | 5.2235 | 7.3670 | 8.1367 | 11.8178 | agent_replay | 464 | decision.persist |
| A-rag | 1.5210 | 0.1334 | qwen-3.8-27b | 6 | 0.1200 | 0.1561 | 0.1918 | 0.3121 | agent_replay | 470 | decision.validate |
| A-rag | 15.3430 | 9.6812 | qwen-3.8-27b | 0 | 9.7870 | 11.0706 | 11.7703 | 13.4326 | agent_replay | 470 | embedding.request |
| A-rag | 22196.9080 | 9063.4491 | qwen-3.8-27b | 0 | 8061.3595 | 14821.0133 | 16013.2511 | 18514.0734 | agent_replay | 940 | model.operation |
| A-rag | 22196.1710 | 9062.8617 | qwen-3.8-27b | 0 | 8060.8790 | 14820.4514 | 16012.6599 | 18513.5756 | agent_replay | 940 | model.request |
| A-rag | 2.4590 | 0.1431 | qwen-3.8-27b | 0 | 0.1290 | 0.1670 | 0.1981 | 0.3783 | agent_replay | 940 | prompt.assemble |
| A-rag | 215.0680 | 97.3410 | qwen-3.8-27b | 0 | 93.7690 | 108.1387 | 115.6193 | 137.1727 | agent_replay | 470 | retrieval.operation |
| A-rag | 216.3210 | 97.9558 | qwen-3.8-27b | 0 | 94.3255 | 108.9050 | 116.2017 | 138.1950 | agent_replay | 470 | tool.operation |
| A-tools | 37820.9050 | 18112.9614 | qwen-3.8-27b | 51 | 17514.6495 | 26320.8052 | 27498.3867 | 30175.1062 | agent_replay | 540 | agent.loop |
| A-tools | 24.9400 | 5.7918 | qwen-3.8-27b | 0 | 4.9800 | 7.4850 | 9.9598 | 16.1480 | agent_replay | 489 | decision.persist |
| A-tools | 2.4620 | 0.1447 | qwen-3.8-27b | 41 | 0.1210 | 0.1603 | 0.2335 | 0.6195 | agent_replay | 530 | decision.validate |
| A-tools | 26.1290 | 10.2640 | qwen-3.8-27b | 0 | 9.9665 | 11.6678 | 12.6380 | 22.1753 | agent_replay | 134 | embedding.request |
| A-tools | 28227.6770 | 8008.8516 | qwen-3.8-27b | 10 | 5118.4700 | 16269.8570 | 17720.5636 | 20345.2984 | agent_replay | 1195 | model.operation |
| A-tools | 28227.1660 | 8008.2269 | qwen-3.8-27b | 10 | 5117.6820 | 16269.2010 | 17720.0321 | 20344.7729 | agent_replay | 1195 | model.request |
| A-tools | 2.4600 | 0.1507 | qwen-3.8-27b | 0 | 0.1360 | 0.1746 | 0.2416 | 0.4607 | agent_replay | 1195 | prompt.assemble |
| A-tools | 142.0200 | 99.2311 | qwen-3.8-27b | 0 | 95.9620 | 109.4544 | 128.8015 | 140.5251 | agent_replay | 134 | retrieval.operation |
| A-tools | 142.5940 | 21.3423 | qwen-3.8-27b | 0 | 1.1390 | 96.5462 | 101.2028 | 129.3026 | agent_replay | 655 | tool.operation |
| A-tools | 11673.6890 | 7996.5769 | qwen-3.8-27b | 4 | 7552.7480 | 10071.3715 | 10621.0435 | 11285.3886 | control_batching_sequential | 48 | agent.loop |
| A-tools | 8.6310 | 6.0700 | qwen-3.8-27b | 0 | 5.8875 | 7.3510 | 7.8209 | 8.3713 | control_batching_sequential | 44 | decision.persist |
| A-tools | 0.5980 | 0.1655 | qwen-3.8-27b | 4 | 0.1490 | 0.1785 | 0.2189 | 0.4852 | control_batching_sequential | 48 | decision.validate |
| A-tools | 10.6840 | 9.8004 | qwen-3.8-27b | 0 | 9.7820 | 10.3676 | 10.5258 | 10.6524 | control_batching_sequential | 8 | embedding.request |
| A-tools | 7755.4600 | 3459.7044 | qwen-3.8-27b | 0 | 1925.5000 | 6150.9965 | 6685.8728 | 7536.9941 | control_batching_sequential | 106 | model.operation |
| A-tools | 7754.7770 | 3458.9012 | qwen-3.8-27b | 0 | 1924.6785 | 6149.9730 | 6685.1130 | 7536.3138 | control_batching_sequential | 106 | model.request |
| A-tools | 0.5910 | 0.1866 | qwen-3.8-27b | 0 | 0.1625 | 0.2170 | 0.4157 | 0.5156 | control_batching_sequential | 106 | prompt.assemble |
| A-tools | 137.3300 | 106.2261 | qwen-3.8-27b | 0 | 95.7585 | 135.4750 | 136.4025 | 137.1445 | control_batching_sequential | 8 | retrieval.operation |
| A-tools | 137.9580 | 17.9949 | qwen-3.8-27b | 0 | 2.0980 | 92.3956 | 100.0836 | 136.4982 | control_batching_sequential | 58 | tool.operation |
| A-tools | 31023.5410 | 20769.6956 | qwen-3.8-27b | 4 | 19492.4835 | 26272.2405 | 28816.0842 | 30601.5310 | control_error_number_mask | 96 | agent.loop |
| A-tools | 17.3550 | 6.4370 | qwen-3.8-27b | 0 | 5.4370 | 9.8810 | 15.0039 | 17.1603 | control_error_number_mask | 92 | decision.persist |
| A-tools | 0.8150 | 0.1479 | qwen-3.8-27b | 3 | 0.1340 | 0.1870 | 0.2367 | 0.3807 | control_error_number_mask | 95 | decision.validate |
| A-tools | 18.3120 | 10.5566 | qwen-3.8-27b | 0 | 10.0850 | 12.1038 | 15.7250 | 17.6627 | control_error_number_mask | 33 | embedding.request |
| A-tools | 20600.5170 | 8650.2351 | qwen-3.8-27b | 1 | 6851.4035 | 15840.2050 | 17710.6983 | 19755.8515 | control_error_number_mask | 226 | model.operation |
| A-tools | 20599.9750 | 8649.1715 | qwen-3.8-27b | 1 | 6850.6650 | 15839.5795 | 17709.9708 | 19754.8662 | control_error_number_mask | 226 | model.request |
| A-tools | 4.6500 | 0.2456 | qwen-3.8-27b | 0 | 0.1470 | 0.2700 | 0.4608 | 2.4697 | control_error_number_mask | 226 | prompt.assemble |
| A-tools | 160.6010 | 102.9404 | qwen-3.8-27b | 0 | 96.0130 | 125.6238 | 159.0530 | 160.5386 | control_error_number_mask | 33 | retrieval.operation |
| A-tools | 163.7160 | 28.5406 | qwen-3.8-27b | 0 | 1.9980 | 97.8612 | 104.1125 | 163.1814 | control_error_number_mask | 130 | tool.operation |
| A-tools | 29599.2090 | 19109.9289 | qwen-3.8-27b | 8 | 18977.3765 | 25241.1885 | 27111.3755 | 29334.3205 | control_shuffled_runbooks | 96 | agent.loop |
| A-tools | 16.5780 | 5.5283 | qwen-3.8-27b | 0 | 4.8785 | 7.6610 | 9.1291 | 14.8319 | control_shuffled_runbooks | 88 | decision.persist |
| A-tools | 0.4040 | 0.1349 | qwen-3.8-27b | 7 | 0.1220 | 0.1626 | 0.2603 | 0.3890 | control_shuffled_runbooks | 95 | decision.validate |
| A-tools | 19428.4160 | 8501.1598 | qwen-3.8-27b | 1 | 6996.7150 | 15160.4280 | 16565.4699 | 18202.7093 | control_shuffled_runbooks | 212 | model.operation |
| A-tools | 19427.9620 | 8499.9579 | qwen-3.8-27b | 1 | 6992.9715 | 15159.8718 | 16564.9489 | 18202.2315 | control_shuffled_runbooks | 212 | model.request |
| A-tools | 6.3590 | 0.2598 | qwen-3.8-27b | 0 | 0.1370 | 0.3258 | 0.7920 | 1.8792 | control_shuffled_runbooks | 212 | prompt.assemble |
| A-tools | 44.1520 | 2.8291 | qwen-3.8-27b | 0 | 1.4465 | 6.3560 | 9.0802 | 14.6131 | control_shuffled_runbooks | 116 | tool.operation |
| A-direct | 2200.1910 | 996.6329 | qwen-smoke | 60 | 1176.7670 | 1664.2274 | 1703.9333 | 1809.6653 | qwen_smoke_replay | 120 | agent.loop |
| A-direct | 13.8150 | 6.7167 | qwen-smoke | 0 | 6.1330 | 9.8575 | 10.1375 | 13.7147 | qwen_smoke_replay | 60 | decision.persist |
| A-direct | 0.9510 | 0.2690 | qwen-smoke | 0 | 0.1720 | 0.5690 | 0.6705 | 0.8831 | qwen_smoke_replay | 60 | decision.validate |
| A-direct | 1628.5010 | 833.5803 | qwen-smoke | 30 | 999.0955 | 1503.0933 | 1524.0060 | 1619.0384 | qwen_smoke_replay | 120 | model.operation |
| A-direct | 1620.6480 | 831.7816 | qwen-smoke | 30 | 994.8985 | 1500.3707 | 1523.1605 | 1612.1448 | qwen_smoke_replay | 120 | model.request |
| A-direct | 4.7390 | 0.3994 | qwen-smoke | 0 | 0.1810 | 0.8040 | 0.9152 | 4.4344 | qwen_smoke_replay | 120 | prompt.assemble |
| A-direct | 3.1820 | 0.9610 | qwen-smoke | 30 | 0.4870 | 2.9072 | 3.0603 | 3.1762 | qwen_smoke_replay | 30 | tool.operation |
| A-rag | 3164.7500 | 1524.5245 | qwen-smoke | 60 | 1479.9015 | 2658.7500 | 2700.6590 | 2794.1313 | qwen_smoke_replay | 120 | agent.loop |
| A-rag | 16.0340 | 6.4027 | qwen-smoke | 0 | 5.6515 | 8.4097 | 11.1605 | 13.4699 | qwen_smoke_replay | 60 | decision.persist |
| A-rag | 0.6290 | 0.1488 | qwen-smoke | 0 | 0.1205 | 0.2139 | 0.2434 | 0.6024 | qwen_smoke_replay | 60 | decision.validate |
| A-rag | 12.8280 | 9.4283 | qwen-smoke | 0 | 9.2285 | 10.9632 | 11.8780 | 12.6085 | qwen_smoke_replay | 60 | embedding.request |
| A-rag | 2066.1270 | 832.7563 | qwen-smoke | 0 | 354.1875 | 1888.7298 | 1944.0587 | 1966.3485 | qwen_smoke_replay | 180 | model.operation |
| A-rag | 2065.5640 | 832.1482 | qwen-smoke | 0 | 352.0830 | 1888.2013 | 1943.4747 | 1965.7575 | qwen_smoke_replay | 180 | model.request |
| A-rag | 0.6160 | 0.1371 | qwen-smoke | 0 | 0.1215 | 0.1763 | 0.2402 | 0.3543 | qwen_smoke_replay | 180 | prompt.assemble |
| A-rag | 161.0090 | 90.0520 | qwen-smoke | 0 | 86.6535 | 93.1363 | 125.6940 | 142.9703 | qwen_smoke_replay | 60 | retrieval.operation |
| A-rag | 162.2150 | 45.5417 | qwen-smoke | 60 | 42.6495 | 90.0726 | 93.5366 | 130.4347 | qwen_smoke_replay | 120 | tool.operation |
| A-tools | 3032.6290 | 1269.1842 | qwen-smoke | 111 | 1099.7530 | 2827.6456 | 2971.9250 | 3017.3741 | qwen_smoke_replay | 120 | agent.loop |
| A-tools | 14.4350 | 5.7100 | qwen-smoke | 0 | 4.4210 | 7.7510 | 11.0930 | 13.7666 | qwen_smoke_replay | 9 | decision.persist |
| A-tools | 0.3340 | 0.1483 | qwen-smoke | 0 | 0.1260 | 0.1844 | 0.2592 | 0.3190 | qwen_smoke_replay | 9 | decision.validate |
| A-tools | 1618.8650 | 337.3279 | qwen-smoke | 21 | 234.5060 | 388.1510 | 1438.9228 | 1533.6861 | qwen_smoke_replay | 307 | model.operation |
| A-tools | 1618.2500 | 336.7250 | qwen-smoke | 21 | 233.9970 | 387.5520 | 1438.2183 | 1533.0409 | qwen_smoke_replay | 307 | model.request |
| A-tools | 0.4500 | 0.1485 | qwen-smoke | 0 | 0.1360 | 0.1922 | 0.2694 | 0.3728 | qwen_smoke_replay | 307 | prompt.assemble |
| A-tools | 45.0130 | 1.1488 | qwen-smoke | 60 | 1.0190 | 1.5346 | 1.7530 | 3.0477 | qwen_smoke_replay | 277 | tool.operation |
| dev-agent-loop-a-direct-v1 | 72.9800 | 72.7790 | qwen-smoke | 2 | 72.7790 | 72.9398 | 72.9599 | 72.9760 | agent_loop_gate | 2 | agent.loop |
| dev-agent-loop-a-direct-v1 | 3.2940 | 3.2155 | qwen-smoke | 2 | 3.2155 | 3.2783 | 3.2861 | 3.2924 | agent_loop_gate | 2 | model.operation |
| dev-agent-loop-a-direct-v1 | 2.5620 | 2.5320 | qwen-smoke | 2 | 2.5320 | 2.5560 | 2.5590 | 2.5614 | agent_loop_gate | 2 | model.request |
| dev-agent-loop-a-direct-v1 | 0.1450 | 0.1435 | qwen-smoke | 0 | 0.1435 | 0.1447 | 0.1448 | 0.1450 | agent_loop_gate | 2 | prompt.assemble |
| dev-agent-loop-a-tools-v1 | 583.4710 | 323.5867 | qwen-smoke | 8 | 257.1520 | 521.2900 | 551.8205 | 577.1409 | agent_loop_gate | 15 | agent.loop |
| dev-agent-loop-a-tools-v1 | 15.1900 | 8.4783 | qwen-smoke | 0 | 7.3290 | 12.6886 | 13.9393 | 14.9399 | agent_loop_gate | 7 | decision.persist |
| dev-agent-loop-a-tools-v1 | 2.4280 | 0.5333 | qwen-smoke | 0 | 0.2210 | 1.1602 | 1.7941 | 2.3012 | agent_loop_gate | 7 | decision.validate |
| dev-agent-loop-a-tools-v1 | 10.2940 | 9.8974 | qwen-smoke | 0 | 9.8840 | 10.2384 | 10.2662 | 10.2884 | agent_loop_gate | 5 | embedding.request |
| dev-agent-loop-a-tools-v1 | 8.9090 | 3.3484 | qwen-smoke | 0 | 2.6750 | 5.9218 | 8.6321 | 8.8237 | agent_loop_gate | 35 | model.operation |
| dev-agent-loop-a-tools-v1 | 7.9170 | 2.6701 | qwen-smoke | 0 | 2.0520 | 4.9752 | 7.6584 | 7.8310 | agent_loop_gate | 35 | model.request |
| dev-agent-loop-a-tools-v1 | 0.3310 | 0.1844 | qwen-smoke | 0 | 0.1660 | 0.2546 | 0.3024 | 0.3279 | agent_loop_gate | 35 | prompt.assemble |
| dev-agent-loop-a-tools-v1 | 147.5140 | 104.8944 | qwen-smoke | 0 | 113.9290 | 135.6224 | 141.5682 | 146.3248 | agent_loop_gate | 5 | retrieval.operation |
| dev-agent-loop-a-tools-v1 | 149.1590 | 19.8222 | qwen-smoke | 6 | 0.7180 | 86.7588 | 117.9391 | 141.0957 | agent_loop_gate | 28 | tool.operation |
| dev-model-client-gate-v1 | 43.7400 | 9.8935 | qwen-smoke | 14 | 3.3220 | 28.7419 | 43.5241 | 43.6968 | fake_gateway_gate | 18 | model.operation |
| dev-model-client-gate-v1 | 42.9520 | 7.9428 | qwen-smoke | 16 | 1.9445 | 22.8365 | 42.8276 | 42.9271 | fake_gateway_gate | 20 | model.request |

## vLLM and GPU residency samples

| cancellation_delta | first_sample_at_utc | generation_token_delta | last_sample_at_utc | max_generation_tokens_per_second | max_gpu_memory_used_mib | max_gpu_power_draw_w | max_gpu_temperature_c | max_gpu_utilization_pct | max_kv_cache_usage_ratio | max_prompt_tokens_per_second | max_running_requests | max_swapped_requests | max_waiting_requests | mean_generation_tokens_per_second | mean_gpu_utilization_pct | mean_prompt_tokens_per_second | mean_running_requests | model_profile_id | phase | preemption_delta | prefix_cache_hit_ratio | prompt_token_delta | request_error_delta | sample_count | residency_window_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | 2026-08-23T09:50:40.099Z | 0 | 2026-08-23T10:10:04.288Z | — | 81952 | 89.7400 | 33 | 2 | — | — | 0 | — | 0 | — | 0.0881 | — | 0.0000 | qwen-smoke | retrieval-evaluation | 0 | — | 0 | — | 159 | 1164.1890 |
| — | 2026-08-23T10:13:37.068Z | 43776 | 2026-08-23T10:45:14.614Z | — | 81956 | 379.3700 | 55 | 100 | — | — | 12 | — | 0 | — | 5.2054 | — | 0.1860 | qwen-smoke | qwen-smoke-replay | 0 | 0.7247 | 938786 | — | 258 | 1897.5460 |
| — | 2026-08-23T11:39:06.616Z | 908467 | 2026-08-23T15:37:39.296Z | — | 82752 | 493.0800 | 74 | 100 | — | — | 16 | — | 0 | — | 65.5871 | — | 5.2783 | muse-glimmer-30b | muse-glimmer-30b-residency | 0 | 0.7413 | 6924913 | — | 1901 | 14312.6800 |
| — | 2026-08-23T16:14:57.431Z | 364367 | 2026-08-23T17:46:40.002Z | — | 86190 | 599.3300 | 81 | 100 | — | — | 16 | — | 2 | — | 37.5549 | — | 4.4231 | gemma-4-31b | gemma-4-31b-residency | 1 | 0.4582 | 4421103 | — | 728 | 5502.5710 |
| — | 2026-08-23T18:01:52.378Z | 3386 | 2026-08-23T18:09:26.840Z | — | 81372 | 408.2200 | 54 | 100 | — | — | 4 | — | 0 | — | 22.9508 | — | 0.3279 | olmo-3.1-32b-instruct | olmo-3.1-32b-instruct-residency | 0 | 0.9206 | 20838 | — | 61 | 454.4620 |
| — | 2026-08-23T18:20:56.357Z | 376467 | 2026-08-23T20:05:45.951Z | — | 81646 | 609.2800 | 80 | 100 | — | — | 16 | — | 5 | — | 38.6631 | — | 4.0903 | qwen-3.8-27b | qwen-3.8-27b-residency | 0 | — | 6019861 | — | 831 | 6289.5940 |
| — | 2026-08-23T20:09:46.666Z | 0 | 2026-08-23T20:09:54.502Z | — | 81646 | 87.0400 | 32 | 0 | — | — | 0 | — | 0 | — | 0.0000 | — | 0.0000 | qwen-3.8-27b | graceful-stop-gate | 0 | — | 0 | — | 2 | 7.8360 |

## SQL queue

| first_sample_at_utc | last_sample_at_utc | max_lease_expired_count | max_leased_count | max_oldest_pending_age_ms | max_pending_count | max_retryable_count | max_worker_count | mean_leased_count | mean_pending_count | observed_arrivals | observed_completions | sample_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-23T04:15:04.086Z | 2026-08-23T20:09:54.502Z | 0 | 16 | 3562963 | 1370 | 1 | 16 | 3.1200 | 101.7833 | 0 | 0 | 5501 |

## SQL resource sampling

| first_sample_at_utc | last_sample_at_utc | max_blocked_request_count | max_data_file_bytes | max_log_file_bytes | max_log_used_pct | max_pending_io_count | max_process_cpu_pct | max_process_memory_kb | max_request_count | max_runnable_task_count | max_target_memory_kb | mean_process_cpu_pct | sample_count | unavailable_sample_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-23T04:15:04.086Z | 2026-08-23T20:09:54.502Z | 14 | 5788139520 | 822083584 | — | 4 | — | 4280320 | 88 | 3 | 7198840 | — | 5501 | 0 |

## Query Store materialization

_No retained rows._

## Metric availability and non-inference policy

| metric_family | status | provenance | note |
|---|---|---|---|
| agent wall clock | available | client monotonic timestamps | p50/p90/p95/p99 retained per profile/arm |
| pipeline phases | available | closed telemetry spans | prompt, model, retrieval, tool, validation, persistence, and loop spans |
| model request total | available | client monotonic timestamps | headers wait, body read, and parse are separately retained |
| true model TTFT | unavailable | endpoint limitation | non-streaming transport exposes no first-token event; headers wait is not relabeled TTFT |
| inter-token latency | unavailable | endpoint limitation | no streaming token timestamps |
| vLLM request/token counters | available | raw /metrics snapshots | running/waiting requests, token deltas, preemptions, and prefix-cache counters where exposed |
| vLLM throughput gauges | unavailable | vLLM 0.27.1 metric surface | prompt/generation throughput gauges were absent; counter deltas remain retained |
| KV-cache occupancy | unavailable | vLLM 0.27.1 metric surface | metric was absent for these service configurations |
| GPU | available | nvidia-smi samples | utilization, memory, power, temperature, clocks, and process snapshots retained |
| queue | available | SQL queue sampler | depth, age, lease, worker, and terminal-state samples retained |
| SQL resources | partial | SQL DMVs | memory/request/blocking/I/O/file sizes available; process CPU and log-used percent unavailable |
| Query Store intervals | unavailable | Tier-2 materialization deferred | settings/history are retained, but no interval rows were materialized for Tier 1 |

Missing service metrics remain unavailable; the report never derives TTFT or inter-token latency from total request time. Residency windows include governed warm-up, calibration, primary replay, and controls and therefore are not model-only benchmark times.
