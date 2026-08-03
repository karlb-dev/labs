# Phase 4.4 execution record

**DEVELOPMENT-ONLY DECISION BLOCK — NOT A FREEZE RECORD**

## Launch and pre-commitment

The isolated `interp_jspace_phase4_4` branch was created at 2026-08-03 02:17:30 UTC from clean, freshly pulled `interp_jspace_part2` commit `901fb4fc7578a913088c7947a2e6240f7fc45aeb`. The required two-paragraph Terminal-B success-state commitment was committed and pushed at `6f23a29896e61cc367ed884a2d840f0e08857f40` before any A1000 functional outcome was opened.

## M0 boundary

Two bootstrap test invocations failed collection: the literal subdirectory command did not expose the sibling Phase 3 and part2 packages, and the first attempted `PYTHONPATH` correction was rooted one directory too high. The corrected invocation used the frozen wrapper's three explicit package roots and passed 279/279 tests; no source was changed to obtain that result. Exact FLA 0.5.2 distributions and `jlens` 0.1.0 at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` were installed without upgrading the already matching Torch, Transformers, or Triton versions.

The pinned Qwen snapshot was downloaded directly from Hugging Face at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. All 23 files, 55,563,006,400 weight bytes, inventory hash `c799b1249667b815a6f78aa85963f30f16d7146a9e04ea7aa0c8f69ef25e8644`, and 48 fused linear-attention module bindings passed. The external published lens was materialized at revision `a4114d7752d11eb546e6cf372213d7e75526d3a1` and matched SHA-256 `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1`.

The fresh-VM tensor pass verified the registered A1000 lens, checkpoint, and header hashes; all 63 source layers; target layer 63; 5,120-by-5,120 fp32/fp16 tensors; full finiteness; and bit-exact quantized checkpoint-mean equality. The whole-registry pass verified 230 of 232 live output references with no unexpected failure or path-pin conflict. The missing historical `state.json` and `capacity_reconstructions_a120.pt` are the only failures and remain explicitly red.

The sealed wrapper was then stopped at the already registered structural boundary without opening the functional gate. It freshly reverified the A1000 binding and structural event and recreated exact local backups for all three fit outputs and all six structural outputs. Their backup-manifest hashes are `cdaae0129c6f2ff55650481bf74713ccfb0af61039ceb4bf8d0dcc0a6ab5397c` and `5bff1706c7684c38d8f103a2d725059a9a1bca8f0b8aaad3a528047390c57e09`, respectively. The repository remained clean and the functional event remained the first incomplete queue stage.

## Pending transactional order

1. Open the functional gate, then bank, back up, commit, and push it.
2. Run selection margin, prompt-323 influence, and the mechanical Q-L decision in order with the same transaction boundary.
3. Admit the study-1 side releases and reproduce the 16/20 P4-P3 block.
4. Run only Q-L-licensed M3/M4 work; no consumed outcome runs without a real independent review.
5. Close durability, review packets, paper artifacts, and the honest freeze-candidate boundary.

No confirmatory or replication intervention outcome exists. No A2000, new bank, new model, new endpoint, or SESOI change is authorized.
