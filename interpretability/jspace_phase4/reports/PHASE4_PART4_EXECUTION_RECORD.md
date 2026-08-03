# Phase 4.4 execution record

**DEVELOPMENT-ONLY DECISION BLOCK — NOT A FREEZE RECORD**

## Launch and pre-commitment

The isolated `interp_jspace_phase4_4` branch was created at 2026-08-03 02:17:30 UTC from clean, freshly pulled `interp_jspace_part2` commit `901fb4fc7578a913088c7947a2e6240f7fc45aeb`. The required two-paragraph Terminal-B success-state commitment was committed and pushed at `6f23a29896e61cc367ed884a2d840f0e08857f40` before any A1000 functional outcome was opened.

## M0 boundary

Two bootstrap test invocations failed collection: the literal subdirectory command did not expose the sibling Phase 3 and part2 packages, and the first attempted `PYTHONPATH` correction was rooted one directory too high. The corrected invocation used the frozen wrapper's three explicit package roots and passed 279/279 tests; no source was changed to obtain that result. Exact FLA 0.5.2 distributions and `jlens` 0.1.0 at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` were installed without upgrading the already matching Torch, Transformers, or Triton versions.

The pinned Qwen snapshot was downloaded directly from Hugging Face at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. All 23 files, 55,563,006,400 weight bytes, inventory hash `c799b1249667b815a6f78aa85963f30f16d7146a9e04ea7aa0c8f69ef25e8644`, and 48 fused linear-attention module bindings passed. The external published lens was materialized at revision `a4114d7752d11eb546e6cf372213d7e75526d3a1` and matched SHA-256 `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1`.

The fresh-VM tensor pass verified the registered A1000 lens, checkpoint, and header hashes; all 63 source layers; target layer 63; 5,120-by-5,120 fp32/fp16 tensors; full finiteness; and bit-exact quantized checkpoint-mean equality. The whole-registry pass verified 230 of 232 live output references with no unexpected failure or path-pin conflict. The missing historical `state.json` and `capacity_reconstructions_a120.pt` are the only failures and remain explicitly red.

The sealed wrapper was then stopped at the already registered structural boundary without opening the functional gate. It freshly reverified the A1000 binding and structural event and recreated exact local backups for all three fit outputs and all six structural outputs. Their backup-manifest hashes are `cdaae0129c6f2ff55650481bf74713ccfb0af61039ceb4bf8d0dcc0a6ab5397c` and `5bff1706c7684c38d8f103a2d725059a9a1bca8f0b8aaad3a528047390c57e09`, respectively. The repository remained clean and the functional event remained the first incomplete queue stage.

## Functional observer hard stop and repair

The functional stage subsequently hard-stopped before its first A500 row because the selection-margin observer used `topk(32)[:10]` as an assumed replay of the inherited `topk(10)`. The real boundary contained an exact score tie, so Torch returned different but equally scoring IDs for the two requested k values. A diagnostic replay proved that a separate observer-side `topk(10)` exactly matched the unchanged parent intervention. No functional evidence event was registered.

The repair separates top-32 diagnostics from the exact top-10 replay and retains hard ID-and-score equality against the parent. The successor validator admits different top-32-prefix IDs only at an exact k/k+1 score tie and otherwise retains all original checks. The incompatible 57 MB partial state was preserved on Drive and locally under the paths and hashes recorded in `PHASE4_PART4_FUNCTIONAL_INSTRUMENTATION_INCIDENT.md`; it was not migrated across code commits. The focused suite passed 27/27 and the full Phase 4 suite passed 282/282 after the repair.

## Registered functional and margin transactions

The repaired functional stage restarted from scratch and registered
`p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1`. Its event was
banked and pushed at `603bbcec271c8a5a0b7ce6f519a4845988e959b3`; all 18 registered
outputs were copied into a 303,287,915-byte local backup whose manifest has
SHA-256 `988844d347c456919864bdffd0d72c35eb71622fbab653674db0e065aeba35e7`.
The structural q50/q05 gates reverified at 0.998702/0.998122. A500–A1000
median selected-ID Jaccard (0.538462), normalized projector overlap
(0.709818), and bridge-rescue difference (-0.294028 nat) fail their frozen
gates. Occupancy, corrected capacity, span-safe specificity, tail rate, G4,
and bridge preference pass. The functional producer therefore emits
provisional branch candidate Q-L4 but explicitly leaves the branch pending
the separate margin and influence inputs and nominates no lens.

The next transaction registered
`p4-qwen-selection-margin-a500-a1000-dev-v1` and was banked and pushed at
`0ce3519b3abfc67ccfb3335e6e371b8a689a0fb5`. All 34 outputs are present in a
14,411,038-byte local backup whose manifest has SHA-256
`12e76809602688529a8fd461ee3f3d3fe4f3050bf8cef46600568114c4b1f451`.
The audit retained all 17,381 functional positions: 15,536 near-tie, 1,845
stable-core, and zero rank-deficient. It exactly reconstructs the captured
top-k intervention and registered geometry. The blinded lexical sheet marks
52,261 rows for optional behavior-blind manual review; no review was performed
because that sheet is nondecisional. Q-L4 remains a candidate, not a canonical
event.

The registered `p4f26` and `p4f27` PNG/PDF pairs were copied byte-for-byte
from Drive into the report figure tree and visually checked. Their registered
hashes remain unchanged.

## Prompt-323 runtime-identity hard stop

The queue then opened the frozen prompt-323 influence stage from clean commit
`0ce3519b3abfc67ccfb3335e6e371b8a689a0fb5`. Its first full backward pass
returned `max ||J|| / sqrt(d) = 181.826310` against the frozen fit-log value
173.345 and prospective absolute tolerance 0.5. The producer stopped before
writing a contribution, layer metric, figure, result envelope, or registry
event. Consequently the canonical producer never opened.

A second clean prompt-323 process returned 181.785516. A replay that evaluated
prompt 322 immediately before prompt 323 returned 59.545969 versus historical
52.150 and left prompt 323 at 181.854247. Two independent current-runtime
evaluations of the already registered prompt-112 control returned 55.544060
and 55.587600, whereas its fit-log value is 159.952 and its registered clean
recompute is 160.070954. Each current pair agrees within 0.044, all values are
finite, and all maxima are at layer 0. The mismatch is therefore stable on the
current VM and broader than prompt 323 or log rounding.

The historical and current records match nominal GPU, driver, CUDA, Torch,
Transformers, Triton, FLA, fused bindings, exact model/corpus, and `jlens`
revision. The historical record did not preserve installed-distribution
content, wheel/build, or compiled Triton/FLA cache hashes. The current
distribution contents are now fully inventoried. No single unrecorded surface
is asserted as the cause.

Both failed states have `contribution: null`, zero completed layers, and the
same 1,146-byte SHA-256 `4eef3124...239eb3`. Each is preserved under a distinct
Drive and local unregistered-backup name. The queue lock is free, the fresh
canonical output path is absent, and no GPU process remains. The exact record
is `PHASE4_PART4_PROMPT323_RUNTIME_BLOCK.md` plus its machine-readable JSON.

## Blocked transactional order

1. Obtain independent scientific/governance review; do not rerun blindly or
   alter the frozen tolerance.
2. Either reconstruct a historically content-pinned backward runtime or
   approve a prospective runtime-contract amendment before any fresh attempt.
3. Only a passing fresh influence transaction may be registered, backed up,
   committed, and pushed.
4. Only after that live event exists may the unchanged mechanical canonical
   producer open.
5. M2 side admission, M3, and M4 remain closed until the registered canonical
   branch actually licenses them.

No confirmatory or replication intervention outcome exists. No A2000, new bank, new model, new endpoint, or SESOI change is authorized.
