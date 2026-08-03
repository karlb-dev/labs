# Phase 4.4 prompt-323 runtime-identity block

**DEVELOPMENT EXECUTION BOUNDARY — NO INFLUENCE OR CANONICAL EVENT**

## Hard stop

The sealed queue registered and backed up the A500–A1000 functional event and
its selection-margin audit, then opened the frozen prompt-323 influence stage
from clean commit `0ce3519b3abfc67ccfb3335e6e371b8a689a0fb5`. Before writing a prompt
contribution or any layer result, the producer recomputed
`max ||J|| / sqrt(d)` as 181.826310. The frozen fit-log value is 173.345 and
the prospective absolute tolerance is 0.5, so the producer raised
`recomputed prompt-323 norm exceeds frozen tolerance`. No influence event was
registered and the queue did not open the canonical decision.

Two independent clean-process prompt-323 evaluations returned 181.826310 and
181.785516, only 0.040794 apart but 8.481310 and 8.440516 above the frozen
value. Replaying prompt 322 immediately before prompt 323 produced 59.545969
versus its historical 52.150 and left prompt 323 at 181.854247. The preceding
prompt therefore does not restore the historical computation.

## Independent registered control

Prompt 112 is a stronger control because its earlier influence event contains
a successful clean recomputation of 160.070954 against the fit-log value
159.952. On this VM, two independent clean processes instead returned
55.544060 and 55.587600. They agree to 0.043540 while missing the registered
recompute by 104.526894 and 104.483354. All four prompt-323 evaluations and
both prompt-112 controls were finite, used sequence length 128 and 111 valid
positions, and reached their maximum at source layer 0.

This rules out a prompt-323-only problem, decimal log rounding, and a one-off
current-process excursion. It establishes a stable incompatibility between
the current backward computation and the computation that produced the fit
logs and registered prompt-112 audit.

## Runtime identity boundary

The surviving contracts agree on the nominal GPU and software surface:
NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, driver
580.82.07, CUDA 12.8, Torch 2.11.0+cu128, Transformers 5.13.1, Triton 3.6.0,
FLA 0.5.2, the same 48 fused Qwen bindings, exact Qwen revision
`6a9e13bd...`, and exact `jlens` commit `581d398...`. Model, corpus, lens,
fit-config, and producer hashes also pass.

Those contracts did not preserve hashes of the historical installed
distribution contents, wheels/builds, or compiled Triton/FLA cache. The
current non-pyc content inventories are now recorded in
`PHASE4_PART4_PROMPT323_RUNTIME_IDENTITY.json`, including file counts, byte
counts, distribution `METADATA`/`RECORD` hashes, and whole-content inventory
hashes for Torch, Transformers, Triton, `fla-core`, and
`flash-linear-attention`. A build/content difference, compiled-kernel/cache
difference, or another unrecorded runtime state remains possible. That list
is an inference boundary, not a selected root cause.

## Governance disposition

The functional and margin events mechanically emit provisional branch
candidate Q-L4. They do not constitute a canonical Q-L event: the canonical
producer requires a live prompt-323 influence event and admits only its frozen
influence decisions. It would therefore be invalid to widen the 0.5
tolerance, omit the influence input, manufacture a contribution from the fit
log, or register Q-L4 directly.

The GPU-stop rule applies. Both failed states contain `contribution: null` and
zero completed layers; their identical 1,146-byte state has SHA-256
`4eef3124...239eb3`. They are preserved under separately named Drive and
local unregistered-backup directories. The canonical output path is empty,
the queue lock is free, and no model process remains. M2 side admission and
all M3/M4 work stay closed because the registered canonical event does not
exist.

The next lawful action requires independent scientific/governance review:
either reconstruct a historically content-pinned backward runtime and repeat
the unchanged influence gate, or approve a prospective runtime-contract
amendment before any fresh attempt. The preserved null attempts can never be
promoted to evidence. Exact paths, hashes, observations, flags, and current
distribution inventories are in the companion JSON; diagnostic scripts and
logs are archived at Drive
`diagnostics/prompt323_runtime_identity_20260803/`.
