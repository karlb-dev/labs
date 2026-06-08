# Lab 8: Chunked And Pipelined Ring

Goal: turn a correct ring into a performance experiment without pretending that
chunking automatically means overlap.

Lab 8 is the first lab where performance engineering becomes the main character.
Labs 1-2 taught one-hop movement and dependency chains. Labs 5-7 used those
pieces to build all-gather, reduce-scatter, and all-reduce. Now we take the same
ring idea and ask a sharper question:

```text
How does the schedule change when the payload is split into chunks?
```

Lab 8 is about **building a real fused collective kernel** and learning to
characterize it. It ships three implementations of the same token-ring reduction
so students can see the schedule, build the fused version that actually overlaps
communication with work, and measure honestly where that overlap helps:

```text
serialized teaching path  ->  fused Pallas double-buffered ring  ->  XLA reference
```

The serialized path is the clarity-first microscope: chunking without pipelining.
The double-buffered path is the real thing — one fused Pallas TPU kernel that
keeps an async remote DMA in flight while it accumulates locally, with HBM buffer
slots and a capacity semaphore. The XLA path is a reference point.

This is a *systems-skills* lab, not a "beat the compiler" contest. The goal is to
write a correct fused double-buffered ring with real RDMA overlap and then read
its trace and byte model to explain exactly what the overlap buys — and what it
does not. Matching XLA on large transfers is explicitly **not** a goal here (see
"How fast is it, really?" below). 🧪

## Implemented Happy Path

Run (a single sweep compares all four):

```bash
python collective_bench.py --lab lab8
```

Implemented operations:

- `pmap_token_ring`: dependency-chain reference from Lab 2.
- `pallas_chunked_token_ring`: **serialized** chunked ring built from Lab 1 hops
  (the clarity-first teaching baseline; one remote-DMA hop per chunk/hop pair).
- `pallas_db_token_ring`: **fused custom double-buffered** ring. One Pallas TPU
  kernel uses async remote DMA into HBM buffer slots, capacity semaphores to
  prevent run-ahead, and an inner HBM↔VMEM accumulation pipeline that runs while
  the remote copy is in flight. This is where the real overlap lives.
- `xla_token_ring`: an **XLA reference** (`lax.psum` for full rings,
  `lax.ppermute` for partial-hop experiments).
- `lab8_chunked_pipeline_spec`: source-history, byte-model, collective-ID, and
  buffer-slot artifact.

## How Fast Is It, Really?

Measured on the 4-chip v5e (bf16, 4 chunks, buffer_count=2; median latency, lower
is better):

| payload | pmap_token_ring | serialized | pallas_db | xla |
|---:|---:|---:|---:|---:|
| 64 KiB | 547 us | 230 us | 214 us | 206 us |
| 1 MiB | 766 us | 300 us | 282 us | 232 us |
| 4 MiB | 1375 us | 523 us | 511 us | 344 us |

Read these honestly — two trends, and they *are* the lesson:

1. **The fused kernel's edge over the serialized path shrinks as payloads grow**
   (≈7% at 64 KiB → ≈2% at 4 MiB). The overlap mostly hides *fixed per-phase
   overhead* — it folds `chunks × hops` separately-dispatched Lab 1 calls into one
   fused kernel with `hops + 1` steps. That overhead matters at small/medium
   payloads; at 4 MiB the run is bandwidth-bound and the cheap local accumulation
   it overlaps is no longer the bottleneck, so the win is small. That is expected,
   not a disappointment.

2. **XLA pulls further ahead as payloads grow**, and the reason is algorithmic,
   not overlap quality. This whole-token ring moves `hops · B = 3·B` bytes per
   device; an optimal all-reduce (what `lax.psum` does) moves only `~1.5·B` by
   sending one *chunk* per hop. XLA wins large transfers because it moves about
   **half the bytes** — a property of the algorithm, not the kernel. (Per byte
   actually moved, the fused kernel is competitive.)

So the takeaway is not "the custom kernel is fast." It is: *double buffering buys
overhead hiding, biggest where fixed costs dominate; closing the gap to a tuned
collective on large transfers needs a lower-byte algorithm, not better
overlap.* Making the ring bandwidth-optimal (chunk-per-hop reduce-scatter +
all-gather) is a natural follow-on lab, not this one.

## Chunking Versus Pipelining

Chunking cuts a payload into pieces.

Pipelining keeps multiple pieces in flight.

They are cousins, not twins. The `serialized` path is chunked but each chunk
completes its whole ring before the next starts — chunking without pipelining.
The `pallas_db` path is the pipelined version: inside one fused kernel it starts
the remote DMA, then does useful accumulation **while that copy is in flight**.

`--lab8-buffer-count` reflects this split. For the serialized path it is only a
planning artifact. For `pallas_db` it allocates real HBM scratch slots
(`working_slot = step % buffer_count`, `receiving_slot = (step+1) % buffer_count`)
and a capacity semaphore guards slot reuse, so it must be `>= 2`.

Useful question for students:

```text
Did the benchmark prove overlap, or did the code merely configure a buffer count?
```

The only acceptable answer comes from a trace — and for `pallas_db` the trace
shows the remote-copy start before the accumulation pipeline and the wait after.

## Mental Model

Each device starts with a multi-chunk tile:

```text
x: [device, chunk, row, col]
```

The lab uses a visible coordinate pattern:

```text
x[source, chunk, row, col]
  = 10 * source + chunk + 0.25 * (row mod 8) + 0.03125 * (col mod 16)
```

The first element is hand-checkable:

```text
x[source, chunk, 0, 0] = 10 * source + chunk
```

For a four-device full ring, every receiver sees sources `0, 1, 2, 3` exactly
once. Therefore, at `[row=0, col=0]`:

```text
chunk 0 expected value = 10 * (0 + 1 + 2 + 3) + 4 * 0 = 60
chunk 1 expected value = 10 * (0 + 1 + 2 + 3) + 4 * 1 = 64
chunk 2 expected value = 10 * (0 + 1 + 2 + 3) + 4 * 2 = 68
chunk 3 expected value = 10 * (0 + 1 + 2 + 3) + 4 * 3 = 72
```

The full tile is checked, not only `[:, :, 0, 0]`. That matters because this lab
is preparing students to reason about chunk boundaries and buffer slots. A one
scalar check is a peephole; a full-tile check opens the door.

## Direction Convention

Direction is from the sender's point of view, matching Lab 1:

```text
right: source i -> destination i + 1 mod N
left:  source i -> destination i - 1 mod N
```

For a right-moving ring, receiver `r` sees:

```text
[r, r - 1, r - 2, ...] mod N
```

For a left-moving ring, receiver `r` sees:

```text
[r, r + 1, r + 2, ...] mod N
```

For a four-device right-moving full ring, the source-history table is:

| receiver | sources seen by arrival order |
|---:|---|
| 0 | `[0, 3, 2, 1]` |
| 1 | `[1, 0, 3, 2]` |
| 2 | `[2, 1, 0, 3]` |
| 3 | `[3, 2, 1, 0]` |

The sum is the same for every receiver in a full ring, but the arrival order is
not. That difference matters once chunks overlap and buffer slots are reused.

## Byte Model

Let:

```text
N = number of devices
C = number of chunks
H = number of hops
B = full payload bytes per device
b = chunk payload bytes per device, roughly B / C
```

The serialized implementation sends:

```text
send bytes per device    = H * B
receive bytes per device = H * B
remote-copy phases/device = C * H
```

Chunking does **not** reduce the payload volume of the ring. If the total input
is 4 MiB per device, the ring still moves 4 MiB per hop per device. What changes
is the size of each remote-DMA operation and the number of phases.

That tradeoff has teeth:

```text
large chunks -> fewer phases, less synchronization overhead, less pipeline flexibility
small chunks -> more phases, more synchronization overhead, more overlap opportunity later
```

In this lab, smaller chunks can easily be slower. That is not a failure. That is
the measurement telling the truth with a tiny brass trumpet.

## Collective-ID Schedule

Every chunk/hop remote-copy phase gets a unique collective ID:

```text
collective_id = base_collective_id + chunk_idx * max(1, hops) + hop
```

For `base = 100`, `chunks = 4`, and `hops = 3`:

| chunk | hop | collective_id |
|---:|---:|---:|
| 0 | 0 | 100 |
| 0 | 1 | 101 |
| 0 | 2 | 102 |
| 1 | 0 | 103 |
| 1 | 1 | 104 |
| 1 | 2 | 105 |
| 2 | 0 | 106 |
| 2 | 1 | 107 |
| 2 | 2 | 108 |
| 3 | 0 | 109 |
| 3 | 1 | 110 |
| 3 | 2 | 111 |

This is the Lab 4 discipline carried forward: different communication phases
should not casually alias semaphore state through a reused collective ID.

## Buffer-Slot Ownership

For the serialized path `--lab8-buffer-count` is only a planning artifact. For
the fused `pallas_db` path it allocates **real** HBM scratch slots that the ring
rotates through, one step at a time:

```text
working_slot   = step % buffer_count   # the token accumulated this step
receiving_slot = (step + 1) % buffer_count  # where the neighbor's next token lands
```

With two buffers the steady-state alternation is:

| step | working slot | receiving slot | meaning |
|---:|---:|---:|---|
| 0 | 0 | 1 | accumulate local token; RDMA local token into neighbor slot 1 |
| 1 | 1 | 0 | accumulate token received from neighbor; send it onward to slot 0 |
| 2 | 0 | 1 | slot 0 reused only after its prior occupant drained |
| 3 | 1 | 0 | steady-state alternation continues |

The hazard rule:

```text
A slot may be reused only after local reads/writes, send waits, and receive waits for the previous occupant have drained.
```

In the serialized path this is trivially safe (the next chunk does not start
until the current one completes). In `pallas_db` it is enforced for real by a
**capacity semaphore**: before a device starts the RDMA into its neighbor's
receiving slot, it signals the upstream neighbor that its own receiving slot is
free and waits for the same signal from downstream. That handshake is what keeps
a faster device from overwriting a slot the receiver has not drained — the
difference between correct overlap and a stale buffer carnival. Setting
`--lab8-buffer-count 3` or `4` adds run-ahead capacity; the capacity handshake
still bounds it, so more buffers may not always help — a useful result, not a
failure.

## Run Commands

Default run:

```bash
cd ~/labs/collective_communication
python collective_bench.py --lab lab8
```

Short smoke run:

```bash
python collective_bench.py \
  --lab lab8 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Chunk-count sweep:

```bash
for chunks in 1 2 4 8 16; do
  python collective_bench.py \
    --lab lab8 \
    --sizes 1MiB,4MiB,16MiB \
    --lab8-chunks "${chunks}" \
    --lab8-buffer-count 2
done
```

Direction sweep:

```bash
for direction in right left; do
  python collective_bench.py \
    --lab lab8 \
    --sizes 4MiB \
    --lab8-chunks 8 \
    --neighbor-direction "${direction}"
done
```

Buffer-count sweep (real double buffering for `pallas_db`):

```bash
for buffers in 2 3 4; do
  python collective_bench.py \
    --lab lab8 \
    --ops pallas_db_token_ring \
    --sizes 4MiB \
    --lab8-chunks 8 \
    --lab8-buffer-count "${buffers}"
done
```

Inner-block sweep — see the micro-transfer penalty of an undersized VMEM block.
`--lab8-inner-cols 0` (the default) auto-sizes the largest VMEM-safe block; a
small value forces tiny blocks:

```bash
for inner in 0 128 1024 8192; do
  python collective_bench.py \
    --lab lab8 \
    --ops pallas_db_token_ring,pallas_chunked_token_ring \
    --sizes 4MiB \
    --lab8-inner-cols "${inner}"
done
```

Profile the fused double-buffered kernel (look for RDMA start before the
accumulation pipeline, and the wait after):

```bash
python collective_bench.py \
  --lab lab8 \
  --sizes 4MiB \
  --lab8-chunks 8 \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_db_token_ring \
  --trace-size 4MiB
```

Useful flags:

```bash
--lab8-chunks 8
--lab8-buffer-count 2
--token-hops 0
--token-hops 1
--neighbor-direction left
--neighbor-direction right
--pallas-memory-space HBM
--pallas-memory-space VMEM
```

`--token-hops 0` is a good sanity check: each chunk should equal the local input
chunk converted to `float32`. `--token-hops 1` checks a single movement. The
default full-ring setting is usually `device_count - 1`.

## What To Inspect

Start with:

```text
results_summary.md
csvs/results.csv
plots/latency_by_payload.png
plots/bandwidth_by_payload.png
logs/console.log
lab_artifacts/*lab8_chunked_pipeline_spec*
```

When profiling is enabled, inspect:

```text
traces/
artifact_index.json
```

Questions for the CSV:

```text
By how much does pallas_db_token_ring edge out the serialized path, and does
  that edge shrink as payload grows? Why would it?
Does the gap to xla_token_ring widen as payload grows? Tie it to bytes moved.
Where is the chunk-count knee for the serialized path?
With --lab8-inner-cols 128, how badly does the fused kernel regress, and why?
Does buffer_count = 3 or 4 help pallas_db, or does the capacity handshake dominate?
```

Questions for the trace (`--trace-op pallas_db_token_ring`):

```text
Is there one fused custom kernel rather than C * H separate Lab 1 calls?
Does the remote DMA start before the local accumulation pipeline?
Does the accumulation run before the remote-copy wait (i.e. real overlap)?
Are the capacity-semaphore waits visible and bounded?
```

Questions for the trace:

```text
Do you see separate chunk-hop phases?
Are there gaps between phases?
Do named scopes make the serialized structure obvious?
Is there any evidence of overlap, or only many small blocking copies?
```

If there is no evidence of overlap, do not claim overlap. Wizards keep receipts.

## Correctness Contract

For every receiver device `r`, chunk `c`, row `i`, and column `j`:

```text
expected[r, c, i, j]
  = sum over sources seen by r:
      10 * source + c + 0.25 * (i mod 8) + 0.03125 * (j mod 16)
```

For a full ring, every receiver sees every source exactly once. For partial
hops, each receiver sees only a prefix of the ring schedule.

If `y[:, :, 0, 0]` looks correct but the full check fails, suspect:

```text
partial-tile copy
wrong chunk axis
stale buffer slot
accidental broadcasting
wrong dtype conversion
```

## Common Failure Modes

### Too many chunks create too much overhead

Symptom:

```text
latency gets worse as chunk count increases
```

Likely explanation:

```text
The serialized implementation performs C * H remote-copy phases per device.
```

Fix:

```text
Do not call this a bad result. Plot it. Explain it.
```

### The fused kernel is slower than the serialized path

Symptom:

```text
pallas_db_token_ring latency is much worse than pallas_chunked_token_ring, especially at large payloads
```

Likely explanation:

```text
The inner VMEM block is too small, so emit_pipeline does thousands of micro-transfers and overlap cannot pay for the overhead.
```

Fix:

```text
Use --lab8-inner-cols 0 (auto, the default). A fixed small value like 128 is for *demonstrating* the penalty, not for fast runs.
```

### Buffer count and double buffering

Symptom:

```text
--lab8-buffer-count 2 does not change the serialized path's runtime.
```

Likely explanation:

```text
Buffer count is a planning artifact for the serialized path. It only becomes real HBM double buffering in pallas_db_token_ring.
```

Fix:

```text
Compare buffer counts against the pallas_db op, not the serialized op, and confirm overlap in the trace.
```

### Collective IDs collide

Symptom:

```text
wrong data, stale semaphore state, or a hang-prone run
```

Likely explanation:

```text
Two chunk-hop phases reused a collective ID while their communication patterns were not safely separated.
```

Fix:

```text
Use the collective-ID plan in the spec artifact.
```

### VMEM fails at large chunk sizes

Symptom:

```text
compile-time memory failure or early validation error
```

Likely explanation:

```text
A single chunk tile is too large for the selected memory space.
```

Fix:

```text
Use HBM, reduce payload size, or increase chunk count.
```

## Pass Condition

```text
reference token-ring correctness remains stable across payload sizes
serialized Pallas chunked ring matches per-chunk full-tile token sums on TPU
fused pallas_db_token_ring matches the same full-tile sums on TPU
xla_token_ring matches the same correctness contract and serves as a reference
the spec artifact records source history, byte model, collective-ID plan, and buffer ownership rules
a profile trace shows the remote copy overlapped with the local accumulation pipeline
students can explain where the overlap helps, why its edge shrinks with payload, and why XLA moves fewer bytes
```

## What the Fused Kernel Does

`pallas_db_token_ring` runs one Pallas TPU kernel with an outer grid over ring
steps. Per device, per step `s`:

```text
prologue (step 0): local neighbor barrier; stage local token into hbm_scratch[0];
                   initialize the float32 accumulator from the first token
each step s in 0..hops:
  build a remote copy: hbm_scratch[working_slot] -> neighbor.hbm_scratch[receiving_slot]
  if not last step:
    signal capacity to the upstream neighbor; wait for downstream capacity
    start the async remote DMA
  OVERLAP: accumulate hbm_scratch[working_slot] into the output through an inner
           HBM<->VMEM emit_pipeline while the DMA is in flight
  if not last step:
    wait for the remote copy to complete before the slot is reused
```

The accumulator is read-modify-written into HBM via `emit_pipeline` with
`should_accumulate_out=True`, and the inner column block is auto-sized
(`--lab8-inner-cols 0`) to the largest VMEM-safe window so the pipeline does a
few large transfers rather than many tiny ones.

## Deferred Work

```text
add a bidirectional pallas_db that uses both ICI directions
carry the fused double-buffered schedule into reduce-scatter and all-reduce
explore controlled run-ahead for buffer_count > 2
add interpret-mode race tests for stale-slot hazards
```

## Bridge To Lab 9

Lab 8 asks, "How should one ring be chunked and pipelined?" Lab 9 asks, "Should
this have been one flat ring at all?"

The next lab moves from chunking on a flat ring to topology-aware collectives on
a 2D mesh. The same discipline carries forward:

```text
state the schedule
state the ownership rule
state the bytes
state the synchronization story
prove correctness
profile before bragging
```
