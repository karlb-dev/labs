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

The implemented custom path is **chunked but serialized**. Each chunk runs
through the Lab 1 remote-DMA hop sequence independently. This is deliberate. A
fully fused, double-buffered, overlapped ring is a later optimization. Lab 8 is
the bridge: students learn chunk size, buffer ownership, collective-ID planning,
and profile evidence before they claim pipeline magic. 🧪

## Implemented Happy Path

Run:

```bash
python collective_bench.py --lab lab8
```

Implemented operations:

- `pmap_token_ring`: dependency-chain reference from Lab 2.
- `pallas_chunked_token_ring`: serialized chunked ring built from Lab 1 hops.
- `lab8_chunked_pipeline_spec`: source-history, byte-model, collective-ID, and buffer-slot artifact.

The custom path does this:

```text
for chunk in chunks:
    token = local_chunk[chunk]
    seen_sum = token
    for hop in ring_hops:
        token = one Lab 1 neighbor_copy(token)
        seen_sum += token
    output[chunk] = seen_sum
```

That is not the final optimized kernel. It is a teaching implementation that
exposes the performance ingredients one at a time.

## Chunking Versus Pipelining

Chunking cuts a payload into pieces.

Pipelining keeps multiple pieces in flight.

They are cousins, not twins.

The current Pallas path is chunked, but each chunk completes its whole ring
before the next chunk starts. Therefore `--lab8-buffer-count` does **not** make
the current path double-buffered. It records the buffer-slot contract a future
fused kernel must obey.

Useful question for students:

```text
Did the benchmark prove overlap, or did the code merely configure a buffer count?
```

The only acceptable answer comes from a trace.

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

`--lab8-buffer-count` does not create real overlap yet. It records the planned
slot assignment:

```text
buffer_slot = chunk_idx % buffer_count
```

With two buffers:

| chunk | slot | epoch |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 1 |
| 4 | 0 | 2 |
| 5 | 1 | 2 |

The hazard rule:

```text
A slot may be reused only after local reads/writes, send waits, and receive waits for the previous occupant have drained.
```

In the current serialized implementation, this rule is trivially safe because
the next chunk does not start until the current one completes. In a fused
pipeline, this rule becomes the difference between correct overlap and a stale
buffer carnival.

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

Buffer-count planning sweep:

```bash
for buffers in 1 2 3 4; do
  python collective_bench.py \
    --lab lab8 \
    --sizes 4MiB \
    --lab8-chunks 8 \
    --lab8-buffer-count "${buffers}"
done
```

Profile the custom path:

```bash
python collective_bench.py \
  --lab lab8 \
  --sizes 4MiB \
  --lab8-chunks 8 \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_chunked_token_ring \
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
At fixed payload size, does latency increase as chunk count increases?
At fixed payload size, does bandwidth improve, degrade, or stay flat?
Where is the chunk-count knee?
Does the Pallas chunked path beat the Lab 2 unchunked token ring?
Does it beat anything built-in, or only teach us something?
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

### Buffer count is mistaken for real double buffering

Symptom:

```text
Students expect --lab8-buffer-count 2 to improve runtime by itself.
```

Likely explanation:

```text
The current code records the schedule but does not overlap chunks inside one fused kernel.
```

Fix:

```text
Use the spec artifact as a contract for the later fused implementation.
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
the spec artifact records source history, byte model, collective-ID plan, and buffer ownership rules
students can explain why smaller chunks can be slower before overlap exists
```

## Deferred Work

```text
fuse the chunk schedule into one Pallas kernel
split start, wait_send, wait_recv, and consume phases
use buffer slots for real double buffering
add capacity semaphores or equivalent flow control
insert local compute between enqueue and wait
prove overlap in traces rather than assuming it
carry the optimized chunked schedule into reduce-scatter and all-reduce
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
