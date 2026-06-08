# Lab 2: Token-Passing Ring

Lab 2 turns the Lab 1 single-hop remote copy into the first real coordination
computation in the course.

In Lab 1, each device pushed one tile to one neighbor. In Lab 2, the received
tile becomes the input to the next hop. The result is a dependency chain around
the logical ring:

```text
hop 0: every device starts with its own rank-valued token
hop 1: every device receives a neighbor token and accumulates it
hop 2: every device receives the next token and accumulates it
...
```

For the default `hops = device_count - 1`, every device should see every rank
exactly once. On four devices, every output element should equal:

```text
0 + 1 + 2 + 3 = 6
```

This lab is not trying to be the fastest possible ring. The custom Pallas path
intentionally composes multiple Lab 1 one-hop kernels. That makes the dataflow,
correctness invariant, and per-hop synchronization visible before later labs
move loops, chunking, buffering, and flow control deeper into custom kernels.

## Where This Fits In The Course

Lab 1 answered:

```text
Can I push one tile to one neighbor with explicit remote DMA?
```

Lab 2 answers:

```text
Can I use that one-hop primitive to build an ordered distributed computation?
```

That is the bridge from copy primitive to collective algorithm. Ring
all-gather, reduce-scatter, all-reduce, pipelined rings, and mesh collectives
all reuse the same basic idea: a schedule of local ownership changes plus
synchronization.

## Mental Model

Each device owns one rank-valued token tile:

```text
token_i = full_tile(rank=i)
seen_sum_i = token_i
```

Then each hop performs:

```text
token_i = token received from logical neighbor
seen_sum_i += token_i
```

The important detail is that `token` is updated after every hop. Hop `k + 1`
depends on hop `k`. This gives you a latency microscope: repeated small copies
make ordering overhead and synchronization cost much easier to see.

## Direction Convention

Direction is from the sender's point of view, matching Lab 1.

For `--neighbor-direction right`:

```text
device 0 sends to device 1
device 1 sends to device 2
device 2 sends to device 3
device 3 sends to device 0
```

Therefore device `i` receives from `i - 1 mod N` on the first hop.

For `--neighbor-direction left`, device `i` receives from `i + 1 mod N` on the
first hop.

## Four-Device Example

With four devices and `direction=right`:

| Output device | Hop 0 | Hop 1 | Hop 2 | Hop 3 | Final sum |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 3 | 2 | 1 | 6 |
| 1 | 1 | 0 | 3 | 2 | 6 |
| 2 | 2 | 1 | 0 | 3 | 6 |
| 3 | 3 | 2 | 1 | 0 | 6 |

With `hops = 1`, device `0` would output `0 + 3 = 3` for the right-moving ring.
With `hops = 4`, device `0` would output `0 + 3 + 2 + 1 + 0 = 6`, because its
own token has wrapped around and been counted a second time.

## Run

From the TPU VM checkout:

```bash
cd ~/labs/collective_communication
python collective_bench.py --lab lab2
```

Short smoke run:

```bash
python collective_bench.py \
  --lab lab2 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Run only one hop to compare against the Lab 1 building block plus accumulation:

```bash
python collective_bench.py \
  --lab lab2 \
  --token-hops 1 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Run the full four-device ring explicitly:

```bash
python collective_bench.py \
  --lab lab2 \
  --token-hops 3
```

Profile the custom Pallas token ring:

```bash
python collective_bench.py \
  --lab lab2 \
  --sizes 4MiB \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_token_ring \
  --trace-size 4MiB
```

## Implementations

The benchmark should compare two versions:

| Operation | What it does | Why it exists |
|---|---|---|
| `pmap_token_ring` | Repeated `lax.ppermute` plus accumulation | XLA-managed reference |
| `pallas_token_ring` | Repeated Lab 1 Pallas remote-DMA hops plus accumulation | Teaching implementation with explicit one-hop building blocks |

The reference path is important. Students should never debug a custom
communication kernel without an executable specification nearby.

## Algorithm

For each device `i`:

```python
token = rank_tile(i)
seen_sum = token.astype(float32)

for hop in range(hops):
    token = neighbor_copy(token)
    seen_sum += token.astype(float32)

return seen_sum
```

In the Pallas path, `neighbor_copy` is the Lab 1 remote-DMA hop. Each hop uses:

```text
collective_id = base_collective_id + hop
```

This is intentionally conservative. It makes barrier and semaphore ownership
visible and avoids teaching students to casually reuse one communication ID for
different phases.

## Correctness Invariant

Let `N = device_count`.

For `direction=right`, output device `i` should have seen:

```text
i, i - 1, i - 2, ..., i - hops    mod N
```

For `direction=left`, output device `i` should have seen:

```text
i, i + 1, i + 2, ..., i + hops    mod N
```

So the expected scalar on device `i` is:

```text
sum(seen ranks for device i)
```

The updated `check_result` checks the full output tile, not just `y[:, 0, 0]`.
That matters because this lab is still proving that every rank-valued tile is
copied and accumulated correctly across the whole payload.

## Byte Model

For token size `B` bytes per device and `H` hops:

```text
remote sends per device       = H
remote receives per device    = H
send bytes per device         = H * B
receive bytes per device      = H * B
total ring send bytes         = N * H * B
```

For the default `H = N - 1`:

```text
send bytes per device = (N - 1) * B
```

This is a useful model to compare with latency and bandwidth plots. If doubling
`H` nearly doubles latency for small payloads, you are seeing the dependency
chain. If large payloads flatten into a bandwidth regime, the wire starts
speaking louder than the per-hop overhead.

## What This Lab Teaches

- A ring algorithm is repeated single-hop ownership transfer.
- Dependency chains amplify latency.
- Per-hop correctness can be expressed as a simple rank-history invariant.
- The custom Pallas version is intentionally didactic, not yet optimized.
- `collective_id` discipline matters once one logical algorithm has many
  communication phases.
- Timing distributions matter: read `p50`, `p90`, and `p99`, not just one mean.
- Profiles should be interpreted together with the run's size, hop count, dtype,
  direction, memory space, and correctness status.

## Artifacts To Inspect

Each run creates a directory under `runs/`.

Start with:

```text
results_summary.md
csvs/results.csv
plots/latency_by_payload.png
plots/bandwidth_by_payload.png
logs/console.log
```

For Lab 2, pay special attention to columns or notes showing:

```text
op
payload_bytes
token_hops
neighbor_direction
p50_us
p90_us
p99_us
bandwidth_GBps
ok
note
```

When profiling is enabled, inspect:

```text
traces/
artifact_index.json
```

Look for per-hop names such as:

```text
lab2_token_hop_00
lab2_token_hop_01
lab2_token_hop_02
```

Exact trace labels may vary by JAX/XLA version, but named scopes are included in
the code so students have a fighting chance of seeing the algorithm in the
trace rather than one anonymous slab of runtime soup.

## Useful Flags

```bash
--token-hops 0
--token-hops 1
--token-hops 3
--neighbor-direction left
--neighbor-direction right
--pallas-collective-id 10
--pallas-memory-space HBM
--pallas-tile-rows 4
--pallas-min-cols 128
```

Use `--token-hops 0` as a no-communication sanity check. It should return the
original rank on every device.

Use `--neighbor-direction left` to verify that your mental model is based on the
sender direction, not the receiver direction.

Use `--pallas-collective-id` when running variants in the same process and you
want to make communication-phase identity explicit.

## Suggested Experiments

### 1. Hop scaling

Run fixed payload sizes while sweeping hop count:

```bash
python collective_bench.py \
  --lab lab2 \
  --sizes 1KiB,64KiB,4MiB \
  --token-hops 0,1,2,3
```

Question:

```text
Does latency grow roughly linearly with hops for small payloads?
```

### 2. Direction reversal

Run:

```bash
python collective_bench.py \
  --lab lab2 \
  --neighbor-direction left \
  --token-hops 3
```

Question:

```text
Does the final full-ring sum stay the same? What changes for partial hops?
```

### 3. Reference versus custom

Compare `pmap_token_ring` and `pallas_token_ring`.

Questions:

```text
Which one wins at tiny payloads?
Which one wins at larger payloads?
How much of the Pallas cost is likely from one-hop kernel composition?
```

### 4. Wraparound

Use more hops than `N - 1`.

For four devices and `direction=right`, predict the expected output for:

```text
--token-hops 4
--token-hops 5
```

Then run it and check your prediction.

## Common Failure Modes

### All devices output their own rank

Likely causes:

```text
--token-hops 0 was used intentionally
neighbor_copy was bypassed
Pallas path failed and the harness used the wrong result
```

### Right and left direction seem swapped

Remember: direction is from the sender's point of view.

```text
right send means receiver sees rank i - 1
left send means receiver sees rank i + 1
```

### Full-ring case passes, partial-hop case fails

This is a classic ring bug. Full rings can hide direction errors because every
device sees every rank eventually. Test `--token-hops 1` and `--token-hops 2`
when debugging neighbor maps.

### Small payloads look surprisingly slow

That is expected. Lab 2 composes multiple one-hop communication phases. For tiny
payloads, the fixed cost of synchronization, dispatch, and dependency ordering
can dominate the actual data movement.

### Large VMEM runs fail

Lab 2 still uses whole-token copies. Use HBM for the normal payload sweep:

```bash
--pallas-memory-space HBM
```

Small VMEM experiments are useful, but large whole-token VMEM staging belongs in
the later memory-space and pipelining labs.

## Pass Criteria

A successful Lab 2 submission should include:

```text
1. pmap_token_ring correctness for at least hops 0, 1, and N - 1.
2. pallas_token_ring correctness for at least hops 0, 1, and N - 1.
3. A short explanation of right versus left sender direction.
4. A table or plot showing latency versus payload size.
5. A table or plot showing latency versus hop count.
6. One paragraph comparing the built-in reference and custom Pallas path.
7. One profiler trace for pallas_token_ring, with the profiled payload and hop
   count recorded.
```

## Bridge To The Next Labs

Lab 2 is the last lab where the ring only carries a rank token. Next, students
should learn the memory hierarchy and synchronization failure modes more
explicitly, then build ring all-gather:

```text
Lab 3: Pallas memory spaces and local movement
Lab 4: semaphore bug zoo
Lab 5: ring all-gather from repeated hops
```

The conceptual upgrade from Lab 2 to ring all-gather is small but important:
Lab 2 accumulates scalar rank identities. Ring all-gather accumulates full shard
ownership.

## Reference Reading

- JAX Pallas distributed TPU tutorial: https://docs.jax.dev/en/latest/pallas/tpu/distributed.html
- Pallas TPU API reference: https://docs.jax.dev/en/latest/jax.experimental.pallas.tpu.html
- JAX profiling guide: https://docs.jax.dev/en/latest/profiling.html
