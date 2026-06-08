# Lab 4: Semaphore Bug Zoo

Goal: learn the synchronization failure modes around TPU remote DMA without
making the default lab run hang-prone.

Lab 4 is the course's little synchronization museum. Labs 1 and 2 showed that a
single-hop remote copy can move data correctly. Lab 3 separated local memory
movement from remote movement. Lab 4 asks the scarier question:

```text
What exactly breaks when the synchronization contract is wrong?
```

The answer is not just "the output is wrong." With custom TPU communication,
some mistakes fail fast, some crash at kernel completion, some silently corrupt
data, and some can leave the process waiting for bytes that will never arrive.
The goal of this lab is to make those failure modes concrete while keeping the
normal benchmark safe to run.

## Course Placement

```text
Lab 1: one remote DMA hop
Lab 2: repeated hops as a token ring
Lab 3: local HBM <-> VMEM movement
Lab 4: semaphore and synchronization failure modes
Lab 5: ring all-gather from scratch
```

Lab 4 is deliberately placed before all-gather. Once a one-hop copy becomes a
multi-hop collective, the number of possible races multiplies. This lab gives
students a checklist before the ring grows teeth.

## Operations

Lab 4 has three operations:

- `pallas_semaphore_correct`: the correct Lab 1 single-hop remote-copy probe.
  It exercises a phase-entry barrier plus DMA send/receive semaphores.
- `pallas_semaphore_bug`: a **runnable** reproduction of one *safe* catalog bug
  (`wrong_neighbor_map` by default). It runs the correct kernel with the wrong
  neighbor direction, so it completes cleanly but the correctness oracle catches
  the rotated ranks. This is the lab's "see a bug for real" operation. It records
  `ok=True` when the documented symptom is reproduced, so a healthy Lab 4 run
  stays green even though this row deliberately produces wrong data — read the
  `observed_ranks` vs `expected_ranks` columns and the note to see the mismatch.
- `semaphore_bug_zoo`: a safe catalog of broken-kernel mutations, expected
  symptoms, diagnostics, prevention rules, and recovery rules. It writes JSON
  and Markdown under `lab_artifacts/` in the run directory.

Only the *correctness-class* bug runs in-process. Hang/crash/race variants are
not executed by default, and on purpose. A kernel with an over-wait can hang the
process. A kernel with nonzero semaphore state can crash at completion. A buffer
ownership race can produce a plausible-looking but wrong tensor. Those run only
through a guarded subprocess with a hard timeout (see "Running a dangerous bug"
below); the default lab run teaches these hazards rather than summoning them.

## Mental Model

A Pallas remote DMA has three separate ideas that students should not blur
together:

```text
remote copy descriptor: the operation that starts and waits for a transfer
DMA send semaphore:      source-side progress that says sending is complete
DMA receive semaphore:   receiver-side progress that says bytes arrived
```

Then there are synchronization semaphores that are not themselves the DMA data
movement:

```text
barrier semaphore:   make sure peers have entered the cross-device phase
regular semaphore:   explicit flow control or buffer ownership between devices
collective_id:       namespace for barrier semaphore use inside pallas_call
```

The Lab 4 mantra:

```text
A custom collective is a data movement schedule plus a semaphore ledger.
```

If the ledger does not balance, the benchmark dragon starts coughing sparks.

## Run

```bash
cd ~/labs/collective_communication
python collective_bench.py --lab lab4
```

Short run with only the catalog:

```bash
python collective_bench.py \
  --lab lab4 \
  --ops semaphore_bug_zoo \
  --no-plots
```

Profile the correct semaphore probe:

```bash
python collective_bench.py \
  --lab lab4 \
  --ops pallas_semaphore_correct \
  --sizes 1KiB \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_semaphore_correct \
  --trace-size 1KiB
```

Small correctness-only run:

```bash
python collective_bench.py \
  --lab lab4 \
  --ops pallas_semaphore_correct,semaphore_bug_zoo \
  --sizes 1KiB \
  --iters 5 \
  --warmup 1 \
  --no-plots
```

## Running a Real Bug (Safe)

The default `--lab lab4` run includes `pallas_semaphore_bug`, which actually
reproduces the `wrong_neighbor_map` failure instead of only describing it. To run
just the bug demo:

```bash
python collective_bench.py \
  --lab lab4 \
  --ops pallas_semaphore_bug \
  --sizes 1KiB --iters 5 --warmup 1 --no-plots
```

What you should see: the row completes (no hang), `ok=True` (the demo behaved as
designed), and `observed_ranks` does **not** equal `expected_ranks` — for four
devices, the kernel sent `left` while the intended map was `right`, so it shows
`observed=[1, 2, 3, 0]` against `expected=[3, 0, 1, 2]`. That is the documented
symptom: *fast correctness failure with otherwise clean completion*. The lesson
is that a wrong topology map is caught only because there is a correctness
oracle; without one, this bug ships silently.

`--lab4-run-bug <id>` selects which catalog bug to reproduce. Only
correctness-class bugs are runnable in-process (currently `wrong_neighbor_map`);
asking for a hang/crash/race id here returns a clear "use the guarded path"
message instead of executing it.

## Running a Dangerous Bug (Guarded, Opt-In)

Hang/crash/race bugs are **not** run by the normal sweep and are **not** shipped
as ready-to-run kernels, because a real over-wait can deadlock the TPU process
and force a runtime restart. The harness provides the safe execution mechanism
for them — `run_guarded_subprocess` — which launches the repro in its own process
group under a hard wall-clock timeout and SIGKILLs the whole group on expiry, so
even a true deadlock cannot keep the device wedged:

```bash
# Disposable VM only. Requires you to add the broken kernel (see below).
python collective_bench.py \
  --lab lab4 \
  --ops pallas_semaphore_bug \
  --lab4-run-bug overwait_dma \
  --lab4-allow-dangerous \
  --lab4-bug-timeout 20 \
  --sizes 1KiB --no-plots
```

To author a dangerous repro, add a broken kernel builder to
`labs/lab4_semaphore_bug_zoo.py` and register its id, then route it through
`run_guarded_subprocess`. Do this in a disposable environment with a small
payload and a short timeout — never in a shared session, and never in the default
op list. The guard's timeout/teardown is unit-checkable without a TPU via the
hidden `--__lab4-selftest-hang` entrypoint.

## What This Lab Teaches

- Every remote DMA needs one intended sender and one intended receiver.
- DMA byte counts must match exactly.
- `start()` means a copy has been issued, not that its result is safe to read.
- Send-side waits and receive-side waits answer different questions.
- Barrier semaphores are about phase entry, not data payload bytes.
- Regular semaphores are counters and must drain before completion.
- `collective_id` values are part of the synchronization contract.
- Buffer slots need ownership rules, especially once pipelining begins.
- Some bugs are safe correctness failures; others belong only in isolated repros.

## Correct Probe: `pallas_semaphore_correct`

The correct probe reuses the Lab 1 ownership model:

```text
device 0 -> device 1
device 1 -> device 2
device 2 -> device 3
device 3 -> device 0
```

If every device sends right, then device `i` receives from device `i - 1 mod N`.
The output tile on device `i` should be filled with the left-neighbor rank.

The important synchronization structure is:

```text
1. compute destination device
2. enter phase-entry barrier
3. start remote DMA
4. wait for send completion
5. wait for receive completion
6. validate output ownership
```

Depending on the exact Lab 1 implementation in the repo, the remote copy may be
written as one combined `.wait()` or as explicit `wait_send()` and
`wait_recv()` calls. For teaching, students should still be able to explain both
halves:

```text
wait_send(): source-side proof that this device is done sending
wait_recv(): destination-side proof that this device received its bytes
```

## Bug Zoo Scenarios

The Python file records the scenarios as structured data so the benchmark can
write both JSON and Markdown artifacts.

| id | safety | lesson |
| --- | --- | --- |
| `overwait_dma` | dangerous, hang risk | Waiting for bytes that no sender can produce can block forever. |
| `undersend_dma` | dangerous, hang risk | Source and destination slices must agree on size. |
| `oversignal_regular` | dangerous, crash risk | Semaphores must drain to zero at program completion. |
| `overwait_regular` | dangerous, hang risk | Regular semaphore waits must have matching peer signals. |
| `missing_entry_barrier` | dangerous, race risk | Peers must enter the communication phase before remote writes begin. |
| `collective_id_reuse` | dangerous, race risk | A `collective_id` names a compatible communication pattern, not a casual label. |
| `mismatched_collective_order` | dangerous, hang risk | All peers must execute phases in compatible order (mismatch deadlocks). |
| `two_writers_one_slot` | dangerous, race risk | One destination slot cannot have two simultaneous writers. |
| `wrong_neighbor_map` | safe correctness failure | The sender map and expected-output oracle must agree. |
| `missing_dma_wait_before_use` | dangerous, race risk | Do not consume a destination before the DMA receive has completed. |
| `missing_wait_send_before_source_reuse` | dangerous, race risk | Do not overwrite a source slot before the send side is done. |
| `buffer_slot_run_ahead` | dangerous, race risk | Pipelines need per-slot ownership, not vibes. |
| `single_recv_semaphore_multiple_inflight` | dangerous, race risk | Independent in-flight DMAs need unambiguous completion tracking. |
| `device_id_type_or_axis_mismatch` | isolated repro only | Mesh coordinates, logical IDs, and axis names must match. |

Only `wrong_neighbor_map` is safe to execute as a default broken variant because
it completes cleanly and merely fails correctness. It is exactly the bug the
`pallas_semaphore_bug` operation reproduces in the default run (see "Running a
Real Bug"). The other scenarios are cataloged for study and run only as guarded,
isolated repros.

## Semaphore Ledger

Before writing any custom communication phase, students should be able to fill
this table.

| Field | Question | Example |
| --- | --- | --- |
| `phase_name` | What communication phase is this? | `lab5_all_gather_hop_2` |
| `collective_id` | Which barrier namespace does this phase use? | `base_collective_id + hop` |
| `participants` | Which devices participate? | all devices on mesh axis `x` |
| `source_ref` | Which local ref or slot is read by the send? | `send_buffer[working_slot, ...]` |
| `destination_ref` | Which remote ref or slot is written? | `recv_buffer[receiving_slot, ...]` |
| `bytes` | How many bytes can be sent and received? | `rows * cols * dtype.itemsize` |
| `send_wait` | What proves the source can be reused? | `remote_copy.wait_send()` |
| `recv_wait` | What proves the destination can be consumed? | `remote_copy.wait_recv()` |
| `buffer_owner_after_phase` | Who owns each buffer slot after the phase? | device `i` owns shard from device `i - hop` |

This is the worksheet students should reuse in Labs 5 through 8.

## Debugging Rules

### Rule 1: Separate send-side and receive-side reasoning

A sender being done sending does not mean the receiver's later computation is
correct unless the receive side also waited for the bytes it needs.

```text
source reuse safety:        wait_send()
destination consume safety: wait_recv()
```

### Rule 2: Make byte counts boring

Byte counts should come from shapes and dtypes, not from hand-written constants.
Manual byte constants are tiny foot-traps wearing tap shoes.

```text
bytes = rows * cols * dtype.itemsize
```

For chunked labs:

```text
chunk_bytes = chunk_rows * chunk_cols * dtype.itemsize
```

### Rule 3: Treat `collective_id` as a scarce namespace

A `collective_id` is not a decorative number. It is part of the barrier
semaphore contract. Reusing one across incompatible phases can create confusing
races.

Suggested convention for the course:

```text
Lab 1 correct single hop:      1
Lab 2 token hop h:             base + h
Lab 5 all-gather hop h:        base + h
Lab 8 pipeline phase p, hop h: base + p * 100 + h
```

The exact numbers matter less than the written map.

### Rule 4: Draw ownership, not just arrows

A communication diagram that only shows arrows is half a spell. Add buffer
ownership:

```text
before phase:  who may read each slot?
during phase:  who may write each slot?
after phase:   who owns each slot's valid data?
```

### Rule 5: Isolate dangerous repros

A dangerous scenario should use:

```text
one tiny payload
one iteration
one fresh process
one timeout
one fresh run directory
one explicit cleanup/restart plan
```

Do not hide dangerous repros in a payload sweep.

## Default Artifacts

Look for:

```text
lab_artifacts/*_semaphore_bug_zoo.json
lab_artifacts/*_semaphore_bug_zoo.md
results.jsonl
csvs/results.csv
errors/*.txt
logs/console.log
traces/..., if profiling was enabled
```

The catalog marks each scenario as safe or dangerous for default execution.
Dangerous scenarios are still useful, but they should be implemented later as
separate subprocess or timeout-controlled repros.

## Suggested Student Exercises

### Exercise 1: Fill the ledger for the correct probe

For each device in a four-device right ring, fill:

```text
rank:
destination rank:
expected source rank:
barrier signal target:
barrier wait count:
DMA bytes:
send wait:
receive wait:
final output rank:
```

### Exercise 2: Classify the symptom

Pick the most likely scenario:

```text
Symptom: kernel completes useful work, then reports nonzero semaphore state.
Likely scenario: oversignal_regular.
```

```text
Symptom: works for one chunk, fails intermittently for 8 chunks.
Likely scenario: buffer_slot_run_ahead or missing_dma_wait_before_use.
```

```text
Symptom: operation completes quickly but rank markers are rotated the wrong way.
Likely scenario: wrong_neighbor_map.
```

### Exercise 3: Add one new bug card

Add a new `BugScenario` to the Python file with:

```text
id
invariant
mutation
expected_symptom
safe_to_run_by_default
diagnostic
prevention
recovery
worksheet_questions
profiler_clue
```

Then verify that `scenario_rows()`, `render_json()`, and `render_markdown()`
include it.

### Exercise 4: Design an isolated repro, but do not run it in the sweep

Pick one dangerous scenario and write a plan:

```text
payload size: 1 KiB
iterations: 1
subprocess timeout: 30 seconds
fresh collective_id: yes
expected failure mode:
logs to capture:
cleanup/restart instruction:
```

The harness now provides the hardened runner (`run_guarded_subprocess`: isolated
process group + hard timeout + group SIGKILL on expiry), so the remaining work is
to author the broken kernel itself and run it in a disposable VM — see "Running a
Dangerous Bug". Keep dangerous repros out of the default sweep.

## Pass Condition

```text
the correct semaphore probe matches the Lab 1 ppermute ownership model
the pallas_semaphore_bug demo reproduces wrong_neighbor_map (observed != expected) and completes cleanly
the bug zoo artifact lists invariant, mutation, symptom, diagnostic, prevention, and recovery
no hang-prone broken kernel runs by default; dangerous repros run only under the guarded subprocess
students can fill the semaphore ledger for the next ring all-gather lab
```

## Review Questions

1. Why is an over-wait more dangerous for a default benchmark than a wrong-neighbor map?
2. Why does a barrier semaphore need `collective_id` discipline?
3. Why is a DMA semaphore easier to reason about in bytes than a regular semaphore?
4. Why can a double-buffered algorithm still race if one sender runs ahead?
5. What artifact would you inspect first for a correctness failure? What about a hang?

## Next Step

Lab 5 is ring all-gather. Its checklist should be inherited from this lab:

```text
before adding N - 1 hops, prove one hop
before adding chunks, prove one slot
before overlapping copies, prove every wait
before profiling speed, prove ownership
```

That is the quiet magic of Lab 4: it does not add a new collective. It gives
every later collective a safety rail.
