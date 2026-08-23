# LogWarden observability contract

Status: binding pre-inference implementation contract, added before data in
response to the PI's 2026-08-23 instruction. It promotes the useful Tier 2
telemetry surfaces from the addendum into a gate for long model runs without
promoting the live systems experiment itself.

## Persistence rule

Every long inference worker writes a local append-only JSONL journal first and
persists the corresponding normalized row to SQL during the run. A SQL outage
may delay ingestion but may not erase the journal. The journal has a monotonic
sequence, event ID, run/job/episode/attempt identity, trace/span identity,
wall-clock UTC, client monotonic time, payload hash, and schema version.

Recovery replays only journal records after the committed SQL ingestion cursor.
Unique event/span/sample keys make replay idempotent. Finalization reconciles
journal counts and hashes against SQL; a mismatch blocks campaign completion.

For operator ergonomics, the unified journal can be projected into semantic
streams (`transcript`, `model`, `tool`, `validation`, `decision`, `metrics`,
and `state-transition`) without creating new sources of truth. Each projection
retains the source event ID and sequence. Post-run diagnostics and aggregate
tables must be reconstructable from the journal plus frozen manifests; derived
files carry an input-set hash and are safe to delete and regenerate.

## Required timing path

The following stages receive start/end or point events as appropriate:

```text
schedule ready -> queue enqueue -> queue claim -> packet load -> prompt build
-> HTTP request start -> model queue/decode -> raw response durable
-> parse/repair -> policy -> tool dispatch -> tool SQL/retrieval
-> tool result durable -> next model turn -> decision validation
-> decision persistence -> proposal persistence -> lease completion
```

Retries, timeouts, cancellations, lease heartbeats/expiry, cache hits, snapshot
misses, truncation, repairs, and every failure transition are explicit events.
No duration is reconstructed from unrelated wall clocks when a monotonic clock
is available.

Every model request retains hashes for the exact sent messages, prompt bytes,
tool registry, response schema, and decode configuration. The client-generated
request ID is sent to the model service when its protocol permits and is the
join key for raw vLLM samples. Client wall time, connection/write time, time to
headers, first-content time when streaming, body-read time, parse time, and
retry backoff are distinct. Server queue/service/TTFT/decode fields are only
populated from a server observation with that request ID; aggregate histogram
deltas are labeled as interval-level evidence.

## vLLM evidence

At each port gate, discover and retain the exact `/metrics` exposition and its
SHA-256 instead of assuming metric names from a different image. During a run,
retain timestamped raw snapshots and promote all available request-count,
running/waiting/swapped sequence, prompt/generation throughput, KV/prefix-cache,
preemption, request latency, queue time, TTFT, time-per-output-token/inter-token,
end-to-end latency, token, and error/cancellation measures. Missing metrics stay
null and are reported as unavailable.

Replay remains non-streaming, so request-level TTFT is null unless the server
exposes it; service histograms are not misrepresented as per-request values.
The live plane may use streaming and measures first content chunk client-side.

## Agent and tool evidence

Every semantic operation is an `agent.agent_steps` row and a trace span. Raw
request and response bytes are persisted before parsing. Tool rows include
canonical arguments, policy result, execution/cache/snapshot-miss status,
result hash and truncation, row count, latency, and failure. Validation emits
one row per layer. Decisions and transitions are immutable.

The loop has frozen limits for turns, tool calls, repeated no-action responses,
rejected terminal decisions, response bytes/tokens, wall time, and retries.
Limit exhaustion never fabricates success: it deterministically persists a
terminal `needs_human_review`, `contract_rejected`, or other governed failure
with the reason and incomplete work state. OOM, empty output, invalid JSON,
schema rejection, policy rejection, snapshot miss, HTTP error, timeout,
cancellation, output limit, and telemetry failure are separate error classes.

Raw token counts from the OpenAI-compatible response are tagged
`server_reported`; tokenizer-derived counts are tagged `tokenizer_counted`;
byte- or whitespace-based approximations are tagged `estimated`. Aggregates do
not mix these provenances without a visible qualification.

## Systems samples

At a frozen interval, sample:

- queue depth/state, oldest age, arrivals, completions, lease expiry and retry;
- raw vLLM `/metrics` plus model container/image/profile identity;
- GPU utilization, memory, power, clocks, and process allocations from
  `nvidia-smi` where available;
- SQL process memory/CPU, requests, waits, file I/O, database/log size, and
  Query Store interval deltas;
- XE cursor/newest-event/ingest lag, batch size, parse errors, rollover, and
  dropped-event counters;
- host disk and journal/SQL storage growth.

The development tier measures telemetry-on versus bounded telemetry-off
overhead. The chosen interval and any unavailable signal are frozen in the run
configuration before test inference.

Diagnostics separate end-to-end critical-path wall time from additive component
time and report unexplained wall time rather than forcing components to sum.
They include per-turn waterfalls, per-tool/per-model-step breakdowns, cold/warm
status, concurrency and overlap where timestamps support it, p50/p95/tails,
failure and missingness funnels, and per-job/model/arm/role/regime aggregates.

## Gate before a long LLM run

- fake-gateway tests prove every success/failure route closes its trace and
  records the correct terminal transition;
- crash/replay tests prove journal-to-SQL idempotency and cursor recovery;
- a real port gate retains raw `/metrics` before, during, and after canaries and
  records which expected measures exist;
- journal, SQL, job, request, response, decision, and tool counts reconcile;
- the sampler survives a worker restart without duplicate samples;
- telemetry overhead and sampling interval are recorded;
- the checkpoint watchdog mirrors journals and database backups within the
  durability ceiling.
- diagnostics rebuilt from journals match their persisted input-set hashes and
  do not depend on mutable live state;
- each exhausted loop bound produces the expected deterministic terminal row,
  and OOM/empty/truncated/invalid/retried responses retain distinct evidence.

Any unexplained observability gap blocks the long run rather than becoming a
post-hoc limitation.
