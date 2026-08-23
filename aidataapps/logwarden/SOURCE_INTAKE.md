# LogWarden source intake

Recorded 2026-08-23 UTC before implementation or scientific data collection.
The governing addendum wins over the specification. Labs 1–2 and the frozen
`interpretability/` tree are read-only predecessors.

| Input | SHA-256 | Disposition |
|---|---|---|
| Drive Lab 3 specification | `fc101329474ac9bd5866c719017721cd6207c6db3ccc94ab4121e6a1d86bc144` | vendored byte-identically as `docs/SPEC.md` |
| Drive Lab 3 addendum | `7d4a84cda08344907436552cff9de72f999e5cacf8e43c69dcc4d27f42d14d51` | vendored byte-identically as `docs/SPEC_ADDENDUM.md` |
| inherited ModelPrint registry | `eaa204e9d9134c62ef5b327f1bef3987e9300b92f8e4e462b7fa1354ae147a30` | vendored byte-identically as `config/models.json` |
| Lab 1 Colab host bootstrap | `c62ca9df5a5d45b77c3e977de9cf95a1cb44ae5654460676255e3587ba83510c` | adapted under Lab 3 paths and names |
| Lab 2 checkpoint watchdog | `5ca836e724057b1df1b898511a94ece19cc65a62fdba8949d6fcedaa2193e0d2` | adapted to Lab 3 ownership and Drive subtree |
| `cloud-deploy-agent` reference | commit `273221e97e36abea34e47961d4deb42455750f02` | inspected read-only; design ideas selectively adapted, no source copied |

## Repository identity

- source branch: `origin/aidataapps-modelprint`
- source/base commit: `88ea443092cd27226272776e6c6fcf8fbce329de`
- Lab 3 branch at intake: `aidataapps-logwarden`
- model registry's own recorded Lab 1 source commit:
  `a11a433c82b28a85f572bdc41eb6d20e9534d267`

## Binding intake decisions

- Use the addendum's Tier 1 Minimum State of Record and defer Tier 2/3 until
  row-only reconstruction, terminal job accounting, and safety gates pass.
- Use structured JSON for every primary model; native tools and guided JSON are
  separate Tier 2 cells.
- Target at least 600 captured episodes; never retain an infeasible scenario
  merely to preserve a round count.
- Build the custom SQL Server full-text image first. The governed app-side BM25
  fallback is allowed only if the nested runtime cannot build it, and such a
  fallback narrows lexical/full-text claims as specified.
- Keep replay model quality separate from live systems measurements.

## Instrumentation reference disposition

The PI supplied
`karlb-dev/documents/uw/csep590a-26sp/project/cloud-deploy-agent` as a design
reference. The intake clone was pinned at commit
`273221e97e36abea34e47961d4deb42455750f02`. The most relevant reviewed file
hashes were:

| File | SHA-256 |
|---|---|
| `dbagent/telemetry.py` | `6169ce8b694aacfbe8ff6e481f01fea9d11b78f9289d825a49617a45d5d85f1c` |
| `dbagent/agent/runner.py` | `da67d12680b5cba4a69ff8be55a84733c14a58b760f16853bc57ca71a378337f` |
| `dbagent/agent/transcript.py` | `0b68fab39a0999bce0908ba55c49b3c7db1659bc88141592ce3eeb03b5073ea2` |
| `dbagent/service/run_diagnostics.py` | `e6142ae2632d3d53bf86c665864cd8640c316af11494031505eabfc7b2ed0579` |
| `dbagent/service/benchmark.py` | `4b0942a2e8602f689a3d5e6125edeb7d32b2503e99650a32ed52929b983f604a` |
| `dbagent/service/_gemma_server_ingest.py` | `95fcc4e1f880de7740caa59e07eb10b685fd0d8f94b0a08138c9e14dccfe45df` |

Adapted concepts are semantic streams plus a unified journal, stable
prompt/tool hashes per request, post-run reconstruction from immutable files,
request/run correlation between client and server, explicit output-limit and
failure categories, cold/warm annotations, per-job and grouped summaries, and
deterministic safe terminal handling when an agent exhausts its bounds. The
TPU/JAX implementation, timing heuristics, and token estimates are not copied
or treated as vLLM observations. LogWarden records metric provenance and keeps
unavailable fields null.

## Version-sensitive method references checked at intake

- Microsoft Learn, “Install SQL Server Full-Text Search on Linux” (SQL Server
  2025 view; updated 2026-06-11).
- Microsoft Learn, `CREATE EVENT SESSION` (`MAX_DISPATCH_LATENCY` minimum one
  second and default 30 seconds).
- Microsoft Learn, Query Store management and supported one-minute statistics
  interval.
- vLLM official structured-output protocol/examples for OpenAI-compatible
  `response_format.type = json_schema`.

These pages constrain implementation choices but are not substitutes for the
retained SQL/vLLM runtime probes.
