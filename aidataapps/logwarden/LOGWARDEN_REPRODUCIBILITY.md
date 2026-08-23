# LogWarden reproducibility

The row-only input bundle contains 85,411 rows across 46 query/evidence artifacts.

- Input-manifest receipt: `f62ca2de7ec965d3a2d2b571d86ddbbdc20e6c0f38e3b8f9343d0fff37855a71`
- Query-bundle hash: `6ac31332676a99962baf0f9a480d87d06611a1e87a6a781cd42165993cdcd796`
- Deterministic outputs: scorecard, taxonomy, claims, report Markdown, and figure source CSVs.
- Rendering tolerance: PNG bytes may vary across Matplotlib/font builds; source CSV hashes and captions must match.
- `repro.sh --mode rows` verifies every input hash, rebuilds outputs without SQL/GPU/inference, and checks expected hashes.
- `repro.sh --mode restore` starts an isolated SQL Server container, restores the final `.bak` pair, regenerates row exports, compares them byte-for-byte, then runs row-only reconstruction.
