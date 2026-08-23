# LogWarden reproducibility

The row-only input bundle contains 85,596 rows across 50 query/evidence artifacts.

- Input-manifest receipt: `02835d283cb931566633e9551ddafd248d116348c65b1909d009fe0db32977be`
- Query-bundle hash: `5774657af144896634eba445a78991531eeeab12635e5f75b1e7753ee6337fce`
- Deterministic outputs: scorecard, taxonomy, claims, report Markdown, and figure source CSVs.
- Rendering tolerance: PNG bytes may vary across Matplotlib/font builds; source CSV hashes and captions must match.
- `repro.sh --mode rows` verifies every input hash, rebuilds outputs without SQL/GPU/inference, and checks expected hashes.
- `repro.sh --mode restore` starts an isolated SQL Server container, restores the final `.bak` pair, regenerates row exports, compares them byte-for-byte, then runs row-only reconstruction.
- Row-only gate: `PASS` (`cfddf486ec4fea5e94ff1cf47f059f5f23226f22d48bc5133c6a89e61f7e313d`).
- Fresh-restore gate: `PASS` (`843c0c7efa97f02bb2dd99300eaa663b17ed4e90708e476dd07d213af4c5df0e`).
