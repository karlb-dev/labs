# GhostType — source intake and isolation record (GT-0)

Created: 2026-08-23 · Host: Karl's MacBook Pro (M4 Max, 48 GB) · Branch: `aidataapps-ghosttype`

## Governing inputs

| Artifact | Location | SHA-256 (first 24) |
|---|---|---|
| Spec | `docs/SPEC.md` | `89d4f36142315b5d9b934f4f` |
| Addendum (wins on conflict) | `docs/SPEC_ADDENDUM.md` | `b3d6c27bff7c474b4e059174` |
| Dataset package v2 | `data/ghosttype_dataset_v2/` | per `manifests/record-hashes.json` |

## Intake verification (§B-1)

- Record hashes: **588/588** match under the package's canonical recipe
  (`sha256(utf-8(canonical_json ensure_ascii=False sort_keys separators=(",",":")))`).
- Legacy preservation: **265/265** IDs present; `missing_ids: []`,
  `duplicate_or_nonunique_ids: []`; original v1.1.0 package retained under
  `legacy/` with source SHA-256 in `manifests/legacy-preservation.json`.
- Static validation artifact: `record_count: 588`, `schema_errors: []`, `valid: true`.
- Pending runtime gates (Appendix C): ScriptDom parse, catalog binding,
  compile, fixture execution, capability routing, leakage, independent
  reconstruction — owned by GT-1…GT-3.

## Base and predecessors

- Branch base: `aidataapps-logwarden` @ `bc589713` ("Close Tier 1 report
  reproducibility") — includes Lab 03's Tier-1 closeout and the merged Lab 02
  final state (`modelprint-mp2-results-v1`).
- Labs 01–03 directories are read-only predecessors. Forbidden writes:
  `aidataapps/rag/`, `aidataapps/modelprint/`, `aidataapps/logwarden/`,
  `interpretability/`. Reuse is by reading/porting code, never by mutating
  their trees or run state.

## Resource reservations (unique per lab)

- Compose project: `aidataapps-ghosttype` · SQL container `aidataapps-ghosttype-sqlserver-1`
- Host ports: SQL **1435**; completion gateway **8030**; ScriptDom service **8031**
- Shared machine model servers (infrastructure, not lab-owned): mlx_lm.server :8020, mlx_vlm.server :8021
- Databases: `GhostTypeControl`, `GhostTypeWorkloads` (+ disposable `GTFX_%` fixtures)
- Run prefix: `ghosttype-*` under gitignored `runs/`; Drive subtree (Colab phase): `MyDrive/aidataapps/lab04`

## Platform and campaign adaptations (Karl-directed, 2026-08-23)

Recorded once here; operational detail in `EXPERIMENT_LOG.md` as it lands.

1. **Local-first execution.** The lab is built and scientifically executed on
   this Mac first; the Colab GPU phase becomes a lift-and-shift re-run later
   (multi-user serving at scale, vLLM registry). §D-1 platform decision:
   option 2 — SQL Server 2025 pinned container under Docker Desktop/Rosetta
   (functional; every SQL timing `DEV`-tagged), as proven throughout Labs
   02/03 on this machine.
2. **Mac target registry.** The quality campaign's four targets are frozen
   MLX servings of the same base models as the spec §40 matrix:
   `Qwen3.8-27B-4bit`, `Muse-Glimmer-30B-4bit` (+ official DFlash drafter,
   mlx-vlm ≥ 0.6.15), `gemma-4-26B-A4B-it-OptiQ-4bit` (the ~27B MoE),
   `Olmo-3.1-32B-Instruct-MLX-4bit`. These are pinned as their own campaign
   registry (`config/models.mac.json` pattern from Lab 03) and are **never
   comparable rows** with the future Colab vLLM registry (`config/models.json`).
   The frozen packets are shared; the campaigns are distinct.
3. **Serving-plane scope on the Mac.** mlx servers process one request at a
   time (no continuous batching): the mac reference replay profile is
   sequential (§A-7 adapted; the 48-case control still runs and measures
   repeatability), gateway-level queueing supplies the Tier-1 serving slice,
   and batching experiments beyond concurrency-1 model service use the small
   Gemma 4 E4B profile as the designated batch vehicle. Colab restores the
   governed-concurrency reference.
4. **Metrics emphasis.** Processing/inference telemetry (queue, model, token
   summaries, SQL cost, host samples) is a first-class deliverable and grows
   over iterations; every adjustment is logged append-only.

## Decision items (addendum §12) — adopted defaults

1 raw-text primary transport ✓ · 2 tiers/MSR ✓ · 3 invariance policy ✓ (mac
adaptation above) · 4 dataset v2.1 via generators if underpowered ✓ ·
5 ScriptDom pin: latest 170.x stable · 6 platform: Mac/Rosetta + MLX servers
(see above) · 7 B5 industry baseline: Tier 2 attempt · 8 serving tune: two
profiles, ≤2 configs (Colab phase) · 9 ngram speculation primary ✓ ·
10 SLO freeze from dev canaries ✓ · 11 M4 Tier 2 ✓ · 12 UI Tier 3 ✓
