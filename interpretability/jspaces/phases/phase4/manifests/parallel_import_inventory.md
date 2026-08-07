# Phase 4 parallel import inventory

**DEVELOPMENT / METHODS PROVENANCE — NO NATIVE SIDE EVENT IS A PHASE 4 EVENT**

State date: 2026-08-03; updated 2026-08-04 with the Phase 4.5 terminal
Study-2 admissions. Admission follows
`protocol/SIDE_TRACK_IMPORT_BUNDLE_CONTRACT.md`; the Study-2 layer uses the
dedicated closeout importer
`experiments/p4_import_sidelines_study2.py`. All selected source outputs
were freshly rehashed, normalized, registered by one Phase 4 import event, and
copied to local and Drive content-preservation storage before the next
transaction.

| Mainline event | Source commit / boundary | Tier | Selected outputs | Admission result and claim boundary |
|---|---|---|---:|---|
| `p4-import-olmo-bank-w-capability-v1` | `d76e937d2e6294b92d3d599581bd0fb029f5735c`; early release | side-development-import | 14 event outputs, including 11 strict source outputs plus envelope/validation/registry boundary | Capability only. OLMo Think and Instruct pass independently; no intervention outcome. Registered at `bcf537e...`. |
| `p4-bank-w-capability-joint-imported-dev-v1` | Binds the early import plus registered Qwen baseline | phase4-development | 3 | Fresh mainline replay: 16/20 common-capable families, `baseline_capability_ready=false`, no intervention. Producer `922ef98...`; event `fddab68...`. |
| `p4-import-gemma-transport-v1` | `b0425a441f1b87c33d9bb0b4d08d221942f11923`; terminal state of record | side-development-import | 24 event outputs, including 21 strict source outputs plus envelope/validation/source registry | Methods blocker only. Selected-slot tangents are bit-identical; frozen all-slot relative error 0.002458 fails the 1e-5 ceiling. No mechanism, nondifferentiability, late-band, workspace, confirmatory, replication, or intervention claim. Registered at `6ac62d2...`. |
| `p4-import-olmo-lineage-final-v1` | `a28cdd54dda335daf55f468e5be8cc65b2fc5253`; terminal methods release | side-development-import | 16 event outputs, including 13 strict terminal outputs plus envelope/validation/source registry | O1 remains blocked at 16/20; O2 is consistent with broadly conserved capacity recruitment; O3 is a dictionary-formation development pattern; O5 has no identifiable estimand. No new scientific cell or intervention. Registered at `a8e218c...`. |
| `p4-import-gemma-transport-study2-v1` | `aba2e01460dde32e5c2ca1478a5502950e2448ec`; terminal Study-2 bundle, frozen prefix `2a144bcf...` | side-development-import | 43 event outputs: bundle JSON/MD, saved validation, frozen registry-prefix snapshot, 12 release artifacts, and all 8 admitted events' outputs | Methods-only calibrated relicense. Target-blind pooled ceiling 0.07870368901355948; preserved all-slot error 0.0024581113830208778 in-envelope; selected slot bit-identical; five-layer classifier a closed finite-scale methods result. Historical Study-1 blocker preserved. Depends on `p4-import-gemma-transport-v1`. |
| `p4-import-olmo-lineage-study2-v1` | `80213290125a56ad75bd9a23a638211a0dc1c618`; terminal Study-2 bundle, frozen prefix `0a8973e0...` | side-development-import | 56 event outputs: bundle JSON/MD, saved validation, frozen registry-prefix snapshot, 12 release artifacts, and all 12 admitted events' outputs | Methods-only boundary closure. Capability-gated SFT/DPO wedge (effects missing, not zero); H6 in-band failure with Think-only L56 epsilon-0.10 late anchor; registered-dose coverage unavailable, not zero; Bank-W pair power 0.7788 at 16 families (first passing count 18). Depends on `p4-import-olmo-lineage-final-v1` and the Gemma Study-2 admission. |

The exact source envelopes and saved validations remain:

- OLMo early bundle `d3c2f94e...`, validation `57d4ac60...`.
- Gemma terminal bundle `e946a59b...`, validation `143fda2e...`.
- OLMo terminal bundle `be1870d0...`, validation `875ce846...`.
- Gemma Study-2 bundle `9ef48b8a...` (payload `1751d22a...`), validation
  `reports/gemma_transport_study2_import_validation_v1.json`.
- OLMo Study-2 bundle `c213dc74...` (payload `b5cfdf92...`), validation
  `reports/olmo_lineage_study2_import_validation_v1.json`.

Bank W's load intervention is not lawfully runnable as a cross-model Phase 4
primary at current capability support; it routes to Phase 5B as per-model
estimation.

The Phase 4 registry contains no native `ol-*`, `ol2-*`, `gm-*`, or `gm2-*`
evidence IDs. Imports do not fill reviewer/PI fields, upgrade tiers, alter
16/20, or authorize a confirmatory or replication outcome.
