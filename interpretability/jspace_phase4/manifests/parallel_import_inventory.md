# Phase 4 parallel import inventory

**DEVELOPMENT / METHODS PROVENANCE — NO NATIVE SIDE EVENT IS A PHASE 4 EVENT**

State date: 2026-08-03. Admission follows
`protocol/SIDE_TRACK_IMPORT_BUNDLE_CONTRACT.md`. All selected source outputs
were freshly rehashed, normalized, registered by one Phase 4 import event, and
copied to local content-preservation storage before the next transaction.

| Mainline event | Source commit / boundary | Tier | Selected outputs | Admission result and claim boundary |
|---|---|---|---:|---|
| `p4-import-olmo-bank-w-capability-v1` | `d76e937d2e6294b92d3d599581bd0fb029f5735c`; early release | side-development-import | 14 event outputs, including 11 strict source outputs plus envelope/validation/registry boundary | Capability only. OLMo Think and Instruct pass independently; no intervention outcome. Registered at `bcf537e...`. |
| `p4-bank-w-capability-joint-imported-dev-v1` | Binds the early import plus registered Qwen baseline | phase4-development | 3 | Fresh mainline replay: 16/20 common-capable families, `baseline_capability_ready=false`, no intervention. Producer `922ef98...`; event `fddab68...`. |
| `p4-import-gemma-transport-v1` | `b0425a441f1b87c33d9bb0b4d08d221942f11923`; terminal state of record | side-development-import | 24 event outputs, including 21 strict source outputs plus envelope/validation/source registry | Methods blocker only. Selected-slot tangents are bit-identical; frozen all-slot relative error 0.002458 fails the 1e-5 ceiling. No mechanism, nondifferentiability, late-band, workspace, confirmatory, replication, or intervention claim. Registered at `6ac62d2...`. |
| `p4-import-olmo-lineage-final-v1` | `a28cdd54dda335daf55f468e5be8cc65b2fc5253`; terminal methods release | side-development-import | 16 event outputs, including 13 strict terminal outputs plus envelope/validation/source registry | O1 remains blocked at 16/20; O2 is consistent with broadly conserved capacity recruitment; O3 is a dictionary-formation development pattern; O5 has no identifiable estimand. No new scientific cell or intervention. Registered at `a8e218c...`. |

The exact source envelopes and saved validations remain:

- OLMo early bundle `d3c2f94e...`, validation `57d4ac60...`.
- Gemma terminal bundle `e946a59b...`, validation `143fda2e...`.
- OLMo terminal bundle `be1870d0...`, validation `875ce846...`.

Bank W's load intervention is not lawfully runnable as a cross-model Phase 4
primary at current capability support; it routes to Phase 5B as per-model
estimation.

The Phase 4 registry contains no native `ol-*` or `gm-*` evidence IDs. Imports
do not fill reviewer/PI fields, upgrade tiers, alter 16/20, or authorize a
confirmatory or replication outcome.
