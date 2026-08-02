# Phase 4 parallel import inventory

**DEVELOPMENT / METHODS PROVENANCE — NO NATIVE SIDE EVENT IS A PHASE 4 EVENT**

State date: 2026-08-02. The acceptance contract is
`protocol/SIDE_TRACK_IMPORT_BUNDLE_CONTRACT.md`. A side result becomes
mainline evidence only after its complete source branch is ancestry-merged,
the strict normalized envelope and saved validation match a fresh replay, and
one `p4-import-*` event is appended from a clean mainline tree.

| Source | Source boundary | Strict validation | Target event | State |
|---|---|---|---|---|
| OLMo Bank-W early service | `d76e937d2e6294b92d3d599581bd0fb029f5735c` | 5 events / 11 outputs; saved validation SHA-256 `57d4ac60aeef972a26a19a4d2b159e93d62a53822b40a836fa27401c36ebccf2` | `p4-import-olmo-bank-w-capability-v1` | **VALIDATED / NOT REGISTERED** |
| Gemma terminal transport | source `b0425a441f1b87c33d9bb0b4d08d221942f11923`; ancestry merged to main at `c9021e5c2405eaa34a16a07ba0ea903c53b81ee6` | 5 events / 21 outputs; saved validation SHA-256 `143fda2e9d75c7cce0b0c3e38837d60a5ed05b7c191a29870e885a81ba1e79d3` | `p4-import-gemma-transport-v1` | **VALIDATED / NOT REGISTERED** |
| OLMo final lineage | terminal source `a28cdd54dda335daf55f468e5be8cc65b2fc5253`; ancestry merged through `aa6663a` / prep merge `b0f74d6` | 1 methods event / 13 outputs; saved validation SHA-256 `875ce846417e661df298153a5bd9f93ef8f1d647f6867764dc829476ea525e46` | `p4-import-olmo-lineage-final-v1` | **VALIDATED / NOT REGISTERED** |

## OLMo early service boundary

Normalized bundle:
`reports/OLMO_BANK_W_CAPABILITY_IMPORT_V1.json`, SHA-256
`d3c2f94e23251c64edcdd643e1ab13a94ad9263a7ccb6fe36c2c2e1a367b0536`.
Its six-event source-registry boundary is preserved byte-for-byte at
`reports/source_registries/olmo_bank_w_early_d76e937.jsonl`, SHA-256
`1e66b35068dc6489de10cccad206899a726d522872bec3f5fe3586aa0a20cbca`,
so later OLMo registry appends cannot invalidate the early admission record.
Its selected source outputs independently verify. Qwen, OLMo Think, and OLMo
Instruct pass independently; the exact common support is 16/20. The source
and mainline algorithms agree that the service is blocked. Import is still a
required provenance action, never a license to alter that decision or open a
Bank-W intervention.

## Gemma terminal boundary

Native release:
`/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802/release/IMPORT_BUNDLE_PHASE4.json`,
SHA-256
`005532754166644e42a369358565b9ce72235151e64559a9f12254d987ff7729`.
Strict normalized envelope:
`reports/GEMMA_TRANSPORT_IMPORT_V1.json`, SHA-256
`e946a59ba25aa02ec0415625ee9863d2a3f9f6226f02842266d720760431ad56`.

The importable result is a methods blocker. The selected actual-model replay
is bit-identical, but the precommitted all-slot forward/fallback tangent
relative error is 0.002458 versus the 1e-5 ceiling. The source stops G2/G3 and
licenses no mechanism, nondifferentiability, late-band, workspace,
confirmatory, or replication conclusion. Phase 4 does not adopt even that
methods statement until its target import event is registered.

## OLMo terminal boundary

Native release:
`/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801/release/IMPORT_BUNDLE_PHASE4.json`,
SHA-256
`a2486ec5a4759a1f5b21643e7c60766824c48f13ff43240d458ba72147165a2a`.
Strict normalized envelope:
`reports/OLMO_LINEAGE_IMPORT_V1.json`, SHA-256
`be1870d0b4a562a314809171313957d12e8b176a97189429355b576915a8ba09`.

The terminal source event is methods-only and registers thirteen outputs: the
state and claims records, JSON/Markdown import bundle, live inventory,
environment lock, isolated TeX/PDF paper, and five registered PDF figures.
The source terminal verifier passes 24 live events / 101 outputs against exact
registry prefix `db3fe202...e80a`. It preserves the 16/20 Bank-W service block,
the descriptive O2/O3 results, the no-identifiable-O5-estimand disposition,
and all forbidden-use boundaries. It opens no new scientific cell or
intervention. The strict Phase-4 validation selects only this self-verifying
terminal methods event and rehashes all thirteen outputs. Registration remains
queued behind the A1000 branch. DriveFS uploads are still rate-limited, so the
exact local recovery staging must remain preserved until a fresh cloud remount
proves durability.

## Admission order and invariants

1. Do not touch the live A1000 worktree to land an import.
2. After the frozen A1000 branch resolves, integrate the prospective Phase 4
   preparation and register each ready strict bundle from a clean tree.
3. Re-run the imported Bank-W joint-support calculation from the three
   registered capability artifacts.
4. Import any later OLMo final release only after it is terminal, reviewed as
   development/methods evidence, normalized, and freshly validated.
5. Re-run the pre-freeze inventory and two-pass durability audit after the
   registry is final.

The Phase 4 registry rejects native `ol-*` and `gm-*` creation. Side-track
commits never fill independent-review or PI fields, and imports never upgrade
their source tier.
