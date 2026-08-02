# J-space OLMo lineage side study

This package is the isolated development/methods track for the OLMo 32B
lineage. It starts from the Phase 4 import boundary `3b041735d8b842de46a9c0a474fccd0c44e0841a`
and runs on branch `interp_jspace_olmo_lineage`.

It must never write the Phase 3, Phase 4, or Gemma registries, run roots,
reports, figures, or preregistrations. The only pre-freeze Phase 4 service
outputs are the two OLMo Bank-W baseline capability gates and their early,
hash-pinned import bundle. Every native result is `development` or `methods`.

The durable run root is:

```text
/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801
```

Bootstrap and verify:

```bash
bash interpretability/jspace_olmo_lineage/repro.sh
```

Register the clean foundation:

```bash
python -m jspace_olmo_lineage.experiments.foundation \
  --config interpretability/jspace_olmo_lineage/configs/ol_foundation_v1.yaml
```

The live scientific narrative is maintained in
`reports/OLMO_LINEAGE_DEVELOPMENT_REPORT.md`; recovery state is maintained in
`reports/INPROGRESS_OLMO_LINEAGE.md`, with the stable restart procedure in
`reports/OLMO_LINEAGE_RESUME.md`. The three documents are mirrored below the
Drive run root by:

```bash
python -m jspace_olmo_lineage.recovery
```

Heavy and resumable artifacts live only in the side-study Drive root.

Inventory official 32B post-training stages without downloading weights or
opening model outcomes:

```bash
python -m jspace_olmo_lineage.experiments.checkpoint_inventory \
  --config interpretability/jspace_olmo_lineage/configs/ol_checkpoint_inventory_v2.yaml
```

Version 1 is an immutable conservative result that used byte-identical
tokenizer files. Version 2 supersedes it with full token-ID, normalized-BPE,
processing-component, and frozen-corpus encoding checks while retaining
BOS/chat-template differences as explicit qualifications.

Audit whether the registered evidence can identify the O5 crossed causal
decomposition without substituting structural proxies:

```bash
python -m jspace_olmo_lineage.experiments.o5_feasibility \
  --config interpretability/jspace_olmo_lineage/configs/ol_o5_feasibility_v1.yaml
```

Freeze and register the exact Phase 4 Bank-W compatibility contract before
opening either OLMo baseline:

```bash
python -m jspace_olmo_lineage.experiments.bank_w_capability \
  --config interpretability/jspace_olmo_lineage/configs/ol_bank_w_capability_v1.yaml \
  --freeze-protocol
```

After committing and pushing that registry event, stage and run one model at a
time from a clean tree:

```bash
bash interpretability/jspace_olmo_lineage/run_bank_w_capability_model.sh \
  olmo31-think
bash interpretability/jspace_olmo_lineage/run_bank_w_capability_model.sh \
  olmo31-instruct
```
