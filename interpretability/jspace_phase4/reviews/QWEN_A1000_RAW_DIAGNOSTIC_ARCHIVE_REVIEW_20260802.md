# Qwen A1000 raw-diagnostic archive review

**OPEN ENGINEERING DEVIATION — INDEPENDENT REVIEW REQUIRED — NOT A FREEZE OR
APPROVAL RECORD**

Date: 2026-08-02. Evidence target:
`p4-qwen-lens-fit-drawA-n1000-dev-v1`.

## Binding criterion and observed archive

The governing Phase 4.3 plan asks the A1000 registration review to account for
all 1,000 per-prompt diagnostic rows, with every row finite or covered by an
already precommitted skip rule. The frozen producer prints those diagnostics
to its terminal but does not store them inside the checkpoint.

The materialized archive on VM13 is continuous for prompts 181--1000 once the
final fit completes:

- prompts 181--250 are in `qwen_fit_drawA_n250_20260801.log`;
- prompts 251--500 are in `qwen_continuation_draw_a_n500_20260801.log`;
- prompts 501--554 and 717--1000 are in
  `qwen_continuation_draw_a_n1000_20260801.log`;
- prompts 555--716 are recovered from the locally retained Codex tool-output
  transcript for the active VM13 session. The admissible immutable prefix is
  lines 1--7,073: 18,232,092 bytes, SHA-256
  `e142b280e01622e2d4e4214083804de26f204c4fdfe6db05f656d0218536a943`.
  Its diagnostic-only extract covers exactly 162 unique prompts and hashes to
  `6adb27484077dd577fffa5da9efd0a1db48d65c8faaff1979babee57bc960838`.
  Later transcript output is excluded because it displays test fixtures that
  intentionally resemble diagnostics; reading the entire append-only session
  would contaminate provenance.

Exact prefix copies are preserved at:

```text
/content/sl4_work/qwen_fit_diagnostics/provenance/codex_tool_output_prefix_lines_1_7073.jsonl
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/diagnostics/qwen_a1000_fit/provenance/codex_tool_output_prefix_lines_1_7073.jsonl
```

Both copies were rehashed to the prefix digest above. DriveFS presence on this
VM is a cache-local copy, not fresh cloud-materialization proof.

This yields 820 distinct raw rows. Terminal text for prompts 1--180 was
emitted on an earlier VM and is not present in the mounted run root, local
work roots, or retained Codex JSONL sessions available on VM13. Exact local
searches for representative first-block diagnostic strings returned no
candidate file. No producer `skipping prompt` record exists, and no skip rule
is being invented after the fact.

## What the final checkpoint can and cannot prove

After A1000, the independent engineering checks must establish all of the
following from fresh hashes and tensor bytes:

- the checkpoint header and tensor payload both have
  `n_done == next_idx == 1000`;
- the fit contract, nested corpus, model snapshot, runtime, and `jlens`
  bindings are unchanged;
- source layers 0--62, target layer 63, and d_model 5120 are exact;
- every fp32 cumulative tensor and registered fp16 lens tensor is finite;
- every fp16 lens layer is bit-exactly
  `(checkpoint_jacobian_sum / 1000).to(float16)`;
- the registered event, output hashes, and content-addressed local backup
  agree.

Those invariants prove that the frozen estimator accepted all 1,000 ordered
nested-prefix prompts and that no prompt was silently skipped or trimmed. A
nonfinite contribution would persist in its cumulative tensor. They do **not**
recreate the missing prompts' printed sequence lengths, valid-token counts,
wall times, Jacobian norms, or running-mean changes. Therefore the project may
report 1,000 accepted finite contributions, but must not report a complete
1,000-row raw diagnostic archive.

## Required disposition

The frozen producer may create its append-only development event when its
implemented registration contract passes. This memo does not silently turn
the separate raw-log archival criterion green. The Phase 4.4 reviewer should
choose one of two honest dispositions:

1. recover the exact earlier terminal text from a genuinely new archival
   source and rehash/rebuild the normalized archive; or
2. accept an explicit deviation in which the 820-row raw archive plus the
   stronger checkpoint/tensor invariants satisfy the scientific integrity
   intent while raw per-prompt QA remains incomplete for prompts 1--180.

Do not rerun prompts 1--180 and label the new diagnostics as original text,
do not synthesize values from the cumulative checkpoint, and do not weaken or
reinterpret any frozen A1000 decision threshold. The fit and every successor
remain development-only pending this review and the other freeze gates.

## Final closeout fields

| field | value |
|---|---|
| normalized raw rows | **PENDING; expected 820** |
| raw missing indices | **PENDING; expected 1--180 only** |
| producer skip rows | **PENDING; expected zero** |
| checkpoint acceptance invariant | **PENDING** |
| tensor-integrity audit | **PENDING** |
| exact source/extract hashes | **PENDING** |
| independent disposition | **PENDING — implementation agent must not sign** |
