# INCIDENT or1-002 — BOS text shifts char-offset span mapping on OLMo
# raw renders

**Discovered:** 2026-08-09 11:34Z, when the ignition cell's span audit
(correctly) hard-stopped the OLMo battery. **Registered output
affected:** none — the only affected banked file was unregistered.

## What happened

The span finders (`find_token_span`, `position_before_substring`, and
selectivity-language's inline question-span walk) located character
offsets in the RAW prompt text but walked token boundaries in the
DECODED stream. Upstream `HFLensModel.encode` forces
`add_bos_token=True`; OLMo's BOS decodes as `<|endoftext|>` (22 chars),
so on OLMo raw renders every decoded prefix is 22 chars longer than the
raw offset — spans landed ~4–6 tokens early. Qwen has no BOS: all Qwen
cells are unaffected. OLMo chat renders embed their special tokens in
the render text consistently: unaffected. OLMo cells using only token
positions (core swaps, evals, capacity, linecount full-span):
unaffected.

## Blast radius

- `ignition` (OLMo): crashed at the audit before any output — no data.
- `selectivity-language` (OLMo): banked with a mis-anchored question
  span (started ~5 tokens early, inside the passage tail), **not yet
  registered**. Quarantined as
  `selectivity_language_olmo.json.defective-or1-002` (bytes preserved);
  the cell reruns under the fixed finder before Group A registration.
- Groups A(qwen)/B(both)/C(capacity both): verified unaffected.

## Correction (prospective)

All char-offset location now happens in the decoded token stream
(single source of truth for offsets and walk), with the existing
decode-back audits retained. Regression test added
(`test_render_goldens.py::test_span_finders_survive_bos_prefix`).
The OLMo battery reruns after the current chain drains (banked cells
skip; the quarantined cell and ignition/top-down run fresh).
