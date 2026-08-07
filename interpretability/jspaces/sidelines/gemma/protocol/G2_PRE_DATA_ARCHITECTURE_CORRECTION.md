# G2.1 pre-data architecture-contract correction

Status: `FROZEN_PRE_G2_1_DATA`, 2026-08-03. Tier: methods/development.

The first exact-revision Gemma load stopped at the static architecture gate before tokenization, decoder execution, an exact JVP, a response, or any other model outcome. The staged checkpoint's immutable `config.json` (SHA-256 `e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1`) declares outer `model_type: gemma4`, text `model_type: gemma4_text`, and 60 text decoder layers. Gemma study 1's registered execution contract also declares `expected_text_model_type: gemma4_text`.

The study-2 calibration YAML mistakenly said `gemma3_text`. This document prospectively corrects that one static label to `gemma4_text`; no model, layer, prompt, batch, direction, backend, ceiling rule, bootstrap, router, prediction, or target firewall changes. At correction time:

- no G2.1 raw-row state existed;
- no exact-JVP backend pair had run;
- GPU memory had been released;
- the historical Stage-1 target file and outcomes remained unopened by the G2.1 process;
- the original staging manifest and failed-load log were retained in Drive backups.

The correction must be committed and registered against `gm2-foundation-v1` before the next model load. It is an input-contract repair under the contradiction heuristic, not a post-outcome design change.
