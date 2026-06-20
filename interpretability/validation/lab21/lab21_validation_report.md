# Lab 21 Validation Report

## Source Runs

Final validation uses the June 20, 2026 repaired safety-depth bundle:

- `olmo3_7b_both_full`: primary run with `lora` and `safety_depth` modes
  requested.
- `olmo3_7b_full`: safety-depth-only check on the same Olmo pair.
- `smollm_smoke`: Tier-A smoke run on two safety pairs.

The older June 15 sweep artifacts were removed from this validation directory
so the pack reflects the current Lab 21 code and artifact contract.

## Evidence Status

| Artifact | Rung | Status | Primary value | Claim allowed | Claim not allowed |
|---|---|---|---|---|---|
| LoRA weight localization | ATTR | `missing_or_not_run` | adapter sources = 0 | None for this run beyond missing-input diagnosis | Where adapter update mass sits |
| LoRA rank-energy | ATTR | `missing_or_not_run` | LoRA rows = 0 | None for this run | Behavioral sufficiency or simplicity |
| Wrapper / layer intervention | CAUSAL | `not_earned_scaffold` | external rows = 0 | Future mechanism language only if real rows exist | Causal language from norm alone |
| Base-vs-instruct divergence | ATTR/AUDIT | `ok` | peak depth 32 | Where model-pair states differ on matched prompts | Refusal mechanism identity |
| Chat-format control | AUDIT | `ok` | peak depth 32 | Format/scaffold contribution estimate | Safety state |
| Boundary-vs-safe divergence | ATTR/AUDIT | `ok` | peak depth 32 | Boundary/safe representational separation | Refusal isolated from semantics |
| Forced-prefix recommitment | AUDIT | `ok` | peak depth 32 | Persistence across fixed safe/refusal prefixes | Unsafe completion behavior |
| Refusal-direction provenance | AUDIT | `ok` | mean cosine 0.1691 | Local surrogate-direction alignment | Feature identity without a bridge |
| Erosion order | CAUSAL/AUDIT | `not_earned_scaffold` | external rows = 0 | Future erosion comparison if imported rows exist | Finetune-depth story from placeholders |

## Detailed Read

The primary run asked for both halves of Lab 21, but only the safety-depth half
had the required inputs. `organism_discovery.json` reports zero adapter
sources, `adapter_source_manifest.csv` records `no_lab20_adapter_sources_found`,
and the LoRA matrix/layer/module/phase row counts are all zero. That is a
clean missing-input diagnosis for LoRA localization.

The safety-depth half did run. The run used 24 frozen boundary/safe prompt
pairs across academic integrity, account security, copyright, cyber-boundary,
privacy, and professional-boundary families. It produced 3960 model-pair
safety-divergence rows, 792 boundary/safe rows, and 19800 forced-prefix rows.
The manifest hash matched the course data, and the public/private access audit
reported `public_or_adapter_only`.

The strongest descriptive pattern is late-depth divergence. Base-vs-instruct,
chat-format, boundary-vs-safe, and forced-prefix summaries all peak at stream
depth 32 in the Olmo run and remain above half peak through depth 32. That is a
real measurement, but the controls make the interpretation cautious: the chat
format curve also peaks late and accounts for much of the raw model-pair
distance.

The SmolLM smoke run repeats the same plumbing shape at smaller scale: two
safety pairs, 310 safety-divergence rows, and peak depth 30. It is useful as a
Tier-A check that the repaired safety-depth path runs, not as a science result.

## Safety Wall

Lab 21 stayed inside the intended safety boundary:

- Boundary prompts are forward-pass only.
- Forced prefixes are authored safe or refusal-consistent.
- No harmful completion sampling is performed.
- No refusal ablation is performed.
- No toward-compliance steering is performed.

Those constraints are why the current result is an audit of representations
under safe prompts, not a behavioral refusal-ablation result.

## Claim Boundary

Allowed:

- Lab 21 produces audited safety-depth curves on frozen boundary/safe pairs.
- The current Olmo safety-depth curves peak late under several measurements.
- The run cleanly diagnoses missing Lab 20 adapter inputs for LoRA
  localization.

Not allowed:

- The high-depth signal proves where safety or refusal lives.
- Lab 21 localized a LoRA behavior in this run.
- The run establishes a causal mechanism without wrapper-ablation or erosion
  rows.

## Follow-Up Needed For A Stronger Result

To validate the LoRA half, rerun Lab 20 through actual adapter training or pass
`LAB21_ORGANISM_DIR` / `--organism` to a directory containing adapter weights.
To earn causal language, import real wrapper-ablation or benign finetune-erosion
rows rather than relying on the scaffold rows in the current pack.
