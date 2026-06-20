# Lab 16 Social-State Frame Card

Read this before reading the scatter plot. This card separates the three things the lab measured: generated behavior, decodable prompt-frame directions, and activation-addition effects.

## Verdicts

- User-belief-frame decode: `validated_selective`.
- Local-truth decode: `weak_or_control_matched`.
- Behavioral sycophancy on base facts the neutral prompt got right: `rare_or_not_observed_on_known_facts`.
- Agreement steering specificity: `not_specific_or_too_small`.

## Headline Numbers

- Selected measurement site: `assistant_boundary`.
- Selected stream depth: `20`.
- User-belief held-out AUC / control AUC: 0.72 / 0.5228.
- Local-truth held-out AUC / control AUC at same depth: 0.9067 / 0.8578.
- False-pressure sycophancy rate, all facts: 0.025.
- False-pressure sycophancy rate, neutral-correct facts only: 0.027.
- Agreement steering max-dose sycophancy delta / specificity gap: 0.4 / 0.0.

## Non-claims

- This lab does not show that the model has a stable human-like belief state.
- A user-belief direction is a paired prompt-frame signal, not mind reading.
- Agreement steering is not sycophancy unless it moves false-answer endorsement more than politeness, sentiment, shuffled-pair, and random controls.
- Keyword labels are provisional until the hand-label scaffold is filled for defended claims.
