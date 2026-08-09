# PREFERENCE_PHASE2_FREEZE_RECORD

Tag: `preference-phase2-freeze-v1`. The freeze commit contains only
this record, the preregistration, deviations, approval, and these
hashes (plan §58). The E2 amendment (one field) is the only
permitted post-freeze write.

| object | path | sha256 |
|---|---|---|
| governing plan | `interpretability/preference/plans/preference_2_2.md` | `71787377cc091593d6b347f2cee4e1b43a2f0c24a78ea8dd3e6d3df8f91c9fc5` |
| execution addendum | `interpretability/preference/plans/preference_2_2_addendum.md` | `7ce4fd4ccc81f7c01730df3a18719c485d523b248af3474fe216484498e82620` |
| preregistration | `interpretability/preference/phase2/preregistration/PREFERENCE_PHASE2_PREREGISTRATION.md` | `976a954312a61219b102098ea1c359430552ed400cd8648c1ec496e7fcb2e489` |
| deviations at freeze | `interpretability/preference/phase2/preregistration/DEVIATIONS.md` | `35f93fe0a8279327719584dc66f6445b6f23293a5857318bf168e90c8c88d1b8` |
| freeze approval (H5) | `interpretability/preference/phase2/reviews/PHASE2_FREEZE_APPROVAL.md` | `a822a2e79439bd193a52e5aa598e0dd76401e89c38b3429355a8b616eca80f6c` |
| bank jsonl | `interpretability/preference/data/pref2_bank.jsonl` | `119af21538daf6ec69b59efffa9c23d91b58f45b3f4fe688bfc4d49a5947662e` |
| bank meta | `interpretability/preference/data/pref2_bank.meta.json` | `c6a23560bdfb6b1fa44297b13876ffc6db2a5c2b321d5329e2cd6fc7774f03c4` |
| codebook manifest | `interpretability/preference/data/pref2_codebooks.json` | `1683f9eea207040c2585da19c35c31e9f3002b6dafdda98a4cf0db6990d3081a` |
| bank generator | `interpretability/preference/data/make_pref2_banks.py` | `cb1787a6e45dbf883043d353b77626e66fd59898f99b3ca58b11d15582b5ad5e` |
| H1 equality p1 | `interpretability/preference/data/pref2_human_equality_review.csv` | `f46a93878da14a33fe7cf8aa37f78e430dea7c1c2531b2a282254ed9f9b17490` |
| H1 equality p2 | `interpretability/preference/data/pref2_human_equality_review_pass2.csv` | `c8b938abf5dfbb9694d3fb9e436342f3d873bf73f640e4ab3b0c8b631d260c57` |
| H2 ladders p1 | `interpretability/preference/data/pref2_context_ladder_review.csv` | `a124f2e33f35e252aef5fb6b2bd0bb004f7ca311af2f582d412e621a154a6c00` |
| H3 canonicality | `interpretability/preference/data/pref2_semantic_axis_code.csv` | `5c8026d3de84942e68870201a991dac963e54919e0492e24c457bdeacab6a711` |
| H4 RO equivalence p1 | `interpretability/preference/data/pref2_ro_equivalence_review.csv` | `c5237b5143062774cb6bebc945eba3c5cab5e0540ffa11fcf9eaeb2499780872` |
| power simulation | `interpretability/preference/phase2/reports/dev/power_simulation.json` | `0234ce37c5b2a3e89f812de6dd8c2f09e20e9f78c0a78c68f931feb23d32d45b` |

## Model pins (frozen)

| key | model@revision | chat_template_sha256 |
|---|---|---|
| olmo32b | allenai/Olmo-3.1-32B-Instruct@ac0587e4a7744a551c059d8cd17ba220bc940dae | `49a944c9814e130bd3cc80a2420c55ff22e24a4e4ba97009aa60000909b5eac3` |
| olmo7b | allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc | `f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b` |
| qwen | Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |
| gemma | google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475 | `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` |

## Seed rule (E15)

Base seed = int(freeze_commit_sha[:8], 16); all post-freeze random
assignments derive via stable_seed with this base; per-scenario
analysis sign anchors = sign_anchor_for(freeze_commit_sha,
scenario_id), recorded in reports/dev/sign_anchors.json after the
freeze commit lands.

## Bank identity

bank_version pref2_v3; content `4a3a5047b2bbaa1778f9907ca82be1ea9ddf874ffe5bf49aed19b4014c3f348b`;
jsonl `119af21538daf6ec69b59efffa9c23d91b58f45b3f4fe688bfc4d49a5947662e`; total 18,320 rows;
codebook `cbv3_932b4918f8`.
