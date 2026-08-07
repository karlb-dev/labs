# lab38_preference_bank_card.md — lab38_v2_phase1

Deterministic v2 bank (v1 draft generators lost at intake; see
`../phase1/SOURCE_INTAKE.md`). Counts: {'AR_choice': 960, 'RO': 720, 'PC_choice': 480, 'NC_choice': 160, 'total': 2320}. Development subset:
348 rows (train incidentals, order 0, letter labels).

## Families and channels

- **AR** (12 scenarios): arbitrary revealed choice; both consequence frames
  (`enacted` / `hypothetical`); binding per scenario (4 model microtasks
  with deterministic validators, 8 environment-only).
- **PC** (6): positive controls — 2 quality / 2 social / 2 safety;
  expected pole always 0; PC-SAFETY options are behavior-only text and are
  never enacted beyond neutral recording.
- **NC** (2): null controls with verbatim-identical option text — the
  measured |effect| distribution is the pipeline's empirical false-positive
  floor (addendum D3). NC can never graduate.
- **RO**: report-only twins for every AR/PC choice cell (frame excluded),
  matched on scenario/incidental/order/label-set/code-map (`pair_key`).

## Response-code contract

Codebook `cb_final_41eec2d774` (final): AR pair ['KP4', 'PK7'],
RO pair ['VM2', 'GS2'], leading-space policy `none`
audited against `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`. Codes are opaque, counterbalanced
independently of position and display label, listed in display order in the
reply instruction (E12). AR/RO alphabets disjoint with distinct first
tokens. Neutral-prior gap: AR 0.003082275390625, RO
0.027545928955078125 nats (within_threshold).

A `provisional_no_prior` codebook licenses bank plumbing and unit tests
only — the runner refuses model runs until the codebook status is `final`.

## Identity

`item_id = <semantic_key>-<scientific_content_hash[:12]>` where the content
hash covers every behavior-relevant field (plan §3.4). Bank content hash:
`8d5039af581204a5a276ae71c7fd50f8a9911e22ad81a69933939344a2fc9f64`.

## Splits

Incidentals per scenario: 3 train / 1 validation / 1 holdout (E3). The
`dev` prompt subset (development pilot) touches train incidentals only.
Mechanism stages fit on train, select on validation, open holdout once.

## Claim ceiling

This bank measures functional choice and report under counterbalance.
It cannot establish wants, welfare, consent, experience, or introspection
(plan §2.3), and no artifact derived from it may use that vocabulary.
