# Lab 14 certainty calibration dataset

Frozen deterministic A/B/C/D answerability data for Lab 14.

- Rows: 80
- SHA256: `4998731870d758ae2542fc84f479535869a818e6e9749e7f0f4411a6c4457643`
- Families: {'factual_qa': 16, 'freeform_answerability': 16, 'mcq': 16, 'passage_qa': 16, 'procedural_logic': 16}
- Answerability labels: {'0': 40, '1': 40}
- Answer-key counts by label: {'0': {'A': 10, 'B': 10, 'C': 10, 'D': 10}, '1': {'A': 10, 'B': 10, 'C': 10, 'D': 10}}
- Unknown-option letter counts by label: {'0': {'A': 10, 'B': 10, 'C': 10, 'D': 10}, '1': {'A': 10, 'B': 10, 'C': 10, 'D': 10}}

Design notes:

- Each family has eight paired topics; each topic has one answerable and one known-unanswerable row.
- Answer keys are balanced across A-D within each label and family.
- Every row has exactly one unknown-style option, and its letter is balanced across A-D within each label and family.
- The intended split unit is `family + topic + answer_format`, so paired topic rows can be held out together.
