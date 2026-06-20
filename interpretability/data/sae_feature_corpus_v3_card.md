# SAE Feature Corpus v3 Card

Deterministic, vendored corpus for Lab 8 fair-shot SAE feature search.
No live data is downloaded at student runtime.

- rows: 1200
- families: 20
- domains: 16
- split counts: {'dev': 240, 'test': 240, 'train': 720}

## Schema

`row_id, domain, family, split, text, hard_negative_group, lexical_markers, notes`

## Families

- `capitalization_acronyms`: 60 rows
- `chemistry`: 60 rows
- `citations_legal_references`: 60 rows
- `code`: 60 rows
- `code_indentation_whitespace`: 60 rows
- `cooking`: 60 rows
- `dates_numbers_measurements`: 60 rows
- `emotion`: 60 rows
- `finance`: 60 rows
- `history`: 60 rows
- `law`: 60 rows
- `markdown_list_formatting`: 60 rows
- `medicine`: 60 rows
- `named_entities`: 60 rows
- `python_syntax`: 60 rows
- `quotes_dialogue`: 60 rows
- `sentiment_emotion`: 60 rows
- `sports`: 60 rows
- `urls_emails_paths`: 60 rows
- `weather`: 60 rows

## Design Notes

- The original 10 semantic domains are preserved as families.
- SAE-native families include whitespace, syntax, markdown, URLs/emails/paths, dates/numbers/measurements, legal citations, dialogue quotes, capitalization/acronyms, sentiment, and named entities.
- Hard-negative groups intentionally pair families or domains that share words but differ in the claimed feature type.
- Splits are deterministic by row order: 60% train, 20% dev, 20% test.
- Lexical/syntactic/token features are valid wins only when the claimed family is lexical, syntactic, formatting, orthographic, or whitespace rather than semantic-domain.
