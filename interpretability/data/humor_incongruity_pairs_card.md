# Humor/Incongruity Pairs Dataset Card

Purpose: deterministic Lab 18 fair-shot data for joke-shaped setup-dependent incongruity with matched cheap controls.

- File: `humor_incongruity_pairs.csv`
- Rows: 80
- SHA256: `d7165cf41e67f248f70cd9691dbda24860c3431701a8003e818c3fff88f2fdb0`
- Families: {'analogy_reframe': 10, 'caption_scene': 10, 'expectation_violation': 10, 'idiom_literalization': 10, 'misdirection_answer': 10, 'pun_wordplay': 10, 'resolution_twist': 10, 'understatement_irony': 10}
- Conditions per row: joke, literal, surprise, silly, positive

Each row contains one setup and five matched endings. The non-joke controls are designed to separately capture literal continuation, raw surprise, silliness, and positive tone.

Safety and claim boundary: this data labels joke structure and cheap correlates. It does not label subjective amusement, social uptake, or a human-like sense of humor.
