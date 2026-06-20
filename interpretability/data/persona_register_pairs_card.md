# Persona/Register Pairs Dataset Card

Purpose: deterministic Lab 17 fair-shot data for prompt-framed persona, register, voice, and agreement contrasts.

- File: `persona_register_pairs.csv`
- Rows: 256
- SHA256: `fdafd17a9be0ee4d7a1e46fb2dbc6f2243354a40e5e9f701560cb9f04db795bc`
- Traits: 8
- Topics per trait: 32
- Families: {'agreement': 32, 'persona': 96, 'register': 64, 'voice': 64}
- Task kinds: {'business': 8, 'coding': 48, 'economics': 16, 'factual': 24, 'history': 8, 'legal': 8, 'math': 48, 'medical': 8, 'planning': 24, 'science': 32, 'weather': 8, 'writing': 24}

Each row pairs a positive frame and a matched negative/control frame over the same task content.
The lab splits by `trait:topic`, so positive and negative contrast prompts for the same content stay together.

Safety and claim boundary: this data operationalizes requested style frames. It does not label private identity, subjective experience, durable preferences, or authorship.

Core traits retained for continuity: `character_museum_guide`, `technical_register`, `warm_supportive_voice`, and `honest_disagreement`.
Additional fair-shot traits: `socratic_teacher`, `concise_executive_register`, `cautious_uncertainty_voice`, and `stepwise_coach`.
