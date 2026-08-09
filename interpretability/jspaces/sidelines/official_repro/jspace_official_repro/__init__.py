"""jspace_official_repro — Anthropic released-materials reproduction, Study 1.

Runs the released Jacobian-lens evaluations and experiment prompt sets on
Qwen 3.6 27B (published lens) and OLMo 3.1 32B Instruct (prospective
official-estimator fit), then separates prompt, lens, intervention,
capability, and model effects.

Governing plan: `interpret/jspace_lab_official_repro_1.md` plus its
addendum (Drive). Evidence prefix: ``or1-``. Everything scientific goes
through :mod:`jspace_official_repro.registry`.
"""

__version__ = "0.1.0"

STUDY_ID = "jspace-official-repro-1"
EVIDENCE_PREFIX = "or1-"
