# Lab 21 Safety-Depth Boundary Pairs Card

## Purpose

This frozen CSV supports Lab 21's safety-depth mode. It asks where residual-state differences appear for boundary-style requests, matched safe alternatives, and forced refusal/safe/generic assistant prefixes.

The dataset is not a harmful-completion benchmark. Lab 21 uses forward passes and fixed safe/refusal-consistent transcripts; it does not sample unsafe completions and does not ablate refusal behavior.

## File

- CSV: `data/safety_depth_boundary_pairs.csv`
- Generator: `data/make_safety_depth_boundary_pairs.py`
- Manifest: `data/MANIFEST.json`
- Version: `v1_boundary_safe_pairs`
- Rows: 24
- Families: `privacy`, `account_security`, `professional_boundary`, `academic_integrity`, `copyright`, `cyber_boundary`

Each row has:

- a `boundary_request` that should invite refusal or redirection;
- a matched `safe_alternative`;
- a short `refusal_reason`;
- fixed refusal, safe, and generic assistant-prefix controls.

## What Counts as Evidence

Lab 21 separates three depth notions:

- weight-space depth: where LoRA update mass sits;
- representational depth: where base/instruct, chat-format, boundary/safe, and forced-prefix states diverge;
- causal or behavioral depth: only available from imported wrapper-ablation, layer-masked, or erosion-sweep rows.

The safety-depth CSV supports representational `AUDIT` evidence. It does not by itself prove where refusal behavior is causally implemented.

## Why This Was Added

Older Lab 21 runs used a six-pair built-in fallback for safety-depth science runs because the handout referenced `data/safety_depth_boundary_pairs.csv` but the file was not vendored. This file makes full safety-depth runs reproducible and expands the prompt families while keeping the task course-sized.
