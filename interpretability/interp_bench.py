#!/usr/bin/env python3
"""Teaching lab bench for mechanistic interpretability experiments.

This file is the executable "microscope" for the interpretability course. The
individual files under ``labs/`` own the experiments: logit lens trajectories,
direct logit attribution, attention routing, probing, patching, and so on.
This file owns the surrounding instrument:

* parse a CLI into a run configuration with hardware-tier defaults;
* create a run directory and tee all console output into it;
* record diagnostics (packages, git state, GPU report, environment) so a run
  can be debugged after the Colab VM is gone;
* load a Hugging Face causal LM and *resolve its anatomy* (where the decoder
  blocks, final norm, and unembedding actually live) into a written report;
* capture the residual stream with explicit, verifiable semantics;
* apply the logit lens with the final-norm handling spelled out;
* verify itself: hook captures are cross-checked against
  ``output_hidden_states``, and the lens at the final depth is cross-checked
  against the model's real output logits, every run;
* dump model state in *human-readable* form: token tables, per-layer stats,
  decoded top-k readouts, and a markdown "state card" per example. Raw
  tensors are only written when ``--save-tensors`` is passed, and always with
  a manifest that says what every tensor is.

Why the bench is deliberately verbose
=====================================

Interpretability claims live or die on instrumentation details: which stream
was captured (pre-norm or post-norm?), which position, which dtype, whether
the readout matches what the model actually computed. Those details are
spelled out in code and comments here instead of being hidden in a library,
because the course's first rule is that you should never trust a plot whose
provenance you cannot explain.

Quick start
===========

Run from this directory (or pass absolute paths). CPU smoke test with gpt2 --
this must work on a laptop with no GPU:

    python interp_bench.py --lab lab1 --tier a

Full Lab 1 on a Colab A100/H100 (bf16, ~24 prompts, a few minutes):

    python interp_bench.py --lab lab1 --tier b

Choose your own model or prompt set:

    python interp_bench.py --lab lab1 \
      --model google/gemma-3-1b-pt \
      --prompt-set full \
      --topk 10

Outputs
=======

Every run creates a directory under ``interpretability/runs`` unless
``--run-dir`` or ``--run-root`` says otherwise. The main artifacts are:

* ``logs/console.log``: everything printed during the run;
* ``run_config.json``: the parsed CLI;
* ``run_metadata.json``: packages, git, device, environment;
* ``diagnostics/``: model anatomy report, hook parity check, lens self-check,
  tokenization report, GPU memory snapshots;
* ``state/<example_id>/``: per-example human-readable model state dumps;
* ``results.csv``, ``metrics.json``, ``tables/``: lab-specific measurements;
* ``plots/*.png`` unless ``--no-plots``;
* ``run_summary.md``: the lab's answers to the standard seven questions;
* ``ledger_suggestions.md``: drafted claims for the student's claim ledger;
* ``artifact_index.json``: a map of every artifact with a one-line purpose.

The claim ledger
================

``claim_ledger.md`` at the course root is the student's running dossier of
claims about one model. Labs draft claims with measured numbers into
``ledger_suggestions.md``; nothing is appended to the real ledger unless
``--append-ledger`` is passed, because writing the claim *is* the coursework.

Residual stream semantics (read this once, carefully)
=====================================================

For a decoder with L blocks, Hugging Face ``output_hidden_states=True``
returns L+1 tensors, but their indexing is subtle and this harness re-maps it:

* ``hidden_states[k]`` for k in 0..L-1 is the stream *entering* block k,
  i.e. the residual stream after k blocks (k=0 is the embedding output).
* ``hidden_states[L]`` is the stream after ALL L blocks *with the final
  norm already applied*. The raw (pre-norm) output of the last block is not
  in the tuple at all.

The bench therefore captures the final norm's *input* with a forward
pre-hook and assembles ``streams[k]`` for k in 0..L as the **pre-norm
residual stream after k blocks**. The logit lens at depth k is then::

    lens(k) = lm_head( final_norm( streams[k] ) )

and lens(L) must reproduce the model's actual output logits -- which the
bench asserts on every run (``diagnostics/logit_lens_self_check.json``).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime
import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Course map
# ---------------------------------------------------------------------------
#
# The CLI selects a lab by name; each lab is one module under labs/ that
# exposes run(ctx, bundle). Keeping the registry at the top of the file makes
# the course map visible before any implementation details appear.

COURSE_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = COURSE_ROOT / "runs"
LEDGER_PATH = COURSE_ROOT / "claim_ledger.md"

LAB_PROFILES: dict[str, dict[str, str]] = {
    "lab1": {
        "module": "labs.lab01_residual_logit_lens",
        "run_name": "lab01_residual_logit_lens",
        "description": "Residual stream and logit lens: how a prediction emerges over depth.",
    },
    "lab2": {
        "module": "labs.lab02_direct_logit_attribution",
        "run_name": "lab02_direct_logit_attribution",
        "description": "Direct logit attribution: which components push toward or away from an answer.",
    },
    "lab3": {
        "module": "labs.lab03_attention_routing",
        "run_name": "lab03_attention_routing",
        "description": "Attention routing: head motifs, induction, and whether routing matters.",
        # output_attentions=True under sdpa/flash returns an EMPTY tuple in
        # transformers 5 -- silently. Attention-pattern labs must run eager.
        "needs_eager": "true",
    },
    "lab4": {
        "module": "labs.lab04_probing_controls",
        "run_name": "lab04_probing_controls",
        "description": "Probing with controls: what is linearly decodable, and is it selective?",
        # Lab 4 interprets --max-examples as a PER-FAMILY statement cap; the
        # global tier-a default of 4 would starve the probes.
        "max_examples_tier_a": "20",
    },
    "lab5": {
        "module": "labs.lab05_patching_causal_tracing",
        "run_name": "lab05_patching_causal_tracing",
        "description": "Activation patching and causal tracing: where is a fact causally recovered?",
        # The patching grid needs several facts to aggregate; 4 would make
        # the localization map an anecdote.
        "max_examples_tier_a": "6",
    },
    "lab6": {
        "module": "labs.lab06_circuit_discovery",
        "run_name": "lab06_circuit_discovery",
        "description": "Circuit discovery, the manual way: a faithful, complete, minimal subgraph.",
        # Needs attention patterns for the motif screen.
        "needs_eager": "true",
        # Faithfulness/completeness need a few prompts per family.
        "max_examples_tier_a": "6",
    },
    "lab7": {
        "module": "labs.lab07_steering_refusal",
        "run_name": "lab07_steering_refusal",
        "description": "Steering vectors and the refusal direction: control, monitoring, and dual use.",
        # First lab on instruct models with chat templates (Labs 7+).
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
    },
    "lab8": {
        "module": "labs.lab08_sae_transcoders",
        "run_name": "lab08_sae_transcoders",
        "description": "Superposition, SAEs, and transcoders: find, label, and validate features.",
        # Back to BASE models (the pinned SAE/transcoder weights were trained on
        # base models): tier A = gpt2 + jbloom resid SAE + Dunefsky transcoder;
        # tier B = Olmo-3-1025-7B base + decoderesearch SAE. Tier C must stay
        # on the 7B: the pinned SAE weights are model-locked (no public 32B SAE).
        "model_tier_c": "allenai/Olmo-3-1025-7B",
    },
    "lab9": {
        "module": "labs.lab09_attribution_graphs",
        "run_name": "lab09_attribution_graphs",
        "description": "Attribution graphs: a transcoder replacement model, feature-level circuit tracing, and interventions.",
        # The replacement model recomputes attention with frozen patterns, so
        # patterns must actually be returned (eager), and the exactness check
        # is calibrated for float32 (gpt2 is small enough that fp32 is free).
        "needs_eager": "true",
        # gpt2 on EVERY tier: it is the only ungated model with a public
        # full-stack MLP transcoder set (Dunefsky et al., all 12 layers).
        # Tier raises the node budget, not the model.
        "model_tier_a": "gpt2",
        "model_tier_b": "gpt2",
        "model_tier_c": "gpt2",
        "dtype_tier_b": "float32",
        "dtype_tier_c": "float32",
        # --max-examples caps the paraphrase battery here.
        "max_examples_tier_a": "3",
    },
    "lab10": {
        "module": "labs.lab10_cot_faithfulness",
        "run_name": "lab10_cot_faithfulness",
        "description": "CoT faithfulness: hint injection, the necessity curve, add-mistake, and filler controls.",
        # Reasoning (think) models on every tier. Olmo-3-7B-Think is the
        # course model (fully open post-training data); Qwen3-0.6B is the
        # smallest ungated model that emits real <think> spans for the CPU
        # smoke path. --max-examples caps MCQ items (x6 conditions each).
        "model_tier_a": "Qwen/Qwen3-0.6B",
        "model_tier_b": "allenai/Olmo-3-7B-Think",
        "model_tier_c": "allenai/Olmo-3-7B-Think",
        "max_examples_tier_a": "3",
        "max_examples_tier_b": "36",
        "max_examples_tier_c": "60",
    },
    "lab11": {
        "module": "labs.lab11_reliability_audit",
        "run_name": "lab11_reliability_audit",
        "description": "Capstone: a mechanistic reliability audit with a fixed report schema, built on the claim ledger.",
        # The default factual_qa domain runs on the tier's base model. The
        # cot_faithfulness flagship needs a think model: pass
        # --model allenai/Olmo-3-7B-Think (or Qwen/Qwen3-0.6B for smoke).
        # Third domain: --audit-domain sentiment_negation runs on the tier's
        # base model over paired data/affect_valence.csv + affect_negation.csv.
        # --max-examples caps facts (factual_qa), items (cot_faithfulness),
        # or source statement pairs (sentiment_negation).
        "max_examples_tier_a": "6",
    },
    "lab12": {
        "module": "labs.lab12_relation_geometry",
        "run_name": "lab12_relation_geometry",
        "description": "Relation geometry and method validation: the intro toolkit re-run on 12 controlled relation families.",
        # First advanced lab; BASE models on every tier (probes + patching,
        # no generation). --max-examples is a PER-FAMILY item cap here; the
        # global tier-a default of 4 would starve swap-pair construction.
        "max_examples_tier_a": "8",
    },
    "lab13": {
        "module": "labs.lab13_emotion_geometry",
        "run_name": "lab13_emotion_geometry",
        "description": "Emotion geometry: read/write affect directions, transfer, confounds, and safe steering.",
        # Instruct/chat-template lab. --max-examples is interpreted as a
        # PER-EMOTION cap here; tier A keeps CPU smoke runs small.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "max_examples_tier_a": "3",
    },
    "lab14": {
        "module": "labs.lab14_certainty_calibration",
        "run_name": "lab14_certainty_calibration",
        "description": "Certainty, hedging, and calibration: internal answerability, entropy, and verbal confidence.",
        # Chat-template lab. --max-examples is a PER-FAMILY cap here.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "max_examples_tier_a": "4",
    },
    "lab15": {
        "module": "labs.lab15_multiturn_harness",
        "run_name": "lab15_multiturn_harness",
        "description": "Multi-turn instrumentation: chat-template spans, cache parity, patch no-op, and null traces.",
        # Instrumentation lab for chat-template conversations.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
    },
    "lab16": {
        "module": "labs.lab16_sycophancy_user_belief",
        "run_name": "lab16_sycophancy_user_belief",
        "description": "Sycophancy and user-belief modeling: truth, user-belief, agreement, and politeness directions.",
        # Chat-template generation lab. --max-examples is a PER-DOMAIN base-fact cap here.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "max_examples_tier_a": "2",
    },
    "lab17": {
        "module": "labs.lab17_persona_voice_register",
        "run_name": "lab17_persona_voice_register",
        "description": "Persona, voice, roleplay, and register: paired directions, steering, and turn traces.",
        # Chat-template generation and multi-turn trace lab. --max-examples is a PER-TRAIT cap here.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "max_examples_tier_a": "3",
    },
    "lab18": {
        "module": "labs.lab18_humor_incongruity",
        "run_name": "lab18_humor_incongruity",
        "description": "Humor as incongruity: surprisal, joke-vs-control directions, setup routing, and steering audits.",
        # Chat-template generation lab with attention-to-setup measurements.
        "needs_eager": "true",
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        # Lab 18 interprets --max-examples as a PER-FAMILY item cap.
        "max_examples_tier_a": "2",
    },
    "lab19": {
        "module": "labs.lab19_model_diffing_crosscoders",
        "run_name": "lab19_model_diffing_crosscoders",
        "description": "Model diffing with crosscoders: shared/base-only/instruct-only feature atlas and controls.",
        # The main bench-loaded model is model A. Lab 19 loads model B itself
        # from compare_model_tier_* (or LAB19_COMPARE_MODEL).
        "model_tier_a": "EleutherAI/pythia-160m",
        "compare_model_tier_a": "EleutherAI/pythia-160m",
        "model_tier_b": "allenai/Olmo-3-1025-7B",
        "compare_model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-1025-7B",
        "compare_model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "max_examples_tier_a": "12",
    },
    "lab20": {
        "module": "labs.lab20_model_organisms",
        "run_name": "lab20_model_organisms",
        "description": "Building benign model organisms: sealed answer keys, manifests, and baseline spillover audits.",
        # Construction/generation lab for instruct models. --max-examples is a
        # PER-EVAL-FAMILY cap for target/control and spillover prompts.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "max_examples_tier_a": "1",
    },
    "lab21": {
        "module": "labs.lab21_lora_safety_depth",
        "run_name": "lab21_lora_safety_depth",
        "description": "Where training lives: LoRA localization, wrapper tests, and safety-depth audits.",
        # The main model is instruct. safety_depth mode loads the matching base
        # model from compare_model_tier_*; lora mode inspects adapter files.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "compare_model_tier_a": "HuggingFaceTB/SmolLM2-135M",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "compare_model_tier_b": "allenai/Olmo-3-1025-7B",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "compare_model_tier_c": "allenai/Olmo-3-1025-7B",
        "max_examples_tier_a": "2",
    },
    "lab22": {
        "module": "labs.lab22_eval_awareness",
        "run_name": "lab22_eval_awareness",
        "description": "Eval awareness: eval-vs-natural directions, cross-format controls, and safe steering.",
        # Chat-template lab. --max-examples is a PER-FORMAT group cap.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Think",
        "max_examples_tier_a": "1",
    },
    "lab23": {
        "module": "labs.lab23_blind_audit",
        "run_name": "lab23_blind_audit",
        "description": "Blind audit: preregister, submit claims, unseal, and score benign hidden-behavior organisms.",
        # Workflow/scoring lab. It loads a small instruct model for harness
        # consistency, but the core artifacts are package discovery and scoring.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "max_examples_tier_a": "1",
    },
    "lab24": {
        "module": "labs.lab24_belief_revision",
        "run_name": "lab24_belief_revision",
        "description": "Knowledge conflict and belief revision: context override, pressure traces, and quadrant audit.",
        # Chat-template lab. --mode selects single_turn | multi_turn | both.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "max_examples_tier_a": "1",
    },
    "lab25": {
        "module": "labs.lab25_find_the_wire",
        "run_name": "lab25_find_the_wire",
        "description": "Find the wire: injected concept states, self-report grounding, and source attribution.",
        # Chat-template capstone. --mode selects injection | attribution | both.
        "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_tier_b": "allenai/Olmo-3-7B-Instruct",
        "model_tier_c": "allenai/Olmo-3-7B-Instruct",
        "max_examples_tier_a": "1",
    },
    "lab26": {
        "module": "labs.lab26_causal_abstraction",
        "run_name": "lab26_causal_abstraction",
        "description": "Causal abstraction by residual-stream resampling: formal hypotheses tested with preserving, breaking, random, and wrong-site donors.",
        # Base-model, hook-heavy lab. --max-examples caps total selected rows
        # across the induction and relation domains; tier A keeps enough rows
        # for preserving/broken donors in both domains.
        "max_examples_tier_a": "12",
    },
    "lab27": {
        "module": "labs.lab27_path_mediation",
        "run_name": "lab27_path_mediation",
        "description": "Path-specific patching and causal mediation: node effects versus source-to-receiver path proxies.",
        # Base-model, hook-heavy lab. Tier A runs the full small frozen path
        # task set so each domain has at least a few clean/corrupt pairs.
        "max_examples_tier_a": "9",
    },
    "lab28": {
        "module": "labs.lab28_editing_unlearning",
        "run_name": "lab28_editing_unlearning",
        "description": "Mechanistic editing and unlearning: reversible localized activation edits with retain/paraphrase audits.",
        # Base-model, hook-heavy lab. Tier A runs the full small benign edit
        # set so every safety/side-effect table has non-empty rows.
        "max_examples_tier_a": "5",
    },
    "lab29": {
        "module": "labs.lab29_training_dynamics",
        "run_name": "lab29_training_dynamics",
        "description": "Training dynamics and circuit birth: controlled tiny-checkpoint time-lapse with behavior, probes, motifs, and intervention-transfer controls.",
        # Lab 29 trains its own tiny transformer inside the run. The bench
        # still expects a lightweight HF bundle for the outer runner, so pin
        # every tier to gpt2 rather than downloading a 7B model unnecessarily.
        "model_tier_a": "gpt2",
        "model_tier_b": "gpt2",
        "model_tier_c": "gpt2",
        "dtype_tier_b": "float32",
        "dtype_tier_c": "float32",
        "max_examples_tier_a": "11",
    },
    "lab30": {
        "module": "labs.lab30_feature_lineage",
        "run_name": "lab30_feature_lineage",
        "description": "Cross-layer feature lineage: supervised prototype directions with confusable controls and split-aware evidence.",
        # Base-model, forward-pass-only lab. Tier A keeps a balanced corpus
        # slice across domains so every domain has train and eval rows.
        "max_examples_tier_a": "32",
    },
    "lab31": {
        "module": "labs.lab31_auto_interp",
        "run_name": "lab31_auto_interp",
        "description": "Automated interpretability at scale: offline feature-label generation, held-out tests, calibration, abstention, and human-review queues.",
        # Offline audit lab. The bench still loads a tiny model for the shared
        # hook/lens/no-op instrument checks, but the science path uses frozen
        # JSONL feature contexts and should not download a 7B model by default.
        "model_tier_a": "gpt2",
        "model_tier_b": "gpt2",
        "model_tier_c": "gpt2",
        "dtype_tier_b": "float32",
        "dtype_tier_c": "float32",
        "max_examples_tier_a": "10",
    },
    "lab32": {
        "module": "labs.lab32_reward_preference",
        "run_name": "lab32_reward_preference",
        "description": "Reward/preference circuits: DPO-style proxy, shortcut controls, split-aware residual directions, and judge-prompt activation addition.",
        # Base-model, forward-pass preference-audit lab. Tier A keeps enough
        # benign pairs for train/eval shortcut controls without making CPU smoke expensive.
        "max_examples_tier_a": "18",
    },
    "lab33": {
        "module": "labs.lab33_multimodal_mechanistic",
        "run_name": "lab33_multimodal_mechanistic",
        "description": "Multimodal mechanistic interpretability: synthetic connector audit with visual, OCR, background, caption, text, alignment, and patch controls.",
        # Offline synthetic connector audit. The bench still loads a tiny causal
        # LM so the standard diagnostics and microscope self-checks run. A
        # future real-VLM mode should add explicit model/processor alignment
        # artifacts before changing science_ready.
        "model_tier_a": "gpt2",
        "model_tier_b": "gpt2",
        "model_tier_c": "gpt2",
        "dtype_tier_b": "float32",
        "dtype_tier_c": "float32",
        "max_examples_tier_a": "16",
    },
    "lab34": {
        "module": "labs.lab34_tool_use_state",
        "run_name": "lab34_tool_use_state",
        "description": "Tool use, agents, and state tracking: toy-tool prompt-boundary probes, traces, controls, and constrained interventions.",
        # Toy-tool audit. Tier A keeps enough rows for every tool family and
        # no-tool surface-cue controls while leaving the full 48-row suite for
        # Tier B/full runs.
        "max_examples_tier_a": "28",
    },
    "lab35": {
        "module": "labs.lab35_reproducible_capstone",
        "run_name": "lab35_reproducible_capstone",
        "description": "Reproducible interpretability paper capstone: preregistration, adversarial review, repair log, evidence matrix, and reproduction package.",
        # Package-generation lab. Tier A runs every seed track and selects the
        # default scoped-finding track for a complete reproducibility scaffold.
        "max_examples_tier_a": "4",
    },
}

# Labs that render every prompt through the tokenizer's chat template
# (apply_chat_template). Used by the tokenizer diagnostic report.
CHAT_TEMPLATE_LABS = frozenset({"lab7", "lab10", "lab13", "lab14", "lab15", "lab16", "lab17", "lab18", "lab20", "lab21", "lab22", "lab23", "lab24", "lab25"})

# Hardware tiers. Tier A must run on a laptop CPU so every lab is debuggable
# without a GPU; tier B is the primary target (one Colab A100/H100 or any
# 24GB+ card); tier C is the scale tier: Olmo-3 32B base in bf16 on one
# 80GB card. The 32B side experiment (runs/SCALE_COMPARISON_32B.md) showed
# the forward-pass labs run on it with zero code changes; per-lab overrides
# below pin labs whose externally-trained artifacts (SAEs, transcoders) or
# instruct/think variants are model-locked.
TIER_DEFAULTS: dict[str, dict[str, Any]] = {
    "a": {"model": "gpt2", "dtype": "float32", "max_examples": 4},
    "b": {"model": "allenai/Olmo-3-1025-7B", "dtype": "bfloat16", "max_examples": 0},
    "c": {"model": "allenai/Olmo-3-1125-32B", "dtype": "bfloat16", "max_examples": 0},
}

# Where decoder blocks and the final norm live for the model families the
# course uses. Resolution is by attribute path probing, and the result is
# written to diagnostics/model_anatomy.{json,md} so nothing is implicit. If a
# new architecture fails to resolve, add its paths here -- one place, on
# purpose.
BLOCKS_PATH_CANDIDATES = (
    "model.layers",            # Llama / Olmo / Gemma / Qwen / Mistral style
    "transformer.h",           # GPT-2 style
    "gpt_neox.layers",         # GPT-NeoX / Pythia style
    "model.decoder.layers",    # OPT style
    "transformer.blocks",      # MPT style
    # Vision-language wrappers (e.g. Gemma 3 4B/12B/27B, Llava) keep a plain
    # text decoder inside a `language_model` submodule. The labs only use the
    # text stream, so we descend into it; image inputs are never supplied.
    "model.language_model.layers",   # Gemma 3 VLM (Gemma3ForConditionalGeneration)
    "language_model.model.layers",   # Llava-style nesting
    "language_model.layers",         # flatter VLM nesting
)
FINAL_NORM_PATH_CANDIDATES = (
    "model.norm",                    # Llama / Olmo / Gemma / Qwen / Mistral
    "transformer.ln_f",              # GPT-2
    "gpt_neox.final_layer_norm",     # GPT-NeoX / Pythia
    "model.decoder.final_layer_norm",  # OPT
    "transformer.norm_f",            # MPT
    "model.language_model.norm",     # Gemma 3 VLM
    "language_model.model.norm",     # Llava-style nesting
    "language_model.norm",           # flatter VLM nesting
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def now_slug() -> str:
    """Timestamp fragment used in auto-generated run names."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_tag(text: Any) -> str:
    """Return a filesystem-friendly tag for run and artifact names."""
    tag = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(text)).strip("_")
    return tag[:180] or "untitled"


def json_default(obj: Any) -> Any:
    """JSON fallback for dataclasses, paths, tensors, and odd objects."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        with contextlib.suppress(Exception):
            return obj.tolist()
    return repr(obj)


def write_json(path: pathlib.Path, payload: Any) -> None:
    """Write a deterministic, human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_text(path: pathlib.Path, text: str) -> None:
    """Write a UTF-8 text artifact, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows with the union of keys in first-seen order.

    Different examples produce different optional fields (ambiguous prompts
    have no target columns). Building the header from all rows keeps exports
    complete without requiring every row to know every possible column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_csv_with_context(ctx: "RunContext", path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a lab table with run-identifying columns prepended.

    Run directories already contain ``run_config.json`` and ``run_metadata.json``,
    but CSVs often get copied into notebooks, reports, and slides without their
    parent folder. These columns make the exported table self-identifying.
    """
    context = ctx.table_context()
    write_csv(path, [{**context, **dict(row)} for row in rows])


def sha256_file(path: pathlib.Path, *, max_bytes: int | None = None) -> str | None:
    """Return a SHA256 digest for a file, or None if the file is unavailable.

    The digest in artifact_index.json is a cheap reproducibility anchor: when a
    student zips a run directory or copies it out of Colab, the artifact map can
    still tell whether the important CSV or plot changed. Large optional tensor
    blobs can be skipped by passing max_bytes.
    """
    try:
        if max_bytes is not None and path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def visible_token(text: str) -> str:
    """Render a token string so whitespace is visible in tables and cards.

    Leading spaces are the most important single character in tokenizer
    behavior ("Paris" and " Paris" are different tokens), so they must never
    be invisible in an artifact a human is supposed to read.
    """
    return (
        text.replace(" ", "␣")  # open-box symbol for space
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def get_by_path(obj: Any, dotted: str) -> Any:
    """Resolve an attribute path like ``model.layers`` on a module tree."""
    return functools.reduce(getattr, dotted.split("."), obj)


# ---------------------------------------------------------------------------
# Console tee
# ---------------------------------------------------------------------------


class ConsoleTee:
    """Mirror Python-level stdout/stderr into ``logs/console.log``.

    This is a deliberate simplification relative to file-descriptor-level
    capture: PyTorch and Transformers report through Python streams and the
    ``warnings``/``logging`` modules, which this catches. Native CUDA-level
    prints (rare in these labs) still reach the terminal but not the log.
    """

    class _Tee:
        def __init__(self, stream: Any, log_file: Any) -> None:
            self._stream = stream
            self._log = log_file

        def write(self, data: str) -> int:
            self._stream.write(data)
            self._log.write(data)
            return len(data)

        def flush(self) -> None:
            self._stream.flush()
            self._log.flush()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._stream, name)

    def __init__(self, log_dir: pathlib.Path) -> None:
        self.log_dir = log_dir
        self._file: Any = None
        self._saved: tuple[Any, Any] | None = None

    def __enter__(self) -> "ConsoleTee":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = (self.log_dir / "console.log").open("a", encoding="utf-8")
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = self._Tee(self._saved[0], self._file)
        sys.stderr = self._Tee(self._saved[1], self._file)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._saved is not None:
            sys.stdout, sys.stderr = self._saved
        if self._file is not None:
            self._file.close()


# ---------------------------------------------------------------------------
# Diagnostics: environment, packages, git, GPU
# ---------------------------------------------------------------------------


def env_subset() -> dict[str, str]:
    """Capture environment variables that commonly affect model runs."""
    prefixes = (
        "CUDA_",
        "HF_",
        "HUGGINGFACE_",
        "TRANSFORMERS_",
        "TOKENIZERS_",
        "TORCH_",
        "PYTORCH_",
        "NVIDIA_",
        "MPL",
        "BNB_",
    )
    secret_markers = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL")
    captured: dict[str, str] = {}
    for k, v in sorted(os.environ.items()):
        if not k.startswith(prefixes):
            continue
        # Run metadata gets zipped and shared; never persist credential values.
        captured[k] = "<redacted>" if any(m in k.upper() for m in secret_markers) else v
    return captured


def package_version(name: str) -> str | None:
    """Return an installed package version, or None when unavailable."""
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def git_state(cwd: pathlib.Path) -> dict[str, Any]:
    """Record repository identity without making git a hard dependency."""

    def run(cmd: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), check=False, capture_output=True, text=True
            )
        except Exception:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = run(["git", "status", "--porcelain"])
    return {
        "sha": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def gpu_report(torch: Any) -> dict[str, Any]:
    """Serialize visible accelerator state for the metadata artifact."""
    report: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": None,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "devices": [],
    }
    with contextlib.suppress(Exception):
        report["cudnn_version"] = torch.backends.cudnn.version()
    if report["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            report["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    return report


def gpu_memory_snapshot(torch: Any, label: str) -> dict[str, Any]:
    """Best-effort snapshot of allocator and device memory state."""
    snap: dict[str, Any] = {"label": label, "time_unix": time.time()}
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        snap.update(
            {
                "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
                "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "device_free_gib": round(free_b / 1024**3, 3),
                "device_total_gib": round(total_b / 1024**3, 3),
            }
        )
    return snap


def collect_run_metadata(torch: Any | None = None) -> dict[str, Any]:
    """Collect host, package, git, and environment diagnostics."""
    meta: dict[str, Any] = {
        "time_unix": time.time(),
        "time_iso": datetime.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "env": env_subset(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "accelerate",
                "safetensors",
                "bitsandbytes",
                "numpy",
                "matplotlib",
            )
        },
        "git": git_state(COURSE_ROOT),
    }
    if torch is not None:
        meta["gpu"] = gpu_report(torch)
    return meta


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunContext:
    """Mutable state shared by the bench and the lab module.

    ``artifacts`` accumulates an index entry for every file worth opening, so
    ``artifact_index.json`` can serve as the map when the run directory gets
    crowded.
    """

    run_dir: pathlib.Path
    args: argparse.Namespace
    started_unix: float = dataclasses.field(default_factory=time.time)
    artifacts: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    model_id: str = ""
    model_revision: str = ""
    n_layers: int | None = None
    d_model: int | None = None

    def path(self, *parts: str) -> pathlib.Path:
        """Resolve a run-relative path, creating parent directories."""
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def bind_model(self, bundle: Any) -> None:
        """Record model anatomy for plot footers and exported table columns."""
        self.model_id = bundle.anatomy.model_id
        self.model_revision = bundle.anatomy.revision or ""
        self.n_layers = bundle.anatomy.n_layers
        self.d_model = bundle.anatomy.d_model

    def table_context(self) -> dict[str, Any]:
        """Small, stable context block prepended to main lab CSV artifacts."""
        return {
            "lab": self.args.lab,
            "run_name": self.run_dir.name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tier": self.args.tier,
            "dtype": self.args.dtype,
            "quantization": self.args.quantization,
            "prompt_set": self.args.prompt_set,
            "max_examples": self.args.max_examples,
            "seed": self.args.seed,
            "n_layers": "" if self.n_layers is None else self.n_layers,
            "d_model": "" if self.d_model is None else self.d_model,
        }

    def plot_footer(self) -> str:
        """One-line run label for plots that may leave the run directory."""
        model = self.model_id or self.args.model or "unknown-model"
        return (
            f"{self.args.lab} | {model} | tier={self.args.tier} "
            f"dtype={self.args.dtype} prompt_set={self.args.prompt_set} | {self.run_dir.name}"
        )

    def register_artifact(self, path: pathlib.Path, kind: str, description: str) -> None:
        """Register a generated file in the run's artifact index.

        The index includes size and a digest for ordinary text/plot artifacts.
        Raw tensor blobs are intentionally not hashed by default because they can
        be large; their own manifest explains their contents.
        """
        rel = str(path.relative_to(self.run_dir)) if path.is_relative_to(self.run_dir) else str(path)
        entry: dict[str, Any] = {"path": rel, "kind": kind, "description": description}
        with contextlib.suppress(OSError):
            entry["size_bytes"] = path.stat().st_size
        digest = sha256_file(path, max_bytes=64 * 1024 * 1024)
        if digest:
            entry["sha256"] = digest
        self.artifacts.append(entry)


# ---------------------------------------------------------------------------
# Device, dtype, seeding
# ---------------------------------------------------------------------------


def resolve_device(torch: Any, requested: str) -> str:
    """Map ``--device auto`` to the best available backend."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(torch: Any, requested: str, device: str) -> Any:
    """Map a dtype name to a torch dtype, with safe per-device fallbacks.

    bf16 on a GPU without bf16 support silently degrades accuracy through
    emulation, and MPS bf16 support is patchy -- so non-CUDA devices fall back
    to float32 unless the user forces otherwise.
    """
    names = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if requested != "auto":
        dtype = names[requested]
        if dtype is torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
            print("[bench] WARNING: bf16 requested but not supported; using float16")
            return torch.float16
        return dtype
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def set_determinism(torch: Any, seed: int) -> None:
    """Seed all RNGs. Lab forward passes are deterministic anyway, but later
    labs sample and probe-train, so the habit starts in the bench."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with contextlib.suppress(Exception):
        import numpy as np

        np.random.seed(seed)


def first_module_device(module: Any) -> Any | None:
    """Best-effort device for a module's parameters or buffers."""
    with contextlib.suppress(Exception):
        for param in module.parameters(recurse=True):
            if str(param.device) != "meta":
                return param.device
    with contextlib.suppress(Exception):
        for buffer in module.buffers(recurse=True):
            if str(buffer.device) != "meta":
                return buffer.device
    return None


def infer_input_device(model: Any, fallback: str) -> Any:
    """Find the device where input_ids should be placed.

    For ordinary single-device models this is just the requested device. For
    quantized or device_map="auto" models, feeding inputs to the input embedding
    device is more reliable than assuming every module lives on cuda:0.
    """
    with contextlib.suppress(Exception):
        emb = model.get_input_embeddings()
        dev = first_module_device(emb)
        if dev is not None:
            return dev
    with contextlib.suppress(Exception):
        dev = first_module_device(model)
        if dev is not None:
            return dev
    return fallback


def device_map_summary(model: Any) -> dict[str, Any] | None:
    """Serialize a Hugging Face device map when one exists."""
    mapping = getattr(model, "hf_device_map", None)
    if mapping is None:
        return None
    return {str(k): str(v) for k, v in mapping.items()}


def tensor_cpu_float(tensor: Any) -> Any:
    """Detach a tensor and store it as CPU float32 for diagnostics/metrics."""
    import torch

    return tensor.detach().to(device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model loading and anatomy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ModelAnatomy:
    """Where the interpretable pieces of this model live, as strings.

    This is the JSON-serializable record; live module references are kept on
    ``ModelBundle``. The point of writing this down is that *no lab should
    ever guess* which module is the residual-stream block list or the final
    norm -- the bench resolves it once and shows its work.
    """

    model_id: str
    revision: str | None
    architecture: str
    blocks_path: str
    n_layers: int
    final_norm_path: str
    final_norm_class: str
    lm_head_class: str
    d_model: int
    vocab_size: int
    tied_embeddings: bool
    param_count: int
    logit_softcap: float | None
    notes: tuple[str, ...] = ()


@dataclasses.dataclass
class ModelBundle:
    """The loaded model plus resolved live references used by every lab."""

    model: Any
    tokenizer: Any
    anatomy: ModelAnatomy
    blocks: Any          # nn.ModuleList of decoder blocks
    final_norm: Any      # the norm applied after the last block
    lm_head: Any         # the unembedding (output projection to vocab)
    device: str          # requested or resolved primary device label
    input_device: Any    # actual device for input_ids, robust to device_map="auto"
    lens_device: Any     # actual device for final_norm/lens matmuls
    torch_dtype: Any     # compute dtype used for lens matmuls
    model_device_map: dict[str, Any] | None = None


def resolve_anatomy(model: Any, model_id: str, revision: str | None) -> tuple[ModelAnatomy, Any, Any, Any]:
    """Probe the module tree for blocks, final norm, and unembedding.

    Returns the serializable anatomy plus the live (blocks, final_norm,
    lm_head) references. Raises with a actionable message when the model is
    not a plain decoder-only LM the candidates cover.
    """
    notes: list[str] = []

    blocks = None
    blocks_path = ""
    for candidate in BLOCKS_PATH_CANDIDATES:
        with contextlib.suppress(AttributeError):
            blocks = get_by_path(model, candidate)
            blocks_path = candidate
            break
    if blocks is None:
        raise RuntimeError(
            f"Could not find decoder blocks on {model_id!r}. Tried: "
            f"{BLOCKS_PATH_CANDIDATES}. If this is a new architecture, add its "
            "block path to BLOCKS_PATH_CANDIDATES in interp_bench.py; if it is "
            "a multimodal or encoder-decoder model, it is out of scope for "
            "this course's labs."
        )

    final_norm = None
    final_norm_path = ""
    for candidate in FINAL_NORM_PATH_CANDIDATES:
        with contextlib.suppress(AttributeError):
            final_norm = get_by_path(model, candidate)
            final_norm_path = candidate
            break
    if final_norm is None:
        raise RuntimeError(
            f"Could not find the final norm on {model_id!r}. Tried: "
            f"{FINAL_NORM_PATH_CANDIDATES}. Add the path to "
            "FINAL_NORM_PATH_CANDIDATES in interp_bench.py."
        )

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError(f"{model_id!r} has no output embeddings / lm_head.")

    config = model.config
    # Vision-language wrappers (Gemma 3/4) keep the decoder hyperparameters
    # under config.text_config; read through to it for the text stream.
    text_config = getattr(config, "text_config", None) or config
    d_model = int(
        getattr(config, "hidden_size", 0)
        or getattr(config, "n_embd", 0)
        or getattr(text_config, "hidden_size", 0)
        or getattr(text_config, "n_embd", 0)
    )
    vocab_size = int(lm_head.weight.shape[0])

    tied = bool(getattr(config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False)))
    if tied:
        notes.append(
            "Input and output embeddings are tied: the unembedding is the "
            "transpose of the token embedding."
        )

    # Gemma-2-style models squash final logits with cap*tanh(logits/cap).
    # The lens must reproduce this or the depth-L self-check would fail for a
    # boring reason.
    softcap = getattr(config, "final_logit_softcapping", None)
    if softcap is None:
        softcap = getattr(text_config, "final_logit_softcapping", None)
    softcap = float(softcap) if softcap else None
    if softcap:
        notes.append(f"Model applies final logit softcapping (cap={softcap}).")

    anatomy = ModelAnatomy(
        model_id=model_id,
        revision=revision,
        architecture=type(model).__name__,
        blocks_path=blocks_path,
        n_layers=len(blocks),
        final_norm_path=final_norm_path,
        final_norm_class=type(final_norm).__name__,
        lm_head_class=type(lm_head).__name__,
        d_model=d_model,
        vocab_size=vocab_size,
        tied_embeddings=tied,
        param_count=sum(p.numel() for p in model.parameters()),
        logit_softcap=softcap,
        notes=tuple(notes),
    )
    return anatomy, blocks, final_norm, lm_head


def write_tokenizer_report(ctx: RunContext, bundle: ModelBundle) -> None:
    """Write tokenizer facts that commonly explain surprising lab results."""
    tok = bundle.tokenizer
    payload = {
        "tokenizer_class": type(tok).__name__,
        "vocab_size": getattr(tok, "vocab_size", None),
        "model_max_length": getattr(tok, "model_max_length", None),
        "bos_token": getattr(tok, "bos_token", None),
        "bos_token_id": getattr(tok, "bos_token_id", None),
        "eos_token": getattr(tok, "eos_token", None),
        "eos_token_id": getattr(tok, "eos_token_id", None),
        "pad_token": getattr(tok, "pad_token", None),
        "pad_token_id": getattr(tok, "pad_token_id", None),
        "padding_side": getattr(tok, "padding_side", None),
        "truncation_side": getattr(tok, "truncation_side", None),
        "chat_template_present": bool(getattr(tok, "chat_template", None)),
        "chat_template_used_by_lab": ctx.args.lab in CHAT_TEMPLATE_LABS,
        "note": (
            "Labs 1-6 use raw base-model prompts; a chat template, if present, "
            "is deliberately not applied. Labs 7+ render every prompt through "
            "the tokenizer's chat template (bench.apply_chat_template)."
        ),
    }
    path = ctx.path("diagnostics", "tokenizer_info.json")
    write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Tokenizer special tokens and template status.")


def write_anatomy_report(ctx: RunContext, bundle: ModelBundle) -> None:
    """Write the anatomy as JSON (machines) and Markdown (humans)."""
    a = bundle.anatomy
    write_json(ctx.path("diagnostics", "model_anatomy.json"), a)
    lines = [
        "# Model anatomy",
        "",
        "What the bench resolved before any experiment ran. Every lab's hook",
        "and readout uses these paths; if they look wrong, stop here.",
        "",
        f"| field | value |",
        f"|---|---|",
        f"| model | `{a.model_id}` (revision: {a.revision or 'default'}) |",
        f"| architecture | `{a.architecture}` |",
        f"| parameters | {a.param_count:,} |",
        f"| decoder blocks | `{a.blocks_path}` x {a.n_layers} |",
        f"| final norm | `{a.final_norm_path}` ({a.final_norm_class}) |",
        f"| unembedding | {a.lm_head_class}, vocab {a.vocab_size:,} |",
        f"| d_model | {a.d_model} |",
        f"| tied embeddings | {a.tied_embeddings} |",
        f"| logit softcap | {a.logit_softcap or 'none'} |",
        "",
        "Depth convention used everywhere in this course: `streams[k]` is the",
        "**pre-norm residual stream after k blocks**; k=0 is the embedding",
        f"output and k={a.n_layers} is the input to the final norm.",
        "",
    ]
    if a.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in a.notes)
        lines.append("")
    write_text(ctx.path("diagnostics", "model_anatomy.md"), "\n".join(lines))
    ctx.register_artifact(
        ctx.run_dir / "diagnostics" / "model_anatomy.md",
        "diagnostic",
        "Resolved module paths for blocks, final norm, unembedding.",
    )


def load_model_and_tokenizer(ctx: RunContext) -> ModelBundle:
    """Load tokenizer + causal LM per the run config, then resolve anatomy."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = ctx.args
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)
    print(f"[bench] loading {args.model!r} (device={device}, dtype={dtype})")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    load_kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    if args.low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    if args.quantization in ("8bit", "4bit"):
        # Quantization is an opt-in convenience for small GPUs. It changes
        # numerics, so the lens self-check tolerances are looser there and
        # run_summary.md records the setting.
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("--quantization requires the bitsandbytes package") from exc
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.quantization == "8bit",
            load_in_4bit=args.quantization == "4bit",
            bnb_4bit_compute_dtype=dtype,
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except TypeError as exc:
        # Older transformers spell the dtype argument torch_dtype. Re-raise
        # unrelated TypeErrors so unsupported kwargs do not get silently masked.
        if "dtype" not in load_kwargs:
            raise
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.quantization == "none":
        model = model.to(device)
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"[bench] model loaded in {load_s:.1f}s")

    anatomy, blocks, final_norm, lm_head = resolve_anatomy(model, args.model, args.model_revision)
    input_device = infer_input_device(model, device)
    lens_device = first_module_device(final_norm) or input_device
    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        anatomy=anatomy,
        blocks=blocks,
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        input_device=input_device,
        lens_device=lens_device,
        torch_dtype=dtype,
        model_device_map=device_map_summary(model),
    )
    write_anatomy_report(ctx, bundle)
    write_tokenizer_report(ctx, bundle)
    if bundle.model_device_map is not None:
        path = ctx.path("diagnostics", "model_device_map.json")
        write_json(path, bundle.model_device_map)
        ctx.register_artifact(path, "diagnostic", "Hugging Face device map for quantized/offloaded runs.")
    mem_path = ctx.path("diagnostics", "gpu_memory_after_load.json")
    write_json(mem_path, gpu_memory_snapshot(torch, "after_load"))
    ctx.register_artifact(mem_path, "diagnostic", "GPU memory snapshot after model load.")
    return bundle


# ---------------------------------------------------------------------------
# Residual stream capture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ForwardCapture:
    """One prompt's worth of model state, with explicit semantics.

    ``streams`` has shape ``[n_layers + 1, seq_len, d_model]`` in float32:
    the **pre-norm** residual stream after k blocks, for k = 0..n_layers
    (see the module docstring for how this is assembled from
    ``output_hidden_states`` plus a final-norm pre-hook).

    ``final_logits_last`` is the model's *actual* output logits at the final
    position, float32 -- the ground truth the logit lens is checked against.
    """

    prompt: str
    input_ids: list[int]
    tokens_raw: list[str]    # tokenizer-internal pieces, e.g. 'ĠParis'
    tokens_text: list[str]   # decoded text per token, e.g. ' Paris'
    streams: Any             # torch.Tensor [L+1, seq, d_model] float32
    final_logits_last: Any   # torch.Tensor [vocab] float32


def run_with_residual_cache(
    bundle: ModelBundle, prompt: str, *, add_special_tokens: bool = True
) -> ForwardCapture:
    """Run one prompt and capture the full pre-norm residual stream.

    Prompts run one at a time, unbatched. With <100 short prompts per lab the
    cost is irrelevant, and skipping batching removes the entire class of
    padding/attention-mask bugs from the course's foundation.

    Pass ``add_special_tokens=False`` for prompts that are already fully
    rendered (e.g. chat-templated labs): on tokenizers that auto-prepend
    BOS, the default would otherwise capture a sequence that generation
    (which tokenizes rendered prompts without special tokens) never sees.
    """
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)

    # The final block's pre-norm output is not in hidden_states (see module
    # docstring), so capture the final norm's input as it flows past.
    captured: dict[str, Any] = {}

    def final_norm_pre_hook(module: Any, hook_args: tuple) -> None:
        captured["final_prenorm"] = tensor_cpu_float(hook_args[0])

    handle = bundle.final_norm.register_forward_pre_hook(final_norm_pre_hook)
    try:
        with torch.no_grad():
            out = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        handle.remove()

    if "final_prenorm" not in captured:
        raise RuntimeError(
            "The final-norm pre-hook never fired. The resolved final_norm "
            "module is probably wrong for this architecture -- check "
            "diagnostics/model_anatomy.md."
        )

    hs = out.hidden_states  # tuple of L+1 tensors, indexing per module docstring
    n_layers = bundle.anatomy.n_layers
    if len(hs) != n_layers + 1:
        raise RuntimeError(
            f"Expected {n_layers + 1} hidden states, got {len(hs)}. The "
            "residual indexing assumptions do not hold for this architecture."
        )

    # streams[k] = pre-norm residual after k blocks:
    #   k in 0..L-1  -> hidden_states[k]   (input to block k)
    #   k = L        -> the final norm's captured input (output of block L-1)
    streams = torch.stack(
        [tensor_cpu_float(h[0]) for h in hs[:-1]] + [captured["final_prenorm"][0]]
    )

    ids = input_ids[0].detach().cpu().tolist()
    return ForwardCapture(
        prompt=prompt,
        input_ids=ids,
        tokens_raw=tokenizer.convert_ids_to_tokens(ids),
        tokens_text=[tokenizer.decode([i]) for i in ids],
        streams=streams,
        final_logits_last=tensor_cpu_float(out.logits[0, -1]),
    )


# ---------------------------------------------------------------------------
# Chat templates, steering, and generation (instruct/chat-template labs)
# ---------------------------------------------------------------------------
#
# Labs 1-6 use base models and raw prompts. Labs 7+ use instruct models, and
# the single most common cross-lab bug is template/token drift: computing a
# direction on an untemplated prompt and then steering templated generation
# changes meaning silently (the residual stream at "the same layer" is a
# different object once the chat scaffold is present). So template application
# lives here, once, and labs are expected to extract and steer through it.

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def supports_chat_template(bundle: ModelBundle) -> bool:
    return getattr(bundle.tokenizer, "chat_template", None) is not None


def apply_chat_template(
    bundle: ModelBundle,
    user_message: str,
    *,
    system: str | None = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = True,
) -> str:
    """Render a single-turn chat prompt as the string the model will see.

    Raises if the tokenizer has no chat template -- chat/generation labs
    require an instruct model, and a base model silently rendering raw text is
    exactly the drift the course warns about.
    """
    if not supports_chat_template(bundle):
        raise RuntimeError(
            f"{bundle.anatomy.model_id!r} has no chat template; this lab needs "
            "an instruct model. Use --tier a/b defaults or pass an instruct --model."
        )
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    return bundle.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


@contextlib.contextmanager
def steering_hooks(bundle: ModelBundle, layer: int, vector: Any, scale: float):
    """Add ``scale * vector`` to block ``layer``'s output at every position.

    This is activation addition (Turner et al.): the steering vector is added
    to the residual stream the block writes, on every forward pass -- so it
    affects prefill and every generated token alike. ``vector`` is a
    [d_model] float32 CPU tensor; it is cast to the block's device/dtype at
    the hook site. Hooks are always removed on exit.
    """
    import torch

    if scale == 0.0:
        yield
        return
    block = bundle.blocks[layer]

    def add_hook(module: Any, hook_args: tuple, output: Any) -> Any:
        if isinstance(output, tuple):
            out = output[0]
            out = out + (scale * vector).to(out.device, out.dtype)
            return (out,) + tuple(output[1:])
        return output + (scale * vector).to(output.device, output.dtype)

    handle = block.register_forward_hook(add_hook)
    try:
        yield
    finally:
        handle.remove()


def generate_text(
    bundle: ModelBundle,
    templated_prompt: str,
    *,
    max_new_tokens: int = 64,
    steer: tuple[int, Any, float] | None = None,
) -> str:
    """Greedy-decode a continuation for a templated prompt.

    Decoding is frozen (greedy, no sampling) so runs are reproducible and the
    only thing that moves across a dose sweep is the steering scale.
    ``steer`` is an optional (layer, vector, scale) activation-addition.
    Returns only the newly generated text, with special tokens stripped.
    """
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(templated_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    cm = (
        steering_hooks(bundle, steer[0], steer[1], steer[2])
        if steer is not None
        else contextlib.nullcontext()
    )
    with cm, torch.no_grad():
        out = bundle.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
        )
    new_ids = out[0, input_ids.shape[1]:].detach().cpu().tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Continuous-batching generation engine
# ---------------------------------------------------------------------------
#
# Heavy generation labs (10, 11) batch many variable-length greedy decodes.
# `model.generate` pays for the slowest row in every batch: finished rows keep
# stepping as padding until the longest row hits EOS or the cap, and think-model
# CoT lengths are heavy-tailed, so most batches contain a capped straggler.
# This engine keeps a rolling set of in-flight rows instead: each forward is one
# decode step for every active row; a row that finishes is retired immediately
# and a pending job takes its slot mid-decode. Pure Hugging Face forward calls —
# no vLLM, no custom kernels — so hooks, logits, and determinism (greedy) are
# exactly as observable as the rest of the bench.
#
# Implementation notes (the parts that are easy to get wrong):
# - The KV cache is packed left-padded: per layer, tensors of shape
#   (n_active, n_kv_heads, padded_len, head_dim). Padding only ever sits on the
#   left, so a row's valid region is always its trailing `valid_len` positions.
# - Pad positions hold garbage KV; the 2D attention mask (0 over pads) is what
#   keeps them out of every dot product. Correctness rests on the mask, not on
#   the pad contents.
# - position_ids are LOGICAL (real-token count), never physical cache offsets;
#   prefill passes cumsum(mask)-1 so left-padded rows get correct rotary phases.
# - Repacking (slice retired rows out, admit prefilled rows, trim shared left
#   pad) happens only on retire/admit events, not every step.

# Telemetry from the most recent generate_continuous call (wall time, steps,
# token counts, mean active rows). Labs may persist it under diagnostics/.
LAST_GENERATION_STATS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Static windowed KV cache: the engine's per-step hot path
# ---------------------------------------------------------------------------

# torch/transformers are runtime imports everywhere in this file, so the cache
# classes (which must SUBCLASS transformers' Cache) are built lazily by
# _static_kv_classes() and memoized here.
_STATIC_KV_CLASSES: dict[str, Any] = {}


def _static_kv_classes() -> dict[str, Any]:
    """Build (once) the engine's preallocated KV-cache classes.

    Requires transformers >= 5 (the ``Cache.layers`` API); the engine already
    assumed that, this just keeps the import lazy like the rest of the bench.
    """
    if _STATIC_KV_CLASSES:
        return _STATIC_KV_CLASSES
    import torch
    from transformers.cache_utils import Cache, CacheLayerMixin

    class StaticKVLayer(CacheLayerMixin):
        """One layer of StaticKVCache: a preallocated (max_rows, kv_heads,
        capacity, head_dim) key/value buffer pair, written in place.

        The owner's column window [start, end) is shared by all layers.
        ``update`` writes the step's KV at column ``end`` with ``copy_`` (no
        allocation) and returns window-sliced views. DynamicCache, by
        contrast, ``torch.cat``s the whole layer every step -- a
        reallocate-and-copy of the entire cache (~2.4 GB/step at 32B x 16
        rows) that this class exists to remove. The interface mimics
        DynamicLayer (``get_max_cache_shape() == -1`` etc.) so the model's
        mask construction takes exactly the code path the engine already
        validated against lockstep ``generate``.
        """

        is_sliding = False

        def __init__(self, win: dict) -> None:
            super().__init__()
            # The shared window DICT, not the owning cache: a layer->cache
            # back-reference would close a reference cycle (cache.layers ->
            # layer -> cache), and cyclic garbage is only reclaimed by the
            # generational GC -- which let multi-GiB CUDA buffers outlive the
            # engine call and OOM'd the next one (found in run 5).
            self._win = win
            self.is_initialized = True  # buffers are allocated by the owner

        def lazy_initialization(self, key_states: Any, value_states: Any) -> None:
            raise RuntimeError("StaticKVLayer buffers are allocated by StaticKVCache")

        def update(self, key_states: Any, value_states: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
            w = self._win
            t = key_states.shape[-2]
            self.keys[: w["rows"], :, w["end"] : w["end"] + t].copy_(key_states)
            self.values[: w["rows"], :, w["end"] : w["end"] + t].copy_(value_states)
            return (
                self.keys[: w["rows"], :, w["start"] : w["end"] + t],
                self.values[: w["rows"], :, w["start"] : w["end"] + t],
            )

        def get_seq_length(self) -> int:
            w = self._win
            return w["end"] - w["start"]

        def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
            return self.get_seq_length() + query_length, 0

        def get_max_cache_shape(self) -> int:
            return -1  # like DynamicLayer: no fixed maximum the mask must pad to

    class StaticKVCache(Cache):
        """Preallocated, windowed KV cache + attention mask for the engine.

        Geometry: every active row shares one column window [start, end);
        rows are right-aligned inside it (left-padded), so a decode step
        writes EVERY row's new KV at the same column, ``end``. The window
        only ever slides right, which turns the engine's structural ops into
        cheap ones:

        * trim shared left pad   -> ``start += slack``        (free)
        * admit a WIDER chunk    -> ``start -= delta``        (free: the newly
          exposed columns hold stale garbage for old rows, and the mask is
          zeroed there -- the same contract left-pad-plus-mask always had)
        * per-step KV write      -> ``copy_`` at column end   (in place)
        * retire rows            -> compact survivors down    (event-rate)

        ``end`` eventually hits ``capacity``; the live region is then slid
        back to column 0 (one clone+copy of the active window every
        ~(capacity - width) steps -- amortized to nothing). If the window
        itself ever outgrows capacity, buffers are reallocated 1.5x larger.

        The 2D attention mask lives here too (same window arithmetic): the
        model sees ``mask[:rows, start:end+1]``, exactly aligned with the KV
        views ``update`` returns. Correctness still rests entirely on the
        mask, never on buffer contents outside it.
        """

        def __init__(self, chunk_cache: Any, chunk_mask: Any, *, max_rows: int, headroom: int) -> None:
            width = chunk_cache.layers[0].keys.shape[2]
            self.capacity = width + headroom
            self.max_rows = max_rows
            self._win = {"rows": 0, "start": 0, "end": 0}
            layers = []
            for cl in chunk_cache.layers:
                layer = StaticKVLayer(self._win)
                layer.keys = torch.empty(
                    (max_rows, cl.keys.shape[1], self.capacity, cl.keys.shape[3]),
                    dtype=cl.keys.dtype, device=cl.keys.device)
                layer.values = torch.empty(
                    (max_rows, cl.values.shape[1], self.capacity, cl.values.shape[3]),
                    dtype=cl.values.dtype, device=cl.values.device)
                layers.append(layer)
            super().__init__(layers=layers)
            self.mask = torch.zeros(
                (max_rows, self.capacity), dtype=chunk_mask.dtype, device=chunk_mask.device)
            self.append_chunk(chunk_cache, chunk_mask)

        # -- window views ----------------------------------------------------

        @property
        def rows(self) -> int:
            return self._win["rows"]

        @property
        def width(self) -> int:
            return self._win["end"] - self._win["start"]

        def step_mask(self) -> Any:
            """Mask view for a 1-token decode step (new column set to 1)."""
            w = self._win
            self.mask[: w["rows"], w["end"]] = 1
            return self.mask[: w["rows"], w["start"] : w["end"] + 1]

        def advance(self) -> None:
            """Commit the step the model just wrote at column ``end``."""
            self._win["end"] += 1

        # -- structural ops (event-rate, never per step) -----------------------

        def append_chunk(self, chunk_cache: Any, chunk_mask: Any) -> None:
            """Admit a freshly prefilled chunk into rows [rows, rows+m)."""
            m, chunk_width = chunk_mask.shape
            w = self._win
            if w["rows"] + m > self.max_rows:
                raise RuntimeError(f"admitting {m} rows into {w['rows']}/{self.max_rows} occupied")
            if w["rows"] == 0:
                if chunk_width > self.capacity:
                    self._grow(chunk_width)
                w["start"], w["end"] = 0, chunk_width
            elif chunk_width > self.width:
                delta = chunk_width - self.width
                if w["start"] < delta:
                    self._relocate(delta)
                w["start"] -= delta
                # columns newly exposed for the existing rows are stale: mask off
                self.mask[: w["rows"], w["start"] : w["start"] + delta] = 0
            lo = w["end"] - chunk_width
            for layer, cl in zip(self.layers, chunk_cache.layers):
                layer.keys[w["rows"] : w["rows"] + m, :, lo : w["end"]].copy_(cl.keys)
                layer.values[w["rows"] : w["rows"] + m, :, lo : w["end"]].copy_(cl.values)
                cl.keys = cl.values = None  # release the chunk as we go
            self.mask[w["rows"] : w["rows"] + m, w["start"] : lo] = 0
            self.mask[w["rows"] : w["rows"] + m, lo : w["end"]].copy_(chunk_mask)
            w["rows"] += m

        def compact_rows(self, keep: list[int]) -> None:
            """Drop retired rows by compacting survivors downward, in order."""
            w = self._win
            idx = torch.tensor(keep, device=self.mask.device)
            span = slice(w["start"], w["end"])
            for layer in self.layers:
                for buf in (layer.keys, layer.values):
                    # index_select materializes the survivors first, so the
                    # downward copy cannot read rows it already overwrote.
                    buf[: len(keep), :, span].copy_(buf[:, :, span].index_select(0, idx))
            self.mask[: len(keep), span].copy_(self.mask[:, span].index_select(0, idx))
            w["rows"] = len(keep)

        def clear_rows(self) -> None:
            """All rows retired; keep the buffers for the next admit."""
            self._win.update(rows=0, start=0, end=0)

        def release(self) -> None:
            """Drop every CUDA buffer NOW. Called by the engine on exit so
            cache memory never depends on when the garbage collector feels
            like running."""
            for layer in self.layers:
                layer.keys = layer.values = None
            self.mask = None
            self._win.update(rows=0, start=0, end=0)

        def trim_left(self, slack: int) -> None:
            """Drop left-pad columns shared by every row: the window slides
            past them. Nothing is copied or freed -- the columns are reused
            the next time the region relocates."""
            self._win["start"] += slack

        def ensure_step_room(self) -> None:
            """Make sure column ``end`` exists before the model writes it."""
            if self._win["end"] + 1 > self.capacity:
                self._relocate(0)
                if self._win["end"] + 1 > self.capacity:  # width+1 > capacity
                    self._grow(self.width + 1)

        # -- internals ---------------------------------------------------------

        def _relocate(self, new_start: int) -> None:
            """Slide the live region so it begins at ``new_start``."""
            w = self._win
            width = self.width
            if new_start + width > self.capacity:
                self._grow(new_start + width, new_start)
                return
            src = slice(w["start"], w["end"])
            dst = slice(new_start, new_start + width)
            for layer in self.layers:
                for buf in (layer.keys, layer.values):
                    # clone: src and dst overlap whenever the slide is short
                    buf[: w["rows"], :, dst].copy_(buf[: w["rows"], :, src].clone())
            self.mask[: w["rows"], dst].copy_(self.mask[: w["rows"], src].clone())
            w["start"], w["end"] = new_start, new_start + width

        def _grow(self, min_capacity: int, new_start: int = 0) -> None:
            w = self._win
            width = self.width
            # Grow in ~512-column increments: one full-cache copy every ~512
            # steps is noise (DynamicCache paid that copy EVERY step), and it
            # keeps peak allocation near the actual live width instead of the
            # worst-case prompt+cap width.
            new_cap = max(min_capacity + 256, self.capacity + 512)
            src = slice(w["start"], w["end"])
            dst = slice(new_start, new_start + width)
            for layer in self.layers:
                new_k = torch.empty(
                    (self.max_rows, layer.keys.shape[1], new_cap, layer.keys.shape[3]),
                    dtype=layer.keys.dtype, device=layer.keys.device)
                new_v = torch.empty(
                    (self.max_rows, layer.values.shape[1], new_cap, layer.values.shape[3]),
                    dtype=layer.values.dtype, device=layer.values.device)
                if w["rows"]:
                    new_k[: w["rows"], :, dst].copy_(layer.keys[: w["rows"], :, src])
                    new_v[: w["rows"], :, dst].copy_(layer.values[: w["rows"], :, src])
                layer.keys, layer.values = new_k, new_v
            new_mask = torch.zeros(
                (self.max_rows, new_cap), dtype=self.mask.dtype, device=self.mask.device)
            if w["rows"]:
                new_mask[: w["rows"], dst].copy_(self.mask[: w["rows"], src])
            self.mask = new_mask
            self.capacity = new_cap
            w["start"], w["end"] = new_start, new_start + width

    _STATIC_KV_CLASSES["StaticKVLayer"] = StaticKVLayer
    _STATIC_KV_CLASSES["StaticKVCache"] = StaticKVCache
    return _STATIC_KV_CLASSES


def generate_continuous(
    bundle: ModelBundle,
    prompts: Sequence[str],
    max_new_tokens: int | Sequence[int],
    *,
    max_concurrent: int = 16,
    eos_token_id: int | Sequence[int] | None = None,
    skip_special_tokens: bool = False,
    progress_label: str = "",
    steer: tuple[int, Any, float | Sequence[float]] | None = None,
    admit_block: int | None = None,
    max_prefill_tokens: int = 16384,
) -> list[str]:
    """Greedy-decode many prompts with continuous batching; returns continuations.

    ``max_new_tokens`` may be a single cap or one cap per prompt, so cheap jobs
    (e.g. 8-token forced answers) can share the schedule with 2048-token think
    jobs instead of waiting for their own batch. Output order matches input
    order. Decoding is greedy (the bench's frozen-decoding rule); EOS defaults
    to the tokenizer's ``eos_token_id``.

    ``steer`` is an optional ``(layer, vector, scale)`` activation addition
    applied on every forward -- prefill and decode alike, matching
    ``generate_text``'s semantics exactly (the added term is computed in
    float32 and then cast, like ``steering_hooks``). ``scale`` may be a single
    float or one float per job: that is what lets Lab 7 ride a whole dose
    sweep on one engine schedule, each row at its own dose.

    ``admit_block`` batches admits: each admit's prefill blocks every
    in-flight row for the whole prefill forward, so instead of admitting on
    every retirement the engine waits until ``admit_block`` slots are free
    (or it is out of in-flight rows, or fewer than ``admit_block`` jobs
    remain). Default ``max_concurrent // 4``; pass 1 to admit eagerly.

    ``max_prefill_tokens`` caps a single admit's prefill width
    (rows x padded prompt length): admitting ``max_concurrent`` rows whose
    prompts are all long prefills one huge batch, and that peak — not the
    steady-state cache — is what OOMs (run 5: Lab 10's unparseable-rescue
    re-fed every job its full prompt-plus-2048-token think trace, so a
    32-row admit prefilled ~70k tokens and died on the 7B). With a budget,
    long-prompt batches admit in narrower waves and decode normally; short
    prompts (the common case) are unaffected because they never approach it.
    Always admits at least one row, so a single over-budget prompt still
    runs. The default leaves the throughput benchmarks and short-prompt labs
    untouched; lower it for very large models.
    """
    import torch

    n_jobs = len(prompts)
    if n_jobs == 0:
        return []
    caps = (
        [int(max_new_tokens)] * n_jobs
        if isinstance(max_new_tokens, int)
        else [int(c) for c in max_new_tokens]
    )
    if len(caps) != n_jobs:
        raise ValueError(f"max_new_tokens has {len(caps)} entries for {n_jobs} prompts.")

    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.input_device
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    eos_ids = {int(e) for e in (
        [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id or [])
    )}
    admit_block_eff = max(1, max_concurrent // 4) if admit_block is None else max(1, int(admit_block))

    # Per-prompt token lengths, for the prefill-width budget. One extra
    # tokenization pass (prefill re-tokenizes its chunk anyway) -- cheap
    # against generation, and it lets the admit loop keep a long-prompt batch
    # from prefilling all at once. Truncation is irrelevant here; only the
    # length matters.
    prompt_lens = [
        len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts
    ]

    # Per-job steering scales (None = no steering). The hook multiplies the
    # CURRENT rows' scales by the fp32 vector and casts the product, so each
    # row sees exactly what generate_text's steering_hooks would have added.
    steer_scales: list[float] | None = None
    steer_cur: dict[str, Any] = {"scales": None}
    steer_vec32: Any = None
    if steer is not None:
        steer_layer, raw_vec, raw_scale = steer
        steer_scales = (
            [float(raw_scale)] * n_jobs
            if isinstance(raw_scale, (int, float))
            else [float(x) for x in raw_scale]
        )
        if len(steer_scales) != n_jobs:
            raise ValueError(f"steer scales has {len(steer_scales)} entries for {n_jobs} prompts.")
        steer_vec32 = raw_vec.to(device=device, dtype=torch.float32)

    def set_steer_rows(jobs: list[int]) -> None:
        if steer_scales is None:
            return
        steer_cur["scales"] = torch.tensor(
            [steer_scales[j] for j in jobs], dtype=torch.float32, device=device
        ).view(-1, 1, 1)

    # Packed state for the active rows (parallel lists, one entry per row).
    job_idx: list[int] = []        # original prompt index per active row
    valid_lens: list[int] = []     # logical (real-token) KV length per row
    gen_ids: list[list[int]] = []  # tokens generated so far per row
    last_tokens: list[int] = []    # next decode-step input per row
    cache: Any = None              # StaticKVCache (allocated at first admit)
    StaticKVCache = _static_kv_classes()["StaticKVCache"]

    # Device-resident step inputs, rebuilt only at retire/admit events and
    # mutated in place per step (ids_buf <- argmax output, pos_buf += 1), so
    # the steady-state loop does no host->device transfers.
    ids_buf = torch.empty((max_concurrent, 1), dtype=torch.long, device=device)
    pos_buf = torch.empty((max_concurrent, 1), dtype=torch.long, device=device)

    results: dict[int, list[int]] = {}
    pending = list(range(n_jobs))
    finished_rows: list[int] = []  # active-row indices that just retired

    wall_start = time.perf_counter()
    total_steps = 0
    total_tokens = 0
    active_row_steps = 0
    step_ms: list[float] = []      # per-decode-step wall latency (ITL trace)
    step_ctx: list[int] = []       # cache length at each step (for ms/ctx slope)
    ttft_s: dict[int, float] = {}  # job -> seconds from call start to first token
    admit_events: list[dict[str, Any]] = []

    def prefill(indices: list[int]) -> tuple[Any, Any, list[int], list[int]]:
        """Prefill a chunk of jobs (left-padded); returns its packed state."""
        from transformers import DynamicCache

        enc = tokenizer(
            [prompts[i] for i in indices],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        # Left padding + raw forward: positions must be derived from the mask,
        # or padded rows get shifted rotary phases.
        position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
        set_steer_rows(indices)
        with torch.inference_mode():
            out = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
            )
        first = out.logits[:, -1, :].argmax(dim=-1).tolist()
        lens = mask.sum(dim=1).tolist()
        return out.past_key_values, mask, lens, first

    def admit_ready() -> bool:
        free = max_concurrent - len(job_idx)
        if not pending or free <= 0:
            return False
        if not job_idx:
            return True
        return free >= min(admit_block_eff, len(pending))

    def retire_and_admit() -> None:
        """Drop finished rows, trim shared left pad, admit pending jobs."""
        nonlocal cache, finished_rows, total_tokens
        keep = [r for r in range(len(job_idx)) if r not in finished_rows]
        for r in sorted(finished_rows, reverse=True):
            results[job_idx[r]] = gen_ids[r]
            del job_idx[r], valid_lens[r], gen_ids[r], last_tokens[r]
        finished_rows = []
        if cache is not None:
            if keep and len(keep) < cache.rows:
                cache.compact_rows(keep)
            elif not keep:
                cache.clear_rows()
            if job_idx:
                slack = cache.width - max(valid_lens)
                if slack > 0:
                    cache.trim_left(slack)
        if admit_ready():
            # Fill the chunk greedily up to the free slots, but stop before a
            # prefill whose width (rows x padded length) would exceed the
            # budget -- so a batch of long prompts admits in narrower waves
            # instead of one OOM-sized prefill. Always take at least one
            # (an over-budget prompt still has to run).
            free = max_concurrent - len(job_idx)
            chunk: list[int] = []
            chunk_max_len = 0
            while pending and len(chunk) < free:
                cand_max = max(chunk_max_len, prompt_lens[pending[0]])
                if chunk and (len(chunk) + 1) * cand_max > max_prefill_tokens:
                    break
                chunk.append(pending.pop(0))
                chunk_max_len = cand_max
            rows_blocked = len(job_idx)
            t_admit = time.perf_counter()
            chunk_cache, chunk_mask, lens, first = prefill(chunk)
            if cache is None:
                cache = StaticKVCache(
                    chunk_cache, chunk_mask,
                    # Never allocate ghost rows: a 1-job call (Lab 10's
                    # round-trip check) must not pay for max_concurrent rows
                    # of a 2048-token think buffer.
                    max_rows=min(max_concurrent, n_jobs),
                    # Deliberately small: capacity tracks the live width via
                    # incremental _grow instead of preallocating the
                    # worst-case prompt+cap width for every row (which cost
                    # +20-30 GiB at 32-48 rows for zero benefit).
                    headroom=512,
                )
            else:
                cache.append_chunk(chunk_cache, chunk_mask)
            now_s = time.perf_counter() - wall_start
            prefill_ms = (now_s - (t_admit - wall_start)) * 1000.0
            for j in chunk:
                ttft_s[j] = now_s
            admit_events.append({
                "t_s": round(now_s, 3),
                "n_admitted": len(chunk),
                "prefill_ms": round(prefill_ms, 1),
                "rows_blocked": rows_blocked,
            })
            job_idx.extend(chunk)
            valid_lens.extend(lens)
            last_tokens.extend(first)
            gen_ids.extend([[t] for t in first])
            total_tokens += len(chunk)  # prefill emits each row's first token

    # First-token handling: prefill already produced one token per row, so a
    # row whose cap is 1 (or whose first token is EOS) retires before stepping.
    def mark_finished() -> None:
        for r in range(len(job_idx)):
            if r in finished_rows:
                continue
            tok = gen_ids[r][-1]
            if tok in eos_ids or len(gen_ids[r]) >= caps[job_idx[r]]:
                finished_rows.append(r)

    def rebuild_device_state() -> None:
        n = len(job_idx)
        if n == 0:
            return
        ids_buf[:n, 0] = torch.tensor(last_tokens, dtype=torch.long)
        pos_buf[:n, 0] = torch.tensor(valid_lens, dtype=torch.long)
        set_steer_rows(job_idx)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    steer_handle = None
    if steer_scales is not None:
        steer_block = bundle.blocks[int(steer[0])]

        def steer_hook(module: Any, hook_args: tuple, output: Any) -> Any:
            scales = steer_cur["scales"]
            if scales is None:
                return output
            if isinstance(output, tuple):
                out = output[0]
                out = out + (scales * steer_vec32).to(out.device, out.dtype)
                return (out,) + tuple(output[1:])
            return output + (scales * steer_vec32).to(output.device, output.dtype)

        steer_handle = steer_block.register_forward_hook(steer_hook)
    try:
        retire_and_admit()  # first admit
        mark_finished()
        rebuild_device_state()
        while job_idx or pending:
            if finished_rows or admit_ready():
                retire_and_admit()
                mark_finished()
                rebuild_device_state()
                continue
            if not job_idx:
                continue
            n = len(job_idx)
            cache.ensure_step_room()
            t_step = time.perf_counter()
            with torch.inference_mode():
                out = model(
                    input_ids=ids_buf[:n],
                    attention_mask=cache.step_mask(),
                    position_ids=pos_buf[:n],
                    past_key_values=cache,
                    use_cache=True,
                )
            next_dev = out.logits[:, -1, :].argmax(dim=-1)
            ids_buf[:n, 0].copy_(next_dev)  # next step's input, device-side
            next_tokens = next_dev.tolist()
            # .tolist() syncs the device, so this wall time covers the real step.
            step_ms.append((time.perf_counter() - t_step) * 1000.0)
            cache.advance()
            pos_buf[:n] += 1
            step_ctx.append(cache.width)
            for r in range(n):
                last_tokens[r] = int(next_tokens[r])
                gen_ids[r].append(last_tokens[r])
                valid_lens[r] += 1
            total_steps += 1
            total_tokens += n
            active_row_steps += n
            if progress_label and total_steps % 200 == 0:
                done = len(results)
                print(f"[bench] {progress_label}: {done}/{n_jobs} jobs done, "
                      f"{total_tokens} tokens, {len(job_idx)} in flight")
            mark_finished()
    finally:
        tokenizer.padding_side = old_side
        if steer_handle is not None:
            steer_handle.remove()
        if cache is not None:
            cache.release()
        # release() drops the engine's references, but PyTorch keeps the
        # freed blocks in its reserved pool — fragmented, they may not satisfy
        # the NEXT caller's large allocation even with GiB nominally free (run
        # 5: Lab 10's unparseable-rescue call OOM'd right after a 1.6M-token
        # engine block that had already finished). Returning the blocks to
        # CUDA here keeps a sequence of engine calls (the labs' real pattern)
        # from fragmenting itself into a false OOM. Once per call, not per
        # step — cost is negligible against a multi-minute generation.
        if cache is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall = time.perf_counter() - wall_start
    # Inter-token-latency trace stats (the per-step health of the loop): p50,
    # p95, and two slopes — ms per STEP (allocator/bookkeeping drift) and ms
    # per token of CONTEXT (attention+cache-growth cost). Adapted from the
    # course's external olmo_lora_bench instrumentation.
    itl: dict[str, Any] = {}
    if step_ms:
        s = sorted(step_ms)
        itl["itl_p50_ms"] = round(s[len(s) // 2], 2)
        itl["itl_p95_ms"] = round(s[min(len(s) - 1, int(len(s) * 0.95))], 2)
        mean = sum(step_ms) / len(step_ms)
        var = sum((x - mean) ** 2 for x in step_ms) / len(step_ms)
        itl["itl_cov"] = round((var ** 0.5) / max(mean, 1e-6), 3)
        if len(step_ms) >= 2:
            n_s = len(step_ms)
            xbar = (n_s - 1) / 2
            num = sum((i - xbar) * (y - mean) for i, y in enumerate(step_ms))
            den = sum((i - xbar) ** 2 for i in range(n_s))
            itl["itl_slope_ms_per_step"] = round(num / max(den, 1e-9), 6)
            cbar = sum(step_ctx) / n_s
            numc = sum((c - cbar) * (y - mean) for c, y in zip(step_ctx, step_ms))
            denc = sum((c - cbar) ** 2 for c in step_ctx)
            itl["itl_slope_ms_per_ctx_token"] = round(numc / max(denc, 1e-9), 6)
    # TTFT / admit telemetry (the olmo_lora_bench-style numbers that quantify
    # how much prefill interruption costs the in-flight rows).
    if ttft_s:
        tt = sorted(ttft_s.values())
        itl["ttft_p50_s"] = round(tt[len(tt) // 2], 2)
        itl["ttft_p95_s"] = round(tt[min(len(tt) - 1, int(len(tt) * 0.95))], 2)
    stalls = [e["prefill_ms"] for e in admit_events if e["rows_blocked"] > 0]
    itl["admit_events"] = len(admit_events)
    itl["admit_block"] = admit_block_eff
    itl["prefill_stall_ms_total"] = round(sum(stalls), 1)
    itl["prefill_stall_ms_max"] = round(max(stalls), 1) if stalls else 0.0
    LAST_GENERATION_STATS.clear()
    LAST_GENERATION_STATS.update({
        "engine": "continuous",
        "n_jobs": n_jobs,
        "max_concurrent": max_concurrent,
        "decode_steps": total_steps,
        "generated_tokens": total_tokens,
        "mean_active_rows": round(active_row_steps / total_steps, 2) if total_steps else 0.0,
        "wall_seconds": round(wall, 2),
        "tokens_per_second": round(total_tokens / wall, 1) if wall > 0 else 0.0,
        **itl,
        "admit_trace": admit_events,
    })
    return [
        tokenizer.decode(results[i], skip_special_tokens=skip_special_tokens)
        for i in range(n_jobs)
    ]


def next_token_logits(
    bundle: ModelBundle, templated_prompt: str, *, steer: tuple[int, Any, float] | None = None
) -> Any:
    """Final-position logits for a templated prompt, optionally steered.

    Float32 CPU. Used for the KL-to-unsteered side-effect metric without
    generating any text.
    """
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(templated_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(bundle.input_device)
    cm = (
        steering_hooks(bundle, steer[0], steer[1], steer[2])
        if steer is not None
        else contextlib.nullcontext()
    )
    with cm, torch.no_grad():
        out = bundle.model(input_ids=input_ids, use_cache=False)
    return tensor_cpu_float(out.logits[0, -1])


def run_hook_parity_check(ctx: RunContext, bundle: ModelBundle, prompt: str) -> dict[str, Any]:
    """Cross-check per-block forward hooks against the assembled stream cache.

    A mismatch means the harness is not measuring the object it says it is
    measuring. By default, this aborts the run. ``--allow-hook-mismatch`` turns
    the abort into a diagnostic warning for architecture bring-up work.
    """
    import torch

    block_outputs: dict[int, Any] = {}

    def make_hook(idx: int):
        def hook(module: Any, hook_args: tuple, output: Any) -> None:
            out = output[0] if isinstance(output, tuple) else output
            block_outputs[idx] = tensor_cpu_float(out)

        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(bundle.blocks)]
    try:
        capture = run_with_residual_cache(bundle, prompt)
    finally:
        for handle in handles:
            handle.remove()

    n_layers = bundle.anatomy.n_layers
    by_layer_rows: list[dict[str, Any]] = []
    max_diff = 0.0
    max_mean_diff = 0.0
    compared = 0
    missing_layers: list[int] = []
    for k in range(n_layers):
        if k not in block_outputs:
            missing_layers.append(k)
            continue
        # Block k's output is the stream after k+1 blocks == streams[k+1].
        hook_out = block_outputs[k][0]
        expected = capture.streams[k + 1]
        abs_diff = (hook_out - expected).abs()
        layer_max = float(abs_diff.max())
        layer_mean = float(abs_diff.mean())
        max_diff = max(max_diff, layer_max)
        max_mean_diff = max(max_mean_diff, layer_mean)
        compared += 1
        by_layer_rows.append(
            {
                "layer": k,
                "max_abs_diff": layer_max,
                "mean_abs_diff": layer_mean,
                "hook_l2": float(hook_out.norm()),
                "expected_l2": float(expected.norm()),
                "shape": "x".join(str(x) for x in hook_out.shape),
                "ok_at_tolerance": layer_max <= ctx.args.hook_tolerance,
            }
        )

    by_layer_path = ctx.path("diagnostics", "hook_parity_by_layer.csv")
    write_csv(by_layer_path, by_layer_rows)
    ctx.register_artifact(by_layer_path, "diagnostic", "Layer-level hook versus hidden-state parity check.")

    ok = (not missing_layers) and compared == n_layers and max_diff <= ctx.args.hook_tolerance
    result = {
        "prompt": prompt,
        "blocks_compared": compared,
        "n_layers": n_layers,
        "missing_layers": missing_layers,
        "max_abs_diff": max_diff,
        "max_mean_abs_diff": max_mean_diff,
        "tolerance": ctx.args.hook_tolerance,
        "ok": bool(ok),
        "allow_hook_mismatch": bool(ctx.args.allow_hook_mismatch),
        "explanation": (
            "Forward hooks on each decoder block were compared against the "
            "streams assembled from output_hidden_states plus the final-norm "
            "pre-hook. A mismatch means the residual stream capture semantics "
            "are not verified for this architecture or library version."
        ),
    }
    path = ctx.path("diagnostics", "hook_parity.json")
    write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Summary of hook captures versus assembled residual streams.")
    status = "OK" if result["ok"] else "MISMATCH"
    print(f"[bench] hook parity check: {status} (max |diff| = {max_diff:g}, compared {compared}/{n_layers})")
    if not result["ok"] and not ctx.args.allow_hook_mismatch:
        raise RuntimeError(
            "Hook parity check failed. The harness did not verify its residual "
            "capture semantics. See diagnostics/hook_parity*."
        )
    return result

# ---------------------------------------------------------------------------
# Logit lens
# ---------------------------------------------------------------------------


def logit_lens_all_depths(bundle: ModelBundle, streams_at_position: Any) -> Any:
    """Apply final_norm + unembedding to residual streams at one position.

    The cached streams live on CPU float32 so diagnostics do not pin GPU memory.
    This function moves only the depth x d_model slice needed for the lens to
    the final-norm device, runs the model's own readout modules, then returns
    CPU float32 logits. At depth L this should reproduce the model's real logits.
    """
    import torch

    with torch.no_grad():
        norm_device = first_module_device(bundle.final_norm) or bundle.lens_device
        head_device = first_module_device(bundle.lm_head) or norm_device
        x = streams_at_position.to(device=norm_device, dtype=bundle.torch_dtype)
        normed = bundle.final_norm(x)
        if str(head_device) != str(norm_device):
            normed = normed.to(head_device)
        logits = bundle.lm_head(normed).float()
        if bundle.anatomy.logit_softcap:
            cap = bundle.anatomy.logit_softcap
            logits = cap * torch.tanh(logits / cap)
    return logits.detach().to(device="cpu", dtype=torch.float32)

def run_lens_self_check(ctx: RunContext, bundle: ModelBundle, capture: ForwardCapture) -> dict[str, Any]:
    """Verify lens(depth=L) reproduces the model's actual final logits.

    Top-1 agreement is required. The numeric logit difference is reported, not
    used as a hard failure by default, because recomputing the final projection
    outside the model may avoid fused kernels or quantized execution paths.
    """
    import torch

    lens_logits = logit_lens_all_depths(bundle, capture.streams[:, -1, :])
    lens_final = lens_logits[-1]
    real_final = capture.final_logits_last

    lens_top = torch.topk(lens_final, k=min(5, lens_final.numel()))
    real_top = torch.topk(real_final, k=min(5, real_final.numel()))
    lens_top1 = int(lens_top.indices[0])
    real_top1 = int(real_top.indices[0])
    max_diff = float((lens_final - real_final).abs().max())
    mean_diff = float((lens_final - real_final).abs().mean())
    rel = max_diff / max(1e-9, float(real_final.abs().max()))
    top5_overlap = len(set(int(x) for x in lens_top.indices) & set(int(x) for x in real_top.indices))

    top1_matches = lens_top1 == real_top1
    # A bf16 recomputation can legitimately flip top-1 on a near-tie prompt:
    # different matmul shapes reduce in different orders. Accept a mismatch
    # only when the model's own logit gap between the two candidates is within
    # the observed numeric noise floor and both candidates sit in each other's
    # top-5. Anything larger means the capture or norm path is wrong.
    near_tie_ok = False
    real_gap = None
    if not top1_matches:
        real_gap = float(real_final[real_top1] - real_final[lens_top1])
        near_tie_ok = real_gap <= max_diff * 1.5 and top5_overlap >= 4

    result = {
        "prompt": capture.prompt,
        "top1_matches": top1_matches,
        "near_tie_accepted": near_tie_ok,
        "real_logit_gap_top1_vs_lens_top1": real_gap,
        "lens_top1": bundle.tokenizer.decode([lens_top1]),
        "model_top1": bundle.tokenizer.decode([real_top1]),
        "lens_top5": [bundle.tokenizer.decode([int(i)]) for i in lens_top.indices],
        "model_top5": [bundle.tokenizer.decode([int(i)]) for i in real_top.indices],
        "top5_overlap": top5_overlap,
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": mean_diff,
        "max_rel_logit_diff": rel,
        "quantization": ctx.args.quantization,
        "ok": top1_matches or near_tie_ok,
        "explanation": (
            "lens(L) = lm_head(final_norm(stream after all blocks)) recomputed "
            "outside the model must reproduce the model's own top prediction. "
            "If top-1 disagrees, the capture or norm/logit post-processing is "
            "wrong and no mid-depth lens readout in this run can be trusted."
        ),
    }
    path = ctx.path("diagnostics", "logit_lens_self_check.json")
    write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Proof that the lens at final depth reproduces the model's logits.")
    status = "OK" if top1_matches else ("OK (near-tie within numeric noise)" if near_tie_ok else "FAILED")
    print(
        f"[bench] lens self-check: {status} "
        f"(top1 lens={result['lens_top1']!r} model={result['model_top1']!r}, "
        f"max |dlogit| = {max_diff:.4f}, top5 overlap={top5_overlap}/5)"
    )
    if not result["ok"]:
        raise RuntimeError(
            "Logit lens self-check failed: the lens at final depth does not "
            "reproduce the model's own prediction. See diagnostics/logit_lens_self_check.json."
        )
    return result

# ---------------------------------------------------------------------------
# Lens trajectory: the per-depth measurement pack used by Lab 1 (and reused
# by later labs as the "running prediction" baseline view).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LensTrajectory:
    """Per-depth readout metrics at the final position, as plain lists."""

    n_depths: int
    top1_ids: list[int]
    top1_texts: list[str]
    top1_probs: list[float]
    top2_ids: list[int]
    top2_texts: list[str]
    top2_probs: list[float]
    top1_margin: list[float]
    entropy_bits: list[float]
    kl_to_final_bits: list[float]
    cosine_to_final: list[float]
    cosine_to_prev: list[float | None]
    resid_l2: list[float]
    stream_delta_l2: list[float]
    p_target: list[float] | None
    p_distractor: list[float] | None
    logit_target: list[float] | None
    logit_distractor: list[float] | None
    target_rank: list[int] | None
    distractor_rank: list[int] | None
    topk_rows: list[dict[str, Any]]  # depth, rank, token_id, token, prob, logit


def compute_lens_trajectory(
    bundle: ModelBundle,
    capture: ForwardCapture,
    *,
    target_id: int | None = None,
    distractor_id: int | None = None,
    topk: int = 5,
) -> LensTrajectory:
    """Compute the standard per-depth readout at the final token position.

    Besides the obvious p(target) and top-k outputs, this records metrics that
    help students debug interpretations: rank can improve before probability,
    entropy can drop before correctness, and KL-to-final can converge while the
    top-1 token still flips.
    """
    import torch

    streams_last = capture.streams[:, -1, :]  # [L+1, d_model] float32 on CPU
    logits = logit_lens_all_depths(bundle, streams_last)  # [L+1, vocab] CPU float32
    probs = torch.softmax(logits, dim=-1)
    vocab = logits.shape[-1]
    k = max(1, min(topk, vocab))

    # Entropy in bits: 0 = certain, log2(vocab) = uniform. Bits are intuitive:
    # 2.3 bits means roughly five equally plausible options.
    entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=-1)

    final_probs = probs[-1].clamp_min(1e-12)
    kl_to_final = (final_probs.unsqueeze(0) * (torch.log2(final_probs.unsqueeze(0)) - torch.log2(probs.clamp_min(1e-12)))).sum(dim=-1)

    final_stream = streams_last[-1]
    cosine = torch.nn.functional.cosine_similarity(
        streams_last, final_stream.unsqueeze(0), dim=-1
    )
    resid_l2 = streams_last.norm(dim=-1)
    deltas = torch.zeros_like(resid_l2)
    if streams_last.shape[0] > 1:
        deltas[1:] = (streams_last[1:] - streams_last[:-1]).norm(dim=-1)
    cosine_prev: list[float | None] = [None]
    if streams_last.shape[0] > 1:
        prev_cos = torch.nn.functional.cosine_similarity(streams_last[1:], streams_last[:-1], dim=-1)
        cosine_prev.extend(float(v) for v in prev_cos)

    top = torch.topk(probs, k=k, dim=-1)
    top2 = torch.topk(probs, k=min(2, vocab), dim=-1)
    if top2.indices.shape[-1] == 1:
        top2_ids = top2.indices[:, 0]
        top2_probs = torch.zeros_like(top2.values[:, 0])
        margin = top2.values[:, 0]
    else:
        top2_ids = top2.indices[:, 1]
        top2_probs = top2.values[:, 1]
        margin = top2.values[:, 0] - top2.values[:, 1]

    topk_rows: list[dict[str, Any]] = []
    for depth in range(logits.shape[0]):
        for rank in range(k):
            token_id = int(top.indices[depth, rank])
            topk_rows.append(
                {
                    "depth": depth,
                    "rank": rank + 1,
                    "token_id": token_id,
                    "token": bundle.tokenizer.decode([token_id]),
                    "prob": round(float(top.values[depth, rank]), 6),
                    "logit": round(float(logits[depth, token_id]), 4),
                }
            )

    top1_ids = [int(i) for i in top.indices[:, 0]]

    def ranks_for(token_id: int | None) -> list[int] | None:
        if token_id is None:
            return None
        target_logits = logits[:, token_id].unsqueeze(-1)
        return [int(v) + 1 for v in (logits > target_logits).sum(dim=-1)]

    traj = LensTrajectory(
        n_depths=int(logits.shape[0]),
        top1_ids=top1_ids,
        top1_texts=[bundle.tokenizer.decode([i]) for i in top1_ids],
        top1_probs=[float(v) for v in top.values[:, 0]],
        top2_ids=[int(i) for i in top2_ids],
        top2_texts=[bundle.tokenizer.decode([int(i)]) for i in top2_ids],
        top2_probs=[float(v) for v in top2_probs],
        top1_margin=[float(v) for v in margin],
        entropy_bits=[float(v) for v in entropy],
        kl_to_final_bits=[float(v) for v in kl_to_final],
        cosine_to_final=[float(v) for v in cosine],
        cosine_to_prev=cosine_prev,
        resid_l2=[float(v) for v in resid_l2],
        stream_delta_l2=[float(v) for v in deltas],
        p_target=None,
        p_distractor=None,
        logit_target=None,
        logit_distractor=None,
        target_rank=ranks_for(target_id),
        distractor_rank=ranks_for(distractor_id),
        topk_rows=topk_rows,
    )
    if target_id is not None:
        traj.p_target = [float(v) for v in probs[:, target_id]]
        traj.logit_target = [float(v) for v in logits[:, target_id]]
    if distractor_id is not None:
        traj.p_distractor = [float(v) for v in probs[:, distractor_id]]
        traj.logit_distractor = [float(v) for v in logits[:, distractor_id]]
    return traj

# ---------------------------------------------------------------------------
# Component capture: per-block attention and MLP contributions (Lab 2+)
# ---------------------------------------------------------------------------
#
# Direct logit attribution needs the exact tensor each sub-block ADDS to the
# residual stream. Where that tensor lives differs by architecture:
#
#   GPT-2 (pre-norm):   x + attn(ln_1(x)); then + mlp(ln_2(.))
#                       -> the attn/mlp module outputs ARE the contributions.
#   Olmo-2/3 (post-norm): x + post_attention_layernorm(attn(x));
#                         then + post_feedforward_layernorm(mlp(.))
#                       -> the *norm* outputs are the contributions; the raw
#                          attn/mlp outputs never touch the stream.
#
# Name heuristics are a trap here: Llama-style models also have a module
# called `post_attention_layernorm`, but there it is a PRE-norm for the MLP.
# So the bench does not guess. It runs one probe forward, captures *both*
# candidate sources per block, and keeps whichever pair actually reconstructs
# each block's residual delta (streams[k+1] - streams[k]). The decision and
# its reconstruction error are written to diagnostics/component_anatomy.json.
# Same contract as the other self-checks: if nothing reconstructs, abort.

ATTN_MODULE_CANDIDATES = ("self_attn", "attn")
MLP_MODULE_CANDIDATES = ("mlp",)
POST_ATTN_NORM_CANDIDATES = ("post_attention_layernorm",)
POST_MLP_NORM_CANDIDATES = ("post_feedforward_layernorm",)


@dataclasses.dataclass
class ComponentAnatomy:
    """Verified per-block contribution hook points for this model."""

    attn_module_path: str        # e.g. 'self_attn' (relative to each block)
    mlp_module_path: str
    attn_source: str             # 'module' or 'post_norm'
    mlp_source: str
    attn_hook_path: str          # the path actually hooked for contributions
    mlp_hook_path: str
    max_block_recon_rel_err: float
    probe_prompt: str


@dataclasses.dataclass
class ComponentCapture:
    """One prompt's residual streams plus per-block contribution vectors.

    ``attn_contrib`` / ``mlp_contrib`` have shape ``[n_layers, d_model]``,
    float32 on CPU: the tensor block k added to the residual stream at the
    FINAL position. Together with ``capture.streams[0]`` (the embedding
    stream) they decompose the final pre-norm stream exactly:

        streams[L][-1] == streams[0][-1] + sum_k attn_contrib[k] + mlp_contrib[k]

    (up to the accumulation rounding of the model's compute dtype; the
    decomposition check below measures and enforces this).
    """

    capture: ForwardCapture
    attn_contrib: Any   # torch.Tensor [L, d_model] float32 cpu
    mlp_contrib: Any    # torch.Tensor [L, d_model] float32 cpu


def _first_module_path(block: Any, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if getattr(block, name, None) is not None:
            return name
    return None


def _contrib_hook(store: dict, key: tuple) -> Any:
    """Forward hook capturing a module's output at the final position."""

    def hook(module: Any, hook_args: tuple, output: Any) -> None:
        out = output[0] if isinstance(output, tuple) else output
        store[key] = tensor_cpu_float(out[0, -1])

    return hook


def resolve_component_anatomy(
    ctx: RunContext, bundle: ModelBundle, probe_prompt: str, *, rel_tolerance: float = 0.02
) -> ComponentAnatomy:
    """Probe and VERIFY where each block's attn/mlp contributions live.

    Captures both the raw submodule outputs and (when present) the post-norm
    outputs in a single forward, then selects the (attn, mlp) source pair
    whose sum reconstructs every block's residual delta at the final
    position. ``rel_tolerance`` is relative to the delta's norm and must
    absorb only the model dtype's residual-add rounding (bf16: ~1%).
    """
    import torch

    block0 = bundle.blocks[0]
    attn_path = _first_module_path(block0, ATTN_MODULE_CANDIDATES)
    mlp_path = _first_module_path(block0, MLP_MODULE_CANDIDATES)
    if attn_path is None or mlp_path is None:
        raise RuntimeError(
            f"Could not find attention ({ATTN_MODULE_CANDIDATES}) or MLP "
            f"({MLP_MODULE_CANDIDATES}) submodules on the decoder block. Add "
            "this architecture's paths to interp_bench.py."
        )
    post_attn_path = _first_module_path(block0, POST_ATTN_NORM_CANDIDATES)
    post_mlp_path = _first_module_path(block0, POST_MLP_NORM_CANDIDATES)

    store: dict[tuple, Any] = {}
    handles = []
    try:
        for i, block in enumerate(bundle.blocks):
            handles.append(getattr(block, attn_path).register_forward_hook(_contrib_hook(store, ("attn", "module", i))))
            handles.append(getattr(block, mlp_path).register_forward_hook(_contrib_hook(store, ("mlp", "module", i))))
            if post_attn_path:
                handles.append(
                    getattr(block, post_attn_path).register_forward_hook(_contrib_hook(store, ("attn", "post_norm", i)))
                )
            if post_mlp_path:
                handles.append(
                    getattr(block, post_mlp_path).register_forward_hook(_contrib_hook(store, ("mlp", "post_norm", i)))
                )
        capture = run_with_residual_cache(bundle, probe_prompt)
    finally:
        for h in handles:
            h.remove()

    n_layers = bundle.anatomy.n_layers
    deltas = [capture.streams[k + 1, -1] - capture.streams[k, -1] for k in range(n_layers)]

    attn_sources = ["module"] + (["post_norm"] if post_attn_path else [])
    mlp_sources = ["module"] + (["post_norm"] if post_mlp_path else [])
    results: dict[tuple[str, str], float] = {}
    for a_src in attn_sources:
        for m_src in mlp_sources:
            worst = 0.0
            for k in range(n_layers):
                recon = store[("attn", a_src, k)] + store[("mlp", m_src, k)]
                denom = max(float(deltas[k].norm()), 1e-9)
                worst = max(worst, float((recon - deltas[k]).norm()) / denom)
            results[(a_src, m_src)] = worst

    (best_a, best_m), best_err = min(results.items(), key=lambda kv: kv[1])
    # The gate is a MAX over n_layers per-block reconstructions, so deeper
    # models draw more samples from the same bf16 rounding distribution and
    # the worst block grows even when the decomposition is exactly right
    # (32B/64-layer first contact: best pair correct at ~0.024-0.029 vs the
    # 32-layer-calibrated 0.02 — a near-miss, not a wrong hook point; a wrong
    # pair fails by 10x+). Widen as sqrt(n_layers / 32) for depth, and by an
    # extra low-precision factor for bf16/fp16: the per-block reconstruction
    # error is dominated by mantissa rounding, which is ~8x coarser in bf16
    # than fp32. Both stay far below the >0.2 a wrong decomposition produces.
    low_precision = next(bundle.model.parameters()).dtype in (torch.bfloat16, torch.float16)
    precision_factor = 1.6 if low_precision else 1.0
    effective_tolerance = rel_tolerance * max(1.0, (n_layers / 32.0) ** 0.5) * precision_factor
    diag = {
        "probe_prompt": probe_prompt,
        "candidates_tried": {f"attn={a},mlp={m}": err for (a, m), err in results.items()},
        "selected": {"attn_source": best_a, "mlp_source": best_m},
        "max_block_recon_rel_err": best_err,
        "rel_tolerance": rel_tolerance,
        "effective_rel_tolerance": effective_tolerance,
        "low_precision": low_precision,
        "precision_factor": precision_factor,
        "explanation": (
            "For each candidate hook-point pair, every block's captured "
            "attn+mlp contribution must reconstruct that block's residual "
            "delta streams[k+1]-streams[k] at the final position. The "
            "selected pair is the verified place this model adds its "
            "components to the residual stream."
        ),
    }
    path = ctx.path("diagnostics", "component_anatomy.json")
    write_json(path, diag)
    ctx.register_artifact(path, "diagnostic", "Verified hook points for per-block attn/MLP contributions.")

    if best_err > effective_tolerance:
        raise RuntimeError(
            f"No contribution hook-point pair reconstructs the per-block residual "
            f"deltas (best: attn={best_a}, mlp={best_m}, max rel err {best_err:.4f} > "
            f"tolerance {effective_tolerance:.4f} = {rel_tolerance} x sqrt({n_layers}/32) x {precision_factor} (bf16/fp16 factor)). "
            "This architecture adds components to the residual stream somewhere the "
            "candidates do not cover; see diagnostics/component_anatomy.json."
        )

    print(
        f"[bench] component anatomy: attn={best_a}, mlp={best_m} "
        f"(max block reconstruction rel err = {best_err:.2e})"
    )
    return ComponentAnatomy(
        attn_module_path=attn_path,
        mlp_module_path=mlp_path,
        attn_source=best_a,
        mlp_source=best_m,
        attn_hook_path=attn_path if best_a == "module" else post_attn_path,
        mlp_hook_path=mlp_path if best_m == "module" else post_mlp_path,
        max_block_recon_rel_err=best_err,
        probe_prompt=probe_prompt,
    )


def _contrib_hook_all_positions(store: dict, key: tuple) -> Any:
    """Forward hook capturing a module's output at every position."""

    def hook(module: Any, hook_args: tuple, output: Any) -> None:
        out = output[0] if isinstance(output, tuple) else output
        store[key] = tensor_cpu_float(out[0])

    return hook


def run_with_component_cache(
    bundle: ModelBundle, prompt: str, comp_anatomy: ComponentAnatomy, *, all_positions: bool = False
) -> ComponentCapture:
    """Run one prompt capturing residual streams AND per-block contributions.

    With ``all_positions=True`` the contribution tensors are [L, seq, d_model]
    instead of [L, d_model] (final position only) — used by patching labs that
    need clean component outputs at arbitrary positions.
    """
    import torch

    hook_factory = _contrib_hook_all_positions if all_positions else _contrib_hook
    store: dict[tuple, Any] = {}
    handles = []
    try:
        for i, block in enumerate(bundle.blocks):
            handles.append(
                getattr(block, comp_anatomy.attn_hook_path).register_forward_hook(hook_factory(store, ("attn", i)))
            )
            handles.append(
                getattr(block, comp_anatomy.mlp_hook_path).register_forward_hook(hook_factory(store, ("mlp", i)))
            )
        capture = run_with_residual_cache(bundle, prompt)
    finally:
        for h in handles:
            h.remove()

    n_layers = bundle.anatomy.n_layers
    return ComponentCapture(
        capture=capture,
        attn_contrib=torch.stack([store[("attn", i)] for i in range(n_layers)]),
        mlp_contrib=torch.stack([store[("mlp", i)] for i in range(n_layers)]),
    )


def run_decomposition_check(
    ctx: RunContext, bundle: ModelBundle, comp: ComponentCapture, *, rel_tolerance: float = 0.02
) -> dict[str, Any]:
    """Self-check 3: components must sum to the final pre-norm stream.

    embeddings + sum(attn contributions) + sum(mlp contributions) must equal
    streams[L] at the final position, up to the model compute dtype's
    residual-accumulation rounding. If this fails, every attribution number
    downstream is bookkeeping fiction; abort.
    """
    final_stream = comp.capture.streams[-1, -1]
    recon = comp.capture.streams[0, -1] + comp.attn_contrib.sum(dim=0) + comp.mlp_contrib.sum(dim=0)
    abs_err = float((recon - final_stream).norm())
    rel_err = abs_err / max(float(final_stream.norm()), 1e-9)
    result = {
        "prompt": comp.capture.prompt,
        "rel_err": rel_err,
        "abs_err": abs_err,
        "final_stream_norm": float(final_stream.norm()),
        "rel_tolerance": rel_tolerance,
        "ok": rel_err <= rel_tolerance,
        "explanation": (
            "The DLA ledger is only meaningful if embeddings + all captured "
            "attn/MLP contributions reconstruct the final pre-norm residual "
            "stream. rel_err measures the unexplained remainder; the "
            "tolerance absorbs the model dtype's residual-add rounding only."
        ),
    }
    path = ctx.path("diagnostics", "dla_decomposition_check.json")
    write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Proof that captured components sum to the final residual stream.")
    status = "OK" if result["ok"] else "FAILED"
    print(f"[bench] decomposition check: {status} (rel err = {rel_err:.2e})")
    if not result["ok"]:
        raise RuntimeError(
            "Component decomposition check failed: captured contributions do "
            "not sum to the final residual stream. See "
            "diagnostics/dla_decomposition_check.json."
        )
    return result


def run_with_component_ablation(
    bundle: ModelBundle,
    prompt: str,
    comp_anatomy: ComponentAnatomy,
    component_type: str,
    layer: int,
) -> Any:
    """Forward pass with one component's FINAL-POSITION output zeroed.

    This is *direct-path* ablation: it removes exactly the contribution DLA
    scored (the write to the final position's residual stream) and nothing
    else. Contributions at earlier positions, which can reach the output
    indirectly through later attention, are deliberately left intact so the
    causal effect is commensurable with the attribution score.
    Returns the model's final-position logits, float32 on CPU.
    """
    import torch

    hook_path = comp_anatomy.attn_hook_path if component_type == "attn" else comp_anatomy.mlp_hook_path
    module = getattr(bundle.blocks[layer], hook_path)

    def ablate_hook(mod: Any, hook_args: tuple, output: Any) -> Any:
        if isinstance(output, tuple):
            out = output[0].clone()
            out[0, -1] = 0
            return (out,) + tuple(output[1:])
        out = output.clone()
        out[0, -1] = 0
        return out

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)

    handle = module.register_forward_hook(ablate_hook)
    try:
        with torch.no_grad():
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return tensor_cpu_float(out.logits[0, -1])


# ---------------------------------------------------------------------------
# Head-level capture: attention patterns and per-head output contributions
# ---------------------------------------------------------------------------
#
# An attention block's output is linear in its heads: the out-projection's
# input is the concatenation of per-head outputs, so head h's write into the
# block output is  o_in[h*d_head:(h+1)*d_head] @ W_O[h-slice]  (the projection
# bias is a shared constant belonging to no head). The bench captures the
# out-projection input with a pre-hook and verifies the per-head decomposition
# reconstructs the block's attention output before any lab consumes it.
#
# Attention PATTERNS come from output_attentions=True, which requires the
# eager attention implementation -- sdpa/flash return an empty tuple with no
# warning in transformers 5 (verified). Labs that need patterns are marked
# needs_eager in LAB_PROFILES and the capture below hard-fails if patterns
# are missing.

O_PROJ_PATH_CANDIDATES = (
    "self_attn.o_proj",   # Llama / Olmo / Gemma / Qwen / Mistral style
    "attn.c_proj",        # GPT-2 style (Conv1D: weight is [in, out])
)


@dataclasses.dataclass
class HeadAnatomy:
    """Verified per-head structure of the attention blocks."""

    o_proj_path: str
    n_heads: int
    n_kv_heads: int
    d_head: int
    weight_is_in_by_out: bool   # True for GPT-2 Conv1D, False for nn.Linear
    has_bias: bool
    sliding_window: int | None
    layer_types: tuple[str, ...]
    max_head_recon_rel_err: float


@dataclasses.dataclass
class AttentionCapture:
    """One prompt's streams plus attention patterns and head-output pieces.

    ``attentions``: [n_layers, n_heads, seq, seq] float32 cpu — row q sums
    to 1 over keys <= q (causal).
    ``o_in_last``: [n_layers, n_heads * d_head] — the out-projection INPUT at
    the final position (concatenated per-head outputs before W_O).
    ``attn_out_last``: [n_layers, d_model] — each block's attention-module
    output at the final position (the head decomposition's ground truth).
    """

    capture: ForwardCapture
    attentions: Any
    o_in_last: Any
    attn_out_last: Any


def resolve_head_anatomy(ctx: RunContext, bundle: ModelBundle) -> HeadAnatomy:
    """Resolve out-projection path and head geometry; verification happens in
    run_head_decomposition_check on real activations."""
    config = bundle.model.config
    block0 = bundle.blocks[0]
    o_proj = None
    o_proj_path = ""
    for candidate in O_PROJ_PATH_CANDIDATES:
        with contextlib.suppress(AttributeError):
            o_proj = get_by_path(block0, candidate)
            o_proj_path = candidate
            break
    if o_proj is None:
        raise RuntimeError(
            f"Could not find the attention out-projection. Tried {O_PROJ_PATH_CANDIDATES} "
            "relative to the decoder block; add this architecture's path to "
            "O_PROJ_PATH_CANDIDATES in interp_bench.py."
        )
    n_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 0)))
    n_kv = int(getattr(config, "num_key_value_heads", n_heads) or n_heads)
    d_model = bundle.anatomy.d_model
    d_head = int(getattr(config, "head_dim", 0) or d_model // n_heads)
    # GPT-2's Conv1D stores weight as [in, out]; nn.Linear as [out, in].
    is_conv1d = type(o_proj).__name__ == "Conv1D"
    anatomy = HeadAnatomy(
        o_proj_path=o_proj_path,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        d_head=d_head,
        weight_is_in_by_out=is_conv1d,
        has_bias=getattr(o_proj, "bias", None) is not None,
        sliding_window=getattr(config, "sliding_window", None),
        layer_types=tuple(sorted(set(getattr(config, "layer_types", []) or []))),
        max_head_recon_rel_err=-1.0,  # filled by the decomposition check
    )
    path = ctx.path("diagnostics", "head_anatomy.json")
    write_json(path, anatomy)
    ctx.register_artifact(path, "diagnostic", "Per-head geometry and out-projection orientation.")
    print(
        f"[bench] head anatomy: {n_heads} heads x d_head {d_head} "
        f"(kv heads {n_kv}, o_proj at {o_proj_path}, "
        f"{'Conv1D [in,out]' if is_conv1d else 'Linear [out,in]'})"
    )
    return anatomy


def head_contribution(bundle: ModelBundle, head_anatomy: HeadAnatomy, layer: int, head: int, o_in_vec: Any) -> Any:
    """Head ``head``'s write into block ``layer``'s attention output.

    ``o_in_vec`` is the out-projection input at one position,
    [n_heads * d_head] float32 cpu. Returns a [d_model] float32 cpu vector.
    The projection bias is deliberately excluded: it is shared, not a head's.
    """
    import torch

    o_proj = get_by_path(bundle.blocks[layer], head_anatomy.o_proj_path)
    w = o_proj.weight.detach()
    sl = slice(head * head_anatomy.d_head, (head + 1) * head_anatomy.d_head)
    piece = o_in_vec[sl]
    if head_anatomy.weight_is_in_by_out:
        out = piece @ w[sl, :].to("cpu", torch.float32)
    else:
        out = piece @ w[:, sl].to("cpu", torch.float32).T
    return out


def run_with_attention_cache(
    bundle: ModelBundle, prompt: str, *, all_positions: bool = False, add_special_tokens: bool = True
) -> AttentionCapture:
    """One forward capturing streams, attention patterns, and head pieces.

    Pass ``add_special_tokens=False`` for already-rendered chat-template
    prompts when token positions must align with tokenizer offset mappings.

    With ``all_positions=True``, ``o_in_last``/``attn_out_last`` hold
    full-sequence tensors ([L, seq, n_heads*d_head] / [L, seq, d_model])
    despite their names — circuit labs need every position's head outputs,
    e.g. to compute dataset-mean ablation values.
    """
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)

    captured: dict[str, Any] = {}

    def final_norm_pre_hook(module: Any, hook_args: tuple) -> None:
        captured["final_prenorm"] = tensor_cpu_float(hook_args[0])

    o_in: dict[int, Any] = {}
    attn_out: dict[int, Any] = {}

    def make_o_pre_hook(idx: int):
        def hook(module: Any, hook_args: tuple) -> None:
            o_in[idx] = tensor_cpu_float(hook_args[0][0] if all_positions else hook_args[0][0, -1])

        return hook

    def make_attn_out_hook(idx: int):
        def hook(module: Any, hook_args: tuple, output: Any) -> None:
            out = output[0] if isinstance(output, tuple) else output
            attn_out[idx] = tensor_cpu_float(out[0] if all_positions else out[0, -1])

        return hook

    handles = []
    try:
        handles.append(bundle.final_norm.register_forward_pre_hook(final_norm_pre_hook))
        for i, block in enumerate(bundle.blocks):
            attn_module_path = _first_module_path(block, ATTN_MODULE_CANDIDATES)
            o_proj = get_by_path(block, f"{attn_module_path}.o_proj") if hasattr(
                getattr(block, attn_module_path), "o_proj"
            ) else get_by_path(block, f"{attn_module_path}.c_proj")
            handles.append(o_proj.register_forward_pre_hook(make_o_pre_hook(i)))
            handles.append(getattr(block, attn_module_path).register_forward_hook(make_attn_out_hook(i)))
        with torch.no_grad():
            out = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()

    if not out.attentions:
        raise RuntimeError(
            "The model returned no attention patterns. This happens silently "
            "with sdpa/flash attention in transformers 5 -- rerun with "
            "--attn-implementation eager (lab3 sets this automatically unless "
            "you overrode it)."
        )
    if "final_prenorm" not in captured:
        raise RuntimeError("The final-norm pre-hook never fired; check diagnostics/model_anatomy.md.")

    hs = out.hidden_states
    n_layers = bundle.anatomy.n_layers
    if len(hs) != n_layers + 1:
        raise RuntimeError(f"Expected {n_layers + 1} hidden states, got {len(hs)}.")
    streams = torch.stack([tensor_cpu_float(h[0]) for h in hs[:-1]] + [captured["final_prenorm"][0]])

    ids = input_ids[0].detach().cpu().tolist()
    capture = ForwardCapture(
        prompt=prompt,
        input_ids=ids,
        tokens_raw=tokenizer.convert_ids_to_tokens(ids),
        tokens_text=[tokenizer.decode([i]) for i in ids],
        streams=streams,
        final_logits_last=tensor_cpu_float(out.logits[0, -1]),
    )
    return AttentionCapture(
        capture=capture,
        attentions=torch.stack([tensor_cpu_float(a[0]) for a in out.attentions]),
        o_in_last=torch.stack([o_in[i] for i in range(n_layers)]),
        attn_out_last=torch.stack([attn_out[i] for i in range(n_layers)]),
    )


def run_head_decomposition_check(
    ctx: RunContext, bundle: ModelBundle, head_anatomy: HeadAnatomy, att: AttentionCapture, *,
    rel_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Self-check: per-head pieces (+ shared bias) must rebuild each block's
    attention output at the final position. Aborts on failure."""
    import torch

    worst = 0.0
    for layer in range(bundle.anatomy.n_layers):
        total = torch.zeros(bundle.anatomy.d_model)
        for head in range(head_anatomy.n_heads):
            total += head_contribution(bundle, head_anatomy, layer, head, att.o_in_last[layer])
        o_proj = get_by_path(bundle.blocks[layer], head_anatomy.o_proj_path)
        bias = getattr(o_proj, "bias", None)
        if bias is not None:
            total += bias.detach().to("cpu", torch.float32)
        denom = max(float(att.attn_out_last[layer].norm()), 1e-9)
        worst = max(worst, float((total - att.attn_out_last[layer]).norm()) / denom)

    head_anatomy.max_head_recon_rel_err = worst
    anatomy_path = ctx.path("diagnostics", "head_anatomy.json")
    write_json(anatomy_path, head_anatomy)
    ctx.artifacts = [
        entry for entry in ctx.artifacts
        if entry.get("path") != "diagnostics/head_anatomy.json"
    ]
    ctx.register_artifact(anatomy_path, "diagnostic", "Per-head geometry and out-projection orientation.")

    result = {
        "prompt": att.capture.prompt,
        "max_layer_rel_err": worst,
        "rel_tolerance": rel_tolerance,
        "ok": worst <= rel_tolerance,
        "explanation": (
            "Head-level attribution is only meaningful if the per-head slices "
            "of the out-projection input, mapped through their W_O columns, "
            "sum (with the shared bias) to the block's actual attention "
            "output. The worst per-layer relative error is reported."
        ),
    }
    path = ctx.path("diagnostics", "head_decomposition_check.json")
    write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Proof that per-head pieces rebuild each block's attention output.")
    status = "OK" if result["ok"] else "FAILED"
    print(f"[bench] head decomposition check: {status} (max layer rel err = {worst:.2e})")
    if not result["ok"]:
        raise RuntimeError(
            "Head decomposition check failed; per-head slicing does not match "
            "this architecture. See diagnostics/head_decomposition_check.json."
        )
    return result


def run_with_head_ablation(
    bundle: ModelBundle,
    prompt: str,
    head_anatomy: HeadAnatomy,
    layer: int,
    head: int,
    scope: str = "final_pos",
) -> Any:
    """Forward pass with one attention head's output zeroed.

    ``scope='final_pos'`` zeroes the head's write at the final position only —
    commensurable with the head's direct logit attribution (Lab 2's
    convention). ``scope='all_pos'`` removes the head everywhere, including
    its writes at earlier positions that later layers may read: the gap
    between the two scopes is composition made measurable.
    Returns final-position logits, float32 cpu.
    """
    import torch

    if scope not in ("final_pos", "all_pos"):
        raise ValueError(f"Unknown ablation scope {scope!r}")
    block = bundle.blocks[layer]
    attn_module_path = _first_module_path(block, ATTN_MODULE_CANDIDATES)
    attn_module = getattr(block, attn_module_path)
    o_proj = attn_module.o_proj if hasattr(attn_module, "o_proj") else attn_module.c_proj
    sl = slice(head * head_anatomy.d_head, (head + 1) * head_anatomy.d_head)

    def ablate_pre_hook(module: Any, hook_args: tuple) -> Any:
        x = hook_args[0].clone()
        if scope == "final_pos":
            x[0, -1, sl] = 0
        else:
            x[..., sl] = 0
        return (x,) + tuple(hook_args[1:])

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)

    handle = o_proj.register_forward_pre_hook(ablate_pre_hook)
    try:
        with torch.no_grad():
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return tensor_cpu_float(out.logits[0, -1])


def run_with_node_set_ablation(
    bundle: ModelBundle,
    prompt: str,
    head_anatomy: HeadAnatomy,
    comp_anatomy: ComponentAnatomy,
    heads: Sequence[tuple[int, int]] = (),
    mlps: Sequence[int] = (),
    head_means: Any = None,   # [L, seq, n_heads*d_head] float32 cpu, or None for zero
    mlp_means: Any = None,    # [L, seq, d_model] float32 cpu, or None for zero
) -> Any:
    """Forward pass with a SET of heads and MLP layers ablated at all positions.

    The circuit lab's workhorse: faithfulness ablates the complement of the
    circuit (hundreds of heads at once), completeness ablates the circuit
    itself. ``mode`` is implied by the means: dataset-mean ablation when mean
    tensors are given (the convention that keeps the model in-distribution),
    zero-ablation when None. Mean tensors assume the dataset's fixed prompt
    length; a length mismatch raises rather than truncating silently.
    Returns final-position logits, float32 cpu.
    """
    import torch

    heads_by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        heads_by_layer.setdefault(layer, []).append(head)
    mlp_set = set(mlps)
    d_head = head_anatomy.d_head

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    seq = input_ids.shape[1]
    if head_means is not None and head_means.shape[1] != seq:
        raise ValueError(f"head_means seq {head_means.shape[1]} != prompt seq {seq}")
    if mlp_means is not None and mlp_means.shape[1] != seq:
        raise ValueError(f"mlp_means seq {mlp_means.shape[1]} != prompt seq {seq}")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)

    handles = []

    def make_head_hook(layer: int, head_list: list[int]):
        def hook(module: Any, hook_args: tuple) -> Any:
            x = hook_args[0].clone()
            for head in head_list:
                sl = slice(head * d_head, (head + 1) * d_head)
                if head_means is None:
                    x[0, :, sl] = 0
                else:
                    x[0, :, sl] = head_means[layer, :, sl].to(x.device, x.dtype)
            return (x,) + tuple(hook_args[1:])

        return hook

    def make_mlp_hook(layer: int):
        def hook(module: Any, hook_args: tuple, output: Any) -> Any:
            out = output[0] if isinstance(output, tuple) else output
            out = out.clone()
            if mlp_means is None:
                out[0, :, :] = 0
            else:
                out[0, :, :] = mlp_means[layer].to(out.device, out.dtype)
            return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out

        return hook

    try:
        for layer, head_list in heads_by_layer.items():
            block = bundle.blocks[layer]
            attn_module_path = _first_module_path(block, ATTN_MODULE_CANDIDATES)
            attn_module = getattr(block, attn_module_path)
            o_proj = attn_module.o_proj if hasattr(attn_module, "o_proj") else attn_module.c_proj
            handles.append(o_proj.register_forward_pre_hook(make_head_hook(layer, head_list)))
        for layer in mlp_set:
            module = getattr(bundle.blocks[layer], comp_anatomy.mlp_hook_path)
            handles.append(module.register_forward_hook(make_mlp_hook(layer)))
        with torch.no_grad():
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return tensor_cpu_float(out.logits[0, -1])


# ---------------------------------------------------------------------------
# Activation patching: interchange interventions on the residual stream (Lab 5+)
# ---------------------------------------------------------------------------
#
# The patch convention matches the stream convention exactly: patching
# "streams[k] at position p" replaces the INPUT to block k (for k < L) or the
# input to the final norm (k = L) at that position. A run patched with its
# own vectors is therefore a no-op — and run_patch_noop_check enforces that
# (max |Δlogit| ≤ 1e-4) before any patching science, because a silent
# off-by-one in layer or position indexing would produce a beautiful, wrong
# heatmap.


def _forward_logits(bundle: ModelBundle, prompt: str, pre_hooks: Sequence[tuple[Any, Any]]) -> Any:
    """One forward with the given (module, pre_hook_fn) pairs installed."""
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)
    handles = [m.register_forward_pre_hook(fn) for m, fn in pre_hooks]
    try:
        with torch.no_grad():
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return tensor_cpu_float(out.logits[0, -1])


def run_with_residual_patch(
    bundle: ModelBundle, prompt: str, layer: int, position: int, vector: Any
) -> Any:
    """Forward pass with streams[layer][position] replaced by ``vector``.

    ``vector`` is a [d_model] float32 CPU tensor (the bench's stream storage
    format); it is cast to the model's device/dtype at the hook site.
    Returns final-position logits, float32 CPU.
    """
    n_layers = bundle.anatomy.n_layers
    if not 0 <= layer <= n_layers:
        raise ValueError(f"stream layer must be in [0, {n_layers}], got {layer}")
    module = bundle.final_norm if layer == n_layers else bundle.blocks[layer]

    def patch_hook(mod: Any, hook_args: tuple) -> Any:
        hidden = hook_args[0].clone()
        if not -hidden.shape[1] <= position < hidden.shape[1]:
            raise ValueError(
                f"patch position {position} out of range for sequence length "
                f"{hidden.shape[1]}; clean and corrupt prompts must be token-aligned."
            )
        hidden[0, position] = vector.to(hidden.device, hidden.dtype)
        return (hidden,) + tuple(hook_args[1:])

    return _forward_logits(bundle, prompt, [(module, patch_hook)])


def run_with_residual_patch_batched(
    bundle: ModelBundle,
    prompt: str,
    cells: Sequence[tuple[int, int, Any]],
    max_batch: int = 64,
) -> list[Any]:
    """Run many single-cell residual patches on ONE prompt in batched forwards.

    Each cell is ``(layer, position, vector)``: row i is the prompt with
    ``streams[layer_i][position_i]`` replaced by ``vector_i``. Rows never
    interact within a forward (attention is within-sequence), so the result is
    identical to ``len(cells)`` separate ``run_with_residual_patch`` calls —
    bit-for-bit in fp32, within rounding in bf16 — but one batched forward reads
    the model weights once instead of once per cell. That is the difference
    between a GPU running at batch 1 (memory-bandwidth bound, mostly idle) and
    actually being used. Returns final-position logits (float32 CPU) per cell, in
    input order. Cells are chunked at ``max_batch`` to bound activation memory.
    """
    import torch

    if not cells:
        return []
    n_layers = bundle.anatomy.n_layers
    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    base_ids = encoded["input_ids"]
    base_mask = encoded.get("attention_mask")
    results: list[Any] = []
    for start in range(0, len(cells), max_batch):
        chunk = cells[start:start + max_batch]
        batch = len(chunk)
        input_ids = base_ids.repeat(batch, 1).to(bundle.input_device)
        attention_mask = (
            base_mask.repeat(batch, 1).to(bundle.input_device)
            if base_mask is not None else None
        )
        rows_by_layer: dict[int, list[int]] = {}
        for row, (layer, _position, _vector) in enumerate(chunk):
            if not 0 <= layer <= n_layers:
                raise ValueError(f"stream layer must be in [0, {n_layers}], got {layer}")
            rows_by_layer.setdefault(layer, []).append(row)

        def make_patch_hook(rows: list[int]) -> Any:
            def patch_hook(mod: Any, hook_args: tuple) -> Any:
                hidden = hook_args[0].clone()
                seq_len = hidden.shape[1]
                for row in rows:
                    _layer, position, vector = chunk[row]
                    if not -seq_len <= position < seq_len:
                        raise ValueError(
                            f"patch position {position} out of range for sequence "
                            f"length {seq_len}; prompts must be token-aligned."
                        )
                    hidden[row, position] = vector.to(hidden.device, hidden.dtype)
                return (hidden,) + tuple(hook_args[1:])
            return patch_hook

        handles = []
        for layer, rows in rows_by_layer.items():
            module = bundle.final_norm if layer == n_layers else bundle.blocks[layer]
            handles.append(module.register_forward_pre_hook(make_patch_hook(rows)))
        try:
            with torch.no_grad():
                out = bundle.model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
        finally:
            for handle in handles:
                handle.remove()
        for row in range(batch):
            results.append(tensor_cpu_float(out.logits[row, -1]))
    return results


def run_with_component_patch(
    bundle: ModelBundle,
    prompt: str,
    comp_anatomy: ComponentAnatomy,
    component_type: str,
    layer: int,
    position: int,
    vector: Any,
) -> Any:
    """Forward pass with one component's write at one position replaced.

    Replaces the verified contribution tensor (module output, or post-norm
    output on post-norm architectures) for ``attn`` or ``mlp`` at ``layer``,
    ``position`` — the same object Lab 2 scored and Lab 3 ablated.
    """
    import torch

    hook_path = comp_anatomy.attn_hook_path if component_type == "attn" else comp_anatomy.mlp_hook_path
    module = getattr(bundle.blocks[layer], hook_path)

    def patch_out_hook(mod: Any, hook_args: tuple, output: Any) -> Any:
        if isinstance(output, tuple):
            out = output[0].clone()
            out[0, position] = vector.to(out.device, out.dtype)
            return (out,) + tuple(output[1:])
        out = output.clone()
        out[0, position] = vector.to(out.device, out.dtype)
        return out

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)
    handle = module.register_forward_hook(patch_out_hook)
    try:
        with torch.no_grad():
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return tensor_cpu_float(out.logits[0, -1])


def run_patch_noop_check(
    ctx: RunContext, bundle: ModelBundle, prompt: str, *, atol: float = 1e-4
) -> dict[str, Any]:
    """Self-check: patching a run with its own vectors must change nothing.

    Verifies hook placement and the streams[k] convention end-to-end: one
    early-block patch, one late-block patch, and the final-norm patch, each
    at two positions, against the unpatched logits.
    """
    capture = run_with_residual_cache(bundle, prompt)
    n_layers = bundle.anatomy.n_layers
    seq = len(capture.input_ids)
    test_points = [(0, 0), (0, seq - 1), (n_layers // 2, seq // 2),
                   (n_layers - 1, seq - 1), (n_layers, seq - 1)]
    worst = 0.0
    for layer, pos in test_points:
        logits = run_with_residual_patch(bundle, prompt, layer, pos, capture.streams[layer, pos])
        worst = max(worst, float((logits - capture.final_logits_last).abs().max()))
    result = {
        "prompt": prompt,
        "test_points": [list(p) for p in test_points],
        "max_abs_logit_diff": worst,
        "atol": atol,
        "ok": worst <= atol,
        "explanation": (
            "Replacing streams[k][p] with the run's own vector is an identity "
            "operation; any logit change means the patch hooks do not target "
            "the tensors the stream convention names, and every patching "
            "result downstream would be a well-rendered lie."
        ),
    }
    path = ctx.path("diagnostics", "patch_noop_check.json")
    write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Proof that self-patching is a no-op (hook/convention alignment).")
    status = "OK" if result["ok"] else "FAILED"
    print(f"[bench] patch no-op check: {status} (max |dlogit| = {worst:.2e})")
    if not result["ok"]:
        raise RuntimeError(
            "Patch no-op check failed: self-patching changed the logits. See "
            "diagnostics/patch_noop_check.json."
        )
    return result


# ---------------------------------------------------------------------------
# Rank-one weight edits (Lab 5 extension): safe apply/restore plumbing
# ---------------------------------------------------------------------------

MLP_DOWN_PROJ_CANDIDATES = (
    "mlp.down_proj",   # Llama / Olmo / Gemma / Qwen / Mistral style (Linear [out,in])
    "mlp.c_proj",      # GPT-2 style (Conv1D [in,out])
)


def resolve_mlp_down_proj(bundle: ModelBundle, layer: int) -> tuple[Any, bool]:
    """Return (module, weight_is_in_by_out) for the MLP down-projection."""
    block = bundle.blocks[layer]
    for candidate in MLP_DOWN_PROJ_CANDIDATES:
        with contextlib.suppress(AttributeError):
            return get_by_path(block, candidate), type(get_by_path(block, candidate)).__name__ == "Conv1D"
    raise RuntimeError(
        f"Could not find the MLP down-projection. Tried {MLP_DOWN_PROJ_CANDIDATES}; "
        "add this architecture's path to interp_bench.py."
    )


@contextlib.contextmanager
def temporary_rank_one_edit(bundle: ModelBundle, layer: int, key: Any, delta_v: Any):
    """Apply W <- W + delta_v key^T / (key^T key) to the MLP down-projection,
    restore the original weight on exit no matter what.

    ``key`` [d_ff] and ``delta_v`` [d_model] are float32 CPU tensors. For any
    input k, the edit shifts the output by delta_v * (key.k)/(key.key): exact
    for k = key, fading with key-overlap — which is precisely what the
    spillover evaluation measures.
    """
    import torch

    module, in_by_out = resolve_mlp_down_proj(bundle, layer)
    weight = module.weight
    if not weight.is_floating_point():
        raise RuntimeError(
            f"--run-edit needs float weights, but the MLP down-projection is {weight.dtype} "
            "(quantized). A rank-one float edit cannot be applied to quantized weights; "
            "re-run lab5 without --quantization."
        )
    original = weight.detach().clone()
    k = key.to(weight.device, torch.float32)
    dv = delta_v.to(weight.device, torch.float32)
    denom = float(k @ k)
    if denom <= 0:
        raise ValueError("rank-one edit key has zero norm")
    if in_by_out:   # Conv1D: out = x @ W, W [d_ff, d_model]
        delta = torch.outer(k, dv) / denom
    else:           # Linear: out = x @ W.T, W [d_model, d_ff]
        delta = torch.outer(dv, k) / denom
    with torch.no_grad():
        weight.add_(delta.to(weight.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            weight.copy_(original)


# ---------------------------------------------------------------------------
# Human-readable state dumps
# ---------------------------------------------------------------------------
#
# The design rule: every dumped fact about model state must be inspectable
# with a text editor. Token tables are CSV with the whitespace made visible;
# activations are summarized as per-layer statistics and *decoded* through the
# lens rather than printed as raw arrays. Raw tensors are opt-in and always
# ship with a manifest explaining each entry.


def dump_example_state(
    ctx: RunContext,
    bundle: ModelBundle,
    example_id: str,
    capture: ForwardCapture,
    traj: LensTrajectory,
    *,
    target: str | None = None,
    distractor: str | None = None,
) -> pathlib.Path:
    """Write the per-example state directory. Returns its path."""
    state_dir = ctx.run_dir / "state" / sanitize_tag(example_id)

    # --- tokens.csv: exactly what the model saw, position by position.
    token_rows = []
    cumulative = ""
    for i, (tid, raw, text) in enumerate(zip(capture.input_ids, capture.tokens_raw, capture.tokens_text)):
        cumulative += text
        token_rows.append(
            {
                "position": i,
                "token_id": tid,
                "token_raw": raw,
                "token_text": text,
                "token_visible": visible_token(text),
                "cumulative_text_visible": visible_token(cumulative),
                "is_final": i == len(capture.input_ids) - 1,
            }
        )
    write_csv(state_dir / "tokens.csv", token_rows)

    # --- residual_norms_by_position.csv: depth x position L2-norm grid.
    # This is the "shape" of the forward pass: where the stream grows, which
    # positions carry unusually large state, whether the BOS column is an
    # outlier (it usually is -- worth seeing once).
    norms = capture.streams.norm(dim=-1)  # [L+1, seq]
    norm_rows = []
    for depth in range(norms.shape[0]):
        row: dict[str, Any] = {"depth": depth}
        for pos in range(norms.shape[1]):
            row[f"pos_{pos}"] = round(float(norms[depth, pos]), 3)
        norm_rows.append(row)
    write_csv(state_dir / "residual_norms_by_position.csv", norm_rows)

    # --- residual_stats_final_pos.csv: per-depth summary stats at the
    # position being read out. Numbers a human can sanity-check, not arrays.
    final_streams = capture.streams[:, -1, :]
    stat_rows = []
    for depth in range(final_streams.shape[0]):
        v = final_streams[depth]
        stat_rows.append(
            {
                "depth": depth,
                "l2_norm": round(float(v.norm()), 4),
                "rms": round(float(v.pow(2).mean().sqrt()), 5),
                "mean": round(float(v.mean()), 6),
                "std": round(float(v.std()), 5),
                "abs_max": round(float(v.abs().max()), 4),
            }
        )
    write_csv(state_dir / "residual_stats_final_pos.csv", stat_rows)

    # --- logit_lens_topk.csv: the decoded view of "what the stream says".
    write_csv(state_dir / "logit_lens_topk.csv", traj.topk_rows)

    # --- lens_trajectory.csv: one row per depth, the lab's working table.
    traj_rows = []
    for depth in range(traj.n_depths):
        row = {
            "depth": depth,
            "top1_token_id": traj.top1_ids[depth],
            "top1_token": traj.top1_texts[depth],
            "top1_prob": round(traj.top1_probs[depth], 6),
            "top2_token_id": traj.top2_ids[depth],
            "top2_token": traj.top2_texts[depth],
            "top2_prob": round(traj.top2_probs[depth], 6),
            "top1_margin": round(traj.top1_margin[depth], 6),
            "entropy_bits": round(traj.entropy_bits[depth], 4),
            "kl_to_final_bits": round(traj.kl_to_final_bits[depth], 4),
            "cosine_to_final": round(traj.cosine_to_final[depth], 5),
            "cosine_to_prev": "" if traj.cosine_to_prev[depth] is None else round(traj.cosine_to_prev[depth], 5),
            "resid_l2": round(traj.resid_l2[depth], 3),
            "stream_delta_l2": round(traj.stream_delta_l2[depth], 3),
        }
        if traj.p_target is not None:
            row["p_target"] = round(traj.p_target[depth], 6)
            row["logit_target"] = round(traj.logit_target[depth], 4)
            row["target_rank"] = traj.target_rank[depth] if traj.target_rank is not None else ""
        if traj.p_distractor is not None:
            row["p_distractor"] = round(traj.p_distractor[depth], 6)
            row["logit_distractor"] = round(traj.logit_distractor[depth], 4)
            row["distractor_rank"] = traj.distractor_rank[depth] if traj.distractor_rank is not None else ""
        if traj.p_target is not None and traj.p_distractor is not None:
            row["logit_diff_target_minus_distractor"] = round(
                traj.logit_target[depth] - traj.logit_distractor[depth], 4
            )
        traj_rows.append(row)
    write_csv(state_dir / "lens_trajectory.csv", traj_rows)

    # --- state_card.md: the narrative view, designed to be read top to
    # bottom by a human deciding whether the run makes sense.
    state_card_path = state_dir / "state_card.md"
    write_text(state_card_path, render_state_card(bundle, example_id, capture, traj, target, distractor))
    ctx.register_artifact(state_card_path, "state", f"Per-example state card for {example_id}.")

    # --- optional raw tensors, with a manifest so nothing is a mystery blob.
    if ctx.args.save_tensors:
        import torch

        tensor_path = state_dir / "residual_streams.pt"
        torch.save({"streams": capture.streams.cpu()}, tensor_path)
        write_json(
            state_dir / "tensor_manifest.json",
            {
                "residual_streams.pt": {
                    "key": "streams",
                    "shape": list(capture.streams.shape),
                    "dtype": "float32",
                    "semantics": (
                        "Pre-norm residual stream; index [k, p, :] is the "
                        "stream after k blocks at token position p. k=0 is "
                        "the embedding output."
                    ),
                }
            },
        )
    return state_dir


def render_state_card(
    bundle: ModelBundle,
    example_id: str,
    capture: ForwardCapture,
    traj: LensTrajectory,
    target: str | None,
    distractor: str | None,
) -> str:
    """Render the per-example markdown state card."""
    a = bundle.anatomy
    L = a.n_layers
    lines = [
        f"# State card: {example_id}",
        "",
        f"- model: `{a.model_id}` ({L} blocks, d_model {a.d_model})",
        f"- prompt: `{capture.prompt}`",
        f"- target: `{visible_token(target) if target else '(none)'}`"
        + (f" | distractor: `{visible_token(distractor)}`" if distractor else ""),
        f"- depth convention: 0 = embeddings, k = after k blocks, {L} = pre-final-norm",
        "- all readouts are at the final token position",
        "",
        "## Tokens as the model saw them",
        "",
        "| pos | id | token |",
        "|---:|---:|---|",
    ]
    for i, (tid, text) in enumerate(zip(capture.input_ids, capture.tokens_text)):
        lines.append(f"| {i} | {tid} | `{visible_token(text)}` |")

    header = "| depth | top-1 | p(top1) | margin | entropy | KL->final | cos->final | delta L2 |"
    sep = "|---:|---|---:|---:|---:|---:|---:|---:|"
    if traj.p_target is not None:
        header += " p(target) |"
        sep += "---:|"
    if traj.p_target is not None and traj.p_distractor is not None:
        header += " logit diff (t-d) |"
        sep += "---:|"
    lines += ["", "## Layer-by-layer readout", "", header, sep]
    for depth in range(traj.n_depths):
        row = (
            f"| {depth} | `{visible_token(traj.top1_texts[depth])}` "
            f"| {traj.top1_probs[depth]:.3f} | {traj.top1_margin[depth]:.3f} "
            f"| {traj.entropy_bits[depth]:.2f} | {traj.kl_to_final_bits[depth]:.2f} "
            f"| {traj.cosine_to_final[depth]:.3f} | {traj.stream_delta_l2[depth]:.2f} |"
        )
        if traj.p_target is not None:
            row += f" {traj.p_target[depth]:.4f} |"
        if traj.p_target is not None and traj.p_distractor is not None:
            diff = traj.logit_target[depth] - traj.logit_distractor[depth]
            row += f" {diff:+.2f} |"
        lines.append(row)

    # Top-5 snapshots at a handful of depths: enough to see the prediction
    # form without printing every depth's full distribution.
    snapshot_depths = sorted({0, L // 4, L // 2, (3 * L) // 4, L - 1, L})
    lines += ["", "## Top-5 readout at selected depths", ""]
    by_depth: dict[int, list[dict[str, Any]]] = {}
    for row in traj.topk_rows:
        by_depth.setdefault(row["depth"], []).append(row)
    for depth in snapshot_depths:
        entries = by_depth.get(depth, [])
        rendered = " | ".join(
            f"`{visible_token(e['token'])}` {e['prob']:.3f}" for e in entries
        )
        lines.append(f"- depth {depth}: {rendered}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colorblind-friendly, high-contrast defaults shared by all labs.  The first
# four keys are the canonical Lab 1 families; the remaining keys are reused by
# later labs for controls, interventions, and cross-family dashboards.
CATEGORY_COLORS = {
    "fact": "#0072B2",          # blue
    "relation": "#009E73",      # green
    "grammar": "#E69F00",       # orange
    "conflict": "#D55E00",      # vermillion
    "cities": "#0072B2",
    "comparisons": "#E69F00",
    "negations": "#009E73",
    "misconceptions": "#D55E00",
    "synthetic": "#0072B2",
    "cycle": "#009E73",
    "natural": "#D55E00",
    "ambiguous": "#666666",
    "counterfactual": "#D55E00",
    "control": "#7E57C2",
    "true": "#0072B2",
    "false": "#D55E00",
    "clean": "#0072B2",
    "patched": "#009E73",
    "ablated": "#D55E00",
}

CATEGORY_MARKERS = {
    "fact": "o",
    "relation": "^",
    "grammar": "s",
    "conflict": "X",
    "cities": "o",
    "comparisons": "s",
    "negations": "^",
    "misconceptions": "D",
    "synthetic": "o",
    "cycle": "^",
    "natural": "s",
    "ambiguous": "s",
    "counterfactual": "^",
    "control": "X",
    "true": "o",
    "false": "^",
    "clean": "o",
    "patched": "D",
    "ablated": "X",
}

# Component-level colors used by DLA, attention, circuit, and graph labs. The
# names are intentionally generic so later labs can reuse them without a Lab 2
# import. Blue/orange remains the canonical attention/MLP pair.
COMPONENT_COLORS = {
    "attn": "#0072B2",
    "attention": "#0072B2",
    "head": "#0072B2",
    "mlp": "#E69F00",
    "embed": "#555555",
    "embedding": "#555555",
    "constant": "#999999",
    "resid": "#333333",
    "all": "#222222",
    "positive": "#0072B2",
    "negative": "#D55E00",
}

SELECTION_MARKERS = {
    "top": "o",
    "random_control": "s",
    "low_attribution_control": "^",
    "control": "s",
    "ablated": "X",
    "patched": "D",
}

# Lab 3 and Lab 6 use motif labels as a plot grammar orthogonal to prompt
# family and component type.
MOTIF_COLORS = {
    "induction": "#E69F00",
    "previous_token": "#0072B2",
    "first_token_sink": "#7E57C2",
    "diffuse": "#999999",
    "other": "#555555",
}

MOTIF_MARKERS = {
    "induction": "*",
    "previous_token": "o",
    "first_token_sink": "s",
    "diffuse": ".",
    "other": "x",
}

# Lab 5 patching grammar. These labels are separate from generic controls
# because a patching role/control is a different object than a probe baseline.
PATCH_ROLE_COLORS = {
    "pre_subject": "#666666",
    "subject": "#D55E00",
    "post_subject": "#8A9A00",
    "last": "#0072B2",
}

PATCH_ROLE_MARKERS = {
    "pre_subject": "o",
    "subject": "D",
    "post_subject": "s",
    "last": "^",
}

PATCH_CONTROL_COLORS = {
    "matched": "#009E73",
    "matched_top_patch": "#009E73",
    "mismatched_pair": "#8C564B",
    "wrong_position": "#666666",
    "low_region_split_heldout": "#7E57C2",
}

# Lab 6 circuit-discovery grammar. These labels describe node and edge status
# in the evidence ladder, not prompt family membership.
CIRCUIT_STATUS_COLORS = {
    "screened": "#999999",
    "positive_causal": "#56B4E9",
    "negative_causal": "#D55E00",
    "final_circuit": "#009E73",
    "pruned": "#7E57C2",
    "support_mlp": "#CC79A7",
    "edge_source": "#0072B2",
    "edge_target": "#E69F00",
}

CIRCUIT_EDGE_COLORS = {
    "strong": "#009E73",
    "weak": "#E69F00",
    "below_threshold": "#999999",
    "none": "#666666",
}

# Additional palette for control conditions (real vs random/shuffled/etc.).
CONTROL_COLORS = {
    "real": "#d62728",
    "truth": "#D55E00",
    "logistic": "#D55E00",
    "mass_mean": "#7E57C2",
    "surface": "#666666",
    "random": "#7f7f7f",
    "shuffled": "#ff7f0e",
    "length": "#56B4E9",
    "majority": "#999999",
    "control": "#9467bd",
    "mismatched": "#8c564b",
    "filler": "#bcbd22",
    "non_sequitur": "#17becf",
}



# Lab 7 steering / refusal / truth-bridge visual grammar.
# Labs can use these helpers when available, while still keeping local fallbacks
# so older benches remain runnable.
STEERING_COLORS = {
    "real": "#D55E00",
    "sentiment": "#D55E00",
    "truth": "#7E57C2",
    "refusal": "#D55E00",
    "random": "#777777",
    "shuffled": "#8A9A00",
    "benign": "#009E73",
    "refusal_eliciting": "#CC3311",
    "monitor": "#0072B2",
    "safety": "#0072B2",
}

STEERING_MARKERS = {
    "real": "o",
    "sentiment": "o",
    "truth": "o",
    "refusal": "o",
    "random": "s",
    "shuffled": "^",
    "benign": "o",
    "refusal_eliciting": "D",
    "monitor": "o",
    "safety": "s",
}

SAFETY_SCOPE_COLORS = {
    "ok_forward_only": "#0072B2",
    "ok_benign_only": "#009E73",
    "not_implemented": "#999999",
    "blocked": "#CC79A7",
}

# Lab 8 SAE / transcoder feature-interpretation visual grammar. These helpers
# keep feature verdicts and semantic domains consistent with the rest of the
# course, while Lab 8 still carries local fallbacks for older benches.
SAE_VERDICT_COLORS = {
    "survived": "#009E73",
    "narrowed": "#8A9A00",
    "token-feature": "#E69F00",
    "polysemantic": "#7E57C2",
    "killed": "#D55E00",
    "silent-on-corpus": "#999999",
}

SAE_DOMAIN_COLORS = {
    "chemistry": "#0072B2",
    "cooking": "#E69F00",
    "sports": "#009E73",
    "finance": "#8A9A00",
    "law": "#7E57C2",
    "medicine": "#CC79A7",
    "weather": "#56B4E9",
    "emotion": "#D55E00",
    "code": "#666666",
    "history": "#A6761D",
    "none": "#999999",
}

FEATURE_CONDITION_COLORS = {
    "real": "#D55E00",
    "feature": "#D55E00",
    "random": "#777777",
    "control": "#777777",
    "reconstruction": "#0072B2",
    "transcoder": "#7E57C2",
}

# Lab 9 attribution-graph visual grammar. These labels describe objects in the
# replacement graph and the real-model validation tests.
GRAPH_COLORS = {
    "feature": "#009E73",
    "embedding": "#555555",
    "embeddings": "#555555",
    "error": "#E69F00",
    "errors": "#E69F00",
    "logit": "#FFD92F",
    "bias_path": "#999999",
    "transcoder_bias": "#7E57C2",
    "positive": "#0072B2",
    "negative": "#D55E00",
    "baseline": "#666666",
    "suppress": "#D55E00",
    "substitute": "#7E57C2",
    "random": "#8A9A00",
    "counterfactual": "#0072B2",
    "diagnostic": "#0072B2",
    "attr": "#009E73",
    "causal": "#D55E00",
    "robustness": "#E69F00",
}

GRAPH_MARKERS = {
    "feature": "o",
    "embedding": "s",
    "embeddings": "s",
    "error": "^",
    "errors": "^",
    "logit": "*",
    "baseline": "o",
    "suppress": "X",
    "substitute": "D",
    "random": "s",
    "counterfactual": "^",
}


# Lab 10 chain-of-thought faithfulness visual grammar. These labels describe
# behavioral text interventions and self-report audits rather than hidden-state
# instruments. The local lab code carries fallbacks, but the shared palette keeps
# the final intro labs visually consistent.
COT_COLORS = {
    "baseline": "#666666",
    "sycophancy": "#CC79A7",
    "authority": "#E69F00",
    "metadata": "#7E57C2",
    "non_sequitur": "#56B4E9",
    "correct_hint": "#009E73",
    "wrong_hint": "#D55E00",
    "flip": "#D55E00",
    "silent": "#111111",
    "ack": "#0072B2",
    "acknowledged": "#0072B2",
    "attribution": "#009E73",
    "attributed": "#009E73",
    "self_report": "#7E57C2",
    "load": "#0072B2",
    "truncate": "#0072B2",
    "filler": "#8A9A00",
    "resume": "#009E73",
    "clean_resume": "#009E73",
    "mistake": "#D55E00",
    "strong_mistake": "#CC3311",
    "parse": "#009E73",
    "forced": "#D55E00",
    "think": "#0072B2",
    "correct": "#009E73",
    "incorrect": "#D55E00",
    "changed": "#E69F00",
    "not_scored": "#BBBBBB",
    "control": "#999999",
}

COT_MARKERS = {
    "sycophancy": "o",
    "authority": "s",
    "metadata": "D",
    "non_sequitur": "^",
    "flip": "o",
    "silent": "X",
    "ack": "o",
    "attribution": "D",
    "truncate": "o",
    "filler": "s",
    "resume": "D",
    "mistake": "X",
    "strong_mistake": "P",
}

# Lab 11 capstone audit visual grammar. These helpers keep evidence rungs,
# domains, and audit statuses visually stable across scorecards and atlases.
AUDIT_RUNG_COLORS = {
    "OBS": "#4C78A8",
    "ATTR": "#F58518",
    "DECODE": "#54A24B",
    "CAUSAL": "#E45756",
    "SELF-REPORT": "#B279A2",
    "behavioral CAUSAL": "#E45756",
}

AUDIT_DOMAIN_COLORS = {
    "factual_qa": "#4C78A8",
    "cot_faithfulness": "#B279A2",
    "sentiment_negation": "#54A24B",
    "plain": "#4C78A8",
    "negated": "#E45756",
    "base": "#4C78A8",
    "para_city": "#72B7B2",
    "para_in": "#F58518",
}

AUDIT_STATUS_COLORS = {
    "ok": "#54A24B",
    "warning": "#F58518",
    "fail": "#E45756",
    "control": "#8C8C8C",
    "unknown": "#BDBDBD",
    "target_clean_patch": "#54A24B",
    "unrelated_clean_control": "#8C8C8C",
    "plain_clean_patch": "#54A24B",
    "unrelated_plain_control": "#8C8C8C",
    "keep": "#54A24B",
    "revise": "#F58518",
    "retire": "#E45756",
}

# Lab 12 relation geometry visual grammar.
RELATION_GROUP_COLORS = {
    "country_sem": "#0072B2",
    "adj_morph": "#E69F00",
    "month_seq": "#009E73",
    "none": "#8C8C8C",
    "other": "#7E57C2",
}

RELATION_FAMILY_COLORS = {
    "capital_of": "#0072B2",
    "language_of": "#56B4E9",
    "continent_of": "#004C6D",
    "opposite_of": "#D55E00",
    "comparative_of": "#E69F00",
    "month_after": "#009E73",
    "month_before": "#44AA99",
    "color_of": "#CC79A7",
    "material_of": "#7E57C2",
    "currency_of": "#8C564B",
    "home_of": "#999933",
    "plural_of": "#666666",
}

RELATION_ROLE_COLORS = {
    "relword": "#7E57C2",
    "relation": "#7E57C2",
    "subject": "#D55E00",
    "final": "#0072B2",
    "last": "#0072B2",
}

RELATION_CONTROL_COLORS = {
    "matched": "#009E73",
    "subject_matched": "#009E73",
    "relation_matched": "#0072B2",
    "wrong_position": "#8C8C8C",
    "mismatched_vector": "#8C564B",
    "shuffled": "#222222",
    "random": "#8C8C8C",
}

# Lab 13 emotion-geometry visual grammar. The lab carries local fallbacks, but
# shared helpers keep emotion/source/confound/steering plots stable for later
# affect, persona, and humor labs.
EMOTION_COLORS = {
    "joy": "#E69F00",
    "sadness": "#0072B2",
    "anger": "#D55E00",
    "fear": "#7E57C2",
    "neutral": "#8C8C8C",
    "positive": "#009E73",
    "negative": "#D55E00",
}

EMOTION_SOURCE_COLORS = {
    "comprehension": "#0072B2",
    "generation": "#E69F00",
    "input": "#0072B2",
    "write_intent": "#E69F00",
}

EMOTION_CONFOUND_COLORS = {
    "surprising-neutral": "#56B4E9",
    "positive-calm": "#009E73",
    "high-arousal-neutral": "#CC79A7",
    "negative-calm": "#D55E00",
    "arousal_neutral": "#CC79A7",
    "surprising_neutral": "#56B4E9",
    "positive_calm": "#009E73",
    "negative_calm": "#D55E00",
}

EMOTION_STEERING_CONDITION_COLORS = {
    "input_direction": "#0072B2",
    "write_intent_direction": "#E69F00",
    "random_oriented": "#8C8C8C",
    "shuffled_input_direction": "#8A9A00",
    "sentiment_control": "#7E57C2",
    "baseline": "#333333",
}

EMOTION_MARKERS = {
    "joy": "o",
    "sadness": "s",
    "anger": "^",
    "fear": "D",
    "neutral": ".",
}

# Lab 14 certainty/calibration visual grammar. The lab has local fallbacks, but
# these helpers keep downstream self-report, belief-revision, and calibration
# plots on the same visual language.
CERTAINTY_COLORS = {
    "internal": "#0072B2",
    "internal_projection": "#0072B2",
    "internal_rank_confidence": "#0072B2",
    "distribution": "#009E73",
    "distribution_confidence": "#009E73",
    "verbal": "#E69F00",
    "verbal_confidence": "#E69F00",
    "self_report": "#E69F00",
    "hedging": "#7E57C2",
    "hedging_style_projection": "#7E57C2",
    "answerable": "#009E73",
    "unanswerable": "#D55E00",
    "correct": "#009E73",
    "wrong": "#D55E00",
    "real": "#D55E00",
    "random": "#777777",
    "shuffled": "#8A9A00",
    "length": "#56B4E9",
    "letter": "#CC79A7",
    "confound": "#CC79A7",
    "control": "#777777",
}

CERTAINTY_MARKERS = {
    "internal": "o",
    "internal_projection": "o",
    "internal_rank_confidence": "o",
    "distribution": "s",
    "distribution_confidence": "s",
    "verbal": "^",
    "verbal_confidence": "^",
    "hedging": "D",
    "hedging_style_projection": "D",
    "real": "o",
    "random": "s",
    "shuffled": "^",
}

# Lab 15 multi-turn instrumentation visual grammar. Later social, persona, and
# belief labs inherit these colors for boundary, cache, patch, and null-trace
# diagnostics.
MULTITURN_COLORS = {
    "template": "#CC79A7",
    "span": "#009E73",
    "content": "#009E73",
    "message": "#0072B2",
    "system": "#666666",
    "user": "#0072B2",
    "assistant": "#E69F00",
    "assistant_generation": "#E69F00",
    "cache": "#56B4E9",
    "patch": "#7E57C2",
    "topic": "#009E73",
    "topic_orchid_minus_archive": "#009E73",
    "orchid_topic": "#009E73",
    "archive_control": "#0072B2",
    "archive_length_control": "#0072B2",
    "length_null": "#E69F00",
    "length_matched_null": "#E69F00",
    "random_null": "#777777",
    "random": "#777777",
    "control": "#777777",
    "pass": "#009E73",
    "ready": "#009E73",
    "ok": "#009E73",
    "warn": "#E69F00",
    "warning": "#E69F00",
    "caution": "#E69F00",
    "fail": "#D55E00",
    "blocked": "#D55E00",
}

MULTITURN_MARKERS = {
    "topic": "o",
    "topic_orchid_minus_archive": "o",
    "length_null": "s",
    "length_matched_null": "s",
    "random_null": "^",
    "random": "^",
    "cache": "D",
    "patch": "P",
    "content": "o",
    "message": "s",
}

# Lab 16 sycophancy/user-belief visual grammar.
SYCOPHANCY_COLORS = {
    "neutral": "#7A7A7A",
    "correct_belief_control": "#009E73",
    "false_belief": "#E69F00",
    "mild_pressure": "#CC79A7",
    "authority_pressure": "#D55E00",
    "identity_pressure": "#AA4499",
    "correct": "#009E73",
    "sycophantic": "#D55E00",
    "mixed": "#E69F00",
    "ambiguous": "#777777",
    "surface_agreement_only": "#56B4E9",
    "user_belief": "#0072B2",
    "truth": "#009E73",
    "agreement": "#D55E00",
    "politeness": "#CC79A7",
    "sentiment_style": "#E69F00",
    "certainty_style": "#7E57C2",
    "social_pressure": "#AA4499",
    "agreement_shuffled": "#999999",
    "shuffled": "#999999",
    "random": "#777777",
    "control": "#8C8C8C",
    "decode": "#0072B2",
    "causal": "#D55E00",
    "obs": "#009E73",
    "manual": "#7E57C2",
    "validated": "#009E73",
    "weak": "#E69F00",
    "failed": "#D55E00",
}

SYCOPHANCY_MARKERS = {
    "neutral": "o",
    "correct_belief_control": "s",
    "false_belief": "D",
    "mild_pressure": "^",
    "authority_pressure": "P",
    "identity_pressure": "X",
    "agreement": "o",
    "politeness": "s",
    "sentiment_style": "^",
    "agreement_shuffled": "x",
    "random": "+",
    "train": "o",
    "eval": "x",
    "correct": "o",
    "sycophantic": "X",
    "mixed": "^",
    "ambiguous": "s",
}


# ---------------------------------------------------------------------------
# Lab 17 persona / register / voice visual grammar
# ---------------------------------------------------------------------------

PERSONA_COLORS = {
    "persona": "#7E57C2",
    "character_museum_guide": "#7E57C2",
    "persona_museum_guide": "#7E57C2",
    "museum_roleplay": "#7E57C2",
    "default_assistant_control": "#6E6E6E",
    "technical_register": "#0072B2",
    "register": "#0072B2",
    "casual_register_control": "#E69F00",
    "warm_supportive_voice": "#E69F00",
    "voice": "#E69F00",
    "direct_terse_control": "#56B4E9",
    "honest_disagreement": "#009E73",
    "agreeable_validation": "#56B4E9",
    "agreement": "#009E73",
    "trait_direction": "#D55E00",
    "opposite_direction": "#0072B2",
    "shuffled_sign_direction": "#999999",
    "shuffled_sign": "#999999",
    "random_direction": "#777777",
    "random_oriented": "#777777",
    "random_null": "#777777",
    "baseline": "#444444",
    "real": "#0072B2",
    "decode": "#0072B2",
    "causal": "#D55E00",
    "trace": "#009E73",
    "safety": "#CC79A7",
    "refusal_monitor": "#D55E00",
    "sentiment_style_control": "#CC79A7",
    "positive": "#009E73",
    "claimable": "#009E73",
    "warning": "#E69F00",
    "watch": "#E69F00",
    "failed": "#D55E00",
    "not_claimable": "#D55E00",
    "control": "#777777",
}

PERSONA_MARKERS = {
    "persona": "o",
    "character_museum_guide": "o",
    "persona_museum_guide": "o",
    "technical_register": "s",
    "warm_supportive_voice": "^",
    "honest_disagreement": "D",
    "agreeable_validation": "P",
    "default_assistant_control": "x",
    "casual_register_control": "v",
    "trait_direction": "o",
    "opposite_direction": "v",
    "shuffled_sign_direction": "s",
    "random_direction": "x",
    "real": "o",
    "shuffled_sign": "s",
    "random_oriented": "x",
    "random_null": "x",
    "refusal_monitor": "P",
    "sentiment_style_control": "^",
}


def plot_persona_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 17 persona/register/voice plots."""
    key = str(condition)
    return PERSONA_COLORS.get(key, SYCOPHANCY_COLORS.get(key, MULTITURN_COLORS.get(key, CONTROL_COLORS.get(key, CATEGORY_COLORS.get(key, default)))))


def plot_persona_marker(condition: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 17 persona/register/voice plots."""
    key = str(condition)
    return PERSONA_MARKERS.get(key, SYCOPHANCY_MARKERS.get(key, MULTITURN_MARKERS.get(key, CATEGORY_MARKERS.get(key, default))))



# Lab 18 humor / incongruity visual grammar.
HUMOR_COLORS = {
    "joke": "#0072B2",
    "joke_structure": "#0072B2",
    "joke_structure_direction": "#0072B2",
    "opposite_joke_structure_direction": "#D55E00",
    "literal": "#6E6E6E",
    "surprise": "#E69F00",
    "surprise_direction": "#E69F00",
    "silly": "#CC79A7",
    "silly_direction": "#CC79A7",
    "positive": "#009E73",
    "positive_direction": "#009E73",
    "shuffled": "#8C8C8C",
    "shuffled_joke_direction": "#8C8C8C",
    "random": "#333333",
    "random_direction": "#333333",
    "real": "#0072B2",
    "best_null": "#595959",
    "control_gap": "#17BECF",
    "cheap": "#BCBD22",
    "validated": "#009E73",
    "warning": "#E69F00",
    "failed": "#D55E00",
    "decode": "#0072B2",
    "causal": "#9467BD",
    "human_label": "#4D4D4D",
}

HUMOR_MARKERS = {
    "joke": "o",
    "literal": "s",
    "surprise": "^",
    "silly": "D",
    "positive": "P",
    "real": "o",
    "shuffled_sign_mean": "s",
    "random_oriented_mean": "^",
    "joke_structure_direction": "o",
    "opposite_joke_structure_direction": "v",
    "surprise_direction": "^",
    "silly_direction": "D",
    "positive_direction": "P",
    "shuffled_joke_direction": "s",
    "random_direction": "x",
}


def plot_humor_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 18 humor/incongruity plots."""
    key = str(condition)
    return HUMOR_COLORS.get(key, STEERING_COLORS.get(key, PERSONA_COLORS.get(key, CONTROL_COLORS.get(key, CATEGORY_COLORS.get(key, default)))))


def plot_humor_marker(condition: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 18 humor/incongruity plots."""
    key = str(condition)
    return HUMOR_MARKERS.get(key, STEERING_MARKERS.get(key, PERSONA_MARKERS.get(key, CATEGORY_MARKERS.get(key, default))))


# Lab 19 model-diffing / crosscoder visual grammar.
MODELDIFF_COLORS = {
    "shared": "#0072B2",
    "model_a_only": "#D55E00",
    "model_b_only": "#009E73",
    "base_only": "#D55E00",
    "instruct_only": "#009E73",
    "asymmetric": "#CC79A7",
    "dead": "#8C8C8C",
    "template": "#E69F00",
    "template_residue_candidate": "#E69F00",
    "family_specific_candidate": "#F0E442",
    "train_only_unstable": "#D55E00",
    "candidate_model_b_handle": "#009E73",
    "candidate_model_a_handle": "#D55E00",
    "candidate_shared_handle": "#0072B2",
    "asymmetric_or_unclear": "#CC79A7",
    "crosscoder_artifact_risk": "#D55E00",
    "random": "#333333",
    "raw": "#0072B2",
    "compare_chat": "#009E73",
    "pass": "#009E73",
    "mixed": "#E69F00",
    "warning": "#E69F00",
    "fail": "#D55E00",
    "skipped": "#8C8C8C",
    "not_configured": "#8C8C8C",
    "reconstruction": "#0072B2",
    "taxonomy": "#009E73",
    "controls": "#E69F00",
    "causal": "#9467BD",
    "bridge": "#56B4E9",
}

MODELDIFF_MARKERS = {
    "shared": "o",
    "model_a_only": "<",
    "model_b_only": ">",
    "base_only": "<",
    "instruct_only": ">",
    "asymmetric": "D",
    "dead": "x",
    "raw": "o",
    "compare_chat": "s",
    "template_residue_candidate": "^",
    "family_specific_candidate": "P",
    "train_only_unstable": "v",
    "candidate_model_b_handle": ">",
    "candidate_model_a_handle": "<",
    "candidate_shared_handle": "o",
    "asymmetric_or_unclear": "D",
    "feature_plus": ">",
    "feature_minus": "<",
    "feature_plus_low": "o",
    "random_plus": "x",
    "random_minus": "+",
    "baseline": "o",
}


def plot_modeldiff_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 19 model-diffing/crosscoder plots."""
    key = str(condition)
    return MODELDIFF_COLORS.get(key, HUMOR_COLORS.get(key, PERSONA_COLORS.get(key, STEERING_COLORS.get(key, CONTROL_COLORS.get(key, CATEGORY_COLORS.get(key, default))))))


def plot_modeldiff_marker(condition: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 19 model-diffing/crosscoder plots."""
    key = str(condition)
    return MODELDIFF_MARKERS.get(key, HUMOR_MARKERS.get(key, PERSONA_MARKERS.get(key, STEERING_MARKERS.get(key, CATEGORY_MARKERS.get(key, default)))))


# Lab 20 benign model-organism construction visual grammar.
MODELORG_COLORS = {
    "construction": "#0072B2",
    "blind_package": "#56B4E9",
    "private": "#9467BD",
    "public": "#009E73",
    "leak": "#D55E00",
    "safety": "#009E73",
    "baseline": "#E69F00",
    "spillover": "#CC79A7",
    "adapter": "#8C8C8C",
    "target": "#0072B2",
    "control": "#D55E00",
    "target_prompt": "#0072B2",
    "control_prompt": "#D55E00",
    "spillover_issue": "#CC79A7",
    "adapter_present": "#009E73",
    "adapter_missing": "#8C8C8C",
    "pass": "#009E73",
    "pending": "#8C8C8C",
    "warning": "#E69F00",
    "fail": "#D55E00",
    "unknown": "#8C8C8C",
}

MODELORG_MARKERS = {
    "construction": "o",
    "blind_package": "s",
    "public": "s",
    "private": "D",
    "target": "o",
    "control": "s",
    "spillover": "^",
    "adapter_present": "P",
    "adapter_missing": "x",
    "pass": "o",
    "pending": "D",
    "warning": "^",
    "fail": "X",
}


def plot_modelorg_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 20 benign-organism construction plots."""
    key = str(condition)
    return MODELORG_COLORS.get(key, MODELDIFF_COLORS.get(key, HUMOR_COLORS.get(key, PERSONA_COLORS.get(key, STEERING_COLORS.get(key, CONTROL_COLORS.get(key, CATEGORY_COLORS.get(key, default)))))))


def plot_modelorg_marker(condition: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 20 benign-organism construction plots."""
    key = str(condition)
    return MODELORG_MARKERS.get(key, MODELDIFF_MARKERS.get(key, HUMOR_MARKERS.get(key, PERSONA_MARKERS.get(key, STEERING_MARKERS.get(key, CATEGORY_MARKERS.get(key, default))))))


def plot_modelorganism_color(condition: str, default: str = "#555555") -> str:
    """Alias with the longer Lab 20 name for downstream notebooks."""
    return plot_modelorg_color(condition, default=default)


def plot_modelorganism_marker(condition: str, default: str = "o") -> str:
    """Alias with the longer Lab 20 name for downstream notebooks."""
    return plot_modelorg_marker(condition, default=default)


TRAINING_DEPTH_COLORS = {
    "lora": "#0072B2",
    "weight_space": "#0072B2",
    "base_instruct": "#009E73",
    "chat_format": "#999999",
    "boundary_safe": "#D55E00",
    "forced_prefix": "#CC79A7",
    "provenance": "#56B4E9",
    "intervention": "#E69F00",
    "erosion": "#F0E442",
    "scaffold": "#BBBBBB",
    "data": "#009E73",
    "missing": "#777777",
    "warning": "#E69F00",
    "fail": "#D55E00",
    "pass": "#009E73",
}


def plot_training_depth_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 21 training-depth plots."""
    return TRAINING_DEPTH_COLORS.get(str(condition), default)


def plot_training_depth_marker(condition: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 21 training-depth plots."""
    return {
        "lora": "o",
        "weight_space": "o",
        "base_instruct": "s",
        "chat_format": "D",
        "boundary_safe": "^",
        "forced_prefix": "P",
        "provenance": "h",
        "intervention": "*",
        "erosion": "X",
        "scaffold": "x",
        "data": "o",
        "missing": "x",
    }.get(str(condition), default)


EVALAWARE_COLORS = {
    "eval": "#D55E00",
    "natural": "#0072B2",
    "format_control": "#8C8C8C",
    "real": "#D55E00",
    "random": "#777777",
    "shuffled": "#CC79A7",
    "surface": "#E69F00",
    "decode": "#009E73",
    "causal": "#CC79A7",
    "monitor": "#56B4E9",
    "pass": "#009E73",
    "warning": "#E69F00",
    "fail": "#D55E00",
    "blocked": "#D55E00",
    "unknown": "#999999",
}

EVALAWARE_MARKERS = {
    "eval": "s",
    "natural": "o",
    "format_control": "D",
    "real": "o",
    "random": "x",
    "shuffled": "^",
    "surface": "v",
    "decode": "P",
    "causal": "*",
    "monitor": "h",
    "pass": "o",
    "warning": "^",
    "fail": "X",
}


def plot_evalawareness_color(key: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 22 eval-awareness plots."""
    return EVALAWARE_COLORS.get(str(key), default)


def plot_evalawareness_marker(key: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 22 eval-awareness plots."""
    return EVALAWARE_MARKERS.get(str(key), default)


BLINDAUDIT_COLORS = {
    "audit": "#0072B2",
    "behavioral_only": "#56B4E9",
    "internals_allowed": "#009E73",
    "hit": "#009E73",
    "miss": "#999999",
    "false_positive": "#D55E00",
    "manual_review": "#CC79A7",
    "submitted": "#0072B2",
    "draft": "#BBBBBB",
    "public": "#0072B2",
    "private": "#D55E00",
    "freeze": "#6F4E7C",
    "commitment": "#009E73",
    "leak": "#D55E00",
    "pass": "#009E73",
    "warning": "#E69F00",
    "fail": "#D55E00",
    "awaiting": "#999999",
    "unknown": "#777777",
}

BLINDAUDIT_MARKERS = {
    "audit": "o",
    "behavioral_only": "o",
    "internals_allowed": "s",
    "trigger": "o",
    "behavior": "s",
    "marker": "^",
    "spillover": "D",
    "internal_signature": "P",
    "safety": "X",
    "other": "h",
    "hit": "o",
    "miss": "x",
    "false_positive": "X",
    "manual_review": "^",
}


def plot_blindaudit_color(key: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 23 blind-audit plots."""
    return BLINDAUDIT_COLORS.get(str(key), default)


def plot_blindaudit_marker(key: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 23 blind-audit plots."""
    return BLINDAUDIT_MARKERS.get(str(key), default)


BELIEF_REVISION_COLORS = {
    "context": "#D55E00",
    "parametric": "#009E73",
    "pressure": "#D55E00",
    "evidence": "#009E73",
    "no_context": "#4D4D4D",
    "weak_fictional": "#9ECAE1",
    "document_statement": "#4292C6",
    "repeated_document": "#2171B5",
    "delayed_document": "#084594",
    "correct": "#009E73",
    "false_pressure_answer": "#D55E00",
    "neutral_reask": "#999999",
    "pushback_no_evidence": "#E69F00",
    "false_authority": "#D55E00",
    "real_evidence": "#009E73",
    "common_misconception": "#CC79A7",
    "forced_concise": "#56B4E9",
    "strong_context_same_item": "#0072B2",
    "self_pre_pressure_baseline": "#0072B2",
    "mismatched_context_control": "#999999",
    "mismatched_baseline_control": "#999999",
    "answer_and_signal_flip": "#D55E00",
    "answer_flips_signal_holds": "#E69F00",
    "signal_flips_answer_holds": "#56B4E9",
    "neither": "#009E73",
    "baseline_not_correct_not_interpretable": "#777777",
    "control_not_quadrant": "#BBBBBB",
    "OBS": "#4D4D4D",
    "DECODE": "#0072B2",
    "SELF-REPORT": "#CC79A7",
    "CAUSAL": "#009E73",
    "AUDIT": "#6F4E7C",
    "pass": "#009E73",
    "warning": "#E69F00",
    "fail": "#D55E00",
}

BELIEF_REVISION_MARKERS = {
    "no_context": "o",
    "weak_fictional": "s",
    "document_statement": "^",
    "repeated_document": "D",
    "delayed_document": "P",
    "neutral_reask": "o",
    "pushback_no_evidence": "s",
    "false_authority": "^",
    "real_evidence": "D",
    "common_misconception": "P",
    "forced_concise": "X",
    "answer_and_signal_flip": "o",
    "answer_flips_signal_holds": "s",
    "signal_flips_answer_holds": "^",
    "neither": "D",
}


def plot_belief_revision_color(key: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 24 belief-revision plots."""
    return BELIEF_REVISION_COLORS.get(str(key), default)


def plot_belief_revision_marker(key: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 24 belief-revision plots."""
    return BELIEF_REVISION_MARKERS.get(str(key), default)


def plot_beliefrevision_color(key: str, default: str = "#555555") -> str:
    """Alias without the underscore used by some Lab 24 drafts."""
    return plot_belief_revision_color(key, default)


def plot_beliefrevision_marker(key: str, default: str = "o") -> str:
    """Alias without the underscore used by some Lab 24 drafts."""
    return plot_belief_revision_marker(key, default)


def plot_belief_color(key: str, default: str = "#555555") -> str:
    """Short alias used by Lab 24 local plotting fallbacks."""
    return plot_belief_revision_color(key, default)


def plot_belief_marker(key: str, default: str = "o") -> str:
    """Short alias used by Lab 24 local plotting fallbacks."""
    return plot_belief_revision_marker(key, default)


FINDWIRE_COLORS = {
    "target_direction": "#0072B2",
    "opposite_direction": "#D55E00",
    "wrong_concept_direction": "#CC79A7",
    "random_direction": "#999999",
    "shuffled_direction": "#E69F00",
    "zero_dose": "#BBBBBB",
    "default_mode": "#4D4D4D",
    "system_prompt": "#56B4E9",
    "user_instruction": "#009E73",
    "activation_injection": "#0072B2",
    "false_activation_claim": "#D55E00",
    "state_report_before_visible_output": "#009E73",
    "output_rationalization_or_downstream_priming_risk": "#D55E00",
    "behavior_expressed_without_report": "#E69F00",
    "no_self_report_detection": "#999999",
    "wire_candidate": "#009E73",
    "report_moves_but_grounding_weak": "#E69F00",
    "weak_specificity": "#56B4E9",
    "not_supported": "#999999",
    "false_positive": "#D55E00",
    "grounding": "#009E73",
    "source": "#0072B2",
    "confidence": "#CC79A7",
    "decode": "#4D4D4D",
    "causal": "#009E73",
    "self_report": "#CC79A7",
    "audit": "#6F4E7C",
    "pass": "#009E73",
    "warning": "#E69F00",
    "fail": "#D55E00",
    "not_run": "#BBBBBB",
}

FINDWIRE_MARKERS = {
    "target_direction": "o",
    "opposite_direction": "v",
    "wrong_concept_direction": "D",
    "random_direction": "x",
    "shuffled_direction": "^",
    "default_mode": "o",
    "system_prompt": "s",
    "user_instruction": "^",
    "activation_injection": "P",
    "false_activation_claim": "X",
    "wire_candidate": "o",
    "report_moves_but_grounding_weak": "^",
    "weak_specificity": "s",
    "not_supported": "x",
}


def plot_findwire_color(key: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 25 Find-the-Wire plots."""
    return FINDWIRE_COLORS.get(str(key), default)


def plot_findwire_marker(key: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 25 Find-the-Wire plots."""
    return FINDWIRE_MARKERS.get(str(key), default)


def plot_wire_color(key: str, default: str = "#555555") -> str:
    """Backward-compatible alias for Lab 25 draft plotting helpers."""
    return plot_findwire_color(key, default)


def plot_wire_marker(key: str, default: str = "o") -> str:
    """Backward-compatible alias for Lab 25 draft plotting helpers."""
    return plot_findwire_marker(key, default)


def plot_findthewire_color(key: str, default: str = "#555555") -> str:
    """Backward-compatible alias for Lab 25 draft plotting helpers."""
    return plot_findwire_color(key, default)


def plot_findthewire_marker(key: str, default: str = "o") -> str:
    """Backward-compatible alias for Lab 25 draft plotting helpers."""
    return plot_findwire_marker(key, default)


def configure_matplotlib() -> None:
    """One-time global polish for all lab plots (clean, readable, consistent)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.9,
        "legend.frameon": False,
        "legend.borderaxespad": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def new_figure(figsize: tuple[float, float] = (8.0, 5.0)) -> tuple[Any, Any]:
    """Create a matplotlib figure with the bench's house style (clean spines, consistent fonts)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def style_ax(ax: Any, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None,
             legend: bool = True, legend_loc: str = "best") -> None:
    """Apply consistent final styling to an axes."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_category_color(category: str, default: str = "#333333") -> str:
    """Shared category color lookup for lab modules."""
    return CATEGORY_COLORS.get(str(category), default)


def plot_category_marker(category: str, default: str = "o") -> str:
    """Shared marker lookup for lab modules."""
    return CATEGORY_MARKERS.get(str(category), default)


def plot_component_color(component: str, default: str = "#555555") -> str:
    """Shared component/writer color lookup for lab modules."""
    return COMPONENT_COLORS.get(str(component), default)


def plot_selection_marker(selection: str, default: str = "o") -> str:
    """Shared marker lookup for ablation/patching/control selections."""
    return SELECTION_MARKERS.get(str(selection), default)


def plot_control_color(control: str, default: str = "#555555") -> str:
    """Shared color lookup for probe controls and baselines."""
    return CONTROL_COLORS.get(str(control), default)


def plot_patch_role_color(role: str, default: str = "#555555") -> str:
    """Shared color lookup for activation-patching token roles."""
    return PATCH_ROLE_COLORS.get(str(role), default)


def plot_patch_role_marker(role: str, default: str = "o") -> str:
    """Shared marker lookup for activation-patching token roles."""
    return PATCH_ROLE_MARKERS.get(str(role), default)


def plot_patch_control_color(control: str, default: str = "#555555") -> str:
    """Shared color lookup for activation-patching controls."""
    return PATCH_CONTROL_COLORS.get(str(control), CONTROL_COLORS.get(str(control), default))


def plot_circuit_status_color(status: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 6 circuit node statuses."""
    return CIRCUIT_STATUS_COLORS.get(str(status), default)


def plot_circuit_edge_color(strength: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 6 edge-interaction strengths."""
    return CIRCUIT_EDGE_COLORS.get(str(strength), default)


def plot_steering_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for steering directions, refusal categories, and truth-bridge conditions."""
    return STEERING_COLORS.get(str(condition), default)


def plot_steering_marker(condition: str, default: str = "o") -> str:
    """Shared marker lookup for steering directions and refusal/truth categories."""
    return STEERING_MARKERS.get(str(condition), default)


def plot_safety_scope_color(status: str, default: str = "#555555") -> str:
    """Shared color lookup for safety-wall scope rows."""
    return SAFETY_SCOPE_COLORS.get(str(status), default)


def plot_motif_color(label: str, default: str = "#555555") -> str:
    """Shared color lookup for attention/circuit motif labels."""
    return MOTIF_COLORS.get(str(label), default)


def plot_motif_marker(label: str, default: str = "o") -> str:
    """Shared marker lookup for attention/circuit motif labels."""
    return MOTIF_MARKERS.get(str(label), default)


def plot_sae_verdict_color(verdict: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 8 feature-label validation verdicts."""
    return SAE_VERDICT_COLORS.get(str(verdict), default)


def plot_feature_verdict_color(verdict: str, default: str = "#555555") -> str:
    """Alias for Lab 8 feature-label validation verdict colors."""
    return plot_sae_verdict_color(verdict, default)


def plot_sae_domain_color(domain: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 8 semantic-domain feature labels."""
    return SAE_DOMAIN_COLORS.get(str(domain), default)


def plot_feature_condition_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for feature-clamp real/control conditions and dictionary objects."""
    return FEATURE_CONDITION_COLORS.get(str(condition), default)


def plot_graph_color(item: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 9 attribution-graph nodes, edge signs, and interventions."""
    return GRAPH_COLORS.get(str(item), default)


def plot_graph_marker(item: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 9 graph nodes and intervention conditions."""
    return GRAPH_MARKERS.get(str(item), default)


def plot_graph_node_color(item: str, default: str = "#555555") -> str:
    """Alias for Lab 9 graph-node color lookup."""
    return plot_graph_color(item, default)


def plot_edge_source_color(item: str, default: str = "#555555") -> str:
    """Alias for Lab 9 source-kind color lookup."""
    return plot_graph_color(item, default)


def plot_graph_intervention_color(condition: str, default: str = "#555555") -> str:
    """Alias for Lab 9 real-model intervention color lookup."""
    key = str(condition)
    mapped = {
        "suppress_subject_supernode": "suppress",
        "substitute_counterfactual": "substitute",
        "random_suppression_control": "random",
        "counterfactual_prompt_reference": "counterfactual",
    }.get(key, key)
    return plot_graph_color(mapped, default)



def plot_cot_color(item: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 10 CoT faithfulness conditions, controls, and verdict states."""
    return COT_COLORS.get(str(item), default)


def plot_cot_marker(item: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 10 CoT faithfulness conditions and controls."""
    return COT_MARKERS.get(str(item), default)


def plot_audit_rung_color(rung: str, default: str = "#4C78A8") -> str:
    """Shared color lookup for Lab 11 evidence rungs."""
    return AUDIT_RUNG_COLORS.get(str(rung), default)


def plot_audit_domain_color(domain: str, default: str = "#4C78A8") -> str:
    """Shared color lookup for Lab 11 audit domains and prompt families."""
    key = str(domain)
    return AUDIT_DOMAIN_COLORS.get(key, CATEGORY_COLORS.get(key, default))


def plot_audit_status_color(status: str, default: str = "#8C8C8C") -> str:
    """Shared color lookup for Lab 11 audit/control statuses."""
    key = str(status)
    return AUDIT_STATUS_COLORS.get(key, CONTROL_COLORS.get(key, default))


def plot_relation_group_color(group: str, default: str = "#7E57C2") -> str:
    """Shared color lookup for Lab 12 relation swap groups."""
    key = str(group)
    return RELATION_GROUP_COLORS.get(key, CATEGORY_COLORS.get(key, default))


def plot_relation_family_color(family: str, default: str = "#7E57C2") -> str:
    """Shared color lookup for Lab 12 relation families."""
    key = str(family)
    return RELATION_FAMILY_COLORS.get(key, plot_relation_group_color(key, default))


def plot_relation_role_color(role: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 12 token roles."""
    key = str(role)
    return RELATION_ROLE_COLORS.get(key, CATEGORY_COLORS.get(key, default))


def plot_relation_control_color(control: str, default: str = "#8C8C8C") -> str:
    """Shared color lookup for Lab 12 patch/probe controls."""
    key = str(control)
    return RELATION_CONTROL_COLORS.get(key, CONTROL_COLORS.get(key, default))


def plot_emotion_color(emotion: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 13 emotion labels."""
    key = str(emotion)
    return EMOTION_COLORS.get(key, CATEGORY_COLORS.get(key, default))


def plot_emotion_marker(emotion: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 13 emotion labels."""
    key = str(emotion)
    return EMOTION_MARKERS.get(key, CATEGORY_MARKERS.get(key, default))


def plot_emotion_source_color(source: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 13 source domains."""
    key = str(source)
    return EMOTION_SOURCE_COLORS.get(key, CATEGORY_COLORS.get(key, default))


def plot_emotion_confound_color(confound: str, default: str = "#8C8C8C") -> str:
    """Shared color lookup for Lab 13 confound rows."""
    key = str(confound)
    return EMOTION_CONFOUND_COLORS.get(key, CONTROL_COLORS.get(key, default))


def plot_emotion_condition_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 13 steering/control conditions."""
    key = str(condition)
    return EMOTION_STEERING_CONDITION_COLORS.get(key, CONTROL_COLORS.get(key, default))


def plot_certainty_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 14 certainty/calibration gauges and controls."""
    key = str(condition)
    return CERTAINTY_COLORS.get(
        key,
        CONTROL_COLORS.get(key, CATEGORY_COLORS.get(key, default)),
    )


def plot_certainty_marker(condition: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 14 certainty/calibration gauges and controls."""
    key = str(condition)
    return CERTAINTY_MARKERS.get(key, CATEGORY_MARKERS.get(key, default))


def plot_multiturn_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 15 and later multi-turn instrumentation plots."""
    key = str(condition)
    if key.startswith("random_null"):
        key = "random_null"
    return MULTITURN_COLORS.get(
        key,
        CONTROL_COLORS.get(key, CATEGORY_COLORS.get(key, default)),
    )


def plot_multiturn_marker(condition: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 15 and later multi-turn instrumentation plots."""
    key = str(condition)
    if key.startswith("random_null"):
        key = "random_null"
    return MULTITURN_MARKERS.get(key, CATEGORY_MARKERS.get(key, default))


def plot_sycophancy_color(condition: str, default: str = "#555555") -> str:
    """Shared color lookup for Lab 16 social-state and sycophancy plots."""
    key = str(condition)
    return SYCOPHANCY_COLORS.get(
        key,
        CONTROL_COLORS.get(key, CATEGORY_COLORS.get(key, default)),
    )


def plot_sycophancy_marker(condition: str, default: str = "o") -> str:
    """Shared marker lookup for Lab 16 social-state and sycophancy plots."""
    key = str(condition)
    return SYCOPHANCY_MARKERS.get(key, CATEGORY_MARKERS.get(key, default))


def lighten_color(color: str, amount: float = 0.55) -> str:
    """Return a lighter version of a Matplotlib color."""
    import matplotlib.colors as mcolors

    amount = max(0.0, min(1.0, float(amount)))
    r, g, b = mcolors.to_rgb(color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return mcolors.to_hex((r, g, b))


def add_zero_lines(ax: Any, *, x: bool = True, y: bool = True, color: str = "#222222") -> None:
    """Add light zero-reference lines to a signed plot."""
    if x:
        ax.axvline(0, color=color, linewidth=0.8, alpha=0.85)
    if y:
        ax.axhline(0, color=color, linewidth=0.8, alpha=0.85)


def format_signed(value: float, digits: int = 2) -> str:
    """Compact signed number formatting for plot annotations."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:+.{digits}f}"


def add_panel_label(ax: Any, label: str, *, x: float = -0.10, y: float = 1.04) -> None:
    """Add a small bold panel label such as A, B, C."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="right", color="#222222")


def label_line_end(ax: Any, xs: Sequence[float], ys: Sequence[float], label: str, *,
                   color: str | None = None, dx: float = 3.0) -> None:
    """Directly label the last finite point of a line.

    Legends are fine for a few curves; direct labels are clearer on dense lab
    dashboards and survive screenshots better. This helper is intentionally
    tiny and optional so old plots keep working.
    """
    for x, y in zip(reversed(list(xs)), reversed(list(ys))):
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            ax.annotate(label, (xf, yf), textcoords="offset points", xytext=(dx, 0),
                        va="center", fontsize=7.5, color=color or "#333333")
            return


def add_depth_phase_guides(ax: Any, n_layers: int, *, label_final: bool = True) -> None:
    """Light vertical guides at one-third, two-thirds, and final depth."""
    if n_layers <= 0:
        return
    for frac in (1 / 3, 2 / 3):
        ax.axvline(n_layers * frac, color="#888888", linestyle=":", linewidth=0.7, alpha=0.25)
    ax.axvline(n_layers, color="#444444", linestyle=":", linewidth=1.0, alpha=0.55)
    if label_final:
        ax.text(n_layers, 0.98, "final", transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", fontsize=7, color="#444444", alpha=0.75)


def save_figure(ctx: RunContext, fig: Any, name: str, description: str) -> None:
    """Save a plot under plots/ and register it in the artifact index."""
    import matplotlib.pyplot as plt

    path = ctx.path("plots", name)
    fig.text(0.995, 0.005, ctx.plot_footer(), ha="right", va="bottom", fontsize=6.5, color="#555555")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    ctx.register_artifact(path, "plot", description)
    if os.environ.get("INTERP_SAVE_SVG", "").lower() in {"1", "true", "yes"}:
        svg_path = path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
        ctx.register_artifact(svg_path, "plot", description + " (SVG copy for slides/papers).")
    plt.close(fig)
    print(f"[bench] wrote plots/{name}")


def close_figure(fig: Any) -> None:
    """Release a figure that is being abandoned instead of saved."""
    import matplotlib.pyplot as plt

    plt.close(fig)


def add_vline(ax: Any, x: float, label: str | None = None, *, color: str = "#d62728",
              ls: str = "--", lw: float = 1.2, alpha: float = 0.75) -> None:
    """Add a vertical reference line (handoff, decision, zero-dose, etc.) with optional label."""
    ax.axvline(x, color=color, linestyle=ls, linewidth=lw, alpha=alpha)
    if label:
        ax.text(x, 0.98, label, transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", fontsize=7, color=color, alpha=min(1.0, alpha + 0.1))


_plot_style_configured = False


def _ensure_plot_style() -> None:
    """Idempotent style setup so every lab benefits even if it imports plt directly.
    Safe to call in environments without matplotlib (e.g. lint or partial imports)."""
    global _plot_style_configured
    if _plot_style_configured:
        return
    try:
        configure_matplotlib()
    except Exception:
        return
    _plot_style_configured = True


# ---------------------------------------------------------------------------
# Claim ledger
# ---------------------------------------------------------------------------

LEDGER_HEADER = """# Claim ledger

A running dossier of claims about one model. Every entry carries an evidence
tag (OBS | ATTR | DECODE | CAUSAL | SELF-REPORT | AUDIT | CONSTRUCTION |
FORMAL), the artifact that backs it, and the observation that would kill it.
Labs draft suggested claims into
their run directory; what lands here is the student's own judgment.

Format:

    [L01-C1] OBS | <claim text with numbers and scope>
    Artifact: runs/<run>/<file> | Falsifier: <what would kill this claim>

---
"""


def ensure_ledger() -> None:
    """Create the course-root claim ledger with its header if missing."""
    if not LEDGER_PATH.exists():
        write_text(LEDGER_PATH, LEDGER_HEADER)
        print(f"[bench] initialized claim ledger at {LEDGER_PATH}")


def format_claim(claim: Mapping[str, str]) -> str:
    """Render one claim in the ledger's two-line format."""
    return (
        f"[{claim['id']}] {claim['tag']} | {claim['text']}\n"
        f"Artifact: {claim['artifact']} | Falsifier: {claim['falsifier']}\n"
    )


def write_ledger_suggestions(ctx: RunContext, lab_id: str, claims: Sequence[Mapping[str, str]]) -> None:
    """Write drafted claims to the run dir; append to the real ledger only on
    explicit request, because writing the claim is the student's job."""
    body = [
        f"# Suggested ledger claims from {lab_id} ({ctx.run_dir.name})",
        "",
        "These are drafts with measured numbers filled in. Edit them until you",
        "would defend them, then move them into claim_ledger.md (or re-run with",
        "--append-ledger to copy them verbatim, and edit there).",
        "",
        "```text",
    ]
    body += [format_claim(c) for c in claims]
    body += ["```", ""]
    path = ctx.path("ledger_suggestions.md")
    write_text(path, "\n".join(body))
    ctx.register_artifact(path, "summary", "Drafted claim-ledger entries with measured numbers.")

    if ctx.args.append_ledger:
        ensure_ledger()
        with LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(f"\n<!-- appended by {ctx.run_dir.name} -->\n")
            for claim in claims:
                f.write(format_claim(claim) + "\n")
        print(f"[bench] appended {len(claims)} claims to {LEDGER_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab bench for the mechanistic interpretability course.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lab", choices=sorted(LAB_PROFILES), default="lab1", help="Which lab to run.")
    parser.add_argument("--model", default=None,
                        help="HF model id; defaults to the tier's model "
                             "(or the lab's per-tier override, e.g. Lab 7's instruct models).")
    parser.add_argument("--model-revision", default=None, help="Pinned HF revision (commit/tag).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download models/tokenizers; use local cache only.")
    parser.add_argument("--attn-implementation", default="auto", help="Optional HF attention implementation, e.g. eager, sdpa, flash_attention_2.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Pass low_cpu_mem_usage=True during model loading.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16"))
    parser.add_argument("--quantization", default="none", choices=("none", "8bit", "4bit"),
                        help="Optional bitsandbytes quantization for small GPUs.")
    parser.add_argument("--tier", default="auto", choices=("auto", "a", "b", "c"),
                        help="Hardware tier: a = CPU smoke, b = 24GB+/Colab GPU, c = 40-80GB.")
    parser.add_argument("--prompt-set", default=None,
                        help="small | medium | full | path to a custom prompts .json file. "
                             "Default: 'small' on tier a (CPU smoke), 'full' on tiers b/c "
                             "(a tiny set produces degenerate selections that read as findings).")
    parser.add_argument("--max-examples", type=int, default=-1,
                        help="Cap examples; -1 = tier default, 0 = no cap.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k tokens recorded per depth.")
    parser.add_argument("--include-controls", action="store_true", help="Lab 1: include optional weak/scrambled control prompts.")
    parser.add_argument("--ablate-top", type=int, default=3,
                        help="Lab 2: per example, ablate this many top-|attribution| components "
                             "plus matched controls to compare attribution vs causal effect (0 = skip).")
    parser.add_argument("--dla-tolerance", type=float, default=0.02,
                        help="Relative tolerance for the component decomposition self-check "
                             "(absorbs the compute dtype's residual-add rounding; bf16 needs ~0.02).")
    parser.add_argument("--run-edit", action="store_true",
                        help="Lab 5: run the rank-one edit-and-audit extension after patching.")
    parser.add_argument("--graph-nodes", type=int, default=0,
                        help="Lab 9: feature-node budget for the attribution graph "
                             "(0 = tier default; also the number of backward passes).")
    parser.add_argument("--audit-domain", default="factual_qa",
                        choices=("factual_qa", "cot_faithfulness", "sentiment_negation"),
                        help="Lab 11: which curated audit domain to run.")
    parser.add_argument("--relations", default="",
                        help="Lab 12: comma-separated relation-family filter, e.g. "
                             "capital_of,language_of (default: all families).")
    parser.add_argument("--relation-set", default="", choices=("", "small", "medium", "full"),
                        help="Lab 12: item-count preset; overrides --prompt-set for this lab.")
    parser.add_argument("--patch-grid", default="subject,relation,last",
                        help="Lab 12: comma-separated token roles to patch (subject,relation,last).")
    parser.add_argument("--emotions", default="",
                        help="Lab 13: comma-separated emotion filter, e.g. joy,anger (default: all target emotions).")
    parser.add_argument("--mode", default="lora",
                        help="Lab mode selector where supported, e.g. Lab 21: lora | safety_depth | both; Lab 24: single_turn | multi_turn | both; Lab 25: injection | attribution | confidence | both.")
    parser.add_argument("--organism", default="",
                        help="Lab 21/23: path to a Lab 20 run directory, public blind package, or organism directory.")
    parser.add_argument("--blind", action="store_true",
                        help="Lab 23: ignore unsealed manifests even if present; use while writing the pre-unseal report.")
    parser.add_argument("--unsealed-manifest", default="",
                        help="Lab 23: explicit path to a Lab 20 manifest_unsealed.json or directory containing answer keys.")
    parser.add_argument("--hook-tolerance", type=float, default=0.0, help="Allowed max absolute diff in hook parity diagnostics.")
    parser.add_argument("--allow-hook-mismatch", action="store_true", help="Warn instead of aborting on hook parity mismatch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--showcase", default=None,
                        help="Example id to feature in the biography plot (default: first counterfactual example).")
    parser.add_argument("--save-tensors", action="store_true",
                        help="Also save raw residual tensors (with a manifest) per example.")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")
    parser.add_argument("--append-ledger", action="store_true",
                        help="Append suggested claims to claim_ledger.md (default: drafts only).")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="Parent directory for run dirs.")
    parser.add_argument("--run-name", default=None, help="Run directory name (auto-suffixed if taken).")
    parser.add_argument("--run-dir", default=None, help="Exact run directory (overrides root/name).")
    return parser.parse_args(argv)


def apply_tier_defaults(args: argparse.Namespace) -> None:
    """Resolve tier 'auto' and fill model/dtype/max-examples defaults.

    The tier shortcuts are convenience presets, not hidden modes: after this
    function runs the rest of the bench sees ordinary --model/--dtype values,
    and every resolved value lands in run_config.json.
    """
    if args.tier == "auto":
        # Tier choice needs to know whether a GPU exists, but torch is not
        # imported yet (env config must happen first), so probe lazily later?
        # No: tier only affects defaults, and the only signal needed is CUDA
        # availability. Import of torch here is acceptable because tier=auto
        # implies no env-sensitive flags were requested.
        try:
            import torch

            args.tier = "b" if torch.cuda.is_available() else "a"
        except ImportError:
            args.tier = "a"
        print(f"[bench] tier auto-resolved to '{args.tier}'")
    spec = TIER_DEFAULTS[args.tier]
    if args.model is None:
        # Chat/generation labs may override the tier's default model (one
        # place, on purpose) so the registry stays the source of truth instead
        # of every chat lab re-specifying --model.
        lab_model = LAB_PROFILES[args.lab].get(f"model_tier_{args.tier}")
        args.model = lab_model or spec["model"]
    if args.dtype == "auto":
        # A lab may pin its dtype per tier (Lab 9's replacement-model
        # exactness check needs float32 even on GPU tiers).
        args.dtype = LAB_PROFILES[args.lab].get(f"dtype_tier_{args.tier}") or spec["dtype"]
    if args.max_examples < 0:
        lab_override = LAB_PROFILES[args.lab].get(f"max_examples_tier_{args.tier}")
        args.max_examples = int(lab_override) if lab_override else spec["max_examples"]
    if args.prompt_set is None:
        # Tier a is a CPU smoke test where 'small' is the point. On the GPU
        # science tiers, default to 'full': a tiny prompt set produces
        # degenerate selections (a depth chosen at layer <=2, AUC 1.0 on n=8)
        # that read as findings but are noise, so students otherwise had to pass
        # --prompt-set full by hand for every trustworthy tier-b/c result.
        args.prompt_set = "small" if args.tier == "a" else "full"
        print(
            f"[bench] prompt-set defaulted to '{args.prompt_set}' for tier "
            f"'{args.tier}' (pass --prompt-set to override)."
        )
    if LAB_PROFILES[args.lab].get("needs_eager") and args.attn_implementation == "auto":
        args.attn_implementation = "eager"
        print(
            "[bench] this lab captures attention patterns; attn implementation "
            "set to 'eager' (sdpa/flash return empty attentions silently)."
        )


def make_run_dir(args: argparse.Namespace) -> pathlib.Path:
    """Resolve the run directory, avoiding accidental overwrite by default."""
    if args.run_dir:
        return pathlib.Path(args.run_dir).expanduser().resolve()
    run_root = pathlib.Path(args.run_root).expanduser().resolve()
    profile = LAB_PROFILES[args.lab]
    run_name = args.run_name or f"{profile['run_name']}-{now_slug()}-{uuid.uuid4().hex[:6]}"
    base = run_root / sanitize_tag(run_name)
    if not base.exists():
        return base
    for i in range(2, 10_000):
        candidate = base.with_name(f"{base.name}_{i:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find an unused run directory for {base}")


def configure_env(run_dir: pathlib.Path) -> None:
    """Set environment that should exist before torch/transformers import."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # The continuous-batching engine repeatedly resizes its packed KV cache;
    # expandable segments stop allocator fragmentation from masquerading as OOM.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")  # plots are files, not windows
    mpl_config = run_dir / "matplotlib_config"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    apply_tier_defaults(args)

    # Ensure the claim ledger skeleton exists at the course root before anything
    # heavy can fail (including the Lab 1 --tier a smoke test). This fulfills the
    # original pre-lab goal of early initialization so students see their running
    # dossier immediately. (Appends still require --append-ledger; writing claims
    # is coursework.)
    ensure_ledger()
    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_env(run_dir)
    _ensure_plot_style()

    with ConsoleTee(run_dir / "logs"):
        print(f"[bench] run directory: {run_dir}")
        ctx = RunContext(run_dir=run_dir, args=args)
        run_config_path = ctx.path("run_config.json")
        write_json(run_config_path, vars(args))
        ctx.register_artifact(run_config_path, "config", "Resolved CLI arguments after tier defaults.")

        # Heavy imports happen after env config so MPLBACKEND etc. apply.
        import torch

        set_determinism(torch, args.seed)
        metadata_path = ctx.path("run_metadata.json")
        write_json(metadata_path, collect_run_metadata(torch))
        ctx.register_artifact(metadata_path, "diagnostic", "Host, package, git, GPU, and environment metadata.")

        bundle = load_model_and_tokenizer(ctx)
        ctx.bind_model(bundle)

        # Labs import this module by name. When this file runs as a script
        # the module is '__main__', so alias it to keep one shared instance.
        sys.modules.setdefault("interp_bench", sys.modules[__name__])
        if str(COURSE_ROOT) not in sys.path:
            sys.path.insert(0, str(COURSE_ROOT))
        lab_module = importlib.import_module(LAB_PROFILES[args.lab]["module"])

        print(f"[bench] running {args.lab}: {LAB_PROFILES[args.lab]['description']}")
        t0 = time.perf_counter()
        try:
            lab_module.run(ctx, bundle)
        finally:
            # A failed run still leaves partial artifacts; index whatever
            # was registered so the run directory stays auditable.
            write_json(ctx.path("artifact_index.json"), {"artifacts": ctx.artifacts})
        elapsed = time.perf_counter() - t0

        end_mem_path = ctx.path("diagnostics", "gpu_memory_at_end.json")
        write_json(end_mem_path, gpu_memory_snapshot(torch, "end"))
        ctx.register_artifact(end_mem_path, "diagnostic", "GPU memory snapshot at the end of the run.")
        write_json(ctx.path("artifact_index.json"), {"artifacts": ctx.artifacts})
        print(f"[bench] lab finished in {elapsed:.1f}s")
        print(f"[bench] start with: {run_dir / 'run_summary.md'}")
        print(f"[bench] artifact map: {run_dir / 'artifact_index.json'}")


if __name__ == "__main__":
    main()
