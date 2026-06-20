"""Lab 8: Superposition, sparse autoencoders, and transcoders.

Why dense activations resist neuron-level reading, and whether sparse
dictionaries recover units worth naming. Three parts under a strict time
budget, plus a causal extension and two bridges back to earlier labs.

Prerequisites lineage:
- Lab 1: residual stream sites, instrument hygiene, pre/post conventions.
- Lab 4: "decodable does not mean used"; the truth-direction bridge asks whether
  a probe direction is recoverable as a single SAE feature (expect low cosine;
  distributed or no single atom captures it).
- Lab 5/7: causal interventions with controls and measured costs. The clamp here
  uses the same dose discipline as Lab 7 (multiples of observed peak activation,
  not a raw unit vector) plus random control + fluency proxy.
- Lab 6: manual circuit discovery as baseline; here the "validation battery" is
  the automated analogue that forces you to kill labels.
- Lab 9: the transcoder section exists only to give you input→output objects
  (reconstructs the MLP map, not a site snapshot) so that attribution graphs
  can have edges ("this feature causes that feature") rather than just nouns.

* **Part 0 — superposition in a jar (CPU, model-free).** Train the Elhage et
  al. toy model: a small autoencoder packing more sparse features than it has
  dimensions. Sweep sparsity and watch features go from orthogonal-and-
  monosemantic (dense) to overlapping-in-superposition (sparse). This is the
  *why* of everything after it, and it runs in seconds on a laptop.

* **Part 1 — the feature atlas (the core).** Load a pretrained SAE for the
  tier's model (gpt2 + jbloom resid SAE on Tier A; Olmo-3-1025-7B + decoderesearch
  SAE on Tier B), run it over the frozen domain-tagged corpus, rank features
  two ways (max activation vs firing frequency — the rankings disagree, and
  that disagreement is a lesson), retrieve top-activating contexts, propose a
  label, and then **validate the label** with apparatus instead of vibes:
  held-out AUC against domain membership, an adversarial confusable-pair test
  that separates a *concept* feature from a *token* feature, a polysemanticity
  score, and a verdict — survived / narrowed / token-feature / polysemantic.
  The atlas is required to contain at least one label that died.

* **Part 2 — the transcoder (the bridge to Lab 9).** Load a gpt2 MLP
  transcoder. An SAE reconstructs activations *at a site*; a transcoder
  reconstructs the MLP's *computation* (input -> output). Verify reconstruction
  (FVU, and the downstream-logit KL when its output is spliced in for the real
  MLP), inspect a few transcoder features by de-embedding them, and state why
  feature-level circuit tracing wants input->output objects, not site snapshots.

* **Bridges & extension.** Cosine between the best SAE feature's decoder
  direction and Lab 4's truth direction (decodable concept vs SAE coordinate);
  and a CAUSAL feature-clamp: steer with one *validated* feature's decoder
  direction and measure the behavioral push toward its domain against a
  random-feature control.

Evidence level: OBSERVATION / DECODABILITY at the feature level, upgraded to
CAUSAL for the one clamped-and-controlled feature.
"""

from __future__ import annotations

import csv
import copy
import math
import pathlib
import random
import statistics
from collections import Counter, defaultdict
from typing import Any

import interp_bench as bench

LAB_ID = "L08"

# ---------------------------------------------------------------------------
# Pins. Pretrained, ungated SAE/transcoder weights matched to the course's
# tier models. Validated at authoring time (FVU/L0 in the run summary); the
# conventions below — centering for the TL-trained gpt2 SAEs, bare LayerNorm
# for the transcoder, jumprelu threshold for the Olmo SAE — were each settled
# empirically, not guessed, because guessing them silently triples the FVU.
# ---------------------------------------------------------------------------

# Tier A: gpt2-small. jbloom resid_pre SAE (TransformerLens-trained: the
# residual stream was centered, so the SAE only reconstructs well on a
# per-token-demeaned input — that is `center_input=True`).
GPT2_SAE = {
    "repo": "jbloom/GPT2-Small-SAEs-Reformatted",
    "subdir": "blocks.8.hook_resid_pre",
    "weights": "sae_weights.safetensors",
    "layer": 8, "site": "pre", "center_input": True, "sub_b_dec": True, "jumprelu": False,
}
# gpt2 MLP transcoder (Dunefsky et al.). Input is the *bare* LayerNorm of the
# pre-MLP residual (no affine gamma/beta), output is the MLP output.
GPT2_TRANSCODER = {
    "repo": "jacobdunefsky/gpt2small-transcoders",
    "subdir": "gpt2-small-dun-chl-mlp-tc8",
    "weights": "sae.safetensors",
    "layer": 8,
}
# Tier B: Olmo-3-1025-7B. decoderesearch jumprelu SAE on a mid layer's output.
OLMO_SAE = {
    "repo": "decoderesearch/olmo-3-saes",
    "subdir": "olmo-3-1025-7b/btk-mat-layer-16-k-100",
    "weights": "sae_weights.safetensors",
    "layer": 16, "site": "post", "center_input": False, "sub_b_dec": True, "jumprelu": True,
}

SAE_REGISTRY = {
    "gpt2-jbloom-resid-pre-l8": {
        **GPT2_SAE,
        "id": "gpt2-jbloom-resid-pre-l8",
        "model_substrings": ("gpt2",),
        "description": "jbloom gpt2-small resid_pre SAE, block 8.",
        "convention_note": "TransformerLens-trained gpt2 residual SAE; centered input and b_dec subtraction reproduce the authoring FVU.",
    },
    "olmo3-1025-7b-decoderesearch-l16-post": {
        **OLMO_SAE,
        "id": "olmo3-1025-7b-decoderesearch-l16-post",
        "model_substrings": ("olmo-3-1025-7b",),
        "description": "decoderesearch Olmo-3-1025-7B jumprelu SAE, layer 16 resid_post.",
        "convention_note": "decoderesearch jumprelu SAE; b_dec subtraction plus thresholded jumprelu is the documented/validated convention.",
    },
}

N_ATLAS_FEATURES = 20        # features to label and validate (bumped for more statistical power on verdicts/rankings)
N_TOP_CONTEXTS = 6           # top-activating lines retrieved per feature
N_LABEL_CONTEXTS = 4         # of those, how many inform the proposed label
# Clamp dose as multiples of the feature's peak activation. The window is
# narrow: a clean concept feature induces its concept around 1x peak and
# collapses into repetition by ~3x, so the sweep straddles both.
CLAMP_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)

# Confusable domain pairs (same surface tokens, different concept). The
# adversarial near-miss test asks whether a feature separates these.
CONFUSABLE_PAIRS = {
    "chemistry": "cooking", "cooking": "chemistry",
    "finance": "sports", "sports": "finance",
    "law": "medicine", "medicine": "law",
    "weather": "emotion", "emotion": "weather",
}

# Keyword batteries for scoring the causal feature-clamp generations. Kept
# broad (the obvious semantic shift uses far more words than a 10-item list),
# so the measured hit count tracks the qualitative effect instead of
# undercounting it. Matching is substring on lowercased text.
DOMAIN_KEYWORDS = {
    "chemistry": ["acid", "base", "reaction", "molecule", "electron", "solution", "atom", "bond", "ph",
                  "ion", "chemical", "compound", "oxid", "catalyst", "proton", "salt", "gas", "element"],
    "cooking": ["sauce", "heat", "salt", "dough", "pan", "flavor", "cook", "season", "simmer", "butter",
                "recipe", "bake", "roast", "spice", "dish", "ingredient", "oven", "boil", "taste"],
    "sports": ["game", "team", "score", "ball", "player", "match", "win", "goal", "race", "field",
               "championship", "coach", "athlete", "season", "tournament", "league", "play", "defense"],
    "finance": ["stock", "market", "revenue", "rate", "investor", "profit", "shares", "fund", "yield",
                "bank", "dollar", "earnings", "trading", "economy", "price", "capital", "debt", "invest"],
    "law": ["court", "trial", "judge", "jury", "law", "contract", "plaintiff", "ruling", "ruled", "statute",
            "legal", "defendant", "guilty", "crime", "charges", "lawyer", "attorney", "verdict", "sentence",
            "appeal", "witness", "evidence", "lawsuit", "prosecut", "convict", "justice"],
    "medicine": ["patient", "drug", "blood", "disease", "doctor", "treatment", "dose", "clinical", "surgery",
                 "tumor", "diagnos", "symptom", "medical", "hospital", "therapy", "cancer", "vaccine", "heart"],
    "weather": ["storm", "rain", "wind", "cloud", "snow", "temperature", "fog", "sky", "cold", "hurricane",
                "forecast", "weather", "thunder", "lightning", "humid", "frost", "warm", "degrees"],
    "emotion": ["fear", "joy", "grief", "anger", "hope", "sadness", "pride", "dread", "relief", "loneliness",
                "happy", "afraid", "angry", "sad", "love", "anxious", "feel", "emotion", "lonely", "cried"],
    "code": ["def", "return", "for ", "import", "class", "function", "value", "array", "loop", "print",
             "variable", "string", "method", "code", "compile", "syntax", "object", "integer"],
    "history": ["empire", "war", "century", "treaty", "king", "ancient", "revolution", "nation", "rome",
                "history", "battle", "emperor", "dynasty", "medieval", "conquest", "kingdom", "civilization"],
}
CLAMP_PROMPTS = ["Here is a short paragraph.\n\n", "Let me tell you a story. ", "Today I want to talk about "]
CAUSAL_NEUTRAL_PROMPTS = [
    "Write one short paragraph about a situation that has not been specified yet.\n\n",
    "Here is a neutral opening sentence. Continue it with concrete details:\n\n",
    "A person looked at the notes on the desk and began to explain.\n\n",
    "The report was brief, factual, and open-ended:\n\n",
    "Continue this passage without using a list:\n\n",
    "The meeting started with a simple observation.\n\n",
    "In a plain paragraph, describe what happened next.\n\n",
    "The room was quiet while the speaker chose the next topic.\n\n",
    "A short memo began with the following sentence:\n\n",
    "The example was intentionally ordinary and unspecialized.\n\n",
    "Someone asked for a concise explanation.\n\n",
    "The document opened with a general statement.\n\n",
    "A narrator described the scene in neutral terms.\n\n",
    "The first draft avoided naming any specific field.\n\n",
    "The paragraph continued with a practical detail.\n\n",
    "The speaker wrote a few careful sentences.\n\n",
    "The note did not yet have a topic, but it needed one.\n\n",
    "A generic example can be completed in many ways.\n\n",
    "The next sentence made the idea more specific.\n\n",
    "Continue with a sober, factual paragraph.\n\n",
]

SEMANTIC_FAMILIES = {
    "chemistry", "cooking", "sports", "finance", "law",
    "medicine", "weather", "emotion", "code", "history",
}
LABEL_TYPES = {
    "code_indentation_whitespace": "position/whitespace",
    "python_syntax": "syntax/format",
    "markdown_list_formatting": "syntax/format",
    "urls_emails_paths": "lexical-token",
    "dates_numbers_measurements": "lexical-token",
    "citations_legal_references": "syntax/format",
    "quotes_dialogue": "syntax/format",
    "capitalization_acronyms": "lexical-token",
    "sentiment_emotion": "semantic-domain",
    "named_entities": "semantic-domain",
}
TARGETED_CANDIDATE_POOL = 80
TARGETED_REPORT_TOPK = 20
MIN_SPLIT_POSITIVES = 5
MIN_SPLIT_NEGATIVES = 20


# ---------------------------------------------------------------------------
# SAE / transcoder container
# ---------------------------------------------------------------------------


class LoadedSAE:
    """A pretrained dictionary (SAE or transcoder) with its encode/decode
    convention baked in. ``encode`` returns feature activations [..., d_sae];
    ``decode`` maps features back to the reconstructed activation [..., d_in].

    The conventions (centering, b_dec subtraction, jumprelu thresholding) are
    NOT cosmetic: each was validated to minimize FVU at authoring time, and
    getting one wrong silently wrecks reconstruction (see the lab handout's
    debugging table)."""

    def __init__(self, weights: dict[str, Any], *, center_input: bool, sub_b_dec: bool,
                 jumprelu: bool, kind: str, layer: int, site: str):
        import torch

        self.W_enc = weights["W_enc"].float()      # [d_in, d_sae]
        self.b_enc = weights["b_enc"].float()      # [d_sae]
        self.W_dec = weights["W_dec"].float()      # [d_sae, d_in]
        self.b_dec = weights["b_dec"].float()      # [d_in]
        self.threshold = weights["threshold"].float() if "threshold" in weights else None
        self.center_input = center_input
        self.sub_b_dec = sub_b_dec
        self.jumprelu = jumprelu and self.threshold is not None
        self.kind = kind
        self.layer = layer
        self.site = site
        self.d_in = self.W_enc.shape[0]
        self.d_sae = self.W_enc.shape[1]
        self.dec_norms = torch.linalg.vector_norm(self.W_dec, dim=-1)  # [d_sae]

    def to(self, device: Any) -> "LoadedSAE":
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_dec = self.b_dec.to(device)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device)
        self.dec_norms = self.dec_norms.to(device)
        return self

    def _prep(self, x: Any) -> Any:
        if self.center_input:
            x = x - x.mean(-1, keepdim=True)
        return x

    def encode(self, x: Any) -> Any:
        import torch

        x = self._prep(x).to(self.W_enc.dtype)
        xin = x - self.b_dec if self.sub_b_dec else x
        pre = xin @ self.W_enc + self.b_enc
        if self.jumprelu:
            return pre * (pre > self.threshold)
        return torch.relu(pre)

    def decode(self, feats: Any) -> Any:
        return feats @ self.W_dec + self.b_dec

    def reconstruct(self, x: Any) -> Any:
        """Reconstruction in the SAE's own (possibly centered) target space."""
        return self.decode(self.encode(x))


def _download(spec: dict[str, str], cfg_name: str | None = None) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    filename = f"{spec['subdir'].strip('/')}/{spec['weights']}" if spec.get("subdir") else spec["weights"]
    wpath = hf_hub_download(spec["repo"], filename)
    return load_file(wpath)


def _arg_value(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _bool_arg_value(args: Any, name: str) -> bool | None:
    value = getattr(args, name, None)
    return value if value is None else bool(value)


def resolve_sae_spec(args: Any, bundle: bench.ModelBundle) -> dict[str, Any]:
    """Resolve the exact SAE spec from the registry plus CLI overrides."""
    model_id = bundle.anatomy.model_id.lower()
    requested_id = _arg_value(args, "sae_id", "") or "auto"
    explicit_repo = bool(_arg_value(args, "sae_repo", ""))

    if requested_id == "auto" and explicit_repo:
        base: dict[str, Any] = {
            "id": "custom",
            "repo": _arg_value(args, "sae_repo", ""),
            "subdir": _arg_value(args, "sae_subdir", ""),
            "weights": _arg_value(args, "sae_weights", "") or "sae_weights.safetensors",
            "layer": _arg_value(args, "sae_layer", None),
            "site": _arg_value(args, "sae_site", "post"),
            "center_input": False,
            "sub_b_dec": True,
            "jumprelu": False,
            "model_substrings": (),
            "description": "CLI-specified custom SAE.",
            "convention_note": "Custom SAE; unresolved conventions are chosen by the loading calibration.",
            "auto_convention": True,
        }
    elif requested_id == "auto":
        matches = [
            copy.deepcopy(spec)
            for spec in SAE_REGISTRY.values()
            if any(fragment in model_id for fragment in spec.get("model_substrings", ()))
        ]
        if not matches:
            available = ", ".join(sorted(SAE_REGISTRY))
            raise RuntimeError(
                f"No default SAE registered for model {bundle.anatomy.model_id!r}. "
                f"Pass --sae-id or --sae-repo/--sae-subdir/--sae-weights. Available ids: {available}"
            )
        base = matches[0]
    else:
        if requested_id not in SAE_REGISTRY:
            available = ", ".join(sorted(SAE_REGISTRY))
            raise RuntimeError(f"Unknown --sae-id {requested_id!r}. Available ids: {available}")
        base = copy.deepcopy(SAE_REGISTRY[requested_id])

    overrides = {
        "repo": _arg_value(args, "sae_repo", "") or None,
        "subdir": _arg_value(args, "sae_subdir", "") or None,
        "weights": _arg_value(args, "sae_weights", "") or None,
        "layer": _arg_value(args, "sae_layer", None),
        "site": _arg_value(args, "sae_site", None),
        "center_input": _bool_arg_value(args, "sae_center_input"),
        "sub_b_dec": _bool_arg_value(args, "sae_sub_b_dec"),
        "jumprelu": _bool_arg_value(args, "sae_jumprelu"),
    }
    convention_overrides = {key for key in ("center_input", "sub_b_dec", "jumprelu") if overrides[key] is not None}
    for key, value in overrides.items():
        if value is not None:
            base[key] = value
    if convention_overrides:
        base["auto_convention"] = False
    base["convention_overrides"] = sorted(convention_overrides)

    missing = [key for key in ("repo", "weights", "layer", "site") if base.get(key) in ("", None)]
    if missing:
        raise RuntimeError(f"Incomplete SAE spec; missing {missing}. Pass explicit --sae-* flags.")
    if base["site"] not in {"pre", "post"}:
        raise RuntimeError(f"Unsupported SAE site {base['site']!r}; expected 'pre' or 'post'.")
    base["layer"] = int(base["layer"])
    base["center_input"] = bool(base.get("center_input", False))
    base["sub_b_dec"] = bool(base.get("sub_b_dec", False))
    base["jumprelu"] = bool(base.get("jumprelu", False))
    return base


def _sae_from_spec(weights: dict[str, Any], spec: dict[str, Any]) -> LoadedSAE:
    return LoadedSAE(
        weights,
        center_input=bool(spec["center_input"]),
        sub_b_dec=bool(spec["sub_b_dec"]),
        jumprelu=bool(spec["jumprelu"]),
        kind="sae",
        layer=int(spec["layer"]),
        site=str(spec["site"]),
    )


def load_model_sae(bundle: bench.ModelBundle, spec: dict[str, Any], weights: dict[str, Any] | None = None) -> LoadedSAE:
    """Load a specific SAE spec. No implicit family fallback is allowed."""
    print(f"[lab8] loading SAE {spec['repo']}/{spec.get('subdir', '')} (layer {spec['layer']}, {spec['site']})")
    weights = weights or _download(spec)
    sae = _sae_from_spec(weights, spec)
    if sae.d_in != bundle.anatomy.d_model:
        raise RuntimeError(f"SAE d_in {sae.d_in} != model d_model {bundle.anatomy.d_model}; wrong SAE for this model.")
    print(f"[lab8]   SAE d_in={sae.d_in} d_sae={sae.d_sae} jumprelu={sae.jumprelu}")
    return sae


def calibration_sites(spec: dict[str, Any], n_layers: int) -> list[tuple[int, str]]:
    layer = int(spec["layer"])
    site = str(spec["site"])
    candidates = [(layer, site), (layer, "post" if site == "pre" else "pre")]
    if layer > 0:
        candidates.append((layer - 1, site))
    if layer + 1 < n_layers:
        candidates.append((layer + 1, site))
    out: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for item in candidates:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def calibrate_sae_loading(ctx: bench.RunContext, bundle: bench.ModelBundle, spec: dict[str, Any],
                          weights: dict[str, Any], corpus: list[dict[str, str]],
                          sample_size: int = 4) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Try plausible loading conventions and write a reconstruction health report."""
    import torch

    sample = corpus[:max(1, min(sample_size, len(corpus)))]
    has_threshold = "threshold" in weights
    relu_modes = [False, True] if has_threshold else [False]
    locations = calibration_sites(spec, len(bundle.blocks))
    acts_by_loc: dict[tuple[int, str], list[Any]] = {}
    print(f"[lab8]   calibrating SAE loading on {len(sample)} corpus rows")
    for layer, site in locations:
        loc_acts = []
        for row in sample:
            _, acts = capture_site(bundle, row["text"], layer, site)
            loc_acts.append(acts.to(bundle.input_device))
        acts_by_loc[(layer, site)] = loc_acts

    rows: list[dict[str, Any]] = []
    for layer, site in locations:
        for center in (False, True):
            for sub_b_dec in (False, True):
                for jumprelu in relu_modes:
                    test_spec = {
                        **spec,
                        "layer": layer,
                        "site": site,
                        "center_input": center,
                        "sub_b_dec": sub_b_dec,
                        "jumprelu": jumprelu,
                    }
                    sae = _sae_from_spec(weights, test_spec).to(bundle.input_device)
                    num = den = 0.0
                    l0s = []
                    peak = 0.0
                    ok = True
                    try:
                        with torch.no_grad():
                            for acts in acts_by_loc[(layer, site)]:
                                feats = sae.encode(acts)
                                recon = sae.decode(feats)
                                target = sae._prep(acts)
                                num += float((target - recon).pow(2).sum())
                                den += float((target - target.mean(0, keepdim=True)).pow(2).sum())
                                l0s.append(float((feats > 1e-6).float().sum(-1).mean()))
                                peak = max(peak, float(feats.max()))
                    except Exception as exc:
                        ok = False
                        rows.append({
                            "layer": layer, "site": site, "center_input": center, "sub_b_dec": sub_b_dec,
                            "jumprelu": jumprelu, "ok": False, "error": repr(exc),
                        })
                    if ok:
                        rows.append({
                            "layer": layer,
                            "site": site,
                            "center_input": center,
                            "sub_b_dec": sub_b_dec,
                            "jumprelu": jumprelu,
                            "ok": True,
                            "fvu": round(num / max(den, 1e-9), 6),
                            "mean_l0": round(sum(l0s) / max(len(l0s), 1), 3),
                            "peak_activation": round(peak, 6),
                            "selected_location": layer == int(spec["layer"]) and site == spec["site"],
                            "selected_convention": (
                                center == bool(spec["center_input"])
                                and sub_b_dec == bool(spec["sub_b_dec"])
                                and jumprelu == bool(spec["jumprelu"] and has_threshold)
                            ),
                        })

    ok_rows = [r for r in rows if r.get("ok") and math.isfinite(float(r.get("fvu", 999.0)))]
    if not ok_rows:
        raise RuntimeError("SAE loading calibration failed for every tested convention.")
    best = min(ok_rows, key=lambda r: float(r["fvu"]))
    if spec.get("auto_convention"):
        selected_rows = [
            r for r in ok_rows
            if int(r["layer"]) == int(spec["layer"]) and r["site"] == spec["site"]
        ]
        if not selected_rows:
            raise RuntimeError("No successful calibration rows at the requested SAE layer/site.")
        chosen = min(selected_rows, key=lambda r: float(r["fvu"]))
        spec["center_input"] = bool(chosen["center_input"])
        spec["sub_b_dec"] = bool(chosen["sub_b_dec"])
        spec["jumprelu"] = bool(chosen["jumprelu"])
        spec["convention_note"] = (
            spec.get("convention_note", "")
            + f" Calibration selected center_input={spec['center_input']}, "
              f"sub_b_dec={spec['sub_b_dec']}, jumprelu={spec['jumprelu']} at the requested site."
        ).strip()
    else:
        chosen = next(
            (
                r for r in ok_rows
                if int(r["layer"]) == int(spec["layer"])
                and r["site"] == spec["site"]
                and bool(r["center_input"]) == bool(spec["center_input"])
                and bool(r["sub_b_dec"]) == bool(spec["sub_b_dec"])
                and bool(r["jumprelu"]) == bool(spec["jumprelu"] and has_threshold)
            ),
            None,
        )
    if chosen is None:
        raise RuntimeError("Chosen SAE convention was not among successful calibration rows.")

    report = {
        "sae_spec": spec,
        "sample_rows": [r.get("text_id", r.get("row_id", "")) for r in sample],
        "has_threshold": has_threshold,
        "best_by_fvu": best,
        "chosen": chosen,
        "known_convention_note": spec.get("convention_note", ""),
        "chosen_is_best": (
            int(chosen["layer"]) == int(best["layer"])
            and chosen["site"] == best["site"]
            and bool(chosen["center_input"]) == bool(best["center_input"])
            and bool(chosen["sub_b_dec"]) == bool(best["sub_b_dec"])
            and bool(chosen["jumprelu"]) == bool(best["jumprelu"])
        ),
        "rows": rows,
    }
    bench.write_json(ctx.path("sae_loading_calibration.json"), report)
    ctx.register_artifact(ctx.path("sae_loading_calibration.json"), "diagnostic",
                          "SAE convention calibration: layer/site, centering, b_dec subtraction, and jumprelu.")
    if float(chosen["fvu"]) > 1.05 or float(chosen["mean_l0"]) <= 0.0:
        raise RuntimeError(
            f"SAE reconstruction looks broken under chosen convention: FVU={chosen['fvu']} "
            f"L0={chosen['mean_l0']}. See sae_loading_calibration.json."
        )
    return report, rows


# ---------------------------------------------------------------------------
# Activation capture at the SAE hook site
# ---------------------------------------------------------------------------


def capture_site(bundle: bench.ModelBundle, text: str, layer: int, site: str) -> tuple[list[str], Any]:
    """Return (token_texts, activations[seq, d_model]) at the SAE's hook site.

    ``site='pre'`` captures the block's INPUT (resid_pre); ``site='post'`` its
    OUTPUT (resid_post). Hooking the exact module the SAE was trained on avoids
    every off-by-one the residual-stream index convention could introduce."""
    import torch

    tok = bundle.tokenizer
    ids = tok(text, return_tensors="pt")["input_ids"].to(bundle.input_device)
    block = bundle.blocks[layer]
    grabbed: dict[str, Any] = {}

    def pre_hook(module: Any, args: tuple) -> None:
        grabbed["act"] = bench.tensor_cpu_float(args[0][0])

    def post_hook(module: Any, args: tuple, output: Any) -> None:
        out = output[0] if isinstance(output, tuple) else output
        grabbed["act"] = bench.tensor_cpu_float(out[0])

    handle = block.register_forward_pre_hook(pre_hook) if site == "pre" \
        else block.register_forward_hook(post_hook)
    try:
        with torch.no_grad():
            bundle.model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    token_texts = [tok.decode([i]) for i in ids[0].tolist()]
    return token_texts, grabbed["act"]


# ---------------------------------------------------------------------------
# Part 0: toy model of superposition
# ---------------------------------------------------------------------------


def run_toy_superposition(ctx: bench.RunContext, seed: int) -> dict[str, Any]:
    """Elhage et al. toy model: n_features > d_hidden, sparse inputs.

    As sparsity rises the autoencoder packs more features than it has
    dimensions, in superposition, accepting interference. Returns geometry
    stats and (unless --no-plots) writes the geometry figure."""
    import torch

    n_features, d_hidden, steps = 20, 5, 6000
    importance = torch.tensor([0.85 ** i for i in range(n_features)])
    sparsities = [0.0, 0.7, 0.9, 0.97, 0.99]
    gen = torch.Generator().manual_seed(seed)

    results: dict[float, dict[str, Any]] = {}
    for sparsity in sparsities:
        # Small init + AdamW + enough steps is what makes the canonical collapse
        # appear: dense, the model represents exactly d_hidden features
        # orthogonally and drops the rest; as sparsity rises it packs more than
        # d_hidden in superposition, accepting interference.
        W = (torch.randn(d_hidden, n_features, generator=gen) * 0.1).requires_grad_(True)
        b = torch.zeros(n_features, requires_grad=True)
        opt = torch.optim.AdamW([W, b], lr=1e-3, weight_decay=0.0)
        loss_val = 0.0
        for _ in range(steps):
            x = torch.rand(2048, n_features, generator=gen)
            mask = (torch.rand(2048, n_features, generator=gen) > sparsity).float()
            x = x * mask
            h = x @ W.T
            xr = torch.relu(h @ W + b)
            loss = (importance * (x - xr) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.detach())
        Wf = W.detach()
        norms = torch.linalg.vector_norm(Wf, dim=0)               # per-feature norm
        unit = Wf / norms.clamp_min(1e-6)
        cos_gram = unit.T @ unit                                   # cosine Gram in [-1, 1]
        represented = int((norms > 0.3).sum())
        rep_idx = (norms > 0.3).nonzero().flatten()
        if len(rep_idx) > 1:
            g = cos_gram[rep_idx][:, rep_idx].abs().clone()
            g.fill_diagonal_(0.0)
            interference = float(g.max(dim=1).values.mean())      # mean nearest-neighbor overlap
        else:
            interference = 0.0
        results[sparsity] = {
            "norms": norms.tolist(), "represented": represented,
            "interference": round(interference, 4), "gram": cos_gram.tolist(),
            "final_loss": round(loss_val, 6),
        }
        print(f"[lab8]   toy sparsity {sparsity:.2f}: {represented}/{n_features} features represented "
              f"in {d_hidden} dims, interference {interference:.3f}")

    stats = {
        "n_features": n_features, "d_hidden": d_hidden, "steps": steps,
        "sparsities": sparsities,
        "represented_by_sparsity": {str(s): results[s]["represented"] for s in sparsities},
        "interference_by_sparsity": {str(s): results[s]["interference"] for s in sparsities},
    }
    bench.write_json(ctx.path("toy_superposition_stats.json"), stats)
    ctx.register_artifact(ctx.path("toy_superposition_stats.json"), "metrics",
                          "Toy-model geometry: features represented and interference vs sparsity.")
    if not ctx.args.no_plots:
        plot_toy_geometry(ctx, results, sparsities, n_features, d_hidden)
    return stats


def plot_toy_geometry(ctx, results, sparsities, n_features, d_hidden) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    bench._ensure_plot_style()
    fig = plt.figure(figsize=(15.0, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    for s in sparsities:
        ax1.plot(range(n_features), sorted(results[s]["norms"], reverse=True),
                 marker="o", markersize=3, linewidth=1.6, label=f"sparsity {s:.2f}")
    bench.add_vline(ax1, d_hidden - 0.5, label=f"d_hidden = {d_hidden}", color="black", ls="--", lw=0.8, alpha=0.8)
    ax1.set_xlabel("feature (sorted by norm)")
    ax1.set_ylabel("‖W column‖ (is the feature represented?)")
    ax1.set_title("Dense: only d_hidden features survive.\nSparse: more, via superposition.")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    # style the first ax lightly (subplots in toy are custom)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    for idx, s in [(2, sparsities[0]), (3, sparsities[-1])]:
        ax = fig.add_subplot(1, 3, idx)
        gram = np.array(results[s]["gram"])
        im = ax.imshow(gram, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"WᵀW at sparsity {s:.2f}\n(off-diagonal = interference)")
        ax.set_xlabel("feature"); ax.set_ylabel("feature")
        fig.colorbar(im, ax=ax, fraction=0.046)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Toy model of superposition: dense → orthogonal/monosemantic, sparse → overlapping")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "toy_superposition_geometry.png",
                      "Feature norms and WᵀW interference across sparsity levels.")


# ---------------------------------------------------------------------------
# Part 1: encode the corpus, rank, label, validate
# ---------------------------------------------------------------------------


def load_corpus(path_override: str | None = None) -> list[dict[str, str]]:
    if path_override:
        path = pathlib.Path(path_override).expanduser()
        if not path.is_absolute():
            path = bench.COURSE_ROOT / path
    else:
        path = bench.COURSE_ROOT / "data" / "sae_feature_corpus.csv"
    if not path.exists():
        raise RuntimeError(f"Frozen corpus missing: {path}. Run data/make_sae_corpus.py once at authoring time.")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, str]] = []
    for i, row in enumerate(rows, start=1):
        normalized = dict(row)
        normalized.setdefault("text_id", normalized.get("row_id", f"T{i:03d}"))
        normalized.setdefault("row_id", normalized["text_id"])
        normalized.setdefault("domain", normalized.get("family", "unknown"))
        normalized.setdefault("family", normalized.get("domain", "unknown"))
        normalized.setdefault("split", "all")
        normalized.setdefault("hard_negative_group", "")
        normalized.setdefault("lexical_markers", "")
        normalized.setdefault("notes", "")
        if not normalized.get("text"):
            raise RuntimeError(f"Corpus row {i} in {path} is missing required text.")
        out.append(normalized)
    return out


def encode_corpus(bundle, sae, corpus) -> dict[str, Any]:
    """One forward pass per line; accumulate per-line max activation and
    per-token firing frequency without storing the full dense activation
    tensor (which is d_sae-wide and would be wasteful at 65k features)."""
    import torch

    n = len(corpus)
    line_max = torch.zeros(n, sae.d_sae)
    line_mean = torch.zeros(n, sae.d_sae)
    fire_count = torch.zeros(sae.d_sae)
    recon_fvu_num = recon_fvu_den = 0.0
    n_tokens = 0
    for i, row in enumerate(corpus):
        toks, acts = capture_site(bundle, row["text"], sae.layer, sae.site)
        acts = acts.to(sae.W_enc.device)
        feats = sae.encode(acts)                      # [seq, d_sae]
        recon = sae.decode(feats)
        target = sae._prep(acts)
        recon_fvu_num += float((target - recon).pow(2).sum())
        recon_fvu_den += float((target - target.mean(0, keepdim=True)).pow(2).sum())
        line_max[i] = feats.max(0).values.cpu()
        line_mean[i] = feats.mean(0).cpu()
        fire_count += (feats > 1e-6).float().sum(0).cpu()
        n_tokens += feats.shape[0]
        if (i + 1) % 40 == 0:
            print(f"[lab8]   encoded {i + 1}/{n} lines")
    l0 = float((line_max > 1e-6).float().sum(-1).mean())  # rough; refined below per-token
    return {
        "line_max": line_max, "line_mean": line_mean, "fire_count": fire_count,
        "n_tokens": n_tokens, "fvu": recon_fvu_num / max(recon_fvu_den, 1e-9),
    }


def per_token_l0(bundle, sae, corpus, sample: int = 12) -> float:
    """Honest L0 (mean active features per token) over a sample of lines."""
    import torch

    counts = []
    for row in corpus[:sample]:
        _, acts = capture_site(bundle, row["text"], sae.layer, sae.site)
        feats = sae.encode(acts.to(sae.W_enc.device))
        counts.append(float((feats > 1e-6).float().sum(-1).mean()))
    return sum(counts) / len(counts)


def roc_auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def domain_of(corpus, i) -> str:
    return corpus[i]["domain"]


def propose_and_validate(bundle, sae, corpus, enc, feature: int) -> dict[str, Any]:
    """The graded function: label a feature from its top contexts, then test
    the label with held-out AUC, a confusable-pair adversarial check, and a
    polysemanticity score. Returns one atlas row."""
    import torch

    line_max = enc["line_max"]
    acts = line_max[:, feature]
    order = torch.argsort(acts, descending=True).tolist()
    single = [i for i in order if "+" not in corpus[i]["domain"]]

    # --- propose: majority domain of the top label-contexts ---
    label_ctx = single[:N_LABEL_CONTEXTS]
    label_domains = [domain_of(corpus, i) for i in label_ctx]
    proposed = max(set(label_domains), key=label_domains.count) if label_domains else "none"
    label_purity = label_domains.count(proposed) / max(len(label_domains), 1)

    # --- polysemanticity: domain entropy over the top contexts ---
    top_ctx = single[:N_TOP_CONTEXTS]
    top_domains = [domain_of(corpus, i) for i in top_ctx]
    dist = {d: top_domains.count(d) / len(top_domains) for d in set(top_domains)} if top_domains else {}
    entropy = -sum(p * math.log(p, 2) for p in dist.values()) if dist else 0.0

    # --- held-out AUC: domain membership, EXCLUDING the label contexts ---
    held_idx = [i for i in single if i not in set(label_ctx)]
    pos = [float(acts[i]) for i in held_idx if domain_of(corpus, i) == proposed]
    neg = [float(acts[i]) for i in held_idx if domain_of(corpus, i) != proposed]
    held_auc = roc_auc(pos, neg)

    # --- adversarial confusable pair: concept feature vs token feature ---
    confuse = CONFUSABLE_PAIRS.get(proposed)
    pair_auc = None
    pair_margin = None
    if confuse:
        in_dom = [float(acts[i]) for i in single if domain_of(corpus, i) == proposed]
        near = [float(acts[i]) for i in single if domain_of(corpus, i) == confuse]
        if in_dom and near:
            pair_auc = roc_auc(in_dom, near)
            pair_margin = (sum(in_dom) / len(in_dom)) - (sum(near) / len(near))

    # --- verdict ---
    # Ordering matters and encodes the epistemics: a feature that separates its
    # domain from everything ELSE at high AUC is monosemantic-enough for that
    # domain; if it still cannot separate the *confusable twin* (chemistry vs
    # cooking) it is firing on the shared surface token, not the concept. Only
    # when the domain AUC itself is low do we ask whether the spread is
    # polysemantic (high domain entropy) or just noise (killed).
    max_act = float(acts.max())
    fire_frac = float(enc["fire_count"][feature]) / enc["n_tokens"]
    token_feature = confuse is not None and pair_auc is not None and pair_auc < 0.65
    if max_act < 1e-5 or fire_frac == 0.0:
        verdict = "silent-on-corpus"
    elif held_auc >= 0.85 and label_purity >= 0.75:
        verdict = "token-feature" if token_feature else "survived"
    elif held_auc >= 0.70:
        verdict = "token-feature" if token_feature else "narrowed"
    elif entropy > 1.5:
        verdict = "polysemantic"
    else:
        verdict = "killed"

    # --- top-context evidence with the peak-activating token highlighted ---
    evidence = []
    for i in top_ctx[:3]:
        toks, a = capture_site(bundle, corpus[i]["text"], sae.layer, sae.site)
        feats = sae.encode(a.to(sae.W_enc.device))[:, feature]
        peak = int(feats.argmax())
        hl = "".join((f"⟦{t}⟧" if j == peak else t) for j, t in enumerate(toks))
        evidence.append({"text_id": corpus[i]["text_id"], "domain": corpus[i]["domain"],
                         "peak_act": round(float(feats[peak]), 3), "highlight": hl[:160]})

    return {
        "feature": feature, "proposed_label": proposed, "label_purity": round(label_purity, 3),
        "max_activation": round(max_act, 4), "fire_fraction": round(fire_frac, 5),
        "held_out_auc": round(held_auc, 4),
        "confusable_with": confuse or "",
        "confusable_auc": round(pair_auc, 4) if pair_auc is not None else "",
        "confusable_margin": round(pair_margin, 4) if pair_margin is not None else "",
        "polysemy_entropy_bits": round(entropy, 3),
        "top_domains": ",".join(top_domains),
        "verdict": verdict,
        "evidence": evidence,
    }


def rank_features(enc, k: int) -> dict[str, list[int]]:
    """Two rankings that famously disagree: by peak activation vs by how often
    the feature fires across the corpus."""
    import torch

    by_max = torch.argsort(enc["line_max"].max(0).values, descending=True)[:k].tolist()
    by_freq = torch.argsort(enc["fire_count"], descending=True)[:k].tolist()
    return {"by_max": by_max, "by_freq": by_freq}


# ---------------------------------------------------------------------------
# Part 2: transcoder
# ---------------------------------------------------------------------------


def load_gpt2_transcoder() -> LoadedSAE:
    weights = _download(GPT2_TRANSCODER)
    # Transcoder: bare-LN input (no centering flag — we feed it the normalized
    # pre-MLP residual directly), no b_dec subtraction, relu.
    sae = LoadedSAE(weights, center_input=False, sub_b_dec=False, jumprelu=False,
                    kind="transcoder", layer=GPT2_TRANSCODER["layer"], site="mlp")
    return sae


def get_gpt2_for_transcoder(bundle: bench.ModelBundle):
    """The transcoder is pinned to gpt2 (the ungated MLP transcoder weights the
    course uses). On Tier A the loaded model already IS gpt2; on Tier B we load
    a small auxiliary gpt2 so the transcoder lesson is available on every tier."""
    if "gpt2" in bundle.anatomy.model_id.lower():
        return bundle.model, bundle.tokenizer, "reused loaded gpt2"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[lab8] Part 2: loading auxiliary gpt2 for the transcoder demo (main model is not gpt2)")
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
    model.eval()
    return model, tok, "loaded auxiliary gpt2"


def verify_transcoder(model, tok, tc: LoadedSAE, corpus) -> dict[str, Any]:
    """Reconstruct the MLP's computation (bare-LN(pre-MLP resid) -> mlp_out) and
    measure both FVU and the downstream-logit KL when the reconstruction is
    spliced in for the real MLP output."""
    import torch

    dev = next(model.parameters()).device
    tc.to(dev)
    block = model.transformer.h[tc.layer]
    ln2, mlp = block.ln_2, block.mlp
    num = den = 0.0
    l0s = []
    kls = []
    for row in corpus[:24]:
        ids = tok(row["text"], return_tensors="pt")["input_ids"].to(dev)
        grab: dict[str, Any] = {}
        h_in = ln2.register_forward_pre_hook(lambda m, a: grab.__setitem__("mid", a[0][0].detach()))
        h_out = mlp.register_forward_hook(
            lambda m, a, o: grab.__setitem__("mlp_out", (o[0] if isinstance(o, tuple) else o)[0].detach()))
        with torch.no_grad():
            base = model(ids)
        h_in.remove(); h_out.remove()
        mid = grab["mid"]                                   # pre-MLP residual
        mu = mid.mean(-1, keepdim=True)
        var = mid.var(-1, keepdim=True, unbiased=False)
        bare_ln = (mid - mu) / torch.sqrt(var + ln2.eps)    # no affine
        feats = tc.encode(bare_ln)
        recon = tc.decode(feats)
        target = grab["mlp_out"]
        num += float((target - recon).pow(2).sum())
        den += float((target - target.mean(0, keepdim=True)).pow(2).sum())
        l0s.append(float((feats > 1e-6).float().sum(-1).mean()))

        # splice: replace the real MLP output with the transcoder reconstruction
        def splice(m, a, o, r=recon):
            base_o = o[0] if isinstance(o, tuple) else o
            new = base_o.clone()
            new[0] = r.to(new.dtype)
            return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
        hs = mlp.register_forward_hook(splice)
        with torch.no_grad():
            spliced = model(ids)
        hs.remove()
        p = torch.log_softmax(base.logits[0, -1].float(), -1)
        q = torch.log_softmax(spliced.logits[0, -1].float(), -1)
        kls.append(float((p.exp() * (p - q)).sum()))
    return {
        "fvu": round(num / max(den, 1e-9), 4),
        "mean_l0": round(sum(l0s) / len(l0s), 2),
        "mean_splice_kl": round(sum(kls) / len(kls), 4),
        "max_splice_kl": round(max(kls), 4),
    }


def inspect_transcoder_features(model, tok, tc: LoadedSAE, corpus, n: int = 3) -> list[dict[str, Any]]:
    """De-embed a few transcoder output features: a transcoder feature's
    decoder row lives in MLP-output space, so projecting it through the
    unembedding says which tokens the feature *promotes* — a direct read of
    what the computation does, which an SAE-at-a-site cannot give you."""
    import torch

    dev = next(model.parameters()).device
    tc.to(dev)
    block = model.transformer.h[tc.layer]
    ln2 = block.ln_2
    W_U = model.lm_head.weight                # [vocab, d_model]
    ln_f = model.transformer.ln_f

    # rank transcoder features by peak activation across a sample
    peak = torch.zeros(tc.d_sae, device=dev)
    for row in corpus[:40]:
        ids = tok(row["text"], return_tensors="pt")["input_ids"].to(dev)
        grab = {}
        h = ln2.register_forward_pre_hook(lambda m, a: grab.__setitem__("mid", a[0][0].detach()))
        with torch.no_grad():
            model(ids)
        h.remove()
        mid = grab["mid"]
        bare = (mid - mid.mean(-1, keepdim=True)) / torch.sqrt(mid.var(-1, keepdim=True, unbiased=False) + ln2.eps)
        feats = tc.encode(bare)
        peak = torch.maximum(peak, feats.max(0).values)
    top = torch.argsort(peak, descending=True)[:n].tolist()

    out = []
    for f in top:
        dec = tc.W_dec[f]                                   # [d_model] in mlp-out space
        with torch.no_grad():
            normed = ln_f(dec.unsqueeze(0)).squeeze(0)      # approximate path to logits
            logits = normed @ W_U.T
        toptok = torch.argsort(logits, descending=True)[:8].tolist()
        out.append({
            "feature": f, "peak_activation": round(float(peak[f]), 3),
            "promotes_tokens": [tok.decode([t]).strip() for t in toptok],
        })
    return out


# ---------------------------------------------------------------------------
# Bridge & causal extension
# ---------------------------------------------------------------------------


def truth_direction_bridge(sae: LoadedSAE, bundle) -> dict[str, Any]:
    """Cosine of every SAE feature's decoder direction with Lab 4's truth
    direction; report the best-aligned feature. Decodable concept or just a
    convenient coordinate?"""
    import torch

    runs = sorted((bench.COURSE_ROOT / "runs").glob("lab04*/tables/truth_direction.pt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return {"found": False, "note": "no Lab 4 truth_direction.pt found"}
    meta = torch.load(runs[0], map_location="cpu", weights_only=False)
    vec = meta.get("direction")
    if vec is None or len(vec) != sae.d_in:
        return {"found": True, "path": str(runs[0].relative_to(bench.COURSE_ROOT)),
                "note": f"truth direction dim {None if vec is None else len(vec)} != SAE d_in {sae.d_in} "
                        f"(computed on {meta.get('model_id')}); cosine skipped — directions are model-specific"}
    v = torch.as_tensor(vec, dtype=torch.float32)
    v = v / v.norm().clamp_min(1e-9)
    dec_unit = sae.W_dec.cpu() / sae.dec_norms.cpu().clamp_min(1e-9)[:, None]
    cos = dec_unit @ v
    abs_cos = cos.abs()
    best = int(abs_cos.argmax())
    top = torch.argsort(abs_cos, descending=True)[:12].tolist()
    qs = torch.quantile(abs_cos, torch.tensor([0.50, 0.90, 0.99], dtype=abs_cos.dtype)).tolist()
    return {
        "found": True,
        "path": str(runs[0].relative_to(bench.COURSE_ROOT)),
        "saved_on_model": meta.get("model_id"),
        "best_feature": best,
        "best_cosine": round(float(cos[best]), 4),
        "abs_cosine_quantiles": {"p50": round(float(qs[0]), 4), "p90": round(float(qs[1]), 4), "p99": round(float(qs[2]), 4)},
        "top_cosines": [
            {"rank": i + 1, "feature": int(f), "cosine": round(float(cos[f]), 4), "abs_cosine": round(float(abs_cos[f]), 4)}
            for i, f in enumerate(top)
        ],
    }


def clamp_feature_steering(bundle, sae, corpus, feature: int, label: str, peak_act: float,
                           seed: int) -> dict[str, Any]:
    """CAUSAL: clamp one validated feature ON during generation and measure the
    push toward its domain's vocabulary, against a random-feature control.

    Clamping feature f to activation ``a`` means adding ``a * W_dec[f]`` to the
    residual stream — the feature's own decoder contribution at that activation.
    Doses are multiples of the feature's observed PEAK activation, so the scale
    is physically meaningful and transfers across models (a unit direction times
    a small constant does nothing in a stream whose norm is in the hundreds —
    the same mistake Lab 7 calls out). The random control uses another feature's
    raw decoder row at the same multiples."""
    import torch

    real_dir = sae.W_dec[feature].detach().cpu()                # raw decoder row
    gen = torch.Generator().manual_seed(seed * 7 + feature)
    rand_feat = int(torch.randint(0, sae.d_sae, (1,), generator=gen))
    rand_dir = sae.W_dec[rand_feat].detach().cpu()
    kws = DOMAIN_KEYWORDS.get(label, [])

    def domain_hits(text: str) -> int:
        low = text.lower()
        return sum(1 for k in kws if k in low)

    def distinct_ratio(text: str) -> float:
        """Fluency proxy: unique-token fraction. Steering past the window
        collapses generation into repetition, which this catches."""
        words = text.split()
        return len(set(words)) / max(len(words), 1)

    rows = []
    for cond, direction in (("real", real_dir), ("random", rand_dir)):
        for mult in CLAMP_SCALES:
            coef = mult * peak_act                              # clamp to mult × peak activation
            hits = 0
            distinct = []
            sample = ""
            for prompt in CLAMP_PROMPTS:
                gen_text = bench.generate_text(bundle, prompt, max_new_tokens=40,
                                               steer=(sae.layer, direction, coef))
                hits += domain_hits(gen_text)
                distinct.append(distinct_ratio(gen_text))
                if not sample:
                    sample = gen_text[:160]
            rows.append({"condition": cond, "clamp_mult": mult, "coef": round(coef, 3),
                         "domain_keyword_hits": hits,
                         "distinct_ratio": round(sum(distinct) / len(distinct), 3),
                         "label": label, "sample": sample})

    base = next(r for r in rows if r["condition"] == "real" and r["clamp_mult"] == 0.0)
    # A fluent dose keeps generation from collapsing into repetition.
    fluent_real = [r for r in rows if r["condition"] == "real" and r["distinct_ratio"] >= 0.4]
    best = max(fluent_real, key=lambda r: r["domain_keyword_hits"]) if fluent_real else base
    rand_top = max(r["domain_keyword_hits"] for r in rows if r["condition"] == "random")
    return {"feature": feature, "label": label, "random_feature": rand_feat, "peak_act": round(peak_act, 3),
            "rows": rows, "real_base_hits": base["domain_keyword_hits"],
            "real_max_hits": best["domain_keyword_hits"], "best_mult": best["clamp_mult"],
            "random_max_hits": rand_top,
            "causal": best["domain_keyword_hits"] > base["domain_keyword_hits"]
            and best["domain_keyword_hits"] > rand_top}


def text_terms(text: str) -> list[str]:
    import re

    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+|[0-9]+(?:\.[0-9]+)?|https?://|@|/|\\\\|§", text.lower())


def keywords_for_family(corpus: list[dict[str, str]], family: str) -> list[str]:
    if family in DOMAIN_KEYWORDS:
        return DOMAIN_KEYWORDS[family]
    markers: list[str] = []
    for row in corpus:
        if family_of(row) == family:
            markers.extend(str(row.get("lexical_markers", "")).split(";"))
    out = []
    seen = set()
    for marker in markers:
        marker = marker.strip().lower()
        if marker and marker not in seen:
            out.append(marker)
            seen.add(marker)
    return out[:24]


def train_family_probe(corpus: list[dict[str, str]], family: str):
    """Tiny one-vs-rest lexical probe trained on the frozen corpus."""
    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()
    n_pos = n_neg = 0
    for row in corpus:
        terms = text_terms(row["text"])
        if family_of(row) == family:
            pos_counts.update(terms)
            n_pos += 1
        else:
            neg_counts.update(terms)
            n_neg += 1
    vocab = set(pos_counts) | set(neg_counts)
    alpha = 1.0
    pos_total = sum(pos_counts.values()) + alpha * max(len(vocab), 1)
    neg_total = sum(neg_counts.values()) + alpha * max(len(vocab), 1)
    prior = math.log((n_pos + 1) / max(n_neg + 1, 1))

    def score(text: str) -> float:
        total = prior
        for term in text_terms(text):
            lp = math.log((pos_counts.get(term, 0) + alpha) / pos_total)
            ln = math.log((neg_counts.get(term, 0) + alpha) / neg_total)
            total += lp - ln
        return total

    return score


def distinct_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / max(len(words), 1)


def domain_hits_for_keywords(text: str, keywords: list[str]) -> int:
    low = text.lower()
    return sum(1 for k in keywords if k and k.lower() in low)


def matched_control_features(sae, enc, feature: int, n: int) -> list[int]:
    import torch

    fire = enc["fire_count"].float() / max(enc["n_tokens"], 1)
    peak = enc["line_max"].max(0).values.float()
    norm = sae.dec_norms.detach().cpu().float()
    target = torch.tensor([norm[feature], fire[feature], peak[feature]], dtype=torch.float32)
    cols = torch.stack([norm, fire, peak], dim=1)
    scale = cols.std(0).clamp_min(1e-6)
    dist = (((cols - target) / scale) ** 2).sum(1)
    dist[feature] = float("inf")
    dist[fire <= 0] = float("inf")
    return [int(x) for x in torch.argsort(dist)[:n].tolist()]


def positive_prompts_for_family(corpus: list[dict[str, str]], family: str, n: int) -> list[str]:
    rows = [r for r in corpus if family_of(r) == family and split_for_row(r, 0) == "test"]
    if len(rows) < n:
        rows.extend(r for r in corpus if family_of(r) == family and r not in rows)
    prompts = []
    for row in rows[:n]:
        text = str(row["text"]).replace("\n", " ")[:220]
        prompts.append(f"Continue this {family} passage in the same style:\n\n{text}\n\n")
    return prompts


def aggregate_generation_rows(bundle, layer: int, direction: Any, coef: float, prompts: list[str],
                              keywords: list[str], probe_score, *, feature: int, family: str,
                              condition: str, phase: str, dose_mult: float,
                              control_feature: int | str = "") -> dict[str, Any]:
    hits = []
    probes = []
    distincts = []
    sample = ""
    for prompt in prompts:
        text = bench.generate_text(bundle, prompt, max_new_tokens=32, steer=(layer, direction, coef))
        hits.append(domain_hits_for_keywords(text, keywords))
        probes.append(probe_score(text))
        distincts.append(distinct_ratio(text))
        if not sample:
            sample = text[:240]
    return {
        "feature": feature,
        "family": family,
        "phase": phase,
        "condition": condition,
        "control_feature": control_feature,
        "dose_mult": dose_mult,
        "coef": round(coef, 4),
        "n_prompts": len(prompts),
        "total_keyword_hits": sum(hits),
        "mean_keyword_hits": round(sum(hits) / max(len(hits), 1), 4),
        "mean_probe_score": round(sum(probes) / max(len(probes), 1), 4),
        "mean_distinct_ratio": round(sum(distincts) / max(len(distincts), 1), 4),
        "sample": sample,
    }


def enhanced_causal_feature_test(ctx, bundle, sae, corpus, enc, feature: int, family: str,
                                 seed: int, *, n_prompts: int, n_controls: int) -> dict[str, Any]:
    """Matched-control causal suite for one validated feature."""
    import matplotlib.pyplot as plt

    peak_act = float(enc["line_max"][:, feature].max())
    direction = sae.W_dec[feature].detach().cpu()
    controls = matched_control_features(sae, enc, feature, n_controls)
    keywords = keywords_for_family(corpus, family)
    probe_score = train_family_probe(corpus, family)
    neutral = CAUSAL_NEUTRAL_PROMPTS[:n_prompts]
    positive = positive_prompts_for_family(corpus, family, n_prompts)
    rows: list[dict[str, Any]] = []

    for mult in (0.0, 0.5, 1.0, 1.5, 2.0):
        rows.append(aggregate_generation_rows(
            bundle, sae.layer, direction, mult * peak_act, neutral, keywords, probe_score,
            feature=feature, family=family, condition="real", phase="clamp_on", dose_mult=mult,
        ))
        for control in controls:
            rows.append(aggregate_generation_rows(
                bundle, sae.layer, sae.W_dec[control].detach().cpu(), mult * peak_act, neutral, keywords, probe_score,
                feature=feature, family=family, condition="matched_control", phase="clamp_on",
                dose_mult=mult, control_feature=control,
            ))

    for mult in (0.0, 0.5, 1.0):
        rows.append(aggregate_generation_rows(
            bundle, sae.layer, direction, -mult * peak_act, positive, keywords, probe_score,
            feature=feature, family=family, condition="real", phase="suppress", dose_mult=-mult,
        ))

    real_rows = [r for r in rows if r["phase"] == "clamp_on" and r["condition"] == "real"]
    control_rows = [r for r in rows if r["phase"] == "clamp_on" and r["condition"] == "matched_control"]
    base = next(r for r in real_rows if float(r["dose_mult"]) == 0.0)
    fluent = [r for r in real_rows if _safe_float(r["mean_distinct_ratio"], 0.0) >= 0.4]
    best = max(fluent, key=lambda r: (_safe_float(r["mean_probe_score"]), _safe_float(r["mean_keyword_hits"]))) if fluent else base
    controls_same_dose = [r for r in control_rows if float(r["dose_mult"]) == float(best["dose_mult"])]
    control_probe = max((_safe_float(r["mean_probe_score"], -1e9) for r in controls_same_dose), default=-1e9)
    control_hits = max((_safe_float(r["mean_keyword_hits"], 0.0) for r in controls_same_dose), default=0.0)
    suppress_rows = [r for r in rows if r["phase"] == "suppress" and r["condition"] == "real"]
    suppress_base = next(r for r in suppress_rows if float(r["dose_mult"]) == 0.0)
    suppress_best = min(suppress_rows, key=lambda r: _safe_float(r["mean_probe_score"], 0.0))
    causal = (
        _safe_float(best["mean_probe_score"]) > control_probe
        and _safe_float(best["mean_keyword_hits"]) >= control_hits
        and _safe_float(best["mean_probe_score"]) > _safe_float(base["mean_probe_score"])
        and _safe_float(suppress_best["mean_probe_score"]) < _safe_float(suppress_base["mean_probe_score"])
        and _safe_float(best["mean_distinct_ratio"]) >= 0.4
    )
    summary = {
        "feature": feature,
        "family": family,
        "peak_act": round(peak_act, 4),
        "matched_controls": controls,
        "keywords": keywords,
        "real_base_probe": base["mean_probe_score"],
        "real_best_probe": best["mean_probe_score"],
        "real_best_hits": best["mean_keyword_hits"],
        "best_dose_mult": best["dose_mult"],
        "control_max_probe_same_dose": round(control_probe, 4),
        "control_max_hits_same_dose": round(control_hits, 4),
        "suppress_base_probe": suppress_base["mean_probe_score"],
        "suppress_min_probe": suppress_best["mean_probe_score"],
        "suppress_best_dose_mult": suppress_best["dose_mult"],
        "causal": causal,
    }

    bench.write_csv_with_context(ctx, ctx.path("tables", "causal_feature_tests.csv"), rows)
    ctx.register_artifact(ctx.path("tables", "causal_feature_tests.csv"), "table",
                          "Matched-control causal feature tests: clamp-on and suppression with probe/keyword/fluency scores.")
    bench.write_json(ctx.path("causal_feature_tests_summary.json"), summary)
    ctx.register_artifact(ctx.path("causal_feature_tests_summary.json"), "metrics",
                          "Summary of the matched-control causal feature test.")

    lines = [
        f"# Causal Feature Card: Feature {feature} ({family})",
        "",
        f"- matched controls: {controls}",
        f"- best real dose: {best['dose_mult']}x peak",
        f"- real probe: {base['mean_probe_score']} -> {best['mean_probe_score']}",
        f"- matched-control max probe at same dose: {summary['control_max_probe_same_dose']}",
        f"- suppression probe: {suppress_base['mean_probe_score']} -> {suppress_best['mean_probe_score']}",
        f"- causal claim passed: {causal}",
        "",
        "## Sample At Best Dose",
        "",
        str(best.get("sample", "")),
        "",
        "## Sample Under Strongest Suppression",
        "",
        str(suppress_best.get("sample", "")),
        "",
    ]
    bench.write_text(ctx.path("causal_feature_card.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("causal_feature_card.md"), "summary",
                          "Causal feature card for the matched-control suite.")

    bench._ensure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=False)
    xs = [r["dose_mult"] for r in real_rows]
    axes[0].plot(xs, [r["mean_probe_score"] for r in real_rows], marker="o", label="real", color=_clamp_color("real"))
    by_dose: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in control_rows:
        by_dose[float(row["dose_mult"])].append(row)
    ctrl_x = sorted(by_dose)
    ctrl_y = [max(_safe_float(r["mean_probe_score"], -1e9) for r in by_dose[x]) for x in ctrl_x]
    axes[0].plot(ctrl_x, ctrl_y, marker="s", label="best matched control", color=_clamp_color("random"))
    axes[0].set_title("Clamp-on probe score")
    axes[0].set_xlabel("dose (x peak activation)")
    axes[0].set_ylabel("corpus-trained probe score")
    axes[0].legend(fontsize=8)
    supp_x = [abs(float(r["dose_mult"])) for r in suppress_rows]
    axes[1].plot(supp_x, [r["mean_probe_score"] for r in suppress_rows], marker="o", color=_clamp_color("real"))
    axes[1].set_title("Clamp-off / suppression")
    axes[1].set_xlabel("negative dose (x peak activation)")
    axes[1].set_ylabel("probe score on positive prompts")
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(f"Causal operating window: feature {feature} ({family})")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "causal_operating_window.png",
                      "Matched-control causal operating window for the selected feature.")
    return summary


def choose_causal_candidate(targeted_rows: list[dict[str, Any]], atlas: list[dict[str, Any]]) -> tuple[int, str] | None:
    grade_rank = {"survived_strong": 5, "survived_weak": 4, "lexical_valid": 3, "narrowed": 2}
    candidates = [
        r for r in targeted_rows
        if r.get("status") == "ok" and grade_rank.get(str(r.get("claim_grade")), 0) > 0
    ]
    if candidates:
        best = max(
            candidates,
            key=lambda r: (
                grade_rank.get(str(r.get("claim_grade")), 0),
                _safe_float(r.get("test_auc"), 0.5),
                -_safe_float(r.get("fire_fraction"), 1.0),
            ),
        )
        return int(best["feature"]), str(best["family"])
    survivors = [r for r in atlas if r.get("verdict") in ("survived", "narrowed")]
    if survivors:
        best = max(survivors, key=lambda r: _safe_float(r.get("held_out_auc"), 0.5) - _safe_float(r.get("fire_fraction"), 0.0))
        return int(best["feature"]), str(best["proposed_label"])
    return None



# ---------------------------------------------------------------------------
# Lab 8 visualization helpers and synthesis tables
# ---------------------------------------------------------------------------

VERDICT_ORDER = ("survived", "narrowed", "token-feature", "polysemantic", "killed", "silent-on-corpus")
FEATURE_EVIDENCE_COLUMNS = ("held_out_auc", "confusable_auc", "label_purity", "polysemy_clean", "rarity_clean")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x == "" or x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _verdict_color(verdict: str) -> str:
    getter = getattr(bench, "plot_sae_verdict_color", None)
    if callable(getter):
        return getter(verdict)
    return {
        "survived": "#009E73",
        "narrowed": "#8A9A00",
        "token-feature": "#E69F00",
        "polysemantic": "#7E57C2",
        "killed": "#D55E00",
        "silent-on-corpus": "#999999",
    }.get(str(verdict), "#555555")


def _domain_color(domain: str) -> str:
    getter = getattr(bench, "plot_sae_domain_color", None)
    if callable(getter):
        return getter(domain)
    palette = {
        "chemistry": "#0072B2", "cooking": "#E69F00", "sports": "#009E73", "finance": "#8A9A00",
        "law": "#7E57C2", "medicine": "#CC79A7", "weather": "#56B4E9", "emotion": "#D55E00",
        "code": "#666666", "history": "#A6761D", "none": "#999999",
    }
    return palette.get(str(domain), "#555555")


def _clamp_color(condition: str) -> str:
    getter = getattr(bench, "plot_feature_condition_color", None)
    if callable(getter):
        return getter(condition)
    return {"real": "#D55E00", "random": "#777777", "feature": "#D55E00", "control": "#777777"}.get(str(condition), "#555555")


def _concept_score(row: dict[str, Any]) -> float:
    """Score for sorting feature rows: validate strongly, fire sparsely, avoid polysemy."""
    auc = _safe_float(row.get("held_out_auc"), 0.5)
    purity = _safe_float(row.get("label_purity"), 0.0)
    conf = _safe_float(row.get("confusable_auc"), 0.5)
    fire = _safe_float(row.get("fire_fraction"), 0.0)
    poly = _safe_float(row.get("polysemy_entropy_bits"), 0.0)
    return round(0.45 * auc + 0.25 * purity + 0.20 * conf + 0.10 * max(0.0, 1.0 - fire * 20.0) - 0.08 * poly, 4)


def feature_evidence_rows(atlas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in atlas:
        conf_auc = _safe_float(r.get("confusable_auc"), 0.5)
        poly = _safe_float(r.get("polysemy_entropy_bits"), 0.0)
        fire = _safe_float(r.get("fire_fraction"), 0.0)
        rows.append({
            "feature": r.get("feature"),
            "proposed_label": r.get("proposed_label", ""),
            "verdict": r.get("verdict", ""),
            "held_out_auc": r.get("held_out_auc", ""),
            "confusable_with": r.get("confusable_with", ""),
            "confusable_auc": r.get("confusable_auc", ""),
            "confusable_margin": r.get("confusable_margin", ""),
            "label_purity": r.get("label_purity", ""),
            "polysemy_entropy_bits": r.get("polysemy_entropy_bits", ""),
            "polysemy_clean": round(max(0.0, 1.0 - poly / 2.5), 4),
            "fire_fraction": r.get("fire_fraction", ""),
            "rarity_clean": round(max(0.0, 1.0 - fire * 20.0), 4),
            "max_activation": r.get("max_activation", ""),
            "concept_score": _concept_score(r),
            "top_domains": r.get("top_domains", ""),
        })
    return sorted(rows, key=lambda x: (-_safe_float(x.get("concept_score")), str(x.get("feature"))))


def domain_validation_summary(atlas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in atlas:
        by_domain[str(r.get("proposed_label", "none"))].append(r)
    rows = []
    for domain, rs in sorted(by_domain.items()):
        aucs = [_safe_float(r.get("held_out_auc"), 0.5) for r in rs]
        fires = [_safe_float(r.get("fire_fraction"), 0.0) for r in rs]
        verdicts = Counter(str(r.get("verdict", "")) for r in rs)
        rows.append({
            "proposed_label": domain,
            "n_features": len(rs),
            "median_held_out_auc": round(statistics.median(aucs), 4) if aucs else "",
            "max_held_out_auc": round(max(aucs), 4) if aucs else "",
            "median_fire_fraction": round(statistics.median(fires), 5) if fires else "",
            "n_survived": verdicts.get("survived", 0),
            "n_narrowed": verdicts.get("narrowed", 0),
            "n_token_feature": verdicts.get("token-feature", 0),
            "n_polysemantic": verdicts.get("polysemantic", 0),
            "n_killed": verdicts.get("killed", 0),
            "n_silent": verdicts.get("silent-on-corpus", 0),
        })
    return rows


def label_type_for(family: str) -> str:
    if family in LABEL_TYPES:
        return LABEL_TYPES[family]
    if family in SEMANTIC_FAMILIES:
        return "semantic-domain"
    return "broad-basis/polysemantic"


def split_for_row(row: dict[str, str], i: int) -> str:
    split = str(row.get("split", "")).strip().lower()
    if split in {"train", "dev", "test"}:
        return split
    bucket = i % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "dev"
    return "test"


def family_of(row: dict[str, str]) -> str:
    return str(row.get("family") or row.get("domain") or "unknown")


def auc_by_feature(scores: Any, labels: Any, chunk_size: int = 4096) -> Any:
    """Vectorized rank-AUC for every feature column in scores[n_rows, d_sae]."""
    import torch

    labels = labels.bool()
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return torch.full((scores.shape[1],), 0.5, dtype=torch.float32)
    aucs = []
    rank_base = torch.arange(1, scores.shape[0] + 1, dtype=torch.float32).unsqueeze(1)
    label_float = labels.float().unsqueeze(1)
    for start in range(0, scores.shape[1], chunk_size):
        chunk = scores[:, start:start + chunk_size].float()
        order = torch.argsort(chunk, dim=0, stable=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks.scatter_(0, order, rank_base.expand(-1, chunk.shape[1]))
        sum_pos_ranks = (ranks * label_float).sum(0)
        auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
        aucs.append(auc.cpu())
    return torch.cat(aucs, dim=0)


def auc_for_feature(scores: Any, labels: list[bool] | Any, feature: int) -> float:
    vals = scores[:, feature].tolist()
    pos = [float(v) for v, y in zip(vals, labels) if bool(y)]
    neg = [float(v) for v, y in zip(vals, labels) if not bool(y)]
    return roc_auc(pos, neg)


def bootstrap_auc_ci(pos: list[float], neg: list[float], seed: int, draws: int = 200) -> dict[str, Any]:
    if not pos or not neg:
        return {"auc": 0.5, "ci_low": "", "ci_high": "", "draws": 0}
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        bs_pos = [pos[rng.randrange(len(pos))] for _ in pos]
        bs_neg = [neg[rng.randrange(len(neg))] for _ in neg]
        vals.append(roc_auc(bs_pos, bs_neg))
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return {"auc": roc_auc(pos, neg), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "draws": draws}


def permutation_auc_null(pos: list[float], neg: list[float], seed: int, draws: int = 200) -> dict[str, Any]:
    if not pos or not neg:
        return {"null_mean": "", "p_ge_observed": "", "draws": 0}
    rng = random.Random(seed)
    values = list(pos) + list(neg)
    labels = [True] * len(pos) + [False] * len(neg)
    observed = roc_auc(pos, neg)
    nulls = []
    for _ in range(draws):
        rng.shuffle(labels)
        null_pos = [v for v, y in zip(values, labels) if y]
        null_neg = [v for v, y in zip(values, labels) if not y]
        nulls.append(roc_auc(null_pos, null_neg))
    p_ge = (1 + sum(1 for v in nulls if v >= observed)) / (len(nulls) + 1)
    return {"null_mean": round(sum(nulls) / len(nulls), 4), "p_ge_observed": round(p_ge, 4), "draws": draws}


def subset_stability_auc(pos: list[float], neg: list[float], seed: int, draws: int = 20) -> dict[str, Any]:
    if len(pos) < 4 or len(neg) < 4:
        return {"subset_auc_mean": "", "subset_auc_std": "", "draws": 0}
    rng = random.Random(seed)
    vals = []
    n_pos = max(2, int(len(pos) * 0.8))
    n_neg = max(2, int(len(neg) * 0.8))
    for _ in range(draws):
        vals.append(roc_auc(rng.sample(pos, n_pos), rng.sample(neg, n_neg)))
    return {
        "subset_auc_mean": round(sum(vals) / len(vals), 4),
        "subset_auc_std": round(statistics.pstdev(vals), 4),
        "draws": draws,
    }


def family_entropy(families: list[str]) -> float:
    if not families:
        return 0.0
    counts = Counter(families)
    total = len(families)
    return -sum((n / total) * math.log(n / total, 2) for n in counts.values())


def targeted_grade(row: dict[str, Any]) -> str:
    if row.get("status") == "insufficient_data":
        return "insufficient_data"
    label_type = row.get("label_type", "")
    test_auc = _safe_float(row.get("test_auc"), 0.5)
    ci_low = _safe_float(row.get("test_auc_ci_low"), 0.0)
    conf = _safe_float(row.get("test_confusable_auc"), 0.75)
    purity = _safe_float(row.get("test_top20_purity"), 0.0)
    entropy = _safe_float(row.get("test_top20_family_entropy"), 0.0)
    fire = _safe_float(row.get("fire_fraction"), 0.0)
    lexical_like = label_type in {"lexical-token", "syntax/format", "position/whitespace"}
    if lexical_like and test_auc >= 0.85 and purity >= 0.55:
        return "lexical_valid"
    if label_type == "semantic-domain" and test_auc >= 0.78 and row.get("test_confusable_auc") not in ("", None) and conf < 0.65:
        return "token_feature_mislabeled"
    if label_type == "semantic-domain" and test_auc >= 0.88 and ci_low >= 0.75 and conf >= 0.75 and purity >= 0.55 and fire <= 0.25:
        return "survived_strong"
    if label_type == "semantic-domain" and test_auc >= 0.78 and conf >= 0.65 and purity >= 0.45:
        return "survived_weak"
    if test_auc >= 0.68:
        return "narrowed"
    if entropy >= 2.0:
        return "polysemantic"
    return "killed"


def write_targeted_feature_card(ctx, corpus, enc, row: dict[str, Any]) -> None:
    feature = int(row["feature"])
    family = str(row["family"])
    acts = enc["line_max"][:, feature]
    order = sorted(range(len(corpus)), key=lambda i: float(acts[i]), reverse=True)[:12]
    lines = [
        f"# Feature {feature}: {family}",
        "",
        f"- label type: `{row.get('label_type')}`",
        f"- claim grade: `{row.get('claim_grade')}`",
        f"- train AUC: {row.get('train_auc')} | dev AUC: {row.get('dev_auc')} | test AUC: {row.get('test_auc')} "
        f"[{row.get('test_auc_ci_low')}, {row.get('test_auc_ci_high')}]",
        f"- confusable test AUC: {row.get('test_confusable_auc')}",
        f"- fire fraction: {row.get('fire_fraction')} | purity top20: {row.get('test_top20_purity')} | "
        f"family entropy top20: {row.get('test_top20_family_entropy')}",
        f"- permutation null mean: {row.get('permutation_null_mean')} | p>=observed: {row.get('permutation_p_ge_observed')}",
        "",
        "## Top Activating Rows",
        "",
    ]
    for i in order:
        text = str(corpus[i]["text"]).replace("\n", " ")[:260]
        lines.append(
            f"- `{corpus[i].get('row_id', corpus[i].get('text_id', i))}` split={split_for_row(corpus[i], i)} "
            f"family={family_of(corpus[i])} domain={corpus[i].get('domain')} act={float(acts[i]):.3f}: {text}"
        )
    path = ctx.path("feature_cards", f"{feature}_{family}.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", f"Targeted feature card for feature {feature} labeled {family}.")


def targeted_feature_search(ctx, corpus, enc, seed: int) -> list[dict[str, Any]]:
    """Supervised fair-shot search with train/dev/test separation."""
    import torch

    families = sorted({family_of(r) for r in corpus if "+" not in family_of(r)})
    split_indices = {
        split: [i for i, r in enumerate(corpus) if split_for_row(r, i) == split]
        for split in ("train", "dev", "test")
    }
    line_max = enc["line_max"].float()
    candidate_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    prereg = {
        "candidate_pool": TARGETED_CANDIDATE_POOL,
        "report_topk": TARGETED_REPORT_TOPK,
        "min_split_positives": MIN_SPLIT_POSITIVES,
        "min_split_negatives": MIN_SPLIT_NEGATIVES,
        "grade_rules": {
            "lexical_valid": "lexical/syntax/whitespace label, test_auc >= 0.85, top20 purity >= 0.55",
            "token_feature_mislabeled": "semantic label with test_auc >= 0.78 but confusable_auc < 0.65",
            "survived_strong": "semantic label, test_auc >= 0.88, ci_low >= 0.75, confusable_auc >= 0.75, purity >= 0.55, fire <= 0.25",
            "survived_weak": "semantic label, test_auc >= 0.78, confusable_auc >= 0.65, purity >= 0.45",
            "narrowed": "test_auc >= 0.68 but not stronger",
        },
    }
    bench.write_json(ctx.path("targeted_search_preregistration.json"), prereg)
    ctx.register_artifact(ctx.path("targeted_search_preregistration.json"), "diagnostic",
                          "Preregistered targeted search thresholds and grade rules.")

    for family in families:
        label_type = label_type_for(family)
        by_split: dict[str, dict[str, Any]] = {}
        sufficient = True
        for split, idxs in split_indices.items():
            labels = [family_of(corpus[i]) == family for i in idxs]
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            by_split[split] = {"idxs": idxs, "labels": labels, "n_pos": n_pos, "n_neg": n_neg}
            if n_pos < MIN_SPLIT_POSITIVES or n_neg < MIN_SPLIT_NEGATIVES:
                sufficient = False
        if not sufficient:
            best_rows.append({
                "family": family,
                "label_type": label_type,
                "status": "insufficient_data",
                "train_pos": by_split["train"]["n_pos"],
                "dev_pos": by_split["dev"]["n_pos"],
                "test_pos": by_split["test"]["n_pos"],
                "claim_grade": "insufficient_data",
            })
            continue

        train_scores = line_max[by_split["train"]["idxs"]]
        train_labels = torch.tensor(by_split["train"]["labels"], dtype=torch.bool)
        train_auc_all = auc_by_feature(train_scores, train_labels)
        top_features = torch.argsort(train_auc_all, descending=True)[:TARGETED_CANDIDATE_POOL].tolist()

        dev_scores = line_max[by_split["dev"]["idxs"]]
        test_scores = line_max[by_split["test"]["idxs"]]
        dev_labels = by_split["dev"]["labels"]
        test_labels = by_split["test"]["labels"]
        scored: list[dict[str, Any]] = []
        for feature in top_features:
            train_auc = float(train_auc_all[feature])
            dev_auc = auc_for_feature(dev_scores, dev_labels, feature)
            fire = float(enc["fire_count"][feature]) / max(enc["n_tokens"], 1)
            ubiq_penalty = min(fire * 2.0, 0.5)
            score = dev_auc + 0.25 * train_auc - ubiq_penalty
            scored.append({
                "family": family,
                "label_type": label_type,
                "feature": int(feature),
                "train_auc": round(train_auc, 4),
                "dev_auc": round(dev_auc, 4),
                "selection_score": round(score, 4),
                "fire_fraction": round(fire, 6),
                "selected_for_test": False,
                "test_auc": "",
                "claim_grade": "",
            })
        scored.sort(key=lambda r: (_safe_float(r["selection_score"]), _safe_float(r["dev_auc"])), reverse=True)
        selected = dict(scored[0])
        selected["selected_for_test"] = True
        feature = int(selected["feature"])

        test_vals = test_scores[:, feature].tolist()
        test_pos = [float(v) for v, y in zip(test_vals, test_labels) if y]
        test_neg = [float(v) for v, y in zip(test_vals, test_labels) if not y]
        ci = bootstrap_auc_ci(test_pos, test_neg, seed + feature)
        null = permutation_auc_null(test_pos, test_neg, seed + 17 + feature)
        stability = subset_stability_auc(test_pos, test_neg, seed + 31 + feature)

        group = next((corpus[i].get("hard_negative_group", "") for i in by_split["test"]["idxs"] if family_of(corpus[i]) == family), "")
        conf_neg_idx = [
            i for i in by_split["test"]["idxs"]
            if group and corpus[i].get("hard_negative_group", "") == group and family_of(corpus[i]) != family
        ]
        conf_pos_idx = [i for i in by_split["test"]["idxs"] if family_of(corpus[i]) == family]
        conf_auc = ""
        if len(conf_pos_idx) >= MIN_SPLIT_POSITIVES and len(conf_neg_idx) >= MIN_SPLIT_POSITIVES:
            conf_auc = round(roc_auc(
                [float(line_max[i, feature]) for i in conf_pos_idx],
                [float(line_max[i, feature]) for i in conf_neg_idx],
            ), 4)

        top_test = sorted(by_split["test"]["idxs"], key=lambda i: float(line_max[i, feature]), reverse=True)[:20]
        top_families = [family_of(corpus[i]) for i in top_test]
        purity = top_families.count(family) / max(len(top_families), 1)
        entropy = family_entropy(top_families)

        selected.update({
            "status": "ok",
            "train_pos": by_split["train"]["n_pos"],
            "train_neg": by_split["train"]["n_neg"],
            "dev_pos": by_split["dev"]["n_pos"],
            "dev_neg": by_split["dev"]["n_neg"],
            "test_pos": by_split["test"]["n_pos"],
            "test_neg": by_split["test"]["n_neg"],
            "test_auc": round(ci["auc"], 4),
            "test_auc_ci_low": ci["ci_low"],
            "test_auc_ci_high": ci["ci_high"],
            "test_confusable_auc": conf_auc,
            "test_top20_purity": round(purity, 4),
            "test_top20_family_entropy": round(entropy, 4),
            "permutation_null_mean": null["null_mean"],
            "permutation_p_ge_observed": null["p_ge_observed"],
            "subset_auc_mean": stability["subset_auc_mean"],
            "subset_auc_std": stability["subset_auc_std"],
            "hard_negative_group": group,
        })
        selected["claim_grade"] = targeted_grade(selected)
        best_rows.append(selected)
        write_targeted_feature_card(ctx, corpus, enc, selected)

        for i, row in enumerate(scored[:TARGETED_REPORT_TOPK]):
            if int(row["feature"]) == feature:
                row.update(selected)
            candidate_rows.append({"candidate_rank": i + 1, **row})

    bench.write_csv_with_context(ctx, ctx.path("tables", "targeted_feature_search.csv"), candidate_rows)
    ctx.register_artifact(ctx.path("tables", "targeted_feature_search.csv"), "table",
                          "Targeted fair-shot feature candidates: train discovery, dev selection, test only for selected features.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "best_feature_per_family.csv"), best_rows)
    ctx.register_artifact(ctx.path("tables", "best_feature_per_family.csv"), "table",
                          "Best selected targeted feature per corpus family with split-aware validation and claim grade.")
    return best_rows


def write_lab8_synthesis_tables(ctx, atlas, enc, ranks, toy, tc_report, bridge, clamp) -> None:
    """Write the tables that make the plots auditable, not just ornamental."""
    import numpy as np
    import torch

    evidence = feature_evidence_rows(atlas)
    bench.write_csv_with_context(ctx, ctx.path("tables", "feature_evidence_matrix.csv"), evidence)
    ctx.register_artifact(ctx.path("tables", "feature_evidence_matrix.csv"), "table",
                          "Joined feature evidence: label validation, confusable checks, sparsity, polysemy, and concept score.")

    dom = domain_validation_summary(atlas)
    bench.write_csv_with_context(ctx, ctx.path("tables", "domain_validation_summary.csv"), dom)
    ctx.register_artifact(ctx.path("tables", "domain_validation_summary.csv"), "table",
                          "Per-label domain summary of atlas verdicts and validation scores.")

    maxv = enc["line_max"].max(0).values.detach().cpu()
    freq = (enc["fire_count"] / max(enc["n_tokens"], 1)).detach().cpu()
    qs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    rows = []
    for q in qs:
        rows.append({
            "quantile": q,
            "feature_peak_activation": round(float(torch.quantile(maxv, q)), 5),
            "feature_fire_fraction": round(float(torch.quantile(freq, q)), 8),
        })
    bench.write_csv_with_context(ctx, ctx.path("tables", "feature_activity_distribution.csv"), rows)
    ctx.register_artifact(ctx.path("tables", "feature_activity_distribution.csv"), "table",
                          "Feature-level activity quantiles: peak activation and firing frequency.")

    if tc_report.get("inspected_features"):
        tc_rows = []
        for r in tc_report["inspected_features"]:
            tc_rows.append({
                "feature": r.get("feature"),
                "peak_activation": r.get("peak_activation"),
                "promotes_tokens": ", ".join(r.get("promotes_tokens", [])),
            })
        bench.write_csv_with_context(ctx, ctx.path("tables", "transcoder_feature_promotes.csv"), tc_rows)
        ctx.register_artifact(ctx.path("tables", "transcoder_feature_promotes.csv"), "table",
                              "De-embedded transcoder output features and promoted tokens.")

    if bridge.get("top_cosines"):
        bench.write_csv_with_context(ctx, ctx.path("tables", "truth_bridge_feature_cosines.csv"), bridge["top_cosines"])
        ctx.register_artifact(ctx.path("tables", "truth_bridge_feature_cosines.csv"), "table",
                              "Top SAE decoder directions by absolute cosine with the saved Lab 4 truth direction.")

    if clamp:
        real_base = next((r for r in clamp["rows"] if r["condition"] == "real" and float(r["clamp_mult"]) == 0.0), None)
        base_hits = real_base["domain_keyword_hits"] if real_base else 0
        random_by_dose = {float(r["clamp_mult"]): r for r in clamp["rows"] if r["condition"] == "random"}
        clamp_rows = []
        for r in clamp["rows"]:
            same_random = random_by_dose.get(float(r["clamp_mult"]))
            random_hits = same_random["domain_keyword_hits"] if same_random is not None else ""
            gap = r["domain_keyword_hits"] - random_hits if same_random is not None and r["condition"] == "real" else ""
            gain = r["domain_keyword_hits"] - base_hits if r["condition"] == "real" else ""
            clamp_rows.append({
                "condition": r["condition"],
                "clamp_mult": r["clamp_mult"],
                "domain_keyword_hits": r["domain_keyword_hits"],
                "real_gain_vs_base": gain,
                "random_hits_same_dose": random_hits if r["condition"] == "real" else "",
                "real_minus_random_same_dose": gap,
                "distinct_ratio": r["distinct_ratio"],
                "claimable_window": bool(r["condition"] == "real" and gain != "" and gain > 0 and _safe_float(r["distinct_ratio"], 0.0) >= 0.4 and (gap == "" or gap > 0)),
                "sample": str(r.get("sample", ""))[:260],
            })
        bench.write_csv_with_context(ctx, ctx.path("tables", "clamp_operating_points.csv"), clamp_rows)
        ctx.register_artifact(ctx.path("tables", "clamp_operating_points.csv"), "table",
                              "Feature-clamp operating points with target gain, same-dose random control gap, fluency proxy, and sample text.")

    plot_guide = [
        {"artifact": "plots/feature_evidence_dashboard.png", "question": "What is the whole Lab 8 evidence packet?", "look_for": "Toy geometry, SAE health, label verdicts, clamp or bridge status on one page."},
        {"artifact": "plots/feature_validation_matrix.png", "question": "Which labels earned their name?", "look_for": "Rows with high AUC, high purity, confusable separation, low polysemy, and sparse firing."},
        {"artifact": "plots/sae_activity_dashboard.png", "question": "Is this dictionary sparse and mostly unused on the corpus?", "look_for": "Long fire-frequency tail, silent mass, and atlas points outside the ordinary cloud."},
        {"artifact": "plots/domain_validation_summary.png", "question": "Which semantic domains are robust rather than lucky?", "look_for": "Survived/narrowed counts and domain-level median AUC, not a single heroic feature."},
        {"artifact": "plots/clamp_operating_window.png", "question": "Where does causal sufficiency become side-effect soup?", "look_for": "Real feature moves hits before distinct-ratio collapses; random control stays flat."},
        {"artifact": "plots/truth_bridge_feature_cosines.png", "question": "Is Lab 4 truth a single SAE atom?", "look_for": "Best absolute cosine and the tiny top-cosine distribution."},
        {"artifact": "plots/transcoder_feature_cards.png", "question": "What makes transcoders edge-ready for Lab 9?", "look_for": "FVU/KL plus de-embedded output-token tendencies."},
    ]
    bench.write_csv_with_context(ctx, ctx.path("tables", "plot_reading_guide.csv"), plot_guide)
    ctx.register_artifact(ctx.path("tables", "plot_reading_guide.csv"), "table",
                          "Plot-by-plot reading guide for Lab 8.")


def plot_superposition_phase_diagram(ctx, toy) -> None:
    import matplotlib.pyplot as plt

    if not toy or "sparsities" not in toy:
        return
    bench._ensure_plot_style()
    sparsities = [float(s) for s in toy["sparsities"]]
    represented = [toy["represented_by_sparsity"].get(str(s), toy["represented_by_sparsity"].get(f"{s:.2f}", 0)) for s in sparsities]
    interference = [toy["interference_by_sparsity"].get(str(s), toy["interference_by_sparsity"].get(f"{s:.2f}", 0)) for s in sparsities]
    fig, ax1 = plt.subplots(figsize=(8.4, 5.0))
    ax1.plot(sparsities, represented, marker="o", linewidth=2.2, label="features represented")
    ax1.axhline(toy.get("d_hidden", 0), color="black", linewidth=0.8, linestyle="--", alpha=0.8, label="hidden dimensions")
    ax1.set_xlabel("input sparsity")
    ax1.set_ylabel("represented features")
    ax2 = ax1.twinx()
    ax2.plot(sparsities, interference, marker="s", linewidth=1.8, color="#D55E00", label="mean nearest-neighbor interference")
    ax2.set_ylabel("interference")
    ax1.set_title("Toy phase diagram: more represented features are bought with interference")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    bench.save_figure(ctx, fig, "toy_superposition_phase_diagram.png",
                      "Toy superposition phase diagram: represented features and interference vs sparsity.")


def plot_feature_validation_matrix(ctx, atlas) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = feature_evidence_rows(atlas)
    if not rows:
        return
    top = rows[:min(len(rows), 28)]
    data = []
    labels = []
    for r in top:
        conf = _safe_float(r.get("confusable_auc"), 0.5)
        data.append([
            _safe_float(r.get("held_out_auc"), 0.5),
            conf,
            _safe_float(r.get("label_purity"), 0.0),
            _safe_float(r.get("polysemy_clean"), 0.0),
            _safe_float(r.get("rarity_clean"), 0.0),
        ])
        labels.append(f"F{r.get('feature')} {r.get('proposed_label')}\n{r.get('verdict')}")
    arr = np.array(data, dtype=float)
    bench._ensure_plot_style()
    fig, ax = plt.subplots(figsize=(9.6, max(5.2, 0.32 * len(top) + 1.8)))
    im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(len(FEATURE_EVIDENCE_COLUMNS)))
    ax.set_xticklabels(["held-out\nAUC", "confusable\nAUC", "label\npurity", "low\npolysemy", "rare\nfiring"], fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="white" if arr[i, j] < 0.45 else "black", fontsize=6.5)
    ax.set_title("Feature validation matrix: a label must survive several locks")
    fig.colorbar(im, ax=ax, fraction=0.035, label="better / cleaner")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "feature_validation_matrix.png",
                      "Feature-by-feature validation battery: AUC, confusable check, purity, polysemy, and firing rarity.")


def plot_sae_activity_dashboard(ctx, enc, atlas) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    maxv = enc["line_max"].max(0).values.detach().cpu()
    freq = (enc["fire_count"] / max(enc["n_tokens"], 1)).detach().cpu()
    # Keep plots fast and legible for 65k dictionaries.
    finite_max = maxv[torch.isfinite(maxv)].numpy()
    finite_freq = freq[torch.isfinite(freq)].numpy()
    bench._ensure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    axes = axes.flatten()
    axes[0].hist(np.log10(np.maximum(finite_max, 1e-8)), bins=60, alpha=0.85)
    axes[0].set_title("Peak activation distribution")
    axes[0].set_xlabel("log10 peak activation")
    axes[0].set_ylabel("features")
    axes[1].hist(np.log10(np.maximum(finite_freq, 1e-10)), bins=60, alpha=0.85)
    axes[1].set_title("Firing-frequency distribution")
    axes[1].set_xlabel("log10 fire fraction")
    axes[1].set_ylabel("features")
    sample = torch.argsort(maxv, descending=True)[:min(1200, len(maxv))]
    axes[2].scatter(maxv[sample], freq[sample], s=8, alpha=0.35, color="#777777", label="dictionary sample")
    for r in atlas:
        f = int(r["feature"])
        axes[2].scatter(float(maxv[f]), float(freq[f]), s=52, color=_verdict_color(r["verdict"]), edgecolor="black", linewidth=0.3)
    axes[2].set_xlabel("peak activation")
    axes[2].set_ylabel("fire fraction")
    axes[2].set_title("Atlas features inside the whole dictionary cloud")
    silent = float((freq <= 0).float().mean())
    active = 1.0 - silent
    axes[3].bar(["silent on corpus", "fires at least once"], [silent, active], color=[_verdict_color("silent-on-corpus"), "#0072B2"])
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("fraction of dictionary")
    axes[3].set_title("Silent ≠ dead: corpus coverage audit")
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("SAE activity dashboard: sparsity, tails, and atlas selection", fontsize=14)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "sae_activity_dashboard.png",
                      "Activity distribution of the SAE dictionary and where selected atlas features sit.")


def plot_domain_validation_summary(ctx, atlas) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = domain_validation_summary(atlas)
    if not rows:
        return
    labels = [r["proposed_label"] for r in rows]
    x = np.arange(len(rows))
    survived = np.array([r["n_survived"] for r in rows], dtype=float)
    narrowed = np.array([r["n_narrowed"] for r in rows], dtype=float)
    failed = np.array([r["n_token_feature"] + r["n_polysemantic"] + r["n_killed"] + r["n_silent"] for r in rows], dtype=float)
    auc = np.array([_safe_float(r["median_held_out_auc"], 0.5) for r in rows])
    bench._ensure_plot_style()
    fig, ax1 = plt.subplots(figsize=(10.6, 5.4))
    ax1.bar(x, survived, label="survived", color=_verdict_color("survived"))
    ax1.bar(x, narrowed, bottom=survived, label="narrowed", color=_verdict_color("narrowed"))
    ax1.bar(x, failed, bottom=survived + narrowed, label="failed / token / polysemantic", color=_verdict_color("killed"), alpha=0.75)
    ax1.set_ylabel("atlas features")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, auc, color="black", marker="o", linewidth=1.6, label="median held-out AUC")
    ax2.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("median held-out AUC")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.set_title("Domain validation summary: labels need population evidence, not one good context")
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "domain_validation_summary.png",
                      "Per-domain verdict counts and median held-out AUC across atlas features.")


def plot_clamp_operating_window(ctx, clamp) -> None:
    import matplotlib.pyplot as plt

    if not clamp:
        return
    bench._ensure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True)
    for cond in sorted({r["condition"] for r in clamp["rows"]}):
        rows = sorted([r for r in clamp["rows"] if r["condition"] == cond], key=lambda r: r["clamp_mult"])
        xs = [r["clamp_mult"] for r in rows]
        color = _clamp_color(cond)
        axes[0].plot(xs, [r["domain_keyword_hits"] for r in rows], marker="o", linewidth=2.0, color=color, label=cond)
        axes[1].plot(xs, [r["distinct_ratio"] for r in rows], marker="o", linewidth=2.0, color=color, label=cond)
    axes[0].axvline(clamp.get("best_mult", 0), color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[0].set_title(f"Target concept hits: feature {clamp['feature']} ({clamp['label']})")
    axes[0].set_ylabel("domain keyword hits")
    axes[1].axhline(0.4, color="black", linewidth=0.9, linestyle=":", alpha=0.8, label="fluency floor")
    axes[1].set_title("Side-effect guardrail: distinct-token ratio")
    axes[1].set_ylabel("distinct ratio")
    for ax in axes:
        ax.set_xlabel("dose (× observed peak activation)")
        ax.legend(fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Clamp operating window: causal handle before repetition soup")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "clamp_operating_window.png",
                      "Feature-clamp operating window: target concept movement versus fluency side effects.")


def plot_truth_bridge_cosines(ctx, bridge) -> None:
    import matplotlib.pyplot as plt

    if not bridge or not bridge.get("top_cosines"):
        return
    rows = bridge["top_cosines"]
    labels = [f"F{r['feature']}" for r in rows]
    vals = [r["cosine"] for r in rows]
    colors = ["#0072B2" if v >= 0 else "#D55E00" for v in vals]
    fig, ax = bench.new_figure(figsize=(9.0, 5.0))
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    q = bridge.get("abs_cosine_quantiles", {})
    subtitle = " | ".join(f"{k}={v}" for k, v in q.items()) if q else ""
    ax.set_title("Truth-direction bridge: best SAE atoms barely align" + (f"\nabs cosine quantiles: {subtitle}" if subtitle else ""))
    ax.set_ylabel("cosine with Lab 4 truth direction")
    ax.set_xlabel("top SAE decoder directions by |cosine|")
    ax.tick_params(axis="x", rotation=35)
    bench.save_figure(ctx, fig, "truth_bridge_feature_cosines.png",
                      "Top SAE feature decoder cosines with the saved Lab 4 truth direction.")


def plot_transcoder_feature_cards(ctx, tc_report) -> None:
    import matplotlib.pyplot as plt

    feats = tc_report.get("inspected_features", []) if tc_report else []
    if not feats:
        return
    fig, ax = bench.new_figure(figsize=(10.0, max(4.0, 1.2 * len(feats) + 1.5)))
    y = list(range(len(feats)))
    peaks = [_safe_float(f.get("peak_activation"), 0.0) for f in feats]
    ax.barh(y, peaks, color="#0072B2", alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"F{f.get('feature')}" for f in feats])
    ax.invert_yaxis()
    ax.set_xlabel("peak transcoder activation on corpus sample")
    ax.set_title(f"Transcoder features: reconstruct the MLP map, then de-embed outputs\nFVU {tc_report.get('fvu')} | splice KL {tc_report.get('mean_splice_kl')}")
    xmax = max(peaks + [1.0])
    for i, f in enumerate(feats):
        toks = ", ".join(str(t) for t in f.get("promotes_tokens", [])[:8])
        ax.text(xmax * 1.02, i, toks, va="center", fontsize=8)
    ax.set_xlim(0, xmax * 2.4)
    bench.save_figure(ctx, fig, "transcoder_feature_cards.png",
                      "Inspected transcoder features with de-embedded promoted tokens.")


def plot_feature_evidence_dashboard(ctx, atlas, enc, ranks, toy, tc_report, bridge, clamp) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    bench._ensure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    ax = axes[0, 0]
    sparsities = [float(s) for s in toy.get("sparsities", [])]
    represented = [toy.get("represented_by_sparsity", {}).get(str(s), toy.get("represented_by_sparsity", {}).get(f"{s:.2f}", 0)) for s in sparsities]
    interference = [toy.get("interference_by_sparsity", {}).get(str(s), toy.get("interference_by_sparsity", {}).get(f"{s:.2f}", 0)) for s in sparsities]
    ax.plot(sparsities, represented, marker="o", label="represented features")
    if toy.get("d_hidden") is not None:
        ax.axhline(toy.get("d_hidden"), color="black", linestyle="--", linewidth=0.8, alpha=0.7, label="hidden dims")
    ax.set_title("0. Why sparse codes overlap")
    ax.set_xlabel("input sparsity"); ax.set_ylabel("features represented")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    verdicts = Counter(str(r.get("verdict", "")) for r in atlas)
    counts = [verdicts.get(v, 0) for v in VERDICT_ORDER]
    ax.bar(range(len(VERDICT_ORDER)), counts, color=[_verdict_color(v) for v in VERDICT_ORDER])
    ax.set_xticks(range(len(VERDICT_ORDER))); ax.set_xticklabels(VERDICT_ORDER, rotation=35, ha="right")
    ax.set_ylabel("features")
    ax.set_title("1. Label validation verdicts")

    ax = axes[1, 0]
    maxv = enc["line_max"].max(0).values.detach().cpu()
    freq = (enc["fire_count"] / max(enc["n_tokens"], 1)).detach().cpu()
    sample = torch.argsort(maxv, descending=True)[:min(800, len(maxv))]
    ax.scatter(maxv[sample], freq[sample], s=7, alpha=0.25, color="#777777")
    for r in atlas:
        f = int(r["feature"])
        ax.scatter(float(maxv[f]), float(freq[f]), s=55, color=_verdict_color(r["verdict"]), edgecolor="black", linewidth=0.25)
    ax.set_xlabel("peak activation")
    ax.set_ylabel("fire fraction")
    ax.set_title("2. Atlas points live in a skewed dictionary")

    ax = axes[1, 1]
    ax.axis("off")
    silent = float((freq <= 0).float().mean())
    bridge_text = "not available"
    if bridge.get("best_cosine") is not None:
        bridge_text = f"best truth cosine {bridge.get('best_cosine')} (F{bridge.get('best_feature')})"
    elif bridge.get("note"):
        bridge_text = str(bridge.get("note"))[:72]
    clamp_text = "clamp skipped"
    if clamp:
        clamp_text = f"clamp F{clamp['feature']} {clamp['label']}: {clamp['real_base_hits']}→{clamp['real_max_hits']} hits, random {clamp['random_max_hits']}"
    summary = [
        "3. Evidence packet",
        f"SAE FVU: {enc.get('fvu', float('nan')):.3f}",
        f"per-token L0: {enc.get('per_token_l0', 'n/a')}",
        f"silent on this corpus: {silent * 100:.1f}%",
        f"ranking overlap top-N: {len(set(ranks.get('by_max', [])) & set(ranks.get('by_freq', [])))}",
        f"transcoder FVU: {tc_report.get('fvu')} | splice KL: {tc_report.get('mean_splice_kl')}",
        bridge_text,
        clamp_text,
        "Claim grammar: OBS for reconstruction, DECODE for labels, CAUSAL only for clamp.",
    ]
    y = 0.96
    for i, line in enumerate(summary):
        size = 13 if i == 0 else 9.5
        weight = "bold" if i == 0 else "normal"
        ax.text(0.02, y, line, transform=ax.transAxes, fontsize=size, fontweight=weight, va="top")
        y -= 0.105 if i == 0 else 0.095

    for ax in axes.flatten()[:3]:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Lab 8 feature evidence dashboard: sparse geometry → labels → causality", fontsize=15)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "feature_evidence_dashboard.png",
                      "Lab 8 dashboard combining toy geometry, SAE activity, atlas verdicts, transcoder, truth bridge, and clamp summary.")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_ranking_disagreement(ctx, enc, ranks) -> None:
    import torch

    fig, ax = bench.new_figure(figsize=(8.8, 5.6))
    maxv = enc["line_max"].max(0).values.detach().cpu()
    freq = (enc["fire_count"] / max(enc["n_tokens"], 1)).detach().cpu()
    sample = torch.argsort(maxv, descending=True)[:min(900, len(maxv))]
    ax.scatter(maxv[sample], freq[sample], s=9, alpha=0.28, color="#777777", label="dictionary features")
    overlap = set(ranks["by_max"]) & set(ranks["by_freq"])
    for f in ranks["by_max"][:10]:
        ax.scatter(maxv[f], freq[f], s=70, color="#D55E00", marker="^", edgecolor="black", linewidth=0.3,
                   label="top by peak" if f == ranks["by_max"][0] else None)
    for f in ranks["by_freq"][:10]:
        ax.scatter(maxv[f], freq[f], s=70, color="#009E73", marker="s", edgecolor="black", linewidth=0.3,
                   label="top by frequency" if f == ranks["by_freq"][0] else None)
    first_overlap = next(iter(overlap), None)
    for f in overlap:
        ax.scatter(maxv[f], freq[f], s=130, facecolors="none", edgecolors="black", linewidth=1.3,
                   label="overlap" if f == first_overlap else None)
    ax.set_xlabel("peak activation (max-activation ranking →)")
    ax.set_ylabel("firing frequency (frequency ranking ↑)")
    ax.set_title(f"Two rankings disagree: peak events vs workhorse features (overlap={len(overlap)})")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "ranking_disagreement.png",
                      "Peak activation vs firing frequency; the top features by each metric differ.")


def plot_atlas_verdicts(ctx, atlas) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, 5.2))
    verdicts = [r["verdict"] for r in atlas]
    order = list(VERDICT_ORDER)
    counts = [verdicts.count(v) for v in order]
    bars = ax.bar(order, counts, color=[_verdict_color(v) for v in order])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, str(c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("features")
    ax.set_title("Label-validation verdicts across the atlas: killed labels are evidence, not embarrassment")
    ax.tick_params(axis="x", rotation=30)
    bench.save_figure(ctx, fig, "atlas_verdicts.png", "Distribution of validation verdicts.")


def plot_clamp(ctx, clamp) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, 5.2))
    for cond in ("real", "random"):
        pts = sorted((r["clamp_mult"], r["domain_keyword_hits"]) for r in clamp["rows"] if r["condition"] == cond)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=_clamp_color(cond), linewidth=2.2,
                label=f"{cond} feature")
    ax.axvline(clamp.get("best_mult", 0), color="black", linestyle="--", linewidth=0.8, alpha=0.7, label="chosen fluent dose")
    ax.set_xlabel("clamp dose (× feature peak activation)")
    ax.set_ylabel(f"'{clamp['label']}' keyword hits in generations")
    ax.set_title(f"Feature clamp (CAUSAL): feature {clamp['feature']} vs random control")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "feature_clamp.png",
                      "Domain-keyword push under clamping the validated feature vs a random feature.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    device = bundle.input_device

    # ----- Part 0: toy superposition (CPU, model-free) -------------------------
    print("[lab8] Part 0: toy model of superposition")
    toy = run_toy_superposition(ctx, seed=args.seed)

    # ----- Part 1: feature atlas ----------------------------------------------
    print("[lab8] Part 1: feature atlas (SAE feature interpretation + label validation)")
    corpus = load_corpus(getattr(args, "corpus_path", "") or None)
    sae_spec = resolve_sae_spec(args, bundle)
    weights = _download(sae_spec)
    provisional_sae = load_model_sae(bundle, sae_spec, weights=weights)
    calibration_report, _ = calibrate_sae_loading(ctx, bundle, sae_spec, weights, corpus)
    sae = load_model_sae(bundle, sae_spec, weights=weights).to(device)
    # Update run_config.json with the exact resolved SAE spec after registry/default
    # resolution and any custom auto-convention calibration. The bench wrote the
    # CLI config before the lab-specific registry could be consulted.
    run_config = {**vars(args), "resolved_sae_spec": sae_spec}
    bench.write_json(ctx.path("run_config.json"), run_config)
    loading_report = {
        "model_id": bundle.anatomy.model_id,
        "model_d_model": bundle.anatomy.d_model,
        "sae_spec": sae_spec,
        "weight_keys": sorted(weights.keys()),
        "d_in": sae.d_in,
        "d_sae": sae.d_sae,
        "threshold_present": sae.threshold is not None,
        "dimensionality_ok": sae.d_in == bundle.anatomy.d_model,
        "calibration_chosen": calibration_report["chosen"],
        "calibration_best_by_fvu": calibration_report["best_by_fvu"],
    }
    bench.write_json(ctx.path("sae_loading_report.json"), loading_report)
    ctx.register_artifact(ctx.path("sae_loading_report.json"), "diagnostic",
                          "Exact SAE spec, loaded tensor keys, dimensions, compatibility, and chosen calibration.")
    n_single = sum(1 for r in corpus if "+" not in r["domain"])
    print(f"[lab8]   corpus: {len(corpus)} lines ({n_single} single-domain) at SAE layer {sae.layer}")
    print(f"[lab8]   loading calibration: chosen FVU={calibration_report['chosen']['fvu']} "
          f"L0={calibration_report['chosen']['mean_l0']} best FVU={calibration_report['best_by_fvu']['fvu']}")

    enc = encode_corpus(bundle, sae, corpus)
    enc["per_token_l0"] = round(per_token_l0(bundle, sae, corpus), 2)
    fvu = enc["fvu"]
    print(f"[lab8]   SAE reconstruction FVU={fvu:.4f}  L0(per-token)≈{enc['per_token_l0']}")

    atlas_budget = int(getattr(args, "atlas_budget", 0) or N_ATLAS_FEATURES)
    ranks = rank_features(enc, atlas_budget)
    overlap = len(set(ranks["by_max"]) & set(ranks["by_freq"]))
    # atlas feature set: top by max-activation, padded with top-frequency ones
    atlas_features: list[int] = list(ranks["by_max"])
    for f in ranks["by_freq"]:
        if len(atlas_features) >= atlas_budget + 5:
            break
        if f not in atlas_features:
            atlas_features.append(f)

    dead_corpus = int((enc["fire_count"] == 0).sum())
    print(f"[lab8]   ranking overlap (max vs freq, top {atlas_budget}): {overlap}; "
          f"features silent on corpus: {dead_corpus}/{sae.d_sae}")

    atlas = [propose_and_validate(bundle, sae, corpus, enc, f) for f in atlas_features]
    verdict_counts = {v: sum(1 for r in atlas if r["verdict"] == v) for r in atlas for v in [r["verdict"]]}
    print(f"[lab8]   atlas verdicts: {verdict_counts}")

    # feature_rankings.csv
    rank_rows = []
    maxv = enc["line_max"].max(0).values
    freq = enc["fire_count"] / enc["n_tokens"]
    for rank, f in enumerate(ranks["by_max"]):
        rank_rows.append({"rank": rank, "metric": "max_activation", "feature": f,
                          "peak_activation": round(float(maxv[f]), 4),
                          "fire_fraction": round(float(freq[f]), 5)})
    for rank, f in enumerate(ranks["by_freq"]):
        rank_rows.append({"rank": rank, "metric": "fire_frequency", "feature": f,
                          "peak_activation": round(float(maxv[f]), 4),
                          "fire_fraction": round(float(freq[f]), 5)})
    bench.write_csv_with_context(ctx, ctx.path("tables", "feature_rankings.csv"), rank_rows)
    ctx.register_artifact(ctx.path("tables", "feature_rankings.csv"), "table",
                          "Top features by peak activation and by firing frequency (the two disagree).")

    # atlas table (flat) + results alias
    atlas_rows = [{k: v for k, v in r.items() if k != "evidence"} for r in atlas]
    bench.write_csv_with_context(ctx, ctx.path("tables", "feature_atlas.csv"), atlas_rows)
    ctx.register_artifact(ctx.path("tables", "feature_atlas.csv"), "table",
                          "Per-feature label, validation metrics, and verdict.")
    bench.write_csv_with_context(ctx, ctx.path("results.csv"), atlas_rows)
    ctx.register_artifact(ctx.path("results.csv"), "results", "Alias of feature_atlas.csv for the run contract.")

    targeted_rows: list[dict[str, Any]] = []
    if getattr(args, "feature_search", "blind") in {"targeted", "both"}:
        print("[lab8] Part 1b: targeted fair-shot feature search (train/dev/test)")
        targeted_rows = targeted_feature_search(ctx, corpus, enc, args.seed)
        targeted_counts = Counter(str(r.get("claim_grade", "")) for r in targeted_rows)
        print(f"[lab8]   targeted claim grades: {dict(targeted_counts)}")

    dead_stats = {
        "d_sae": sae.d_sae, "n_tokens": enc["n_tokens"],
        "silent_on_corpus": dead_corpus,
        "silent_fraction": round(dead_corpus / sae.d_sae, 4),
        "per_token_l0": enc["per_token_l0"],
        "reconstruction_fvu": round(fvu, 4),
        "ranking_overlap_topN": overlap,
    }
    bench.write_json(ctx.path("dead_feature_stats.json"), dead_stats)
    ctx.register_artifact(ctx.path("dead_feature_stats.json"), "metrics",
                          "Dead/silent feature counts, L0, FVU, ranking overlap.")

    # ----- Part 2: transcoder (gpt2) ------------------------------------------
    if getattr(args, "skip_transcoder", False):
        print("[lab8] Part 2: transcoder skipped by --skip-transcoder")
        tc_report = {
            "skipped": True,
            "reason": "--skip-transcoder",
            "model": "skipped",
            "fvu": "",
            "mean_l0": "",
            "mean_splice_kl": "",
            "max_splice_kl": "",
            "inspected_features": [],
        }
        bench.write_json(ctx.path("transcoder_reconstruction_report.json"), tc_report)
        ctx.register_artifact(ctx.path("transcoder_reconstruction_report.json"), "metrics",
                              "Transcoder skipped for SAE-focused sweep.")
    else:
        print("[lab8] Part 2: transcoder reconstruction + feature inspection (gpt2)")
        tc = load_gpt2_transcoder()
        tc_model, tc_tok, tc_note = get_gpt2_for_transcoder(bundle)
        tc_report = verify_transcoder(tc_model, tc_tok, tc, corpus)
        tc_report["model"] = tc_note
        tc_features = inspect_transcoder_features(tc_model, tc_tok, tc, corpus)
        tc_report["inspected_features"] = tc_features
        bench.write_json(ctx.path("transcoder_reconstruction_report.json"), tc_report)
        ctx.register_artifact(ctx.path("transcoder_reconstruction_report.json"), "metrics",
                              "Transcoder FVU, splice-in KL, and de-embedded features.")
        print(f"[lab8]   transcoder FVU={tc_report['fvu']} mean_splice_KL={tc_report['mean_splice_kl']} "
              f"L0={tc_report['mean_l0']}")

    # ----- Bridge: SAE feature vs Lab 4 truth direction ------------------------
    bridge = truth_direction_bridge(sae, bundle)
    print(f"[lab8]   truth-direction bridge: {bridge.get('best_cosine', bridge.get('note'))}")

    # ----- Causal extension: clamp a validated feature -------------------------
    # Pick the cleanest CONCEPT feature to clamp: validated, with a keyword
    # battery, and — critically — low firing frequency. A feature that fires on
    # 85% of tokens is a broadly-active basis vector, not a concept handle;
    # clamping it just degrades fluency. concept_score rewards domain AUC and
    # penalizes ubiquity, which is exactly what makes a feature steerable.
    survivors = [r for r in atlas
                 if r["verdict"] in ("survived", "narrowed") and r["proposed_label"] in DOMAIN_KEYWORDS]
    clamp = None
    if survivors:
        chosen = max(survivors, key=lambda r: r["held_out_auc"] - r["fire_fraction"])
        print(f"[lab8]   causal clamp: feature {chosen['feature']} ('{chosen['proposed_label']}', "
              f"AUC {chosen['held_out_auc']}, fires {chosen['fire_fraction']*100:.2f}%)")
        clamp = clamp_feature_steering(bundle, sae, corpus, chosen["feature"], chosen["proposed_label"],
                                       chosen["max_activation"], args.seed)
        bench.write_csv_with_context(ctx, ctx.path("tables", "feature_clamp.csv"), clamp["rows"])
        ctx.register_artifact(ctx.path("tables", "feature_clamp.csv"), "table",
                              "Feature-clamp dose vs domain-keyword hits, real feature vs random control.")
        print(f"[lab8]   clamp causal={clamp['causal']} (real {clamp['real_base_hits']}→{clamp['real_max_hits']} "
              f"at {clamp['best_mult']}× peak, random max {clamp['random_max_hits']})")
    else:
        print("[lab8]   causal clamp: no survivor with a keyword battery; extension skipped")

    causal_suite = None
    if getattr(args, "causal_suite", False):
        candidate = choose_causal_candidate(targeted_rows, atlas)
        if candidate is None:
            print("[lab8]   causal suite: no validated candidate available; skipped")
        else:
            feature, family = candidate
            print(f"[lab8]   causal suite: feature {feature} ({family}), "
                  f"{getattr(args, 'causal_prompts', 20)} prompts, {getattr(args, 'causal_controls', 10)} controls")
            causal_suite = enhanced_causal_feature_test(
                ctx, bundle, sae, corpus, enc, feature, family, args.seed,
                n_prompts=int(getattr(args, "causal_prompts", 20)),
                n_controls=int(getattr(args, "causal_controls", 10)),
            )
            print(f"[lab8]   causal suite passed={causal_suite['causal']} "
                  f"(real probe {causal_suite['real_base_probe']}→{causal_suite['real_best_probe']}, "
                  f"control max {causal_suite['control_max_probe_same_dose']})")

    # ----- synthesis tables + plots --------------------------------------------
    write_lab8_synthesis_tables(ctx, atlas, enc, ranks, toy, tc_report, bridge, clamp)
    if not args.no_plots:
        plot_superposition_phase_diagram(ctx, toy)
        plot_ranking_disagreement(ctx, enc, ranks)
        plot_atlas_verdicts(ctx, atlas)
        plot_feature_validation_matrix(ctx, atlas)
        plot_sae_activity_dashboard(ctx, enc, atlas)
        plot_domain_validation_summary(ctx, atlas)
        plot_feature_evidence_dashboard(ctx, atlas, enc, ranks, toy, tc_report, bridge, clamp)
        plot_truth_bridge_cosines(ctx, bridge)
        plot_transcoder_feature_cards(ctx, tc_report)
        if clamp is not None:
            plot_clamp(ctx, clamp)
            plot_clamp_operating_window(ctx, clamp)

    # ----- metrics, atlas, claims, summary -------------------------------------
    survived = [r for r in atlas if r["verdict"] == "survived"]
    killed = [r for r in atlas if r["verdict"] in ("killed", "token-feature", "polysemantic", "silent-on-corpus")]
    metrics = {
        "model_id": bundle.anatomy.model_id,
        "sae_layer": sae.layer, "sae_d_sae": sae.d_sae,
        "reconstruction_fvu": round(fvu, 4), "per_token_l0": enc["per_token_l0"],
        "silent_feature_fraction": round(dead_corpus / sae.d_sae, 4),
        "ranking_overlap_topN": overlap,
        "atlas_size": len(atlas), "n_survived": len(survived), "n_killed": len(killed),
        "sae_spec": sae_spec,
        "sae_loading_chosen_fvu": calibration_report["chosen"]["fvu"],
        "sae_loading_best_fvu": calibration_report["best_by_fvu"]["fvu"],
        "feature_search": getattr(args, "feature_search", "blind"),
        "targeted_search_families": len(targeted_rows),
        "targeted_claim_grades": dict(Counter(str(r.get("claim_grade", "")) for r in targeted_rows)),
        "causal_suite": causal_suite,
        "toy_represented_dense": toy["represented_by_sparsity"][str(toy["sparsities"][0])],
        "toy_represented_sparse": toy["represented_by_sparsity"][str(toy["sparsities"][-1])],
        "transcoder_fvu": tc_report["fvu"], "transcoder_splice_kl": tc_report["mean_splice_kl"],
        "truth_bridge": bridge,
        "clamp_causal": None if clamp is None else clamp["causal"],
    }
    bench.write_json(ctx.path("metrics.json"), metrics)
    ctx.register_artifact(ctx.path("metrics.json"), "metrics", "Aggregate Lab 8 metrics.")

    write_feature_atlas(ctx, bundle, sae, atlas, enc, toy, tc_report, bridge, clamp)
    claims = build_claims(ctx, bundle, sae, atlas, enc, toy, tc_report, clamp, survived, killed)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    write_summary(ctx, bundle, metrics, atlas, toy, tc_report, bridge, clamp, claims)
    print(f"[lab8] wrote feature_atlas.md, run_summary.md, and {len(claims)} drafted ledger claims")


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------


def build_claims(ctx, bundle, sae, atlas, enc, toy, tc_report, clamp, survived, killed) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    validish = [r for r in atlas if r.get("verdict") in ("survived", "narrowed")]
    best = max(validish or atlas, key=lambda r: _safe_float(r.get("held_out_auc"), 0.5)) if atlas else None
    claims = [
        {
            "id": f"{LAB_ID}-C1", "tag": "OBS",
            "text": (
                f"A sparse autoencoder at layer {sae.layer} of {bundle.anatomy.model_id} reconstructs "
                f"its activations at FVU {enc['fvu']:.3f} with ~{enc['per_token_l0']} active features per "
                f"token out of {sae.d_sae}, and {int((enc['fire_count']==0).sum())} features stay silent on "
                f"the {enc['n_tokens']}-token corpus — superposition made into a usable, sparse code."
            ),
            "artifact": f"runs/{run_name}/dead_feature_stats.json",
            "falsifier": "FVU is no better than reconstructing from the same number of random directions, or L0≈d_sae (no sparsity).",
        },
    ]
    if best is not None:
        claims.append({
            "id": f"{LAB_ID}-C2", "tag": "DECODE",
            "text": (
                f"SAE feature {best['feature']} is labeled '{best['proposed_label']}' and the label {str(best['verdict']).upper()} "
                f"under validation: held-out AUC {_safe_float(best['held_out_auc'], 0.5):.2f} against domain membership"
                + (f", and it separates the confusable pair '{best['proposed_label']}' vs "
                   f"'{best['confusable_with']}' at AUC {best['confusable_auc']} (a concept feature, not a "
                   f"token feature)" if best['confusable_with'] else "")
                + f". Of {len(atlas)} labeled features, {len(survived)} survived and {len(killed)} were killed "
                  f"by the same battery."
            ),
            "artifact": f"runs/{run_name}/feature_atlas.md",
            "falsifier": "On a fresh corpus the held-out AUC collapses, or the label fires equally on the confusable domain — it tracked a token.",
        })
    if clamp is not None and clamp["causal"]:
        claims.append({
            "id": f"{LAB_ID}-C3", "tag": "CAUSAL",
            "text": (
                f"Clamping validated feature {clamp['feature']} ('{clamp['label']}') along its decoder "
                f"direction at {clamp['best_mult']}× its peak activation pushes generations toward "
                f"{clamp['label']} vocabulary ({clamp['real_base_hits']}→{clamp['real_max_hits']} keyword hits) "
                f"while staying fluent, where a random feature's direction reaches only "
                f"{clamp['random_max_hits']} — the feature is causally sufficient to move the behavior, not "
                f"just decodable. Past ~3× peak the clamp collapses generation into repetition (see the CSV)."
            ),
            "artifact": f"runs/{run_name}/plots/feature_clamp.png",
            "falsifier": "The random-feature control matches the clamped feature — the effect was generic perturbation, not this feature.",
        })
    if not tc_report.get("skipped"):
        claims.append({
            "id": f"{LAB_ID}-C{len(claims)+1}", "tag": "OBS",
            "text": (
                f"A gpt2 MLP transcoder reconstructs the layer-{tc_report.get('model') and GPT2_TRANSCODER['layer']} "
                f"MLP's input→output map at FVU {tc_report['fvu']}, and splicing its reconstruction in for the real "
                f"MLP output shifts next-token logits by only KL {tc_report['mean_splice_kl']} on average — it "
                f"reconstructs the computation, not just a snapshot, which is what Lab 9's circuit tracing needs."
            ),
            "artifact": f"runs/{run_name}/transcoder_reconstruction_report.json",
            "falsifier": "Splice-in KL is large — the transcoder reconstructs the vector but breaks the downstream computation.",
        })
    return claims


def write_feature_atlas(ctx, bundle, sae, atlas, enc, toy, tc_report, bridge, clamp) -> None:
    lines = [
        "# Feature atlas",
        "",
        f"- **Model:** `{bundle.anatomy.model_id}` | SAE layer {sae.layer} ({sae.site}) | d_sae {sae.d_sae}",
        f"- **Reconstruction:** FVU {enc['fvu']:.4f}, per-token L0 ≈ {enc['per_token_l0']}, "
        f"{int((enc['fire_count']==0).sum())}/{sae.d_sae} features silent on the corpus",
        f"- **Run:** `{ctx.run_dir.name}`",
        "",
        "Each row is a proposed label and the verdict of an automated validation battery:",
        "held-out AUC against domain membership (label excluded from its own test set), an",
        "adversarial confusable-pair check (concept feature vs token feature), and a",
        "polysemanticity entropy over the top contexts. Verdicts: **survived** (AUC≥0.85, pure",
        "label), **narrowed** (AUC≥0.7), **token-feature** (fails the confusable pair),",
        "**polysemantic** (top contexts span unrelated domains), **killed**, **silent-on-corpus**.",
        "",
    ]
    for r in atlas:
        lines.append(f"## Feature {r['feature']} — “{r['proposed_label']}” → **{r['verdict']}**")
        lines.append("")
        conf = (f"confusable vs `{r['confusable_with']}` AUC {r['confusable_auc']} "
                f"(margin {r['confusable_margin']})" if r["confusable_with"] else "no confusable pair")
        lines.append(f"- held-out AUC **{r['held_out_auc']}**, label purity {r['label_purity']}, "
                     f"peak act {r['max_activation']}, fires on {r['fire_fraction']*100:.2f}% of tokens")
        lines.append(f"- {conf}; polysemy entropy {r['polysemy_entropy_bits']} bits; "
                     f"top domains: {r['top_domains']}")
        for ev in r["evidence"]:
            lines.append(f"  - `{ev['text_id']}` [{ev['domain']}] act {ev['peak_act']}: {ev['highlight']}")
        lines.append("")
    lines += [
        "## Bridges",
        "",
        f"- **Lab 4 truth direction:** {bridge.get('best_cosine', bridge.get('note', 'n/a'))}"
        + (f" (feature {bridge['best_feature']})" if bridge.get("best_feature") is not None else ""),
        "",
        "## What the atlas does NOT show",
        "",
        "- A surviving label means the feature *predicts* its domain on this corpus; only the clamped",
        "  feature carries a CAUSAL claim. Decodability is not use.",
        "- 'Silent on corpus' is not 'dead': the feature may fire richly on text this corpus never samples.",
        "- The validation is only as good as the corpus. A narrow corpus cannot tell 'fires on chemistry'",
        "  from 'fires on the word acid' — which is exactly why the confusable pairs are built in.",
        "",
    ]
    bench.write_text(ctx.path("feature_atlas.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("feature_atlas.md"), "summary",
                          "The feature atlas: labels, validation verdicts, evidence, and limits.")


def write_summary(ctx, bundle, metrics, atlas, toy, tc_report, bridge, clamp, claims) -> None:
    lines = [
        "# Lab 8 run summary: superposition, SAEs, and transcoders",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` (base model; SAE/transcoder are pretrained, pinned)",
        f"- SAE layer {metrics['sae_layer']}, d_sae {metrics['sae_d_sae']}; transcoder on gpt2",
        "- evidence level: OBS/DECODE at the feature level, CAUSAL for the one clamped feature",
        "",
        "## 1. Superposition, demonstrated (Part 0)",
        "",
        f"- toy model: {metrics['toy_represented_dense']} features represented when dense vs "
        f"{metrics['toy_represented_sparse']} when sparse, in only {toy['d_hidden']} dimensions — "
        "more features than dimensions, packed in superposition as sparsity rises.",
        "",
        "## 2. Feature atlas (Part 1)",
        "",
        f"- reconstruction FVU {metrics['reconstruction_fvu']}, per-token L0 ≈ {metrics['per_token_l0']}, "
        f"{metrics['silent_feature_fraction']*100:.1f}% of features silent on the corpus",
        f"- ranking overlap (max-activation vs frequency, top N): {metrics['ranking_overlap_topN']} — the two "
        "rankings surface largely different features (the disagreement is the lesson, not a bug)",
        f"- of {metrics['atlas_size']} labeled features, {metrics['n_survived']} survived validation and "
        f"{metrics['n_killed']} were killed (token-feature / polysemantic / low-AUC). The killed count is required; a clean sheet is a warning.",
    ]
    if metrics.get("targeted_search_families"):
        grades = metrics.get("targeted_claim_grades", {})
        grade_text = ", ".join(f"{k}={v}" for k, v in sorted(grades.items())) if grades else "none"
        lines += [
            "- targeted fair-shot search used train for discovery, dev for selection, and test for the selected",
            f"  feature per family across {metrics['targeted_search_families']} families; grades: {grade_text}",
        ]
    lines += [
        "",
        "## 3. Transcoder (Part 2)",
        "",
    ]
    if tc_report.get("skipped"):
        lines.append("- skipped by `--skip-transcoder` for the SAE fair-shot sweep.")
    else:
        lines += [
            f"- FVU {tc_report['fvu']}, per-token L0 {tc_report['mean_l0']}, mean splice-in KL "
            f"{tc_report['mean_splice_kl']} (max {tc_report['max_splice_kl']})",
            "- a transcoder reconstructs the MLP's input->output map, so its features can be de-embedded and",
            "  wired into a circuit - which is why Lab 9's tracing is built on transcoders, not site SAEs.",
        ]
    lines += [
        "",
        "## 4. Bridges and causal extension",
        "",
        f"- Lab 4 truth direction: {bridge.get('best_cosine', bridge.get('note', 'n/a'))}",
    ]
    if clamp is not None:
        lines.append(f"- feature clamp (CAUSAL): feature {clamp['feature']} ('{clamp['label']}') "
                     f"{clamp['real_base_hits']}→{clamp['real_max_hits']} keyword hits at "
                     f"{clamp['best_mult']}× peak vs random {clamp['random_max_hits']}; causal={clamp['causal']}")
    if metrics.get("causal_suite"):
        suite = metrics["causal_suite"]
        lines.append(f"- matched-control causal suite: feature {suite['feature']} ('{suite['family']}') "
                     f"probe {suite['real_base_probe']}→{suite['real_best_probe']} at {suite['best_dose_mult']}× peak; "
                     f"same-dose control max {suite['control_max_probe_same_dose']}; "
                     f"suppression {suite['suppress_base_probe']}→{suite['suppress_min_probe']}; causal={suite['causal']}")
    lines += [
        "",
        "## 5. Claims",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## 6. The reading order",
        "",
        "Diagnostics first, then the artifacts that make the distinctions visible.",
        "",
        "1. `diagnostics/model_anatomy.json` — confirm base model + layer; loading conventions matter.",
        "2. `feature_atlas.md` + `tables/feature_atlas.csv` — the deliverable. **Look for** the required",
        "   dead labels (in the reference gpt2 run, high-purity 'code' features with held-out AUC ~0.57",
        "   were the teaching case), the confusable-pair numbers that separate concept from token, and",
        "   the explicit 'What the atlas does NOT show' section. A clean sheet of 'survived' is a",
        "   warning sign.",
        "3. `plots/feature_evidence_dashboard.png` — the whole lab packet on one page: toy geometry,",
        "   SAE health, label verdicts, transcoder, truth bridge, and clamp status.",
        "4. `plots/feature_validation_matrix.png` + `tables/feature_evidence_matrix.csv` — the label",
        "   locks side by side: held-out AUC, confusable AUC, purity, low-polysemy score, and sparse firing.",
        "5. `plots/toy_superposition_geometry.png` and `plots/toy_superposition_phase_diagram.png` —",
        "   predict the geometry (dense: exactly d_hidden orthogonal; sparse: more features via",
        "   accepted interference) before you look.",
        "6. `plots/ranking_disagreement.png` + `tables/feature_rankings.csv` — **look for little or no",
        "   overlap** between the red (max-act, rare high-peak outliers) and green (freq, broad basis",
        "   vectors); the reference run had 0.",
        "7. `plots/sae_activity_dashboard.png` and `plots/domain_validation_summary.png` — separate",
        "   ordinary dictionary sparsity from the few features you are tempted to name.",
        "8. `plots/atlas_verdicts.png` — count the killed bar; the lab wants dead labels.",
        "9. `transcoder_reconstruction_report.json`, `plots/transcoder_feature_cards.png`, and",
        "   `tables/transcoder_feature_promotes.csv` — FVU + splice-in KL + de-embedded promoted tokens.",
        "10. `plots/truth_bridge_feature_cosines.png` — the Lab 4 truth direction is compared against",
        "    SAE decoder atoms instead of being assumed to be one feature.",
        "11. `plots/feature_clamp.png`, `plots/clamp_operating_window.png`, and `tables/feature_clamp.csv`",
        "    — the single CAUSAL claim. **Read the sample generations** at each dose (not just hits).",
        "    Expect a narrow window (reference run: induce ~1× peak, collapse by ~3×); random stays at",
        "    or near 0; the distinct ratio flags repetition.",
        "",
        "## 7. Caveats",
        "",
        "- Validation is corpus-bound: a label that survives here can die on different text. The confusable",
        "  pairs are the built-in guard against mistaking a token for a concept.",
        "- 'Silent on corpus' ≠ dead; most of the dictionary simply never gets the inputs that fire it here.",
        "- Decodability is not causality. Only the clamped, control-tested feature earns a CAUSAL tag.",
        "- The SAE conventions (centering, b_dec, jumprelu) are validated, not assumed; a wrong convention",
        "  inflates FVU silently. See the handout's debugging table.",
        "",
    ]
    bench.write_text(ctx.path("run_summary.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("run_summary.md"), "summary", "The seven standard questions answered.")
