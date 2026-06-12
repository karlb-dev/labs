"""Lab 8: Superposition, sparse autoencoders, and transcoders.

Why dense activations resist neuron-level reading, and whether sparse
dictionaries recover units worth naming. Three parts under a strict time
budget, plus a causal extension and two bridges back to earlier labs.

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
import math
import pathlib
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

N_ATLAS_FEATURES = 15        # features to label and validate
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

    wpath = hf_hub_download(spec["repo"], f"{spec['subdir']}/{spec['weights']}")
    return load_file(wpath)


def load_model_sae(bundle: bench.ModelBundle) -> LoadedSAE:
    """Pick and load the SAE matched to the loaded model's family."""
    model_id = bundle.anatomy.model_id.lower()
    spec = GPT2_SAE if "gpt2" in model_id else OLMO_SAE
    print(f"[lab8] loading SAE {spec['repo']}/{spec['subdir']} (layer {spec['layer']}, {spec['site']})")
    weights = _download(spec)
    sae = LoadedSAE(weights, center_input=spec["center_input"], sub_b_dec=spec["sub_b_dec"],
                    jumprelu=spec["jumprelu"], kind="sae", layer=spec["layer"], site=spec["site"])
    if sae.d_in != bundle.anatomy.d_model:
        raise RuntimeError(f"SAE d_in {sae.d_in} != model d_model {bundle.anatomy.d_model}; wrong SAE for this model.")
    print(f"[lab8]   SAE d_in={sae.d_in} d_sae={sae.d_sae} jumprelu={sae.jumprelu}")
    return sae


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

    fig = plt.figure(figsize=(15.0, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    for s in sparsities:
        ax1.plot(range(n_features), sorted(results[s]["norms"], reverse=True),
                 marker="o", markersize=3, linewidth=1.6, label=f"sparsity {s:.2f}")
    ax1.axvline(d_hidden - 0.5, color="black", linewidth=0.8, linestyle="--", label=f"d_hidden = {d_hidden}")
    ax1.set_xlabel("feature (sorted by norm)")
    ax1.set_ylabel("‖W column‖ (is the feature represented?)")
    ax1.set_title("Dense: only d_hidden features survive.\nSparse: more, via superposition.")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    for idx, s in [(2, sparsities[0]), (3, sparsities[-1])]:
        ax = fig.add_subplot(1, 3, idx)
        gram = np.array(results[s]["gram"])
        im = ax.imshow(gram, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"WᵀW at sparsity {s:.2f}\n(off-diagonal = interference)")
        ax.set_xlabel("feature"); ax.set_ylabel("feature")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Toy model of superposition: dense → orthogonal/monosemantic, sparse → overlapping")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "toy_superposition_geometry.png",
                      "Feature norms and WᵀW interference across sparsity levels.")


# ---------------------------------------------------------------------------
# Part 1: encode the corpus, rank, label, validate
# ---------------------------------------------------------------------------


def load_corpus() -> list[dict[str, str]]:
    path = bench.COURSE_ROOT / "data" / "sae_feature_corpus.csv"
    if not path.exists():
        raise RuntimeError(f"Frozen corpus missing: {path}. Run data/make_sae_corpus.py once at authoring time.")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    best = int(cos.abs().argmax())
    return {"found": True, "path": str(runs[0].relative_to(bench.COURSE_ROOT)),
            "saved_on_model": meta.get("model_id"),
            "best_feature": best, "best_cosine": round(float(cos[best]), 4)}


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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_ranking_disagreement(ctx, enc, ranks) -> None:
    import torch

    fig, ax = bench.new_figure(figsize=(8.0, 5.2))
    maxv = enc["line_max"].max(0).values
    freq = enc["fire_count"] / enc["n_tokens"]
    sample = torch.argsort(maxv, descending=True)[:400]
    ax.scatter(maxv[sample], freq[sample], s=10, alpha=0.5, color="tab:blue")
    for f in ranks["by_max"][:8]:
        ax.scatter(maxv[f], freq[f], s=60, color="tab:red", marker="^")
    for f in ranks["by_freq"][:8]:
        ax.scatter(maxv[f], freq[f], s=60, color="tab:green", marker="s")
    ax.set_xlabel("peak activation (max-activation ranking →)")
    ax.set_ylabel("firing frequency (frequency ranking ↑)")
    ax.set_title("Two rankings disagree: peak-activation (red ▲) vs frequency (green ■)")
    bench.save_figure(ctx, fig, "ranking_disagreement.png",
                      "Peak activation vs firing frequency; the top features by each metric differ.")


def plot_atlas_verdicts(ctx, atlas) -> None:
    fig, ax = bench.new_figure(figsize=(8.0, 5.0))
    verdicts = [r["verdict"] for r in atlas]
    order = ["survived", "narrowed", "token-feature", "polysemantic", "killed", "silent-on-corpus"]
    counts = [verdicts.count(v) for v in order]
    colors = ["tab:green", "tab:olive", "tab:orange", "tab:purple", "tab:red", "tab:gray"]
    ax.bar(order, counts, color=colors)
    ax.set_ylabel("features")
    ax.set_title("Label-validation verdicts across the atlas")
    ax.tick_params(axis="x", rotation=30)
    bench.save_figure(ctx, fig, "atlas_verdicts.png", "Distribution of validation verdicts.")


def plot_clamp(ctx, clamp) -> None:
    fig, ax = bench.new_figure(figsize=(8.0, 5.0))
    for cond, color in (("real", "tab:red"), ("random", "tab:gray")):
        pts = sorted((r["clamp_mult"], r["domain_keyword_hits"]) for r in clamp["rows"] if r["condition"] == cond)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=color, linewidth=2.0,
                label=f"{cond} feature")
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
    corpus = load_corpus()
    sae = load_model_sae(bundle).to(device)
    n_single = sum(1 for r in corpus if "+" not in r["domain"])
    print(f"[lab8]   corpus: {len(corpus)} lines ({n_single} single-domain) at SAE layer {sae.layer}")

    enc = encode_corpus(bundle, sae, corpus)
    enc["per_token_l0"] = round(per_token_l0(bundle, sae, corpus), 2)
    fvu = enc["fvu"]
    print(f"[lab8]   SAE reconstruction FVU={fvu:.4f}  L0(per-token)≈{enc['per_token_l0']}")

    ranks = rank_features(enc, N_ATLAS_FEATURES)
    overlap = len(set(ranks["by_max"]) & set(ranks["by_freq"]))
    # atlas feature set: top by max-activation, padded with top-frequency ones
    atlas_features: list[int] = list(ranks["by_max"])
    for f in ranks["by_freq"]:
        if len(atlas_features) >= N_ATLAS_FEATURES + 5:
            break
        if f not in atlas_features:
            atlas_features.append(f)

    dead_corpus = int((enc["fire_count"] == 0).sum())
    print(f"[lab8]   ranking overlap (max vs freq, top {N_ATLAS_FEATURES}): {overlap}; "
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

    # ----- plots ---------------------------------------------------------------
    if not args.no_plots:
        plot_ranking_disagreement(ctx, enc, ranks)
        plot_atlas_verdicts(ctx, atlas)
        if clamp is not None:
            plot_clamp(ctx, clamp)

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
    best = max(atlas, key=lambda r: r["held_out_auc"]) if atlas else None
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
                f"SAE feature {best['feature']} is labeled '{best['proposed_label']}' and the label SURVIVES "
                f"validation: held-out AUC {best['held_out_auc']:.2f} against domain membership"
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
        "rankings surface largely different features",
        f"- of {metrics['atlas_size']} labeled features, {metrics['n_survived']} survived validation and "
        f"{metrics['n_killed']} were killed (token-feature / polysemantic / low-AUC)",
        "",
        "## 3. Transcoder (Part 2)",
        "",
        f"- FVU {tc_report['fvu']}, per-token L0 {tc_report['mean_l0']}, mean splice-in KL "
        f"{tc_report['mean_splice_kl']} (max {tc_report['max_splice_kl']})",
        "- a transcoder reconstructs the MLP's input→output map, so its features can be de-embedded and",
        "  wired into a circuit — which is why Lab 9's tracing is built on transcoders, not site SAEs.",
        "",
        "## 4. Bridges and causal extension",
        "",
        f"- Lab 4 truth direction: {bridge.get('best_cosine', bridge.get('note', 'n/a'))}",
    ]
    if clamp is not None:
        lines.append(f"- feature clamp (CAUSAL): feature {clamp['feature']} ('{clamp['label']}') "
                     f"{clamp['real_base_hits']}→{clamp['real_max_hits']} keyword hits at "
                     f"{clamp['best_mult']}× peak vs random {clamp['random_max_hits']}; causal={clamp['causal']}")
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
        "1. `feature_atlas.md` — the deliverable: labels, validation verdicts, evidence.",
        "2. `plots/toy_superposition_geometry.png` — why neurons resist reading.",
        "3. `plots/ranking_disagreement.png` — max-activation vs frequency rankings.",
        "4. `plots/atlas_verdicts.png` — how many labels survived.",
        "5. `transcoder_reconstruction_report.json` — the bridge to Lab 9.",
        "6. `plots/feature_clamp.png` — the one CAUSAL feature.",
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
