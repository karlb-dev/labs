"""Lab 9: attribution graphs and circuit tracing.

Same destination as Lab 6, different vehicle: a circuit-shaped explanation
of one behavior, now written as a graph over transcoder features rather than
attention heads. This implementation is the **inspectable miniature** backend:
GPT-2 small plus the full Dunefsky MLP-transcoder stack. A separate registry or
course build can swap in circuit-tracer/Gemma; the evidence contract here stays
the same: replacement-model graph first, real-model interventions second.

The lab has four pieces.

* **Local replacement model.** Run the real model once, freeze attention
  patterns and LayerNorm denominators, replace every MLP with its transcoder,
  and add an ERROR node at every (layer, position) for whatever the transcoder
  failed to reconstruct. Exact replacement logits are a diagnostic, not a
  miracle: the error nodes pay the unreconstructed bill.

* **Direct-attribution edges.** With attention and norms frozen, the network is
  linear in its source terms. For a target scalar t, one backward pass gives
  gradients at every MLP-write injection site; feature edge(s -> t) is
  activation_s * (w_dec_s dot grad). The edge-reconstruction check makes the
  accounting identity visible: bias path + embeddings + features + error nodes
  + transcoder bias = target value.

* **Graph as hypothesis, interventions as test.** The graph is attribution on a
  replacement model. It becomes evidence about the real model only when feature
  edits, suppressions, substitutions, and random matched controls are run on
  the real GPT-2 forward pass.

* **Lab 6 confrontation.** The same signed ledger is computed for an induction
  prompt. Frozen-attention attribution graphs are expected to be least helpful
  exactly where Lab 6's manual attention-head circuit was strongest. That is a
  feature of the lesson, not a bug in the microscope.

Evidence level: ATTRIBUTION for the graph itself; CAUSAL only for successful,
controlled interventions on the real model.
"""

from __future__ import annotations

from typing import Any

import interp_bench as bench

LAB_ID = "L09"

# ---------------------------------------------------------------------------
# Pins. The full Dunefsky et al. MLP-transcoder stack for gpt2-small: one
# transcoder per layer, bare-LayerNorm input convention (no affine, no b_dec
# subtraction, plain ReLU) - the convention this lab records in its diagnostics.
# ---------------------------------------------------------------------------

TC_REPO = "jacobdunefsky/gpt2small-transcoders"
TC_SUBDIR = "gpt2-small-dun-chl-mlp-tc{layer}"
TC_WEIGHTS = "sae.safetensors"

# The behavior: one-token factual recall, the same domain Lab 5 patched.
# Every prompt uses single-token subjects and single-token answers so the
# subject position is unambiguous and the metric is a clean logit difference.
PRIMARY_FACT = {
    "id": "france", "prompt": "The capital of France is",
    "subject": " France", "target": " Paris", "distractor": " Berlin",
}
# The substitution intervention donor: same 5-token template, so positions
# align and the counterfactual features can be written at the same site.
COUNTERFACTUAL_FACT = {
    "id": "germany", "prompt": "The capital of Germany is",
    "subject": " Germany", "target": " Berlin", "distractor": " Paris",
}
# Surface variants of the primary fact (paraphrase battery) plus other
# countries (counterfactual battery). The baseline gate drops any prompt the
# model does not already solve (logit diff <= 0), with a recorded count.
PARAPHRASES = [
    {"id": "para_city", "prompt": "The capital city of France is",
     "subject": " France", "target": " Paris", "distractor": " Berlin"},
    {"id": "para_in", "prompt": "In France, the capital city is",
     "subject": " France", "target": " Paris", "distractor": " Berlin"},
    {"id": "para_possessive", "prompt": "France's capital city is",
     "subject": "France", "target": " Paris", "distractor": " Berlin"},
    {"id": "para_country_of", "prompt": "France is a country. The capital of France is",
     "subject": " France", "target": " Paris", "distractor": " Berlin"},
    {"id": "para_largest", "prompt": "The largest city in France is",
     "subject": " France", "target": " Paris", "distractor": " Berlin"},
]
COUNTERFACTUALS = [
    {"id": "italy", "prompt": "The capital of Italy is",
     "subject": " Italy", "target": " Rome", "distractor": " Paris"},
    {"id": "japan", "prompt": "The capital of Japan is",
     "subject": " Japan", "target": " Tokyo", "distractor": " Paris"},
    {"id": "spain", "prompt": "The capital of Spain is",
     "subject": " Spain", "target": " Madrid", "distractor": " Paris"},
    {"id": "russia", "prompt": "The capital of Russia is",
     "subject": " Russia", "target": " Moscow", "distractor": " Paris"},
]

# Lab 6's behavior, revisited with this lab's instrument. The graph should be
# nearly silent here - induction is attention routing, and the replacement
# model freezes attention into the wiring. That silence is the point.
INDUCTION_VIGNETTE = {
    "id": "induction", "prompt": "red blue green red blue green red blue",
    "target": " green", "distractor": " red",
}

# Node budgets by tier: how many feature nodes the backward-flow selection
# keeps (one backward pass per kept node, so this is also the compute knob).
GRAPH_NODES_BY_TIER = {"a": 16, "b": 28, "c": 40}
INTERVENTION_K = 25          # features suppressed/substituted at the subject site
EXPAND_DEPTH = 2             # backward-flow expansion rounds beyond the logit node
EDGE_KEEP_PER_TARGET = 8     # incoming edges kept per node for the adjacency/plot
TOP_LOGIT_SOURCES = 50       # rows saved in tables/logit_edge_sources.csv
BUDGET_CURVE_POINTS = (4, 8, 12, 16, 24, 32, 48, 64)


def _arg(args: Any, name: str, default: Any) -> Any:
    """Read an optional lab-specific CLI arg without requiring bench changes.

    Some course builds add ``--graph-nodes`` to the shared bench registry. The
    standalone file should also run against a bench that has not learned that
    flag yet, so Lab 9 treats missing attributes as defaults.
    """
    return getattr(args, name, default)


def single_token_id(bundle: bench.ModelBundle, text: str, role: str) -> int:
    ids = bundle.tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"{role} {text!r} must be exactly one token for this lab; got ids={ids}")
    return int(ids[0])


def _find_subsequence(xs: list[int], ys: list[int]) -> list[int]:
    if not ys or len(ys) > len(xs):
        return []
    return [i for i in range(len(xs) - len(ys) + 1) if xs[i:i + len(ys)] == ys]


def tokenization_report_rows(bundle: bench.ModelBundle, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tokenizer audit for all prompts whose positions or answers matter.

    Attribution graphs are position-indexed. This report makes the alignment
    assumption auditable before the graph gets pretty enough to seduce anyone.
    """
    tok = bundle.tokenizer
    rows = []
    for fact in facts:
        prompt_ids = tok(fact["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tok(fact["target"], add_special_tokens=False)["input_ids"]
        distractor_ids = tok(fact["distractor"], add_special_tokens=False)["input_ids"]
        subject_text = fact.get("subject", "")
        subject_ids = tok(subject_text, add_special_tokens=False)["input_ids"] if subject_text else []
        matches = _find_subsequence(prompt_ids, subject_ids) if subject_ids else []
        rows.append({
            "id": fact.get("id", ""),
            "prompt": fact["prompt"],
            "n_prompt_tokens": len(prompt_ids),
            "prompt_token_ids": " ".join(str(i) for i in prompt_ids),
            "prompt_tokens": " | ".join(bundle.tokenizer.decode([i]) for i in prompt_ids),
            "subject": subject_text,
            "subject_token_ids": " ".join(str(i) for i in subject_ids),
            "subject_match_positions": " ".join(str(i) for i in matches),
            "target": fact["target"],
            "target_token_ids": " ".join(str(i) for i in target_ids),
            "distractor": fact["distractor"],
            "distractor_token_ids": " ".join(str(i) for i in distractor_ids),
            "ok_single_token_subject": len(subject_ids) == 1,
            "ok_subject_found": bool(matches),
            "ok_single_token_answers": len(target_ids) == len(distractor_ids) == 1,
            "ok_distinct_answer_tokens": len(target_ids) == len(distractor_ids) == 1 and target_ids[0] != distractor_ids[0],
        })
    return rows


# ---------------------------------------------------------------------------
# Transcoder stack
# ---------------------------------------------------------------------------


class TranscoderStack:
    """All 12 gpt2 MLP transcoders, with the validated input convention.

    Each maps bare-LN(pre-MLP residual) -> MLP output:
        feats = relu(bare @ W_enc + b_enc);  recon = feats @ W_dec + b_dec
    ``bare`` is (x - mean(x)) / sqrt(var(x) + eps) with NO affine gamma/beta.
    The run records this convention beside the FVU numbers so students can see
    that transcoder loading is not a vibes-based ritual.
    """

    def __init__(self, weights: list[dict[str, Any]], device: Any):
        self.layers = []
        for w in weights:
            self.layers.append({k: w[k].float().to(device) for k in ("W_enc", "b_enc", "W_dec", "b_dec")})
        self.n_layers = len(self.layers)
        self.d_in = self.layers[0]["W_enc"].shape[0]
        self.d_sae = self.layers[0]["W_enc"].shape[1]
        self.device = device

    def encode_pre(self, layer: int, bare: Any) -> Any:
        """Encoder PRE-activations (before ReLU) - the graph's target scalars."""
        tc = self.layers[layer]
        return bare @ tc["W_enc"] + tc["b_enc"]

    def w_dec(self, layer: int) -> Any:
        return self.layers[layer]["W_dec"]


def load_transcoder_stack(bundle: bench.ModelBundle) -> TranscoderStack:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if "gpt2" not in bundle.anatomy.model_id.lower():
        raise RuntimeError(
            f"This inspectable Lab 9 backend requires GPT-2 small because the public "
            f"Dunefsky full-stack MLP transcoders are GPT-2-specific; got "
            f"{bundle.anatomy.model_id!r}. Use the circuit-tracer/Gemma backend in "
            "a registry build, or run this miniature path with --model gpt2."
        )
    print(f"[lab9] loading {bundle.anatomy.n_layers}-layer transcoder stack from {TC_REPO}")
    weights = []
    for layer in range(bundle.anatomy.n_layers):
        path = hf_hub_download(TC_REPO, f"{TC_SUBDIR.format(layer=layer)}/{TC_WEIGHTS}")
        weights.append(load_file(path))
    stack = TranscoderStack(weights, bundle.input_device)
    if stack.d_in != bundle.anatomy.d_model:
        raise RuntimeError(f"transcoder d_in {stack.d_in} != model d_model {bundle.anatomy.d_model}")
    print(f"[lab9]   d_in={stack.d_in} d_sae={stack.d_sae} x {stack.n_layers} layers")
    return stack


# ---------------------------------------------------------------------------
# Real-pass capture: everything the replacement model freezes
# ---------------------------------------------------------------------------


def _sigma(x: Any, eps: float) -> Any:
    import torch

    return torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + eps)


def real_pass(bundle: bench.ModelBundle, prompt: str) -> dict[str, Any]:
    """One real forward, capturing the quantities the replacement model needs:
    attention patterns, every LayerNorm input (for the frozen denominators),
    every MLP output (for the error nodes), and the true logits."""
    import torch

    model, tok = bundle.model, bundle.tokenizer
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bundle.input_device)
    grabbed: dict[str, Any] = {"ln1_in": {}, "ln2_in": {}, "mlp_out": {}}
    handles = []
    for layer, blk in enumerate(bundle.blocks):
        handles.append(blk.ln_1.register_forward_pre_hook(
            lambda m, a, l=layer: grabbed["ln1_in"].__setitem__(l, a[0][0].detach().clone())))
        handles.append(blk.ln_2.register_forward_pre_hook(
            lambda m, a, l=layer: grabbed["ln2_in"].__setitem__(l, a[0][0].detach().clone())))
        handles.append(blk.mlp.register_forward_hook(
            lambda m, a, o, l=layer: grabbed["mlp_out"].__setitem__(
                l, (o[0] if isinstance(o, tuple) else o)[0].detach().clone())))
    handles.append(bundle.final_norm.register_forward_pre_hook(
        lambda m, a: grabbed.__setitem__("lnf_in", a[0][0].detach().clone())))
    try:
        with torch.no_grad():
            out = model(ids, output_attentions=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if not out.attentions:
        raise RuntimeError("no attention patterns returned; lab9 must run with eager attention")
    grabbed["attn"] = [a[0].detach().clone() for a in out.attentions]
    grabbed["ids"] = ids
    grabbed["prompt"] = prompt
    grabbed["tokens"] = [tok.decode([i]) for i in ids[0].tolist()]
    grabbed["logits_last"] = out.logits[0, -1].detach().clone()
    return grabbed


# ---------------------------------------------------------------------------
# The frozen replacement forward
# ---------------------------------------------------------------------------


def frozen_forward(bundle: bench.ModelBundle, tcs: TranscoderStack, cap: dict[str, Any],
                   *, zero_inputs: bool = False) -> dict[str, Any]:
    """Differentiable forward of the LOCAL REPLACEMENT MODEL.

    Frozen at the real pass's values: attention patterns, every LayerNorm
    1/sigma, and the per-(layer, position) error vectors. MLP writes use the
    transcoder reconstruction with feature activations DETACHED, plus a
    zero injection leaf per layer - so gradients flow only through the
    linear skeleton (residual adds, frozen attention, frozen LN), and the
    gradient at any injection site is exactly the direct-path read-off
    vector for that (layer, position).

    ``zero_inputs=True`` zeroes the embeddings and every MLP write: what
    remains is the frozen network's bias path (attention value/projection
    biases, LN betas), the constant term of the linear map.
    """
    import torch

    model = bundle.model
    cfg = model.config
    n_head, d_model = cfg.n_head, cfg.n_embd
    d_head = d_model // n_head
    ids = cap["ids"][0]
    seq = ids.shape[0]
    wte, wpe = model.transformer.wte.weight, model.transformer.wpe.weight
    emb0 = wte[ids] + wpe[torch.arange(seq, device=ids.device)]
    if zero_inputs:
        emb0 = torch.zeros_like(emb0)
    emb = emb0.detach().clone().requires_grad_(True)

    resid = emb
    h_list, injs, errs, feats_list = [], [], [], []
    for layer, blk in enumerate(bundle.blocks):
        # Attention with frozen pattern: linear in the residual stream.
        ln1 = blk.ln_1
        inv1 = 1.0 / _sigma(cap["ln1_in"][layer], ln1.eps)            # frozen [seq, 1]
        x = resid
        xln = ln1.weight * ((x - x.mean(-1, keepdim=True)) * inv1) + ln1.bias
        W, b = blk.attn.c_attn.weight, blk.attn.c_attn.bias           # Conv1D [d, 3d]
        v = xln @ W[:, 2 * d_model:] + b[2 * d_model:]
        vh = v.view(seq, n_head, d_head).permute(1, 0, 2)
        av = torch.bmm(cap["attn"][layer], vh)                        # frozen patterns
        attn_out = av.permute(1, 0, 2).reshape(seq, d_model) @ blk.attn.c_proj.weight + blk.attn.c_proj.bias
        x_mid = resid + attn_out

        # Transcoder in place of the MLP, on the frozen bare LN.
        inv2 = 1.0 / _sigma(cap["ln2_in"][layer], blk.ln_2.eps)       # frozen
        bare = (x_mid - x_mid.mean(-1, keepdim=True)) * inv2
        pre = tcs.encode_pre(layer, bare)
        h_list.append(pre)                                            # live: graph targets
        feats = torch.relu(pre)
        feats_list.append(feats.detach())

        # Error node: what the transcoder missed at the REAL input. Constant.
        mid_real = cap["ln2_in"][layer]
        bare_real = (mid_real - mid_real.mean(-1, keepdim=True)) / _sigma(mid_real, blk.ln_2.eps)
        recon_real = torch.relu(tcs.encode_pre(layer, bare_real)) @ tcs.w_dec(layer) + tcs.layers[layer]["b_dec"]
        err = (cap["mlp_out"][layer] - recon_real).detach()
        errs.append(err)

        write = feats.detach() @ tcs.w_dec(layer) + tcs.layers[layer]["b_dec"] + err
        if zero_inputs:
            write = torch.zeros_like(write)
        inj = torch.zeros_like(write).requires_grad_(True)            # gradient probe
        injs.append(inj)
        resid = x_mid + write + inj

    lnf = bundle.final_norm
    invf = 1.0 / _sigma(cap["lnf_in"], lnf.eps)                       # frozen
    final = lnf.weight * ((resid - resid.mean(-1, keepdim=True)) * invf) + lnf.bias
    logits = final @ bundle.lm_head.weight.T
    return {"logits": logits, "h_list": h_list, "emb": emb, "emb0": emb0.detach(),
            "injs": injs, "errs": errs, "feats": feats_list, "seq": seq}


def run_replacement_exactness_check(ctx: bench.RunContext, bundle, tcs, cap, *, atol: float = 5e-3) -> dict:
    """Self-check 1: the replacement model (frozen patterns + frozen sigmas +
    transcoders + error nodes) must reproduce the real final logits. If it
    does not, the 'replacement model' is some other network and every edge
    downstream describes nothing. Aborts on failure."""
    import torch

    with torch.enable_grad():
        fr = frozen_forward(bundle, tcs, cap)
    diff = float((fr["logits"][-1].detach() - cap["logits_last"]).abs().max())
    result = {
        "prompt": cap["prompt"], "max_abs_logit_diff": diff, "atol": atol, "ok": diff <= atol,
        "error_node_l2_by_layer": [round(float(e.norm()), 2) for e in fr["errs"]],
        "explanation": (
            "The local replacement model is exact BY CONSTRUCTION because the "
            "error nodes absorb the transcoders' residual mistakes and all "
            "nonlinearities are frozen at the real pass's values. Any gap "
            "beyond float rounding means the frozen forward is wrong."
        ),
    }
    path = ctx.path("diagnostics", "replacement_exactness.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Replacement-model logits vs real logits (must match).")
    print(f"[bench] replacement exactness check: {'OK' if result['ok'] else 'FAILED'} (max |dlogit| = {diff:.2e})")
    if not result["ok"]:
        raise RuntimeError("Replacement-model forward does not reproduce the real logits; "
                           "see diagnostics/replacement_exactness.json.")
    return result


# ---------------------------------------------------------------------------
# Edges: one backward per target node through the frozen network
# ---------------------------------------------------------------------------


def edges_for_target(bundle, tcs, fr: dict[str, Any], target_scalar: Any) -> dict[str, Any]:
    """All direct incoming edges of one target scalar.

    Returns per-source-family tensors:
      emb_edges    [seq]            token-embedding -> target
      feat_edges   list of [seq, d_sae] per layer (a_f * w_dec_f . grad)
      err_edges    [L, seq]         error-node -> target
      bdec_edges   [L]              transcoder output bias -> target (bias bucket)
    """
    import torch

    grads = torch.autograd.grad(target_scalar, [fr["emb"]] + fr["injs"],
                                retain_graph=True, allow_unused=True)
    g_emb = grads[0] if grads[0] is not None else torch.zeros_like(fr["emb"])
    n_layers = len(fr["injs"])
    feat_edges, err_edges, bdec_edges = [], [], []
    for layer in range(n_layers):
        g = grads[1 + layer]
        if g is None:
            g = torch.zeros_like(fr["injs"][layer])
        dot = tcs.w_dec(layer) @ g.T                       # [d_sae, seq]
        feat_edges.append(fr["feats"][layer] * dot.T)      # [seq, d_sae]
        err_edges.append((fr["errs"][layer] * g).sum(-1))  # [seq]
        bdec_edges.append(float((tcs.layers[layer]["b_dec"] * g).sum()))
    return {
        "emb": (fr["emb0"] * g_emb).sum(-1),               # [seq]
        "feat": feat_edges,
        "err": torch.stack(err_edges),                     # [L, seq]
        "bdec": bdec_edges,
    }


def edge_totals(edges: dict[str, Any]) -> dict[str, float]:
    return {
        "emb": float(edges["emb"].sum()),
        "feat": sum(float(e.sum()) for e in edges["feat"]),
        "err": float(edges["err"].sum()),
        "bdec": sum(edges["bdec"]),
    }


def budget_curve_from_logit_edges(edges: dict[str, Any], budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    """Direct-logit feature-mass coverage as a function of node budget.

    This is not the full backward-flow graph budget, but it is a cheap and
    transparent sensitivity curve for the first pruning decision.
    """
    import torch

    vals = torch.cat([e.abs().flatten() for e in edges["feat"]])
    total = float(vals.sum())
    if vals.numel() == 0 or total <= 0:
        return [{"budget": b, "coverage_of_direct_feature_mass": 0.0} for b in budgets]
    top = torch.sort(vals, descending=True).values
    rows = []
    for b in budgets:
        b2 = min(int(b), int(top.numel()))
        rows.append({
            "budget": int(b),
            "coverage_of_direct_feature_mass": round(float(top[:b2].sum()) / total, 4),
            "direct_feature_mass_total": round(total, 4),
        })
    return rows


def top_logit_source_rows(bundle, tcs, cap: dict[str, Any], edges: dict[str, Any],
                           *, limit: int = TOP_LOGIT_SOURCES) -> list[dict[str, Any]]:
    """Human table of the largest direct sources into the logit node.

    The pruned graph is readable because it is small. This table is the antidote
    to mistaking small for complete: it shows the largest raw direct sources
    before display pruning.
    """
    rows: list[dict[str, Any]] = []
    for pos, weight in enumerate(edges["emb"]):
        w = float(weight)
        rows.append({
            "kind": "embedding", "layer": -1, "pos": pos, "feature": "",
            "token": cap["tokens"][pos], "signed_edge": round(w, 6),
            "abs_edge": round(abs(w), 6), "promotes": "",
        })
    for layer, e in enumerate(edges["feat"]):
        vals, idx = e.abs().flatten().topk(min(limit, e.numel()))
        for _, flat_i in zip(vals.tolist(), idx.tolist()):
            pos, feat = divmod(flat_i, e.shape[1])
            w = float(e[pos, feat])
            rows.append({
                "kind": "feature", "layer": layer, "pos": pos, "feature": feat,
                "token": cap["tokens"][pos], "signed_edge": round(w, 6),
                "abs_edge": round(abs(w), 6),
                "promotes": " | ".join(deembed_feature(bundle, tcs, layer, feat, k=3)),
            })
    for layer in range(edges["err"].shape[0]):
        for pos in range(edges["err"].shape[1]):
            w = float(edges["err"][layer, pos])
            rows.append({
                "kind": "error", "layer": layer, "pos": pos, "feature": "",
                "token": cap["tokens"][pos], "signed_edge": round(w, 6),
                "abs_edge": round(abs(w), 6), "promotes": "",
            })
    for layer, w0 in enumerate(edges.get("bdec", [])):
        w = float(w0)
        rows.append({
            "kind": "transcoder_bias", "layer": layer, "pos": "all", "feature": "",
            "token": "", "signed_edge": round(w, 6), "abs_edge": round(abs(w), 6),
            "promotes": "",
        })
    rows.sort(key=lambda r: -float(r["abs_edge"]))
    for i, r in enumerate(rows[:limit], start=1):
        r["rank_by_abs_edge"] = i
    return rows[:limit]


def influence_ledger_rows(label: str, influence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    signed = influence["signed_contributions"]
    for source, val in signed.items():
        rows.append({
            "case": label, "source": source, "signed_contribution": val,
            "metric_logit_diff": round(float(influence["metric_logit_diff"]), 6),
        })
    rows.extend([
        {"case": label, "source": "abs_share_features", "signed_contribution": influence["share_features"],
         "metric_logit_diff": round(float(influence["metric_logit_diff"]), 6)},
        {"case": label, "source": "abs_share_embeddings", "signed_contribution": influence["share_embeddings"],
         "metric_logit_diff": round(float(influence["metric_logit_diff"]), 6)},
        {"case": label, "source": "abs_share_errors", "signed_contribution": influence["share_errors"],
         "metric_logit_diff": round(float(influence["metric_logit_diff"]), 6)},
        {"case": label, "source": "kept_coverage_of_feature_mass",
         "signed_contribution": influence["kept_coverage_of_feature_mass"],
         "metric_logit_diff": round(float(influence["metric_logit_diff"]), 6)},
    ])
    return rows


# ---------------------------------------------------------------------------
# Graph construction: backward-flow node selection + adjacency
# ---------------------------------------------------------------------------


def deembed_feature(bundle, tcs, layer: int, feature: int, k: int = 4) -> list[str]:
    """Project a feature's decoder row (an MLP-output-space write) through the
    final norm + unembedding: which tokens does this feature promote? The
    same de-embedding read Lab 8 introduced."""
    import torch

    dec = tcs.w_dec(layer)[feature]
    with torch.no_grad():
        normed = bundle.final_norm(dec.unsqueeze(0)).squeeze(0)
        logits = normed @ bundle.lm_head.weight.T
    top = torch.argsort(logits, descending=True)[:k].tolist()
    return [bundle.tokenizer.decode([t]) for t in top]


def build_graph(ctx, bundle, tcs, cap, *, node_budget: int, fact: dict[str, Any],
                check_reconstruction: bool = False) -> dict[str, Any]:
    """The full attribution-graph pipeline for one prompt.

    1. frozen forward; logit node = logit(target) - logit(distractor)
    2. backward from the logit node -> direct edges from every feature
    3. backward-flow expansion: keep the strongest feature nodes, backward
       from each, add their strongest sources (depth-limited, budget-capped)
    4. adjacency among kept nodes + emb/error nodes; influence accounting
    """
    import torch

    tok = bundle.tokenizer
    tid = single_token_id(bundle, fact["target"], f"target for {fact.get('id', fact['prompt'])}")
    did = single_token_id(bundle, fact["distractor"], f"distractor for {fact.get('id', fact['prompt'])}")
    with torch.enable_grad():
        fr = frozen_forward(bundle, tcs, cap)
        metric = fr["logits"][-1, tid] - fr["logits"][-1, did]
        metric_value = float(metric.detach())

        # --- logit node edges + the bias path of the frozen-linear network --
        logit_edges = edges_for_target(bundle, tcs, fr, metric)
        totals = edge_totals(logit_edges)
        fr0 = frozen_forward(bundle, tcs, cap, zero_inputs=True)
        bias = float((fr0["logits"][-1, tid] - fr0["logits"][-1, did]).detach())
        if check_reconstruction:
            recon = bias + totals["emb"] + totals["feat"] + totals["err"] + totals["bdec"]
            rel = abs(recon - metric_value) / max(abs(metric_value), 1e-6)
            result = {
                "prompt": cap["prompt"], "metric_logit_diff": metric_value,
                "reconstructed": recon, "rel_err": rel,
                "bias_path": bias, "edge_totals": totals, "ok": rel <= 0.01,
                "explanation": (
                    "The frozen replacement network is linear, so the logit-diff "
                    "metric must equal its bias path plus the sum of EVERY direct "
                    "edge (embeddings, features, error nodes, transcoder output "
                    "biases). An imbalance means the edge gradients are not "
                    "measuring what the graph claims."
                ),
            }
            path = ctx.path("diagnostics", "edge_reconstruction_check.json")
            bench.write_json(path, result)
            ctx.register_artifact(path, "diagnostic", "Edges + bias must sum to the metric (linearity audit).")
            print(f"[bench] edge reconstruction check: {'OK' if result['ok'] else 'FAILED'} "
                  f"(metric {metric_value:+.4f} vs reconstructed {recon:+.4f})")
            if not result["ok"]:
                raise RuntimeError("Edge reconstruction check failed; see diagnostics/edge_reconstruction_check.json.")

        # --- backward-flow node selection ---------------------------------
        # Rank candidate feature nodes by |direct edge to the logit node|,
        # then expand: backward from each kept node and pull in strong
        # sources the logit pass alone would miss (multi-hop paths).
        def top_feature_edges(edge_pack, k):
            rows = []
            for layer, e in enumerate(edge_pack["feat"]):
                vals, idx = e.abs().flatten().topk(min(k, e.numel()))
                for v, i in zip(vals.tolist(), idx.tolist()):
                    if v <= 0:
                        continue
                    pos, feat = divmod(i, e.shape[1])
                    rows.append((v, layer, pos, feat))
            rows.sort(key=lambda r: -r[0])
            return rows[:k]

        kept: dict[tuple[int, int, int], dict[str, Any]] = {}
        edge_packs: dict[str, Any] = {"logit": logit_edges}
        frontier = []
        for v, layer, pos, feat in top_feature_edges(logit_edges, node_budget):
            key = (layer, pos, feat)
            kept[key] = {"selected_by": "logit", "abs_edge_to_selector": v}
            frontier.append(key)

        for depth in range(EXPAND_DEPTH):
            new_frontier = []
            for key in frontier:
                if len(kept) >= node_budget:
                    break
                layer, pos, feat = key
                pack = edges_for_target(bundle, tcs, fr, fr["h_list"][layer][pos, feat])
                edge_packs[f"f{layer}.{pos}.{feat}"] = pack
                for v, sl, sp, sf in top_feature_edges(pack, 4):
                    skey = (sl, sp, sf)
                    if skey not in kept and len(kept) < node_budget:
                        kept[skey] = {"selected_by": f"f{layer}.{pos}.{feat}", "abs_edge_to_selector": v}
                        new_frontier.append(skey)
            frontier = new_frontier
            if not frontier:
                break

        # edge packs for any kept node not yet expanded (needed for adjacency)
        for (layer, pos, feat) in kept:
            name = f"f{layer}.{pos}.{feat}"
            if name not in edge_packs:
                edge_packs[name] = edges_for_target(bundle, tcs, fr, fr["h_list"][layer][pos, feat])

    # --- node table --------------------------------------------------------
    seq = fr["seq"]
    nodes: list[dict[str, Any]] = []
    for pos in range(seq):
        nodes.append({"name": f"emb.{pos}", "kind": "embedding", "layer": -1, "pos": pos,
                      "label": f"emb {cap['tokens'][pos]!r}"})
    for (layer, pos, feat), meta in sorted(kept.items()):
        act = float(fr["feats"][layer][pos, feat])
        promotes = deembed_feature(bundle, tcs, layer, feat)
        edge_to_logit = float(logit_edges["feat"][layer][pos, feat])
        nodes.append({"name": f"f{layer}.{pos}.{feat}", "kind": "feature", "layer": layer,
                      "pos": pos, "feature": feat, "activation": round(act, 3),
                      "token": cap["tokens"][pos], "promotes": promotes,
                      "edge_to_logit": round(edge_to_logit, 6),
                      "abs_edge_to_logit": round(abs(edge_to_logit), 6),
                      "selected_by": meta["selected_by"],
                      "abs_edge_to_selector": round(float(meta["abs_edge_to_selector"]), 6)})
    err_l2 = [[float(fr["errs"][layer][pos].norm()) for pos in range(seq)]
              for layer in range(tcs.n_layers)]
    for layer in range(tcs.n_layers):
        for pos in range(seq):
            nodes.append({"name": f"err.{layer}.{pos}", "kind": "error", "layer": layer, "pos": pos,
                          "label": f"error L{layer} @ {cap['tokens'][pos]!r}", "l2": round(err_l2[layer][pos], 2)})
    nodes.append({"name": "logit", "kind": "logit", "layer": tcs.n_layers, "pos": seq - 1,
                  "label": f"logit({fact['target']!r}) - logit({fact['distractor']!r})"})

    # --- adjacency: incoming edges per target node -------------------------
    edges_rows: list[dict[str, Any]] = []

    def harvest(target_name: str, pack: dict[str, Any]) -> None:
        cand: list[tuple[float, str]] = []
        for pos in range(seq):
            cand.append((float(pack["emb"][pos]), f"emb.{pos}"))
        for (layer, pos, feat) in kept:
            cand.append((float(pack["feat"][layer][pos, feat]), f"f{layer}.{pos}.{feat}"))
        for layer in range(tcs.n_layers):
            for pos in range(seq):
                cand.append((float(pack["err"][layer, pos]), f"err.{layer}.{pos}"))
        cand = [c for c in cand if c[1] != target_name and abs(c[0]) > 1e-6]
        cand.sort(key=lambda c: -abs(c[0]))
        for weight, source in cand[:EDGE_KEEP_PER_TARGET]:
            edges_rows.append({"source": source, "target": target_name, "weight": round(weight, 5)})

    harvest("logit", logit_edges)
    for (layer, pos, feat) in kept:
        harvest(f"f{layer}.{pos}.{feat}", edge_packs[f"f{layer}.{pos}.{feat}"])

    # --- influence accounting ----------------------------------------------
    # Direct shares of the logit node, over ALL sources (not just kept nodes):
    # the honest completeness line of the graph card.
    abs_feat_total = sum(float(e.abs().sum()) for e in logit_edges["feat"])
    abs_kept = sum(abs(float(logit_edges["feat"][l][p, f])) for (l, p, f) in kept)
    direct_abs = {
        "features_all": abs_feat_total,
        "features_kept": abs_kept,
        "embeddings": float(logit_edges["emb"].abs().sum()),
        "errors": float(logit_edges["err"].abs().sum()),
    }
    denom = direct_abs["features_all"] + direct_abs["embeddings"] + direct_abs["errors"]
    # The discriminating view is the SIGNED decomposition: bias + emb + feat
    # + err + b_dec sums to the metric exactly (the reconstruction check).
    # |edge|-mass shares can look similar for very different mechanisms; the
    # signed ledger says who actually paid for the logit difference.
    signed = {
        "bias_path": round(bias, 4),
        "embeddings": round(totals["emb"], 4),
        "features": round(totals["feat"], 4),
        "errors": round(totals["err"], 4),
        "transcoder_bias": round(totals["bdec"], 4),
    }
    influence = {
        "metric_logit_diff": metric_value,
        "signed_contributions": signed,
        "feature_signed_fraction_of_metric": round(totals["feat"] / metric_value, 4)
        if abs(metric_value) > 1e-6 else None,
        "direct_abs_edge_mass": {k: round(v, 4) for k, v in direct_abs.items()},
        "share_features": round(direct_abs["features_all"] / max(denom, 1e-9), 4),
        "share_embeddings": round(direct_abs["embeddings"] / max(denom, 1e-9), 4),
        "share_errors": round(direct_abs["errors"] / max(denom, 1e-9), 4),
        "kept_coverage_of_feature_mass": round(abs_kept / max(abs_feat_total, 1e-9), 4),
        "budget_curve": budget_curve_from_logit_edges(logit_edges, BUDGET_CURVE_POINTS),
    }
    return {"fact": fact, "cap": cap, "fr": fr, "kept": kept, "nodes": nodes,
            "edges": edges_rows, "edge_sources": top_logit_source_rows(bundle, tcs, cap, logit_edges),
            "influence": influence, "metric": metric_value}


# ---------------------------------------------------------------------------
# Interventions on the REAL model
# ---------------------------------------------------------------------------


def feature_acts_real(bundle, tcs, prompt: str) -> tuple[Any, list[Any], Any]:
    """Transcoder feature activations on the real model (no replacement)."""
    import torch

    tok = bundle.tokenizer
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bundle.input_device)
    mids: dict[int, Any] = {}
    handles = [blk.ln_2.register_forward_pre_hook(
        lambda m, a, l=layer: mids.__setitem__(l, a[0][0].detach().clone()))
        for layer, blk in enumerate(bundle.blocks)]
    try:
        with torch.no_grad():
            out = bundle.model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    acts = []
    for layer, blk in enumerate(bundle.blocks):
        bare = (mids[layer] - mids[layer].mean(-1, keepdim=True)) / _sigma(mids[layer], blk.ln_2.eps)
        acts.append(torch.relu(tcs.encode_pre(layer, bare)))
    return ids, acts, out.logits[0, -1].detach()


def run_with_feature_edits(bundle, tcs, prompt: str, edits: list[tuple[int, int, int, float]]) -> Any:
    """Real-model forward with feature edits applied as decoder-direction
    deltas to the MLP output: mlp_out[pos] += (new_act - real_act) * w_dec[f].

    This is the crucial epistemics of the lab: the GRAPH lives in the
    replacement model, but the intervention is measured on the REAL model -
    a graph hypothesis that only works in its own idealization is not an
    explanation of the model anyone deployed.
    """
    import torch

    ids, acts, _ = feature_acts_real(bundle, tcs, prompt)
    # ``edits`` are assignments, not additive nudges. If the same feature is
    # listed twice (common in substitution, where a donor feature can also be
    # in the suppressed original supernode), the LAST assignment wins.
    assignments: dict[tuple[int, int, int], float] = {}
    for layer, pos, feat, new_act in edits:
        assignments[(int(layer), int(pos), int(feat))] = float(new_act)
    by_layer: dict[int, list[tuple[int, int, float]]] = {}
    for (layer, pos, feat), new_act in assignments.items():
        delta = new_act - float(acts[layer][pos, feat])
        by_layer.setdefault(layer, []).append((pos, feat, delta))

    def make_hook(layer: int):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.clone()
            for pos, feat, delta in by_layer[layer]:
                out[0, pos] += delta * tcs.w_dec(layer)[feat]
            return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
        return hook

    handles = [bundle.blocks[layer].mlp.register_forward_hook(make_hook(layer)) for layer in by_layer]
    try:
        with torch.no_grad():
            out = bundle.model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return out.logits[0, -1].detach()


def run_feature_edit_noop_check(ctx, bundle, tcs, fact, *, atol: float = 1e-4) -> dict:
    """Self-check 3: editing features to their own observed activations must
    be a numerical identity on the real model. Aborts on failure."""
    _, acts, base = feature_acts_real(bundle, tcs, fact["prompt"])
    edits = []
    for layer in (0, tcs.n_layers // 2, tcs.n_layers - 1):
        last = acts[layer].shape[0] - 1
        a = acts[layer][last]
        for feat in (a > 0).nonzero().flatten().tolist()[:4]:
            edits.append((layer, last, feat, float(a[feat])))
    edited = run_with_feature_edits(bundle, tcs, fact["prompt"], edits)
    diff = float((edited - base).abs().max())
    result = {"prompt": fact["prompt"], "n_noop_assignments": len(edits),
              "max_abs_logit_diff": diff, "atol": atol, "ok": diff <= atol,
              "explanation": "Setting features to their own activations is a no-op; any logit "
                             "change means the edit hooks or activation capture are misaligned."}
    path = ctx.path("diagnostics", "feature_edit_noop_check.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Self-edit no-op check for the feature-intervention hooks.")
    print(f"[bench] feature-edit no-op check: {'OK' if result['ok'] else 'FAILED'} (max |dlogit| = {diff:.2e})")
    if not result["ok"]:
        raise RuntimeError("Feature-edit no-op check failed; see diagnostics/feature_edit_noop_check.json.")
    return result


def subject_position(bundle, fact: dict[str, Any]) -> int:
    tok = bundle.tokenizer
    ids = tok(fact["prompt"], add_special_tokens=False)["input_ids"]
    sid = tok(fact["subject"], add_special_tokens=False)["input_ids"]
    matches = _find_subsequence(ids, sid)
    if len(sid) != 1:
        raise RuntimeError(f"subject {fact['subject']!r} must be one token in this lab; got ids={sid}")
    if not matches:
        texts = [tok.decode([i]) for i in ids]
        raise RuntimeError(f"subject token {fact['subject']!r} ids={sid} not found in {texts}")
    return matches[-1]


def run_interventions(ctx, bundle, tcs, graph, fact, counterfact, seed: int,
                      k: int = INTERVENTION_K) -> list[dict[str, Any]]:
    """Suppress / substitute / random-control interventions at the subject site.

    The supernode is graph-informed: kept feature nodes at the subject
    position, padded (if needed) with the strongest remaining subject-site
    activations by activation x decoder-norm - and the same padding rule
    defines the donor features from the counterfactual prompt.
    """
    import torch

    tok = bundle.tokenizer
    subj = subject_position(bundle, fact)
    tid = single_token_id(bundle, fact["target"], f"target for {fact['id']}")
    did = single_token_id(bundle, fact["distractor"], f"distractor for {fact['id']}")

    _, acts, base = feature_acts_real(bundle, tcs, fact["prompt"])

    def ld(logits) -> float:
        return float(logits[tid] - logits[did])

    def p(logits, token_id) -> float:
        return float(torch.softmax(logits, -1)[token_id])

    # the subject supernode: graph-kept features at the subject position first
    supernode = [(layer, feat) for (layer, pos, feat) in graph["kept"] if pos == subj]
    ranked = []
    for layer in range(tcs.n_layers):
        a = acts[layer][subj]
        for feat in (a > 0).nonzero().flatten().tolist():
            ranked.append((float(a[feat]) * float(tcs.w_dec(layer)[feat].norm()), layer, feat))
    ranked.sort(key=lambda r: -r[0])
    for _, layer, feat in ranked:
        if len(supernode) >= k:
            break
        if (layer, feat) not in supernode:
            supernode.append((layer, feat))

    # donor features from the counterfactual prompt at its (aligned) subject site
    cf_subj = subject_position(bundle, counterfact)
    _, cf_acts, cf_base = feature_acts_real(bundle, tcs, counterfact["prompt"])
    donors = []
    for layer in range(tcs.n_layers):
        a = cf_acts[layer][cf_subj]
        for feat in (a > 0).nonzero().flatten().tolist():
            donors.append((float(a[feat]) * float(tcs.w_dec(layer)[feat].norm()), layer, feat, float(a[feat])))
    donors.sort(key=lambda r: -r[0])
    donors = donors[:k]

    # random control: matched count of OTHER active subject-site features
    gen = torch.Generator().manual_seed(seed * 31 + 9)
    sn_set = set(supernode)
    pool = [(layer, feat) for _, layer, feat in ranked if (layer, feat) not in sn_set]
    perm = torch.randperm(len(pool), generator=gen).tolist()
    random_set = [pool[i] for i in perm[:min(k, len(pool))]]

    cf_tid = single_token_id(bundle, counterfact["target"], f"target for {counterfact['id']}")

    suppress = [(layer, subj, feat, 0.0) for (layer, feat) in supernode]
    substitute_assignments = {(layer, subj, feat): 0.0 for (layer, feat) in supernode}
    for _, layer, feat, act in donors:
        substitute_assignments[(layer, subj, feat)] = act
    substitute = [(layer, pos, feat, act) for (layer, pos, feat), act in substitute_assignments.items()]
    random_edits = [(layer, subj, feat, 0.0) for (layer, feat) in random_set]

    manifest = {
        "primary_prompt": fact["prompt"],
        "counterfactual_prompt": counterfact["prompt"],
        "subject_position_primary": subj,
        "subject_position_counterfactual": cf_subj,
        "requested_k": k,
        "n_supernode_features": len(supernode),
        "n_donor_features": len(donors),
        "n_random_features": len(random_set),
        "n_duplicate_substitution_assignments_collapsed": len(suppress) + len(donors) - len(substitute),
        "note": (
            "Feature edits are assignments to target activations. If suppression and donor "
            "sets contain the same feature, the donor assignment wins for substitution; "
            "the duplicate count records that collapse."
        ),
    }
    bench.write_json(ctx.path("diagnostics", "feature_intervention_manifest.json"), manifest)
    ctx.register_artifact(ctx.path("diagnostics", "feature_intervention_manifest.json"), "diagnostic",
                          "Which features were suppressed/substituted, and how duplicate assignments were handled.")

    feature_rows = []
    graph_subject_set = {(layer, feat) for (layer, pos, feat) in graph["kept"] if pos == subj}
    for layer, feat in supernode:
        feature_rows.append({
            "condition": "suppress_subject_supernode", "layer": layer, "pos": subj, "feature": feat,
            "old_activation": round(float(acts[layer][subj, feat]), 6), "new_activation": 0.0,
            "decoder_norm": round(float(tcs.w_dec(layer)[feat].norm()), 6),
            "selected_by_graph": (layer, feat) in graph_subject_set,
        })
    for _, layer, feat, act in donors:
        feature_rows.append({
            "condition": "substitute_counterfactual_donor", "layer": layer, "pos": subj, "feature": feat,
            "old_activation": round(float(acts[layer][subj, feat]), 6), "new_activation": round(float(act), 6),
            "decoder_norm": round(float(tcs.w_dec(layer)[feat].norm()), 6),
            "selected_by_graph": (layer, feat) in graph_subject_set,
        })
    for layer, feat in random_set:
        feature_rows.append({
            "condition": "random_suppression_control", "layer": layer, "pos": subj, "feature": feat,
            "old_activation": round(float(acts[layer][subj, feat]), 6), "new_activation": 0.0,
            "decoder_norm": round(float(tcs.w_dec(layer)[feat].norm()), 6),
            "selected_by_graph": False,
        })
    bench.write_csv_with_context(ctx, ctx.path("tables", "supernode_features.csv"), feature_rows)
    ctx.register_artifact(ctx.path("tables", "supernode_features.csv"), "table",
                          "Features edited by the graph-guided suppression/substitution/control interventions.")

    rows = []

    def record(condition, logits, n_edited):
        rows.append({
            "condition": condition, "n_features_edited": n_edited,
            "logit_diff": round(ld(logits), 4),
            "p_target": round(p(logits, tid), 5),
            "p_distractor": round(p(logits, did), 5),
            "p_counterfactual_target": round(p(logits, cf_tid), 5),
            "top1": tok.decode([int(logits.argmax())]),
        })

    record("baseline", base, 0)
    record("suppress_subject_supernode", run_with_feature_edits(bundle, tcs, fact["prompt"], suppress),
           len({(l, p, f) for l, p, f, _ in suppress}))
    record("substitute_counterfactual", run_with_feature_edits(bundle, tcs, fact["prompt"], substitute),
           len({(l, p, f) for l, p, f, _ in substitute}))
    record("random_suppression_control", run_with_feature_edits(bundle, tcs, fact["prompt"], random_edits),
           len({(l, p, f) for l, p, f, _ in random_edits}))
    rows.append({
        "condition": "counterfactual_prompt_reference", "n_features_edited": 0,
        "logit_diff": round(ld(cf_base), 4), "p_target": round(p(cf_base, tid), 5),
        "p_distractor": round(p(cf_base, did), 5),
        "p_counterfactual_target": round(p(cf_base, cf_tid), 5),
        "top1": tok.decode([int(cf_base.argmax())]),
    })
    base_ld = next(r["logit_diff"] for r in rows if r["condition"] == "baseline")
    for r in rows:
        r["delta_logit_diff_vs_baseline"] = round(r["logit_diff"] - base_ld, 4)
    return rows


# ---------------------------------------------------------------------------
# Paraphrase battery and the induction vignette
# ---------------------------------------------------------------------------


def baseline_gate(bundle, facts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only facts the model already solves (logit diff > 0), Lab 6's rule:
    you cannot trace a mechanism for a behavior the model is not doing."""
    import torch

    tok = bundle.tokenizer
    kept, dropped = [], []
    for fact in facts:
        ids = tok(fact["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(bundle.input_device)
        with torch.no_grad():
            logits = bundle.model(ids, use_cache=False).logits[0, -1]
        tid = tok(fact["target"], add_special_tokens=False)["input_ids"]
        did = tok(fact["distractor"], add_special_tokens=False)["input_ids"]
        subject_ok = True
        subject_reason = ""
        try:
            subj_pos = subject_position(bundle, fact)
        except Exception as exc:  # recorded rather than hidden behind a traceback
            subj_pos = ""
            subject_ok = False
            subject_reason = str(exc)
        row = {**fact, "subject_position": subj_pos}
        if len(tid) != 1 or len(did) != 1:
            dropped.append({**row, "reason": "multi-token answer",
                            "target_token_ids": " ".join(str(i) for i in tid),
                            "distractor_token_ids": " ".join(str(i) for i in did)})
            continue
        if tid[0] == did[0]:
            dropped.append({**row, "reason": "target and distractor collapse to same token",
                            "target_token_id": tid[0], "distractor_token_id": did[0]})
            continue
        if not subject_ok:
            dropped.append({**row, "reason": "subject tokenization/alignment failed",
                            "subject_failure": subject_reason})
            continue
        probs = torch.softmax(logits, -1)
        diff = float(logits[tid[0]] - logits[did[0]])
        top1 = int(logits.argmax())
        rank_target = int((logits > logits[tid[0]]).sum().item() + 1)
        row = {**row,
               "target_token_id": tid[0], "distractor_token_id": did[0],
               "baseline_logit_diff": round(diff, 4),
               "p_target": round(float(probs[tid[0]]), 6),
               "p_distractor": round(float(probs[did[0]]), 6),
               "target_rank": rank_target,
               "top1": tok.decode([top1])}
        (kept if diff > 0 else dropped).append(row if diff > 0 else {**row, "reason": "model fails baseline"})
    return kept, dropped


def paraphrase_recurrence(graphs: list[dict[str, Any]], bundle) -> list[dict[str, Any]]:
    """Which (layer, feature) pairs at the SUBJECT position recur across
    surface variants? Recurring features are mechanism candidates; one-off
    features are template artifacts. Positions differ across paraphrases, so
    recurrence is counted on (layer, feature) identity at each prompt's own
    subject position."""
    counts: dict[tuple[int, int], dict[str, Any]] = {}
    for g in graphs:
        subj = subject_position_from_graph(g, bundle)
        seen = set()
        for (layer, pos, feat) in g["kept"]:
            if pos != subj or (layer, feat) in seen:
                continue
            seen.add((layer, feat))
            entry = counts.setdefault((layer, feat), {"layer": layer, "feature": feat, "n_prompts": 0,
                                                      "prompts": []})
            entry["n_prompts"] += 1
            entry["prompts"].append(g["fact"]["id"])
    rows = sorted(counts.values(), key=lambda r: -r["n_prompts"])
    for r in rows:
        r["prompts"] = ",".join(r["prompts"])
    return rows


def subject_position_from_graph(g: dict[str, Any], bundle) -> int:
    return subject_position(bundle, g["fact"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_graph(ctx, bundle, graph, name: str, title: str) -> None:
    """Layered rendering: x = token position, y = layer; features are dots,
    embeddings squares, the logit node a star; edge width tracks |weight|."""
    import matplotlib.pyplot as plt

    cap = graph["cap"]
    seq = len(cap["tokens"])
    fig, ax = plt.subplots(figsize=(max(8.0, 1.7 * seq), 8.0))
    coords: dict[str, tuple[float, float]] = {}
    for n in graph["nodes"]:
        if n["kind"] == "embedding":
            coords[n["name"]] = (n["pos"], -1.2)
        elif n["kind"] == "feature":
            coords[n["name"]] = (n["pos"], n["layer"])
        elif n["kind"] == "logit":
            coords[n["name"]] = (seq - 1, n["layer"] + 1.2)
    # error nodes are drawn only if they carry a kept edge
    used_err = {e["source"] for e in graph["edges"] if e["source"].startswith("err.")}
    for name_ in used_err:
        _, layer, pos = name_.split(".")
        coords[name_] = (int(pos) + 0.28, int(layer) + 0.28)

    max_w = max((abs(e["weight"]) for e in graph["edges"]), default=1.0)
    for e in graph["edges"]:
        if e["source"] not in coords or e["target"] not in coords:
            continue
        x0, y0 = coords[e["source"]]
        x1, y1 = coords[e["target"]]
        w = abs(e["weight"]) / max_w
        ax.plot([x0, x1], [y0, y1], color=("tab:blue" if e["weight"] > 0 else "tab:red"),
                linewidth=0.5 + 3.0 * w, alpha=0.25 + 0.55 * w, zorder=1)
    for n in graph["nodes"]:
        if n["name"] not in coords:
            continue
        x, y = coords[n["name"]]
        if n["kind"] == "embedding":
            ax.scatter([x], [y], marker="s", s=90, color="#444444", zorder=2)
        elif n["kind"] == "feature":
            ax.scatter([x], [y], marker="o", s=110, color="tab:green", edgecolor="black",
                       linewidth=0.5, zorder=3)
            ax.annotate(f"L{n['layer']} f{n['feature']}\n→{n['promotes'][0]!r}", (x, y),
                        textcoords="offset points", xytext=(6, 4), fontsize=5.5)
        elif n["kind"] == "logit":
            ax.scatter([x], [y], marker="*", s=380, color="gold", edgecolor="black", zorder=3)
    for name_ in used_err:
        x, y = coords[name_]
        ax.scatter([x], [y], marker="^", s=55, color="tab:orange", alpha=0.8, zorder=2)
    ax.set_xticks(range(seq))
    ax.set_xticklabels([bench.visible_token(t) for t in cap["tokens"]], rotation=30, fontsize=8)
    # Feature/error layers are 0..L-1 and the logit node sits above them. Use
    # observed feature layers rather than assuming GPT-2's 12 blocks in the plot.
    feature_layers = sorted({int(n["layer"]) for n in graph["nodes"]
                             if n.get("kind") in {"feature", "error"}})
    logit_y = max(y for _, y in coords.values()) if coords else (max(feature_layers, default=0) + 2.2)
    yticks = [-1.2] + feature_layers + [logit_y]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["emb"] + [f"L{i}" for i in feature_layers] + ["logit"], fontsize=8)
    ax.set_ylim(-2, logit_y + 1.0)
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.set_title(title + "\n(● feature  ■ embedding  ▲ error  ★ logit node; blue +, red −)")
    ax.grid(True, alpha=0.2)
    bench.save_figure(ctx, fig, name, f"Pruned attribution graph: {title}")


def plot_influence_composition(ctx, fact_inf, ind_inf) -> None:
    """The signed ledger of the logit node, fact vs induction. The five bars
    sum to the metric exactly (the reconstruction check), so 'who paid for
    the logit difference' is an accounting identity, not a vibe."""
    import matplotlib.pyplot as plt
    import numpy as np

    cats = ["bias_path", "embeddings", "features", "errors", "transcoder_bias"]
    labels = ["bias\npath", "token\nembeddings", "features", "error\nnodes", "transcoder\nbias"]
    colors = ["#bbbbbb", "#444444", "tab:green", "tab:orange", "#8888bb"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=False)
    for ax, inf, title in (
        (axes[0], fact_inf, "capital fact (MLP recall)"),
        (axes[1], ind_inf, "induction (attention routing)"),
    ):
        vals = [inf["signed_contributions"][c] for c in cats]
        ax.bar(np.arange(len(cats)), vals, color=colors)
        ax.axhline(inf["metric_logit_diff"], color="black", linestyle="--", linewidth=1.0,
                   label=f"metric = {inf['metric_logit_diff']:+.2f}")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("signed contribution to the logit diff")
    fig.suptitle("Who pays for the logit difference? Features carry the fact; "
                 "frozen attention routes embeddings for induction")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "influence_composition.png",
                      "Signed decomposition of the logit node, fact prompt vs induction prompt.")


def plot_edge_mass_shares(ctx, fact_inf, ind_inf) -> None:
    """Absolute direct-edge mass shares into the logit node.

    Signed ledgers answer who paid for the logit difference. Absolute shares
    answer how much explanatory load is outside the feature dictionary.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cases = [("capital fact", fact_inf), ("induction", ind_inf)]
    cats = ["share_features", "share_embeddings", "share_errors"]
    labels = ["features", "embeddings", "error nodes"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(cases))
    width = 0.22
    for j, cat in enumerate(cats):
        ax.bar(x + (j - 1) * width, [case[1][cat] for case in cases], width, label=labels[j])
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cases])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of absolute direct edge mass into the logit node")
    ax.set_title("Graph visibility: features vs embeddings vs error nodes")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    bench.save_figure(ctx, fig, "edge_mass_shares.png",
                      "Absolute direct-edge mass shares for fact and induction prompts.")


def plot_interventions(ctx, rows, fact) -> None:
    fig, ax = bench.new_figure(figsize=(9.0, 5.0))
    conds = [r["condition"] for r in rows]
    vals = [r["logit_diff"] for r in rows]
    colors = {"baseline": "tab:gray", "suppress_subject_supernode": "tab:red",
              "substitute_counterfactual": "tab:purple", "random_suppression_control": "tab:olive",
              "counterfactual_prompt_reference": "tab:blue"}
    ax.bar(range(len(rows)), vals, color=[colors.get(c, "tab:gray") for c in conds])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=7.5)
    ax.set_ylabel(f"logit({fact['target']!r}) − logit({fact['distractor']!r})")
    ax.set_title("Graph-guided interventions on the REAL model")
    bench.save_figure(ctx, fig, "intervention_effects.png",
                      "Logit-diff under supernode suppression/substitution vs the random control.")


def plot_paraphrase_recurrence(ctx, rec_rows, n_prompts) -> None:
    fig, ax = bench.new_figure(figsize=(9.0, 5.0))
    top = rec_rows[:20]
    labels = [f"L{r['layer']} f{r['feature']}" for r in top]
    ax.bar(range(len(top)), [r["n_prompts"] for r in top], color="tab:green")
    ax.axhline(n_prompts, color="black", linestyle="--", linewidth=0.8,
               label=f"all {n_prompts} prompts")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.set_ylabel("prompts where the feature is a kept subject-site node")
    ax.set_title("Paraphrase robustness: recurring features vs template artifacts")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "paraphrase_recurrence.png",
                      "Subject-site graph features recurring across surface variants.")


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------


def intervention_deltas(interventions: list[dict[str, Any]]) -> dict[str, float | bool | str]:
    base = next(r for r in interventions if r["condition"] == "baseline")
    sup = next(r for r in interventions if r["condition"] == "suppress_subject_supernode")
    sub = next(r for r in interventions if r["condition"] == "substitute_counterfactual")
    ctl = next(r for r in interventions if r["condition"] == "random_suppression_control")
    suppress_drop = float(base["logit_diff"]) - float(sup["logit_diff"])
    random_drop = float(base["logit_diff"]) - float(ctl["logit_diff"])
    substitution_shift = float(base["logit_diff"]) - float(sub["logit_diff"])
    cf_prob_gain = float(sub["p_counterfactual_target"]) - float(base["p_counterfactual_target"])
    specificity_gap = suppress_drop - random_drop
    validated = suppress_drop > 0.25 and specificity_gap > 0.25
    verdict = "validated" if validated else "not validated by this run"
    return {
        "suppress_drop": round(suppress_drop, 4),
        "random_drop": round(random_drop, 4),
        "substitution_shift": round(substitution_shift, 4),
        "cf_prob_gain": round(cf_prob_gain, 6),
        "specificity_gap": round(specificity_gap, 4),
        "validated": validated,
        "verdict": verdict,
    }


def write_graph_card(ctx, bundle, graph, interventions, rec_rows, n_para, ind_graph) -> None:
    fact = graph["fact"]
    inf = graph["influence"]
    base = next(r for r in interventions if r["condition"] == "baseline")
    sup = next(r for r in interventions if r["condition"] == "suppress_subject_supernode")
    sub = next(r for r in interventions if r["condition"] == "substitute_counterfactual")
    ctl = next(r for r in interventions if r["condition"] == "random_suppression_control")
    recurring = [r for r in rec_rows if r["n_prompts"] >= max(2, n_para - 1)]
    feat_nodes = [n for n in graph["nodes"] if n["kind"] == "feature"]
    effects = intervention_deltas(interventions)
    lines = [
        "# Graph card - attribution graph for one-hop factual recall",
        "",
        "The Lab 9 counterpart of Lab 6's circuit card. Same schema, different",
        "method: features instead of heads, automated attribution instead of",
        "manual screening, and a replacement model in the small print.",
        "",
        f"- **Task:** `{fact['prompt']}` → `{fact['target']}` (distractor `{fact['distractor']}`)",
        f"- **Model:** `{bundle.anatomy.model_id}` + {TC_REPO} (12-layer MLP transcoder stack)",
        f"- **Metric:** logit({fact['target']!r}) − logit({fact['distractor']!r}) = {graph['metric']:+.3f}",
        f"- **Graph:** {len(feat_nodes)} feature nodes kept (backward-flow selection), "
        f"{len(graph['edges'])} edges retained for display",
        f"- **Visibility:** kept nodes cover {inf['kept_coverage_of_feature_mass']:.0%} "
        f"of direct feature edge mass; error-node share is {inf['share_errors']:.0%}",
        "",
        "## Implied mechanism (HYPOTHESIS, written before the interventions)",
        "",
        f"Subject-token features (`{fact['subject']}` position) carry the country",
        "identity; mid/late-layer features there and at the final position",
        f"promote `{fact['target']}` directly. Input → subject features → say-target.",
        "",
        "## Intervention results (on the REAL model, not the replacement)",
        "",
        "| condition | logit diff | p(target) | top-1 |",
        "|---|---:|---:|---|",
        f"| baseline | {base['logit_diff']:+.2f} | {base['p_target']:.4f} | `{base['top1']}` |",
        f"| suppress subject supernode ({sup['n_features_edited']} feats) | {sup['logit_diff']:+.2f} "
        f"| {sup['p_target']:.4f} | `{sup['top1']}` |",
        f"| substitute counterfactual | {sub['logit_diff']:+.2f} | {sub['p_target']:.4f} | `{sub['top1']}` |",
        f"| random suppression control | {ctl['logit_diff']:+.2f} | {ctl['p_target']:.4f} | `{ctl['top1']}` |",
        "",
        f"Substitution pushes p(counterfactual target) to {sub['p_counterfactual_target']:.4f} "
        f"(baseline {base['p_counterfactual_target']:.5f}).",
        f"",
        f"**Intervention verdict:** {effects['verdict']} - suppression drop "
        f"{effects['suppress_drop']:+.2f} logits; random-control drop "
        f"{effects['random_drop']:+.2f}; specificity gap {effects['specificity_gap']:+.2f}.",
        "",
        "## Paraphrase robustness",
        "",
        f"- {len(recurring)} subject-site features recur in ≥{max(2, n_para - 1)} of {n_para} "
        "surface variants (mechanism candidates); single-prompt features are template artifacts.",
        "",
        "## Error-node accounting (the limitations line)",
        "",
        f"- Direct |edge| shares into the logit node: features {inf['share_features']:.0%}, "
        f"errors {inf['share_errors']:.0%}, embeddings {inf['share_embeddings']:.0%}.",
        f"- The kept nodes cover {inf['kept_coverage_of_feature_mass']:.0%} of the total feature edge mass.",
        "- Whatever routes through error nodes is computation the transcoders failed to",
        "  re-describe: the graph is an idealization whose residue is measured, not hidden.",
        "",
        "## Lab 6 vs Lab 9 (the confrontation)",
        "",
        f"- Signed ledger here: features contribute {inf['signed_contributions']['features']:+.2f} "
        f"of the {inf['metric_logit_diff']:+.2f} metric. On Lab 6's induction prompt, features "
        f"contribute only {ind_graph['influence']['signed_contributions']['features']:+.2f} of "
        f"{ind_graph['influence']['metric_logit_diff']:+.2f} - copied token embeddings "
        f"({ind_graph['influence']['signed_contributions']['embeddings']:+.2f}) and error nodes "
        f"({ind_graph['influence']['signed_contributions']['errors']:+.2f}) carry that behavior.",
        "- Lab 6's circuit card names attention heads and earns faithfulness/completeness",
        "  by mean-ablation; this graph names MLP features and earns its causality by",
        "  feature interventions. Each method is structurally blind where the other works:",
        "  the graph freezes attention into wiring; the head circuit treated MLPs as support.",
        "",
        "## What this card does NOT claim",
        "",
        "- The graph is computed on a replacement model; only the intervention rows are",
        "  claims about the real model.",
        "- Edges are direct, linearized attributions at one prompt (plus paraphrases);",
        "  they are not invariances over a prompt population.",
        "- Pruning thresholds and the node budget do silent work; the coverage and",
        "  error-share numbers above are the honest residue.",
        "",
    ]
    bench.write_text(ctx.path("graph_card.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("graph_card.md"), "summary",
                          "The deliverable: graph, mechanism hypothesis, interventions, limits.")


def build_claims(ctx, bundle, graph, interventions, rec_rows, n_para, ind_graph) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    fact = graph["fact"]
    inf = graph["influence"]
    base = next(r for r in interventions if r["condition"] == "baseline")
    sup = next(r for r in interventions if r["condition"] == "suppress_subject_supernode")
    sub = next(r for r in interventions if r["condition"] == "substitute_counterfactual")
    ctl = next(r for r in interventions if r["condition"] == "random_suppression_control")
    effects = intervention_deltas(interventions)
    if effects["validated"]:
        causal_text = (
            f"Suppressing the graph's subject supernode ({sup['n_features_edited']} features at the "
            f"'{fact['subject']}' position) on the REAL model drops the target logit diff from "
            f"{base['logit_diff']:+.2f} to {sup['logit_diff']:+.2f}; the random matched "
            f"suppression moves it to {ctl['logit_diff']:+.2f}, leaving a graph-specific gap of "
            f"{effects['specificity_gap']:+.2f} logits. Substituting counterfactual-country "
            f"features drives the diff to {sub['logit_diff']:+.2f} and raises p(counterfactual "
            f"capital) by {effects['cf_prob_gain']:+.3f}."
        )
        causal_falsifier = "The random matched control produces a comparable drop - the effect was generic perturbation."
    else:
        causal_text = (
            f"The graph-guided real-model intervention did NOT validate a specific subject "
            f"supernode in this run: suppression changed the target logit diff by "
            f"{-effects['suppress_drop']:+.2f} relative to baseline, while the random matched "
            f"control changed it by {-effects['random_drop']:+.2f}; specificity gap "
            f"{effects['specificity_gap']:+.2f}. This is causal evidence about the tested "
            f"intervention, but not a successful mechanism validation."
        )
        causal_falsifier = "A rerun with a better-labeled supernode and matched controls shows a large specificity gap."
    claims = [
        {
            "id": f"{LAB_ID}-C1", "tag": "ATTR",
            "text": (
                f"An attribution graph over a transcoder replacement model of {bundle.anatomy.model_id} "
                f"(exact to the real logits; error nodes absorb FVU) attributes the "
                f"'{fact['prompt']}'→'{fact['target']}' logit diff mostly to features at the subject "
                f"token: direct edge shares are features {inf['share_features']:.0%}, errors "
                f"{inf['share_errors']:.0%}, embeddings {inf['share_embeddings']:.0%}."
            ),
            "artifact": f"runs/{run_name}/graphs/pruned_graph.json",
            "falsifier": "The edge-reconstruction check fails on reruns, or the top edges vanish under a different node budget.",
        },
        {
            "id": f"{LAB_ID}-C2", "tag": "CAUSAL",
            "text": causal_text,
            "artifact": f"runs/{run_name}/tables/intervention_results.csv",
            "falsifier": causal_falsifier,
        },
        {
            "id": f"{LAB_ID}-C3", "tag": "OBS",
            "text": (
                f"In the signed edge ledger, features pay {inf['signed_contributions']['features']:+.2f} "
                f"of the fact's {inf['metric_logit_diff']:+.2f} logit diff but only "
                f"{ind_graph['influence']['signed_contributions']['features']:+.2f} of the induction "
                f"prompt's {ind_graph['influence']['metric_logit_diff']:+.2f}, where copied token "
                f"embeddings ({ind_graph['influence']['signed_contributions']['embeddings']:+.2f}) and "
                f"error nodes ({ind_graph['influence']['signed_contributions']['errors']:+.2f}) dominate: "
                f"the replacement model freezes attention, so a routing behavior shows up as embedding "
                f"mass moved by invisible wiring, not as feature structure - Lab 6's head circuit and "
                f"this graph see complementary slices of the mechanism."
            ),
            "artifact": f"runs/{run_name}/plots/influence_composition.png",
            "falsifier": "On reruns the induction metric's embedding/error contributions do not exceed its feature contribution, or QK-attribution variants show comparable feature ledgers for both behaviors.",
        },
    ]
    recurring = [r for r in rec_rows if r["n_prompts"] >= max(2, n_para - 1)]
    if rec_rows:
        claims.append({
            "id": f"{LAB_ID}-C4", "tag": "ATTR",
            "text": (
                f"{len(recurring)} subject-site features recur as kept graph nodes in at least "
                f"{max(2, n_para - 1)} of {n_para} paraphrases of the fact; the rest are "
                f"single-template artifacts - recurrence under paraphrase is the cheap robustness "
                f"screen the graph card requires before any feature is named in the mechanism."
            ),
            "artifact": f"runs/{run_name}/tables/paraphrase_robustness.csv",
            "falsifier": "Recurring features fail to recur on fresh paraphrases, or recur equally for unrelated facts.",
        })
    return claims


def write_summary(ctx, bundle, graph, interventions, rec_rows, n_para, ind_graph,
                  stack_report, gate_dropped, claims) -> None:
    inf = graph["influence"]
    fact = graph["fact"]
    base = next(r for r in interventions if r["condition"] == "baseline")
    sup = next(r for r in interventions if r["condition"] == "suppress_subject_supernode")
    sub = next(r for r in interventions if r["condition"] == "substitute_counterfactual")
    ctl = next(r for r in interventions if r["condition"] == "random_suppression_control")
    effects = intervention_deltas(interventions)
    lines = [
        "# Lab 9 run summary: attribution graphs and circuit tracing",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` + `{TC_REPO}` (full 12-layer MLP transcoder stack)",
        f"- primary fact: `{fact['prompt']}` → `{fact['target']}` (vs `{fact['distractor']}`)",
        "- evidence level: ATTR for the graph, CAUSAL only for the intervention rows",
        "",
        "## 1. What behavior was studied?",
        "",
        f"One-hop factual recall (Lab 5's domain) with logit-diff metric {graph['metric']:+.3f}; "
        f"{len(gate_dropped)} prompts dropped by the baseline gate (see tables/baseline_gate.csv).",
        "",
        "## 2. What internal object was measured?",
        "",
        f"- A LOCAL REPLACEMENT MODEL: frozen attention patterns + frozen LN denominators,",
        f"  MLPs replaced by transcoders (mean FVU {stack_report['mean_fvu']:.3f}) plus error nodes.",
        "  It reproduces the real logits exactly (diagnostics/replacement_exactness.json),",
        "  and is linear - so direct edges are well-defined and must sum to the metric",
        "  (diagnostics/edge_reconstruction_check.json).",
        f"- Backward-flow selection kept {len([n for n in graph['nodes'] if n['kind'] == 'feature'])} "
        f"feature nodes covering {inf['kept_coverage_of_feature_mass']:.0%} of feature edge mass.",
        "",
        "## 3. What intervention or control was used?",
        "",
        f"- suppress the subject supernode on the REAL model: {base['logit_diff']:+.2f} → {sup['logit_diff']:+.2f}",
        f"- substitute the counterfactual country's features: → {sub['logit_diff']:+.2f} "
        f"(p of counterfactual capital {sub['p_counterfactual_target']:.4f})",
        f"- random suppression of matched size: → {ctl['logit_diff']:+.2f} (the control that makes it causal)",
        f"- intervention verdict: {effects['verdict']} (specificity gap {effects['specificity_gap']:+.2f} logits)",
        "",
        "## 4. What metric changed?",
        "",
        f"- target-vs-distractor logit diff and p(target) ({base['p_target']:.4f} → {sup['p_target']:.4f} "
        "under suppression); see tables/intervention_results.csv.",
        "",
        "## 5. What claim is supported?",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## 6. What claim is NOT supported?",
        "",
        f"- {inf['share_errors']:.0%} of direct logit-edge mass routes through error nodes -",
        "  computation the transcoders did not re-describe. The graph explains the part of",
        "  the mechanism its dictionary can see, and the share is measured, not hidden.",
        "- Edges are linearized attributions on ONE replacement model at a handful of",
        "  prompts; no claim of invariance over a prompt population is made.",
        "- The induction vignette shows the instrument's blind spot, not a fact about",
        "  induction: frozen attention cannot appear as graph structure.",
        "",
        "## 7. What would falsify the interpretation?",
        "",
        "- Suppression effects matched by the random control on reruns/other facts.",
        "- Recurring paraphrase features failing on fresh surface forms.",
        "- A replacement model with lower-FVU transcoders attributing the behavior to",
        "  different features entirely (dictionary-dependence).",
        "",
        "## Reading order",
        "",
        "1. `graph_card.md` - the deliverable.",
        "2. `plots/attribution_graph.png` then `graphs/pruned_graph.json` - the object itself.",
        "3. `tables/intervention_results.csv`, `plots/intervention_effects.png` - the causal test.",
        "4. `plots/influence_composition.png` - Lab 6 vs Lab 9, one picture.",
        "5. `tables/paraphrase_robustness.csv` - what survives surface change.",
        "6. `tables/logit_edge_sources.csv`, `tables/node_budget_curve.csv`, and `tables/supernode_features.csv` - pruning and intervention audits.",
        "7. `diagnostics/` - exactness, edge-reconstruction, no-op checks, tokenization, and build manifest.",
        "",
    ]
    bench.write_text(ctx.path("run_summary.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("run_summary.md"), "summary", "The seven standard questions answered.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if bundle.torch_dtype != torch.float32:
        print("[lab9] WARNING: replacement-model exactness is calibrated for float32; "
              f"running in {bundle.torch_dtype} loosens every check.")
    bundle.model.requires_grad_(False)

    all_facts_for_audit = [PRIMARY_FACT, COUNTERFACTUAL_FACT, *PARAPHRASES, *COUNTERFACTUALS,
                           {**INDUCTION_VIGNETTE, "subject": INDUCTION_VIGNETTE["prompt"].split()[0]}]
    tok_rows = tokenization_report_rows(bundle, all_facts_for_audit)
    bench.write_csv_with_context(ctx, ctx.path("diagnostics", "tokenization_report.csv"), tok_rows)
    ctx.register_artifact(ctx.path("diagnostics", "tokenization_report.csv"), "diagnostic",
                          "Tokenizer audit for subjects, targets, distractors, and prompt positions.")
    prompt_manifest = {
        "backend": "inspectable_gpt2_dunefsky_transcoder_stack",
        "primary_fact": PRIMARY_FACT,
        "counterfactual_fact": COUNTERFACTUAL_FACT,
        "n_paraphrases_available": len(PARAPHRASES),
        "n_counterfactuals_available": len(COUNTERFACTUALS),
        "induction_vignette": INDUCTION_VIGNETTE,
        "node_budgets_by_tier": GRAPH_NODES_BY_TIER,
        "expand_depth": EXPAND_DEPTH,
        "edge_keep_per_target": EDGE_KEEP_PER_TARGET,
        "intervention_k": INTERVENTION_K,
        "evidence_contract": (
            "Graph edges are ATTR on a replacement model; only feature interventions "
            "run on the real model can support CAUSAL claims."
        ),
    }
    bench.write_json(ctx.path("diagnostics", "graph_build_manifest.json"), prompt_manifest)
    ctx.register_artifact(ctx.path("diagnostics", "graph_build_manifest.json"), "diagnostic",
                          "Lab 9 backend assumptions, budgets, prompt set, and evidence contract.")

    # ----- transcoder stack + reconstruction report ---------------------------
    tcs = load_transcoder_stack(bundle)
    cap0 = real_pass(bundle, PRIMARY_FACT["prompt"])
    fvus = []
    for layer, blk in enumerate(bundle.blocks):
        mid = cap0["ln2_in"][layer]
        bare = (mid - mid.mean(-1, keepdim=True)) / _sigma(mid, blk.ln_2.eps)
        feats = torch.relu(tcs.encode_pre(layer, bare))
        recon = feats @ tcs.w_dec(layer) + tcs.layers[layer]["b_dec"]
        t = cap0["mlp_out"][layer]
        fvu = float(((t - recon) ** 2).sum() / ((t - t.mean(0, keepdim=True)) ** 2).sum())
        fvus.append({"layer": layer, "fvu": round(fvu, 4),
                     "l0": round(float((feats > 0).float().sum(-1).mean()), 1)})
    stack_report = {"repo": TC_REPO, "n_layers": tcs.n_layers, "d_sae": tcs.d_sae,
                    "per_layer": fvus, "mean_fvu": round(sum(r["fvu"] for r in fvus) / len(fvus), 4)}
    bench.write_json(ctx.path("transcoder_stack_report.json"), stack_report)
    bench.write_json(ctx.path("diagnostics", "transcoder_stack_report.json"), stack_report)
    bench.write_csv_with_context(ctx, ctx.path("tables", "transcoder_reconstruction_by_layer.csv"), fvus)
    ctx.register_artifact(ctx.path("transcoder_stack_report.json"), "metrics",
                          "Per-layer transcoder FVU/L0 on the primary prompt.")
    ctx.register_artifact(ctx.path("diagnostics", "transcoder_stack_report.json"), "diagnostic",
                          "Copy of the transcoder reconstruction report inside diagnostics.")
    ctx.register_artifact(ctx.path("tables", "transcoder_reconstruction_by_layer.csv"), "table",
                          "Per-layer transcoder reconstruction quality and sparsity.")
    print(f"[lab9] transcoder stack mean FVU {stack_report['mean_fvu']} "
          f"(worst layer {max(fvus, key=lambda r: r['fvu'])['fvu']})")

    # ----- baseline gate -------------------------------------------------------
    n_para = max(2, args.max_examples) if args.max_examples > 0 else len(PARAPHRASES) + 1
    battery = [PRIMARY_FACT] + PARAPHRASES[: n_para - 1]
    gate_kept, gate_dropped = baseline_gate(bundle, battery + [COUNTERFACTUAL_FACT] + COUNTERFACTUALS)
    bench.write_csv_with_context(ctx, ctx.path("tables", "baseline_gate.csv"),
                                 gate_kept + gate_dropped)
    ctx.register_artifact(ctx.path("tables", "baseline_gate.csv"), "table",
                          "Baseline logit-diffs; prompts the model fails are dropped with a reason.")
    kept_ids = {f["id"] for f in gate_kept}
    if PRIMARY_FACT["id"] not in kept_ids or COUNTERFACTUAL_FACT["id"] not in kept_ids:
        raise RuntimeError("The model fails the primary or counterfactual fact at baseline; "
                           "there is no behavior to trace. See tables/baseline_gate.csv.")
    para_facts = [f for f in gate_kept if f["id"] == "france" or f["id"].startswith("para_")]
    print(f"[lab9] baseline gate: {len(gate_kept)} kept, {len(gate_dropped)} dropped; "
          f"{len(para_facts)} paraphrases in the battery")

    # ----- self-checks ---------------------------------------------------------
    run_replacement_exactness_check(ctx, bundle, tcs, cap0)
    run_feature_edit_noop_check(ctx, bundle, tcs, PRIMARY_FACT)

    # ----- the primary graph ----------------------------------------------------
    requested_graph_nodes = int(_arg(args, "graph_nodes", 0) or 0)
    node_budget = requested_graph_nodes if requested_graph_nodes > 0 else GRAPH_NODES_BY_TIER.get(args.tier, 24)
    print(f"[lab9] building primary attribution graph (node budget {node_budget})")
    graph = build_graph(ctx, bundle, tcs, cap0, node_budget=node_budget,
                        fact=PRIMARY_FACT, check_reconstruction=True)
    bench.write_json(ctx.path("graphs", "pruned_graph.json"),
                     {"fact": PRIMARY_FACT, "metric_logit_diff": graph["metric"],
                      "node_budget": node_budget, "expand_depth": EXPAND_DEPTH,
                      "influence": graph["influence"],
                      "nodes": [n for n in graph["nodes"] if n["kind"] != "error" or
                                any(e["source"] == n["name"] for e in graph["edges"])],
                      "edges": graph["edges"],
                      "top_logit_sources": graph["edge_sources"]})
    ctx.register_artifact(ctx.path("graphs", "pruned_graph.json"), "results",
                          "The pruned attribution graph: nodes, edges, influence accounting.")

    # supernode map: auto-proposed groupings the student is expected to edit
    subj = subject_position(bundle, PRIMARY_FACT)
    supernodes = {"subject": [], "say_target": [], "other": []}
    for n in graph["nodes"]:
        if n["kind"] != "feature":
            continue
        entry = {"name": n["name"], "layer": n["layer"], "feature": n["feature"],
                 "token": n["token"], "promotes": n["promotes"]}
        if n["pos"] == subj:
            supernodes["subject"].append(entry)
        elif n["pos"] == graph["fr"]["seq"] - 1 and PRIMARY_FACT["target"].strip() in " ".join(n["promotes"]):
            supernodes["say_target"].append(entry)
        else:
            supernodes["other"].append(entry)
    bench.write_json(ctx.path("graphs", "supernode_map.json"),
                     {"note": "Auto-proposed groupings. Edit before citing: validate each feature's "
                              "membership by its top contexts, not its de-embedding alone.",
                      "supernodes": supernodes})
    ctx.register_artifact(ctx.path("graphs", "supernode_map.json"), "results",
                          "Auto-proposed supernode groupings (subject / say-target / other).")

    # node table for humans
    feat_rows = [{k: v for k, v in n.items() if k != "name"} | {"name": n["name"]}
                 for n in graph["nodes"] if n["kind"] == "feature"]
    for r in feat_rows:
        r["promotes"] = " | ".join(r["promotes"])
    bench.write_csv_with_context(ctx, ctx.path("tables", "graph_nodes.csv"), feat_rows)
    ctx.register_artifact(ctx.path("tables", "graph_nodes.csv"), "table",
                          "Kept feature nodes with activations, peak tokens, and de-embeddings.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "graph_edges.csv"), graph["edges"])
    ctx.register_artifact(ctx.path("tables", "graph_edges.csv"), "table",
                          "Retained direct-attribution edges (source, target, signed weight).")
    bench.write_csv_with_context(ctx, ctx.path("tables", "logit_edge_sources.csv"), graph["edge_sources"])
    ctx.register_artifact(ctx.path("tables", "logit_edge_sources.csv"), "table",
                          "Largest raw direct sources into the logit node before display pruning.")
    bench.write_csv_with_context(ctx, ctx.path("tables", "node_budget_curve.csv"),
                                 graph["influence"]["budget_curve"])
    ctx.register_artifact(ctx.path("tables", "node_budget_curve.csv"), "table",
                          "How much direct feature mass is covered by top-N logit-edge nodes.")

    # ----- interventions on the real model -------------------------------------
    print("[lab9] running graph-guided interventions on the real model")
    interventions = run_interventions(ctx, bundle, tcs, graph, PRIMARY_FACT,
                                      COUNTERFACTUAL_FACT, args.seed)
    bench.write_csv_with_context(ctx, ctx.path("tables", "intervention_results.csv"), interventions)
    ctx.register_artifact(ctx.path("tables", "intervention_results.csv"), "table",
                          "Suppression / substitution / random-control intervention outcomes.")
    bench.write_csv_with_context(ctx, ctx.path("results.csv"), interventions)
    ctx.register_artifact(ctx.path("results.csv"), "results",
                          "Alias of intervention_results.csv for the run contract.")
    for r in interventions:
        print(f"[lab9]   {r['condition']:34s} ld={r['logit_diff']:+6.2f} p(target)={r['p_target']:.4f}")

    # ----- paraphrase battery ---------------------------------------------------
    print(f"[lab9] paraphrase battery: {len(para_facts)} prompts")
    para_graphs = [graph]
    for fact in para_facts:
        if fact["id"] == PRIMARY_FACT["id"]:
            continue
        capp = real_pass(bundle, fact["prompt"])
        para_graphs.append(build_graph(ctx, bundle, tcs, capp,
                                       node_budget=max(10, node_budget // 2), fact=fact))
    rec_rows = paraphrase_recurrence(para_graphs, bundle)
    bench.write_csv_with_context(ctx, ctx.path("tables", "paraphrase_robustness.csv"), rec_rows)
    ctx.register_artifact(ctx.path("tables", "paraphrase_robustness.csv"), "table",
                          "Subject-site feature recurrence across surface variants.")

    # ----- the induction vignette (Lab 6 comparison) ----------------------------
    print("[lab9] induction vignette (the Lab 6 confrontation)")
    cap_ind = real_pass(bundle, INDUCTION_VIGNETTE["prompt"])
    ind_graph = build_graph(ctx, bundle, tcs, cap_ind, node_budget=max(10, node_budget // 2),
                            fact={**INDUCTION_VIGNETTE, "subject": INDUCTION_VIGNETTE["prompt"].split()[0]})
    bench.write_json(ctx.path("graphs", "induction_graph.json"),
                     {"vignette": INDUCTION_VIGNETTE, "metric_logit_diff": ind_graph["metric"],
                      "influence": ind_graph["influence"]})
    ctx.register_artifact(ctx.path("graphs", "induction_graph.json"), "results",
                          "Influence accounting for Lab 6's induction prompt under this instrument.")
    ledger_rows = influence_ledger_rows("capital_fact", graph["influence"]) + \
        influence_ledger_rows("induction_vignette", ind_graph["influence"])
    bench.write_csv_with_context(ctx, ctx.path("tables", "influence_ledger.csv"), ledger_rows)
    ctx.register_artifact(ctx.path("tables", "influence_ledger.csv"), "table",
                          "Signed and absolute-share influence accounting for fact vs induction prompts.")
    print(f"[lab9]   signed feature contribution: fact "
          f"{graph['influence']['signed_contributions']['features']:+.2f}/"
          f"{graph['influence']['metric_logit_diff']:+.2f} vs induction "
          f"{ind_graph['influence']['signed_contributions']['features']:+.2f}/"
          f"{ind_graph['influence']['metric_logit_diff']:+.2f}")

    # ----- plots -----------------------------------------------------------------
    if not args.no_plots:
        plot_graph(ctx, bundle, graph, "attribution_graph.png",
                   f"{PRIMARY_FACT['prompt']!r} → {PRIMARY_FACT['target']!r}")
        plot_influence_composition(ctx, graph["influence"], ind_graph["influence"])
        plot_edge_mass_shares(ctx, graph["influence"], ind_graph["influence"])
        plot_interventions(ctx, interventions, PRIMARY_FACT)
        if rec_rows:
            plot_paraphrase_recurrence(ctx, rec_rows, len(para_graphs))

    # ----- metrics, card, claims, summary ----------------------------------------
    metrics = {
        "model_id": bundle.anatomy.model_id,
        "transcoder_mean_fvu": stack_report["mean_fvu"],
        "metric_logit_diff": graph["metric"],
        "node_budget": node_budget,
        "n_feature_nodes": len([n for n in graph["nodes"] if n["kind"] == "feature"]),
        "signed_contributions": graph["influence"]["signed_contributions"],
        "share_features": graph["influence"]["share_features"],
        "share_errors": graph["influence"]["share_errors"],
        "share_embeddings": graph["influence"]["share_embeddings"],
        "kept_coverage_of_feature_mass": graph["influence"]["kept_coverage_of_feature_mass"],
        "intervention": {r["condition"]: r["logit_diff"] for r in interventions},
        "intervention_effects": intervention_deltas(interventions),
        "induction_signed_contributions": ind_graph["influence"]["signed_contributions"],
        "induction_metric_logit_diff": ind_graph["influence"]["metric_logit_diff"],
        "n_paraphrases": len(para_graphs),
        "n_recurring_features": len([r for r in rec_rows
                                     if r["n_prompts"] >= max(2, len(para_graphs) - 1)]),
        "baseline_gate_dropped": len(gate_dropped),
    }
    bench.write_json(ctx.path("metrics.json"), metrics)
    ctx.register_artifact(ctx.path("metrics.json"), "metrics", "Aggregate Lab 9 metrics.")

    write_graph_card(ctx, bundle, graph, interventions, rec_rows, len(para_graphs), ind_graph)
    claims = build_claims(ctx, bundle, graph, interventions, rec_rows, len(para_graphs), ind_graph)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    write_summary(ctx, bundle, graph, interventions, rec_rows, len(para_graphs), ind_graph,
                  stack_report, gate_dropped, claims)
    print(f"[lab9] wrote graph_card.md, run_summary.md, and {len(claims)} drafted ledger claims")
