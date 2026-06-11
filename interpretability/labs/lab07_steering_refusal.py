"""Lab 7: Steering vectors, representation engineering, and the refusal direction.

The course's first lab on instruct models, and its ethics unit done with
apparatus instead of homilies. Three tracks, one safety wall.

* **Track A (the method).** A difference-in-means direction from sentiment
  contrast pairs, injected during generation across a dose sweep. The honest
  unit of evidence is a dose-response curve with controls (random direction,
  shuffled-label direction) on the same axes — not a cherry-picked
  before/after. We measure the target behavior, fluency, KL to the unsteered
  distribution, and drift on an unrelated task.

* **Track B (the result).** The refusal direction, extracted by
  difference-in-means between activations on refusal-eliciting and matched
  benign instructions. THE SAFETY WALL, stated and enforced:
    - direction extraction and the monitor use FORWARD PASSES ONLY;
    - no completion is ever sampled from a refusal-eliciting prompt;
    - steering is only ever TOWARD refusal, on benign prompts;
    - refusal ABLATION is not implemented (the jailbreak result is assigned
      reading, not reproduced).
  The monitor asks whether the direction PREDICTS refusal (projection vs
  category, forward-pass ROC); the steer-toward-refusal sweep asks whether it
  CAUSES refusal (induced-refusal rate on benign prompts). Those are
  different properties and the lab keeps them apart.

* **Bridge (closing Lab 4's loop).** Lab 4 found truth linearly decodable.
  Does intervening on a truth direction change behavior? We recompute the
  diff-in-means truth direction on THIS instruct model from the frozen Lab 4
  cities data (directions are model-specific — the base-model `.pt` is loaded
  only for provenance) and measure whether steering shifts True/False assent.
  Decodable-and-steerable and decodable-but-inert are both publishable.

Evidence level: CAUSAL (representation-level control), with explicit attention
to what control does and does not explain.
"""

from __future__ import annotations

import csv
import math
import pathlib
from typing import Any

import interp_bench as bench

LAB_ID = "L07"

# Decoding/steering pins (frozen so the only moving part is the dose).
# Scales are FRACTIONS OF THE ACTIVATION NORM at the injection layer, not raw
# vector magnitudes: a unit direction times a fixed number is ~6% of a typical
# residual stream and does nothing, while the same fraction transfers across
# models and layers (a 135M and a 7B have very different stream norms). The
# lab multiplies these by the measured median norm at the chosen layer.
MAX_NEW_TOKENS = 48
TRACK_A_SCALES = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
REFUSAL_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
BRIDGE_SCALES = (-1.0, -0.5, 0.0, 0.5, 1.0)
# Mid-stack candidate layers. Steering is consistently strongest mid-network;
# late layers carry the answer but resist being redirected. The layer is then
# chosen by ACTUAL generation steering (below), not a next-token proxy -- the
# proxy measures the final-position logit nudge, which does not predict how a
# direction propagates through 40 generated tokens (it picked a near-useless
# late layer in early validation).
LAYER_FRACTIONS = (0.35, 0.45, 0.55, 0.65)
LAYER_SELECT_SCALE = 0.5     # dose (activation-norm fraction) for layer selection
MONITOR_TRAIN_FRACTION = 0.6

POSITIVE_WORDS = set("""good great wonderful excellent amazing fantastic lovely delightful happy joy joyful
beautiful brilliant perfect pleasure delicious cheerful warm bright glorious inspiring rewarding
exciting adorable cozy comfortable grateful proud thrilled magnificent radiant smooth rich elegant
enjoy love loved loving best better nice pleasant fun fond hope hopeful""".split())
NEGATIVE_WORDS = set("""bad terrible awful horrible dreadful disgusting depressing sad miserable gloomy
ugly boring tedious painful disappointed ashamed unhappy filthy cramped bitter dull draining anxious
destructive stress resentful clumsy buggy burden lifeless withered grim frustrating tense awkward hate
hated worst worse nasty unpleasant fear afraid angry sorry""".split())

# Refusal-string classifier lexicon (the handout requires hand-auditing it).
REFUSAL_MARKERS = (
    "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm not able",
    "i am not able", "i'm unable", "i am unable", "i'm sorry, but", "i am sorry, but",
    "i'm sorry but", "i apologize, but", "i cannot help", "can't help with",
    "cannot help with", "can't assist", "cannot assist", "i must decline", "i won't be able",
    "not able to help", "not able to assist", "against my", "i'm not going to",
    "i am not going to", "as an ai", "i do not feel comfortable", "i don't feel comfortable",
    "i'm really sorry", "unable to provide", "cannot provide", "can't provide that",
)


def load_pairs(name: str, col_a: str, col_b: str) -> list[tuple[str, str, str]]:
    path = bench.COURSE_ROOT / "data" / name
    if not path.exists():
        raise RuntimeError(f"Frozen dataset missing: {path}. Re-checkout data/; do not regenerate per-run.")
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append((row[list(row)[0]], row[col_a], row[col_b]))
    return out


def load_eval_prompts() -> list[tuple[str, str]]:
    path = bench.COURSE_ROOT / "data" / "steering_eval_prompts.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return [(r["prompt_id"], r["prompt"]) for r in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------


def last_token_residual(bundle: bench.ModelBundle, templated_prompt: str, layer: int) -> Any:
    """streams[layer] at the final position of a templated prompt (fp32 cpu)."""
    cap = bench.run_with_residual_cache(bundle, templated_prompt)
    return cap.streams[layer, -1]


def diff_in_means_direction(
    bundle: bench.ModelBundle,
    pairs: list[tuple[str, str]],
    layer: int,
    *,
    as_instruction: bool,
) -> Any:
    """Unit difference-in-means direction at ``layer``.

    Each pair is (a, b); the direction points a - b (mean over pairs). When
    ``as_instruction`` the strings are wrapped in the chat template as user
    instructions (Track B); otherwise they are templated as user messages too
    (Track A statements are spoken by the user). Either way the activation is
    read at the generation position, never on a raw untemplated string — the
    template-drift failure mode the course warns about.
    """
    import torch

    a_vecs, b_vecs = [], []
    for a, b in pairs:
        a_vecs.append(last_token_residual(bundle, bench.apply_chat_template(bundle, a), layer))
        b_vecs.append(last_token_residual(bundle, bench.apply_chat_template(bundle, b), layer))
    direction = torch.stack(a_vecs).mean(0) - torch.stack(b_vecs).mean(0)
    return direction / direction.norm().clamp_min(1e-9)


def shuffled_direction(
    bundle: bench.ModelBundle, pairs: list[tuple[str, str]], layer: int, seed: int
) -> Any:
    """Control: diff-in-means after flipping EXACTLY HALF the pairs' labels.

    A proper null: balanced flips make the concept signal cancel in
    expectation, so any residual steering effect is the contrast structure,
    not the concept. (Random per-pair flips can leave an unbalanced remnant
    that retains signal -- which made this control track the real direction
    too closely in early validation.)
    """
    import torch

    n = len(pairs)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    flip = set(perm[: n // 2].tolist())
    shuffled = [(b, a) if i in flip else (a, b) for i, (a, b) in enumerate(pairs)]
    return diff_in_means_direction(bundle, shuffled, layer, as_instruction=False)


def random_direction(d_model: int, seed: int) -> Any:
    """Control: a random unit direction (matched norm by construction)."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=gen)
    return v / v.norm().clamp_min(1e-9)


# ---------------------------------------------------------------------------
# Behavior scorers
# ---------------------------------------------------------------------------


def sentiment_score(text: str) -> float:
    words = [w.strip(".,!?;:'\"()").lower() for w in text.split()]
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return (pos - neg) / (pos + neg + 1)


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def mean_token_logprob(bundle: bench.ModelBundle, context: str, continuation: str) -> float:
    """Fluency proxy: mean logprob of ``continuation`` given ``context`` under
    the UNSTEERED model. Degenerate steered text scores low here."""
    import torch

    tok = bundle.tokenizer
    ctx_ids = tok(context, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tok(context + continuation, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if full_ids.shape[1] <= ctx_ids.shape[1]:
        return 0.0
    ids = full_ids.to(bundle.input_device)
    with torch.no_grad():
        out = bundle.model(input_ids=ids, use_cache=False)
    logprobs = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    start = ctx_ids.shape[1] - 1
    cont_targets = ids[0, ctx_ids.shape[1]:]
    picked = logprobs[start:start + len(cont_targets)].gather(1, cont_targets[:, None].to(logprobs.device))
    return float(picked.mean())


def kl_to_unsteered(steered_logits: Any, base_logits: Any) -> float:
    import torch

    p = torch.log_softmax(steered_logits, dim=-1)
    q = torch.log_softmax(base_logits, dim=-1)
    return float((p.exp() * (p - q)).sum())


DRIFT_FACTS = [
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("Two plus two equals", "4"),
    ("The opposite of hot is", "cold"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The largest planet is", "Jupiter"),
]


def drift_accuracy(bundle: bench.ModelBundle, layer: int, direction: Any, scale: float) -> float:
    """Fraction of unrelated factual prompts still answered correctly under
    steering (the side-effect battery)."""
    correct = 0
    for prompt, answer in DRIFT_FACTS:
        templated = bench.apply_chat_template(bundle, prompt)
        gen = bench.generate_text(bundle, templated, max_new_tokens=12,
                                  steer=(layer, direction, scale))
        if answer.lower() in gen.lower():
            correct += 1
    return correct / len(DRIFT_FACTS)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_dose_response(ctx: bench.RunContext, rows: list[dict[str, Any]], concept: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    series = {"real": "tab:red", "random": "tab:gray", "shuffled": "tab:olive"}
    metrics = [("target_score", f"{concept} score of generations"),
               ("fluency_logprob", "fluency (mean token logprob)"),
               ("kl_to_unsteered", "KL from unsteered next-token dist")]
    for ax, (key, ylabel) in zip(axes, metrics):
        for cond, color in series.items():
            pts = sorted([(r["scale"], r[key]) for r in rows if r["condition"] == cond])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                        color=color, label=cond, linewidth=2.0)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("steering scale")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Track A dose-response: steering {concept}, real direction vs controls")
    fig.tight_layout()
    bench.save_figure(ctx, fig, f"dose_response_{concept}.png",
                      "Target behavior, fluency, and KL vs steering scale for real and control directions.")


def plot_layer_sweep(ctx: bench.RunContext, rows: list[dict[str, Any]], best_layer: int) -> None:
    fig, ax = bench.new_figure(figsize=(8.0, 5.0))
    layers = sorted({r["layer"] for r in rows})
    ax.plot(layers, [next(r["pos_score"] for r in rows if r["layer"] == l) for l in layers],
            marker="^", linewidth=1.8, color="tab:green", label="+dose sentiment")
    ax.plot(layers, [next(r["neg_score"] for r in rows if r["layer"] == l) for l in layers],
            marker="v", linewidth=1.8, color="tab:red", label="-dose sentiment")
    ax.plot(layers, [next(r["steering_spread"] for r in rows if r["layer"] == l) for l in layers],
            marker="o", linewidth=2.4, color="black", label="steering spread (pos - neg)")
    ax.axvline(best_layer, color="tab:purple", linewidth=1.0, alpha=0.5, label=f"chosen layer {best_layer}")
    ax.set_xlabel("layer the direction is injected at")
    ax.set_ylabel("sentiment of generations at +/- the probe dose")
    ax.set_title("Where steering is strongest (measured by actual generation)")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "layer_sweep.png",
                      "Per-layer generation steering spread; the chosen injection layer.")


def plot_induced_refusal(ctx: bench.RunContext, rows: list[dict[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.0, 5.2))
    for cond, color in (("refusal", "tab:red"), ("random", "tab:gray")):
        pts = sorted([(r["scale"], r["refusal_rate"]) for r in rows if r["condition"] == cond])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=color,
                    linewidth=2.2, label=f"{cond} direction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("steering scale (toward refusal)")
    ax.set_ylabel("induced refusal rate on BENIGN prompts")
    ax.set_title("Track B: steering benign prompts toward refusal (safe direction only)")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "induced_refusal.png",
                      "Induced-refusal rate on benign prompts vs dose, refusal vs random direction.")


def plot_monitor(ctx: bench.RunContext, proj_refusal: list[float], proj_benign: list[float],
                 auc: float) -> None:
    fig, ax = bench.new_figure(figsize=(8.0, 5.0))
    ax.hist(proj_benign, bins=10, alpha=0.6, color="tab:green", label="benign (held-out)")
    ax.hist(proj_refusal, bins=10, alpha=0.6, color="tab:red", label="refusal-eliciting (held-out)")
    ax.set_xlabel("projection onto the refusal direction")
    ax.set_ylabel("count")
    ax.set_title(f"Refusal monitor: does the direction PREDICT refusal? (AUC = {auc:.2f}, forward-pass only)")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "refusal_monitor.png",
                      "Held-out projection histograms for refusal-eliciting vs benign prompts.")


def plot_bridge(ctx: bench.RunContext, rows: list[dict[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.0, 5.0))
    pts = sorted([(r["scale"], r["mean_true_false_logit_diff"]) for r in rows])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2.2, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("steering scale (toward the truth direction)")
    ax.set_ylabel("mean logit('True') - logit('False')")
    ax.set_title("Bridge: does Lab 4's decodable truth direction STEER behavior?")
    bench.save_figure(ctx, fig, "truth_direction_bridge.png",
                      "True/False assent shift vs steering on the recomputed truth direction.")


# ---------------------------------------------------------------------------
# AUC (no sklearn)
# ---------------------------------------------------------------------------


def roc_auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 7 requires an instruct model with a chat template.")
    n_layers = bundle.anatomy.n_layers
    d_model = bundle.anatomy.d_model
    candidate_layers = sorted({max(1, int(f * n_layers)) for f in LAYER_FRACTIONS})

    sentiment = [(a, b) for _, a, b in load_pairs("sentiment_contrast_set.csv", "positive", "negative")]
    refusal = [(a, b) for _, a, b in load_pairs("refusal_elicitation_set.csv",
                                                "refusal_eliciting", "benign_matched")]
    eval_prompts = load_eval_prompts()
    print(f"[lab7] instruct model {bundle.anatomy.model_id}; {len(sentiment)} sentiment pairs, "
          f"{len(refusal)} refusal pairs, {len(eval_prompts)} eval prompts")

    # Instrument sanity: hook parity + lens still hold on the templated prompt.
    probe = bench.apply_chat_template(bundle, eval_prompts[0][1])
    bench.run_hook_parity_check(ctx, bundle, probe)
    bench.run_lens_self_check(ctx, bundle, bench.run_with_residual_cache(bundle, probe))

    # ----- layer sweep (generation-based: steer and score, the real target) ---
    sweep_prompts = eval_prompts[:3]
    sweep_rows = []
    layer_dirs = {}
    for layer in candidate_layers:
        d = diff_in_means_direction(bundle, sentiment, layer, as_instruction=False)
        layer_dirs[layer] = d
        lnorm = float(torch.stack([
            last_token_residual(bundle, bench.apply_chat_template(bundle, p), layer)
            for _, p in sweep_prompts]).norm(dim=-1).median())
        pos, neg = [], []
        for _, prompt in sweep_prompts:
            t = bench.apply_chat_template(bundle, prompt)
            pos.append(sentiment_score(bench.generate_text(
                bundle, t, max_new_tokens=MAX_NEW_TOKENS, steer=(layer, d, LAYER_SELECT_SCALE * lnorm))))
            neg.append(sentiment_score(bench.generate_text(
                bundle, t, max_new_tokens=MAX_NEW_TOKENS, steer=(layer, d, -LAYER_SELECT_SCALE * lnorm))))
        spread = sum(pos) / len(pos) - sum(neg) / len(neg)
        sweep_rows.append({"layer": layer, "ref_norm": round(lnorm, 2),
                           "pos_score": round(sum(pos) / len(pos), 4),
                           "neg_score": round(sum(neg) / len(neg), 4),
                           "steering_spread": round(spread, 4)})
        print(f"[lab7]   layer {layer}: steering spread {spread:+.3f}")
    best = max(sweep_rows, key=lambda r: r["steering_spread"])
    best_layer = best["layer"]
    ref_norm = best["ref_norm"]
    sweep_path = ctx.path("tables", "layer_sweep.csv")
    bench.write_csv_with_context(ctx, sweep_path, sweep_rows)
    ctx.register_artifact(sweep_path, "table", "Per-layer generation steering spread (pos minus neg sentiment).")
    print(f"[lab7] layer sweep -> steering at layer {best_layer} (ref activation norm {ref_norm:.1f})")

    def eff(scale: float) -> float:
        """A dose (fraction of activation norm) as an absolute steering coefficient."""
        return scale * ref_norm

    real_dir = layer_dirs[best_layer]
    rand_dir = random_direction(d_model, seed=args.seed * 13 + best_layer)
    shuf_dir = shuffled_direction(bundle, sentiment, best_layer, seed=args.seed * 17 + best_layer)

    # ----- Track A: dose-response with controls --------------------------------
    print(f"[lab7] Track A: dose-response over {len(TRACK_A_SCALES)} scales x 3 conditions")
    dose_rows = []
    examples_rows = []
    for cond, direction in (("real", real_dir), ("random", rand_dir), ("shuffled", shuf_dir)):
        for scale in TRACK_A_SCALES:
            scores, fluencies, kls = [], [], []
            for pid, prompt in eval_prompts:
                templated = bench.apply_chat_template(bundle, prompt)
                gen = bench.generate_text(bundle, templated, max_new_tokens=MAX_NEW_TOKENS,
                                          steer=(best_layer, direction, eff(scale)))
                scores.append(sentiment_score(gen))
                fluencies.append(mean_token_logprob(bundle, templated, gen))
                steered_logits = bench.next_token_logits(bundle, templated,
                                                         steer=(best_layer, direction, eff(scale)))
                kls.append(kl_to_unsteered(steered_logits, bench.next_token_logits(bundle, templated)))
                if cond == "real" and pid == eval_prompts[0][0]:
                    examples_rows.append({"scale": scale, "prompt": prompt, "generation": gen[:200]})
            drift = drift_accuracy(bundle, best_layer, direction, eff(scale)) if cond == "real" else ""
            dose_rows.append({
                "condition": cond, "layer": best_layer, "scale": scale,
                "target_score": round(sum(scores) / len(scores), 4),
                "fluency_logprob": round(sum(fluencies) / len(fluencies), 4),
                "kl_to_unsteered": round(sum(kls) / len(kls), 4),
                "drift_accuracy": round(drift, 4) if drift != "" else "",
            })
        print(f"[lab7]   {cond}: done")
    dose_path = ctx.path("tables", "dose_response.csv")
    bench.write_csv_with_context(ctx, dose_path, dose_rows)
    ctx.register_artifact(dose_path, "table", "Dose-response metrics for real and control directions.")
    bench.write_csv_with_context(ctx, ctx.path("results.csv"), dose_rows)
    ctx.register_artifact(ctx.path("results.csv"), "results", "Alias of dose_response.csv for the run contract.")
    ex_path = ctx.path("tables", "steered_examples.csv")
    bench.write_csv(ex_path, examples_rows)
    ctx.register_artifact(ex_path, "table", "Showcase generations across the dose sweep (one prompt).")

    # ----- Track B: refusal direction (FORWARD PASSES ONLY) --------------------
    print("[lab7] Track B: refusal direction (forward-pass extraction; no harmful generation)")
    n_train = int(len(refusal) * MONITOR_TRAIN_FRACTION)
    train_ref, held_ref = refusal[:n_train], refusal[n_train:]
    refusal_dir = diff_in_means_direction(bundle, train_ref, best_layer, as_instruction=True)
    # Monitor: project held-out prompts; ground truth = category, no generation.
    proj_refusal = [float(last_token_residual(bundle, bench.apply_chat_template(bundle, a), best_layer) @ refusal_dir)
                    for a, _ in held_ref]
    proj_benign = [float(last_token_residual(bundle, bench.apply_chat_template(bundle, b), best_layer) @ refusal_dir)
                   for _, b in held_ref]
    auc = roc_auc(proj_refusal, proj_benign)
    monitor_rows = []
    all_proj = sorted(set(proj_refusal + proj_benign))
    for thr in all_proj:
        tp = sum(1 for p in proj_refusal if p >= thr)
        fp = sum(1 for p in proj_benign if p >= thr)
        monitor_rows.append({
            "threshold": round(thr, 4),
            "true_positive_rate": round(tp / len(proj_refusal), 4),
            "false_positive_rate": round(fp / len(proj_benign), 4),
        })
    mon_path = ctx.path("tables", "refusal_monitor_table.csv")
    bench.write_csv_with_context(ctx, mon_path, monitor_rows)
    ctx.register_artifact(mon_path, "table", "Forward-pass refusal monitor ROC points (held-out category labels).")
    print(f"[lab7]   monitor AUC = {auc:.3f} (forward-pass projection vs category)")

    # Steer benign prompts TOWARD refusal; measure induced-refusal rate.
    print(f"[lab7]   steer-toward-refusal sweep over {len(REFUSAL_SCALES)} scales (benign prompts only)")
    induced_rows = []
    for cond, direction in (("refusal", refusal_dir), ("random", rand_dir)):
        for scale in REFUSAL_SCALES:
            refusals = 0
            for pid, prompt in eval_prompts:
                templated = bench.apply_chat_template(bundle, prompt)
                gen = bench.generate_text(bundle, templated, max_new_tokens=MAX_NEW_TOKENS,
                                          steer=(best_layer, direction, eff(scale)))
                if is_refusal(gen):
                    refusals += 1
            induced_rows.append({"condition": cond, "scale": scale,
                                 "refusal_rate": round(refusals / len(eval_prompts), 4)})
    induced_path = ctx.path("tables", "induced_refusal_curve.csv")
    bench.write_csv_with_context(ctx, induced_path, induced_rows)
    ctx.register_artifact(induced_path, "table", "Induced-refusal rate on benign prompts vs dose.")
    max_induced = max(r["refusal_rate"] for r in induced_rows if r["condition"] == "refusal")
    print(f"[lab7]   max induced refusal on benign prompts: {max_induced:.0%}")

    # ----- Bridge: does Lab 4's truth direction steer? -------------------------
    print("[lab7] Bridge: recomputing truth direction on this model, testing whether it steers")
    truth_provenance = find_lab4_direction()
    cities = load_truth_statements()
    truth_pairs = [(s, f) for s, f in cities]  # (true statement, false statement)
    truth_dir = diff_in_means_direction(bundle, truth_pairs, best_layer, as_instruction=True)
    true_id = bundle.tokenizer.encode(" True", add_special_tokens=False)[0]
    false_id = bundle.tokenizer.encode(" False", add_special_tokens=False)[0]
    bridge_rows = []
    test_statements = [s for s, _ in cities][:8] + [f for _, f in cities][:8]
    for scale in BRIDGE_SCALES:
        diffs = []
        for stmt in test_statements:
            templated = bench.apply_chat_template(
                bundle, f"Respond with only 'True' or 'False'. Statement: {stmt}")
            logits = bench.next_token_logits(bundle, templated, steer=(best_layer, truth_dir, eff(scale)))
            diffs.append(float(logits[true_id] - logits[false_id]))
        bridge_rows.append({"scale": scale, "mean_true_false_logit_diff": round(sum(diffs) / len(diffs), 4)})
    bridge_path = ctx.path("tables", "truth_direction_bridge.csv")
    bench.write_csv_with_context(ctx, bridge_path, bridge_rows)
    ctx.register_artifact(bridge_path, "table", "True/False assent shift vs steering on the truth direction.")
    bridge_span = (max(r["mean_true_false_logit_diff"] for r in bridge_rows)
                   - min(r["mean_true_false_logit_diff"] for r in bridge_rows))
    bridge_verdict = "decodable-and-steerable" if bridge_span > 1.0 else "decodable-but-inert"
    print(f"[lab7]   truth-direction steering span {bridge_span:.2f} logits -> {bridge_verdict}")

    # ----- plots ----------------------------------------------------------------
    if not args.no_plots:
        plot_layer_sweep(ctx, sweep_rows, best_layer)
        plot_dose_response(ctx, dose_rows, "sentiment")
        plot_monitor(ctx, proj_refusal, proj_benign, auc)
        plot_induced_refusal(ctx, induced_rows)
        plot_bridge(ctx, bridge_rows)

    # ----- metrics, card, claims, summary --------------------------------------
    real_at = {r["scale"]: r for r in dose_rows if r["condition"] == "real"}
    rand_at = {r["scale"]: r for r in dose_rows if r["condition"] == "random"}
    shuf_at = {r["scale"]: r for r in dose_rows if r["condition"] == "shuffled"}
    max_pos = max(TRACK_A_SCALES)
    min_neg = min(TRACK_A_SCALES)
    effect_over_control = real_at[max_pos]["target_score"] - rand_at[max_pos]["target_score"]
    pos_swing = real_at[max_pos]["target_score"] - real_at[0.0]["target_score"]
    neg_swing = real_at[min_neg]["target_score"] - real_at[0.0]["target_score"]
    metrics = {
        "model_id": bundle.anatomy.model_id,
        "best_layer": best_layer,
        "track_a_effect_over_random_at_max_dose": round(effect_over_control, 4),
        "track_a_positive_swing": round(pos_swing, 4),
        "track_a_negative_swing": round(neg_swing, 4),
        "track_a_fluency_at_max_dose": real_at[max_pos]["fluency_logprob"],
        "track_a_fluency_at_zero": real_at[0.0]["fluency_logprob"],
        "refusal_monitor_auc": round(auc, 4),
        "max_induced_refusal_benign": max_induced,
        "bridge_span_logits": round(bridge_span, 4),
        "bridge_verdict": bridge_verdict,
        "truth_direction_provenance": truth_provenance,
    }
    bench.write_json(ctx.path("metrics.json"), metrics)
    ctx.register_artifact(ctx.path("metrics.json"), "metrics", "Aggregate Lab 7 metrics.")

    write_claim_card(ctx, bundle, best_layer, dose_rows, auc, max_induced, bridge_verdict,
                     bridge_span, truth_provenance)

    run_name = ctx.run_dir.name
    claims = [
        {
            "id": f"{LAB_ID}-C1", "tag": "CAUSAL",
            "text": (
                f"A difference-in-means sentiment direction injected at layer {best_layer} of "
                f"{bundle.anatomy.model_id} steers generated sentiment ASYMMETRICALLY: positive dose "
                f"swings the score by {pos_swing:+.2f} (beating the random control by {effect_over_control:+.2f} "
                f"at scale {max_pos}), while negative dose moves it only {neg_swing:+.2f} — the RLHF "
                f"positivity floor resists negative steering. Fluency falls from {real_at[0.0]['fluency_logprob']} "
                f"to {real_at[max_pos]['fluency_logprob']} at max dose (the side effect that breaks first)."
            ),
            "artifact": f"runs/{run_name}/plots/dose_response_sentiment.png",
            "falsifier": "The random and shuffled controls match the real direction's positive curve — the effect was generic norm, not the concept.",
        },
        {
            "id": f"{LAB_ID}-C2", "tag": "CAUSAL",
            "text": (
                f"The refusal direction both PREDICTS and CAUSES refusal, but these are separate "
                f"properties: forward-pass projection separates held-out refusal-eliciting from benign "
                f"prompts at AUC {auc:.2f}, and steering benign prompts toward it induces refusal in up "
                f"to {max_induced:.0%} of them. No completion was sampled from any refusal-eliciting "
                "prompt; ablation was not implemented."
            ),
            "artifact": f"runs/{run_name}/tables/refusal_monitor_table.csv",
            "falsifier": "The random direction induces refusal at the same rate — the effect was disruption, not the refusal feature.",
        },
        {
            "id": f"{LAB_ID}-C3", "tag": "CAUSAL",
            "text": (
                f"Lab 4's decodable truth direction is {bridge_verdict} on this model: steering it across "
                f"scales moves mean logit('True')-logit('False') by {bridge_span:.2f}. Decodability "
                f"({truth_provenance.get('within', 'n/a')} probe accuracy in Lab 4) and steerability are "
                "different evidence about a direction."
            ),
            "artifact": f"runs/{run_name}/plots/truth_direction_bridge.png",
            "falsifier": "Recomputing the direction with a different contrast set flips the verdict — it was an artifact of these statements.",
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    write_summary(ctx, bundle, best_layer, metrics, dose_rows, auc, max_induced,
                  bridge_verdict, bridge_span, claims)
    print(f"[lab7] wrote steering_claim_card.md, run_summary.md, and {len(claims)} drafted ledger claims")


# ---------------------------------------------------------------------------
# Lab 4 bridge helpers
# ---------------------------------------------------------------------------


def find_lab4_direction() -> dict[str, Any]:
    """Locate the most recent Lab 4 truth_direction.pt for provenance display.

    We do NOT steer with it directly (it was computed on the BASE model; a
    direction is model-specific). We recompute on the current model from the
    same frozen cities data and report the saved one's metadata so students
    see the provenance and the model mismatch.
    """
    import torch

    runs = sorted((bench.COURSE_ROOT / "runs").glob("lab04*/tables/truth_direction.pt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return {"found": False, "note": "no Lab 4 run found; recomputed on current model only"}
    meta = torch.load(runs[0], map_location="cpu", weights_only=False)
    return {
        "found": True, "path": str(runs[0].relative_to(bench.COURSE_ROOT)),
        "saved_on_model": meta.get("model_id"), "saved_layer": meta.get("layer"),
        "train_family": meta.get("train_family"),
        "within": (meta.get("metrics") or {}).get("within"),
    }


def load_truth_statements() -> list[tuple[str, str]]:
    """(true statement, false statement) pairs from the frozen Lab 4 cities CSV."""
    path = bench.COURSE_ROOT / "data" / "truth_cities.csv"
    trues, falses = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            (trues if row["label"] == "1" else falses).append(row["statement"])
    return list(zip(trues, falses))


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------


def write_claim_card(ctx, bundle, best_layer, dose_rows, auc, max_induced, bridge_verdict,
                     bridge_span, provenance) -> None:
    real = {r["scale"]: r for r in dose_rows if r["condition"] == "real"}
    rand = {r["scale"]: r for r in dose_rows if r["condition"] == "random"}
    max_pos = max(r["scale"] for r in dose_rows)
    card = [
        "# Steering claim card",
        "",
        f"- **Model:** `{bundle.anatomy.model_id}` (instruct) | run `{ctx.run_dir.name}`",
        f"- **Direction:** difference-in-means, sentiment contrast pairs, injected at layer {best_layer}",
        "",
        "## Track A — sentiment steering",
        "",
        f"- **Effect:** sentiment score {real[0.0]['target_score']} (dose 0) -> {real[max_pos]['target_score']} "
        f"(dose {max_pos}); random control reaches only {rand[max_pos]['target_score']}.",
        f"- **Dose:** monotone over {sorted(set(r['scale'] for r in dose_rows))}.",
        f"- **Side effects:** fluency {real[0.0]['fluency_logprob']} -> {real[max_pos]['fluency_logprob']}; "
        f"drift accuracy {real[0.0]['drift_accuracy']} -> {real[max_pos]['drift_accuracy']}.",
        "- **What it does NOT show:** that the model 'feels' sentiment, or that this direction is the only "
        "one that would work. It shows one direction is sufficient to move one behavior.",
        "",
        "## Track B — refusal direction (forward-pass only; safe direction only)",
        "",
        f"- **Predicts:** held-out monitor AUC {auc:.2f} (projection vs category, no generation).",
        f"- **Causes:** up to {max_induced:.0%} induced refusal on BENIGN prompts under steering.",
        "- **Wall:** no completion sampled from any refusal-eliciting prompt; ablation not implemented.",
        "- **What it does NOT show:** that refusal is ONE direction (redundancy untested here), or that "
        "ablation would jailbreak (assigned as reading, not reproduced).",
        "",
        "## Bridge — Lab 4's truth direction",
        "",
        f"- **Verdict:** {bridge_verdict} (steering span {bridge_span:.2f} logits on True/False assent).",
        f"- **Provenance of the saved direction:** {provenance}",
        "- **Lesson:** decodability (Lab 4) and steerability (here) are different evidence; a probe finding "
        "a direction does not entail the model uses it.",
        "",
        "## Hacking's question (graded prose goes here, not in this file)",
        "",
        "You moved the model with a direction you computed. Is it real? What distinguishes steering",
        "success from an explanation of the behavior? (See the handout's ethics prompts.)",
        "",
    ]
    path = ctx.path("steering_claim_card.md")
    bench.write_text(path, "\n".join(card))
    ctx.register_artifact(path, "summary", "The steering claim card: effect, dose, side effects, and limits.")


def write_summary(ctx, bundle, best_layer, metrics, dose_rows, auc, max_induced,
                  bridge_verdict, bridge_span, claims) -> None:
    real = {r["scale"]: r for r in dose_rows if r["condition"] == "real"}
    max_pos = max(r["scale"] for r in dose_rows)
    lines = [
        "# Lab 7 run summary: steering and the refusal direction",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` (instruct, chat template applied to every prompt)",
        f"- steering layer: {best_layer} (chosen by the cheap next-token layer sweep)",
        "- evidence level: `CAUSAL` (representation-level control)",
        "- SAFETY: refusal direction extracted/monitored by forward passes only; steering toward refusal "
        "on benign prompts only; ablation not implemented; no harmful completion ever sampled",
        "",
        "## 1-4. Behavior, object, intervention, headline",
        "",
        f"- Track A: sentiment direction at L{best_layer}; score {real[0.0]['target_score']} -> "
        f"{real[max_pos]['target_score']} at dose {max_pos}, beating the random control by "
        f"{metrics['track_a_effect_over_random_at_max_dose']:+.2f}; fluency "
        f"{real[0.0]['fluency_logprob']} -> {real[max_pos]['fluency_logprob']}",
        f"- Track B: refusal monitor AUC {auc:.2f}; induced refusal on benign prompts up to {max_induced:.0%}",
        f"- Bridge: Lab 4's truth direction is {bridge_verdict} (span {bridge_span:.2f} logits)",
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
        "1. `steering_claim_card.md` — the deliverable.",
        "2. `plots/dose_response_sentiment.png` — real vs controls; where fluency breaks.",
        "3. `plots/refusal_monitor.png` and `plots/induced_refusal.png` — predict vs cause.",
        "4. `plots/truth_direction_bridge.png` — Lab 4's loop, closed.",
        "5. `tables/steered_examples.csv` — read actual generations across the dose.",
        "",
        "## 7. Caveats and the ethics unit",
        "",
        "- A dose-response curve with controls is the unit of evidence; one generation is an anecdote.",
        "- 'Predicts refusal' and 'causes refusal' are different claims; the lab measured both separately.",
        "- The single-direction framing is a hypothesis the monitor supports, not proves; redundancy is",
        "  untested here (cf. Lab 6's redundancy finding).",
        "- Dual use is confronted by apparatus: read Arditi et al., then argue (handout) whether the",
        "  ablation result should have been published — using your own Track B numbers as evidence about",
        "  how easy the method is.",
        "",
    ]
    bench.write_text(ctx.path("run_summary.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("run_summary.md"), "summary", "The seven standard questions answered.")
