# §15.3 scoring + statistics suites.
import numpy as np
import pandas as pd
import pytest

from jspace_phase3.scoring import DEFAULT_SPEC, ScoringSession
from jspace_phase3.stats import (family_cluster_bootstrap_ci,
                                 exact_signflip_test, family_signflip_test,
                                 leave_one_family_out, monte_carlo_pvalue,
                                 paired_specific_effects, plant_interaction,
                                 signflip_confidence_set,
                                 within_fact_composition,
                                 within_fact_model_diff,
                                 within_item_label_exchange_tail,
                                 wild_cluster_percentile_t_ci,
                                 wild_cluster_bootstrap_t)


# ------------------------------------------------------------- scoring
class MockTok:
    """Word-level tokenizer with a BOS id; enough to test the spec."""
    bos_token_id = 0
    add_bos_token = True

    def __call__(self, text, add_special_tokens=True, return_tensors=None,
                 truncation=False, max_length=None):
        import torch
        toks = [hash(w) % 1000 + 2 for w in text.split()]
        if add_special_tokens and self.add_bos_token:
            toks = [self.bos_token_id] + toks
        if max_length:
            toks = toks[:max_length]

        class R:
            input_ids = torch.tensor([toks])
        return R()


def test_bos_asserted_and_prompt_rules():
    s = ScoringSession(MockTok(), DEFAULT_SPEC)
    assert s.bos_prefixed
    ids = s.prompt_ids("the capital of France is")
    assert int(ids[0, 0]) == 0
    with pytest.raises(ValueError, match="trailing whitespace"):
        s.prompt_ids("the capital of France is ")


def test_no_bos_tokenizer_scores_native():
    """Qwen case: no bos_token_id exists, so jlens.from_hf(force_bos=True)
    is a no-op and the assay-wide unit system is native tokenization —
    the session must construct and must NOT prefix anything."""
    tok = MockTok()
    tok.bos_token_id = None
    tok.add_bos_token = False
    s = ScoringSession(tok, DEFAULT_SPEC)
    assert not s.bos_prefixed
    ids = s.prompt_ids("the capital of France is")
    assert ids.shape[1] == 5 and int(ids[0, 0]) != 0
    assert not tok.add_bos_token          # session did not mutate it


def test_piecewise_concatenation():
    import torch
    s = ScoringSession(MockTok(), DEFAULT_SPEC)
    full, n_prompt = s.full_ids("a b c", " Paris")
    p = s.prompt_ids("a b c")
    a = s.answer_ids(" Paris")
    assert torch.equal(full, torch.cat([p, a], dim=1))
    assert n_prompt == p.shape[1]
    assert int(a[0, 0]) != 0                 # no BOS inside the answer


def test_alias_prefix_overlap_flagged():
    s = ScoringSession(MockTok(), DEFAULT_SPEC)
    audit = s.alias_audit([" Paris", " Paris France", " Rome"])
    assert audit["prefix_overlaps"] == [[" Paris", " Paris France"]]
    assert not audit["ok"]
    assert s.alias_audit([" Paris", " Rome"])["ok"]


def test_generation_grading_deterministic():
    s = ScoringSession(MockTok(), DEFAULT_SPEC)
    assert s.grade_generation("  Paris!", [" Paris"])["correct"]
    assert s.grade_generation("paris is the capital", [" Paris"])["correct"]
    assert not s.grade_generation("Lyon", [" Paris"])["correct"]
    assert not s.grade_generation("Indian state", [" India"])["correct"]
    assert not s.grade_generation("Dutchman", [" Dutch"])["correct"]


def test_answer_seq_lp_matches_manual():
    import torch
    s = ScoringSession(MockTok(), DEFAULT_SPEC)
    full, n_prompt = s.full_ids("a b", "x y")
    V = 1001
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(full.shape[1], V, generator=g)
    lp = s.answer_seq_lp(full, logits, n_prompt)
    logp = torch.log_softmax(logits.float(), dim=-1)
    manual = sum(float(logp[t, full[0, t + 1]])
                 for t in range(n_prompt - 1, full.shape[1] - 1))
    assert abs(lp - manual) < 1e-5


# ---------------------------------------------------------------- stats
def test_planted_interaction_recovered():
    rng = np.random.default_rng(7)
    df = plant_interaction(rng, n_families=40, effect=-0.8)
    comp = within_fact_composition(df)
    diff = within_fact_model_diff(comp, model_a="A", model_b="B")
    fam = diff.groupby("canonical_family")["diff"].mean()
    res = family_signflip_test(fam.to_numpy(), draws=20_000, seed=1)
    assert res["p"] < 0.01
    assert -1.2 < res["estimate"] < -0.4
    ci = family_cluster_bootstrap_ci(diff, "diff")
    assert ci["ci95"][0] < -0.8 < ci["ci95"][1] or ci["ci95"][1] < -0.4


def test_null_calibration_signflip():
    """Type-I ≈ α under a true null (200 sims, α=0.05 ⇒ expect ~10±)."""
    rng = np.random.default_rng(11)
    rejections = 0
    for _ in range(200):
        vals = rng.normal(0, 1, size=25)
        if family_signflip_test(vals, draws=2000, seed=3)["p"] < 0.05:
            rejections += 1
    assert rejections <= 25            # binomial(200, .05): P(>25) ~ 1e-5


def test_two_family_failure_caught():
    with pytest.raises(ValueError, match="families"):
        family_signflip_test(np.array([1.0, -0.5]))
    df = pd.DataFrame({"delta_J": [-2.0, -1.5], "delta_C": [0.1, 0.0],
                       "canonical_family": ["a", "b"]})
    with pytest.raises(ValueError, match="families"):
        within_item_label_exchange_tail(df)


def test_label_exchange_tail_detects_and_calibrates():
    rng = np.random.default_rng(13)
    n = 200
    fams = np.repeat([f"f{i}" for i in range(20)], n // 20)
    # real J tail: 30% of items lose 2 nats under J, control never
    tail = rng.random(n) < 0.3
    df = pd.DataFrame({
        "delta_J": np.where(tail, -2.0, 0.0) + rng.normal(0, 0.1, n),
        "delta_C": rng.normal(0, 0.1, n),
        "canonical_family": fams})
    res = within_item_label_exchange_tail(df, draws=4000, seed=5)
    assert res["p"] < 0.01 and res["estimate"] > 0.2
    # null: identical distributions ⇒ p uniform-ish, not tiny
    df0 = pd.DataFrame({
        "delta_J": rng.normal(0, 1, n), "delta_C": rng.normal(0, 1, n),
        "canonical_family": fams})
    res0 = within_item_label_exchange_tail(df0, draws=4000, seed=6)
    assert res0["p"] > 0.01


def test_wild_cluster_bootstrap_and_weighting():
    rng = np.random.default_rng(17)
    df = plant_interaction(rng, n_families=12, effect=-1.0,
                           facts_per_family=6)
    comp = within_fact_composition(df)
    diff = within_fact_model_diff(comp, model_a="A", model_b="B")
    res = wild_cluster_bootstrap_t(diff, "diff")
    assert res["p"] < 0.05
    # family vs item weighting differ when one family is huge
    big = diff.copy()
    extra = big[big.canonical_family == "fam000"].sample(
        40, replace=True, random_state=1)
    extra = extra.assign(diff=extra["diff"] + 5.0)
    big = pd.concat([big, extra])
    fam_w = big.groupby("canonical_family")["diff"].mean().mean()
    item_w = big["diff"].mean()
    assert abs(fam_w - item_w) > 0.3


def test_leave_one_family_out_shape():
    rng = np.random.default_rng(19)
    df = plant_interaction(rng, n_families=8)
    comp = within_fact_composition(df)
    diff = within_fact_model_diff(comp, model_a="A", model_b="B")
    lofo = leave_one_family_out(diff, "diff")
    assert len(lofo) == 8
    assert {"left_out", "estimate"} <= set(lofo.columns)


def test_exact_signflip_and_inverted_confidence_set():
    vals = np.array([-1.2, -0.9, -0.8, -0.7, -0.6, -0.4, -0.2, 0.1])
    res = exact_signflip_test(vals)
    brute = []
    for bits in range(2**len(vals)):
        signs = 1 - 2 * ((bits >> np.arange(len(vals))) & 1)
        brute.append(float((signs * vals).mean()))
    expected = np.mean(np.abs(brute) >= abs(vals.mean()) - 1e-15)
    assert res["p"] == expected
    assert res["n_patterns"] == 256
    inv = signflip_confidence_set(vals, grid_points=1001)
    assert inv["confidence_set"][0] <= vals.mean() \
        <= inv["confidence_set"][1]
    assert not inv["range_truncated"]


def test_plus_one_monte_carlo_pvalue():
    null = np.zeros(99)
    assert monte_carlo_pvalue(null, 1.0, alternative="greater") == 0.01


def test_wild_cluster_percentile_t_is_named_and_deterministic():
    df = pd.DataFrame({
        "canonical_family": [f"f{i}" for i in range(6)],
        "d": [-1.2, -0.7, -0.6, -0.4, 0.1, -0.8],
    })
    a = wild_cluster_percentile_t_ci(df, "d")
    b = wild_cluster_percentile_t_ci(df, "d")
    assert a["method"] == "wild-cluster-percentile-t"
    assert a["exact"] and a["n_randomizations"] == 64
    assert a["t_distribution_sha256"] == b["t_distribution_sha256"]
    assert a["ci"][0] < a["estimate"] < a["ci"][1]


def test_paired_specific_effects_requires_conditions():
    df = pd.DataFrame({
        "fact_id": ["f1"] * 3, "canonical_family": ["a"] * 3,
        "model": ["A"] * 3, "variant": ["direct"] * 3,
        "condition": ["baseline", "meanJ_span_safe", "matched"],
        "lp_logsumexp": [-1.0, -2.0, -1.2]})
    eff = paired_specific_effects(df, j_condition="meanJ_span_safe",
                                  control_condition="matched")
    assert abs(float(eff.J_effect.iloc[0]) + 1.0) < 1e-9
    assert abs(float(eff.specific.iloc[0]) + 0.8) < 1e-9
    with pytest.raises(ValueError, match="absent"):
        paired_specific_effects(df, j_condition="nope",
                                control_condition="matched")


def test_within_item_exchange_mean_calibration_and_power():
    from jspace_phase3.stats import within_item_exchange_mean
    rng = np.random.default_rng(11)
    n = 120
    fam = [f"f{i % 20}" for i in range(n)]
    null = pd.DataFrame({"a": rng.normal(0, 1, n),
                         "b": rng.normal(0, 1, n),
                         "canonical_family": fam})
    r0 = within_item_exchange_mean(null, a_col="a", b_col="b", draws=4000)
    assert r0["p"] > 0.01
    alt = null.copy()
    alt["a"] = alt["b"] + 0.8 + rng.normal(0, 0.3, n)
    r1 = within_item_exchange_mean(alt, a_col="a", b_col="b", draws=4000,
                                   alternative="greater")
    assert r1["p"] < 0.01 and r1["estimate"] > 0.5


def test_normalize_spaces_newlines():
    """Regression: 'the\\nBaht' must grade as 'the baht', not 'thebaht'
    (the v1 deletion behavior failed correct newline-led generations)."""
    s = ScoringSession(MockTok(), DEFAULT_SPEC)
    assert DEFAULT_SPEC.normalize(" the\nBaht\nThe official") \
        == "the baht the official"
    assert s.grade_generation(" the\nBaht", [" the baht"])["correct"]
