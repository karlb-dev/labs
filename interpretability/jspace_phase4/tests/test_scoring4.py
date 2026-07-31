import hashlib

import pytest
import torch

from jspace_phase4.scoring4 import (
    DEFAULT_SPEC,
    ScoringSession,
    aggregate_alias_lps,
    canonical_alias_for,
    logsumexp_reference,
    prefix_disjoint_aliases,
)


class MockTokenizer:
    bos_token_id = 0
    add_bos_token = True

    @staticmethod
    def _id(token):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:2], "big") + 2

    def __call__(self, text, *, add_special_tokens=True,
                 return_tensors=None, truncation=False, max_length=None):
        tokens = [self._id(token) for token in text.split()]
        if add_special_tokens and self.add_bos_token:
            tokens.insert(0, self.bos_token_id)
        if max_length is not None:
            tokens = tokens[:max_length]

        class Result:
            input_ids = torch.tensor([tokens], dtype=torch.long)
        return Result()


def test_scoring_piecewise_bos_and_boundary():
    session = ScoringSession(MockTokenizer())
    full, prompt_length = session.full_ids("capital of France is", " Paris")
    prompt = session.prompt_ids("capital of France is")
    answer = session.answer_ids(" Paris")
    assert session.bos_prefixed
    assert torch.equal(full, torch.cat([prompt, answer], dim=1))
    assert prompt_length == prompt.shape[1]
    assert int(answer[0, 0]) != 0
    with pytest.raises(ValueError, match="trailing"):
        session.prompt_ids("capital of France is ")


def test_prefix_disjoint_selection_and_primary_aggregation():
    aliases = [" a", " a long", " b", " c"]
    token_ids = {
        " a": [1],
        " a long": [1, 2],
        " b": [3],
        " c": [4],
    }
    selected = prefix_disjoint_aliases(
        aliases, token_ids, canonical_alias=" a long")
    assert selected == [" a long", " b", " c"]
    values = {" a long": -3.0, " b": -4.0, " c": -5.0}
    actual = aggregate_alias_lps(values, selected)
    expected = logsumexp_reference(list(values.values()))
    assert actual == pytest.approx(expected)
    with pytest.raises(ValueError, match="frozen"):
        aggregate_alias_lps(values, selected, method="max")


def test_alias_manifest_is_tokenizer_pinned():
    session = ScoringSession(MockTokenizer())
    first = session.freeze_alias_manifest(
        [" a", " a long", " b"], canonical_alias=" a long")
    second = session.freeze_alias_manifest(
        [" a", " a long", " b"], canonical_alias=" a long")
    assert first == second
    assert len(first["token_manifest_sha256"]) == 64
    assert first["aggregation"] == "prefix-disjoint-logsumexp"


def test_canonical_alias_is_not_assumed_to_be_alias_zero():
    session = ScoringSession(MockTokenizer())
    assert canonical_alias_for(
        session,
        [" the baht", " baht", " Thai baht"],
        "baht",
    ) == " baht"


def test_canonical_alias_prefers_exact_spelling_before_normalization():
    session = ScoringSession(MockTokenizer())
    assert canonical_alias_for(
        session,
        [" the Río de la Plata", " the Rio de la Plata"],
        "the Río de la Plata",
    ) == " the Río de la Plata"


def test_generation_boundary_and_counterfactual_trichotomy():
    session = ScoringSession(MockTokenizer())
    assert session.grade_alias("Bogotá is large", [" Bogota"]) == " Bogota"
    assert session.grade_alias("Indian state", [" India"]) is None
    assert session.grade_alias("Dutchman", [" Dutch"]) is None
    assert session.grade_counterfactual_generation(
        "euro is used", original_aliases=[" dollar"],
        counterfactual_aliases=[" euro"])["outcome"] == "counterfactual"
    assert session.grade_counterfactual_generation(
        "unknown", original_aliases=[" dollar"],
        counterfactual_aliases=[" euro"])["outcome"] == "other_invalid"


def test_clean_rank_requires_and_reports_alias_metadata():
    logits = torch.tensor([0.0, 3.0, 2.0, 1.0])
    result = ScoringSession.clean_first_token_ranks(
        logits, {" first": 2, " second": 3})
    assert result["rank_metadata_present"]
    assert result["rank_by_alias"] == {" first": 2, " second": 3}
    assert result["min_rank"] == 2
    with pytest.raises(ValueError, match="requires alias metadata"):
        ScoringSession.clean_first_token_ranks(logits, {})


def test_teacher_forced_answer_score_stays_on_logits_device():
    session = ScoringSession(MockTokenizer())
    full, prompt_length = session.full_ids("a b", " c d")
    vocabulary = 65540
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(full.shape[1], vocabulary, generator=generator)
    actual = session.answer_sequence_lp(full, logits, prompt_length)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    expected = sum(
        float(log_probs[index, full[0, index + 1]])
        for index in range(prompt_length - 1, full.shape[1] - 1)
    )
    assert actual == pytest.approx(expected, abs=1e-5)
