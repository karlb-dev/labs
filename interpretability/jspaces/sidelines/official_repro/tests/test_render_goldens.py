"""Tokenizer-level render goldens against the pinned snapshots (CPU).

These pin the template facts the render manifest records: Qwen
non-thinking closed-empty think span; OLMo default system preamble;
prefill preservation; BOS behavior; position-finder audits on released
items. Skipped when the pinned snapshots are absent (fresh clone without
model downloads)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jspace_official_repro.paths import (
    EVALUATIONS_DIR,
    EXPERIMENTS_DIR,
    OLMO_MODEL_ID,
    OLMO_MODEL_REVISION,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    model_snapshot,
)
from jspace_official_repro.rendering import (
    Rendered,
    position_before_substring,
    preferred_token,
    render_chat,
    single_token_forms,
)

sys.path.insert(0, str(Path(__file__).parent))


def _tokenizer(model_id, revision):
    snapshot = model_snapshot(model_id, revision)
    if not snapshot.exists():
        pytest.skip(f"pinned snapshot not downloaded: {snapshot}")
    import transformers

    return transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True
    )


@pytest.fixture(scope="module")
def qwen_tok():
    return _tokenizer(QWEN_MODEL_ID, QWEN_MODEL_REVISION)


@pytest.fixture(scope="module")
def olmo_tok():
    return _tokenizer(OLMO_MODEL_ID, OLMO_MODEL_REVISION)


def _fake_model(tokenizer):
    return SimpleNamespace(tokenizer=tokenizer, input_device="cpu")


MESSAGES = [{"role": "user", "content": "Think of a country. Answer in one word."}]


def test_qwen_nothink_renders_closed_empty_think_span(qwen_tok):
    text = qwen_tok.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    assert text.endswith("<think>\n\n</think>\n\n")
    default = qwen_tok.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True
    )
    assert default.endswith("<think>\n")  # default OPENS a span: must not use


def test_qwen_has_no_bos_and_raw_matches(qwen_tok):
    assert qwen_tok.bos_token is None
    a = qwen_tok("Fact: The currency", add_special_tokens=True).input_ids
    b = qwen_tok("Fact: The currency", add_special_tokens=False).input_ids
    assert a == b


def test_olmo_template_injects_default_system_preamble(olmo_tok):
    text = olmo_tok.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True
    )
    assert "<functions></functions>" in text  # checkpoint-official preamble
    assert text.endswith("<|im_start|>assistant\n")


def test_olmo_prefill_preserved_verbatim(olmo_tok):
    prefill = ' Yes, I detect an injected thought. The thought is about "'
    messages = [
        {"role": "user", "content": "Trial 1: Do you detect an injected thought?"},
        {"role": "assistant", "content": prefill},
    ]
    text = olmo_tok.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )
    assert text.endswith(prefill)


def test_qwen_prefill_open_quote_is_final_token(qwen_tok):
    prefill = ' Yes, I detect an injected thought. The thought is about "'
    messages = [
        {"role": "user", "content": "Trial 1: Do you detect an injected thought?"},
        {"role": "assistant", "content": prefill},
    ]
    rendered = render_chat(_fake_model(qwen_tok), messages, continue_final=True,
                           extra_template_kwargs={"enable_thinking": False})
    last = qwen_tok.decode([rendered.input_ids[0, -1].item()])
    assert last.endswith('"')  # open quote scored position survives render


def test_lens_eval_targets_are_continuations_so_position_is_final(qwen_tok):
    # D7: `target` is the expected next word — prompts end immediately
    # before it, so the readout position is the final prompt token.
    data = json.loads((EVALUATIONS_DIR / "lens-eval-multihop.json").read_text())
    for item in data["items"][:10]:
        assert not item["prompt"].rstrip().endswith(item["target"])
    item = data["items"][0]
    ids = qwen_tok(item["prompt"], return_tensors="pt",
                   add_special_tokens=False).input_ids
    rendered = Rendered(text=item["prompt"], input_ids=ids, form="raw",
                        template_kwargs={})
    round_trip = qwen_tok.decode(ids[0].tolist())
    assert round_trip == item["prompt"]  # no token loss at the boundary
    assert rendered.final_position == ids.shape[1] - 1


def test_position_before_substring_works_for_inline_targets(qwen_tok):
    # The substring finder remains in use for span location (probe-swap
    # intermediates, top-down stimuli); verify on a synthetic case.
    text = "Fact: The capital of Japan is Tokyo. The currency there is the"
    ids = qwen_tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    rendered = Rendered(text=text, input_ids=ids, form="raw", template_kwargs={})
    position = position_before_substring(rendered, qwen_tok, "Tokyo")
    decoded_next = qwen_tok.decode(ids[0, position + 1 : position + 3].tolist())
    assert "Tokyo" in decoded_next


def test_single_token_targets_on_released_words(qwen_tok, olmo_tok):
    words = ["France", "Paris", "Ottawa", "Beijing", "Cairo", "lion", "seven"]
    for tokenizer in (qwen_tok, olmo_tok):
        for word in words:
            forms = single_token_forms(tokenizer, word)
            assert forms, f"{word} has no single-token form"
            assert preferred_token(tokenizer, word) is not None


def test_verbal_report_prompt_final_position_is_generation_boundary(qwen_tok):
    rendered = render_chat(_fake_model(qwen_tok), MESSAGES,
                           extra_template_kwargs={"enable_thinking": False})
    # The scored position's next token is the model's first answer token;
    # the render must end exactly at the generation boundary.
    assert rendered.text.endswith("<think>\n\n</think>\n\n")
    assert rendered.final_position == rendered.seq_len - 1


def test_span_finders_survive_bos_prefix(olmo_tok):
    # INCIDENT or1-002: OLMo raw renders carry BOS whose decoded text
    # shifted char offsets. Finders must locate spans correctly with the
    # special-token prefix present.
    import torch

    from jspace_official_repro.rendering import (
        find_token_span,
        position_before_substring,
    )

    text = "My sister has always wanted to visit France"
    ids = olmo_tok(text, add_special_tokens=False).input_ids
    with_bos = [olmo_tok.bos_token_id] + ids
    rendered = Rendered(text=text, input_ids=torch.tensor([with_bos]),
                        form="raw", template_kwargs={})
    start, end = find_token_span(rendered, olmo_tok, "France")
    decoded = olmo_tok.decode(with_bos[start:end + 1])
    assert "France" in decoded
    position = position_before_substring(rendered, olmo_tok, "France")
    after = olmo_tok.decode(with_bos[position + 1:position + 3])
    assert "France" in after
