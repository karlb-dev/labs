from pathlib import Path
import subprocess

import yaml


PHASE4_ROOT = Path(__file__).resolve().parents[1]
QUEUE = PHASE4_ROOT / "run_qwen_a1000_postfit_queue.sh"


def test_a1000_queue_is_syntax_valid_and_probe_is_nonmutating():
    subprocess.run(["bash", "-n", str(QUEUE)], check=True)
    result = subprocess.run(
        [str(QUEUE), "--approval-probe"], check=True,
        text=True, capture_output=True)
    assert "no file or GPU work started" in result.stdout


def test_a1000_queue_freezes_stage_order_and_stops_before_mode_pilot():
    text = QUEUE.read_text()
    stages = [
        "run_stage qwen_lens_convergence_a500_a1000",
        "run_stage qwen_multilens_functional_gate_a500_a1000",
        "run_stage qwen_selection_margin_a500_a1000",
        "run_stage qwen_lens_influence_prompt323",
        "run_stage qwen_canonical_lens_decision_a1000",
    ]
    positions = [text.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "p4_qwen_mode_variance_gpu" not in text
    fit_backup = (
        "preserve_registered_outputs "
        "p4-qwen-lens-fit-drawA-n1000-dev-v1")
    assert text.rindex(fit_backup) < positions[0]
    bank = text[text.index("bank_registry_event()"):
                text.index("preserve_registered_outputs()")]
    assert bank.index("git pull --ff-only") < bank.index("git push origin")


def test_exactly_three_successor_configs_share_the_a1000_binding_slot():
    paths = [
        PHASE4_ROOT / "configs/p4_qwen_lens_convergence_drawA_n500_n1000_dev.yaml",
        PHASE4_ROOT / "configs/p4_qwen_multilens_functional_gate_a500_a1000_dev.yaml",
        PHASE4_ROOT / "configs/p4_qwen_lens_influence_prompt323_dev.yaml",
    ]
    expected_uri = (
        "artifact://phase4/lens/qwen36-27b/nested_fit/draw_a/"
        "qwen36-27b_jlens_drawA_n1000.pt")
    for path in paths:
        lens = yaml.safe_load(path.read_text())["lenses"]["a1000"]
        assert lens["evidence_id"] == \
            "p4-qwen-lens-fit-drawA-n1000-dev-v1"
        assert lens["lens_uri"] == expected_uri
        assert lens["n_prompts"] == 1000
        digest = str(lens["lens_sha256"])
        assert digest == "BIND_REGISTERED_A1000_SHA256_AFTER_FIT_COMPLETION" \
            or (len(digest) == 64
                and all(value in "0123456789abcdef" for value in digest))
