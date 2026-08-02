from pathlib import Path
import subprocess


PHASE4_ROOT = Path(__file__).resolve().parents[1]
QUEUE = PHASE4_ROOT / "run_phase4_post_a1000_import_queue.sh"


def test_post_a1000_import_queue_is_syntax_valid_and_probe_is_nonmutating():
    subprocess.run(["bash", "-n", str(QUEUE)], check=True)
    result = subprocess.run(
        [str(QUEUE), "--approval-probe"], check=True,
        text=True, capture_output=True)
    assert "no file work started" in result.stdout


def test_post_a1000_import_queue_freezes_methods_only_order():
    text = QUEUE.read_text()
    stages = [
        "register_import olmo_bank_w_capability",
        "register_joint_replay",
        "register_import gemma_transport_terminal",
        "register_import olmo_lineage_terminal",
    ]
    positions = [text.rindex(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "p4-import-olmo-bank-w-capability-v1" in text
    assert "p4-bank-w-capability-joint-imported-dev-v1" in text
    assert "p4-import-gemma-transport-v1" in text
    assert "p4-import-olmo-lineage-final-v1" in text
    assert "startswith((\"ol-\", \"gm-\"))" in text


def test_post_a1000_import_queue_pulls_before_every_push():
    text = QUEUE.read_text()
    assert text.count("git push origin") == 2
    assert text.count("git pull --ff-only origin") == 3
    for block_name, next_name in [
            ("bank_registry_event()", "verify_existing_import()"),
            ("bank_joint_outputs()", "register_joint_replay()")]:
        block = text[text.index(block_name):text.index(next_name)]
        assert block.index("git pull --ff-only origin") < \
            block.index("git push origin")


def test_post_a1000_import_queue_requires_full_frozen_decision_boundary():
    text = QUEUE.read_text()
    required = [
        "p4-qwen-lens-fit-drawA-n1000-dev-v1",
        "p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1",
        "p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1",
        "p4-qwen-selection-margin-a500-a1000-dev-v1",
        "p4-qwen-lens-influence-prompt323-dev-v1",
        "p4-qwen-canonical-lens-decision-a1000-dev-v1",
    ]
    assert all(evidence_id in text for evidence_id in required)
    assert "p4_qwen_mode_variance_gpu" not in text
