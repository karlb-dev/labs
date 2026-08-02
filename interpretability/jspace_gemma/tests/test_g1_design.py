import json
from pathlib import Path

import yaml

from jspace_gemma.experiments.gm_band_convention import mapped_band
from jspace_gemma.manifests import file_sha256


def test_g1_thresholds_are_registered_before_target_execution_is_allowed():
    root = Path(__file__).resolve().parents[1]
    design = yaml.safe_load((root / "configs/gm_g1_design.yaml").read_text())
    assert len(design["stage1_prompt_ids"]) == 4
    assert len(design["stage2_prompt_ids"]) == 16
    assert design["models"]["gemma_target"]["layers_zero_indexed"] == [22, 30, 37, 44, 52]
    assert design["models"]["olmo_positive_control"]["layers_zero_indexed"] == [4, 24, 32, 40, 47, 56, 60]
    assert design["relative_epsilon_ladder"] == [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    assert design["delivery_gate"] == {
        "cosine_floor": 0.999,
        "relative_norm_error_ceiling": 0.01,
        "below_gate": "unmeasurable",
    }
    calibration = design["threshold_calibration"]
    assert calibration["status"] == "FROZEN_PRE_GEMMA_REGISTERED"
    assert calibration["gemma_execution_allowed"] is True
    assert calibration["positive_control_artifact_sha256"] == (
        "fc957c9a6f6f397cbaf3274193713ebec332bb81dbb99b61cf9f56d058cd1942"
    )
    assert calibration["tangent_vs_secant"] == {
        "primary_decision_cosine_floor": 0.98,
        "primary_decision_relative_error_ceiling": 0.20,
        "declared_finite_dose": 0.10,
    }
    assert design["forbidden_exact_backend"] == "finite_difference"
    assert design["batch_alignment"]["finite_baseline_same_batch_shape_and_slot"]
    assert design["batch_alignment"]["exact_jvp_same_batch_shape_and_slot"]
    assert design["response_snr_contract"] == {
        "signal_norm": "min_finite_response_and_exact_jvp_norm",
        "noise_norm": "max_in_batch_clean_repeat_and_target_dtype_half_step_norm",
        "numeric_floor_is_calibrated_on_olmo": True,
    }
    assert all(design["realized_delivery_adjustment"].values())
    thresholds = yaml.safe_load(
        (root / "configs/gm_g1_thresholds_frozen.yaml").read_text()
    )
    assert thresholds["status"] == "FROZEN_PRE_GEMMA"
    assert thresholds["frozen_before_first_gemma_result"] is True
    assert thresholds["target_model_opened_at_freeze"] is False
    assert thresholds["response_snr_floor"] == 12.0
    assert thresholds["response_snr"]["primary_decision_floor"] == 20.0
    assert thresholds["smallest_faithful_secant"]["tangent_cosine_floor"] == 0.98
    assert (
        thresholds["smallest_faithful_secant"]["tangent_relative_error_ceiling"]
        == 0.20
    )
    assert thresholds["finite_dose_gate"]["declared_relative_epsilon"] == 0.10
    assert thresholds["floor_curvature_partition"]["curvature_slope_b_floor"] == 0.15
    events = [
        json.loads(line)
        for line in (root / "reports/evidence_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    positive = [
        row
        for row in events
        if row["evidence_id"] == "gm-jvp-olmo-positive-control-v1"
        and row["event"] == "evidence_created"
    ]
    assert len(positive) == 1
    assert positive[0]["positive_control_pass"] is True
    assert positive[0]["target_model_opened"] is False


def test_prompt_bank_membership_and_strata_are_exact():
    root = Path(__file__).resolve().parents[1]
    rows = [
        json.loads(line)
        for line in (root / "data/g1_prompts_v1.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["prompt_id"] for row in rows] == [f"gm-p{i:03d}" for i in range(1, 17)]
    stage1 = [row for row in rows if row["stage"] == 1]
    assert {row["stratum"] for row in stage1} == {
        "factual", "multi-hop", "neutral-prose", "code-sql"
    }
    assert len({row["text"] for row in rows}) == 16


def test_primary_paper_band_maps_to_the_addendum_gemma_range():
    root = Path(__file__).resolve().parents[1]
    convention = yaml.safe_load(
        (root / "configs/gm_g6_band_convention.yaml").read_text()
    )
    resolution = convention["resolution"]
    mapped = mapped_band(
        resolution["transferable_workspace_depth_fraction"],
        resolution["gemma_decoder_blocks"],
    )
    assert mapped == [23, 55]
    assert all(
        mapped[0] <= layer <= mapped[1]
        for layer in resolution["g6_candidate_layers_zero_indexed"]
    )


def test_stage1_execution_manifest_binds_staging_and_frozen_inputs():
    root = Path(__file__).resolve().parents[1]
    execution = yaml.safe_load(
        (root / "configs/gm_g1_stage1_execution.yaml").read_text()
    )
    assert execution["status"] == "READY_PRE_TARGET"
    assert execution["evidence_id"] == "gm-jvp-gemma-stage1-v1"
    assert execution["snapshot_manifest"] == {
        "path": (
            "/content/drive/MyDrive/interpret/special-lab-1/"
            "gemma_transport_20260802/manifests/"
            "gemma_target_local_snapshot_v1.json"
        ),
        "file_sha256": (
            "5b8d26a91b5cdc74e7fbc982d89bbf6d661233ee3da81705d165ef31cf6e308a"
        ),
        "payload_sha256": (
            "cfb98f55c3453319f19aedce51419445260b03355632d13c2402897ffbab4ec1"
        ),
        "remote_inventory_sha256": (
            "357a1bed087996cb6e9e171ec02961a6430879b1204aee0b1830997ac62d80c6"
        ),
        "staging_code_commit": "4f00d43e0810b98dfe1c281d260548ac791a14ab",
        "all_content_hashes_verified": True,
        "weight_shards": 2,
        "target_model_loaded_during_staging": False,
        "target_response_created_during_staging": False,
    }
    frozen = execution["frozen_inputs"]
    assert frozen["design_sha256"] == file_sha256(
        root / "configs/gm_g1_design.yaml"
    )
    assert frozen["threshold_sha256"] == file_sha256(
        root / "configs/gm_g1_thresholds_frozen.yaml"
    )
    assert frozen["prompt_bank_sha256"] == file_sha256(
        root / "data/g1_prompts_v1.jsonl"
    )
    grid = execution["grid"]
    assert grid["expected_cells"] == 40
    assert grid["expected_rows_per_cell"] == 28
    assert grid["expected_rows"] == 1120
    assert grid["expected_clean_parity_rows"] == 20
    assert execution["smoke"]["cell_id"] == "gm-p001-L52-single_position"


def test_stage1_result_is_registered_under_the_frozen_target_firewall():
    root = Path(__file__).resolve().parents[1]
    events = [
        json.loads(line)
        for line in (root / "reports/evidence_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    stage1 = [
        row
        for row in events
        if row["evidence_id"] == "gm-jvp-gemma-stage1-v1"
        and row["event"] == "evidence_created"
    ]
    assert len(stage1) == 1
    event = stage1[0]
    assert event["code_commit"] == "036e55233babcabacae061ab41d1410a35715aea"
    assert event["target_model_opened"] is True
    assert event["model_response_data_created"] is True
    assert event["thresholds_frozen_before_target"] is True
    assert event["positive_control_evidence_id"] == (
        "gm-jvp-olmo-positive-control-v1"
    )
    outputs = {Path(row["path"]).name: row["sha256"] for row in event["outputs"]}
    assert outputs["gemma_stage1_summary.json"] == (
        "0f28372591bc1ece4472b103d74d645416b1ddba59a08ae0688c19fccb56e384"
    )


def test_backend_parity_diagnostic_replays_the_frozen_stage1_batch_slot():
    root = Path(__file__).resolve().parents[1]
    diagnostic = yaml.safe_load(
        (root / "configs/gm_g1_backend_parity.yaml").read_text()
    )
    design = yaml.safe_load((root / "configs/gm_g1_design.yaml").read_text())
    execution = yaml.safe_load(
        (root / "configs/gm_g1_stage1_execution.yaml").read_text()
    )
    assert diagnostic["status"] == "FROZEN_PRE_DIAGNOSTIC"
    assert diagnostic["evidence_id"] == "gm-jvp-gemma-backend-parity-v1"
    assert diagnostic["source_stage1"] == {
        "evidence_id": "gm-jvp-gemma-stage1-v1",
        "compute_commit": "036e55233babcabacae061ab41d1410a35715aea",
        "summary_sha256": (
            "0f28372591bc1ece4472b103d74d645416b1ddba59a08ae0688c19fccb56e384"
        ),
        "state_sha256": (
            "5d902ae4b7b2dd6a5d2073ca1238041e321dcee537457a22ab6847d9c5d2df65"
        ),
        "cell_id": "gm-p001-L52-single_position",
        "source_layer": 52,
        "prompt_id": "gm-p001",
        "perturbation_mode": "single_position",
        "direction_id": "random-rademacher-0",
        "relative_epsilon": 0.05,
        "stored_exact_backend": "torch.func.jvp",
    }
    assert diagnostic["exact_backends"] == {
        "primary": "torch.func.jvp",
        "independent_fallback": "torch.autograd.functional.jvp",
        "finite_difference_as_exact": "forbidden",
    }
    batch = diagnostic["batch_replay"]
    assert batch["batch_size"] == execution["runtime"][
        "finite_response_batch_size"
    ]
    epsilon_index = design["relative_epsilon_ladder"].index(
        diagnostic["source_stage1"]["relative_epsilon"]
    )
    absolute_request_index = epsilon_index * 3
    assert batch["chunk_start_index"] == (
        absolute_request_index // batch["batch_size"]
    ) * batch["batch_size"]
    assert batch["selected_offset"] == (
        absolute_request_index - batch["chunk_start_index"]
    )
    assert batch["expected_request_key"] == [
        diagnostic["source_stage1"]["direction_id"],
        diagnostic["source_stage1"]["relative_epsilon"],
        "positive",
    ]
    acceptance = diagnostic["acceptance"]
    assert acceptance["backend_tangent_cosine_floor"] == 0.999999
    assert acceptance["backend_tangent_relative_error_ceiling"] == 1.0e-5
    assert max(
        acceptance["stored_source_activation_relative_error_ceiling"],
        acceptance["stored_clean_target_relative_error_ceiling"],
        acceptance["stored_forward_tangent_relative_error_ceiling"],
        acceptance["stored_finite_response_relative_error_ceiling"],
        acceptance["stored_metric_absolute_tolerance"],
    ) == 1.0e-6
