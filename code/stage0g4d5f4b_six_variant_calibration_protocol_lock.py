#!/usr/bin/env python3
"""Lock the six-variant post-hoc calibration protocol before inference.

This stage is deliberately metadata-only. It verifies the completed six-run
training freeze, inherits the already frozen primary calibration method, and
locks one resumable CPU/FP32 calibration workflow for each selected checkpoint.
It does not instantiate a model, load a checkpoint tensor payload, parse split
or calibration-manifest rows, access an image, fit a temperature, or touch the
locked internal test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


WIDTH = 124
ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

POSTHOC_PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
FINAL_TRAINING_REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_final_training_registry_v1.csv"
FINAL_TRAINING_COMPLETION = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_final_training_completion_v1.json"
FINAL_TRAINING_EVIDENCE = ROOT / "Evidence/stage0g4d5f3d_six_variant_final_training_completion_freeze.json"
FINAL_TRAINING_FREEZE_SOURCE = (
    ROOT
    / "Code/FOVEA_DR/v1_1/stage0g4d5f3d_r3_legacy_status_aware_final_training_completion_freeze.py"
)
FINAL_TRAINING_RUNNER = (
    ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f3a_resumable_final_variant_training.py"
)
COMPARATOR_IMPLEMENTATION = (
    ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2a_comparator_implementation_audit.py"
)

DEVELOPMENT_SPLIT = ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
CALIBRATION_MANIFEST = (
    ROOT / "Audit/mmrdr_fovea_dr_CALIBRATION_local_cache_manifest_v1.csv"
)
PRIMARY_CALIBRATION_PROTOCOL = (
    ROOT / "Metadata/mmrdr_fovea_dr_post_selection_calibration_protocol_v1.json"
)
PRIMARY_CALIBRATION_PROTOCOL_EVIDENCE = (
    ROOT / "Evidence/stage0g4d5b_calibration_protocol_lock.json"
)
CALIBRATION_CACHE_SOURCE = (
    ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5a_calibration_only_cache_preparation.py"
)
PRIMARY_CALIBRATION_LOCK_SOURCE = (
    ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5b_calibration_protocol_lock.py"
)
PRIMARY_CALIBRATION_RUNNER = (
    ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5c_resumable_cpu_calibration_inference.py"
)

CALIBRATION_RUN_BASE = (
    ROOT / "Runs/FOVEA_DR/posthoc_comparisons/calibration_seed2026_cpu_fp32"
)
CHECKPOINT_BASE = (
    ROOT / "Checkpoints/FOVEA_DR/posthoc_comparisons/final_seed2026"
)

PROTOCOL_OUTPUT = (
    ROOT / "Metadata/mmrdr_fovea_dr_posthoc_six_variant_calibration_protocol_v1.json"
)
EVIDENCE_OUTPUT = (
    ROOT / "Evidence/stage0g4d5f4b_six_variant_calibration_protocol_lock.json"
)
FROZEN_SOURCE = (
    ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4b_six_variant_calibration_protocol_lock.py"
)
PLANNED_RUNNER = (
    ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4c_resumable_six_variant_cpu_calibration.py"
)

EXPECTED_SHA256 = {
    POSTHOC_PROTOCOL: "f946964923e08722dbb486aa3782b08880446040e07fb6b268ea9c28c08e6ae1",
    FINAL_TRAINING_REGISTRY: "de8b3d0c098128510d185960f81af772c37f34ba877580b7160606153fa6ad75",
    FINAL_TRAINING_COMPLETION: "10659f793a7350c127e2485a10d723095cba914db7aa4b6cdecc7d59fbde5987",
    FINAL_TRAINING_EVIDENCE: "2260a55a9509ad0d649d174d85c3c5ee7b83ebd558c3af960f8c7658e2dc8c94",
    FINAL_TRAINING_FREEZE_SOURCE: "61fa64944f24b9925be7f175f95447e1854e23ad96bc73289cd3b11943918c57",
    FINAL_TRAINING_RUNNER: "b3bc733b2204a1657a6d1f62226fb13f04bc3c61d953334bfd04329fc62e3495",
    COMPARATOR_IMPLEMENTATION: "987f170cbfd4ef048161d29a12a9cfd6e715b45338c539b0c9ba7dfe5a04379f",
    DEVELOPMENT_SPLIT: "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140",
    CALIBRATION_MANIFEST: "f8e6a62b96e330a1b21bb9ed56897cbf04b730987cf9e0c31141927abbb84655",
    PRIMARY_CALIBRATION_PROTOCOL: "f569f63c9b9c44816f19ce5f49f724fc5b1e4dc202d6d4a8694250f7b33995e7",
    CALIBRATION_CACHE_SOURCE: "59a5ed886e146461f4cc0c66df3bd61eb51e0eed76441bba61513b4910745473",
    PRIMARY_CALIBRATION_LOCK_SOURCE: "ff0bdcf837bba70dc5b2fcc147bfd0ed5e156949b5de7e96f36f06339100d282",
    PRIMARY_CALIBRATION_RUNNER: "9d8050264362b7af12aa8962c9cc16528c0b75f11eac26040974068ad4e2d522",
}

VARIANT_IDS = [
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
    "B2_RESNET50_MULTITASK",
    "B3_CONVNEXT_TINY_MULTITASK",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
]

EXPECTED_SELECTED_EPOCHS = {
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK": 19,
    "B2_RESNET50_MULTITASK": 13,
    "B3_CONVNEXT_TINY_MULTITASK": 21,
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS": 10,
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT": 8,
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE": 17,
}

EXPECTED_CALIBRATION_ROWS = 2840
EXPECTED_MODALITY_COUNTS = {"CFP": 1313, "UWF": 1171, "OCT": 356}
EXPECTED_GRADE_UNITS = [
    "CFP_FINAL_FUSED_GRADE",
    "UWF_FINAL_FUSED_GRADE",
    "OCT_DME_GRADE",
]
EXPECTED_LESION_UNITS = [
    "lesion_microaneurysm",
    "lesion_hard_exudate",
    "lesion_intraretinal_hemorrhage",
    "lesion_VB_IRMA",
    "lesion_neovascularization",
    "lesion_vitreous_hemorrhage",
    "lesion_retinal_detachment",
]


def banner(text: str) -> None:
    print("=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def section(text: str) -> None:
    print(f"\n{text}")
    print("-" * WIDTH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def all_evidence_gates_pass(payload: dict[str, Any]) -> bool:
    gates = payload.get("gates")
    return (
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and bool(gates)
        and all(bool(value) for value in gates.values())
    )


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    data = json_bytes(payload)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact exists with different content: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    atomic_write(path, data)
    return "CREATED_AND_FROZEN"


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError(
                "A different six-variant calibration protocol source is already frozen."
            )
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    if sha256_file(temporary) != source_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Frozen-source copy verification failed.")
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def mount_type() -> str:
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", "/content/drive"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip().lower()


def no_scientific_calibration_outputs_exist() -> bool:
    if not CALIBRATION_RUN_BASE.exists():
        return True
    prohibited_names = {
        "inference_state.json",
        "calibration_raw_predictions.csv",
        "calibration_temperature_scaled_predictions.csv",
        "temperature_parameters.json",
        "calibration_metrics_before_after_temperature.csv",
        "calibration_completion.json",
    }
    for path in CALIBRATION_RUN_BASE.rglob("*"):
        if path.is_file() and (
            path.name in prohibited_names
            or path.suffix.lower() in {".pt", ".pth"}
            or ("raw_inference_parts" in path.parts and path.suffix.lower() == ".csv")
        ):
            return False
    return True


def main() -> None:
    banner("STAGE 0G-4D-5F-4B — SIX-VARIANT POST-HOC CALIBRATION PROTOCOL LOCK")

    section("[1] REAL GOOGLE DRIVE AND FROZEN PARENT ARTIFACTS")
    filesystem = mount_type()
    gates: dict[str, bool] = {
        "real_Google_Drive_mounted": ROOT.is_dir() and "drive" in filesystem,
    }
    print(f"Mounted filesystem type: {filesystem or '[NONE]'}")
    print(f"{'real_Google_Drive_mounted':<94}: {gates['real_Google_Drive_mounted']}")
    if not gates["real_Google_Drive_mounted"]:
        raise RuntimeError(
            "The real Google Drive is not mounted. Mount Drive in the notebook "
            "process before launching this script."
        )

    for path, expected_sha in EXPECTED_SHA256.items():
        key = f"hash_verified::{path.name}"
        verified = path.is_file() and sha256_file(path) == expected_sha
        gates[key] = verified
        print(f"{path.name:<78}: exists={path.is_file()}, hash_verified={verified}")
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError("A frozen parent artifact changed:\n" + "\n".join(failed))

    section("[2] SIX-RUN COMPLETION AND LOCKED ORDER")
    posthoc = read_json(POSTHOC_PROTOCOL)
    completion = read_json(FINAL_TRAINING_COMPLETION)
    completion_evidence = read_json(FINAL_TRAINING_EVIDENCE)
    primary_calibration = read_json(PRIMARY_CALIBRATION_PROTOCOL)
    primary_calibration_evidence = read_json(PRIMARY_CALIBRATION_PROTOCOL_EVIDENCE)
    primary_calibration_evidence_sha = sha256_file(PRIMARY_CALIBRATION_PROTOCOL_EVIDENCE)
    rows = read_registry(FINAL_TRAINING_REGISTRY)

    protocol_ids = [item["id"] for item in posthoc.get("comparators", [])]
    protocol_ids += [item["id"] for item in posthoc.get("ablations", [])]
    registry_ids = [row.get("variant_id", "") for row in rows]

    completion_gates = {
        "posthoc_protocol_order_exact": protocol_ids == VARIANT_IDS,
        "registry_has_exactly_six_rows": len(rows) == 6,
        "registry_order_exact": registry_ids == VARIANT_IDS,
        "registry_positions_exact": [int(row["variant_order"]) for row in rows] == list(range(1, 7)),
        "all_registry_statuses_COMPLETE": all(row.get("status") == "COMPLETE" for row in rows),
        "all_registry_completion_checks_passed": all(
            parse_bool(row.get("all_completion_checks_passed")) for row in rows
        ),
        "all_registry_runs_early_stopped": all(
            parse_bool(row.get("early_stopped")) and int(row.get("bad_epochs", -1)) == 6
            for row in rows
        ),
        "all_selected_epochs_match_frozen_values": all(
            int(row.get("selected_best_epoch", -1))
            == EXPECTED_SELECTED_EPOCHS[row["variant_id"]]
            for row in rows
        ),
        "six_distinct_best_checkpoint_hashes": len(
            {row.get("best_checkpoint_sha256", "") for row in rows}
        ) == 6,
        "all_best_checkpoint_hashes_well_formed": all(
            is_sha256(row.get("best_checkpoint_sha256")) for row in rows
        ),
        "completion_status_exact": completion.get("status")
        == "SIX_VARIANT_FINAL_TRAINING_COMPLETE",
        "completion_variant_order_exact": completion.get("variants_in_locked_order")
        == VARIANT_IDS,
        "completion_all_six_complete": completion.get("all_six_variants_complete") is True,
        "completion_best_epochs_selected_on_VALIDATION": completion.get(
            "all_best_epochs_selected_on_VALIDATION"
        ) is True,
        "completion_authorizes_CALIBRATION": completion.get("calibration_may_now_begin")
        is True,
        "completion_CALIBRATION_dataset_exact": completion.get("calibration_dataset")
        == "CALIBRATION",
        "completion_locked_test_prohibited": completion.get(
            "locked_test_access_for_new_variants"
        ) == "PROHIBITED",
        "completion_registry_hash_linked": completion.get("final_training_registry_sha256")
        == EXPECTED_SHA256[FINAL_TRAINING_REGISTRY],
        "completion_evidence_PASS": all_evidence_gates_pass(completion_evidence),
        "completion_evidence_reports_no_CALIBRATION_access": completion_evidence.get(
            "calibration_accessed"
        ) is False,
        "completion_evidence_reports_no_locked_test_access": completion_evidence.get(
            "locked_test_accessed"
        ) is False,
    }
    gates.update(completion_gates)
    for name, passed in completion_gates.items():
        print(f"{name:<94}: {passed}")
    if not all(completion_gates.values()):
        failed = [name for name, passed in completion_gates.items() if not passed]
        raise RuntimeError("Six-run completion verification failed:\n" + "\n".join(failed))

    section("[3] INHERITED CALIBRATION METHOD")
    grade_method = primary_calibration.get("grade_temperature_scaling", {})
    lesion_method = primary_calibration.get("lesion_temperature_scaling", {})
    reporting = primary_calibration.get("reporting", {})
    acceptance = primary_calibration.get("acceptance_rules", {})
    partition = primary_calibration.get("calibration_partition", {})
    prohibited = primary_calibration.get("prohibited_actions", [])

    method_gates = {
        "primary_calibration_protocol_locked_before_inference": primary_calibration.get(
            "status"
        ) == "LOCKED_BEFORE_CALIBRATION_INFERENCE",
        "primary_calibration_evidence_PASS": all_evidence_gates_pass(
            primary_calibration_evidence
        ),
        "primary_calibration_evidence_links_frozen_protocol": primary_calibration_evidence.get(
            "calibration_protocol_sha256"
        ) == EXPECTED_SHA256[PRIMARY_CALIBRATION_PROTOCOL],
        "primary_calibration_evidence_links_frozen_source": primary_calibration_evidence.get(
            "source_sha256"
        ) == EXPECTED_SHA256[PRIMARY_CALIBRATION_LOCK_SOURCE],
        "primary_protocol_lock_accessed_no_images_or_models": primary_calibration_evidence.get(
            "medical_images_accessed"
        ) is False
        and primary_calibration_evidence.get("model_loaded") is False
        and primary_calibration_evidence.get("inference_started") is False,
        "primary_protocol_lock_did_not_access_locked_test": primary_calibration_evidence.get(
            "locked_test_accessed"
        ) is False,
        "CALIBRATION_rows_exactly_2840": int(partition.get("rows", -1))
        == EXPECTED_CALIBRATION_ROWS,
        "CALIBRATION_modality_counts_exact": partition.get("modality_counts")
        == EXPECTED_MODALITY_COUNTS,
        "three_grade_temperature_units_exact": grade_method.get("temperature_units")
        == EXPECTED_GRADE_UNITS,
        "seven_lesion_temperature_units_exact": lesion_method.get("temperature_units")
        == EXPECTED_LESION_UNITS,
        "grade_transformation_exact": grade_method.get("transformation")
        == "softmax(log(clamp(p,1e-12,1))/T)",
        "lesion_transformation_exact": lesion_method.get("transformation")
        == "sigmoid(logit/T)",
        "grade_temperature_bounds_exact": grade_method.get("temperature_bounds")
        == [0.05, 20.0],
        "lesion_temperature_bounds_exact": lesion_method.get("temperature_bounds")
        == [0.05, 20.0],
        "grade_log_temperature_bounds_exact": grade_method.get("log_temperature_bounds")
        == [math.log(0.05), math.log(20.0)],
        "lesion_log_temperature_bounds_exact": lesion_method.get("log_temperature_bounds")
        == [math.log(0.05), math.log(20.0)],
        "bounded_optimizer_exact": grade_method.get("optimizer")
        == "SCIPY_MINIMIZE_SCALAR_BOUNDED_ON_LOG_T"
        and lesion_method.get("optimizer")
        == "SCIPY_MINIMIZE_SCALAR_BOUNDED_ON_LOG_T",
        "optimizer_tolerance_and_iterations_exact": grade_method.get("xatol") == 1e-6
        and lesion_method.get("xatol") == 1e-6
        and int(grade_method.get("maximum_iterations", -1)) == 500
        and int(lesion_method.get("maximum_iterations", -1)) == 500,
        "identity_candidate_exact": grade_method.get("identity_candidate_T") == 1.0
        and lesion_method.get("identity_candidate_T") == 1.0,
        "identity_fallback_rule_exact": grade_method.get("fallback_rule")
        == "USE_T_1_IF_OPTIMIZED_NLL_IS_NOT_STRICTLY_LOWER"
        and lesion_method.get("fallback_rule")
        == "USE_T_1_IF_OPTIMIZED_NLL_IS_NOT_STRICTLY_LOWER",
        "grade_argmax_invariance_locked": grade_method.get("decision_rule")
        == "ARGMAX_UNCHANGED_BY_POSITIVE_SCALAR_TEMPERATURE",
        "lesion_threshold_0_5_locked": lesion_method.get("decision_threshold") == 0.5,
        "lesion_threshold_invariance_locked": lesion_method.get("threshold_invariance")
        == "PRESERVED_BY_POSITIVE_SCALAR_TEMPERATURE",
        "metrics_before_after_exact": reporting.get("metrics_before_and_after")
        == ["NLL", "BRIER_SCORE", "ECE_15_EQUAL_WIDTH_BINS"],
        "probability_epsilon_exact": reporting.get("probability_epsilon") == 1e-12,
        "ECE_bins_exact": int(reporting.get("ece_bins", -1)) == 15,
        "acceptance_rules_inherited": all(
            acceptance.get(name) is True
            for name in [
                "all_temperatures_finite_and_within_bounds",
                "calibrated_NLL_must_not_exceed_identity_NLL",
                "grade_argmax_predictions_must_be_unchanged",
                "lesion_predictions_at_0_5_must_be_unchanged",
                "all_saved_probabilities_finite_bounded_and_normalized",
            ]
        ),
        "weight_update_prohibited": "MODEL_WEIGHT_UPDATE" in prohibited,
        "checkpoint_reselection_prohibited": "CHECKPOINT_RESELECTION" in prohibited,
        "architecture_change_prohibited": "ARCHITECTURE_CHANGE" in prohibited,
        "VALIDATION_reuse_prohibited": "VALIDATION_REUSE_FOR_CALIBRATION_FIT" in prohibited,
        "locked_test_access_prohibited": "LOCKED_TEST_ACCESS" in prohibited,
        "threshold_optimization_prohibited": "DECISION_THRESHOLD_OPTIMIZATION_IN_THIS_STAGE"
        in prohibited,
    }
    gates.update(method_gates)
    for name, passed in method_gates.items():
        print(f"{name:<94}: {passed}")
    if not all(method_gates.values()):
        failed = [name for name, passed in method_gates.items() if not passed]
        raise RuntimeError("Inherited calibration-method verification failed:\n" + "\n".join(failed))

    section("[4] SELECTED CHECKPOINTS AND RESUMABLE OUTPUT CONTRACT")
    variant_contracts: list[dict[str, Any]] = []
    checkpoint_presence: dict[str, bool] = {}
    for row in rows:
        variant_id = row["variant_id"]
        checkpoint = CHECKPOINT_BASE / variant_id / "best_model.pt"
        run_root = CALIBRATION_RUN_BASE / variant_id
        slug = variant_id.lower()
        exists_nonempty = checkpoint.is_file() and checkpoint.stat().st_size > 0
        checkpoint_presence[variant_id] = exists_nonempty
        print(f"{variant_id:<78}: selected_epoch={row['selected_best_epoch']}, exists_nonempty={exists_nonempty}")
        variant_contracts.append(
            {
                "variant_order": int(row["variant_order"]),
                "variant_id": variant_id,
                "family": row["family"],
                "backbone": row["backbone"],
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": row["best_checkpoint_sha256"],
                "selected_epoch": int(row["selected_best_epoch"]),
                "selection_partition": "VALIDATION",
                "pre_execution_checkpoint_SHA256_verification_required": True,
                "run_root": str(run_root),
                "raw_part_root": str(run_root / "raw_inference_parts"),
                "inference_state": str(run_root / "inference_state.json"),
                "raw_predictions": str(run_root / "calibration_raw_predictions.csv"),
                "calibrated_predictions": str(
                    run_root / "calibration_temperature_scaled_predictions.csv"
                ),
                "temperature_parameters": str(run_root / "temperature_parameters.json"),
                "calibration_metrics": str(
                    run_root / "calibration_metrics_before_after_temperature.csv"
                ),
                "latest_session_evidence": str(
                    ROOT / f"Evidence/stage0g4d5f4c_{slug}_cpu_calibration_latest_session.json"
                ),
                "completion_evidence": str(
                    ROOT / f"Evidence/stage0g4d5f4c_{slug}_cpu_calibration_completion.json"
                ),
            }
        )

    output_precondition = no_scientific_calibration_outputs_exist()
    execution_gates = {
        "all_six_selected_checkpoints_exist_nonempty": all(checkpoint_presence.values()),
        "all_selected_checkpoint_paths_distinct": len(
            {item["selected_checkpoint"] for item in variant_contracts}
        ) == 6,
        "all_selected_checkpoint_hashes_distinct": len(
            {item["selected_checkpoint_sha256"] for item in variant_contracts}
        ) == 6,
        "posthoc_calibration_not_started_before_protocol_lock": output_precondition,
        "execution_backend_locked_to_CPU_FLOAT32": True,
        "one_variant_at_a_time_locked": True,
        "resumable_parts_locked": True,
        "model_parameters_may_not_update": True,
        "locked_test_remains_prohibited": True,
    }
    gates.update(execution_gates)
    for name, passed in execution_gates.items():
        print(f"{name:<94}: {passed}")
    if not all(execution_gates.values()):
        failed = [name for name, passed in execution_gates.items() if not passed]
        raise RuntimeError("Calibration execution-contract verification failed:\n" + "\n".join(failed))

    section("[5] PROTOCOL, SOURCE, AND EVIDENCE FREEZE")
    source_action, source_sha = freeze_source()
    gates["protocol_lock_source_frozen_and_verified"] = (
        FROZEN_SOURCE.is_file() and sha256_file(FROZEN_SOURCE) == source_sha
    )

    protocol_payload: dict[str, Any] = {
        "protocol_id": "MMRDR_FOVEA_DR_POSTHOC_SIX_VARIANT_CALIBRATION_V1",
        "stage": "0G-4D-5F-4B",
        "status": "LOCKED_BEFORE_SIX_VARIANT_CALIBRATION_INFERENCE",
        "classification": "POST_HOC_CALIBRATION_OF_SIX_FROZEN_VALIDATION_SELECTED_MODELS",
        "frozen_parent_artifacts": {
            str(path): sha for path, sha in EXPECTED_SHA256.items()
        },
        "primary_calibration_protocol_evidence": {
            "path": str(PRIMARY_CALIBRATION_PROTOCOL_EVIDENCE),
            "sha256": primary_calibration_evidence_sha,
            "status": "PASS",
        },
        "final_training_registry": {
            "path": str(FINAL_TRAINING_REGISTRY),
            "sha256": EXPECTED_SHA256[FINAL_TRAINING_REGISTRY],
        },
        "protocol_lock_source": {
            "path": str(FROZEN_SOURCE),
            "sha256": source_sha,
        },
        "planned_calibration_runner": str(PLANNED_RUNNER),
        "execution_order": VARIANT_IDS,
        "execution": {
            "device": "CPU",
            "precision": "FLOAT32",
            "one_variant_at_a_time": True,
            "run_seed": 2026,
            "input_size": "512x512_RGB",
            "batch_size": 4,
            "workers": 2,
            "rows_per_atomic_part": 32,
            "default_rows_per_session": 64,
            "sample_order": "SORTED_NORMALIZED_RELATIVE_PATH",
            "resume_unit": "VERIFIED_ATOMIC_32_ROW_PART",
            "cache_root": "/content/MMRDR_calibration_v1",
            "cache_is_temporary_and_must_be_restored_each_new_runtime": True,
            "checkpoint_tensor_payload_must_be_loaded_on_CPU": True,
            "checkpoint_SHA256_must_be_verified_immediately_before_load": True,
            "model_eval_mode_required": True,
            "torch_no_grad_required": True,
        },
        "calibration_partition": {
            "name": "CALIBRATION",
            "rows": EXPECTED_CALIBRATION_ROWS,
            "modality_counts": EXPECTED_MODALITY_COUNTS,
            "manifest": str(CALIBRATION_MANIFEST),
            "manifest_sha256": EXPECTED_SHA256[CALIBRATION_MANIFEST],
            "development_split": str(DEVELOPMENT_SPLIT),
            "development_split_sha256": EXPECTED_SHA256[DEVELOPMENT_SPLIT],
            "may_update_model_parameters": False,
            "may_select_model_checkpoint": False,
            "may_change_architecture": False,
            "may_change_decision_threshold": False,
        },
        "grade_temperature_scaling": grade_method,
        "lesion_temperature_scaling": lesion_method,
        "reporting": reporting,
        "acceptance_rules": acceptance,
        "variants": variant_contracts,
        "completion_rule_per_variant": {
            "all_2840_rows_inferred_exactly_once": True,
            "raw_predictions_frozen_and_hashed": True,
            "ten_temperatures_fitted_under_inherited_method": True,
            "calibrated_predictions_frozen_and_hashed": True,
            "metrics_before_and_after_frozen_and_hashed": True,
            "grade_and_lesion_decision_invariance_verified": True,
            "completion_evidence_PASS_required": True,
        },
        "global_completion_rule": {
            "all_six_variants_must_complete": True,
            "all_six_results_must_be_reported": True,
            "no_variant_may_be_omitted_for_unfavorable_calibration_results": True,
            "calibration_comparison_registry_freezes_after_all_six_complete": True,
        },
        "prohibited_actions": [
            "MODEL_WEIGHT_UPDATE",
            "OPTIMIZER_CREATION",
            "CHECKPOINT_RESELECTION",
            "ARCHITECTURE_CHANGE",
            "HYPERPARAMETER_SELECTION",
            "VALIDATION_REUSE_FOR_CALIBRATION_FIT",
            "DECISION_THRESHOLD_OPTIMIZATION",
            "LOCKED_INTERNAL_TEST_IMAGE_ACCESS",
            "LOCKED_INTERNAL_TEST_LABEL_ACCESS",
            "LOCKED_INTERNAL_TEST_PREDICTION_GENERATION",
            "LOCKED_INTERNAL_TEST_METRIC_RECOMPUTATION",
        ],
        "existing_primary_FOVEA_DR_calibration_result": "PRESERVED_UNCHANGED",
        "existing_primary_FOVEA_DR_locked_test_result": "PRESERVED_UNCHANGED",
        "next_required_stage": "RESTORE_CALIBRATION_CACHE_THEN_RUN_B1_POSTHOC_CPU_CALIBRATION",
    }

    protocol_action = frozen_json(PROTOCOL_OUTPUT, protocol_payload)
    protocol_sha = sha256_file(PROTOCOL_OUTPUT)
    gates.update(
        {
            "protocol_artifact_frozen_and_verified": PROTOCOL_OUTPUT.is_file()
            and sha256_file(PROTOCOL_OUTPUT) == protocol_sha,
            "model_not_instantiated_this_stage": True,
            "checkpoint_tensor_payload_not_loaded_this_stage": True,
            "split_rows_not_parsed_this_stage": True,
            "calibration_manifest_rows_not_parsed_this_stage": True,
            "images_not_accessed_this_stage": True,
            "temperature_not_fitted_this_stage": True,
            "optimizer_not_created_this_stage": True,
            "parameters_not_updated_this_stage": True,
            "VALIDATION_not_accessed_this_stage": True,
            "CALIBRATION_not_accessed_this_stage": True,
            "locked_test_not_accessed_this_stage": True,
        }
    )
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError("Final calibration-protocol gates failed:\n" + "\n".join(failed))

    evidence_payload: dict[str, Any] = {
        "stage": "0G-4D-5F-4B",
        "status": "PASS",
        "purpose": "LOCK_SIX_VARIANT_POSTHOC_CALIBRATION_METHOD_BEFORE_INFERENCE",
        "source_path": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "protocol_path": str(PROTOCOL_OUTPUT),
        "protocol_sha256": protocol_sha,
        "variant_order": VARIANT_IDS,
        "selected_checkpoints": {
            item["variant_id"]: {
                "path": item["selected_checkpoint"],
                "sha256": item["selected_checkpoint_sha256"],
                "selected_epoch": item["selected_epoch"],
            }
            for item in variant_contracts
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "model_instantiated": False,
        "checkpoint_tensor_payload_loaded": False,
        "split_rows_parsed": False,
        "calibration_manifest_rows_parsed": False,
        "images_accessed": False,
        "temperature_fitted": False,
        "optimizer_created": False,
        "parameters_updated": False,
        "validation_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "next_required_stage": "RESTORE_CALIBRATION_CACHE_THEN_RUN_B1_POSTHOC_CPU_CALIBRATION",
    }
    evidence_action = frozen_json(EVIDENCE_OUTPUT, evidence_payload)
    evidence_sha = sha256_file(EVIDENCE_OUTPUT)
    os.sync()

    print("Source action   :", source_action)
    print("Source          :", FROZEN_SOURCE)
    print("Source SHA-256  :", source_sha)
    print("Protocol action :", protocol_action)
    print("Protocol        :", PROTOCOL_OUTPUT)
    print("Protocol SHA-256:", protocol_sha)
    print("Evidence action :", evidence_action)
    print("Evidence        :", EVIDENCE_OUTPUT)
    print("Evidence SHA-256:", evidence_sha)

    section("SIX-VARIANT CALIBRATION PROTOCOL GATES")
    for name, passed in gates.items():
        print(f"{name:<102}: {passed}")

    banner("STAGE 0G-4D-5F-4B STATUS: PASS")
    print("Six frozen checkpoints locked : YES")
    print("Calibration method inherited  : YES — UNCHANGED")
    print("Execution backend             : CPU / FLOAT32")
    print("Model instantiated            : NO")
    print("Checkpoint tensors loaded     : NO")
    print("Images accessed               : NO")
    print("Temperatures fitted           : NO")
    print("Parameters updated            : NO")
    print("CALIBRATION accessed          : NO")
    print("Locked test accessed          : NO")
    print("\nNEXT ACTION: RESTORE CALIBRATION CACHE, THEN RUN B1 POST-HOC CALIBRATION")


if __name__ == "__main__":
    main()
