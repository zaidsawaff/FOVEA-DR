#!/usr/bin/env python3
"""Schema-safe repair for the six-variant calibration completion freeze.

The original 5F-4D run completed every scientific and provenance gate, froze
its source, registry, and completion contract, then stopped while serializing
the final evidence because nested NumPy boolean scalars are not JSON-native.
This repair re-verifies the already frozen artifacts, converts diagnostic
scalars to Python-native values, and freezes the missing completion evidence.
It performs no model load, image access, inference, fitting, or parameter update.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WIDTH = 124
ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

ORIGINAL_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4d_six_variant_calibration_completion_freeze.py"
REPAIR_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4d_r1_six_variant_calibration_completion_freeze_repair.py"
PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_six_variant_calibration_protocol_v1.json"
CALIBRATION_RUNNER = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4c_resumable_six_variant_cpu_calibration.py"
REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_six_variant_calibration_registry_v1.csv"
COMPLETION_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_six_variant_calibration_completion_v1.json"
REPAIR_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_six_variant_calibration_completion_repair_v1.json"
FINAL_EVIDENCE = ROOT / "Evidence/stage0g4d5f4d_six_variant_calibration_completion_freeze.json"
LOG_ROOT = ROOT / "Logs/FOVEA_DR/posthoc_calibration"

ORIGINAL_SOURCE_SHA256 = "3bb166b67f89e3991b7044ada3c0078add6846ef1efa5286801e9bb3fa1429ea"
PROTOCOL_SHA256 = "9bdd3a5cf4f4ead1dc3055fd5c84b62ca1bea929a95bf96372ec8bea66487c4d"
RUNNER_SHA256 = "a21db2d7a3e247ddfdb9a25bcc12d4931076fd37c3e41982453b611a94b5fd0e"

VARIANT_IDS = [
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
    "B2_RESNET50_MULTITASK",
    "B3_CONVNEXT_TINY_MULTITASK",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
]

EXPECTED_ROWS = 2840
EXPECTED_TOTAL_ROWS = 17040


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
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def json_bytes(payload: dict[str, Any]) -> bytes:
    normalized = to_builtin(payload)
    return (json.dumps(normalized, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    data = json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact exists with different content: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    atomic_write(path, data)
    return "CREATED_AND_FROZEN"


def freeze_repair_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    REPAIR_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if REPAIR_SOURCE.exists():
        if sha256_file(REPAIR_SOURCE) != source_sha:
            raise RuntimeError("A different 5F-4D-R1 repair source is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = REPAIR_SOURCE.with_name(REPAIR_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    if sha256_file(temporary) != source_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Repair-source copy verification failed.")
    os.replace(temporary, REPAIR_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def real_drive_mounted() -> bool:
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", "/content/drive"],
        capture_output=True,
        text=True,
        check=False,
    )
    return ROOT.is_dir() and "drive" in result.stdout.strip().lower()


def import_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to import frozen source: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def truthy_series(series: pd.Series) -> bool:
    accepted = {"true", "1", "yes", "y", "t"}
    return bool(series.astype(str).str.strip().str.lower().isin(accepted).all())


def locate_failure_log() -> Path | None:
    candidates = sorted(
        LOG_ROOT.glob("six_variant_calibration_completion_freeze_*.log"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    return candidates[-1] if candidates else None


def compare_registry(registry: pd.DataFrame, summaries: list[dict[str, Any]]) -> bool:
    if len(registry) != 6 or list(registry["variant_id"].astype(str)) != VARIANT_IDS:
        return False
    if "all_checks_passed" not in registry.columns or not truthy_series(registry["all_checks_passed"]):
        return False
    summary_by_variant = {item["variant_id"]: item for item in summaries}
    exact_columns = [
        "selected_checkpoint_sha256",
        "raw_predictions_sha256",
        "calibrated_predictions_sha256",
        "temperature_parameters_sha256",
        "calibration_metrics_sha256",
        "inference_state_sha256",
        "completion_evidence_sha256",
    ]
    for row in registry.to_dict(orient="records"):
        summary = summary_by_variant.get(str(row["variant_id"]))
        if summary is None:
            return False
        if int(row["completed_rows"]) != EXPECTED_ROWS or int(row["temperature_count"]) != 10:
            return False
        if any(str(row[column]) != str(summary[column]) for column in exact_columns):
            return False
        for column in [
            "grade_weighted_NLL_before",
            "grade_weighted_NLL_after",
            "lesion_weighted_NLL_before",
            "lesion_weighted_NLL_after",
            "CFP_temperature",
            "UWF_temperature",
            "OCT_temperature",
        ]:
            if not math.isclose(float(row[column]), float(summary[column]), rel_tol=0.0, abs_tol=1e-12):
                return False
    return True


def main() -> None:
    banner("STAGE 0G-4D-5F-4D-R1 — CALIBRATION COMPLETION EVIDENCE SERIALIZATION REPAIR")
    if not real_drive_mounted():
        raise RuntimeError("The real Google Drive is not mounted in this notebook process.")

    section("[1] ORIGINAL FREEZE OUTPUTS AND FAILURE CAUSE")
    failure_log = locate_failure_log()
    failure_text = (
        failure_log.read_text(encoding="utf-8", errors="replace")
        if failure_log is not None
        else ""
    )
    initial_gates: dict[str, bool] = {
        "original_source_exists": ORIGINAL_SOURCE.is_file(),
        "original_source_hash_verified": ORIGINAL_SOURCE.is_file()
        and sha256_file(ORIGINAL_SOURCE) == ORIGINAL_SOURCE_SHA256,
        "protocol_hash_verified": PROTOCOL.is_file() and sha256_file(PROTOCOL) == PROTOCOL_SHA256,
        "runner_hash_verified": CALIBRATION_RUNNER.is_file()
        and sha256_file(CALIBRATION_RUNNER) == RUNNER_SHA256,
        "registry_exists_nonempty": REGISTRY.is_file() and REGISTRY.stat().st_size > 0,
        "completion_contract_exists_nonempty": COMPLETION_CONTRACT.is_file()
        and COMPLETION_CONTRACT.stat().st_size > 0,
        "failure_log_exists": failure_log is not None and failure_log.is_file(),
        "failure_occurred_only_at_evidence_JSON_serialization": (
            "TypeError: Object of type bool is not JSON serializable" in failure_text
            and "STAGE 0G-4D-5F-4D STATUS: FAIL" in failure_text
        ),
        "failure_log_reports_all_six_pairs_PASS": all(
            f"{variant_id}" in failure_text and "PASS=True" in failure_text
            for variant_id in VARIANT_IDS
        ),
    }
    for name, passed in initial_gates.items():
        print(f"{name:<92}: {passed}")
    if not all(initial_gates.values()):
        failed = [name for name, passed in initial_gates.items() if not passed]
        raise RuntimeError("Original freeze recovery gate failed:\n" + "\n".join(failed))

    original = import_module(ORIGINAL_SOURCE, "fovea_calibration_completion_original")
    protocol = read_json(PROTOCOL)
    variant_contracts = protocol.get("variants", [])

    section("[2] RE-RUN READ-ONLY SIX-VARIANT VERIFICATION")
    summaries: list[dict[str, Any]] = []
    variant_checks: dict[str, dict[str, bool]] = {}
    for position, variant_id in enumerate(VARIANT_IDS):
        matches = [
            item for item in variant_contracts
            if isinstance(item, dict) and item.get("variant_id") == variant_id
        ]
        if len(matches) != 1:
            raise RuntimeError(f"No unique protocol contract for {variant_id}.")
        summary, raw_checks = original.inspect_variant(
            position,
            matches[0],
            PROTOCOL_SHA256,
            RUNNER_SHA256,
        )
        checks = {str(name): bool(value) for name, value in raw_checks.items()}
        checks["unique_protocol_contract"] = True
        summaries.append(to_builtin(summary))
        variant_checks[variant_id] = checks
        print(f"{variant_id:<76}: PASS={all(checks.values())}")
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"Variant verification failed for {variant_id}:\n" + "\n".join(failed))

    registry = pd.read_csv(REGISTRY)
    completion_contract = read_json(COMPLETION_CONTRACT)
    registry_sha = sha256_file(REGISTRY)
    completion_contract_sha = sha256_file(COMPLETION_CONTRACT)
    existing_final_evidence_valid = True
    if FINAL_EVIDENCE.exists():
        try:
            existing_final_evidence = read_json(FINAL_EVIDENCE)
            existing_final_evidence_valid = bool(
                existing_final_evidence.get("stage") == "0G-4D-5F-4D-R1"
                and existing_final_evidence.get("status") == "PASS"
                and existing_final_evidence.get("repair_scope")
                == "JSON_NATIVE_BOOLEAN_NORMALIZATION_ONLY"
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_final_evidence_valid = False

    section("[3] FROZEN REGISTRY AND COMPLETION CONTRACT")
    contract_variants = completion_contract.get("variants", [])
    artifact_gates = {
        "registry_has_exact_six_verified_rows": compare_registry(registry, summaries),
        "contract_protocol_exact": completion_contract.get("protocol")
        == "FOVEA_DR_POSTHOC_SIX_VARIANT_CALIBRATION_COMPLETION_V1",
        "contract_stage_exact": completion_contract.get("stage") == "0G-4D-5F-4D",
        "contract_status_complete_and_frozen": completion_contract.get("status")
        == "COMPLETE_AND_FROZEN",
        "contract_execution_order_exact": completion_contract.get("execution_order") == VARIANT_IDS,
        "contract_backend_CPU_FLOAT32": completion_contract.get("execution_backend") == "CPU_FLOAT32",
        "contract_rows_per_variant_exact": int(
            completion_contract.get("calibration_rows_per_variant", -1)
        ) == EXPECTED_ROWS,
        "contract_total_rows_exact": int(completion_contract.get("total_calibration_rows", -1))
        == EXPECTED_TOTAL_ROWS,
        "contract_temperatures_per_variant_exact": int(
            completion_contract.get("temperatures_per_variant", -1)
        ) == 10,
        "contract_protocol_hash_verified": completion_contract.get("calibration_protocol_sha256")
        == PROTOCOL_SHA256,
        "contract_runner_hash_verified": completion_contract.get("calibration_runner_sha256")
        == RUNNER_SHA256,
        "contract_original_source_hash_verified": completion_contract.get(
            "completion_freeze_source_sha256"
        ) == ORIGINAL_SOURCE_SHA256,
        "contract_registry_hash_verified": completion_contract.get("registry_sha256") == registry_sha,
        "contract_has_exact_six_variant_summaries": isinstance(contract_variants, list)
        and len(contract_variants) == 6,
        "contract_variant_order_exact": isinstance(contract_variants, list)
        and [item.get("variant_id") for item in contract_variants] == VARIANT_IDS,
        "contract_no_parameter_updates": completion_contract.get("model_parameters_updated") is False,
        "contract_no_checkpoint_reselection": completion_contract.get(
            "checkpoint_reselection_performed"
        ) is False,
        "contract_no_VALIDATION_fit": completion_contract.get(
            "validation_used_for_calibration_fit"
        ) is False,
        "contract_no_locked_test": completion_contract.get("locked_test_accessed") is False,
        "final_evidence_absent_or_valid_R1": existing_final_evidence_valid,
    }
    for name, passed in artifact_gates.items():
        print(f"{name:<92}: {passed}")
    if not all(artifact_gates.values()):
        failed = [name for name, passed in artifact_gates.items() if not passed]
        raise RuntimeError("Frozen completion artifact gate failed:\n" + "\n".join(failed))

    repair_source_action, repair_source_sha = freeze_repair_source()
    repair_contract_payload = {
        "protocol": "FOVEA_DR_POSTHOC_SIX_VARIANT_CALIBRATION_COMPLETION_REPAIR_V1",
        "stage": "0G-4D-5F-4D-R1",
        "status": "EVIDENCE_SERIALIZATION_REPAIRED",
        "repair_scope": "JSON_NATIVE_BOOLEAN_NORMALIZATION_ONLY",
        "original_source": str(ORIGINAL_SOURCE),
        "original_source_sha256": ORIGINAL_SOURCE_SHA256,
        "repair_source": str(REPAIR_SOURCE),
        "repair_source_sha256": repair_source_sha,
        "failure_log": str(failure_log),
        "failure_log_sha256": sha256_file(failure_log),
        "registry": str(REGISTRY),
        "registry_sha256": registry_sha,
        "completion_contract": str(COMPLETION_CONTRACT),
        "completion_contract_sha256": completion_contract_sha,
        "scientific_results_changed": False,
        "registry_changed": False,
        "completion_contract_changed": False,
        "calibration_artifacts_changed": False,
        "model_loaded": False,
        "images_accessed": False,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "validation_used_for_calibration_fit": False,
        "locked_test_accessed": False,
    }
    repair_contract_action = frozen_json(REPAIR_CONTRACT, repair_contract_payload)
    repair_contract_sha = sha256_file(REPAIR_CONTRACT)

    final_gates = {
        **initial_gates,
        **artifact_gates,
        "all_six_variant_checks_passed": all(
            all(checks.values()) for checks in variant_checks.values()
        ),
        "all_nested_diagnostic_values_JSON_native": all(
            isinstance(value, bool)
            for checks in variant_checks.values()
            for value in checks.values()
        ),
        "repair_source_frozen_and_verified": REPAIR_SOURCE.is_file()
        and sha256_file(REPAIR_SOURCE) == repair_source_sha,
        "repair_contract_frozen_and_verified": REPAIR_CONTRACT.is_file()
        and sha256_file(REPAIR_CONTRACT) == repair_contract_sha,
        "scientific_results_unchanged": True,
        "calibration_artifacts_unchanged": True,
        "model_not_loaded_this_stage": True,
        "images_not_accessed_this_stage": True,
        "optimizer_not_created_this_stage": True,
        "parameters_not_updated_this_stage": True,
        "VALIDATION_not_used_for_fit_this_stage": True,
        "locked_test_not_accessed_this_stage": True,
    }

    section("[4] CORRECTED COMPLETION EVIDENCE")
    for name, passed in final_gates.items():
        print(f"{name:<100}: {passed}")
    if not all(final_gates.values()):
        failed = [name for name, passed in final_gates.items() if not passed]
        raise RuntimeError("Corrected evidence gate failed:\n" + "\n".join(failed))

    evidence_payload = {
        "stage": "0G-4D-5F-4D-R1",
        "status": "PASS",
        "completion_status": "SIX_VARIANT_CALIBRATION_COMPLETE_AND_FROZEN",
        "repair_scope": "JSON_NATIVE_BOOLEAN_NORMALIZATION_ONLY",
        "calibration_protocol": str(PROTOCOL),
        "calibration_protocol_sha256": PROTOCOL_SHA256,
        "calibration_runner": str(CALIBRATION_RUNNER),
        "calibration_runner_sha256": RUNNER_SHA256,
        "original_completion_freeze_source": str(ORIGINAL_SOURCE),
        "original_completion_freeze_source_sha256": ORIGINAL_SOURCE_SHA256,
        "repair_source": str(REPAIR_SOURCE),
        "repair_source_sha256": repair_source_sha,
        "registry": str(REGISTRY),
        "registry_sha256": registry_sha,
        "completion_contract": str(COMPLETION_CONTRACT),
        "completion_contract_sha256": completion_contract_sha,
        "repair_contract": str(REPAIR_CONTRACT),
        "repair_contract_sha256": repair_contract_sha,
        "failure_log": str(failure_log),
        "failure_log_sha256": sha256_file(failure_log),
        "variant_order": VARIANT_IDS,
        "variant_checks": variant_checks,
        "gates": final_gates,
        "model_loaded": False,
        "images_accessed": False,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "validation_used_for_calibration_fit": False,
        "locked_test_accessed": False,
        "next_required_stage": "GENERATE_SIX_VARIANT_POSTHOC_COMPARISON_OUTPUTS",
    }
    evidence_action = frozen_json(FINAL_EVIDENCE, evidence_payload)
    evidence_sha = sha256_file(FINAL_EVIDENCE)
    os.sync()

    section("[5] REPAIR CONTRACT, SOURCE, AND FINAL EVIDENCE")
    print("Original registry SHA-256 :", registry_sha)
    print("Original contract SHA-256 :", completion_contract_sha)
    print("Repair source action       :", repair_source_action)
    print("Repair source              :", REPAIR_SOURCE)
    print("Repair source SHA-256      :", repair_source_sha)
    print("Repair contract action     :", repair_contract_action)
    print("Repair contract            :", REPAIR_CONTRACT)
    print("Repair contract SHA-256    :", repair_contract_sha)
    print("Final evidence action      :", evidence_action)
    print("Final evidence             :", FINAL_EVIDENCE)
    print("Final evidence SHA-256     :", evidence_sha)

    banner("STAGE 0G-4D-5F-4D-R1 STATUS: PASS")
    print("Six calibrations complete : YES")
    print("Total calibration rows    : 17040")
    print("Temperatures fitted       : 60 / 60")
    print("Scientific results changed: NO")
    print("Registry changed          : NO")
    print("Completion contract changed: NO")
    print("Model loaded              : NO")
    print("Images accessed           : NO")
    print("Parameters updated        : NO")
    print("VALIDATION used for fit   : NO")
    print("Locked test accessed      : NO")
    print("\nNEXT ACTION: GENERATE SIX-VARIANT POST-HOC COMPARISON OUTPUTS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        banner("STAGE 0G-4D-5F-4D-R1 STATUS: FAIL")
        raise
