#!/usr/bin/env python3
"""Freeze the label-free prediction schema before one-time test scoring."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

MEMBERSHIP = ROOT / "Splits/mmrdr_locked_test_membership_v1_FROZEN.csv"
RAW = ROOT / "Runs/FOVEA_DR/locked_test_seed2026_cpu_fp32/locked_test_raw_predictions_v1.csv"
CAL = ROOT / "Runs/FOVEA_DR/locked_test_seed2026_cpu_fp32/locked_test_calibrated_predictions_v1.csv"
LABEL_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_locked_test_label_source_contract_v1.json"
EVALUATION_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_locked_test_single_evaluation_contract_v1_1.json"
R3_EVIDENCE = ROOT / "Evidence/stage0g4d5e3a_r3_source_index_bridged_label_contract.json"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5e3b0_prediction_schema_preflight.py"
SCHEMA_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_locked_test_prediction_schema_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5e3b0_prediction_schema_preflight.json"

EXPECTED_SHA = {
    MEMBERSHIP: "56913e5067d371f8631ef8deb56db585bb21f8591f97a88bdbedc07b29e4f45b",
    RAW: "67b612d48fea47256ad75c7482572b4037807b6f2c3f087a5f11fbc0f9d75e23",
    CAL: "65ecfa3d102c3d2c08551c3799084b02d28e4a44cbf475c6d7757daa772e1f78",
    LABEL_CONTRACT: "fdd8151603c6afc482f26892d47b791809513880708cefff18f767fa4039958b",
    EVALUATION_CONTRACT: "984d0e13fbdd178bbeff42ff97e69465cd760b9434e86cad414ca93e5c0d7f10",
    R3_EVIDENCE: "4ab72b63e9a10f5f8fa71199da62892bc9fd0949c9fbf30443bd13115a335595",
}

FORBIDDEN_LABEL_TOKENS = (
    "true_grade",
    "target",
    "label",
    "ground_truth",
    "lesion_loss_mask",
    "grade_loss_mask",
)

FINAL_OUTPUTS = [
    ROOT / "Audit/mmrdr_fovea_dr_LOCKED_TEST_final_metrics_v1.csv",
    ROOT / "Audit/mmrdr_fovea_dr_LOCKED_TEST_bootstrap_CI_v1.csv",
    ROOT / "Evidence/stage0g4d5e3b_locked_test_single_evaluation_results.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def csv_data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        total_lines = sum(block.count(b"\n") for block in iter(lambda: handle.read(1024 * 1024), b""))
    return max(0, total_lines - 1)


def frozen_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact exists with different content: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return "CREATED_AND_FROZEN"


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    current_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != current_sha:
            raise RuntimeError("Frozen schema-preflight source differs from the executed source.")
        action = "EXISTING_IDENTICAL_SOURCE_VERIFIED"
    else:
        temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
        shutil.copyfile(current, temporary)
        os.replace(temporary, FROZEN_SOURCE)
        action = "COPIED_AND_FROZEN"
    return action, current_sha


def contract_subset(value: object, prefix: str = "") -> dict[str, object]:
    selected = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).casefold()
            if any(token in lowered for token in ("metric", "endpoint", "bootstrap", "confidence", "ece", "threshold", "replicate", "seed")):
                selected[path] = child
            else:
                selected.update(contract_subset(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            selected.update(contract_subset(child, f"{prefix}[{index}]"))
    return selected


def classify_columns(header: list[str]) -> dict[str, list[str]]:
    groups = {
        "identity": [],
        "grade_logits": [],
        "grade_probabilities": [],
        "lesion_logits": [],
        "lesion_probabilities": [],
        "other": [],
    }
    for column in header:
        lowered = column.casefold()
        if lowered in {"locked_test_index", "test_index", "row_index", "relative_path", "modality", "image", "image_id"}:
            group = "identity"
        elif "grade" in lowered and "logit" in lowered:
            group = "grade_logits"
        elif "grade" in lowered and any(token in lowered for token in ("prob", "softmax")):
            group = "grade_probabilities"
        elif "lesion" in lowered and "logit" in lowered:
            group = "lesion_logits"
        elif "lesion" in lowered and any(token in lowered for token in ("prob", "sigmoid")):
            group = "lesion_probabilities"
        else:
            group = "other"
        groups[group].append(column)
    return groups


def main() -> None:
    print("=" * 80)
    print("STAGE 0G-4D-5E-3B-0 — LABEL-FREE FINAL SCORING SCHEMA PREFLIGHT")
    print("=" * 80)

    print("\n[1] FROZEN INPUT VERIFICATION")
    print("-" * 80)
    hash_gates = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        hash_gates[f"hash_verified::{path.name}"] = verified
        print(f"{path.name:76s}: exists={exists}, hash_verified={verified}")
    if not all(hash_gates.values()):
        raise RuntimeError("A frozen prerequisite is missing or changed.")

    with R3_EVIDENCE.open("r", encoding="utf-8") as handle:
        r3 = json.load(handle)
    r3_pass = r3.get("status") == "PASS" and not r3.get("single_scientific_evaluation_consumed", True)
    results_absent = not any(path.exists() for path in FINAL_OUTPUTS)
    print("R3 evidence PASS                    :", r3_pass)
    print("Final scientific outputs absent    :", results_absent)
    if not (r3_pass and results_absent):
        raise RuntimeError("Single-evaluation precondition failed.")

    print("\n[2] LABEL-FREE PREDICTION HEADERS")
    print("-" * 80)
    raw_header = csv_header(RAW)
    cal_header = csv_header(CAL)
    raw_rows = csv_data_rows(RAW)
    cal_rows = csv_data_rows(CAL)
    raw_groups = classify_columns(raw_header)
    cal_groups = classify_columns(cal_header)

    print("Raw rows       :", raw_rows)
    print("Calibrated rows:", cal_rows)
    print("\nRAW COLUMNS:")
    for index, column in enumerate(raw_header):
        print(f"  [{index:02d}] {column}")
    print("\nCALIBRATED COLUMNS:")
    for index, column in enumerate(cal_header):
        print(f"  [{index:02d}] {column}")

    forbidden_raw = [column for column in raw_header if any(token in column.casefold() for token in FORBIDDEN_LABEL_TOKENS)]
    forbidden_cal = [column for column in cal_header if any(token in column.casefold() for token in FORBIDDEN_LABEL_TOKENS)]
    common_columns = [column for column in raw_header if column in set(cal_header)]

    print("\n[3] AUTOMATIC COLUMN GROUPING")
    print("-" * 80)
    print("Raw grouping:")
    print(json.dumps(raw_groups, indent=2))
    print("Calibrated grouping:")
    print(json.dumps(cal_groups, indent=2))
    print("Forbidden raw columns       :", forbidden_raw)
    print("Forbidden calibrated columns:", forbidden_cal)
    print("Common column count         :", len(common_columns))

    print("\n[4] REGISTERED EVALUATION SETTINGS")
    print("-" * 80)
    with EVALUATION_CONTRACT.open("r", encoding="utf-8") as handle:
        evaluation_contract = json.load(handle)
    settings = contract_subset(evaluation_contract)
    print(json.dumps(settings, indent=2, ensure_ascii=False))

    source_action, source_sha = freeze_source()
    print("\n[5] SOURCE AND SCHEMA FREEZE")
    print("-" * 80)
    print("Source action :", source_action)
    print("Frozen source :", FROZEN_SOURCE)
    print("Source SHA-256:", source_sha)

    gates = {
        **hash_gates,
        "R3_label_contract_evidence_passed": r3_pass,
        "single_evaluation_outputs_absent": results_absent,
        "raw_predictions_have_5380_rows": raw_rows == 5380,
        "calibrated_predictions_have_5380_rows": cal_rows == 5380,
        "raw_prediction_header_nonempty": len(raw_header) > 0,
        "calibrated_prediction_header_nonempty": len(cal_header) > 0,
        "raw_predictions_label_free": len(forbidden_raw) == 0,
        "calibrated_predictions_label_free": len(forbidden_cal) == 0,
        "raw_identity_columns_detected": len(raw_groups["identity"]) >= 2,
        "calibrated_identity_columns_detected": len(cal_groups["identity"]) >= 2,
        "registered_evaluation_settings_found": len(settings) > 0,
        "schema_preflight_source_frozen": sha256_file(FROZEN_SOURCE) == source_sha,
        "test_label_values_not_loaded": True,
        "scientific_metrics_not_computed": True,
        "single_scientific_evaluation_not_consumed": True,
        "test_images_not_opened": True,
        "model_not_loaded": True,
    }
    if not all(gates.values()):
        for name, passed in gates.items():
            print(f"{name:76s}: {passed}")
        raise RuntimeError("Prediction-schema preflight gate failed.")

    schema = {
        "contract_id": "MMRDR_FOVEA_DR_LOCKED_TEST_PREDICTION_SCHEMA_V1",
        "stage": "0G-4D-5E-3B-0",
        "status": "FROZEN_BEFORE_TEST_LABEL_VALUE_ACCESS",
        "raw_predictions": {"path": str(RAW), "sha256": EXPECTED_SHA[RAW], "rows": raw_rows, "columns": raw_header, "groups": raw_groups},
        "calibrated_predictions": {"path": str(CAL), "sha256": EXPECTED_SHA[CAL], "rows": cal_rows, "columns": cal_header, "groups": cal_groups},
        "common_columns": common_columns,
        "evaluation_settings_snapshot": settings,
        "evaluation_contract": {"path": str(EVALUATION_CONTRACT), "sha256": EXPECTED_SHA[EVALUATION_CONTRACT]},
        "label_source_contract": {"path": str(LABEL_CONTRACT), "sha256": EXPECTED_SHA[LABEL_CONTRACT]},
        "runner_source": {"path": str(FROZEN_SOURCE), "sha256": source_sha},
        "test_label_values_loaded": False,
        "scientific_metrics_computed": False,
        "single_scientific_evaluation_consumed": False,
    }
    schema_action = frozen_json(SCHEMA_CONTRACT, schema)
    schema_sha = sha256_file(SCHEMA_CONTRACT)

    evidence = {
        "stage": "0G-4D-5E-3B-0",
        "status": "PASS",
        "schema_contract": str(SCHEMA_CONTRACT),
        "schema_contract_sha256": schema_sha,
        "runner_source_sha256": source_sha,
        "gates": gates,
        "test_label_values_loaded": False,
        "scientific_metrics_computed": False,
        "single_scientific_evaluation_consumed": False,
        "next_required_stage": "FREEZE_AND_RUN_ONE_TIME_LOCKED_TEST_SCORING_IMPLEMENTATION",
    }
    evidence_action = frozen_json(EVIDENCE, evidence)

    print("Schema action    :", schema_action)
    print("Schema contract  :", SCHEMA_CONTRACT)
    print("Schema SHA-256   :", schema_sha)
    print("Evidence action  :", evidence_action)
    print("Evidence         :", EVIDENCE)
    print("Evidence SHA-256 :", sha256_file(EVIDENCE))

    print("\n" + "=" * 80)
    print("PREDICTION-SCHEMA PREFLIGHT GATES")
    print("=" * 80)
    for name, passed in gates.items():
        print(f"{name:76s}: {passed}")

    print("\nTest label values loaded   : NO")
    print("Scientific metrics computed: NO")
    print("Single evaluation consumed : NO")
    print("\nNEXT REQUIRED STAGE:")
    print("Freeze and run the one-time locked-test scoring implementation.")
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-5E-3B-0 STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
