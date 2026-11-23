#!/usr/bin/env python3
"""
STAGE 0G-4D-5E-3B
One-time locked-test label join and preregistered scientific scoring.

The executed source is frozen before any test-label values are loaded. A
same-source recovery state permits deterministic technical recovery only; it
does not permit changed code, inputs, metric definitions, or a second analysis.
No image or model is opened by this program.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
RUN_ROOT = ROOT / "Runs/FOVEA_DR/locked_test_seed2026_cpu_fp32"

MEMBERSHIP = ROOT / "Splits/mmrdr_locked_test_membership_v1_FROZEN.csv"
SOURCE_INDEX = ROOT / "Audit/mmrdr_integrity_protocol_index_v3.csv"
RAW_PREDICTIONS = RUN_ROOT / "locked_test_raw_predictions_v1.csv"
CAL_PREDICTIONS = RUN_ROOT / "locked_test_calibrated_predictions_v1.csv"
LABEL_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_locked_test_label_source_contract_v1.json"
EVALUATION_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_locked_test_single_evaluation_contract_v1_1.json"
SCHEMA_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_locked_test_prediction_schema_v1.json"
R3_EVIDENCE = ROOT / "Evidence/stage0g4d5e3a_r3_source_index_bridged_label_contract.json"
SCHEMA_EVIDENCE = ROOT / "Evidence/stage0g4d5e3b0_prediction_schema_preflight.json"
DEPLOYMENT_PACKAGE = ROOT / "Metadata/mmrdr_fovea_dr_calibrated_deployment_package_v1.json"
TEMPERATURES = ROOT / "Metadata/mmrdr_fovea_dr_temperature_parameters_v1.json"
CHECKPOINT = ROOT / "Checkpoints/FOVEA_DR/baseline_seed2026/selected_epoch10_frozen.pt"

QC_PATHS = {
    "CFP": ROOT / "Metadata/MMRDR_QC_v1/CFP_standardized_qc.csv",
    "OCT": ROOT / "Metadata/MMRDR_QC_v1/OCT_standardized_qc.csv",
    "UWF": ROOT / "Metadata/MMRDR_QC_v1/UWF_standardized_qc.csv",
}

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5e3b_one_time_locked_test_scoring.py"
SCORING_TABLE = RUN_ROOT / "locked_test_scoring_table_v1.csv"
METRICS = ROOT / "Audit/mmrdr_fovea_dr_LOCKED_TEST_final_metrics_v1.csv"
BOOTSTRAP = ROOT / "Audit/mmrdr_fovea_dr_LOCKED_TEST_bootstrap_CI_v1.csv"
SUMMARY = RUN_ROOT / "locked_test_final_summary_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5e3b_locked_test_single_evaluation_results.json"
STATE = RUN_ROOT / "single_evaluation_state_v1.json"

EXPECTED_SHA = {
    MEMBERSHIP: "56913e5067d371f8631ef8deb56db585bb21f8591f97a88bdbedc07b29e4f45b",
    SOURCE_INDEX: "c1609072eb96f893898da62d90828f4b9683249ed439f34526b856532099be12",
    RAW_PREDICTIONS: "67b612d48fea47256ad75c7482572b4037807b6f2c3f087a5f11fbc0f9d75e23",
    CAL_PREDICTIONS: "65ecfa3d102c3d2c08551c3799084b02d28e4a44cbf475c6d7757daa772e1f78",
    LABEL_CONTRACT: "fdd8151603c6afc482f26892d47b791809513880708cefff18f767fa4039958b",
    EVALUATION_CONTRACT: "984d0e13fbdd178bbeff42ff97e69465cd760b9434e86cad414ca93e5c0d7f10",
    SCHEMA_CONTRACT: "1389035976a076bd33279fff5e06ea3d9e24c15fdef9c66022f925182d669c9e",
    R3_EVIDENCE: "4ab72b63e9a10f5f8fa71199da62892bc9fd0949c9fbf30443bd13115a335595",
    SCHEMA_EVIDENCE: "12c75b94c808b36da04b90c65e2bff0c00ce4ad5dc5cc33e647f0ccb79559458",
    DEPLOYMENT_PACKAGE: "bf79997cd9afe922e957b904ee074b7fe5bf9308fd081562ee424e9e1f174867",
    TEMPERATURES: "b694a32f84126516b0c6dc1ef0e32e130ffe6e6919433bf751399e21d10894c8",
    CHECKPOINT: "89f5e87b647680a9094c55bf647dc1dd2abb624681e733ffb453e30e4c11ff1f",
    QC_PATHS["CFP"]: "d91bd3fa6a01c271e2a44bbb1203bfdcae6eb47d252afa488c46557bf8728bba",
    QC_PATHS["OCT"]: "628329c09b15d60a395d8154f8db199976e5ce645662a47083c0cc86ec1a1a93",
    QC_PATHS["UWF"]: "5c6ede59edb78d28030e1931266a3773204d265ea6a958c069e71056ade9c033",
}

EXPECTED_COUNTS = {"CFP": 2221, "OCT": 562, "UWF": 2597}
EXPECTED_QC_COUNTS = {"CFP": 2225, "OCT": 562, "UWF": 2597}
CLASS_COUNTS = {"CFP": 5, "UWF": 5, "OCT": 3}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260906
CONFIDENCE_LEVEL = 0.95
ECE_BINS = 15
LESION_THRESHOLD = 0.5
EPSILON = 1e-12

LESIONS = [
    "lesion_microaneurysm",
    "lesion_hard_exudate",
    "lesion_intraretinal_hemorrhage",
    "lesion_VB_IRMA",
    "lesion_neovascularization",
    "lesion_vitreous_hemorrhage",
    "lesion_retinal_detachment",
]

LABEL_COLUMNS = [
    "image",
    "image_id",
    "source_split",
    "protocol_role",
    "grade",
    "grade_loss_mask",
    "lesion_applicable",
    "lesion_vector_valid",
    "lesion_loss_mask",
    "lesion_sensitivity_mask",
    *LESIONS,
]

RAW_GRADE_COLUMNS = [f"raw_probability_{index}" for index in range(5)]
CAL_GRADE_COLUMNS = [f"calibrated_probability_{index}" for index in range(5)]
RAW_LESION_COLUMNS = [f"raw_{lesion}_probability" for lesion in LESIONS]
CAL_LESION_COLUMNS = [f"calibrated_{lesion}_probability" for lesion in LESIONS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def json_bytes(payload: dict) -> bytes:
    return (
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
            default=json_scalar,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_bytes(path, json_bytes(payload))


def frozen_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing frozen output differs: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    atomic_bytes(path, data)
    return "CREATED_AND_FROZEN"


def dataframe_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different scoring implementation is already frozen.")
        action = "EXISTING_IDENTICAL_SOURCE_VERIFIED"
    else:
        temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
        shutil.copyfile(current, temporary)
        os.replace(temporary, FROZEN_SOURCE)
        action = "COPIED_AND_FROZEN_BEFORE_LABEL_ACCESS"
    return action, source_sha


def normalize_modality(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def normalize_path(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace("\\\\", "/", regex=False)
        .str.replace(r"^\./", "", regex=True)
    )


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def numeric(series: pd.Series, name: str) -> np.ndarray:
    converted = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(converted).all():
        raise RuntimeError(f"Non-finite or non-numeric values in required column: {name}")
    return converted


def binary_mask(series: pd.Series, name: str) -> np.ndarray:
    text = series.astype("string").str.strip().str.casefold()
    mapped = text.map({"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0})
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    values = values.where(values.notna(), mapped)
    array = values.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all() or not np.isin(array, [0.0, 1.0]).all():
        raise RuntimeError(f"Mask {name} contains values outside {{0,1}}.")
    return array.astype(bool)


def grade_probability_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> np.ndarray:
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    modality_values = normalize_modality(frame["modality"]).to_numpy()
    for modality, classes in CLASS_COUNTS.items():
        rows = modality_values == modality
        relevant = matrix[rows, :classes]
        if not np.isfinite(relevant).all():
            raise RuntimeError(f"Non-finite relevant values in {name} for {modality}.")
        if np.any(relevant < 0.0) or np.any(relevant > 1.0):
            raise RuntimeError(f"Probabilities outside [0,1] in {name} for {modality}.")
    # Irrelevant fundus-only class slots for OCT may be serialized as NaN.
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix


def lesion_probability_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> np.ndarray:
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    fundus_rows = normalize_modality(frame["modality"]).isin(["CFP", "UWF"]).to_numpy()
    relevant = matrix[fundus_rows]
    if not np.isfinite(relevant).all():
        raise RuntimeError(f"Non-finite fundus values in {name}.")
    if np.any(relevant < 0.0) or np.any(relevant > 1.0):
        raise RuntimeError(f"Probabilities outside [0,1] in {name}.")
    # OCT has no lesion endpoint; any absent OCT entries are never scored.
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix


def multiclass_nll(y: np.ndarray, probabilities: np.ndarray) -> float:
    if len(y) == 0:
        return math.nan
    chosen = probabilities[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(chosen, EPSILON, 1.0))))


def multiclass_brier(y: np.ndarray, probabilities: np.ndarray, classes: int) -> float:
    if len(y) == 0:
        return math.nan
    one_hot = np.eye(classes, dtype=np.float64)[y]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def multiclass_ece(y: np.ndarray, probabilities: np.ndarray) -> float:
    if len(y) == 0:
        return math.nan
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correctness = (predictions == y).astype(np.float64)
    bins = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    total = float(len(y))
    result = 0.0
    for bin_index in range(ECE_BINS):
        selected = bins == bin_index
        if np.any(selected):
            result += float(np.sum(selected) / total) * abs(
                float(np.mean(correctness[selected])) - float(np.mean(confidence[selected]))
            )
    return result


def accuracy(y: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(y == predictions)) if len(y) else math.nan


def macro_f1(y: np.ndarray, predictions: np.ndarray, classes: int) -> float:
    if len(y) == 0:
        return math.nan
    values = []
    for label in range(classes):
        true_positive = np.sum((y == label) & (predictions == label))
        false_positive = np.sum((y != label) & (predictions == label))
        false_negative = np.sum((y == label) & (predictions != label))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(float(2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(values))


def quadratic_weighted_kappa(y: np.ndarray, predictions: np.ndarray, classes: int) -> float:
    if len(y) == 0:
        return math.nan
    confusion = np.zeros((classes, classes), dtype=np.float64)
    np.add.at(confusion, (y, predictions), 1.0)
    actual = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    expected = np.outer(actual, predicted) / float(len(y))
    indices = np.arange(classes, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) ** 2) / float((classes - 1) ** 2)
    denominator = float(np.sum(weights * expected))
    if denominator <= 0.0:
        return math.nan
    return float(1.0 - np.sum(weights * confusion) / denominator)


def grade_metrics(y: np.ndarray, probabilities: np.ndarray, classes: int) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    return {
        "NLL": multiclass_nll(y, probabilities),
        "Brier_score": multiclass_brier(y, probabilities, classes),
        "ECE_15_EQUAL_WIDTH": multiclass_ece(y, probabilities),
        "accuracy": accuracy(y, predictions),
        "macro_F1": macro_f1(y, predictions, classes),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(y, predictions, classes),
    }


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {name: math.nan for name in ["binary_NLL", "Brier_score", "ECE_15_EQUAL_WIDTH", "accuracy_at_0_5", "F1_at_0_5"]}
    probability = np.clip(probability, EPSILON, 1.0 - EPSILON)
    prediction = (probability >= LESION_THRESHOLD).astype(np.int64)
    nll = -np.mean(y * np.log(probability) + (1 - y) * np.log(1.0 - probability))
    brier = np.mean((probability - y) ** 2)
    bins = np.minimum((probability * ECE_BINS).astype(int), ECE_BINS - 1)
    ece = 0.0
    for bin_index in range(ECE_BINS):
        selected = bins == bin_index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(y[selected])) - float(np.mean(probability[selected]))
            )
    true_positive = np.sum((y == 1) & (prediction == 1))
    false_positive = np.sum((y == 0) & (prediction == 1))
    false_negative = np.sum((y == 1) & (prediction == 0))
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = float(2 * true_positive / denominator) if denominator else 0.0
    return {
        "binary_NLL": float(nll),
        "Brier_score": float(brier),
        "ECE_15_EQUAL_WIDTH": float(ece),
        "accuracy_at_0_5": accuracy(y, prediction),
        "F1_at_0_5": f1,
    }


def metric_id(record: dict) -> str:
    return "|".join(
        str(record[key])
        for key in ["task_family", "unit", "modality_scope", "calibration_state", "metric"]
    )


def endpoint_flags(task_family: str, unit: str, state: str, metric: str) -> tuple[bool, bool]:
    primary = task_family == "aggregate" and unit == "FUNDUS" and metric == "mean_CFP_UWF_quadratic_weighted_kappa"
    secondary = False
    if task_family == "grade" and metric == "quadratic_weighted_kappa" and unit in {"CFP", "UWF", "OCT"}:
        secondary = True
    if task_family == "grade" and unit == "OCT" and metric in {"macro_F1", "NLL"}:
        secondary = True
    if task_family == "aggregate" and unit == "FUNDUS" and metric == "mean_CFP_UWF_NLL":
        secondary = True
    return primary, secondary


def build_record(task_family: str, unit: str, scope: str, state: str, metric: str, value: float, n: int) -> dict:
    primary, secondary = endpoint_flags(task_family, unit, state, metric)
    record = {
        "task_family": task_family,
        "unit": unit,
        "modality_scope": scope,
        "calibration_state": state,
        "metric": metric,
        "value": float(value),
        "eligible_rows": int(n),
        "decision_threshold": LESION_THRESHOLD if task_family == "lesion" else math.nan,
        "is_primary_endpoint": bool(primary),
        "is_secondary_endpoint": bool(secondary),
    }
    record["metric_id"] = metric_id(record)
    return record


def evaluate_records(arrays: dict, sampled: dict[str, np.ndarray]) -> list[dict]:
    records = []
    grade_lookup = {}
    for modality in ["CFP", "UWF", "OCT"]:
        indices = sampled[modality]
        eligible = arrays["grade_mask"][indices]
        selected = indices[eligible]
        y = arrays["grade"][selected]
        classes = CLASS_COUNTS[modality]
        for state, probability_key in [("RAW", "raw_grade"), ("CALIBRATED", "cal_grade")]:
            probabilities = arrays[probability_key][selected, :classes]
            values = grade_metrics(y, probabilities, classes)
            for metric, value in values.items():
                record = build_record("grade", modality, modality, state, metric, value, len(y))
                records.append(record)
                grade_lookup[(modality, state, metric)] = value

    fundus_indices = np.concatenate([sampled["CFP"], sampled["UWF"]])
    for lesion_index, lesion in enumerate(LESIONS):
        eligible = arrays["lesion_mask"][fundus_indices]
        selected = fundus_indices[eligible]
        y = arrays["lesion_labels"][selected, lesion_index]
        for state, probability_key in [("RAW", "raw_lesion"), ("CALIBRATED", "cal_lesion")]:
            values = binary_metrics(y, arrays[probability_key][selected, lesion_index])
            for metric, value in values.items():
                records.append(build_record("lesion", lesion, "CFP+UWF", state, metric, value, len(y)))

    for state in ["RAW", "CALIBRATED"]:
        mean_qwk = np.mean([
            grade_lookup[("CFP", state, "quadratic_weighted_kappa")],
            grade_lookup[("UWF", state, "quadratic_weighted_kappa")],
        ])
        mean_nll = np.mean([
            grade_lookup[("CFP", state, "NLL")],
            grade_lookup[("UWF", state, "NLL")],
        ])
        fundus_n = int(np.sum(arrays["grade_mask"][sampled["CFP"]]) + np.sum(arrays["grade_mask"][sampled["UWF"]]))
        records.append(build_record("aggregate", "FUNDUS", "CFP+UWF_UNWEIGHTED", state, "mean_CFP_UWF_quadratic_weighted_kappa", mean_qwk, fundus_n))
        records.append(build_record("aggregate", "FUNDUS", "CFP+UWF_UNWEIGHTED", state, "mean_CFP_UWF_NLL", mean_nll, fundus_n))
    return records


def state_payload(status: str, source_sha: str, output_hashes: dict | None = None) -> dict:
    return {
        "run": "FOVEA_DR_LOCKED_TEST_SINGLE_EVALUATION_SEED2026",
        "status": status,
        "runner_source_sha256": source_sha,
        "input_hashes": {str(path): digest for path, digest in EXPECTED_SHA.items()},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "output_hashes": output_hashes or {},
        "test_label_values_access_authorized": True,
        "metric_or_threshold_changes_permitted": False,
        "technical_recovery_permitted_only_with_identical_source_and_inputs": True,
        "single_evaluation_authorization_consumed": True,
        "single_scientific_evaluation_completed": status == "COMPLETE",
        "single_scientific_evaluation_consumed": status == "COMPLETE",
    }


def main() -> None:
    started = time.perf_counter()
    print("=" * 80)
    print("STAGE 0G-4D-5E-3B — ONE-TIME LOCKED-TEST SCIENTIFIC SCORING")
    print("=" * 80)

    print("\n[1] FROZEN PREREQUISITE VERIFICATION")
    print("-" * 80)
    gates = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(verified)
        print(f"{path.name:76s}: exists={exists}, hash_verified={verified}")
    if not all(gates.values()):
        raise RuntimeError("A frozen scoring prerequisite is missing or changed.")

    with R3_EVIDENCE.open("r", encoding="utf-8") as handle:
        r3_evidence = json.load(handle)
    with SCHEMA_EVIDENCE.open("r", encoding="utf-8") as handle:
        schema_evidence = json.load(handle)
    gates["R3_label_contract_evidence_passed"] = r3_evidence.get("status") == "PASS"
    gates["prediction_schema_evidence_passed"] = schema_evidence.get("status") == "PASS"
    gates["labels_reported_unused_before_scoring"] = not r3_evidence.get("test_label_values_loaded", True)
    gates["metrics_reported_uncomputed_before_scoring"] = not schema_evidence.get("scientific_metrics_computed", True)
    if not all(gates.values()):
        raise RuntimeError("Pre-scoring evidence gate failed.")

    print("\n[2] EXECUTED SOURCE FREEZE — BEFORE LABEL ACCESS")
    print("-" * 80)
    source_action, source_sha = freeze_source()
    print("Source action :", source_action)
    print("Frozen source :", FROZEN_SOURCE)
    print("Source SHA-256:", source_sha)
    gates["scoring_source_frozen_before_label_access"] = sha256_file(FROZEN_SOURCE) == source_sha

    final_outputs = [SCORING_TABLE, METRICS, BOOTSTRAP, SUMMARY, EVIDENCE]
    if STATE.exists():
        with STATE.open("r", encoding="utf-8") as handle:
            existing_state = json.load(handle)
        same_source = existing_state.get("runner_source_sha256") == source_sha
        same_inputs = existing_state.get("input_hashes") == {str(path): digest for path, digest in EXPECTED_SHA.items()}
        if not (same_source and same_inputs):
            raise RuntimeError("A scoring state exists for different code or inputs.")
        if existing_state.get("status") == "COMPLETE":
            output_hashes = existing_state.get("output_hashes", {})
            complete_verified = all(
                Path(path).is_file() and sha256_file(Path(path)) == digest
                for path, digest in output_hashes.items()
            )
            if not complete_verified:
                raise RuntimeError("Completed scoring state has missing or changed outputs.")
            print("\nThe single evaluation is already COMPLETE and verified.")
            print("No label values were reloaded and no metrics were recomputed.")
            print("Summary:", SUMMARY)
            print("Evidence:", EVIDENCE)
            return
        if existing_state.get("status") != "AUTHORIZED_SCORING_STARTED":
            raise RuntimeError("Unknown existing scoring-state status.")
        print("Identical-source technical recovery: AUTHORIZED")
    else:
        if any(path.exists() for path in final_outputs):
            raise RuntimeError("Final outputs exist without a matching scoring state.")
        atomic_json(STATE, state_payload("AUTHORIZED_SCORING_STARTED", source_sha))
        print("Single-evaluation state: AUTHORIZED_SCORING_STARTED")

    print("\n[3] ONE-TIME TEST LABEL LOAD AND AUTHORITATIVE JOIN")
    print("-" * 80)
    # No standardized-QC label value is read before this point.
    membership_header = pd.read_csv(MEMBERSHIP, nrows=0).columns.tolist()
    index_candidates = [name for name in ["locked_test_index", "test_index"] if name in membership_header]
    if len(index_candidates) != 1:
        raise RuntimeError(f"Could not identify one membership index column: {index_candidates}")
    membership_index = index_candidates[0]
    membership = pd.read_csv(
        MEMBERSHIP,
        usecols=[membership_index, "relative_path", "modality"],
        dtype={"relative_path": "string", "modality": "string"},
    ).rename(columns={membership_index: "locked_test_index"})
    membership["modality"] = normalize_modality(membership["modality"])
    membership["relative_path_key"] = normalize_path(membership["relative_path"])
    membership["locked_test_index"] = pd.to_numeric(membership["locked_test_index"], errors="raise").astype(int)

    source = pd.read_csv(
        SOURCE_INDEX,
        usecols=["modality", "relative_path", "image_id", "protocol_role"],
        dtype="string",
    )
    source["modality"] = normalize_modality(source["modality"])
    source["relative_path_key"] = normalize_path(source["relative_path"])
    source["image_id_key"] = normalize_id(source["image_id"])
    source = source[["modality", "relative_path_key", "image_id", "image_id_key", "protocol_role"]]

    joined = membership.merge(
        source,
        on=["modality", "relative_path_key"],
        how="left",
        validate="one_to_one",
        indicator="source_join",
    )
    if len(joined) != 5380 or not (joined["source_join"] == "both").all():
        raise RuntimeError("Membership-to-source-index label bridge is incomplete.")

    qc_frames = []
    for modality, path in QC_PATHS.items():
        frame = pd.read_csv(
            path,
            usecols=LABEL_COLUMNS,
            dtype={
                "image": "string",
                "image_id": "string",
                "source_split": "string",
                "protocol_role": "string",
            },
        )
        frame["modality"] = modality
        role = frame["protocol_role"].astype("string").str.strip().str.upper()
        frame = frame.loc[role == "LOCKED_INTERNAL_TEST"].copy()
        frame["image_id_key"] = normalize_id(frame["image_id"])
        qc_frames.append(frame)
        print(f"{modality}: loaded authoritative locked-test label rows={len(frame)}")
    qc = pd.concat(qc_frames, ignore_index=True)
    qc["modality"] = normalize_modality(qc["modality"])

    qc_key_unique = not qc.duplicated(["modality", "image_id_key"]).any()
    if not qc_key_unique:
        raise RuntimeError("Standardized-QC test label keys are not unique.")
    label_table = joined.merge(
        qc,
        on=["modality", "image_id_key"],
        how="left",
        validate="one_to_one",
        indicator="label_join",
        suffixes=("_source", "_qc"),
    )
    label_complete = len(label_table) == 5380 and (label_table["label_join"] == "both").all()
    label_counts = label_table["modality"].value_counts().sort_index().to_dict()
    qc_only = qc.merge(
        label_table[["modality", "image_id_key"]],
        on=["modality", "image_id_key"],
        how="left",
        indicator=True,
    )
    qc_only = qc_only.loc[qc_only["_merge"] == "left_only"]
    qc_only_counts = qc_only["modality"].value_counts().sort_index().to_dict()
    print("Joined label rows      :", len(label_table))
    print("Joined modality counts :", label_counts)
    print("QC-only excluded rows  :", len(qc_only), qc_only_counts)
    if not (label_complete and label_counts == EXPECTED_COUNTS and len(qc_only) == 4 and qc_only_counts == {"CFP": 4}):
        raise RuntimeError("Final label-table structural gate failed.")

    print("\n[4] FROZEN PREDICTION JOIN AND NUMERIC VALIDATION")
    print("-" * 80)
    raw = pd.read_csv(RAW_PREDICTIONS)
    calibrated = pd.read_csv(CAL_PREDICTIONS)
    identity_columns = ["locked_test_index", "relative_path", "image_sha256", "modality"]
    raw_columns = identity_columns + ["run_signature", "raw_prediction", "raw_expected_grade", *RAW_GRADE_COLUMNS, *RAW_LESION_COLUMNS]
    cal_columns = identity_columns + ["run_signature", "calibrated_prediction", "calibrated_expected_grade", *CAL_GRADE_COLUMNS, *CAL_LESION_COLUMNS]
    missing_raw = sorted(set(raw_columns) - set(raw.columns))
    missing_cal = sorted(set(cal_columns) - set(calibrated.columns))
    if missing_raw or missing_cal:
        raise RuntimeError(f"Prediction schema mismatch: raw={missing_raw}, calibrated={missing_cal}")

    for frame in [raw, calibrated]:
        frame["modality"] = normalize_modality(frame["modality"])
        frame["relative_path_key"] = normalize_path(frame["relative_path"])
        frame["locked_test_index"] = pd.to_numeric(frame["locked_test_index"], errors="raise").astype(int)
        frame.sort_values("locked_test_index", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    raw_identity = raw[["locked_test_index", "relative_path_key", "image_sha256", "modality"]].astype("string")
    cal_identity = calibrated[["locked_test_index", "relative_path_key", "image_sha256", "modality"]].astype("string")
    prediction_identity_match = raw_identity.equals(cal_identity)
    indices_contiguous = raw["locked_test_index"].tolist() == list(range(5380))
    if not (prediction_identity_match and indices_contiguous and len(raw) == len(calibrated) == 5380):
        raise RuntimeError("Raw/calibrated prediction identity gate failed.")

    prediction_table = raw[raw_columns + ["relative_path_key"]].merge(
        calibrated[["locked_test_index", *[column for column in cal_columns if column not in identity_columns and column != "run_signature"]]],
        on="locked_test_index",
        how="left",
        validate="one_to_one",
    )
    scoring = label_table.merge(
        prediction_table,
        on=["locked_test_index", "modality", "relative_path_key"],
        how="left",
        validate="one_to_one",
        indicator="prediction_join",
        suffixes=("", "_prediction"),
    ).sort_values("locked_test_index").reset_index(drop=True)
    prediction_join_complete = len(scoring) == 5380 and (scoring["prediction_join"] == "both").all()
    if not prediction_join_complete:
        raise RuntimeError("Label-to-prediction join is incomplete.")

    grade = numeric(scoring["grade"], "grade").astype(np.int64)
    grade_mask = binary_mask(scoring["grade_loss_mask"], "grade_loss_mask")
    lesion_applicable = binary_mask(scoring["lesion_applicable"], "lesion_applicable")
    lesion_vector_valid = binary_mask(scoring["lesion_vector_valid"], "lesion_vector_valid")
    lesion_loss_mask = binary_mask(scoring["lesion_loss_mask"], "lesion_loss_mask")
    fundus = scoring["modality"].isin(["CFP", "UWF"]).to_numpy()
    lesion_mask = fundus & lesion_applicable & lesion_vector_valid & lesion_loss_mask

    raw_grade = grade_probability_matrix(scoring, RAW_GRADE_COLUMNS, "raw grade probabilities")
    cal_grade = grade_probability_matrix(scoring, CAL_GRADE_COLUMNS, "calibrated grade probabilities")
    raw_lesion = lesion_probability_matrix(scoring, RAW_LESION_COLUMNS, "raw lesion probabilities")
    cal_lesion = lesion_probability_matrix(scoring, CAL_LESION_COLUMNS, "calibrated lesion probabilities")
    lesion_labels_all = scoring[LESIONS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(lesion_labels_all[lesion_mask]).all() or not np.isin(lesion_labels_all[lesion_mask], [0.0, 1.0]).all():
        raise RuntimeError("Eligible lesion labels are missing or non-binary.")
    lesion_labels = np.nan_to_num(lesion_labels_all, nan=0.0).astype(np.int64)

    probability_sums_valid = True
    grade_ranges_valid = True
    for modality, classes in CLASS_COUNTS.items():
        selected = scoring["modality"].eq(modality).to_numpy()
        for matrix in [raw_grade, cal_grade]:
            relevant = matrix[selected, :classes]
            probability_sums_valid &= bool(np.allclose(relevant.sum(axis=1), 1.0, atol=2e-5, rtol=0.0))
        labels = grade[selected & grade_mask]
        grade_ranges_valid &= bool(np.isin(labels, np.arange(classes)).all())
    raw_decisions = np.zeros(5380, dtype=np.int64)
    cal_decisions = np.zeros(5380, dtype=np.int64)
    for modality, classes in CLASS_COUNTS.items():
        selected = scoring["modality"].eq(modality).to_numpy()
        raw_decisions[selected] = np.argmax(raw_grade[selected, :classes], axis=1)
        cal_decisions[selected] = np.argmax(cal_grade[selected, :classes], axis=1)
    grade_decisions_invariant = bool(np.array_equal(raw_decisions, cal_decisions))
    lesion_decisions_invariant = bool(np.array_equal(raw_lesion[lesion_mask] >= 0.5, cal_lesion[lesion_mask] >= 0.5))
    print("Prediction identity match :", prediction_identity_match)
    print("Indices contiguous        :", indices_contiguous)
    print("Probability sums valid    :", probability_sums_valid)
    print("Grade ranges valid        :", grade_ranges_valid)
    print("Grade decisions invariant :", grade_decisions_invariant)
    print("Lesion decisions invariant:", lesion_decisions_invariant)
    print("Grade eligible rows       :", int(grade_mask.sum()))
    print("Lesion eligible rows      :", int(lesion_mask.sum()))

    arrays = {
        "grade": grade,
        "grade_mask": grade_mask,
        "lesion_mask": lesion_mask,
        "lesion_labels": lesion_labels,
        "raw_grade": raw_grade,
        "cal_grade": cal_grade,
        "raw_lesion": raw_lesion,
        "cal_lesion": cal_lesion,
    }
    observed_indices = {
        modality: np.flatnonzero(scoring["modality"].eq(modality).to_numpy())
        for modality in ["CFP", "UWF", "OCT"]
    }

    print("\n[5] PREREGISTERED OBSERVED METRICS")
    print("-" * 80)
    observed_records = evaluate_records(arrays, observed_indices)
    metrics_frame = pd.DataFrame(observed_records)
    for _, row in metrics_frame.loc[
        metrics_frame["is_primary_endpoint"] | metrics_frame["is_secondary_endpoint"]
    ].iterrows():
        flag = "PRIMARY" if row["is_primary_endpoint"] else "SECONDARY"
        print(f"{flag:9s} | {row['calibration_state']:10s} | {row['unit']:7s} | {row['metric']:48s} = {row['value']:.6f}")

    print("\n[6] STRATIFIED NONPARAMETRIC BOOTSTRAP")
    print("-" * 80)
    print("Replicates       :", BOOTSTRAP_REPLICATES)
    print("Seed             :", BOOTSTRAP_SEED)
    print("Confidence level :", CONFIDENCE_LEVEL)
    print("Strata           : CFP, UWF, OCT")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_keys = metrics_frame["metric_id"].tolist()
    bootstrap_values = {key: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for key in metric_keys}
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = {
            modality: rng.choice(indices, size=len(indices), replace=True)
            for modality, indices in observed_indices.items()
        }
        replicate_records = evaluate_records(arrays, sampled)
        replicate_map = {record["metric_id"]: record["value"] for record in replicate_records}
        if set(replicate_map) != set(metric_keys):
            raise RuntimeError("Bootstrap metric-key drift detected.")
        for key in metric_keys:
            bootstrap_values[key][replicate] = replicate_map[key]
        if (replicate + 1) % 100 == 0 or replicate == 0:
            print(f"Bootstrap {replicate + 1:4d}/{BOOTSTRAP_REPLICATES}")

    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    ci_records = []
    for record in observed_records:
        values = bootstrap_values[record["metric_id"]]
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            lower = upper = math.nan
        else:
            lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
        ci_records.append({
            **record,
            "bootstrap_method": "STRATIFIED_NONPARAMETRIC_PERCENTILE",
            "bootstrap_strata": "MODALITY",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
            "bootstrap_replicates_finite": int(len(finite)),
            "confidence_level": CONFIDENCE_LEVEL,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        })
    bootstrap_frame = pd.DataFrame(ci_records)

    gates.update({
        "exactly_5380_label_rows_joined": label_complete,
        "locked_test_modality_counts_exact": label_counts == EXPECTED_COUNTS,
        "exactly_four_excluded_QC_rows": len(qc_only) == 4,
        "all_excluded_QC_rows_are_CFP": qc_only_counts == {"CFP": 4},
        "prediction_identity_and_order_match": prediction_identity_match and indices_contiguous,
        "prediction_join_complete": prediction_join_complete,
        "all_relevant_probability_distributions_valid": probability_sums_valid,
        "all_grade_labels_in_declared_ranges": grade_ranges_valid,
        "all_eligible_lesion_labels_binary": True,
        "grade_decisions_invariant_after_calibration": grade_decisions_invariant,
        "lesion_threshold_decisions_invariant_after_calibration": lesion_decisions_invariant,
        "all_observed_metrics_finite": bool(np.isfinite(metrics_frame["value"].to_numpy(dtype=float)).all()),
        "exactly_2000_bootstrap_replicates_completed": all(len(values) == BOOTSTRAP_REPLICATES for values in bootstrap_values.values()),
        "all_bootstrap_intervals_finite": bool(np.isfinite(bootstrap_frame[["ci_lower", "ci_upper"]].to_numpy(dtype=float)).all()),
        "bootstrap_seed_preserved": BOOTSTRAP_SEED == 20260906,
        "lesion_threshold_preserved": LESION_THRESHOLD == 0.5,
        "test_images_not_opened": True,
        "model_not_loaded": True,
        "model_parameters_not_updated": True,
        "temperatures_not_refitted": True,
    })
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Final scientific scoring gates failed: {failed}")

    print("\n[7] IMMUTABLE RESULT FREEZE")
    print("-" * 80)
    scoring_columns = [
        "locked_test_index", "relative_path", "image_sha256", "modality", "image_id_qc", "grade",
        "grade_loss_mask", "lesion_applicable", "lesion_vector_valid", "lesion_loss_mask",
        "lesion_sensitivity_mask", *LESIONS,
        "raw_prediction", "raw_expected_grade", *RAW_GRADE_COLUMNS, *RAW_LESION_COLUMNS,
        "calibrated_prediction", "calibrated_expected_grade", *CAL_GRADE_COLUMNS, *CAL_LESION_COLUMNS,
    ]
    missing_scoring_columns = [column for column in scoring_columns if column not in scoring.columns]
    if missing_scoring_columns:
        raise RuntimeError(f"Scoring-table output columns missing: {missing_scoring_columns}")
    scoring_output = scoring[scoring_columns].copy()
    scoring_action = frozen_bytes(SCORING_TABLE, dataframe_bytes(scoring_output))
    metrics_action = frozen_bytes(METRICS, dataframe_bytes(metrics_frame))
    bootstrap_action = frozen_bytes(BOOTSTRAP, dataframe_bytes(bootstrap_frame))

    endpoint_rows = bootstrap_frame.loc[
        bootstrap_frame["is_primary_endpoint"] | bootstrap_frame["is_secondary_endpoint"]
    ]
    primary_calibrated = endpoint_rows.loc[
        endpoint_rows["is_primary_endpoint"] & endpoint_rows["calibration_state"].eq("CALIBRATED")
    ].iloc[0]
    summary = {
        "run": "FOVEA_DR_LOCKED_TEST_SINGLE_EVALUATION_SEED2026",
        "status": "COMPLETE",
        "primary_endpoint": {
            "name": primary_calibrated["metric"],
            "calibration_state": "CALIBRATED",
            "value": float(primary_calibrated["value"]),
            "confidence_level": CONFIDENCE_LEVEL,
            "ci_lower": float(primary_calibrated["ci_lower"]),
            "ci_upper": float(primary_calibrated["ci_upper"]),
        },
        "all_primary_and_secondary_endpoints": endpoint_rows.to_dict(orient="records"),
        "test_rows": 5380,
        "modality_counts": EXPECTED_COUNTS,
        "grade_eligible_rows": int(grade_mask.sum()),
        "lesion_eligible_rows": int(lesion_mask.sum()),
        "selected_epoch": 10,
        "bootstrap": {
            "method": "STRATIFIED_NONPARAMETRIC_PERCENTILE",
            "strata": ["CFP", "UWF", "OCT"],
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "metric_definitions": {
            "multiclass_Brier_score": "MEAN_OVER_ROWS_OF_SUM_OVER_FIXED_CLASSES_OF_SQUARED_ERROR",
            "binary_Brier_score": "MEAN_SQUARED_ERROR_OF_POSITIVE_CLASS_PROBABILITY",
            "multiclass_ECE": "15_EQUAL_WIDTH_BINS_OF_MAXIMUM_CLASS_CONFIDENCE_VERSUS_ACCURACY",
            "binary_ECE": "15_EQUAL_WIDTH_BINS_OF_POSITIVE_CLASS_PROBABILITY_VERSUS_PREVALENCE",
            "macro_F1": "UNWEIGHTED_MEAN_OVER_ALL_FIXED_TASK_CLASSES_WITH_ZERO_FOR_UNDEFINED_CLASS_F1",
            "QWK": "QUADRATIC_WEIGHTED_COHEN_KAPPA_OVER_ALL_FIXED_TASK_CLASSES",
            "lesion_eligibility": "CFP_OR_UWF_AND_lesion_applicable_AND_lesion_vector_valid_AND_lesion_loss_mask",
        },
        "runner_source": str(FROZEN_SOURCE),
        "runner_source_sha256": source_sha,
        "test_images_opened": False,
        "model_loaded": False,
        "parameters_updated": False,
        "temperatures_refitted": False,
        "single_scientific_evaluation_consumed": True,
    }
    summary_action = frozen_bytes(SUMMARY, json_bytes(summary))

    preliminary_hashes = {
        str(SCORING_TABLE): sha256_file(SCORING_TABLE),
        str(METRICS): sha256_file(METRICS),
        str(BOOTSTRAP): sha256_file(BOOTSTRAP),
        str(SUMMARY): sha256_file(SUMMARY),
    }
    evidence = {
        "stage": "0G-4D-5E-3B",
        "status": "PASS",
        "purpose": "ONE_TIME_PREREGISTERED_LOCKED_TEST_SCIENTIFIC_EVALUATION",
        "runner_source": str(FROZEN_SOURCE),
        "runner_source_sha256": source_sha,
        "input_hashes": {str(path): digest for path, digest in EXPECTED_SHA.items()},
        "output_hashes": preliminary_hashes,
        "gates": gates,
        "primary_endpoint": summary["primary_endpoint"],
        "test_label_values_loaded": True,
        "scientific_metrics_computed": True,
        "single_scientific_evaluation_consumed": True,
        "test_images_opened": False,
        "model_loaded": False,
        "model_parameters_updated": False,
        "temperatures_refitted": False,
        "further_model_selection_permitted": False,
        "further_threshold_tuning_permitted": False,
        "repeat_scientific_test_evaluation_permitted": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    evidence_action = frozen_bytes(EVIDENCE, json_bytes(evidence))
    output_hashes = {**preliminary_hashes, str(EVIDENCE): sha256_file(EVIDENCE)}
    atomic_json(STATE, state_payload("COMPLETE", source_sha, output_hashes))

    for path, action in [
        (SCORING_TABLE, scoring_action),
        (METRICS, metrics_action),
        (BOOTSTRAP, bootstrap_action),
        (SUMMARY, summary_action),
        (EVIDENCE, evidence_action),
    ]:
        print(f"{path.name:62s}: {action}")
        print("  SHA-256:", sha256_file(path))

    print("\n" + "=" * 80)
    print("FINAL LOCKED-TEST SCIENTIFIC EVALUATION GATES")
    print("=" * 80)
    for name, passed in gates.items():
        print(f"{name:76s}: {passed}")

    print("\nPRIMARY ENDPOINT — CALIBRATED DEPLOYMENT")
    print("-" * 80)
    print("Mean CFP/UWF QWK:", f"{primary_calibrated['value']:.6f}")
    print(
        "95% bootstrap CI :",
        f"[{primary_calibrated['ci_lower']:.6f}, {primary_calibrated['ci_upper']:.6f}]",
    )
    print("\nTest labels loaded          : YES — ONE-TIME FINAL SCORING ONLY")
    print("Scientific metrics computed : YES — FROZEN")
    print("Single evaluation consumed  : YES")
    print("Images/model accessed        : NO")
    print("Parameters updated           : NO")
    print("Temperature refitted         : NO")
    print("\nResults:", METRICS)
    print("Bootstrap intervals:", BOOTSTRAP)
    print("Summary:", SUMMARY)
    print("Evidence:", EVIDENCE)
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-5E-3B STATUS: PASS — SINGLE EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
