#!/usr/bin/env python3
"""Generate the frozen, descriptive FOVEA-DR post-hoc comparison outputs.

R1 is schema-aware for the original full-model VALIDATION prediction artifact:
that legacy table contains grade probabilities but not lesion probabilities.
The full-model lesion accuracy/F1 values are therefore read from the verified
selected-checkpoint metadata on CPU; probability-based VALIDATION lesion
metrics are explicitly marked unavailable instead of being reconstructed.

This stage reads frozen JSON/CSV metadata, selected-epoch VALIDATION prediction
tables, already-computed CALIBRATION metric tables, and one hash-verified
selected checkpoint on CPU solely to recover legacy lesion summary metadata.
It never opens an image, instantiates or executes a model, creates an optimizer,
updates parameters, or reads any locked-test artifact.

The comparison protocol explicitly prohibits significance claims from these
overlapping single-seed validation runs.  Consequently, every contrast emitted
here is descriptive only; no p-values, confidence intervals, bootstrap
resampling, or multiplicity claims are generated.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)


WIDTH = 124
ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

POSTHOC_PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
FINAL_TRAINING_REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_final_training_registry_v1.csv"
CALIBRATION_REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_six_variant_calibration_registry_v1.csv"
CALIBRATION_COMPLETION = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_six_variant_calibration_completion_v1.json"
CALIBRATION_COMPLETION_EVIDENCE = ROOT / "Evidence/stage0g4d5f4d_six_variant_calibration_completion_freeze.json"
BASELINE_SELECTION = ROOT / "Metadata/mmrdr_fovea_dr_baseline_seed2026_selection_freeze_v1.json"
BASELINE_SELECTION_EVIDENCE = ROOT / "Evidence/stage0g4d4c_baseline_completion_and_selection_freeze.json"
PRIMARY_CALIBRATION_EVIDENCE = ROOT / "Evidence/stage0g4d5c_cpu_calibration_inference_and_temperature_fit.json"
IMPLEMENTATION_EVIDENCE = ROOT / "Evidence/stage0g4d5f2a_comparator_implementation_audit.json"
DEVELOPMENT_SPLIT = ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f5a_r1_generate_posthoc_comparison_outputs.py"

VALIDATION_SUMMARY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_selected_validation_summary_v1.csv"
VALIDATION_GRADE_LONG = ROOT / "Audit/mmrdr_fovea_dr_posthoc_selected_validation_grade_metrics_v1.csv"
VALIDATION_LESION_LONG = ROOT / "Audit/mmrdr_fovea_dr_posthoc_selected_validation_lesion_metrics_v1.csv"
CALIBRATION_LONG = ROOT / "Audit/mmrdr_fovea_dr_posthoc_calibration_metrics_v1.csv"
CALIBRATION_SUMMARY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_calibration_summary_v1.csv"
CONTRASTS = ROOT / "Audit/mmrdr_fovea_dr_posthoc_descriptive_contrasts_v1.csv"
COMPUTATION = ROOT / "Audit/mmrdr_fovea_dr_posthoc_computational_characteristics_v1.csv"
REPORT = ROOT / "Reports/FOVEA_DR/posthoc_comparison/mmrdr_fovea_dr_posthoc_comparison_results_v1.md"
OUTPUT_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_outputs_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5f5a_posthoc_comparison_outputs.json"

EXPECTED_SHA256 = {
    POSTHOC_PROTOCOL: "f946964923e08722dbb486aa3782b08880446040e07fb6b268ea9c28c08e6ae1",
    FINAL_TRAINING_REGISTRY: "de8b3d0c098128510d185960f81af772c37f34ba877580b7160606153fa6ad75",
    CALIBRATION_REGISTRY: "c746140f4c549a78875876f5df34a1322704ee9fa7f838120eaff14385541e20",
    CALIBRATION_COMPLETION: "c9a910d4bbf51460982fd157b8bf649da6638c88c1f5528dd36b1828f6e8cebe",
    CALIBRATION_COMPLETION_EVIDENCE: "396344021608272beab7d7989ef4c1c76fa84950a7059920621bfd8791207447",
    BASELINE_SELECTION: "505e7cb52155e0c8656a81c57754e890dff81d4b53e9ada8920de999fd11239d",
    DEVELOPMENT_SPLIT: "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140",
}

VARIANT_IDS = [
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
    "B2_RESNET50_MULTITASK",
    "B3_CONVNEXT_TINY_MULTITASK",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
]
CONFIG_IDS = ["FOVEA_DR_FULL", *VARIANT_IDS]

DISPLAY_NAMES = {
    "FOVEA_DR_FULL": "FOVEA-DR (full)",
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK": "B1 Plain EfficientNetV2-S multitask",
    "B2_RESNET50_MULTITASK": "B2 ResNet-50 multitask",
    "B3_CONVNEXT_TINY_MULTITASK": "B3 ConvNeXt-Tiny multitask",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS": "A1 FOVEA-DR without counterfactual loss",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT": (
        "A2 FOVEA-DR without clinical alignment and route agreement"
    ),
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE": (
        "A3 FOVEA-DR without learned reliability gate"
    ),
}

GROUPS = {
    "FOVEA_DR_FULL": "REFERENCE",
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK": "PLAIN_COMPARATOR",
    "B2_RESNET50_MULTITASK": "PLAIN_COMPARATOR",
    "B3_CONVNEXT_TINY_MULTITASK": "PLAIN_COMPARATOR",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS": "FOVEA_ABLATION",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT": "FOVEA_ABLATION",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE": "FOVEA_ABLATION",
}

LESION_COLUMNS = [
    "lesion_microaneurysm",
    "lesion_hard_exudate",
    "lesion_intraretinal_hemorrhage",
    "lesion_VB_IRMA",
    "lesion_neovascularization",
    "lesion_vitreous_hemorrhage",
    "lesion_retinal_detachment",
]

EXPECTED_VALIDATION_ROWS = 2839
EXPECTED_VALIDATION_COUNTS = {"CFP": 1314, "UWF": 1168, "OCT": 357}
EXPECTED_CALIBRATION_ROWS = 2840
PRIMARY_PARAMETER_COUNT = 20_695_974
METRIC_TOLERANCE = 2e-7


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
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def real_drive_mounted() -> bool:
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", "/content/drive"],
        text=True,
        capture_output=True,
        check=False,
    )
    return ROOT.is_dir() and "drive" in result.stdout.strip().lower()


def all_evidence_gates_pass(payload: dict[str, Any]) -> bool:
    gates = payload.get("gates")
    return bool(
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and gates
        and all(bool(value) for value in gates.values())
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def close(a: Any, b: Any, tolerance: float = METRIC_TOLERANCE) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def frozen_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact exists with different content: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return "CREATED_AND_FROZEN"


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")


def frozen_csv(path: Path, frame: pd.DataFrame) -> str:
    return frozen_bytes(path, csv_bytes(frame))


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    data = (
        json.dumps(json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    return frozen_bytes(path, data)


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different 5F-5A comparison-output source is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    if sha256_file(temporary) != source_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Frozen source copy verification failed.")
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def multiclass_ece(target: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == target
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(result)


def binary_ece(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(target[selected].mean()) - float(probability[selected].mean())
            )
    return float(result)


def grade_metrics(frame: pd.DataFrame, modality: str, classes: int) -> dict[str, Any]:
    subset = frame[frame["modality"].astype(str) == modality].copy()
    probability_columns = [f"probability_{index}" for index in range(classes)]
    target = pd.to_numeric(subset["target"], errors="raise").to_numpy(dtype=int)
    prediction = pd.to_numeric(subset["prediction"], errors="raise").to_numpy(dtype=int)
    expected = pd.to_numeric(subset["expected_grade"], errors="raise").to_numpy(dtype=float)
    probabilities = subset[probability_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    one_hot = np.eye(classes, dtype=float)[target]
    valid = bool(
        len(subset) > 0
        and np.isfinite(probabilities).all()
        and ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
        and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-3)
        and np.array_equal(probabilities.argmax(axis=1), prediction)
        and ((target >= 0) & (target < classes)).all()
    )
    if not valid:
        raise RuntimeError(f"Invalid selected prediction probabilities for {modality}.")
    nll = -np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-7, 1.0)).mean()
    return {
        "rows": int(len(subset)),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_F1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "QWK": float(cohen_kappa_score(target, prediction, weights="quadratic")),
        "NLL": float(nll),
        "Brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ECE15": multiclass_ece(target, probabilities),
        "expected_grade_MAE": float(mean_absolute_error(target, expected)),
    }


def validation_metrics(
    configuration_id: str,
    prediction_path: Path,
    split_validation: pd.DataFrame,
    primary_lesion_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions = pd.read_csv(prediction_path)
    predictions["relative_path"] = (
        predictions["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    required = {
        "relative_path", "modality", "target", "prediction", "expected_grade",
        *[f"probability_{index}" for index in range(5)],
    }
    lesion_probability_columns = {
        f"{lesion}_probability" for lesion in LESION_COLUMNS
    }
    has_lesion_probabilities = lesion_probability_columns.issubset(predictions.columns)
    if configuration_id != "FOVEA_DR_FULL":
        required.update(lesion_probability_columns)
    if not required.issubset(predictions.columns):
        missing = sorted(required - set(predictions.columns))
        raise RuntimeError(f"Selected prediction schema is incomplete for {configuration_id}: {missing}")
    if len(predictions) != EXPECTED_VALIDATION_ROWS or not predictions["relative_path"].is_unique:
        raise RuntimeError(f"Selected prediction row identity failed for {configuration_id}.")
    counts = predictions["modality"].astype(str).value_counts().astype(int).to_dict()
    if counts != EXPECTED_VALIDATION_COUNTS:
        raise RuntimeError(f"Selected prediction modality counts failed for {configuration_id}: {counts}")

    label_columns = [
        "relative_path", "modality", "grade", "lesion_loss_mask", *LESION_COLUMNS
    ]
    labels = split_validation[label_columns].copy()
    merged = predictions.merge(
        labels,
        on="relative_path",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_split"),
    )
    if len(merged) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(f"Prediction-to-split linkage failed for {configuration_id}.")
    if not merged["modality"].astype(str).eq(merged["modality_split"].astype(str)).all():
        raise RuntimeError(f"Modality linkage failed for {configuration_id}.")
    if not np.array_equal(
        pd.to_numeric(merged["target"], errors="raise").to_numpy(dtype=int),
        pd.to_numeric(merged["grade"], errors="raise").to_numpy(dtype=int),
    ):
        raise RuntimeError(f"Grade-target linkage failed for {configuration_id}.")

    grade_rows: list[dict[str, Any]] = []
    grade_lookup: dict[str, dict[str, Any]] = {}
    for modality, classes in (("CFP", 5), ("UWF", 5), ("OCT", 3)):
        values = grade_metrics(merged, modality, classes)
        grade_lookup[modality] = values
        grade_rows.append(
            {
                "configuration_order": CONFIG_IDS.index(configuration_id),
                "configuration_id": configuration_id,
                "configuration": DISPLAY_NAMES[configuration_id],
                "comparison_group": GROUPS[configuration_id],
                "modality": modality,
                **values,
            }
        )

    lesion_rows: list[dict[str, Any]] = []
    if has_lesion_probabilities:
        valid = (
            merged["modality"].astype(str).isin(["CFP", "UWF"])
            & (pd.to_numeric(merged["lesion_loss_mask"], errors="raise") > 0.5)
        )
        for lesion in LESION_COLUMNS:
            target = pd.to_numeric(merged.loc[valid, lesion], errors="raise").to_numpy(dtype=int)
            probability = pd.to_numeric(
                merged.loc[valid, f"{lesion}_probability"], errors="raise"
            ).to_numpy(dtype=float)
            if not (
                len(target) > 0
                and np.isfinite(probability).all()
                and ((probability >= 0.0) & (probability <= 1.0)).all()
                and np.isin(target, [0, 1]).all()
            ):
                raise RuntimeError(f"Invalid lesion predictions: {configuration_id} / {lesion}")
            prediction = (probability >= 0.5).astype(int)
            nll = -(
                target * np.log(np.clip(probability, 1e-7, 1.0))
                + (1 - target) * np.log(np.clip(1.0 - probability, 1e-7, 1.0))
            ).mean()
            lesion_rows.append(
                {
                    "configuration_order": CONFIG_IDS.index(configuration_id),
                    "configuration_id": configuration_id,
                    "configuration": DISPLAY_NAMES[configuration_id],
                    "comparison_group": GROUPS[configuration_id],
                    "lesion": lesion,
                    "valid_rows": int(len(target)),
                    "positive_rows": int(target.sum()),
                    "binary_NLL": float(nll),
                    "Brier_score": float(np.square(probability - target).mean()),
                    "ECE15": binary_ece(target, probability),
                    "accuracy_at_0_5": float(accuracy_score(target, prediction)),
                    "F1_at_0_5": float(f1_score(target, prediction, zero_division=0)),
                    "probability_metrics_available": True,
                    "metric_source": "FROZEN_SELECTED_VALIDATION_PREDICTIONS",
                }
            )
    else:
        if configuration_id != "FOVEA_DR_FULL":
            raise RuntimeError(f"Unexpected missing lesion probabilities for {configuration_id}.")
        if not isinstance(primary_lesion_metadata, dict) or set(primary_lesion_metadata) != set(LESION_COLUMNS):
            raise RuntimeError("Frozen full-model lesion metadata is missing or incomplete.")
        for lesion in LESION_COLUMNS:
            item = primary_lesion_metadata[lesion]
            lesion_rows.append(
                {
                    "configuration_order": CONFIG_IDS.index(configuration_id),
                    "configuration_id": configuration_id,
                    "configuration": DISPLAY_NAMES[configuration_id],
                    "comparison_group": GROUPS[configuration_id],
                    "lesion": lesion,
                    "valid_rows": int(item["valid_rows"]),
                    "positive_rows": int(item["positive_rows"]),
                    "binary_NLL": math.nan,
                    "Brier_score": math.nan,
                    "ECE15": math.nan,
                    "accuracy_at_0_5": float(item["accuracy_at_0_5"]),
                    "F1_at_0_5": float(item["F1_at_0_5"]),
                    "probability_metrics_available": False,
                    "metric_source": "FROZEN_SELECTED_CHECKPOINT_METADATA",
                }
            )

    lesion_frame = pd.DataFrame(lesion_rows)
    summary = {
        "configuration_order": CONFIG_IDS.index(configuration_id),
        "configuration_id": configuration_id,
        "configuration": DISPLAY_NAMES[configuration_id],
        "comparison_group": GROUPS[configuration_id],
        "CFP_QWK": grade_lookup["CFP"]["QWK"],
        "UWF_QWK": grade_lookup["UWF"]["QWK"],
        "mean_fundus_QWK": 0.5 * (grade_lookup["CFP"]["QWK"] + grade_lookup["UWF"]["QWK"]),
        "CFP_NLL": grade_lookup["CFP"]["NLL"],
        "UWF_NLL": grade_lookup["UWF"]["NLL"],
        "mean_fundus_NLL": 0.5 * (grade_lookup["CFP"]["NLL"] + grade_lookup["UWF"]["NLL"]),
        "OCT_QWK": grade_lookup["OCT"]["QWK"],
        "OCT_NLL": grade_lookup["OCT"]["NLL"],
        "OCT_macro_F1": grade_lookup["OCT"]["macro_F1"],
        "mean_lesion_binary_NLL": float(lesion_frame["binary_NLL"].mean()),
        "mean_lesion_Brier": float(lesion_frame["Brier_score"].mean()),
        "mean_lesion_ECE15": float(lesion_frame["ECE15"].mean()),
        "mean_lesion_accuracy_at_0_5": float(lesion_frame["accuracy_at_0_5"].mean()),
        "mean_lesion_F1_at_0_5": float(lesion_frame["F1_at_0_5"].mean()),
        "validation_lesion_probability_metrics_available": bool(has_lesion_probabilities),
    }
    return grade_rows, lesion_rows, summary


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    weights = pd.to_numeric(frame["rows"], errors="raise").to_numpy(dtype=float)
    if not (len(values) and np.isfinite(values).all() and np.isfinite(weights).all() and weights.sum() > 0):
        raise RuntimeError(f"Invalid weighted metric: {column}")
    return float(np.average(values, weights=weights))


def calibration_summary(configuration_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    grade = metrics[metrics["task_type"].astype(str) == "GRADE"].copy()
    lesion = metrics[metrics["task_type"].astype(str) == "LESION"].copy()
    if not (
        len(metrics) == 10
        and len(grade) == 3
        and set(grade["unit"].astype(str)) == {"CFP", "UWF", "OCT"}
        and len(lesion) == 7
        and set(lesion["unit"].astype(str)) == set(LESION_COLUMNS)
    ):
        raise RuntimeError(f"Calibration metric schema failed for {configuration_id}.")
    if not metrics["decision_invariant"].map(parse_bool).all():
        raise RuntimeError(f"Calibration decision invariance failed for {configuration_id}.")
    if not metrics["NLL_nonworsening"].map(parse_bool).all():
        raise RuntimeError(f"Calibration NLL gate failed for {configuration_id}.")
    before = pd.to_numeric(metrics["NLL_before"], errors="raise")
    after = pd.to_numeric(metrics["NLL_after"], errors="raise")
    if not bool((after <= before + 1e-12).all()):
        raise RuntimeError(f"Calibration NLL values worsened for {configuration_id}.")
    result: dict[str, Any] = {
        "configuration_order": CONFIG_IDS.index(configuration_id),
        "configuration_id": configuration_id,
        "configuration": DISPLAY_NAMES[configuration_id],
        "comparison_group": GROUPS[configuration_id],
    }
    for prefix, subset in (("grade", grade), ("lesion", lesion)):
        for metric in ("NLL", "Brier", "ECE15"):
            result[f"{prefix}_weighted_{metric}_before"] = weighted_mean(subset, f"{metric}_before")
            result[f"{prefix}_weighted_{metric}_after"] = weighted_mean(subset, f"{metric}_after")
    return result


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> list[str]:
    headers = [heading for _, heading, _ in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.to_dict("records"):
        cells: list[str] = []
        for key, _, spec in columns:
            value = row[key]
            if spec == "text":
                cells.append(str(value).replace("|", "/"))
            elif spec == "int":
                cells.append(str(int(value)))
            else:
                cells.append(format(float(value), spec))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    banner("STAGE 0G-4D-5F-5A-R1 — SCHEMA-AWARE FROZEN POST-HOC COMPARISON OUTPUTS")
    if not real_drive_mounted():
        raise RuntimeError("The real Google Drive is not mounted in this notebook process.")

    section("[1] FROZEN PARENT ARTIFACTS")
    gates: dict[str, bool] = {}
    for path, expected in EXPECTED_SHA256.items():
        passed = path.is_file() and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(passed)
        print(f"{path.name:<92}: {passed}")
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError("A frozen parent artifact changed:\n" + "\n".join(failed))

    protocol = read_json(POSTHOC_PROTOCOL)
    training_registry = pd.read_csv(FINAL_TRAINING_REGISTRY)
    calibration_registry = pd.read_csv(CALIBRATION_REGISTRY)
    calibration_completion = read_json(CALIBRATION_COMPLETION)
    calibration_completion_evidence = read_json(CALIBRATION_COMPLETION_EVIDENCE)
    baseline = read_json(BASELINE_SELECTION)
    baseline_evidence = read_json(BASELINE_SELECTION_EVIDENCE)
    primary_calibration_evidence = read_json(PRIMARY_CALIBRATION_EVIDENCE)
    implementation_evidence = read_json(IMPLEMENTATION_EVIDENCE)

    protocol_order = [item["id"] for item in protocol.get("comparators", [])]
    protocol_order += [item["id"] for item in protocol.get("ablations", [])]
    parent_semantic_gates = {
        "posthoc_protocol_order_exact": protocol_order == VARIANT_IDS,
        "report_all_six_required": protocol.get("reporting_policy", {}).get("report_all_six_frozen_runs") is True,
        "no_significance_claim_required": protocol.get("reporting_policy", {}).get(
            "no_significance_claim_from_overlapping_validation_runs"
        ) is True,
        "locked_test_prohibited": str(protocol.get("data_access_policy", {}).get(
            "LOCKED_INTERNAL_TEST", ""
        )).startswith("PROHIBITED"),
        "training_registry_order_exact": training_registry["variant_id"].astype(str).tolist() == VARIANT_IDS,
        "training_registry_all_complete": len(training_registry) == 6
        and training_registry["status"].astype(str).eq("COMPLETE").all()
        and training_registry["all_completion_checks_passed"].map(parse_bool).all(),
        "calibration_registry_order_exact": calibration_registry["variant_id"].astype(str).tolist() == VARIANT_IDS,
        "calibration_registry_all_complete": len(calibration_registry) == 6
        and calibration_registry["status"].astype(str).eq("COMPLETE_AND_TEMPERATURE_FITTED").all()
        and calibration_registry["all_checks_passed"].map(parse_bool).all(),
        "calibration_completion_frozen": calibration_completion.get("status") == "COMPLETE_AND_FROZEN",
        "calibration_completion_evidence_PASS": all_evidence_gates_pass(calibration_completion_evidence),
        "baseline_selection_frozen": baseline.get("status") == "FROZEN"
        and int(baseline.get("selected_epoch", -1)) == 10,
        "baseline_selection_evidence_PASS": all_evidence_gates_pass(baseline_evidence),
        "primary_calibration_evidence_PASS": all_evidence_gates_pass(primary_calibration_evidence)
        and primary_calibration_evidence.get("session_status") == "COMPLETE_AND_TEMPERATURE_FITTED"
        and primary_calibration_evidence.get("validation_used_for_calibration_fit") is False
        and primary_calibration_evidence.get("locked_test_accessed") is False,
        "implementation_evidence_PASS": all_evidence_gates_pass(implementation_evidence),
    }
    gates.update(parent_semantic_gates)
    for name, passed in parent_semantic_gates.items():
        print(f"{name:<92}: {passed}")
    if not all(parent_semantic_gates.values()):
        failed = [name for name, passed in parent_semantic_gates.items() if not passed]
        raise RuntimeError("Frozen parent semantics failed:\n" + "\n".join(failed))

    source_action, source_sha = freeze_source()
    gates["source_frozen_and_verified"] = FROZEN_SOURCE.is_file() and sha256_file(FROZEN_SOURCE) == source_sha

    section("[2] FROZEN VALIDATION LABEL TABLE")
    split = pd.read_csv(DEVELOPMENT_SPLIT)
    split["relative_path"] = split["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    validation = split[split["development_split"].astype(str) == "VALIDATION"].copy()
    validation_counts = validation["modality"].astype(str).value_counts().astype(int).to_dict()
    validation_gates = {
        "validation_rows_exact": len(validation) == EXPECTED_VALIDATION_ROWS,
        "validation_modality_counts_exact": validation_counts == EXPECTED_VALIDATION_COUNTS,
        "validation_paths_unique": validation["relative_path"].is_unique,
        "lesion_columns_present": {"lesion_loss_mask", *LESION_COLUMNS}.issubset(validation.columns),
    }
    gates.update(validation_gates)
    for name, passed in validation_gates.items():
        print(f"{name:<92}: {passed}")
    if not all(validation_gates.values()):
        raise RuntimeError("Frozen VALIDATION table gate failed.")

    section("[3] SEVEN SELECTED-EPOCH VALIDATION RESULTS")
    selected_epochs: dict[str, int] = {"FOVEA_DR_FULL": int(baseline["selected_epoch"])}
    prediction_paths: dict[str, Path] = {
        "FOVEA_DR_FULL": Path(str(baseline["frozen_selected_validation_predictions"]))
    }
    prediction_hashes: dict[str, str] = {
        "FOVEA_DR_FULL": str(baseline["frozen_selected_validation_predictions_sha256"])
    }
    history_paths: dict[str, Path] = {"FOVEA_DR_FULL": Path(str(baseline["history"]))}
    history_hashes: dict[str, str] = {"FOVEA_DR_FULL": str(baseline["history_sha256"])}
    expected_selected_metrics: dict[str, dict[str, Any]] = {
        "FOVEA_DR_FULL": dict(baseline["selected_metrics"])
    }

    # The legacy full-model prediction CSV intentionally contains grade outputs
    # only.  Its selected checkpoint froze per-lesion accuracy/F1 metadata at
    # training time.  Read that trusted, hash-verified checkpoint on CPU and
    # discard all tensor payloads immediately after extracting the metadata.
    import torch

    primary_checkpoint_path = Path(str(baseline["frozen_selected_checkpoint"]))
    primary_checkpoint_sha = str(baseline["frozen_selected_checkpoint_sha256"])
    if not (
        primary_checkpoint_path.is_file()
        and sha256_file(primary_checkpoint_path) == primary_checkpoint_sha
    ):
        raise RuntimeError("The frozen full-model selected checkpoint changed.")

    primary_checkpoint = torch.load(
        primary_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    def all_tensors_on_cpu(value: Any) -> bool:
        if torch.is_tensor(value):
            return value.device.type == "cpu"
        if isinstance(value, dict):
            return all(all_tensors_on_cpu(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(all_tensors_on_cpu(item) for item in value)
        return True

    checkpoint_cpu_only = all_tensors_on_cpu(primary_checkpoint)
    primary_lesion_metadata = json_ready(primary_checkpoint.get("lesion_metrics"))
    checkpoint_metadata_valid = bool(
        isinstance(primary_checkpoint, dict)
        and primary_checkpoint.get("format") == "FOVEA_DR_BASELINE_BEST_MODEL_V1"
        and int(primary_checkpoint.get("epoch", -1)) == 10
        and isinstance(primary_lesion_metadata, dict)
        and set(primary_lesion_metadata) == set(LESION_COLUMNS)
        and all(
            isinstance(primary_lesion_metadata[lesion], dict)
            and {
                "valid_rows", "positive_rows", "accuracy_at_0_5", "F1_at_0_5"
            }.issubset(primary_lesion_metadata[lesion])
            for lesion in LESION_COLUMNS
        )
    )
    del primary_checkpoint
    gc.collect()
    gates.update(
        {
            "primary_checkpoint_hash_verified_for_metadata": True,
            "primary_checkpoint_loaded_on_CPU_only": checkpoint_cpu_only,
            "primary_frozen_lesion_metadata_complete": checkpoint_metadata_valid,
        }
    )
    if not (checkpoint_cpu_only and checkpoint_metadata_valid):
        raise RuntimeError("Frozen full-model lesion metadata verification failed.")

    training_lookup = training_registry.set_index("variant_id")
    for variant_id in VARIANT_IDS:
        row = training_lookup.loc[variant_id]
        run_root = ROOT / "Runs/FOVEA_DR/posthoc_comparisons/final_seed2026" / variant_id
        selected_epochs[variant_id] = int(row["selected_best_epoch"])
        prediction_paths[variant_id] = run_root / "best_validation_predictions.csv"
        prediction_hashes[variant_id] = str(row["best_validation_predictions_sha256"])
        history_paths[variant_id] = run_root / "epoch_history.csv"
        history_hashes[variant_id] = str(row["history_sha256"])
        expected_selected_metrics[variant_id] = {
            "mean_fundus_QWK": float(row["best_mean_fundus_QWK"]),
            "mean_fundus_NLL": float(row["best_mean_fundus_NLL"]),
        }

    grade_records: list[dict[str, Any]] = []
    lesion_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}
    per_configuration_gates: dict[str, bool] = {}

    for configuration_id in CONFIG_IDS:
        prediction_path = prediction_paths[configuration_id]
        history_path = history_paths[configuration_id]
        artifacts_verified = bool(
            prediction_path.is_file()
            and sha256_file(prediction_path) == prediction_hashes[configuration_id]
            and history_path.is_file()
            and sha256_file(history_path) == history_hashes[configuration_id]
        )
        if not artifacts_verified:
            raise RuntimeError(f"Selected VALIDATION artifacts changed for {configuration_id}.")
        history = pd.read_csv(history_path)
        histories[configuration_id] = history
        selected_epoch = selected_epochs[configuration_id]
        selected_rows = history[pd.to_numeric(history["epoch"], errors="raise").astype(int) == selected_epoch]
        if len(selected_rows) != 1:
            raise RuntimeError(f"Selected epoch is not unique for {configuration_id}.")
        selected_history = selected_rows.iloc[0]
        grade_rows, lesion_rows, summary = validation_metrics(
            configuration_id,
            prediction_path,
            validation,
            primary_lesion_metadata=(
                primary_lesion_metadata
                if configuration_id == "FOVEA_DR_FULL"
                else None
            ),
        )
        summary["selected_epoch"] = selected_epoch
        summary["selection_partition"] = "VALIDATION"
        summary["selected_predictions_sha256"] = prediction_hashes[configuration_id]
        recomputed_matches_history = all(
            close(summary[key], selected_history[key])
            for key in [
                "CFP_QWK", "UWF_QWK", "mean_fundus_QWK", "CFP_NLL", "UWF_NLL",
                "mean_fundus_NLL", "OCT_QWK", "OCT_NLL", "OCT_macro_F1",
            ]
        )
        frozen_reference_matches = all(
            close(summary[key], value)
            for key, value in expected_selected_metrics[configuration_id].items()
        )
        per_configuration_gates[configuration_id] = bool(
            artifacts_verified and recomputed_matches_history and frozen_reference_matches
        )
        if not per_configuration_gates[configuration_id]:
            raise RuntimeError(f"Selected metric reproduction failed for {configuration_id}.")
        grade_records.extend(grade_rows)
        lesion_records.extend(lesion_rows)
        summary_records.append(summary)
        print(
            f"{configuration_id:<68} | epoch={selected_epoch:2d} | "
            f"mean Fundus QWK={summary['mean_fundus_QWK']:.6f} | PASS=True"
        )

    grade_frame = pd.DataFrame(grade_records)
    lesion_frame = pd.DataFrame(lesion_records)
    validation_summary = pd.DataFrame(summary_records).sort_values("configuration_order")
    gates.update({f"validation_reproduced::{key}": value for key, value in per_configuration_gates.items()})
    gates.update(
        {
            "seven_validation_summaries_exact": len(validation_summary) == 7,
            "twenty_one_grade_rows_exact": len(grade_frame) == 21,
            "forty_nine_lesion_rows_exact": len(lesion_frame) == 49,
        }
    )

    section("[4] SEVEN POST-SELECTION CALIBRATION RESULTS")
    primary_metrics_path = Path(str(primary_calibration_evidence["metrics"]))
    primary_metrics_sha = str(primary_calibration_evidence["metrics_sha256"])
    calibration_lookup = calibration_registry.set_index("variant_id")
    calibration_metric_paths: dict[str, Path] = {"FOVEA_DR_FULL": primary_metrics_path}
    calibration_metric_hashes: dict[str, str] = {"FOVEA_DR_FULL": primary_metrics_sha}
    for variant_id in VARIANT_IDS:
        row = calibration_lookup.loc[variant_id]
        calibration_metric_paths[variant_id] = Path(str(row["calibration_metrics"]))
        calibration_metric_hashes[variant_id] = str(row["calibration_metrics_sha256"])

    calibration_long_records: list[dict[str, Any]] = []
    calibration_summary_records: list[dict[str, Any]] = []
    for configuration_id in CONFIG_IDS:
        metrics_path = calibration_metric_paths[configuration_id]
        if not (
            metrics_path.is_file()
            and sha256_file(metrics_path) == calibration_metric_hashes[configuration_id]
        ):
            raise RuntimeError(f"Calibration metrics changed for {configuration_id}.")
        metrics = pd.read_csv(metrics_path)
        summary = calibration_summary(configuration_id, metrics)
        summary["selected_epoch"] = selected_epochs[configuration_id]
        summary["calibration_rows"] = EXPECTED_CALIBRATION_ROWS
        summary["calibration_metrics_sha256"] = calibration_metric_hashes[configuration_id]
        calibration_summary_records.append(summary)
        for item in metrics.to_dict("records"):
            item.pop("variant_id", None)
            calibration_long_records.append(
                {
                    "configuration_order": CONFIG_IDS.index(configuration_id),
                    "configuration_id": configuration_id,
                    "configuration": DISPLAY_NAMES[configuration_id],
                    "comparison_group": GROUPS[configuration_id],
                    "selected_epoch": selected_epochs[configuration_id],
                    **item,
                }
            )
        print(
            f"{configuration_id:<68} | rows=2840 | units=10 | "
            f"grade NLL {summary['grade_weighted_NLL_before']:.6f} -> "
            f"{summary['grade_weighted_NLL_after']:.6f}"
        )

    calibration_long = pd.DataFrame(calibration_long_records)
    calibration_summary_frame = pd.DataFrame(calibration_summary_records).sort_values("configuration_order")
    gates.update(
        {
            "seventy_calibration_metric_rows_exact": len(calibration_long) == 70,
            "seven_calibration_summaries_exact": len(calibration_summary_frame) == 7,
            "all_calibration_decisions_invariant": calibration_long["decision_invariant"].map(parse_bool).all(),
            "all_calibration_NLL_nonworsening": calibration_long["NLL_nonworsening"].map(parse_bool).all(),
        }
    )

    section("[5] DESCRIPTIVE CONTRASTS AND COMPUTATIONAL CHARACTERISTICS")
    combined = validation_summary.merge(
        calibration_summary_frame.drop(
            columns=["configuration", "comparison_group", "selected_epoch"],
            errors="ignore",
        ),
        on=["configuration_order", "configuration_id"],
        how="inner",
        validate="one_to_one",
    )
    contrast_metrics = {
        "mean_fundus_QWK": "HIGHER_IS_BETTER",
        "CFP_QWK": "HIGHER_IS_BETTER",
        "UWF_QWK": "HIGHER_IS_BETTER",
        "mean_fundus_NLL": "LOWER_IS_BETTER",
        "OCT_QWK": "HIGHER_IS_BETTER",
        "OCT_macro_F1": "HIGHER_IS_BETTER",
        "OCT_NLL": "LOWER_IS_BETTER",
        "mean_lesion_accuracy_at_0_5": "HIGHER_IS_BETTER",
        "mean_lesion_F1_at_0_5": "HIGHER_IS_BETTER",
        "grade_weighted_NLL_after": "LOWER_IS_BETTER",
        "grade_weighted_ECE15_after": "LOWER_IS_BETTER",
        "lesion_weighted_NLL_after": "LOWER_IS_BETTER",
        "lesion_weighted_ECE15_after": "LOWER_IS_BETTER",
    }
    reference = combined[combined["configuration_id"] == "FOVEA_DR_FULL"].iloc[0]
    contrast_records: list[dict[str, Any]] = []
    for variant_id in VARIANT_IDS:
        comparison = combined[combined["configuration_id"] == variant_id].iloc[0]
        for metric, direction in contrast_metrics.items():
            reference_value = float(reference[metric])
            comparison_value = float(comparison[metric])
            delta = reference_value - comparison_value
            favourable = delta if direction == "HIGHER_IS_BETTER" else -delta
            contrast_records.append(
                {
                    "contrast_order": VARIANT_IDS.index(variant_id) + 1,
                    "contrast_id": f"FOVEA_DR_FULL_MINUS_{variant_id}",
                    "contrast_family": (
                        "FULL_VS_PLAIN_COMPARATOR"
                        if GROUPS[variant_id] == "PLAIN_COMPARATOR"
                        else "MECHANISTIC_ABLATION"
                    ),
                    "reference_id": "FOVEA_DR_FULL",
                    "comparison_id": variant_id,
                    "metric": metric,
                    "direction": direction,
                    "reference_value": reference_value,
                    "comparison_value": comparison_value,
                    "delta_full_minus_comparison": delta,
                    "favourability_aligned_delta_positive_favours_full": favourable,
                    "inference_scope": "DESCRIPTIVE_DEVELOPMENT_SET_ONLY",
                    "p_value": "NOT_COMPUTED_BY_FROZEN_PROTOCOL",
                    "confidence_interval": "NOT_COMPUTED_BY_FROZEN_PROTOCOL",
                }
            )
    contrast_frame = pd.DataFrame(contrast_records)

    parameter_records = implementation_evidence.get("variant_records", [])
    parameter_lookup = {
        str(item["variant_id"]): int(item["total_parameters"])
        for item in parameter_records
        if isinstance(item, dict) and "variant_id" in item and "total_parameters" in item
    }
    computation_records: list[dict[str, Any]] = []
    for configuration_id in CONFIG_IDS:
        history = histories[configuration_id]
        train_seconds = pd.to_numeric(history["train_seconds"], errors="raise")
        validation_seconds = pd.to_numeric(history["validation_seconds"], errors="raise")
        peak_vram = pd.to_numeric(history["peak_reserved_VRAM_GB"], errors="raise")
        parameters = (
            PRIMARY_PARAMETER_COUNT
            if configuration_id == "FOVEA_DR_FULL"
            else parameter_lookup.get(configuration_id)
        )
        if parameters is None:
            raise RuntimeError(f"Missing audited parameter count for {configuration_id}.")
        computation_records.append(
            {
                "configuration_order": CONFIG_IDS.index(configuration_id),
                "configuration_id": configuration_id,
                "configuration": DISPLAY_NAMES[configuration_id],
                "comparison_group": GROUPS[configuration_id],
                "parameters": int(parameters),
                "selected_epoch": selected_epochs[configuration_id],
                "completed_epochs": int(len(history)),
                "global_optimizer_steps": int(pd.to_numeric(history["global_optimizer_step"], errors="raise").iloc[-1]),
                "total_training_seconds": float(train_seconds.sum()),
                "total_training_hours": float(train_seconds.sum() / 3600.0),
                "total_validation_seconds": float(validation_seconds.sum()),
                "total_validation_hours": float(validation_seconds.sum() / 3600.0),
                "peak_reserved_VRAM_GB": float(peak_vram.max()),
                "run_seed": 2026,
                "single_seed": True,
            }
        )
    computation_frame = pd.DataFrame(computation_records)
    gates.update(
        {
            "seventy_eight_descriptive_contrasts_exact": len(contrast_frame) == 78,
            "no_inferential_statistics_generated": contrast_frame["p_value"].eq(
                "NOT_COMPUTED_BY_FROZEN_PROTOCOL"
            ).all(),
            "seven_computational_rows_exact": len(computation_frame) == 7,
            "all_parameter_counts_positive": (computation_frame["parameters"] > 0).all(),
            "all_times_finite_nonnegative": np.isfinite(
                computation_frame[["total_training_seconds", "total_validation_seconds"]].to_numpy(dtype=float)
            ).all()
            and (computation_frame[["total_training_seconds", "total_validation_seconds"]] >= 0).all().all(),
        }
    )

    section("[6] FREEZE TABLES, REPORT, CONTRACT, AND EVIDENCE")
    output_frames = {
        VALIDATION_SUMMARY: validation_summary,
        VALIDATION_GRADE_LONG: grade_frame.sort_values(["configuration_order", "modality"]),
        VALIDATION_LESION_LONG: lesion_frame.sort_values(["configuration_order", "lesion"]),
        CALIBRATION_LONG: calibration_long.sort_values(["configuration_order", "task_type", "unit"]),
        CALIBRATION_SUMMARY: calibration_summary_frame,
        CONTRASTS: contrast_frame.sort_values(["contrast_order", "metric"]),
        COMPUTATION: computation_frame,
    }
    output_actions: dict[str, str] = {}
    for path, frame in output_frames.items():
        output_actions[str(path)] = frozen_csv(path, frame)

    primary_contrasts = contrast_frame[contrast_frame["metric"] == "mean_fundus_QWK"].copy()
    primary_contrasts["comparison"] = primary_contrasts["comparison_id"].map(DISPLAY_NAMES)
    report_lines = [
        "# FOVEA-DR post-hoc comparator and ablation results",
        "",
        "> **Scope:** descriptive, development-only, single-seed post-hoc comparison. ",
        "> No comparator or ablation was evaluated on the locked internal test, and no ",
        "> p-values, confidence intervals, or significance claims are generated here.",
        "",
        "## Selected-epoch VALIDATION performance",
        "",
    ]
    report_lines += markdown_table(
        validation_summary,
        [
            ("configuration", "Configuration", "text"),
            ("selected_epoch", "Epoch", "int"),
            ("CFP_QWK", "CFP QWK", ".6f"),
            ("UWF_QWK", "UWF QWK", ".6f"),
            ("mean_fundus_QWK", "Mean fundus QWK", ".6f"),
            ("mean_fundus_NLL", "Mean fundus NLL", ".6f"),
            ("OCT_QWK", "OCT QWK", ".6f"),
            ("OCT_macro_F1", "OCT macro-F1", ".6f"),
        ],
    )
    report_lines += [
        "",
        "### VALIDATION lesion-metric availability",
        "",
        "The legacy frozen full-model VALIDATION prediction table did not store lesion probabilities. Its per-lesion accuracy and F1 values were recovered from the hash-verified selected-checkpoint metadata on CPU. Full-model VALIDATION lesion NLL, Brier score, and ECE are therefore reported as unavailable rather than recomputed. All five lesion metrics remain available for every configuration on the post-selection CALIBRATION partition.",
        "",
        "## Primary descriptive contrasts",
        "",
        "The delta is defined as full FOVEA-DR minus the compared configuration. These deltas are descriptive only.",
        "",
    ]
    report_lines += markdown_table(
        primary_contrasts,
        [
            ("comparison", "Compared configuration", "text"),
            ("reference_value", "Full QWK", ".6f"),
            ("comparison_value", "Other QWK", ".6f"),
            ("delta_full_minus_comparison", "Delta", "+.6f"),
        ],
    )
    report_lines += ["", "## Post-selection CALIBRATION summary", ""]
    report_lines += markdown_table(
        calibration_summary_frame,
        [
            ("configuration", "Configuration", "text"),
            ("grade_weighted_NLL_before", "Grade NLL raw", ".6f"),
            ("grade_weighted_NLL_after", "Grade NLL calibrated", ".6f"),
            ("grade_weighted_ECE15_before", "Grade ECE raw", ".6f"),
            ("grade_weighted_ECE15_after", "Grade ECE calibrated", ".6f"),
            ("lesion_weighted_NLL_before", "Lesion NLL raw", ".6f"),
            ("lesion_weighted_NLL_after", "Lesion NLL calibrated", ".6f"),
        ],
    )
    report_lines += ["", "## Computational characteristics", ""]
    report_lines += markdown_table(
        computation_frame,
        [
            ("configuration", "Configuration", "text"),
            ("parameters", "Parameters", "int"),
            ("selected_epoch", "Best epoch", "int"),
            ("completed_epochs", "Completed epochs", "int"),
            ("total_training_hours", "Training hours", ".3f"),
            ("total_validation_hours", "Validation hours", ".3f"),
            ("peak_reserved_VRAM_GB", "Peak VRAM (GB)", ".3f"),
        ],
    )
    report_lines += [
        "",
        "## Interpretation boundary",
        "",
        "All seven configurations are reported. The six added runs were post hoc and used one seed on the same frozen development partitions. Their differences therefore support descriptive method comparison and ablation analysis only. The already-frozen primary FOVEA-DR locked-test result is deliberately not reproduced in these comparator tables.",
        "",
    ]
    report_action = frozen_bytes(REPORT, "\n".join(report_lines).encode("utf-8"))
    output_actions[str(REPORT)] = report_action

    output_hashes = {str(path): sha256_file(path) for path in [*output_frames, REPORT]}
    gates.update(
        {
            "all_eight_outputs_exist_nonempty": all(
                Path(path).is_file() and Path(path).stat().st_size > 0 for path in output_hashes
            ),
            "models_not_loaded": True,
            "only_frozen_primary_checkpoint_loaded_on_CPU_for_metadata": True,
            "no_checkpoint_used_for_inference": True,
            "medical_images_not_accessed": True,
            "optimizer_not_created": True,
            "parameters_not_updated": True,
            "locked_test_artifacts_not_read": True,
        }
    )
    failed = [name for name, passed in gates.items() if not bool(passed)]
    if failed:
        raise RuntimeError("Final comparison-output gate failed:\n" + "\n".join(failed))

    parent_hashes = {str(path): expected for path, expected in EXPECTED_SHA256.items()}
    parent_hashes[str(PRIMARY_CALIBRATION_EVIDENCE)] = sha256_file(PRIMARY_CALIBRATION_EVIDENCE)
    parent_hashes[str(BASELINE_SELECTION_EVIDENCE)] = sha256_file(BASELINE_SELECTION_EVIDENCE)
    parent_hashes[str(IMPLEMENTATION_EVIDENCE)] = sha256_file(IMPLEMENTATION_EVIDENCE)
    contract = {
        "contract_id": "MMRDR_FOVEA_DR_POSTHOC_COMPARISON_OUTPUTS_V1",
        "stage": "0G-4D-5F-5A-R1",
        "status": "COMPLETE_AND_FROZEN",
        "classification": "DESCRIPTIVE_DEVELOPMENT_ONLY_POST_HOC_COMPARISON",
        "configuration_order": CONFIG_IDS,
        "reference_configuration": "FOVEA_DR_FULL",
        "primary_endpoint": "UNWEIGHTED_MEAN_CFP_UWF_QWK_ON_FROZEN_VALIDATION",
        "contrast_definition": "FULL_FOVEA_DR_MINUS_COMPARED_CONFIGURATION",
        "inferential_statistics": "NOT_COMPUTED_BY_FROZEN_PROTOCOL",
        "parent_artifact_sha256": parent_hashes,
        "source": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "outputs": {path: sha for path, sha in output_hashes.items()},
        "all_six_posthoc_runs_reported": True,
        "single_seed_limitation": True,
        "full_model_validation_lesion_probability_metrics": "UNAVAILABLE_IN_LEGACY_FROZEN_PREDICTION_ARTIFACT",
        "full_model_validation_lesion_accuracy_and_F1_source": "HASH_VERIFIED_SELECTED_CHECKPOINT_METADATA_CPU_READ_ONLY",
        "locked_test_result_included_in_comparator_tables": False,
        "model_loaded": False,
        "checkpoint_loaded_on_CPU_for_metadata_only": True,
        "checkpoint_used_for_inference": False,
        "medical_images_accessed": False,
        "model_parameters_updated": False,
        "locked_test_accessed": False,
        "next_required_stage": "MANUSCRIPT_RESULTS_SYNTHESIS_AND_FIGURE_GENERATION",
    }
    contract_action = frozen_json(OUTPUT_CONTRACT, contract)
    contract_sha = sha256_file(OUTPUT_CONTRACT)
    evidence = {
        "stage": "0G-4D-5F-5A-R1",
        "status": "PASS",
        "purpose": "GENERATE_FROZEN_DESCRIPTIVE_POSTHOC_COMPARISON_OUTPUTS",
        "source": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "output_contract": str(OUTPUT_CONTRACT),
        "output_contract_sha256": contract_sha,
        "outputs": {path: sha for path, sha in output_hashes.items()},
        "gates": {name: bool(value) for name, value in gates.items()},
        "validation_prediction_rows_read": 7 * EXPECTED_VALIDATION_ROWS,
        "calibration_metric_rows_read": 70,
        "medical_images_accessed": False,
        "models_loaded": False,
        "checkpoint_tensors_loaded_on_CPU_for_frozen_metadata": True,
        "checkpoint_used_for_inference": False,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "p_values_computed": False,
        "confidence_intervals_computed": False,
        "locked_test_artifacts_read": False,
        "locked_test_accessed": False,
        "next_required_stage": "MANUSCRIPT_RESULTS_SYNTHESIS_AND_FIGURE_GENERATION",
    }
    evidence_action = frozen_json(EVIDENCE, evidence)
    evidence_sha = sha256_file(EVIDENCE)
    os.sync()

    for path in [*output_frames, REPORT]:
        print(f"{path.name:<74}: {output_actions[str(path)]}")
        print("  SHA-256:", output_hashes[str(path)])
    print(f"{'Source':<74}: {source_action}")
    print("  SHA-256:", source_sha)
    print(f"{'Contract':<74}: {contract_action}")
    print("  SHA-256:", contract_sha)
    print(f"{'Evidence':<74}: {evidence_action}")
    print("  SHA-256:", evidence_sha)

    banner("STAGE 0G-4D-5F-5A-R1 STATUS: PASS")
    print("Configurations reported      : 7 / 7")
    print("Selected VALIDATION rows read:", 7 * EXPECTED_VALIDATION_ROWS)
    print("CALIBRATION metric rows read : 70")
    print("Descriptive contrasts        :", len(contrast_frame))
    print("Inferential statistics       : NOT COMPUTED")
    print("Medical images accessed      : NO")
    print("Models loaded                : NO")
    print("Parameters updated           : NO")
    print("Locked test accessed         : NO")
    print("\nManuscript-ready report:")
    print(REPORT)
    print("\nNEXT ACTION: MANUSCRIPT RESULTS SYNTHESIS AND FIGURE GENERATION")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        banner("STAGE 0G-4D-5F-5A-R1 STATUS: FAIL")
        raise
