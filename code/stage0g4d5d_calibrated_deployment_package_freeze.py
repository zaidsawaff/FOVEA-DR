from pathlib import Path
import hashlib
import json
import math
import os
import shutil

import numpy as np
import pandas as pd


# =============================================================================
# STAGE 0G-4D-5D — CALIBRATED DEPLOYMENT PACKAGE FREEZE
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

FROZEN_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "Checkpoints/FOVEA_DR/baseline_seed2026/selected_epoch10_frozen.pt"
)
PRODUCTION_SOURCE_PATH = (
    PROJECT_ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
)
CALIBRATION_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_post_selection_calibration_protocol_v1.json"
)
TEMPERATURES_PATH = (
    PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_temperature_parameters_v1.json"
)
CALIBRATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Audit/mmrdr_fovea_dr_CALIBRATION_local_cache_manifest_v1.csv"
)
RAW_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "Runs/FOVEA_DR/calibration_seed2026_cpu_fp32/calibration_raw_predictions.csv"
)
CALIBRATED_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "Runs/FOVEA_DR/calibration_seed2026_cpu_fp32/calibration_temperature_scaled_predictions.csv"
)
METRICS_PATH = (
    PROJECT_ROOT
    / "Audit/mmrdr_fovea_dr_CALIBRATION_metrics_before_after_temperature_v1.csv"
)
INFERENCE_STATE_PATH = (
    PROJECT_ROOT
    / "Runs/FOVEA_DR/calibration_seed2026_cpu_fp32/inference_state.json"
)
CALIBRATION_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5c_cpu_calibration_inference_and_temperature_fit.json"
)
CALIBRATION_RUNNER_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/stage0g4d5c_resumable_cpu_calibration_inference.py"
)

FROZEN_DEPLOYMENT_SOURCE_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/stage0g4d5d_calibrated_deployment_package_freeze.py"
)
CALIBRATED_TRANSFORM_SOURCE_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/fovea_dr_calibrated_probability_transform_v1.py"
)
PACKAGE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_calibrated_deployment_package_v1.json"
)
EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5d_calibrated_deployment_package_freeze.json"
)

EXPECTED_HASHES = {
    "frozen_checkpoint": "89f5e87b647680a9094c55bf647dc1dd2abb624681e733ffb453e30e4c11ff1f",
    "production_source": "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9",
    "calibration_protocol": "f569f63c9b9c44816f19ce5f49f724fc5b1e4dc202d6d4a8694250f7b33995e7",
    "temperature_parameters": "b694a32f84126516b0c6dc1ef0e32e130ffe6e6919433bf751399e21d10894c8",
    "calibration_manifest": "f8e6a62b96e330a1b21bb9ed56897cbf04b730987cf9e0c31141927abbb84655",
    "raw_predictions": "cf10c19b9f56b0512444015722857d756563b2c5f9fbf7388fd39f5ce8cefd03",
    "calibrated_predictions": "9b0810953067fb6995d35f1fc63deefbdc847126330523bc07aaed0e4408c85c",
    "calibration_runner": "9d8050264362b7af12aa8962c9cc16528c0b75f11eac26040974068ad4e2d522",
}

EXPECTED_ROWS = 2840
EXPECTED_MODALITY_COUNTS = {"CFP": 1313, "UWF": 1171, "OCT": 356}
EXPECTED_RUN_SIGNATURE = (
    "6e5a0bdbea838b952a601dc6cda8407ab31637ede2ba7e184500db632d02c960"
)
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 20.0
EPSILON = 1e-12

LESION_COLUMNS = [
    "lesion_microaneurysm",
    "lesion_hard_exudate",
    "lesion_intraretinal_hemorrhage",
    "lesion_VB_IRMA",
    "lesion_neovascularization",
    "lesion_vitreous_hemorrhage",
    "lesion_retinal_detachment",
]

CALIBRATED_TRANSFORM_SOURCE = '''"""Frozen FOVEA-DR v1.1 post-selection probability calibration."""

import math
import numpy as np

EPSILON = 1e-12


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def apply_grade_temperature(probabilities, temperature):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    log_probabilities = np.log(np.clip(probabilities, EPSILON, 1.0))
    return _softmax(log_probabilities / temperature)


def apply_lesion_temperature(logits, temperature):
    logits = np.asarray(logits, dtype=np.float64)
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    return _sigmoid(logits / temperature)
'''


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(path) as file:
        return json.load(file)


def atomic_json_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary, path)


def create_text_once_verified(text, path):
    expected_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != expected_sha:
            raise RuntimeError(
                f"Frozen transform source exists with a different hash: {path}"
            )
        return "ALREADY_FROZEN_IDENTICAL", expected_sha

    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with open(temporary, "w") as file:
        file.write(text)
    if sha256_file(temporary) != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Transform source creation verification failed.")
    os.replace(temporary, path)
    return "CREATED_AND_FROZEN", expected_sha


def copy_once_verified(source, destination, expected_sha):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_sha:
            raise RuntimeError(
                f"Frozen source exists with a different hash: {destination}"
            )
        return "ALREADY_FROZEN_IDENTICAL"

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Deployment source freeze verification failed.")
    os.replace(temporary, destination)
    return "COPIED_AND_FROZEN"


def all_evidence_gates_pass(payload):
    gates = payload.get("gates", {})
    return (
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and len(gates) > 0
        and all(bool(value) for value in gates.values())
    )


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def apply_grade_temperature(probabilities, temperature):
    """Apply the frozen scalar temperature to final class probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("Temperature must be finite and positive.")
    log_probabilities = np.log(np.clip(probabilities, EPSILON, 1.0))
    return softmax(log_probabilities / float(temperature))


def apply_lesion_temperature(logits, temperature):
    """Apply the frozen scalar temperature to binary lesion logits."""
    logits = np.asarray(logits, dtype=np.float64)
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("Temperature must be finite and positive.")
    return sigmoid(logits / float(temperature))


def multiclass_nll(probabilities, targets):
    selected = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(selected, EPSILON, 1.0)).mean())


def binary_nll(probabilities, targets):
    probabilities = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return float(
        -(
            targets * np.log(probabilities)
            + (1.0 - targets) * np.log(1.0 - probabilities)
        ).mean()
    )


print("=" * 80)
print("STAGE 0G-4D-5D — CALIBRATED DEPLOYMENT PACKAGE FREEZE")
print("=" * 80)


# =============================================================================
# 1. CALIBRATION OUTPUT VERIFICATION
# =============================================================================

print("\n[1] CALIBRATION OUTPUT VERIFICATION")
print("-" * 80)

paths = {
    "frozen_checkpoint": FROZEN_CHECKPOINT_PATH,
    "production_source": PRODUCTION_SOURCE_PATH,
    "calibration_protocol": CALIBRATION_PROTOCOL_PATH,
    "temperature_parameters": TEMPERATURES_PATH,
    "calibration_manifest": CALIBRATION_MANIFEST_PATH,
    "raw_predictions": RAW_PREDICTIONS_PATH,
    "calibrated_predictions": CALIBRATED_PREDICTIONS_PATH,
    "calibration_runner": CALIBRATION_RUNNER_PATH,
    "metrics": METRICS_PATH,
    "inference_state": INFERENCE_STATE_PATH,
    "calibration_evidence": CALIBRATION_EVIDENCE_PATH,
}

for name, path in paths.items():
    print(f"{name:27s}: exists={path.exists()}")

if not all(path.exists() for path in paths.values()):
    raise FileNotFoundError("One or more calibrated-package inputs are missing.")

actual_hashes = {
    name: sha256_file(paths[name])
    for name in EXPECTED_HASHES
}
hash_gates = {
    f"{name}_hash_verified": actual_hashes[name] == expected
    for name, expected in EXPECTED_HASHES.items()
}

for name, passed in hash_gates.items():
    print(f"{name:51s}: {passed}")

if not all(hash_gates.values()):
    raise RuntimeError("A calibrated-package input hash changed.")

temperatures = load_json(TEMPERATURES_PATH)
inference_state = load_json(INFERENCE_STATE_PATH)
calibration_evidence = load_json(CALIBRATION_EVIDENCE_PATH)

metrics_hash = sha256_file(METRICS_PATH)
calibration_evidence_ok = (
    all_evidence_gates_pass(calibration_evidence)
    and calibration_evidence.get("session_status")
    == "COMPLETE_AND_TEMPERATURE_FITTED"
    and calibration_evidence.get("execution_backend") == "CPU_FLOAT32"
    and int(calibration_evidence.get("completed_rows", -1)) == EXPECTED_ROWS
    and calibration_evidence.get("model_parameters_updated") is False
    and calibration_evidence.get("validation_used_for_calibration_fit") is False
    and calibration_evidence.get("locked_test_accessed") is False
)
inference_state_ok = (
    inference_state.get("status") == "COMPLETE_AND_TEMPERATURE_FITTED"
    and inference_state.get("execution_backend") == "CPU_FLOAT32"
    and inference_state.get("run_signature") == EXPECTED_RUN_SIGNATURE
    and int(inference_state.get("completed_rows", -1)) == EXPECTED_ROWS
    and inference_state.get("locked_test_accessed") is False
)
temperature_provenance_ok = (
    temperatures.get("status") == "FITTED_ON_CALIBRATION_ONLY"
    and temperatures.get("execution_backend") == "CPU_FLOAT32"
    and temperatures.get("run_signature") == EXPECTED_RUN_SIGNATURE
    and temperatures.get("selected_checkpoint_sha256")
    == EXPECTED_HASHES["frozen_checkpoint"]
    and temperatures.get("calibration_protocol_sha256")
    == EXPECTED_HASHES["calibration_protocol"]
    and temperatures.get("calibration_manifest_sha256")
    == EXPECTED_HASHES["calibration_manifest"]
    and temperatures.get("raw_predictions_sha256")
    == EXPECTED_HASHES["raw_predictions"]
    and temperatures.get("calibrated_predictions_sha256")
    == EXPECTED_HASHES["calibrated_predictions"]
    and temperatures.get("metrics_sha256") == metrics_hash
    and temperatures.get("locked_test_accessed") is False
)

print("\nCalibration evidence PASS :", calibration_evidence_ok)
print("Inference state COMPLETE  :", inference_state_ok)
print("Temperature provenance OK :", temperature_provenance_ok)

if not all([
    calibration_evidence_ok,
    inference_state_ok,
    temperature_provenance_ok,
]):
    raise RuntimeError("Calibration completion/provenance verification failed.")


# =============================================================================
# 2. TEMPERATURE STRUCTURE AND RANGE
# =============================================================================

print("\n[2] TEMPERATURE STRUCTURE AND RANGE")
print("-" * 80)

grade_parameters = temperatures.get("grade_temperatures", {})
lesion_parameters = temperatures.get("lesion_temperatures", {})

grade_units_exact = set(grade_parameters) == {"CFP", "UWF", "OCT"}
lesion_units_exact = set(lesion_parameters) == set(LESION_COLUMNS)

all_temperature_values = [
    float(value["temperature"])
    for value in grade_parameters.values()
] + [
    float(value["temperature"])
    for value in lesion_parameters.values()
]
temperatures_valid = all(
    math.isfinite(value)
    and TEMPERATURE_MIN <= value <= TEMPERATURE_MAX
    for value in all_temperature_values
)

print("Grade temperature units:")
for unit in ("CFP", "UWF", "OCT"):
    if unit in grade_parameters:
        print(f"  {unit:3s}: {float(grade_parameters[unit]['temperature']):.9f}")

print("\nLesion temperature units:")
for lesion in LESION_COLUMNS:
    if lesion in lesion_parameters:
        print(f"  {lesion:36s}: {float(lesion_parameters[lesion]['temperature']):.9f}")

print("\nGrade units exact  :", grade_units_exact)
print("Lesion units exact :", lesion_units_exact)
print("All values in range:", temperatures_valid)


# =============================================================================
# 3. INDEPENDENT TRANSFORMATION REPLAY
# =============================================================================

print("\n[3] INDEPENDENT TRANSFORMATION REPLAY")
print("-" * 80)

raw = pd.read_csv(RAW_PREDICTIONS_PATH)
calibrated = pd.read_csv(CALIBRATED_PREDICTIONS_PATH)
metrics = pd.read_csv(METRICS_PATH)

rows_and_order_match = (
    len(raw) == EXPECTED_ROWS
    and len(calibrated) == EXPECTED_ROWS
    and raw["calibration_index"].astype(int).tolist() == list(range(EXPECTED_ROWS))
    and calibrated["calibration_index"].astype(int).tolist()
    == list(range(EXPECTED_ROWS))
    and raw["relative_path"].astype(str).tolist()
    == calibrated["relative_path"].astype(str).tolist()
    and raw["image_sha256"].astype(str).tolist()
    == calibrated["image_sha256"].astype(str).tolist()
)

modality_counts = {
    str(key): int(value)
    for key, value in raw["modality"].value_counts().to_dict().items()
}

grade_replay_exact = True
grade_decisions_invariant = True
grade_nll_nonworsening = True
replay_records = []

for modality, classes in (("CFP", 5), ("UWF", 5), ("OCT", 3)):
    mask = raw["modality"].eq(modality).to_numpy()
    raw_columns = [f"raw_probability_{index}" for index in range(classes)]
    calibrated_columns = [
        f"calibrated_probability_{index}"
        for index in range(classes)
    ]
    raw_probability = raw.loc[mask, raw_columns].to_numpy(dtype=float)
    stored_calibrated = calibrated.loc[
        mask, calibrated_columns
    ].to_numpy(dtype=float)
    targets = raw.loc[mask, "target"].to_numpy(dtype=int)
    temperature = float(grade_parameters[modality]["temperature"])
    replayed = apply_grade_temperature(raw_probability, temperature)

    replay_match = bool(
        np.allclose(replayed, stored_calibrated, rtol=0.0, atol=5e-12)
    )
    invariant = bool(
        np.array_equal(raw_probability.argmax(axis=1), replayed.argmax(axis=1))
    )
    raw_nll = multiclass_nll(raw_probability, targets)
    calibrated_nll = multiclass_nll(replayed, targets)
    nonworsening = calibrated_nll <= raw_nll + 1e-12

    grade_replay_exact = grade_replay_exact and replay_match
    grade_decisions_invariant = grade_decisions_invariant and invariant
    grade_nll_nonworsening = grade_nll_nonworsening and nonworsening
    replay_records.append({
        "task": "GRADE",
        "unit": modality,
        "temperature": temperature,
        "raw_NLL": raw_nll,
        "calibrated_NLL": calibrated_nll,
        "replay_matches_saved": replay_match,
        "decision_invariant": invariant,
    })
    print(
        f"{modality:3s} | replay={replay_match} | "
        f"decision={invariant} | NLL={raw_nll:.6f}->{calibrated_nll:.6f}"
    )

fundus_mask = raw["modality"].isin(["CFP", "UWF"]).to_numpy()
valid_mask = fundus_mask & (raw["lesion_loss_mask"].to_numpy(dtype=float) > 0.5)

lesion_replay_exact = True
lesion_decisions_invariant = True
lesion_nll_nonworsening = True

for lesion in LESION_COLUMNS:
    logits = raw.loc[valid_mask, f"raw_{lesion}_logit"].to_numpy(dtype=float)
    raw_probability = raw.loc[
        valid_mask, f"raw_{lesion}_probability"
    ].to_numpy(dtype=float)
    stored_calibrated = calibrated.loc[
        valid_mask, f"calibrated_{lesion}_probability"
    ].to_numpy(dtype=float)
    targets = raw.loc[valid_mask, f"{lesion}_target"].to_numpy(dtype=int)
    temperature = float(lesion_parameters[lesion]["temperature"])
    replayed = apply_lesion_temperature(logits, temperature)

    replay_match = bool(
        np.allclose(replayed, stored_calibrated, rtol=0.0, atol=5e-12)
    )
    invariant = bool(
        np.array_equal(raw_probability >= 0.5, replayed >= 0.5)
    )
    raw_nll = binary_nll(raw_probability, targets)
    calibrated_nll = binary_nll(replayed, targets)
    nonworsening = calibrated_nll <= raw_nll + 1e-12

    lesion_replay_exact = lesion_replay_exact and replay_match
    lesion_decisions_invariant = lesion_decisions_invariant and invariant
    lesion_nll_nonworsening = lesion_nll_nonworsening and nonworsening
    replay_records.append({
        "task": "LESION",
        "unit": lesion,
        "temperature": temperature,
        "raw_NLL": raw_nll,
        "calibrated_NLL": calibrated_nll,
        "replay_matches_saved": replay_match,
        "decision_invariant": invariant,
    })
    print(
        f"{lesion:36s} | replay={replay_match} | "
        f"decision={invariant} | NLL={raw_nll:.6f}->{calibrated_nll:.6f}"
    )

metrics_structure_ok = (
    len(metrics) == 10
    and int((metrics["task_type"] == "GRADE").sum()) == 3
    and int((metrics["task_type"] == "LESION").sum()) == 7
    and bool(metrics["NLL_nonworsening"].all())
    and bool(metrics["decision_invariant"].all())
)

print("\nRows and order match       :", rows_and_order_match)
print("Modality counts            :", modality_counts)
print("Grade replay matches       :", grade_replay_exact)
print("Lesion replay matches      :", lesion_replay_exact)
print("Grade decisions invariant  :", grade_decisions_invariant)
print("Lesion decisions invariant :", lesion_decisions_invariant)
print("All NLL values nonworsening:", grade_nll_nonworsening and lesion_nll_nonworsening)
print("Metrics artifact valid     :", metrics_structure_ok)


# =============================================================================
# 4. DEPLOYMENT PACKAGE GATES
# =============================================================================

gates = {
    **hash_gates,
    "calibration_completion_evidence_passed": calibration_evidence_ok,
    "inference_state_complete": inference_state_ok,
    "temperature_provenance_verified": temperature_provenance_ok,
    "exactly_three_grade_temperature_units": grade_units_exact,
    "exactly_seven_lesion_temperature_units": lesion_units_exact,
    "all_temperature_values_finite_and_in_range": temperatures_valid,
    "raw_and_calibrated_rows_and_order_match": rows_and_order_match,
    "CALIBRATION_modality_counts_exact": modality_counts == EXPECTED_MODALITY_COUNTS,
    "grade_transform_replay_matches_saved_probabilities": grade_replay_exact,
    "lesion_transform_replay_matches_saved_probabilities": lesion_replay_exact,
    "grade_argmax_decisions_invariant": grade_decisions_invariant,
    "lesion_threshold_decisions_invariant": lesion_decisions_invariant,
    "all_calibrated_NLL_values_nonworsening": (
        grade_nll_nonworsening and lesion_nll_nonworsening
    ),
    "metrics_artifact_structure_verified": metrics_structure_ok,
    "model_parameters_not_updated": calibration_evidence.get("model_parameters_updated") is False,
    "VALIDATION_not_used_for_calibration_fit": calibration_evidence.get("validation_used_for_calibration_fit") is False,
    "locked_test_not_accessed": calibration_evidence.get("locked_test_accessed") is False,
}

print("\n" + "=" * 80)
print("CALIBRATED DEPLOYMENT PACKAGE GATES")
print("=" * 80)
for name, passed in gates.items():
    print(f"{name:70s}: {passed}")

if not all(gates.values()):
    raise RuntimeError("Calibrated deployment package gate failed.")


# =============================================================================
# 5. FREEZE SOURCE, PACKAGE MANIFEST, AND EVIDENCE
# =============================================================================

source_path = Path(__file__).resolve()
source_sha = sha256_file(source_path)
source_action = copy_once_verified(
    source_path,
    FROZEN_DEPLOYMENT_SOURCE_PATH,
    source_sha,
)
frozen_source_sha = sha256_file(FROZEN_DEPLOYMENT_SOURCE_PATH)
source_frozen_ok = frozen_source_sha == source_sha

if not source_frozen_ok:
    raise RuntimeError("Calibrated deployment source freeze failed.")

transform_action, transform_source_sha = create_text_once_verified(
    CALIBRATED_TRANSFORM_SOURCE,
    CALIBRATED_TRANSFORM_SOURCE_PATH,
)
transform_source_ok = (
    CALIBRATED_TRANSFORM_SOURCE_PATH.exists()
    and sha256_file(CALIBRATED_TRANSFORM_SOURCE_PATH) == transform_source_sha
)

if not transform_source_ok:
    raise RuntimeError("Calibrated probability-transform source freeze failed.")

package_manifest = {
    "package": "FOVEA_DR_CALIBRATED_DEPLOYMENT_PACKAGE_V1",
    "status": "FROZEN_BEFORE_LOCKED_TEST",
    "model": {
        "architecture": "FOVEA_DR_V1_1_EFFICIENTNETV2_S",
        "selected_training_epoch": 10,
        "checkpoint": str(FROZEN_CHECKPOINT_PATH),
        "checkpoint_sha256": EXPECTED_HASHES["frozen_checkpoint"],
        "production_source": str(PRODUCTION_SOURCE_PATH),
        "production_source_sha256": EXPECTED_HASHES["production_source"],
    },
    "input_contract": {
        "shape": "512x512_RGB",
        "resize_interpolation": "BILINEAR_ANTIALIAS",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "inference_precision": "CPU_FLOAT32",
    },
    "calibration": {
        "partition": "CALIBRATION_ONLY",
        "rows": EXPECTED_ROWS,
        "protocol": str(CALIBRATION_PROTOCOL_PATH),
        "protocol_sha256": EXPECTED_HASHES["calibration_protocol"],
        "parameters": str(TEMPERATURES_PATH),
        "parameters_sha256": EXPECTED_HASHES["temperature_parameters"],
        "grade_decision_rule": "ARGMAX_AFTER_POSITIVE_SCALAR_TEMPERATURE",
        "lesion_decision_rule": "PROBABILITY_AT_LEAST_0_5_AFTER_POSITIVE_SCALAR_TEMPERATURE",
    },
    "calibrated_transform_source": {
        "path": str(CALIBRATED_TRANSFORM_SOURCE_PATH),
        "sha256": transform_source_sha,
        "grade_function": "apply_grade_temperature",
        "lesion_function": "apply_lesion_temperature",
    },
    "calibration_evidence": {
        "path": str(CALIBRATION_EVIDENCE_PATH),
        "sha256": sha256_file(CALIBRATION_EVIDENCE_PATH),
        "raw_predictions_sha256": EXPECTED_HASHES["raw_predictions"],
        "calibrated_predictions_sha256": EXPECTED_HASHES["calibrated_predictions"],
        "metrics_sha256": metrics_hash,
    },
    "prohibited_after_freeze": [
        "MODEL_WEIGHT_UPDATE",
        "CHECKPOINT_RESELECTION",
        "TEMPERATURE_REFIT",
        "DECISION_THRESHOLD_CHANGE",
        "PREPROCESSING_CHANGE",
        "ARCHITECTURE_CHANGE",
    ],
    "validation_role": "MODEL_SELECTION_ONLY_ALREADY_COMPLETE",
    "calibration_role": "TEMPERATURE_FIT_ONLY_ALREADY_COMPLETE",
    "locked_test_accessed": False,
    "next_required_stage": "LOCKED_TEST_SINGLE_EVALUATION_READINESS_AUDIT",
}
atomic_json_save(package_manifest, PACKAGE_MANIFEST_PATH)
package_manifest_sha = sha256_file(PACKAGE_MANIFEST_PATH)

final_gates = {
    **gates,
    "deployment_freeze_audit_source_frozen_and_verified": source_frozen_ok,
    "calibrated_probability_transform_source_frozen_and_verified": transform_source_ok,
    "deployment_package_manifest_created": PACKAGE_MANIFEST_PATH.exists(),
}

evidence = {
    "stage": "0G-4D-5D",
    "status": "PASS",
    "purpose": "CALIBRATED_DEPLOYMENT_PACKAGE_FREEZE_BEFORE_LOCKED_TEST",
    "selected_model_epoch": 10,
    "selected_model_sha256": EXPECTED_HASHES["frozen_checkpoint"],
    "temperature_parameters_sha256": EXPECTED_HASHES["temperature_parameters"],
    "calibration_rows": EXPECTED_ROWS,
    "transformation_replay": replay_records,
    "freeze_audit_source": str(FROZEN_DEPLOYMENT_SOURCE_PATH),
    "freeze_audit_source_sha256": frozen_source_sha,
    "freeze_audit_source_action": source_action,
    "calibrated_probability_transform_source": str(CALIBRATED_TRANSFORM_SOURCE_PATH),
    "calibrated_probability_transform_source_sha256": transform_source_sha,
    "calibrated_probability_transform_source_action": transform_action,
    "deployment_package_manifest": str(PACKAGE_MANIFEST_PATH),
    "deployment_package_manifest_sha256": package_manifest_sha,
    "gates": {name: bool(value) for name, value in final_gates.items()},
    "medical_images_accessed": False,
    "model_loaded": False,
    "optimizer_created": False,
    "model_parameters_updated": False,
    "temperatures_refitted": False,
    "validation_accessed": False,
    "calibration_predictions_read_for_audit": True,
    "locked_test_accessed": False,
    "next_required_stage": "LOCKED_TEST_SINGLE_EVALUATION_READINESS_AUDIT",
}
atomic_json_save(evidence, EVIDENCE_PATH)

print("\nFrozen calibrated transform source:")
print(FROZEN_DEPLOYMENT_SOURCE_PATH)
print("SHA-256:", frozen_source_sha)
print("Source action:", source_action)

print("\nFrozen calibrated probability-transform source:")
print(CALIBRATED_TRANSFORM_SOURCE_PATH)
print("SHA-256:", transform_source_sha)
print("Transform action:", transform_action)

print("\nCalibrated deployment package manifest:")
print(PACKAGE_MANIFEST_PATH)
print("SHA-256:", package_manifest_sha)

print("\nEvidence:")
print(EVIDENCE_PATH)

print("\nImages accessed         : NO")
print("Model loaded            : NO")
print("Parameters updated      : NO")
print("Temperatures refitted   : NO")
print("VALIDATION accessed     : NO")
print("Locked test accessed    : NO")

print("\nNEXT REQUIRED STAGE:")
print("Locked-test single-evaluation readiness audit — without opening test images.")

print("\n" + "=" * 80)
print("STAGE 0G-4D-5D STATUS: PASS")
print("=" * 80)


if __name__ == "__main__":
    pass
