from pathlib import Path
import hashlib
import json
import math
import os
import shutil

import pandas as pd


# =============================================================================
# STAGE 0G-4D-5B — POST-SELECTION CALIBRATION PROTOCOL LOCK
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

SPLIT_PATH = PROJECT_ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
CALIBRATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Audit/mmrdr_fovea_dr_CALIBRATION_local_cache_manifest_v1.csv"
)
CALIBRATION_CACHE_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5a_calibration_only_cache_preparation.json"
)
BASELINE_FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_baseline_seed2026_selection_freeze_v1.json"
)
BASELINE_FREEZE_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d4c_baseline_completion_and_selection_freeze.json"
)
FROZEN_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "Checkpoints/FOVEA_DR/baseline_seed2026/selected_epoch10_frozen.pt"
)
RUN_STATE_PATH = (
    PROJECT_ROOT / "Runs/FOVEA_DR/baseline_seed2026/run_state.json"
)

GRADE_COUNTS_PATH = (
    PROJECT_ROOT
    / "Audit/mmrdr_fovea_dr_CALIBRATION_modality_grade_counts_v1.csv"
)
LESION_COUNTS_PATH = (
    PROJECT_ROOT
    / "Audit/mmrdr_fovea_dr_CALIBRATION_lesion_counts_v1.csv"
)
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_post_selection_calibration_protocol_v1.json"
)
FROZEN_SOURCE_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/stage0g4d5b_calibration_protocol_lock.py"
)
EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5b_calibration_protocol_lock.json"
)

EXPECTED_SPLIT_SHA = (
    "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140"
)
EXPECTED_CALIBRATION_MANIFEST_SHA = (
    "f8e6a62b96e330a1b21bb9ed56897cbf04b730987cf9e0c31141927abbb84655"
)
EXPECTED_BASELINE_FREEZE_MANIFEST_SHA = (
    "505e7cb52155e0c8656a81c57754e890dff81d4b53e9ada8920de999fd11239d"
)
EXPECTED_FROZEN_CHECKPOINT_SHA = (
    "89f5e87b647680a9094c55bf647dc1dd2abb624681e733ffb453e30e4c11ff1f"
)

EXPECTED_CALIBRATION_ROWS = 2840
EXPECTED_MODALITY_COUNTS = {"CFP": 1313, "UWF": 1171, "OCT": 356}
EXPECTED_GRADE_CLASSES = {"CFP": {0, 1, 2, 3, 4}, "UWF": {0, 1, 2, 3, 4}, "OCT": {0, 1, 2}}

LESION_COLUMNS = [
    "lesion_microaneurysm",
    "lesion_hard_exudate",
    "lesion_intraretinal_hemorrhage",
    "lesion_VB_IRMA",
    "lesion_neovascularization",
    "lesion_vitreous_hemorrhage",
    "lesion_retinal_detachment",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(path) as file:
        return json.load(file)


def all_evidence_gates_pass(evidence):
    gates = evidence.get("gates", {})
    return (
        evidence.get("status") == "PASS"
        and isinstance(gates, dict)
        and len(gates) > 0
        and all(bool(value) for value in gates.values())
    )


def atomic_json_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary, path)


def atomic_csv_save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


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
        raise RuntimeError("Source freeze verification failed.")
    os.replace(temporary, destination)
    return "COPIED_AND_FROZEN"


print("=" * 80)
print("STAGE 0G-4D-5B — POST-SELECTION CALIBRATION PROTOCOL LOCK")
print("=" * 80)


# =============================================================================
# 1. PREREQUISITE VERIFICATION
# =============================================================================

print("\n[1] PREREQUISITE VERIFICATION")
print("-" * 80)

required = {
    "split": SPLIT_PATH,
    "CALIBRATION_manifest": CALIBRATION_MANIFEST_PATH,
    "CALIBRATION_cache_evidence": CALIBRATION_CACHE_EVIDENCE_PATH,
    "baseline_freeze_manifest": BASELINE_FREEZE_MANIFEST_PATH,
    "baseline_freeze_evidence": BASELINE_FREEZE_EVIDENCE_PATH,
    "frozen_epoch10_checkpoint": FROZEN_CHECKPOINT_PATH,
    "run_state": RUN_STATE_PATH,
}

for name, path in required.items():
    print(f"{name:31s}: exists={path.exists()}")

if not all(path.exists() for path in required.values()):
    raise FileNotFoundError("One or more calibration-protocol inputs are missing.")

split_hash_ok = sha256_file(SPLIT_PATH) == EXPECTED_SPLIT_SHA
calibration_manifest_hash_ok = (
    sha256_file(CALIBRATION_MANIFEST_PATH)
    == EXPECTED_CALIBRATION_MANIFEST_SHA
)
baseline_freeze_manifest_hash_ok = (
    sha256_file(BASELINE_FREEZE_MANIFEST_PATH)
    == EXPECTED_BASELINE_FREEZE_MANIFEST_SHA
)
frozen_checkpoint_hash_ok = (
    sha256_file(FROZEN_CHECKPOINT_PATH)
    == EXPECTED_FROZEN_CHECKPOINT_SHA
)

calibration_cache_evidence = load_json(CALIBRATION_CACHE_EVIDENCE_PATH)
baseline_freeze_evidence = load_json(BASELINE_FREEZE_EVIDENCE_PATH)
baseline_freeze_manifest = load_json(BASELINE_FREEZE_MANIFEST_PATH)
run_state = load_json(RUN_STATE_PATH)

calibration_cache_pass = (
    all_evidence_gates_pass(calibration_cache_evidence)
    and calibration_cache_evidence.get("locked_test_accessed") is False
)
baseline_freeze_pass = (
    all_evidence_gates_pass(baseline_freeze_evidence)
    and baseline_freeze_manifest.get("status") == "FROZEN"
    and int(baseline_freeze_manifest.get("selected_epoch", -1)) == 10
    and baseline_freeze_manifest.get("calibration_used_for_selection") is False
)
baseline_run_complete = (
    run_state.get("status") == "COMPLETE"
    and int(run_state.get("last_completed_epoch", -1)) == 16
    and int(run_state.get("best_epoch", -1)) == 10
    and run_state.get("early_stopped") is True
)

print("Split hash verified             :", split_hash_ok)
print("CALIBRATION manifest verified   :", calibration_manifest_hash_ok)
print("Baseline freeze manifest verified:", baseline_freeze_manifest_hash_ok)
print("Frozen checkpoint verified      :", frozen_checkpoint_hash_ok)
print("CALIBRATION cache evidence PASS :", calibration_cache_pass)
print("Baseline freeze evidence PASS   :", baseline_freeze_pass)
print("Baseline run COMPLETE           :", baseline_run_complete)

if not all([
    split_hash_ok,
    calibration_manifest_hash_ok,
    baseline_freeze_manifest_hash_ok,
    frozen_checkpoint_hash_ok,
    calibration_cache_pass,
    baseline_freeze_pass,
    baseline_run_complete,
]):
    raise RuntimeError("Calibration-protocol prerequisite verification failed.")


# =============================================================================
# 2. CALIBRATION COHORT AND LABEL SUPPORT
# =============================================================================

print("\n[2] CALIBRATION COHORT AND LABEL SUPPORT")
print("-" * 80)

split = pd.read_csv(SPLIT_PATH)
calibration = split[
    split["development_split"] == "CALIBRATION"
].copy()
calibration["relative_path"] = (
    calibration["relative_path"].astype(str).str.replace("\\", "/", regex=False)
)

manifest = pd.read_csv(CALIBRATION_MANIFEST_PATH)
manifest["relative_path"] = (
    manifest["relative_path"].astype(str).str.replace("\\", "/", regex=False)
)

manifest_small = manifest[
    ["relative_path", "local_path", "file_size_bytes", "image_sha256"]
].copy()
table = calibration.merge(
    manifest_small,
    on="relative_path",
    how="inner",
    validate="one_to_one",
)
table = table.sort_values("relative_path").reset_index(drop=True)

modality_counts = {
    str(key): int(value)
    for key, value in table["modality"].value_counts().to_dict().items()
}

grade_counts = (
    table.groupby(["modality", "grade"])
    .size()
    .rename("count")
    .reset_index()
    .sort_values(["modality", "grade"])
    .reset_index(drop=True)
)
grade_counts["within_modality_fraction"] = grade_counts.apply(
    lambda row: row["count"] / modality_counts[str(row["modality"])],
    axis=1,
)

grade_support_exact = all(
    set(
        table.loc[table["modality"] == modality, "grade"]
        .astype(int)
        .unique()
    ) == classes
    for modality, classes in EXPECTED_GRADE_CLASSES.items()
)

fundus = table[table["modality"].isin(["CFP", "UWF"])].copy()
lesion_records = []

for lesion in LESION_COLUMNS:
    if lesion not in fundus.columns:
        raise KeyError(lesion)

    valid = fundus[
        fundus["lesion_loss_mask"].astype(float) > 0.5
    ][lesion].dropna().astype(int)

    positive = int((valid == 1).sum())
    negative = int((valid == 0).sum())
    lesion_records.append({
        "lesion": lesion,
        "valid_rows": int(len(valid)),
        "positive_count": positive,
        "negative_count": negative,
        "prevalence": float(positive / len(valid)) if len(valid) else math.nan,
    })

lesion_counts = pd.DataFrame(lesion_records)
lesion_support_adequate = bool(
    (lesion_counts["valid_rows"] > 0).all()
    and (lesion_counts["positive_count"] >= 10).all()
    and (lesion_counts["negative_count"] >= 10).all()
)

print("CALIBRATION rows       :", len(table))
print("Modality counts        :", modality_counts)
print("All grade classes seen :", grade_support_exact)

print("\nModality × grade counts:")
print(grade_counts.to_string(index=False))

print("\nLesion label support:")
print(lesion_counts.to_string(index=False))
print("At least 10 positives and negatives per lesion:", lesion_support_adequate)


# =============================================================================
# 3. LOCK POST-SELECTION CALIBRATION CONTRACT
# =============================================================================

print("\n[3] POST-SELECTION CALIBRATION CONTRACT")
print("-" * 80)

protocol = {
    "protocol": "FOVEA_DR_POST_SELECTION_CALIBRATION_V1",
    "status": "LOCKED_BEFORE_CALIBRATION_INFERENCE",
    "selected_model": {
        "checkpoint": str(FROZEN_CHECKPOINT_PATH),
        "checkpoint_sha256": EXPECTED_FROZEN_CHECKPOINT_SHA,
        "selected_epoch": 10,
        "selection_partition": "VALIDATION",
    },
    "calibration_partition": {
        "role": "CALIBRATION",
        "rows": EXPECTED_CALIBRATION_ROWS,
        "manifest": str(CALIBRATION_MANIFEST_PATH),
        "manifest_sha256": EXPECTED_CALIBRATION_MANIFEST_SHA,
        "modality_counts": modality_counts,
        "may_update_model_parameters": False,
        "may_select_model_checkpoint": False,
        "may_change_architecture": False,
    },
    "grade_temperature_scaling": {
        "input": "LOG_CLAMPED_FINAL_CLASS_PROBABILITIES",
        "transformation": "softmax(log(clamp(p,1e-12,1))/T)",
        "temperature_units": [
            "CFP_FINAL_FUSED_GRADE",
            "UWF_FINAL_FUSED_GRADE",
            "OCT_DME_GRADE",
        ],
        "one_positive_scalar_temperature_per_unit": True,
        "objective": "MINIMIZE_MULTICLASS_NLL_ON_CALIBRATION_ONLY",
        "optimizer": "SCIPY_MINIMIZE_SCALAR_BOUNDED_ON_LOG_T",
        "log_temperature_bounds": [float(math.log(0.05)), float(math.log(20.0))],
        "temperature_bounds": [0.05, 20.0],
        "xatol": 1e-6,
        "maximum_iterations": 500,
        "identity_candidate_T": 1.0,
        "fallback_rule": "USE_T_1_IF_OPTIMIZED_NLL_IS_NOT_STRICTLY_LOWER",
        "decision_rule": "ARGMAX_UNCHANGED_BY_POSITIVE_SCALAR_TEMPERATURE",
    },
    "lesion_temperature_scaling": {
        "input": "RAW_LESION_PRESENCE_LOGITS",
        "transformation": "sigmoid(logit/T)",
        "temperature_units": LESION_COLUMNS,
        "pool_modalities": ["CFP", "UWF"],
        "one_positive_scalar_temperature_per_lesion": True,
        "validity_mask": "lesion_loss_mask",
        "objective": "MINIMIZE_BINARY_NLL_ON_VALID_CALIBRATION_ROWS_ONLY",
        "optimizer": "SCIPY_MINIMIZE_SCALAR_BOUNDED_ON_LOG_T",
        "log_temperature_bounds": [float(math.log(0.05)), float(math.log(20.0))],
        "temperature_bounds": [0.05, 20.0],
        "xatol": 1e-6,
        "maximum_iterations": 500,
        "identity_candidate_T": 1.0,
        "fallback_rule": "USE_T_1_IF_OPTIMIZED_NLL_IS_NOT_STRICTLY_LOWER",
        "decision_threshold": 0.5,
        "threshold_invariance": "PRESERVED_BY_POSITIVE_SCALAR_TEMPERATURE",
    },
    "reporting": {
        "metrics_before_and_after": [
            "NLL",
            "BRIER_SCORE",
            "ECE_15_EQUAL_WIDTH_BINS",
        ],
        "grade_discrimination_metrics_for_invariance_audit": [
            "ACCURACY",
            "MACRO_F1",
            "QUADRATIC_WEIGHTED_KAPPA",
        ],
        "lesion_discrimination_metrics_for_invariance_audit": [
            "ACCURACY_AT_0_5",
            "F1_AT_0_5",
        ],
        "probability_epsilon": 1e-12,
        "ece_bins": 15,
    },
    "acceptance_rules": {
        "all_temperatures_finite_and_within_bounds": True,
        "calibrated_NLL_must_not_exceed_identity_NLL": True,
        "grade_argmax_predictions_must_be_unchanged": True,
        "lesion_predictions_at_0_5_must_be_unchanged": True,
        "all_saved_probabilities_finite_bounded_and_normalized": True,
    },
    "prohibited_actions": [
        "MODEL_WEIGHT_UPDATE",
        "CHECKPOINT_RESELECTION",
        "ARCHITECTURE_CHANGE",
        "VALIDATION_REUSE_FOR_CALIBRATION_FIT",
        "LOCKED_TEST_ACCESS",
        "POST_HOC_CHANGE_OF_CALIBRATION_METHOD",
        "DECISION_THRESHOLD_OPTIMIZATION_IN_THIS_STAGE",
    ],
    "locked_test_policy": "UNTOUCHED_UNTIL_FINAL_SINGLE_EVALUATION",
    "next_required_stage": "RESUMABLE_CALIBRATION_ONLY_INFERENCE_AND_TEMPERATURE_FIT",
}


# =============================================================================
# 4. PROTOCOL-LOCK GATES
# =============================================================================

gates = {
    "frozen_split_hash_verified": split_hash_ok,
    "CALIBRATION_manifest_hash_verified": calibration_manifest_hash_ok,
    "CALIBRATION_cache_evidence_passed": calibration_cache_pass,
    "baseline_freeze_manifest_hash_verified": baseline_freeze_manifest_hash_ok,
    "baseline_freeze_evidence_passed": baseline_freeze_pass,
    "frozen_epoch10_checkpoint_hash_verified": frozen_checkpoint_hash_ok,
    "baseline_run_complete_before_calibration": baseline_run_complete,
    "exactly_2840_CALIBRATION_rows_joined": len(table) == EXPECTED_CALIBRATION_ROWS,
    "CALIBRATION_modality_counts_exact": modality_counts == EXPECTED_MODALITY_COUNTS,
    "all_required_grade_classes_represented": grade_support_exact,
    "all_seven_lesion_columns_present": all(column in table.columns for column in LESION_COLUMNS),
    "each_lesion_has_at_least_10_positive_and_negative_examples": lesion_support_adequate,
    "three_grade_temperature_units_locked": len(protocol["grade_temperature_scaling"]["temperature_units"]) == 3,
    "seven_lesion_temperature_units_locked": len(protocol["lesion_temperature_scaling"]["temperature_units"]) == 7,
    "temperature_bounds_locked": protocol["grade_temperature_scaling"]["temperature_bounds"] == [0.05, 20.0],
    "identity_fallback_rule_locked": protocol["grade_temperature_scaling"]["identity_candidate_T"] == 1.0,
    "model_weight_updates_prohibited": "MODEL_WEIGHT_UPDATE" in protocol["prohibited_actions"],
    "checkpoint_reselection_prohibited": "CHECKPOINT_RESELECTION" in protocol["prohibited_actions"],
    "decision_threshold_optimization_deferred": "DECISION_THRESHOLD_OPTIMIZATION_IN_THIS_STAGE" in protocol["prohibited_actions"],
    "locked_test_access_prohibited": "LOCKED_TEST_ACCESS" in protocol["prohibited_actions"],
}

print("\n" + "=" * 80)
print("CALIBRATION-PROTOCOL LOCK GATES")
print("=" * 80)
for name, passed in gates.items():
    print(f"{name:70s}: {passed}")

if not all(gates.values()):
    raise RuntimeError("STAGE 0G-4D-5B protocol-lock gate failed.")


# =============================================================================
# 5. SAVE LOCKED ARTIFACTS AND EVIDENCE
# =============================================================================

atomic_csv_save(grade_counts, GRADE_COUNTS_PATH)
atomic_csv_save(lesion_counts, LESION_COUNTS_PATH)

protocol["support_artifacts"] = {
    "grade_counts": str(GRADE_COUNTS_PATH),
    "grade_counts_sha256": sha256_file(GRADE_COUNTS_PATH),
    "lesion_counts": str(LESION_COUNTS_PATH),
    "lesion_counts_sha256": sha256_file(LESION_COUNTS_PATH),
}
atomic_json_save(protocol, PROTOCOL_PATH)
protocol_sha = sha256_file(PROTOCOL_PATH)

source_path = Path(__file__).resolve()
source_sha = sha256_file(source_path)
source_action = copy_once_verified(
    source_path,
    FROZEN_SOURCE_PATH,
    source_sha,
)
frozen_source_sha = sha256_file(FROZEN_SOURCE_PATH)

final_gates = {
    **gates,
    "grade_count_artifact_created": GRADE_COUNTS_PATH.exists(),
    "lesion_count_artifact_created": LESION_COUNTS_PATH.exists(),
    "calibration_protocol_created": PROTOCOL_PATH.exists(),
    "protocol_lock_source_frozen_and_verified": frozen_source_sha == source_sha,
}

if not all(final_gates.values()):
    raise RuntimeError("Final calibration-protocol artifact gate failed.")

evidence = {
    "stage": "0G-4D-5B",
    "status": "PASS",
    "purpose": "LOCK_POST_SELECTION_CALIBRATION_METHOD_BEFORE_INFERENCE",
    "selected_model_epoch": 10,
    "selected_model_sha256": EXPECTED_FROZEN_CHECKPOINT_SHA,
    "CALIBRATION_rows": int(len(table)),
    "CALIBRATION_modality_counts": modality_counts,
    "grade_count_artifact": str(GRADE_COUNTS_PATH),
    "grade_count_artifact_sha256": sha256_file(GRADE_COUNTS_PATH),
    "lesion_count_artifact": str(LESION_COUNTS_PATH),
    "lesion_count_artifact_sha256": sha256_file(LESION_COUNTS_PATH),
    "calibration_protocol": str(PROTOCOL_PATH),
    "calibration_protocol_sha256": protocol_sha,
    "source": str(FROZEN_SOURCE_PATH),
    "source_sha256": frozen_source_sha,
    "source_action": source_action,
    "gates": {name: bool(value) for name, value in final_gates.items()},
    "medical_images_accessed": False,
    "model_loaded": False,
    "inference_started": False,
    "optimizer_created": False,
    "model_parameters_updated": False,
    "validation_used_for_calibration_fit": False,
    "locked_test_accessed": False,
    "next_required_stage": "RESUMABLE_CALIBRATION_ONLY_INFERENCE_AND_TEMPERATURE_FIT",
}
atomic_json_save(evidence, EVIDENCE_PATH)

print("\nGrade counts:")
print(GRADE_COUNTS_PATH)

print("\nLesion counts:")
print(LESION_COUNTS_PATH)

print("\nLocked calibration protocol:")
print(PROTOCOL_PATH)
print("SHA-256:", protocol_sha)

print("\nFrozen source:")
print(FROZEN_SOURCE_PATH)
print("SHA-256:", frozen_source_sha)
print("Source action:", source_action)

print("\nEvidence:")
print(EVIDENCE_PATH)

print("\nImages accessed             : NO")
print("Model loaded                : NO")
print("Inference started           : NO")
print("Optimizer created           : NO")
print("Model parameters updated    : NO")
print("VALIDATION used for fitting : NO")
print("Locked test accessed        : NO")

print("\nNEXT REQUIRED STAGE:")
print("Resumable CALIBRATION-only inference and locked temperature fitting")

print("\n" + "=" * 80)
print("STAGE 0G-4D-5B STATUS: PASS")
print("=" * 80)
