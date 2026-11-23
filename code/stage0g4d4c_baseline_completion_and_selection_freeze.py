from pathlib import Path
import hashlib
import json
import math
import os
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, f1_score


# =============================================================================
# STAGE 0G-4D-4C — BASELINE COMPLETION AND BEST-SELECTION FREEZE
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

RUN_ROOT = PROJECT_ROOT / "Runs/FOVEA_DR/baseline_seed2026"
CHECKPOINT_ROOT = PROJECT_ROOT / "Checkpoints/FOVEA_DR/baseline_seed2026"

RUNNER_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/stage0g4d4b_r2_resumable_baseline_seed2026.py"
)
CANDIDATE_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
)
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_training_protocol_v1.json"
)
PRODUCTION_SOURCE_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
)
TRAINING_CONTRACT_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_training_implementation_contract_v1.json"
)

RUN_STATE_PATH = RUN_ROOT / "run_state.json"
HISTORY_PATH = RUN_ROOT / "epoch_history.csv"
STEP_LOG_PATH = RUN_ROOT / "optimizer_step_log.csv"
BEST_PREDICTIONS_PATH = RUN_ROOT / "best_validation_predictions.csv"
LATEST_SESSION_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d4b_r2_baseline_seed2026_latest_session.json"
)

LAST_CHECKPOINT_PATH = CHECKPOINT_ROOT / "last_resume.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_ROOT / "best_model.pt"

FROZEN_BEST_CHECKPOINT_PATH = (
    CHECKPOINT_ROOT / "selected_epoch10_frozen.pt"
)
FROZEN_BEST_PREDICTIONS_PATH = (
    RUN_ROOT / "selected_epoch10_validation_predictions_frozen.csv"
)
FROZEN_AUDIT_SOURCE_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/stage0g4d4c_baseline_completion_and_selection_freeze.py"
)

FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_baseline_seed2026_selection_freeze_v1.json"
)
EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d4c_baseline_completion_and_selection_freeze.json"
)

EXPECTED_HASHES = {
    "runner": "72b82966837e553e8684526bc9d9e7e6ad7a6bded9120eec02f6f900954dbe47",
    "candidate": "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181",
    "protocol": "5b16e0c8565b86d3cac603f416ee0b7f03055ab9a0eb9ac9e3fb007632329b40",
    "production_source": "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9",
    "training_contract": "7de76cc722220e601a73125874dd9dee979527cfce847d441933ca8bfcb29ea4",
    "epoch16_last_checkpoint": "0ccab345cfb6a343a9b14cb9af348fc89c80f53d2f6d9dc8056b36bf98b9187c",
    "epoch10_best_checkpoint": "89f5e87b647680a9094c55bf647dc1dd2abb624681e733ffb453e30e4c11ff1f",
}

EXPECTED_FINAL_EPOCH = 16
EXPECTED_BEST_EPOCH = 10
EXPECTED_GLOBAL_STEP = 6640
EXPECTED_STEPS_PER_EPOCH = 415
EXPECTED_PATIENCE = 6
PRIMARY_MIN_DELTA = 0.002
EXPECTED_RUN_SEED = 2026
EXPECTED_VALIDATION_ROWS = 2839
EXPECTED_MODALITY_COUNTS = {"CFP": 1314, "UWF": 1168, "OCT": 357}
FLOAT_TOLERANCE = 1e-9


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


def copy_once_verified(source, destination, expected_sha):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination_sha = sha256_file(destination)
        if destination_sha != expected_sha:
            raise RuntimeError(
                f"Frozen destination already exists with a different hash: {destination}"
            )
        return "ALREADY_PRESENT_IDENTICAL"

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Temporary copy verification failed: {destination}")
    os.replace(temporary, destination)
    return "COPIED_AND_HASH_VERIFIED"


def all_evidence_gates_pass(payload):
    gates = payload.get("gates", {})
    return (
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and len(gates) > 0
        and all(bool(value) for value in gates.values())
    )


def close(a, b, tolerance=FLOAT_TOLERANCE):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def replay_selection(history):
    best_epoch = 0
    best_qwk = -math.inf
    best_nll = math.inf
    bad_epochs = 0
    replay = []

    for row in history.itertuples(index=False):
        epoch = int(row.epoch)
        current_qwk = float(row.mean_fundus_QWK)
        current_nll = float(row.mean_fundus_NLL)

        primary_improved = current_qwk > best_qwk + PRIMARY_MIN_DELTA
        tie_break_improved = (
            abs(current_qwk - best_qwk) <= PRIMARY_MIN_DELTA
            and current_nll < best_nll
        )
        selected = best_epoch == 0 or primary_improved or tie_break_improved

        if selected:
            best_epoch = epoch
            best_qwk = current_qwk
            best_nll = current_nll
            bad_epochs = 0
        else:
            bad_epochs += 1

        replay.append({
            "epoch": epoch,
            "selected": bool(selected),
            "best_epoch_after_epoch": best_epoch,
            "bad_epochs_after_epoch": bad_epochs,
        })

    return {
        "best_epoch": best_epoch,
        "best_qwk": best_qwk,
        "best_nll": best_nll,
        "bad_epochs": bad_epochs,
        "replay": replay,
    }


def probability_and_metric_audit(predictions):
    required_columns = {
        "relative_path",
        "modality",
        "target",
        "prediction",
        "expected_grade",
        "probability_0",
        "probability_1",
        "probability_2",
        "probability_3",
        "probability_4",
    }
    columns_ok = required_columns.issubset(predictions.columns)
    if not columns_ok:
        return {
            "columns_ok": False,
            "probabilities_valid": False,
            "metrics": {},
        }

    all_probabilities_valid = True
    metrics = {}

    for modality, classes in (("CFP", 5), ("UWF", 5), ("OCT", 3)):
        subset = predictions[predictions["modality"] == modality].copy()
        probability_columns = [f"probability_{index}" for index in range(classes)]
        probabilities = subset[probability_columns].to_numpy(dtype=float)
        targets = subset["target"].to_numpy(dtype=int)
        predicted = subset["prediction"].to_numpy(dtype=int)

        finite = bool(np.isfinite(probabilities).all())
        bounded = bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
        normalized = bool(
            np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-3)
        )
        target_range = bool(((targets >= 0) & (targets < classes)).all())
        prediction_range = bool(((predicted >= 0) & (predicted < classes)).all())
        argmax_matches = bool((probabilities.argmax(axis=1) == predicted).all())

        modality_valid = all(
            [finite, bounded, normalized, target_range, prediction_range, argmax_matches]
        )
        all_probabilities_valid = all_probabilities_valid and modality_valid

        nll = float(
            -np.log(
                np.clip(probabilities[np.arange(len(targets)), targets], 1e-7, 1.0)
            ).mean()
        )
        qwk = float(cohen_kappa_score(targets, predicted, weights="quadratic"))
        macro_f1 = float(
            f1_score(targets, predicted, average="macro", zero_division=0)
        )

        metrics[modality] = {
            "rows": int(len(subset)),
            "QWK": qwk,
            "NLL": nll,
            "macro_F1": macro_f1,
            "probabilities_valid": bool(modality_valid),
        }

    metrics["mean_fundus_QWK"] = float(
        0.5 * (metrics["CFP"]["QWK"] + metrics["UWF"]["QWK"])
    )
    metrics["mean_fundus_NLL"] = float(
        0.5 * (metrics["CFP"]["NLL"] + metrics["UWF"]["NLL"])
    )

    return {
        "columns_ok": True,
        "probabilities_valid": bool(all_probabilities_valid),
        "metrics": metrics,
    }


print("=" * 80)
print("STAGE 0G-4D-4C — BASELINE COMPLETION AND BEST-SELECTION FREEZE")
print("=" * 80)


# =============================================================================
# 1. REQUIRED ARTIFACT VERIFICATION
# =============================================================================

print("\n[1] REQUIRED ARTIFACT VERIFICATION")
print("-" * 80)

required_paths = {
    "runner": RUNNER_PATH,
    "candidate": CANDIDATE_PATH,
    "protocol": PROTOCOL_PATH,
    "production_source": PRODUCTION_SOURCE_PATH,
    "training_contract": TRAINING_CONTRACT_PATH,
    "run_state": RUN_STATE_PATH,
    "history": HISTORY_PATH,
    "step_log": STEP_LOG_PATH,
    "best_predictions": BEST_PREDICTIONS_PATH,
    "latest_session_evidence": LATEST_SESSION_EVIDENCE_PATH,
    "last_checkpoint": LAST_CHECKPOINT_PATH,
    "best_checkpoint": BEST_CHECKPOINT_PATH,
}

for name, path in required_paths.items():
    print(f"{name:28s}: exists={path.exists()}")

if not all(path.exists() for path in required_paths.values()):
    raise FileNotFoundError("One or more required baseline artifacts are missing.")

actual_hashes = {
    "runner": sha256_file(RUNNER_PATH),
    "candidate": sha256_file(CANDIDATE_PATH),
    "protocol": sha256_file(PROTOCOL_PATH),
    "production_source": sha256_file(PRODUCTION_SOURCE_PATH),
    "training_contract": sha256_file(TRAINING_CONTRACT_PATH),
    "epoch16_last_checkpoint": sha256_file(LAST_CHECKPOINT_PATH),
    "epoch10_best_checkpoint": sha256_file(BEST_CHECKPOINT_PATH),
}

hash_gates = {
    name: actual_hashes[name] == expected
    for name, expected in EXPECTED_HASHES.items()
}

print("\nHash verification:")
for name, passed in hash_gates.items():
    print(f"{name:36s}: {passed}")

if not all(hash_gates.values()):
    raise RuntimeError("A frozen input hash changed; selection freeze aborted.")


# =============================================================================
# 2. COMPLETION STATE AND SESSION EVIDENCE
# =============================================================================

print("\n[2] COMPLETION STATE AND SESSION EVIDENCE")
print("-" * 80)

run_state = load_json(RUN_STATE_PATH)
session_evidence = load_json(LATEST_SESSION_EVIDENCE_PATH)

run_state_hashes_match = (
    run_state.get("runner_source_sha256") == actual_hashes["runner"]
    and run_state.get("last_checkpoint_sha256")
    == actual_hashes["epoch16_last_checkpoint"]
    and run_state.get("best_checkpoint_sha256")
    == actual_hashes["epoch10_best_checkpoint"]
    and run_state.get("history_sha256") == sha256_file(HISTORY_PATH)
    and run_state.get("step_log_sha256") == sha256_file(STEP_LOG_PATH)
)

session_evidence_ok = (
    all_evidence_gates_pass(session_evidence)
    and session_evidence.get("run_complete") is True
    and session_evidence.get("early_stopped") is True
    and session_evidence.get("calibration_used") is False
    and session_evidence.get("locked_test_used") is False
)

print("Run status               :", run_state.get("status"))
print("Last completed epoch     :", run_state.get("last_completed_epoch"))
print("Global optimizer step    :", run_state.get("global_optimizer_step"))
print("Best epoch               :", run_state.get("best_epoch"))
print("Bad epochs               :", run_state.get("bad_epochs"))
print("Early stopped            :", run_state.get("early_stopped"))
print("Run-state hashes current :", run_state_hashes_match)
print("Latest session PASS      :", session_evidence_ok)


# =============================================================================
# 3. HISTORY AND OPTIMIZER-STEP REPLAY
# =============================================================================

print("\n[3] HISTORY AND OPTIMIZER-STEP REPLAY")
print("-" * 80)

history = pd.read_csv(HISTORY_PATH)
steps = pd.read_csv(STEP_LOG_PATH)

expected_epochs = list(range(1, EXPECTED_FINAL_EPOCH + 1))
epochs_exact = history["epoch"].astype(int).tolist() == expected_epochs
history_global_steps_exact = bool(
    (
        history["global_optimizer_step"].astype(int).to_numpy()
        == np.arange(1, EXPECTED_FINAL_EPOCH + 1) * EXPECTED_STEPS_PER_EPOCH
    ).all()
)

step_rows_exact = len(steps) == EXPECTED_GLOBAL_STEP
step_global_sequence_exact = bool(
    step_rows_exact
    and (
        steps["global_optimizer_step"].astype(int).to_numpy()
        == np.arange(1, EXPECTED_GLOBAL_STEP + 1)
    ).all()
)
step_epoch_counts_exact = (
    steps.groupby("epoch").size().to_dict()
    == {
        epoch: EXPECTED_STEPS_PER_EPOCH
        for epoch in expected_epochs
    }
)

finite_step_columns = [
    "loss",
    "gradient_norm_before_clip",
    "grad_scaler_scale",
    "backbone_lr",
    "new_module_lr",
    "step_seconds",
]
all_step_values_finite = bool(
    np.isfinite(steps[finite_step_columns].to_numpy(dtype=float)).all()
)

selection = replay_selection(history)
history_flags_match_replay = all(
    bool(row.is_best) == replay["selected"]
    for row, replay in zip(history.itertuples(index=False), selection["replay"])
)
history_bad_epochs_match_replay = all(
    int(row.bad_epochs) == replay["bad_epochs_after_epoch"]
    for row, replay in zip(history.itertuples(index=False), selection["replay"])
)

best_history_row = history.loc[
    history["epoch"].astype(int) == EXPECTED_BEST_EPOCH
].iloc[0]
final_history_row = history.iloc[-1]

print("History rows              :", len(history))
print("Epoch sequence exact      :", epochs_exact)
print("History global steps exact:", history_global_steps_exact)
print("Optimizer-step rows       :", len(steps))
print("Step sequence exact       :", step_global_sequence_exact)
print("415 steps per epoch       :", step_epoch_counts_exact)
print("All step values finite    :", all_step_values_finite)
print("Replayed best epoch       :", selection["best_epoch"])
print("Replayed bad epochs       :", selection["bad_epochs"])
print("Stored is_best flags match:", history_flags_match_replay)


# =============================================================================
# 4. CHECKPOINT INTERNAL AUDIT
# =============================================================================

print("\n[4] CHECKPOINT INTERNAL AUDIT")
print("-" * 80)

last_checkpoint = torch.load(
    LAST_CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)
best_checkpoint = torch.load(
    BEST_CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

last_checkpoint_metadata_ok = (
    last_checkpoint.get("format") == "FOVEA_DR_BASELINE_RESUME_V1"
    and int(last_checkpoint.get("run_seed", -1)) == EXPECTED_RUN_SEED
    and int(last_checkpoint.get("epoch_completed", -1)) == EXPECTED_FINAL_EPOCH
    and int(last_checkpoint.get("global_optimizer_step", -1))
    == EXPECTED_GLOBAL_STEP
    and int(last_checkpoint.get("best_epoch", -1)) == EXPECTED_BEST_EPOCH
    and int(last_checkpoint.get("bad_epochs", -1)) == EXPECTED_PATIENCE
    and last_checkpoint.get("runner_source_sha256") == actual_hashes["runner"]
    and last_checkpoint.get("candidate_sha256") == actual_hashes["candidate"]
    and last_checkpoint.get("protocol_sha256") == actual_hashes["protocol"]
    and last_checkpoint.get("production_source_sha256")
    == actual_hashes["production_source"]
)

best_checkpoint_metadata_ok = (
    best_checkpoint.get("format") == "FOVEA_DR_BASELINE_BEST_MODEL_V1"
    and int(best_checkpoint.get("run_seed", -1)) == EXPECTED_RUN_SEED
    and int(best_checkpoint.get("epoch", -1)) == EXPECTED_BEST_EPOCH
    and best_checkpoint.get("runner_source_sha256") == actual_hashes["runner"]
    and best_checkpoint.get("production_source_sha256")
    == actual_hashes["production_source"]
    and close(best_checkpoint.get("mean_fundus_QWK"), best_history_row["mean_fundus_QWK"])
    and close(best_checkpoint.get("mean_fundus_NLL"), best_history_row["mean_fundus_NLL"])
)

model_state = best_checkpoint.get("model_state_dict", {})
model_state_nonempty = isinstance(model_state, dict) and len(model_state) > 0
model_float_tensors = [
    value
    for value in model_state.values()
    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
]
all_selected_model_tensors_finite = (
    model_state_nonempty
    and len(model_float_tensors) > 0
    and all(bool(torch.isfinite(value).all()) for value in model_float_tensors)
)

print("Last checkpoint format    :", last_checkpoint.get("format"))
print("Last checkpoint epoch     :", last_checkpoint.get("epoch_completed"))
print("Last checkpoint global step:", last_checkpoint.get("global_optimizer_step"))
print("Last metadata verified    :", last_checkpoint_metadata_ok)
print("Best checkpoint format    :", best_checkpoint.get("format"))
print("Best checkpoint epoch     :", best_checkpoint.get("epoch"))
print("Best metadata verified    :", best_checkpoint_metadata_ok)
print("Selected state entries    :", len(model_state))
print("All model tensors finite  :", all_selected_model_tensors_finite)


# =============================================================================
# 5. SELECTED VALIDATION-PREDICTION AUDIT
# =============================================================================

print("\n[5] SELECTED VALIDATION-PREDICTION AUDIT")
print("-" * 80)

best_predictions = pd.read_csv(BEST_PREDICTIONS_PATH)
prediction_counts = (
    best_predictions["modality"].value_counts().astype(int).to_dict()
)
prediction_audit = probability_and_metric_audit(best_predictions)
prediction_metrics = prediction_audit["metrics"]

predictions_match_best_history = (
    prediction_audit["columns_ok"]
    and close(
        prediction_metrics["mean_fundus_QWK"],
        best_history_row["mean_fundus_QWK"],
        tolerance=2e-7,
    )
    and close(
        prediction_metrics["mean_fundus_NLL"],
        best_history_row["mean_fundus_NLL"],
        tolerance=2e-7,
    )
    and close(
        prediction_metrics["OCT"]["QWK"],
        best_history_row["OCT_QWK"],
        tolerance=2e-7,
    )
    and close(
        prediction_metrics["OCT"]["macro_F1"],
        best_history_row["OCT_macro_F1"],
        tolerance=2e-7,
    )
)

print("Prediction rows           :", len(best_predictions))
print("Modality counts           :", prediction_counts)
print("Unique relative paths     :", best_predictions["relative_path"].is_unique)
print("All distributions valid  :", prediction_audit["probabilities_valid"])
print("Metrics match epoch 10    :", predictions_match_best_history)
print("Recomputed mean Fundus QWK:", f"{prediction_metrics['mean_fundus_QWK']:.12f}")
print("Recomputed mean Fundus NLL:", f"{prediction_metrics['mean_fundus_NLL']:.12f}")


# =============================================================================
# 6. COMPLETION GATES BEFORE COPYING
# =============================================================================

pre_freeze_gates = {
    "all_required_artifacts_exist":
        all(path.exists() for path in required_paths.values()),
    "all_frozen_input_hashes_verified":
        all(hash_gates.values()),
    "run_status_is_COMPLETE":
        run_state.get("status") == "COMPLETE",
    "last_completed_epoch_is_16":
        int(run_state.get("last_completed_epoch", -1)) == EXPECTED_FINAL_EPOCH,
    "global_optimizer_step_is_6640":
        int(run_state.get("global_optimizer_step", -1)) == EXPECTED_GLOBAL_STEP,
    "early_stopping_triggered_at_patience_6":
        run_state.get("early_stopped") is True
        and int(run_state.get("bad_epochs", -1)) == EXPECTED_PATIENCE,
    "run_state_hashes_match_current_files":
        run_state_hashes_match,
    "latest_training_session_evidence_passed":
        session_evidence_ok,
    "history_contains_exactly_epochs_1_through_16":
        len(history) == EXPECTED_FINAL_EPOCH and epochs_exact,
    "history_global_steps_are_exact":
        history_global_steps_exact,
    "optimizer_step_log_contains_exactly_6640_steps":
        step_rows_exact and step_global_sequence_exact,
    "each_epoch_contains_exactly_415_optimizer_steps":
        step_epoch_counts_exact,
    "all_optimizer_step_values_finite":
        all_step_values_finite,
    "selection_rule_replays_to_epoch_10":
        selection["best_epoch"] == EXPECTED_BEST_EPOCH,
    "selection_flags_match_replay":
        history_flags_match_replay,
    "bad_epoch_counts_match_replay":
        history_bad_epochs_match_replay
        and selection["bad_epochs"] == EXPECTED_PATIENCE,
    "epoch16_last_checkpoint_metadata_verified":
        last_checkpoint_metadata_ok,
    "epoch10_best_checkpoint_metadata_verified":
        best_checkpoint_metadata_ok,
    "selected_model_state_is_nonempty_and_finite":
        model_state_nonempty and all_selected_model_tensors_finite,
    "best_validation_prediction_row_count_exact":
        len(best_predictions) == EXPECTED_VALIDATION_ROWS,
    "best_validation_modality_counts_exact":
        prediction_counts == EXPECTED_MODALITY_COUNTS,
    "best_validation_paths_unique":
        bool(best_predictions["relative_path"].is_unique),
    "best_validation_probability_distributions_valid":
        prediction_audit["probabilities_valid"],
    "recomputed_metrics_match_selected_epoch":
        predictions_match_best_history,
    "CALIBRATION_not_accessed":
        True,
    "locked_test_not_accessed":
        True,
}

print("\n" + "=" * 80)
print("BASELINE COMPLETION PRE-FREEZE GATES")
print("=" * 80)
for name, passed in pre_freeze_gates.items():
    print(f"{name:64s}: {passed}")

if not all(pre_freeze_gates.values()):
    raise RuntimeError("Baseline completion audit failed; no freeze copy was created.")


# =============================================================================
# 7. BYTE-FOR-BYTE SELECTION FREEZE
# =============================================================================

print("\n[7] BYTE-FOR-BYTE SELECTION FREEZE")
print("-" * 80)

checkpoint_copy_action = copy_once_verified(
    BEST_CHECKPOINT_PATH,
    FROZEN_BEST_CHECKPOINT_PATH,
    EXPECTED_HASHES["epoch10_best_checkpoint"],
)

best_predictions_sha = sha256_file(BEST_PREDICTIONS_PATH)
predictions_copy_action = copy_once_verified(
    BEST_PREDICTIONS_PATH,
    FROZEN_BEST_PREDICTIONS_PATH,
    best_predictions_sha,
)

audit_source = Path(__file__).resolve()
audit_source_sha = sha256_file(audit_source)
audit_source_action = copy_once_verified(
    audit_source,
    FROZEN_AUDIT_SOURCE_PATH,
    audit_source_sha,
)

frozen_checkpoint_sha = sha256_file(FROZEN_BEST_CHECKPOINT_PATH)
frozen_predictions_sha = sha256_file(FROZEN_BEST_PREDICTIONS_PATH)
frozen_audit_source_sha = sha256_file(FROZEN_AUDIT_SOURCE_PATH)

print("Checkpoint action         :", checkpoint_copy_action)
print("Frozen checkpoint         :", FROZEN_BEST_CHECKPOINT_PATH)
print("Frozen checkpoint SHA-256 :", frozen_checkpoint_sha)
print("Predictions action        :", predictions_copy_action)
print("Frozen predictions SHA-256:", frozen_predictions_sha)
print("Audit-source action       :", audit_source_action)
print("Frozen audit source SHA   :", frozen_audit_source_sha)


# =============================================================================
# 8. FREEZE MANIFEST, FINAL GATES, AND EVIDENCE
# =============================================================================

freeze_manifest = {
    "artifact": "FOVEA_DR_BASELINE_SEED2026_SELECTED_MODEL",
    "version": "1",
    "status": "FROZEN",
    "run_seed": EXPECTED_RUN_SEED,
    "training_completion": {
        "termination": "EARLY_STOPPING",
        "last_completed_epoch": EXPECTED_FINAL_EPOCH,
        "global_optimizer_step": EXPECTED_GLOBAL_STEP,
        "early_stopping_patience": EXPECTED_PATIENCE,
    },
    "selection_contract": {
        "primary_metric": "MEAN_CFP_UWF_QUADRATIC_WEIGHTED_KAPPA",
        "primary_min_delta": PRIMARY_MIN_DELTA,
        "tie_breaker": "LOWER_MEAN_CFP_UWF_NLL",
        "selection_partition": "VALIDATION",
    },
    "selected_epoch": EXPECTED_BEST_EPOCH,
    "selected_metrics": {
        "CFP_QWK": float(best_history_row["CFP_QWK"]),
        "UWF_QWK": float(best_history_row["UWF_QWK"]),
        "mean_fundus_QWK": float(best_history_row["mean_fundus_QWK"]),
        "CFP_NLL": float(best_history_row["CFP_NLL"]),
        "UWF_NLL": float(best_history_row["UWF_NLL"]),
        "mean_fundus_NLL": float(best_history_row["mean_fundus_NLL"]),
        "OCT_QWK": float(best_history_row["OCT_QWK"]),
        "OCT_NLL": float(best_history_row["OCT_NLL"]),
        "OCT_macro_F1": float(best_history_row["OCT_macro_F1"]),
    },
    "source_best_checkpoint": str(BEST_CHECKPOINT_PATH),
    "source_best_checkpoint_sha256": actual_hashes["epoch10_best_checkpoint"],
    "frozen_selected_checkpoint": str(FROZEN_BEST_CHECKPOINT_PATH),
    "frozen_selected_checkpoint_sha256": frozen_checkpoint_sha,
    "source_best_validation_predictions": str(BEST_PREDICTIONS_PATH),
    "source_best_validation_predictions_sha256": best_predictions_sha,
    "frozen_selected_validation_predictions": str(FROZEN_BEST_PREDICTIONS_PATH),
    "frozen_selected_validation_predictions_sha256": frozen_predictions_sha,
    "history": str(HISTORY_PATH),
    "history_sha256": sha256_file(HISTORY_PATH),
    "optimizer_step_log": str(STEP_LOG_PATH),
    "optimizer_step_log_sha256": sha256_file(STEP_LOG_PATH),
    "run_state": str(RUN_STATE_PATH),
    "run_state_sha256": sha256_file(RUN_STATE_PATH),
    "last_resume_checkpoint": str(LAST_CHECKPOINT_PATH),
    "last_resume_checkpoint_sha256": actual_hashes["epoch16_last_checkpoint"],
    "runner": str(RUNNER_PATH),
    "runner_sha256": actual_hashes["runner"],
    "audit_source": str(FROZEN_AUDIT_SOURCE_PATH),
    "audit_source_sha256": frozen_audit_source_sha,
    "calibration_used_for_selection": False,
    "locked_test_used": False,
    "next_required_stage": "CALIBRATION_ONLY_CACHE_PREPARATION",
}

atomic_json_save(freeze_manifest, FREEZE_MANIFEST_PATH)
freeze_manifest_sha = sha256_file(FREEZE_MANIFEST_PATH)

final_gates = {
    **pre_freeze_gates,
    "frozen_checkpoint_exists":
        FROZEN_BEST_CHECKPOINT_PATH.exists(),
    "frozen_checkpoint_is_byte_identical_to_selected_best":
        frozen_checkpoint_sha == actual_hashes["epoch10_best_checkpoint"],
    "frozen_predictions_exist":
        FROZEN_BEST_PREDICTIONS_PATH.exists(),
    "frozen_predictions_are_byte_identical_to_selected_best":
        frozen_predictions_sha == best_predictions_sha,
    "audit_source_frozen_and_hash_verified":
        FROZEN_AUDIT_SOURCE_PATH.exists()
        and frozen_audit_source_sha == audit_source_sha,
    "selection_freeze_manifest_created":
        FREEZE_MANIFEST_PATH.exists(),
}

print("\n" + "=" * 80)
print("BASELINE COMPLETION AND FREEZE GATES")
print("=" * 80)
for name, passed in final_gates.items():
    print(f"{name:64s}: {passed}")

if not all(final_gates.values()):
    raise RuntimeError("Final baseline selection-freeze gate failed.")

evidence = {
    "stage": "0G-4D-4C",
    "status": "PASS",
    "purpose": "BASELINE_COMPLETION_AND_BEST_SELECTION_FREEZE",
    "run": "FOVEA_DR_BASELINE_SEED2026",
    "training_termination": "EARLY_STOPPING",
    "last_completed_epoch": EXPECTED_FINAL_EPOCH,
    "global_optimizer_step": EXPECTED_GLOBAL_STEP,
    "selected_epoch": EXPECTED_BEST_EPOCH,
    "selected_mean_fundus_QWK": float(best_history_row["mean_fundus_QWK"]),
    "selected_mean_fundus_NLL": float(best_history_row["mean_fundus_NLL"]),
    "final_epoch_mean_train_loss": float(final_history_row["mean_train_loss"]),
    "final_epoch_mean_fundus_QWK": float(final_history_row["mean_fundus_QWK"]),
    "final_epoch_mean_fundus_NLL": float(final_history_row["mean_fundus_NLL"]),
    "frozen_selected_checkpoint": str(FROZEN_BEST_CHECKPOINT_PATH),
    "frozen_selected_checkpoint_sha256": frozen_checkpoint_sha,
    "frozen_selected_validation_predictions": str(
        FROZEN_BEST_PREDICTIONS_PATH
    ),
    "frozen_selected_validation_predictions_sha256": frozen_predictions_sha,
    "selection_freeze_manifest": str(FREEZE_MANIFEST_PATH),
    "selection_freeze_manifest_sha256": freeze_manifest_sha,
    "audit_source": str(FROZEN_AUDIT_SOURCE_PATH),
    "audit_source_sha256": frozen_audit_source_sha,
    "gates": {name: bool(value) for name, value in final_gates.items()},
    "optimizer_created": False,
    "model_parameters_updated": False,
    "medical_images_accessed": False,
    "validation_predictions_read_for_selection_audit": True,
    "calibration_accessed": False,
    "locked_test_accessed": False,
    "next_required_stage": "CALIBRATION_ONLY_CACHE_PREPARATION",
}

atomic_json_save(evidence, EVIDENCE_PATH)

print("\nSelected baseline checkpoint:")
print(FROZEN_BEST_CHECKPOINT_PATH)
print("SHA-256:", frozen_checkpoint_sha)

print("\nSelection-freeze manifest:")
print(FREEZE_MANIFEST_PATH)
print("SHA-256:", freeze_manifest_sha)

print("\nEvidence:")
print(EVIDENCE_PATH)

print("\nOptimizer created       : NO")
print("Model parameters updated: NO")
print("Medical images accessed : NO")
print("CALIBRATION accessed     : NO")
print("Locked test accessed     : NO")

print("\nNEXT REQUIRED STAGE:")
print("CALIBRATION-only local cache preparation")

print("\n" + "=" * 80)
print("STAGE 0G-4D-4C STATUS: PASS")
print("=" * 80)
