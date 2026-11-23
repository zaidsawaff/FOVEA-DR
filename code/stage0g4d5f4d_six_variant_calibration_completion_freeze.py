#!/usr/bin/env python3
"""Freeze and audit completion of all six post-hoc calibration runs.

This stage is intentionally CPU/read-only with respect to model checkpoints and
images.  It validates the protocol-locked outputs produced by stage 0G-4D-5F-4C,
creates a deterministic six-row registry, and freezes a completion contract and
evidence record.  It never imports the model implementation, opens an image,
creates an optimizer, updates parameters, or accesses the locked test set.
"""

from __future__ import annotations

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


WIDTH = 124
ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_six_variant_calibration_protocol_v1.json"
PROTOCOL_EVIDENCE = ROOT / "Evidence/stage0g4d5f4b_six_variant_calibration_protocol_lock.json"
PROTOCOL_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4b_six_variant_calibration_protocol_lock.py"
CALIBRATION_RUNNER = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4c_resumable_six_variant_cpu_calibration.py"
FINAL_TRAINING_REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_final_training_registry_v1.csv"
CALIBRATION_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_CALIBRATION_local_cache_manifest_v1.csv"
DEVELOPMENT_SPLIT = ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4d_six_variant_calibration_completion_freeze.py"
REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_six_variant_calibration_registry_v1.csv"
CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_six_variant_calibration_completion_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5f4d_six_variant_calibration_completion_freeze.json"

EXPECTED_SHA256 = {
    PROTOCOL: "9bdd3a5cf4f4ead1dc3055fd5c84b62ca1bea929a95bf96372ec8bea66487c4d",
    PROTOCOL_EVIDENCE: "eb78d67a25728eadff334a9f20a6bb94d42704781f591ad153b1ece2484abf5c",
    PROTOCOL_SOURCE: "0a59899a1abb238aacfb5b18d99c6444725493cf8f6c1eb1cafd2899dbeb9fec",
    CALIBRATION_RUNNER: "a21db2d7a3e247ddfdb9a25bcc12d4931076fd37c3e41982453b611a94b5fd0e",
    FINAL_TRAINING_REGISTRY: "de8b3d0c098128510d185960f81af772c37f34ba877580b7160606153fa6ad75",
    CALIBRATION_MANIFEST: "f8e6a62b96e330a1b21bb9ed56897cbf04b730987cf9e0c31141927abbb84655",
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

LESION_COLUMNS = [
    "lesion_microaneurysm",
    "lesion_hard_exudate",
    "lesion_intraretinal_hemorrhage",
    "lesion_VB_IRMA",
    "lesion_neovascularization",
    "lesion_vitreous_hemorrhage",
    "lesion_retinal_detachment",
]

EXPECTED_ROWS = 2840
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS * len(VARIANT_IDS)
EXPECTED_GRADE_UNITS = {"CFP", "UWF", "OCT"}
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 20.0


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
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def frozen_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact exists with different content: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    atomic_write(path, data)
    return "CREATED_AND_FROZEN"


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    return frozen_bytes(path, json_bytes(payload))


def frozen_csv(path: Path, frame: pd.DataFrame) -> str:
    return frozen_bytes(path, csv_bytes(frame))


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different 5F-4D completion-freeze source is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    if sha256_file(temporary) != source_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Completion-freeze source copy verification failed.")
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def real_drive_mounted() -> bool:
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", "/content/drive"],
        capture_output=True,
        text=True,
        check=False,
    )
    return ROOT.is_dir() and "drive" in result.stdout.strip().lower()


def truthy_series(series: pd.Series) -> bool:
    accepted = {"true", "1", "yes", "y", "t"}
    return bool(series.astype(str).str.strip().str.lower().isin(accepted).all())


def all_evidence_gates_pass(payload: dict[str, Any]) -> bool:
    gates = payload.get("gates")
    return bool(
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and gates
        and all(bool(value) for value in gates.values())
    )


def exact_sha(path: Path, expected: Any) -> bool:
    return bool(path.is_file() and isinstance(expected, str) and sha256_file(path) == expected)


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(frame["rows"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError(f"Cannot compute weighted mean for {column}.")
    return float(np.average(values, weights=weights))


def inspect_variant(
    position: int,
    contract: dict[str, Any],
    protocol_sha: str,
    runner_sha: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    variant_id = VARIANT_IDS[position]
    run_root = Path(str(contract["run_root"]))
    state_path = Path(str(contract["inference_state"]))
    raw_path = Path(str(contract["raw_predictions"]))
    calibrated_path = Path(str(contract["calibrated_predictions"]))
    temperatures_path = Path(str(contract["temperature_parameters"]))
    metrics_path = Path(str(contract["calibration_metrics"]))
    latest_evidence_path = Path(str(contract["latest_session_evidence"]))
    completion_evidence_path = Path(str(contract["completion_evidence"]))
    checkpoint_path = Path(str(contract["selected_checkpoint"]))
    checkpoint_sha = str(contract["selected_checkpoint_sha256"])

    required = {
        "state": state_path,
        "raw_predictions": raw_path,
        "calibrated_predictions": calibrated_path,
        "temperature_parameters": temperatures_path,
        "metrics": metrics_path,
        "latest_evidence": latest_evidence_path,
        "completion_evidence": completion_evidence_path,
        "selected_checkpoint": checkpoint_path,
    }
    checks: dict[str, bool] = {
        f"{name}_exists_nonempty": path.is_file() and path.stat().st_size > 0
        for name, path in required.items()
    }
    if not all(checks.values()):
        return {"position": position + 1, "variant_id": variant_id}, checks

    state = read_json(state_path)
    temperatures = read_json(temperatures_path)
    evidence = read_json(completion_evidence_path)
    latest_evidence = read_json(latest_evidence_path)
    final_hashes = state.get("final_artifact_sha256", {})
    if not isinstance(final_hashes, dict):
        final_hashes = {}

    artifact_paths = {
        "raw_predictions": raw_path,
        "calibrated_predictions": calibrated_path,
        "temperature_parameters": temperatures_path,
        "metrics": metrics_path,
    }
    actual_hashes = {name: sha256_file(path) for name, path in artifact_paths.items()}

    raw = pd.read_csv(raw_path)
    calibrated = pd.read_csv(calibrated_path)
    metrics = pd.read_csv(metrics_path)

    expected_indices = list(range(EXPECTED_ROWS))
    raw_indices = pd.to_numeric(raw.get("calibration_index"), errors="coerce")
    calibrated_indices = pd.to_numeric(calibrated.get("calibration_index"), errors="coerce")
    grade_metrics = metrics[metrics.get("task_type", pd.Series(dtype=str)).astype(str) == "GRADE"].copy()
    lesion_metrics = metrics[metrics.get("task_type", pd.Series(dtype=str)).astype(str) == "LESION"].copy()

    nll_before = pd.to_numeric(metrics.get("NLL_before"), errors="coerce")
    nll_after = pd.to_numeric(metrics.get("NLL_after"), errors="coerce")
    temperature_values = pd.to_numeric(metrics.get("temperature"), errors="coerce")

    decision_invariance = False
    if "raw_prediction" in calibrated.columns and "calibrated_prediction" in calibrated.columns:
        decision_invariance = bool(
            np.array_equal(
                calibrated["raw_prediction"].astype(int).to_numpy(),
                calibrated["calibrated_prediction"].astype(int).to_numpy(),
            )
        )

    lesion_decisions_invariant = True
    valid_mask = (
        calibrated["modality"].astype(str).isin(["CFP", "UWF"])
        & (pd.to_numeric(calibrated["lesion_loss_mask"], errors="coerce") > 0.5)
    )
    for lesion in LESION_COLUMNS:
        raw_column = f"raw_{lesion}_probability"
        calibrated_column = f"calibrated_{lesion}_probability"
        if raw_column not in calibrated.columns or calibrated_column not in calibrated.columns:
            lesion_decisions_invariant = False
            break
        before = pd.to_numeric(calibrated.loc[valid_mask, raw_column], errors="coerce").to_numpy()
        after = pd.to_numeric(calibrated.loc[valid_mask, calibrated_column], errors="coerce").to_numpy()
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            lesion_decisions_invariant = False
            break
        if not np.array_equal(before >= 0.5, after >= 0.5):
            lesion_decisions_invariant = False
            break

    grade_temperatures = temperatures.get("grade_temperatures", {})
    lesion_temperatures = temperatures.get("lesion_temperatures", {})
    json_temperatures: list[float] = []
    if isinstance(grade_temperatures, dict) and isinstance(lesion_temperatures, dict):
        try:
            json_temperatures = [
                float(item["temperature"]) for item in grade_temperatures.values()
            ] + [float(item["temperature"]) for item in lesion_temperatures.values()]
        except (KeyError, TypeError, ValueError):
            json_temperatures = []

    parts = state.get("parts")
    parts_valid = isinstance(parts, list) and bool(parts)
    expected_start = 0
    if parts_valid:
        for record in parts:
            if not isinstance(record, dict):
                parts_valid = False
                break
            part_path = Path(str(record.get("path", "")))
            try:
                start = int(record.get("start_index", -1))
                end = int(record.get("end_index", -1))
                rows = int(record.get("rows", -1))
            except (TypeError, ValueError):
                parts_valid = False
                break
            if not (
                start == expected_start
                and end >= start
                and rows == end - start + 1
                and part_path.is_file()
                and sha256_file(part_path) == record.get("sha256")
            ):
                parts_valid = False
                break
            expected_start = end + 1
        parts_valid = bool(parts_valid and expected_start == EXPECTED_ROWS)

    checks.update(
        {
            "contract_variant_exact": contract.get("variant_id") == variant_id,
            "contract_position_exact": position == VARIANT_IDS.index(variant_id),
            "checkpoint_hash_verified": sha256_file(checkpoint_path) == checkpoint_sha,
            "state_variant_exact": state.get("variant_id") == variant_id,
            "state_status_complete": state.get("status") == "COMPLETE_AND_TEMPERATURE_FITTED",
            "state_backend_CPU_FLOAT32": state.get("execution_backend") == "CPU_FLOAT32",
            "state_rows_exact": int(state.get("completed_rows", -1)) == EXPECTED_ROWS
            and int(state.get("expected_rows", -1)) == EXPECTED_ROWS,
            "state_part_rows_32": int(state.get("part_rows", -1)) == 32,
            "state_parts_contiguous_and_hash_verified": parts_valid,
            "state_protocol_hash_verified": state.get("calibration_protocol_sha256") == protocol_sha,
            "state_runner_hash_verified": state.get("runner_source_sha256") == runner_sha,
            "state_checkpoint_hash_verified": state.get("selected_checkpoint_sha256") == checkpoint_sha,
            "state_manifest_hash_verified": state.get("calibration_manifest_sha256")
            == EXPECTED_SHA256[CALIBRATION_MANIFEST],
            "state_optimizer_not_created": state.get("optimizer_created") is False,
            "state_parameters_not_updated": state.get("model_parameters_updated") is False,
            "state_VALIDATION_not_used_for_fit": state.get("validation_used_for_calibration_fit") is False,
            "state_locked_test_not_accessed": state.get("locked_test_accessed") is False,
            "final_artifact_hashes_exact": all(
                final_hashes.get(name) == actual_hashes[name] for name in artifact_paths
            ),
            "raw_has_exactly_2840_rows": len(raw) == EXPECTED_ROWS,
            "calibrated_has_exactly_2840_rows": len(calibrated) == EXPECTED_ROWS,
            "raw_indices_contiguous": raw_indices.notna().all()
            and raw_indices.astype(int).tolist() == expected_indices,
            "calibrated_indices_contiguous": calibrated_indices.notna().all()
            and calibrated_indices.astype(int).tolist() == expected_indices,
            "raw_variant_exact": "variant_id" in raw.columns
            and raw["variant_id"].astype(str).eq(variant_id).all(),
            "calibrated_variant_exact": "variant_id" in calibrated.columns
            and calibrated["variant_id"].astype(str).eq(variant_id).all(),
            "raw_and_calibrated_rows_linked": all(
                column in raw.columns and column in calibrated.columns
                and raw[column].astype(str).tolist() == calibrated[column].astype(str).tolist()
                for column in ["relative_path", "image_sha256", "modality", "target", "raw_prediction"]
            ),
            "grade_decisions_invariant": decision_invariance,
            "lesion_decisions_invariant": lesion_decisions_invariant,
            "metrics_has_exactly_10_rows": len(metrics) == 10,
            "metrics_variant_exact": "variant_id" in metrics.columns
            and metrics["variant_id"].astype(str).eq(variant_id).all(),
            "three_grade_metric_rows": len(grade_metrics) == 3
            and set(grade_metrics["unit"].astype(str)) == EXPECTED_GRADE_UNITS,
            "seven_lesion_metric_rows": len(lesion_metrics) == 7
            and set(lesion_metrics["unit"].astype(str)) == set(LESION_COLUMNS),
            "all_metric_NLL_values_finite": len(nll_before) == 10
            and np.isfinite(nll_before.to_numpy(dtype=float)).all()
            and np.isfinite(nll_after.to_numpy(dtype=float)).all(),
            "all_metric_NLL_values_nonworsening": len(nll_before) == 10
            and bool((nll_after <= nll_before + 1e-12).all()),
            "all_metric_decision_flags_true": "decision_invariant" in metrics.columns
            and truthy_series(metrics["decision_invariant"]),
            "all_metric_NLL_flags_true": "NLL_nonworsening" in metrics.columns
            and truthy_series(metrics["NLL_nonworsening"]),
            "all_metric_temperatures_within_bounds": len(temperature_values) == 10
            and np.isfinite(temperature_values.to_numpy(dtype=float)).all()
            and bool(temperature_values.between(TEMPERATURE_MIN, TEMPERATURE_MAX).all()),
            "temperature_schema_exact": temperatures.get("protocol")
            == "FOVEA_DR_POSTHOC_VARIANT_TEMPERATURE_PARAMETERS_V1"
            and temperatures.get("status") == "FITTED_ON_CALIBRATION_ONLY",
            "temperature_variant_exact": temperatures.get("variant_id") == variant_id,
            "temperature_backend_CPU_FLOAT32": temperatures.get("execution_backend") == "CPU_FLOAT32",
            "temperature_checkpoint_hash_verified": temperatures.get("selected_checkpoint_sha256")
            == checkpoint_sha,
            "temperature_protocol_hash_verified": temperatures.get("calibration_protocol_sha256")
            == protocol_sha,
            "three_grade_temperatures_exact": isinstance(grade_temperatures, dict)
            and set(grade_temperatures) == EXPECTED_GRADE_UNITS,
            "seven_lesion_temperatures_exact": isinstance(lesion_temperatures, dict)
            and set(lesion_temperatures) == set(LESION_COLUMNS),
            "all_JSON_temperatures_within_bounds": len(json_temperatures) == 10
            and all(math.isfinite(value) and TEMPERATURE_MIN <= value <= TEMPERATURE_MAX for value in json_temperatures),
            "temperature_artifact_links_verified": temperatures.get("raw_predictions_sha256")
            == actual_hashes["raw_predictions"]
            and temperatures.get("calibrated_predictions_sha256")
            == actual_hashes["calibrated_predictions"]
            and temperatures.get("metrics_sha256") == actual_hashes["metrics"],
            "completion_evidence_PASS": all_evidence_gates_pass(evidence),
            "completion_evidence_variant_exact": evidence.get("variant_id") == variant_id,
            "completion_evidence_status_exact": evidence.get("session_status")
            == "COMPLETE_AND_TEMPERATURE_FITTED",
            "completion_evidence_rows_exact": int(evidence.get("completed_rows", -1)) == EXPECTED_ROWS,
            "completion_evidence_artifact_hashes_verified": all(
                evidence.get(f"{name}_sha256") == actual_hashes[name]
                for name in artifact_paths
            ),
            "completion_evidence_state_hash_verified": evidence.get("inference_state_sha256")
            == sha256_file(state_path),
            "completion_evidence_runner_hash_verified": evidence.get("runner_source_sha256") == runner_sha,
            "completion_evidence_checkpoint_hash_verified": evidence.get("selected_checkpoint_sha256")
            == checkpoint_sha,
            "completion_evidence_no_optimizer": evidence.get("optimizer_created") is False,
            "completion_evidence_no_parameter_updates": evidence.get("model_parameters_updated") is False,
            "completion_evidence_no_VALIDATION_fit": evidence.get("validation_used_for_calibration_fit") is False,
            "completion_evidence_no_locked_test": evidence.get("locked_test_accessed") is False,
            "latest_evidence_matches_frozen_completion": latest_evidence == evidence,
            "run_root_matches_protocol": run_root == state_path.parent,
        }
    )

    grade_weighted_before = weighted_mean(grade_metrics, "NLL_before")
    grade_weighted_after = weighted_mean(grade_metrics, "NLL_after")
    lesion_weighted_before = weighted_mean(lesion_metrics, "NLL_before")
    lesion_weighted_after = weighted_mean(lesion_metrics, "NLL_after")

    grade_lookup = grade_metrics.set_index("unit")
    summary: dict[str, Any] = {
        "position": position + 1,
        "variant_id": variant_id,
        "status": "COMPLETE_AND_TEMPERATURE_FITTED",
        "execution_backend": "CPU_FLOAT32",
        "selected_epoch": int(contract["selected_epoch"]),
        "selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_sha256": checkpoint_sha,
        "completed_rows": EXPECTED_ROWS,
        "temperature_count": 10,
        "grade_weighted_NLL_before": grade_weighted_before,
        "grade_weighted_NLL_after": grade_weighted_after,
        "lesion_weighted_NLL_before": lesion_weighted_before,
        "lesion_weighted_NLL_after": lesion_weighted_after,
        "CFP_temperature": float(grade_lookup.loc["CFP", "temperature"]),
        "UWF_temperature": float(grade_lookup.loc["UWF", "temperature"]),
        "OCT_temperature": float(grade_lookup.loc["OCT", "temperature"]),
        "raw_predictions": str(raw_path),
        "raw_predictions_sha256": actual_hashes["raw_predictions"],
        "calibrated_predictions": str(calibrated_path),
        "calibrated_predictions_sha256": actual_hashes["calibrated_predictions"],
        "temperature_parameters": str(temperatures_path),
        "temperature_parameters_sha256": actual_hashes["temperature_parameters"],
        "calibration_metrics": str(metrics_path),
        "calibration_metrics_sha256": actual_hashes["metrics"],
        "inference_state": str(state_path),
        "inference_state_sha256": sha256_file(state_path),
        "completion_evidence": str(completion_evidence_path),
        "completion_evidence_sha256": sha256_file(completion_evidence_path),
        "all_checks_passed": bool(all(checks.values())),
    }
    return summary, checks


def main() -> None:
    banner("STAGE 0G-4D-5F-4D — SIX-VARIANT CALIBRATION COMPLETION FREEZE")
    if not real_drive_mounted():
        raise RuntimeError("The real Google Drive is not mounted in this notebook process.")

    section("[1] FROZEN PARENT ARTIFACTS")
    gates: dict[str, bool] = {}
    for path, expected_sha in EXPECTED_SHA256.items():
        verified = path.is_file() and sha256_file(path) == expected_sha
        gates[f"hash_verified::{path.name}"] = verified
        print(f"{path.name:<82}: exists={path.is_file()}, hash_verified={verified}")
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError("A frozen calibration parent changed:\n" + "\n".join(failed))

    protocol_sha = EXPECTED_SHA256[PROTOCOL]
    runner_sha = EXPECTED_SHA256[CALIBRATION_RUNNER]
    protocol = read_json(PROTOCOL)
    protocol_evidence = read_json(PROTOCOL_EVIDENCE)
    variant_contracts = protocol.get("variants", [])
    protocol_gates = {
        "protocol_status_locked": protocol.get("status")
        == "LOCKED_BEFORE_SIX_VARIANT_CALIBRATION_INFERENCE",
        "protocol_evidence_PASS": all_evidence_gates_pass(protocol_evidence),
        "protocol_execution_order_exact": protocol.get("execution_order") == VARIANT_IDS,
        "protocol_has_exactly_six_variant_contracts": isinstance(variant_contracts, list)
        and len(variant_contracts) == 6,
        "protocol_variant_contract_order_exact": isinstance(variant_contracts, list)
        and [item.get("variant_id") for item in variant_contracts] == VARIANT_IDS,
        "protocol_CPU_FLOAT32_exact": protocol.get("execution", {}).get("device") == "CPU"
        and protocol.get("execution", {}).get("precision") == "FLOAT32",
        "protocol_part_rows_exact": int(protocol.get("execution", {}).get("rows_per_atomic_part", -1)) == 32,
        "protocol_forbids_parameter_updates": protocol.get("calibration_partition", {}).get(
            "may_update_model_parameters"
        ) is False,
        "protocol_forbids_checkpoint_reselection": protocol.get("calibration_partition", {}).get(
            "may_select_model_checkpoint"
        ) is False,
    }
    gates.update(protocol_gates)
    for name, passed in protocol_gates.items():
        print(f"{name:<100}: {passed}")
    if not all(protocol_gates.values()):
        failed = [name for name, passed in protocol_gates.items() if not passed]
        raise RuntimeError("Calibration protocol gate failed:\n" + "\n".join(failed))

    source_action, source_sha = freeze_source()
    gates["completion_freeze_source_frozen_and_verified"] = (
        FROZEN_SOURCE.is_file() and sha256_file(FROZEN_SOURCE) == source_sha
    )

    section("[2] SIX CALIBRATION COMPLETION PAIRS")
    summaries: list[dict[str, Any]] = []
    variant_checks: dict[str, dict[str, bool]] = {}
    for position, variant_id in enumerate(VARIANT_IDS):
        matches = [
            item for item in variant_contracts
            if isinstance(item, dict) and item.get("variant_id") == variant_id
        ]
        if len(matches) != 1:
            summary = {"position": position + 1, "variant_id": variant_id}
            checks = {"unique_protocol_contract": False}
        else:
            summary, checks = inspect_variant(position, matches[0], protocol_sha, runner_sha)
            checks["unique_protocol_contract"] = True
        summaries.append(summary)
        variant_checks[variant_id] = checks
        passed = bool(checks and all(checks.values()))
        print(f"{variant_id:<76}: PASS={passed}")
        if not passed:
            for name, value in checks.items():
                if not value:
                    print(f"  FAILED::{name}")

    all_variant_checks = bool(
        len(summaries) == 6
        and all(checks and all(checks.values()) for checks in variant_checks.values())
    )
    gates.update(
        {
            f"calibration_complete::{variant_id}": bool(
                variant_checks[variant_id] and all(variant_checks[variant_id].values())
            )
            for variant_id in VARIANT_IDS
        }
    )
    gates.update(
        {
            "exactly_six_completion_summaries": len(summaries) == 6,
            "summary_order_matches_protocol": [item.get("variant_id") for item in summaries]
            == VARIANT_IDS,
            "all_six_variants_complete": all_variant_checks,
            "all_six_statuses_complete_and_temperature_fitted": all(
                item.get("status") == "COMPLETE_AND_TEMPERATURE_FITTED" for item in summaries
            ),
            "all_six_have_2840_rows": all(item.get("completed_rows") == EXPECTED_ROWS for item in summaries),
            "total_calibration_rows_equal_17040": sum(
                int(item.get("completed_rows", 0)) for item in summaries
            ) == EXPECTED_TOTAL_ROWS,
            "all_six_have_ten_temperatures": all(item.get("temperature_count") == 10 for item in summaries),
            "six_distinct_selected_checkpoints": len(
                {item.get("selected_checkpoint_sha256") for item in summaries}
            ) == 6,
            "six_distinct_raw_prediction_artifacts": len(
                {item.get("raw_predictions_sha256") for item in summaries}
            ) == 6,
            "six_distinct_calibrated_prediction_artifacts": len(
                {item.get("calibrated_predictions_sha256") for item in summaries}
            ) == 6,
            "six_distinct_temperature_artifacts": len(
                {item.get("temperature_parameters_sha256") for item in summaries}
            ) == 6,
            "all_grade_weighted_NLL_values_nonworsening": all(
                float(item.get("grade_weighted_NLL_after", math.inf))
                <= float(item.get("grade_weighted_NLL_before", -math.inf)) + 1e-12
                for item in summaries
            ),
            "all_lesion_weighted_NLL_values_nonworsening": all(
                float(item.get("lesion_weighted_NLL_after", math.inf))
                <= float(item.get("lesion_weighted_NLL_before", -math.inf)) + 1e-12
                for item in summaries
            ),
            "model_not_loaded_this_stage": True,
            "images_not_accessed_this_stage": True,
            "optimizer_not_created_this_stage": True,
            "parameters_not_updated_this_stage": True,
            "VALIDATION_not_used_for_calibration_fit_this_stage": True,
            "locked_test_not_accessed_this_stage": True,
        }
    )

    section("[3] SIX-VARIANT CALIBRATION COMPLETION GATES")
    for name, passed in gates.items():
        print(f"{name:<108}: {passed}")
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError("Six-variant calibration completion gate failed:\n" + "\n".join(failed))

    registry_frame = pd.DataFrame(summaries)
    registry_action = frozen_csv(REGISTRY, registry_frame)
    registry_sha = sha256_file(REGISTRY)

    contract_payload = {
        "protocol": "FOVEA_DR_POSTHOC_SIX_VARIANT_CALIBRATION_COMPLETION_V1",
        "stage": "0G-4D-5F-4D",
        "status": "COMPLETE_AND_FROZEN",
        "execution_order": VARIANT_IDS,
        "execution_backend": "CPU_FLOAT32",
        "calibration_rows_per_variant": EXPECTED_ROWS,
        "total_calibration_rows": EXPECTED_TOTAL_ROWS,
        "temperatures_per_variant": 10,
        "calibration_protocol": str(PROTOCOL),
        "calibration_protocol_sha256": protocol_sha,
        "calibration_runner": str(CALIBRATION_RUNNER),
        "calibration_runner_sha256": runner_sha,
        "completion_freeze_source": str(FROZEN_SOURCE),
        "completion_freeze_source_sha256": source_sha,
        "registry": str(REGISTRY),
        "registry_sha256": registry_sha,
        "variants": summaries,
        "model_parameters_updated": False,
        "checkpoint_reselection_performed": False,
        "validation_used_for_calibration_fit": False,
        "locked_test_accessed": False,
        "next_required_stage": "GENERATE_SIX_VARIANT_POSTHOC_COMPARISON_OUTPUTS",
    }
    contract_action = frozen_json(CONTRACT, contract_payload)
    contract_sha = sha256_file(CONTRACT)

    evidence_payload = {
        "stage": "0G-4D-5F-4D",
        "status": "PASS",
        "completion_status": "SIX_VARIANT_CALIBRATION_COMPLETE_AND_FROZEN",
        "calibration_protocol": str(PROTOCOL),
        "calibration_protocol_sha256": protocol_sha,
        "calibration_runner": str(CALIBRATION_RUNNER),
        "calibration_runner_sha256": runner_sha,
        "completion_freeze_source": str(FROZEN_SOURCE),
        "completion_freeze_source_sha256": source_sha,
        "registry": str(REGISTRY),
        "registry_sha256": registry_sha,
        "completion_contract": str(CONTRACT),
        "completion_contract_sha256": contract_sha,
        "variant_order": VARIANT_IDS,
        "variant_checks": variant_checks,
        "gates": {name: bool(value) for name, value in gates.items()},
        "model_loaded": False,
        "images_accessed": False,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "validation_used_for_calibration_fit": False,
        "locked_test_accessed": False,
        "next_required_stage": "GENERATE_SIX_VARIANT_POSTHOC_COMPARISON_OUTPUTS",
    }
    evidence_action = frozen_json(EVIDENCE, evidence_payload)
    evidence_sha = sha256_file(EVIDENCE)
    os.sync()

    section("[4] REGISTRY, CONTRACT, SOURCE, AND EVIDENCE FREEZE")
    print("Registry action   :", registry_action)
    print("Registry          :", REGISTRY)
    print("Registry SHA-256  :", registry_sha)
    print("Contract action   :", contract_action)
    print("Contract          :", CONTRACT)
    print("Contract SHA-256  :", contract_sha)
    print("Source action     :", source_action)
    print("Source            :", FROZEN_SOURCE)
    print("Source SHA-256    :", source_sha)
    print("Evidence action   :", evidence_action)
    print("Evidence          :", EVIDENCE)
    print("Evidence SHA-256  :", evidence_sha)

    banner("STAGE 0G-4D-5F-4D STATUS: PASS")
    print("Six calibrations complete : YES")
    print("Total calibration rows    :", EXPECTED_TOTAL_ROWS)
    print("Temperatures fitted       : 60 / 60")
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
        banner("STAGE 0G-4D-5F-4D STATUS: FAIL")
        raise
