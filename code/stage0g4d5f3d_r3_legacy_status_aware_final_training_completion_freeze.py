#!/usr/bin/env python3
"""Freeze the completed final-training results of all six post-hoc variants."""

from __future__ import annotations

from google.colab import drive
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
from typing import Any


WIDTH = 124
ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
RUNNER = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f3a_resumable_final_variant_training.py"
FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f3d_r3_legacy_status_aware_final_training_completion_freeze.py"
REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_final_training_registry_v1.csv"
COMPLETION_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_final_training_completion_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5f3d_six_variant_final_training_completion_freeze.json"
RUN_BASE = ROOT / "Runs/FOVEA_DR/posthoc_comparisons/final_seed2026"
CHECKPOINT_BASE = ROOT / "Checkpoints/FOVEA_DR/posthoc_comparisons/final_seed2026"

PROTOCOL_SHA = "f946964923e08722dbb486aa3782b08880446040e07fb6b268ea9c28c08e6ae1"
RUNNER_SHA = "b3bc733b2204a1657a6d1f62226fb13f04bc3c61d953334bfd04329fc62e3495"

VARIANTS = [
    {
        "variant_id": "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
        "family": "PLAIN_COMPARATOR",
        "backbone": "EfficientNetV2-S",
        "contract": ROOT / "Metadata/mmrdr_fovea_dr_posthoc_B1_completion_v1.json",
        "contract_sha256": "3a67bf812f1d0dc2b7eb1089cc9a0f3327f8e8f9b8d7f86c007f14429ce26fa4",
        "evidence": ROOT / "Evidence/stage0g4d5f3c_b1_completion_freeze.json",
        "evidence_sha256": "e965db2f56b3d63e101208eacf3fba9969df0921b458605d1a812df68c98b474",
        "completion_epoch": 25,
        "global_optimizer_step": 10375,
        "best_epoch": 19,
        "best_qwk": 0.9196794652059448,
        "best_nll": 0.7672031672199291,
        "completion_qwk": 0.9214882407737576,
        "AMP_scale": 4096.0,
        "AMP_growth_interval": 1000000,
    },
    {
        "variant_id": "B2_RESNET50_MULTITASK",
        "family": "PLAIN_COMPARATOR",
        "backbone": "ResNet-50",
        "contract": ROOT / "Metadata/mmrdr_fovea_dr_posthoc_B2_completion_v1.json",
        "contract_sha256": "19171fa0bb531407424bce818395f4eb05bf479d02cd9aabb7d9c005d24284fd",
        "evidence": ROOT / "Evidence/stage0g4d5f3c_b2_completion_freeze.json",
        "evidence_sha256": "2a17b8b516ec7aebdb256be8f5fe8e587f5baf43f3aceb33345f93d5b8b2c116",
        "completion_epoch": 19,
        "global_optimizer_step": 7885,
        "best_epoch": 13,
        "best_qwk": 0.9039859173852121,
        "best_nll": 0.6623852801682953,
        "completion_qwk": 0.9053049215585331,
        "AMP_scale": 4096.0,
        "AMP_growth_interval": 1000000,
    },
    {
        "variant_id": "B3_CONVNEXT_TINY_MULTITASK",
        "family": "PLAIN_COMPARATOR",
        "backbone": "ConvNeXt-Tiny",
        "contract": ROOT / "Metadata/mmrdr_fovea_dr_posthoc_B3_completion_v1.json",
        "contract_sha256": "146cf297e679675f86f544bb243a8be134b893ba1ece94300c4ebf2184d9b3ca",
        "evidence": ROOT / "Evidence/stage0g4d5f3c_b3_completion_freeze.json",
        "evidence_sha256": "55152a9583d9d7fa1ac05b42850b213e78da0245c594bb7541699ff62054cb7b",
        "completion_epoch": 27,
        "global_optimizer_step": 11205,
        "best_epoch": 21,
        "best_qwk": 0.9187861201162492,
        "best_nll": 0.7257978542928547,
        "completion_qwk": 0.9165972446011296,
        "AMP_scale": 1024.0,
        "AMP_growth_interval": 1000000,
    },
    {
        "variant_id": "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
        "family": "FOVEA_ABLATION",
        "backbone": "EfficientNetV2-S",
        "contract": ROOT / "Metadata/mmrdr_fovea_dr_posthoc_A1_completion_v1.json",
        "contract_sha256": "5a02ba2276e1aa8366e65782da06078a22ac3fa67528aeefee08beb7354a2097",
        "evidence": ROOT / "Evidence/stage0g4d5f3c_a1_completion_freeze.json",
        "evidence_sha256": "503a78243a41157a54f63582c62045789c742d32f6b894c54568fa4db2d02d5e",
        "completion_epoch": 16,
        "global_optimizer_step": 6640,
        "best_epoch": 10,
        "best_qwk": 0.9216415913851501,
        "best_nll": 0.5742359435180183,
        "completion_qwk": 0.9218392538599391,
        "AMP_scale": 4096.0,
        "AMP_growth_interval": 1000000,
    },
    {
        "variant_id": "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
        "family": "FOVEA_ABLATION",
        "backbone": "EfficientNetV2-S",
        "contract": ROOT / "Metadata/mmrdr_fovea_dr_posthoc_A2_completion_v1.json",
        "contract_sha256": "f4acb32413bd1c0b5fe043abbcb388b097246b53a1e786bf459490c16bc1a93e",
        "evidence": ROOT / "Evidence/stage0g4d5f3c_a2_completion_freeze.json",
        "evidence_sha256": "829b7e4686bcaf92b24dbe6432ef2e22abb5698fd30e905ad9121d40a736ee35",
        "completion_epoch": 14,
        "global_optimizer_step": 5810,
        "best_epoch": 8,
        "best_qwk": 0.9170038335963082,
        "best_nll": 0.5915676464916253,
        "completion_qwk": 0.9144422059356694,
        "AMP_scale": 4096.0,
        "AMP_growth_interval": 1000000,
    },
    {
        "variant_id": "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
        "family": "FOVEA_ABLATION",
        "backbone": "EfficientNetV2-S",
        "contract": ROOT / "Metadata/mmrdr_fovea_dr_posthoc_A3_completion_v1.json",
        "contract_sha256": "e09ffb8e5c8a643e8979def4b84a18be773df572bf3d21d93838195a12df3bdc",
        "evidence": ROOT / "Evidence/stage0g4d5f3c_a3_completion_freeze.json",
        "evidence_sha256": "7285f7444782514496d0f303241f2d3514bdc5228b0a2eeb2bcd9eaecb666703",
        "completion_epoch": 23,
        "global_optimizer_step": 9545,
        "best_epoch": 17,
        "best_qwk": 0.9234807937342082,
        "best_nll": 0.6138728559156931,
        "completion_qwk": 0.9220844652880966,
        "AMP_scale": 4096.0,
        "AMP_growth_interval": 1000000,
    },
]

REGISTRY_FIELDS = [
    "variant_order",
    "variant_id",
    "family",
    "backbone",
    "status",
    "contract_status_value",
    "contract_status_source",
    "completion_epoch",
    "global_optimizer_step",
    "selected_best_epoch",
    "best_mean_fundus_QWK",
    "best_mean_fundus_NLL",
    "completion_mean_fundus_QWK",
    "bad_epochs",
    "early_stopped",
    "AMP_scale",
    "AMP_growth_interval",
    "last_checkpoint_sha256",
    "best_checkpoint_sha256",
    "history_sha256",
    "step_log_sha256",
    "best_validation_predictions_sha256",
    "completion_validation_predictions_sha256",
    "run_state_sha256",
    "completion_contract_sha256",
    "completion_evidence_sha256",
    "all_completion_checks_passed",
]

MISSING = object()


def banner(text: str) -> None:
    print("=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def section(text: str) -> None:
    print(f"\n{text}")
    print("-" * WIDTH)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _recursive_values(payload: Any, key: str) -> list[Any]:
    """Return values for an exact key anywhere in a JSON-compatible tree."""
    found: list[Any] = []
    if isinstance(payload, dict):
        for current_key, value in payload.items():
            if current_key == key:
                found.append(value)
            found.extend(_recursive_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_recursive_values(value, key))
    return found


def resolve_value(payload: dict[str, Any], aliases: tuple[str, ...], default: Any = MISSING) -> Any:
    """Resolve a schema alias without silently accepting contradictory values."""
    for alias in aliases:
        if alias in payload:
            return payload[alias]
    for alias in aliases:
        values = _recursive_values(payload, alias)
        if not values:
            continue
        encoded: dict[str, Any] = {}
        for value in values:
            marker = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
            encoded[marker] = value
        if len(encoded) == 1:
            return next(iter(encoded.values()))
    return default


def as_int(value: Any) -> int | None:
    if value is MISSING or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def as_float(value: Any) -> float | None:
    if value is MISSING or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"TRUE", "YES", "PASS", "COMPLETE", "COMPLETED"}:
            return True
        if normalized in {"FALSE", "NO", "FAIL", "FAILED", "INCOMPLETE"}:
            return False
    if value in (0, 1):
        return bool(value)
    return None


def valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def close_to(value: float | None, expected: float) -> bool:
    return value is not None and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def optional_text_matches(value: Any, expected: str) -> bool:
    return value is MISSING or str(value).strip().upper() == expected.upper()


def completion_status_not_contradictory(value: Any) -> bool:
    """Accept positive legacy freeze statuses while rejecting incomplete states."""
    if value is MISSING:
        return True
    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    negative_markers = (
        "FAIL", "ERROR", "INCOMPLETE", "RESUMABLE", "IN_PROGRESS",
        "PENDING", "ABORT", "CANCEL",
    )
    if any(marker in normalized for marker in negative_markers):
        return False
    positive_markers = ("PASS", "COMPLETE", "COMPLETED", "FROZEN", "VERIFIED")
    return any(marker in normalized for marker in positive_markers)


def optional_int_matches(value: Any, expected: int) -> bool:
    return value is MISSING or as_int(value) == expected


def optional_float_matches(value: Any, expected: float) -> bool:
    return value is MISSING or close_to(as_float(value), expected)


def optional_bool_matches(value: Any, expected: bool) -> bool:
    return value is MISSING or as_bool(value) is expected


def optional_sha_matches(value: Any, expected: str) -> bool:
    return value is MISSING or value == expected


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_amp_state(path: Path) -> tuple[float | None, int | None, bool]:
    """Read only the scaler metadata from a trusted frozen checkpoint on CPU."""
    import gc
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    scaler = None
    for key in ("scaler_state_dict", "grad_scaler_state_dict", "scaler_state"):
        candidate = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(candidate, dict):
            scaler = candidate
            break
    if scaler is None:
        del checkpoint
        gc.collect()
        return None, None, False

    scale = as_float(scaler.get("scale", MISSING))
    interval = as_int(scaler.get("growth_interval", MISSING))
    cpu_only = True
    for value in scaler.values():
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            cpu_only = False
            break

    del scaler
    del checkpoint
    gc.collect()
    return scale, interval, cpu_only


def frozen_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return "CREATED_AND_FROZEN"


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return frozen_bytes(path, data)


def frozen_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return frozen_bytes(path, stream.getvalue().encode("utf-8"))


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different 5F-3D source is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def inspect_variant(position: int, declared: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    variant_id = declared["variant_id"]
    run_root = RUN_BASE / variant_id
    checkpoint_root = CHECKPOINT_BASE / variant_id
    paths = {
        "contract": declared["contract"],
        "evidence": declared["evidence"],
        "state": run_root / "run_state.json",
        "history": run_root / "epoch_history.csv",
        "steps": run_root / "optimizer_step_log.csv",
        "last": checkpoint_root / "last_resume.pt",
        "best": checkpoint_root / "best_model.pt",
        "latest_predictions": run_root / "latest_validation_predictions.csv",
        "best_predictions": run_root / "best_validation_predictions.csv",
    }
    checks: dict[str, bool] = {
        f"exists::{name}": path.is_file() for name, path in paths.items()
    }
    if not all(checks.values()):
        return {}, checks

    contract = read_json(paths["contract"])
    evidence = read_json(paths["evidence"])
    state = read_json(paths["state"])
    history = csv_rows(paths["history"])
    steps = csv_rows(paths["steps"])

    actual_hashes = {
        name: sha256_file(paths[name])
        for name in (
            "contract", "evidence", "state", "history", "steps", "last", "best",
            "latest_predictions", "best_predictions",
        )
    }

    evidence_gates = evidence.get("gates", {})
    evidence_gate_pass = (
        isinstance(evidence_gates, dict)
        and bool(evidence_gates)
        and all(value is True for value in evidence_gates.values())
    )

    completion_epoch = as_int(state.get("last_completed_epoch", MISSING))
    global_step = as_int(state.get("global_optimizer_step", MISSING))
    best_epoch = as_int(state.get("best_epoch", MISSING))
    best_qwk = as_float(state.get("best_mean_fundus_QWK", MISSING))
    best_nll = as_float(state.get("best_mean_fundus_NLL", MISSING))
    bad_epochs = as_int(state.get("bad_epochs", MISSING))
    early_stopped = as_bool(state.get("early_stopped", MISSING))

    history_epochs = [as_int(row.get("epoch", MISSING)) for row in history]
    step_numbers = [as_int(row.get("global_optimizer_step", MISSING)) for row in steps]
    completion_row = history[-1] if history else {}
    best_rows = [
        row for row in history if as_int(row.get("epoch", MISSING)) == best_epoch
    ]
    best_row = best_rows[0] if len(best_rows) == 1 else {}
    completion_qwk = as_float(completion_row.get("mean_fundus_QWK", MISSING))
    completion_step = as_int(completion_row.get("global_optimizer_step", MISSING))
    history_best_qwk = as_float(best_row.get("mean_fundus_QWK", MISSING))
    history_best_nll = as_float(best_row.get("mean_fundus_NLL", MISSING))

    short_id = variant_id.split("_", 1)[0]
    expected_contract_id = f"MMRDR_FOVEA_DR_POSTHOC_{short_id}_COMPLETION_V1"
    contract_id = resolve_value(contract, ("contract_id", "completion_contract_id", "artifact_id"))
    contract_variant = resolve_value(contract, ("variant_id", "variant"))
    contract_status = resolve_value(
        contract,
        ("status", "training_status", "run_status", "completion_status"),
    )
    contract_epoch = resolve_value(
        contract,
        ("last_completed_epoch", "completion_epoch", "completed_epoch", "epoch_completed"),
    )
    contract_step = resolve_value(
        contract,
        ("global_optimizer_step", "global_step", "optimizer_step"),
    )
    contract_best_epoch = resolve_value(contract, ("selected_best_epoch", "best_epoch"))
    contract_best_qwk = resolve_value(
        contract,
        (
            "best_mean_fundus_QWK", "best_mean_fundus_qwk",
            "selected_best_mean_fundus_QWK", "selected_best_QWK", "best_QWK",
        ),
    )
    contract_best_nll = resolve_value(
        contract,
        (
            "best_mean_fundus_NLL", "best_mean_fundus_nll",
            "selected_best_mean_fundus_NLL", "selected_best_NLL", "best_NLL",
        ),
    )
    contract_completion_qwk = resolve_value(
        contract,
        (
            "completion_mean_fundus_QWK", "completion_mean_fundus_qwk",
            "completion_QWK", "final_mean_fundus_QWK", "last_mean_fundus_QWK",
        ),
    )
    contract_bad_epochs = resolve_value(contract, ("bad_epochs", "final_bad_epochs"))
    contract_early_stopped = resolve_value(
        contract,
        ("early_stopped", "early_stopping_triggered", "early_stop_triggered"),
    )
    contract_runner_sha = resolve_value(
        contract,
        ("runner_source_sha256", "runner_sha256", "final_training_runner_sha256"),
    )
    contract_last_sha = resolve_value(contract, ("last_checkpoint_sha256", "last_resume_sha256"))
    contract_best_sha = resolve_value(contract, ("best_checkpoint_sha256", "best_model_sha256"))
    contract_history_sha = resolve_value(contract, ("history_sha256", "epoch_history_sha256"))
    contract_steps_sha = resolve_value(contract, ("step_log_sha256", "optimizer_step_log_sha256"))
    contract_best_predictions_sha = resolve_value(
        contract,
        ("best_validation_predictions_sha256", "best_predictions_sha256"),
    )
    contract_completion_predictions_sha = resolve_value(
        contract,
        (
            "completion_validation_predictions_sha256",
            "latest_validation_predictions_sha256",
            "completion_predictions_sha256",
            "latest_predictions_sha256",
        ),
    )
    contract_amp_scale = resolve_value(contract, ("AMP_scale", "amp_scale"))
    contract_amp_interval = resolve_value(
        contract,
        ("AMP_growth_interval", "amp_growth_interval", "growth_interval"),
    )

    amp_scale = as_float(contract_amp_scale)
    amp_interval = as_int(contract_amp_interval)
    checkpoint_amp_read = False
    checkpoint_amp_cpu_only = True
    if amp_scale is None or amp_interval is None:
        checkpoint_amp_read = True
        checkpoint_scale, checkpoint_interval, checkpoint_amp_cpu_only = checkpoint_amp_state(paths["last"])
        if amp_scale is None:
            amp_scale = checkpoint_scale
        if amp_interval is None:
            amp_interval = checkpoint_interval

    contract_validation_used = resolve_value(
        contract,
        (
            "validation_used_for_model_selection",
            "validation_used_only_for_selection",
            "VALIDATION_used_only_for_selection",
        ),
    )
    contract_calibration_accessed = resolve_value(
        contract,
        ("calibration_accessed", "calibration_used", "CALIBRATION_accessed", "CALIBRATION_used"),
    )
    contract_locked_accessed = resolve_value(
        contract,
        ("locked_test_accessed", "locked_test_used"),
    )

    evidence_status = resolve_value(evidence, ("status", "evidence_status"))
    evidence_variant = resolve_value(evidence, ("variant_id", "variant"))
    evidence_images_accessed = resolve_value(evidence, ("images_accessed",))
    evidence_parameters_updated = resolve_value(evidence, ("parameters_updated",))
    evidence_calibration_accessed = resolve_value(
        evidence,
        ("calibration_accessed", "calibration_used", "CALIBRATION_accessed"),
    )
    evidence_locked_accessed = resolve_value(
        evidence,
        ("locked_test_accessed", "locked_test_used"),
    )

    checks.update({
        "completion_contract_hash_verified": actual_hashes["contract"] == declared["contract_sha256"],
        "completion_evidence_hash_verified": actual_hashes["evidence"] == declared["evidence_sha256"],
        "contract_id_exact_or_legacy_absent": contract_id is MISSING or contract_id == expected_contract_id,
        "contract_variant_exact": contract_variant == variant_id,
        "contract_status_not_contradictory": completion_status_not_contradictory(contract_status),
        "contract_completion_epoch_not_contradictory": optional_int_matches(contract_epoch, declared["completion_epoch"]),
        "contract_global_step_not_contradictory": optional_int_matches(contract_step, declared["global_optimizer_step"]),
        "contract_best_epoch_not_contradictory": optional_int_matches(contract_best_epoch, declared["best_epoch"]),
        "contract_best_QWK_not_contradictory": optional_float_matches(contract_best_qwk, declared["best_qwk"]),
        "contract_best_NLL_not_contradictory": optional_float_matches(contract_best_nll, declared["best_nll"]),
        "contract_completion_QWK_not_contradictory": optional_float_matches(contract_completion_qwk, declared["completion_qwk"]),
        "contract_bad_epochs_not_contradictory": optional_int_matches(contract_bad_epochs, 6),
        "contract_early_stopped_not_contradictory": optional_bool_matches(contract_early_stopped, True),
        "contract_runner_hash_not_contradictory": optional_sha_matches(contract_runner_sha, RUNNER_SHA),
        "contract_last_hash_not_contradictory": optional_sha_matches(contract_last_sha, actual_hashes["last"]),
        "contract_best_hash_not_contradictory": optional_sha_matches(contract_best_sha, actual_hashes["best"]),
        "contract_history_hash_not_contradictory": optional_sha_matches(contract_history_sha, actual_hashes["history"]),
        "contract_step_log_hash_not_contradictory": optional_sha_matches(contract_steps_sha, actual_hashes["steps"]),
        "contract_best_predictions_hash_not_contradictory": optional_sha_matches(
            contract_best_predictions_sha, actual_hashes["best_predictions"]
        ),
        "contract_completion_predictions_hash_not_contradictory": optional_sha_matches(
            contract_completion_predictions_sha, actual_hashes["latest_predictions"]
        ),
        "contract_validation_policy_not_contradictory": optional_bool_matches(contract_validation_used, True),
        "contract_CALIBRATION_policy_not_contradictory": optional_bool_matches(contract_calibration_accessed, False),
        "contract_locked_test_policy_not_contradictory": optional_bool_matches(contract_locked_accessed, False),
        "evidence_status_PASS": str(evidence_status).strip().upper() == "PASS",
        "evidence_variant_exact": evidence_variant == variant_id,
        "all_completion_evidence_gates_PASS": evidence_gate_pass,
        "evidence_reports_no_image_access": as_bool(evidence_images_accessed) is False,
        "evidence_reports_no_parameter_update": as_bool(evidence_parameters_updated) is False,
        "evidence_reports_CALIBRATION_not_accessed": as_bool(evidence_calibration_accessed) is False,
        "evidence_reports_locked_test_not_accessed": as_bool(evidence_locked_accessed) is False,
        "run_state_variant_exact": state.get("variant_id") == variant_id,
        "run_state_status_COMPLETE": state.get("status") == "COMPLETE",
        "run_state_completion_epoch_exact": completion_epoch == declared["completion_epoch"],
        "run_state_global_step_exact": global_step == declared["global_optimizer_step"],
        "run_state_best_epoch_exact": best_epoch == declared["best_epoch"],
        "run_state_best_QWK_exact": close_to(best_qwk, declared["best_qwk"]),
        "run_state_best_NLL_exact": close_to(best_nll, declared["best_nll"]),
        "run_state_bad_epochs_is_6": bad_epochs == 6,
        "run_state_early_stopped": early_stopped is True,
        "run_state_runner_hash_verified": state.get("runner_source_sha256") == RUNNER_SHA,
        "run_state_validation_used_for_selection": state.get("validation_used_for_model_selection") is True,
        "run_state_CALIBRATION_not_used": state.get("calibration_used") is False,
        "run_state_locked_test_not_used": state.get("locked_test_used") is False,
        "history_has_exact_completion_epochs": len(history) == declared["completion_epoch"],
        "history_epochs_contiguous": history_epochs == list(range(1, declared["completion_epoch"] + 1)),
        "history_completion_step_exact": completion_step == declared["global_optimizer_step"],
        "history_completion_QWK_exact": close_to(completion_qwk, declared["completion_qwk"]),
        "history_has_one_selected_best_epoch": len(best_rows) == 1,
        "history_best_QWK_matches_state": close_to(history_best_qwk, declared["best_qwk"]),
        "history_best_NLL_matches_state": close_to(history_best_nll, declared["best_nll"]),
        "step_log_has_exact_rows": len(steps) == declared["global_optimizer_step"],
        "step_log_contiguous": step_numbers == list(range(1, declared["global_optimizer_step"] + 1)),
        "last_checkpoint_matches_run_state": state.get("last_checkpoint_sha256") == actual_hashes["last"],
        "best_checkpoint_matches_run_state": state.get("best_checkpoint_sha256") == actual_hashes["best"],
        "history_matches_run_state": state.get("history_sha256") == actual_hashes["history"],
        "step_log_matches_run_state": state.get("step_log_sha256") == actual_hashes["steps"],
        "latest_predictions_match_run_state": (
            state.get("latest_validation_predictions_sha256") == actual_hashes["latest_predictions"]
        ),
        "best_predictions_match_run_state": (
            state.get("best_validation_predictions_sha256") == actual_hashes["best_predictions"]
        ),
        "AMP_scale_matches_frozen_completion": close_to(amp_scale, declared["AMP_scale"]),
        "AMP_growth_interval_matches_frozen_completion": amp_interval == declared["AMP_growth_interval"],
        "checkpoint_AMP_loaded_on_CPU_only_if_required": (
            not checkpoint_amp_read or checkpoint_amp_cpu_only
        ),
    })

    passed = all(checks.values())
    if not passed:
        return {}, checks

    normalized_contract_status = (
        "FIELD_ABSENT" if contract_status is MISSING else str(contract_status)
    )
    status_source = (
        "EXPLICIT_COMPLETE_STATUS_AND_CANONICAL_RUN_STATE"
        if contract_status is not MISSING
        and str(contract_status).strip().upper() in {"COMPLETE", "COMPLETED"}
        else "POSITIVE_LEGACY_STATUS_AND_CANONICAL_RUN_STATE"
        if contract_status is not MISSING
        else "CANONICAL_RUN_STATE_WITH_HASH_LOCKED_LEGACY_CONTRACT"
    )
    summary = {
        "variant_order": position,
        "variant_id": variant_id,
        "family": declared["family"],
        "backbone": declared["backbone"],
        "status": "COMPLETE",
        "contract_status_value": normalized_contract_status,
        "contract_status_source": status_source,
        "completion_epoch": completion_epoch,
        "global_optimizer_step": global_step,
        "selected_best_epoch": best_epoch,
        "best_mean_fundus_QWK": best_qwk,
        "best_mean_fundus_NLL": best_nll,
        "completion_mean_fundus_QWK": completion_qwk,
        "bad_epochs": bad_epochs,
        "early_stopped": early_stopped,
        "AMP_scale": amp_scale,
        "AMP_growth_interval": amp_interval,
        "last_checkpoint_sha256": actual_hashes["last"],
        "best_checkpoint_sha256": actual_hashes["best"],
        "history_sha256": actual_hashes["history"],
        "step_log_sha256": actual_hashes["steps"],
        "best_validation_predictions_sha256": actual_hashes["best_predictions"],
        "completion_validation_predictions_sha256": actual_hashes["latest_predictions"],
        "run_state_sha256": actual_hashes["state"],
        "completion_contract_sha256": actual_hashes["contract"],
        "completion_evidence_sha256": actual_hashes["evidence"],
        "all_completion_checks_passed": True,
    }
    return summary, checks


def main() -> None:
    banner("STAGE 0G-4D-5F-3D-R3 — LEGACY-STATUS-AWARE FINAL-TRAINING COMPLETION FREEZE")

    if not ROOT.exists():
        raise RuntimeError(
            "The real Google Drive must be mounted in the Colab notebook process "
            "before this script is launched."
        )
    mount_type = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", "/content/drive"],
        text=True,
        capture_output=True,
    ).stdout.strip().lower()
    if not ROOT.exists() or "drive" not in mount_type:
        raise RuntimeError("The real Google Drive is not mounted. Run the safe Drive recovery cell first.")

    section("[1] FROZEN PROTOCOL AND TRAINING RUNNER")
    gates: dict[str, bool] = {
        "comparison_protocol_hash_verified": PROTOCOL.is_file() and sha256_file(PROTOCOL) == PROTOCOL_SHA,
        "final_training_runner_hash_verified": RUNNER.is_file() and sha256_file(RUNNER) == RUNNER_SHA,
    }
    for name, passed in gates.items():
        print(f"{name:<90}: {passed}")
    if not all(gates.values()):
        raise RuntimeError("A frozen protocol or final-training runner changed.")

    protocol = read_json(PROTOCOL)
    protocol_ids = [item["id"] for item in protocol["comparators"]]
    protocol_ids += [item["id"] for item in protocol["ablations"]]
    declared_ids = [item["variant_id"] for item in VARIANTS]
    gates.update({
        "six_variants_match_protocol_order": declared_ids == protocol_ids,
        "CALIBRATION_allowed_only_after_selection": protocol["data_access_policy"]["CALIBRATION"].startswith("ALLOWED_ONLY_AFTER"),
        "locked_test_prohibited_for_new_variants": protocol["data_access_policy"]["LOCKED_INTERNAL_TEST"].startswith("PROHIBITED"),
        "report_all_six_variants_required": protocol["reporting_policy"]["report_all_six_frozen_runs"] is True,
    })

    section("[2] SIX FINAL-TRAINING COMPLETION PAIRS")
    summaries: list[dict[str, Any]] = []
    variant_checks: dict[str, dict[str, bool]] = {}
    for position, declared in enumerate(VARIANTS, start=1):
        summary, checks = inspect_variant(position, declared)
        variant_id = declared["variant_id"]
        variant_checks[variant_id] = checks
        passed = bool(checks) and all(checks.values())
        gates[f"final_training_complete::{variant_id}"] = passed
        print(f"{variant_id:<78}: PASS={passed}")
        if not passed:
            for check_name, check_passed in checks.items():
                if not check_passed:
                    print(f"  FAILED::{check_name}")
        if summary:
            summaries.append(summary)

    exactly_six = len(summaries) == 6
    gates.update({
        "exactly_six_completion_summaries": len(summaries) == 6,
        "registry_order_matches_protocol": [row["variant_id"] for row in summaries] == protocol_ids,
        "three_plain_comparators_complete": sum(row["family"] == "PLAIN_COMPARATOR" for row in summaries) == 3,
        "three_FOVEA_ablations_complete": sum(row["family"] == "FOVEA_ABLATION" for row in summaries) == 3,
        "all_statuses_COMPLETE": exactly_six and all(row["status"] == "COMPLETE" for row in summaries),
        "all_runs_early_stopped_at_patience_6": exactly_six and all(
            row["early_stopped"] and row["bad_epochs"] == 6 for row in summaries
        ),
        "six_distinct_best_checkpoints": len({row["best_checkpoint_sha256"] for row in summaries}) == 6,
        "six_distinct_last_checkpoints": len({row["last_checkpoint_sha256"] for row in summaries}) == 6,
        "all_completion_checks_passed": exactly_six and all(
            row["all_completion_checks_passed"] for row in summaries
        ),
        "CALIBRATION_not_accessed_during_training": exactly_six,
        "locked_test_not_accessed_by_new_variants": exactly_six,
        "model_not_instantiated_this_stage": True,
        "checkpoint_metadata_loaded_on_CPU_only": True,
        "images_not_accessed_this_stage": True,
        "optimizer_not_created_this_stage": True,
        "parameters_not_updated_this_stage": True,
    })

    section("[3] SIX-VARIANT FINAL-TRAINING COMPLETION GATES")
    for name, passed in gates.items():
        print(f"{name:<98}: {passed}")
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError("Six-variant final-training completion gate failed:\n" + "\n".join(failed))

    section("[4] REGISTRY, CONTRACT, SOURCE, AND EVIDENCE FREEZE")
    registry_action = frozen_csv(REGISTRY, summaries)
    registry_sha = sha256_file(REGISTRY)
    source_action, source_sha = freeze_source()

    completion_payload = {
        "contract_id": "MMRDR_FOVEA_DR_POSTHOC_FINAL_TRAINING_COMPLETION_V1",
        "stage": "0G-4D-5F-3D-R3",
        "status": "SIX_VARIANT_FINAL_TRAINING_COMPLETE",
        "comparison_protocol_path": str(PROTOCOL),
        "comparison_protocol_sha256": PROTOCOL_SHA,
        "final_training_runner_path": str(RUNNER),
        "final_training_runner_sha256": RUNNER_SHA,
        "final_training_registry_path": str(REGISTRY),
        "final_training_registry_sha256": registry_sha,
        "variants_in_locked_order": protocol_ids,
        "all_six_variants_complete": True,
        "all_best_epochs_selected_on_VALIDATION": True,
        "calibration_may_now_begin": True,
        "calibration_dataset": "CALIBRATION",
        "locked_test_access_for_new_variants": "PROHIBITED",
        "existing_primary_FOVEA_DR_locked_test_result_remains_unchanged": True,
        "variant_completion_contract_sha256": {
            row["variant_id"]: row["completion_contract_sha256"] for row in summaries
        },
        "variant_completion_evidence_sha256": {
            row["variant_id"]: row["completion_evidence_sha256"] for row in summaries
        },
    }
    completion_action = frozen_json(COMPLETION_CONTRACT, completion_payload)
    completion_sha = sha256_file(COMPLETION_CONTRACT)

    gates["completion_source_frozen_and_verified"] = sha256_file(FROZEN_SOURCE) == source_sha
    evidence_payload = {
        "stage": "0G-4D-5F-3D-R3",
        "status": "PASS",
        "purpose": "SIX_VARIANT_FINAL_TRAINING_COMPLETION_FREEZE",
        "source_path": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "registry_path": str(REGISTRY),
        "registry_sha256": registry_sha,
        "completion_contract_path": str(COMPLETION_CONTRACT),
        "completion_contract_sha256": completion_sha,
        "variant_summaries": summaries,
        "variant_checks": variant_checks,
        "gates": {name: bool(value) for name, value in gates.items()},
        "model_instantiated": False,
        "checkpoint_metadata_loaded_on_CPU_read_only": True,
        "images_accessed": False,
        "optimizer_created": False,
        "parameters_updated": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "next_required_stage": "POSTHOC_SIX_VARIANT_CALIBRATION_AND_VALIDATION_COMPARISON",
    }
    evidence_action = frozen_json(EVIDENCE, evidence_payload)
    evidence_sha = sha256_file(EVIDENCE)
    os.sync()

    print("Registry action   :", registry_action)
    print("Registry          :", REGISTRY)
    print("Registry SHA-256  :", registry_sha)
    print("Contract action   :", completion_action)
    print("Contract          :", COMPLETION_CONTRACT)
    print("Contract SHA-256  :", completion_sha)
    print("Source action     :", source_action)
    print("Source SHA-256    :", source_sha)
    print("Evidence action   :", evidence_action)
    print("Evidence          :", EVIDENCE)
    print("Evidence SHA-256  :", evidence_sha)

    banner("STAGE 0G-4D-5F-3D-R3 STATUS: PASS")
    print("Six final-training variants complete : YES")
    print("Three plain comparators complete     : YES")
    print("Three FOVEA ablations complete       : YES")
    print("CALIBRATION accessed this stage      : NO")
    print("Locked test accessed                 : NO")
    print("GPU training remaining               : NONE")
    print("\nNEXT ACTION: PREPARE POST-HOC CALIBRATION FOR ALL SIX FROZEN BEST CHECKPOINTS.")


if __name__ == "__main__":
    main()
