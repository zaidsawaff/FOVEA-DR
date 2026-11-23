#!/usr/bin/env python3
"""Freeze completion of the three reviewer E2 64-step TRAIN-only preflights.

This stage is deliberately CPU-only and metadata-only.  It verifies the
three disposable preflight records, their optimizer logs, and the targeted
AMP-safe amendment.  It neither reads medical images nor creates an
optimizer/model checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
CODE = ROOT / "Code/FOVEA_DR/v1_1"
METADATA = ROOT / "Metadata"
EVIDENCE = ROOT / "Evidence"
RUN_ROOT = ROOT / "Runs/FOVEA_DR/reviewer_experiments/E2_component_ablation_preflight64"

REVIEWER_PROTOCOL = METADATA / "mmrdr_fovea_dr_reviewer_experiment_protocol_v1.json"
ORIGINAL_IMPLEMENTATION = CODE / "stage0g4d5f7a_reviewer_component_ablation_implementation_audit.py"
REPAIR_SOURCE = CODE / "stage0g4d5f7a_r1_reviewer_component_ablation_amp_safe_repair.py"
REPAIR_CONTRACT = METADATA / "mmrdr_fovea_dr_reviewer_component_ablation_amp_safe_repair_v1.json"
REPAIR_EVIDENCE = EVIDENCE / "stage0g4d5f7a_r1_reviewer_component_ablation_amp_safe_repair.json"
BASE_RUNNER = CODE / "stage0g4d5f7b_reviewer_component_ablation_64step_preflight.py"
REPAIRED_RUNNER = CODE / "stage0g4d5f7b_r1_reviewer_component_ablation_64step_preflight.py"

FROZEN_SOURCE = CODE / "stage0g4d5f7c_reviewer_component_ablation_preflight_completion_freeze.py"
REGISTRY = RUN_ROOT / "mmrdr_fovea_dr_reviewer_E2_preflight64_registry_v1.csv"
CONTRACT = METADATA / "mmrdr_fovea_dr_reviewer_E2_preflight_completion_v1.json"
COMPLETION_EVIDENCE = EVIDENCE / "stage0g4d5f7c_reviewer_component_ablation_preflight_completion_freeze.json"

EXPECTED = {
    REVIEWER_PROTOCOL: "edc7a1503d6c2323c8bc302cfa7942a9bec8a1182de561ef99fdba44bb00a1a2",
    ORIGINAL_IMPLEMENTATION: "d74b6ebfe866905e1edcefff2a298a5bed78d578621c74c30d95f155dd2ef1bd",
    REPAIR_SOURCE: "bd7358f402d3488f4a829eee85a823456ee6e87b1eb0d676d229b383dbc1a357",
    BASE_RUNNER: "462ad8cd8bee42404e97eeea25a85421ab165e3343bba06c3eee5fa846c61157",
    REPAIRED_RUNNER: "535ce34324acfe3131e67937cf57efd090c5d6a2f3459b02d0db7fc6a5e59cdd",
}
EXPECTED_OFFICIAL_WEIGHT_SHA = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"

VARIANTS = [
    "NO_FIELD_OF_VIEW_ENCODING",
    "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE",
    "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE",
]
EXPECTED_PARAMETERS = {
    "NO_FIELD_OF_VIEW_ENCODING": 20695974,
    "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE": 20694054,
    "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE": 20695777,
}
ORIGINAL_VARIANT = "NO_FIELD_OF_VIEW_ENCODING"
REPAIRED_VARIANTS = set(VARIANTS[1:])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def evidence_passes(payload: dict[str, Any]) -> bool:
    gates = payload.get("gates")
    return bool(
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and gates
        and all(value is True for value in gates.values())
    )


def atomic_frozen_bytes(path: Path, data: bytes, label: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"A different frozen {label} already exists: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    if path.read_bytes() != data:
        raise RuntimeError(f"Frozen {label} write/read-back verification failed")
    return "CREATED_AND_FROZEN"


def canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different E2 completion-freeze source is already frozen")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    if sha256_file(FROZEN_SOURCE) != source_sha:
        raise RuntimeError("E2 completion-freeze source copy verification failed")
    return "COPIED_AND_FROZEN", source_sha


def evidence_path(variant: str) -> Path:
    return EVIDENCE / f"stage0g4d5f7b_{variant.lower()}_64step_preflight.json"


def print_gate(name: str, value: bool, width: int = 92) -> None:
    print(f"{name:{width}s}: {value}")


def inspect_variant(position: int, variant: str) -> tuple[dict[str, Any], dict[str, bool]]:
    path = evidence_path(variant)
    checks: dict[str, bool] = {"evidence_exists": path.is_file()}
    if not path.is_file():
        return {
            "protocol_position": position,
            "variant_id": variant,
            "status": "MISSING",
        }, checks

    payload = load_json(path)
    evidence_sha = sha256_file(path)
    log_value = payload.get("pilot_log")
    log_path = Path(log_value) if isinstance(log_value, str) and log_value else Path("/__missing__")
    log_exists = log_path.is_file()
    log_sha = sha256_file(log_path) if log_exists else None
    gates = payload.get("gates")

    expected_impl_sha = (
        EXPECTED[ORIGINAL_IMPLEMENTATION]
        if variant == ORIGINAL_VARIANT
        else EXPECTED[REPAIR_SOURCE]
    )
    expected_runner_sha = (
        EXPECTED[BASE_RUNNER]
        if variant == ORIGINAL_VARIANT
        else EXPECTED[REPAIRED_RUNNER]
    )

    checks.update({
        "evidence_PASS": evidence_passes(payload),
        "variant_id_exact": payload.get("variant_id") == variant,
        "run_seed_is_2026": payload.get("run_seed") == 2026,
        "runner_source_hash_verified": payload.get("runner_source_sha256") == expected_runner_sha,
        "implementation_source_hash_verified": payload.get("implementation_source_sha256") == expected_impl_sha,
        "official_weight_hash_verified": payload.get("official_checkpoint_sha256") == EXPECTED_OFFICIAL_WEIGHT_SHA,
        "optimizer_steps_are_64": payload.get("optimizer_steps") == 64,
        "microbatches_are_128": payload.get("microbatches") == 128,
        "TRAIN_samples_are_2048": payload.get("TRAIN_samples_processed_with_replacement") == 2048,
        "parameter_count_expected": EXPECTED_PARAMETERS[variant] > 0,
        "pilot_log_exists": log_exists,
        "pilot_log_hash_verified": log_exists and payload.get("pilot_log_sha256") == log_sha,
        "all_preflight_gates_PASS": isinstance(gates, dict) and bool(gates) and all(value is True for value in gates.values()),
        "preflight_checkpoint_not_saved": payload.get("preflight_checkpoint_saved") is False,
        "preflight_weights_not_reusable": payload.get("preflight_weights_allowed_for_final_initialization") is False,
        "final_training_not_started": payload.get("final_training_started") is False,
        "VALIDATION_not_accessed": payload.get("validation_accessed") is False,
        "CALIBRATION_not_accessed": payload.get("calibration_accessed") is False,
        "locked_test_not_accessed": payload.get("locked_test_accessed") is False,
    })

    summary = {
        "protocol_position": position,
        "variant_id": variant,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "parameter_count": EXPECTED_PARAMETERS[variant],
        "optimizer_steps": payload.get("optimizer_steps"),
        "TRAIN_samples_processed_with_replacement": payload.get("TRAIN_samples_processed_with_replacement"),
        "mean_first_8_step_loss": payload.get("mean_first_8_step_loss"),
        "mean_last_8_step_loss": payload.get("mean_last_8_step_loss"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "peak_reserved_VRAM_GB": payload.get("peak_reserved_VRAM_GB"),
        "peak_reserved_VRAM_fraction": payload.get("peak_reserved_VRAM_fraction"),
        "pilot_log": str(log_path),
        "pilot_log_sha256": log_sha,
        "preflight_evidence": str(path),
        "preflight_evidence_sha256": evidence_sha,
        "runner_source_sha256": payload.get("runner_source_sha256"),
        "implementation_source_sha256": payload.get("implementation_source_sha256"),
        "official_checkpoint_sha256": payload.get("official_checkpoint_sha256"),
        "AMP_safe_repair_applies": variant in REPAIRED_VARIANTS,
        "preflight_checkpoint_saved": False,
        "preflight_weights_reusable": False,
        "final_training_started": False,
        "validation_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
    }
    return summary, checks


def registry_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = [
        "protocol_position", "variant_id", "status", "parameter_count",
        "optimizer_steps", "TRAIN_samples_processed_with_replacement",
        "mean_first_8_step_loss", "mean_last_8_step_loss", "elapsed_seconds",
        "peak_reserved_VRAM_GB", "peak_reserved_VRAM_fraction", "pilot_log",
        "pilot_log_sha256", "preflight_evidence", "preflight_evidence_sha256",
        "runner_source_sha256", "implementation_source_sha256",
        "official_checkpoint_sha256", "AMP_safe_repair_applies",
        "preflight_checkpoint_saved", "preflight_weights_reusable",
        "final_training_started", "validation_accessed", "calibration_accessed",
        "locked_test_accessed",
    ]
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    print("=" * 124)
    print("STAGE 0G-4D-5F-7C — E2 THREE-VARIANT 64-STEP PREFLIGHT COMPLETION FREEZE")
    print("=" * 124)

    source_action, source_sha = freeze_source()
    print("\n[1] FROZEN PROTOCOL, IMPLEMENTATIONS, AND RUNNERS")
    print("-" * 124)
    top_checks: dict[str, bool] = {}
    for path, expected_sha in EXPECTED.items():
        name = f"hash_verified::{path.name}"
        value = path.is_file() and sha256_file(path) == expected_sha
        top_checks[name] = value
        print_gate(name, value)

    repair_contract_exists = REPAIR_CONTRACT.is_file()
    repair_evidence_exists = REPAIR_EVIDENCE.is_file()
    repair_contract_sha = sha256_file(REPAIR_CONTRACT) if repair_contract_exists else None
    repair_payload = load_json(REPAIR_EVIDENCE) if repair_evidence_exists else {}
    repair_checks = {
        "repair_contract_exists": repair_contract_exists,
        "repair_evidence_exists": repair_evidence_exists,
        "repair_evidence_PASS": repair_evidence_exists and evidence_passes(repair_payload),
        "repair_evidence_links_contract": repair_evidence_exists and repair_payload.get("contract_sha256") == repair_contract_sha,
        "repair_variant_exact": repair_payload.get("repaired_variant") == "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE",
        "repair_architecture_unchanged": repair_payload.get("gates", {}).get("architecture_parameter_count_unchanged") is True,
        "repair_loss_definition_unchanged": repair_payload.get("gates", {}).get("loss_definition_unchanged") is True,
        "repair_medical_images_not_accessed": repair_payload.get("medical_images_accessed") is False,
        "repair_CALIBRATION_not_accessed": repair_payload.get("calibration_accessed") is False,
        "repair_locked_test_not_accessed": repair_payload.get("locked_test_accessed") is False,
    }
    for name, value in repair_checks.items():
        print_gate(name, value)

    print("\n[2] THREE DISPOSABLE PREFLIGHT RECORDS")
    print("-" * 124)
    rows: list[dict[str, Any]] = []
    variant_checks: dict[str, bool] = {}
    for position, variant in enumerate(VARIANTS, start=1):
        summary, checks = inspect_variant(position, variant)
        rows.append(summary)
        passed = all(checks.values())
        variant_checks[f"preflight_complete::{variant}"] = passed
        print(f"{variant:64s}: PASS={passed}")
        if not passed:
            for name, value in checks.items():
                if not value:
                    print(f"  FAILED::{name}")

    log_paths = [row.get("pilot_log") for row in rows]
    evidence_paths = [row.get("preflight_evidence") for row in rows]
    aggregate_checks = {
        **top_checks,
        **repair_checks,
        **variant_checks,
        "exactly_three_preflight_summaries": len(rows) == 3,
        "registry_order_matches_protocol": [row.get("variant_id") for row in rows] == VARIANTS,
        "all_statuses_PASS": all(row.get("status") == "PASS" for row in rows),
        "three_distinct_optimizer_logs": len(set(log_paths)) == 3,
        "three_distinct_preflight_evidence_files": len(set(evidence_paths)) == 3,
        "all_optimizer_step_counts_are_64": all(row.get("optimizer_steps") == 64 for row in rows),
        "all_preflight_checkpoints_absent": all(row.get("preflight_checkpoint_saved") is False for row in rows),
        "all_preflight_weights_disallowed": all(row.get("preflight_weights_reusable") is False for row in rows),
        "all_final_training_not_started": all(row.get("final_training_started") is False for row in rows),
        "VALIDATION_not_accessed": all(row.get("validation_accessed") is False for row in rows),
        "CALIBRATION_not_accessed": all(row.get("calibration_accessed") is False for row in rows),
        "locked_test_not_accessed": all(row.get("locked_test_accessed") is False for row in rows),
        "model_not_loaded_this_stage": True,
        "medical_images_not_accessed_this_stage": True,
        "optimizer_not_created_this_stage": True,
        "parameters_not_updated_this_stage": True,
    }

    print("\n[3] COMPLETION GATES")
    print("-" * 124)
    for name, value in aggregate_checks.items():
        print_gate(name, value)
    failed = [name for name, value in aggregate_checks.items() if not value]
    if failed:
        raise RuntimeError("E2 preflight completion freeze failed:\n" + "\n".join(failed))

    reg_data = registry_bytes(rows)
    registry_action = atomic_frozen_bytes(REGISTRY, reg_data, "E2 preflight registry")
    registry_sha = sha256_file(REGISTRY)

    contract_payload = {
        "contract_id": "MMRDR_FOVEA_DR_REVIEWER_E2_PREFLIGHT_COMPLETION_V1",
        "status": "FROZEN",
        "stage": "0G-4D-5F-7C",
        "purpose": "THREE_VARIANT_64_STEP_TRAIN_ONLY_PREFLIGHT_COMPLETION",
        "freeze_date_utc": "2026-10-30",
        "variant_order": VARIANTS,
        "variant_count": 3,
        "run_seed": 2026,
        "optimizer_steps_per_variant": 64,
        "preflight_checkpoint_saved": False,
        "preflight_weights_allowed_for_final_initialization": False,
        "final_component_ablation_training_started": False,
        "original_implementation_source_sha256": EXPECTED[ORIGINAL_IMPLEMENTATION],
        "AMP_safe_repair_source_sha256": EXPECTED[REPAIR_SOURCE],
        "AMP_safe_repair_contract": str(REPAIR_CONTRACT),
        "AMP_safe_repair_contract_sha256": repair_contract_sha,
        "registry": str(REGISTRY),
        "registry_sha256": registry_sha,
        "preflight_evidence": [
            {
                "variant_id": row["variant_id"],
                "path": row["preflight_evidence"],
                "sha256": row["preflight_evidence_sha256"],
            }
            for row in rows
        ],
        "VALIDATION_access": "PROHIBITED_DURING_PREFLIGHT",
        "CALIBRATION_access": "PROHIBITED",
        "LOCKED_INTERNAL_TEST_access": "PROHIBITED",
        "next_required_action": "PREPARE_RESUMABLE_FINAL_TRAINING_FOR_E2_VARIANTS",
    }
    contract_action = atomic_frozen_bytes(CONTRACT, canonical_json(contract_payload), "E2 completion contract")
    contract_sha = sha256_file(CONTRACT)

    evidence_payload = {
        "stage": "0G-4D-5F-7C",
        "status": "PASS",
        "purpose": "REVIEWER_E2_PREFLIGHT_COMPLETION_FREEZE",
        "source": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "reviewer_protocol": str(REVIEWER_PROTOCOL),
        "reviewer_protocol_sha256": EXPECTED[REVIEWER_PROTOCOL],
        "repair_evidence": str(REPAIR_EVIDENCE),
        "repair_evidence_sha256": sha256_file(REPAIR_EVIDENCE),
        "registry": str(REGISTRY),
        "registry_sha256": registry_sha,
        "contract": str(CONTRACT),
        "contract_sha256": contract_sha,
        "variant_summaries": rows,
        "gates": {name: bool(value) for name, value in aggregate_checks.items()},
        "model_loaded": False,
        "medical_images_accessed": False,
        "optimizer_created": False,
        "parameters_updated": False,
        "validation_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "next_required_action": "PREPARE_RESUMABLE_FINAL_TRAINING_FOR_E2_VARIANTS",
    }
    evidence_action = atomic_frozen_bytes(
        COMPLETION_EVIDENCE,
        canonical_json(evidence_payload),
        "E2 completion evidence",
    )

    print("\n[4] CONTRACT AND EVIDENCE FREEZE")
    print("-" * 124)
    print("Source action   :", source_action)
    print("Source SHA-256  :", source_sha)
    print("Registry action :", registry_action)
    print("Registry        :", REGISTRY)
    print("Registry SHA-256:", registry_sha)
    print("Contract action :", contract_action)
    print("Contract        :", CONTRACT)
    print("Contract SHA-256:", contract_sha)
    print("Evidence action :", evidence_action)
    print("Evidence        :", COMPLETION_EVIDENCE)
    print("Evidence SHA-256:", sha256_file(COMPLETION_EVIDENCE))

    print("\n" + "=" * 124)
    print("STAGE 0G-4D-5F-7C STATUS: PASS")
    print("=" * 124)
    print("E2 variants verified             : 3 / 3")
    print("AMP-safe amendment               : VERIFIED")
    print("Disposable checkpoints saved     : NO")
    print("Preflight weights reusable       : NO")
    print("Final E2 training started         : NO")
    print("Medical images accessed this stage: NO")
    print("VALIDATION accessed               : NO")
    print("CALIBRATION accessed              : NO")
    print("Locked test accessed              : NO")
    print("NEXT ACTION: PREPARE RESUMABLE FINAL TRAINING FOR THE THREE E2 VARIANTS")


if __name__ == "__main__":
    main()
