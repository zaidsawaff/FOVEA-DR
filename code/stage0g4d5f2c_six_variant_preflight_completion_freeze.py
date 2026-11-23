#!/usr/bin/env python3
"""Verify and freeze the six post-hoc 64-step preflight results."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
IMPLEMENTATION_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2a_comparator_implementation_audit.py"
IMPLEMENTATION_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparator_implementation_contract_v1.json"
IMPLEMENTATION_EVIDENCE = ROOT / "Evidence/stage0g4d5f2a_comparator_implementation_audit.json"
PREFLIGHT_RUNNER = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2b_resumable_variant_64step_preflight.py"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2c_six_variant_preflight_completion_freeze.py"
REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_preflight64_registry_v1.csv"
COMPLETION_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_preflight_completion_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5f2c_six_variant_preflight_completion_freeze.json"

EXPECTED_SHA = {
    PROTOCOL: "f946964923e08722dbb486aa3782b08880446040e07fb6b268ea9c28c08e6ae1",
    IMPLEMENTATION_SOURCE: "987f170cbfd4ef048161d29a12a9cfd6e715b45338c539b0c9ba7dfe5a04379f",
    PREFLIGHT_RUNNER: "dbe713a8050ef0d454488fb18932e6e68f3cf7cf0395a4cbe344b3005d56396c",
}

VARIANT_RECORDS = [
    {
        "variant_id": "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
        "family": "PLAIN_COMPARATOR",
        "backbone": "EfficientNetV2-S",
        "log_sha256": "bd7bdb5b727475ae6237de691229e2481ebf3dfb35f475aedbdd85767c2d2727",
        "evidence_sha256": "a93fe1f257e75e27b93965c97508d742fbf0dfc901d51c3131ca6a63f2e58ba5",
    },
    {
        "variant_id": "B2_RESNET50_MULTITASK",
        "family": "PLAIN_COMPARATOR",
        "backbone": "ResNet-50",
        "log_sha256": "c0cb552409ac3a82fa3bb387001478caf0da1cbee0c6b5ac8e75cfb464da029e",
        "evidence_sha256": "d7db4cd9a33bcc619bf220159f303d0bf3730262b9a16fcee70f6be85b5439ba",
    },
    {
        "variant_id": "B3_CONVNEXT_TINY_MULTITASK",
        "family": "PLAIN_COMPARATOR",
        "backbone": "ConvNeXt-Tiny",
        "log_sha256": "ca565671633db050dcbf6d8d9d660e866520abf7b61104945529c32db5d6225f",
        "evidence_sha256": "c6315bbdd21d22b223ebbdb3b7c06cedd9140cc3a604fe22340f06844ec5b317",
    },
    {
        "variant_id": "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
        "family": "FOVEA_ABLATION",
        "backbone": "EfficientNetV2-S",
        "log_sha256": "935342c1aef2ca85a003da8056b1561188d6d64016b341cd7cd2225b12571d85",
        "evidence_sha256": "7b535f596a0737c618434c880c7541c0f94e19a996fe483b1fdca84ad552fb02",
    },
    {
        "variant_id": "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
        "family": "FOVEA_ABLATION",
        "backbone": "EfficientNetV2-S",
        "log_sha256": "928dea465ad0a07ede45b7394d28541ae6fd87c0c6ce3bcad8df1392a71d0953",
        "evidence_sha256": "8b8be406e145512c71f12a6009f32d77dea2a241d3c674560e8969dea5d992fc",
    },
    {
        "variant_id": "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
        "family": "FOVEA_ABLATION",
        "backbone": "EfficientNetV2-S",
        "log_sha256": "d62536380600d814242315a2a6629649877b55600ff3502ed2aabd43e8421418",
        "evidence_sha256": "9cfdf390ed51e31c9030037ae9dac6cee8b61e2794ce5742a2bfc2fed5d3d89c",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return frozen_bytes(path, data)


def frozen_csv(path: Path, frame: pd.DataFrame) -> str:
    data = frame.to_csv(index=False).encode("utf-8")
    return frozen_bytes(path, data)


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different 5F-2C source is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def artifact_paths(variant_id: str) -> tuple[Path, Path]:
    log_path = (
        ROOT
        / "Runs/FOVEA_DR/posthoc_comparisons/preflight64"
        / variant_id
        / "optimizer_step_log.csv"
    )
    evidence_path = (
        ROOT
        / "Evidence"
        / f"stage0g4d5f2b_{variant_id.lower()}_64step_preflight.json"
    )
    return log_path, evidence_path


def inspect_variant(
    declared: dict[str, str],
    parameter_map: dict[str, int],
) -> tuple[dict[str, Any], dict[str, bool]]:
    variant_id = declared["variant_id"]
    log_path, evidence_path = artifact_paths(variant_id)
    checks: dict[str, bool] = {
        "log_exists": log_path.is_file(),
        "evidence_exists": evidence_path.is_file(),
    }
    if not all(checks.values()):
        return {}, checks

    actual_log_sha = sha256_file(log_path)
    actual_evidence_sha = sha256_file(evidence_path)
    checks.update({
        "log_hash_verified": actual_log_sha == declared["log_sha256"],
        "evidence_hash_verified": actual_evidence_sha == declared["evidence_sha256"],
    })

    with evidence_path.open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)
    evidence_gates = evidence.get("gates", {})
    checks.update({
        "evidence_status_PASS": evidence.get("status") == "PASS",
        "variant_id_exact": evidence.get("variant_id") == variant_id,
        "runner_hash_verified": evidence.get("runner_source_sha256") == EXPECTED_SHA[PREFLIGHT_RUNNER],
        "implementation_hash_verified": evidence.get("implementation_source_sha256") == EXPECTED_SHA[IMPLEMENTATION_SOURCE],
        "all_variant_evidence_gates_passed": (
            isinstance(evidence_gates, dict)
            and bool(evidence_gates)
            and all(bool(value) for value in evidence_gates.values())
        ),
        "evidence_links_log": evidence.get("pilot_log_sha256") == actual_log_sha,
        "exact_64_steps_in_evidence": int(evidence.get("optimizer_steps", -1)) == 64,
        "exact_128_microbatches_in_evidence": int(evidence.get("microbatches", -1)) == 128,
        "exact_2048_samples_in_evidence": int(evidence.get("TRAIN_samples_processed_with_replacement", -1)) == 2048,
        "preflight_checkpoint_absent": evidence.get("preflight_checkpoint_saved") is False,
        "final_training_not_started": evidence.get("final_training_started") is False,
        "validation_not_accessed": evidence.get("validation_accessed") is False,
        "calibration_not_accessed": evidence.get("calibration_accessed") is False,
        "locked_test_not_accessed": evidence.get("locked_test_accessed") is False,
    })

    log = pd.read_csv(log_path)
    required_columns = {
        "optimizer_step", "total_loss", "gradient_norm_before_clip",
        "grad_scaler_scale", "peak_reserved_vram_gb",
        "CFP_samples", "UWF_samples", "OCT_samples",
    }
    numeric_columns = sorted(required_columns - {"optimizer_step"})
    missing_columns = sorted(required_columns - set(log.columns))
    numeric_finite = False
    steps_exact = False
    composition_exact = False
    if not missing_columns:
        numeric = log[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric_finite = bool(np.isfinite(numeric.to_numpy()).all())
        steps_exact = (
            len(log) == 64
            and log["optimizer_step"].astype(int).tolist() == list(range(1, 65))
        )
        composition_exact = bool(
            (
                (log["CFP_samples"] == 15)
                & (log["UWF_samples"] == 13)
                & (log["OCT_samples"] == 4)
            ).all()
        )
    checks.update({
        "log_schema_complete": not missing_columns,
        "log_numeric_values_finite": numeric_finite,
        "log_steps_exactly_1_through_64": steps_exact,
        "log_modality_composition_exact": composition_exact,
    })

    mean_first = float(log.head(8)["total_loss"].mean())
    mean_last = float(log.tail(8)["total_loss"].mean())
    checks.update({
        "mean_first_matches_evidence": math.isclose(
            mean_first, float(evidence.get("mean_first_8_step_loss", math.nan)),
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "mean_last_matches_evidence": math.isclose(
            mean_last, float(evidence.get("mean_last_8_step_loss", math.nan)),
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "parameter_count_available": variant_id in parameter_map,
    })
    summary = {
        "variant_id": variant_id,
        "family": declared["family"],
        "backbone": declared["backbone"],
        "total_parameters": parameter_map.get(variant_id),
        "optimizer_steps": int(evidence["optimizer_steps"]),
        "TRAIN_samples_with_replacement": int(evidence["TRAIN_samples_processed_with_replacement"]),
        "mean_first_8_loss": mean_first,
        "mean_last_8_loss": mean_last,
        "last_to_first_loss_ratio": mean_last / mean_first,
        "elapsed_seconds": float(evidence["elapsed_seconds"]),
        "peak_reserved_VRAM_GB": float(evidence["peak_reserved_VRAM_GB"]),
        "peak_reserved_VRAM_fraction": float(evidence["peak_reserved_VRAM_fraction"]),
        "preflight_log_sha256": actual_log_sha,
        "preflight_evidence_sha256": actual_evidence_sha,
        "all_checks_passed": bool(all(checks.values())),
    }
    return summary, checks


def main() -> None:
    print("=" * 96)
    print("STAGE 0G-4D-5F-2C — SIX-VARIANT PREFLIGHT COMPLETION FREEZE")
    print("=" * 96)

    print("\n[1] FROZEN PARENT ARTIFACTS")
    print("-" * 96)
    gates: dict[str, bool] = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(verified)
        print(f"{path.name:78s}: exists={exists}, hash_verified={verified}")
    for path in [IMPLEMENTATION_CONTRACT, IMPLEMENTATION_EVIDENCE]:
        exists = path.is_file() and path.stat().st_size > 0
        gates[f"exists_nonempty::{path.name}"] = bool(exists)
        print(f"{path.name:78s}: exists_nonempty={exists}")
    if not all(gates.values()):
        raise RuntimeError("A frozen parent artifact is missing or changed.")

    with PROTOCOL.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    protocol_ids = [item["id"] for item in protocol["comparators"]]
    protocol_ids += [item["id"] for item in protocol["ablations"]]
    declared_ids = [record["variant_id"] for record in VARIANT_RECORDS]
    gates["six_declared_variants_match_protocol_order"] = declared_ids == protocol_ids
    gates["locked_test_prohibited"] = protocol["data_access_policy"][
        "LOCKED_INTERNAL_TEST"
    ].startswith("PROHIBITED")

    with IMPLEMENTATION_EVIDENCE.open("r", encoding="utf-8") as handle:
        implementation_evidence = json.load(handle)
    implementation_gates = implementation_evidence.get("gates", {})
    implementation_pass = (
        implementation_evidence.get("status") == "PASS"
        and bool(implementation_gates)
        and all(bool(value) for value in implementation_gates.values())
        and implementation_evidence.get("locked_test_accessed") is False
    )
    parameter_map = {
        record["variant_id"]: int(record["total_parameters"])
        for record in implementation_evidence.get("variant_records", [])
    }
    gates["implementation_audit_passed"] = bool(implementation_pass)
    gates["parameter_map_has_exact_six_variants"] = set(parameter_map) == set(protocol_ids)

    print("\n[2] SIX PREFLIGHT ARTIFACT PAIRS")
    print("-" * 96)
    summaries: list[dict[str, Any]] = []
    variant_checks: dict[str, dict[str, bool]] = {}
    for declared in VARIANT_RECORDS:
        summary, checks = inspect_variant(declared, parameter_map)
        variant_id = declared["variant_id"]
        variant_checks[variant_id] = checks
        passed = bool(checks) and all(checks.values())
        gates[f"preflight_complete::{variant_id}"] = passed
        print(f"{variant_id:68s}: PASS={passed}")
        if summary:
            summaries.append(summary)

    gates.update({
        "exactly_six_preflight_summaries_created": len(summaries) == 6,
        "three_plain_comparators_complete": sum(
            record.get("family") == "PLAIN_COMPARATOR" for record in summaries
        ) == 3,
        "three_FOVEA_ablations_complete": sum(
            record.get("family") == "FOVEA_ABLATION" for record in summaries
        ) == 3,
        "all_preflight_weights_disposable": True,
        "no_preflight_checkpoint_available_for_initialization": True,
        "final_comparator_training_not_started": True,
        "VALIDATION_not_accessed": True,
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
        "model_not_loaded_this_stage": True,
        "images_not_accessed_this_stage": True,
    })

    print("\n" + "=" * 96)
    print("SIX-VARIANT PREFLIGHT COMPLETION GATES")
    print("=" * 96)
    for name, passed in gates.items():
        print(f"{name:86s}: {passed}")
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Six-variant completion gate failed: {failed}")

    print("\n[3] REGISTRY, COMPLETION CONTRACT, SOURCE, AND EVIDENCE FREEZE")
    print("-" * 96)
    registry_frame = pd.DataFrame(summaries)
    registry_action = frozen_csv(REGISTRY, registry_frame)
    registry_sha = sha256_file(REGISTRY)
    source_action, source_sha = freeze_source()

    completion = {
        "contract_id": "MMRDR_FOVEA_DR_POSTHOC_PREFLIGHT_COMPLETION_V1",
        "stage": "0G-4D-5F-2C",
        "status": "SIX_VARIANTS_READY_FOR_INDEPENDENT_FINAL_TRAINING",
        "comparison_protocol": str(PROTOCOL),
        "comparison_protocol_sha256": EXPECTED_SHA[PROTOCOL],
        "implementation_source_sha256": EXPECTED_SHA[IMPLEMENTATION_SOURCE],
        "preflight_runner_sha256": EXPECTED_SHA[PREFLIGHT_RUNNER],
        "preflight_registry": str(REGISTRY),
        "preflight_registry_sha256": registry_sha,
        "variants_in_locked_order": declared_ids,
        "each_final_run_must_initialize_from_official_ImageNet_weights": True,
        "preflight_weights_must_not_be_reused": True,
        "final_runs_must_be_reported_even_if_unsuccessful": True,
        "validation_only_for_early_stopping_and_selection": True,
        "calibration_only_after_epoch_selection": True,
        "locked_test_access": "PROHIBITED",
    }
    completion_action = frozen_json(COMPLETION_CONTRACT, completion)
    completion_sha = sha256_file(COMPLETION_CONTRACT)
    gates["completion_source_frozen_and_verified"] = sha256_file(FROZEN_SOURCE) == source_sha
    evidence = {
        "stage": "0G-4D-5F-2C",
        "status": "PASS",
        "purpose": "SIX_VARIANT_PREFLIGHT_COMPLETION_FREEZE",
        "source": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "registry": str(REGISTRY),
        "registry_sha256": registry_sha,
        "completion_contract": str(COMPLETION_CONTRACT),
        "completion_contract_sha256": completion_sha,
        "variant_summaries": summaries,
        "variant_checks": variant_checks,
        "gates": {name: bool(value) for name, value in gates.items()},
        "model_loaded": False,
        "images_accessed": False,
        "optimizer_created": False,
        "parameters_updated": False,
        "final_training_started": False,
        "validation_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "next_required_stage": "FREEZE_RESUMABLE_FINAL_COMPARATOR_TRAINING_RUNNER",
    }
    evidence_action = frozen_json(EVIDENCE, evidence)
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
    print("Evidence SHA-256  :", sha256_file(EVIDENCE))

    print("\nModel loaded          : NO")
    print("Images accessed       : NO")
    print("Optimizer created     : NO")
    print("Parameters updated    : NO")
    print("Final training started: NO")
    print("VALIDATION accessed   : NO")
    print("CALIBRATION accessed  : NO")
    print("Locked test accessed  : NO")
    print("\nNEXT REQUIRED STAGE:")
    print("Freeze the resumable final comparator-training runner.")
    print("\n" + "=" * 96)
    print("STAGE 0G-4D-5F-2C STATUS: PASS")
    print("=" * 96)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 96, file=sys.stderr)
        print("STAGE 0G-4D-5F-2C STATUS: FAIL", file=sys.stderr)
        print("=" * 96, file=sys.stderr)
        raise
