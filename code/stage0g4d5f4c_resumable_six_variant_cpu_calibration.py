#!/usr/bin/env python3
"""Resumable CPU/FP32 post-hoc calibration for one frozen variant at a time.

The runner consumes only the frozen CALIBRATION partition, writes immutable
32-row inference parts, fits the ten protocol-locked scalar temperatures after
all 2,840 rows are complete, and never creates an optimizer or updates model
parameters. The locked internal test is outside every path used here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


WIDTH = 124
ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
CALIBRATION_CACHE_ROOT = Path("/content/MMRDR_calibration_v1")

PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_six_variant_calibration_protocol_v1.json"
PROTOCOL_EVIDENCE = ROOT / "Evidence/stage0g4d5f4b_six_variant_calibration_protocol_lock.json"
PROTOCOL_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4b_six_variant_calibration_protocol_lock.py"
FINAL_TRAINING_REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_final_training_registry_v1.csv"
IMPLEMENTATION_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2a_comparator_implementation_audit.py"
PRODUCTION_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
PRIMARY_CALIBRATION_RUNNER = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5c_resumable_cpu_calibration_inference.py"
CALIBRATION_CACHE_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5a_calibration_only_cache_preparation.py"
CALIBRATION_CACHE_EVIDENCE = ROOT / "Evidence/stage0g4d5a_calibration_only_cache_preparation.json"
DEVELOPMENT_SPLIT = ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
CALIBRATION_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_CALIBRATION_local_cache_manifest_v1.csv"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f4c_resumable_six_variant_cpu_calibration.py"
RUN_BASE = ROOT / "Runs/FOVEA_DR/posthoc_comparisons/calibration_seed2026_cpu_fp32"

EXPECTED_SHA256 = {
    PROTOCOL: "9bdd3a5cf4f4ead1dc3055fd5c84b62ca1bea929a95bf96372ec8bea66487c4d",
    PROTOCOL_EVIDENCE: "eb78d67a25728eadff334a9f20a6bb94d42704781f591ad153b1ece2484abf5c",
    PROTOCOL_SOURCE: "0a59899a1abb238aacfb5b18d99c6444725493cf8f6c1eb1cafd2899dbeb9fec",
    FINAL_TRAINING_REGISTRY: "de8b3d0c098128510d185960f81af772c37f34ba877580b7160606153fa6ad75",
    IMPLEMENTATION_SOURCE: "987f170cbfd4ef048161d29a12a9cfd6e715b45338c539b0c9ba7dfe5a04379f",
    PRODUCTION_SOURCE: "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9",
    PRIMARY_CALIBRATION_RUNNER: "9d8050264362b7af12aa8962c9cc16528c0b75f11eac26040974068ad4e2d522",
    CALIBRATION_CACHE_SOURCE: "59a5ed886e146461f4cc0c66df3bd61eb51e0eed76441bba61513b4910745473",
    DEVELOPMENT_SPLIT: "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140",
    CALIBRATION_MANIFEST: "f8e6a62b96e330a1b21bb9ed56897cbf04b730987cf9e0c31141927abbb84655",
}

VARIANT_IDS = [
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
    "B2_RESNET50_MULTITASK",
    "B3_CONVNEXT_TINY_MULTITASK",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
]
PLAIN_VARIANTS = set(VARIANT_IDS[:3])
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
EXPECTED_MODALITY_COUNTS = {"CFP": 1313, "UWF": 1171, "OCT": 356}
INPUT_SIZE = 512
RUN_SEED = 2026
PART_ROWS = 32
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


def all_evidence_gates_pass(payload: dict[str, Any]) -> bool:
    gates = payload.get("gates")
    return (
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and bool(gates)
        and all(bool(value) for value in gates.values())
    )


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, json_bytes(payload))


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
            raise RuntimeError("A different 5F-4C calibration runner is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    if sha256_file(temporary) != source_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Calibration-runner source copy verification failed.")
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def import_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to import frozen source: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def stable_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def real_drive_mounted() -> bool:
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", "/content/drive"],
        capture_output=True,
        text=True,
        check=False,
    )
    return ROOT.is_dir() and "drive" in result.stdout.strip().lower()


def run_paths(variant_id: str, variant_contract: dict[str, Any]) -> dict[str, Path]:
    run_root = Path(variant_contract["run_root"])
    return {
        "root": run_root,
        "parts": Path(variant_contract["raw_part_root"]),
        "state": Path(variant_contract["inference_state"]),
        "raw": Path(variant_contract["raw_predictions"]),
        "calibrated": Path(variant_contract["calibrated_predictions"]),
        "temperatures": Path(variant_contract["temperature_parameters"]),
        "metrics": Path(variant_contract["calibration_metrics"]),
        "latest_evidence": Path(variant_contract["latest_session_evidence"]),
        "completion_evidence": Path(variant_contract["completion_evidence"]),
    }


def verify_predecessors(variant_id: str, protocol_sha: str) -> bool:
    position = VARIANT_IDS.index(variant_id)
    for predecessor in VARIANT_IDS[:position]:
        state_path = RUN_BASE / predecessor / "inference_state.json"
        slug = predecessor.lower()
        evidence_path = ROOT / f"Evidence/stage0g4d5f4c_{slug}_cpu_calibration_completion.json"
        if not state_path.is_file() or not evidence_path.is_file():
            raise RuntimeError(f"Previous calibration is incomplete: {predecessor}")
        state = read_json(state_path)
        evidence = read_json(evidence_path)
        valid = (
            state.get("variant_id") == predecessor
            and state.get("status") == "COMPLETE_AND_TEMPERATURE_FITTED"
            and int(state.get("completed_rows", -1)) == EXPECTED_ROWS
            and state.get("calibration_protocol_sha256") == protocol_sha
            and evidence.get("variant_id") == predecessor
            and evidence.get("session_status") == "COMPLETE_AND_TEMPERATURE_FITTED"
            and all_evidence_gates_pass(evidence)
            and evidence.get("locked_test_accessed") is False
        )
        if not valid:
            raise RuntimeError(f"Previous calibration failed verification: {predecessor}")
    return True


def verify_existing_completion(
    paths: dict[str, Path],
    variant_id: str,
    checkpoint_sha: str,
    protocol_sha: str,
    runner_sha: str,
) -> bool:
    if not paths["state"].is_file():
        return False
    state = read_json(paths["state"])
    if state.get("status") != "COMPLETE_AND_TEMPERATURE_FITTED":
        return False
    final_hashes = state.get("final_artifact_sha256")
    if not isinstance(final_hashes, dict):
        return False
    artifacts = {
        "raw_predictions": paths["raw"],
        "calibrated_predictions": paths["calibrated"],
        "temperature_parameters": paths["temperatures"],
        "metrics": paths["metrics"],
    }
    artifacts_valid = all(
        path.is_file()
        and isinstance(final_hashes.get(name), str)
        and sha256_file(path) == final_hashes[name]
        for name, path in artifacts.items()
    )
    parts = state.get("parts")
    parts_valid = isinstance(parts, list) and bool(parts) and all(
        isinstance(record, dict)
        and Path(str(record.get("path", ""))).is_file()
        and sha256_file(Path(str(record["path"]))) == record.get("sha256")
        for record in parts
    )
    evidence_valid = False
    if paths["completion_evidence"].is_file():
        evidence = read_json(paths["completion_evidence"])
        evidence_valid = (
            evidence.get("variant_id") == variant_id
            and evidence.get("session_status") == "COMPLETE_AND_TEMPERATURE_FITTED"
            and all_evidence_gates_pass(evidence)
            and evidence.get("locked_test_accessed") is False
        )
    return bool(
        state.get("variant_id") == variant_id
        and int(state.get("completed_rows", -1)) == EXPECTED_ROWS
        and state.get("selected_checkpoint_sha256") == checkpoint_sha
        and state.get("calibration_protocol_sha256") == protocol_sha
        and state.get("runner_source_sha256") == runner_sha
        and artifacts_valid
        and parts_valid
        and evidence_valid
    )


def load_calibration_table() -> tuple[pd.DataFrame, dict[str, int], bool]:
    split = pd.read_csv(DEVELOPMENT_SPLIT)
    calibration = split[split["development_split"] == "CALIBRATION"].copy()
    calibration["relative_path"] = (
        calibration["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    manifest = pd.read_csv(CALIBRATION_MANIFEST)
    manifest["relative_path"] = (
        manifest["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    manifest_small = manifest[
        ["relative_path", "local_path", "file_size_bytes", "image_sha256"]
    ].copy()
    table = (
        calibration.merge(manifest_small, on="relative_path", how="inner", validate="one_to_one")
        .sort_values("relative_path")
        .reset_index(drop=True)
    )
    table["calibration_index"] = np.arange(len(table), dtype=int)
    modality_counts = {
        str(key): int(value)
        for key, value in table["modality"].value_counts().to_dict().items()
    }
    cache_ready = bool(
        len(table) == EXPECTED_ROWS
        and all(
            Path(str(row.local_path)).is_file()
            and Path(str(row.local_path)).stat().st_size == int(row.file_size_bytes)
            for row in table.itertuples(index=False)
        )
    )
    return table, modality_counts, cache_ready


def scan_parts(
    table: pd.DataFrame,
    part_root: Path,
    run_signature: str,
) -> tuple[int, list[pd.DataFrame], list[dict[str, Any]]]:
    part_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^part_(\d{4})_(\d{4})\.csv$")
    located: list[tuple[int, int, Path]] = []
    for path in part_root.iterdir():
        match = pattern.match(path.name)
        if match:
            located.append((int(match.group(1)), int(match.group(2)), path))
    located.sort(key=lambda item: item[0])

    expected_start = 0
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for start, end, path in located:
        if start != expected_start or end < start or end >= EXPECTED_ROWS:
            raise RuntimeError(f"Non-contiguous calibration part: {path.name}")
        frame = pd.read_csv(path)
        expected_indices = list(range(start, end + 1))
        if frame["calibration_index"].astype(int).tolist() != expected_indices:
            raise RuntimeError(f"Index mismatch in calibration part: {path.name}")
        if not frame["run_signature"].astype(str).eq(run_signature).all():
            raise RuntimeError(f"Run-signature mismatch in calibration part: {path.name}")
        expected_rows = table.iloc[start : end + 1]
        if frame["relative_path"].astype(str).tolist() != expected_rows["relative_path"].astype(str).tolist():
            raise RuntimeError(f"Relative-path mismatch in calibration part: {path.name}")
        if frame["image_sha256"].astype(str).tolist() != expected_rows["image_sha256"].astype(str).tolist():
            raise RuntimeError(f"Image-hash mismatch in calibration part: {path.name}")
        frames.append(frame)
        records.append(
            {
                "path": str(path),
                "start_index": start,
                "end_index": end,
                "rows": int(len(frame)),
                "sha256": sha256_file(path),
            }
        )
        expected_start = end + 1
    return expected_start, frames, records


def save_part(records: list[dict[str, Any]], part_root: Path, run_signature: str) -> Path:
    frame = pd.DataFrame(records).sort_values("calibration_index").reset_index(drop=True)
    start = int(frame["calibration_index"].iloc[0])
    end = int(frame["calibration_index"].iloc[-1])
    if frame["calibration_index"].astype(int).tolist() != list(range(start, end + 1)):
        raise RuntimeError("Attempted to save a non-contiguous calibration part.")
    if not frame["run_signature"].astype(str).eq(run_signature).all():
        raise RuntimeError("Attempted to save a mixed-signature calibration part.")
    path = part_root / f"part_{start:04d}_{end:04d}.csv"
    frozen_bytes(path, csv_bytes(frame))
    return path


def write_state(
    path: Path,
    variant_id: str,
    status: str,
    run_signature: str,
    runner_sha: str,
    checkpoint_sha: str,
    protocol_sha: str,
    batch_size: int,
    completed_rows: int,
    part_records: list[dict[str, Any]],
    final_hashes: dict[str, str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run": "FOVEA_DR_POSTHOC_CALIBRATION_SEED2026_CPU_FP32",
        "variant_id": variant_id,
        "status": status,
        "execution_backend": "CPU_FLOAT32",
        "batch_size": int(batch_size),
        "input_size": INPUT_SIZE,
        "part_rows": PART_ROWS,
        "run_signature": run_signature,
        "runner_source_sha256": runner_sha,
        "selected_checkpoint_sha256": checkpoint_sha,
        "calibration_protocol_sha256": protocol_sha,
        "calibration_manifest_sha256": EXPECTED_SHA256[CALIBRATION_MANIFEST],
        "completed_rows": int(completed_rows),
        "expected_rows": EXPECTED_ROWS,
        "parts": part_records,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "validation_used_for_calibration_fit": False,
        "locked_test_accessed": False,
    }
    if final_hashes:
        payload["final_artifact_sha256"] = final_hashes
    atomic_json(path, payload)


def extract_model_outputs(
    variant_id: str,
    output: dict[str, Any],
    fundus_positions: torch.Tensor,
    oct_positions: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    grade_by_position: dict[int, torch.Tensor] = {}
    expected_by_position: dict[int, torch.Tensor] = {}
    lesion_by_position: dict[int, torch.Tensor] = {}
    oct_by_position: dict[int, torch.Tensor] = {}

    if len(fundus_positions) > 0:
        if variant_id in PLAIN_VARIANTS:
            fundus_node = output["fundus"]["grade"]
            lesion_logits = output["fundus"]["lesion_logits"]
        else:
            fundus_node = output["fundus"]["original"]["fusion"]
            lesion_logits = output["fundus"]["lesions"]["presence_logits"]
        probabilities = fundus_node["probabilities"].float().cpu()
        expected = fundus_node["expected_grade"].float().cpu()
        lesion_logits = lesion_logits.float().cpu()
        for local, position in enumerate(fundus_positions.tolist()):
            grade_by_position[position] = probabilities[local]
            expected_by_position[position] = expected[local]
            lesion_by_position[position] = lesion_logits[local]

    if len(oct_positions) > 0:
        probabilities = output["oct"]["probabilities"].float().cpu()
        expected = output["oct"]["expected_grade"].float().cpu()
        for local, position in enumerate(oct_positions.tolist()):
            oct_by_position[position] = probabilities[local]
            expected_by_position[position] = expected[local]

    return grade_by_position, expected_by_position, lesion_by_position, oct_by_position


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANT_IDS)
    parser.add_argument("--session-rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.session_rows < 0 or (args.session_rows != 0 and args.session_rows % PART_ROWS != 0):
        raise ValueError("session-rows must be zero or a positive multiple of 32.")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if args.workers < 0:
        raise ValueError("workers cannot be negative.")
    if not real_drive_mounted():
        raise RuntimeError("The real Google Drive is not mounted in the notebook process.")

    variant_id = args.variant
    banner("STAGE 0G-4D-5F-4C — RESUMABLE SIX-VARIANT CPU CALIBRATION")
    print("Variant:", variant_id)

    section("[1] FROZEN PARENT ARTIFACTS")
    gates: dict[str, bool] = {}
    for path, expected_sha in EXPECTED_SHA256.items():
        verified = path.is_file() and sha256_file(path) == expected_sha
        gates[f"hash_verified::{path.name}"] = verified
        print(f"{path.name:<78}: exists={path.is_file()}, hash_verified={verified}")
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError("A frozen calibration parent changed:\n" + "\n".join(failed))

    protocol = read_json(PROTOCOL)
    protocol_evidence = read_json(PROTOCOL_EVIDENCE)
    cache_evidence = read_json(CALIBRATION_CACHE_EVIDENCE)
    protocol_sha = EXPECTED_SHA256[PROTOCOL]
    variant_contracts = protocol.get("variants", [])
    matches = [item for item in variant_contracts if item.get("variant_id") == variant_id]
    if len(matches) != 1:
        raise RuntimeError("The selected variant has no unique frozen calibration contract.")
    variant_contract = matches[0]
    selected_checkpoint = Path(variant_contract["selected_checkpoint"])
    selected_checkpoint_sha = str(variant_contract["selected_checkpoint_sha256"])
    selected_epoch = int(variant_contract["selected_epoch"])
    paths = run_paths(variant_id, variant_contract)

    identity_gates = {
        "protocol_status_locked": protocol.get("status")
        == "LOCKED_BEFORE_SIX_VARIANT_CALIBRATION_INFERENCE",
        "protocol_evidence_PASS": all_evidence_gates_pass(protocol_evidence),
        "protocol_execution_order_exact": protocol.get("execution_order") == VARIANT_IDS,
        "protocol_CPU_FLOAT32_exact": protocol.get("execution", {}).get("device") == "CPU"
        and protocol.get("execution", {}).get("precision") == "FLOAT32",
        "protocol_part_rows_exact": int(
            protocol.get("execution", {}).get("rows_per_atomic_part", -1)
        ) == PART_ROWS,
        "protocol_forbids_parameter_updates": protocol.get("calibration_partition", {}).get(
            "may_update_model_parameters"
        ) is False,
        "protocol_forbids_checkpoint_reselection": protocol.get(
            "calibration_partition", {}
        ).get("may_select_model_checkpoint") is False,
        "selected_variant_contract_unique": len(matches) == 1,
        "selected_checkpoint_exists_nonempty": selected_checkpoint.is_file()
        and selected_checkpoint.stat().st_size > 0,
        "selected_checkpoint_hash_verified": selected_checkpoint.is_file()
        and sha256_file(selected_checkpoint) == selected_checkpoint_sha,
        "selected_checkpoint_was_selected_on_VALIDATION": variant_contract.get(
            "selection_partition"
        ) == "VALIDATION",
        "predecessor_calibrations_complete": verify_predecessors(variant_id, protocol_sha),
        "CALIBRATION_cache_evidence_PASS": all_evidence_gates_pass(cache_evidence),
        "CALIBRATION_cache_evidence_locked_test_not_accessed": cache_evidence.get(
            "locked_test_accessed"
        ) is False,
    }
    gates.update(identity_gates)
    for name, passed in identity_gates.items():
        print(f"{name:<96}: {passed}")
    if not all(identity_gates.values()):
        failed = [name for name, passed in identity_gates.items() if not passed]
        raise RuntimeError("Calibration identity gate failed:\n" + "\n".join(failed))

    source_action, source_sha = freeze_source()
    gates["runner_source_frozen_and_verified"] = (
        FROZEN_SOURCE.is_file() and sha256_file(FROZEN_SOURCE) == source_sha
    )
    if verify_existing_completion(
        paths,
        variant_id,
        selected_checkpoint_sha,
        protocol_sha,
        source_sha,
    ):
        banner("STAGE 0G-4D-5F-4C STATUS: PASS — ALREADY COMPLETE")
        print("Variant                  :", variant_id)
        print("CALIBRATION rows         : 2840 / 2840")
        print("Final artifacts verified : YES")
        print("Model loaded             : NO")
        print("Images accessed          : NO")
        print("Locked test accessed     : NO")
        print(
            "NEXT ACTION              :",
            "FREEZE_SIX_VARIANT_CALIBRATION_COMPLETION"
            if variant_id == VARIANT_IDS[-1]
            else f"CALIBRATE_{VARIANT_IDS[VARIANT_IDS.index(variant_id) + 1]}",
        )
        return

    section("[2] CALIBRATION TABLE AND TEMPORARY CACHE")
    table, modality_counts, cache_ready = load_calibration_table()
    table_gates = {
        "exactly_2840_CALIBRATION_rows": len(table) == EXPECTED_ROWS,
        "CALIBRATION_modality_counts_exact": modality_counts == EXPECTED_MODALITY_COUNTS,
        "CALIBRATION_relative_paths_unique": table["relative_path"].is_unique,
        "all_CALIBRATION_cache_files_exist_with_frozen_sizes": cache_ready,
        "cache_root_is_CALIBRATION_only": str(CALIBRATION_CACHE_ROOT)
        == "/content/MMRDR_calibration_v1",
    }
    gates.update(table_gates)
    print("Rows            :", len(table))
    print("Modality counts :", modality_counts)
    print("Cache ready     :", cache_ready)
    if not all(table_gates.values()):
        failed = [name for name, passed in table_gates.items() if not passed]
        raise RuntimeError("CALIBRATION cache is not ready:\n" + "\n".join(failed))

    execution_contract = {
        "variant_id": variant_id,
        "execution_backend": "CPU_FLOAT32",
        "batch_size": int(args.batch_size),
        "input_size": INPUT_SIZE,
        "part_rows": PART_ROWS,
        "selected_epoch": selected_epoch,
        "selected_checkpoint_sha256": selected_checkpoint_sha,
        "calibration_protocol_sha256": protocol_sha,
        "calibration_manifest_sha256": EXPECTED_SHA256[CALIBRATION_MANIFEST],
        "implementation_source_sha256": EXPECTED_SHA256[IMPLEMENTATION_SOURCE],
        "runner_source_sha256": source_sha,
        "row_order": "SORTED_NORMALIZED_RELATIVE_PATH",
    }
    run_signature = stable_signature(execution_contract)
    next_index, _, part_records = scan_parts(table, paths["parts"], run_signature)
    if next_index > EXPECTED_ROWS:
        raise RuntimeError("Persisted calibration rows exceed the frozen cohort.")

    print("Runner source action :", source_action)
    print("Runner source SHA    :", source_sha)
    print("Run signature        :", run_signature)
    print("Rows already complete:", next_index)
    print("Rows remaining       :", EXPECTED_ROWS - next_index)

    session_start = next_index
    session_end = (
        EXPECTED_ROWS
        if args.session_rows == 0
        else min(EXPECTED_ROWS, session_start + args.session_rows)
    )

    model_loaded = False
    strict_load = True
    parameters_finite = True
    all_tensors_cpu = True
    completed_this_session = 0
    session_seconds = 0.0

    if session_start < session_end:
        section("[3] STRICT SELECTED-CHECKPOINT LOAD ON CPU")
        cpu_threads = max(1, min(4, os.cpu_count() or 1))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        random.seed(RUN_SEED)
        np.random.seed(RUN_SEED)
        torch.manual_seed(RUN_SEED)
        torch.use_deterministic_algorithms(True)

        implementation = import_module(IMPLEMENTATION_SOURCE, "fovea_posthoc_impl_calibration")
        primary = import_module(PRIMARY_CALIBRATION_RUNNER, "fovea_primary_calibration_helpers")
        model = implementation.build_variant(variant_id)
        checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        strict_load = (
            not incompatible.missing_keys
            and not incompatible.unexpected_keys
            and checkpoint.get("format") == "FOVEA_DR_POSTHOC_VARIANT_BEST_V1"
            and checkpoint.get("variant_id") == variant_id
            and int(checkpoint.get("epoch", -1)) == selected_epoch
            and checkpoint.get("implementation_source_sha256")
            == EXPECTED_SHA256[IMPLEMENTATION_SOURCE]
        )
        state_tensors = list(model.parameters()) + list(model.buffers())
        parameters_finite = all(bool(torch.isfinite(tensor).all()) for tensor in state_tensors)
        all_tensors_cpu = all(tensor.device.type == "cpu" for tensor in state_tensors)
        no_trainable_parameters = all(not parameter.requires_grad for parameter in model.parameters())
        model_loaded = True
        print("Device                    : CPU")
        print("Precision                 : FLOAT32")
        print("Selected epoch            :", checkpoint.get("epoch"))
        print("Strict state load         :", strict_load)
        print("All parameters finite     :", parameters_finite)
        print("All parameters on CPU     :", all_tensors_cpu)
        print("Trainable parameters      :", sum(p.numel() for p in model.parameters() if p.requires_grad))
        if not (strict_load and parameters_finite and all_tensors_cpu and no_trainable_parameters):
            raise RuntimeError("Selected checkpoint could not be loaded safely on CPU.")
        del checkpoint

        section("[4] RESUMABLE CPU FLOAT32 INFERENCE")
        print(f"This session will process indices {session_start}..{session_end - 1}.")
        session_table = table.iloc[session_start:session_end].copy().reset_index(drop=True)
        generator = torch.Generator().manual_seed(RUN_SEED)
        loader = DataLoader(
            primary.CalibrationDataset(session_table),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            persistent_workers=args.workers > 0,
            worker_init_fn=primary.worker_seed_factory(RUN_SEED),
            generator=generator,
        )
        buffer: list[dict[str, Any]] = []
        started = time.perf_counter()

        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                output = model.forward_mixed(
                    batch["image"],
                    batch["fundus_mask"],
                    batch["fov_mask"],
                    batch["modality_index"],
                    batch["quality"],
                )
                fundus_positions = torch.where(batch["fundus_mask"])[0]
                oct_positions = torch.where(~batch["fundus_mask"])[0]
                grade_map, expected_map, lesion_map, oct_map = extract_model_outputs(
                    variant_id, output, fundus_positions, oct_positions
                )

                for position in range(len(batch["relative_path"])):
                    modality = str(batch["modality"][position])
                    is_fundus = modality in {"CFP", "UWF"}
                    probability = grade_map[position] if is_fundus else oct_map[position]
                    expected_grade = expected_map[position]
                    classes = 5 if is_fundus else 3
                    probability_values = probability.numpy().astype(float)
                    valid_probability = (
                        np.isfinite(probability_values).all()
                        and np.all(probability_values >= 0.0)
                        and np.all(probability_values <= 1.0)
                        and abs(float(probability_values.sum()) - 1.0) <= 2e-4
                    )
                    if not valid_probability:
                        raise FloatingPointError("Invalid raw grade probability distribution.")

                    record: dict[str, Any] = {
                        "run_signature": run_signature,
                        "variant_id": variant_id,
                        "calibration_index": int(batch["calibration_index"][position]),
                        "relative_path": str(batch["relative_path"][position]),
                        "image_sha256": str(batch["image_sha256"][position]),
                        "modality": modality,
                        "target": int(batch["grade"][position]),
                        "raw_prediction": int(probability.argmax()),
                        "raw_expected_grade": float(expected_grade),
                        "lesion_loss_mask": float(batch["lesion_loss_mask"][position]),
                    }
                    for class_index in range(5):
                        record[f"raw_probability_{class_index}"] = (
                            float(probability[class_index]) if class_index < classes else math.nan
                        )
                    for lesion_index, lesion in enumerate(LESION_COLUMNS):
                        lesion_logit = lesion_map[position][lesion_index] if is_fundus else None
                        record[f"{lesion}_target"] = (
                            float(batch["lesion_target"][position, lesion_index])
                            if is_fundus
                            else math.nan
                        )
                        record[f"raw_{lesion}_logit"] = (
                            float(lesion_logit) if is_fundus else math.nan
                        )
                        record[f"raw_{lesion}_probability"] = (
                            float(torch.sigmoid(lesion_logit)) if is_fundus else math.nan
                        )
                    buffer.append(record)
                    completed_this_session += 1
                    if len(buffer) == PART_ROWS:
                        save_part(buffer, paths["parts"], run_signature)
                        buffer = []

                elapsed = time.perf_counter() - started
                throughput = completed_this_session / max(elapsed, 1e-12)
                total = session_start + completed_this_session
                eta_minutes = (EXPECTED_ROWS - total) / max(throughput, 1e-12) / 60.0
                if batch_number == 1 or batch_number % 8 == 0 or total == session_end:
                    print(
                        f"Batch {batch_number:04d} | session_rows={completed_this_session:4d} | "
                        f"total={total:4d}/{EXPECTED_ROWS} | {throughput:.3f} img/s | "
                        f"full-run ETA={eta_minutes:.1f} min"
                    )

        if buffer:
            if len(buffer) != session_end - (session_start + completed_this_session - len(buffer)):
                raise RuntimeError("Unexpected terminal inference buffer length.")
            save_part(buffer, paths["parts"], run_signature)
        session_seconds = time.perf_counter() - started
        del loader, model, implementation, primary
    else:
        print("\nAll 2,840 raw-inference rows are already persisted; model load skipped.")

    next_index, frames, part_records = scan_parts(table, paths["parts"], run_signature)
    run_complete = next_index == EXPECTED_ROWS
    status = "INFERENCE_COMPLETE" if run_complete else "RESUMABLE_PARTIAL"
    write_state(
        paths["state"], variant_id, status, run_signature, source_sha,
        selected_checkpoint_sha, protocol_sha, args.batch_size, next_index, part_records
    )

    session_gates = {
        **gates,
        **identity_gates,
        **table_gates,
        "runner_source_frozen_and_verified": FROZEN_SOURCE.is_file()
        and sha256_file(FROZEN_SOURCE) == source_sha,
        "selected_model_loaded_strictly_if_required": strict_load,
        "all_model_parameters_finite_if_loaded": parameters_finite,
        "all_model_tensors_on_CPU_if_loaded": all_tensors_cpu,
        "completed_at_least_one_row_or_inference_already_complete": completed_this_session > 0
        or run_complete,
        "persisted_parts_contiguous": next_index >= session_end,
        "completed_rows_do_not_exceed_2840": next_index <= EXPECTED_ROWS,
        "optimizer_not_created": True,
        "model_parameters_not_updated": True,
        "VALIDATION_not_used_for_calibration_fit": True,
        "locked_test_not_accessed": True,
    }
    if not all(session_gates.values()):
        failed = [name for name, passed in session_gates.items() if not passed]
        raise RuntimeError("Calibration session gate failed:\n" + "\n".join(failed))

    if not run_complete:
        throughput = (
            completed_this_session / session_seconds
            if completed_this_session > 0 and session_seconds > 0
            else None
        )
        evidence = {
            "stage": "0G-4D-5F-4C",
            "status": "PASS",
            "session_status": "RESUMABLE_PARTIAL",
            "variant_id": variant_id,
            "execution_backend": "CPU_FLOAT32",
            "run_signature": run_signature,
            "completed_before_session": session_start,
            "completed_this_session": completed_this_session,
            "completed_total": next_index,
            "expected_total": EXPECTED_ROWS,
            "session_seconds": session_seconds,
            "session_throughput_images_per_second": throughput,
            "inference_state": str(paths["state"]),
            "inference_state_sha256": sha256_file(paths["state"]),
            "runner_source": str(FROZEN_SOURCE),
            "runner_source_sha256": source_sha,
            "gates": {name: bool(value) for name, value in session_gates.items()},
            "calibration_images_accessed": completed_this_session > 0,
            "model_loaded": model_loaded,
            "optimizer_created": False,
            "model_parameters_updated": False,
            "temperature_fitting_started": False,
            "validation_used_for_calibration_fit": False,
            "locked_test_accessed": False,
            "next_required_stage": f"CONTINUE_{variant_id}_CPU_CALIBRATION_INFERENCE",
        }
        atomic_json(paths["latest_evidence"], evidence)
        os.sync()
        banner("STAGE 0G-4D-5F-4C STATUS: PASS — RESUMABLE PARTIAL")
        print("Variant       :", variant_id)
        print("Completed rows:", next_index, "/", EXPECTED_ROWS)
        print("Rows remaining:", EXPECTED_ROWS - next_index)
        print("Throughput    :", throughput, "img/s")
        print("Parameters updated : NO")
        print("Locked test accessed: NO")
        print("\nNEXT ACTION: RERUN THIS SAME FROZEN RUNNER FOR THE SAME VARIANT")
        return

    section("[5] COMPLETE RAW INFERENCE CONSOLIDATION")
    raw = pd.concat(frames, ignore_index=True).sort_values("calibration_index").reset_index(drop=True)
    raw_complete = (
        len(raw) == EXPECTED_ROWS
        and raw["calibration_index"].astype(int).tolist() == list(range(EXPECTED_ROWS))
        and raw["relative_path"].astype(str).tolist() == table["relative_path"].astype(str).tolist()
        and raw["image_sha256"].astype(str).tolist() == table["image_sha256"].astype(str).tolist()
        and raw["variant_id"].astype(str).eq(variant_id).all()
        and raw["relative_path"].is_unique
    )
    if not raw_complete:
        raise RuntimeError("Complete raw calibration inference consolidation failed.")
    raw_action = frozen_csv(paths["raw"], raw)
    raw_sha = sha256_file(paths["raw"])
    print("Raw action    :", raw_action)
    print("Raw rows      :", len(raw))
    print("Raw SHA-256   :", raw_sha)

    section("[6] LOCKED TEMPERATURE FIT")
    primary = import_module(PRIMARY_CALIBRATION_RUNNER, "fovea_primary_calibration_fit_helpers")
    calibrated = raw.copy()
    temperature_parameters: dict[str, Any] = {
        "protocol": "FOVEA_DR_POSTHOC_VARIANT_TEMPERATURE_PARAMETERS_V1",
        "status": "FITTED_ON_CALIBRATION_ONLY",
        "variant_id": variant_id,
        "execution_backend": "CPU_FLOAT32",
        "run_signature": run_signature,
        "selected_epoch": selected_epoch,
        "selected_checkpoint_sha256": selected_checkpoint_sha,
        "calibration_protocol_sha256": protocol_sha,
        "calibration_manifest_sha256": EXPECTED_SHA256[CALIBRATION_MANIFEST],
        "raw_predictions_sha256": raw_sha,
        "grade_temperatures": {},
        "lesion_temperatures": {},
    }
    metric_records: list[dict[str, Any]] = []
    nll_nonworsening = True
    grade_argmax_invariant = True
    lesion_threshold_invariant = True

    for modality, classes in (("CFP", 5), ("UWF", 5), ("OCT", 3)):
        mask = calibrated["modality"].eq(modality).to_numpy()
        columns = [f"raw_probability_{index}" for index in range(classes)]
        probabilities = calibrated.loc[mask, columns].to_numpy(dtype=float)
        targets = calibrated.loc[mask, "target"].to_numpy(dtype=int)
        temperature, adjusted, fit_info = primary.fit_grade_temperature(probabilities, targets)
        before = primary.grade_metrics(probabilities, targets)
        after = primary.grade_metrics(adjusted, targets)
        for class_index in range(5):
            column = f"calibrated_probability_{class_index}"
            if column not in calibrated.columns:
                calibrated[column] = math.nan
            if class_index < classes:
                calibrated.loc[mask, column] = adjusted[:, class_index]
        calibrated.loc[mask, "calibrated_prediction"] = adjusted.argmax(axis=1)
        calibrated.loc[mask, "calibrated_expected_grade"] = (
            adjusted * np.arange(classes, dtype=float)[None, :]
        ).sum(axis=1)
        invariant = bool(np.array_equal(probabilities.argmax(axis=1), adjusted.argmax(axis=1)))
        nonworsening = bool(after["NLL"] <= before["NLL"] + 1e-12)
        grade_argmax_invariant = grade_argmax_invariant and invariant
        nll_nonworsening = nll_nonworsening and nonworsening
        temperature_parameters["grade_temperatures"][modality] = {
            "temperature": float(temperature), **fit_info, "rows": int(len(targets))
        }
        metric_records.append(
            {
                "variant_id": variant_id,
                "task_type": "GRADE",
                "unit": modality,
                "rows": int(len(targets)),
                "temperature": float(temperature),
                "NLL_before": before["NLL"], "NLL_after": after["NLL"],
                "Brier_before": before["Brier_score"], "Brier_after": after["Brier_score"],
                "ECE15_before": before["ECE_15"], "ECE15_after": after["ECE_15"],
                "accuracy_before": before["accuracy"], "accuracy_after": after["accuracy"],
                "macro_F1_before": before["macro_F1"], "macro_F1_after": after["macro_F1"],
                "QWK_before": before["QWK"], "QWK_after": after["QWK"],
                "decision_invariant": invariant, "NLL_nonworsening": nonworsening,
            }
        )
        print(
            f"{modality:3s} grade | T={temperature:.6f} | "
            f"NLL {before['NLL']:.6f} -> {after['NLL']:.6f} | "
            f"ECE {before['ECE_15']:.6f} -> {after['ECE_15']:.6f}"
        )

    fundus = calibrated["modality"].isin(["CFP", "UWF"]).to_numpy()
    valid = fundus & (calibrated["lesion_loss_mask"].to_numpy(dtype=float) > 0.5)
    for lesion in LESION_COLUMNS:
        logits = calibrated.loc[valid, f"raw_{lesion}_logit"].to_numpy(dtype=float)
        raw_probabilities = calibrated.loc[valid, f"raw_{lesion}_probability"].to_numpy(dtype=float)
        targets = calibrated.loc[valid, f"{lesion}_target"].to_numpy(dtype=int)
        temperature, adjusted, fit_info = primary.fit_lesion_temperature(logits, targets)
        before = primary.lesion_metrics(raw_probabilities, targets)
        after = primary.lesion_metrics(adjusted, targets)
        column = f"calibrated_{lesion}_probability"
        calibrated[column] = math.nan
        calibrated.loc[valid, column] = adjusted
        invariant = bool(np.array_equal(raw_probabilities >= 0.5, adjusted >= 0.5))
        nonworsening = bool(after["NLL"] <= before["NLL"] + 1e-12)
        lesion_threshold_invariant = lesion_threshold_invariant and invariant
        nll_nonworsening = nll_nonworsening and nonworsening
        temperature_parameters["lesion_temperatures"][lesion] = {
            "temperature": float(temperature), **fit_info,
            "valid_rows": int(len(targets)), "positive_rows": int(targets.sum()),
        }
        metric_records.append(
            {
                "variant_id": variant_id,
                "task_type": "LESION", "unit": lesion, "rows": int(len(targets)),
                "temperature": float(temperature),
                "NLL_before": before["NLL"], "NLL_after": after["NLL"],
                "Brier_before": before["Brier_score"], "Brier_after": after["Brier_score"],
                "ECE15_before": before["ECE_15"], "ECE15_after": after["ECE_15"],
                "accuracy_before": before["accuracy_at_0_5"],
                "accuracy_after": after["accuracy_at_0_5"],
                "macro_F1_before": before["F1_at_0_5"],
                "macro_F1_after": after["F1_at_0_5"],
                "QWK_before": math.nan, "QWK_after": math.nan,
                "decision_invariant": invariant, "NLL_nonworsening": nonworsening,
            }
        )
        print(
            f"{lesion:36s} | T={temperature:.6f} | "
            f"NLL {before['NLL']:.6f} -> {after['NLL']:.6f} | "
            f"ECE {before['ECE_15']:.6f} -> {after['ECE_15']:.6f}"
        )

    temperatures = [
        item["temperature"] for item in temperature_parameters["grade_temperatures"].values()
    ] + [
        item["temperature"] for item in temperature_parameters["lesion_temperatures"].values()
    ]
    temperatures_valid = all(
        math.isfinite(value) and TEMPERATURE_MIN <= value <= TEMPERATURE_MAX
        for value in temperatures
    )
    calibrated["calibrated_prediction"] = calibrated["calibrated_prediction"].astype(int)
    probabilities_valid = primary.all_probability_rows_valid(calibrated)
    metrics = pd.DataFrame(metric_records)
    final_fit_gates = {
        "raw_inference_has_exactly_2840_rows": raw_complete,
        "three_grade_temperatures_fitted": len(temperature_parameters["grade_temperatures"]) == 3,
        "seven_lesion_temperatures_fitted": len(temperature_parameters["lesion_temperatures"]) == 7,
        "all_temperatures_finite_and_within_locked_bounds": temperatures_valid,
        "all_unit_NLL_values_nonworsening": nll_nonworsening,
        "grade_argmax_predictions_unchanged": grade_argmax_invariant,
        "lesion_predictions_at_0_5_unchanged": lesion_threshold_invariant,
        "all_calibrated_probabilities_valid": probabilities_valid,
        "optimizer_not_created": True,
        "model_parameters_not_updated": True,
        "VALIDATION_not_used_for_calibration_fit": True,
        "locked_test_not_accessed": True,
    }
    for name, passed in final_fit_gates.items():
        print(f"{name:<94}: {passed}")
    if not all(final_fit_gates.values()):
        failed = [name for name, passed in final_fit_gates.items() if not passed]
        raise RuntimeError("Locked temperature-fit gate failed:\n" + "\n".join(failed))

    calibrated_action = frozen_csv(paths["calibrated"], calibrated)
    metrics_action = frozen_csv(paths["metrics"], metrics)
    calibrated_sha = sha256_file(paths["calibrated"])
    metrics_sha = sha256_file(paths["metrics"])
    temperature_parameters.update(
        {
            "calibrated_predictions": str(paths["calibrated"]),
            "calibrated_predictions_sha256": calibrated_sha,
            "metrics": str(paths["metrics"]),
            "metrics_sha256": metrics_sha,
            "optimizer_created": False,
            "model_parameters_updated": False,
            "validation_used_for_calibration_fit": False,
            "locked_test_accessed": False,
        }
    )
    temperatures_action = frozen_json(paths["temperatures"], temperature_parameters)
    temperatures_sha = sha256_file(paths["temperatures"])
    final_hashes = {
        "raw_predictions": raw_sha,
        "calibrated_predictions": calibrated_sha,
        "temperature_parameters": temperatures_sha,
        "metrics": metrics_sha,
    }
    write_state(
        paths["state"], variant_id, "COMPLETE_AND_TEMPERATURE_FITTED", run_signature,
        source_sha, selected_checkpoint_sha, protocol_sha, args.batch_size,
        EXPECTED_ROWS, part_records, final_hashes
    )
    final_gates = {**session_gates, **final_fit_gates}
    final_evidence = {
        "stage": "0G-4D-5F-4C",
        "status": "PASS",
        "session_status": "COMPLETE_AND_TEMPERATURE_FITTED",
        "variant_id": variant_id,
        "execution_backend": "CPU_FLOAT32",
        "run_signature": run_signature,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": selected_checkpoint_sha,
        "completed_rows": EXPECTED_ROWS,
        "raw_predictions": str(paths["raw"]),
        "raw_predictions_sha256": raw_sha,
        "calibrated_predictions": str(paths["calibrated"]),
        "calibrated_predictions_sha256": calibrated_sha,
        "temperature_parameters": str(paths["temperatures"]),
        "temperature_parameters_sha256": temperatures_sha,
        "metrics": str(paths["metrics"]),
        "metrics_sha256": metrics_sha,
        "runner_source": str(FROZEN_SOURCE),
        "runner_source_sha256": source_sha,
        "inference_state": str(paths["state"]),
        "inference_state_sha256": sha256_file(paths["state"]),
        "gates": {name: bool(value) for name, value in final_gates.items()},
        "calibration_images_accessed": True,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "validation_used_for_calibration_fit": False,
        "locked_test_accessed": False,
        "next_required_stage": (
            "FREEZE_SIX_VARIANT_CALIBRATION_COMPLETION"
            if variant_id == VARIANT_IDS[-1]
            else f"CALIBRATE_{VARIANT_IDS[VARIANT_IDS.index(variant_id) + 1]}"
        ),
    }
    evidence_action = frozen_json(paths["completion_evidence"], final_evidence)
    atomic_json(paths["latest_evidence"], final_evidence)
    os.sync()

    print("\nCalibrated predictions action:", calibrated_action)
    print("Metrics action                :", metrics_action)
    print("Temperature action            :", temperatures_action)
    print("Completion evidence action    :", evidence_action)
    print("Raw predictions SHA-256       :", raw_sha)
    print("Calibrated predictions SHA-256:", calibrated_sha)
    print("Temperatures SHA-256          :", temperatures_sha)
    print("Metrics SHA-256               :", metrics_sha)
    banner("STAGE 0G-4D-5F-4C STATUS: PASS — COMPLETE")
    print("Variant                  :", variant_id)
    print("CALIBRATION rows         : 2840 / 2840")
    print("Temperatures fitted      : 10 / 10")
    print("Model parameters updated : NO")
    print("VALIDATION used for fit  : NO")
    print("Locked test accessed     : NO")
    print("NEXT ACTION              :", final_evidence["next_required_stage"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        banner("STAGE 0G-4D-5F-4C STATUS: FAIL")
        raise
