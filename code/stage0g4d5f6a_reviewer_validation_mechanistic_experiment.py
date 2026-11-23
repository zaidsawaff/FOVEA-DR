#!/usr/bin/env python3
"""Reviewer-requested FOVEA-DR validation mechanistic experiment.

This permanent, idempotent stage does four things in a fixed order:
1) freezes a reviewer-experiment protocol before image access;
2) restores and verifies only the frozen VALIDATION cache;
3) loads the frozen seed-2026 full-model checkpoint without an optimizer;
4) exports gate, counterfactual, and lesion-map analyses on VALIDATION only.

CALIBRATION and LOCKED_INTERNAL_TEST are never read. No parameter is updated.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
RAW_DIR = ROOT / "Data/Raw/MMRDR_Figshare_v2_Archives"
VALIDATION_CACHE = Path("/content/MMRDR_validation_v1")
VOLUME_VIEW = Path("/content/MMRDR_reviewer_validation_volume_view")

SPLIT = ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
VALIDATION_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_VALIDATION_local_cache_manifest_v1.csv"
CORE_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
BASELINE_FREEZE = ROOT / "Metadata/mmrdr_fovea_dr_baseline_seed2026_selection_freeze_v1.json"
CHECKPOINT = ROOT / "Checkpoints/FOVEA_DR/baseline_seed2026/selected_epoch10_frozen.pt"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f6a_reviewer_validation_mechanistic_experiment.py"
PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_reviewer_experiment_protocol_v1.json"
PROTOCOL_EVIDENCE = ROOT / "Evidence/stage0g4d5f6a_reviewer_experiment_protocol_lock.json"
CACHE_EVIDENCE = ROOT / "Evidence/stage0g4d5f6a_validation_only_cache_restore.json"
CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_validation_mechanistic_audit_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5f6b_validation_mechanistic_audit.json"

RESULT_ROOT = ROOT / "Results/FOVEA_DR/reviewer_experiments/validation_mechanistic_v1"
SHARD_ROOT = RESULT_ROOT / "numeric_shards"
FIGURE_ROOT = RESULT_ROOT / "figures"
PANEL_ROOT = FIGURE_ROOT / "qualitative_panels"

EXPECTED_SPLIT_SHA = "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140"
EXPECTED_VALIDATION_MANIFEST_SHA = "d94c5613b6d030dd6e99246b1b72f77746b5223139a4597bef733eb38a5786af"
EXPECTED_CORE_SHA = "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9"
EXPECTED_CHECKPOINT_SHA = "89f5e87b647680a9094c55bf647dc1dd2abb624681e733ffb453e30e4c11ff1f"

EXPECTED_VALIDATION_ROWS = 2839
EXPECTED_FUNDUS_ROWS = 2482
EXPECTED_MODALITY_COUNTS = {"CFP": 1314, "OCT": 357, "UWF": 1168}
INPUT_SIZE = 512
BATCH_SIZE = 16
COUNTERFACTUAL_MARGIN = 0.25
PERFORMANCE_IDENTITY_TOLERANCE = 1.0e-4

EXPECTED_VOLUME_SIZES = {
    **{f"MMRDR.zip.{index:03d}": 2_097_152_000 for index in range(1, 9)},
    "MMRDR.zip.009": 1_833_101_731,
}

ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    data = json_bytes(payload)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    atomic_bytes(path, data)
    return "CREATED_AND_FROZEN"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError(f"A different reviewer-experiment source exists: {FROZEN_SOURCE}")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    if sha256_file(FROZEN_SOURCE) != source_sha:
        raise RuntimeError("Source freeze verification failed")
    return "COPIED_AND_FROZEN", source_sha


def normalized_path(value: Any) -> str:
    raw = str(value).strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise RuntimeError(f"Unsafe relative path: {raw!r}")
    return str(path)


def has_locked_test_token(value: str) -> bool:
    forbidden = {"ts", "test", "testing", "locked_test", "locked_internal_test"}
    return any(part.lower() in forbidden for part in PurePosixPath(value).parts)


def protocol_payload(source_sha: str) -> dict[str, Any]:
    baseline = load_json(BASELINE_FREEZE)
    return {
        "protocol_id": "MMRDR_FOVEA_DR_REVIEWER_EXPERIMENTS_V1",
        "status": "FROZEN_BEFORE_REVIEWER_EXPERIMENTS",
        "freeze_date_utc": "2026-10-30",
        "source_sha256": source_sha,
        "frozen_parents": {
            "development_split_sha256": EXPECTED_SPLIT_SHA,
            "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA,
            "production_model_source_sha256": EXPECTED_CORE_SHA,
            "selected_full_model_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "baseline_selection_contract_sha256": sha256_file(BASELINE_FREEZE),
            "selected_epoch": int(baseline["selected_epoch"]),
            "run_seed": int(baseline["run_seed"]),
        },
        "data_access_policy": {
            "TRAIN": "ALLOWED_FOR_PRESPECIFIED_RETRAINING_EXPERIMENTS_ONLY",
            "VALIDATION": "ALLOWED_FOR_SELECTION_AND_MECHANISTIC_CHARACTERIZATION",
            "CALIBRATION": "PROHIBITED_FOR_STAGE_E1; ALLOWED_ONLY_AFTER_MODEL_SELECTION_IN_LATER_STAGES",
            "LOCKED_INTERNAL_TEST": "PROHIBITED_FOR_ALL_REVIEWER-REQUESTED DEVELOPMENT EXPERIMENTS",
            "EXTERNAL_DATA": "ALLOWED_ONLY_AFTER_MODEL_AND_MAPPING_FREEZE",
        },
        "experiments_in_locked_order": [
            {
                "id": "E1_VALIDATION_MECHANISTIC_AUDIT",
                "purpose": "Quantify reliability-gate behavior, lesion-evidence maps, and counterfactual directional consistency",
                "partition": "VALIDATION_ONLY",
                "parameter_updates": False,
                "primary_outputs": [
                    "per-image gate and route-disagreement table",
                    "counterfactual direction and margin-satisfaction rates",
                    "predeclared qualitative lesion-evidence panels",
                ],
                "checkpoint_identity_rule": {
                    "metrics": ["CFP_QWK", "UWF_QWK", "mean_fundus_QWK", "CFP_NLL", "UWF_NLL", "mean_fundus_NLL"],
                    "absolute_tolerance": PERFORMANCE_IDENTITY_TOLERANCE,
                },
                "selection_rules": [
                    "highest route disagreement among correct predictions per fundus modality",
                    "highest route disagreement among incorrect predictions per fundus modality",
                    "highest and lowest clinical-route gate per fundus modality",
                    "largest add-PDR and remove-PDR expected-grade shifts per fundus modality",
                ],
            },
            {
                "id": "E2_COMPONENT_ABLATIONS",
                "variants": [
                    "NO_FIELD_OF_VIEW_ENCODING",
                    "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE",
                    "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE",
                ],
                "partitions": ["TRAIN", "VALIDATION"],
                "locked_test": False,
            },
            {
                "id": "E3_MULTI_SEED_ROBUSTNESS",
                "configurations": ["FOVEA_DR_FULL", "B1", "A1", "A3"],
                "seeds": [2026, 3407, 7919],
                "reporting": "per-seed values plus mean and standard deviation; no p-value claim from n=3",
                "locked_test": False,
            },
            {
                "id": "E4_RETINAL_FOUNDATION_BASELINE",
                "minimum_requirement": "one officially released retinal foundation encoder trained under the same frozen TRAIN/VALIDATION protocol",
                "locked_test": False,
            },
            {
                "id": "E5_EXTERNAL_CFP_VALIDATION",
                "minimum_requirement": "externally sourced five-class CFP cohort with mapping frozen before inference",
                "internal_locked_test": False,
            },
        ],
        "primary_metrics": ["mean_CFP_UWF_QWK", "mean_CFP_UWF_NLL"],
        "secondary_metrics": ["modality_QWK", "macro_F1", "macro_ROC_AUC", "ECE", "Brier"],
        "claim_policy": "No universal-superiority, causal-component, localization-accuracy, or deployment claim without the corresponding experiment",
    }


def freeze_protocol(source_sha: str) -> tuple[str, str]:
    payload = protocol_payload(source_sha)
    action = frozen_json(PROTOCOL, payload)
    protocol_sha = sha256_file(PROTOCOL)
    gates = {
        "source_frozen": FROZEN_SOURCE.is_file() and sha256_file(FROZEN_SOURCE) == source_sha,
        "split_hash_verified": SPLIT.is_file() and sha256_file(SPLIT) == EXPECTED_SPLIT_SHA,
        "validation_manifest_hash_verified": VALIDATION_MANIFEST.is_file() and sha256_file(VALIDATION_MANIFEST) == EXPECTED_VALIDATION_MANIFEST_SHA,
        "core_source_hash_verified": CORE_SOURCE.is_file() and sha256_file(CORE_SOURCE) == EXPECTED_CORE_SHA,
        "baseline_freeze_status_verified": BASELINE_FREEZE.is_file() and load_json(BASELINE_FREEZE).get("status") == "FROZEN",
        "selected_checkpoint_hash_verified": CHECKPOINT.is_file() and sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA,
        "locked_test_prohibited": payload["data_access_policy"]["LOCKED_INTERNAL_TEST"].startswith("PROHIBITED"),
        "experiment_order_locked": len(payload["experiments_in_locked_order"]) == 5,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Reviewer protocol gates failed: {[k for k, v in gates.items() if not v]}")
    evidence = {
        "stage": "0G-4D-5F-6A",
        "purpose": "REVIEWER_EXPERIMENT_PROTOCOL_LOCK",
        "status": "PASS",
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_sha,
        "source_sha256": source_sha,
        "gates": gates,
        "model_loaded": False,
        "images_accessed": False,
        "parameters_updated": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
    }
    frozen_json(PROTOCOL_EVIDENCE, evidence)
    return action, protocol_sha


def completed_result_is_valid() -> bool:
    if not EVIDENCE.is_file():
        return False
    try:
        evidence = load_json(EVIDENCE)
        if evidence.get("status") != "PASS":
            return False
        if not all(bool(value) for value in evidence.get("gates", {}).values()):
            return False
        contract = Path(str(evidence.get("contract", "")))
        if not contract.is_file() or sha256_file(contract) != evidence.get("contract_sha256"):
            return False
        for record in evidence.get("output_files", []):
            path = Path(str(record.get("path", "")))
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                return False
        return True
    except Exception:
        return False


def load_validation_manifest() -> pd.DataFrame:
    if sha256_file(VALIDATION_MANIFEST) != EXPECTED_VALIDATION_MANIFEST_SHA:
        raise RuntimeError("Frozen VALIDATION manifest hash mismatch")
    frame = pd.read_csv(VALIDATION_MANIFEST)
    required = {"relative_path", "modality", "image_sha256", "file_size_bytes"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"VALIDATION manifest missing columns: {missing}")
    frame = frame.copy()
    frame["relative_path"] = frame["relative_path"].map(normalized_path)
    frame["modality"] = frame["modality"].astype(str).str.upper().str.strip()
    frame["image_sha256"] = frame["image_sha256"].astype(str).str.lower()
    frame["file_size_bytes"] = pd.to_numeric(frame["file_size_bytes"], errors="raise").astype("int64")
    counts = {str(k): int(v) for k, v in frame["modality"].value_counts().to_dict().items()}
    checks = {
        "row_count": len(frame) == EXPECTED_VALIDATION_ROWS,
        "modality_counts": counts == EXPECTED_MODALITY_COUNTS,
        "unique_paths": bool(frame["relative_path"].is_unique),
        "hash_schema": bool(frame["image_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()),
        "positive_sizes": bool((frame["file_size_bytes"] > 0).all()),
        "no_locked_test_tokens": not bool(frame["relative_path"].map(has_locked_test_token).any()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"VALIDATION manifest gates failed: {checks}")
    return frame.sort_values("relative_path").reset_index(drop=True)


def prepare_volume_view() -> tuple[Path, list[dict[str, Any]]]:
    if shutil.which("7z") is None:
        raise RuntimeError("7z is unavailable in the current runtime")
    VOLUME_VIEW.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name, expected_size in EXPECTED_VOLUME_SIZES.items():
        final_path = RAW_DIR / name
        part_path = RAW_DIR / f"{name}.part"
        if final_path.is_file() and final_path.stat().st_size == expected_size:
            source, source_type = final_path, "FINAL"
        elif part_path.is_file() and part_path.stat().st_size == expected_size:
            source, source_type = part_path, "COMPLETE_PART_ALIAS"
        else:
            raise RuntimeError(f"Complete archive volume unavailable: {name}")
        alias = VOLUME_VIEW / name
        if alias.is_symlink():
            alias.unlink()
        elif alias.exists():
            raise RuntimeError(f"Refusing to replace non-symlink volume alias: {alias}")
        os.symlink(str(source), str(alias))
        records.append({"volume": name, "source": source.name, "source_type": source_type, "size_bytes": int(expected_size)})
    archive = VOLUME_VIEW / "MMRDR.zip.001"
    listing = subprocess.run(["7z", "l", str(archive)], capture_output=True, text=True, check=False)
    if listing.returncode != 0 or "Volumes = 9" not in listing.stdout:
        raise RuntimeError("Nine-volume archive verification failed")
    return archive, records


def extract_members(archive: Path, members: list[str], list_path: Path) -> None:
    VALIDATION_CACHE.mkdir(parents=True, exist_ok=True)
    if not members:
        print("VALIDATION cache already present; extraction not required.")
        return
    atomic_bytes(list_path, ("\n".join(members) + "\n").encode("utf-8"))
    result = subprocess.run(
        ["7z", "x", "-y", "-aoa", str(archive), f"-o{VALIDATION_CACHE}", f"@{list_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    print("Selective extraction return code:", result.returncode)
    print(result.stdout[-1800:])
    if result.returncode != 0:
        print(result.stderr[-1800:])
        raise RuntimeError("VALIDATION selective extraction failed")


def verify_validation_cache(frame: pd.DataFrame) -> tuple[list[str], int]:
    failures: list[str] = []
    verified_bytes = 0
    for index, row in enumerate(frame.to_dict("records"), start=1):
        relative = str(row["relative_path"])
        path = VALIDATION_CACHE / relative
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(row["file_size_bytes"]):
                raise RuntimeError("size mismatch")
            if sha256_file(path) != str(row["image_sha256"]):
                raise RuntimeError("SHA-256 mismatch")
            with Image.open(path) as image:
                image.verify()
            verified_bytes += int(row["file_size_bytes"])
        except Exception:
            failures.append(relative)
        if index % 500 == 0 or index == len(frame):
            print(f"Verified {index:4d}/{len(frame)} VALIDATION images | failures={len(failures)}")
    return failures, verified_bytes


def restore_validation_cache(source_sha: str, protocol_sha: str) -> pd.DataFrame:
    frame = load_validation_manifest()
    archive, volumes = prepare_volume_view()
    missing = [
        str(row["relative_path"])
        for row in frame.to_dict("records")
        if not (VALIDATION_CACHE / str(row["relative_path"])).is_file()
        or (VALIDATION_CACHE / str(row["relative_path"])).stat().st_size != int(row["file_size_bytes"])
    ]
    size_map = dict(zip(frame["relative_path"], frame["file_size_bytes"]))
    extraction_bytes = sum(int(size_map[path]) for path in missing)
    free_bytes = shutil.disk_usage("/content").free
    if free_bytes < extraction_bytes + 5 * 1024**3:
        raise RuntimeError("Insufficient local disk for VALIDATION cache plus 5 GiB headroom")
    print("VALIDATION images needing extraction:", len(missing))
    print("Required extraction GB             :", f"{extraction_bytes / 1024**3:.3f}")
    extract_members(archive, missing, Path("/content/fovea_dr_reviewer_validation_restore_list.txt"))
    failures, verified_bytes = verify_validation_cache(frame)
    if failures:
        print("Repairing failed VALIDATION members:", len(failures))
        extract_members(archive, failures, Path("/content/fovea_dr_reviewer_validation_repair_list.txt"))
        failures, verified_bytes = verify_validation_cache(frame)
    if failures:
        raise RuntimeError(f"VALIDATION cache verification failed for {len(failures)} images")
    payload = {
        "stage": "0G-4D-5F-6A-CACHE",
        "purpose": "VALIDATION_ONLY_CACHE_RESTORE_FOR_REVIEWER_EXPERIMENT",
        "status": "PASS",
        "source_sha256": source_sha,
        "protocol_sha256": protocol_sha,
        "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA,
        "validation_rows": int(len(frame)),
        "validation_modality_counts": {str(k): int(v) for k, v in frame["modality"].value_counts().to_dict().items()},
        "verified_bytes": int(verified_bytes),
        "archive_volumes": volumes,
        "train_images_accessed": False,
        "validation_images_accessed": True,
        "calibration_accessed": False,
        "locked_test_accessed": False,
    }
    frozen_json(CACHE_EVIDENCE, payload)
    return frame


def import_core() -> Any:
    specification = importlib.util.spec_from_file_location("fovea_dr_reviewer_core_v1", CORE_SOURCE)
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not import the frozen FOVEA-DR source")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ValidationFundusDataset(Dataset):
    def __init__(self, table: pd.DataFrame):
        self.table = table.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        path = Path(str(row["local_path"]))
        if not path.is_file() or path.stat().st_size != int(row["file_size_bytes"]):
            raise RuntimeError(f"VALIDATION cache mismatch: {path}")
        with Image.open(path) as image:
            raw = TF.to_tensor(image.convert("RGB"))
        raw = TF.resize(raw, [INPUT_SIZE, INPUT_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
        quality = raw.std().clamp(0.0, 1.0)
        fov = (raw.amax(dim=0, keepdim=True) > (8.0 / 255.0)).to(raw.dtype)
        normalized = TF.normalize(raw, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        modality = str(row["modality"])
        return {
            "image": normalized,
            "fov_mask": fov,
            "quality": torch.tensor(float(quality), dtype=torch.float32),
            "fundus_mask": torch.tensor(True, dtype=torch.bool),
            "modality_index": torch.tensor(0 if modality == "CFP" else 1, dtype=torch.long),
            "grade": torch.tensor(int(row["grade"]), dtype=torch.long),
            "modality": modality,
            "relative_path": str(row["relative_path"]),
        }


def build_validation_table(manifest: pd.DataFrame, lesion_columns: list[str]) -> pd.DataFrame:
    if sha256_file(SPLIT) != EXPECTED_SPLIT_SHA:
        raise RuntimeError("Frozen development split hash mismatch")
    split = pd.read_csv(SPLIT)
    split = split[split["development_split"].astype(str) == "VALIDATION"].copy()
    split["relative_path"] = split["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    manifest_small = manifest[["relative_path", "file_size_bytes", "image_sha256"]].copy()
    table = split.merge(manifest_small, on="relative_path", how="inner", validate="one_to_one")
    table["local_path"] = table["relative_path"].map(lambda value: str(VALIDATION_CACHE / str(value)))
    required = {"relative_path", "modality", "grade", *lesion_columns}
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f"Frozen validation table missing columns: {missing}")
    if len(table) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("Frozen validation table row count mismatch")
    fundus = table[table["modality"].isin(["CFP", "UWF"])].copy()
    if len(fundus) != EXPECTED_FUNDUS_ROWS:
        raise RuntimeError("Fundus validation row count mismatch")
    return fundus.sort_values("relative_path").reset_index(drop=True)


def build_model(core: Any, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen selected checkpoint hash mismatch")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    checks = {
        "format": checkpoint.get("format") == "FOVEA_DR_BASELINE_BEST_MODEL_V1",
        "seed": int(checkpoint.get("run_seed", -1)) == 2026,
        "epoch": int(checkpoint.get("epoch", -1)) == 10,
        "production_source": checkpoint.get("production_source_sha256") == EXPECTED_CORE_SHA,
        "state_dict_present": isinstance(checkpoint.get("model_state_dict"), dict),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen checkpoint identity failed: {checks}")
    original_builder = core.efficientnet_v2_s

    def weightless_builder(*args: Any, **kwargs: Any) -> torch.nn.Module:
        kwargs["weights"] = None
        kwargs["progress"] = False
        return original_builder(*args, **kwargs)

    core.efficientnet_v2_s = weightless_builder
    try:
        model = core.FOVEADRTrainingModel()
    finally:
        core.efficientnet_v2_s = original_builder
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {incompatible}")
    model.to(device).eval()
    return model, checkpoint


def valid_existing_shard(
    path: Path,
    expected_paths: list[str],
    source_sha: str,
    protocol_sha: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
        return (
            frame["relative_path"].astype(str).tolist() == expected_paths
            and set(frame["analysis_source_sha256"].astype(str)) == {source_sha}
            and set(frame["analysis_protocol_sha256"].astype(str)) == {protocol_sha}
            and set(frame["checkpoint_sha256"].astype(str)) == {EXPECTED_CHECKPOINT_SHA}
        )
    except Exception:
        return False


def numeric_records(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    lesion_columns: list[str],
    source_sha: str,
    protocol_sha: str,
) -> pd.DataFrame:
    tensor_batch = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        output = model.forward_mixed(
            tensor_batch["image"],
            tensor_batch["fundus_mask"],
            tensor_batch["fov_mask"],
            tensor_batch["modality_index"],
            tensor_batch["quality"],
        )
        fundus = output["fundus"]
        evidence = fundus["lesions"]["evidence_strength"]
        latent_probabilities = fundus["latent"]["probabilities"]
        modality = tensor_batch["modality_index"]
        quality = tensor_batch["quality"]
        original = fundus["original"]
        no_lesion = model.fuse(torch.zeros_like(evidence), latent_probabilities, modality, quality)
        remove_pdr_evidence = torch.cat([evidence[:, :4], torch.zeros_like(evidence[:, 4:])], dim=1)
        add_pdr_evidence = torch.cat([evidence[:, :4], torch.ones_like(evidence[:, 4:])], dim=1)
        remove_pdr = model.fuse(remove_pdr_evidence, latent_probabilities, modality, quality)
        add_pdr = model.fuse(add_pdr_evidence, latent_probabilities, modality, quality)

    def cpu(value: torch.Tensor) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    fused_prob = cpu(original["fusion"]["probabilities"])
    clinical_prob = cpu(original["clinical"]["probabilities"])
    latent_prob = cpu(fundus["latent"]["probabilities"])
    evidence_np = cpu(evidence)
    burden_np = cpu(fundus["lesions"]["burden"])
    presence_np = cpu(fundus["lesions"]["presence"])
    entropy_np = cpu(fundus["lesions"]["spatial_entropy"])
    fused_expected = cpu(original["fusion"]["expected_grade"])
    clinical_expected = cpu(original["clinical"]["expected_grade"])
    latent_expected = cpu(fundus["latent"]["expected_grade"])
    gate = cpu(original["fusion"]["gate"])
    js = cpu(original["fusion"]["js_disagreement"])
    route_lesion_entropy = cpu(original["fusion"]["lesion_entropy"])
    no_expected = cpu(no_lesion["fusion"]["expected_grade"])
    remove_expected = cpu(remove_pdr["fusion"]["expected_grade"])
    add_expected = cpu(add_pdr["fusion"]["expected_grade"])
    no_gate = cpu(no_lesion["fusion"]["gate"])
    remove_gate = cpu(remove_pdr["fusion"]["gate"])
    add_gate = cpu(add_pdr["fusion"]["gate"])
    targets = cpu(tensor_batch["grade"]).astype(int)
    qualities = cpu(tensor_batch["quality"])

    records: list[dict[str, Any]] = []
    for index, relative_path in enumerate(batch["relative_path"]):
        record: dict[str, Any] = {
            "analysis_source_sha256": source_sha,
            "analysis_protocol_sha256": protocol_sha,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "relative_path": str(relative_path),
            "modality": str(batch["modality"][index]),
            "target": int(targets[index]),
            "fused_prediction": int(fused_prob[index].argmax()),
            "fused_expected_grade": float(fused_expected[index]),
            "clinical_expected_grade": float(clinical_expected[index]),
            "latent_expected_grade": float(latent_expected[index]),
            "clinical_route_gate": float(gate[index]),
            "route_js_disagreement": float(js[index]),
            "route_lesion_entropy": float(route_lesion_entropy[index]),
            "image_quality_proxy": float(qualities[index]),
            "remove_all_expected_grade": float(no_expected[index]),
            "remove_pdr_expected_grade": float(remove_expected[index]),
            "add_pdr_expected_grade": float(add_expected[index]),
            "remove_all_gate": float(no_gate[index]),
            "remove_pdr_gate": float(remove_gate[index]),
            "add_pdr_gate": float(add_gate[index]),
            "any_lesion_support": float(1.0 - np.prod(1.0 - evidence_np[index])),
            "pdr_support": float(1.0 - np.prod(1.0 - evidence_np[index, 4:])),
        }
        for class_index in range(5):
            record[f"fused_probability_{class_index}"] = float(fused_prob[index, class_index])
            record[f"clinical_probability_{class_index}"] = float(clinical_prob[index, class_index])
            record[f"latent_probability_{class_index}"] = float(latent_prob[index, class_index])
        for lesion_index, lesion in enumerate(lesion_columns):
            record[f"{lesion}_evidence_strength"] = float(evidence_np[index, lesion_index])
            record[f"{lesion}_presence"] = float(presence_np[index, lesion_index])
            record[f"{lesion}_burden"] = float(burden_np[index, lesion_index])
            record[f"{lesion}_spatial_entropy"] = float(entropy_np[index, lesion_index])
        records.append(record)
    return pd.DataFrame(records)


def run_resumable_numeric_inference(
    model: torch.nn.Module,
    table: pd.DataFrame,
    device: torch.device,
    lesion_columns: list[str],
    source_sha: str,
    protocol_sha: str,
) -> pd.DataFrame:
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    total_batches = math.ceil(len(table) / BATCH_SIZE)
    shard_paths: list[Path] = []
    for batch_index in range(total_batches):
        start = batch_index * BATCH_SIZE
        end = min(len(table), start + BATCH_SIZE)
        subset = table.iloc[start:end].copy()
        expected_paths = subset["relative_path"].astype(str).tolist()
        shard = SHARD_ROOT / f"batch_{batch_index:04d}_{start:04d}_{end - 1:04d}.csv"
        shard_paths.append(shard)
        if valid_existing_shard(shard, expected_paths, source_sha, protocol_sha):
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
                print(f"Numeric batch {batch_index + 1:3d}/{total_batches}: reused verified shard")
            continue
        loader = DataLoader(
            ValidationFundusDataset(subset),
            batch_size=len(subset),
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        batch = next(iter(loader))
        frame = numeric_records(
            model,
            batch,
            device,
            lesion_columns,
            source_sha,
            protocol_sha,
        )
        if frame["relative_path"].astype(str).tolist() != expected_paths:
            raise RuntimeError("Inference shard path order mismatch")
        atomic_csv(shard, frame)
        if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
            print(f"Numeric batch {batch_index + 1:3d}/{total_batches}: committed")
    combined = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
    if len(combined) != len(table) or not combined["relative_path"].is_unique:
        raise RuntimeError("Combined mechanistic inference table is incomplete")
    expected = table["relative_path"].astype(str).tolist()
    if combined["relative_path"].astype(str).tolist() != expected:
        raise RuntimeError("Combined mechanistic inference order mismatch")
    return combined


def summarize_mechanisms(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame["correct"] = frame["fused_prediction"].astype(int) == frame["target"].astype(int)
    frame["remove_all_shift"] = frame["remove_all_expected_grade"] - frame["fused_expected_grade"]
    frame["remove_pdr_shift"] = frame["remove_pdr_expected_grade"] - frame["fused_expected_grade"]
    frame["add_pdr_shift"] = frame["add_pdr_expected_grade"] - frame["fused_expected_grade"]
    tolerance = 1e-6
    frame["remove_all_direction_satisfied"] = frame["remove_all_shift"] <= tolerance
    frame["remove_pdr_direction_satisfied"] = frame["remove_pdr_shift"] <= tolerance
    frame["add_pdr_direction_satisfied"] = frame["add_pdr_shift"] >= -tolerance
    frame["remove_all_margin_satisfied"] = (
        frame["fused_expected_grade"] - frame["remove_all_expected_grade"]
        >= COUNTERFACTUAL_MARGIN * frame["any_lesion_support"] - tolerance
    )
    frame["remove_pdr_margin_satisfied"] = (
        frame["fused_expected_grade"] - frame["remove_pdr_expected_grade"]
        >= COUNTERFACTUAL_MARGIN * frame["pdr_support"] - tolerance
    )
    frame["add_pdr_margin_satisfied"] = (
        frame["add_pdr_expected_grade"] - frame["fused_expected_grade"]
        >= COUNTERFACTUAL_MARGIN * (1.0 - frame["pdr_support"]) - tolerance
    )

    gate_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    counter_rows: list[dict[str, Any]] = []
    groups = [("ALL", frame)] + [(str(modality), subset) for modality, subset in frame.groupby("modality", sort=True)]
    for modality, subset in groups:
        gate_rows.append({
            "modality": modality,
            "n": int(len(subset)),
            "gate_mean": float(subset["clinical_route_gate"].mean()),
            "gate_sd": float(subset["clinical_route_gate"].std(ddof=1)),
            "gate_median": float(subset["clinical_route_gate"].median()),
            "gate_q25": float(subset["clinical_route_gate"].quantile(0.25)),
            "gate_q75": float(subset["clinical_route_gate"].quantile(0.75)),
            "route_js_mean": float(subset["route_js_disagreement"].mean()),
            "quality_mean": float(subset["image_quality_proxy"].mean()),
            "fused_accuracy": float(subset["correct"].mean()),
        })
        for variable in ["route_js_disagreement", "image_quality_proxy", "route_lesion_entropy"]:
            coefficient, p_value = spearmanr(subset["clinical_route_gate"], subset[variable])
            correlation_rows.append({
                "modality": modality,
                "n": int(len(subset)),
                "gate_correlate": variable,
                "spearman_rho": float(coefficient),
                "two_sided_p_value_descriptive": float(p_value),
            })
        counter_rows.append({
            "modality": modality,
            "n": int(len(subset)),
            "remove_all_direction_rate": float(subset["remove_all_direction_satisfied"].mean()),
            "remove_pdr_direction_rate": float(subset["remove_pdr_direction_satisfied"].mean()),
            "add_pdr_direction_rate": float(subset["add_pdr_direction_satisfied"].mean()),
            "remove_all_margin_rate": float(subset["remove_all_margin_satisfied"].mean()),
            "remove_pdr_margin_rate": float(subset["remove_pdr_margin_satisfied"].mean()),
            "add_pdr_margin_rate": float(subset["add_pdr_margin_satisfied"].mean()),
            "mean_remove_all_shift": float(subset["remove_all_shift"].mean()),
            "mean_remove_pdr_shift": float(subset["remove_pdr_shift"].mean()),
            "mean_add_pdr_shift": float(subset["add_pdr_shift"].mean()),
        })
    return pd.DataFrame(gate_rows), pd.DataFrame(correlation_rows), pd.DataFrame(counter_rows)


def validation_performance_identity(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    baseline = load_json(BASELINE_FREEZE)
    expected = baseline["selected_metrics"]
    rows: list[dict[str, Any]] = []
    actual: dict[str, float] = {}
    probability_columns = [f"fused_probability_{index}" for index in range(5)]
    for modality in ["CFP", "UWF"]:
        subset = frame[frame["modality"] == modality]
        target = subset["target"].to_numpy(dtype=int)
        prediction = subset["fused_prediction"].to_numpy(dtype=int)
        probabilities = subset[probability_columns].to_numpy(dtype=float)
        actual[f"{modality}_QWK"] = float(
            cohen_kappa_score(target, prediction, weights="quadratic")
        )
        actual[f"{modality}_NLL"] = float(
            -np.log(np.clip(probabilities[np.arange(len(target)), target], 1.0e-7, 1.0)).mean()
        )
    actual["mean_fundus_QWK"] = 0.5 * (actual["CFP_QWK"] + actual["UWF_QWK"])
    actual["mean_fundus_NLL"] = 0.5 * (actual["CFP_NLL"] + actual["UWF_NLL"])

    checks: dict[str, bool] = {}
    for metric in [
        "CFP_QWK", "UWF_QWK", "mean_fundus_QWK",
        "CFP_NLL", "UWF_NLL", "mean_fundus_NLL",
    ]:
        expected_value = float(expected[metric])
        difference = abs(actual[metric] - expected_value)
        passed = difference <= PERFORMANCE_IDENTITY_TOLERANCE
        checks[f"checkpoint_identity::{metric}"] = passed
        rows.append({
            "metric": metric,
            "recomputed_value": actual[metric],
            "frozen_expected_value": expected_value,
            "absolute_difference": difference,
            "absolute_tolerance": PERFORMANCE_IDENTITY_TOLERANCE,
            "passed": passed,
        })
    return pd.DataFrame(rows), checks


def select_examples(frame: pd.DataFrame) -> pd.DataFrame:
    selections: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(subset: pd.DataFrame, sort_column: str, ascending: bool, reason: str) -> None:
        if subset.empty:
            return
        for _, row in subset.sort_values([sort_column, "relative_path"], ascending=[ascending, True]).iterrows():
            path = str(row["relative_path"])
            if path not in seen:
                seen.add(path)
                record = row.to_dict()
                record["selection_reason"] = reason
                selections.append(record)
                return

    for modality in ["CFP", "UWF"]:
        subset = frame[frame["modality"] == modality].copy()
        add(subset[subset["correct"]], "route_js_disagreement", False, f"{modality}:highest_disagreement_correct")
        add(subset[~subset["correct"]], "route_js_disagreement", False, f"{modality}:highest_disagreement_incorrect")
        add(subset, "clinical_route_gate", False, f"{modality}:highest_clinical_gate")
        add(subset, "clinical_route_gate", True, f"{modality}:lowest_clinical_gate")
        add(subset, "add_pdr_shift", False, f"{modality}:largest_add_pdr_shift")
        add(subset, "remove_pdr_shift", True, f"{modality}:largest_remove_pdr_shift")
    selected = pd.DataFrame(selections)
    if len(selected) < 8:
        raise RuntimeError("Predeclared qualitative selection produced too few distinct examples")
    selected.insert(0, "panel_id", [f"P{index:02d}" for index in range(1, len(selected) + 1)])
    return selected


def infer_maps_for_row(
    model: torch.nn.Module,
    row: pd.Series,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    dataset = ValidationFundusDataset(pd.DataFrame([row]))
    item = dataset[0]
    image = item["image"].unsqueeze(0).to(device)
    fov = item["fov_mask"].unsqueeze(0).to(device)
    fundus_mask = torch.ones(1, dtype=torch.bool, device=device)
    modality = item["modality_index"].view(1).to(device)
    quality = item["quality"].view(1).to(device)
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        output = model.forward_mixed(image, fundus_mask, fov, modality, quality)["fundus"]
    maps = output["lesions"]["spatial_maps"][0].detach().float().cpu().numpy()
    evidence = output["lesions"]["evidence_strength"][0].detach().float().cpu().numpy()
    probabilities = output["original"]["fusion"]["probabilities"][0].detach().float().cpu().numpy()
    gate = float(output["original"]["fusion"]["gate"][0].detach().float().cpu())
    with Image.open(Path(str(row["local_path"]))) as opened:
        original = np.asarray(opened.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR))
    return original, maps, evidence, gate, int(probabilities.argmax())


def normalized_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lower, upper = float(values.min()), float(values.max())
    if upper <= lower:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


def render_figures(
    model: torch.nn.Module,
    selected: pd.DataFrame,
    table: pd.DataFrame,
    frame: pd.DataFrame,
    gate_summary: pd.DataFrame,
    counter_summary: pd.DataFrame,
    lesion_columns: list[str],
    device: torch.device,
) -> list[Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    lookup = table.set_index("relative_path")

    gate_figure = FIGURE_ROOT / "gate_behavior_summary.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    values = [frame.loc[frame["modality"] == modality, "clinical_route_gate"].to_numpy() for modality in ["CFP", "UWF"]]
    axes[0].boxplot(values, tick_labels=["CFP", "UWF"], showfliers=False)
    axes[0].set_ylabel("Clinical-route gate")
    axes[0].set_title("Reliability-gate distribution")
    colors = {"CFP": "#2878B5", "UWF": "#E07A1F"}
    for modality in ["CFP", "UWF"]:
        subset = frame[frame["modality"] == modality]
        axes[1].scatter(subset["route_js_disagreement"], subset["clinical_route_gate"], s=8, alpha=0.22, color=colors[modality], label=modality)
    axes[1].set_xlabel("Clinical–latent Jensen–Shannon disagreement")
    axes[1].set_ylabel("Clinical-route gate")
    axes[1].set_title("Gate response to route disagreement")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(gate_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(gate_figure)

    counter_figure = FIGURE_ROOT / "counterfactual_consistency_summary.png"
    plot = counter_summary[counter_summary["modality"].isin(["CFP", "UWF", "ALL"])].copy()
    x = np.arange(len(plot))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, (column, label, color) in enumerate([
        ("remove_all_margin_rate", "Remove all lesions", "#2878B5"),
        ("remove_pdr_margin_rate", "Remove proliferative evidence", "#E07A1F"),
        ("add_pdr_margin_rate", "Add proliferative evidence", "#2A9D8F"),
    ]):
        ax.bar(x + (offset - 1) * width, plot[column], width, label=label, color=color)
    ax.set_xticks(x, plot["modality"].tolist())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Prespecified margin-satisfaction rate")
    ax.set_title("Counterfactual directional consistency on frozen validation data")
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(counter_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(counter_figure)

    overview_cache: list[tuple[pd.Series, np.ndarray, np.ndarray, float, int]] = []
    for _, selection in selected.iterrows():
        table_row = lookup.loc[str(selection["relative_path"])]
        original, maps, evidence_values, gate, prediction = infer_maps_for_row(model, table_row, device)
        panel_path = PANEL_ROOT / f"{selection['panel_id']}_{selection['modality']}.png"
        fig, axes = plt.subplots(3, 3, figsize=(12, 11))
        axes = axes.ravel()
        axes[0].imshow(original)
        axes[0].set_title("Original validation image")
        axes[1].barh(np.arange(len(lesion_columns)), evidence_values, color="#2A9D8F")
        axes[1].set_yticks(np.arange(len(lesion_columns)), [name.replace("lesion_", "").replace("_", " ") for name in lesion_columns], fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1)
        axes[1].set_title("Lesion evidence strength")
        for lesion_index, lesion in enumerate(lesion_columns):
            axis = axes[lesion_index + 2]
            resized = np.asarray(Image.fromarray((normalized_heatmap(maps[lesion_index]) * 255).astype(np.uint8)).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)) / 255.0
            axis.imshow(original)
            axis.imshow(resized, cmap="magma", alpha=0.50, vmin=0, vmax=1)
            axis.set_title(lesion.replace("lesion_", "").replace("_", " "), fontsize=9)
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(
            f"{selection['panel_id']} | {selection['selection_reason']} | target={int(selection['target'])}, prediction={prediction}, gate={gate:.3f}\n"
            "Attention-derived evidence maps; not validated segmentation masks",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(panel_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(panel_path)
        if len(overview_cache) < 6:
            overview_cache.append((selection, original, maps, gate, prediction))

    overview_path = FIGURE_ROOT / "publication_qualitative_overview.png"
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for axis, (selection, original, maps, gate, prediction) in zip(axes.ravel(), overview_cache):
        maximum = normalized_heatmap(maps.max(axis=0))
        resized = np.asarray(Image.fromarray((maximum * 255).astype(np.uint8)).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)) / 255.0
        axis.imshow(original)
        axis.imshow(resized, cmap="magma", alpha=0.48, vmin=0, vmax=1)
        axis.set_title(f"{selection['panel_id']} {selection['modality']} | y={int(selection['target'])}, ŷ={prediction}, g={gate:.2f}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Predeclared validation examples: maximum lesion-evidence response\nMaps are explanatory hypotheses, not pixel-level ground truth", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(overview_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(overview_path)
    return outputs


def output_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}


def main() -> None:
    print("=" * 120)
    print("STAGE 0G-4D-5F-6A/6B — REVIEWER VALIDATION MECHANISTIC EXPERIMENT")
    print("=" * 120)
    if not ROOT.is_dir():
        raise RuntimeError("The real Google Drive project root is not mounted")

    print("\n[1] SOURCE AND REVIEWER-PROTOCOL FREEZE")
    print("-" * 120)
    source_action, source_sha = freeze_source()
    protocol_action, protocol_sha = freeze_protocol(source_sha)
    print("Source action  :", source_action)
    print("Source SHA-256:", source_sha)
    print("Protocol action:", protocol_action)
    print("Protocol SHA  :", protocol_sha)

    if completed_result_is_valid():
        print("\nExisting mechanistic experiment and every recorded output hash are valid.")
        print("No image was reread and no model was reloaded.")
        print("STATUS: PASS — EXISTING COMPLETE")
        return

    print("\n[2] VALIDATION-ONLY CACHE RESTORATION")
    print("-" * 120)
    manifest = restore_validation_cache(source_sha, protocol_sha)

    print("\n[3] FROZEN FULL-MODEL LOADING")
    print("-" * 120)
    if sha256_file(CORE_SOURCE) != EXPECTED_CORE_SHA:
        raise RuntimeError("Frozen production source hash mismatch")
    core = import_core()
    table = build_validation_table(manifest, list(core.LESION_COLUMNS))
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    if cuda_available and torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU is required when CUDA is enabled")
    model, checkpoint = build_model(core, device)
    print("Device          :", device)
    print("GPU             :", torch.cuda.get_device_name(0) if cuda_available else "CPU")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Fundus rows     :", len(table))
    print("Optimizer       : NOT CREATED")

    print("\n[4] RESUMABLE NUMERIC MECHANISTIC INFERENCE")
    print("-" * 120)
    frame = run_resumable_numeric_inference(
        model,
        table,
        device,
        list(core.LESION_COLUMNS),
        source_sha,
        protocol_sha,
    )
    performance_identity, performance_identity_checks = validation_performance_identity(frame)
    if not all(performance_identity_checks.values()):
        failed_identity = [key for key, value in performance_identity_checks.items() if not value]
        raise RuntimeError(f"Frozen checkpoint performance identity failed: {failed_identity}")
    gate_summary, correlations, counter_summary = summarize_mechanisms(frame)
    selected = select_examples(frame)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    per_image_path = RESULT_ROOT / "validation_mechanistic_per_image_v1.csv"
    gate_path = RESULT_ROOT / "validation_gate_summary_v1.csv"
    correlation_path = RESULT_ROOT / "validation_gate_correlations_v1.csv"
    counter_path = RESULT_ROOT / "validation_counterfactual_consistency_v1.csv"
    selected_path = RESULT_ROOT / "validation_qualitative_selection_v1.csv"
    identity_path = RESULT_ROOT / "validation_checkpoint_performance_identity_v1.csv"
    atomic_csv(per_image_path, frame)
    atomic_csv(gate_path, gate_summary)
    atomic_csv(correlation_path, correlations)
    atomic_csv(counter_path, counter_summary)
    atomic_csv(selected_path, selected)
    atomic_csv(identity_path, performance_identity)

    print("\n[5] PREDECLARED QUALITATIVE PANELS")
    print("-" * 120)
    figures = render_figures(
        model, selected, table, frame, gate_summary, counter_summary,
        list(core.LESION_COLUMNS), device,
    )

    output_paths = [
        per_image_path,
        gate_path,
        correlation_path,
        counter_path,
        selected_path,
        identity_path,
        *figures,
    ]
    output_files = [output_record(path) for path in output_paths]
    overall_counter = counter_summary[counter_summary["modality"] == "ALL"].iloc[0].to_dict()
    contract_payload = {
        "contract_id": "MMRDR_FOVEA_DR_VALIDATION_MECHANISTIC_AUDIT_V1",
        "status": "COMPLETE",
        "protocol_sha256": protocol_sha,
        "source_sha256": source_sha,
        "full_model_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "selected_epoch": 10,
        "analysis_partition": "VALIDATION_ONLY",
        "fundus_rows": int(len(frame)),
        "modality_counts": {str(k): int(v) for k, v in frame["modality"].value_counts().to_dict().items()},
        "qualitative_examples": int(len(selected)),
        "gate_summary": gate_summary.to_dict("records"),
        "gate_correlations": correlations.to_dict("records"),
        "counterfactual_summary": counter_summary.to_dict("records"),
        "checkpoint_performance_identity": performance_identity.to_dict("records"),
        "interpretation_limits": [
            "Lesion maps are weakly supervised attention-derived evidence maps, not segmentation masks.",
            "Counterfactual interventions act in learned evidence space and do not establish biological causality.",
            "Example selection followed frozen validation rules and did not use locked-test outcomes.",
        ],
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "parameters_updated": False,
        "output_files": output_files,
    }
    contract_action = frozen_json(CONTRACT, contract_payload)
    contract_sha = sha256_file(CONTRACT)
    gates = {
        "protocol_frozen_before_image_access": PROTOCOL.is_file() and sha256_file(PROTOCOL) == protocol_sha,
        "source_hash_verified": FROZEN_SOURCE.is_file() and sha256_file(FROZEN_SOURCE) == source_sha,
        "split_hash_verified": sha256_file(SPLIT) == EXPECTED_SPLIT_SHA,
        "validation_manifest_hash_verified": sha256_file(VALIDATION_MANIFEST) == EXPECTED_VALIDATION_MANIFEST_SHA,
        "checkpoint_hash_verified": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA,
        "checkpoint_reports_seed2026_epoch10": int(checkpoint.get("run_seed", -1)) == 2026 and int(checkpoint.get("epoch", -1)) == 10,
        "exact_fundus_validation_rows": len(frame) == EXPECTED_FUNDUS_ROWS,
        "fundus_paths_unique": bool(frame["relative_path"].is_unique),
        "all_numeric_outputs_finite": bool(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()),
        "all_correlation_outputs_finite": bool(
            np.isfinite(correlations.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "frozen_checkpoint_performance_reproduced": all(performance_identity_checks.values()),
        "counterfactual_summary_present": len(counter_summary) == 3,
        "gate_summary_present": len(gate_summary) == 3,
        "predeclared_examples_exported": len(selected) >= 8 and all(path.is_file() for path in figures),
        "optimizer_not_created": True,
        "parameters_not_updated": True,
        "calibration_not_accessed": True,
        "locked_test_not_accessed": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Mechanistic audit gates failed: {[k for k, v in gates.items() if not v]}")
    evidence_payload = {
        "stage": "0G-4D-5F-6B",
        "purpose": "FROZEN_VALIDATION_MECHANISTIC_AUDIT",
        "status": "PASS",
        "contract": str(CONTRACT),
        "contract_sha256": contract_sha,
        "contract_action": contract_action,
        "protocol_sha256": protocol_sha,
        "source_sha256": source_sha,
        "gates": gates,
        "output_files": output_files,
        "headline_results": {
            "overall_remove_all_direction_rate": float(overall_counter["remove_all_direction_rate"]),
            "overall_remove_pdr_direction_rate": float(overall_counter["remove_pdr_direction_rate"]),
            "overall_add_pdr_direction_rate": float(overall_counter["add_pdr_direction_rate"]),
            "overall_remove_all_margin_rate": float(overall_counter["remove_all_margin_rate"]),
            "overall_remove_pdr_margin_rate": float(overall_counter["remove_pdr_margin_rate"]),
            "overall_add_pdr_margin_rate": float(overall_counter["add_pdr_margin_rate"]),
        },
        "validation_images_accessed": True,
        "train_images_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "parameters_updated": False,
    }
    frozen_json(EVIDENCE, evidence_payload)

    print("\n" + "=" * 120)
    print("STAGE 0G-4D-5F-6B STATUS: PASS")
    print("=" * 120)
    print("Validation fundus rows        :", len(frame))
    print("Qualitative panels            :", len(selected))
    print("Remove-all direction rate     :", f"{overall_counter['remove_all_direction_rate']:.4f}")
    print("Remove-PDR direction rate     :", f"{overall_counter['remove_pdr_direction_rate']:.4f}")
    print("Add-PDR direction rate        :", f"{overall_counter['add_pdr_direction_rate']:.4f}")
    print("Parameters updated            : NO")
    print("CALIBRATION accessed          : NO")
    print("Locked test accessed          : NO")
    print("Results                       :", RESULT_ROOT)
    print("NEXT ACTION: FREEZE AND PREFLIGHT THE THREE NEW COMPONENT ABLATIONS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 120)
        print("STAGE 0G-4D-5F-6A/6B STATUS: FAIL")
        print("=" * 120)
        raise
