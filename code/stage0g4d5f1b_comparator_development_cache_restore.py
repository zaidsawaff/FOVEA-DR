#!/usr/bin/env python3
"""Restore and verify TRAIN/VALIDATION caches for post-hoc comparisons.

This stage reads only the frozen development-cache manifests. It never reads
CALIBRATION or LOCKED_INTERNAL_TEST membership, images, labels, or predictions.
Raw multi-volume archives are exposed through temporary symlinks and are never
modified.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd
from PIL import Image, ImageFile
import torch


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
RAW_DIR = ROOT / "Data/Raw/MMRDR_Figshare_v2_Archives"

PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
PROTOCOL_EVIDENCE = ROOT / "Evidence/stage0g4d5f0_posthoc_comparison_protocol_freeze.json"
TRAIN_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_TRAIN_local_cache_manifest_v1.csv"
VALIDATION_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_VALIDATION_local_cache_manifest_v1.csv"

TRAIN_CACHE = Path("/content/MMRDR_training_v1")
VALIDATION_CACHE = Path("/content/MMRDR_validation_v1")
VOLUME_VIEW = Path("/content/MMRDR_comparator_volume_view")

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f1b_comparator_development_cache_restore.py"
EVIDENCE = ROOT / "Evidence/stage0g4d5f1b_comparator_development_cache_restore.json"

EXPECTED_SHA = {
    PROTOCOL: "f946964923e08722dbb486aa3782b08880446040e07fb6b268ea9c28c08e6ae1",
    PROTOCOL_EVIDENCE: "8e55afece9ba66813f057a8547b46c2da6b80c520120c78b6dc0cd63c336ec94",
    TRAIN_MANIFEST: "32f9a3b6b17916b78f591007a85ce2e1910c9c4d2b1294af1c391b19f476bd15",
    VALIDATION_MANIFEST: "d94c5613b6d030dd6e99246b1b72f77746b5223139a4597bef733eb38a5786af",
}

EXPECTED_COUNTS = {
    "TRAIN": {"TOTAL": 13251, "CFP": 6128, "OCT": 1663, "UWF": 5460},
    "VALIDATION": {"TOTAL": 2839, "CFP": 1314, "OCT": 357, "UWF": 1168},
}

EXPECTED_VOLUME_SIZES = {
    **{f"MMRDR.zip.{index:03d}": 2_097_152_000 for index in range(1, 9)},
    "MMRDR.zip.009": 1_833_101_731,
}

HASH_COLUMNS = ("image_sha256", "sha256", "file_sha256")
SIZE_COLUMNS = ("file_size_bytes", "archive_size_bytes", "size_bytes")
DISK_HEADROOM_BYTES = 5 * 1024**3

ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    data = json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return "CREATED_AND_FROZEN"


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different 5F-1B source is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def first_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise RuntimeError(f"{role} manifest lacks one of: {list(candidates)}")


def normalized_path(value: Any) -> str:
    raw = str(value).strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise RuntimeError(f"Unsafe relative path: {raw!r}")
    return str(path)


def contains_locked_test_token(value: str) -> bool:
    forbidden = {"ts", "test", "testing", "locked_test", "locked_internal_test"}
    return any(part.lower() in forbidden for part in PurePosixPath(value).parts)


def load_manifest(path: Path, role: str) -> tuple[pd.DataFrame, str, str, dict[str, int]]:
    frame = pd.read_csv(path)
    required = {"relative_path", "modality"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{role} manifest missing columns: {missing}")

    frame = frame.copy()
    frame["relative_path"] = frame["relative_path"].map(normalized_path)
    frame["modality"] = frame["modality"].astype(str).str.strip().str.upper()
    hash_column = first_column(frame, HASH_COLUMNS, role)
    size_column = first_column(frame, SIZE_COLUMNS, role)

    hashes = frame[hash_column].astype(str).str.lower()
    sizes = pd.to_numeric(frame[size_column], errors="coerce")
    if not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise RuntimeError(f"Invalid SHA-256 value in {role} manifest")
    if sizes.isna().any() or (sizes <= 0).any():
        raise RuntimeError(f"Invalid file size in {role} manifest")
    frame[hash_column] = hashes
    frame[size_column] = sizes.astype("int64")

    expected = EXPECTED_COUNTS[role]
    counts = {
        key: int(value)
        for key, value in frame["modality"].value_counts().sort_index().to_dict().items()
    }
    expected_modalities = {key: expected[key] for key in ["CFP", "OCT", "UWF"]}
    if len(frame) != expected["TOTAL"] or counts != expected_modalities:
        raise RuntimeError(
            f"{role} count mismatch: rows={len(frame)}, modalities={counts}"
        )
    if not frame["relative_path"].is_unique:
        raise RuntimeError(f"Duplicate relative paths in {role} manifest")
    if frame["relative_path"].map(contains_locked_test_token).any():
        raise RuntimeError(f"Locked-test token found in {role} manifest")
    if "development_split" in frame.columns:
        roles = set(frame["development_split"].astype(str).str.strip().str.upper())
        if roles != {role}:
            raise RuntimeError(f"Unexpected roles in {role} manifest: {roles}")

    return frame, hash_column, size_column, counts


def prepare_volume_view() -> tuple[Path, list[dict[str, Any]]]:
    if shutil.which("7z") is None:
        raise RuntimeError("7z is not installed in this runtime")
    VOLUME_VIEW.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for volume_name, expected_size in EXPECTED_VOLUME_SIZES.items():
        final_path = RAW_DIR / volume_name
        part_path = RAW_DIR / f"{volume_name}.part"
        if final_path.is_file() and final_path.stat().st_size == expected_size:
            source = final_path
            source_type = "FINAL"
        elif part_path.is_file() and part_path.stat().st_size == expected_size:
            source = part_path
            source_type = "COMPLETE_PART_ALIAS"
        else:
            raise RuntimeError(f"Complete archive volume unavailable: {volume_name}")

        alias = VOLUME_VIEW / volume_name
        if alias.is_symlink():
            alias.unlink()
        elif alias.exists():
            raise RuntimeError(f"Refusing to replace non-symlink: {alias}")
        os.symlink(str(source), str(alias))
        records.append({
            "volume": volume_name,
            "source": source.name,
            "source_type": source_type,
            "size_bytes": int(source.stat().st_size),
        })
        print(f"{volume_name} <- {source.name} [{source_type}]")

    archive = VOLUME_VIEW / "MMRDR.zip.001"
    listing = subprocess.run(
        ["7z", "l", str(archive)], capture_output=True, text=True, check=False
    )
    volumes_seen = listing.returncode == 0 and "Volumes = 9" in listing.stdout
    print("7-Zip listing return code:", listing.returncode)
    print("All nine volumes detected:", volumes_seen)
    if not volumes_seen:
        print(listing.stdout[-2500:])
        print(listing.stderr[-2500:])
        raise RuntimeError("Nine-volume archive verification failed")
    return archive, records


def files_needing_extraction(
    frame: pd.DataFrame, root: Path, size_column: str
) -> list[str]:
    result: list[str] = []
    for row in frame.to_dict("records"):
        relative_path = str(row["relative_path"])
        local_path = root / relative_path
        if not local_path.is_file() or local_path.stat().st_size != int(row[size_column]):
            result.append(relative_path)
    return result


def extraction_bytes(
    frame: pd.DataFrame, size_column: str, members: list[str]
) -> int:
    size_map = dict(zip(frame["relative_path"], frame[size_column]))
    return sum(int(size_map[member]) for member in members)


def extract_members(
    archive: Path, members: list[str], output_root: Path, list_path: Path, role: str
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if not members:
        print(f"{role}: no extraction required; sizes already match.")
        return
    list_path.write_text("\n".join(members) + "\n", encoding="utf-8")
    print(f"{role}: extracting {len(members)} members...")
    result = subprocess.run(
        [
            "7z", "x", "-y", "-aoa", str(archive),
            f"-o{output_root}", f"@{list_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"{role} extraction return code:", result.returncode)
    print(result.stdout[-2200:])
    if result.returncode != 0:
        print(result.stderr[-2200:])
        raise RuntimeError(f"{role} selective extraction failed")


def verify_cache(
    frame: pd.DataFrame,
    root: Path,
    hash_column: str,
    size_column: str,
    role: str,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    verified_bytes = 0
    total = len(frame)
    for index, row in enumerate(frame.to_dict("records"), start=1):
        relative_path = str(row["relative_path"])
        local_path = root / relative_path
        try:
            if not local_path.is_file():
                raise FileNotFoundError(local_path)
            expected_size = int(row[size_column])
            actual_size = local_path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(f"size expected={expected_size}, actual={actual_size}")
            actual_hash = sha256_file(local_path)
            if actual_hash != str(row[hash_column]).lower():
                raise RuntimeError("SHA-256 mismatch")
            with Image.open(local_path) as image:
                image.verify()
            verified_bytes += actual_size
        except Exception as error:
            failures.append({"relative_path": relative_path, "error": repr(error)})
        if index % 1000 == 0 or index == total:
            print(f"{role:10s} verified {index:5d}/{total} | failures={len(failures)}")
    return {
        "role": role,
        "expected_rows": int(total),
        "verified_rows": int(total - len(failures)),
        "verified_bytes": int(verified_bytes),
        "failures": failures,
    }


def main() -> None:
    print("=" * 80)
    print("STAGE 0G-4D-5F-1B — COMPARATOR DEVELOPMENT-CACHE RESTORATION")
    print("=" * 80)

    print("\n[1] GPU AND FROZEN COMPARISON PROTOCOL")
    print("-" * 80)
    gates: dict[str, bool] = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(verified)
        print(f"{path.name:76s}: exists={exists}, hash_verified={verified}")
    if not all(gates.values()):
        raise RuntimeError("A frozen comparison prerequisite is missing or changed.")

    with PROTOCOL.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    with PROTOCOL_EVIDENCE.open("r", encoding="utf-8") as handle:
        protocol_evidence = json.load(handle)
    protocol_frozen = protocol.get("status") == "FROZEN_BEFORE_COMPARATOR_TRAINING"
    evidence_pass = protocol_evidence.get("status") == "PASS"
    test_prohibited = protocol["data_access_policy"]["LOCKED_INTERNAL_TEST"].startswith("PROHIBITED")

    cuda_available = torch.cuda.is_available()
    visible_devices = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if visible_devices == 1 else None
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if visible_devices == 1 else 0.0
    )
    capability = torch.cuda.get_device_capability(0) if visible_devices == 1 else None
    gates.update({
        "comparison_protocol_frozen": bool(protocol_frozen),
        "comparison_protocol_evidence_passed": bool(evidence_pass),
        "locked_test_prohibited": bool(test_prohibited),
        "CUDA_available": bool(cuda_available),
        "exactly_one_GPU_visible": visible_devices == 1,
        "GPU_memory_at_least_10GB": vram_gb >= 10.0,
    })
    print("Protocol frozen       :", protocol_frozen)
    print("Protocol evidence PASS:", evidence_pass)
    print("Locked test prohibited:", test_prohibited)
    print("CUDA available        :", cuda_available)
    print("GPU                   :", gpu_name)
    print("VRAM GB               :", f"{vram_gb:.3f}")
    if not all(gates.values()):
        raise RuntimeError("GPU/protocol readiness failed.")

    print("\n[2] FROZEN DEVELOPMENT MANIFESTS")
    print("-" * 80)
    train, train_hash_col, train_size_col, train_counts = load_manifest(TRAIN_MANIFEST, "TRAIN")
    validation, val_hash_col, val_size_col, validation_counts = load_manifest(
        VALIDATION_MANIFEST, "VALIDATION"
    )
    paths_disjoint = set(train["relative_path"]).isdisjoint(validation["relative_path"])
    print("TRAIN rows/counts      :", len(train), train_counts)
    print("VALIDATION rows/counts :", len(validation), validation_counts)
    print("Path overlap           :", not paths_disjoint)
    if not paths_disjoint:
        raise RuntimeError("TRAIN/VALIDATION path overlap detected.")

    print("\n[3] NINE-VOLUME READ-ONLY VIEW")
    print("-" * 80)
    archive, volume_records = prepare_volume_view()

    print("\n[4] SELECTIVE EXTRACTION CAPACITY")
    print("-" * 80)
    train_missing = files_needing_extraction(train, TRAIN_CACHE, train_size_col)
    validation_missing = files_needing_extraction(validation, VALIDATION_CACHE, val_size_col)
    required_bytes = extraction_bytes(train, train_size_col, train_missing)
    required_bytes += extraction_bytes(validation, val_size_col, validation_missing)
    free_bytes = shutil.disk_usage("/content").free
    capacity_ok = free_bytes >= required_bytes + DISK_HEADROOM_BYTES
    print("TRAIN needing extraction     :", len(train_missing))
    print("VALIDATION needing extraction:", len(validation_missing))
    print("Extraction size GiB          :", f"{required_bytes / 1024**3:.3f}")
    print("Free disk GiB                :", f"{free_bytes / 1024**3:.3f}")
    print("Five-GiB headroom retained   :", capacity_ok)
    if not capacity_ok:
        raise RuntimeError("Insufficient /content disk capacity.")

    print("\n[5] SELECTIVE DEVELOPMENT EXTRACTION")
    print("-" * 80)
    extract_members(
        archive, train_missing, TRAIN_CACHE,
        Path("/content/fovea_dr_comparator_TRAIN_restore_list.txt"), "TRAIN",
    )
    extract_members(
        archive, validation_missing, VALIDATION_CACHE,
        Path("/content/fovea_dr_comparator_VALIDATION_restore_list.txt"), "VALIDATION",
    )

    print("\n[6] FULL SHA-256 AND IMAGE-DECODE VERIFICATION")
    print("-" * 80)
    train_result = verify_cache(train, TRAIN_CACHE, train_hash_col, train_size_col, "TRAIN")
    validation_result = verify_cache(
        validation, VALIDATION_CACHE, val_hash_col, val_size_col, "VALIDATION"
    )

    # One automatic repair pass is permitted only for failed development members.
    train_failures = [item["relative_path"] for item in train_result["failures"]]
    validation_failures = [item["relative_path"] for item in validation_result["failures"]]
    if train_failures or validation_failures:
        print("\n[7] AUTOMATIC DEVELOPMENT-ONLY REPAIR PASS")
        print("-" * 80)
        extract_members(
            archive, train_failures, TRAIN_CACHE,
            Path("/content/fovea_dr_comparator_TRAIN_repair_list.txt"), "TRAIN repair",
        )
        extract_members(
            archive, validation_failures, VALIDATION_CACHE,
            Path("/content/fovea_dr_comparator_VALIDATION_repair_list.txt"), "VALIDATION repair",
        )
        train_result = verify_cache(train, TRAIN_CACHE, train_hash_col, train_size_col, "TRAIN")
        validation_result = verify_cache(
            validation, VALIDATION_CACHE, val_hash_col, val_size_col, "VALIDATION"
        )
    else:
        print("\n[7] AUTOMATIC DEVELOPMENT-ONLY REPAIR PASS")
        print("-" * 80)
        print("No repair was required.")

    gates.update({
        "exactly_13251_TRAIN_rows": len(train) == 13251,
        "TRAIN_modality_counts_exact": train_counts == {"CFP": 6128, "OCT": 1663, "UWF": 5460},
        "exactly_2839_VALIDATION_rows": len(validation) == 2839,
        "VALIDATION_modality_counts_exact": validation_counts == {"CFP": 1314, "OCT": 357, "UWF": 1168},
        "TRAIN_VALIDATION_paths_disjoint": bool(paths_disjoint),
        "no_locked_test_path_tokens": not (
            train["relative_path"].map(contains_locked_test_token).any()
            or validation["relative_path"].map(contains_locked_test_token).any()
        ),
        "all_nine_archive_volumes_verified": len(volume_records) == 9,
        "disk_capacity_with_5GiB_headroom": bool(capacity_ok),
        "all_TRAIN_images_hash_and_decode_verified": (
            train_result["verified_rows"] == 13251 and not train_result["failures"]
        ),
        "all_VALIDATION_images_hash_and_decode_verified": (
            validation_result["verified_rows"] == 2839 and not validation_result["failures"]
        ),
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
        "model_not_instantiated": True,
        "optimizer_not_created": True,
        "training_not_started": True,
        "raw_archives_not_modified": True,
    })

    print("\n" + "=" * 80)
    print("COMPARATOR DEVELOPMENT-CACHE RESTORATION GATES")
    print("=" * 80)
    for name, passed in gates.items():
        print(f"{name:78s}: {passed}")
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Development-cache restoration failed: {failed}")

    print("\n[8] SOURCE AND EVIDENCE FREEZE")
    print("-" * 80)
    source_action, source_sha = freeze_source()
    gates["restoration_source_frozen_and_verified"] = sha256_file(FROZEN_SOURCE) == source_sha
    evidence = {
        "stage": "0G-4D-5F-1B",
        "status": "PASS",
        "purpose": "POSTHOC_COMPARATOR_DEVELOPMENT_CACHE_RESTORATION",
        "comparison_protocol": str(PROTOCOL),
        "comparison_protocol_sha256": EXPECTED_SHA[PROTOCOL],
        "TRAIN_manifest": str(TRAIN_MANIFEST),
        "TRAIN_manifest_sha256": EXPECTED_SHA[TRAIN_MANIFEST],
        "VALIDATION_manifest": str(VALIDATION_MANIFEST),
        "VALIDATION_manifest_sha256": EXPECTED_SHA[VALIDATION_MANIFEST],
        "GPU": {
            "name": gpu_name,
            "visible_devices": int(visible_devices),
            "VRAM_GB": float(vram_gb),
            "compute_capability": capability,
        },
        "TRAIN_cache": train_result,
        "VALIDATION_cache": validation_result,
        "archive_aliases": volume_records,
        "source": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "gates": {name: bool(value) for name, value in gates.items()},
        "development_images_accessed": True,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "model_instantiated": False,
        "optimizer_created": False,
        "training_started": False,
        "raw_archives_modified": False,
        "next_required_stage": "RE-RUN_0G-4D-5F-1A_READINESS_GATE",
    }
    evidence_action = frozen_json(EVIDENCE, evidence)
    print("Source action   :", source_action)
    print("Source SHA-256  :", source_sha)
    print("Evidence action :", evidence_action)
    print("Evidence        :", EVIDENCE)
    print("Evidence SHA-256:", sha256_file(EVIDENCE))

    print("\nDevelopment images accessed: YES — CACHE VERIFICATION ONLY")
    print("CALIBRATION accessed        : NO")
    print("Locked test accessed        : NO")
    print("Model instantiated          : NO")
    print("Optimizer created           : NO")
    print("Training started            : NO")
    print("Raw archives modified       : NO")
    print("\nNEXT REQUIRED STAGE:")
    print("Re-run the frozen 5F-1A readiness gate.")
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-5F-1B STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 80, file=sys.stderr)
        print("STAGE 0G-4D-5F-1B STATUS: FAIL", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        raise
