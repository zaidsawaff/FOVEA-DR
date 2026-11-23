#!/usr/bin/env python3
"""GPU and development-cache readiness gate for post-hoc comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import torch


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
PROTOCOL_EVIDENCE = ROOT / "Evidence/stage0g4d5f0_posthoc_comparison_protocol_freeze.json"
TRAIN_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_TRAIN_local_cache_manifest_v1.csv"
VALIDATION_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_VALIDATION_local_cache_manifest_v1.csv"

TRAIN_ROOT = Path("/content/MMRDR_training_v1")
VALIDATION_ROOT = Path("/content/MMRDR_validation_v1")

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f1a_comparator_runtime_cache_readiness.py"
EVIDENCE = ROOT / "Evidence/stage0g4d5f1a_comparator_runtime_cache_readiness.json"

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

WEIGHT_CACHE = Path.home() / ".cache/torch/hub/checkpoints"
EXPECTED_WEIGHT_FILES = {
    "EfficientNetV2-S": "efficientnet_v2_s-dd5fe13b.pth",
    "ResNet-50 V2": "resnet50-11ad3fa6.pth",
    "ConvNeXt-Tiny V1": "convnext_tiny-983f1562.pth",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def frozen_json(path: Path, payload: dict) -> str:
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
            raise RuntimeError("A different readiness source is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def inspect_cache(manifest_path: Path, cache_root: Path, role: str) -> dict:
    header = pd.read_csv(manifest_path, nrows=0).columns.tolist()
    required = ["relative_path", "modality", "file_size_bytes"]
    missing = sorted(set(required) - set(header))
    if missing:
        raise RuntimeError(f"{role} manifest columns missing: {missing}")
    frame = pd.read_csv(
        manifest_path,
        usecols=required,
        dtype={"relative_path": "string", "modality": "string"},
    )
    frame["modality"] = frame["modality"].str.strip().str.upper()
    frame["file_size_bytes"] = pd.to_numeric(frame["file_size_bytes"], errors="raise").astype(int)
    counts = {key: int(value) for key, value in frame["modality"].value_counts().sort_index().to_dict().items()}
    expected = EXPECTED_COUNTS[role]
    rows_correct = len(frame) == expected["TOTAL"]
    counts_correct = counts == {key: expected[key] for key in ["CFP", "OCT", "UWF"]}

    present = 0
    size_match = 0
    for row in frame.itertuples(index=False):
        path = cache_root / str(row.relative_path)
        if path.is_file():
            present += 1
            if path.stat().st_size == int(row.file_size_bytes):
                size_match += 1
    return {
        "manifest_rows": int(len(frame)),
        "manifest_counts": counts,
        "manifest_rows_correct": bool(rows_correct),
        "manifest_counts_correct": bool(counts_correct),
        "cache_root": str(cache_root),
        "cache_root_exists": cache_root.is_dir(),
        "files_present": int(present),
        "files_size_matched": int(size_match),
        "expected_files": int(expected["TOTAL"]),
        "cache_complete": bool(present == expected["TOTAL"] and size_match == expected["TOTAL"]),
    }


def main() -> None:
    print("=" * 80)
    print("STAGE 0G-4D-5F-1A — COMPARATOR GPU AND DEVELOPMENT-CACHE READINESS")
    print("=" * 80)

    print("\n[1] FROZEN COMPARISON PROTOCOL")
    print("-" * 80)
    gates = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(verified)
        print(f"{path.name:76s}: exists={exists}, hash_verified={verified}")
    if not all(gates.values()):
        raise RuntimeError("A comparator prerequisite is missing or changed.")

    with PROTOCOL.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    with PROTOCOL_EVIDENCE.open("r", encoding="utf-8") as handle:
        protocol_evidence = json.load(handle)
    protocol_pass = protocol.get("status") == "FROZEN_BEFORE_COMPARATOR_TRAINING"
    evidence_pass = protocol_evidence.get("status") == "PASS"
    test_prohibited = protocol["data_access_policy"]["LOCKED_INTERNAL_TEST"].startswith("PROHIBITED")
    gates.update({
        "comparison_protocol_frozen": protocol_pass,
        "comparison_protocol_evidence_passed": evidence_pass,
        "locked_test_prohibited": test_prohibited,
    })
    print("Protocol frozen       :", protocol_pass)
    print("Protocol evidence PASS:", evidence_pass)
    print("Locked test prohibited:", test_prohibited)

    print("\n[2] GPU RUNTIME")
    print("-" * 80)
    cuda_available = torch.cuda.is_available()
    visible_devices = torch.cuda.device_count() if cuda_available else 0
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        capability = torch.cuda.get_device_capability(0)
    else:
        gpu_name = None
        total_vram_gb = 0.0
        capability = None
    gpu_ready = bool(cuda_available and visible_devices == 1 and total_vram_gb >= 10.0)
    print("CUDA available :", cuda_available)
    print("Visible devices:", visible_devices)
    print("GPU            :", gpu_name)
    print("VRAM GB        :", f"{total_vram_gb:.3f}")
    print("Capability     :", capability)
    print("GPU ready      :", gpu_ready)

    print("\n[3] DEVELOPMENT-ONLY LOCAL CACHES")
    print("-" * 80)
    train = inspect_cache(TRAIN_MANIFEST, TRAIN_ROOT, "TRAIN")
    validation = inspect_cache(VALIDATION_MANIFEST, VALIDATION_ROOT, "VALIDATION")
    for role, result in [("TRAIN", train), ("VALIDATION", validation)]:
        print(f"{role} root          : {result['cache_root']}")
        print(f"{role} manifest rows : {result['manifest_rows']}")
        print(f"{role} modality counts: {result['manifest_counts']}")
        print(f"{role} files present : {result['files_present']} / {result['expected_files']}")
        print(f"{role} size matched  : {result['files_size_matched']} / {result['expected_files']}")
        print(f"{role} cache complete: {result['cache_complete']}")

    print("\n[4] OFFICIAL-WEIGHT CACHE STATUS")
    print("-" * 80)
    weight_status = {}
    for model, filename in EXPECTED_WEIGHT_FILES.items():
        path = WEIGHT_CACHE / filename
        exists = path.is_file() and path.stat().st_size > 0
        weight_status[model] = {"path": str(path), "cached": bool(exists)}
        print(f"{model:24s}: cached={exists} | {path}")
    print("Missing official weights may be downloaded only during a later verified acquisition stage.")

    gates.update({
        "CUDA_available": bool(cuda_available),
        "exactly_one_GPU_visible": visible_devices == 1,
        "GPU_memory_at_least_10GB": total_vram_gb >= 10.0,
        "TRAIN_manifest_structure_exact": train["manifest_rows_correct"] and train["manifest_counts_correct"],
        "VALIDATION_manifest_structure_exact": validation["manifest_rows_correct"] and validation["manifest_counts_correct"],
        "TRAIN_cache_complete": train["cache_complete"],
        "VALIDATION_cache_complete": validation["cache_complete"],
        "locked_test_not_accessed": True,
        "images_not_decoded": True,
        "model_not_instantiated": True,
        "training_not_started": True,
    })

    print("\n" + "=" * 80)
    print("COMPARATOR PREFLIGHT READINESS GATES")
    print("=" * 80)
    for name, passed in gates.items():
        print(f"{name:76s}: {passed}")

    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        print("\nTraining preflight ready: False")
        if not gpu_ready:
            print("NEXT ACTION: Select a T4 GPU runtime.")
        if not train["cache_complete"] or not validation["cache_complete"]:
            print("NEXT ACTION: Restore TRAIN and VALIDATION caches from the frozen manifests.")
        print("Locked test remains untouched.")
        raise RuntimeError(f"Comparator readiness incomplete: {failed}")

    print("\n[5] READINESS EVIDENCE FREEZE")
    print("-" * 80)
    source_action, source_sha = freeze_source()
    gates["readiness_source_frozen"] = sha256_file(FROZEN_SOURCE) == source_sha
    evidence = {
        "stage": "0G-4D-5F-1A",
        "status": "PASS",
        "purpose": "COMPARATOR_GPU_AND_DEVELOPMENT_CACHE_READINESS",
        "protocol": str(PROTOCOL),
        "protocol_sha256": EXPECTED_SHA[PROTOCOL],
        "GPU": {
            "name": gpu_name,
            "visible_devices": visible_devices,
            "VRAM_GB": total_vram_gb,
            "compute_capability": capability,
        },
        "TRAIN_cache": train,
        "VALIDATION_cache": validation,
        "official_weight_cache": weight_status,
        "source": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "gates": gates,
        "locked_test_accessed": False,
        "images_decoded": False,
        "model_instantiated": False,
        "training_started": False,
        "next_required_stage": "COMPARATOR_IMPLEMENTATION_AND_64_STEP_TRAIN_VALIDATION_PREFLIGHT",
    }
    evidence_action = frozen_json(EVIDENCE, evidence)
    print("Source action   :", source_action)
    print("Source SHA-256  :", source_sha)
    print("Evidence action :", evidence_action)
    print("Evidence        :", EVIDENCE)
    print("Evidence SHA-256:", sha256_file(EVIDENCE))
    print("\nTraining preflight ready: True")
    print("Locked test accessed     : NO")
    print("Images decoded           : NO")
    print("Model instantiated       : NO")
    print("Training started         : NO")
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-5F-1A STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
