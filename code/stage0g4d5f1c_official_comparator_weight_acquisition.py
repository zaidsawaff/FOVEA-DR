#!/usr/bin/env python3
"""Acquire and verify official ImageNet weights for frozen comparators."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
from urllib.parse import urlparse

import torch
from torch import nn
import torchvision
from torchvision import models


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
PROTOCOL_EVIDENCE = ROOT / "Evidence/stage0g4d5f0_posthoc_comparison_protocol_freeze.json"
READINESS_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f1a_comparator_runtime_cache_readiness.py"
READINESS_EVIDENCE = ROOT / "Evidence/stage0g4d5f1a_comparator_runtime_cache_readiness.json"
CACHE_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f1b_comparator_development_cache_restore.py"
CACHE_EVIDENCE = ROOT / "Evidence/stage0g4d5f1b_comparator_development_cache_restore.json"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f1c_official_comparator_weight_acquisition.py"
EVIDENCE = ROOT / "Evidence/stage0g4d5f1c_official_comparator_weight_acquisition.json"

EXPECTED_SHA = {
    PROTOCOL: "f946964923e08722dbb486aa3782b08880446040e07fb6b268ea9c28c08e6ae1",
    PROTOCOL_EVIDENCE: "8e55afece9ba66813f057a8547b46c2da6b80c520120c78b6dc0cd63c336ec94",
    READINESS_SOURCE: "0187d2916844b76eb238e7f2e3e25896118d4115b981980f5c89a88ae833bc1c",
    CACHE_SOURCE: "cdd350270818e9413eed027262d89771a61a97a2f37902fbec82aae4a8f075b5",
}

OFFICIAL_CONTRACTS = [
    {
        "name": "EfficientNetV2-S",
        "protocol_id": "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
        "constructor": models.efficientnet_v2_s,
        "weight": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "enum": "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
        "url": "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
        "filename": "efficientnet_v2_s-dd5fe13b.pth",
        "hash_prefix": "dd5fe13b",
    },
    {
        "name": "ResNet-50",
        "protocol_id": "B2_RESNET50_MULTITASK",
        "constructor": models.resnet50,
        "weight": models.ResNet50_Weights.IMAGENET1K_V2,
        "enum": "ResNet50_Weights.IMAGENET1K_V2",
        "url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "filename": "resnet50-11ad3fa6.pth",
        "hash_prefix": "11ad3fa6",
    },
    {
        "name": "ConvNeXt-Tiny",
        "protocol_id": "B3_CONVNEXT_TINY_MULTITASK",
        "constructor": models.convnext_tiny,
        "weight": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "url": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        "filename": "convnext_tiny-983f1562.pth",
        "hash_prefix": "983f1562",
    },
]


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
            raise RuntimeError("A different 5F-1C source is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def load_pass_evidence(path: Path, label: str) -> tuple[dict[str, Any], bool]:
    if not path.is_file():
        print(f"{label:28s}: exists=False")
        return {}, False
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    gates = payload.get("gates", {})
    passed = (
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and bool(gates)
        and all(bool(value) for value in gates.values())
    )
    print(f"{label:28s}: exists=True, PASS={passed}")
    return payload, bool(passed)


def parameters_all_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def inspect_official_model(contract: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    name = str(contract["name"])
    weight = contract["weight"]
    constructor = contract["constructor"]
    declared_url = str(weight.url)
    derived_filename = Path(urlparse(declared_url).path).name
    cache_path = cache_dir / str(contract["filename"])
    cached_before = cache_path.is_file()

    print(f"\n{name}")
    print("-" * 80)
    print("Protocol comparator :", contract["protocol_id"])
    print("Weight enum         :", contract["enum"])
    print("Official URL        :", declared_url)
    print("Checkpoint filename :", derived_filename)
    print("Cached before stage :", cached_before)
    print("Loading official model; download occurs if needed...")

    model = constructor(weights=weight, progress=True)
    model.eval()

    if not cache_path.is_file() or cache_path.stat().st_size <= 0:
        raise RuntimeError(f"Official checkpoint unavailable after load: {cache_path}")
    checkpoint_sha = sha256_file(cache_path)
    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not conv_layers or not linear_layers:
        raise RuntimeError(f"Could not inspect convolution/classifier for {name}")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    first_conv_channels = int(conv_layers[0].in_channels)
    classifier_input_features = int(linear_layers[-1].in_features)
    classifier_outputs = int(linear_layers[-1].out_features)
    all_finite = parameters_all_finite(model)

    checks = {
        "enum_url_exact": declared_url == contract["url"],
        "filename_exact": derived_filename == contract["filename"],
        "checkpoint_exists": cache_path.is_file(),
        "checkpoint_nonempty": cache_path.stat().st_size > 0,
        "checkpoint_hash_prefix_verified": checkpoint_sha.startswith(contract["hash_prefix"]),
        "official_model_loaded": True,
        "first_conv_accepts_RGB": first_conv_channels == 3,
        "ImageNet_classifier_has_1000_outputs": classifier_outputs == 1000,
        "all_parameters_finite": all_finite,
    }

    print("Checkpoint path      :", cache_path)
    print("Checkpoint size bytes:", cache_path.stat().st_size)
    print("Checkpoint SHA-256   :", checkpoint_sha)
    print("Expected hash prefix :", contract["hash_prefix"])
    print("Hash prefix verified :", checks["checkpoint_hash_prefix_verified"])
    print("Total parameters     :", total_parameters)
    print("Trainable parameters :", trainable_parameters)
    print("First-conv channels  :", first_conv_channels)
    print("Classifier input     :", classifier_input_features)
    print("Classifier outputs   :", classifier_outputs)
    print("All parameters finite:", all_finite)

    result = {
        "name": name,
        "protocol_comparator_id": contract["protocol_id"],
        "weight_enum": contract["enum"],
        "official_url": declared_url,
        "checkpoint_filename": derived_filename,
        "checkpoint_path": str(cache_path),
        "checkpoint_size_bytes": int(cache_path.stat().st_size),
        "checkpoint_sha256": checkpoint_sha,
        "expected_hash_prefix": contract["hash_prefix"],
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "first_conv_input_channels": first_conv_channels,
        "classifier_input_features": classifier_input_features,
        "classifier_outputs": classifier_outputs,
        "checks": {key: bool(value) for key, value in checks.items()},
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    print("=" * 80)
    print("STAGE 0G-4D-5F-1C — OFFICIAL COMPARATOR WEIGHT ACQUISITION")
    print("=" * 80)

    print("\n[1] FROZEN COMPARISON READINESS")
    print("-" * 80)
    gates: dict[str, bool] = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(verified)
        print(f"{path.name:76s}: exists={exists}, hash_verified={verified}")

    protocol_payload, protocol_pass = load_pass_evidence(
        PROTOCOL_EVIDENCE, "comparison protocol evidence"
    )
    readiness_payload, readiness_pass = load_pass_evidence(
        READINESS_EVIDENCE, "runtime/cache readiness"
    )
    cache_payload, cache_pass = load_pass_evidence(
        CACHE_EVIDENCE, "cache restoration evidence"
    )
    gates.update({
        "comparison_protocol_evidence_passed": protocol_pass,
        "runtime_and_cache_readiness_passed": readiness_pass,
        "development_cache_restoration_passed": cache_pass,
        "readiness_source_linked": readiness_payload.get("source_sha256") == EXPECTED_SHA[READINESS_SOURCE],
        "cache_source_linked": cache_payload.get("source_sha256") == EXPECTED_SHA[CACHE_SOURCE],
        "protocol_reports_locked_test_not_reused": (
            protocol_payload.get("locked_test_reused") is False
            and protocol_payload.get("locked_test_artifact_rows_read") is False
        ),
        "readiness_reports_locked_test_not_accessed": (
            readiness_payload.get("locked_test_accessed") is False
        ),
        "cache_restore_reports_locked_test_not_accessed": (
            cache_payload.get("locked_test_accessed") is False
        ),
    })

    cuda_available = torch.cuda.is_available()
    visible_devices = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if visible_devices == 1 else None
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if visible_devices == 1 else 0.0
    )
    gates.update({
        "CUDA_available": bool(cuda_available),
        "exactly_one_GPU_visible": visible_devices == 1,
        "GPU_memory_at_least_10GB": vram_gb >= 10.0,
    })
    print("CUDA available :", cuda_available)
    print("Visible devices:", visible_devices)
    print("GPU            :", gpu_name)
    print("VRAM GB        :", f"{vram_gb:.3f}")

    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Frozen comparator readiness failed: {failed}")

    print("\n[2] SOFTWARE ENVIRONMENT")
    print("-" * 80)
    print("Python      :", sys.version.split()[0])
    print("PyTorch     :", torch.__version__)
    print("torchvision :", torchvision.__version__)
    print("CUDA build  :", torch.version.cuda)

    print("\n[3] DOWNLOAD / LOAD / VERIFY OFFICIAL WEIGHTS")
    print("-" * 80)
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = [inspect_official_model(contract, cache_dir) for contract in OFFICIAL_CONTRACTS]

    all_model_checks = all(
        all(result["checks"].values())
        for result in results
    )
    three_distinct_checkpoints = len(
        {result["checkpoint_sha256"] for result in results}
    ) == 3
    gates.update({
        "exactly_three_official_models_verified": len(results) == 3,
        "all_official_model_checks_passed": bool(all_model_checks),
        "three_distinct_official_checkpoints": bool(three_distinct_checkpoints),
        "development_images_not_accessed": True,
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
        "optimizer_not_created": True,
        "model_parameters_not_updated": True,
        "training_not_started": True,
    })

    print("\n" + "=" * 80)
    print("OFFICIAL COMPARATOR WEIGHT GATES")
    print("=" * 80)
    for name, passed in gates.items():
        print(f"{name:78s}: {passed}")
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Official comparator weight gate failed: {failed}")

    print("\n[4] SOURCE AND EVIDENCE FREEZE")
    print("-" * 80)
    source_action, source_sha = freeze_source()
    gates["weight_acquisition_source_frozen_and_verified"] = (
        sha256_file(FROZEN_SOURCE) == source_sha
    )
    evidence = {
        "stage": "0G-4D-5F-1C",
        "status": "PASS",
        "purpose": "OFFICIAL_COMPARATOR_WEIGHT_ACQUISITION",
        "comparison_protocol": str(PROTOCOL),
        "comparison_protocol_sha256": EXPECTED_SHA[PROTOCOL],
        "readiness_evidence": str(READINESS_EVIDENCE),
        "readiness_evidence_sha256": sha256_file(READINESS_EVIDENCE),
        "cache_evidence": str(CACHE_EVIDENCE),
        "cache_evidence_sha256": sha256_file(CACHE_EVIDENCE),
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "CUDA_build": torch.version.cuda,
            "GPU": gpu_name,
            "visible_devices": int(visible_devices),
            "VRAM_GB": float(vram_gb),
        },
        "official_models": results,
        "source": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "gates": {name: bool(value) for name, value in gates.items()},
        "medical_images_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "training_started": False,
        "next_required_stage": "COMPARATOR_IMPLEMENTATION_AND_64_STEP_PREFLIGHT",
    }
    evidence_action = frozen_json(EVIDENCE, evidence)
    print("Source action   :", source_action)
    print("Source SHA-256  :", source_sha)
    print("Evidence action :", evidence_action)
    print("Evidence        :", EVIDENCE)
    print("Evidence SHA-256:", sha256_file(EVIDENCE))

    print("\nMedical images accessed : NO")
    print("CALIBRATION accessed     : NO")
    print("Locked test accessed     : NO")
    print("Optimizer created        : NO")
    print("Parameters updated       : NO")
    print("Training started         : NO")
    print("\nNEXT REQUIRED STAGE:")
    print("Comparator implementation and 64-step TRAIN/VALIDATION preflight.")
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-5F-1C STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 80, file=sys.stderr)
        print("STAGE 0G-4D-5F-1C STATUS: FAIL", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        raise
