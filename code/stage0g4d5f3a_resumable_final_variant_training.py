#!/usr/bin/env python3
"""Protocol-locked resumable final training for the six post-hoc variants.

The runner trains exactly one frozen comparator/ablation at a time on TRAIN,
uses VALIDATION only for early stopping and epoch selection, and never reads
CALIBRATION or LOCKED_INTERNAL_TEST.  Every fresh run initializes from the
official torchvision ImageNet checkpoint; disposable 64-step weights are not
loadable by this source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
TRAIN_CACHE_ROOT = Path("/content/MMRDR_training_v1")
VALIDATION_CACHE_ROOT = Path("/content/MMRDR_validation_v1")

PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
CANDIDATE = ROOT / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
TRAINING_PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_training_protocol_v1.json"
SPLIT = ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
SAMPLER = ROOT / "Splits/mmrdr_fovea_dr_TRAIN_sampler_weights_v1.csv"
LESION_WEIGHTS = ROOT / "Metadata/mmrdr_fovea_dr_lesion_pos_weights_v1.csv"
TRAIN_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_TRAIN_local_cache_manifest_v1.csv"
VALIDATION_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_VALIDATION_local_cache_manifest_v1.csv"

IMPLEMENTATION_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2a_comparator_implementation_audit.py"
IMPLEMENTATION_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparator_implementation_contract_v1.json"
IMPLEMENTATION_EVIDENCE = ROOT / "Evidence/stage0g4d5f2a_comparator_implementation_audit.json"
WEIGHT_EVIDENCE = ROOT / "Evidence/stage0g4d5f1c_official_comparator_weight_acquisition.json"
CACHE_EVIDENCE = ROOT / "Evidence/stage0g4d5f1b_comparator_development_cache_restore.json"
PREFLIGHT_RUNNER = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2b_resumable_variant_64step_preflight.py"
COMPLETION_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2c_six_variant_preflight_completion_freeze.py"
PREFLIGHT_REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_posthoc_preflight64_registry_v1.csv"
COMPLETION_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_preflight_completion_v1.json"
COMPLETION_EVIDENCE = ROOT / "Evidence/stage0g4d5f2c_six_variant_preflight_completion_freeze.json"

FROZEN_RUNNER = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f3a_resumable_final_variant_training.py"
RUNNER_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_final_training_runner_contract_v1.json"
RUNNER_FREEZE_EVIDENCE = ROOT / "Evidence/stage0g4d5f3a_final_training_runner_freeze.json"
RUN_BASE = ROOT / "Runs/FOVEA_DR/posthoc_comparisons/final_seed2026"
CHECKPOINT_BASE = ROOT / "Checkpoints/FOVEA_DR/posthoc_comparisons/final_seed2026"

EXPECTED_SHA = {
    PROTOCOL: "f946964923e08722dbb486aa3782b08880446040e07fb6b268ea9c28c08e6ae1",
    CANDIDATE: "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181",
    TRAINING_PROTOCOL: "5b16e0c8565b86d3cac603f416ee0b7f03055ab9a0eb9ac9e3fb007632329b40",
    SPLIT: "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140",
    TRAIN_MANIFEST: "32f9a3b6b17916b78f591007a85ce2e1910c9c4d2b1294af1c391b19f476bd15",
    VALIDATION_MANIFEST: "d94c5613b6d030dd6e99246b1b72f77746b5223139a4597bef733eb38a5786af",
    IMPLEMENTATION_SOURCE: "987f170cbfd4ef048161d29a12a9cfd6e715b45338c539b0c9ba7dfe5a04379f",
    WEIGHT_EVIDENCE: "84e5f89d29c11175169118b7010e49a91e36eb2987dc84c60961796f54894d54",
    CACHE_EVIDENCE: "1dadfa4463347b97e36cca003649991c585c48e8027b539e67047584526fee34",
    PREFLIGHT_RUNNER: "dbe713a8050ef0d454488fb18932e6e68f3cf7cf0395a4cbe344b3005d56396c",
    COMPLETION_SOURCE: "f82d5b42c5727ba29896234ebde10e5b53a7376c5a497f368f08296e0d7a4c25",
    PREFLIGHT_REGISTRY: "4104b9d37ca1dfc49735184bf8f5c3dc1e2651f822874cc632431803bd623ae4",
    COMPLETION_CONTRACT: "ce0e6184ce4e4daf5bff01c36664b85fe006c57f97962e9b132946ee797f724f",
    COMPLETION_EVIDENCE: "24a397e4786f8e32cbee5eb2981285b715acb14e0bf206558267efc840f083f4",
}

VARIANTS = [
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
    "B2_RESNET50_MULTITASK",
    "B3_CONVNEXT_TINY_MULTITASK",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
]
BACKBONE_FOR_VARIANT = {
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK": "EfficientNetV2-S",
    "B2_RESNET50_MULTITASK": "ResNet-50",
    "B3_CONVNEXT_TINY_MULTITASK": "ConvNeXt-Tiny",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS": "EfficientNetV2-S",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT": "EfficientNetV2-S",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE": "EfficientNetV2-S",
}

RUN_SEED = 2026
MAX_EPOCHS = 30
OPTIMIZER_STEPS_PER_EPOCH = 415
GRADIENT_ACCUMULATION = 2
MICRO_BATCH = 16
EFFECTIVE_BATCH = 32
EARLY_STOPPING_PATIENCE = 6
PRIMARY_MIN_DELTA = 0.002
INPUT_SIZE = 512
VALIDATION_BATCH_SIZE = 16
TOTAL_OPTIMIZER_STEPS = MAX_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH
WARMUP_STEPS = 623
MIN_LR_RATIO = 0.01
GRAD_CLIP_NORM = 1.0

EXPECTED_TRAIN_ROWS = 13251
EXPECTED_VALIDATION_ROWS = 2839
EXPECTED_CALIBRATION_ROWS = 2840
EXPECTED_TRAIN_COUNTS = {"CFP": 6128, "UWF": 5460, "OCT": 1663}
EXPECTED_VALIDATION_COUNTS = {"CFP": 1314, "UWF": 1168, "OCT": 357}

ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    atomic_bytes(path, data)
    return "CREATED_AND_FROZEN"


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def freeze_runner() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_RUNNER.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_RUNNER.exists():
        if sha256_file(FROZEN_RUNNER) != source_sha:
            raise RuntimeError("A different final-variant runner is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_RUNNER.with_name(FROZEN_RUNNER.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_RUNNER)
    if sha256_file(FROZEN_RUNNER) != source_sha:
        raise RuntimeError("Final-variant runner freeze verification failed.")
    return "COPIED_AND_FROZEN_BEFORE_FINAL_TRAINING", source_sha


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evidence_passes(payload: dict[str, Any]) -> bool:
    gates = payload.get("gates", {})
    return bool(
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and gates
        and all(bool(value) for value in gates.values())
    )


def import_implementation():
    specification = importlib.util.spec_from_file_location(
        "fovea_dr_posthoc_comparison_v1", IMPLEMENTATION_SOURCE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not import frozen comparator implementation.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def lr_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return float(step + 1) / float(WARMUP_STEPS)
    progress = float(step - WARMUP_STEPS) / float(
        max(1, TOTAL_OPTIMIZER_STEPS - WARMUP_STEPS)
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


def worker_seed_factory(seed: int):
    def seed_worker(worker_id: int) -> None:
        value = seed + worker_id
        random.seed(value)
        np.random.seed(value % (2**32))
        torch.manual_seed(value)
    return seed_worker


class ValidationDataset(Dataset):
    def __init__(self, table: pd.DataFrame, lesion_columns: list[str]):
        self.table = table.reset_index(drop=True)
        self.lesion_columns = list(lesion_columns)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        path = Path(str(row["local_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["file_size_bytes"]):
            raise RuntimeError(f"VALIDATION cache size mismatch: {path}")
        with Image.open(path) as image:
            raw = TF.to_tensor(image.convert("RGB"))
        raw = TF.resize(
            raw,
            [INPUT_SIZE, INPUT_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        quality = raw.std().clamp(0.0, 1.0)
        fov_mask = (raw.amax(dim=0, keepdim=True) > (8.0 / 255.0)).to(raw.dtype)
        normalized = TF.normalize(
            raw, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        modality = str(row["modality"])
        is_fundus = modality in {"CFP", "UWF"}
        modality_index = 0 if modality == "CFP" else (1 if modality == "UWF" else 2)
        lesion_values = [
            0.0 if pd.isna(row[column]) else float(row[column])
            for column in self.lesion_columns
        ]
        return {
            "image": normalized,
            "fov_mask": fov_mask,
            "quality": torch.tensor(float(quality)),
            "fundus_mask": torch.tensor(is_fundus, dtype=torch.bool),
            "modality_index": torch.tensor(modality_index, dtype=torch.long),
            "grade": torch.tensor(int(row["grade"]), dtype=torch.long),
            "lesion_loss_mask": torch.tensor(float(row["lesion_loss_mask"])),
            "lesion_target": torch.tensor(lesion_values, dtype=torch.float32),
            "modality": modality,
            "relative_path": str(row["relative_path"]),
        }


def build_validation_table() -> pd.DataFrame:
    split = pd.read_csv(SPLIT)
    validation = split[split["development_split"] == "VALIDATION"].copy()
    validation["relative_path"] = validation["relative_path"].astype(str).str.replace(
        "\\", "/", regex=False
    )
    manifest = pd.read_csv(VALIDATION_MANIFEST)
    manifest["relative_path"] = manifest["relative_path"].astype(str).str.replace(
        "\\", "/", regex=False
    )
    manifest_small = manifest[
        ["relative_path", "local_path", "file_size_bytes", "image_sha256"]
    ].copy()
    table = validation.merge(
        manifest_small, on="relative_path", how="inner", validate="one_to_one"
    )
    return table.sort_values("relative_path").reset_index(drop=True)


def multiclass_ece(target: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == target
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(result)


def binary_ece(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(target[selected].mean()) - float(probability[selected].mean())
            )
    return float(result)


def classification_metrics(frame: pd.DataFrame, modality: str, classes: int) -> dict[str, Any]:
    subset = frame[frame["modality"] == modality].copy()
    target = subset["target"].to_numpy(dtype=int)
    prediction = subset["prediction"].to_numpy(dtype=int)
    expected = subset["expected_grade"].to_numpy(dtype=float)
    probabilities = subset[
        [f"probability_{index}" for index in range(classes)]
    ].to_numpy(dtype=float)
    one_hot = np.eye(classes, dtype=float)[target]
    nll = -np.log(
        np.clip(probabilities[np.arange(len(target)), target], 1e-7, 1.0)
    ).mean()
    return {
        "rows": int(len(subset)),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_F1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(target, prediction, weights="quadratic")
        ),
        "NLL": float(nll),
        "Brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ECE_15_equal_width": multiclass_ece(target, probabilities),
        "expected_grade_MAE": float(mean_absolute_error(target, expected)),
    }


@torch.inference_mode()
def evaluate_validation(
    variant_id: str,
    implementation: Any,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lesion_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    lesion_targets: list[torch.Tensor] = []
    lesion_probabilities: list[torch.Tensor] = []
    lesion_masks: list[torch.Tensor] = []
    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model.forward_mixed(
                batch["image"], batch["fundus_mask"], batch["fov_mask"],
                batch["modality_index"], batch["quality"],
            )
        fundus_positions = torch.where(batch["fundus_mask"])[0]
        oct_positions = torch.where(~batch["fundus_mask"])[0]
        if len(fundus_positions) > 0:
            if variant_id in implementation.PLAIN_VARIANTS:
                probabilities = output["fundus"]["grade"]["probabilities"]
                expected = output["fundus"]["grade"]["expected_grade"]
                lesions = torch.sigmoid(output["fundus"]["lesion_logits"])
            else:
                probabilities = output["fundus"]["original"]["fusion"]["probabilities"]
                expected = output["fundus"]["original"]["fusion"]["expected_grade"]
                lesions = output["fundus"]["lesions"]["presence"]
            probabilities = probabilities.float().cpu()
            expected = expected.float().cpu()
            lesions = lesions.float().cpu()
            lesion_probabilities.append(lesions)
            lesion_targets.append(batch["lesion_target"][fundus_positions].float().cpu())
            lesion_masks.append(batch["lesion_loss_mask"][fundus_positions].float().cpu())
            for local_index, batch_position in enumerate(fundus_positions.cpu().tolist()):
                probability = probabilities[local_index].numpy()
                record = {
                    "relative_path": batch["relative_path"][batch_position],
                    "modality": batch["modality"][batch_position],
                    "target": int(batch["grade"][batch_position].cpu()),
                    "prediction": int(probability.argmax()),
                    "expected_grade": float(expected[local_index]),
                }
                for class_index, value in enumerate(probability):
                    record[f"probability_{class_index}"] = float(value)
                for lesion_index, column in enumerate(lesion_columns):
                    record[f"{column}_probability"] = float(lesions[local_index, lesion_index])
                records.append(record)
        if len(oct_positions) > 0:
            probabilities = output["oct"]["probabilities"].float().cpu()
            expected = output["oct"]["expected_grade"].float().cpu()
            for local_index, batch_position in enumerate(oct_positions.cpu().tolist()):
                probability = probabilities[local_index].numpy()
                record = {
                    "relative_path": batch["relative_path"][batch_position],
                    "modality": "OCT",
                    "target": int(batch["grade"][batch_position].cpu()),
                    "prediction": int(probability.argmax()),
                    "expected_grade": float(expected[local_index]),
                    "probability_0": float(probability[0]),
                    "probability_1": float(probability[1]),
                    "probability_2": float(probability[2]),
                    "probability_3": np.nan,
                    "probability_4": np.nan,
                }
                for column in lesion_columns:
                    record[f"{column}_probability"] = np.nan
                records.append(record)
    frame = pd.DataFrame(records).sort_values("relative_path").reset_index(drop=True)
    metrics = {
        "CFP": classification_metrics(frame, "CFP", 5),
        "UWF": classification_metrics(frame, "UWF", 5),
        "OCT": classification_metrics(frame, "OCT", 3),
    }
    metrics["mean_fundus_QWK"] = float(
        0.5 * (
            metrics["CFP"]["quadratic_weighted_kappa"]
            + metrics["UWF"]["quadratic_weighted_kappa"]
        )
    )
    metrics["mean_fundus_NLL"] = float(
        0.5 * (metrics["CFP"]["NLL"] + metrics["UWF"]["NLL"])
    )
    metrics["all_primary_metrics_finite"] = bool(
        math.isfinite(metrics["mean_fundus_QWK"])
        and math.isfinite(metrics["mean_fundus_NLL"])
    )
    lesion_summary: dict[str, Any] = {}
    if lesion_probabilities:
        targets = torch.cat(lesion_targets).numpy()
        probabilities = torch.cat(lesion_probabilities).numpy()
        masks = torch.cat(lesion_masks).numpy() > 0.5
        for index, column in enumerate(lesion_columns):
            target = targets[masks, index].astype(int)
            probability = probabilities[masks, index].astype(float)
            prediction = (probability >= 0.5).astype(int)
            lesion_summary[column] = {
                "valid_rows": int(len(target)),
                "positive_rows": int(target.sum()),
                "binary_NLL": float(
                    -(
                        target * np.log(np.clip(probability, 1e-7, 1.0))
                        + (1 - target) * np.log(np.clip(1.0 - probability, 1e-7, 1.0))
                    ).mean()
                ),
                "Brier_score": float(np.square(probability - target).mean()),
                "ECE_15_equal_width": binary_ece(target, probability),
                "accuracy_at_0_5": float(accuracy_score(target, prediction)),
                "F1_at_0_5": float(f1_score(target, prediction, zero_division=0)),
            }
    return frame, metrics, lesion_summary


def verify_official_weight(weight_evidence: dict[str, Any], variant_id: str) -> dict[str, Any]:
    target = BACKBONE_FOR_VARIANT[variant_id]
    matches = [
        record for record in weight_evidence.get("official_models", [])
        if record.get("name") == target
    ]
    if len(matches) != 1:
        raise RuntimeError(f"No unique official-weight record for {target}.")
    record = matches[0]
    path = Path(str(record.get("checkpoint_path", "")))
    valid = bool(
        path.is_file()
        and path.stat().st_size == int(record.get("checkpoint_size_bytes", -1))
        and sha256_file(path) == record.get("checkpoint_sha256")
        and all(bool(value) for value in record.get("checks", {}).values())
    )
    if not valid:
        raise RuntimeError(
            "Official checkpoint is missing or changed; rerun frozen stage 5F-1C."
        )
    return record


def run_paths(variant_id: str) -> dict[str, Path]:
    run_root = RUN_BASE / variant_id
    checkpoint_root = CHECKPOINT_BASE / variant_id
    return {
        "run_root": run_root,
        "checkpoint_root": checkpoint_root,
        "last": checkpoint_root / "last_resume.pt",
        "best": checkpoint_root / "best_model.pt",
        "history": run_root / "epoch_history.csv",
        "steps": run_root / "optimizer_step_log.csv",
        "prediction_root": run_root / "validation_predictions_by_epoch",
        "latest_predictions": run_root / "latest_validation_predictions.csv",
        "best_predictions": run_root / "best_validation_predictions.csv",
        "state": run_root / "run_state.json",
        "evidence": ROOT / "Evidence" / f"stage0g4d5f3a_{variant_id.lower()}_latest_session.json",
    }


def verify_execution_order(variant_id: str) -> bool:
    position = VARIANTS.index(variant_id)
    for predecessor in VARIANTS[:position]:
        paths = run_paths(predecessor)
        if not paths["state"].is_file():
            raise RuntimeError(f"Predecessor run is not complete: {predecessor}")
        state = load_json(paths["state"])
        required = {
            "last": state.get("last_checkpoint_sha256"),
            "best": state.get("best_checkpoint_sha256"),
            "history": state.get("history_sha256"),
            "steps": state.get("step_log_sha256"),
        }
        valid = bool(
            state.get("variant_id") == predecessor
            and state.get("status") == "COMPLETE"
            and state.get("locked_test_used") is False
            and all(
                paths[name].is_file()
                and expected
                and sha256_file(paths[name]) == expected
                for name, expected in required.items()
            )
        )
        if not valid:
            raise RuntimeError(f"Predecessor state failed verification: {predecessor}")
    return True


def selected_preflight_paths(variant_id: str) -> tuple[Path, Path]:
    log_path = (
        ROOT / "Runs/FOVEA_DR/posthoc_comparisons/preflight64"
        / variant_id / "optimizer_step_log.csv"
    )
    evidence_path = (
        ROOT / "Evidence"
        / f"stage0g4d5f2b_{variant_id.lower()}_64step_preflight.json"
    )
    return log_path, evidence_path


def make_best_payload(resume: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "FOVEA_DR_POSTHOC_VARIANT_BEST_V1",
        "variant_id": resume["variant_id"],
        "run_seed": resume["run_seed"],
        "epoch": resume["epoch_completed"],
        "mean_fundus_QWK": resume["best_mean_fundus_QWK"],
        "mean_fundus_NLL": resume["best_mean_fundus_NLL"],
        "validation_metrics": resume["completed_epoch_metrics"],
        "lesion_metrics": resume["completed_epoch_lesion_metrics"],
        "runner_source_sha256": resume["runner_source_sha256"],
        "implementation_source_sha256": resume["implementation_source_sha256"],
        "model_state_dict": resume["model_state_dict"],
    }


def reconcile_from_checkpoint(
    paths: dict[str, Path], resume: dict[str, Any]
) -> list[str]:
    repairs: list[str] = []
    epoch = int(resume["epoch_completed"])
    global_step = int(resume["global_optimizer_step"])
    history = pd.DataFrame(resume["history_records"])
    steps = pd.DataFrame(resume["step_records"])
    expected_epochs = list(range(1, epoch + 1))
    if history["epoch"].astype(int).tolist() != expected_epochs:
        raise RuntimeError("Checkpoint history is not contiguous through its epoch.")
    if steps["global_optimizer_step"].astype(int).tolist() != list(range(1, global_step + 1)):
        raise RuntimeError("Checkpoint step log is not contiguous through its global step.")
    prediction_path = Path(resume["completed_epoch_predictions"])
    if not prediction_path.is_file() or sha256_file(prediction_path) != resume[
        "completed_epoch_predictions_sha256"
    ]:
        raise RuntimeError("Checkpoint-linked immutable validation predictions changed.")
    expected_history = history.to_csv(index=False).encode("utf-8")
    if not paths["history"].is_file() or paths["history"].read_bytes() != expected_history:
        atomic_bytes(paths["history"], expected_history)
        repairs.append("HISTORY_REBUILT_FROM_CHECKPOINT")
    expected_steps = steps.to_csv(index=False).encode("utf-8")
    if not paths["steps"].is_file() or paths["steps"].read_bytes() != expected_steps:
        atomic_bytes(paths["steps"], expected_steps)
        repairs.append("STEP_LOG_REBUILT_FROM_CHECKPOINT")
    if (
        not paths["latest_predictions"].is_file()
        or sha256_file(paths["latest_predictions"])
        != resume["completed_epoch_predictions_sha256"]
    ):
        atomic_copy(prediction_path, paths["latest_predictions"])
        repairs.append("LATEST_PREDICTIONS_REBUILT_FROM_CHECKPOINT")
    best_epoch = int(resume["best_epoch"])
    best_valid = False
    if paths["best"].is_file():
        best = torch.load(paths["best"], map_location="cpu", weights_only=False)
        best_valid = bool(
            best.get("format") == "FOVEA_DR_POSTHOC_VARIANT_BEST_V1"
            and best.get("variant_id") == resume["variant_id"]
            and int(best.get("epoch", -1)) == best_epoch
            and best.get("runner_source_sha256") == resume["runner_source_sha256"]
            and math.isclose(
                float(best.get("mean_fundus_QWK", math.nan)),
                float(resume["best_mean_fundus_QWK"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(best.get("mean_fundus_NLL", math.nan)),
                float(resume["best_mean_fundus_NLL"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    if not best_valid:
        if best_epoch != epoch:
            raise RuntimeError("Older best checkpoint is missing or inconsistent.")
        atomic_torch(paths["best"], make_best_payload(resume))
        repairs.append("BEST_CHECKPOINT_REBUILT_FROM_LAST")
    best_prediction_path = paths["prediction_root"] / f"epoch_{best_epoch:03d}.csv"
    if not best_prediction_path.is_file():
        raise RuntimeError("Immutable best-epoch validation predictions are missing.")
    if (
        not paths["best_predictions"].is_file()
        or sha256_file(paths["best_predictions"]) != sha256_file(best_prediction_path)
    ):
        atomic_copy(best_prediction_path, paths["best_predictions"])
        repairs.append("BEST_PREDICTIONS_REBUILT")
    return repairs


def checkpoint_payload(
    variant_id: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_qwk: float,
    best_nll: float,
    best_epoch: int,
    bad_epochs: int,
    runner_sha: str,
    history_records: list[dict[str, Any]],
    step_records: list[dict[str, Any]],
    metrics: dict[str, Any],
    lesion_metrics: dict[str, Any],
    prediction_path: Path,
    prediction_sha: str,
) -> dict[str, Any]:
    return {
        "format": "FOVEA_DR_POSTHOC_VARIANT_RESUME_V1",
        "variant_id": variant_id,
        "run_seed": RUN_SEED,
        "epoch_completed": epoch,
        "global_optimizer_step": global_step,
        "best_mean_fundus_QWK": best_qwk,
        "best_mean_fundus_NLL": best_nll,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "runner_source_sha256": runner_sha,
        "comparison_protocol_sha256": EXPECTED_SHA[PROTOCOL],
        "implementation_source_sha256": EXPECTED_SHA[IMPLEMENTATION_SOURCE],
        "completion_contract_sha256": EXPECTED_SHA[COMPLETION_CONTRACT],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
        "history_records": history_records,
        "step_records": step_records,
        "completed_epoch_metrics": metrics,
        "completed_epoch_lesion_metrics": lesion_metrics,
        "completed_epoch_predictions": str(prediction_path),
        "completed_epoch_predictions_sha256": prediction_sha,
    }


def write_run_state(
    paths: dict[str, Path],
    variant_id: str,
    runner_sha: str,
    epoch: int,
    global_step: int,
    best_epoch: int,
    best_qwk: float,
    best_nll: float,
    bad_epochs: int,
    complete: bool,
) -> dict[str, Any]:
    state = {
        "run": "FOVEA_DR_POSTHOC_VARIANT_SEED2026",
        "variant_id": variant_id,
        "status": "COMPLETE" if complete else "RESUMABLE_CHECKPOINT_READY",
        "last_completed_epoch": epoch,
        "global_optimizer_step": global_step,
        "best_epoch": best_epoch,
        "best_mean_fundus_QWK": best_qwk,
        "best_mean_fundus_NLL": best_nll,
        "bad_epochs": bad_epochs,
        "early_stopped": bad_epochs >= EARLY_STOPPING_PATIENCE,
        "runner_source_sha256": runner_sha,
        "last_checkpoint_sha256": sha256_file(paths["last"]),
        "best_checkpoint_sha256": sha256_file(paths["best"]),
        "history_sha256": sha256_file(paths["history"]),
        "step_log_sha256": sha256_file(paths["steps"]),
        "latest_validation_predictions_sha256": sha256_file(paths["latest_predictions"]),
        "best_validation_predictions_sha256": sha256_file(paths["best_predictions"]),
        "validation_used_for_model_selection": True,
        "calibration_used": False,
        "locked_test_used": False,
    }
    atomic_json(paths["state"], state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--session-epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.session_epochs < 1 or args.session_epochs > 4:
        raise RuntimeError("session-epochs must be between 1 and 4.")
    if args.workers < 0 or args.workers > 4:
        raise RuntimeError("workers must be between 0 and 4.")
    if args.freeze_only and args.variant is not None:
        raise RuntimeError("freeze-only must not be combined with --variant.")
    if not args.freeze_only and args.variant is None:
        raise RuntimeError("--variant is required unless --freeze-only is used.")
    variant_id = args.variant
    paths = run_paths(variant_id) if variant_id is not None else None

    print("=" * 96)
    print("STAGE 0G-4D-5F-3A — RESUMABLE FINAL POST-HOC VARIANT TRAINING")
    print("=" * 96)
    print("Mode   :", "FREEZE_ONLY" if args.freeze_only else "FINAL_TRAINING")
    if variant_id is not None:
        print("Variant:", variant_id)

    print("\n[1] FROZEN ARTIFACT AND EXECUTION-ORDER VERIFICATION")
    print("-" * 96)
    gates: dict[str, bool] = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(verified)
        print(f"{path.name:82s}: exists={exists}, hash_verified={verified}")
    for path in [SAMPLER, LESION_WEIGHTS, IMPLEMENTATION_CONTRACT, IMPLEMENTATION_EVIDENCE]:
        exists = path.is_file() and path.stat().st_size > 0
        gates[f"exists_nonempty::{path.name}"] = bool(exists)
        print(f"{path.name:82s}: exists_nonempty={exists}")
    if not all(gates.values()):
        raise RuntimeError("A frozen parent artifact is missing or changed.")

    protocol = load_json(PROTOCOL)
    protocol_ids = [item["id"] for item in protocol["comparators"]]
    protocol_ids += [item["id"] for item in protocol["ablations"]]
    completion = load_json(COMPLETION_CONTRACT)
    completion_evidence = load_json(COMPLETION_EVIDENCE)
    implementation_evidence = load_json(IMPLEMENTATION_EVIDENCE)
    weight_evidence = load_json(WEIGHT_EVIDENCE)
    cache_evidence = load_json(CACHE_EVIDENCE)
    registry = pd.read_csv(PREFLIGHT_REGISTRY)
    gates.update({
        "six_variants_match_protocol_order": protocol_ids == VARIANTS,
        "locked_test_prohibited": protocol["data_access_policy"]["LOCKED_INTERNAL_TEST"].startswith("PROHIBITED"),
        "completion_contract_ready": completion.get("status") == "SIX_VARIANTS_READY_FOR_INDEPENDENT_FINAL_TRAINING",
        "completion_variant_order_exact": completion.get("variants_in_locked_order") == VARIANTS,
        "completion_evidence_passed": evidence_passes(completion_evidence),
        "implementation_evidence_passed": evidence_passes(implementation_evidence),
        "official_weight_evidence_passed": evidence_passes(weight_evidence),
        "development_cache_evidence_passed": evidence_passes(cache_evidence),
        "preflight_registry_has_exact_six_rows": len(registry) == 6,
        "preflight_registry_order_exact": registry["variant_id"].tolist() == VARIANTS,
        "all_registry_checks_passed": registry["all_checks_passed"].astype(str).str.lower().eq("true").all(),
        "preflight_weights_disallowed": bool(completion.get("preflight_weights_must_not_be_reused")),
    })
    if variant_id is not None:
        selected_registry = registry[registry["variant_id"] == variant_id]
        preflight_log, preflight_evidence_path = selected_preflight_paths(variant_id)
        preflight_evidence = load_json(preflight_evidence_path) if preflight_evidence_path.is_file() else {}
        row = selected_registry.iloc[0] if len(selected_registry) == 1 else None
        gates.update({
            "selected_variant_is_protocol_locked": variant_id in protocol_ids,
            "exactly_one_selected_preflight_registry_row": len(selected_registry) == 1,
            "selected_preflight_all_checks_passed": (
                row is not None and str(row["all_checks_passed"]).lower() == "true"
            ),
            "selected_preflight_log_hash_verified": (
                row is not None and preflight_log.is_file()
                and sha256_file(preflight_log) == str(row["preflight_log_sha256"])
            ),
            "selected_preflight_evidence_hash_verified": (
                row is not None and preflight_evidence_path.is_file()
                and sha256_file(preflight_evidence_path) == str(row["preflight_evidence_sha256"])
            ),
            "selected_preflight_evidence_passed": evidence_passes(preflight_evidence),
            "sampler_hash_linked_to_selected_preflight": (
                SAMPLER.is_file()
                and sha256_file(SAMPLER) == preflight_evidence.get("sampler_artifact_sha256")
            ),
            "lesion_weight_hash_linked_to_selected_preflight": (
                LESION_WEIGHTS.is_file()
                and sha256_file(LESION_WEIGHTS) == preflight_evidence.get("lesion_weight_artifact_sha256")
            ),
            "execution_order_verified": verify_execution_order(variant_id),
        })
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Final-training parent gate failed: {failed}")
    runner_action, runner_sha = freeze_runner()
    print("Runner freeze action:", runner_action)
    print("Runner SHA-256     :", runner_sha)

    if args.freeze_only:
        gates.update({
            "no_final_variant_checkpoint_exists": not any(
                CHECKPOINT_BASE.glob("*/last_resume.pt")
            ),
            "model_not_instantiated": True,
            "optimizer_not_created": True,
            "medical_images_not_accessed": True,
            "final_training_not_started": True,
            "VALIDATION_not_accessed": True,
            "CALIBRATION_not_accessed": True,
            "locked_test_not_accessed": True,
        })
        if not all(gates.values()):
            failed = [name for name, passed in gates.items() if not passed]
            raise RuntimeError(f"Final-runner freeze gate failed: {failed}")
        contract = {
            "contract_id": "MMRDR_FOVEA_DR_POSTHOC_FINAL_TRAINING_RUNNER_V1",
            "stage": "0G-4D-5F-3A",
            "status": "FROZEN_BEFORE_FINAL_VARIANT_TRAINING",
            "runner": str(FROZEN_RUNNER),
            "runner_source_sha256": runner_sha,
            "variants_in_mandatory_order": VARIANTS,
            "run_seed": RUN_SEED,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
            "micro_batch": MICRO_BATCH,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch": EFFECTIVE_BATCH,
            "fresh_initialization": "OFFICIAL_TORCHVISION_IMAGENET_ONLY",
            "preflight_checkpoint_initialization_allowed": False,
            "TRAIN_access": "OPTIMIZATION_ONLY",
            "VALIDATION_access": "EARLY_STOPPING_AND_MODEL_SELECTION_ONLY",
            "CALIBRATION_access": "PROHIBITED_DURING_FINAL_TRAINING",
            "LOCKED_INTERNAL_TEST_access": "PROHIBITED",
            "save_order_recovery": "LAST_CHECKPOINT_CONTAINS_AUTHORITATIVE_FULL_HISTORY_AND_STEP_LOG",
        }
        contract_action = frozen_json(RUNNER_CONTRACT, contract)
        contract_sha = sha256_file(RUNNER_CONTRACT)
        evidence = {
            "stage": "0G-4D-5F-3A",
            "status": "PASS",
            "purpose": "RESUMABLE_FINAL_VARIANT_TRAINING_RUNNER_FREEZE",
            "runner": str(FROZEN_RUNNER),
            "runner_source_sha256": runner_sha,
            "runner_contract": str(RUNNER_CONTRACT),
            "runner_contract_sha256": contract_sha,
            "gates": {name: bool(value) for name, value in gates.items()},
            "model_instantiated": False,
            "optimizer_created": False,
            "medical_images_accessed": False,
            "final_training_started": False,
            "validation_accessed": False,
            "calibration_accessed": False,
            "locked_test_accessed": False,
            "next_required_stage": "START_B1_FINAL_TRAINING_FROM_OFFICIAL_IMAGENET_WEIGHTS",
        }
        evidence_action = frozen_json(RUNNER_FREEZE_EVIDENCE, evidence)
        print("\n[2] RUNNER CONTRACT AND EVIDENCE FREEZE")
        print("-" * 96)
        print("Contract action :", contract_action)
        print("Contract        :", RUNNER_CONTRACT)
        print("Contract SHA-256:", contract_sha)
        print("Evidence action :", evidence_action)
        print("Evidence        :", RUNNER_FREEZE_EVIDENCE)
        print("Evidence SHA-256:", sha256_file(RUNNER_FREEZE_EVIDENCE))
        print("Model instantiated    : NO")
        print("Medical images accessed: NO")
        print("Final training started : NO")
        print("CALIBRATION accessed   : NO")
        print("Locked test accessed   : NO")
        print("\nNEXT REQUIRED STAGE:")
        print("Start B1 final training from official ImageNet weights.")
        print("\n" + "=" * 96)
        print("STAGE 0G-4D-5F-3A RUNNER FREEZE STATUS: PASS")
        print("=" * 96)
        return

    freeze_evidence = load_json(RUNNER_FREEZE_EVIDENCE) if RUNNER_FREEZE_EVIDENCE.is_file() else {}
    runner_contract = load_json(RUNNER_CONTRACT) if RUNNER_CONTRACT.is_file() else {}
    freeze_gates = freeze_evidence.get("gates", {})
    if not (
        evidence_passes(freeze_evidence)
        and freeze_evidence.get("runner_source_sha256") == runner_sha
        and RUNNER_CONTRACT.is_file()
        and freeze_evidence.get("runner_contract_sha256") == sha256_file(RUNNER_CONTRACT)
        and runner_contract.get("runner_source_sha256") == runner_sha
        and runner_contract.get("status") == "FROZEN_BEFORE_FINAL_VARIANT_TRAINING"
        and runner_contract.get("variants_in_mandatory_order") == VARIANTS
        and bool(freeze_gates.get("final_training_not_started"))
    ):
        raise RuntimeError("Frozen final-training runner contract/evidence gate failed.")

    print("\n[2] GPU, OFFICIAL INITIALIZATION, AND DEVELOPMENT TABLES")
    print("-" * 96)
    cuda_available = torch.cuda.is_available()
    visible_devices = torch.cuda.device_count() if cuda_available else 0
    if not cuda_available or visible_devices != 1:
        raise RuntimeError("Exactly one CUDA GPU is required.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_vram_gb < 10.0:
        raise RuntimeError("At least 10 GiB of GPU memory is required.")
    official_weight = verify_official_weight(weight_evidence, variant_id)

    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    torch.cuda.manual_seed_all(RUN_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not TRAIN_CACHE_ROOT.is_dir() or not VALIDATION_CACHE_ROOT.is_dir():
        raise RuntimeError("Temporary TRAIN/VALIDATION caches are missing; rerun 5F-1B.")
    implementation = import_implementation()
    core = implementation.get_core()
    train_table, sampler_weight_column = core.build_training_table()
    validation_table = build_validation_table()
    train_counts = {
        str(key): int(value) for key, value in train_table["modality"].value_counts().to_dict().items()
    }
    validation_counts = {
        str(key): int(value) for key, value in validation_table["modality"].value_counts().to_dict().items()
    }
    split_roles = pd.read_csv(SPLIT, usecols=["development_split"])["development_split"].value_counts().to_dict()
    table_gates = {
        "exact_TRAIN_rows_and_counts": len(train_table) == EXPECTED_TRAIN_ROWS and train_counts == EXPECTED_TRAIN_COUNTS,
        "all_training_rows_are_TRAIN": bool(train_table["development_split"].eq("TRAIN").all()),
        "exact_VALIDATION_rows_and_counts": len(validation_table) == EXPECTED_VALIDATION_ROWS and validation_counts == EXPECTED_VALIDATION_COUNTS,
        "all_validation_rows_are_VALIDATION": bool(validation_table["development_split"].eq("VALIDATION").all()),
        "three_way_split_counts_exact": split_roles == {"TRAIN": EXPECTED_TRAIN_ROWS, "CALIBRATION": EXPECTED_CALIBRATION_ROWS, "VALIDATION": EXPECTED_VALIDATION_ROWS},
        "all_TRAIN_cache_files_size_matched": all(Path(str(row["local_path"])).is_file() and Path(str(row["local_path"])).stat().st_size == int(row["file_size_bytes"]) for row in train_table.to_dict("records")),
        "all_VALIDATION_cache_files_size_matched": all(Path(str(row["local_path"])).is_file() and Path(str(row["local_path"])).stat().st_size == int(row["file_size_bytes"]) for row in validation_table.to_dict("records")),
    }
    if not all(table_gates.values()):
        raise RuntimeError(f"Development table/cache gate failed: {[k for k, v in table_gates.items() if not v]}")

    lesion_table = pd.read_csv(LESION_WEIGHTS)
    lesion_pos_weight = torch.tensor(
        lesion_table["sqrt_pos_weight"].to_numpy(dtype=np.float32), device=device
    )
    if not (
        len(lesion_pos_weight) == 7
        and bool(torch.isfinite(lesion_pos_weight).all())
        and bool(torch.all(lesion_pos_weight > 0))
    ):
        raise RuntimeError("Frozen lesion positive weights are invalid.")
    candidate_weights = load_json(CANDIDATE)["candidate_loss"]
    active_loss_weights = implementation.variant_loss_weights(variant_id, candidate_weights)
    expected_active_loss_weights = preflight_evidence.get("active_loss_weights", {})
    if active_loss_weights != expected_active_loss_weights:
        raise RuntimeError("Final active loss weights differ from the selected preflight.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = implementation.build_variant(variant_id).to(device)
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("Freshly initialized model contains non-finite parameters.")
    optimizer, optimizer_group_counts = implementation.build_comparison_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    expected_parameters = int(selected_registry.iloc[0]["total_parameters"])
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != expected_parameters:
        raise RuntimeError("Final model parameter count differs from the audited implementation.")
    print("GPU                       :", gpu_name)
    print("VRAM GB                   :", f"{total_vram_gb:.3f}")
    print("Official checkpoint SHA   :", official_weight["checkpoint_sha256"])
    print("TRAIN rows/counts         :", len(train_table), train_counts)
    print("VALIDATION rows/counts    :", len(validation_table), validation_counts)
    print("Sampler weight column     :", sampler_weight_column)
    print("Parameters                :", actual_parameters)
    print("Optimizer groups          :", optimizer_group_counts)
    print("Active loss weights       :", active_loss_weights)
    print("Disposable preflight used : NO")

    for key in ["run_root", "checkpoint_root", "prediction_root"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    global_step = 0
    best_qwk = -math.inf
    best_nll = math.inf
    best_epoch = 0
    bad_epochs = 0
    history_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    resumed = False
    recovery_actions: list[str] = []

    if paths["last"].is_file():
        print("\n[3] RESUME CHECKPOINT")
        print("-" * 96)
        resume = torch.load(paths["last"], map_location=device, weights_only=False)
        resume_checks = {
            "format": resume.get("format") == "FOVEA_DR_POSTHOC_VARIANT_RESUME_V1",
            "variant": resume.get("variant_id") == variant_id,
            "seed": resume.get("run_seed") == RUN_SEED,
            "runner": resume.get("runner_source_sha256") == runner_sha,
            "protocol": resume.get("comparison_protocol_sha256") == EXPECTED_SHA[PROTOCOL],
            "implementation": resume.get("implementation_source_sha256") == EXPECTED_SHA[IMPLEMENTATION_SOURCE],
            "completion": resume.get("completion_contract_sha256") == EXPECTED_SHA[COMPLETION_CONTRACT],
        }
        if not all(resume_checks.values()):
            raise RuntimeError(f"Resume checkpoint identity gate failed: {resume_checks}")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        random.setstate(resume["python_random_state"])
        np.random.set_state(resume["numpy_random_state"])
        torch.set_rng_state(resume["torch_random_state"].detach().cpu())
        torch.cuda.set_rng_state_all([
            state.detach().cpu() for state in resume["cuda_random_state"]
        ])
        start_epoch = int(resume["epoch_completed"]) + 1
        global_step = int(resume["global_optimizer_step"])
        best_qwk = float(resume["best_mean_fundus_QWK"])
        best_nll = float(resume["best_mean_fundus_NLL"])
        best_epoch = int(resume["best_epoch"])
        bad_epochs = int(resume["bad_epochs"])
        history_records = list(resume["history_records"])
        step_records = list(resume["step_records"])
        recovery_actions = reconcile_from_checkpoint(paths, resume)
        resumed = True
        print("Resumed after epoch       :", start_epoch - 1)
        print("Global optimizer step     :", global_step)
        print("Best epoch                :", best_epoch)
        print("Best mean fundus QWK      :", best_qwk)
        print("Save-order recovery       :", recovery_actions or "NOT_REQUIRED")
    else:
        existing_outputs = [
            path for name, path in paths.items()
            if name not in {"run_root", "checkpoint_root", "prediction_root"}
            and path.exists()
        ]
        immutable_predictions = list(paths["prediction_root"].glob("epoch_*.csv"))
        if existing_outputs or immutable_predictions:
            raise RuntimeError(
                "Fresh run has residual final-training artifacts but no resume checkpoint."
            )
        print("\n[3] FRESH OFFICIAL INITIALIZATION")
        print("-" * 96)
        print("Official torchvision ImageNet weights loaded.")
        print("No disposable preflight checkpoint was read.")

    if start_epoch > MAX_EPOCHS or bad_epochs >= EARLY_STOPPING_PATIENCE:
        complete = True
        write_run_state(
            paths, variant_id, runner_sha, start_epoch - 1, global_step,
            best_epoch, best_qwk, best_nll, bad_epochs, complete,
        )
        print("Run is already complete; no training was repeated.")
        print("STAGE 0G-4D-5F-3A SESSION STATUS: PASS — EXISTING COMPLETE")
        return

    validation_loader = DataLoader(
        ValidationDataset(validation_table, list(core.LESION_COLUMNS)),
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed_factory(RUN_SEED + 900000),
    )

    session_start = time.perf_counter()
    epochs_completed_this_session = 0
    all_losses_finite = True
    all_gradients_finite = True
    all_gradients_nonzero = True
    no_amp_overflow = True
    all_step_compositions_exact = True
    last_epoch_allowed = min(MAX_EPOCHS, start_epoch + args.session_epochs - 1)

    for epoch in range(start_epoch, last_epoch_allowed + 1):
        print("\n" + "=" * 96)
        print(f"{variant_id} — EPOCH {epoch}/{MAX_EPOCHS}")
        print("=" * 96)
        epoch_start = time.perf_counter()
        core.SEED = RUN_SEED + epoch * 1000
        batch_sampler = core.LockedWeightedBatchSampler(
            train_table, RUN_SEED + epoch, OPTIMIZER_STEPS_PER_EPOCH
        )
        if len(batch_sampler) != OPTIMIZER_STEPS_PER_EPOCH * GRADIENT_ACCUMULATION:
            raise RuntimeError("Locked sampler emitted an unexpected micro-batch count.")
        train_loader = DataLoader(
            core.FOVEADRTrainDataset(train_table),
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            worker_init_fn=core.worker_seed if args.workers > 0 else None,
            generator=torch.Generator().manual_seed(RUN_SEED + epoch),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        iterator = iter(train_loader)
        epoch_losses: list[float] = []
        epoch_step_records: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats()

        for step_in_epoch in range(1, OPTIMIZER_STEPS_PER_EPOCH + 1):
            step_start = time.perf_counter()
            step_loss = 0.0
            components = {
                "grade_NLL": 0.0,
                "ordinal_EMD": 0.0,
                "lesion_BCE": 0.0,
                "clinical_alignment": 0.0,
                "counterfactual": 0.0,
                "route_agreement": 0.0,
                "OCT_NLL": 0.0,
            }
            modalities = {"CFP": 0, "UWF": 0, "OCT": 0}
            for _ in range(GRADIENT_ACCUMULATION):
                batch = core.move_batch(next(iterator), device)
                for modality in batch["modality"]:
                    modalities[str(modality)] += 1
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    output = model.forward_mixed(
                        batch["image"], batch["fundus_mask"], batch["fov_mask"],
                        batch["modality_index"], batch["quality"],
                    )
                    total_loss, parts = implementation.compute_variant_losses(
                        variant_id, output, batch, lesion_pos_weight, candidate_weights
                    )
                    scaled_loss = total_loss / GRADIENT_ACCUMULATION
                finite_loss = bool(torch.isfinite(total_loss).all())
                all_losses_finite = all_losses_finite and finite_loss
                if not finite_loss:
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch}, step={step_in_epoch}"
                    )
                scaler.scale(scaled_loss).backward()
                step_loss += float(scaled_loss.detach().cpu())
                for name, value in parts.items():
                    components[name] += float(value.detach().cpu()) / GRADIENT_ACCUMULATION

            scaler.unscale_(optimizer)
            gradients = [
                parameter.grad for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            gradients_finite = bool(gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            gradients_nonzero = bool(gradients) and any(
                bool(torch.count_nonzero(gradient)) for gradient in gradients
            )
            all_gradients_finite = all_gradients_finite and gradients_finite
            all_gradients_nonzero = all_gradients_nonzero and gradients_nonzero
            if not gradients_finite or not gradients_nonzero:
                raise FloatingPointError(
                    f"Invalid gradients at epoch={epoch}, step={step_in_epoch}"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if scale_after < scale_before:
                no_amp_overflow = False
                raise FloatingPointError(
                    f"AMP overflow at epoch={epoch}, step={step_in_epoch}"
                )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            epoch_losses.append(step_loss)
            learning_rates = {
                group["group_name"]: float(group["lr"])
                for group in optimizer.param_groups
            }
            record = {
                "variant_id": variant_id,
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_optimizer_step": global_step,
                "total_loss": step_loss,
                **components,
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "grad_scaler_scale": scale_after,
                "backbone_lr": learning_rates["backbone_decay"],
                "new_module_lr": learning_rates["new_decay"],
                "step_seconds": time.perf_counter() - step_start,
                "CFP_samples": modalities["CFP"],
                "UWF_samples": modalities["UWF"],
                "OCT_samples": modalities["OCT"],
            }
            epoch_step_records.append(record)
            all_step_compositions_exact = all_step_compositions_exact and (
                modalities == {"CFP": 15, "UWF": 13, "OCT": 4}
            )
            if step_in_epoch == 1 or step_in_epoch % 50 == 0 or step_in_epoch == OPTIMIZER_STEPS_PER_EPOCH:
                print(
                    f"Train {step_in_epoch:03d}/{OPTIMIZER_STEPS_PER_EPOCH} | "
                    f"loss={step_loss:.6f} | grad={float(gradient_norm):.4f} | "
                    f"scale={scale_after:.0f}"
                )

        del iterator, train_loader
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_start
        print("\nVALIDATION EVALUATION")
        validation_start = time.perf_counter()
        predictions, metrics, lesion_metrics = evaluate_validation(
            variant_id, implementation, model, validation_loader, device,
            list(core.LESION_COLUMNS),
        )
        validation_seconds = time.perf_counter() - validation_start
        if not metrics["all_primary_metrics_finite"]:
            raise FloatingPointError("Non-finite primary VALIDATION metric.")
        current_qwk = float(metrics["mean_fundus_QWK"])
        current_nll = float(metrics["mean_fundus_NLL"])
        primary_improved = current_qwk > best_qwk + PRIMARY_MIN_DELTA
        tie_break_improved = (
            abs(current_qwk - best_qwk) <= PRIMARY_MIN_DELTA
            and current_nll < best_nll
        )
        is_best = best_epoch == 0 or primary_improved or tie_break_improved
        if is_best:
            best_qwk = current_qwk
            best_nll = current_nll
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        epoch_record = {
            "variant_id": variant_id,
            "epoch": epoch,
            "global_optimizer_step": global_step,
            "mean_train_loss": float(np.mean(epoch_losses)),
            "median_train_loss": float(np.median(epoch_losses)),
            "CFP_QWK": metrics["CFP"]["quadratic_weighted_kappa"],
            "UWF_QWK": metrics["UWF"]["quadratic_weighted_kappa"],
            "mean_fundus_QWK": current_qwk,
            "CFP_NLL": metrics["CFP"]["NLL"],
            "UWF_NLL": metrics["UWF"]["NLL"],
            "mean_fundus_NLL": current_nll,
            "OCT_QWK": metrics["OCT"]["quadratic_weighted_kappa"],
            "OCT_NLL": metrics["OCT"]["NLL"],
            "OCT_macro_F1": metrics["OCT"]["macro_F1"],
            "is_best": is_best,
            "bad_epochs": bad_epochs,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "peak_reserved_VRAM_GB": torch.cuda.max_memory_reserved() / 1024**3,
        }
        history_records.append(epoch_record)
        step_records.extend(epoch_step_records)

        prediction_path = paths["prediction_root"] / f"epoch_{epoch:03d}.csv"
        prediction_bytes = predictions.to_csv(index=False).encode("utf-8")
        if prediction_path.exists() and prediction_path.read_bytes() != prediction_bytes:
            raise RuntimeError("Immutable predictions already differ for this epoch.")
        if not prediction_path.exists():
            atomic_bytes(prediction_path, prediction_bytes)
        prediction_sha = sha256_file(prediction_path)
        payload = checkpoint_payload(
            variant_id, model, optimizer, scheduler, scaler, epoch, global_step,
            best_qwk, best_nll, best_epoch, bad_epochs, runner_sha,
            history_records, step_records, metrics, lesion_metrics,
            prediction_path, prediction_sha,
        )
        atomic_torch(paths["last"], payload)
        if is_best:
            atomic_torch(paths["best"], make_best_payload(payload))
            atomic_copy(prediction_path, paths["best_predictions"])
        atomic_csv(paths["history"], pd.DataFrame(history_records))
        atomic_csv(paths["steps"], pd.DataFrame(step_records))
        atomic_copy(prediction_path, paths["latest_predictions"])

        complete = bad_epochs >= EARLY_STOPPING_PATIENCE or epoch >= MAX_EPOCHS
        state = write_run_state(
            paths, variant_id, runner_sha, epoch, global_step, best_epoch,
            best_qwk, best_nll, bad_epochs, complete,
        )
        print("Mean train loss :", f"{epoch_record['mean_train_loss']:.6f}")
        print("CFP QWK         :", f"{metrics['CFP']['quadratic_weighted_kappa']:.6f}")
        print("UWF QWK         :", f"{metrics['UWF']['quadratic_weighted_kappa']:.6f}")
        print("Mean fundus QWK :", f"{current_qwk:.6f}")
        print("Mean fundus NLL :", f"{current_nll:.6f}")
        print("OCT QWK         :", f"{metrics['OCT']['quadratic_weighted_kappa']:.6f}")
        print("OCT macro-F1    :", f"{metrics['OCT']['macro_F1']:.6f}")
        print("Best epoch      :", best_epoch)
        print("Bad epochs      :", bad_epochs, "/", EARLY_STOPPING_PATIENCE)
        print("Last checkpoint :", paths["last"])
        epochs_completed_this_session += 1
        if complete:
            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print("EARLY STOPPING TRIGGERED")
            break

    final_epoch = int(history_records[-1]["epoch"])
    run_complete = bad_epochs >= EARLY_STOPPING_PATIENCE or final_epoch >= MAX_EPOCHS
    session_gates = {
        **gates,
        **table_gates,
        "CUDA_available": cuda_available,
        "exactly_one_GPU_visible": visible_devices == 1,
        "GPU_memory_at_least_10GB": total_vram_gb >= 10.0,
        "official_ImageNet_weight_SHA_verified": True,
        "model_parameter_count_matches_audit": actual_parameters == expected_parameters,
        "preflight_checkpoint_not_used_for_initialization": True,
        "at_least_one_epoch_completed_this_session": epochs_completed_this_session >= 1,
        "all_session_losses_finite": all_losses_finite,
        "all_session_gradients_finite": all_gradients_finite,
        "all_session_gradients_nonzero": all_gradients_nonzero,
        "no_AMP_overflow_or_skipped_step": no_amp_overflow,
        "all_effective_batch_modality_compositions_exact": all_step_compositions_exact,
        "global_optimizer_step_count_correct": global_step == final_epoch * OPTIMIZER_STEPS_PER_EPOCH,
        "history_contiguous": [int(row["epoch"]) for row in history_records] == list(range(1, final_epoch + 1)),
        "step_log_contiguous": [int(row["global_optimizer_step"]) for row in step_records] == list(range(1, global_step + 1)),
        "resume_checkpoint_created": paths["last"].is_file(),
        "best_checkpoint_created": paths["best"].is_file(),
        "runner_frozen_and_hash_verified": FROZEN_RUNNER.is_file() and sha256_file(FROZEN_RUNNER) == runner_sha,
        "VALIDATION_used_only_for_selection": True,
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
    }
    print("\n" + "=" * 96)
    print("FINAL VARIANT TRAINING SESSION GATES")
    print("=" * 96)
    for name, passed in session_gates.items():
        print(f"{name:88s}: {passed}")
    if not all(session_gates.values()):
        raise RuntimeError(
            f"Final-variant session gate failed: {[k for k, v in session_gates.items() if not v]}"
        )
    state = write_run_state(
        paths, variant_id, runner_sha, final_epoch, global_step, best_epoch,
        best_qwk, best_nll, bad_epochs, run_complete,
    )
    evidence = {
        "stage": "0G-4D-5F-3A",
        "status": "PASS",
        "variant_id": variant_id,
        "session_type": "RESUME" if resumed else "FRESH_OFFICIAL_INITIALIZATION",
        "session_epochs_requested": args.session_epochs,
        "epochs_completed_this_session": epochs_completed_this_session,
        "last_completed_epoch": final_epoch,
        "global_optimizer_step": global_step,
        "best_epoch": best_epoch,
        "best_mean_fundus_QWK": best_qwk,
        "best_mean_fundus_NLL": best_nll,
        "bad_epochs": bad_epochs,
        "run_complete": run_complete,
        "save_order_recovery_actions": recovery_actions,
        "runner_source": str(FROZEN_RUNNER),
        "runner_source_sha256": runner_sha,
        "official_checkpoint_sha256": official_weight["checkpoint_sha256"],
        "run_state": str(paths["state"]),
        "run_state_sha256": sha256_file(paths["state"]),
        "last_checkpoint": str(paths["last"]),
        "last_checkpoint_sha256": state["last_checkpoint_sha256"],
        "best_checkpoint": str(paths["best"]),
        "best_checkpoint_sha256": state["best_checkpoint_sha256"],
        "history": str(paths["history"]),
        "history_sha256": state["history_sha256"],
        "step_log": str(paths["steps"]),
        "step_log_sha256": state["step_log_sha256"],
        "session_elapsed_seconds": time.perf_counter() - session_start,
        "gates": {name: bool(value) for name, value in session_gates.items()},
        "optimizer_created": True,
        "parameters_updated": True,
        "final_variant_training_started": True,
        "validation_used_for_model_selection": True,
        "calibration_used": False,
        "locked_test_used": False,
        "next_required_stage": (
            "NEXT_PROTOCOL_VARIANT_OR_FINAL_COMPARISON_FREEZE"
            if run_complete else "CONTINUE_SAME_VARIANT_FROM_LAST_CHECKPOINT"
        ),
    }
    atomic_json(paths["evidence"], evidence)
    print("\nRun state:", paths["state"])
    print("Last checkpoint:", paths["last"])
    print("SHA-256:", state["last_checkpoint_sha256"])
    print("Best checkpoint:", paths["best"])
    print("SHA-256:", state["best_checkpoint_sha256"])
    print("Session evidence:", paths["evidence"])
    print("Final variant training started: YES")
    print("VALIDATION used              : YES — MODEL SELECTION ONLY")
    print("CALIBRATION used             : NO")
    print("Locked test used             : NO")
    print("Run complete                 :", run_complete)
    print("\n" + "=" * 96)
    print("STAGE 0G-4D-5F-3A SESSION STATUS: PASS")
    print("=" * 96)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 96, file=sys.stderr)
        print("STAGE 0G-4D-5F-3A SESSION STATUS: FAIL", file=sys.stderr)
        print("=" * 96, file=sys.stderr)
        raise
