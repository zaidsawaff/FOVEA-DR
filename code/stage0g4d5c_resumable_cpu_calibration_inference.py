from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# =============================================================================
# STAGE 0G-4D-5C — RESUMABLE CPU CALIBRATION INFERENCE + TEMPERATURE FIT
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
CALIBRATION_CACHE_ROOT = Path("/content/MMRDR_calibration_v1")

SPLIT_PATH = PROJECT_ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
CALIBRATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Audit/mmrdr_fovea_dr_CALIBRATION_local_cache_manifest_v1.csv"
)
CALIBRATION_CACHE_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5a_calibration_only_cache_preparation.json"
)
CALIBRATION_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_post_selection_calibration_protocol_v1.json"
)
CALIBRATION_PROTOCOL_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5b_calibration_protocol_lock.json"
)
FROZEN_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "Checkpoints/FOVEA_DR/baseline_seed2026/selected_epoch10_frozen.pt"
)
PRODUCTION_SOURCE_PATH = (
    PROJECT_ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
)

RUN_ROOT = PROJECT_ROOT / "Runs/FOVEA_DR/calibration_seed2026_cpu_fp32"
PART_ROOT = RUN_ROOT / "raw_inference_parts"
INFERENCE_STATE_PATH = RUN_ROOT / "inference_state.json"
RAW_PREDICTIONS_PATH = RUN_ROOT / "calibration_raw_predictions.csv"
CALIBRATED_PREDICTIONS_PATH = (
    RUN_ROOT / "calibration_temperature_scaled_predictions.csv"
)
TEMPERATURES_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_temperature_parameters_v1.json"
)
METRICS_PATH = (
    PROJECT_ROOT
    / "Audit/mmrdr_fovea_dr_CALIBRATION_metrics_before_after_temperature_v1.csv"
)
FROZEN_SOURCE_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/stage0g4d5c_resumable_cpu_calibration_inference.py"
)
LATEST_SESSION_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5c_cpu_calibration_latest_session.json"
)
FINAL_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5c_cpu_calibration_inference_and_temperature_fit.json"
)

EXPECTED_SPLIT_SHA = (
    "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140"
)
EXPECTED_CALIBRATION_MANIFEST_SHA = (
    "f8e6a62b96e330a1b21bb9ed56897cbf04b730987cf9e0c31141927abbb84655"
)
EXPECTED_CALIBRATION_PROTOCOL_SHA = (
    "f569f63c9b9c44816f19ce5f49f724fc5b1e4dc202d6d4a8694250f7b33995e7"
)
EXPECTED_FROZEN_CHECKPOINT_SHA = (
    "89f5e87b647680a9094c55bf647dc1dd2abb624681e733ffb453e30e4c11ff1f"
)
EXPECTED_PRODUCTION_SOURCE_SHA = (
    "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9"
)
EXPECTED_OFFICIAL_WEIGHT_SHA = (
    "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
)

EXPECTED_ROWS = 2840
EXPECTED_MODALITY_COUNTS = {"CFP": 1313, "UWF": 1171, "OCT": 356}
INPUT_SIZE = 512
RUN_SEED = 2026
PART_ROWS = 32
ECE_BINS = 15
PROBABILITY_EPSILON = 1e-12
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 20.0

LESION_COLUMNS = [
    "lesion_microaneurysm",
    "lesion_hard_exudate",
    "lesion_intraretinal_hemorrhage",
    "lesion_VB_IRMA",
    "lesion_neovascularization",
    "lesion_vitreous_hemorrhage",
    "lesion_retinal_detachment",
]

ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(path) as file:
        return json.load(file)


def atomic_json_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary, path)


def atomic_csv_save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def copy_once_verified(source, destination, expected_sha):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_sha:
            raise RuntimeError(
                f"Frozen source exists with a different hash: {destination}"
            )
        return "ALREADY_FROZEN_IDENTICAL"

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Source freeze verification failed.")
    os.replace(temporary, destination)
    return "COPIED_AND_FROZEN"


def all_evidence_gates_pass(payload):
    gates = payload.get("gates", {})
    return (
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and len(gates) > 0
        and all(bool(value) for value in gates.values())
    )


def import_frozen_core():
    spec = importlib.util.spec_from_file_location(
        "fovea_dr_frozen_training_core_calibration",
        PRODUCTION_SOURCE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import frozen production source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker_seed_factory(seed):
    def seed_worker(worker_id):
        value = seed + worker_id
        random.seed(value)
        np.random.seed(value % (2 ** 32))

    return seed_worker


def stable_signature(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CalibrationDataset(Dataset):
    def __init__(self, table):
        self.table = table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        path = Path(str(row["local_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["file_size_bytes"]):
            raise RuntimeError(f"CALIBRATION cache size mismatch: {path}")

        with Image.open(path) as image:
            raw = TF.to_tensor(image.convert("RGB"))

        raw = TF.resize(
            raw,
            [INPUT_SIZE, INPUT_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        quality = raw.std().clamp(0.0, 1.0)
        fov_mask = (
            raw.amax(dim=0, keepdim=True) > (8.0 / 255.0)
        ).to(raw.dtype)
        normalized = TF.normalize(
            raw,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        modality = str(row["modality"])
        is_fundus = modality in {"CFP", "UWF"}
        modality_index = 0 if modality == "CFP" else (1 if modality == "UWF" else 2)
        lesion_values = [
            0.0 if pd.isna(row[column]) else float(row[column])
            for column in LESION_COLUMNS
        ]

        return {
            "calibration_index": torch.tensor(
                int(row["calibration_index"]),
                dtype=torch.long,
            ),
            "image": normalized,
            "fov_mask": fov_mask,
            "quality": torch.tensor(float(quality), dtype=torch.float32),
            "fundus_mask": torch.tensor(is_fundus, dtype=torch.bool),
            "modality_index": torch.tensor(modality_index, dtype=torch.long),
            "grade": torch.tensor(int(row["grade"]), dtype=torch.long),
            "lesion_loss_mask": torch.tensor(
                float(row["lesion_loss_mask"]),
                dtype=torch.float32,
            ),
            "lesion_target": torch.tensor(lesion_values, dtype=torch.float32),
            "modality": modality,
            "relative_path": str(row["relative_path"]),
            "image_sha256": str(row["image_sha256"]),
        }


def ensure_official_weight():
    weight_enum = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    checkpoint_name = Path(weight_enum.url).name
    checkpoint_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    )

    if not checkpoint_path.exists():
        print("Official EfficientNetV2-S weight is not cached; downloading once...")
        temporary_model = efficientnet_v2_s(
            weights=weight_enum,
            progress=True,
        )
        del temporary_model

    exists = checkpoint_path.is_file()
    actual_sha = sha256_file(checkpoint_path) if exists else None
    verified = exists and actual_sha == EXPECTED_OFFICIAL_WEIGHT_SHA
    return checkpoint_path, actual_sha, verified


def scan_and_validate_parts(table, run_signature):
    PART_ROOT.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^part_(\d{4})_(\d{4})\.csv$")
    located = []

    for path in PART_ROOT.iterdir():
        match = pattern.match(path.name)
        if match:
            located.append((int(match.group(1)), int(match.group(2)), path))

    located.sort(key=lambda item: item[0])
    expected_start = 0
    records = []
    part_records = []

    for start, end, path in located:
        if start != expected_start or end < start:
            raise RuntimeError(
                f"Non-contiguous calibration inference part: {path.name}"
            )

        frame = pd.read_csv(path)
        expected_indices = list(range(start, end + 1))
        actual_indices = frame["calibration_index"].astype(int).tolist()
        if actual_indices != expected_indices:
            raise RuntimeError(f"Index mismatch in inference part: {path.name}")
        if not frame["run_signature"].astype(str).eq(run_signature).all():
            raise RuntimeError(f"Run-signature mismatch in: {path.name}")

        expected_paths = (
            table.iloc[start:end + 1]["relative_path"].astype(str).tolist()
        )
        expected_hashes = (
            table.iloc[start:end + 1]["image_sha256"].astype(str).tolist()
        )
        if frame["relative_path"].astype(str).tolist() != expected_paths:
            raise RuntimeError(f"Relative-path mismatch in: {path.name}")
        if frame["image_sha256"].astype(str).tolist() != expected_hashes:
            raise RuntimeError(f"Image-hash mismatch in: {path.name}")

        records.append(frame)
        part_records.append({
            "path": str(path),
            "start_index": start,
            "end_index": end,
            "rows": int(len(frame)),
            "sha256": sha256_file(path),
        })
        expected_start = end + 1

    return expected_start, records, part_records


def save_part(records, run_signature):
    frame = pd.DataFrame(records).sort_values("calibration_index").reset_index(drop=True)
    start = int(frame["calibration_index"].iloc[0])
    end = int(frame["calibration_index"].iloc[-1])
    expected = list(range(start, end + 1))
    if frame["calibration_index"].astype(int).tolist() != expected:
        raise RuntimeError("Attempted to save a non-contiguous inference part.")
    if not frame["run_signature"].astype(str).eq(run_signature).all():
        raise RuntimeError("Attempted to save a mixed-signature inference part.")

    path = PART_ROOT / f"part_{start:04d}_{end:04d}.csv"
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.equals(frame):
            raise RuntimeError(f"Existing inference part differs: {path}")
        return path

    atomic_csv_save(frame, path)
    return path


def write_inference_state(
    run_signature,
    source_sha,
    batch_size,
    completed_rows,
    part_records,
    status,
):
    payload = {
        "run": "FOVEA_DR_CALIBRATION_SEED2026_CPU_FP32",
        "status": status,
        "execution_backend": "CPU_FLOAT32",
        "batch_size": int(batch_size),
        "input_size": INPUT_SIZE,
        "run_signature": run_signature,
        "runner_source_sha256": source_sha,
        "checkpoint_sha256": EXPECTED_FROZEN_CHECKPOINT_SHA,
        "calibration_manifest_sha256": EXPECTED_CALIBRATION_MANIFEST_SHA,
        "calibration_protocol_sha256": EXPECTED_CALIBRATION_PROTOCOL_SHA,
        "completed_rows": int(completed_rows),
        "expected_rows": EXPECTED_ROWS,
        "parts": part_records,
        "locked_test_accessed": False,
    }
    atomic_json_save(payload, INFERENCE_STATE_PATH)


def softmax_numpy(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def sigmoid_numpy(values):
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def multiclass_nll(probabilities, targets):
    selected = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(selected, PROBABILITY_EPSILON, 1.0)).mean())


def binary_nll(probabilities, targets):
    probabilities = np.clip(
        probabilities,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return float(
        -(
            targets * np.log(probabilities)
            + (1.0 - targets) * np.log(1.0 - probabilities)
        ).mean()
    )


def confidence_ece(probabilities, targets, bins=ECE_BINS):
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == targets).astype(np.float64)
    total = len(targets)
    value = 0.0

    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)


def binary_ece(probabilities, targets, bins=ECE_BINS):
    total = len(targets)
    value = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == 0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities > lower) & (probabilities <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(
                float(targets[mask].mean()) - float(probabilities[mask].mean())
            )
    return float(value)


def fit_grade_temperature(probabilities, targets):
    log_probabilities = np.log(
        np.clip(probabilities.astype(np.float64), PROBABILITY_EPSILON, 1.0)
    )

    def objective(log_temperature):
        temperature = math.exp(float(log_temperature))
        calibrated = softmax_numpy(log_probabilities / temperature)
        return multiclass_nll(calibrated, targets)

    identity_nll = objective(0.0)
    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(math.log(TEMPERATURE_MIN), math.log(TEMPERATURE_MAX)),
        options={"xatol": 1e-6, "maxiter": 500},
    )

    optimized_temperature = math.exp(float(result.x))
    optimized_nll = float(result.fun)
    use_optimized = bool(
        result.success
        and math.isfinite(optimized_temperature)
        and TEMPERATURE_MIN <= optimized_temperature <= TEMPERATURE_MAX
        and optimized_nll < identity_nll - 1e-12
    )
    selected_temperature = optimized_temperature if use_optimized else 1.0
    calibrated = softmax_numpy(log_probabilities / selected_temperature)
    return selected_temperature, calibrated, {
        "optimizer_success": bool(result.success),
        "optimized_temperature": optimized_temperature,
        "identity_nll": identity_nll,
        "optimized_nll": optimized_nll,
        "identity_fallback_used": not use_optimized,
    }


def fit_lesion_temperature(logits, targets):
    logits = logits.astype(np.float64)

    def objective(log_temperature):
        temperature = math.exp(float(log_temperature))
        calibrated = sigmoid_numpy(logits / temperature)
        return binary_nll(calibrated, targets)

    identity_nll = objective(0.0)
    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(math.log(TEMPERATURE_MIN), math.log(TEMPERATURE_MAX)),
        options={"xatol": 1e-6, "maxiter": 500},
    )

    optimized_temperature = math.exp(float(result.x))
    optimized_nll = float(result.fun)
    use_optimized = bool(
        result.success
        and math.isfinite(optimized_temperature)
        and TEMPERATURE_MIN <= optimized_temperature <= TEMPERATURE_MAX
        and optimized_nll < identity_nll - 1e-12
    )
    selected_temperature = optimized_temperature if use_optimized else 1.0
    calibrated = sigmoid_numpy(logits / selected_temperature)
    return selected_temperature, calibrated, {
        "optimizer_success": bool(result.success),
        "optimized_temperature": optimized_temperature,
        "identity_nll": identity_nll,
        "optimized_nll": optimized_nll,
        "identity_fallback_used": not use_optimized,
    }


def grade_metrics(probabilities, targets):
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    return {
        "NLL": multiclass_nll(probabilities, targets),
        "Brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ECE_15": confidence_ece(probabilities, targets),
        "accuracy": float(accuracy_score(targets, prediction)),
        "macro_F1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "QWK": float(cohen_kappa_score(targets, prediction, weights="quadratic")),
    }


def lesion_metrics(probabilities, targets):
    prediction = (probabilities >= 0.5).astype(int)
    return {
        "NLL": binary_nll(probabilities, targets),
        "Brier_score": float(np.square(probabilities - targets).mean()),
        "ECE_15": binary_ece(probabilities, targets),
        "accuracy_at_0_5": float(accuracy_score(targets, prediction)),
        "F1_at_0_5": float(f1_score(targets, prediction, zero_division=0)),
    }


def all_probability_rows_valid(frame):
    valid = True
    for modality, classes in (("CFP", 5), ("UWF", 5), ("OCT", 3)):
        subset = frame[frame["modality"] == modality]
        columns = [f"calibrated_probability_{index}" for index in range(classes)]
        values = subset[columns].to_numpy(dtype=float)
        valid = valid and bool(
            np.isfinite(values).all()
            and ((values >= 0.0) & (values <= 1.0)).all()
            and np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)
        )

    fundus = frame[frame["modality"].isin(["CFP", "UWF"])]
    for lesion in LESION_COLUMNS:
        values = fundus[f"calibrated_{lesion}_probability"].to_numpy(dtype=float)
        valid = valid and bool(
            np.isfinite(values).all()
            and ((values >= 0.0) & (values <= 1.0)).all()
        )
    return bool(valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.session_rows < 0:
        raise ValueError("session-rows must be zero or positive; zero means all remaining rows.")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if args.workers < 0:
        raise ValueError("workers cannot be negative.")

    source_path = Path(__file__).resolve()
    source_sha = sha256_file(source_path)

    print("=" * 80)
    print("STAGE 0G-4D-5C — RESUMABLE CPU CALIBRATION INFERENCE")
    print("=" * 80)

    print("\n[1] FROZEN PREREQUISITE VERIFICATION")
    print("-" * 80)

    required = {
        "split": SPLIT_PATH,
        "CALIBRATION_manifest": CALIBRATION_MANIFEST_PATH,
        "CALIBRATION_cache_evidence": CALIBRATION_CACHE_EVIDENCE_PATH,
        "CALIBRATION_protocol": CALIBRATION_PROTOCOL_PATH,
        "CALIBRATION_protocol_evidence": CALIBRATION_PROTOCOL_EVIDENCE_PATH,
        "frozen_epoch10_checkpoint": FROZEN_CHECKPOINT_PATH,
        "production_source": PRODUCTION_SOURCE_PATH,
    }
    for name, path in required.items():
        print(f"{name:33s}: exists={path.exists()}")
    if not all(path.exists() for path in required.values()):
        raise FileNotFoundError("One or more calibration-inference prerequisites are missing.")

    hash_gates = {
        "split_hash_verified": sha256_file(SPLIT_PATH) == EXPECTED_SPLIT_SHA,
        "CALIBRATION_manifest_hash_verified": (
            sha256_file(CALIBRATION_MANIFEST_PATH) == EXPECTED_CALIBRATION_MANIFEST_SHA
        ),
        "CALIBRATION_protocol_hash_verified": (
            sha256_file(CALIBRATION_PROTOCOL_PATH) == EXPECTED_CALIBRATION_PROTOCOL_SHA
        ),
        "frozen_epoch10_checkpoint_hash_verified": (
            sha256_file(FROZEN_CHECKPOINT_PATH) == EXPECTED_FROZEN_CHECKPOINT_SHA
        ),
        "production_source_hash_verified": (
            sha256_file(PRODUCTION_SOURCE_PATH) == EXPECTED_PRODUCTION_SOURCE_SHA
        ),
    }

    cache_evidence = load_json(CALIBRATION_CACHE_EVIDENCE_PATH)
    protocol_evidence = load_json(CALIBRATION_PROTOCOL_EVIDENCE_PATH)
    protocol = load_json(CALIBRATION_PROTOCOL_PATH)
    evidence_gates = {
        "CALIBRATION_cache_evidence_passed": all_evidence_gates_pass(cache_evidence),
        "CALIBRATION_protocol_evidence_passed": all_evidence_gates_pass(protocol_evidence),
        "protocol_locked_before_inference": (
            protocol.get("status") == "LOCKED_BEFORE_CALIBRATION_INFERENCE"
        ),
        "protocol_prohibits_model_weight_update": (
            "MODEL_WEIGHT_UPDATE" in protocol.get("prohibited_actions", [])
        ),
        "protocol_prohibits_locked_test_access": (
            "LOCKED_TEST_ACCESS" in protocol.get("prohibited_actions", [])
        ),
    }

    for name, passed in {**hash_gates, **evidence_gates}.items():
        print(f"{name:51s}: {passed}")
    if not all({**hash_gates, **evidence_gates}.values()):
        raise RuntimeError("Frozen calibration-inference prerequisite failed.")

    print("\n[2] CALIBRATION TABLE AND LOCAL CACHE")
    print("-" * 80)

    split = pd.read_csv(SPLIT_PATH)
    calibration = split[
        split["development_split"] == "CALIBRATION"
    ].copy()
    calibration["relative_path"] = (
        calibration["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    manifest = pd.read_csv(CALIBRATION_MANIFEST_PATH)
    manifest["relative_path"] = (
        manifest["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    manifest_small = manifest[
        ["relative_path", "local_path", "file_size_bytes", "image_sha256"]
    ].copy()
    table = calibration.merge(
        manifest_small,
        on="relative_path",
        how="inner",
        validate="one_to_one",
    ).sort_values("relative_path").reset_index(drop=True)
    table["calibration_index"] = np.arange(len(table), dtype=int)

    modality_counts = {
        str(key): int(value)
        for key, value in table["modality"].value_counts().to_dict().items()
    }
    local_files_ready = bool(
        len(table) == EXPECTED_ROWS
        and all(
            Path(str(row.local_path)).is_file()
            and Path(str(row.local_path)).stat().st_size == int(row.file_size_bytes)
            for row in table.itertuples(index=False)
        )
    )

    print("Joined CALIBRATION rows:", len(table))
    print("Modality counts        :", modality_counts)
    print("Local cache complete   :", local_files_ready)
    if not (
        len(table) == EXPECTED_ROWS
        and modality_counts == EXPECTED_MODALITY_COUNTS
        and local_files_ready
        and table["relative_path"].is_unique
    ):
        raise RuntimeError("CALIBRATION table/cache readiness failed.")

    execution_contract = {
        "execution_backend": "CPU_FLOAT32",
        "batch_size": int(args.batch_size),
        "input_size": INPUT_SIZE,
        "checkpoint_sha256": EXPECTED_FROZEN_CHECKPOINT_SHA,
        "calibration_manifest_sha256": EXPECTED_CALIBRATION_MANIFEST_SHA,
        "calibration_protocol_sha256": EXPECTED_CALIBRATION_PROTOCOL_SHA,
        "production_source_sha256": EXPECTED_PRODUCTION_SOURCE_SHA,
        "runner_source_sha256": source_sha,
        "row_order": "SORTED_RELATIVE_PATH",
    }
    run_signature = stable_signature(execution_contract)

    source_action = copy_once_verified(
        source_path,
        FROZEN_SOURCE_PATH,
        source_sha,
    )
    if sha256_file(FROZEN_SOURCE_PATH) != source_sha:
        raise RuntimeError("Calibration inference source freeze failed.")

    next_index, _, existing_part_records = scan_and_validate_parts(
        table,
        run_signature,
    )
    if next_index > EXPECTED_ROWS:
        raise RuntimeError("Persisted inference rows exceed the calibration cohort.")

    print("Run signature         :", run_signature)
    print("Source action         :", source_action)
    print("Rows already complete :", next_index)
    print("Rows remaining        :", EXPECTED_ROWS - next_index)

    session_start_index = next_index
    if args.session_rows == 0:
        session_end_index = EXPECTED_ROWS
    else:
        session_end_index = min(
            EXPECTED_ROWS,
            session_start_index + args.session_rows,
        )

    if session_start_index < session_end_index:
        print("\n[3] OFFICIAL INITIALIZATION AND FROZEN MODEL LOAD")
        print("-" * 80)

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

        official_path, official_sha, official_ok = ensure_official_weight()
        print("Device                    : cpu")
        print("Precision                 : float32")
        print("CPU threads               :", cpu_threads)
        print("Batch size                :", args.batch_size)
        print("Official checkpoint       :", official_path)
        print("Official checkpoint SHA   :", official_sha)
        print("Official checkpoint verified:", official_ok)
        if not official_ok:
            raise RuntimeError("Official EfficientNetV2-S checkpoint is missing or changed.")

        core = import_frozen_core()
        model = core.FOVEADRTrainingModel()
        checkpoint = torch.load(
            FROZEN_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )
        incompatible = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        model_loaded_strictly = (
            len(incompatible.missing_keys) == 0
            and len(incompatible.unexpected_keys) == 0
            and int(checkpoint.get("epoch", -1)) == 10
        )
        parameters_finite = all(
            bool(torch.isfinite(parameter).all())
            for parameter in model.parameters()
        )
        print("Frozen epoch loaded       :", checkpoint.get("epoch"))
        print("Strict state load         :", model_loaded_strictly)
        print("All parameters finite     :", parameters_finite)
        if not model_loaded_strictly or not parameters_finite:
            raise RuntimeError("Frozen model load verification failed.")

        print("\n[4] RESUMABLE CPU FLOAT32 INFERENCE")
        print("-" * 80)
        print(
            f"This session will process indices "
            f"{session_start_index}..{session_end_index - 1}."
        )

        session_table = table.iloc[
            session_start_index:session_end_index
        ].copy().reset_index(drop=True)
        generator = torch.Generator().manual_seed(RUN_SEED)
        loader = DataLoader(
            CalibrationDataset(session_table),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            persistent_workers=args.workers > 0,
            worker_init_fn=worker_seed_factory(RUN_SEED),
            generator=generator,
        )

        buffer = []
        completed_this_session = 0
        inference_start = time.perf_counter()

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

                fundus_probability_by_position = {}
                fundus_expected_by_position = {}
                lesion_logit_by_position = {}
                lesion_probability_by_position = {}

                if len(fundus_positions) > 0:
                    probabilities = (
                        output["fundus"]["original"]["fusion"]["probabilities"]
                        .float()
                        .cpu()
                    )
                    expected = (
                        output["fundus"]["original"]["fusion"]["expected_grade"]
                        .float()
                        .cpu()
                    )
                    lesion_logits = (
                        output["fundus"]["lesions"]["presence_logits"]
                        .float()
                        .cpu()
                    )
                    lesion_probabilities = torch.sigmoid(lesion_logits)
                    for local, position in enumerate(fundus_positions.tolist()):
                        fundus_probability_by_position[position] = probabilities[local]
                        fundus_expected_by_position[position] = expected[local]
                        lesion_logit_by_position[position] = lesion_logits[local]
                        lesion_probability_by_position[position] = lesion_probabilities[local]

                oct_probability_by_position = {}
                oct_expected_by_position = {}
                if len(oct_positions) > 0:
                    probabilities = output["oct"]["probabilities"].float().cpu()
                    expected = output["oct"]["expected_grade"].float().cpu()
                    for local, position in enumerate(oct_positions.tolist()):
                        oct_probability_by_position[position] = probabilities[local]
                        oct_expected_by_position[position] = expected[local]

                for position in range(len(batch["relative_path"])):
                    modality = str(batch["modality"][position])
                    is_fundus = modality in {"CFP", "UWF"}
                    probability = (
                        fundus_probability_by_position[position]
                        if is_fundus
                        else oct_probability_by_position[position]
                    )
                    expected = (
                        fundus_expected_by_position[position]
                        if is_fundus
                        else oct_expected_by_position[position]
                    )
                    classes = 5 if is_fundus else 3
                    probability_values = probability.numpy().astype(float)
                    if not (
                        np.isfinite(probability_values).all()
                        and np.all(probability_values >= 0.0)
                        and np.all(probability_values <= 1.0)
                        and abs(float(probability_values.sum()) - 1.0) <= 2e-4
                    ):
                        raise FloatingPointError("Invalid raw grade probability distribution.")

                    record = {
                        "run_signature": run_signature,
                        "calibration_index": int(batch["calibration_index"][position]),
                        "relative_path": str(batch["relative_path"][position]),
                        "image_sha256": str(batch["image_sha256"][position]),
                        "modality": modality,
                        "target": int(batch["grade"][position]),
                        "raw_prediction": int(probability.argmax()),
                        "raw_expected_grade": float(expected),
                        "lesion_loss_mask": float(batch["lesion_loss_mask"][position]),
                    }
                    for class_index in range(5):
                        record[f"raw_probability_{class_index}"] = (
                            float(probability[class_index])
                            if class_index < classes
                            else math.nan
                        )

                    for lesion_index, lesion in enumerate(LESION_COLUMNS):
                        record[f"{lesion}_target"] = (
                            float(batch["lesion_target"][position, lesion_index])
                            if is_fundus
                            else math.nan
                        )
                        record[f"raw_{lesion}_logit"] = (
                            float(lesion_logit_by_position[position][lesion_index])
                            if is_fundus
                            else math.nan
                        )
                        record[f"raw_{lesion}_probability"] = (
                            float(lesion_probability_by_position[position][lesion_index])
                            if is_fundus
                            else math.nan
                        )
                    buffer.append(record)
                    completed_this_session += 1

                    if len(buffer) == PART_ROWS:
                        save_part(buffer, run_signature)
                        buffer = []

                elapsed = time.perf_counter() - inference_start
                throughput = completed_this_session / elapsed
                total_complete = session_start_index + completed_this_session
                remaining = EXPECTED_ROWS - total_complete
                eta_minutes = remaining / max(throughput, 1e-12) / 60.0
                if batch_number == 1 or batch_number % 10 == 0 or total_complete == session_end_index:
                    print(
                        f"Batch {batch_number:04d} | "
                        f"session_rows={completed_this_session:4d} | "
                        f"total={total_complete:4d}/{EXPECTED_ROWS} | "
                        f"throughput={throughput:.3f} img/s | "
                        f"full-run ETA={eta_minutes:.1f} min"
                    )

        if buffer:
            save_part(buffer, run_signature)

        del loader, model, checkpoint, core
        session_seconds = time.perf_counter() - inference_start
    else:
        official_path = None
        official_sha = None
        official_ok = True
        model_loaded_strictly = True
        parameters_finite = True
        completed_this_session = 0
        session_seconds = 0.0
        print("\n[3] MODEL INFERENCE")
        print("-" * 80)
        print("All calibration rows were already persisted; model load skipped.")

    next_index, existing_frames, part_records = scan_and_validate_parts(
        table,
        run_signature,
    )
    run_complete = next_index == EXPECTED_ROWS
    status = "INFERENCE_COMPLETE" if run_complete else "RESUMABLE_PARTIAL"
    write_inference_state(
        run_signature,
        source_sha,
        args.batch_size,
        next_index,
        part_records,
        status,
    )

    session_gates = {
        **hash_gates,
        **evidence_gates,
        "exact_CALIBRATION_table_loaded": len(table) == EXPECTED_ROWS,
        "CALIBRATION_modality_counts_exact": modality_counts == EXPECTED_MODALITY_COUNTS,
        "local_CALIBRATION_cache_complete": local_files_ready,
        "CPU_float32_execution_locked": True,
        "runner_source_frozen_and_verified": sha256_file(FROZEN_SOURCE_PATH) == source_sha,
        "official_weight_verified_if_model_loaded": official_ok,
        "selected_epoch10_model_loaded_strictly_if_required": model_loaded_strictly,
        "all_model_parameters_finite_if_loaded": parameters_finite,
        "at_least_one_row_completed_or_run_already_complete": (
            completed_this_session > 0 or run_complete
        ),
        "persisted_parts_contiguous_and_verified": next_index >= session_end_index,
        "completed_rows_do_not_exceed_2840": next_index <= EXPECTED_ROWS,
        "model_parameters_not_updated": True,
        "VALIDATION_not_used_for_calibration_fit": True,
        "locked_test_not_accessed": True,
    }

    print("\n" + "=" * 80)
    print("CALIBRATION INFERENCE SESSION GATES")
    print("=" * 80)
    for name, passed in session_gates.items():
        print(f"{name:68s}: {passed}")
    if not all(session_gates.values()):
        raise RuntimeError("Calibration inference session gate failed.")

    if not run_complete:
        throughput = (
            completed_this_session / session_seconds
            if session_seconds > 0 and completed_this_session > 0
            else None
        )
        evidence = {
            "stage": "0G-4D-5C",
            "status": "PASS",
            "session_status": "RESUMABLE_PARTIAL",
            "execution_backend": "CPU_FLOAT32",
            "run_signature": run_signature,
            "completed_before_session": session_start_index,
            "completed_this_session": completed_this_session,
            "completed_total": next_index,
            "expected_total": EXPECTED_ROWS,
            "session_seconds": session_seconds,
            "session_throughput_images_per_second": throughput,
            "inference_state": str(INFERENCE_STATE_PATH),
            "inference_state_sha256": sha256_file(INFERENCE_STATE_PATH),
            "runner_source": str(FROZEN_SOURCE_PATH),
            "runner_source_sha256": source_sha,
            "gates": {name: bool(value) for name, value in session_gates.items()},
            "model_loaded": completed_this_session > 0,
            "optimizer_created": False,
            "model_parameters_updated": False,
            "temperature_fitting_started": False,
            "validation_used_for_calibration_fit": False,
            "locked_test_accessed": False,
            "next_required_stage": "CONTINUE_SAME_CPU_FP32_CALIBRATION_INFERENCE",
        }
        atomic_json_save(evidence, LATEST_SESSION_EVIDENCE_PATH)

        print("\nInference state:")
        print(INFERENCE_STATE_PATH)
        print("Completed rows:", next_index, "/", EXPECTED_ROWS)
        print("Session throughput img/s:", throughput)
        print("Temperature fitting started: NO")
        print("Model parameters updated    : NO")
        print("Locked test accessed        : NO")
        print("\nNEXT REQUIRED STAGE:")
        print("Continue the same CPU-float32 runner from its persisted next row.")
        print("\n" + "=" * 80)
        print("STAGE 0G-4D-5C SESSION STATUS: PASS — RESUMABLE PARTIAL")
        print("=" * 80)
        return

    print("\n[5] CONSOLIDATE COMPLETE RAW INFERENCE")
    print("-" * 80)

    raw = pd.concat(existing_frames, ignore_index=True)
    raw = raw.sort_values("calibration_index").reset_index(drop=True)
    raw_complete = (
        len(raw) == EXPECTED_ROWS
        and raw["calibration_index"].astype(int).tolist() == list(range(EXPECTED_ROWS))
        and raw["relative_path"].astype(str).tolist()
        == table["relative_path"].astype(str).tolist()
        and raw["image_sha256"].astype(str).tolist()
        == table["image_sha256"].astype(str).tolist()
        and raw["relative_path"].is_unique
    )
    if not raw_complete:
        raise RuntimeError("Complete raw calibration inference consolidation failed.")

    atomic_csv_save(raw, RAW_PREDICTIONS_PATH)
    raw_sha = sha256_file(RAW_PREDICTIONS_PATH)
    print("Raw prediction rows:", len(raw))
    print("Raw predictions SHA:", raw_sha)

    print("\n[6] LOCKED TEMPERATURE FIT")
    print("-" * 80)

    calibrated = raw.copy()
    temperature_parameters = {
        "protocol": "FOVEA_DR_POST_SELECTION_TEMPERATURE_PARAMETERS_V1",
        "status": "FITTED_ON_CALIBRATION_ONLY",
        "execution_backend": "CPU_FLOAT32",
        "run_signature": run_signature,
        "selected_checkpoint_sha256": EXPECTED_FROZEN_CHECKPOINT_SHA,
        "calibration_protocol_sha256": EXPECTED_CALIBRATION_PROTOCOL_SHA,
        "calibration_manifest_sha256": EXPECTED_CALIBRATION_MANIFEST_SHA,
        "raw_predictions_sha256": raw_sha,
        "grade_temperatures": {},
        "lesion_temperatures": {},
    }
    metric_records = []
    nll_nonworsening = True
    grade_argmax_invariant = True
    lesion_threshold_invariant = True

    for modality, classes in (("CFP", 5), ("UWF", 5), ("OCT", 3)):
        mask = calibrated["modality"].eq(modality).to_numpy()
        indices = np.where(mask)[0]
        probability_columns = [f"raw_probability_{index}" for index in range(classes)]
        probabilities = calibrated.loc[mask, probability_columns].to_numpy(dtype=float)
        targets = calibrated.loc[mask, "target"].to_numpy(dtype=int)

        temperature, adjusted, fit_info = fit_grade_temperature(probabilities, targets)
        before = grade_metrics(probabilities, targets)
        after = grade_metrics(adjusted, targets)

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

        invariant = bool(
            np.array_equal(probabilities.argmax(axis=1), adjusted.argmax(axis=1))
        )
        grade_argmax_invariant = grade_argmax_invariant and invariant
        nonworsening = after["NLL"] <= before["NLL"] + 1e-12
        nll_nonworsening = nll_nonworsening and nonworsening

        temperature_parameters["grade_temperatures"][modality] = {
            "temperature": float(temperature),
            **fit_info,
            "rows": int(len(targets)),
        }
        metric_records.append({
            "task_type": "GRADE",
            "unit": modality,
            "rows": int(len(targets)),
            "temperature": float(temperature),
            "NLL_before": before["NLL"],
            "NLL_after": after["NLL"],
            "Brier_before": before["Brier_score"],
            "Brier_after": after["Brier_score"],
            "ECE15_before": before["ECE_15"],
            "ECE15_after": after["ECE_15"],
            "accuracy_before": before["accuracy"],
            "accuracy_after": after["accuracy"],
            "macro_F1_before": before["macro_F1"],
            "macro_F1_after": after["macro_F1"],
            "QWK_before": before["QWK"],
            "QWK_after": after["QWK"],
            "decision_invariant": invariant,
            "NLL_nonworsening": nonworsening,
        })
        print(
            f"{modality:3s} grade | T={temperature:.6f} | "
            f"NLL {before['NLL']:.6f} -> {after['NLL']:.6f} | "
            f"ECE {before['ECE_15']:.6f} -> {after['ECE_15']:.6f}"
        )

    fundus_mask = calibrated["modality"].isin(["CFP", "UWF"]).to_numpy()
    valid_mask = fundus_mask & (calibrated["lesion_loss_mask"].to_numpy(dtype=float) > 0.5)

    for lesion in LESION_COLUMNS:
        logits = calibrated.loc[valid_mask, f"raw_{lesion}_logit"].to_numpy(dtype=float)
        raw_probabilities = calibrated.loc[
            valid_mask, f"raw_{lesion}_probability"
        ].to_numpy(dtype=float)
        targets = calibrated.loc[valid_mask, f"{lesion}_target"].to_numpy(dtype=int)

        temperature, adjusted, fit_info = fit_lesion_temperature(logits, targets)
        before = lesion_metrics(raw_probabilities, targets)
        after = lesion_metrics(adjusted, targets)

        column = f"calibrated_{lesion}_probability"
        calibrated[column] = math.nan
        calibrated.loc[valid_mask, column] = adjusted

        invariant = bool(
            np.array_equal(
                raw_probabilities >= 0.5,
                adjusted >= 0.5,
            )
        )
        lesion_threshold_invariant = lesion_threshold_invariant and invariant
        nonworsening = after["NLL"] <= before["NLL"] + 1e-12
        nll_nonworsening = nll_nonworsening and nonworsening

        temperature_parameters["lesion_temperatures"][lesion] = {
            "temperature": float(temperature),
            **fit_info,
            "valid_rows": int(len(targets)),
            "positive_rows": int(targets.sum()),
        }
        metric_records.append({
            "task_type": "LESION",
            "unit": lesion,
            "rows": int(len(targets)),
            "temperature": float(temperature),
            "NLL_before": before["NLL"],
            "NLL_after": after["NLL"],
            "Brier_before": before["Brier_score"],
            "Brier_after": after["Brier_score"],
            "ECE15_before": before["ECE_15"],
            "ECE15_after": after["ECE_15"],
            "accuracy_before": before["accuracy_at_0_5"],
            "accuracy_after": after["accuracy_at_0_5"],
            "macro_F1_before": before["F1_at_0_5"],
            "macro_F1_after": after["F1_at_0_5"],
            "QWK_before": math.nan,
            "QWK_after": math.nan,
            "decision_invariant": invariant,
            "NLL_nonworsening": nonworsening,
        })
        print(
            f"{lesion:36s} | T={temperature:.6f} | "
            f"NLL {before['NLL']:.6f} -> {after['NLL']:.6f} | "
            f"ECE {before['ECE_15']:.6f} -> {after['ECE_15']:.6f}"
        )

    temperatures = [
        value["temperature"]
        for value in temperature_parameters["grade_temperatures"].values()
    ] + [
        value["temperature"]
        for value in temperature_parameters["lesion_temperatures"].values()
    ]
    temperatures_valid = all(
        math.isfinite(value)
        and TEMPERATURE_MIN <= value <= TEMPERATURE_MAX
        for value in temperatures
    )
    probabilities_valid = all_probability_rows_valid(calibrated)

    calibrated["calibrated_prediction"] = calibrated[
        "calibrated_prediction"
    ].astype(int)
    metrics = pd.DataFrame(metric_records)

    final_fit_gates = {
        "raw_inference_has_exactly_2840_rows": raw_complete,
        "three_grade_temperatures_fitted": (
            len(temperature_parameters["grade_temperatures"]) == 3
        ),
        "seven_lesion_temperatures_fitted": (
            len(temperature_parameters["lesion_temperatures"]) == 7
        ),
        "all_temperatures_finite_and_within_locked_bounds": temperatures_valid,
        "all_unit_NLL_values_nonworsening": nll_nonworsening,
        "grade_argmax_predictions_unchanged": grade_argmax_invariant,
        "lesion_predictions_at_0_5_unchanged": lesion_threshold_invariant,
        "all_calibrated_probabilities_valid": probabilities_valid,
        "model_parameters_not_updated": True,
        "VALIDATION_not_used_for_calibration_fit": True,
        "locked_test_not_accessed": True,
    }

    print("\n" + "=" * 80)
    print("LOCKED TEMPERATURE-FIT GATES")
    print("=" * 80)
    for name, passed in final_fit_gates.items():
        print(f"{name:68s}: {passed}")
    if not all(final_fit_gates.values()):
        raise RuntimeError("Locked calibration temperature-fit gate failed.")

    atomic_csv_save(calibrated, CALIBRATED_PREDICTIONS_PATH)
    atomic_csv_save(metrics, METRICS_PATH)
    temperature_parameters["calibrated_predictions"] = str(
        CALIBRATED_PREDICTIONS_PATH
    )
    temperature_parameters["calibrated_predictions_sha256"] = sha256_file(
        CALIBRATED_PREDICTIONS_PATH
    )
    temperature_parameters["metrics"] = str(METRICS_PATH)
    temperature_parameters["metrics_sha256"] = sha256_file(METRICS_PATH)
    temperature_parameters["locked_test_accessed"] = False
    atomic_json_save(temperature_parameters, TEMPERATURES_PATH)

    write_inference_state(
        run_signature,
        source_sha,
        args.batch_size,
        EXPECTED_ROWS,
        part_records,
        "COMPLETE_AND_TEMPERATURE_FITTED",
    )

    final_gates = {**session_gates, **final_fit_gates}
    final_evidence = {
        "stage": "0G-4D-5C",
        "status": "PASS",
        "session_status": "COMPLETE_AND_TEMPERATURE_FITTED",
        "execution_backend": "CPU_FLOAT32",
        "run_signature": run_signature,
        "completed_rows": EXPECTED_ROWS,
        "raw_predictions": str(RAW_PREDICTIONS_PATH),
        "raw_predictions_sha256": raw_sha,
        "calibrated_predictions": str(CALIBRATED_PREDICTIONS_PATH),
        "calibrated_predictions_sha256": sha256_file(CALIBRATED_PREDICTIONS_PATH),
        "temperature_parameters": str(TEMPERATURES_PATH),
        "temperature_parameters_sha256": sha256_file(TEMPERATURES_PATH),
        "metrics": str(METRICS_PATH),
        "metrics_sha256": sha256_file(METRICS_PATH),
        "runner_source": str(FROZEN_SOURCE_PATH),
        "runner_source_sha256": source_sha,
        "inference_state": str(INFERENCE_STATE_PATH),
        "inference_state_sha256": sha256_file(INFERENCE_STATE_PATH),
        "gates": {name: bool(value) for name, value in final_gates.items()},
        "calibration_images_accessed": True,
        "model_loaded": completed_this_session > 0,
        "optimizer_created": False,
        "model_parameters_updated": False,
        "validation_used_for_calibration_fit": False,
        "locked_test_accessed": False,
        "next_required_stage": "CALIBRATED_MODEL_PACKAGE_FREEZE_BEFORE_LOCKED_TEST",
    }
    atomic_json_save(final_evidence, LATEST_SESSION_EVIDENCE_PATH)
    atomic_json_save(final_evidence, FINAL_EVIDENCE_PATH)

    print("\nRaw predictions:")
    print(RAW_PREDICTIONS_PATH)
    print("SHA-256:", raw_sha)
    print("\nCalibrated predictions:")
    print(CALIBRATED_PREDICTIONS_PATH)
    print("SHA-256:", sha256_file(CALIBRATED_PREDICTIONS_PATH))
    print("\nTemperature parameters:")
    print(TEMPERATURES_PATH)
    print("SHA-256:", sha256_file(TEMPERATURES_PATH))
    print("\nMetrics:")
    print(METRICS_PATH)
    print("\nModel parameters updated    : NO")
    print("VALIDATION used for fitting : NO")
    print("Locked test accessed        : NO")
    print("\nNEXT REQUIRED STAGE:")
    print("Freeze the calibrated deployment package before locked-test access.")
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-5C STATUS: PASS — COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
