from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)


# =============================================================================
# STAGE 0G-4D-4B — RESUMABLE BASELINE TRAINING, SEED 2026
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
TRAIN_CACHE_ROOT = Path("/content/MMRDR_training_v1")
VALIDATION_CACHE_ROOT = Path("/content/MMRDR_validation_v1")

SPLIT_PATH = PROJECT_ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
CANDIDATE_PATH = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
PROTOCOL_PATH = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_training_protocol_v1.json"
PRODUCTION_SOURCE_PATH = PROJECT_ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
TRAINING_CONTRACT_PATH = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_training_implementation_contract_v1.json"
TRAIN_MANIFEST_PATH = PROJECT_ROOT / "Audit/mmrdr_fovea_dr_TRAIN_local_cache_manifest_v1.csv"
VALIDATION_MANIFEST_PATH = PROJECT_ROOT / "Audit/mmrdr_fovea_dr_VALIDATION_local_cache_manifest_v1.csv"
VALIDATION_EVIDENCE_PATH = PROJECT_ROOT / "Evidence/stage0g4d4a_r2_validation_only_cache_preparation.json"

RUN_ROOT = PROJECT_ROOT / "Runs/FOVEA_DR/baseline_seed2026"
CHECKPOINT_ROOT = PROJECT_ROOT / "Checkpoints/FOVEA_DR/baseline_seed2026"
LAST_CHECKPOINT_PATH = CHECKPOINT_ROOT / "last_resume.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_ROOT / "best_model.pt"
HISTORY_PATH = RUN_ROOT / "epoch_history.csv"
STEP_LOG_PATH = RUN_ROOT / "optimizer_step_log.csv"
LATEST_PREDICTIONS_PATH = RUN_ROOT / "latest_validation_predictions.csv"
BEST_PREDICTIONS_PATH = RUN_ROOT / "best_validation_predictions.csv"
SESSION_EVIDENCE_PATH = PROJECT_ROOT / "Evidence/stage0g4d4b_baseline_seed2026_latest_session.json"
RUN_STATE_PATH = RUN_ROOT / "run_state.json"
FROZEN_RUNNER_PATH = PROJECT_ROOT / "Code/FOVEA_DR/v1_1/stage0g4d4b_resumable_baseline_seed2026.py"

EXPECTED_SPLIT_SHA = "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140"
EXPECTED_CANDIDATE_SHA = "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181"
EXPECTED_PROTOCOL_SHA = "5b16e0c8565b86d3cac603f416ee0b7f03055ab9a0eb9ac9e3fb007632329b40"
EXPECTED_PRODUCTION_SOURCE_SHA = "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9"
EXPECTED_TRAINING_CONTRACT_SHA = "7de76cc722220e601a73125874dd9dee979527cfce847d441933ca8bfcb29ea4"
EXPECTED_TRAIN_MANIFEST_SHA = "32f9a3b6b17916b78f591007a85ce2e1910c9c4d2b1294af1c391b19f476bd15"
EXPECTED_VALIDATION_MANIFEST_SHA = "d94c5613b6d030dd6e99246b1b72f77746b5223139a4597bef733eb38a5786af"
EXPECTED_OFFICIAL_WEIGHT_SHA = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"

RUN_SEED = 2026
MAX_EPOCHS = 30
OPTIMIZER_STEPS_PER_EPOCH = 415
GRADIENT_ACCUMULATION = 2
EFFECTIVE_BATCH = 32
EARLY_STOPPING_PATIENCE = 6
PRIMARY_MIN_DELTA = 0.002
INPUT_SIZE = 512
VALIDATION_BATCH_SIZE = 16

EXPECTED_TRAIN_ROWS = 13251
EXPECTED_VALIDATION_ROWS = 2839
EXPECTED_CALIBRATION_ROWS = 2840

ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary, path)


def freeze_runner_after_success(source_path, source_sha):
    FROZEN_RUNNER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_RUNNER_PATH.exists():
        existing_sha = sha256_file(FROZEN_RUNNER_PATH)
        if existing_sha != source_sha:
            raise RuntimeError(
                "A different baseline runner is already frozen: "
                f"{existing_sha}"
            )
        return "ALREADY_FROZEN_IDENTICAL"
    shutil.copy2(source_path, FROZEN_RUNNER_PATH)
    if sha256_file(FROZEN_RUNNER_PATH) != source_sha:
        raise RuntimeError("Baseline runner freeze verification failed.")
    return "COPIED_AND_FROZEN_AFTER_SUCCESSFUL_EPOCH"


def all_evidence_gates_pass(evidence):
    gates = evidence.get("gates", {})
    return (
        evidence.get("status") == "PASS"
        and isinstance(gates, dict)
        and len(gates) > 0
        and all(bool(value) for value in gates.values())
    )


def import_frozen_core():
    spec = importlib.util.spec_from_file_location(
        "fovea_dr_frozen_training_core",
        PRODUCTION_SOURCE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen production source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidationDataset(Dataset):
    def __init__(self, table, lesion_columns):
        self.table = table.reset_index(drop=True)
        self.lesion_columns = list(lesion_columns)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
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


def build_validation_table():
    split = pd.read_csv(SPLIT_PATH)
    validation = split[
        split["development_split"] == "VALIDATION"
    ].copy()
    validation["relative_path"] = (
        validation["relative_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )
    manifest = pd.read_csv(VALIDATION_MANIFEST_PATH)
    manifest["relative_path"] = (
        manifest["relative_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )
    manifest_small = manifest[
        ["relative_path", "local_path", "file_size_bytes", "image_sha256"]
    ].copy()
    table = validation.merge(
        manifest_small,
        on="relative_path",
        how="inner",
        validate="one_to_one",
    )
    return table.sort_values("relative_path").reset_index(drop=True)


def worker_seed_factory(seed):
    def seed_worker(worker_id):
        value = seed + worker_id
        random.seed(value)
        np.random.seed(value % (2 ** 32))
        torch.manual_seed(value)
    return seed_worker


def classification_metrics(frame, modality, classes):
    subset = frame[frame["modality"] == modality].copy()
    target = subset["target"].to_numpy(dtype=int)
    prediction = subset["prediction"].to_numpy(dtype=int)
    expected = subset["expected_grade"].to_numpy(dtype=float)
    probabilities = subset[
        [f"probability_{index}" for index in range(classes)]
    ].to_numpy(dtype=float)
    nll = float(
        -np.log(
            np.clip(probabilities[np.arange(len(target)), target], 1e-7, 1.0)
        ).mean()
    )
    return {
        "rows": int(len(subset)),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_F1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(target, prediction, weights="quadratic")),
        "NLL": nll,
        "expected_grade_MAE": float(mean_absolute_error(target, expected)),
    }


@torch.inference_mode()
def evaluate_validation(model, loader, device, lesion_columns):
    model.eval()
    records = []
    lesion_targets = []
    lesion_probabilities = []
    lesion_masks = []

    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model.forward_mixed(
                batch["image"],
                batch["fundus_mask"],
                batch["fov_mask"],
                batch["modality_index"],
                batch["quality"],
            )

        fundus_positions = torch.where(batch["fundus_mask"])[0]
        oct_positions = torch.where(~batch["fundus_mask"])[0]

        if len(fundus_positions) > 0:
            probabilities = output["fundus"]["original"]["fusion"]["probabilities"].float().cpu()
            expected = output["fundus"]["original"]["fusion"]["expected_grade"].float().cpu()
            lesions = output["fundus"]["lesions"]["presence"].float().cpu()
            targets = batch["lesion_target"][fundus_positions].float().cpu()
            masks = batch["lesion_loss_mask"][fundus_positions].float().cpu()
            lesion_probabilities.append(lesions)
            lesion_targets.append(targets)
            lesion_masks.append(masks)

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

    lesion_summary = {}
    if lesion_probabilities:
        targets = torch.cat(lesion_targets).numpy()
        probabilities = torch.cat(lesion_probabilities).numpy()
        masks = torch.cat(lesion_masks).numpy() > 0.5
        valid_targets = targets[masks]
        valid_probabilities = probabilities[masks]
        for index, column in enumerate(lesion_columns):
            binary_target = valid_targets[:, index]
            binary_prediction = (valid_probabilities[:, index] >= 0.5).astype(int)
            lesion_summary[column] = {
                "valid_rows": int(len(binary_target)),
                "positive_rows": int(binary_target.sum()),
                "accuracy_at_0_5": float(accuracy_score(binary_target, binary_prediction)),
                "F1_at_0_5": float(f1_score(binary_target, binary_prediction, zero_division=0)),
            }

    return frame, metrics, lesion_summary


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_qwk,
    best_nll,
    best_epoch,
    bad_epochs,
    source_sha,
):
    return {
        "format": "FOVEA_DR_BASELINE_RESUME_V1",
        "run_seed": RUN_SEED,
        "epoch_completed": epoch,
        "global_optimizer_step": global_step,
        "best_mean_fundus_QWK": best_qwk,
        "best_mean_fundus_NLL": best_nll,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "runner_source_sha256": source_sha,
        "production_source_sha256": EXPECTED_PRODUCTION_SOURCE_SHA,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.session_epochs < 1:
        raise ValueError("session-epochs must be at least 1.")

    runner_source = Path(__file__).resolve()
    runner_source_sha = sha256_file(runner_source)

    print("=" * 80)
    print("STAGE 0G-4D-4B — RESUMABLE BASELINE TRAINING, SEED 2026")
    print("=" * 80)

    print("\n[1] FROZEN ARTIFACT AND CACHE VERIFICATION")
    print("-" * 80)
    required_hashes = {
        SPLIT_PATH: EXPECTED_SPLIT_SHA,
        CANDIDATE_PATH: EXPECTED_CANDIDATE_SHA,
        PROTOCOL_PATH: EXPECTED_PROTOCOL_SHA,
        PRODUCTION_SOURCE_PATH: EXPECTED_PRODUCTION_SOURCE_SHA,
        TRAINING_CONTRACT_PATH: EXPECTED_TRAINING_CONTRACT_SHA,
        TRAIN_MANIFEST_PATH: EXPECTED_TRAIN_MANIFEST_SHA,
        VALIDATION_MANIFEST_PATH: EXPECTED_VALIDATION_MANIFEST_SHA,
    }
    artifact_hash_gates = {}
    for path, expected_hash in required_hashes.items():
        exists = path.exists()
        verified = exists and sha256_file(path) == expected_hash
        artifact_hash_gates[path.name] = verified
        print(f"{path.name:65s}: exists={exists}, hash_verified={verified}")

    with open(VALIDATION_EVIDENCE_PATH) as file:
        validation_evidence = json.load(file)
    validation_evidence_pass = all_evidence_gates_pass(validation_evidence)

    if not all(artifact_hash_gates.values()) or not validation_evidence_pass:
        raise RuntimeError("Frozen artifact or validation evidence gate failed.")
    if not TRAIN_CACHE_ROOT.exists() or not VALIDATION_CACHE_ROOT.exists():
        raise RuntimeError("Required temporary TRAIN/VALIDATION caches are missing.")

    split = pd.read_csv(SPLIT_PATH)
    role_counts = split["development_split"].value_counts().to_dict()
    three_way_counts_ok = role_counts == {
        "TRAIN": EXPECTED_TRAIN_ROWS,
        "CALIBRATION": EXPECTED_CALIBRATION_ROWS,
        "VALIDATION": EXPECTED_VALIDATION_ROWS,
    }
    print("Validation evidence PASS:", validation_evidence_pass)
    print("Three-way counts exact  :", three_way_counts_ok, role_counts)
    print("Runner source SHA-256   :", runner_source_sha)
    if not three_way_counts_ok:
        raise RuntimeError("Three-way development split changed.")

    print("\n[2] GPU AND OFFICIAL INITIALIZATION")
    print("-" * 80)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    core = import_frozen_core()
    weight_enum = core.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weight_enum.url).name
    official_weight_ok = checkpoint_path.exists() and sha256_file(checkpoint_path) == EXPECTED_OFFICIAL_WEIGHT_SHA
    if not official_weight_ok:
        raise RuntimeError("Official EfficientNetV2-S checkpoint is missing or changed.")

    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    torch.cuda.manual_seed_all(RUN_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    train_table, sampler_weight_column = core.build_training_table()
    validation_table = build_validation_table()
    train_table_ok = len(train_table) == EXPECTED_TRAIN_ROWS
    validation_table_ok = len(validation_table) == EXPECTED_VALIDATION_ROWS
    if not train_table_ok or not validation_table_ok:
        raise RuntimeError("TRAIN or VALIDATION table cardinality failed.")

    model = core.FOVEADRTrainingModel().to(device)
    model.train()
    optimizer, optimizer_group_counts = core.build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=core.lr_multiplier,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)

    lesion_weight_table = pd.read_csv(core.LESION_WEIGHT_PATH)
    lesion_pos_weight = torch.tensor(
        lesion_weight_table["sqrt_pos_weight"].to_numpy(dtype=np.float32),
        device=device,
    )
    with open(CANDIDATE_PATH) as file:
        loss_weights = json.load(file)["candidate_loss"]

    print("GPU                     :", gpu_name)
    print("TRAIN rows              :", len(train_table))
    print("VALIDATION rows         :", len(validation_table))
    print("Sampler weight column   :", sampler_weight_column)
    print("Optimizer groups        :", optimizer_group_counts)
    print("Official weights verified:", official_weight_ok)
    print("Preflight checkpoint used: NO")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_step = 0
    best_qwk = -math.inf
    best_nll = math.inf
    best_epoch = 0
    bad_epochs = 0
    resumed = False

    if LAST_CHECKPOINT_PATH.exists():
        print("\n[3] RESUME CHECKPOINT")
        print("-" * 80)
        resume = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        if resume.get("format") != "FOVEA_DR_BASELINE_RESUME_V1":
            raise RuntimeError("Resume checkpoint format mismatch.")
        if resume.get("run_seed") != RUN_SEED:
            raise RuntimeError("Resume checkpoint seed mismatch.")
        if resume.get("runner_source_sha256") != runner_source_sha:
            raise RuntimeError("Runner source changed since checkpoint creation.")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        random.setstate(resume["python_random_state"])
        np.random.set_state(resume["numpy_random_state"])
        torch.set_rng_state(resume["torch_random_state"])
        torch.cuda.set_rng_state_all(resume["cuda_random_state"])
        start_epoch = int(resume["epoch_completed"]) + 1
        global_step = int(resume["global_optimizer_step"])
        best_qwk = float(resume["best_mean_fundus_QWK"])
        best_nll = float(resume["best_mean_fundus_NLL"])
        best_epoch = int(resume["best_epoch"])
        bad_epochs = int(resume["bad_epochs"])
        resumed = True
        print("Resumed after epoch:", start_epoch - 1)
        print("Global optimizer step:", global_step)
        print("Best epoch          :", best_epoch)
        print("Best mean fundus QWK:", best_qwk)
    else:
        print("\n[3] FRESH BASELINE INITIALIZATION")
        print("-" * 80)
        print("Starting from official ImageNet1K V1 weights.")
        print("Pilot checkpoint is deliberately excluded.")

    if start_epoch > MAX_EPOCHS or bad_epochs >= EARLY_STOPPING_PATIENCE:
        print("Baseline run is already complete or early-stopped.")
        return

    validation_dataset = ValidationDataset(validation_table, core.LESION_COLUMNS)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed_factory(RUN_SEED + 900000),
    )

    history_records = pd.read_csv(HISTORY_PATH).to_dict("records") if HISTORY_PATH.exists() else []
    step_records = pd.read_csv(STEP_LOG_PATH).to_dict("records") if STEP_LOG_PATH.exists() else []
    session_start = time.perf_counter()
    epochs_completed_this_session = 0
    session_all_losses_finite = True
    session_all_gradients_finite = True
    session_no_amp_overflow = True
    stopped_early = False
    runner_freeze_action = None

    last_epoch_allowed = min(MAX_EPOCHS, start_epoch + args.session_epochs - 1)
    for epoch in range(start_epoch, last_epoch_allowed + 1):
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch}/{MAX_EPOCHS}")
        print("=" * 80)
        epoch_start = time.perf_counter()
        core.SEED = RUN_SEED + epoch * 1000
        batch_sampler = core.LockedWeightedBatchSampler(
            train_table,
            RUN_SEED + epoch,
            OPTIMIZER_STEPS_PER_EPOCH,
        )
        train_dataset = core.FOVEADRTrainDataset(train_table)
        train_loader = DataLoader(
            train_dataset,
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
        epoch_losses = []
        torch.cuda.reset_peak_memory_stats()

        for step_in_epoch in range(1, OPTIMIZER_STEPS_PER_EPOCH + 1):
            accumulated_loss = 0.0
            step_start = time.perf_counter()

            for _ in range(GRADIENT_ACCUMULATION):
                batch = core.move_batch(next(iterator), device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    output = model.forward_mixed(
                        batch["image"],
                        batch["fundus_mask"],
                        batch["fov_mask"],
                        batch["modality_index"],
                        batch["quality"],
                    )
                    total_loss, _ = core.compute_mixed_losses(
                        output,
                        batch,
                        lesion_pos_weight,
                        loss_weights,
                    )
                    scaled_loss = total_loss / GRADIENT_ACCUMULATION

                finite_loss = bool(torch.isfinite(total_loss).all())
                session_all_losses_finite = session_all_losses_finite and finite_loss
                if not finite_loss:
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step_in_epoch}")
                scaler.scale(scaled_loss).backward()
                accumulated_loss += float(scaled_loss.detach().cpu())

            scaler.unscale_(optimizer)
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            gradients_finite = len(gradients) > 0 and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            session_all_gradients_finite = session_all_gradients_finite and gradients_finite
            if not gradients_finite:
                raise FloatingPointError(f"Non-finite gradients at epoch={epoch}, step={step_in_epoch}")

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                core.GRAD_CLIP_NORM,
            )
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if scale_after < scale_before:
                session_no_amp_overflow = False
                raise FloatingPointError(f"AMP overflow at epoch={epoch}, step={step_in_epoch}")
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            epoch_losses.append(accumulated_loss)

            step_records.append({
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_optimizer_step": global_step,
                "loss": accumulated_loss,
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "grad_scaler_scale": scale_after,
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "new_module_lr": float(optimizer.param_groups[2]["lr"]),
                "step_seconds": time.perf_counter() - step_start,
            })

            if step_in_epoch == 1 or step_in_epoch % 50 == 0 or step_in_epoch == OPTIMIZER_STEPS_PER_EPOCH:
                print(
                    f"Train {step_in_epoch:03d}/{OPTIMIZER_STEPS_PER_EPOCH} | "
                    f"loss={accumulated_loss:.6f} | "
                    f"grad={float(gradient_norm):.4f} | "
                    f"scale={scale_after:.0f}"
                )

        del iterator, train_loader
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_start

        print("\nVALIDATION EVALUATION")
        validation_start = time.perf_counter()
        predictions, metrics, lesion_metrics = evaluate_validation(
            model,
            validation_loader,
            device,
            core.LESION_COLUMNS,
        )
        validation_seconds = time.perf_counter() - validation_start
        if not metrics["all_primary_metrics_finite"]:
            raise FloatingPointError("Non-finite primary validation metric.")

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
            "peak_reserved_VRAM_GB": torch.cuda.max_memory_reserved() / (1024 ** 3),
        }
        history_records.append(epoch_record)
        pd.DataFrame(history_records).to_csv(HISTORY_PATH, index=False)
        pd.DataFrame(step_records).to_csv(STEP_LOG_PATH, index=False)
        predictions.to_csv(LATEST_PREDICTIONS_PATH, index=False)

        runner_freeze_action = freeze_runner_after_success(
            runner_source,
            runner_source_sha,
        )

        resume_payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            best_qwk,
            best_nll,
            best_epoch,
            bad_epochs,
            runner_source_sha,
        )
        atomic_torch_save(resume_payload, LAST_CHECKPOINT_PATH)

        if is_best:
            best_payload = {
                "format": "FOVEA_DR_BASELINE_BEST_MODEL_V1",
                "run_seed": RUN_SEED,
                "epoch": epoch,
                "mean_fundus_QWK": current_qwk,
                "mean_fundus_NLL": current_nll,
                "validation_metrics": metrics,
                "lesion_metrics": lesion_metrics,
                "runner_source_sha256": runner_source_sha,
                "production_source_sha256": EXPECTED_PRODUCTION_SOURCE_SHA,
                "model_state_dict": model.state_dict(),
            }
            atomic_torch_save(best_payload, BEST_CHECKPOINT_PATH)
            predictions.to_csv(BEST_PREDICTIONS_PATH, index=False)

        print("Mean train loss :", f"{epoch_record['mean_train_loss']:.6f}")
        print("CFP QWK         :", f"{metrics['CFP']['quadratic_weighted_kappa']:.6f}")
        print("UWF QWK         :", f"{metrics['UWF']['quadratic_weighted_kappa']:.6f}")
        print("Mean fundus QWK :", f"{current_qwk:.6f}")
        print("Mean fundus NLL :", f"{current_nll:.6f}")
        print("OCT QWK         :", f"{metrics['OCT']['quadratic_weighted_kappa']:.6f}")
        print("OCT macro-F1    :", f"{metrics['OCT']['macro_F1']:.6f}")
        print("Best epoch      :", best_epoch)
        print("Bad epochs      :", bad_epochs, "/", EARLY_STOPPING_PATIENCE)
        print("Last checkpoint :", LAST_CHECKPOINT_PATH)

        epochs_completed_this_session += 1
        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print("EARLY STOPPING TRIGGERED")
            break

    final_epoch = int(history_records[-1]["epoch"])
    run_complete = stopped_early or final_epoch >= MAX_EPOCHS
    last_checkpoint_sha = sha256_file(LAST_CHECKPOINT_PATH)
    best_checkpoint_sha = sha256_file(BEST_CHECKPOINT_PATH)
    history_sha = sha256_file(HISTORY_PATH)
    step_log_sha = sha256_file(STEP_LOG_PATH)

    gates = {
        "all_frozen_artifact_hashes_verified": all(artifact_hash_gates.values()),
        "validation_cache_evidence_passed": validation_evidence_pass,
        "three_way_split_counts_exact": three_way_counts_ok,
        "TRAIN_cache_loaded_exactly": train_table_ok,
        "VALIDATION_cache_loaded_exactly": validation_table_ok,
        "CALIBRATION_not_loaded": True,
        "locked_test_not_loaded": True,
        "official_ImageNet_weight_SHA_verified": official_weight_ok,
        "pilot_checkpoint_not_used_for_initialization": True,
        "at_least_one_epoch_completed_this_session": epochs_completed_this_session >= 1,
        "all_session_losses_finite": session_all_losses_finite,
        "all_session_gradients_finite": session_all_gradients_finite,
        "no_AMP_overflow": session_no_amp_overflow,
        "global_optimizer_step_count_correct": global_step == final_epoch * OPTIMIZER_STEPS_PER_EPOCH,
        "primary_validation_metrics_finite": math.isfinite(best_qwk) and math.isfinite(best_nll),
        "resume_checkpoint_created": LAST_CHECKPOINT_PATH.exists(),
        "best_checkpoint_created": BEST_CHECKPOINT_PATH.exists(),
        "history_created": HISTORY_PATH.exists(),
        "validation_predictions_created": LATEST_PREDICTIONS_PATH.exists(),
        "baseline_runner_frozen_and_hash_verified": (
            FROZEN_RUNNER_PATH.exists()
            and sha256_file(FROZEN_RUNNER_PATH) == runner_source_sha
        ),
    }

    print("\n" + "=" * 80)
    print("RESUMABLE BASELINE SESSION GATES")
    print("=" * 80)
    for name, passed in gates.items():
        print(f"{name:62s}: {passed}")
    if not all(gates.values()):
        raise RuntimeError("STAGE 0G-4D-4B session gate failed.")

    run_state = {
        "run": "FOVEA_DR_BASELINE_SEED2026",
        "status": "COMPLETE" if run_complete else "RESUMABLE_CHECKPOINT_READY",
        "last_completed_epoch": final_epoch,
        "global_optimizer_step": global_step,
        "best_epoch": best_epoch,
        "best_mean_fundus_QWK": best_qwk,
        "best_mean_fundus_NLL": best_nll,
        "bad_epochs": bad_epochs,
        "early_stopped": stopped_early,
        "runner_source_sha256": runner_source_sha,
        "last_checkpoint_sha256": last_checkpoint_sha,
        "best_checkpoint_sha256": best_checkpoint_sha,
        "history_sha256": history_sha,
        "step_log_sha256": step_log_sha,
    }
    atomic_json_save(run_state, RUN_STATE_PATH)

    evidence = {
        "stage": "0G-4D-4B",
        "status": "PASS",
        "session_type": "RESUME" if resumed else "FRESH_LAUNCH",
        "session_epochs_requested": args.session_epochs,
        "epochs_completed_this_session": epochs_completed_this_session,
        "last_completed_epoch": final_epoch,
        "global_optimizer_step": global_step,
        "best_epoch": best_epoch,
        "best_mean_fundus_QWK": best_qwk,
        "best_mean_fundus_NLL": best_nll,
        "run_complete": run_complete,
        "early_stopped": stopped_early,
        "runner_source": str(runner_source),
        "runner_source_sha256": runner_source_sha,
        "frozen_runner": str(FROZEN_RUNNER_PATH),
        "runner_freeze_action": runner_freeze_action,
        "last_checkpoint": str(LAST_CHECKPOINT_PATH),
        "last_checkpoint_sha256": last_checkpoint_sha,
        "best_checkpoint": str(BEST_CHECKPOINT_PATH),
        "best_checkpoint_sha256": best_checkpoint_sha,
        "history": str(HISTORY_PATH),
        "history_sha256": history_sha,
        "step_log": str(STEP_LOG_PATH),
        "step_log_sha256": step_log_sha,
        "session_elapsed_seconds": time.perf_counter() - session_start,
        "gates": {name: bool(value) for name, value in gates.items()},
        "optimizer_created": True,
        "parameters_updated": True,
        "final_baseline_training_started": True,
        "validation_used_for_model_selection": True,
        "calibration_used": False,
        "locked_test_used": False,
        "next_required_stage": (
            "BASELINE_SEED2026_COMPLETE_REVIEW"
            if run_complete
            else "CONTINUE_RESUMABLE_BASELINE_SEED2026"
        ),
    }
    atomic_json_save(evidence, SESSION_EVIDENCE_PATH)

    print("\nRun state:")
    print(RUN_STATE_PATH)
    print("\nLast resume checkpoint:")
    print(LAST_CHECKPOINT_PATH)
    print("SHA-256:", last_checkpoint_sha)
    print("\nBest checkpoint:")
    print(BEST_CHECKPOINT_PATH)
    print("SHA-256:", best_checkpoint_sha)
    print("\nHistory:")
    print(HISTORY_PATH)
    print("\nSession evidence:")
    print(SESSION_EVIDENCE_PATH)
    print("\nFinal baseline training started: YES")
    print("Validation used               : YES — MODEL SELECTION ONLY")
    print("Calibration used              : NO")
    print("Locked test used              : NO")
    print("Run complete                  :", run_complete)
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-4B SESSION STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
