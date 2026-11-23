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
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# =============================================================================
# STAGE 0G-4D-5E-2B — RESUMABLE LABEL-BLIND CPU LOCKED-TEST INFERENCE
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
TEST_CACHE_ROOT = Path("/content/MMRDR_locked_test_v1")

TEST_MEMBERSHIP_PATH = (
    PROJECT_ROOT / "Splits/mmrdr_locked_test_membership_v1_FROZEN.csv"
)
TEST_CACHE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Audit/mmrdr_fovea_dr_LOCKED_TEST_local_cache_manifest_v1.csv"
)
TEST_CACHE_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5e2a_locked_test_label_free_cache_preparation.json"
)
EVALUATION_CONTRACT_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_locked_test_single_evaluation_contract_v1.json"
)
EVALUATION_CONTRACT_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5e1b_single_evaluation_contract_freeze.json"
)
DEPLOYMENT_PACKAGE_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_calibrated_deployment_package_v1.json"
)
FROZEN_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "Checkpoints/FOVEA_DR/baseline_seed2026/selected_epoch10_frozen.pt"
)
PRODUCTION_SOURCE_PATH = (
    PROJECT_ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
)
TEMPERATURES_PATH = (
    PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_temperature_parameters_v1.json"
)
CALIBRATED_TRANSFORM_SOURCE_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/fovea_dr_calibrated_probability_transform_v1.py"
)

RUN_ROOT = PROJECT_ROOT / "Runs/FOVEA_DR/locked_test_seed2026_cpu_fp32"
PART_ROOT = RUN_ROOT / "raw_inference_parts"
INFERENCE_STATE_PATH = RUN_ROOT / "inference_state.json"
RAW_PREDICTIONS_PATH = RUN_ROOT / "locked_test_raw_predictions_v1.csv"
CALIBRATED_PREDICTIONS_PATH = (
    RUN_ROOT / "locked_test_calibrated_predictions_v1.csv"
)
FROZEN_SOURCE_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/stage0g4d5e2b_resumable_cpu_locked_test_inference.py"
)
LATEST_SESSION_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5e2b_locked_test_inference_latest_session.json"
)
FINAL_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5e2b_locked_test_predictions_frozen.json"
)

EXPECTED_TEST_MEMBERSHIP_SHA = (
    "56913e5067d371f8631ef8deb56db585bb21f8591f97a88bdbedc07b29e4f45b"
)
EXPECTED_TEST_CACHE_MANIFEST_SHA = (
    "89dc85d5491681bbee4bc9fb934cf4500db20295afd49f2771427b5781747699"
)
EXPECTED_EVALUATION_CONTRACT_SHA = (
    "6244056ff1a588c91d7ce96573fd295f0be85473bc626b8528ef59d684cd0bd4"
)
EXPECTED_DEPLOYMENT_PACKAGE_SHA = (
    "bf79997cd9afe922e957b904ee074b7fe5bf9308fd081562ee424e9e1f174867"
)
EXPECTED_FROZEN_CHECKPOINT_SHA = (
    "89f5e87b647680a9094c55bf647dc1dd2abb624681e733ffb453e30e4c11ff1f"
)
EXPECTED_PRODUCTION_SOURCE_SHA = (
    "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9"
)
EXPECTED_TEMPERATURES_SHA = (
    "b694a32f84126516b0c6dc1ef0e32e130ffe6e6919433bf751399e21d10894c8"
)
EXPECTED_CALIBRATED_TRANSFORM_SHA = (
    "4318244fc21c2fbd3dd59f4998f4ad662d17d80d2183119b0acbf409d050bd61"
)
EXPECTED_OFFICIAL_WEIGHT_SHA = (
    "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
)

EXPECTED_ROWS = 5380
EXPECTED_MODALITY_COUNTS = {"CFP": 2221, "OCT": 562, "UWF": 2597}
INPUT_SIZE = 512
RUN_SEED = 2026
PART_ROWS = 32
LOCKED_BATCH_SIZE = 4
LOCKED_WORKERS = 2

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


def create_csv_once_verified(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    expected_sha = sha256_file(temporary)
    if path.exists():
        actual_sha = sha256_file(path)
        temporary.unlink(missing_ok=True)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Frozen prediction file differs: {path}")
        return "ALREADY_FROZEN_IDENTICAL", actual_sha
    os.replace(temporary, path)
    return "CREATED_AND_FROZEN", expected_sha


def copy_once_verified(source, destination, expected_sha):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_sha:
            raise RuntimeError(
                f"Frozen runner exists with a different hash: {destination}"
            )
        return "ALREADY_FROZEN_IDENTICAL"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Runner source freeze verification failed.")
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


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import frozen source: {path}")
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


class LockedTestDataset(Dataset):
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
            raise RuntimeError(f"Locked-test cache size mismatch: {path}")
        if sha256_file(path) != str(row["image_sha256"]):
            raise RuntimeError(f"Locked-test cache hash mismatch: {path}")

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

        return {
            "locked_test_index": torch.tensor(
                int(row["locked_test_index"]), dtype=torch.long
            ),
            "image": normalized,
            "fov_mask": fov_mask,
            "quality": torch.tensor(float(quality), dtype=torch.float32),
            "fundus_mask": torch.tensor(is_fundus, dtype=torch.bool),
            "modality_index": torch.tensor(modality_index, dtype=torch.long),
            "modality": modality,
            "relative_path": str(row["relative_path"]),
            "image_sha256": str(row["image_sha256"]),
        }


def ensure_official_weight():
    weight_enum = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    checkpoint_name = Path(weight_enum.url).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        print("Official EfficientNetV2-S weight is not cached; downloading once...")
        temporary_model = efficientnet_v2_s(weights=weight_enum, progress=True)
        del temporary_model
    exists = checkpoint_path.is_file()
    actual_sha = sha256_file(checkpoint_path) if exists else None
    return checkpoint_path, actual_sha, exists and actual_sha == EXPECTED_OFFICIAL_WEIGHT_SHA


def scan_and_validate_parts(table, run_signature):
    PART_ROOT.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^part_(\d{5})_(\d{5})\.csv$")
    located = []
    for path in PART_ROOT.iterdir():
        match = pattern.match(path.name)
        if match:
            located.append((int(match.group(1)), int(match.group(2)), path))
    located.sort(key=lambda item: item[0])

    expected_start = 0
    frames = []
    part_records = []
    for start, end, path in located:
        if start != expected_start or end < start or end >= EXPECTED_ROWS:
            raise RuntimeError(f"Non-contiguous locked-test part: {path.name}")
        frame = pd.read_csv(path)
        expected_indices = list(range(start, end + 1))
        if frame["locked_test_index"].astype(int).tolist() != expected_indices:
            raise RuntimeError(f"Index mismatch in part: {path.name}")
        if not frame["run_signature"].astype(str).eq(run_signature).all():
            raise RuntimeError(f"Run-signature mismatch in part: {path.name}")
        expected_paths = table.iloc[start:end + 1]["relative_path"].astype(str).tolist()
        expected_hashes = table.iloc[start:end + 1]["image_sha256"].astype(str).tolist()
        if frame["relative_path"].astype(str).tolist() != expected_paths:
            raise RuntimeError(f"Relative-path mismatch in part: {path.name}")
        if frame["image_sha256"].astype(str).tolist() != expected_hashes:
            raise RuntimeError(f"Image-hash mismatch in part: {path.name}")
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if not np.isfinite(numeric[~np.isnan(numeric)]).all():
            raise FloatingPointError(f"Non-finite numeric value in part: {path.name}")
        frames.append(frame)
        part_records.append({
            "path": str(path),
            "start_index": start,
            "end_index": end,
            "rows": int(len(frame)),
            "sha256": sha256_file(path),
        })
        expected_start = end + 1
    return expected_start, frames, part_records


def save_part(records, run_signature):
    frame = pd.DataFrame(records).sort_values("locked_test_index").reset_index(drop=True)
    start = int(frame["locked_test_index"].iloc[0])
    end = int(frame["locked_test_index"].iloc[-1])
    if frame["locked_test_index"].astype(int).tolist() != list(range(start, end + 1)):
        raise RuntimeError("Attempted to save a non-contiguous prediction part.")
    if not frame["run_signature"].astype(str).eq(run_signature).all():
        raise RuntimeError("Attempted to save a mixed-signature prediction part.")
    path = PART_ROOT / f"part_{start:05d}_{end:05d}.csv"
    if path.exists():
        existing = pd.read_csv(path)
        if existing.shape != frame.shape or existing.columns.tolist() != frame.columns.tolist():
            raise RuntimeError(f"Existing prediction part differs: {path}")
        return path
    atomic_csv_save(frame, path)
    return path


def write_inference_state(
    run_signature,
    source_sha,
    completed_rows,
    part_records,
    status,
    raw_sha=None,
    calibrated_sha=None,
):
    payload = {
        "run": "FOVEA_DR_LOCKED_TEST_SEED2026_CPU_FP32",
        "status": status,
        "execution_backend": "CPU_FLOAT32",
        "batch_size": LOCKED_BATCH_SIZE,
        "workers": LOCKED_WORKERS,
        "input_size": INPUT_SIZE,
        "run_signature": run_signature,
        "runner_source_sha256": source_sha,
        "checkpoint_sha256": EXPECTED_FROZEN_CHECKPOINT_SHA,
        "test_membership_sha256": EXPECTED_TEST_MEMBERSHIP_SHA,
        "test_cache_manifest_sha256": EXPECTED_TEST_CACHE_MANIFEST_SHA,
        "evaluation_contract_sha256": EXPECTED_EVALUATION_CONTRACT_SHA,
        "completed_rows": int(completed_rows),
        "expected_rows": EXPECTED_ROWS,
        "parts": part_records,
        "raw_predictions_sha256": raw_sha,
        "calibrated_predictions_sha256": calibrated_sha,
        "test_images_accessed": completed_rows > 0,
        "test_labels_loaded": False,
        "scientific_metrics_computed": False,
    }
    atomic_json_save(payload, INFERENCE_STATE_PATH)


def validate_grade_probabilities(values):
    return bool(
        np.isfinite(values).all()
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and abs(float(values.sum()) - 1.0) <= 2e-4
    )


def all_calibrated_probabilities_valid(frame):
    valid = True
    for modality, classes in (("CFP", 5), ("UWF", 5), ("OCT", 3)):
        subset = frame[frame["modality"] == modality]
        columns = [f"calibrated_probability_{index}" for index in range(classes)]
        values = subset[columns].to_numpy(dtype=float)
        valid = valid and bool(
            np.isfinite(values).all()
            and np.all(values >= 0.0)
            and np.all(values <= 1.0)
            and np.allclose(values.sum(axis=1), 1.0, atol=2e-8, rtol=0.0)
        )
    fundus = frame[frame["modality"].isin(["CFP", "UWF"])]
    for lesion in LESION_COLUMNS:
        values = fundus[f"calibrated_{lesion}_probability"].to_numpy(dtype=float)
        valid = valid and bool(
            np.isfinite(values).all()
            and np.all(values >= 0.0)
            and np.all(values <= 1.0)
        )
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=LOCKED_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=LOCKED_WORKERS)
    args = parser.parse_args()

    if args.session_rows < 0:
        raise ValueError("session-rows must be nonnegative; zero means all remaining rows.")
    if args.batch_size != LOCKED_BATCH_SIZE or args.workers != LOCKED_WORKERS:
        raise RuntimeError("Batch size and workers must match the frozen contract: 4 and 2.")

    source_path = Path(__file__).resolve()
    source_sha = sha256_file(source_path)

    print("=" * 80)
    print("STAGE 0G-4D-5E-2B — RESUMABLE LABEL-BLIND CPU LOCKED-TEST INFERENCE")
    print("=" * 80)

    print("\n[1] FROZEN PREREQUISITE VERIFICATION")
    print("-" * 80)
    required = {
        "test_membership": TEST_MEMBERSHIP_PATH,
        "test_cache_manifest": TEST_CACHE_MANIFEST_PATH,
        "test_cache_evidence": TEST_CACHE_EVIDENCE_PATH,
        "evaluation_contract": EVALUATION_CONTRACT_PATH,
        "evaluation_contract_evidence": EVALUATION_CONTRACT_EVIDENCE_PATH,
        "deployment_package": DEPLOYMENT_PACKAGE_PATH,
        "frozen_epoch10_checkpoint": FROZEN_CHECKPOINT_PATH,
        "production_source": PRODUCTION_SOURCE_PATH,
        "temperature_parameters": TEMPERATURES_PATH,
        "calibrated_transform": CALIBRATED_TRANSFORM_SOURCE_PATH,
    }
    for name, path in required.items():
        print(f"{name:33s}: exists={path.exists()}")
    if not all(path.exists() for path in required.values()):
        raise FileNotFoundError("One or more locked-test inference prerequisites are missing.")

    hash_gates = {
        "test_membership_hash_verified": sha256_file(TEST_MEMBERSHIP_PATH) == EXPECTED_TEST_MEMBERSHIP_SHA,
        "test_cache_manifest_hash_verified": sha256_file(TEST_CACHE_MANIFEST_PATH) == EXPECTED_TEST_CACHE_MANIFEST_SHA,
        "evaluation_contract_hash_verified": sha256_file(EVALUATION_CONTRACT_PATH) == EXPECTED_EVALUATION_CONTRACT_SHA,
        "deployment_package_hash_verified": sha256_file(DEPLOYMENT_PACKAGE_PATH) == EXPECTED_DEPLOYMENT_PACKAGE_SHA,
        "frozen_epoch10_checkpoint_hash_verified": sha256_file(FROZEN_CHECKPOINT_PATH) == EXPECTED_FROZEN_CHECKPOINT_SHA,
        "production_source_hash_verified": sha256_file(PRODUCTION_SOURCE_PATH) == EXPECTED_PRODUCTION_SOURCE_SHA,
        "temperature_parameters_hash_verified": sha256_file(TEMPERATURES_PATH) == EXPECTED_TEMPERATURES_SHA,
        "calibrated_transform_hash_verified": sha256_file(CALIBRATED_TRANSFORM_SOURCE_PATH) == EXPECTED_CALIBRATED_TRANSFORM_SHA,
    }

    cache_evidence = load_json(TEST_CACHE_EVIDENCE_PATH)
    contract_evidence = load_json(EVALUATION_CONTRACT_EVIDENCE_PATH)
    contract = load_json(EVALUATION_CONTRACT_PATH)
    deployment = load_json(DEPLOYMENT_PACKAGE_PATH)
    temperatures = load_json(TEMPERATURES_PATH)
    evidence_gates = {
        "test_cache_evidence_passed": all_evidence_gates_pass(cache_evidence),
        "evaluation_contract_evidence_passed": all_evidence_gates_pass(contract_evidence),
        "contract_frozen_before_test": contract.get("status") == "FROZEN_BEFORE_TEST_IMAGE_OR_LABEL_ACCESS",
        "single_evaluation_locked": contract.get("single_evaluation_policy", {}).get("number_of_scientific_test_evaluations") == 1,
        "CPU_float32_locked": contract.get("inference_contract", {}).get("backend") == "CPU" and contract.get("inference_contract", {}).get("precision") == "FLOAT32",
        "prediction_before_label_sequence_locked": contract.get("inference_contract", {}).get("predictions_frozen_before_label_access") is True,
        "deployment_frozen": deployment.get("status") == "FROZEN_BEFORE_LOCKED_TEST",
        "temperatures_fitted_on_calibration_only": temperatures.get("status") == "FITTED_ON_CALIBRATION_ONLY" and temperatures.get("locked_test_accessed") is False,
    }
    for name, passed in {**hash_gates, **evidence_gates}.items():
        print(f"{name:58s}: {passed}")
    if not all({**hash_gates, **evidence_gates}.values()):
        raise RuntimeError("Frozen locked-test inference prerequisite failed.")

    print("\n[2] LABEL-FREE TEST TABLE AND LOCAL CACHE")
    print("-" * 80)
    membership = pd.read_csv(TEST_MEMBERSHIP_PATH)
    cache = pd.read_csv(TEST_CACHE_MANIFEST_PATH)
    label_tokens = ("grade", "label", "lesion", "diagnosis", "target")
    membership_label_free = not any(
        any(token in column.lower() for token in label_tokens)
        for column in membership.columns
    )
    cache_label_free = not any(
        any(token in column.lower() for token in label_tokens)
        for column in cache.columns
    )
    cache_small = cache[[
        "locked_test_index", "relative_path", "local_path",
        "file_size_bytes", "image_sha256", "modality",
    ]].copy()
    table = membership[["locked_test_index", "relative_path", "modality"]].merge(
        cache_small,
        on=["locked_test_index", "relative_path", "modality"],
        how="inner",
        validate="one_to_one",
    ).sort_values("locked_test_index").reset_index(drop=True)
    modality_counts = {
        str(key): int(value)
        for key, value in table["modality"].value_counts().to_dict().items()
    }
    indices_exact = table["locked_test_index"].astype(int).tolist() == list(range(EXPECTED_ROWS))
    local_files_ready = bool(
        len(table) == EXPECTED_ROWS
        and all(
            Path(str(row.local_path)).is_file()
            and Path(str(row.local_path)).stat().st_size == int(row.file_size_bytes)
            for row in table.itertuples(index=False)
        )
    )
    print("Joined locked-test rows:", len(table))
    print("Modality counts        :", modality_counts)
    print("Indices contiguous     :", indices_exact)
    print("Membership label-free  :", membership_label_free)
    print("Cache manifest label-free:", cache_label_free)
    print("Local cache complete   :", local_files_ready)
    if not (
        len(table) == EXPECTED_ROWS
        and modality_counts == EXPECTED_MODALITY_COUNTS
        and indices_exact
        and membership_label_free
        and cache_label_free
        and local_files_ready
        and table["relative_path"].is_unique
    ):
        raise RuntimeError("Locked-test table/cache readiness failed.")

    execution_contract = {
        "execution_backend": "CPU_FLOAT32",
        "batch_size": LOCKED_BATCH_SIZE,
        "workers": LOCKED_WORKERS,
        "input_size": INPUT_SIZE,
        "checkpoint_sha256": EXPECTED_FROZEN_CHECKPOINT_SHA,
        "test_membership_sha256": EXPECTED_TEST_MEMBERSHIP_SHA,
        "test_cache_manifest_sha256": EXPECTED_TEST_CACHE_MANIFEST_SHA,
        "evaluation_contract_sha256": EXPECTED_EVALUATION_CONTRACT_SHA,
        "temperature_parameters_sha256": EXPECTED_TEMPERATURES_SHA,
        "production_source_sha256": EXPECTED_PRODUCTION_SOURCE_SHA,
        "runner_source_sha256": source_sha,
        "row_order": "LOCKED_TEST_INDEX_ASCENDING",
        "labels_loaded": False,
    }
    run_signature = stable_signature(execution_contract)
    source_action = copy_once_verified(source_path, FROZEN_SOURCE_PATH, source_sha)
    if sha256_file(FROZEN_SOURCE_PATH) != source_sha:
        raise RuntimeError("Locked-test runner source freeze failed.")

    next_index, _, existing_part_records = scan_and_validate_parts(table, run_signature)
    if next_index > EXPECTED_ROWS:
        raise RuntimeError("Persisted prediction rows exceed the locked test.")
    print("Run signature         :", run_signature)
    print("Runner source SHA     :", source_sha)
    print("Source action         :", source_action)
    print("Rows already complete :", next_index)
    print("Rows remaining        :", EXPECTED_ROWS - next_index)

    session_start_index = next_index
    session_end_index = (
        EXPECTED_ROWS
        if args.session_rows == 0
        else min(EXPECTED_ROWS, session_start_index + args.session_rows)
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
        print("Batch size                :", LOCKED_BATCH_SIZE)
        print("Official checkpoint       :", official_path)
        print("Official checkpoint SHA   :", official_sha)
        print("Official checkpoint verified:", official_ok)
        if not official_ok:
            raise RuntimeError("Official EfficientNetV2-S checkpoint is missing or changed.")

        core = import_module(PRODUCTION_SOURCE_PATH, "fovea_dr_frozen_core_test")
        model = core.FOVEADRTrainingModel()
        checkpoint = torch.load(FROZEN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model_loaded_strictly = (
            len(incompatible.missing_keys) == 0
            and len(incompatible.unexpected_keys) == 0
            and int(checkpoint.get("epoch", -1)) == 10
        )
        parameters_finite = all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        )
        print("Frozen epoch loaded       :", checkpoint.get("epoch"))
        print("Strict state load         :", model_loaded_strictly)
        print("All parameters finite     :", parameters_finite)
        if not model_loaded_strictly or not parameters_finite:
            raise RuntimeError("Frozen model load verification failed.")

        print("\n[4] RESUMABLE LABEL-BLIND CPU FLOAT32 INFERENCE")
        print("-" * 80)
        print(f"This session will process indices {session_start_index}..{session_end_index - 1}.")
        session_table = table.iloc[session_start_index:session_end_index].copy().reset_index(drop=True)
        generator = torch.Generator().manual_seed(RUN_SEED)
        loader = DataLoader(
            LockedTestDataset(session_table),
            batch_size=LOCKED_BATCH_SIZE,
            shuffle=False,
            num_workers=LOCKED_WORKERS,
            pin_memory=False,
            persistent_workers=True,
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
                fundus_probabilities = {}
                fundus_expected = {}
                lesion_logits_by_position = {}
                lesion_probabilities_by_position = {}
                if len(fundus_positions) > 0:
                    probabilities = output["fundus"]["original"]["fusion"]["probabilities"].float().cpu()
                    expected = output["fundus"]["original"]["fusion"]["expected_grade"].float().cpu()
                    lesion_logits = output["fundus"]["lesions"]["presence_logits"].float().cpu()
                    lesion_probabilities = torch.sigmoid(lesion_logits)
                    for local, position in enumerate(fundus_positions.tolist()):
                        fundus_probabilities[position] = probabilities[local]
                        fundus_expected[position] = expected[local]
                        lesion_logits_by_position[position] = lesion_logits[local]
                        lesion_probabilities_by_position[position] = lesion_probabilities[local]
                oct_probabilities = {}
                oct_expected = {}
                if len(oct_positions) > 0:
                    probabilities = output["oct"]["probabilities"].float().cpu()
                    expected = output["oct"]["expected_grade"].float().cpu()
                    for local, position in enumerate(oct_positions.tolist()):
                        oct_probabilities[position] = probabilities[local]
                        oct_expected[position] = expected[local]

                for position in range(len(batch["relative_path"])):
                    modality = str(batch["modality"][position])
                    is_fundus = modality in {"CFP", "UWF"}
                    probability = (
                        fundus_probabilities[position] if is_fundus else oct_probabilities[position]
                    )
                    expected = fundus_expected[position] if is_fundus else oct_expected[position]
                    classes = 5 if is_fundus else 3
                    values = probability.numpy().astype(float)
                    if not validate_grade_probabilities(values):
                        raise FloatingPointError("Invalid raw grade probability distribution.")
                    record = {
                        "run_signature": run_signature,
                        "locked_test_index": int(batch["locked_test_index"][position]),
                        "relative_path": str(batch["relative_path"][position]),
                        "image_sha256": str(batch["image_sha256"][position]),
                        "modality": modality,
                        "raw_prediction": int(probability.argmax()),
                        "raw_expected_grade": float(expected),
                    }
                    for class_index in range(5):
                        record[f"raw_probability_{class_index}"] = (
                            float(probability[class_index]) if class_index < classes else math.nan
                        )
                    for lesion_index, lesion in enumerate(LESION_COLUMNS):
                        record[f"raw_{lesion}_logit"] = (
                            float(lesion_logits_by_position[position][lesion_index])
                            if is_fundus else math.nan
                        )
                        record[f"raw_{lesion}_probability"] = (
                            float(lesion_probabilities_by_position[position][lesion_index])
                            if is_fundus else math.nan
                        )
                    buffer.append(record)
                    completed_this_session += 1
                    if len(buffer) == PART_ROWS:
                        save_part(buffer, run_signature)
                        buffer = []

                elapsed = time.perf_counter() - inference_start
                throughput = completed_this_session / max(elapsed, 1e-12)
                total_complete = session_start_index + completed_this_session
                remaining = EXPECTED_ROWS - total_complete
                eta_minutes = remaining / max(throughput, 1e-12) / 60.0
                if batch_number == 1 or batch_number % 10 == 0 or total_complete == session_end_index:
                    print(
                        f"Batch {batch_number:04d} | session_rows={completed_this_session:4d} | "
                        f"total={total_complete:4d}/{EXPECTED_ROWS} | "
                        f"throughput={throughput:.3f} img/s | full-run ETA={eta_minutes:.1f} min"
                    )
        if buffer:
            save_part(buffer, run_signature)
        session_seconds = time.perf_counter() - inference_start
        del loader, model, checkpoint, core
    else:
        official_sha = None
        official_ok = True
        model_loaded_strictly = True
        parameters_finite = True
        completed_this_session = 0
        session_seconds = 0.0
        print("\n[3] MODEL INFERENCE")
        print("-" * 80)
        print("All locked-test prediction rows were already persisted; model load skipped.")

    next_index, existing_frames, part_records = scan_and_validate_parts(table, run_signature)
    run_complete = next_index == EXPECTED_ROWS
    status = "RAW_PREDICTIONS_COMPLETE" if run_complete else "RESUMABLE_PARTIAL"
    write_inference_state(run_signature, source_sha, next_index, part_records, status)

    session_gates = {
        **hash_gates,
        **evidence_gates,
        "exact_locked_test_table_loaded": len(table) == EXPECTED_ROWS,
        "locked_test_modality_counts_exact": modality_counts == EXPECTED_MODALITY_COUNTS,
        "locked_test_indices_contiguous": indices_exact,
        "membership_and_cache_manifests_label_free": membership_label_free and cache_label_free,
        "local_locked_test_cache_complete": local_files_ready,
        "CPU_float32_execution_enforced": True,
        "runner_source_frozen_and_verified": sha256_file(FROZEN_SOURCE_PATH) == source_sha,
        "official_weight_verified_if_model_loaded": official_ok,
        "selected_epoch10_model_loaded_strictly_if_required": model_loaded_strictly,
        "all_model_parameters_finite_if_loaded": parameters_finite,
        "at_least_one_row_completed_or_run_already_complete": completed_this_session > 0 or run_complete,
        "persisted_parts_contiguous_and_verified": next_index >= session_end_index,
        "completed_rows_do_not_exceed_5380": next_index <= EXPECTED_ROWS,
        "test_labels_not_loaded": True,
        "scientific_metrics_not_computed": True,
        "model_parameters_not_updated": True,
    }
    print("\n" + "=" * 80)
    print("LOCKED-TEST INFERENCE SESSION GATES")
    print("=" * 80)
    for name, passed in session_gates.items():
        print(f"{name:70s}: {passed}")
    if not all(session_gates.values()):
        raise RuntimeError("Locked-test inference session gate failed.")

    if not run_complete:
        throughput = (
            completed_this_session / session_seconds
            if session_seconds > 0 and completed_this_session > 0 else None
        )
        evidence = {
            "stage": "0G-4D-5E-2B",
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
            "test_images_accessed": True,
            "test_labels_loaded": False,
            "scientific_metrics_computed": False,
            "model_parameters_updated": False,
            "next_required_stage": "CONTINUE_SAME_LABEL_BLIND_CPU_FP32_INFERENCE",
        }
        atomic_json_save(evidence, LATEST_SESSION_EVIDENCE_PATH)
        print("\nInference state:")
        print(INFERENCE_STATE_PATH)
        print("Completed rows:", next_index, "/", EXPECTED_ROWS)
        print("Session throughput img/s:", throughput)
        print("Test labels loaded          : NO")
        print("Scientific metrics computed : NO")
        print("Model parameters updated    : NO")
        print("\nNEXT REQUIRED STAGE:")
        print("Continue the identical runner from its persisted next row.")
        print("\n" + "=" * 80)
        print("STAGE 0G-4D-5E-2B STATUS: PASS — RESUMABLE PARTIAL")
        print("=" * 80)
        return

    print("\n[5] FREEZE COMPLETE RAW PREDICTIONS")
    print("-" * 80)
    raw = pd.concat(existing_frames, ignore_index=True)
    raw = raw.sort_values("locked_test_index").reset_index(drop=True)
    raw_complete = (
        len(raw) == EXPECTED_ROWS
        and raw["locked_test_index"].astype(int).tolist() == list(range(EXPECTED_ROWS))
        and raw["relative_path"].astype(str).tolist() == table["relative_path"].astype(str).tolist()
        and raw["image_sha256"].astype(str).tolist() == table["image_sha256"].astype(str).tolist()
        and raw["relative_path"].is_unique
        and not any(any(token in column.lower() for token in ("target", "label", "grade_true")) for column in raw.columns)
    )
    if not raw_complete:
        raise RuntimeError("Complete raw locked-test prediction consolidation failed.")
    raw_action, raw_sha = create_csv_once_verified(raw, RAW_PREDICTIONS_PATH)
    print("Raw prediction rows:", len(raw))
    print("Raw action         :", raw_action)
    print("Raw predictions SHA:", raw_sha)

    print("\n[6] APPLY FROZEN CALIBRATION TRANSFORMS")
    print("-" * 80)
    transform = import_module(CALIBRATED_TRANSFORM_SOURCE_PATH, "fovea_dr_calibrated_transform_test")
    calibrated = raw.copy()
    grade_invariant = True
    lesion_invariant = True
    for modality, classes in (("CFP", 5), ("UWF", 5), ("OCT", 3)):
        mask = calibrated["modality"].eq(modality)
        columns = [f"raw_probability_{index}" for index in range(classes)]
        probabilities = calibrated.loc[mask, columns].to_numpy(dtype=float)
        temperature = float(temperatures["grade_temperatures"][modality]["temperature"])
        adjusted = transform.apply_grade_temperature(probabilities, temperature)
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
        grade_invariant = grade_invariant and invariant
        print(f"{modality:3s} grade | rows={int(mask.sum())} | T={temperature:.9f} | invariant={invariant}")

    fundus_mask = calibrated["modality"].isin(["CFP", "UWF"])
    for lesion in LESION_COLUMNS:
        logits = calibrated.loc[fundus_mask, f"raw_{lesion}_logit"].to_numpy(dtype=float)
        raw_probabilities = calibrated.loc[fundus_mask, f"raw_{lesion}_probability"].to_numpy(dtype=float)
        temperature = float(temperatures["lesion_temperatures"][lesion]["temperature"])
        adjusted = transform.apply_lesion_temperature(logits, temperature)
        column = f"calibrated_{lesion}_probability"
        calibrated[column] = math.nan
        calibrated.loc[fundus_mask, column] = adjusted
        invariant = bool(np.array_equal(raw_probabilities >= 0.5, adjusted >= 0.5))
        lesion_invariant = lesion_invariant and invariant
        print(f"{lesion:36s} | T={temperature:.9f} | invariant={invariant}")

    calibrated["calibrated_prediction"] = calibrated["calibrated_prediction"].astype(int)
    calibrated_valid = all_calibrated_probabilities_valid(calibrated)
    calibrated_label_free = not any(
        any(token in column.lower() for token in ("target", "label", "grade_true"))
        for column in calibrated.columns
    )
    final_gates = {
        "raw_predictions_complete": raw_complete,
        "grade_decisions_invariant": grade_invariant,
        "lesion_threshold_decisions_invariant": lesion_invariant,
        "all_calibrated_probabilities_valid": calibrated_valid,
        "calibrated_predictions_label_free": calibrated_label_free,
        "test_labels_not_loaded": True,
        "scientific_metrics_not_computed": True,
        "model_parameters_not_updated": True,
    }
    for name, passed in final_gates.items():
        print(f"{name:70s}: {passed}")
    if not all(final_gates.values()):
        raise RuntimeError("Frozen test calibration-transform gate failed.")

    calibrated_action, calibrated_sha = create_csv_once_verified(
        calibrated, CALIBRATED_PREDICTIONS_PATH
    )
    write_inference_state(
        run_signature,
        source_sha,
        EXPECTED_ROWS,
        part_records,
        "PREDICTIONS_FROZEN_BEFORE_LABEL_ACCESS",
        raw_sha,
        calibrated_sha,
    )
    all_gates = {**session_gates, **final_gates}
    final_evidence = {
        "stage": "0G-4D-5E-2B",
        "status": "PASS",
        "session_status": "PREDICTIONS_FROZEN_BEFORE_LABEL_ACCESS",
        "execution_backend": "CPU_FLOAT32",
        "run_signature": run_signature,
        "completed_rows": EXPECTED_ROWS,
        "raw_predictions": str(RAW_PREDICTIONS_PATH),
        "raw_predictions_sha256": raw_sha,
        "raw_predictions_action": raw_action,
        "calibrated_predictions": str(CALIBRATED_PREDICTIONS_PATH),
        "calibrated_predictions_sha256": calibrated_sha,
        "calibrated_predictions_action": calibrated_action,
        "runner_source": str(FROZEN_SOURCE_PATH),
        "runner_source_sha256": source_sha,
        "inference_state": str(INFERENCE_STATE_PATH),
        "inference_state_sha256": sha256_file(INFERENCE_STATE_PATH),
        "gates": {name: bool(value) for name, value in all_gates.items()},
        "test_images_accessed": True,
        "test_labels_loaded": False,
        "scientific_metrics_computed": False,
        "model_parameters_updated": False,
        "next_required_stage": "ONE_TIME_TEST_LABEL_JOIN_AND_PREREGISTERED_SCORING",
    }
    atomic_json_save(final_evidence, FINAL_EVIDENCE_PATH)
    atomic_json_save(final_evidence, LATEST_SESSION_EVIDENCE_PATH)

    print("\nRaw predictions:")
    print(RAW_PREDICTIONS_PATH)
    print("SHA-256:", raw_sha)
    print("\nCalibrated predictions:")
    print(CALIBRATED_PREDICTIONS_PATH)
    print("SHA-256:", calibrated_sha)
    print("\nFinal evidence:")
    print(FINAL_EVIDENCE_PATH)
    print("Test labels loaded          : NO")
    print("Scientific metrics computed : NO")
    print("Model parameters updated    : NO")
    print("\nNEXT REQUIRED STAGE:")
    print("One-time test-label join and preregistered scoring.")
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-5E-2B STATUS: PASS — PREDICTIONS FROZEN")
    print("=" * 80)


if __name__ == "__main__":
    main()
