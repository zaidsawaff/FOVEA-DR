#!/usr/bin/env python3
"""Permanent automatic supervisor for the three reviewer E2 final runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
from typing import Any

import torch


WIDTH = 124
ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
DRIVE_ROOT = Path("/content/drive")
CODE = ROOT / "Code/FOVEA_DR/v1_1"
LOG_DIR = ROOT / "Logs/FOVEA_DR"

RUNNER = CODE / "stage0g4d5f7d_resumable_reviewer_component_ablation_training.py"
CACHE_SOURCE = CODE / "stage0g4d5f1b_comparator_development_cache_restore.py"
WEIGHT_SOURCE = CODE / "stage0g4d5f1c_official_comparator_weight_acquisition.py"
WEIGHT_EVIDENCE = ROOT / "Evidence/stage0g4d5f1c_official_comparator_weight_acquisition.json"
WEIGHT = Path("/root/.cache/torch/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth")
WEIGHT_URL = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"
TRAIN_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_TRAIN_local_cache_manifest_v1.csv"
VALIDATION_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_VALIDATION_local_cache_manifest_v1.csv"
TRAIN_CACHE = Path("/content/MMRDR_training_v1")
VALIDATION_CACHE = Path("/content/MMRDR_validation_v1")

RUN_BASE = ROOT / "Runs/FOVEA_DR/reviewer_experiments/E2_component_ablation_final_seed2026"
CHECKPOINT_BASE = ROOT / "Checkpoints/FOVEA_DR/reviewer_experiments/E2_component_ablation_final_seed2026"

VARIANTS = [
    "NO_FIELD_OF_VIEW_ENCODING",
    "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE",
    "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE",
]

RUNNER_SHA = "dbed7405c710efb8588443f3e0ab689b26b196ebbd629f0bb57c4f4eed1583bc"
CACHE_SOURCE_SHA = "cdd350270818e9413eed027262d89771a61a97a2f37902fbec82aae4a8f075b5"
WEIGHT_SOURCE_SHA = "b9ccaf24db7255cb73dfd0cacf928556033a9810d5766f73f67ee13ecebd5c8b"
WEIGHT_EVIDENCE_SHA = "84e5f89d29c11175169118b7010e49a91e36eb2987dc84c60961796f54894d54"
WEIGHT_SHA = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"

PARENTS = {
    "E2_preflight_contract": (
        ROOT / "Metadata/mmrdr_fovea_dr_reviewer_E2_preflight_completion_v1.json",
        "ee149800c023fc5130985049bf14115c3b8cafcaac6948c24d1b1cbeba1e6fc4",
    ),
    "E2_preflight_evidence": (
        ROOT / "Evidence/stage0g4d5f7c_reviewer_component_ablation_preflight_completion_freeze.json",
        "674cf3e647338057766a0eda5c2d4a3ca00d954f0dbcbab074c50259a4300b4d",
    ),
    "E2_repair_source": (
        CODE / "stage0g4d5f7a_r1_reviewer_component_ablation_amp_safe_repair.py",
        "bd7358f402d3488f4a829eee85a823456ee6e87b1eb0d676d229b383dbc1a357",
    ),
    "E2_preflight_runner": (
        CODE / "stage0g4d5f7b_r1_reviewer_component_ablation_64step_preflight.py",
        "535ce34324acfe3131e67937cf57efd090c5d6a2f3459b02d0db7fc6a5e59cdd",
    ),
}


def banner(text: str) -> None:
    print("=" * WIDTH, flush=True)
    print(text, flush=True)
    print("=" * WIDTH, flush=True)


def section(text: str) -> None:
    print(f"\n{text}", flush=True)
    print("-" * WIDTH, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filesystem_type(path: Path) -> str:
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip().lower()


def stream_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Command       :", " ".join(command), flush=True)
    print("Persistent log:", log_path, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * WIDTH + "\n")
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.write("UTC START: " + datetime.now(timezone.utc).isoformat() + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        log.write("UTC END: " + datetime.now(timezone.utc).isoformat() + "\n")
        log.write(f"RETURN CODE: {return_code}\n")
        log.flush()
        os.fsync(log.fileno())
    return return_code


def ensure_official_efficientnet_weight() -> str:
    """Verify or atomically restore only the required cached weight.

    This deliberately does not execute the historical weight-acquisition
    stage because that stage owns an immutable evidence file. Re-running it
    in a newer session can legitimately produce different runtime metadata
    and must never be allowed to rewrite that frozen evidence.
    """
    if WEIGHT.is_file() and sha256_file(WEIGHT) == WEIGHT_SHA:
        return "EXISTING_HASH_VERIFIED"

    WEIGHT.parent.mkdir(parents=True, exist_ok=True)
    temporary = WEIGHT.with_name(WEIGHT.name + ".reviewer_e2_download.part")
    if temporary.exists():
        temporary.unlink()
    try:
        torch.hub.download_url_to_file(
            WEIGHT_URL,
            str(temporary),
            hash_prefix=WEIGHT_SHA[:8],
            progress=True,
        )
        actual = sha256_file(temporary)
        if actual != WEIGHT_SHA:
            raise RuntimeError(
                "Downloaded EfficientNetV2-S weight hash mismatch: " + actual
            )
        os.replace(temporary, WEIGHT)
    finally:
        if temporary.exists():
            temporary.unlink()

    if not WEIGHT.is_file() or sha256_file(WEIGHT) != WEIGHT_SHA:
        raise RuntimeError("Atomic EfficientNetV2-S weight restoration failed")
    return "OFFICIAL_WEIGHT_ONLY_ATOMIC_RESTORE"


def cache_matches_manifest(manifest: Path, cache_root: Path, expected_rows: int) -> bool:
    """Fast, read-only readiness check; the training runner repeats size gates."""
    if not manifest.is_file() or not cache_root.is_dir():
        return False
    rows = read_csv(manifest)
    if len(rows) != expected_rows:
        return False
    size_columns = ("file_size_bytes", "archive_size_bytes", "size_bytes")
    size_column = next(
        (name for name in size_columns if rows and name in rows[0]),
        None,
    )
    if size_column is None:
        return False
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("relative_path", "")).strip().replace("\\", "/")
        parts = PurePosixPath(relative)
        if (
            not relative
            or parts.is_absolute()
            or "." in parts.parts
            or ".." in parts.parts
            or relative in seen
        ):
            return False
        seen.add(relative)
        target = cache_root / relative
        try:
            expected_size = int(row[size_column])
        except (TypeError, ValueError):
            return False
        if not target.is_file() or target.stat().st_size != expected_size:
            return False
    return True


def development_caches_ready() -> tuple[bool, bool]:
    return (
        cache_matches_manifest(TRAIN_MANIFEST, TRAIN_CACHE, 13251),
        cache_matches_manifest(VALIDATION_MANIFEST, VALIDATION_CACHE, 2839),
    )


def paths_for(variant: str) -> dict[str, Path]:
    run = RUN_BASE / variant
    checkpoint = CHECKPOINT_BASE / variant
    return {
        "run": run,
        "checkpoint": checkpoint,
        "state": run / "run_state.json",
        "history": run / "epoch_history.csv",
        "steps": run / "optimizer_step_log.csv",
        "last": checkpoint / "last_resume.pt",
        "best": checkpoint / "best_model.pt",
        "latest_predictions": run / "latest_validation_predictions.csv",
        "best_predictions": run / "best_validation_predictions.csv",
    }


def validate_state(variant: str) -> dict[str, Any]:
    paths = paths_for(variant)
    if not paths["state"].is_file():
        residual = [
            path for name, path in paths.items()
            if name not in {"run", "checkpoint", "state"} and path.exists()
        ]
        if residual:
            raise RuntimeError(
                f"{variant} has residual artifacts without run_state.json:\n"
                + "\n".join(str(path) for path in residual)
            )
        return {
            "variant_id": variant,
            "status": "NOT_STARTED",
            "last_completed_epoch": 0,
            "global_optimizer_step": 0,
            "best_epoch": 0,
            "bad_epochs": 0,
            "fresh": True,
        }

    state = load_json(paths["state"])
    if state.get("variant_id") != variant:
        raise RuntimeError(f"Saved state belongs to another variant: {variant}")
    if state.get("status") not in {"RESUMABLE_CHECKPOINT_READY", "COMPLETE"}:
        raise RuntimeError(f"Unsupported saved status for {variant}: {state.get('status')}")
    if state.get("calibration_used") is not False or state.get("locked_test_used") is not False:
        raise RuntimeError(f"{variant} reports prohibited CALIBRATION or locked-test access")

    epoch = int(state.get("last_completed_epoch", -1))
    global_step = int(state.get("global_optimizer_step", -1))
    if epoch < 1 or epoch > 30 or global_step != epoch * 415:
        raise RuntimeError(f"Epoch/global-step state is inconsistent for {variant}")

    required = ["history", "steps", "last", "best", "latest_predictions", "best_predictions"]
    missing = [name for name in required if not paths[name].is_file()]
    if missing:
        raise RuntimeError(f"Persistent artifacts missing for {variant}: {missing}")
    history = read_csv(paths["history"])
    steps = read_csv(paths["steps"])
    checks = {
        "history_rows": len(history) == epoch,
        "history_epochs_contiguous": [int(row["epoch"]) for row in history] == list(range(1, epoch + 1)),
        "history_ends_at_global_step": int(history[-1]["global_optimizer_step"]) == global_step,
        "step_rows": len(steps) == global_step,
        "steps_contiguous": [int(row["global_optimizer_step"]) for row in steps] == list(range(1, global_step + 1)),
        "last_hash": state.get("last_checkpoint_sha256") == sha256_file(paths["last"]),
        "best_hash": state.get("best_checkpoint_sha256") == sha256_file(paths["best"]),
        "history_hash": state.get("history_sha256") == sha256_file(paths["history"]),
        "step_hash": state.get("step_log_sha256") == sha256_file(paths["steps"]),
        "latest_predictions_hash": state.get("latest_validation_predictions_sha256") == sha256_file(paths["latest_predictions"]),
        "best_predictions_hash": state.get("best_validation_predictions_sha256") == sha256_file(paths["best_predictions"]),
    }
    failed = [name for name, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"Saved-state verification failed for {variant}:\n" + "\n".join(failed))

    checkpoint = torch.load(paths["last"], map_location="cpu", weights_only=False)
    scaler = checkpoint.get("scaler_state_dict")
    checkpoint_checks = {
        "checkpoint_variant": checkpoint.get("variant_id") == variant,
        "checkpoint_epoch": int(checkpoint.get("epoch_completed", -1)) == epoch,
        "checkpoint_step": int(checkpoint.get("global_optimizer_step", -1)) == global_step,
        "AMP_state_present": isinstance(scaler, dict),
        "AMP_scale_is_1024": isinstance(scaler, dict) and float(scaler.get("scale", -1)) == 1024.0,
        "AMP_growth_interval_is_1000000": isinstance(scaler, dict) and int(scaler.get("growth_interval", -1)) == 1000000,
    }
    failed = [name for name, value in checkpoint_checks.items() if not value]
    if failed:
        raise RuntimeError(f"Checkpoint verification failed for {variant}:\n" + "\n".join(failed))
    state["fresh"] = False
    return state


def first_unfinished() -> tuple[int | None, str | None, dict[str, Any] | None]:
    seen_unfinished = False
    target: tuple[int | None, str | None, dict[str, Any] | None] = (None, None, None)
    for index, variant in enumerate(VARIANTS):
        state = validate_state(variant)
        complete = state.get("status") == "COMPLETE"
        if not complete and not seen_unfinished:
            seen_unfinished = True
            target = (index, variant, state)
        elif complete and seen_unfinished:
            raise RuntimeError(
                f"Execution-order violation: {variant} is complete while a predecessor is unfinished"
            )
        elif not complete and seen_unfinished and target[1] != variant:
            if int(state.get("last_completed_epoch", 0)) > 0:
                raise RuntimeError(
                    f"Execution-order violation: {variant} started before its predecessor completed"
                )
    return target


def print_state_table() -> None:
    for position, variant in enumerate(VARIANTS, start=1):
        state = validate_state(variant)
        print(
            f"[{position}/3] {variant:55s} | "
            f"status={state['status']:<27s} | "
            f"epoch={int(state['last_completed_epoch']):02d} | "
            f"best={int(state.get('best_epoch', 0)):02d} | "
            f"bad={int(state.get('bad_epochs', 0))}/6",
            flush=True,
        )


def main() -> None:
    banner("FOVEA-DR — REVIEWER E2 AUTOMATIC SEQUENTIAL FINAL-TRAINING SUPERVISOR")
    section("[1] REAL DRIVE, GPU, AND FROZEN PARENTS")
    gates = {
        "real_Google_Drive_mounted": ROOT.is_dir() and "drive" in filesystem_type(DRIVE_ROOT),
        "CUDA_available": torch.cuda.is_available(),
        "exactly_one_GPU_visible": torch.cuda.device_count() == 1,
        "GPU_memory_at_least_10GB": (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory / 1024**3 >= 10.0
        ),
        "runner_hash_verified": RUNNER.is_file() and sha256_file(RUNNER) == RUNNER_SHA,
        "cache_restore_source_verified": CACHE_SOURCE.is_file() and sha256_file(CACHE_SOURCE) == CACHE_SOURCE_SHA,
        "weight_restore_source_verified": WEIGHT_SOURCE.is_file() and sha256_file(WEIGHT_SOURCE) == WEIGHT_SOURCE_SHA,
        "frozen_weight_evidence_verified": (
            WEIGHT_EVIDENCE.is_file()
            and sha256_file(WEIGHT_EVIDENCE) == WEIGHT_EVIDENCE_SHA
        ),
    }
    for label, (path, expected_sha) in PARENTS.items():
        gates[f"{label}_verified"] = path.is_file() and sha256_file(path) == expected_sha
    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))
        print("VRAM GB :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3))
    for name, value in gates.items():
        print(f"{name:76s}: {value}")
    failed = [name for name, value in gates.items() if not value]
    if failed:
        raise RuntimeError("E2 supervisor prerequisites failed:\n" + "\n".join(failed))

    section("[2] PERSISTENT E2 FINAL-TRAINING STATE")
    print_state_table()
    _, starting_variant, starting_state = first_unfinished()
    if starting_variant is None:
        print("\nAll three E2 variants are already complete.")
        print("NEXT ACTION: FREEZE E2 FINAL-TRAINING COMPLETION.")
        return
    print("\nNext variant:", starting_variant)
    print("Next epoch  :", int(starting_state["last_completed_epoch"]) + 1)

    section("[3] DEVELOPMENT-CACHE RESTORATION")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    train_ready, validation_ready = development_caches_ready()
    print("TRAIN cache ready      :", train_ready)
    print("VALIDATION cache ready :", validation_ready)
    if train_ready and validation_ready:
        print("Cache action           : EXISTING_SIZE_VERIFIED — RESTORE SKIPPED")
    else:
        cache_log = LOG_DIR / f"reviewer_e2_cache_restore_{stamp}.log"
        cache_code = stream_command([sys.executable, "-u", str(CACHE_SOURCE)], cache_log)
        print("Cache restoration return code:", cache_code)
        if cache_code != 0:
            raise RuntimeError(f"Development-cache restoration failed with code {cache_code}")
        train_ready, validation_ready = development_caches_ready()
        if not train_ready or not validation_ready:
            raise RuntimeError("Cache restoration returned zero but readiness is incomplete")
        print("Cache action           : FULL_FROZEN_RESTORE_AND_VERIFICATION")

    section("[4] OFFICIAL EFFICIENTNETV2-S WEIGHT")
    weight_action = ensure_official_efficientnet_weight()
    weight_valid = WEIGHT.is_file() and sha256_file(WEIGHT) == WEIGHT_SHA
    print("Weight action   :", weight_action)
    print("Weight path     :", WEIGHT)
    print("Weight SHA-256  :", sha256_file(WEIGHT) if WEIGHT.is_file() else None)
    print("Weight verified :", weight_valid)
    print("Frozen evidence : PRESERVED UNCHANGED")
    if not weight_valid:
        raise RuntimeError("Official EfficientNetV2-S weight is missing or changed")

    section("[5] AUTOMATIC SEQUENTIAL TRAINING")
    print(
        "The supervisor will run one complete epoch at a time and commit it to Drive.\n"
        "It proceeds in the locked E2 order until early stopping, Epoch 30,\n"
        "a runtime interruption, or a verified training failure."
    )
    completed_this_session = 0
    starting_epoch = int(starting_state["last_completed_epoch"])

    while True:
        _, variant, state = first_unfinished()
        if variant is None:
            break
        current_epoch = int(state["last_completed_epoch"])
        target_epoch = current_epoch + 1
        banner(f"STARTING {variant} — EPOCH {target_epoch}")
        print("Current status       :", state["status"])
        print("Global optimizer step:", state["global_optimizer_step"])
        print("Current best epoch   :", state.get("best_epoch", 0))
        print("Bad epochs           :", state.get("bad_epochs", 0), "/ 6")
        print("AMP initial/ceiling  : 1024 / 1024")

        epoch_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        epoch_log = LOG_DIR / f"reviewer_e2_{variant.lower()}_epoch{target_epoch:02d}_{epoch_stamp}.log"
        command = [
            sys.executable,
            "-u",
            str(RUNNER),
            "--variant",
            variant,
            "--session-epochs",
            "1",
            "--workers",
            "2",
        ]
        return_code = stream_command(command, epoch_log)
        print("\nRunner return code:", return_code)
        print("Persistent log   :", epoch_log)
        if return_code != 0:
            raise RuntimeError(
                f"{variant} Epoch {target_epoch} failed with code {return_code}. "
                f"The last committed epoch remains {current_epoch}."
            )

        updated = validate_state(variant)
        if int(updated["last_completed_epoch"]) != target_epoch:
            raise RuntimeError("Runner returned zero but the target epoch was not committed")
        completed_this_session += 1
        history = read_csv(paths_for(variant)["history"])
        row = history[-1]
        print("\nEPOCH COMPLETION SUMMARY")
        print("-" * WIDTH)
        print("Variant               :", variant)
        print("Completed epoch       :", target_epoch)
        print("Global optimizer step :", updated["global_optimizer_step"])
        print("Mean train loss       :", row["mean_train_loss"])
        print("CFP QWK               :", row["CFP_QWK"])
        print("UWF QWK               :", row["UWF_QWK"])
        print("Mean Fundus QWK       :", row["mean_fundus_QWK"])
        print("Mean Fundus NLL       :", row["mean_fundus_NLL"])
        print("OCT QWK               :", row["OCT_QWK"])
        print("OCT macro-F1          :", row["OCT_macro_F1"])
        print("Is best               :", row["is_best"])
        print("Best epoch            :", updated.get("best_epoch"))
        print("Bad epochs            :", updated.get("bad_epochs"), "/ 6")
        print("Run status            :", updated["status"])

    banner("REVIEWER E2 AUTOMATIC SESSION SUMMARY")
    print("Starting variant          :", starting_variant)
    print("Started after epoch       :", starting_epoch)
    print("Epochs completed this cell:", completed_this_session)
    print_state_table()
    print("CALIBRATION used          : NO")
    print("Locked test used          : NO")
    _, remaining_variant, _ = first_unfinished()
    if remaining_variant is None:
        print("\nNEXT ACTION: FREEZE E2 FINAL-TRAINING COMPLETION.")
    else:
        print("\nNEXT ACTION: RERUN THIS SAME PERMANENT SUPERVISOR IN THE NEXT GPU SESSION.")


if __name__ == "__main__":
    main()
