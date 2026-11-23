from google.colab import drive
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

import torch


WIDTH = 120
VARIANT = "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE"
PROJECT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
DRIVE_ROOT = Path("/content/drive")

RUNNER_SHA = "b3bc733b2204a1657a6d1f62226fb13f04bc3c61d953334bfd04329fc62e3495"
CACHE_SOURCE_SHA = "cdd350270818e9413eed027262d89771a61a97a2f37902fbec82aae4a8f075b5"
WEIGHT_SHA = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
WEIGHT_URL = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"

PREDECESSORS = {
    "B1_completion_contract": (
        "Metadata/mmrdr_fovea_dr_posthoc_B1_completion_v1.json",
        "3a67bf812f1d0dc2b7eb1089cc9a0f3327f8e8f9b8d7f86c007f14429ce26fa4",
    ),
    "B1_completion_evidence": (
        "Evidence/stage0g4d5f3c_b1_completion_freeze.json",
        "e965db2f56b3d63e101208eacf3fba9969df0921b458605d1a812df68c98b474",
    ),
    "B2_completion_contract": (
        "Metadata/mmrdr_fovea_dr_posthoc_B2_completion_v1.json",
        "19171fa0bb531407424bce818395f4eb05bf479d02cd9aabb7d9c005d24284fd",
    ),
    "B2_completion_evidence": (
        "Evidence/stage0g4d5f3c_b2_completion_freeze.json",
        "2a17b8b516ec7aebdb256be8f5fe8e587f5baf43f3aceb33345f93d5b8b2c116",
    ),
    "B3_completion_contract": (
        "Metadata/mmrdr_fovea_dr_posthoc_B3_completion_v1.json",
        "146cf297e679675f86f544bb243a8be134b893ba1ece94300c4ebf2184d9b3ca",
    ),
    "B3_completion_evidence": (
        "Evidence/stage0g4d5f3c_b3_completion_freeze.json",
        "55152a9583d9d7fa1ac05b42850b213e78da0245c594bb7541699ff62054cb7b",
    ),
    "A1_completion_contract": (
        "Metadata/mmrdr_fovea_dr_posthoc_A1_completion_v1.json",
        "5a02ba2276e1aa8366e65782da06078a22ac3fa67528aeefee08beb7354a2097",
    ),
    "A1_completion_evidence": (
        "Evidence/stage0g4d5f3c_a1_completion_freeze.json",
        "503a78243a41157a54f63582c62045789c742d32f6b894c54568fa4db2d02d5e",
    ),
    "A2_completion_contract": (
        "Metadata/mmrdr_fovea_dr_posthoc_A2_completion_v1.json",
        "f4acb32413bd1c0b5fe043abbcb388b097246b53a1e786bf459490c16bc1a93e",
    ),
    "A2_completion_evidence": (
        "Evidence/stage0g4d5f3c_a2_completion_freeze.json",
        "829b7e4686bcaf92b24dbe6432ef2e22abb5698fd30e905ad9121d40a736ee35",
    ),
}


def banner(text):
    print("=" * WIDTH, flush=True)
    print(text, flush=True)
    print("=" * WIDTH, flush=True)


def section(text):
    print(f"\n{text}", flush=True)
    print("-" * WIDTH, flush=True)


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filesystem_type(path):
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", str(path)],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip().lower()


def real_drive_ready():
    return PROJECT.exists() and "drive" in filesystem_type(DRIVE_ROOT)


def safe_mount_recovery():
    if real_drive_ready():
        print("Real Google Drive is already mounted.", flush=True)
        return

    print("The real Google Drive is not currently accessible.", flush=True)
    try:
        drive.flush_and_unmount()
        print("Previous Drive connection flushed/unmounted.", flush=True)
    except Exception as exc:
        print(f"Drive flush/unmount note: {exc}", flush=True)

    if DRIVE_ROOT.is_symlink():
        raise RuntimeError("/content/drive is a symlink; refusing an unsafe mount repair.")

    if DRIVE_ROOT.exists() and any(DRIVE_ROOT.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        preserved = Path(f"/content/drive_stale_local_{stamp}")
        if preserved.exists():
            raise RuntimeError(f"Preservation target already exists: {preserved}")
        DRIVE_ROOT.rename(preserved)
        print(f"Preserved non-mounted local entries at: {preserved}", flush=True)

    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    drive.mount(str(DRIVE_ROOT))

    if not real_drive_ready():
        raise RuntimeError("The real Google Drive could not be verified after mounting.")


def stream_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Command       :", " ".join(command), flush=True)
    print("Persistent log:", log_path, flush=True)

    with open(log_path, "a", encoding="utf-8") as log:
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


def download_weight(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    with urllib.request.urlopen(WEIGHT_URL) as response, open(temporary, "wb") as output:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
        output.flush()
        os.fsync(output.fileno())

    if sha256_file(temporary) != WEIGHT_SHA:
        raise RuntimeError("Downloaded EfficientNetV2-S weight failed SHA-256 verification.")
    os.replace(temporary, path)


def validate_state(run_dir, checkpoint_dir):
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        return {
            "fresh": True,
            "status": "NOT_STARTED",
            "last_completed_epoch": 0,
            "global_optimizer_step": 0,
            "best_epoch": 0,
            "bad_epochs": 0,
        }

    state = read_json(state_path)
    if state.get("variant_id") != VARIANT:
        raise RuntimeError("Existing state belongs to a different variant.")
    if state.get("calibration_used") is not False or state.get("locked_test_used") is not False:
        raise RuntimeError("A3 state reports prohibited CALIBRATION or locked-test access.")

    epoch = int(state.get("last_completed_epoch", -1))
    step = int(state.get("global_optimizer_step", -1))
    if epoch < 1 or epoch > 30 or step != epoch * 415:
        raise RuntimeError("A3 epoch/global-step state is inconsistent.")

    history_path = run_dir / "epoch_history.csv"
    step_path = run_dir / "optimizer_step_log.csv"
    last_path = checkpoint_dir / "last_resume.pt"
    best_path = checkpoint_dir / "best_model.pt"
    for path in (history_path, step_path, last_path, best_path):
        if not path.exists():
            raise RuntimeError(f"A3 persistent artifact is missing: {path}")

    history = read_csv(history_path)
    steps = read_csv(step_path)
    checks = {
        "history_rows": len(history) == epoch,
        "history_epoch": int(history[-1]["epoch"]) == epoch,
        "history_step": int(history[-1]["global_optimizer_step"]) == step,
        "step_rows": len(steps) == step,
        "step_end": int(steps[-1]["global_optimizer_step"]) == step,
        "last_hash": state.get("last_checkpoint_sha256") == sha256_file(last_path),
        "best_hash": state.get("best_checkpoint_sha256") == sha256_file(best_path),
        "history_hash": state.get("history_sha256") == sha256_file(history_path),
        "step_hash": state.get("step_log_sha256") == sha256_file(step_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("A3 saved-state verification failed:\n" + "\n".join(failed))

    state["fresh"] = False
    return state


def amp_state(checkpoint_path):
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scaler = checkpoint.get("scaler_state_dict")
    if not isinstance(scaler, dict):
        raise RuntimeError("A3 checkpoint has no scaler_state_dict.")
    return {
        "scale": float(scaler["scale"]),
        "growth_interval": int(scaler["growth_interval"]),
        "growth_tracker": int(scaler["_growth_tracker"]),
    }


safe_mount_recovery()

RUNNER = PROJECT / "Code/FOVEA_DR/v1_1/stage0g4d5f3a_resumable_final_variant_training.py"
CACHE_SOURCE = PROJECT / "Code/FOVEA_DR/v1_1/stage0g4d5f1b_comparator_development_cache_restore.py"
OFFICIAL_EVIDENCE = PROJECT / "Evidence/stage0g4d5f1c_official_comparator_weight_acquisition.json"
RUN_DIR = PROJECT / f"Runs/FOVEA_DR/posthoc_comparisons/final_seed2026/{VARIANT}"
CHECKPOINT_DIR = PROJECT / f"Checkpoints/FOVEA_DR/posthoc_comparisons/final_seed2026/{VARIANT}"
LAST = CHECKPOINT_DIR / "last_resume.pt"
WEIGHT = Path("/root/.cache/torch/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth")
LOG_DIR = PROJECT / "Logs/FOVEA_DR"

banner("FOVEA-DR — A3 AUTOMATIC SEQUENTIAL TRAINING SUPERVISOR")

section("[1] GPU AND FROZEN EXECUTION ORDER")
gpu_gates = {
    "CUDA_available": torch.cuda.is_available(),
    "exactly_one_GPU_visible": torch.cuda.device_count() == 1,
    "GPU_memory_at_least_10GB": (
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory / 1024**3 >= 10
    ),
    "runner_hash_verified": RUNNER.exists() and sha256_file(RUNNER) == RUNNER_SHA,
    "cache_restore_source_verified": (
        CACHE_SOURCE.exists() and sha256_file(CACHE_SOURCE) == CACHE_SOURCE_SHA
    ),
    "original_weight_evidence_exists": OFFICIAL_EVIDENCE.exists(),
}
for label, (relative, expected_sha) in PREDECESSORS.items():
    path = PROJECT / relative
    gpu_gates[f"{label}_verified"] = path.exists() and sha256_file(path) == expected_sha

if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
    print("VRAM GB :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3))
for name, passed in gpu_gates.items():
    print(f"{name:<72}: {passed}")
failed = [name for name, passed in gpu_gates.items() if not passed]
if failed:
    raise RuntimeError("A3 prerequisites failed:\n" + "\n".join(failed))

section("[2] PERSISTENT A3 STATE")
start_state = validate_state(RUN_DIR, CHECKPOINT_DIR)
start_epoch = int(start_state["last_completed_epoch"])
print("A3 action              :", "FRESH_START" if start_state["fresh"] else "RESUME")
print("Last completed epoch   :", start_epoch)
print("Global optimizer step  :", start_state["global_optimizer_step"])
print("Current status         :", start_state["status"])
print("Best epoch             :", start_state.get("best_epoch", 0))
print("Bad epochs             :", start_state.get("bad_epochs", 0), "/ 6")

if start_state["status"] == "COMPLETE":
    print("\nA3 is already complete. NEXT ACTION: FREEZE A3 FINAL COMPLETION.")
    raise SystemExit(0)

section("[3] DEVELOPMENT-CACHE RESTORATION")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
cache_log = LOG_DIR / f"a3_cache_restore_{stamp}.log"
cache_code = stream_command([sys.executable, "-u", str(CACHE_SOURCE)], cache_log)
print("\nCache restoration return code:", cache_code)
if cache_code != 0:
    raise RuntimeError(f"Development-cache restoration failed with code {cache_code}.")

section("[4] OFFICIAL A3 BACKBONE WEIGHT")
if not WEIGHT.exists() or sha256_file(WEIGHT) != WEIGHT_SHA:
    print("Downloading official EfficientNetV2-S weight...", flush=True)
    download_weight(WEIGHT)
print("Weight path     :", WEIGHT)
print("Weight SHA-256  :", sha256_file(WEIGHT))
print("Weight verified :", sha256_file(WEIGHT) == WEIGHT_SHA)
if sha256_file(WEIGHT) != WEIGHT_SHA:
    raise RuntimeError("Official A3 backbone weight verification failed.")

section("[5] STARTING AUTOMATIC A3 TRAINING")
print(
    f"The supervisor will continue from Epoch {start_epoch + 1} up to Epoch 30,\n"
    "unless early stopping, an AMP safety pause, a runtime interruption, or a training failure occurs."
)

completed_this_cell = 0
planned_amp_pause = False

while True:
    current = validate_state(RUN_DIR, CHECKPOINT_DIR)
    current_epoch = int(current["last_completed_epoch"])
    status = current["status"]

    if status == "COMPLETE" or current_epoch >= 30:
        break

    amp = amp_state(LAST) if current_epoch > 0 else None
    banner("PRE-EPOCH SAFETY STATE")
    print("Last completed epoch :", current_epoch)
    print("Next target epoch    :", current_epoch + 1)
    print("Global step          :", current["global_optimizer_step"])
    print("Bad epochs           :", current.get("bad_epochs", 0))
    if amp is None:
        print("AMP state            : FRESH INITIALIZATION")
    else:
        print("AMP scale            :", amp["scale"])
        print("AMP growth interval  :", amp["growth_interval"])
        print("AMP growth tracker   :", amp["growth_tracker"])

        if amp["scale"] >= 4096.0 and amp["growth_interval"] == 2000:
            planned_amp_pause = True
            print("\nAMP SAFETY PAUSE: scale has reached 4096 with the original growth interval.")
            print("No new epoch was started.")
            print("NEXT ACTION: APPLY THE A3 AMP SCALE-CEILING AMENDMENT, THEN RERUN THIS CELL.")
            break

    target_epoch = current_epoch + 1
    banner(f"STARTING A3 EPOCH {target_epoch}")
    epoch_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    epoch_log = LOG_DIR / f"a3_epoch{target_epoch:02d}_automatic_{epoch_stamp}.log"
    command = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--variant",
        VARIANT,
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
            f"A3 Epoch {target_epoch} failed with code {return_code}. "
            f"The last fully completed epoch remains Epoch {current_epoch}."
        )

    updated = validate_state(RUN_DIR, CHECKPOINT_DIR)
    if int(updated["last_completed_epoch"]) != target_epoch:
        raise RuntimeError("Runner returned zero but the expected epoch was not committed.")

    completed_this_cell += 1
    history = read_csv(RUN_DIR / "epoch_history.csv")
    row = history[-1]
    print("\nEPOCH COMPLETION SUMMARY")
    print("-" * WIDTH)
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
    print("Bad epochs            :", updated.get("bad_epochs"))
    print("Run status            :", updated["status"])

banner("A3 AUTOMATIC SESSION SUMMARY")
final_state = validate_state(RUN_DIR, CHECKPOINT_DIR)
print("Started after epoch       :", start_epoch)
print("Epochs completed this cell:", completed_this_cell)
print("Last completed epoch      :", final_state["last_completed_epoch"])
print("Global optimizer step     :", final_state["global_optimizer_step"])
print("Current status            :", final_state["status"])
print("Current best epoch        :", final_state.get("best_epoch"))
print("Best mean Fundus QWK      :", final_state.get("best_mean_fundus_QWK"))
print("Best mean Fundus NLL      :", final_state.get("best_mean_fundus_NLL"))
print("Bad epochs                :", final_state.get("bad_epochs"))
print("CALIBRATION used          : NO")
print("Locked test used          : NO")

if final_state["status"] == "COMPLETE":
    print("\nNEXT ACTION: FREEZE A3 FINAL COMPLETION.")
elif planned_amp_pause:
    print("\nNEXT ACTION: APPLY THE A3 AMP SCALE-CEILING AMENDMENT.")
else:
    print("\nNEXT ACTION: RERUN THIS SAME SUPERVISOR IN THE NEXT GPU SESSION.")
