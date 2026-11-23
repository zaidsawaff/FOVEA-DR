from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import subprocess

import pandas as pd
from PIL import Image, ImageFile


# =============================================================================
# STAGE 0G-4D-5A — CALIBRATION-ONLY LOCAL CACHE PREPARATION
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

SPLIT_PATH = PROJECT_ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
TRAIN_MANIFEST_PATH = (
    PROJECT_ROOT / "Audit/mmrdr_fovea_dr_TRAIN_local_cache_manifest_v1.csv"
)
VALIDATION_MANIFEST_PATH = (
    PROJECT_ROOT / "Audit/mmrdr_fovea_dr_VALIDATION_local_cache_manifest_v1.csv"
)
BASELINE_FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Metadata/mmrdr_fovea_dr_baseline_seed2026_selection_freeze_v1.json"
)
BASELINE_FREEZE_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d4c_baseline_completion_and_selection_freeze.json"
)
FROZEN_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "Checkpoints/FOVEA_DR/baseline_seed2026/selected_epoch10_frozen.pt"
)
RUN_STATE_PATH = (
    PROJECT_ROOT / "Runs/FOVEA_DR/baseline_seed2026/run_state.json"
)

RAW_DIR = PROJECT_ROOT / "Data/Raw/MMRDR_Figshare_v2_Archives"
VOLUME_VIEW = Path("/content/MMRDR_volume_view")
CALIBRATION_CACHE_ROOT = Path("/content/MMRDR_calibration_v1")
EXTRACT_LIST = Path("/content/mmrdr_CALIBRATION_extract_list_v1.txt")
REPAIR_LIST = Path("/content/mmrdr_CALIBRATION_repair_list_v1.txt")

CALIBRATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Audit/mmrdr_fovea_dr_CALIBRATION_local_cache_manifest_v1.csv"
)
FROZEN_SOURCE_PATH = (
    PROJECT_ROOT
    / "Code/FOVEA_DR/v1_1/stage0g4d5a_calibration_only_cache_preparation.py"
)
EVIDENCE_PATH = (
    PROJECT_ROOT
    / "Evidence/stage0g4d5a_calibration_only_cache_preparation.json"
)

EXPECTED_SPLIT_SHA = (
    "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140"
)
EXPECTED_TRAIN_MANIFEST_SHA = (
    "32f9a3b6b17916b78f591007a85ce2e1910c9c4d2b1294af1c391b19f476bd15"
)
EXPECTED_VALIDATION_MANIFEST_SHA = (
    "d94c5613b6d030dd6e99246b1b72f77746b5223139a4597bef733eb38a5786af"
)
EXPECTED_BASELINE_FREEZE_MANIFEST_SHA = (
    "505e7cb52155e0c8656a81c57754e890dff81d4b53e9ada8920de999fd11239d"
)
EXPECTED_FROZEN_CHECKPOINT_SHA = (
    "89f5e87b647680a9094c55bf647dc1dd2abb624681e733ffb453e30e4c11ff1f"
)

EXPECTED_TOTAL_DEVELOPMENT_ROWS = 18930
EXPECTED_TRAIN_ROWS = 13251
EXPECTED_VALIDATION_ROWS = 2839
EXPECTED_CALIBRATION_ROWS = 2840
EXPECTED_MODALITIES = {"CFP", "UWF", "OCT"}
DISK_HEADROOM_BYTES = 5 * 1024 ** 3

EXPECTED_VOLUME_SIZES = {
    f"MMRDR.zip.{index:03d}": 2097152000
    for index in range(1, 9)
}
EXPECTED_VOLUME_SIZES["MMRDR.zip.009"] = 1833101731

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


def all_evidence_gates_pass(evidence):
    gates = evidence.get("gates", {})
    return (
        evidence.get("status") == "PASS"
        and isinstance(gates, dict)
        and len(gates) > 0
        and all(bool(value) for value in gates.values())
    )


def gibibytes(value):
    return value / (1024 ** 3)


def normalize_relative_path(value):
    return str(value).replace("\\", "/")


def safe_relative_path(value):
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
    )


def contains_locked_test_token(value):
    forbidden = {"ts", "test", "testing", "locked_test"}
    return any(
        part.lower() in forbidden
        for part in PurePosixPath(value).parts
    )


def atomic_csv_save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(payload, file, indent=2)
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
        raise RuntimeError("Frozen source copy verification failed.")
    os.replace(temporary, destination)
    return "COPIED_AND_FROZEN"


def extract_members(archive, relative_paths, list_path, label):
    if not relative_paths:
        print(f"{label}: no extraction required.")
        return True

    list_path.write_text(
        "\n".join(relative_paths) + "\n",
        encoding="utf-8",
    )

    print(f"{label}: extracting {len(relative_paths)} members...")

    result = subprocess.run(
        [
            "7z",
            "x",
            "-y",
            "-aoa",
            str(archive),
            f"-o{CALIBRATION_CACHE_ROOT}",
            f"@{list_path}",
        ],
        capture_output=True,
        text=True,
    )

    print(f"{label} return code:", result.returncode)
    print(result.stdout[-2500:])

    if result.returncode != 0:
        print(result.stderr[-2500:])

    return result.returncode == 0


print("=" * 80)
print("STAGE 0G-4D-5A — CALIBRATION-ONLY LOCAL CACHE PREPARATION")
print("=" * 80)


# =============================================================================
# 1. FROZEN BASELINE AND SPLIT VERIFICATION
# =============================================================================

print("\n[1] FROZEN BASELINE AND SPLIT VERIFICATION")
print("-" * 80)

required = {
    "split": SPLIT_PATH,
    "TRAIN_manifest": TRAIN_MANIFEST_PATH,
    "VALIDATION_manifest": VALIDATION_MANIFEST_PATH,
    "baseline_freeze_manifest": BASELINE_FREEZE_MANIFEST_PATH,
    "baseline_freeze_evidence": BASELINE_FREEZE_EVIDENCE_PATH,
    "frozen_selected_checkpoint": FROZEN_CHECKPOINT_PATH,
    "run_state": RUN_STATE_PATH,
}

for name, path in required.items():
    print(f"{name:30s}: exists={path.exists()}")

if not all(path.exists() for path in required.values()):
    raise FileNotFoundError("One or more required frozen artifacts are missing.")

split_hash_ok = sha256_file(SPLIT_PATH) == EXPECTED_SPLIT_SHA
train_manifest_hash_ok = (
    sha256_file(TRAIN_MANIFEST_PATH) == EXPECTED_TRAIN_MANIFEST_SHA
)
validation_manifest_hash_ok = (
    sha256_file(VALIDATION_MANIFEST_PATH) == EXPECTED_VALIDATION_MANIFEST_SHA
)
baseline_freeze_manifest_hash_ok = (
    sha256_file(BASELINE_FREEZE_MANIFEST_PATH)
    == EXPECTED_BASELINE_FREEZE_MANIFEST_SHA
)
frozen_checkpoint_hash_ok = (
    sha256_file(FROZEN_CHECKPOINT_PATH)
    == EXPECTED_FROZEN_CHECKPOINT_SHA
)

baseline_freeze_manifest = load_json(BASELINE_FREEZE_MANIFEST_PATH)
baseline_freeze_evidence = load_json(BASELINE_FREEZE_EVIDENCE_PATH)
run_state = load_json(RUN_STATE_PATH)

baseline_freeze_contract_ok = (
    baseline_freeze_manifest.get("status") == "FROZEN"
    and int(baseline_freeze_manifest.get("selected_epoch", -1)) == 10
    and baseline_freeze_manifest.get("frozen_selected_checkpoint_sha256")
    == EXPECTED_FROZEN_CHECKPOINT_SHA
    and baseline_freeze_manifest.get("calibration_used_for_selection") is False
    and baseline_freeze_manifest.get("locked_test_used") is False
)
baseline_freeze_evidence_ok = (
    all_evidence_gates_pass(baseline_freeze_evidence)
    and int(baseline_freeze_evidence.get("selected_epoch", -1)) == 10
    and baseline_freeze_evidence.get("calibration_accessed") is False
    and baseline_freeze_evidence.get("locked_test_accessed") is False
)
baseline_run_complete = (
    run_state.get("status") == "COMPLETE"
    and int(run_state.get("last_completed_epoch", -1)) == 16
    and int(run_state.get("global_optimizer_step", -1)) == 6640
    and int(run_state.get("best_epoch", -1)) == 10
    and int(run_state.get("bad_epochs", -1)) == 6
    and run_state.get("early_stopped") is True
    and run_state.get("best_checkpoint_sha256")
    == EXPECTED_FROZEN_CHECKPOINT_SHA
)

print("Split hash verified             :", split_hash_ok)
print("TRAIN manifest hash verified    :", train_manifest_hash_ok)
print("VALIDATION manifest verified    :", validation_manifest_hash_ok)
print("Baseline freeze manifest verified:", baseline_freeze_manifest_hash_ok)
print("Frozen checkpoint verified      :", frozen_checkpoint_hash_ok)
print("Baseline freeze contract valid  :", baseline_freeze_contract_ok)
print("Baseline freeze evidence PASS   :", baseline_freeze_evidence_ok)
print("Baseline run COMPLETE           :", baseline_run_complete)

if not all([
    split_hash_ok,
    train_manifest_hash_ok,
    validation_manifest_hash_ok,
    baseline_freeze_manifest_hash_ok,
    frozen_checkpoint_hash_ok,
    baseline_freeze_contract_ok,
    baseline_freeze_evidence_ok,
    baseline_run_complete,
]):
    raise RuntimeError("Frozen-baseline prerequisite verification failed.")


# =============================================================================
# 2. CALIBRATION-ONLY SELECTION AND PARTITION SEPARATION
# =============================================================================

print("\n[2] CALIBRATION-ONLY SELECTION")
print("-" * 80)

split = pd.read_csv(SPLIT_PATH)
if "development_split" not in split.columns:
    raise KeyError("development_split")

available_roles = sorted(
    split["development_split"].dropna().astype(str).unique().tolist()
)
three_way_roles_exact = available_roles == [
    "CALIBRATION",
    "TRAIN",
    "VALIDATION",
]

role_counts = {
    str(key): int(value)
    for key, value in split["development_split"].value_counts().to_dict().items()
}
three_way_counts_exact = (
    len(split) == EXPECTED_TOTAL_DEVELOPMENT_ROWS
    and role_counts == {
        "TRAIN": EXPECTED_TRAIN_ROWS,
        "VALIDATION": EXPECTED_VALIDATION_ROWS,
        "CALIBRATION": EXPECTED_CALIBRATION_ROWS,
    }
)

calibration = split[
    split["development_split"] == "CALIBRATION"
].copy()
calibration["split_row_index"] = calibration.index.astype(int)
calibration["relative_path"] = (
    calibration["relative_path"].astype(str).map(normalize_relative_path)
)
calibration = calibration.sort_values("relative_path").reset_index(drop=True)

calibration_rows_ok = len(calibration) == EXPECTED_CALIBRATION_ROWS
calibration_paths = calibration["relative_path"].astype(str).tolist()
calibration_path_set = set(calibration_paths)
paths_unique = len(calibration_path_set) == EXPECTED_CALIBRATION_ROWS
paths_safe = bool(calibration["relative_path"].map(safe_relative_path).all())
no_test_tokens = not bool(
    calibration["relative_path"].map(contains_locked_test_token).any()
)
actual_modalities = set(calibration["modality"].astype(str))
three_modalities_present = actual_modalities == EXPECTED_MODALITIES
modality_counts = {
    str(key): int(value)
    for key, value in calibration["modality"].value_counts().to_dict().items()
}

train_manifest = pd.read_csv(TRAIN_MANIFEST_PATH)
validation_manifest = pd.read_csv(VALIDATION_MANIFEST_PATH)
for frame in (train_manifest, validation_manifest):
    frame["relative_path"] = (
        frame["relative_path"].astype(str).map(normalize_relative_path)
    )

train_path_set = set(train_manifest["relative_path"])
validation_path_set = set(validation_manifest["relative_path"])
train_overlap = sorted(calibration_path_set & train_path_set)
validation_overlap = sorted(calibration_path_set & validation_path_set)
no_path_overlap_with_train = len(train_overlap) == 0
no_path_overlap_with_validation = len(validation_overlap) == 0

print("Available roles              :", available_roles)
print("Role counts                  :", role_counts)
print("CALIBRATION rows             :", len(calibration))
print("CALIBRATION modality counts  :", modality_counts)
print("Unique paths                 :", len(calibration_path_set))
print("Paths structurally safe      :", paths_safe)
print("Locked-test tokens seen      :", not no_test_tokens)
print("Path overlap with TRAIN      :", len(train_overlap))
print("Path overlap with VALIDATION :", len(validation_overlap))

if not all([
    three_way_roles_exact,
    three_way_counts_exact,
    calibration_rows_ok,
    paths_unique,
    paths_safe,
    no_test_tokens,
    three_modalities_present,
    no_path_overlap_with_train,
    no_path_overlap_with_validation,
]):
    raise RuntimeError("CALIBRATION-only selection safety gate failed.")


# =============================================================================
# 3. NINE-VOLUME ARCHIVE INVENTORY
# =============================================================================

print("\n[3] NINE-VOLUME ARCHIVE INVENTORY")
print("-" * 80)

VOLUME_VIEW.mkdir(parents=True, exist_ok=True)
alias_records = []

for volume_name, expected_size in EXPECTED_VOLUME_SIZES.items():
    final_path = RAW_DIR / volume_name
    part_path = RAW_DIR / f"{volume_name}.part"

    if final_path.exists() and final_path.stat().st_size == expected_size:
        source = final_path
        source_type = "FINAL"
    elif part_path.exists() and part_path.stat().st_size == expected_size:
        source = part_path
        source_type = "COMPLETE_PART_ALIAS"
    else:
        raise RuntimeError(f"Complete archive volume unavailable: {volume_name}")

    alias_path = VOLUME_VIEW / volume_name
    if alias_path.is_symlink():
        alias_path.unlink()
    elif alias_path.exists():
        raise RuntimeError(f"Unexpected regular alias-view file: {alias_path}")

    os.symlink(str(source), str(alias_path))
    alias_records.append({
        "volume": volume_name,
        "source": source.name,
        "source_type": source_type,
        "size_bytes": int(source.stat().st_size),
    })
    print(f"{volume_name} <- {source.name} [{source_type}]")

archive = VOLUME_VIEW / "MMRDR.zip.001"
listing = subprocess.run(
    ["7z", "l", "-slt", str(archive)],
    capture_output=True,
    text=True,
)

archive_verified = listing.returncode == 0 and "Volumes = 9" in listing.stdout
print("\n7-Zip return code       :", listing.returncode)
print("All nine volumes seen  :", archive_verified)

if not archive_verified:
    print(listing.stdout[-3000:])
    print(listing.stderr[-3000:])
    raise RuntimeError("Nine-volume archive verification failed.")

member_sizes = {}
current_member = None

for line in listing.stdout.splitlines():
    if line.startswith("Path = "):
        current_member = line[len("Path = "):]
    elif line.startswith("Size = ") and current_member in calibration_path_set:
        try:
            member_sizes[current_member] = int(line[len("Size = "):])
        except ValueError:
            pass

missing_members = [
    path for path in calibration_paths
    if path not in member_sizes
]
all_calibration_members_found = (
    len(missing_members) == 0
    and len(member_sizes) == EXPECTED_CALIBRATION_ROWS
)

print("Required CALIBRATION members:", EXPECTED_CALIBRATION_ROWS)
print("Members located             :", len(member_sizes))
print("Missing members             :", len(missing_members))

if not all_calibration_members_found:
    print("First missing members:", missing_members[:20])
    raise RuntimeError("CALIBRATION archive inventory is incomplete.")


# =============================================================================
# 4. LOCAL CAPACITY AND SELECTIVE EXTRACTION
# =============================================================================

print("\n[4] LOCAL CACHE CAPACITY AND EXTRACTION")
print("-" * 80)

CALIBRATION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def local_size_matches(relative_path):
    path = CALIBRATION_CACHE_ROOT / relative_path
    return (
        path.is_file()
        and path.stat().st_size == member_sizes[relative_path]
    )


missing_before = [
    path for path in calibration_paths
    if not local_size_matches(path)
]

expected_bytes = sum(member_sizes.values())
missing_bytes = sum(member_sizes[path] for path in missing_before)
free_bytes_before = shutil.disk_usage("/content").free
disk_capacity_ok = (
    len(missing_before) == 0
    or free_bytes_before >= missing_bytes + DISK_HEADROOM_BYTES
)

print("Expected cache size GB:", f"{gibibytes(expected_bytes):.3f}")
print("Already valid locally :", EXPECTED_CALIBRATION_ROWS - len(missing_before))
print("Need extraction       :", len(missing_before))
print("Extraction bytes GB   :", f"{gibibytes(missing_bytes):.3f}")
print("Free disk before GB   :", f"{gibibytes(free_bytes_before):.3f}")
print("5 GiB headroom retained:", disk_capacity_ok)

if not disk_capacity_ok:
    raise RuntimeError(
        "Insufficient local storage for CALIBRATION cache plus 5 GiB headroom."
    )

initial_extraction_ok = extract_members(
    archive,
    missing_before,
    EXTRACT_LIST,
    "Initial CALIBRATION extraction",
)
if not initial_extraction_ok:
    raise RuntimeError("CALIBRATION extraction failed.")


# =============================================================================
# 5. FULL CONTENT-AWARE IMAGE VERIFICATION
# =============================================================================

print("\n[5] FULL CALIBRATION IMAGE VERIFICATION")
print("-" * 80)

row_lookup = {
    str(row["relative_path"]): row
    for _, row in calibration.iterrows()
}


def verify_image(relative_path):
    row = row_lookup[relative_path]
    local_path = CALIBRATION_CACHE_ROOT / relative_path
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    actual_size = int(local_path.stat().st_size)
    expected_size = int(member_sizes[relative_path])
    if actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch: actual={actual_size}, expected={expected_size}"
        )

    with Image.open(local_path) as image:
        image.load()
        image_format = str(image.format)
        image_mode = str(image.mode)
        width, height = image.size

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid dimensions: {width}x{height}")

    return {
        "split_row_index": int(row["split_row_index"]),
        "relative_path": relative_path,
        "modality": str(row["modality"]),
        "image_name": str(row["image_name"]),
        "grade": int(row["grade"]),
        "development_split": str(row["development_split"]),
        "local_path": str(local_path),
        "file_size_bytes": actual_size,
        "archive_size_bytes": expected_size,
        "image_sha256": sha256_file(local_path),
        "actual_format": image_format,
        "image_mode": image_mode,
        "width": int(width),
        "height": int(height),
        "readable": True,
    }


record_map = {}
first_failures = []

for position, relative_path in enumerate(calibration_paths, start=1):
    try:
        record_map[relative_path] = verify_image(relative_path)
    except Exception as error:
        first_failures.append({
            "relative_path": relative_path,
            "error": repr(error),
        })

    if position % 500 == 0 or position == EXPECTED_CALIBRATION_ROWS:
        print(
            f"Verified {position:4d}/{EXPECTED_CALIBRATION_ROWS} | "
            f"failures={len(first_failures)}"
        )


# =============================================================================
# 6. SINGLE AUTOMATIC REPAIR PASS
# =============================================================================

print("\n[6] AUTOMATIC REPAIR PASS")
print("-" * 80)

repair_attempted = len(first_failures) > 0
repair_extraction_ok = True
final_failures = []

if repair_attempted:
    repair_paths = [item["relative_path"] for item in first_failures]
    repair_extraction_ok = extract_members(
        archive,
        repair_paths,
        REPAIR_LIST,
        "CALIBRATION repair extraction",
    )
    if repair_extraction_ok:
        for relative_path in repair_paths:
            try:
                record_map[relative_path] = verify_image(relative_path)
            except Exception as error:
                final_failures.append({
                    "relative_path": relative_path,
                    "error": repr(error),
                })
    else:
        final_failures = first_failures
else:
    print("No repair was required.")

all_files_size_match = all(
    local_size_matches(path)
    for path in calibration_paths
)
all_images_readable = (
    len(record_map) == EXPECTED_CALIBRATION_ROWS
    and len(final_failures) == 0
)

print("First-pass failures :", len(first_failures))
print("Final failures      :", len(final_failures))
print("Verified records    :", len(record_map))
print("All files size-match:", all_files_size_match)
print("All images readable :", all_images_readable)

if not all_images_readable:
    print("First final failures:", final_failures[:20])
    raise RuntimeError("CALIBRATION verification failed after repair.")


# =============================================================================
# 7. MANIFEST AND EXACT-HASH PARTITION SEPARATION
# =============================================================================

print("\n[7] CALIBRATION MANIFEST AND PARTITION SEPARATION")
print("-" * 80)

manifest = pd.DataFrame([
    record_map[path]
    for path in calibration_paths
])

calibration_hashes = set(manifest["image_sha256"].astype(str))
train_hashes = set(train_manifest["image_sha256"].astype(str))
validation_hashes = set(validation_manifest["image_sha256"].astype(str))

train_hash_overlap = sorted(calibration_hashes & train_hashes)
validation_hash_overlap = sorted(calibration_hashes & validation_hashes)
no_exact_hash_overlap_with_train = len(train_hash_overlap) == 0
no_exact_hash_overlap_with_validation = len(validation_hash_overlap) == 0

manifest_modality_counts = {
    str(key): int(value)
    for key, value in manifest["modality"].value_counts().to_dict().items()
}
verified_bytes = int(manifest["file_size_bytes"].sum())

atomic_csv_save(manifest, CALIBRATION_MANIFEST_PATH)
calibration_manifest_sha = sha256_file(CALIBRATION_MANIFEST_PATH)

print("Manifest rows                    :", len(manifest))
print("Manifest modalities              :", manifest_modality_counts)
print("Verified cache size GB           :", f"{gibibytes(verified_bytes):.3f}")
print("Exact SHA overlap with TRAIN     :", len(train_hash_overlap))
print("Exact SHA overlap with VALIDATION:", len(validation_hash_overlap))
print("Manifest SHA-256                 :", calibration_manifest_sha)

if train_hash_overlap:
    print("First TRAIN hash overlaps:", train_hash_overlap[:20])
if validation_hash_overlap:
    print("First VALIDATION hash overlaps:", validation_hash_overlap[:20])


# =============================================================================
# 8. READINESS GATES
# =============================================================================

gates = {
    "frozen_split_hash_verified":
        split_hash_ok,
    "TRAIN_manifest_hash_verified":
        train_manifest_hash_ok,
    "VALIDATION_manifest_hash_verified":
        validation_manifest_hash_ok,
    "baseline_selection_freeze_manifest_hash_verified":
        baseline_freeze_manifest_hash_ok,
    "baseline_selection_freeze_evidence_passed":
        baseline_freeze_evidence_ok,
    "frozen_epoch10_checkpoint_hash_verified":
        frozen_checkpoint_hash_ok,
    "baseline_run_is_complete":
        baseline_run_complete,
    "three_way_development_roles_exact":
        three_way_roles_exact,
    "three_way_development_counts_exact":
        three_way_counts_exact,
    "exactly_2840_CALIBRATION_rows":
        calibration_rows_ok,
    "all_selected_rows_are_CALIBRATION":
        bool(calibration["development_split"].eq("CALIBRATION").all()),
    "all_CALIBRATION_paths_unique":
        paths_unique,
    "all_CALIBRATION_paths_structurally_safe":
        paths_safe,
    "no_locked_test_path_tokens":
        no_test_tokens,
    "all_three_modalities_present":
        three_modalities_present,
    "no_relative_path_overlap_with_TRAIN":
        no_path_overlap_with_train,
    "no_relative_path_overlap_with_VALIDATION":
        no_path_overlap_with_validation,
    "all_nine_archive_volumes_verified":
        archive_verified,
    "all_CALIBRATION_members_found_in_archive":
        all_calibration_members_found,
    "disk_capacity_with_5GiB_headroom":
        disk_capacity_ok,
    "initial_extraction_completed":
        initial_extraction_ok,
    "repair_extraction_completed_if_required":
        repair_extraction_ok,
    "all_local_files_match_archive_sizes":
        all_files_size_match,
    "all_CALIBRATION_images_fully_decodable":
        all_images_readable,
    "no_exact_SHA256_overlap_with_TRAIN":
        no_exact_hash_overlap_with_train,
    "no_exact_SHA256_overlap_with_VALIDATION":
        no_exact_hash_overlap_with_validation,
    "manifest_has_exact_CALIBRATION_row_count":
        len(manifest) == EXPECTED_CALIBRATION_ROWS,
    "manifest_modality_counts_match_selection":
        manifest_modality_counts == modality_counts,
    "CALIBRATION_manifest_created":
        CALIBRATION_MANIFEST_PATH.exists(),
    "locked_test_not_accessed":
        True,
}

print("\n" + "=" * 80)
print("CALIBRATION-CACHE READINESS GATES")
print("=" * 80)
for name, passed in gates.items():
    print(f"{name:66s}: {passed}")

if not all(gates.values()):
    raise RuntimeError("STAGE 0G-4D-5A readiness gate failed.")


# =============================================================================
# 9. SOURCE FREEZE AND EVIDENCE
# =============================================================================

source_path = Path(__file__).resolve()
source_sha = sha256_file(source_path)
source_action = copy_once_verified(
    source_path,
    FROZEN_SOURCE_PATH,
    source_sha,
)
frozen_source_sha = sha256_file(FROZEN_SOURCE_PATH)
source_frozen_ok = frozen_source_sha == source_sha

if not source_frozen_ok:
    raise RuntimeError("CALIBRATION cache source freeze failed.")

evidence_gates = {
    **gates,
    "cache_preparation_source_frozen_and_hash_verified": source_frozen_ok,
}

evidence = {
    "stage": "0G-4D-5A",
    "status": "PASS",
    "purpose": "CALIBRATION_ONLY_LOCAL_CACHE_FOR_POST_SELECTION_CALIBRATION",
    "selected_model_epoch": 10,
    "selected_model_sha256": EXPECTED_FROZEN_CHECKPOINT_SHA,
    "baseline_freeze_manifest": str(BASELINE_FREEZE_MANIFEST_PATH),
    "baseline_freeze_manifest_sha256": EXPECTED_BASELINE_FREEZE_MANIFEST_SHA,
    "cache_root": str(CALIBRATION_CACHE_ROOT),
    "CALIBRATION_rows": int(len(manifest)),
    "modality_counts": manifest_modality_counts,
    "expected_cache_bytes": int(expected_bytes),
    "verified_cache_bytes": verified_bytes,
    "images_extracted_initially": int(len(missing_before)),
    "images_repaired": int(len(first_failures)),
    "final_image_failures": final_failures,
    "TRAIN_path_overlap": int(len(train_overlap)),
    "VALIDATION_path_overlap": int(len(validation_overlap)),
    "TRAIN_exact_SHA256_overlap": int(len(train_hash_overlap)),
    "VALIDATION_exact_SHA256_overlap": int(len(validation_hash_overlap)),
    "archive_aliases": alias_records,
    "calibration_manifest": str(CALIBRATION_MANIFEST_PATH),
    "calibration_manifest_sha256": calibration_manifest_sha,
    "source": str(FROZEN_SOURCE_PATH),
    "source_sha256": frozen_source_sha,
    "source_action": source_action,
    "free_disk_before_bytes": int(free_bytes_before),
    "free_disk_after_bytes": int(shutil.disk_usage("/content").free),
    "gates": {name: bool(value) for name, value in evidence_gates.items()},
    "calibration_images_accessed": True,
    "images_displayed": False,
    "model_loaded": False,
    "optimizer_created": False,
    "parameters_updated": False,
    "validation_used_for_new_model_selection": False,
    "locked_test_accessed": False,
    "raw_Drive_archives_modified": False,
    "next_required_stage": "CALIBRATION_ONLY_INFERENCE_AND_TEMPERATURE_SCALING",
}

atomic_json_save(evidence, EVIDENCE_PATH)

print("\nSource action:", source_action)
print("Frozen source:")
print(FROZEN_SOURCE_PATH)
print("Source SHA-256:", frozen_source_sha)

print("\nCALIBRATION cache root:")
print(CALIBRATION_CACHE_ROOT)

print("\nCALIBRATION manifest:")
print(CALIBRATION_MANIFEST_PATH)
print("SHA-256:", calibration_manifest_sha)

print("\nEvidence:")
print(EVIDENCE_PATH)

print("\nImages displayed          : NO")
print("Model loaded              : NO")
print("Optimizer created         : NO")
print("Parameters updated        : NO")
print("New model selection       : NO")
print("Locked test accessed      : NO")
print("Raw Drive archives changed: NO")

print("\nIMPORTANT:")
print("Keep this runtime connected; the CALIBRATION cache under /content is temporary.")

print("\nNEXT REQUIRED STAGE:")
print("CALIBRATION-only inference and temperature scaling using the frozen epoch-10 model")

print("\n" + "=" * 80)
print("STAGE 0G-4D-5A STATUS: PASS")
print("=" * 80)
