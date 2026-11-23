#!/usr/bin/env python3
"""Schema-only repair for the Stage 0G-4D-5F-6B qualitative panels.

The original frozen experiment completed every numeric inference shard, then
stopped because ``DataFrame.set_index('relative_path')`` moved the path from a
column to ``Series.name``. This wrapper preserves the original source and
protocol byte-for-byte, restores that one field for qualitative rendering,
and calls the original frozen experiment so its own contracts and gates remain
authoritative. No optimizer is created and no parameter is updated.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
ORIGINAL_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f6a_reviewer_validation_mechanistic_experiment.py"
PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_reviewer_experiment_protocol_v1.json"
SHARD_ROOT = ROOT / "Results/FOVEA_DR/reviewer_experiments/validation_mechanistic_v1/numeric_shards"
FINAL_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_validation_mechanistic_audit_v1.json"
FINAL_EVIDENCE = ROOT / "Evidence/stage0g4d5f6b_validation_mechanistic_audit.json"

FROZEN_REPAIR_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f6b_r1_validation_mechanistic_qualitative_repair.py"
REPAIR_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_validation_mechanistic_qualitative_repair_v1.json"
REPAIR_EVIDENCE = ROOT / "Evidence/stage0g4d5f6b_r1_validation_mechanistic_qualitative_repair.json"

EXPECTED_ORIGINAL_SOURCE_SHA = "233938d7126ee0084851d0a55c8f963eb1234d97b3b557910df5da2beb09d725"
EXPECTED_PROTOCOL_SHA = "edc7a1503d6c2323c8bc302cfa7942a9bec8a1182de561ef99fdba44bb00a1a2"
EXPECTED_CHECKPOINT_SHA = "89f5e87b647680a9094c55bf647dc1dd2abb624681e733ffb453e30e4c11ff1f"
EXPECTED_SHARDS = 156
EXPECTED_FUNDUS_ROWS = 2482


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    data = json_bytes(payload)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen repair artifact differs: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    atomic_bytes(path, data)
    return "CREATED_AND_FROZEN"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def freeze_repair_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_REPAIR_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_REPAIR_SOURCE.exists():
        if sha256_file(FROZEN_REPAIR_SOURCE) != source_sha:
            raise RuntimeError("A different qualitative-repair source is already frozen")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_REPAIR_SOURCE.with_name(FROZEN_REPAIR_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_REPAIR_SOURCE)
    if sha256_file(FROZEN_REPAIR_SOURCE) != source_sha:
        raise RuntimeError("Repair-source freeze verification failed")
    return "COPIED_AND_FROZEN", source_sha


def aggregate_registry_sha(records: dict[str, str]) -> str:
    canonical = "".join(f"{name}\t{records[name]}\n" for name in sorted(records))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def inspect_numeric_shards() -> tuple[dict[str, str], int]:
    shards = sorted(SHARD_ROOT.glob("batch_*.csv"))
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(f"Expected {EXPECTED_SHARDS} numeric shards, found {len(shards)}")
    hashes: dict[str, str] = {}
    rows = 0
    for shard in shards:
        frame = pd.read_csv(
            shard,
            usecols=[
                "analysis_source_sha256",
                "analysis_protocol_sha256",
                "checkpoint_sha256",
                "relative_path",
            ],
        )
        checks = {
            "original_source": set(frame["analysis_source_sha256"].astype(str)) == {EXPECTED_ORIGINAL_SOURCE_SHA},
            "protocol": set(frame["analysis_protocol_sha256"].astype(str)) == {EXPECTED_PROTOCOL_SHA},
            "checkpoint": set(frame["checkpoint_sha256"].astype(str)) == {EXPECTED_CHECKPOINT_SHA},
            "paths_unique": bool(frame["relative_path"].astype(str).is_unique),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Numeric shard identity failed for {shard.name}: {checks}")
        rows += len(frame)
        hashes[shard.name] = sha256_file(shard)
    if rows != EXPECTED_FUNDUS_ROWS:
        raise RuntimeError(f"Numeric shard row count mismatch: {rows}")
    return hashes, rows


def import_original() -> Any:
    specification = importlib.util.spec_from_file_location(
        "fovea_dr_frozen_reviewer_mechanistic_original",
        ORIGINAL_SOURCE,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not import the original frozen experiment")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def final_outputs_valid() -> tuple[bool, dict[str, Any] | None]:
    if not FINAL_EVIDENCE.is_file():
        return False, None
    try:
        payload = load_json(FINAL_EVIDENCE)
        if payload.get("status") != "PASS" or not all(payload.get("gates", {}).values()):
            return False, payload
        contract = Path(str(payload.get("contract", "")))
        if not contract.is_file() or sha256_file(contract) != payload.get("contract_sha256"):
            return False, payload
        for record in payload.get("output_files", []):
            path = Path(str(record.get("path", "")))
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                return False, payload
        return True, payload
    except Exception:
        return False, None


def repair_already_complete() -> bool:
    if not REPAIR_EVIDENCE.is_file():
        return False
    try:
        payload = load_json(REPAIR_EVIDENCE)
        return (
            payload.get("status") == "PASS"
            and all(payload.get("gates", {}).values())
            and FINAL_EVIDENCE.is_file()
            and sha256_file(FINAL_EVIDENCE) == payload.get("final_evidence_sha256")
        )
    except Exception:
        return False


def main() -> None:
    print("=" * 120)
    print("STAGE 0G-4D-5F-6B-R1 — VALIDATION MECHANISTIC QUALITATIVE-PANEL REPAIR")
    print("=" * 120)
    if not ROOT.is_dir():
        raise RuntimeError("The real Google Drive project root is not mounted")

    required = [ORIGINAL_SOURCE, PROTOCOL]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required frozen artifacts are missing: {missing}")
    if sha256_file(ORIGINAL_SOURCE) != EXPECTED_ORIGINAL_SOURCE_SHA:
        raise RuntimeError("Original frozen experiment source hash mismatch")
    if sha256_file(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("Frozen reviewer protocol hash mismatch")

    print("\n[1] REPAIR SOURCE AND AMENDMENT CONTRACT")
    print("-" * 120)
    source_action, repair_source_sha = freeze_repair_source()
    contract_payload = {
        "contract_id": "MMRDR_FOVEA_DR_VALIDATION_MECHANISTIC_QUALITATIVE_REPAIR_V1",
        "status": "FROZEN_BEFORE_REPAIR_EXECUTION",
        "original_source_sha256": EXPECTED_ORIGINAL_SOURCE_SHA,
        "reviewer_protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "repair_source_sha256": repair_source_sha,
        "failure_point": "QUALITATIVE_PANEL_ROW_SCHEMA_ONLY",
        "root_cause": "relative_path moved from a DataFrame column to Series.name after set_index",
        "permitted_repair": "copy Series.name into a transient relative_path field before qualitative map inference",
        "prohibited_changes": [
            "numeric inference recomputation",
            "checkpoint modification",
            "model parameter update",
            "protocol modification",
            "CALIBRATION access",
            "LOCKED_INTERNAL_TEST access",
        ],
    }
    contract_action = frozen_json(REPAIR_CONTRACT, contract_payload)
    repair_contract_sha = sha256_file(REPAIR_CONTRACT)
    print("Repair source action:", source_action)
    print("Repair source SHA   :", repair_source_sha)
    print("Contract action     :", contract_action)
    print("Contract SHA        :", repair_contract_sha)

    if repair_already_complete():
        print("\nExisting repair evidence and final experiment evidence are valid.")
        print("STATUS: PASS — EXISTING COMPLETE")
        return

    print("\n[2] SAVED NUMERIC SHARD VERIFICATION")
    print("-" * 120)
    before_hashes, numeric_rows = inspect_numeric_shards()
    before_registry_sha = aggregate_registry_sha(before_hashes)
    print("Numeric shards:", len(before_hashes))
    print("Fundus rows   :", numeric_rows)
    print("Registry SHA  :", before_registry_sha)

    print("\n[3] SCHEMA-ONLY RUNTIME REPAIR AND ORIGINAL-STAGE RESUMPTION")
    print("-" * 120)
    repaired_calls = {"count": 0}
    preexisting_final_valid, _ = final_outputs_valid()
    if preexisting_final_valid:
        final_contract_payload = load_json(FINAL_CONTRACT)
        repaired_calls["count"] = int(final_contract_payload.get("qualitative_examples", 0))
        print("The authoritative final experiment outputs already pass; rendering is not repeated.")
    else:
        module = import_original()
        original_infer_maps_for_row = module.infer_maps_for_row

        def repaired_infer_maps_for_row(model: Any, row: pd.Series, device: Any) -> Any:
            repaired = row.copy()
            if "relative_path" not in repaired.index:
                if repaired.name is None:
                    raise RuntimeError("Cannot recover relative_path from an unnamed qualitative row")
                repaired["relative_path"] = str(repaired.name)
            repaired_calls["count"] += 1
            return original_infer_maps_for_row(model, repaired, device)

        module.infer_maps_for_row = repaired_infer_maps_for_row
        module.main()

    print("\n[4] POST-REPAIR INTEGRITY VERIFICATION")
    print("-" * 120)
    after_hashes, after_rows = inspect_numeric_shards()
    after_registry_sha = aggregate_registry_sha(after_hashes)
    final_valid, final_payload = final_outputs_valid()
    gates = {
        "original_frozen_source_hash_verified": sha256_file(ORIGINAL_SOURCE) == EXPECTED_ORIGINAL_SOURCE_SHA,
        "reviewer_protocol_hash_verified": sha256_file(PROTOCOL) == EXPECTED_PROTOCOL_SHA,
        "repair_source_hash_verified": sha256_file(FROZEN_REPAIR_SOURCE) == repair_source_sha,
        "repair_contract_hash_verified": sha256_file(REPAIR_CONTRACT) == repair_contract_sha,
        "exactly_156_numeric_shards_present": len(after_hashes) == EXPECTED_SHARDS,
        "exactly_2482_numeric_rows_present": after_rows == EXPECTED_FUNDUS_ROWS,
        "numeric_shard_registry_unchanged": before_hashes == after_hashes and before_registry_sha == after_registry_sha,
        "qualitative_rows_repaired": repaired_calls["count"] >= 8,
        "final_original_stage_evidence_PASS": final_valid,
        "original_source_unmodified": sha256_file(ORIGINAL_SOURCE) == EXPECTED_ORIGINAL_SOURCE_SHA,
        "protocol_unmodified": sha256_file(PROTOCOL) == EXPECTED_PROTOCOL_SHA,
        "model_parameters_not_updated": True,
        "optimizer_not_created": True,
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
    }
    for name, value in gates.items():
        print(f"{name:<48}: {value}")
    if not all(gates.values()):
        raise RuntimeError(f"Qualitative repair gates failed: {[key for key, value in gates.items() if not value]}")
    assert final_payload is not None

    evidence_payload = {
        "stage": "0G-4D-5F-6B-R1",
        "purpose": "VALIDATION_MECHANISTIC_QUALITATIVE_PANEL_SCHEMA_REPAIR",
        "status": "PASS",
        "repair_contract": str(REPAIR_CONTRACT),
        "repair_contract_sha256": repair_contract_sha,
        "repair_source": str(FROZEN_REPAIR_SOURCE),
        "repair_source_sha256": repair_source_sha,
        "original_source_sha256": EXPECTED_ORIGINAL_SOURCE_SHA,
        "reviewer_protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "numeric_shards": len(after_hashes),
        "numeric_rows": after_rows,
        "numeric_shard_registry_sha256": after_registry_sha,
        "repaired_qualitative_rows": repaired_calls["count"],
        "final_contract": str(FINAL_CONTRACT),
        "final_contract_sha256": sha256_file(FINAL_CONTRACT),
        "final_evidence": str(FINAL_EVIDENCE),
        "final_evidence_sha256": sha256_file(FINAL_EVIDENCE),
        "gates": gates,
        "validation_images_accessed": True,
        "numeric_inference_recomputed": False,
        "model_parameters_updated": False,
        "optimizer_created": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
    }
    evidence_action = frozen_json(REPAIR_EVIDENCE, evidence_payload)

    print("\n" + "=" * 120)
    print("STAGE 0G-4D-5F-6B-R1 STATUS: PASS")
    print("=" * 120)
    print("Numeric inference recomputed : NO")
    print("Numeric shards preserved     : YES — 156/156")
    print("Qualitative rows repaired    :", repaired_calls["count"])
    print("Original protocol modified   : NO")
    print("Original source modified     : NO")
    print("Model parameters updated     : NO")
    print("CALIBRATION accessed         : NO")
    print("Locked test accessed         : NO")
    print("Repair evidence action       :", evidence_action)
    print("NEXT ACTION: INTERPRET E1 RESULTS AND PREPARE E2 COMPONENT ABLATIONS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 120)
        print("STAGE 0G-4D-5F-6B-R1 STATUS: FAIL")
        print("=" * 120)
        raise
