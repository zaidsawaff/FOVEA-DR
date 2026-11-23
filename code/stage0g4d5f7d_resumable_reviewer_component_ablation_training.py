#!/usr/bin/env python3
"""Resumable final training adapter for the three reviewer E2 ablations.

The adapter reuses the already frozen and field-tested six-variant training
engine after verifying its SHA-256, while replacing only the protocol,
implementation, registry, output namespace, and AMP policy required by E2.
All three runs start from the official EfficientNetV2-S ImageNet checkpoint;
the disposable 64-step preflight weights are never loaded.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import types
from typing import Any


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
CODE = ROOT / "Code/FOVEA_DR/v1_1"
METADATA = ROOT / "Metadata"
EVIDENCE = ROOT / "Evidence"

PARENT_RUNNER = CODE / "stage0g4d5f3a_resumable_final_variant_training.py"
FROZEN_RUNNER = CODE / "stage0g4d5f7d_resumable_reviewer_component_ablation_training.py"
RUNNER_CONTRACT = METADATA / "mmrdr_fovea_dr_reviewer_E2_final_training_runner_v1.json"
RUNNER_EVIDENCE = EVIDENCE / "stage0g4d5f7d_reviewer_E2_final_training_runner_freeze.json"

REVIEWER_PROTOCOL = METADATA / "mmrdr_fovea_dr_reviewer_experiment_protocol_v1.json"
CANDIDATE = METADATA / "mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
TRAINING_PROTOCOL = METADATA / "mmrdr_fovea_dr_training_protocol_v1.json"
SPLIT = ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
SAMPLER = ROOT / "Splits/mmrdr_fovea_dr_TRAIN_sampler_weights_v1.csv"
LESION_WEIGHTS = METADATA / "mmrdr_fovea_dr_lesion_pos_weights_v1.csv"
TRAIN_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_TRAIN_local_cache_manifest_v1.csv"
VALIDATION_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_VALIDATION_local_cache_manifest_v1.csv"

IMPLEMENTATION_SOURCE = CODE / "stage0g4d5f7a_r1_reviewer_component_ablation_amp_safe_repair.py"
IMPLEMENTATION_CONTRACT = METADATA / "mmrdr_fovea_dr_reviewer_component_ablation_amp_safe_repair_v1.json"
IMPLEMENTATION_EVIDENCE = EVIDENCE / "stage0g4d5f7a_r1_reviewer_component_ablation_amp_safe_repair.json"
WEIGHT_EVIDENCE = EVIDENCE / "stage0g4d5f1c_official_comparator_weight_acquisition.json"
CACHE_EVIDENCE = EVIDENCE / "stage0g4d5f1b_comparator_development_cache_restore.json"
PREFLIGHT_RUNNER = CODE / "stage0g4d5f7b_r1_reviewer_component_ablation_64step_preflight.py"
COMPLETION_SOURCE = CODE / "stage0g4d5f7c_reviewer_component_ablation_preflight_completion_freeze.py"
RAW_PREFLIGHT_REGISTRY = (
    ROOT
    / "Runs/FOVEA_DR/reviewer_experiments/E2_component_ablation_preflight64"
    / "mmrdr_fovea_dr_reviewer_E2_preflight64_registry_v1.csv"
)
COMPATIBLE_REGISTRY = ROOT / "Audit/mmrdr_fovea_dr_reviewer_E2_final_training_registry_v1.csv"
COMPLETION_CONTRACT = METADATA / "mmrdr_fovea_dr_reviewer_E2_preflight_completion_v1.json"
COMPLETION_EVIDENCE = EVIDENCE / "stage0g4d5f7c_reviewer_component_ablation_preflight_completion_freeze.json"

RUN_BASE = ROOT / "Runs/FOVEA_DR/reviewer_experiments/E2_component_ablation_final_seed2026"
CHECKPOINT_BASE = ROOT / "Checkpoints/FOVEA_DR/reviewer_experiments/E2_component_ablation_final_seed2026"
PREFLIGHT_ROOT = ROOT / "Runs/FOVEA_DR/reviewer_experiments/E2_component_ablation_preflight64"

VARIANTS = [
    "NO_FIELD_OF_VIEW_ENCODING",
    "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE",
    "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE",
]

EXPECTED_PARENT_RUNNER_SHA = "b3bc733b2204a1657a6d1f62226fb13f04bc3c61d953334bfd04329fc62e3495"
EXPECTED_SHA = {
    REVIEWER_PROTOCOL: "edc7a1503d6c2323c8bc302cfa7942a9bec8a1182de561ef99fdba44bb00a1a2",
    CANDIDATE: "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181",
    TRAINING_PROTOCOL: "5b16e0c8565b86d3cac603f416ee0b7f03055ab9a0eb9ac9e3fb007632329b40",
    SPLIT: "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140",
    TRAIN_MANIFEST: "32f9a3b6b17916b78f591007a85ce2e1910c9c4d2b1294af1c391b19f476bd15",
    VALIDATION_MANIFEST: "d94c5613b6d030dd6e99246b1b72f77746b5223139a4597bef733eb38a5786af",
    IMPLEMENTATION_SOURCE: "bd7358f402d3488f4a829eee85a823456ee6e87b1eb0d676d229b383dbc1a357",
    WEIGHT_EVIDENCE: "84e5f89d29c11175169118b7010e49a91e36eb2987dc84c60961796f54894d54",
    CACHE_EVIDENCE: "1dadfa4463347b97e36cca003649991c585c48e8027b539e67047584526fee34",
    PREFLIGHT_RUNNER: "535ce34324acfe3131e67937cf57efd090c5d6a2f3459b02d0db7fc6a5e59cdd",
    COMPLETION_SOURCE: "6386025030c8f05c849f20e077258f26bccc85d68d706b5cd5c7edabeea17bc8",
    RAW_PREFLIGHT_REGISTRY: "8883da93cd10bb66d115fa322cf55376d3d6eeeb85f32967e6717f97d9f43870",
    COMPLETION_CONTRACT: "ee149800c023fc5130985049bf14115c3b8cafcaac6948c24d1b1cbeba1e6fc4",
    COMPLETION_EVIDENCE: "674cf3e647338057766a0eda5c2d4a3ca00d954f0dbcbab074c50259a4300b4d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def frozen_bytes(path: Path, data: bytes, label: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"A different frozen {label} already exists: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    if path.read_bytes() != data:
        raise RuntimeError(f"Frozen {label} write/read-back failed")
    return "CREATED_AND_FROZEN"


def evidence_passes(payload: dict[str, Any]) -> bool:
    gates = payload.get("gates")
    return bool(
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and gates
        and all(value is True for value in gates.values())
    )


def freeze_adapter_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_RUNNER.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_RUNNER.exists():
        if sha256_file(FROZEN_RUNNER) != source_sha:
            raise RuntimeError("A different E2 final-training runner is already frozen")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_RUNNER.with_name(FROZEN_RUNNER.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_RUNNER)
    if sha256_file(FROZEN_RUNNER) != source_sha:
        raise RuntimeError("E2 final-training runner source freeze failed")
    return "COPIED_AND_FROZEN_BEFORE_E2_FINAL_TRAINING", source_sha


def build_compatible_registry() -> tuple[str, str]:
    with RAW_PREFLIGHT_REGISTRY.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if [row.get("variant_id") for row in rows] != VARIANTS:
        raise RuntimeError("Raw E2 preflight registry order is inconsistent")
    for row in rows:
        if row.get("status") != "PASS":
            raise RuntimeError(f"E2 preflight is not PASS: {row.get('variant_id')}")
        row["all_checks_passed"] = "True"
        row["preflight_log_sha256"] = row["pilot_log_sha256"]
        row["total_parameters"] = row["parameter_count"]
    added = ["all_checks_passed", "preflight_log_sha256", "total_parameters"]
    final_fields = fields + [name for name in added if name not in fields]
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=final_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    action = frozen_bytes(COMPATIBLE_REGISTRY, data, "E2 final-training registry")
    return action, sha256_file(COMPATIBLE_REGISTRY)


def prepare_runner_freeze() -> tuple[str, str, str]:
    parent_checks = {
        "parent_final_training_engine_hash_verified": (
            PARENT_RUNNER.is_file()
            and sha256_file(PARENT_RUNNER) == EXPECTED_PARENT_RUNNER_SHA
        ),
        **{
            f"hash_verified::{path.name}": path.is_file() and sha256_file(path) == expected
            for path, expected in EXPECTED_SHA.items()
        },
        "implementation_contract_exists": IMPLEMENTATION_CONTRACT.is_file(),
        "implementation_evidence_PASS": (
            IMPLEMENTATION_EVIDENCE.is_file()
            and evidence_passes(load_json(IMPLEMENTATION_EVIDENCE))
        ),
    }
    failed = [name for name, value in parent_checks.items() if not value]
    if failed:
        raise RuntimeError("E2 final-training parent verification failed:\n" + "\n".join(failed))

    source_action, source_sha = freeze_adapter_source()
    registry_action, registry_sha = build_compatible_registry()

    existing_contract = load_json(RUNNER_CONTRACT) if RUNNER_CONTRACT.is_file() else None
    existing_evidence = load_json(RUNNER_EVIDENCE) if RUNNER_EVIDENCE.is_file() else None
    if existing_contract is not None or existing_evidence is not None:
        if existing_contract is None or existing_evidence is None:
            raise RuntimeError("Only one of the E2 runner contract/evidence files exists")
        existing_valid = bool(
            existing_contract.get("status") == "FROZEN_BEFORE_FINAL_VARIANT_TRAINING"
            and existing_contract.get("runner_source_sha256") == source_sha
            and existing_contract.get("variants_in_mandatory_order") == VARIANTS
            and existing_contract.get("AMP_initial_scale") == 1024.0
            and existing_contract.get("AMP_growth_interval") == 1000000
            and evidence_passes(existing_evidence)
            and existing_evidence.get("runner_source_sha256") == source_sha
            and existing_evidence.get("runner_contract_sha256") == sha256_file(RUNNER_CONTRACT)
        )
        if not existing_valid:
            raise RuntimeError("Existing E2 final-training runner freeze is inconsistent")
        return source_action, registry_action, source_sha

    no_checkpoints = not any(CHECKPOINT_BASE.glob("*/last_resume.pt"))
    freeze_gates = {
        **parent_checks,
        "compatible_registry_created_and_verified": (
            COMPATIBLE_REGISTRY.is_file()
            and sha256_file(COMPATIBLE_REGISTRY) == registry_sha
        ),
        "exactly_three_variants_locked": len(VARIANTS) == 3,
        "no_E2_final_checkpoint_existed_at_freeze": no_checkpoints,
        "model_not_instantiated": True,
        "optimizer_not_created": True,
        "medical_images_not_accessed": True,
        "final_training_not_started": True,
        "VALIDATION_not_accessed": True,
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
    }
    failed = [name for name, value in freeze_gates.items() if not value]
    if failed:
        raise RuntimeError("E2 runner freeze gates failed:\n" + "\n".join(failed))

    contract = {
        "contract_id": "MMRDR_FOVEA_DR_REVIEWER_E2_FINAL_TRAINING_RUNNER_V1",
        "stage": "0G-4D-5F-7D",
        "status": "FROZEN_BEFORE_FINAL_VARIANT_TRAINING",
        "runner": str(FROZEN_RUNNER),
        "runner_source_sha256": source_sha,
        "parent_training_engine": str(PARENT_RUNNER),
        "parent_training_engine_sha256": EXPECTED_PARENT_RUNNER_SHA,
        "variants_in_mandatory_order": VARIANTS,
        "run_seed": 2026,
        "maximum_epochs": 30,
        "early_stopping_patience": 6,
        "primary_QWK_minimum_delta": 0.002,
        "optimizer_steps_per_epoch": 415,
        "micro_batch": 16,
        "gradient_accumulation": 2,
        "effective_batch": 32,
        "AMP_dtype": "float16",
        "AMP_initial_scale": 1024.0,
        "AMP_growth_interval": 1000000,
        "fresh_initialization": "OFFICIAL_EFFICIENTNETV2S_IMAGENET_ONLY",
        "preflight_checkpoint_initialization_allowed": False,
        "preflight_registry": str(COMPATIBLE_REGISTRY),
        "preflight_registry_sha256": registry_sha,
        "TRAIN_access": "OPTIMIZATION_ONLY",
        "VALIDATION_access": "EARLY_STOPPING_AND_MODEL_SELECTION_ONLY",
        "CALIBRATION_access": "PROHIBITED_DURING_FINAL_TRAINING",
        "LOCKED_INTERNAL_TEST_access": "PROHIBITED",
        "save_order_recovery": "LAST_CHECKPOINT_CONTAINS_AUTHORITATIVE_FULL_HISTORY_AND_STEP_LOG",
    }
    contract_action = frozen_bytes(RUNNER_CONTRACT, json_bytes(contract), "E2 runner contract")
    contract_sha = sha256_file(RUNNER_CONTRACT)
    evidence = {
        "stage": "0G-4D-5F-7D",
        "status": "PASS",
        "purpose": "RESUMABLE_REVIEWER_E2_FINAL_TRAINING_RUNNER_FREEZE",
        "runner": str(FROZEN_RUNNER),
        "runner_source_sha256": source_sha,
        "runner_contract": str(RUNNER_CONTRACT),
        "runner_contract_sha256": contract_sha,
        "compatible_registry": str(COMPATIBLE_REGISTRY),
        "compatible_registry_sha256": registry_sha,
        "gates": {name: bool(value) for name, value in freeze_gates.items()},
        "model_instantiated": False,
        "optimizer_created": False,
        "medical_images_accessed": False,
        "final_training_started": False,
        "validation_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "next_required_stage": "START_NO_FIELD_OF_VIEW_ENCODING_FINAL_TRAINING",
    }
    evidence_action = frozen_bytes(RUNNER_EVIDENCE, json_bytes(evidence), "E2 runner evidence")
    print("Runner source action :", source_action)
    print("Registry action      :", registry_action)
    print("Contract action      :", contract_action)
    print("Evidence action      :", evidence_action)
    print("Runner SHA-256       :", source_sha)
    print("Registry SHA-256     :", registry_sha)
    print("Contract SHA-256     :", contract_sha)
    print("Evidence SHA-256     :", sha256_file(RUNNER_EVIDENCE))
    return source_action, registry_action, source_sha


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
        "evidence": EVIDENCE / f"stage0g4d5f7d_{variant_id.lower()}_latest_session.json",
    }


def selected_preflight_paths(variant_id: str) -> tuple[Path, Path]:
    return (
        PREFLIGHT_ROOT / variant_id / "optimizer_step_log.csv",
        EVIDENCE / f"stage0g4d5f7b_{variant_id.lower()}_64step_preflight.json",
    )


def import_repaired_implementation() -> Any:
    specification = importlib.util.spec_from_file_location(
        "fovea_dr_reviewer_E2_final_implementation",
        IMPLEMENTATION_SOURCE,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not import the frozen E2 implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.PLAIN_VARIANTS = set()
    module.build_comparison_optimizer = module.build_reviewer_optimizer
    return module


def load_parent_engine(source_sha: str) -> types.ModuleType:
    source = PARENT_RUNNER.read_text(encoding="utf-8")
    replacements = {
        '"six_variants_match_protocol_order": protocol_ids == VARIANTS,':
            '"three_E2_variants_match_protocol_order": protocol_ids == VARIANTS,',
        '"preflight_registry_has_exact_six_rows": len(registry) == 6,':
            '"preflight_registry_has_exact_three_rows": len(registry) == 3,',
        'scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)':
            'scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0, growth_interval=1000000)',
        'STAGE 0G-4D-5F-3A': 'STAGE 0G-4D-5F-7D',
        'FOVEA_DR_POSTHOC_VARIANT': 'FOVEA_DR_REVIEWER_E2_VARIANT',
    }
    for old, new in replacements.items():
        count = source.count(old)
        if count < 1:
            raise RuntimeError(f"Expected parent-engine token was not found: {old}")
        if old.startswith('"six_') or old.startswith('"preflight_') or old.startswith("scaler ="):
            if count != 1:
                raise RuntimeError(f"Expected exactly one parent-engine token: {old}; found {count}")
        source = source.replace(old, new)

    module = types.ModuleType("fovea_dr_reviewer_E2_resumable_engine")
    module.__file__ = str(PARENT_RUNNER)
    module.__package__ = None
    exec(compile(source, str(PARENT_RUNNER), "exec"), module.__dict__)

    original_load_json = module.load_json

    def adapted_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        target = Path(path)
        if target == REVIEWER_PROTOCOL:
            e2 = [
                item for item in payload.get("experiments_in_locked_order", [])
                if item.get("id") == "E2_COMPONENT_ABLATIONS"
            ]
            variants = e2[0].get("variants") if len(e2) == 1 else []
            transformed = dict(payload)
            transformed["comparators"] = [{"id": item} for item in variants]
            transformed["ablations"] = []
            return transformed
        if target == COMPLETION_CONTRACT:
            transformed = dict(payload)
            transformed["status"] = "SIX_VARIANTS_READY_FOR_INDEPENDENT_FINAL_TRAINING"
            transformed["variants_in_locked_order"] = payload.get("variant_order")
            transformed["preflight_weights_must_not_be_reused"] = (
                payload.get("preflight_weights_allowed_for_final_initialization") is False
            )
            return transformed
        return payload

    def adapter_freeze_runner() -> tuple[str, str]:
        if not FROZEN_RUNNER.is_file() or sha256_file(FROZEN_RUNNER) != source_sha:
            raise RuntimeError("Frozen E2 final-training runner is missing or changed")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha

    registry_sha = sha256_file(COMPATIBLE_REGISTRY)
    expected = {
        REVIEWER_PROTOCOL: EXPECTED_SHA[REVIEWER_PROTOCOL],
        CANDIDATE: EXPECTED_SHA[CANDIDATE],
        TRAINING_PROTOCOL: EXPECTED_SHA[TRAINING_PROTOCOL],
        SPLIT: EXPECTED_SHA[SPLIT],
        TRAIN_MANIFEST: EXPECTED_SHA[TRAIN_MANIFEST],
        VALIDATION_MANIFEST: EXPECTED_SHA[VALIDATION_MANIFEST],
        IMPLEMENTATION_SOURCE: EXPECTED_SHA[IMPLEMENTATION_SOURCE],
        WEIGHT_EVIDENCE: EXPECTED_SHA[WEIGHT_EVIDENCE],
        CACHE_EVIDENCE: EXPECTED_SHA[CACHE_EVIDENCE],
        PREFLIGHT_RUNNER: EXPECTED_SHA[PREFLIGHT_RUNNER],
        COMPLETION_SOURCE: EXPECTED_SHA[COMPLETION_SOURCE],
        COMPATIBLE_REGISTRY: registry_sha,
        COMPLETION_CONTRACT: EXPECTED_SHA[COMPLETION_CONTRACT],
        COMPLETION_EVIDENCE: EXPECTED_SHA[COMPLETION_EVIDENCE],
    }

    patches = {
        "PROTOCOL": REVIEWER_PROTOCOL,
        "CANDIDATE": CANDIDATE,
        "TRAINING_PROTOCOL": TRAINING_PROTOCOL,
        "SPLIT": SPLIT,
        "SAMPLER": SAMPLER,
        "LESION_WEIGHTS": LESION_WEIGHTS,
        "TRAIN_MANIFEST": TRAIN_MANIFEST,
        "VALIDATION_MANIFEST": VALIDATION_MANIFEST,
        "IMPLEMENTATION_SOURCE": IMPLEMENTATION_SOURCE,
        "IMPLEMENTATION_CONTRACT": IMPLEMENTATION_CONTRACT,
        "IMPLEMENTATION_EVIDENCE": IMPLEMENTATION_EVIDENCE,
        "WEIGHT_EVIDENCE": WEIGHT_EVIDENCE,
        "CACHE_EVIDENCE": CACHE_EVIDENCE,
        "PREFLIGHT_RUNNER": PREFLIGHT_RUNNER,
        "COMPLETION_SOURCE": COMPLETION_SOURCE,
        "PREFLIGHT_REGISTRY": COMPATIBLE_REGISTRY,
        "COMPLETION_CONTRACT": COMPLETION_CONTRACT,
        "COMPLETION_EVIDENCE": COMPLETION_EVIDENCE,
        "FROZEN_RUNNER": FROZEN_RUNNER,
        "RUNNER_CONTRACT": RUNNER_CONTRACT,
        "RUNNER_FREEZE_EVIDENCE": RUNNER_EVIDENCE,
        "RUN_BASE": RUN_BASE,
        "CHECKPOINT_BASE": CHECKPOINT_BASE,
        "EXPECTED_SHA": expected,
        "VARIANTS": list(VARIANTS),
        "BACKBONE_FOR_VARIANT": {variant: "EfficientNetV2-S" for variant in VARIANTS},
        "run_paths": run_paths,
        "selected_preflight_paths": selected_preflight_paths,
        "import_implementation": import_repaired_implementation,
        "load_json": adapted_load_json,
        "freeze_runner": adapter_freeze_runner,
    }
    for name, value in patches.items():
        setattr(module, name, value)
    return module


def main() -> None:
    if "--freeze-only" in sys.argv:
        raise RuntimeError(
            "This E2 adapter freezes its contract automatically before the first "
            "training run; use --variant instead of --freeze-only."
        )
    print("=" * 120)
    print("STAGE 0G-4D-5F-7D — RESUMABLE REVIEWER E2 FINAL TRAINING")
    print("=" * 120)
    _, _, source_sha = prepare_runner_freeze()
    engine = load_parent_engine(source_sha)
    engine.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 120, file=sys.stderr)
        print("STAGE 0G-4D-5F-7D STATUS: FAIL", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
