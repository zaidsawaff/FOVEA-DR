#!/usr/bin/env python3
"""AMP-safe numerical amendment for the global-pooled E2 ablation.

The architecture and loss definition are unchanged. Only the sigmoid evidence
probabilities are promoted to float32 and bounded away from exact 0/1 before
entropy-based reliability calculations. No medical images are accessed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
ORIGINAL_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f7a_reviewer_component_ablation_implementation_audit.py"
ORIGINAL_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_reviewer_component_ablation_implementation_v1.json"
ORIGINAL_EVIDENCE = ROOT / "Evidence/stage0g4d5f7a_reviewer_component_ablation_implementation_audit.json"
CANDIDATE = ROOT / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
WEIGHT_EVIDENCE = ROOT / "Evidence/stage0g4d5f1c_official_comparator_weight_acquisition.json"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f7a_r1_reviewer_component_ablation_amp_safe_repair.py"
CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_reviewer_component_ablation_amp_safe_repair_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5f7a_r1_reviewer_component_ablation_amp_safe_repair.json"

EXPECTED_ORIGINAL_SOURCE_SHA = "d74b6ebfe866905e1edcefff2a298a5bed78d578621c74c30d95f155dd2ef1bd"
EXPECTED_CANDIDATE_SHA = "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181"
EXPECTED_WEIGHT_EVIDENCE_SHA = "84e5f89d29c11175169118b7010e49a91e36eb2987dc84c60961796f54894d54"
EXPECTED_WEIGHT_SHA = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"

REVIEWER_VARIANTS = [
    "NO_FIELD_OF_VIEW_ENCODING",
    "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE",
    "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE",
]
REPAIRED_VARIANT = "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE"
RUN_SEED = 2026
EVIDENCE_EPSILON = 1.0e-6


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
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    atomic_bytes(path, data)
    return "CREATED_AND_FROZEN"


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


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different AMP-safe E2 repair source is already frozen")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    if sha256_file(FROZEN_SOURCE) != source_sha:
        raise RuntimeError("AMP-safe E2 repair source freeze failed")
    return "COPIED_AND_FROZEN", source_sha


def import_original() -> Any:
    specification = importlib.util.spec_from_file_location(
        "fovea_dr_reviewer_e2_original_implementation",
        ORIGINAL_SOURCE,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not import the original frozen E2 implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


ORIGINAL = None


def get_original() -> Any:
    global ORIGINAL
    if ORIGINAL is None:
        ORIGINAL = import_original()
    return ORIGINAL


def get_core() -> Any:
    return get_original().get_core()


class AMPSafeGlobalPooledEvidenceBank(nn.Module):
    """Original global-pooled evidence bank with float32 bounded probabilities."""

    def __init__(self, dimension: int = 128, lesions: int = 7) -> None:
        super().__init__()
        self.lesions = lesions
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
        )
        self.presence_head = nn.Linear(dimension, lesions)

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        pooled = self.projection(pooled)
        presence_logits = self.presence_head(pooled)

        # Minimal numerical amendment: entropy receives representable float32
        # probabilities strictly inside (0, 1), even under outer AMP autocast.
        presence = torch.sigmoid(presence_logits.float()).clamp(
            EVIDENCE_EPSILON,
            1.0 - EVIDENCE_EPSILON,
        )

        height, width = feature_map.shape[-2:]
        spatial_maps = presence[:, :, None, None].expand(-1, -1, height, width)
        embeddings = pooled[:, None, :].expand(-1, self.lesions, -1)
        return {
            "presence_logits": presence_logits,
            "presence": presence,
            "spatial_maps": spatial_maps,
            "burden": presence,
            "embeddings": embeddings,
            "evidence_strength": presence,
            "spatial_entropy": torch.ones_like(presence),
        }


class AMPSafeNoLesionQueriesGlobalPooledEvidenceModel(nn.Module):
    """Exact E2 global-pooling ablation with the numerical amendment."""

    def __init__(self) -> None:
        super().__init__()
        core = get_core()
        self.model = core.FOVEADRTrainingModel()
        self.model.lesion_bank = AMPSafeGlobalPooledEvidenceBank()

    def forward_mixed(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model.forward_mixed(*args, **kwargs)


def build_variant(variant_id: str) -> nn.Module:
    if variant_id == REPAIRED_VARIANT:
        return AMPSafeNoLesionQueriesGlobalPooledEvidenceModel()
    return get_original().build_variant(variant_id)


def variant_loss_weights(
    variant_id: str,
    candidate_weights: dict[str, Any],
) -> dict[str, float]:
    return get_original().variant_loss_weights(variant_id, candidate_weights)


def compute_variant_losses(
    variant_id: str,
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    lesion_pos_weight: torch.Tensor,
    candidate_weights: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return get_original().compute_variant_losses(
        variant_id,
        output,
        batch,
        lesion_pos_weight,
        candidate_weights,
    )


def build_reviewer_optimizer(
    model: nn.Module,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    return get_original().build_reviewer_optimizer(model)


def official_weight_record() -> tuple[dict[str, Any], Path]:
    payload = load_json(WEIGHT_EVIDENCE)
    matches = [
        record for record in payload.get("official_models", [])
        if record.get("name") == "EfficientNetV2-S"
    ]
    if len(matches) != 1:
        raise RuntimeError("Unique EfficientNetV2-S weight record not found")
    record = matches[0]
    path = Path(str(record.get("checkpoint_path", "")))
    if not (
        path.is_file()
        and sha256_file(path) == EXPECTED_WEIGHT_SHA
        and record.get("checkpoint_sha256") == EXPECTED_WEIGHT_SHA
    ):
        raise RuntimeError("Official EfficientNetV2-S weight is missing or changed")
    return record, path


def main() -> None:
    print("=" * 120)
    print("STAGE 0G-4D-5F-7A-R1 — E2 GLOBAL-POOLED EVIDENCE AMP-SAFE REPAIR")
    print("=" * 120)
    if not ROOT.is_dir():
        raise RuntimeError("The real Google Drive project root is not mounted")

    original_contract_sha = (
        sha256_file(ORIGINAL_CONTRACT) if ORIGINAL_CONTRACT.is_file() else None
    )
    original_evidence = load_json(ORIGINAL_EVIDENCE) if ORIGINAL_EVIDENCE.is_file() else {}
    parent_checks = {
        "original_source_hash_verified": (
            ORIGINAL_SOURCE.is_file()
            and sha256_file(ORIGINAL_SOURCE) == EXPECTED_ORIGINAL_SOURCE_SHA
        ),
        "original_contract_exists": ORIGINAL_CONTRACT.is_file(),
        "original_audit_evidence_PASS": evidence_passes(original_evidence),
        "original_evidence_links_contract": (
            original_evidence.get("contract_sha256") == original_contract_sha
        ),
        "candidate_hash_verified": (
            CANDIDATE.is_file() and sha256_file(CANDIDATE) == EXPECTED_CANDIDATE_SHA
        ),
        "weight_evidence_hash_verified": (
            WEIGHT_EVIDENCE.is_file()
            and sha256_file(WEIGHT_EVIDENCE) == EXPECTED_WEIGHT_EVIDENCE_SHA
        ),
    }
    for name, value in parent_checks.items():
        print(f"{name:48s}: {value}")
    if not all(parent_checks.values()):
        raise RuntimeError(
            f"AMP repair parent gates failed: "
            f"{[name for name, value in parent_checks.items() if not value]}"
        )

    source_action, source_sha = freeze_source()
    weight_record, _ = official_weight_record()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU is required for the AMP repair audit")
    device = torch.device("cuda:0")
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    torch.cuda.manual_seed_all(RUN_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("\n[2] TARGETED HIGH-DYNAMIC-RANGE AMP STRESS TEST")
    print("-" * 120)
    bank = AMPSafeGlobalPooledEvidenceBank().to(device).train()
    stress_scales = torch.tensor(
        [1.0, 16.0, 64.0, 256.0],
        device=device,
    ).view(4, 1, 1, 1)
    stress_vectors = torch.randn(4, 128, 1, 1, device=device) * stress_scales
    stress_features = stress_vectors.expand(-1, -1, 64, 64)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        stress_output = bank(stress_features)
        stress_entropy = get_core().binary_entropy(
            stress_output["evidence_strength"]
        ).mean()
        stress_objective = stress_entropy + stress_output["evidence_strength"].mean()
    stress_evidence_min = float(stress_output["evidence_strength"].min())
    stress_evidence_max = float(stress_output["evidence_strength"].max())
    stress_objective.backward()
    stress_gradients = [
        parameter.grad for parameter in bank.parameters()
        if parameter.grad is not None
    ]
    stress_checks = {
        "stress_evidence_is_float32": (
            stress_output["evidence_strength"].dtype == torch.float32
        ),
        "stress_evidence_strictly_above_zero": bool(
            (stress_output["evidence_strength"] >= EVIDENCE_EPSILON).all()
        ),
        "stress_evidence_strictly_below_one": bool(
            (stress_output["evidence_strength"] <= 1.0 - EVIDENCE_EPSILON).all()
        ),
        "stress_entropy_finite": bool(torch.isfinite(stress_entropy)),
        "stress_objective_finite": bool(torch.isfinite(stress_objective)),
        "stress_gradients_present": bool(stress_gradients),
        "stress_gradients_finite": bool(stress_gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in stress_gradients
        ),
        "stress_gradients_nonzero": bool(stress_gradients) and any(
            bool(torch.count_nonzero(gradient)) for gradient in stress_gradients
        ),
    }
    for name, value in stress_checks.items():
        print(f"{name:48s}: {value}")
    del bank, stress_scales, stress_vectors, stress_features
    del stress_output, stress_entropy, stress_objective
    torch.cuda.empty_cache()

    print("\n[3] FULL MIXED-PRECISION SYNTHETIC FORWARD/BACKWARD AUDIT")
    print("-" * 120)
    original = get_original()
    model = build_variant(REPAIRED_VARIANT).to(device).train()
    batch = original.synthetic_batch(device)
    lesion_pos_weight = torch.ones(7, device=device)
    candidate_weights = load_json(CANDIDATE)["candidate_loss"]
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.forward_mixed(
            batch["image"],
            batch["fundus_mask"],
            batch["fov_mask"],
            batch["modality_index"],
            batch["quality"],
        )
        total_loss, parts = compute_variant_losses(
            REPAIRED_VARIANT,
            output,
            batch,
            lesion_pos_weight,
            candidate_weights,
        )
    total_loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    full_checks = {
        "parameter_count_preserved": (
            sum(parameter.numel() for parameter in model.parameters()) == 20694054
        ),
        "fundus_output_present": output.get("fundus") is not None,
        "oct_output_present": output.get("oct") is not None,
        "evidence_probabilities_float32": (
            output["fundus"]["lesions"]["evidence_strength"].dtype == torch.float32
        ),
        "all_loss_components_finite": all(
            bool(torch.isfinite(value)) for value in parts.values()
        ),
        "total_loss_finite": bool(torch.isfinite(total_loss)),
        "gradients_present": bool(gradients),
        "all_gradients_finite": bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        ),
        "at_least_one_gradient_nonzero": bool(gradients) and any(
            bool(torch.count_nonzero(gradient)) for gradient in gradients
        ),
    }
    for name, value in full_checks.items():
        print(f"{name:48s}: {value}")

    gates = {
        **parent_checks,
        **stress_checks,
        **full_checks,
        "repair_source_hash_verified": (
            FROZEN_SOURCE.is_file() and sha256_file(FROZEN_SOURCE) == source_sha
        ),
        "official_weight_verified": (
            weight_record.get("checkpoint_sha256") == EXPECTED_WEIGHT_SHA
        ),
        "architecture_parameter_count_unchanged": full_checks["parameter_count_preserved"],
        "loss_definition_unchanged": True,
        "optimizer_not_created": True,
        "parameters_not_updated": True,
        "medical_images_not_accessed": True,
        "TRAIN_not_accessed": True,
        "VALIDATION_not_accessed": True,
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
    }
    failed = [name for name, value in gates.items() if not value]
    if failed:
        raise RuntimeError(f"AMP-safe repair gates failed: {failed}")

    contract_payload = {
        "contract_id": "MMRDR_FOVEA_DR_REVIEWER_E2_GLOBAL_POOLED_AMP_SAFE_REPAIR_V1",
        "status": "AMP_SAFE_NUMERICAL_REPAIR_VERIFIED",
        "parent_implementation_source": str(ORIGINAL_SOURCE),
        "parent_implementation_source_sha256": EXPECTED_ORIGINAL_SOURCE_SHA,
        "parent_contract": str(ORIGINAL_CONTRACT),
        "parent_contract_sha256": original_contract_sha,
        "repair_source": str(FROZEN_SOURCE),
        "repair_source_sha256": source_sha,
        "repaired_variant": REPAIRED_VARIANT,
        "amendment": (
            "Evidence probabilities are promoted to float32 and clamped to "
            "[1e-6, 1-1e-6] before entropy-based reliability calculations."
        ),
        "architecture_changed": False,
        "parameter_count_changed": False,
        "loss_definition_changed": False,
        "official_checkpoint_sha256": EXPECTED_WEIGHT_SHA,
        "preflight_weights_reusable": False,
        "CALIBRATION_access": "PROHIBITED",
        "LOCKED_INTERNAL_TEST_access": "PROHIBITED",
    }
    contract_action = frozen_json(CONTRACT, contract_payload)
    contract_sha = sha256_file(CONTRACT)
    evidence_payload = {
        "stage": "0G-4D-5F-7A-R1",
        "purpose": "REVIEWER_E2_GLOBAL_POOLED_EVIDENCE_AMP_SAFE_REPAIR",
        "status": "PASS",
        "repair_source": str(FROZEN_SOURCE),
        "repair_source_sha256": source_sha,
        "implementation_source": str(FROZEN_SOURCE),
        "implementation_source_sha256": source_sha,
        "contract": str(CONTRACT),
        "contract_sha256": contract_sha,
        "contract_action": contract_action,
        "repaired_variant": REPAIRED_VARIANT,
        "stress_test": {
            "feature_scales": [1.0, 16.0, 64.0, 256.0],
            "evidence_min": stress_evidence_min,
            "evidence_max": stress_evidence_max,
        },
        "synthetic_total_loss": float(total_loss.detach().cpu()),
        "loss_components": {
            name: float(value.detach().cpu()) for name, value in parts.items()
        },
        "gates": gates,
        "optimizer_created": False,
        "parameters_updated": False,
        "medical_images_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "next_required_stage": "RESUME_E2_64_STEP_PREFLIGHT_WITH_REPAIRED_IMPLEMENTATION",
    }
    evidence_action = frozen_json(EVIDENCE, evidence_payload)

    print("\n" + "=" * 120)
    print("STAGE 0G-4D-5F-7A-R1 STATUS: PASS")
    print("=" * 120)
    print("Repaired variant      :", REPAIRED_VARIANT)
    print("Architecture changed  : NO")
    print("Parameter count changed: NO")
    print("Loss definition changed: NO")
    print("Optimizer created     : NO")
    print("Parameters updated    : NO")
    print("Medical images read   : NO")
    print("CALIBRATION accessed  : NO")
    print("Locked test accessed  : NO")
    print("Repair source action  :", source_action)
    print("Contract action       :", contract_action)
    print("Evidence action       :", evidence_action)
    print("NEXT ACTION: RESUME E2 64-STEP PREFLIGHT FROM VARIANT 2")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 120, file=sys.stderr)
        print("STAGE 0G-4D-5F-7A-R1 STATUS: FAIL", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
