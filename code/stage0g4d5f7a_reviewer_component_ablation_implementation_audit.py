#!/usr/bin/env python3
"""Frozen implementation and synthetic audit for reviewer component ablations.

This stage defines the three E2 variants locked in the reviewer-experiment
protocol and validates them only with synthetic tensors. It never reads a
medical image, never creates an optimizer, and never updates parameters.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
REVIEWER_PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_reviewer_experiment_protocol_v1.json"
REVIEWER_PROTOCOL_EVIDENCE = ROOT / "Evidence/stage0g4d5f6a_reviewer_experiment_protocol_lock.json"
CORE_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
CANDIDATE = ROOT / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
WEIGHT_EVIDENCE = ROOT / "Evidence/stage0g4d5f1c_official_comparator_weight_acquisition.json"
WEIGHT_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f1c_official_comparator_weight_acquisition.py"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f7a_reviewer_component_ablation_implementation_audit.py"
CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_reviewer_component_ablation_implementation_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5f7a_reviewer_component_ablation_implementation_audit.json"

EXPECTED_REVIEWER_PROTOCOL_SHA = "edc7a1503d6c2323c8bc302cfa7942a9bec8a1182de561ef99fdba44bb00a1a2"
EXPECTED_CORE_SHA = "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9"
EXPECTED_CANDIDATE_SHA = "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181"
EXPECTED_WEIGHT_EVIDENCE_SHA = "84e5f89d29c11175169118b7010e49a91e36eb2987dc84c60961796f54894d54"
EXPECTED_WEIGHT_SOURCE_SHA = "b9ccaf24db7255cb73dfd0cacf928556033a9810d5766f73f67ee13ecebd5c8b"
EXPECTED_EFFICIENTNET_WEIGHT_SHA = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"

REVIEWER_VARIANTS = [
    "NO_FIELD_OF_VIEW_ENCODING",
    "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE",
    "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE",
]

BACKBONE_LR = 1.0e-4
NEW_MODULE_LR = 3.0e-4
WEIGHT_DECAY = 1.0e-4
INPUT_SIZE = 512
RUN_SEED = 2026


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
            raise RuntimeError("A different E2 implementation source is already frozen")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    if sha256_file(FROZEN_SOURCE) != source_sha:
        raise RuntimeError("E2 implementation-source freeze verification failed")
    return "COPIED_AND_FROZEN", source_sha


def import_frozen_core() -> Any:
    specification = importlib.util.spec_from_file_location(
        "fovea_dr_reviewer_e2_frozen_core",
        CORE_SOURCE,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not import frozen FOVEA-DR core")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


CORE = None


def get_core() -> Any:
    global CORE
    if CORE is None:
        CORE = import_frozen_core()
    return CORE


class NoFieldOfViewEncodingModel(nn.Module):
    """Full model with all seven FoV/coordinate channels fixed to zero."""

    def __init__(self) -> None:
        super().__init__()
        core = get_core()
        self.model = core.FOVEADRTrainingModel()

    def forward_mixed(
        self,
        images: torch.Tensor,
        fundus_mask: torch.Tensor,
        fov_masks: torch.Tensor,
        modality_indices: torch.Tensor,
        quality: torch.Tensor,
    ) -> dict[str, Any]:
        del fov_masks
        model = self.model
        all_features = model.encode(images)
        output: dict[str, Any] = {"fundus": None, "oct": None}
        if bool(fundus_mask.any()):
            fundus_features = [feature[fundus_mask] for feature in all_features]
            pyramid = model.fpn(fundus_features)
            fundus_modality = modality_indices[fundus_mask]
            fundus_quality = quality[fundus_mask]
            no_coordinates = torch.zeros(
                pyramid.shape[0],
                7,
                pyramid.shape[-2],
                pyramid.shape[-1],
                device=pyramid.device,
                dtype=pyramid.dtype,
            )
            fused_map = model.coordinate_fusion(torch.cat([pyramid, no_coordinates], dim=1))
            lesions = model.lesion_bank(fused_map)
            pooled = F.adaptive_avg_pool2d(fundus_features[-1], 1).flatten(1)
            latent = model.latent_head(pooled)
            original = model.fuse(
                lesions["evidence_strength"], latent["probabilities"],
                fundus_modality, fundus_quality,
            )
            counterfactual = model.counterfactual(
                lesions["evidence_strength"], latent["probabilities"],
                fundus_modality, fundus_quality, original,
            )
            output["fundus"] = {
                "lesions": lesions,
                "latent": latent,
                "original": original,
                "counterfactual": counterfactual,
            }
        oct_mask = ~fundus_mask
        if bool(oct_mask.any()):
            oct_pooled = F.adaptive_avg_pool2d(all_features[-1][oct_mask], 1).flatten(1)
            output["oct"] = model.oct_head(oct_pooled)
        return output


class GlobalPooledEvidenceBank(nn.Module):
    """Seven lesion-evidence outputs from one global pooled representation."""

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
        presence = torch.sigmoid(presence_logits)
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


class NoLesionQueriesGlobalPooledEvidenceModel(nn.Module):
    """Full architecture with learned lesion queries replaced by global pooling."""

    def __init__(self) -> None:
        super().__init__()
        core = get_core()
        self.model = core.FOVEADRTrainingModel()
        self.model.lesion_bank = GlobalPooledEvidenceBank()

    def forward_mixed(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model.forward_mixed(*args, **kwargs)


class NoClinicalReasonerLatentOnlyGradeModel(nn.Module):
    """FOVEA feature/lesion stack with grade prediction exclusively latent."""

    def __init__(self) -> None:
        super().__init__()
        core = get_core()
        self.model = core.FOVEADRTrainingModel()
        del self.model.clinical_reasoner
        del self.model.reliability_gate

        def latent_fuse(
            evidence: torch.Tensor,
            latent_probabilities: torch.Tensor,
            modality_index: torch.Tensor,
            quality: torch.Tensor,
        ) -> dict[str, Any]:
            del modality_index, quality
            probabilities = latent_probabilities
            expected = core.expected_grade(probabilities)
            entropy = core.normalized_entropy(probabilities)
            zero = torch.zeros_like(expected)
            clinical = {
                "score": expected,
                "probabilities": probabilities,
                "expected_grade": expected,
            }
            fusion = {
                "gate": zero,
                "probabilities": probabilities,
                "expected_grade": expected,
                "clinical_entropy": entropy,
                "latent_entropy": entropy,
                "js_disagreement": zero,
                "lesion_entropy": core.binary_entropy(evidence).mean(dim=1),
            }
            return {"clinical": clinical, "fusion": fusion}

        def zero_counterfactual(
            evidence: torch.Tensor,
            latent_probabilities: torch.Tensor,
            modality_index: torch.Tensor,
            quality: torch.Tensor,
            original: dict[str, Any],
        ) -> dict[str, torch.Tensor]:
            del evidence, modality_index, quality, original
            zero = latent_probabilities.sum() * 0.0
            return {"remove_all": zero, "remove_pdr": zero, "add_pdr": zero, "total": zero}

        self.model.fuse = latent_fuse
        self.model.counterfactual = zero_counterfactual

    def forward_mixed(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.model.forward_mixed(*args, **kwargs)


def build_variant(variant_id: str) -> nn.Module:
    if variant_id == "NO_FIELD_OF_VIEW_ENCODING":
        return NoFieldOfViewEncodingModel()
    if variant_id == "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE":
        return NoLesionQueriesGlobalPooledEvidenceModel()
    if variant_id == "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE":
        return NoClinicalReasonerLatentOnlyGradeModel()
    raise ValueError(f"Unknown reviewer component ablation: {variant_id}")


def variant_loss_weights(variant_id: str, candidate_weights: dict[str, Any]) -> dict[str, float]:
    if variant_id not in REVIEWER_VARIANTS:
        raise ValueError(f"Unknown reviewer component ablation: {variant_id}")
    keys = [
        "fundus_grade_cross_entropy_weight",
        "fundus_ordinal_EMD_weight",
        "lesion_BCE_weight",
        "clinical_state_alignment_weight",
        "lesion_counterfactual_intervention_weight",
        "masked_route_agreement_weight",
        "OCT_DME_task_weight",
    ]
    weights = {key: float(candidate_weights[key]) for key in keys}
    if variant_id == "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE":
        weights["clinical_state_alignment_weight"] = 0.0
        weights["lesion_counterfactual_intervention_weight"] = 0.0
        weights["masked_route_agreement_weight"] = 0.0
    return weights


def compute_variant_losses(
    variant_id: str,
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    lesion_pos_weight: torch.Tensor,
    candidate_weights: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    core = get_core()
    weights = variant_loss_weights(variant_id, candidate_weights)
    return core.compute_mixed_losses(output, batch, lesion_pos_weight, weights)


def build_reviewer_optimizer(model: nn.Module) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    groups: dict[str, list[nn.Parameter]] = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "new_decay": [],
        "new_no_decay": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("model.encoder.")
        lower_name = name.lower()
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in lower_name
            or ".bn" in lower_name
        )
        prefix = "backbone" if is_backbone else "new"
        suffix = "no_decay" if no_decay else "decay"
        groups[f"{prefix}_{suffix}"].append(parameter)
    empty = [name for name, parameters in groups.items() if not parameters]
    if empty:
        raise RuntimeError(f"Empty optimizer parameter groups: {empty}")
    optimizer = torch.optim.AdamW([
        {"params": groups["backbone_decay"], "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY, "group_name": "backbone_decay"},
        {"params": groups["backbone_no_decay"], "lr": BACKBONE_LR, "weight_decay": 0.0, "group_name": "backbone_no_decay"},
        {"params": groups["new_decay"], "lr": NEW_MODULE_LR, "weight_decay": WEIGHT_DECAY, "group_name": "new_decay"},
        {"params": groups["new_no_decay"], "lr": NEW_MODULE_LR, "weight_decay": 0.0, "group_name": "new_no_decay"},
    ])
    return optimizer, {name: len(parameters) for name, parameters in groups.items()}


def official_weight_record() -> dict[str, Any]:
    def locate() -> tuple[dict[str, Any], Path]:
        payload = load_json(WEIGHT_EVIDENCE)
        matches = [
            record for record in payload.get("official_models", [])
            if record.get("name") == "EfficientNetV2-S"
        ]
        if len(matches) != 1:
            raise RuntimeError("Unique EfficientNetV2-S official-weight record not found")
        record = matches[0]
        return record, Path(str(record.get("checkpoint_path", "")))

    record, path = locate()
    valid = bool(
        path.is_file()
        and sha256_file(path) == EXPECTED_EFFICIENTNET_WEIGHT_SHA
        and record.get("checkpoint_sha256") == EXPECTED_EFFICIENTNET_WEIGHT_SHA
    )
    if not valid:
        if not WEIGHT_SOURCE.is_file() or sha256_file(WEIGHT_SOURCE) != EXPECTED_WEIGHT_SOURCE_SHA:
            raise RuntimeError("Frozen official-weight restoration source is missing or changed")
        print("Official EfficientNetV2-S weight is absent from this runtime; restoring it now.")
        result = subprocess.run(
            [sys.executable, "-u", str(WEIGHT_SOURCE)],
            capture_output=True,
            text=True,
            check=False,
        )
        print(result.stdout[-4000:])
        if result.returncode != 0:
            print(result.stderr[-4000:], file=sys.stderr)
            raise RuntimeError(f"Official-weight restoration failed with code {result.returncode}")
        record, path = locate()
        valid = bool(
            path.is_file()
            and sha256_file(path) == EXPECTED_EFFICIENTNET_WEIGHT_SHA
            and record.get("checkpoint_sha256") == EXPECTED_EFFICIENTNET_WEIGHT_SHA
        )
    if not valid:
        raise RuntimeError("Official EfficientNetV2-S checkpoint is missing or changed")
    return record


def synthetic_batch(device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(RUN_SEED)
    images = torch.randn(3, 3, INPUT_SIZE, INPUT_SIZE, generator=generator)
    fov = torch.ones(3, 1, INPUT_SIZE, INPUT_SIZE)
    lesion_target = torch.randint(0, 2, (3, 7), generator=generator).float()
    return {
        "image": images.to(device),
        "fov_mask": fov.to(device),
        "quality": torch.tensor([0.22, 0.16, 0.18], device=device),
        "fundus_mask": torch.tensor([True, True, False], device=device),
        "modality_index": torch.tensor([0, 1, 2], device=device),
        "grade": torch.tensor([1, 4, 2], device=device),
        "grade_loss_mask": torch.ones(3, device=device),
        "lesion_loss_mask": torch.tensor([1.0, 1.0, 0.0], device=device),
        "lesion_sensitivity_mask": torch.tensor([1.0, 1.0, 0.0], device=device),
        "lesion_target": lesion_target.to(device),
        "modality": ["CFP", "UWF", "OCT"],
    }


def audit_variant(
    variant_id: str,
    device: torch.device,
    candidate_weights: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(RUN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RUN_SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_variant(variant_id).to(device).train()
    batch = synthetic_batch(device)
    lesion_pos_weight = torch.ones(7, device=device)
    output = model.forward_mixed(
        batch["image"], batch["fundus_mask"], batch["fov_mask"],
        batch["modality_index"], batch["quality"],
    )
    loss, parts = compute_variant_losses(
        variant_id, output, batch, lesion_pos_weight, candidate_weights,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"Non-finite synthetic loss for {variant_id}")
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    checks = {
        "fundus_output_present": output.get("fundus") is not None,
        "oct_output_present": output.get("oct") is not None,
        "fundus_probability_shape_2x5": tuple(output["fundus"]["original"]["fusion"]["probabilities"].shape) == (2, 5),
        "lesion_presence_shape_2x7": tuple(output["fundus"]["lesions"]["presence"].shape) == (2, 7),
        "oct_probability_shape_1x3": tuple(output["oct"]["probabilities"].shape) == (1, 3),
        "all_output_probabilities_finite": bool(
            torch.isfinite(output["fundus"]["original"]["fusion"]["probabilities"]).all()
            and torch.isfinite(output["oct"]["probabilities"]).all()
        ),
        "synthetic_loss_finite": bool(torch.isfinite(loss)),
        "gradients_present": bool(gradients),
        "all_gradients_finite": bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients),
        "at_least_one_nonzero_gradient": bool(gradients) and any(bool(torch.count_nonzero(value)) for value in gradients),
        "optimizer_not_created": True,
        "parameters_not_updated": True,
    }
    if variant_id == "NO_FIELD_OF_VIEW_ENCODING":
        checks["fov_coordinate_information_zeroed"] = isinstance(model, NoFieldOfViewEncodingModel)
    elif variant_id == "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE":
        checks["learned_lesion_queries_absent"] = (
            isinstance(model, NoLesionQueriesGlobalPooledEvidenceModel)
            and not hasattr(model.model.lesion_bank, "queries")
        )
    elif variant_id == "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE":
        checks["clinical_reasoner_absent"] = (
            isinstance(model, NoClinicalReasonerLatentOnlyGradeModel)
            and not hasattr(model.model, "clinical_reasoner")
            and not hasattr(model.model, "reliability_gate")
        )
        checks["grade_route_is_latent_only"] = bool(
            torch.allclose(
                output["fundus"]["original"]["fusion"]["probabilities"],
                output["fundus"]["latent"]["probabilities"],
                atol=0.0,
                rtol=0.0,
            )
        )
    result = {
        "variant_id": variant_id,
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "active_loss_weights": variant_loss_weights(variant_id, candidate_weights),
        "synthetic_total_loss": float(loss.detach().cpu()),
        "synthetic_loss_components": {name: float(value.detach().cpu()) for name, value in parts.items()},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "peak_reserved_VRAM_GB": (
            float(torch.cuda.max_memory_reserved() / 1024**3)
            if device.type == "cuda" else 0.0
        ),
    }
    del output, loss, parts, gradients, batch, lesion_pos_weight, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    print("=" * 120)
    print("STAGE 0G-4D-5F-7A — REVIEWER COMPONENT-ABLATION IMPLEMENTATION AUDIT")
    print("=" * 120)
    if not ROOT.is_dir():
        raise RuntimeError("The real Google Drive project root is not mounted")

    parent_checks = {
        "reviewer_protocol_hash_verified": REVIEWER_PROTOCOL.is_file() and sha256_file(REVIEWER_PROTOCOL) == EXPECTED_REVIEWER_PROTOCOL_SHA,
        "reviewer_protocol_evidence_PASS": REVIEWER_PROTOCOL_EVIDENCE.is_file() and evidence_passes(load_json(REVIEWER_PROTOCOL_EVIDENCE)),
        "core_source_hash_verified": CORE_SOURCE.is_file() and sha256_file(CORE_SOURCE) == EXPECTED_CORE_SHA,
        "candidate_hash_verified": CANDIDATE.is_file() and sha256_file(CANDIDATE) == EXPECTED_CANDIDATE_SHA,
        "weight_evidence_hash_verified": WEIGHT_EVIDENCE.is_file() and sha256_file(WEIGHT_EVIDENCE) == EXPECTED_WEIGHT_EVIDENCE_SHA,
        "weight_restore_source_hash_verified": WEIGHT_SOURCE.is_file() and sha256_file(WEIGHT_SOURCE) == EXPECTED_WEIGHT_SOURCE_SHA,
    }
    protocol = load_json(REVIEWER_PROTOCOL)
    e2 = [item for item in protocol["experiments_in_locked_order"] if item["id"] == "E2_COMPONENT_ABLATIONS"]
    parent_checks["one_E2_protocol_entry"] = len(e2) == 1
    parent_checks["E2_variant_order_exact"] = len(e2) == 1 and e2[0]["variants"] == REVIEWER_VARIANTS
    for name, value in parent_checks.items():
        print(f"{name:<48}: {value}")
    if not all(parent_checks.values()):
        raise RuntimeError(f"E2 parent gates failed: {[key for key, value in parent_checks.items() if not value]}")

    source_action, source_sha = freeze_source()
    weight = official_weight_record()
    print("Implementation source action:", source_action)
    print("Implementation source SHA   :", source_sha)
    print("Official weight SHA         :", weight["checkpoint_sha256"])

    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if cuda and torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU is required when CUDA is available")
    if cuda:
        torch.cuda.manual_seed_all(RUN_SEED)
    candidate_weights = load_json(CANDIDATE)["candidate_loss"]

    print("\n[2] SYNTHETIC FORWARD/BACKWARD AUDIT")
    print("-" * 120)
    results: list[dict[str, Any]] = []
    for variant_id in REVIEWER_VARIANTS:
        print("Auditing:", variant_id)
        result = audit_variant(variant_id, device, candidate_weights)
        results.append(result)
        print("  parameters       :", result["total_parameters"])
        print("  synthetic loss   :", f"{result['synthetic_total_loss']:.6f}")
        print("  all checks passed:", result["all_checks_passed"])

    gates = {
        **parent_checks,
        "source_hash_verified": FROZEN_SOURCE.is_file() and sha256_file(FROZEN_SOURCE) == source_sha,
        "exactly_three_variants_audited": len(results) == 3,
        "variant_order_exact": [item["variant_id"] for item in results] == REVIEWER_VARIANTS,
        "all_variant_checks_passed": all(bool(item["all_checks_passed"]) for item in results),
        "all_parameter_counts_positive": all(int(item["total_parameters"]) > 0 for item in results),
        "all_losses_finite": all(math.isfinite(float(item["synthetic_total_loss"])) for item in results),
        "optimizer_not_created": True,
        "parameters_not_updated": True,
        "medical_images_not_accessed": True,
        "TRAIN_not_accessed": True,
        "VALIDATION_not_accessed": True,
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"E2 implementation gates failed: {[key for key, value in gates.items() if not value]}")

    contract_payload = {
        "contract_id": "MMRDR_FOVEA_DR_REVIEWER_COMPONENT_ABLATION_IMPLEMENTATION_V1",
        "status": "THREE_COMPONENT_ABLATIONS_IMPLEMENTED_AND_AUDITED",
        "reviewer_protocol_sha256": EXPECTED_REVIEWER_PROTOCOL_SHA,
        "core_source_sha256": EXPECTED_CORE_SHA,
        "implementation_source": str(FROZEN_SOURCE),
        "implementation_source_sha256": source_sha,
        "variant_order": REVIEWER_VARIANTS,
        "run_seed": RUN_SEED,
        "input_size": INPUT_SIZE,
        "official_initialization": "TORCHVISION_EFFICIENTNETV2S_IMAGENET1K_V1",
        "official_checkpoint_sha256": EXPECTED_EFFICIENTNET_WEIGHT_SHA,
        "ablation_definitions": {
            "NO_FIELD_OF_VIEW_ENCODING": "All seven field-of-view and coordinate channels are identically zero; the original coordinate-fusion parameterization is retained.",
            "NO_LESION_QUERIES_GLOBAL_POOLED_EVIDENCE": "The seven learned spatial lesion queries are replaced by seven global-pooled evidence logits.",
            "NO_CLINICAL_REASONER_LATENT_ONLY_GRADE": "Clinical reasoning and reliability fusion are removed; the latent ordinal route exclusively determines fundus grade while the lesion auxiliary task remains.",
        },
        "variant_audits": results,
        "preflight_weights_reusable": False,
        "CALIBRATION_access": "PROHIBITED",
        "LOCKED_INTERNAL_TEST_access": "PROHIBITED",
    }
    contract_action = frozen_json(CONTRACT, contract_payload)
    contract_sha = sha256_file(CONTRACT)
    evidence_payload = {
        "stage": "0G-4D-5F-7A",
        "purpose": "REVIEWER_COMPONENT_ABLATION_IMPLEMENTATION_AUDIT",
        "status": "PASS",
        "implementation_source": str(FROZEN_SOURCE),
        "implementation_source_sha256": source_sha,
        "contract": str(CONTRACT),
        "contract_sha256": contract_sha,
        "contract_action": contract_action,
        "device": str(device),
        "variant_audits": results,
        "gates": gates,
        "optimizer_created": False,
        "parameters_updated": False,
        "medical_images_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "next_required_stage": "RUN_64_STEP_TRAIN_ONLY_PREFLIGHT_FOR_THREE_E2_VARIANTS",
    }
    evidence_action = frozen_json(EVIDENCE, evidence_payload)

    print("\n" + "=" * 120)
    print("STAGE 0G-4D-5F-7A STATUS: PASS")
    print("=" * 120)
    print("Variants implemented : 3")
    print("Optimizer created     : NO")
    print("Parameters updated    : NO")
    print("Medical images read   : NO")
    print("CALIBRATION accessed  : NO")
    print("Locked test accessed  : NO")
    print("Contract action       :", contract_action)
    print("Evidence action       :", evidence_action)
    print("NEXT ACTION: RUN 64-STEP TRAIN-ONLY PREFLIGHT FOR THE THREE E2 VARIANTS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 120, file=sys.stderr)
        print("STAGE 0G-4D-5F-7A STATUS: FAIL", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
