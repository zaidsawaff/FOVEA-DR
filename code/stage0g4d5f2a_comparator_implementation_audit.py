#!/usr/bin/env python3
"""Frozen implementations and synthetic integration audit for comparators.

The module exports the six protocol-locked post-hoc variants, their losses,
and their optimizer construction. Executing the file audits every variant on
synthetic 512x512 RGB tensors. It does not read medical images, create an
optimizer, update parameters, or access CALIBRATION/LOCKED_INTERNAL_TEST.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
CANDIDATE = ROOT / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
PRODUCTION_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
WEIGHT_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f1c_official_comparator_weight_acquisition.py"
WEIGHT_EVIDENCE = ROOT / "Evidence/stage0g4d5f1c_official_comparator_weight_acquisition.json"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2a_comparator_implementation_audit.py"
CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparator_implementation_contract_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5f2a_comparator_implementation_audit.json"

EXPECTED_SHA = {
    PROTOCOL: "f946964923e08722dbb486aa3782b08880446040e07fb6b268ea9c28c08e6ae1",
    CANDIDATE: "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181",
    PRODUCTION_SOURCE: "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9",
    WEIGHT_SOURCE: "b9ccaf24db7255cb73dfd0cacf928556033a9810d5766f73f67ee13ecebd5c8b",
}

PLAIN_VARIANTS = {
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK": "efficientnet_v2_s",
    "B2_RESNET50_MULTITASK": "resnet50",
    "B3_CONVNEXT_TINY_MULTITASK": "convnext_tiny",
}

ABLATION_VARIANTS = {
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
}

ALL_VARIANTS = list(PLAIN_VARIANTS) + sorted(ABLATION_VARIANTS)

BACKBONE_LR = 1.0e-4
NEW_MODULE_LR = 3.0e-4
WEIGHT_DECAY = 1.0e-4
INPUT_SIZE = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def frozen_json(path: Path, payload: dict[str, Any]) -> str:
    data = json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return "CREATED_AND_FROZEN"


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different 5F-2A implementation is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def import_frozen_core():
    specification = importlib.util.spec_from_file_location(
        "fovea_dr_training_v1_frozen_core", PRODUCTION_SOURCE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not create frozen production-source import specification.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


CORE = None


def get_core():
    global CORE
    if CORE is None:
        CORE = import_frozen_core()
    return CORE


class PlainBackbone(nn.Module):
    """Official torchvision feature extractor with a uniform map interface."""

    def __init__(self, backbone: str):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "efficientnet_v2_s":
            official = models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
                progress=False,
            )
            self.features = official.features
            self.output_dim = 1280
        elif backbone == "convnext_tiny":
            official = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                progress=False,
            )
            self.features = official.features
            self.output_dim = 768
        elif backbone == "resnet50":
            official = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2,
                progress=False,
            )
            self.conv1 = official.conv1
            self.bn1 = official.bn1
            self.relu = official.relu
            self.maxpool = official.maxpool
            self.layer1 = official.layer1
            self.layer2 = official.layer2
            self.layer3 = official.layer3
            self.layer4 = official.layer4
            self.output_dim = 2048
        else:
            raise ValueError(f"Unsupported frozen backbone: {backbone}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone_name in {"efficientnet_v2_s", "convnext_tiny"}:
            return self.features(images)
        value = self.conv1(images)
        value = self.bn1(value)
        value = self.relu(value)
        value = self.maxpool(value)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        return self.layer4(value)


class PlainMultitaskModel(nn.Module):
    """Backbone plus grade, DME, and seven-lesion heads only."""

    def __init__(self, backbone: str):
        super().__init__()
        core = get_core()
        self.encoder = PlainBackbone(backbone)
        dimension = self.encoder.output_dim
        self.fundus_grade_head = core.OrdinalHead(dimension, 5)
        self.lesion_head = nn.Linear(dimension, 7)
        self.oct_head = core.OrdinalHead(dimension, 3)

    def forward_mixed(
        self,
        images: torch.Tensor,
        fundus_mask: torch.Tensor,
        fov_masks: torch.Tensor,
        modality_indices: torch.Tensor,
        quality: torch.Tensor,
    ) -> dict[str, Any]:
        del fov_masks, modality_indices, quality
        maps = self.encoder(images)
        pooled = F.adaptive_avg_pool2d(maps, 1).flatten(1)
        output: dict[str, Any] = {"fundus": None, "oct": None}
        if bool(fundus_mask.any()):
            fundus_features = pooled[fundus_mask]
            output["fundus"] = {
                "grade": self.fundus_grade_head(fundus_features),
                "lesion_logits": self.lesion_head(fundus_features),
            }
        oct_mask = ~fundus_mask
        if bool(oct_mask.any()):
            output["oct"] = self.oct_head(pooled[oct_mask])
        return output


class FixedClinicalReliabilityGate(nn.Module):
    """Ablation gate: clinical/evidence route weight is fixed to exactly one."""

    def forward(
        self,
        clinical: torch.Tensor,
        latent: torch.Tensor,
        evidence: torch.Tensor,
        modality_index: torch.Tensor,
        quality: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del modality_index, quality
        core = get_core()
        batch = clinical.shape[0]
        gate = torch.ones(batch, device=clinical.device, dtype=clinical.dtype)
        probabilities = clinical / clinical.sum(dim=1, keepdim=True)
        return {
            "gate": gate,
            "probabilities": probabilities,
            "expected_grade": core.expected_grade(probabilities),
            "clinical_entropy": core.normalized_entropy(clinical),
            "latent_entropy": core.normalized_entropy(latent),
            "js_disagreement": core.js_divergence(clinical, latent),
            "lesion_entropy": core.binary_entropy(evidence).mean(dim=1),
        }


def build_variant(variant_id: str) -> nn.Module:
    core = get_core()
    if variant_id in PLAIN_VARIANTS:
        return PlainMultitaskModel(PLAIN_VARIANTS[variant_id])
    if variant_id in ABLATION_VARIANTS:
        model = core.FOVEADRTrainingModel()
        if variant_id == "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE":
            model.reliability_gate = FixedClinicalReliabilityGate()
        return model
    raise ValueError(f"Unknown frozen comparison variant: {variant_id}")


def variant_loss_weights(variant_id: str, candidate_weights: dict[str, Any]) -> dict[str, float]:
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
    if variant_id in PLAIN_VARIANTS:
        weights["clinical_state_alignment_weight"] = 0.0
        weights["lesion_counterfactual_intervention_weight"] = 0.0
        weights["masked_route_agreement_weight"] = 0.0
    elif variant_id == "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS":
        weights["lesion_counterfactual_intervention_weight"] = 0.0
    elif variant_id == "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT":
        weights["clinical_state_alignment_weight"] = 0.0
        weights["masked_route_agreement_weight"] = 0.0
    elif variant_id == "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE":
        pass
    else:
        raise ValueError(f"Unknown frozen comparison variant: {variant_id}")
    return weights


def compute_plain_losses(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    lesion_pos_weight: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    core = get_core()
    device = batch["image"].device
    fundus_mask = batch["fundus_mask"]
    oct_mask = ~fundus_mask
    zero = torch.zeros((), device=device, dtype=torch.float32)
    parts = {
        "grade_NLL": zero,
        "ordinal_EMD": zero,
        "lesion_BCE": zero,
        "clinical_alignment": zero,
        "counterfactual": zero,
        "route_agreement": zero,
        "OCT_NLL": zero,
    }
    if bool(fundus_mask.any()):
        target = batch["grade"][fundus_mask]
        grade_mask = batch["grade_loss_mask"][fundus_mask]
        lesion_mask = batch["lesion_loss_mask"][fundus_mask]
        fundus = output["fundus"]
        probabilities = fundus["grade"]["probabilities"]
        parts["grade_NLL"] = core.masked_mean(
            -probabilities.gather(1, target[:, None]).clamp_min(1e-7).log().squeeze(1),
            grade_mask,
        )
        parts["ordinal_EMD"] = core.masked_mean(
            core.grade_emd(probabilities, target), grade_mask
        )
        lesion_per_sample = F.binary_cross_entropy_with_logits(
            fundus["lesion_logits"],
            batch["lesion_target"][fundus_mask],
            pos_weight=lesion_pos_weight,
            reduction="none",
        ).mean(dim=1)
        parts["lesion_BCE"] = core.masked_mean(lesion_per_sample, lesion_mask)
    if bool(oct_mask.any()):
        oct_probabilities = output["oct"]["probabilities"]
        oct_target = batch["grade"][oct_mask]
        parts["OCT_NLL"] = -oct_probabilities.gather(
            1, oct_target[:, None]
        ).clamp_min(1e-7).log().mean()
    total = (
        weights["fundus_grade_cross_entropy_weight"] * parts["grade_NLL"]
        + weights["fundus_ordinal_EMD_weight"] * parts["ordinal_EMD"]
        + weights["lesion_BCE_weight"] * parts["lesion_BCE"]
        + weights["OCT_DME_task_weight"] * parts["OCT_NLL"]
    )
    return total, parts


def compute_variant_losses(
    variant_id: str,
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    lesion_pos_weight: torch.Tensor,
    candidate_weights: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = variant_loss_weights(variant_id, candidate_weights)
    if variant_id in PLAIN_VARIANTS:
        return compute_plain_losses(output, batch, lesion_pos_weight, weights)
    return get_core().compute_mixed_losses(output, batch, lesion_pos_weight, weights)


def build_comparison_optimizer(model: nn.Module) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    groups: dict[str, list[nn.Parameter]] = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "new_decay": [],
        "new_no_decay": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("encoder.")
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
    if any(len(parameters) == 0 for parameters in groups.values()):
        empty = [name for name, parameters in groups.items() if not parameters]
        raise RuntimeError(f"Empty optimizer parameter groups: {empty}")
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["backbone_decay"], "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY, "group_name": "backbone_decay"},
            {"params": groups["backbone_no_decay"], "lr": BACKBONE_LR, "weight_decay": 0.0, "group_name": "backbone_no_decay"},
            {"params": groups["new_decay"], "lr": NEW_MODULE_LR, "weight_decay": WEIGHT_DECAY, "group_name": "new_decay"},
            {"params": groups["new_no_decay"], "lr": NEW_MODULE_LR, "weight_decay": 0.0, "group_name": "new_no_decay"},
        ]
    )
    return optimizer, {name: len(parameters) for name, parameters in groups.items()}


def synthetic_batch(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(2026)
    images = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE, generator=generator)
    fov = torch.ones(4, 1, INPUT_SIZE, INPUT_SIZE)
    return {
        "image": images.to(device),
        "fov_mask": fov.to(device),
        "quality": torch.tensor([0.20, 0.25, 0.30, 0.35], device=device),
        "fundus_mask": torch.tensor([True, True, False, False], device=device),
        "modality_index": torch.tensor([0, 1, 2, 2], device=device),
        "grade": torch.tensor([0, 4, 1, 2], device=device),
        "grade_loss_mask": torch.ones(4, device=device),
        "lesion_loss_mask": torch.tensor([1.0, 1.0, 0.0, 0.0], device=device),
        "lesion_sensitivity_mask": torch.tensor([1.0, 1.0, 0.0, 0.0], device=device),
        "lesion_target": torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
            device=device,
        ),
    }


def probability_checks(variant_id: str, output: dict[str, Any]) -> dict[str, bool]:
    if variant_id in PLAIN_VARIANTS:
        fundus_probabilities = output["fundus"]["grade"]["probabilities"]
    else:
        fundus_probabilities = output["fundus"]["original"]["fusion"]["probabilities"]
    oct_probabilities = output["oct"]["probabilities"]
    return {
        "fundus_shape_is_2x5": tuple(fundus_probabilities.shape) == (2, 5),
        "OCT_shape_is_2x3": tuple(oct_probabilities.shape) == (2, 3),
        "fundus_probabilities_finite": bool(torch.isfinite(fundus_probabilities).all()),
        "OCT_probabilities_finite": bool(torch.isfinite(oct_probabilities).all()),
        "fundus_probabilities_sum_to_one": bool(torch.allclose(fundus_probabilities.sum(1), torch.ones(2, device=fundus_probabilities.device), atol=1e-4)),
        "OCT_probabilities_sum_to_one": bool(torch.allclose(oct_probabilities.sum(1), torch.ones(2, device=oct_probabilities.device), atol=1e-4)),
    }


def audit_variant(
    variant_id: str,
    candidate_weights: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    model = build_variant(variant_id).to(device)
    model.train()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    batch = synthetic_batch(device)
    lesion_pos_weight = torch.ones(7, device=device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.forward_mixed(
            batch["image"],
            batch["fundus_mask"],
            batch["fov_mask"],
            batch["modality_index"],
            batch["quality"],
        )
        loss, parts = compute_variant_losses(
            variant_id, output, batch, lesion_pos_weight, candidate_weights
        )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"Non-finite synthetic loss for {variant_id}")
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    gradients_exist = bool(gradients)
    gradients_finite = gradients_exist and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    gradients_nonzero = gradients_exist and any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    unchanged = all(
        torch.equal(parameter.detach(), before[name])
        for name, parameter in model.named_parameters()
    )
    probabilities = probability_checks(variant_id, output)
    weights = variant_loss_weights(variant_id, candidate_weights)

    structural = {
        "plain_has_no_clinical_reasoner": (
            variant_id not in PLAIN_VARIANTS or not hasattr(model, "clinical_reasoner")
        ),
        "plain_has_no_reliability_gate": (
            variant_id not in PLAIN_VARIANTS or not hasattr(model, "reliability_gate")
        ),
        "A1_counterfactual_weight_zero": (
            variant_id != "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS"
            or weights["lesion_counterfactual_intervention_weight"] == 0.0
        ),
        "A2_alignment_and_route_weights_zero": (
            variant_id != "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT"
            or (
                weights["clinical_state_alignment_weight"] == 0.0
                and weights["masked_route_agreement_weight"] == 0.0
            )
        ),
        "A3_gate_fixed_to_one": (
            variant_id != "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE"
            or (
                isinstance(model.reliability_gate, FixedClinicalReliabilityGate)
                and bool(torch.equal(
                    output["fundus"]["original"]["fusion"]["gate"],
                    torch.ones_like(output["fundus"]["original"]["fusion"]["gate"]),
                ))
            )
        ),
    }
    checks = {
        **probabilities,
        **structural,
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_exist": gradients_exist,
        "gradients_finite": gradients_finite,
        "gradients_nonzero": gradients_nonzero,
        "parameters_unchanged_without_optimizer": unchanged,
    }
    record = {
        "variant_id": variant_id,
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "loss": float(loss.detach().cpu()),
        "loss_parts": {name: float(value.detach().cpu()) for name, value in parts.items()},
        "loss_weights": weights,
        "checks": {name: bool(value) for name, value in checks.items()},
    }
    del output, loss, parts, batch, before, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> None:
    print("=" * 80)
    print("STAGE 0G-4D-5F-2A — COMPARATOR IMPLEMENTATION AND SYNTHETIC AUDIT")
    print("=" * 80)

    print("\n[1] FROZEN PREREQUISITE VERIFICATION")
    print("-" * 80)
    gates: dict[str, bool] = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(verified)
        print(f"{path.name:76s}: exists={exists}, hash_verified={verified}")
    if not WEIGHT_EVIDENCE.is_file():
        raise FileNotFoundError(WEIGHT_EVIDENCE)
    with WEIGHT_EVIDENCE.open("r", encoding="utf-8") as handle:
        weight_evidence = json.load(handle)
    weight_gates = weight_evidence.get("gates", {})
    weight_evidence_pass = (
        weight_evidence.get("status") == "PASS"
        and bool(weight_gates)
        and all(bool(value) for value in weight_gates.values())
        and weight_evidence.get("locked_test_accessed") is False
    )
    gates["official_weight_evidence_passed"] = bool(weight_evidence_pass)

    with PROTOCOL.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    protocol_ids = [item["id"] for item in protocol["comparators"]]
    protocol_ids += [item["id"] for item in protocol["ablations"]]
    variant_ids_exact = protocol_ids == ALL_VARIANTS
    test_prohibited = protocol["data_access_policy"]["LOCKED_INTERNAL_TEST"].startswith("PROHIBITED")
    gates["six_variant_ids_match_frozen_protocol"] = bool(variant_ids_exact)
    gates["locked_test_prohibited"] = bool(test_prohibited)

    cuda_available = torch.cuda.is_available()
    visible_devices = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if visible_devices == 1 else None
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if visible_devices == 1 else 0.0
    )
    gates.update({
        "CUDA_available": bool(cuda_available),
        "exactly_one_GPU_visible": visible_devices == 1,
        "GPU_memory_at_least_10GB": vram_gb >= 10.0,
    })
    print("Official weight evidence PASS:", weight_evidence_pass)
    print("Six variant IDs exact         :", variant_ids_exact)
    print("Locked test prohibited        :", test_prohibited)
    print("GPU                           :", gpu_name)
    print("VRAM GB                       :", f"{vram_gb:.3f}")
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Implementation prerequisites failed: {failed}")

    print("\n[2] FROZEN LOSS CONTRACT")
    print("-" * 80)
    with CANDIDATE.open("r", encoding="utf-8") as handle:
        candidate_weights = json.load(handle)["candidate_loss"]
    for variant_id in ALL_VARIANTS:
        print(variant_id, variant_loss_weights(variant_id, candidate_weights))

    print("\n[3] SYNTHETIC 512x512 FLOAT16 FORWARD/BACKWARD AUDIT")
    print("-" * 80)
    device = torch.device("cuda:0")
    records = []
    for variant_id in ALL_VARIANTS:
        print(f"Auditing {variant_id}...")
        record = audit_variant(variant_id, candidate_weights, device)
        records.append(record)
        passed = all(record["checks"].values())
        print(
            f"{variant_id} | parameters={record['total_parameters']} | "
            f"loss={record['loss']:.6f} | PASS={passed}"
        )

    gates.update({
        "exactly_six_variants_audited": len(records) == 6,
        "all_synthetic_audit_checks_passed": all(
            all(record["checks"].values()) for record in records
        ),
        "synthetic_input_locked_to_512_RGB": True,
        "float16_autocast_used": True,
        "optimizer_not_created": True,
        "parameters_not_updated": True,
        "medical_images_not_accessed": True,
        "TRAIN_not_accessed": True,
        "VALIDATION_not_accessed": True,
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
        "training_not_started": True,
    })

    print("\n" + "=" * 80)
    print("COMPARATOR IMPLEMENTATION AUDIT GATES")
    print("=" * 80)
    for name, passed in gates.items():
        print(f"{name:80s}: {passed}")
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Comparator implementation audit failed: {failed}")

    print("\n[4] IMPLEMENTATION CONTRACT AND SOURCE FREEZE")
    print("-" * 80)
    source_action, source_sha = freeze_source()
    gates["implementation_source_frozen_and_verified"] = sha256_file(FROZEN_SOURCE) == source_sha
    contract = {
        "contract_id": "MMRDR_FOVEA_DR_POSTHOC_COMPARATOR_IMPLEMENTATION_V1",
        "stage": "0G-4D-5F-2A",
        "status": "FROZEN_BEFORE_REAL_IMAGE_PREFLIGHT",
        "comparison_protocol": str(PROTOCOL),
        "comparison_protocol_sha256": EXPECTED_SHA[PROTOCOL],
        "implementation_source": str(FROZEN_SOURCE),
        "implementation_source_sha256": source_sha,
        "input_contract": "512x512_RGB_IMAGENET_NORMALIZATION",
        "plain_model_contract": {
            "fundus_grade_head": "5_CLASS_ORDINAL",
            "OCT_DME_head": "3_CLASS_ORDINAL",
            "fundus_lesion_head": "7_BINARY_LOGITS_FROM_GLOBAL_BACKBONE_FEATURES",
            "clinical_reasoner": False,
            "counterfactual_module": False,
            "learned_reliability_gate": False,
        },
        "ablation_contracts": {
            "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS": "counterfactual_weight=0.0_only",
            "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT": "clinical_alignment_weight=0.0_and_route_agreement_weight=0.0_only",
            "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE": "clinical_route_gate_fixed_to_1.0_for_original_and_all_interventions",
        },
        "optimizer_contract": {
            "type": "AdamW",
            "backbone_lr": BACKBONE_LR,
            "new_module_lr": NEW_MODULE_LR,
            "weight_decay": WEIGHT_DECAY,
            "four_decay_groups": True,
        },
        "all_variants": ALL_VARIANTS,
        "preflight_checkpoint_initialization_allowed": False,
        "locked_test_access": "PROHIBITED",
    }
    contract_action = frozen_json(CONTRACT, contract)
    contract_sha = sha256_file(CONTRACT)
    evidence_records = [
        {
            "variant_id": record["variant_id"],
            "total_parameters": record["total_parameters"],
            "trainable_parameters": record["trainable_parameters"],
            "loss_weights": record["loss_weights"],
            "checks": record["checks"],
        }
        for record in records
    ]
    evidence = {
        "stage": "0G-4D-5F-2A",
        "status": "PASS",
        "purpose": "COMPARATOR_IMPLEMENTATION_SYNTHETIC_AUDIT",
        "source": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "implementation_contract": str(CONTRACT),
        "implementation_contract_sha256": contract_sha,
        "official_weight_evidence": str(WEIGHT_EVIDENCE),
        "official_weight_evidence_sha256": sha256_file(WEIGHT_EVIDENCE),
        "GPU": {"name": gpu_name, "VRAM_GB": float(vram_gb)},
        "variant_records": evidence_records,
        "gates": {name: bool(value) for name, value in gates.items()},
        "synthetic_data_only": True,
        "medical_images_accessed": False,
        "optimizer_created": False,
        "parameters_updated": False,
        "training_started": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "next_required_stage": "SIX_VARIANT_64_OPTIMIZER_STEP_TRAIN_ONLY_PREFLIGHT",
    }
    evidence_action = frozen_json(EVIDENCE, evidence)
    print("Source action     :", source_action)
    print("Source            :", FROZEN_SOURCE)
    print("Source SHA-256    :", source_sha)
    print("Contract action   :", contract_action)
    print("Contract          :", CONTRACT)
    print("Contract SHA-256  :", contract_sha)
    print("Evidence action   :", evidence_action)
    print("Evidence          :", EVIDENCE)
    print("Evidence SHA-256  :", sha256_file(EVIDENCE))

    print("\nSynthetic tensors used : YES")
    print("Medical images accessed: NO")
    print("Optimizer created       : NO")
    print("Parameters updated      : NO")
    print("Training started        : NO")
    print("CALIBRATION accessed    : NO")
    print("Locked test accessed    : NO")
    print("\nNEXT REQUIRED STAGE:")
    print("Run the six-variant 64-optimizer-step TRAIN-only preflight.")
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-5F-2A STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 80, file=sys.stderr)
        print("STAGE 0G-4D-5F-2A STATUS: FAIL", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        raise
