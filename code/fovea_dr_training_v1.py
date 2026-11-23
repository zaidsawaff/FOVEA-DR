from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# =============================================================================
# STAGE 0G-4D-3B-R2 — LOSS-CONTRACT RECOVERY + 64-STEP TRAIN-ONLY PREFLIGHT
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
CACHE_ROOT = Path("/content/MMRDR_training_v1")

SPLIT_PATH = PROJECT_ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
CANDIDATE_PATH = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
PROTOCOL_PATH = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_training_protocol_v1.json"
SAMPLER_PATH = PROJECT_ROOT / "Splits/mmrdr_fovea_dr_TRAIN_sampler_weights_v1.csv"
LESION_WEIGHT_PATH = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_lesion_pos_weights_v1.csv"
CACHE_MANIFEST_PATH = PROJECT_ROOT / "Audit/mmrdr_fovea_dr_TRAIN_local_cache_manifest_v1.csv"
CACHE_EVIDENCE_PATH = PROJECT_ROOT / "Evidence/stage0g4d3a_train_only_full_cache_preparation.json"
SOURCE_FREEZE_MANIFEST_PATH = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_source_freeze_manifest_v1.json"

FROZEN_PRODUCTION_SOURCE = PROJECT_ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
TRAINING_IMPLEMENTATION_CONTRACT = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_training_implementation_contract_v1.json"
PILOT_LOG_PATH = PROJECT_ROOT / "Audit/mmrdr_fovea_dr_TRAIN_preflight64_r2_log_v1.csv"
PILOT_CHECKPOINT_PATH = PROJECT_ROOT / "Checkpoints/FOVEA_DR/preflight/stage0g4d3b_r2_preflight64_model.pt"
EVIDENCE_PATH = PROJECT_ROOT / "Evidence/stage0g4d3b_r2_train_only_64step_preflight.json"

EXPECTED_CANDIDATE_SHA = "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181"
EXPECTED_PROTOCOL_SHA = "5b16e0c8565b86d3cac603f416ee0b7f03055ab9a0eb9ac9e3fb007632329b40"
EXPECTED_SPLIT_SHA = "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140"
EXPECTED_CACHE_MANIFEST_SHA = "32f9a3b6b17916b78f591007a85ce2e1910c9c4d2b1294af1c391b19f476bd15"
EXPECTED_AUDITED_SOURCE_SHA = "0f738cffc9dd0ff1c3e36690fe2d835636a3e3ec5a76af782feed4f949095097"
EXPECTED_OFFICIAL_WEIGHT_SHA = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"

SEED = 20260806
INPUT_SIZE = 512
MICRO_BATCH = 16
GRADIENT_ACCUMULATION = 2
EFFECTIVE_BATCH = 32
PILOT_OPTIMIZER_STEPS = 64
MICROBATCHES_PER_PILOT = PILOT_OPTIMIZER_STEPS * GRADIENT_ACCUMULATION
MAX_EPOCHS = 30
OPTIMIZER_STEPS_PER_EPOCH = 415
TOTAL_OPTIMIZER_STEPS = 12450
WARMUP_STEPS = 623
BACKBONE_LR = 1.0e-4
NEW_MODULE_LR = 3.0e-4
WEIGHT_DECAY = 1.0e-4
MIN_LR_RATIO = 0.01
GRAD_CLIP_NORM = 1.0

FPN_TAPS = [3, 5, 7]
FPN_CHANNELS = [64, 160, 1280]
FUSION_DIM = 128
TOP_K_FRACTION = 0.10
COUNTERFACTUAL_MARGIN = 0.25

MICROBATCH_A = {"CFP": 8, "UWF": 6, "OCT": 2}
MICROBATCH_B = {"CFP": 7, "UWF": 7, "OCT": 2}
EXPECTED_EFFECTIVE_COMPOSITION = {"CFP": 15, "UWF": 13, "OCT": 4}
EXPECTED_TRAIN_COUNTS = {"CFP": 6128, "UWF": 5460, "OCT": 1663}

LESION_COLUMNS = [
    "lesion_microaneurysm",
    "lesion_hard_exudate",
    "lesion_intraretinal_hemorrhage",
    "lesion_VB_IRMA",
    "lesion_neovascularization",
    "lesion_vitreous_hemorrhage",
    "lesion_retinal_detachment",
]

EXPECTED_LOSS_WEIGHTS = {
    "fundus_grade_cross_entropy_weight": 1.0,
    "fundus_ordinal_EMD_weight": 0.5,
    "lesion_BCE_weight": 0.5,
    "clinical_state_alignment_weight": 0.2,
    "lesion_counterfactual_intervention_weight": 0.1,
    "masked_route_agreement_weight": 0.05,
    "OCT_DME_task_weight": 0.5,
}

EXPECTED_MASK_POLICIES = {
    "lesion_BCE": "lesion_loss_mask",
    "clinical_state_alignment": "lesion_sensitivity_mask",
    "route_agreement": "lesion_sensitivity_mask",
    "OCT_lesion_supervision": False,
}

ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_evidence_gates_pass(evidence):
    gates = evidence.get("gates", {})
    return (
        evidence.get("status") == "PASS"
        and isinstance(gates, dict)
        and len(gates) > 0
        and all(bool(value) for value in gates.values())
    )


def nested_json_contains(value, target):
    if value == target:
        return True
    if isinstance(value, dict):
        return any(
            nested_json_contains(item, target)
            for item in value.values()
        )
    if isinstance(value, list):
        return any(
            nested_json_contains(item, target)
            for item in value
        )
    return False


def inv_softplus(value):
    return math.log(math.expm1(float(value)))


def ordered_distribution(score, thresholds, temperature=1.0):
    cumulative = torch.sigmoid(
        (score.unsqueeze(1) - thresholds.unsqueeze(0)) / temperature
    )
    probabilities = torch.cat(
        [
            1.0 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ],
        dim=1,
    )
    probabilities = probabilities.clamp_min(1e-7)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def expected_grade(probabilities):
    grades = torch.arange(
        probabilities.shape[1],
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return (probabilities * grades.unsqueeze(0)).sum(dim=1)


def normalized_entropy(probabilities):
    values = probabilities.clamp_min(1e-7)
    return -(values * values.log()).sum(dim=1) / math.log(probabilities.shape[1])


def binary_entropy(values):
    probabilities = values.clamp(1e-7, 1.0 - 1e-7)
    return -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    ) / math.log(2.0)


def js_divergence(first, second):
    first = first.clamp_min(1e-7)
    second = second.clamp_min(1e-7)
    mean = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - mean.log())).sum(dim=1)
        + (second * (second.log() - mean.log())).sum(dim=1)
    )


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class OrdinalHead(nn.Module):
    def __init__(self, input_dim, classes, temperature=1.0):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.threshold_base = nn.Parameter(torch.tensor(-0.5))
        self.threshold_deltas = nn.Parameter(
            torch.full((classes - 1,), inv_softplus(1.0))
        )
        self.temperature = temperature

    def thresholds(self):
        return self.threshold_base + torch.cumsum(
            F.softplus(self.threshold_deltas), dim=0
        )

    def forward(self, features):
        score = self.score(features).squeeze(1)
        probabilities = ordered_distribution(
            score, self.thresholds(), self.temperature
        )
        return {
            "score": score,
            "probabilities": probabilities,
            "expected_grade": expected_grade(probabilities),
        }


class ClinicalStateReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        desired = torch.tensor([0.8, 0.9, 1.0, 1.2, 1.6, 1.7, 1.8])
        self.raw_weights = nn.Parameter(torch.log(torch.expm1(desired)))
        self.threshold_base = nn.Parameter(torch.tensor(-0.5))
        self.threshold_deltas = nn.Parameter(
            torch.full((4,), inv_softplus(1.0))
        )

    def thresholds(self):
        return self.threshold_base + torch.cumsum(
            F.softplus(self.threshold_deltas), dim=0
        )

    def forward(self, evidence):
        weights = F.softplus(self.raw_weights)
        score = (evidence * weights.unsqueeze(0)).sum(dim=1)
        probabilities = ordered_distribution(
            score, self.thresholds(), temperature=0.25
        )
        return {
            "score": score,
            "probabilities": probabilities,
            "expected_grade": expected_grade(probabilities),
            "positive_weights": weights,
        }


class MultiScaleFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.laterals = nn.ModuleList(
            [nn.Conv2d(channels, FUSION_DIM, 1) for channels in FPN_CHANNELS]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(FUSION_DIM, FUSION_DIM, 3, padding=1, bias=False),
            nn.GroupNorm(8, FUSION_DIM),
            nn.SiLU(),
        )

    def forward(self, features):
        target_size = features[0].shape[-2:]
        projected = []
        for layer, feature in zip(self.laterals, features):
            value = layer(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(value)
        return self.refine(sum(projected) / math.sqrt(len(projected)))


def make_coordinate_channels(fov_mask, modality_index, size):
    mask = F.interpolate(fov_mask, size=size, mode="nearest")
    batch, _, height, width = mask.shape
    y_values = torch.linspace(
        -1.0, 1.0, height, device=mask.device, dtype=mask.dtype
    )
    x_values = torch.linspace(
        -1.0, 1.0, width, device=mask.device, dtype=mask.dtype
    )
    yy, xx = torch.meshgrid(y_values, x_values, indexing="ij")
    xx = xx.view(1, 1, height, width).expand(batch, -1, -1, -1)
    yy = yy.view(1, 1, height, width).expand(batch, -1, -1, -1)
    mass = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    center_x = (xx * mask).sum(dim=(2, 3), keepdim=True) / mass
    center_y = (yy * mask).sum(dim=(2, 3), keepdim=True) / mass
    delta_x = xx - center_x
    delta_y = yy - center_y
    radius = torch.sqrt(delta_x.square() + delta_y.square() + 1e-8)
    maximum_radius = (
        (radius * mask)
        .flatten(1)
        .amax(dim=1)
        .view(batch, 1, 1, 1)
        .clamp_min(1e-4)
    )
    normalized_radius = (radius / maximum_radius).clamp(0.0, 1.5)
    sin_angle = delta_y / radius.clamp_min(1e-4)
    cos_angle = delta_x / radius.clamp_min(1e-4)
    modality = (
        modality_index
        .to(mask.dtype)
        .view(batch, 1, 1, 1)
        .expand(batch, 1, height, width)
    )
    return torch.cat(
        [
            xx,
            yy,
            normalized_radius,
            sin_angle,
            cos_angle,
            mask,
            modality,
        ],
        dim=1,
    )


class LesionEvidenceBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_projection = nn.Sequential(
            nn.Conv2d(FUSION_DIM, FUSION_DIM, 1, bias=False),
            nn.GroupNorm(8, FUSION_DIM),
            nn.SiLU(),
        )
        self.queries = nn.Parameter(torch.randn(7, FUSION_DIM) * 0.02)
        self.presence_weight = nn.Parameter(torch.randn(7, FUSION_DIM) * 0.02)
        self.presence_bias = nn.Parameter(torch.zeros(7))
        self.embedding_evidence_weight = nn.Parameter(
            torch.randn(7, FUSION_DIM) * 0.02
        )

    def forward(self, feature_map):
        projected = self.map_projection(feature_map)
        batch, dimension, height, width = projected.shape
        tokens = projected.flatten(2).transpose(1, 2)
        queries = F.normalize(self.queries, dim=1)
        normalized_tokens = F.normalize(tokens, dim=2)
        attention_logits = torch.einsum(
            "bnd,qd->bqn", normalized_tokens, queries
        ) / math.sqrt(dimension)
        attention = torch.softmax(attention_logits, dim=2)
        embeddings = torch.einsum("bqn,bnd->bqd", attention, tokens)
        presence_logits = (
            embeddings * self.presence_weight.unsqueeze(0)
        ).sum(dim=2) / math.sqrt(dimension) + self.presence_bias.unsqueeze(0)
        spatial_maps = torch.sigmoid(attention_logits).view(
            batch, 7, height, width
        )
        top_k = max(1, int(height * width * TOP_K_FRACTION))
        burden = (
            spatial_maps
            .flatten(2)
            .topk(top_k, dim=2)
            .values
            .mean(dim=2)
        )
        embedding_signal = torch.sigmoid(
            (
                embeddings
                * self.embedding_evidence_weight.unsqueeze(0)
            ).sum(dim=2) / math.sqrt(dimension)
        )
        presence = torch.sigmoid(presence_logits)
        evidence_strength = (presence + burden + embedding_signal) / 3.0
        spatial_entropy = -(
            attention.clamp_min(1e-8)
            * attention.clamp_min(1e-8).log()
        ).sum(dim=2) / math.log(height * width)
        return {
            "presence_logits": presence_logits,
            "presence": presence,
            "spatial_maps": spatial_maps,
            "burden": burden,
            "embeddings": embeddings,
            "evidence_strength": evidence_strength,
            "spatial_entropy": spatial_entropy,
        }


class ReliabilityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_embedding = nn.Embedding(2, 4)
        self.network = nn.Sequential(
            nn.Linear(9, 16), nn.SiLU(), nn.Linear(16, 1)
        )

    def forward(
        self,
        clinical,
        latent,
        evidence,
        modality_index,
        quality,
    ):
        clinical_entropy = normalized_entropy(clinical)
        latent_entropy = normalized_entropy(latent)
        disagreement = js_divergence(clinical, latent)
        lesion_entropy = binary_entropy(evidence).mean(dim=1)
        modality = self.modality_embedding(modality_index)
        inputs = torch.cat(
            [
                clinical_entropy.unsqueeze(1),
                latent_entropy.unsqueeze(1),
                disagreement.unsqueeze(1),
                lesion_entropy.unsqueeze(1),
                modality,
                quality.view(-1, 1),
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.network(inputs)).squeeze(1)
        final = (
            gate.unsqueeze(1) * clinical
            + (1.0 - gate).unsqueeze(1) * latent
        )
        final = final / final.sum(dim=1, keepdim=True)
        return {
            "gate": gate,
            "probabilities": final,
            "expected_grade": expected_grade(final),
            "clinical_entropy": clinical_entropy,
            "latent_entropy": latent_entropy,
            "js_disagreement": disagreement,
            "lesion_entropy": lesion_entropy,
        }


class FOVEADRTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        official = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1,
            progress=False,
        )
        official.classifier = nn.Identity()
        self.encoder = official
        self.fpn = MultiScaleFPN()
        self.coordinate_fusion = nn.Sequential(
            nn.Conv2d(
                FUSION_DIM + 7,
                FUSION_DIM,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, FUSION_DIM),
            nn.SiLU(),
        )
        self.lesion_bank = LesionEvidenceBank()
        self.clinical_reasoner = ClinicalStateReasoner()
        self.latent_head = OrdinalHead(1280, 5)
        self.reliability_gate = ReliabilityGate()
        self.oct_head = OrdinalHead(1280, 3)

    def encode(self, image):
        value = image
        selected = []
        for index, stage in enumerate(self.encoder.features):
            value = stage(value)
            if index in FPN_TAPS:
                selected.append(value)
        return selected

    def fuse(
        self,
        evidence,
        latent_probabilities,
        modality_index,
        quality,
    ):
        clinical = self.clinical_reasoner(evidence)
        fusion = self.reliability_gate(
            clinical["probabilities"],
            latent_probabilities,
            evidence,
            modality_index,
            quality,
        )
        return {"clinical": clinical, "fusion": fusion}

    def counterfactual(
        self,
        evidence,
        latent_probabilities,
        modality_index,
        quality,
        original,
    ):
        no_lesion = torch.zeros_like(evidence)
        remove_pdr = torch.cat(
            [evidence[:, :4], torch.zeros_like(evidence[:, 4:])],
            dim=1,
        )
        add_pdr = torch.cat(
            [evidence[:, :4], torch.ones_like(evidence[:, 4:])],
            dim=1,
        )
        no_prediction = self.fuse(
            no_lesion,
            latent_probabilities,
            modality_index,
            quality,
        )
        remove_prediction = self.fuse(
            remove_pdr,
            latent_probabilities,
            modality_index,
            quality,
        )
        add_prediction = self.fuse(
            add_pdr,
            latent_probabilities,
            modality_index,
            quality,
        )
        original_grade = original["fusion"]["expected_grade"]
        any_support = 1.0 - torch.prod(1.0 - evidence, dim=1)
        pdr_support = 1.0 - torch.prod(1.0 - evidence[:, 4:], dim=1)
        remove_all_term = F.relu(
            no_prediction["fusion"]["expected_grade"]
            + COUNTERFACTUAL_MARGIN * any_support
            - original_grade
        ).mean()
        remove_pdr_term = F.relu(
            remove_prediction["fusion"]["expected_grade"]
            + COUNTERFACTUAL_MARGIN * pdr_support
            - original_grade
        ).mean()
        add_pdr_term = F.relu(
            original_grade
            + COUNTERFACTUAL_MARGIN * (1.0 - pdr_support)
            - add_prediction["fusion"]["expected_grade"]
        ).mean()
        return {
            "remove_all": remove_all_term,
            "remove_pdr": remove_pdr_term,
            "add_pdr": add_pdr_term,
            "total": remove_all_term + remove_pdr_term + add_pdr_term,
        }

    def forward_mixed(
        self,
        images,
        fundus_mask,
        fov_masks,
        modality_indices,
        quality,
    ):
        # The encoder is deliberately evaluated once on all 16 images so
        # EfficientNet BatchNorm sees the locked full micro-batch.
        all_features = self.encode(images)
        output = {"fundus": None, "oct": None}

        if bool(fundus_mask.any()):
            fundus_features = [
                feature[fundus_mask]
                for feature in all_features
            ]
            pyramid = self.fpn(fundus_features)
            fundus_fov = fov_masks[fundus_mask]
            fundus_modality = modality_indices[fundus_mask]
            fundus_quality = quality[fundus_mask]
            coordinates = make_coordinate_channels(
                fundus_fov,
                fundus_modality,
                pyramid.shape[-2:],
            )
            fused_map = self.coordinate_fusion(
                torch.cat([pyramid, coordinates], dim=1)
            )
            lesions = self.lesion_bank(fused_map)
            pooled = F.adaptive_avg_pool2d(
                fundus_features[-1], 1
            ).flatten(1)
            latent = self.latent_head(pooled)
            original = self.fuse(
                lesions["evidence_strength"],
                latent["probabilities"],
                fundus_modality,
                fundus_quality,
            )
            counterfactual = self.counterfactual(
                lesions["evidence_strength"],
                latent["probabilities"],
                fundus_modality,
                fundus_quality,
                original,
            )
            output["fundus"] = {
                "lesions": lesions,
                "latent": latent,
                "original": original,
                "counterfactual": counterfactual,
            }

        oct_mask = ~fundus_mask
        if bool(oct_mask.any()):
            oct_pooled = F.adaptive_avg_pool2d(
                all_features[-1][oct_mask], 1
            ).flatten(1)
            output["oct"] = self.oct_head(oct_pooled)

        return output


def grade_emd(probabilities, target):
    one_hot = F.one_hot(
        target,
        num_classes=probabilities.shape[1],
    ).to(probabilities.dtype)
    return (
        probabilities.cumsum(dim=1)
        - one_hot.cumsum(dim=1)
    ).abs().mean(dim=1)


def random_uniform(low, high):
    return float(torch.empty(1).uniform_(low, high).item())


def color_jitter_tensor(image, amount):
    operations = ["brightness", "contrast", "saturation"]
    order = torch.randperm(len(operations)).tolist()
    for index in order:
        operation = operations[index]
        factor = random_uniform(1.0 - amount, 1.0 + amount)
        if operation == "brightness":
            image = TF.adjust_brightness(image, factor)
        elif operation == "contrast":
            image = TF.adjust_contrast(image, factor)
        else:
            image = TF.adjust_saturation(image, factor)
    return image.clamp(0.0, 1.0)


class FOVEADRTrainDataset(Dataset):
    def __init__(self, table):
        self.table = table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        path = Path(str(row["local_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = int(row["file_size_bytes"])
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Cache size mismatch: {path}")

        with Image.open(path) as image:
            raw = TF.to_tensor(image.convert("RGB"))

        raw = TF.resize(
            raw,
            [INPUT_SIZE, INPUT_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        modality = str(row["modality"])
        if float(torch.rand(1).item()) < 0.5:
            raw = TF.hflip(raw)

        if modality in {"CFP", "UWF"}:
            angle = random_uniform(-10.0, 10.0)
            raw = TF.rotate(
                raw,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            raw = color_jitter_tensor(raw, 0.10)
        else:
            angle = random_uniform(-5.0, 5.0)
            raw = TF.rotate(
                raw,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            raw = color_jitter_tensor(raw, 0.05)

        quality = raw.std().clamp(0.0, 1.0)
        fov_mask = (
            raw.amax(dim=0, keepdim=True) > (8.0 / 255.0)
        ).to(raw.dtype)
        normalized = TF.normalize(
            raw,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        is_fundus = modality in {"CFP", "UWF"}
        modality_index = 0 if modality == "CFP" else (1 if modality == "UWF" else 2)
        lesion_values = []
        for column in LESION_COLUMNS:
            value = row[column]
            lesion_values.append(
                0.0 if pd.isna(value) else float(value)
            )

        return {
            "image": normalized,
            "fov_mask": fov_mask,
            "quality": torch.tensor(float(quality)),
            "fundus_mask": torch.tensor(is_fundus, dtype=torch.bool),
            "modality_index": torch.tensor(modality_index, dtype=torch.long),
            "grade": torch.tensor(int(row["grade"]), dtype=torch.long),
            "grade_loss_mask": torch.tensor(float(row["grade_loss_mask"])),
            "lesion_loss_mask": torch.tensor(float(row["lesion_loss_mask"])),
            "lesion_sensitivity_mask": torch.tensor(float(row["lesion_sensitivity_mask"])),
            "lesion_target": torch.tensor(lesion_values, dtype=torch.float32),
            "modality": modality,
            "relative_path": str(row["relative_path"]),
        }


class LockedWeightedBatchSampler(Sampler):
    def __init__(self, table, seed, optimizer_steps):
        self.batches = []
        generator = torch.Generator().manual_seed(seed)
        modality_indices = {}
        modality_probabilities = {}

        for modality in ["CFP", "UWF", "OCT"]:
            indices = torch.tensor(
                table.index[table["modality"] == modality].tolist(),
                dtype=torch.long,
            )
            probabilities = torch.tensor(
                table.loc[indices.tolist(), "locked_sampling_weight"].to_numpy(),
                dtype=torch.double,
            )
            probabilities = probabilities / probabilities.sum()
            modality_indices[modality] = indices
            modality_probabilities[modality] = probabilities

        for optimizer_step in range(optimizer_steps):
            for composition in [MICROBATCH_A, MICROBATCH_B]:
                batch = []
                for modality in ["CFP", "UWF", "OCT"]:
                    draws = torch.multinomial(
                        modality_probabilities[modality],
                        num_samples=composition[modality],
                        replacement=True,
                        generator=generator,
                    )
                    selected = modality_indices[modality][draws]
                    batch.extend(selected.tolist())
                permutation = torch.randperm(
                    len(batch), generator=generator
                ).tolist()
                batch = [batch[index] for index in permutation]
                self.batches.append(batch)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


def worker_seed(worker_id):
    value = SEED + worker_id
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)


def resolve_sampler_weight_column(table):
    candidates = [
        "sampling_probability_within_modality",
        "sampler_probability_within_modality",
        "sampling_probability",
        "per_row_weight",
        "sampler_weight",
    ]
    for column in candidates:
        if column in table.columns:
            return column
    raise RuntimeError(
        "No recognized sampling-weight column exists in the locked sampler artifact."
    )


def build_training_table():
    split = pd.read_csv(SPLIT_PATH)
    train = split[split["development_split"] == "TRAIN"].copy()
    train["relative_path"] = train["relative_path"].astype(str).str.replace(
        "\\", "/", regex=False
    )
    cache = pd.read_csv(CACHE_MANIFEST_PATH)
    sampler = pd.read_csv(SAMPLER_PATH)
    cache["relative_path"] = cache["relative_path"].astype(str).str.replace(
        "\\", "/", regex=False
    )
    sampler["relative_path"] = sampler["relative_path"].astype(str).str.replace(
        "\\", "/", regex=False
    )
    weight_column = resolve_sampler_weight_column(sampler)
    sampler_small = sampler[["relative_path", weight_column]].copy()
    sampler_small = sampler_small.rename(
        columns={weight_column: "locked_sampling_weight"}
    )
    cache_small = cache[
        ["relative_path", "local_path", "file_size_bytes", "image_sha256"]
    ].copy()
    table = train.merge(
        cache_small,
        on="relative_path",
        how="inner",
        validate="one_to_one",
    ).merge(
        sampler_small,
        on="relative_path",
        how="inner",
        validate="one_to_one",
    )
    table = table.sort_values("relative_path").reset_index(drop=True)
    table["locked_sampling_weight"] = pd.to_numeric(
        table["locked_sampling_weight"], errors="raise"
    )
    if not np.isfinite(table["locked_sampling_weight"]).all():
        raise RuntimeError("Non-finite locked sampler weights.")
    if not (table["locked_sampling_weight"] > 0).all():
        raise RuntimeError("Non-positive locked sampler weights.")
    return table, weight_column


def build_optimizer(model):
    groups = {
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

    optimizer_groups = [
        {
            "params": groups["backbone_decay"],
            "lr": BACKBONE_LR,
            "weight_decay": WEIGHT_DECAY,
            "group_name": "backbone_decay",
        },
        {
            "params": groups["backbone_no_decay"],
            "lr": BACKBONE_LR,
            "weight_decay": 0.0,
            "group_name": "backbone_no_decay",
        },
        {
            "params": groups["new_decay"],
            "lr": NEW_MODULE_LR,
            "weight_decay": WEIGHT_DECAY,
            "group_name": "new_decay",
        },
        {
            "params": groups["new_no_decay"],
            "lr": NEW_MODULE_LR,
            "weight_decay": 0.0,
            "group_name": "new_no_decay",
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_groups)
    return optimizer, {
        name: len(parameters)
        for name, parameters in groups.items()
    }


def lr_multiplier(step):
    if step < WARMUP_STEPS:
        return float(step + 1) / float(WARMUP_STEPS)
    progress = float(step - WARMUP_STEPS) / float(
        max(1, TOTAL_OPTIMIZER_STEPS - WARMUP_STEPS)
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


def compute_mixed_losses(output, batch, lesion_pos_weight, loss_weights):
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
        sensitivity_mask = batch["lesion_sensitivity_mask"][fundus_mask]
        lesion_target = batch["lesion_target"][fundus_mask]
        fundus = output["fundus"]
        final_probabilities = fundus["original"]["fusion"]["probabilities"]
        clinical_probabilities = fundus["original"]["clinical"]["probabilities"]
        parts["grade_NLL"] = masked_mean(
            -final_probabilities
            .gather(1, target[:, None])
            .clamp_min(1e-7)
            .log()
            .squeeze(1),
            grade_mask,
        )
        parts["ordinal_EMD"] = masked_mean(
            grade_emd(final_probabilities, target),
            grade_mask,
        )
        lesion_per_sample = F.binary_cross_entropy_with_logits(
            fundus["lesions"]["presence_logits"],
            lesion_target,
            pos_weight=lesion_pos_weight,
            reduction="none",
        ).mean(dim=1)
        parts["lesion_BCE"] = masked_mean(
            lesion_per_sample,
            lesion_mask,
        )
        parts["clinical_alignment"] = masked_mean(
            -clinical_probabilities
            .gather(1, target[:, None])
            .clamp_min(1e-7)
            .log()
            .squeeze(1),
            sensitivity_mask,
        )
        parts["counterfactual"] = fundus["counterfactual"]["total"]
        parts["route_agreement"] = masked_mean(
            fundus["original"]["fusion"]["js_disagreement"],
            sensitivity_mask,
        )

    if bool(oct_mask.any()):
        oct_target = batch["grade"][oct_mask]
        oct_probabilities = output["oct"]["probabilities"]
        parts["OCT_NLL"] = -oct_probabilities.gather(
            1, oct_target[:, None]
        ).clamp_min(1e-7).log().mean()

    total = (
        loss_weights["fundus_grade_cross_entropy_weight"] * parts["grade_NLL"]
        + loss_weights["fundus_ordinal_EMD_weight"] * parts["ordinal_EMD"]
        + loss_weights["lesion_BCE_weight"] * parts["lesion_BCE"]
        + loss_weights["clinical_state_alignment_weight"] * parts["clinical_alignment"]
        + loss_weights["lesion_counterfactual_intervention_weight"] * parts["counterfactual"]
        + loss_weights["masked_route_agreement_weight"] * parts["route_agreement"]
        + loss_weights["OCT_DME_task_weight"] * parts["OCT_NLL"]
    )
    return total, parts


def parameter_snapshot(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def module_parameter_deltas(model, initial):
    prefixes = {
        "encoder": "encoder.",
        "fpn": "fpn.",
        "coordinate_fusion": "coordinate_fusion.",
        "lesion_bank": "lesion_bank.",
        "clinical_reasoner": "clinical_reasoner.",
        "latent_head": "latent_head.",
        "reliability_gate": "reliability_gate.",
        "oct_head": "oct_head.",
    }
    squared = {name: 0.0 for name in prefixes}
    for name, parameter in model.named_parameters():
        if name not in initial:
            continue
        difference = parameter.detach().cpu().float() - initial[name].float()
        for module_name, prefix in prefixes.items():
            if name.startswith(prefix):
                squared[module_name] += float(difference.square().sum().item())
                break
    return {
        name: math.sqrt(value)
        for name, value in squared.items()
    }


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def inspect_source_freeze_target():
    source_path = Path(__file__).resolve()
    source_sha = sha256_file(source_path)
    if FROZEN_PRODUCTION_SOURCE.exists():
        frozen_sha = sha256_file(FROZEN_PRODUCTION_SOURCE)
        if frozen_sha != source_sha:
            raise RuntimeError(
                "A different production source is already frozen. "
                f"Existing={frozen_sha}, current={source_sha}"
            )
        action = "ALREADY_FROZEN_IDENTICAL"
    else:
        action = "PENDING_FREEZE_AFTER_SUCCESSFUL_PREFLIGHT"
    return source_path, source_sha, action


def freeze_verified_source(source_path, source_sha):
    FROZEN_PRODUCTION_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_PRODUCTION_SOURCE.exists():
        frozen_sha = sha256_file(FROZEN_PRODUCTION_SOURCE)
        action = "ALREADY_FROZEN_IDENTICAL"
    else:
        shutil.copy2(source_path, FROZEN_PRODUCTION_SOURCE)
        frozen_sha = sha256_file(FROZEN_PRODUCTION_SOURCE)
        action = "COPIED_AND_FROZEN_AFTER_PREFLIGHT"
    if frozen_sha != source_sha:
        raise RuntimeError("Production source freeze verification failed.")
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-steps", type=int, default=PILOT_OPTIMIZER_STEPS)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.pilot_steps != PILOT_OPTIMIZER_STEPS:
        raise RuntimeError("This frozen preflight requires exactly 64 optimizer steps.")

    print("=" * 80)
    print("STAGE 0G-4D-3B-R2 — LOSS-CONTRACT RECOVERY AND 64-STEP TRAIN-ONLY PREFLIGHT")
    print("=" * 80)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("\n[1] FROZEN ARTIFACT VERIFICATION")
    print("-" * 80)
    required = {
        "split": SPLIT_PATH,
        "candidate": CANDIDATE_PATH,
        "protocol": PROTOCOL_PATH,
        "sampler": SAMPLER_PATH,
        "lesion_weights": LESION_WEIGHT_PATH,
        "cache_manifest": CACHE_MANIFEST_PATH,
        "cache_evidence": CACHE_EVIDENCE_PATH,
        "source_freeze_manifest": SOURCE_FREEZE_MANIFEST_PATH,
    }
    for name, path in required.items():
        print(f"{name:24s}: exists={path.exists()}")
        if not path.exists():
            raise FileNotFoundError(path)

    candidate_hash_ok = sha256_file(CANDIDATE_PATH) == EXPECTED_CANDIDATE_SHA
    protocol_hash_ok = sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA
    split_hash_ok = sha256_file(SPLIT_PATH) == EXPECTED_SPLIT_SHA
    cache_manifest_hash_ok = sha256_file(CACHE_MANIFEST_PATH) == EXPECTED_CACHE_MANIFEST_SHA
    with open(CACHE_EVIDENCE_PATH) as file:
        cache_evidence = json.load(file)
    cache_evidence_pass = all_evidence_gates_pass(cache_evidence)
    with open(SOURCE_FREEZE_MANIFEST_PATH) as file:
        source_freeze_manifest = json.load(file)
    audited_source_hash_ok = nested_json_contains(
        source_freeze_manifest,
        EXPECTED_AUDITED_SOURCE_SHA,
    )
    with open(CANDIDATE_PATH) as file:
        candidate = json.load(file)
    loss_weights = candidate.get("candidate_loss", {})
    loss_contract_ok = (
        isinstance(loss_weights, dict)
        and all(
            key in loss_weights
            and isinstance(loss_weights[key], (int, float))
            and math.isclose(
                float(loss_weights[key]),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key, expected_value in EXPECTED_LOSS_WEIGHTS.items()
        )
    )
    mask_policy_contract_ok = (
        isinstance(loss_weights, dict)
        and loss_weights.get("mask_policies") == EXPECTED_MASK_POLICIES
    )

    print("Candidate hash verified  :", candidate_hash_ok)
    print("Protocol hash verified   :", protocol_hash_ok)
    print("Split hash verified      :", split_hash_ok)
    print("Cache manifest verified  :", cache_manifest_hash_ok)
    print("Cache evidence PASS      :", cache_evidence_pass)
    print("Audited source hash known:", audited_source_hash_ok)
    print("Candidate loss preserved :", loss_contract_ok)
    print("Mask policies preserved  :", mask_policy_contract_ok)

    if not all([
        candidate_hash_ok,
        protocol_hash_ok,
        split_hash_ok,
        cache_manifest_hash_ok,
        cache_evidence_pass,
        audited_source_hash_ok,
        loss_contract_ok,
        mask_policy_contract_ok,
    ]):
        raise RuntimeError("Frozen artifact verification failed.")

    print("\n[2] PRODUCTION SOURCE FREEZE")
    print("-" * 80)
    source_path, source_sha, source_action = inspect_source_freeze_target()
    print("Source action :", source_action)
    print("Source SHA-256:", source_sha)
    print("Freeze target :", FROZEN_PRODUCTION_SOURCE)

    print("\n[3] TRAIN-ONLY DATA AND SAMPLER")
    print("-" * 80)
    table, sampler_weight_column = build_training_table()
    modality_counts = {
        str(key): int(value)
        for key, value in table["modality"].value_counts().to_dict().items()
    }
    exact_train_table = len(table) == 13251 and modality_counts == EXPECTED_TRAIN_COUNTS
    only_train = bool(table["development_split"].eq("TRAIN").all())
    all_cache_files_present = bool(
        all(
            Path(str(row["local_path"])).is_file()
            and Path(str(row["local_path"])).stat().st_size == int(row["file_size_bytes"])
            for _, row in table.iterrows()
        )
    )
    lesion_weight_table = pd.read_csv(LESION_WEIGHT_PATH)
    lesion_pos_weight = torch.tensor(
        lesion_weight_table["sqrt_pos_weight"].to_numpy(dtype=np.float32)
    )
    lesion_weights_valid = (
        len(lesion_pos_weight) == 7
        and bool(torch.isfinite(lesion_pos_weight).all())
        and bool(torch.all(lesion_pos_weight > 0))
    )
    sampler = LockedWeightedBatchSampler(table, SEED, args.pilot_steps)
    batch_shapes_ok = (
        len(sampler) == MICROBATCHES_PER_PILOT
        and all(len(batch) == MICRO_BATCH for batch in sampler.batches)
    )
    sampled_counts = {"CFP": 0, "UWF": 0, "OCT": 0}
    for batch_indices in sampler.batches:
        for modality, count in table.loc[batch_indices, "modality"].value_counts().items():
            sampled_counts[str(modality)] += int(count)
    expected_sampled_counts = {
        modality: count * args.pilot_steps
        for modality, count in EXPECTED_EFFECTIVE_COMPOSITION.items()
    }
    composition_ok = sampled_counts == expected_sampled_counts

    print("TRAIN rows              :", len(table), modality_counts)
    print("Sampler weight column   :", sampler_weight_column)
    print("Micro-batches           :", len(sampler))
    print("Sampled pilot counts    :", sampled_counts)
    print("Expected pilot counts   :", expected_sampled_counts)
    print("All cache files present :", all_cache_files_present)

    if not all([
        exact_train_table,
        only_train,
        all_cache_files_present,
        lesion_weights_valid,
        batch_shapes_ok,
        composition_ok,
    ]):
        raise RuntimeError("TRAIN-only production data gate failed.")

    dataset = FOVEADRTrainDataset(table)
    loader_generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed if args.workers > 0 else None,
        generator=loader_generator,
    )

    print("\n[4] GPU MODEL, OPTIMIZER, AMP, AND SCHEDULER")
    print("-" * 80)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this preflight.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        print("WARNING: frozen calibration was performed on Tesla T4; active GPU is", gpu_name)

    weight_enum = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    checkpoint_name = Path(weight_enum.url).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        _ = efficientnet_v2_s(weights=weight_enum, progress=True)
    official_weight_hash_ok = (
        checkpoint_path.exists()
        and sha256_file(checkpoint_path) == EXPECTED_OFFICIAL_WEIGHT_SHA
    )
    if not official_weight_hash_ok:
        raise RuntimeError("Official EfficientNetV2-S checkpoint hash mismatch.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = FOVEADRTrainingModel().to(device)
    model.train()
    initial_parameters = parameter_snapshot(model)
    optimizer, optimizer_group_counts = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    lesion_pos_weight = lesion_pos_weight.to(device)

    print("GPU                    :", gpu_name)
    print("Model parameters       :", sum(p.numel() for p in model.parameters()))
    print("Optimizer              : AdamW")
    print("Optimizer groups       :", optimizer_group_counts)
    print("Gradient accumulation  :", GRADIENT_ACCUMULATION)
    print("Effective batch        :", EFFECTIVE_BATCH)
    print("AMP                    : float16 + dynamic GradScaler")
    print("Pilot optimizer steps  :", args.pilot_steps)
    print("Warm-up contract steps :", WARMUP_STEPS)

    print("\n[5] 64-STEP TRAIN-ONLY OPTIMIZATION PREFLIGHT")
    print("-" * 80)
    logs = []
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    all_losses_finite = True
    all_gradients_finite = True
    all_gradients_nonzero = True
    no_skipped_optimizer_steps = True
    start_time = time.perf_counter()

    for optimizer_step in range(1, args.pilot_steps + 1):
        step_loss = 0.0
        component_sums = {
            "grade_NLL": 0.0,
            "ordinal_EMD": 0.0,
            "lesion_BCE": 0.0,
            "clinical_alignment": 0.0,
            "counterfactual": 0.0,
            "route_agreement": 0.0,
            "OCT_NLL": 0.0,
        }
        step_modalities = {"CFP": 0, "UWF": 0, "OCT": 0}
        step_start = time.perf_counter()

        for accumulation_index in range(GRADIENT_ACCUMULATION):
            batch = move_batch(next(iterator), device)
            for modality in batch["modality"]:
                step_modalities[str(modality)] += 1

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                output = model.forward_mixed(
                    batch["image"],
                    batch["fundus_mask"],
                    batch["fov_mask"],
                    batch["modality_index"],
                    batch["quality"],
                )
                total_loss, parts = compute_mixed_losses(
                    output,
                    batch,
                    lesion_pos_weight,
                    loss_weights,
                )
                scaled_loss = total_loss / GRADIENT_ACCUMULATION

            finite_loss = bool(torch.isfinite(total_loss).all())
            all_losses_finite = all_losses_finite and finite_loss
            if not finite_loss:
                raise FloatingPointError(
                    f"Non-finite loss at optimizer step {optimizer_step}."
                )
            scaler.scale(scaled_loss).backward()
            step_loss += float(scaled_loss.detach().cpu())
            for name, value in parts.items():
                component_sums[name] += float(value.detach().cpu()) / GRADIENT_ACCUMULATION

        scaler.unscale_(optimizer)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradients_finite = len(gradients) > 0 and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        gradients_nonzero = len(gradients) > 0 and any(
            bool(torch.any(gradient != 0)) for gradient in gradients
        )
        all_gradients_finite = all_gradients_finite and gradients_finite
        all_gradients_nonzero = all_gradients_nonzero and gradients_nonzero
        if not gradients_finite or not gradients_nonzero:
            raise FloatingPointError(
                f"Invalid gradients at optimizer step {optimizer_step}."
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP_NORM,
        )
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if scale_after < scale_before:
            no_skipped_optimizer_steps = False
            raise FloatingPointError(
                f"GradScaler overflow at optimizer step {optimizer_step}."
            )
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        step_seconds = time.perf_counter() - step_start
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        learning_rates = {
            group["group_name"]: float(group["lr"])
            for group in optimizer.param_groups
        }
        record = {
            "optimizer_step": optimizer_step,
            "microbatches_completed": optimizer_step * GRADIENT_ACCUMULATION,
            "samples_completed": optimizer_step * EFFECTIVE_BATCH,
            "total_loss": step_loss,
            **component_sums,
            "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
            "grad_scaler_scale": scale_after,
            "backbone_lr": learning_rates["backbone_decay"],
            "new_module_lr": learning_rates["new_decay"],
            "step_seconds": step_seconds,
            "peak_reserved_vram_gb": peak_reserved_gb,
            "CFP_samples": step_modalities["CFP"],
            "UWF_samples": step_modalities["UWF"],
            "OCT_samples": step_modalities["OCT"],
        }
        logs.append(record)

        if optimizer_step == 1 or optimizer_step % 8 == 0 or optimizer_step == args.pilot_steps:
            print(
                f"Step {optimizer_step:02d}/{args.pilot_steps} | "
                f"loss={step_loss:.6f} | "
                f"grad={float(grad_norm):.4f} | "
                f"scale={scale_after:.0f} | "
                f"VRAM={peak_reserved_gb:.3f} GB | "
                f"time={step_seconds:.3f}s"
            )

    elapsed_seconds = time.perf_counter() - start_time
    del iterator
    parameter_deltas = module_parameter_deltas(model, initial_parameters)
    all_required_modules_updated = all(
        math.isfinite(value) and value > 0.0
        for value in parameter_deltas.values()
    )
    log_table = pd.DataFrame(logs)
    loss_start = float(log_table.head(8)["total_loss"].mean())
    loss_end = float(log_table.tail(8)["total_loss"].mean())
    all_step_compositions_exact = bool(
        (
            (log_table["CFP_samples"] == 15)
            & (log_table["UWF_samples"] == 13)
            & (log_table["OCT_samples"] == 4)
        ).all()
    )
    peak_reserved_gb = float(log_table["peak_reserved_vram_gb"].max())
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    peak_fraction = peak_reserved_gb / total_vram_gb

    print("\n[6] PREFLIGHT SUMMARY")
    print("-" * 80)
    print("Optimizer steps completed :", len(log_table))
    print("Micro-batches completed   :", len(log_table) * GRADIENT_ACCUMULATION)
    print("TRAIN samples processed   :", len(log_table) * EFFECTIVE_BATCH)
    print("Mean first-8 loss         :", f"{loss_start:.6f}")
    print("Mean last-8 loss          :", f"{loss_end:.6f}")
    print("Elapsed seconds           :", f"{elapsed_seconds:.3f}")
    print("Peak reserved VRAM GB     :", f"{peak_reserved_gb:.3f}")
    print("Peak VRAM fraction        :", f"{peak_fraction:.3f}")
    print("Parameter L2 deltas:")
    for name, value in parameter_deltas.items():
        print(f"  {name:22s}: {value:.9f}")

    print("\n[7] SAVE PREFLIGHT ARTIFACTS")
    print("-" * 80)
    source_action = freeze_verified_source(source_path, source_sha)
    print("Production source action:", source_action)
    PILOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PILOT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_IMPLEMENTATION_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    log_table.to_csv(PILOT_LOG_PATH, index=False)
    checkpoint = {
        "stage": "0G-4D-3B-R2",
        "purpose": "TRAIN_ONLY_64_STEP_PREFLIGHT_NOT_FINAL_TRAINING",
        "source_sha256": source_sha,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA,
        "training_protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA,
        "optimizer_steps": args.pilot_steps,
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
    }
    torch.save(checkpoint, PILOT_CHECKPOINT_PATH)
    checkpoint_sha = sha256_file(PILOT_CHECKPOINT_PATH)
    log_sha = sha256_file(PILOT_LOG_PATH)

    implementation_contract = {
        "contract": "FOVEA_DR_TRAINING_IMPLEMENTATION_V1",
        "status": "FROZEN_AFTER_TRAIN_ONLY_PREFLIGHT",
        "production_source": str(FROZEN_PRODUCTION_SOURCE),
        "production_source_sha256": source_sha,
        "audited_integration_source_sha256": EXPECTED_AUDITED_SOURCE_SHA,
        "joint_encoder_microbatch": True,
        "joint_encoder_reason": "EfficientNet BatchNorm observes the complete locked micro-batch of 16 before fundus/OCT task branching.",
        "input": "512x512_RGB_IMAGENET_V1_NORMALIZATION",
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch": EFFECTIVE_BATCH,
        "microbatch_A": MICROBATCH_A,
        "microbatch_B": MICROBATCH_B,
        "optimizer": "AdamW",
        "backbone_lr": BACKBONE_LR,
        "new_module_lr": NEW_MODULE_LR,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "AMP": "float16_dynamic_GradScaler",
        "scheduler": "linear_warmup_then_cosine",
        "warmup_steps": WARMUP_STEPS,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "candidate_loss": EXPECTED_LOSS_WEIGHTS,
        "mask_policies": EXPECTED_MASK_POLICIES,
        "lesion_positive_weighting": "sqrt_negative_positive_ratio",
        "validation_accessed": False,
        "locked_test_accessed": False,
    }
    with open(TRAINING_IMPLEMENTATION_CONTRACT, "w") as file:
        json.dump(implementation_contract, file, indent=2)
    implementation_contract_sha = sha256_file(TRAINING_IMPLEMENTATION_CONTRACT)

    gates = {
        "candidate_hash_verified": candidate_hash_ok,
        "training_protocol_hash_verified": protocol_hash_ok,
        "frozen_split_hash_verified": split_hash_ok,
        "full_TRAIN_cache_manifest_hash_verified": cache_manifest_hash_ok,
        "full_TRAIN_cache_evidence_passed": cache_evidence_pass,
        "audited_integration_source_hash_verified": audited_source_hash_ok,
        "candidate_loss_contract_preserved": loss_contract_ok,
        "candidate_mask_policy_contract_preserved": mask_policy_contract_ok,
        "production_source_frozen_and_hash_verified": sha256_file(FROZEN_PRODUCTION_SOURCE) == source_sha,
        "exact_13251_TRAIN_rows_loaded": exact_train_table,
        "all_loaded_rows_are_TRAIN": only_train,
        "all_cache_files_present_and_size_matched": all_cache_files_present,
        "locked_sampler_weights_valid": True,
        "lesion_positive_weights_valid": lesion_weights_valid,
        "exact_128_microbatches_generated": batch_shapes_ok,
        "effective_modality_composition_exact": composition_ok and all_step_compositions_exact,
        "official_EfficientNetV2S_weight_SHA_verified": official_weight_hash_ok,
        "joint_encoder_used_for_full_microbatch_BatchNorm": True,
        "exact_64_optimizer_steps_completed": len(log_table) == PILOT_OPTIMIZER_STEPS,
        "all_pilot_losses_finite": all_losses_finite and bool(np.isfinite(log_table["total_loss"]).all()),
        "all_gradients_finite": all_gradients_finite,
        "all_gradients_nonzero": all_gradients_nonzero,
        "no_AMP_overflow_or_skipped_step": no_skipped_optimizer_steps,
        "all_required_model_modules_updated": all_required_modules_updated,
        "peak_reserved_VRAM_below_75_percent": peak_fraction < 0.75,
        "pilot_log_created": PILOT_LOG_PATH.exists(),
        "pilot_checkpoint_created": PILOT_CHECKPOINT_PATH.exists(),
        "training_implementation_contract_created": TRAINING_IMPLEMENTATION_CONTRACT.exists(),
        "validation_not_accessed": True,
        "locked_test_not_accessed": True,
    }

    print("\n" + "=" * 80)
    print("64-STEP PRODUCTION PREFLIGHT GATES")
    print("=" * 80)
    for name, passed in gates.items():
        print(f"{name:64s}: {passed}")
    if not all(gates.values()):
        raise RuntimeError("STAGE 0G-4D-3B-R2 preflight gate failed.")

    evidence = {
        "stage": "0G-4D-3B-R2",
        "status": "PASS",
        "production_source": str(FROZEN_PRODUCTION_SOURCE),
        "production_source_sha256": source_sha,
        "source_action": source_action,
        "training_implementation_contract": str(TRAINING_IMPLEMENTATION_CONTRACT),
        "training_implementation_contract_sha256": implementation_contract_sha,
        "pilot_log": str(PILOT_LOG_PATH),
        "pilot_log_sha256": log_sha,
        "pilot_checkpoint": str(PILOT_CHECKPOINT_PATH),
        "pilot_checkpoint_sha256": checkpoint_sha,
        "active_GPU": gpu_name,
        "PyTorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "Python": platform.python_version(),
        "optimizer_steps": int(len(log_table)),
        "microbatches": int(len(log_table) * GRADIENT_ACCUMULATION),
        "TRAIN_samples_processed_with_replacement": int(len(log_table) * EFFECTIVE_BATCH),
        "sampled_modality_counts": sampled_counts,
        "mean_first_8_step_loss": loss_start,
        "mean_last_8_step_loss": loss_end,
        "elapsed_seconds": elapsed_seconds,
        "peak_reserved_VRAM_GB": peak_reserved_gb,
        "peak_reserved_VRAM_fraction": peak_fraction,
        "parameter_L2_deltas": parameter_deltas,
        "gates": {name: bool(value) for name, value in gates.items()},
        "optimizer_created": True,
        "parameters_updated": True,
        "TRAIN_only_preflight_optimization_performed": True,
        "final_training_started": False,
        "validation_used": False,
        "locked_test_used": False,
        "external_UWF_accessed": False,
        "next_required_stage": "VALIDATION_ONLY_CACHE_PREPARATION_AND_BASELINE_TRAINING_LAUNCH",
    }
    with open(EVIDENCE_PATH, "w") as file:
        json.dump(evidence, file, indent=2)

    print("\nProduction source:")
    print(FROZEN_PRODUCTION_SOURCE)
    print("SHA-256:", source_sha)
    print("\nTraining implementation contract:")
    print(TRAINING_IMPLEMENTATION_CONTRACT)
    print("SHA-256:", implementation_contract_sha)
    print("\nPilot log:")
    print(PILOT_LOG_PATH)
    print("\nPilot checkpoint:")
    print(PILOT_CHECKPOINT_PATH)
    print("SHA-256:", checkpoint_sha)
    print("\nEvidence:")
    print(EVIDENCE_PATH)
    print("\nOptimizer created                 : YES — PREFLIGHT ONLY")
    print("Parameters updated                : YES — PREFLIGHT ONLY")
    print("Final baseline training started   : NO")
    print("Validation/test accessed          : NO")
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-3B-R2 STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
