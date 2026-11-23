from pathlib import Path
import copy, gc, hashlib, json, math, os, platform, random

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
CANDIDATE_PATH = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
MANIFEST_PATH = PROJECT_ROOT / "Audit/mmrdr_fovea_dr_train_integration_samples_v1.csv"
WEIGHT_EVIDENCE_PATH = PROJECT_ROOT / "Evidence/stage0g4c1b_efficientnetv2s_weight_acquisition.json"
CONTRACT_PATH = PROJECT_ROOT / "Metadata/mmrdr_fovea_dr_integration_implementation_contract_v1.json"
EVIDENCE_PATH = PROJECT_ROOT / "Evidence/stage0g4c2_train_only_end_to_end_backbone_integration_audit.json"
EXPECTED_CANDIDATE_SHA = "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181"
EXPECTED_MANIFEST_SHA = "2f90fd393fb9f738059e16288a827605ec1c4c71bd42b1059ba2d661bcf73fcf"
EXPECTED_WEIGHT_SHA = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
SEED = 20260806
INPUT_SIZE = 512
FPN_TAPS = [3, 5, 7]
FPN_CHANNELS = [64, 160, 1280]
FUSION_DIM = 128
TOP_K_FRACTION = 0.10
MARGIN = 0.25
LESION_COLUMNS = [
    "lesion_microaneurysm", "lesion_hard_exudate",
    "lesion_intraretinal_hemorrhage", "lesion_VB_IRMA",
    "lesion_neovascularization", "lesion_vitreous_hemorrhage",
    "lesion_retinal_detachment",
]

print("=" * 80)
print("STAGE 0G-4C-2 — TRAIN-ONLY END-TO-END BACKBONE INTEGRATION AUDIT")
print("=" * 80)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
DEVICE = torch.device("cpu")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def inv_softplus(value):
    return math.log(math.expm1(float(value)))

def ordered_distribution(score, thresholds, temperature=1.0):
    cumulative = torch.sigmoid(
        (score.unsqueeze(1) - thresholds.unsqueeze(0)) / temperature
    )
    probs = torch.cat(
        [1.0 - cumulative[:, :1],
         cumulative[:, :-1] - cumulative[:, 1:],
         cumulative[:, -1:]], dim=1,
    )
    probs = probs.clamp_min(1e-7)
    return probs / probs.sum(dim=1, keepdim=True)

def expected_grade(probs):
    grades = torch.arange(probs.shape[1], device=probs.device, dtype=probs.dtype)
    return (probs * grades.unsqueeze(0)).sum(dim=1)

def normalized_entropy(probs):
    p = probs.clamp_min(1e-7)
    return -(p * p.log()).sum(dim=1) / math.log(probs.shape[1])

def binary_entropy(values):
    p = values.clamp(1e-7, 1.0 - 1e-7)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log()) / math.log(2.0)

def js_divergence(p, q):
    p = p.clamp_min(1e-7); q = q.clamp_min(1e-7)
    m = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - m.log())).sum(dim=1)
        + (q * (q.log() - m.log())).sum(dim=1)
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
        probs = ordered_distribution(score, self.thresholds(), self.temperature)
        return {"score": score, "probabilities": probs,
                "expected_grade": expected_grade(probs)}

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
        probs = ordered_distribution(score, self.thresholds(), temperature=0.25)
        return {"score": score, "probabilities": probs,
                "expected_grade": expected_grade(probs),
                "positive_weights": weights}

class MultiScaleFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(ch, FUSION_DIM, 1) for ch in FPN_CHANNELS
        ])
        self.refine = nn.Sequential(
            nn.Conv2d(FUSION_DIM, FUSION_DIM, 3, padding=1, bias=False),
            nn.GroupNorm(8, FUSION_DIM), nn.SiLU(),
        )

    def forward(self, features):
        target_size = features[0].shape[-2:]
        projected = [
            F.interpolate(layer(feature), size=target_size, mode="bilinear", align_corners=False)
            if feature.shape[-2:] != target_size else layer(feature)
            for layer, feature in zip(self.laterals, features)
        ]
        return self.refine(sum(projected) / math.sqrt(len(projected)))

def make_coordinate_channels(fov_mask, modality_index, size):
    mask = F.interpolate(fov_mask, size=size, mode="nearest")
    b, _, h, w = mask.shape
    y = torch.linspace(-1.0, 1.0, h, device=mask.device, dtype=mask.dtype)
    x = torch.linspace(-1.0, 1.0, w, device=mask.device, dtype=mask.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.view(1, 1, h, w).expand(b, -1, -1, -1)
    yy = yy.view(1, 1, h, w).expand(b, -1, -1, -1)
    mass = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    cx = (xx * mask).sum(dim=(2, 3), keepdim=True) / mass
    cy = (yy * mask).sum(dim=(2, 3), keepdim=True) / mass
    dx, dy = xx - cx, yy - cy
    radius = torch.sqrt(dx.square() + dy.square() + 1e-8)
    max_radius = (radius * mask).flatten(1).amax(dim=1).view(b, 1, 1, 1).clamp_min(1e-4)
    radius_norm = (radius / max_radius).clamp(0.0, 1.5)
    sin_angle = dy / radius.clamp_min(1e-4)
    cos_angle = dx / radius.clamp_min(1e-4)
    mod = modality_index.to(mask.dtype).view(b, 1, 1, 1).expand(b, 1, h, w)
    return torch.cat([xx, yy, radius_norm, sin_angle, cos_angle, mask, mod], dim=1)

class LesionEvidenceBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_projection = nn.Sequential(
            nn.Conv2d(FUSION_DIM, FUSION_DIM, 1, bias=False),
            nn.GroupNorm(8, FUSION_DIM), nn.SiLU(),
        )
        self.queries = nn.Parameter(torch.randn(7, FUSION_DIM) * 0.02)
        self.presence_weight = nn.Parameter(torch.randn(7, FUSION_DIM) * 0.02)
        self.presence_bias = nn.Parameter(torch.zeros(7))
        self.embedding_evidence_weight = nn.Parameter(torch.randn(7, FUSION_DIM) * 0.02)

    def forward(self, feature_map):
        projected = self.map_projection(feature_map)
        b, d, h, w = projected.shape
        tokens = projected.flatten(2).transpose(1, 2)
        queries = F.normalize(self.queries, dim=1)
        normalized_tokens = F.normalize(tokens, dim=2)
        attention_logits = torch.einsum("bnd,qd->bqn", normalized_tokens, queries) / math.sqrt(d)
        attention = torch.softmax(attention_logits, dim=2)
        embeddings = torch.einsum("bqn,bnd->bqd", attention, tokens)
        presence_logits = (
            embeddings * self.presence_weight.unsqueeze(0)
        ).sum(dim=2) / math.sqrt(d) + self.presence_bias.unsqueeze(0)
        spatial_maps = torch.sigmoid(attention_logits).view(b, 7, h, w)
        k = max(1, int(h * w * TOP_K_FRACTION))
        burden = spatial_maps.flatten(2).topk(k, dim=2).values.mean(dim=2)
        embedding_signal = torch.sigmoid(
            (embeddings * self.embedding_evidence_weight.unsqueeze(0)).sum(dim=2) / math.sqrt(d)
        )
        presence = torch.sigmoid(presence_logits)
        evidence_strength = (presence + burden + embedding_signal) / 3.0
        spatial_entropy = -(
            attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
        ).sum(dim=2) / math.log(h * w)
        return {
            "presence_logits": presence_logits, "presence": presence,
            "spatial_maps": spatial_maps, "burden": burden,
            "embeddings": embeddings, "evidence_strength": evidence_strength,
            "spatial_entropy": spatial_entropy,
        }

class ReliabilityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_embedding = nn.Embedding(2, 4)
        self.network = nn.Sequential(nn.Linear(9, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, clinical, latent, evidence, modality_index, quality):
        clinical_entropy = normalized_entropy(clinical)
        latent_entropy = normalized_entropy(latent)
        disagreement = js_divergence(clinical, latent)
        lesion_entropy = binary_entropy(evidence).mean(dim=1)
        modality = self.modality_embedding(modality_index)
        inputs = torch.cat([
            clinical_entropy.unsqueeze(1), latent_entropy.unsqueeze(1),
            disagreement.unsqueeze(1), lesion_entropy.unsqueeze(1),
            modality, quality.view(-1, 1),
        ], dim=1)
        gate = torch.sigmoid(self.network(inputs)).squeeze(1)
        final = gate.unsqueeze(1) * clinical + (1.0 - gate).unsqueeze(1) * latent
        final = final / final.sum(dim=1, keepdim=True)
        return {"gate": gate, "probabilities": final,
                "expected_grade": expected_grade(final),
                "clinical_entropy": clinical_entropy,
                "latent_entropy": latent_entropy,
                "js_disagreement": disagreement,
                "lesion_entropy": lesion_entropy}

class FOVEADRAuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        official = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1, progress=False
        )
        official.classifier = nn.Identity()
        self.encoder = official
        self.fpn = MultiScaleFPN()
        self.coordinate_fusion = nn.Sequential(
            nn.Conv2d(FUSION_DIM + 7, FUSION_DIM, 3, padding=1, bias=False),
            nn.GroupNorm(8, FUSION_DIM), nn.SiLU(),
        )
        self.lesion_bank = LesionEvidenceBank()
        self.clinical_reasoner = ClinicalStateReasoner()
        self.latent_head = OrdinalHead(1280, 5)
        self.reliability_gate = ReliabilityGate()
        self.oct_head = OrdinalHead(1280, 3)

    def encode(self, image):
        x = image; selected = []
        for index, stage in enumerate(self.encoder.features):
            x = stage(x)
            if index in FPN_TAPS:
                selected.append(x)
        return selected

    def fuse(self, evidence, latent_probs, modality_index, quality):
        clinical = self.clinical_reasoner(evidence)
        fusion = self.reliability_gate(
            clinical["probabilities"], latent_probs, evidence, modality_index, quality
        )
        return {"clinical": clinical, "fusion": fusion}

    def counterfactual(self, evidence, latent_probs, modality_index, quality, original):
        no_lesion = torch.zeros_like(evidence)
        remove_pdr = torch.cat([evidence[:, :4], torch.zeros_like(evidence[:, 4:])], dim=1)
        add_pdr = torch.cat([evidence[:, :4], torch.ones_like(evidence[:, 4:])], dim=1)
        no_pred = self.fuse(no_lesion, latent_probs, modality_index, quality)
        remove_pred = self.fuse(remove_pdr, latent_probs, modality_index, quality)
        add_pred = self.fuse(add_pdr, latent_probs, modality_index, quality)
        original_grade = original["fusion"]["expected_grade"]
        any_support = 1.0 - torch.prod(1.0 - evidence, dim=1)
        pdr_support = 1.0 - torch.prod(1.0 - evidence[:, 4:], dim=1)
        remove_all_term = F.relu(
            no_pred["fusion"]["expected_grade"] + MARGIN * any_support - original_grade
        ).mean()
        remove_pdr_term = F.relu(
            remove_pred["fusion"]["expected_grade"] + MARGIN * pdr_support - original_grade
        ).mean()
        add_pdr_term = F.relu(
            original_grade + MARGIN * (1.0 - pdr_support)
            - add_pred["fusion"]["expected_grade"]
        ).mean()
        return {
            "remove_all": remove_all_term, "remove_pdr": remove_pdr_term,
            "add_pdr": add_pdr_term,
            "total": remove_all_term + remove_pdr_term + add_pdr_term,
            "no_lesion": no_pred, "remove_pdr_prediction": remove_pred,
            "add_pdr_prediction": add_pred,
        }

    def forward_fundus(self, image, fov_mask, modality_index, quality):
        features = self.encode(image)
        pyramid = self.fpn(features)
        coords = make_coordinate_channels(fov_mask, modality_index, pyramid.shape[-2:])
        fused_map = self.coordinate_fusion(torch.cat([pyramid, coords], dim=1))
        lesions = self.lesion_bank(fused_map)
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        latent = self.latent_head(pooled)
        original = self.fuse(
            lesions["evidence_strength"], latent["probabilities"], modality_index, quality
        )
        counterfactual = self.counterfactual(
            lesions["evidence_strength"], latent["probabilities"],
            modality_index, quality, original,
        )
        return {"features": features, "pyramid": pyramid, "coordinates": coords,
                "lesions": lesions, "latent": latent, "original": original,
                "counterfactual": counterfactual}

    def forward_oct(self, image):
        features = self.encode(image)
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        return {"features": features, "oct": self.oct_head(pooled)}

def prepare_image(row):
    path = Path(str(row["local_path"]))
    if sha256_file(path) != str(row["image_sha256"]):
        raise RuntimeError(f"Image hash mismatch: {path}")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        raw = TF.to_tensor(rgb).unsqueeze(0)
    raw = TF.resize(raw, [INPUT_SIZE, INPUT_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
    quality = raw.std(dim=(1, 2, 3)).clamp(0.0, 1.0)
    fov_mask = (raw.amax(dim=1, keepdim=True) > (8.0 / 255.0)).to(raw.dtype)
    normalized = TF.normalize(raw.squeeze(0), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    return normalized.to(DEVICE), fov_mask.to(DEVICE), quality.to(DEVICE)

def grade_emd(probs, target):
    one_hot = F.one_hot(target, num_classes=probs.shape[1]).to(probs.dtype)
    return (probs.cumsum(dim=1) - one_hot.cumsum(dim=1)).abs().mean(dim=1)

def fundus_losses(output, row, weights):
    target = torch.tensor([int(row["grade"])], device=DEVICE)
    grade_mask = torch.tensor([float(row["grade_loss_mask"])], device=DEVICE)
    lesion_mask = torch.tensor([float(row["lesion_loss_mask"])], device=DEVICE)
    sensitivity_mask = torch.tensor([float(row["lesion_sensitivity_mask"])], device=DEVICE)
    lesion_target = torch.tensor(
        [[float(row[column]) for column in LESION_COLUMNS]], device=DEVICE
    )
    final_probs = output["original"]["fusion"]["probabilities"]
    clinical_probs = output["original"]["clinical"]["probabilities"]
    nll = masked_mean(-final_probs.gather(1, target[:, None]).clamp_min(1e-7).log().squeeze(1), grade_mask)
    emd = masked_mean(grade_emd(final_probs, target), grade_mask)
    lesion_per = F.binary_cross_entropy_with_logits(
        output["lesions"]["presence_logits"], lesion_target, reduction="none"
    ).mean(dim=1)
    lesion_bce = masked_mean(lesion_per, lesion_mask)
    clinical_alignment = masked_mean(
        -clinical_probs.gather(1, target[:, None]).clamp_min(1e-7).log().squeeze(1), sensitivity_mask
    )
    route_agreement = masked_mean(
        output["original"]["fusion"]["js_disagreement"], sensitivity_mask
    )
    cf = output["counterfactual"]["total"]
    total = (
        weights["fundus_grade_cross_entropy_weight"] * nll
        + weights["fundus_ordinal_EMD_weight"] * emd
        + weights["lesion_BCE_weight"] * lesion_bce
        + weights["clinical_state_alignment_weight"] * clinical_alignment
        + weights["lesion_counterfactual_intervention_weight"] * cf
        + weights["masked_route_agreement_weight"] * route_agreement
    )
    return {"grade_NLL": nll, "ordinal_EMD": emd, "lesion_BCE": lesion_bce,
            "clinical_alignment": clinical_alignment, "counterfactual": cf,
            "route_agreement": route_agreement, "total": total}

def module_grad_stats(module):
    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    return {"gradient_tensors": len(grads),
            "all_finite": len(grads) > 0 and all(bool(torch.isfinite(g).all()) for g in grads),
            "any_nonzero": len(grads) > 0 and any(bool(torch.any(g != 0)) for g in grads)}

def tensor_finite(tensor):
    return bool(torch.isfinite(tensor).all())

def probabilities_valid(tensor):
    return tensor_finite(tensor) and bool(torch.all(tensor >= 0)) and bool(
        torch.allclose(tensor.sum(dim=1), torch.ones(tensor.shape[0]), atol=1e-5)
    )

print("\n[1] FROZEN INPUT VERIFICATION")
print("-" * 80)
candidate_sha = sha256_file(CANDIDATE_PATH); manifest_sha = sha256_file(MANIFEST_PATH)
if candidate_sha != EXPECTED_CANDIDATE_SHA or manifest_sha != EXPECTED_MANIFEST_SHA:
    raise RuntimeError("Candidate or manifest hash mismatch.")
with open(CANDIDATE_PATH) as f: candidate = json.load(f)
with open(WEIGHT_EVIDENCE_PATH) as f: weight_evidence = json.load(f)
if weight_evidence.get("checkpoint_sha256") != EXPECTED_WEIGHT_SHA:
    raise RuntimeError("Official weight evidence mismatch.")
samples = pd.read_csv(MANIFEST_PATH)
if len(samples) != 3 or not samples["development_split"].eq("TRAIN").all():
    raise RuntimeError("TRAIN-only three-sample contract failed.")
if set(samples["modality"]) != {"CFP", "UWF", "OCT"}:
    raise RuntimeError("Modality contract failed.")
for column in LESION_COLUMNS:
    if column not in samples.columns: raise KeyError(column)
print("Candidate SHA verified :", True)
print("Manifest SHA verified  :", True)
print("Official weight SHA    :", EXPECTED_WEIGHT_SHA)
print("Samples                :", len(samples), sorted(samples["modality"].tolist()))

print("\n[2] MODEL INSTANTIATION")
print("-" * 80)
model = FOVEADRAuditModel().to(DEVICE)
model.eval()
parameter_snapshot = [p.detach().clone() for p in model.parameters()]
total_parameters = sum(p.numel() for p in model.parameters())
print("Device             :", DEVICE)
print("Total parameters   :", total_parameters)
print("Input policy       : 512×512 RGB + ImageNet normalization")
print("Optimizer created  : NO")

loss_weights = candidate["candidate_loss"]
fundus_records, oct_records = [], []
all_feature_shapes = []
connectivity_checks = []

print("\n[3] REAL TRAIN-IMAGE FORWARD/BACKWARD AUDIT")
print("-" * 80)
for _, row in samples.iterrows():
    modality = str(row["modality"])
    image, fov_mask, quality = prepare_image(row)
    model.zero_grad(set_to_none=True)
    if modality in {"CFP", "UWF"}:
        modality_index = torch.tensor([0 if modality == "CFP" else 1], dtype=torch.long)
        output = model.forward_fundus(image, fov_mask, modality_index, quality)
        losses = fundus_losses(output, row, loss_weights)
        feature_shapes = [list(t.shape) for t in output["features"]]
        all_feature_shapes.append(feature_shapes)
        cf_loss = losses["counterfactual"]
        if float(cf_loss.detach()) > 1e-12:
            connectivity_objective = cf_loss
            connectivity_source = "candidate_counterfactual_loss"
        else:
            cf = output["counterfactual"]
            connectivity_objective = (
                output["original"]["fusion"]["expected_grade"].mean()
                + 0.37 * cf["no_lesion"]["fusion"]["expected_grade"].mean()
                + 0.19 * cf["remove_pdr_prediction"]["fusion"]["expected_grade"].mean()
                + 0.11 * cf["add_pdr_prediction"]["fusion"]["expected_grade"].mean()
            )
            connectivity_source = "counterfactual_expected_grade_connectivity_probe"
        gate_parameters = list(model.reliability_gate.parameters())
        connectivity_grads = torch.autograd.grad(
            connectivity_objective,
            [output["lesions"]["presence_logits"], output["features"][-1]] + gate_parameters,
            retain_graph=True, allow_unused=True,
        )
        lesion_grad, backbone_grad = connectivity_grads[:2]
        gate_grads = connectivity_grads[2:]
        connectivity = {
            "modality": modality, "source": connectivity_source,
            "lesion_path_finite_nonzero": lesion_grad is not None and tensor_finite(lesion_grad) and bool(torch.any(lesion_grad != 0)),
            "backbone_path_finite_nonzero": backbone_grad is not None and tensor_finite(backbone_grad) and bool(torch.any(backbone_grad != 0)),
            "gate_path_finite_nonzero": any(g is not None and tensor_finite(g) and bool(torch.any(g != 0)) for g in gate_grads),
        }
        connectivity_checks.append(connectivity)
        losses["total"].backward()
        grad_stats = {
            "encoder": module_grad_stats(model.encoder.features),
            "fpn": module_grad_stats(model.fpn),
            "coordinate_fusion": module_grad_stats(model.coordinate_fusion),
            "lesion_bank": module_grad_stats(model.lesion_bank),
            "clinical_reasoner": module_grad_stats(model.clinical_reasoner),
            "latent_head": module_grad_stats(model.latent_head),
            "reliability_gate": module_grad_stats(model.reliability_gate),
        }
        final_probs = output["original"]["fusion"]["probabilities"]
        record = {
            "modality": modality, "image_name": str(row["image_name"]),
            "grade": int(row["grade"]), "feature_shapes": feature_shapes,
            "pyramid_shape": list(output["pyramid"].shape),
            "coordinate_shape": list(output["coordinates"].shape),
            "fov_coverage": float(fov_mask.mean()), "quality_proxy": float(quality.item()),
            "final_probabilities": [float(x) for x in final_probs.detach().cpu()[0]],
            "expected_grade": float(output["original"]["fusion"]["expected_grade"].detach()),
            "reliability_gate": float(output["original"]["fusion"]["gate"].detach()),
            "losses": {name: float(value.detach()) for name, value in losses.items()},
            "counterfactual_terms": {
                name: float(output["counterfactual"][name].detach())
                for name in ["remove_all", "remove_pdr", "add_pdr", "total"]
            },
            "probabilities_valid": probabilities_valid(final_probs),
            "gradient_stats": grad_stats, "connectivity": connectivity,
        }
        fundus_records.append(record)
        print(f"{modality} | {row['image_name']} | grade={int(row['grade'])} | total={record['losses']['total']:.6f} | gate={record['reliability_gate']:.6f} | PASS")
    else:
        output = model.forward_oct(image)
        feature_shapes = [list(t.shape) for t in output["features"]]
        all_feature_shapes.append(feature_shapes)
        target = torch.tensor([int(row["grade"])], device=DEVICE)
        probs = output["oct"]["probabilities"]
        nll = -probs.gather(1, target[:, None]).clamp_min(1e-7).log().mean()
        total = float(loss_weights["OCT_DME_task_weight"]) * nll
        total.backward()
        grad_stats = {"encoder": module_grad_stats(model.encoder.features),
                      "oct_head": module_grad_stats(model.oct_head)}
        record = {
            "modality": modality, "image_name": str(row["image_name"]),
            "grade": int(row["grade"]), "feature_shapes": feature_shapes,
            "probabilities": [float(x) for x in probs.detach().cpu()[0]],
            "expected_grade": float(output["oct"]["expected_grade"].detach()),
            "DME_NLL": float(nll.detach()), "weighted_total": float(total.detach()),
            "probabilities_valid": probabilities_valid(probs), "gradient_stats": grad_stats,
        }
        oct_records.append(record)
        print(f"OCT | {row['image_name']} | grade={int(row['grade'])} | total={record['weighted_total']:.6f} | PASS")
    del image, fov_mask, quality, output
    gc.collect()

print("\n[4] CONTROLLED CLINICAL-ORDERING TEST")
print("-" * 80)
states = []
for values in [[], [0], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]:
    state = torch.zeros(1, 7)
    if values: state[:, values] = 1.0
    states.append(state)
with torch.no_grad():
    controlled = [float(model.clinical_reasoner(state)["expected_grade"].item()) for state in states]
controlled_names = ["NO_LESION", "MA_ONLY", "NONPDR", "VB_IRMA", "PDR"]
controlled_order = all(b > a for a, b in zip(controlled, controlled[1:]))
for name, value in zip(controlled_names, controlled): print(f"{name:12s}: {value:.6f}")
print("Strict ordering:", controlled_order)

parameters_unchanged = all(
    torch.equal(before, after.detach())
    for before, after in zip(parameter_snapshot, model.parameters())
)
expected_shapes = [[1, 64, 64, 64], [1, 160, 32, 32], [1, 1280, 16, 16]]
fundus_modules_ok = all(
    all(stats["gradient_tensors"] > 0 and stats["all_finite"] and stats["any_nonzero"]
        for stats in record["gradient_stats"].values())
    for record in fundus_records
)
oct_modules_ok = all(
    all(stats["gradient_tensors"] > 0 and stats["all_finite"] and stats["any_nonzero"]
        for stats in record["gradient_stats"].values())
    for record in oct_records
)
connectivity_ok = all(
    item["lesion_path_finite_nonzero"] and item["backbone_path_finite_nonzero"]
    and item["gate_path_finite_nonzero"] for item in connectivity_checks
)
all_probabilities_ok = all(r["probabilities_valid"] for r in fundus_records + oct_records)
all_losses_finite = all(
    math.isfinite(value) for record in fundus_records for value in record["losses"].values()
) and all(math.isfinite(r["weighted_total"]) for r in oct_records)

gates = {
    "candidate_hash_verified": candidate_sha == EXPECTED_CANDIDATE_SHA,
    "manifest_hash_verified": manifest_sha == EXPECTED_MANIFEST_SHA,
    "official_weight_sha_verified": weight_evidence.get("checkpoint_sha256") == EXPECTED_WEIGHT_SHA,
    "exactly_three_TRAIN_samples": len(samples) == 3 and samples["development_split"].eq("TRAIN").all(),
    "real_image_hashes_verified": True,
    "three_modality_forward_passes_completed": len(fundus_records) == 2 and len(oct_records) == 1,
    "official_multiscale_feature_shapes_correct": all(shape == expected_shapes for shape in all_feature_shapes),
    "all_probability_distributions_valid": all_probabilities_ok,
    "all_candidate_losses_finite": all_losses_finite,
    "all_fundus_module_gradients_finite_nonzero": fundus_modules_ok,
    "OCT_encoder_and_head_gradients_finite_nonzero": oct_modules_ok,
    "counterfactual_connectivity_reaches_lesion_backbone_gate": connectivity_ok,
    "controlled_clinical_order_strict": controlled_order,
    "model_parameters_unchanged_without_optimizer": parameters_unchanged,
}

print("\n" + "=" * 80)
print("END-TO-END INTEGRATION GATES")
print("=" * 80)
for name, passed in gates.items(): print(f"{name:62s}: {bool(passed)}")
if not all(gates.values()):
    raise RuntimeError("STAGE 0G-4C-2 integration gate failed.")

implementation_contract = {
    "stage": "0G-4C-2", "status": "AUDIT_IMPLEMENTATION_LOCK",
    "candidate_version": "1.1", "candidate_sha256": candidate_sha,
    "input": {"size": [512, 512], "channels": "RGB", "interpolation": "bilinear_antialias",
              "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]},
    "encoder": {"backbone": "EfficientNetV2-S", "weights": "IMAGENET1K_V1",
                "weight_sha256": EXPECTED_WEIGHT_SHA, "feature_taps": FPN_TAPS,
                "feature_channels": FPN_CHANNELS},
    "FPN": {"fusion_dim": FUSION_DIM, "target_resolution": [64, 64]},
    "FOV_coordinates": ["normalized_x", "normalized_y", "retinal_radius", "sin_retinal_angle",
                         "cos_retinal_angle", "visible_retinal_field_mask", "modality_indicator"],
    "FOV_mask_audit_threshold": "max_RGB > 8/255",
    "lesion_bank": {"queries": 7, "embedding_dim": FUSION_DIM,
                    "top_k_fraction": TOP_K_FRACTION, "target_columns": LESION_COLUMNS},
    "counterfactual_margin": MARGIN,
    "audit_policy": "single_real_TRAIN_image_per_backward; eval_mode; no_optimizer; no_parameter_update",
}
with open(CONTRACT_PATH, "w") as f: json.dump(implementation_contract, f, indent=2)
contract_sha = sha256_file(CONTRACT_PATH)

evidence = {
    "stage": "0G-4C-2", "status": "PASS",
    "scope": "REAL_TRAIN_ONLY_END_TO_END_FORWARD_BACKWARD_AUDIT",
    "candidate_sha256": candidate_sha, "manifest_sha256": manifest_sha,
    "official_weight_sha256": EXPECTED_WEIGHT_SHA,
    "implementation_contract": str(CONTRACT_PATH), "implementation_contract_sha256": contract_sha,
    "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "device": str(DEVICE),
                "threads": torch.get_num_threads()},
    "total_model_parameters": int(total_parameters),
    "fundus_records": fundus_records, "OCT_records": oct_records,
    "controlled_clinical_order": dict(zip(controlled_names, controlled)),
    "gates": {name: bool(value) for name, value in gates.items()},
    "medical_images_accessed": True, "images_displayed": False,
    "backward_pass_performed": True, "optimizer_created": False,
    "training_started": False, "model_parameters_updated": False,
    "validation_used": False, "calibration_used": False,
    "locked_test_used": False, "external_UWF_accessed": False,
    "next_required_stage": "SOURCE_FREEZE_AND_TRAINING_PROTOCOL_LOCK",
}
with open(EVIDENCE_PATH, "w") as f: json.dump(evidence, f, indent=2)

print("\nImplementation contract:")
print(CONTRACT_PATH)
print("SHA-256:", contract_sha)
print("\nEvidence:")
print(EVIDENCE_PATH)
print("\nImages displayed        : NO")
print("Optimizer created       : NO")
print("Parameters updated      : NO")
print("Training started        : NO")
print("Validation/test accessed: NO")
print("\n" + "=" * 80)
print("STAGE 0G-4C-2 STATUS: PASS")
print("=" * 80)
