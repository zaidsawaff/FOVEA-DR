#!/usr/bin/env python3
"""Run one protocol-locked comparator/ablation 64-step TRAIN preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
CANDIDATE = ROOT / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
TRAINING_PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_training_protocol_v1.json"
SPLIT = ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
SAMPLER = ROOT / "Splits/mmrdr_fovea_dr_TRAIN_sampler_weights_v1.csv"
LESION_WEIGHTS = ROOT / "Metadata/mmrdr_fovea_dr_lesion_pos_weights_v1.csv"
TRAIN_MANIFEST = ROOT / "Audit/mmrdr_fovea_dr_TRAIN_local_cache_manifest_v1.csv"
CACHE_EVIDENCE = ROOT / "Evidence/stage0g4d5f1b_comparator_development_cache_restore.json"
WEIGHT_EVIDENCE = ROOT / "Evidence/stage0g4d5f1c_official_comparator_weight_acquisition.json"
IMPLEMENTATION_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2a_comparator_implementation_audit.py"
IMPLEMENTATION_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparator_implementation_contract_v1.json"
IMPLEMENTATION_EVIDENCE = ROOT / "Evidence/stage0g4d5f2a_comparator_implementation_audit.json"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f2b_resumable_variant_64step_preflight.py"
PREFLIGHT_ROOT = ROOT / "Runs/FOVEA_DR/posthoc_comparisons/preflight64"

EXPECTED_SHA = {
    PROTOCOL: "f946964923e08722dbb486aa3782b08880446040e07fb6b268ea9c28c08e6ae1",
    CANDIDATE: "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181",
    TRAINING_PROTOCOL: "5b16e0c8565b86d3cac603f416ee0b7f03055ab9a0eb9ac9e3fb007632329b40",
    SPLIT: "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140",
    TRAIN_MANIFEST: "32f9a3b6b17916b78f591007a85ce2e1910c9c4d2b1294af1c391b19f476bd15",
    IMPLEMENTATION_SOURCE: "987f170cbfd4ef048161d29a12a9cfd6e715b45338c539b0c9ba7dfe5a04379f",
}

VARIANTS = [
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
    "B2_RESNET50_MULTITASK",
    "B3_CONVNEXT_TINY_MULTITASK",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
]

BACKBONE_FOR_VARIANT = {
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK": "EfficientNetV2-S",
    "B2_RESNET50_MULTITASK": "ResNet-50",
    "B3_CONVNEXT_TINY_MULTITASK": "ConvNeXt-Tiny",
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS": "EfficientNetV2-S",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT": "EfficientNetV2-S",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE": "EfficientNetV2-S",
}

RUN_SEED = 2026
PILOT_STEPS = 64
GRADIENT_ACCUMULATION = 2
MICRO_BATCH = 16
EFFECTIVE_BATCH = 32
OPTIMIZER_STEPS_PER_EPOCH = 415
TOTAL_OPTIMIZER_STEPS = 30 * OPTIMIZER_STEPS_PER_EPOCH
WARMUP_STEPS = 623
MIN_LR_RATIO = 0.01
GRAD_CLIP_NORM = 1.0

EXPECTED_TRAIN_COUNTS = {"CFP": 6128, "UWF": 5460, "OCT": 1663}
EXPECTED_STEP_COMPOSITION = {"CFP": 15, "UWF": 13, "OCT": 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different 5F-2B preflight runner is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def load_pass_evidence(path: Path, label: str) -> tuple[dict[str, Any], bool]:
    if not path.is_file():
        print(f"{label:32s}: exists=False")
        return {}, False
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    gates = payload.get("gates", {})
    passed = (
        payload.get("status") == "PASS"
        and isinstance(gates, dict)
        and bool(gates)
        and all(bool(value) for value in gates.values())
    )
    print(f"{label:32s}: exists=True, PASS={passed}")
    return payload, bool(passed)


def import_implementation():
    specification = importlib.util.spec_from_file_location(
        "fovea_dr_posthoc_comparison_v1", IMPLEMENTATION_SOURCE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not import frozen comparison implementation.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def worker_seed(worker_id: int) -> None:
    value = RUN_SEED + worker_id
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)


def lr_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return float(step + 1) / float(WARMUP_STEPS)
    progress = float(step - WARMUP_STEPS) / float(
        max(1, TOTAL_OPTIMIZER_STEPS - WARMUP_STEPS)
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


def parameter_group_snapshot(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    result = {"backbone": {}, "new_modules": {}}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = "backbone" if name.startswith("encoder.") else "new_modules"
            result[group][name] = parameter.detach().cpu().clone()
    return result


def parameter_group_deltas(
    model: torch.nn.Module,
    initial: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    squared = {"backbone": 0.0, "new_modules": 0.0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = "backbone" if name.startswith("encoder.") else "new_modules"
        difference = parameter.detach().cpu().float() - initial[group][name].float()
        squared[group] += float(difference.square().sum())
    return {name: math.sqrt(value) for name, value in squared.items()}


def existing_pass(
    evidence_path: Path,
    log_path: Path,
    variant_id: str,
    runner_sha: str,
) -> bool:
    if not evidence_path.is_file():
        return False
    with evidence_path.open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)
    gates = evidence.get("gates", {})
    valid = (
        evidence.get("status") == "PASS"
        and evidence.get("variant_id") == variant_id
        and evidence.get("runner_source_sha256") == runner_sha
        and isinstance(gates, dict)
        and bool(gates)
        and all(bool(value) for value in gates.values())
        and log_path.is_file()
        and evidence.get("pilot_log_sha256") == sha256_file(log_path)
        and evidence.get("locked_test_accessed") is False
    )
    if valid:
        print("Existing completed preflight verified; no optimization was repeated.")
        print("Evidence:", evidence_path)
        print("Log     :", log_path)
        print("STAGE 0G-4D-5F-2B STATUS: PASS — EXISTING VERIFIED")
    return bool(valid)


def verify_selected_official_weight(
    weight_evidence: dict[str, Any], variant_id: str
) -> tuple[dict[str, Any], bool]:
    target = BACKBONE_FOR_VARIANT[variant_id]
    matches = [
        record
        for record in weight_evidence.get("official_models", [])
        if record.get("name") == target
    ]
    if len(matches) != 1:
        return {}, False
    record = matches[0]
    path = Path(str(record.get("checkpoint_path", "")))
    valid = (
        path.is_file()
        and path.stat().st_size == int(record.get("checkpoint_size_bytes", -1))
        and sha256_file(path) == record.get("checkpoint_sha256")
        and all(bool(value) for value in record.get("checks", {}).values())
    )
    return record, bool(valid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--pilot-steps", type=int, default=PILOT_STEPS)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.pilot_steps != PILOT_STEPS:
        raise RuntimeError("The frozen preflight requires exactly 64 optimizer steps.")
    if args.workers < 0 or args.workers > 4:
        raise RuntimeError("workers must be between 0 and 4.")
    variant_id = args.variant
    variant_dir = PREFLIGHT_ROOT / variant_id
    log_path = variant_dir / "optimizer_step_log.csv"
    evidence_path = ROOT / "Evidence" / f"stage0g4d5f2b_{variant_id.lower()}_64step_preflight.json"

    print("=" * 96)
    print("STAGE 0G-4D-5F-2B — RESUMABLE SINGLE-VARIANT 64-STEP TRAIN-ONLY PREFLIGHT")
    print("=" * 96)
    print("Variant:", variant_id)

    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    torch.cuda.manual_seed_all(RUN_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("\n[1] FROZEN ARTIFACT VERIFICATION")
    print("-" * 96)
    gates: dict[str, bool] = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(verified)
        print(f"{path.name:78s}: exists={exists}, hash_verified={verified}")
    for path in [SAMPLER, LESION_WEIGHTS, IMPLEMENTATION_CONTRACT]:
        exists = path.is_file() and path.stat().st_size > 0
        gates[f"exists_nonempty::{path.name}"] = bool(exists)
        print(f"{path.name:78s}: exists_nonempty={exists}")

    cache_payload, cache_pass = load_pass_evidence(CACHE_EVIDENCE, "development cache restoration")
    weight_payload, weight_pass = load_pass_evidence(WEIGHT_EVIDENCE, "official weight acquisition")
    implementation_payload, implementation_pass = load_pass_evidence(
        IMPLEMENTATION_EVIDENCE, "implementation synthetic audit"
    )
    contract_sha = sha256_file(IMPLEMENTATION_CONTRACT) if IMPLEMENTATION_CONTRACT.is_file() else None
    gates.update({
        "development_cache_evidence_passed": cache_pass,
        "official_weight_evidence_passed": weight_pass,
        "implementation_audit_evidence_passed": implementation_pass,
        "implementation_evidence_links_source": (
            implementation_payload.get("source_sha256") == EXPECTED_SHA[IMPLEMENTATION_SOURCE]
        ),
        "implementation_evidence_links_contract": (
            implementation_payload.get("implementation_contract_sha256") == contract_sha
        ),
        "cache_evidence_reports_locked_test_not_accessed": (
            cache_payload.get("locked_test_accessed") is False
        ),
        "weight_evidence_reports_locked_test_not_accessed": (
            weight_payload.get("locked_test_accessed") is False
        ),
        "implementation_reports_locked_test_not_accessed": (
            implementation_payload.get("locked_test_accessed") is False
        ),
    })

    with PROTOCOL.open("r", encoding="utf-8") as handle:
        comparison_protocol = json.load(handle)
    protocol_ids = [item["id"] for item in comparison_protocol["comparators"]]
    protocol_ids += [item["id"] for item in comparison_protocol["ablations"]]
    gates["selected_variant_is_protocol_locked"] = variant_id in protocol_ids
    gates["locked_test_prohibited"] = comparison_protocol["data_access_policy"][
        "LOCKED_INTERNAL_TEST"
    ].startswith("PROHIBITED")

    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"Frozen artifact verification failed: {failed}")

    source_action, runner_sha = freeze_source()
    print("Runner source action:", source_action)
    print("Runner source SHA   :", runner_sha)
    if existing_pass(evidence_path, log_path, variant_id, runner_sha):
        return
    if evidence_path.exists():
        raise RuntimeError(
            "An existing preflight evidence file failed verification; "
            "refusing to overwrite it."
        )

    print("\n[2] GPU AND OFFICIAL INITIALIZATION")
    print("-" * 96)
    cuda_available = torch.cuda.is_available()
    visible_devices = torch.cuda.device_count() if cuda_available else 0
    if not cuda_available or visible_devices != 1:
        raise RuntimeError("Exactly one CUDA GPU is required.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    selected_weight, official_weight_verified = verify_selected_official_weight(
        weight_payload, variant_id
    )
    print("GPU                       :", gpu_name)
    print("VRAM GB                   :", f"{total_vram_gb:.3f}")
    print("Selected backbone         :", BACKBONE_FOR_VARIANT[variant_id])
    print("Official checkpoint SHA   :", selected_weight.get("checkpoint_sha256"))
    print("Official weight verified  :", official_weight_verified)
    if total_vram_gb < 10.0 or not official_weight_verified:
        raise RuntimeError("GPU or official-weight verification failed.")

    print("\n[3] TRAIN-ONLY TABLE AND LOCKED SAMPLER")
    print("-" * 96)
    implementation = import_implementation()
    core = implementation.get_core()
    table, sampler_weight_column = core.build_training_table()
    modality_counts = {
        str(key): int(value)
        for key, value in table["modality"].value_counts().to_dict().items()
    }
    exact_train_table = len(table) == 13251 and modality_counts == EXPECTED_TRAIN_COUNTS
    only_train = bool(table["development_split"].eq("TRAIN").all())
    all_cache_files_present = all(
        Path(str(row["local_path"])).is_file()
        and Path(str(row["local_path"])).stat().st_size == int(row["file_size_bytes"])
        for row in table.to_dict("records")
    )
    lesion_table = pd.read_csv(LESION_WEIGHTS)
    lesion_pos_weight = torch.tensor(
        lesion_table["sqrt_pos_weight"].to_numpy(dtype=np.float32)
    )
    lesion_weights_valid = (
        len(lesion_pos_weight) == 7
        and bool(torch.isfinite(lesion_pos_weight).all())
        and bool(torch.all(lesion_pos_weight > 0))
    )
    sampler = core.LockedWeightedBatchSampler(table, RUN_SEED, PILOT_STEPS)
    batch_shapes_ok = (
        len(sampler) == PILOT_STEPS * GRADIENT_ACCUMULATION
        and all(len(batch) == MICRO_BATCH for batch in sampler.batches)
    )
    sampled_counts = {"CFP": 0, "UWF": 0, "OCT": 0}
    for indices in sampler.batches:
        counts = table.loc[indices, "modality"].value_counts()
        for modality, count in counts.items():
            sampled_counts[str(modality)] += int(count)
    expected_sampled_counts = {
        modality: count * PILOT_STEPS
        for modality, count in EXPECTED_STEP_COMPOSITION.items()
    }
    composition_ok = sampled_counts == expected_sampled_counts
    print("TRAIN rows/counts      :", len(table), modality_counts)
    print("Sampler weight column :", sampler_weight_column)
    print("Micro-batches          :", len(sampler))
    print("Sampled counts         :", sampled_counts)
    print("All cache files present:", all_cache_files_present)
    if not all([
        exact_train_table, only_train, all_cache_files_present,
        lesion_weights_valid, batch_shapes_ok, composition_ok,
    ]):
        raise RuntimeError("TRAIN-only data/sampler verification failed.")

    dataset = core.FOVEADRTrainDataset(table)
    loader_generator = torch.Generator().manual_seed(RUN_SEED)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed if args.workers > 0 else None,
        generator=loader_generator,
    )

    print("\n[4] MODEL, OPTIMIZER, AMP, AND SCHEDULER")
    print("-" * 96)
    with CANDIDATE.open("r", encoding="utf-8") as handle:
        candidate_weights = json.load(handle)["candidate_loss"]
    active_loss_weights = implementation.variant_loss_weights(
        variant_id, candidate_weights
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = implementation.build_variant(variant_id).to(device)
    model.train()
    initial = parameter_group_snapshot(model)
    optimizer, optimizer_group_counts = implementation.build_comparison_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    lesion_pos_weight = lesion_pos_weight.to(device)
    print("Parameters            :", sum(parameter.numel() for parameter in model.parameters()))
    print("Optimizer groups      :", optimizer_group_counts)
    print("Active loss weights   :", active_loss_weights)
    print("Micro-batch           :", MICRO_BATCH)
    print("Gradient accumulation :", GRADIENT_ACCUMULATION)
    print("Effective batch       :", EFFECTIVE_BATCH)
    print("AMP                   : float16 + dynamic GradScaler")

    print("\n[5] 64-STEP TRAIN-ONLY OPTIMIZATION PREFLIGHT")
    print("-" * 96)
    logs: list[dict[str, Any]] = []
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    all_losses_finite = True
    all_gradients_finite = True
    all_gradients_nonzero = True
    no_amp_overflow = True
    start = time.perf_counter()

    for optimizer_step in range(1, PILOT_STEPS + 1):
        step_start = time.perf_counter()
        step_loss = 0.0
        components = {
            "grade_NLL": 0.0,
            "ordinal_EMD": 0.0,
            "lesion_BCE": 0.0,
            "clinical_alignment": 0.0,
            "counterfactual": 0.0,
            "route_agreement": 0.0,
            "OCT_NLL": 0.0,
        }
        modalities = {"CFP": 0, "UWF": 0, "OCT": 0}
        for _ in range(GRADIENT_ACCUMULATION):
            batch = core.move_batch(next(iterator), device)
            for modality in batch["modality"]:
                modalities[str(modality)] += 1
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.forward_mixed(
                    batch["image"], batch["fundus_mask"], batch["fov_mask"],
                    batch["modality_index"], batch["quality"],
                )
                total_loss, parts = implementation.compute_variant_losses(
                    variant_id, output, batch, lesion_pos_weight, candidate_weights
                )
                scaled_loss = total_loss / GRADIENT_ACCUMULATION
            finite_loss = bool(torch.isfinite(total_loss))
            all_losses_finite = all_losses_finite and finite_loss
            if not finite_loss:
                raise FloatingPointError(f"Non-finite loss at step {optimizer_step}")
            scaler.scale(scaled_loss).backward()
            step_loss += float(scaled_loss.detach().cpu())
            for name, value in parts.items():
                components[name] += float(value.detach().cpu()) / GRADIENT_ACCUMULATION

        scaler.unscale_(optimizer)
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradients_finite = bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        gradients_nonzero = bool(gradients) and any(
            bool(torch.count_nonzero(gradient)) for gradient in gradients
        )
        all_gradients_finite = all_gradients_finite and gradients_finite
        all_gradients_nonzero = all_gradients_nonzero and gradients_nonzero
        if not gradients_finite or not gradients_nonzero:
            raise FloatingPointError(f"Invalid gradients at step {optimizer_step}")

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if scale_after < scale_before:
            no_amp_overflow = False
            raise FloatingPointError(f"AMP overflow at step {optimizer_step}")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        learning_rates = {
            group["group_name"]: float(group["lr"])
            for group in optimizer.param_groups
        }
        record = {
            "optimizer_step": optimizer_step,
            "microbatches_completed": optimizer_step * GRADIENT_ACCUMULATION,
            "samples_completed": optimizer_step * EFFECTIVE_BATCH,
            "total_loss": step_loss,
            **components,
            "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
            "grad_scaler_scale": scale_after,
            "backbone_lr": learning_rates["backbone_decay"],
            "new_module_lr": learning_rates["new_decay"],
            "step_seconds": time.perf_counter() - step_start,
            "peak_reserved_vram_gb": peak_reserved_gb,
            "CFP_samples": modalities["CFP"],
            "UWF_samples": modalities["UWF"],
            "OCT_samples": modalities["OCT"],
        }
        logs.append(record)
        if optimizer_step == 1 or optimizer_step % 8 == 0 or optimizer_step == PILOT_STEPS:
            print(
                f"Step {optimizer_step:02d}/{PILOT_STEPS} | loss={step_loss:.6f} | "
                f"grad={float(grad_norm):.4f} | scale={scale_after:.0f} | "
                f"VRAM={peak_reserved_gb:.3f} GB | time={record['step_seconds']:.3f}s"
            )

    elapsed = time.perf_counter() - start
    del iterator
    log = pd.DataFrame(logs)
    deltas = parameter_group_deltas(model, initial)
    loss_first = float(log.head(8)["total_loss"].mean())
    loss_last = float(log.tail(8)["total_loss"].mean())
    peak_reserved_gb = float(log["peak_reserved_vram_gb"].max())
    peak_fraction = peak_reserved_gb / total_vram_gb
    all_compositions_exact = bool(
        (
            (log["CFP_samples"] == 15)
            & (log["UWF_samples"] == 13)
            & (log["OCT_samples"] == 4)
        ).all()
    )
    atomic_csv(log, log_path)
    log_sha = sha256_file(log_path)

    print("\n[6] PREFLIGHT SUMMARY")
    print("-" * 96)
    print("Optimizer steps completed:", len(log))
    print("TRAIN samples processed  :", len(log) * EFFECTIVE_BATCH)
    print("Mean first-8 loss        :", f"{loss_first:.6f}")
    print("Mean last-8 loss         :", f"{loss_last:.6f}")
    print("Elapsed seconds          :", f"{elapsed:.3f}")
    print("Peak reserved VRAM GB    :", f"{peak_reserved_gb:.3f}")
    print("Peak VRAM fraction       :", f"{peak_fraction:.3f}")
    print("Parameter-group deltas   :", deltas)

    gates.update({
        "CUDA_available": bool(cuda_available),
        "exactly_one_GPU_visible": visible_devices == 1,
        "GPU_memory_at_least_10GB": total_vram_gb >= 10.0,
        "selected_official_weight_verified": official_weight_verified,
        "exact_13251_TRAIN_rows_loaded": exact_train_table,
        "all_loaded_rows_are_TRAIN": only_train,
        "all_TRAIN_cache_files_present_and_size_matched": bool(all_cache_files_present),
        "lesion_positive_weights_valid": lesion_weights_valid,
        "exact_128_microbatches_generated": batch_shapes_ok,
        "effective_modality_composition_exact": composition_ok and all_compositions_exact,
        "all_four_optimizer_groups_nonempty": all(value > 0 for value in optimizer_group_counts.values()),
        "exact_64_optimizer_steps_completed": len(log) == PILOT_STEPS,
        "all_losses_finite": all_losses_finite and bool(np.isfinite(log["total_loss"]).all()),
        "all_gradients_finite": all_gradients_finite,
        "all_gradients_nonzero": all_gradients_nonzero,
        "no_AMP_overflow_or_skipped_step": no_amp_overflow,
        "backbone_parameters_updated_in_preflight": deltas["backbone"] > 0.0,
        "new_module_parameters_updated_in_preflight": deltas["new_modules"] > 0.0,
        "peak_reserved_VRAM_below_75_percent": peak_fraction < 0.75,
        "pilot_log_created_and_hashed": log_path.is_file() and bool(log_sha),
        "preflight_checkpoint_not_saved": True,
        "final_comparator_training_not_started": True,
        "VALIDATION_not_accessed": True,
        "CALIBRATION_not_accessed": True,
        "locked_test_not_accessed": True,
    })
    print("\n" + "=" * 96)
    print("SINGLE-VARIANT 64-STEP PREFLIGHT GATES")
    print("=" * 96)
    for name, passed in gates.items():
        print(f"{name:82s}: {passed}")
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"64-step preflight failed: {failed}")

    evidence = {
        "stage": "0G-4D-5F-2B",
        "status": "PASS",
        "purpose": "SINGLE_VARIANT_64_STEP_TRAIN_ONLY_PREFLIGHT",
        "variant_id": variant_id,
        "run_seed": RUN_SEED,
        "runner_source": str(FROZEN_SOURCE),
        "runner_source_sha256": runner_sha,
        "implementation_source": str(IMPLEMENTATION_SOURCE),
        "implementation_source_sha256": EXPECTED_SHA[IMPLEMENTATION_SOURCE],
        "implementation_contract": str(IMPLEMENTATION_CONTRACT),
        "implementation_contract_sha256": contract_sha,
        "official_checkpoint": selected_weight.get("checkpoint_path"),
        "official_checkpoint_sha256": selected_weight.get("checkpoint_sha256"),
        "sampler_artifact": str(SAMPLER),
        "sampler_artifact_sha256": sha256_file(SAMPLER),
        "lesion_weight_artifact": str(LESION_WEIGHTS),
        "lesion_weight_artifact_sha256": sha256_file(LESION_WEIGHTS),
        "pilot_log": str(log_path),
        "pilot_log_sha256": log_sha,
        "GPU": gpu_name,
        "VRAM_GB": float(total_vram_gb),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "optimizer_steps": int(len(log)),
        "microbatches": int(len(log) * GRADIENT_ACCUMULATION),
        "TRAIN_samples_processed_with_replacement": int(len(log) * EFFECTIVE_BATCH),
        "sampled_modality_counts": sampled_counts,
        "active_loss_weights": active_loss_weights,
        "optimizer_group_counts": optimizer_group_counts,
        "mean_first_8_step_loss": loss_first,
        "mean_last_8_step_loss": loss_last,
        "elapsed_seconds": elapsed,
        "peak_reserved_VRAM_GB": peak_reserved_gb,
        "peak_reserved_VRAM_fraction": peak_fraction,
        "parameter_group_L2_deltas": deltas,
        "gates": {name: bool(value) for name, value in gates.items()},
        "optimizer_created": True,
        "parameters_updated_in_disposable_preflight": True,
        "preflight_checkpoint_saved": False,
        "preflight_weights_allowed_for_final_initialization": False,
        "final_training_started": False,
        "validation_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "next_required_action": "RUN_NEXT_UNFINISHED_PROTOCOL_VARIANT_PREFLIGHT",
    }
    atomic_json(evidence, evidence_path)
    print("\nPilot log:")
    print(log_path)
    print("SHA-256:", log_sha)
    print("\nEvidence:")
    print(evidence_path)
    print("SHA-256:", sha256_file(evidence_path))
    print("\nOptimizer created                   : YES — DISPOSABLE PREFLIGHT ONLY")
    print("Parameters updated                  : YES — DISPOSABLE PREFLIGHT ONLY")
    print("Preflight checkpoint saved          : NO")
    print("Final comparator training started   : NO")
    print("VALIDATION/CALIBRATION accessed     : NO")
    print("Locked test accessed                : NO")
    print("\n" + "=" * 96)
    print("STAGE 0G-4D-5F-2B STATUS: PASS")
    print("=" * 96)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 96, file=sys.stderr)
        print("STAGE 0G-4D-5F-2B STATUS: FAIL", file=sys.stderr)
        print("=" * 96, file=sys.stderr)
        raise
