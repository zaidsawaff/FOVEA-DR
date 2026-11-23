#!/usr/bin/env python3
"""
STAGE 0G-4D-5F-0
Post-evaluation comparator and ablation protocol freeze.

This stage reads no image, label, prediction row, or scientific metric table.
It verifies frozen artifacts by hash, reads only JSON completion metadata, and
locks a validation-centered post-hoc comparison protocol. The already consumed
locked test is explicitly prohibited from comparator development or scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")

CANDIDATE = ROOT / "Metadata/mmrdr_fovea_dr_algorithm_candidate_v1_1.json"
SPLIT = ROOT / "Splits/mmrdr_development_split_v1_FROZEN.csv"
TRAINING_PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_training_protocol_v1.json"
PRODUCTION_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/fovea_dr_training_v1.py"
FINAL_SCORING_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5e3b_one_time_locked_test_scoring.py"
FINAL_SCORING_TABLE = ROOT / "Runs/FOVEA_DR/locked_test_seed2026_cpu_fp32/locked_test_scoring_table_v1.csv"
FINAL_METRICS = ROOT / "Audit/mmrdr_fovea_dr_LOCKED_TEST_final_metrics_v1.csv"
FINAL_BOOTSTRAP = ROOT / "Audit/mmrdr_fovea_dr_LOCKED_TEST_bootstrap_CI_v1.csv"
FINAL_SUMMARY = ROOT / "Runs/FOVEA_DR/locked_test_seed2026_cpu_fp32/locked_test_final_summary_v1.json"
FINAL_EVIDENCE = ROOT / "Evidence/stage0g4d5e3b_locked_test_single_evaluation_results.json"

FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f0_posthoc_comparison_protocol_freeze.py"
PROTOCOL = ROOT / "Metadata/mmrdr_fovea_dr_posthoc_comparison_protocol_v1.json"
EVIDENCE = ROOT / "Evidence/stage0g4d5f0_posthoc_comparison_protocol_freeze.json"

EXPECTED_SHA = {
    CANDIDATE: "4e27eb48ae3118dfd9e7d657b3e4cb48fc7858c31c7467dd59d06b3178904181",
    SPLIT: "8263a3ade3317d1c77ceeadaee0df1c8d245bd30fa7d472c47d2e53914714140",
    TRAINING_PROTOCOL: "5b16e0c8565b86d3cac603f416ee0b7f03055ab9a0eb9ac9e3fb007632329b40",
    PRODUCTION_SOURCE: "228ce4ef9389d98a184d53f82c797f172907f27fa3a8702387eb5afc43be2cc9",
    FINAL_SCORING_SOURCE: "c1d19cc8c1949007b527e30c3ba60f631c7abf144f9e1a648060ac9d2edebd4c",
    FINAL_SCORING_TABLE: "a4f2fa60491b24565ab3879461332e108f0676c5606b8f81b7c9701769a614b0",
    FINAL_METRICS: "042508456075bc148cecdb80bac637d687d5d1e4eb674d4877e71a675f6c5bf0",
    FINAL_BOOTSTRAP: "f14b425ee30d3e795b768d46b2ff449721b4d8efe920eba9e042dbe1979d34d9",
    FINAL_SUMMARY: "85ba1b45a5ec2eff5950a8ed3b14aa6be9825168bf420bb26d8af42480682dd5",
    FINAL_EVIDENCE: "f458f949061a71dc80a51b02424058b4ab366531c1ed336dc01dab5bc82947ee",
}

COMPARATOR_IDS = [
    "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
    "B2_RESNET50_MULTITASK",
    "B3_CONVNEXT_TINY_MULTITASK",
]

ABLATION_IDS = [
    "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
    "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
    "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def frozen_json(path: Path, payload: dict) -> str:
    data = json_bytes(payload)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen artifact exists with different content: {path}")
        return "EXISTING_IDENTICAL_ARTIFACT_VERIFIED"
    atomic_write(path, data)
    return "CREATED_AND_FROZEN"


def freeze_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different post-hoc protocol source is already frozen.")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    return "COPIED_AND_FROZEN", source_sha


def main() -> None:
    print("=" * 80)
    print("STAGE 0G-4D-5F-0 — POST-HOC COMPARATOR AND ABLATION PROTOCOL FREEZE")
    print("=" * 80)

    print("\n[1] FROZEN FOVEA-DR AND FINAL-RESULT VERIFICATION")
    print("-" * 80)
    gates = {}
    for path, expected in EXPECTED_SHA.items():
        exists = path.is_file()
        verified = exists and sha256_file(path) == expected
        gates[f"hash_verified::{path.name}"] = bool(verified)
        print(f"{path.name:76s}: exists={exists}, hash_verified={verified}")
    if not all(gates.values()):
        raise RuntimeError("A frozen FOVEA-DR artifact or final result changed.")

    with FINAL_EVIDENCE.open("r", encoding="utf-8") as handle:
        final_evidence = json.load(handle)
    with FINAL_SUMMARY.open("r", encoding="utf-8") as handle:
        final_summary = json.load(handle)

    final_pass = final_evidence.get("status") == "PASS"
    evaluation_consumed = bool(final_evidence.get("single_scientific_evaluation_consumed"))
    no_repeat_permitted = final_evidence.get("repeat_scientific_test_evaluation_permitted") is False
    no_further_selection = final_evidence.get("further_model_selection_permitted") is False
    summary_complete = final_summary.get("status") == "COMPLETE"

    print("Final evaluation PASS            :", final_pass)
    print("Single test evaluation consumed  :", evaluation_consumed)
    print("Repeat test evaluation prohibited:", no_repeat_permitted)
    print("Further test-based selection prohibited:", no_further_selection)
    print("Final summary COMPLETE           :", summary_complete)

    gates.update({
        "final_locked_test_evaluation_passed": final_pass,
        "single_locked_test_evaluation_consumed": evaluation_consumed,
        "repeat_locked_test_evaluation_prohibited": no_repeat_permitted,
        "test_based_model_selection_prohibited": no_further_selection,
        "final_summary_complete": summary_complete,
    })
    if not all(gates.values()):
        raise RuntimeError("Final-result closure metadata is inconsistent.")

    print("\n[2] COMPARISON SCOPE")
    print("-" * 80)
    print("Classification: POST_HOC_METHOD_COMPARISON")
    print("Development data: TRAIN + VALIDATION + CALIBRATION only")
    print("Locked test     : PROHIBITED for all new comparator work")
    print("Primary comparison domain: frozen VALIDATION metrics")
    print("Seed            : 2026 (matched to the completed FOVEA-DR run)")

    print("\nPlain multitask comparators:")
    for identifier in COMPARATOR_IDS:
        print(" ", identifier)
    print("\nFOVEA-DR ablations:")
    for identifier in ABLATION_IDS:
        print(" ", identifier)

    protocol = {
        "protocol_id": "MMRDR_FOVEA_DR_POSTHOC_COMPARISON_V1",
        "stage": "0G-4D-5F-0",
        "status": "FROZEN_BEFORE_COMPARATOR_TRAINING",
        "classification": "POST_HOC_METHOD_COMPARISON_AFTER_PRIMARY_LOCKED_TEST_EVALUATION",
        "scientific_positioning": {
            "primary_FOVEA_DR_locked_test_result_remains_unchanged": True,
            "comparison_was_not_preregistered_before_primary_test_access": True,
            "comparison_must_be_reported_as_post_hoc": True,
            "comparison_cannot_redefine_the_primary_endpoint": True,
            "comparison_cannot_change_the_completed_FOVEA_DR_claim": True,
        },
        "frozen_parent_artifacts": {
            "candidate": {"path": str(CANDIDATE), "sha256": EXPECTED_SHA[CANDIDATE]},
            "development_split": {"path": str(SPLIT), "sha256": EXPECTED_SHA[SPLIT]},
            "training_protocol": {"path": str(TRAINING_PROTOCOL), "sha256": EXPECTED_SHA[TRAINING_PROTOCOL]},
            "production_source": {"path": str(PRODUCTION_SOURCE), "sha256": EXPECTED_SHA[PRODUCTION_SOURCE]},
            "final_result_evidence": {"path": str(FINAL_EVIDENCE), "sha256": EXPECTED_SHA[FINAL_EVIDENCE]},
        },
        "data_access_policy": {
            "TRAIN": "ALLOWED_FOR_OPTIMIZATION",
            "VALIDATION": "ALLOWED_FOR_EARLY_STOPPING_AND_MODEL_SELECTION",
            "CALIBRATION": "ALLOWED_ONLY_AFTER_EPOCH_SELECTION_FOR_TEMPERATURE_SCALING",
            "LOCKED_INTERNAL_TEST": "PROHIBITED_FOR_ALL_COMPARATOR_AND_ABLATION_EXECUTION",
            "existing_FOVEA_DR_locked_test_aggregate_may_be_reported_unchanged": True,
            "existing_locked_test_scoring_table_may_be_read_by_comparator_code": False,
            "existing_locked_test_metric_table_may_be_used_for_model_choice": False,
        },
        "fairness_controls": {
            "development_split": "IDENTICAL_FROZEN_SPLIT",
            "input": "512x512_RGB",
            "normalization": "ImageNet_V1_mean_std",
            "augmentation": "INHERIT_FROZEN_TRAINING_PROTOCOL_V1",
            "sampling": "INHERIT_FROZEN_MODALITY_GRADE_SAMPLER",
            "micro_batch": 16,
            "gradient_accumulation": 2,
            "effective_batch": 32,
            "optimizer_scheduler_and_early_stopping": "INHERIT_FROZEN_TRAINING_PROTOCOL_V1",
            "maximum_epochs": 30,
            "early_stopping_patience": 6,
            "run_seed": 2026,
            "precision": "FLOAT16_AMP_ON_T4_OR_NUMERICALLY_EQUIVALENT_SUPPORTED_GPU",
            "pretraining": "OFFICIAL_TORCHVISION_IMAGENET1K_V1_WHEN_AVAILABLE",
            "pilot_checkpoint_initialization": False,
        },
        "plain_multitask_head_contract": {
            "fundus_grade_head": "5_CLASS_CFP_AND_UWF",
            "OCT_head": "3_CLASS_DME",
            "fundus_lesion_head": "7_INDEPENDENT_BINARY_LOGITS",
            "clinical_reasoner": False,
            "counterfactual_intervention_loss": False,
            "clinical_state_alignment_loss": False,
            "masked_route_agreement_loss": False,
            "learned_reliability_gate": False,
            "grade_CE_weight": 1.0,
            "ordinal_EMD_weight": 0.5,
            "lesion_BCE_weight": 0.5,
            "OCT_DME_task_weight": 0.5,
        },
        "comparators": [
            {
                "id": "B1_PLAIN_EFFICIENTNETV2S_MULTITASK",
                "family": "BACKBONE_MATCHED_CONTROL",
                "backbone": "EfficientNetV2-S",
                "initialization": "official_torchvision_ImageNet1K_V1",
                "purpose": "Isolate the contribution of FOVEA-DR structured clinical modules from the shared backbone.",
            },
            {
                "id": "B2_RESNET50_MULTITASK",
                "family": "CONVENTIONAL_CNN_CONTROL",
                "backbone": "ResNet-50",
                "initialization": "official_torchvision_ImageNet1K_V2",
                "purpose": "Conventional residual CNN baseline.",
            },
            {
                "id": "B3_CONVNEXT_TINY_MULTITASK",
                "family": "MODERN_CNN_CONTROL",
                "backbone": "ConvNeXt-Tiny",
                "initialization": "official_torchvision_ImageNet1K_V1",
                "purpose": "Modern convolutional architecture baseline.",
            },
        ],
        "ablations": [
            {
                "id": "A1_FOVEA_WITHOUT_COUNTERFACTUAL_LOSS",
                "parent": "FOVEA_DR_V1_1",
                "only_change": "lesion_counterfactual_intervention_weight=0.0",
            },
            {
                "id": "A2_FOVEA_WITHOUT_CLINICAL_ALIGNMENT_AND_ROUTE_AGREEMENT",
                "parent": "FOVEA_DR_V1_1",
                "only_changes": [
                    "clinical_state_alignment_weight=0.0",
                    "masked_route_agreement_weight=0.0",
                ],
            },
            {
                "id": "A3_FOVEA_WITHOUT_LEARNED_RELIABILITY_GATE",
                "parent": "FOVEA_DR_V1_1",
                "only_change": "evidence_weight_fixed_to_1.0_for_all_routes_and_interventions",
            },
        ],
        "selection_and_calibration": {
            "epoch_selection_dataset": "VALIDATION",
            "epoch_selection_primary_metric": "MEAN_OF_CFP_AND_UWF_QUADRATIC_WEIGHTED_KAPPA",
            "epoch_selection_tie_breaker": "MEAN_CFP_UWF_NLL",
            "early_stopping_patience": 6,
            "temperature_fit_dataset": "CALIBRATION",
            "temperature_units": "CFP_GRADE_UWF_GRADE_OCT_GRADE_AND_7_POOLED_FUNDUS_LESIONS",
            "temperature_bounds": "INHERIT_FROZEN_CALIBRATION_PROTOCOL",
            "decision_threshold_change_allowed": False,
            "lesion_threshold": 0.5,
        },
        "validation_comparison_endpoints": {
            "primary": "UNWEIGHTED_MEAN_CFP_UWF_QWK",
            "secondary": [
                "CFP_QWK",
                "UWF_QWK",
                "MEAN_CFP_UWF_NLL",
                "OCT_QWK",
                "OCT_MACRO_F1",
                "OCT_NLL",
            ],
            "grade_metrics": [
                "NLL",
                "Brier_score",
                "ECE_15_EQUAL_WIDTH",
                "accuracy",
                "macro_F1",
                "quadratic_weighted_kappa",
            ],
            "lesion_metrics": [
                "binary_NLL",
                "Brier_score",
                "ECE_15_EQUAL_WIDTH",
                "accuracy_at_0_5",
                "F1_at_0_5",
            ],
            "lesion_units": [
                "microaneurysm",
                "hard_exudate",
                "intraretinal_hemorrhage",
                "VB_IRMA",
                "neovascularization",
                "vitreous_hemorrhage",
                "retinal_detachment",
            ],
        },
        "run_plan": {
            "existing_reference": "FOVEA_DR_V1_1_SEED2026",
            "new_comparator_runs": COMPARATOR_IDS,
            "new_ablation_runs": ABLATION_IDS,
            "total_new_full_runs": 6,
            "mandatory_preflight_steps_per_run": 64,
            "execution_order": [*COMPARATOR_IDS, *ABLATION_IDS],
            "stop_or_omit_rule": "NO_RUN_MAY_BE_OMITTED_BECAUSE_OF_UNFAVORABLE_VALIDATION_PERFORMANCE",
        },
        "reporting_policy": {
            "report_all_six_frozen_runs": True,
            "report_failures_and_resource_limits": True,
            "single_seed_limitation_must_be_disclosed": True,
            "no_significance_claim_from_overlapping_validation_runs": True,
            "published_literature_comparison_must_separate_nonidentical_splits_or_metrics": True,
            "existing_locked_test_result_reported_only_for_FOVEA_DR": True,
            "new_models_receive_no_locked_test_score_under_this_protocol": True,
        },
        "prohibitions": [
            "NO_LOCKED_TEST_IMAGE_ACCESS",
            "NO_LOCKED_TEST_LABEL_ACCESS",
            "NO_LOCKED_TEST_PREDICTION_GENERATION_FOR_NEW_MODELS",
            "NO_SECOND_LOCKED_TEST_SCIENTIFIC_EVALUATION",
            "NO_CHANGE_TO_FOVEA_DR_FROZEN_RESULT",
            "NO_COMPARATOR_OMISSION_AFTER_RESULTS",
            "NO_HYPERPARAMETER_TUNING_UNIQUE_TO_A_SINGLE_COMPARATOR",
            "NO_TEST_BASED_CLAIM_EXPANSION",
        ],
        "training_started_during_protocol_freeze": False,
        "image_rows_loaded_during_protocol_freeze": False,
        "locked_test_artifact_rows_read_during_protocol_freeze": False,
    }

    print("\n[3] PROTOCOL AND SOURCE FREEZE")
    print("-" * 80)
    source_action, source_sha = freeze_source()
    protocol["protocol_freeze_source"] = {
        "path": str(FROZEN_SOURCE),
        "sha256": source_sha,
    }
    protocol_action = frozen_json(PROTOCOL, protocol)
    protocol_sha = sha256_file(PROTOCOL)

    gates.update({
        "protocol_declared_post_hoc": protocol["classification"].startswith("POST_HOC"),
        "TRAIN_ALLOWED": protocol["data_access_policy"]["TRAIN"] == "ALLOWED_FOR_OPTIMIZATION",
        "VALIDATION_only_for_selection": "MODEL_SELECTION" in protocol["data_access_policy"]["VALIDATION"],
        "CALIBRATION_only_after_selection": "ONLY_AFTER_EPOCH_SELECTION" in protocol["data_access_policy"]["CALIBRATION"],
        "locked_test_prohibited_for_new_models": protocol["data_access_policy"]["LOCKED_INTERNAL_TEST"].startswith("PROHIBITED"),
        "exactly_three_plain_comparators_locked": len(protocol["comparators"]) == 3,
        "exactly_three_FOVEA_ablations_locked": len(protocol["ablations"]) == 3,
        "all_six_runs_must_be_reported": protocol["reporting_policy"]["report_all_six_frozen_runs"],
        "seed2026_matched": protocol["fairness_controls"]["run_seed"] == 2026,
        "input512_RGB_preserved": protocol["fairness_controls"]["input"] == "512x512_RGB",
        "effective_batch32_preserved": protocol["fairness_controls"]["effective_batch"] == 32,
        "primary_validation_endpoint_preserved": protocol["validation_comparison_endpoints"]["primary"] == "UNWEIGHTED_MEAN_CFP_UWF_QWK",
        "lesion_threshold_0_5_preserved": protocol["selection_and_calibration"]["lesion_threshold"] == 0.5,
        "source_frozen": sha256_file(FROZEN_SOURCE) == source_sha,
        "protocol_frozen": sha256_file(PROTOCOL) == protocol_sha,
        "training_not_started": True,
        "images_not_accessed": True,
        "locked_test_rows_not_read": True,
    })
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Post-hoc comparison protocol gate failed: {failed}")

    evidence = {
        "stage": "0G-4D-5F-0",
        "status": "PASS",
        "purpose": "POST_HOC_VALIDATION_COMPARATOR_AND_ABLATION_PROTOCOL_FREEZE",
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_sha,
        "source": str(FROZEN_SOURCE),
        "source_sha256": source_sha,
        "gates": gates,
        "training_started": False,
        "images_accessed": False,
        "locked_test_artifact_rows_read": False,
        "locked_test_reused": False,
        "next_required_stage": "COMPARATOR_IMPLEMENTATION_AND_64_STEP_TRAIN_VALIDATION_PREFLIGHT",
    }
    evidence_action = frozen_json(EVIDENCE, evidence)

    print("Source action   :", source_action)
    print("Frozen source   :", FROZEN_SOURCE)
    print("Source SHA-256  :", source_sha)
    print("Protocol action :", protocol_action)
    print("Protocol        :", PROTOCOL)
    print("Protocol SHA-256:", protocol_sha)
    print("Evidence action :", evidence_action)
    print("Evidence        :", EVIDENCE)
    print("Evidence SHA-256:", sha256_file(EVIDENCE))

    print("\n" + "=" * 80)
    print("POST-HOC COMPARISON PROTOCOL GATES")
    print("=" * 80)
    for name, passed in gates.items():
        print(f"{name:76s}: {passed}")

    print("\nTraining started          : NO")
    print("Images accessed           : NO")
    print("Locked-test rows accessed : NO")
    print("Locked test reused        : NO")
    print("\nNEXT REQUIRED STAGE:")
    print("Comparator implementation and 64-step TRAIN/VALIDATION-only preflight.")
    print("\n" + "=" * 80)
    print("STAGE 0G-4D-5F-0 STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
