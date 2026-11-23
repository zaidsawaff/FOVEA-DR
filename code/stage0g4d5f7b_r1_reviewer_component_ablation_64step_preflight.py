#!/usr/bin/env python3
"""Resume reviewer E2 preflights under the frozen AMP-safe amendment."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path("/content/drive/MyDrive/MMRDR_Q1_DR_Project")
BASE_RUNNER = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f7b_reviewer_component_ablation_64step_preflight.py"
REPAIR_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f7a_r1_reviewer_component_ablation_amp_safe_repair.py"
REPAIR_CONTRACT = ROOT / "Metadata/mmrdr_fovea_dr_reviewer_component_ablation_amp_safe_repair_v1.json"
REPAIR_EVIDENCE = ROOT / "Evidence/stage0g4d5f7a_r1_reviewer_component_ablation_amp_safe_repair.json"
FROZEN_SOURCE = ROOT / "Code/FOVEA_DR/v1_1/stage0g4d5f7b_r1_reviewer_component_ablation_64step_preflight.py"

EXPECTED_BASE_RUNNER_SHA = "462ad8cd8bee42404e97eeea25a85421ab165e3343bba06c3eee5fa846c61157"
EXPECTED_REPAIR_SOURCE_SHA = "bd7358f402d3488f4a829eee85a823456ee6e87b1eb0d676d229b383dbc1a357"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_base_runner() -> Any:
    if not BASE_RUNNER.is_file() or sha256_file(BASE_RUNNER) != EXPECTED_BASE_RUNNER_SHA:
        raise RuntimeError("The frozen base E2 preflight runner is missing or changed")
    specification = importlib.util.spec_from_file_location(
        "fovea_dr_reviewer_e2_preflight_base",
        BASE_RUNNER,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not import the frozen base E2 preflight runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def freeze_wrapper_source() -> tuple[str, str]:
    current = Path(__file__).resolve()
    source_sha = sha256_file(current)
    FROZEN_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_SOURCE.exists():
        if sha256_file(FROZEN_SOURCE) != source_sha:
            raise RuntimeError("A different repaired E2 preflight runner is already frozen")
        return "EXISTING_IDENTICAL_SOURCE_VERIFIED", source_sha
    temporary = FROZEN_SOURCE.with_name(FROZEN_SOURCE.name + f".tmp.{os.getpid()}")
    shutil.copyfile(current, temporary)
    os.replace(temporary, FROZEN_SOURCE)
    if sha256_file(FROZEN_SOURCE) != source_sha:
        raise RuntimeError("Repaired E2 preflight source freeze failed")
    return "COPIED_AND_FROZEN", source_sha


def main() -> None:
    print("=" * 120)
    print("STAGE 0G-4D-5F-7B-R1 — E2 64-STEP PREFLIGHT AFTER AMP-SAFE REPAIR")
    print("=" * 120)
    if not REPAIR_SOURCE.is_file() or sha256_file(REPAIR_SOURCE) != EXPECTED_REPAIR_SOURCE_SHA:
        raise RuntimeError("The frozen AMP-safe repair source is missing or changed")
    if not REPAIR_CONTRACT.is_file() or not REPAIR_EVIDENCE.is_file():
        raise RuntimeError("AMP-safe repair contract/evidence is missing")

    base = import_base_runner()
    original_implementation_source = base.IMPLEMENTATION_SOURCE
    expected = dict(base.EXPECTED_SHA)
    expected.pop(original_implementation_source, None)
    expected[REPAIR_SOURCE] = EXPECTED_REPAIR_SOURCE_SHA

    base.IMPLEMENTATION_SOURCE = REPAIR_SOURCE
    base.IMPLEMENTATION_CONTRACT = REPAIR_CONTRACT
    base.IMPLEMENTATION_EVIDENCE = REPAIR_EVIDENCE
    base.FROZEN_SOURCE = FROZEN_SOURCE
    base.EXPECTED_SHA = expected
    base.freeze_source = freeze_wrapper_source

    base.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 120, file=sys.stderr)
        print("STAGE 0G-4D-5F-7B-R1 STATUS: FAIL", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        raise
