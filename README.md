# FOVEA-DR

Research code and aggregate experimental results for field-aware diabetic retinopathy grading across color fundus photography and ultra-widefield images. The implementation also includes an auxiliary OCT task. This research release is not intended for clinical use.

## Contents

| Directory | Contents |
| --- | --- |
| `code/` | Original FOVEA-DR model and training sources, comparator and ablation runners, calibration, locked-test scoring, and analysis scripts. Historical and repaired versions are both retained for provenance. |
| `protocols/` | Algorithm specification, training and comparison protocols, reviewer experiment protocol, and calibration parameters. |
| `results/MMRDR/` | Recorded aggregate results and hashes from the single locked internal evaluation. |
| `results/IDRiD/` | Aggregate external CFP metrics and paired contrasts. B2 was not part of the IDRiD comparison. |
| `results/E2/` | Selected-epoch validation metrics and paired validation contrasts for the component ablations. |
| `results/clinical_boundary/` | Exploratory grade 1–2 adaptation. It did not satisfy the predefined acceptance criteria and was not adopted. |
| `SHA256SUMS.txt` | Integrity hashes for the files in this archive. |

## Data sources

- **MMRDR v2:** [Figshare dataset](https://doi.org/10.6084/m9.figshare.29423747.v2).
- **IDRiD:** [Official challenge page and data access](https://idrid.grand-challenge.org/Home/) and [dataset descriptor](https://doi.org/10.3390/data3030025).

Obtain the images and source annotations from their original providers under the applicable terms. This archive does not redistribute images, patient-level or image-level prediction records, split membership lists, checkpoints, cached data, or the manuscript.

## Reproduction notes

This is a source-and-results snapshot, not a standalone runnable distribution. The archived scripts preserve their original file content and hashes. They reference the original Colab and Google Drive directory layout and may require additional frozen artifacts, manifests, package versions, pretrained weights, or checkpoints. Review each script's prerequisites before running it. The principal implementation is `code/fovea_dr_training_v1.py`; the baseline, comparator, and reviewer-stage filenames identify the corresponding workflows.

The aggregate results distinguish locked internal testing, external evaluation, and exploratory validation. They do not imply that all variants were assessed on the same external cohort or that all components improved performance.
