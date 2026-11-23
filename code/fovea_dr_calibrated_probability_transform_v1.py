"""Frozen FOVEA-DR v1.1 post-selection probability calibration."""

import math
import numpy as np

EPSILON = 1e-12


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def apply_grade_temperature(probabilities, temperature):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    log_probabilities = np.log(np.clip(probabilities, EPSILON, 1.0))
    return _softmax(log_probabilities / temperature)


def apply_lesion_temperature(logits, temperature):
    logits = np.asarray(logits, dtype=np.float64)
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    return _sigmoid(logits / temperature)
