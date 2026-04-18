import os
from collections.abc import Sequence

import numpy as np
import torch


_PLATFORM_KEYWORDS = {
    "ALS": (
        "als",
        "airborne",
        "airborne_laser_scanning",
        "airborne-laser-scanning",
    ),
    "ULS": (
        "uls",
        "uav",
        "uavls",
        "uav_lidar",
        "uav-lidar",
        "drone",
        "drone_lidar",
        "uav laser scanning",
        "uav-based",
    ),
    "MLS": (
        "mls",
        "mobile",
        "mobile_laser_scanning",
        "mobile-laser-scanning",
        "backpack",
        "zeb_horizon",
        "zeb-horizon",
    ),
    "TLS": (
        "tls",
        "terrestrial",
        "terrestrial_laser_scanning",
        "terrestrial-laser-scanning",
        "tripod",
        "ground_based",
        "ground-based",
    ),
}


def _flatten_string_candidates(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [str(value.item())]
        return [str(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [str(value.item())]
        return [str(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, Sequence):
        candidates = []
        for item in value:
            candidates.extend(_flatten_string_candidates(item))
        return candidates
    return [str(value)]


# Backward-compatible aliases preserved so old configs or checkpoints do not break.
_EXACT_ALIASES = {
    "ALS": "ALS",
    "ULS": "ULS",
    "MLS": "MLS",
    "TLS": "TLS",
    "S3DIS": "S3DIS",
    "SCANNET": "ScanNet",
    "STRUCTURED3D": "Structured3D",
}


def canonicalize_condition(value):
    """Return a canonical condition string when possible.

    Known forest-platform conditions are normalized to uppercase (ALS/ULS/MLS/TLS).
    Legacy indoor benchmark names are kept for backward compatibility.
    Unknown strings are returned unchanged after stripping whitespace.
    """
    candidates = _flatten_string_candidates(value)
    if not candidates:
        return None
    raw = str(candidates[0]).strip()
    if raw == "":
        return None
    upper = raw.upper()
    if upper in _EXACT_ALIASES:
        return _EXACT_ALIASES[upper]
    return raw


def infer_condition(*candidates):
    """Infer platform condition from explicit strings or file-system paths."""
    flat_candidates = []
    for candidate in candidates:
        flat_candidates.extend(_flatten_string_candidates(candidate))

    for candidate in flat_candidates:
        if candidate is None:
            continue
        candidate = str(candidate).strip()
        if candidate == "":
            continue
        canonical = canonicalize_condition(candidate)
        if canonical in _PLATFORM_KEYWORDS or canonical in _EXACT_ALIASES.values():
            return canonical

        path_bits = []
        normalized = candidate.replace("\\", "/")
        path_bits.append(normalized.lower())
        path_bits.extend(part.lower() for part in normalized.split("/"))
        path_bits.extend(part.lower() for part in os.path.basename(normalized).split("_"))
        path_bits.extend(part.lower() for part in os.path.basename(normalized).split("-"))

        for platform, keywords in _PLATFORM_KEYWORDS.items():
            for bit in path_bits:
                if bit == platform.lower() or bit in keywords:
                    return platform
                if any(keyword in bit for keyword in keywords):
                    return platform
    return None


def resolve_condition(value, *, fallback=None, error_prefix="condition"):
    """Canonicalize `value`, optionally falling back to path/metadata inference."""
    condition = canonicalize_condition(value)
    if condition is None:
        condition = infer_condition(fallback)
    if condition is None:
        raise ValueError(
            f"Unable to resolve {error_prefix}. Please provide an explicit platform "
            f"condition such as 'ALS', 'ULS', or 'MLS'."
        )
    return condition
