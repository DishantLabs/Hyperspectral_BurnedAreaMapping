"""Spectral Angle Mapper: illumination-invariant spectral similarity.

Assumes inputs are brightness-normalized (unit vectors). Angle in [0, pi];
smaller = more similar. Brute-force matvec is instant for a single scene
(~1e6 pixels x ~400 bands).
"""
from __future__ import annotations
import numpy as np


def sam_angles(X_unit: np.ndarray, query_unit: np.ndarray) -> np.ndarray:
    """Return per-pixel spectral angle (radians) to the query spectrum."""
    dots = np.clip(X_unit @ query_unit, -1.0, 1.0)
    return np.arccos(dots)


def search(X_unit, query_unit, top_k=None, threshold_deg=None):
    """Return (scores, selected_idx). scores are angles in degrees (lower better).
    If top_k given, selected are the k smallest angles; else angles < threshold."""
    ang = np.degrees(sam_angles(X_unit, query_unit))
    if top_k is not None:
        k = min(top_k, ang.shape[0])
        sel = np.argpartition(ang, k - 1)[:k]
    else:
        thr = threshold_deg if threshold_deg is not None else 5.0
        sel = np.nonzero(ang < thr)[0]
    return ang, sel
