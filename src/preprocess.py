"""Band masking and per-spectrum normalization."""
from __future__ import annotations
import numpy as np


def good_band_mask(wavelengths: np.ndarray, drop_ranges) -> np.ndarray:
    """Boolean mask over bands, False inside noisy water-vapor ranges (nm)."""
    keep = np.ones_like(wavelengths, dtype=bool)
    for lo, hi in (drop_ranges or []):
        keep &= ~((wavelengths >= lo) & (wavelengths <= hi))
    return keep


def brightness_normalize(X: np.ndarray, eps=1e-8) -> np.ndarray:
    """L2-normalize each spectrum -> illumination/albedo scaling removed.
    This is what makes SAM robust on raw radiance."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def continuum_removal(X: np.ndarray) -> np.ndarray:
    """Divide each spectrum by its convex-hull continuum. Emphasizes absorption
    depth over overall brightness. Vectorized-lite: fine for a scene, not per-frame."""
    out = np.empty_like(X)
    idx = np.arange(X.shape[1])
    for i in range(X.shape[0]):
        y = X[i]
        hull = _upper_hull(idx, y)
        cont = np.interp(idx, hull[0], hull[1])
        out[i] = y / np.where(cont == 0, 1e-8, cont)
    return out


def _upper_hull(x, y):
    pts = list(zip(x, y))
    hull = []
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    hx, hy = zip(*hull)
    return np.array(hx), np.array(hy)


def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def prepare(X: np.ndarray, wavelengths: np.ndarray, cfg: dict):
    """Apply band mask + normalization. Returns (X_ready, wl_kept, band_mask)."""
    mask = good_band_mask(wavelengths, cfg.get("drop_band_ranges"))
    Xk, wlk = X[:, mask], wavelengths[mask]
    mode = cfg.get("normalize", "brightness")
    if mode == "brightness":
        Xk = brightness_normalize(Xk)
    elif mode == "continuum":
        Xk = continuum_removal(Xk)
    return Xk, wlk, mask
