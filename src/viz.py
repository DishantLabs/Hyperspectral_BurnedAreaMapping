"""Visualization: RGB composite, similarity overlay, spectrum plots."""
from __future__ import annotations
import numpy as np


def nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def rgb_composite(cube_hwb, wavelengths, rgb_nm=(640, 550, 470), pct=(2, 98),
                  bad_value=-9999):
    """Percentile-stretched RGB uint8, ignoring fill/NaN pixels in the stretch."""
    bands = [nearest_band(wavelengths, t) for t in rgb_nm]
    img = cube_hwb[:, :, bands].astype(np.float32)
    valid = np.isfinite(img).all(axis=-1) & (img != bad_value).all(axis=-1)
    out = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        good = ch[valid]
        if good.size == 0:
            continue
        lo, hi = np.percentile(good, pct)
        out[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
    out[~valid] = 0
    return (out * 255).astype(np.uint8)


def overlay_similarity(rgb_uint8, sel_mask_hw, color=(255, 40, 40), alpha=0.55):
    """Tint selected pixels over the RGB base."""
    out = rgb_uint8.astype(np.float32).copy()
    col = np.array(color, dtype=np.float32)
    m = sel_mask_hw
    out[m] = (1 - alpha) * out[m] + alpha * col
    return out.astype(np.uint8)


def spectra_figure(wavelengths, query_spec, match_specs, labels=None,
                   ylabel="reflectance"):
    """Matplotlib figure: clicked spectrum vs a few matches. This is the money shot."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(wavelengths, query_spec, lw=2.2, color="black", label="clicked pixel")
    for i, s in enumerate(match_specs):
        lab = labels[i] if labels else f"match {i+1}"
        ax.plot(wavelengths, s, lw=1.0, alpha=0.8, label=lab)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def draw_marker(img, r, c, color=(0, 255, 255), size=6, thickness=2):
    """Draw a hollow square marker centered on the clicked pixel (row r, col c)."""
    out = img.copy()
    H, W = out.shape[:2]
    col = np.array(color, dtype=out.dtype)
    r0, r1 = max(0, r - size), min(H, r + size + 1)
    c0, c1 = max(0, c - size), min(W, c + size + 1)
    for t in range(thickness):
        if 0 <= r - size + t < H:
            out[r - size + t, c0:c1] = col
        if 0 <= r + size - t < H:
            out[r + size - t, c0:c1] = col
        if 0 <= c - size + t < W:
            out[r0:r1, c - size + t] = col
        if 0 <= c + size - t < W:
            out[r0:r1, c + size - t] = col
    return out
