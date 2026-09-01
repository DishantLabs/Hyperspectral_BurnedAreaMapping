"""Load a Tanager hyperspectral HDF5 scene into a numpy cube + wavelength vector.

Tanager open-data HDF5 layout is not guaranteed stable, so we auto-detect:
- the cube is the largest 3D dataset;
- wavelengths are a 1D numeric array (dataset OR HDF5 attribute) whose length
  matches a cube axis AND whose values look like VSWIR nm (spans ~visible to SWIR),
  so we don't accidentally grab FWHM or band-index arrays.
Override cube_dataset / wavelength_dataset in config.yaml if needed
(run scripts/inspect_hdf5.py to see the tree).
"""
from __future__ import annotations
import numpy as np
import h5py


def print_tree(path: str) -> None:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:55s} shape={obj.shape} dtype={obj.dtype}")
        for k in obj.attrs:
            try:
                a = np.atleast_1d(np.asarray(obj.attrs[k]))
                print(f"  [attr] {name}@{k}  len={a.shape[0]} dtype={a.dtype}")
            except Exception:
                pass
    with h5py.File(path, "r") as f:
        for k in f.attrs:
            a = np.atleast_1d(np.asarray(f.attrs[k]))
            print(f"  [root-attr] {k}  len={a.shape[0]} dtype={a.dtype}")
        f.visititems(show)


def _largest_3d(f: h5py.File):
    best = None
    def walk(name, obj):
        nonlocal best
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            n = int(np.prod(obj.shape))
            if best is None or n > best[1]:
                best = (name, n)
    f.visititems(walk)
    return best[0] if best else None


def _as_wavelengths(a):
    """Return (array_nm, is_wavelength_like). Converts micrometers -> nm."""
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size == 0 or not np.all(np.isfinite(a)):
        return a, False
    lo, hi = float(a.min()), float(a.max())
    if 0.3 < lo and hi < 3.0 and hi > 1.5:      # looks like micrometers
        a = a * 1000.0
        lo, hi = lo * 1000.0, hi * 1000.0
    # VSWIR nm should start below ~1000 and reach well into SWIR (>1800), under 3000
    ok = (lo < 1000.0) and (1800.0 < hi < 3000.0)
    return a, ok


def _collect_1d_candidates(f: h5py.File, lengths):
    """All 1D numeric arrays (datasets + attrs) whose length matches a cube axis."""
    lens = set(int(x) for x in lengths)
    out = []

    def consider(name, arr):
        arr = np.atleast_1d(np.asarray(arr))
        if arr.ndim == 1 and arr.shape[0] in lens and np.issubdtype(arr.dtype, np.number):
            out.append((name, arr))

    for k in f.attrs:
        consider(f"@{k}", f.attrs[k])

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
            consider(name, obj[...])
        for k in obj.attrs:
            consider(f"{name}@{k}", obj.attrs[k])
    f.visititems(walk)
    return out


def _pick_wavelengths(f: h5py.File, cube_shape):
    """Best wavelength array by value-range validity, then spectral-looking name."""
    best = None  # (score, array_nm)
    for name, arr in _collect_1d_candidates(f, cube_shape):
        nm, ok = _as_wavelengths(arr)
        if not ok:
            continue
        low = name.lower()
        spectral = any(k in low for k in ("wavelength", "wave", "wvl", "wl", "center"))
        score = 2 if spectral else 1
        if best is None or score > best[0]:
            best = (score, nm)
    return best[1] if best else None


def _to_hwb(cube, nbands_hint):
    if nbands_hint is not None and nbands_hint in cube.shape:
        band_axis = cube.shape.index(nbands_hint)
    else:
        band_axis = int(np.argmin(cube.shape))
    return np.moveaxis(cube, band_axis, -1)


def load_scene(path, cube_dataset=None, wavelength_dataset=None, bad_value=-9999):
    """Returns (cube[H,W,B] float32, wavelengths[B] float32, valid_mask[H,W] bool)."""
    with h5py.File(path, "r") as f:
        cds = cube_dataset or _largest_3d(f)
        if cds is None:
            raise ValueError("No 3D dataset found; set cube_dataset in config.yaml")
        raw = np.asarray(f[cds][...])

        if wavelength_dataset:
            wl = np.asarray(f[wavelength_dataset][...], dtype=np.float32).ravel()
        else:
            wl = _pick_wavelengths(f, raw.shape)
            wl = None if wl is None else wl.astype(np.float32)

    nb_hint = wl.shape[0] if wl is not None else None
    cube = _to_hwb(raw, nb_hint).astype(np.float32)
    B = cube.shape[-1]
    if wl is None:
        print("WARNING: no wavelength array detected; falling back to band index. "
              "Set wavelength_dataset in config.yaml (see scripts/inspect_hdf5.py).")
        wl = np.arange(B, dtype=np.float32)

    valid = np.isfinite(cube).all(axis=-1) & (cube != bad_value).all(axis=-1)
    return cube, wl, valid


def to_matrix(cube):
    H, W, B = cube.shape
    return cube.reshape(-1, B), (H, W)
