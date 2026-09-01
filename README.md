# Tanager Spectral Search

Click any pixel in a Planet Tanager hyperspectral scene and instantly retrieve
every spectrally similar pixel across the scene. It is a reverse-image-search for
burned area mapping or any other surface material, powered by Spectral Angle Mapper (SAM).

Built for the Planet Hackathon on the Tanager Open Data program
(~426 bands, ~5 nm sampling, 380–2500 nm VSWIR, 30 m GSD).

---

## Caveats
- **Needs hyperspectral.** Meaningless with multispectral imagery's limited broad bands.
- **What makes it different.** One click gives instant material map. 
- **Robust by design.** SAM on brightness-normalized spectra is illumination-
  invariant, so it works on radiance and reflectance with zero training.

## How it works
Each spectrum is normalized to a unit vector, so overall brightness (sun angle,
albedo, shadow) drops out. The spectral angle between the clicked pixel and every
other pixel is just `arccos` of their dot product where small angle signifies same material.

---


## Repo layout
```
src/data.py         load Tanager HDF5 -> cube (H,W,B) + wavelengths (auto-detected)
src/preprocess.py   band + water-vapor masking, brightness/continuum normalize
src/sam.py          spectral angle mapper search
src/viz.py          RGB composite, overlay, spectrum plots
app.py              Streamlit click-to-search UI
scripts/            inspect_hdf5, download_data
```

## Notes on the data
Prefer **surface reflectance** scenes for the cleanest matches. Water-vapor bands
(~1350–1450, ~1800–1950 nm) are masked by default as they're noisy in VSWIR and
would pollute similarity.
