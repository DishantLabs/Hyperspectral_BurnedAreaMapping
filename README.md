# Tanager Spectral Search

Click any pixel in a Planet Tanager hyperspectral scene and instantly retrieve
every spectrally similar pixel across the scene — a reverse-image-search for
*materials*, powered by Spectral Angle Mapper (SAM).

Built for the Planet Hackathon on the Tanager Open Data program
(~426 bands, ~5 nm sampling, 380–2500 nm VSWIR, 30 m GSD).

---

## Why this wins
- **Needs hyperspectral.** Meaningless with 4 broad bands. Judges reward
  capabilities impossible without the sensor.
- **Wow-per-minute.** One click → instant material map. No slide required.
- **Robust by design.** SAM on brightness-normalized spectra is illumination-
  invariant, so it works on radiance *and* reflectance with zero training.

## How it works
Each spectrum is normalized to a unit vector, so overall brightness (sun angle,
albedo, shadow) drops out. The spectral angle between the clicked pixel and every
other pixel is just `arccos` of their dot product — small angle = same material.
Brute-force over a full scene (~600k pixels × ~400 bands) is instant.

---

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put a Tanager surface-reflectance .h5 at data/scene.h5, then:
PYTHONPATH=. python scripts/inspect_hdf5.py data/scene.h5   # sanity-check the file
streamlit run app.py
```
Click a pixel → it highlights similar pixels and plots the clicked spectrum
against its top matches. That spectra plot is the money shot: it shows the
judge *why* two pixels matched (shared absorption features).

## Build plan (time-boxed)
- **0–1h** — Data in hand. Drop a scene in `data/`, run `inspect_hdf5.py`. If the
  cube/wavelengths auto-detect wrong, set `cube_dataset` / `wavelength_dataset`
  in `config.yaml`.
- **1–2h** — App running, click-to-search working on SAM. This is the demo.
- **2–4h** — Polish the money shot: clean RGB stretch, overlay color, annotate
  the absorption features in the spectra plot.
- **Stretch** — Add a "top-K vs angle-threshold" toggle; try `normalize:
  continuum` in config to emphasize absorption depth over brightness; swap the
  Streamlit UI for React + Leaflet + FastAPI if you want it to look like a product.

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
(~1350–1450, ~1800–1950 nm) are masked by default — they're noisy in VSWIR and
would pollute similarity. Verify the STAC catalog URL in `scripts/download_data.py`
against Planet's open-data page before relying on auto-download.
