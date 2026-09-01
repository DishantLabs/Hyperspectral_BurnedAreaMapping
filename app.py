"""Streamlit click-to-search demo (spectral / SAM only).

    streamlit run app.py

Click a pixel on the RGB preview -> selects the whole spectral CLASS (every pixel
within a spectral-angle threshold), marks the click, plots the spectra, and reports
pixel count + ground area for the selection.
"""
import numpy as np
import yaml
import streamlit as st

from src import data as D
from src import preprocess as P
from src import sam as S
from src import viz as V

st.set_page_config(page_title="Tanager Spectral Search", layout="wide")


@st.cache_data(show_spinner="Loading scene...")
def load(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    cube, wl, valid = D.load_scene(
        cfg["scene_path"], cfg.get("cube_dataset"),
        cfg.get("wavelength_dataset"), cfg.get("bad_value", -9999),
    )
    X, (H, W) = D.to_matrix(cube)
    Xr, wlk, band_mask = P.prepare(X, wl, cfg)
    rgb = V.rgb_composite(cube, wl, tuple(cfg.get("rgb_bands", (640, 550, 470))),
                          bad_value=cfg.get("bad_value", -9999))
    return cfg, valid, Xr, wlk, rgb, (H, W)


cfg, valid, Xr, wlk, rgb, (H, W) = load()

# y-axis label reflects the actual variable + normalization applied in preprocess.
_QTY = "Reflectance"  # open scenes here are surface reflectance
YLABEL = {
    "brightness": f"{_QTY} (L2-normalized)",
    "continuum": f"{_QTY} (continuum-removed)",
    "none": _QTY,
}.get(cfg.get("normalize", "brightness"), _QTY)

PIXEL_M = float(cfg.get("pixel_size_m", 30))  # Tanager GSD (m)

st.title("Tanager Spectral Search")
st.caption(f"{H}x{W} px · {wlk.shape[0]} bands used · {PIXEL_M:g} m · Spectral Angle Mapper")

# Class selector: every pixel within this spectral angle of the click is the "class".
# Larger angle -> broader class; crank it up to select the whole scene.
thr = st.sidebar.slider("Class similarity threshold (spectral angle °)",
                        0.5, 60.0, float(cfg["search"].get("sam_threshold_deg", 5.0)),
                        step=0.5)
st.sidebar.caption("Higher = looser match. Optimal Threshold is 4 to 8.")

col_img, col_plot = st.columns([3, 2])

with col_img:
    st.subheader("Click a pixel")
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        click = streamlit_image_coordinates(rgb, key="scene")
    except Exception:
        st.image(rgb, caption="Install streamlit-image-coordinates for click support")
        cx = st.number_input("col", 0, W - 1, W // 2)
        cy = st.number_input("row", 0, H - 1, H // 2)
        click = {"x": cx, "y": cy}

if click:
    c, r = int(click["x"]), int(click["y"])
    pidx = r * W + c
    q = Xr[pidx]

    # Select the class: all pixels within `thr` degrees. No fixed pixel cap.
    ang, sel = S.search(Xr, q, top_k=None, threshold_deg=thr)

    sel_mask = np.zeros(H * W, dtype=bool)
    sel_mask[sel] = True
    sel_mask &= valid.reshape(-1)
    sel_mask = sel_mask.reshape(H, W)

    with col_img:
        overlay = V.overlay_similarity(rgb, sel_mask)
        overlay = V.draw_marker(overlay, r, c)  # highlight the clicked pixel
        st.image(overlay, caption=f"Class selection @ row {r}, col {c}")

    with col_plot:
        st.subheader("Spectral Signature")
        match_ids = [i for i in sel[:3] if i != pidx][:3]
        fig = V.spectra_figure(wlk, q, [Xr[i] for i in match_ids], ylabel=YLABEL)
        st.pyplot(fig)

        # ---- Dashboard: selected-class stats ----
        n_sel = int(sel_mask.sum())
        valid_n = int(valid.sum())
        area_m2 = n_sel * PIXEL_M * PIXEL_M
        pct = (100.0 * n_sel / valid_n) if valid_n else 0.0
        st.markdown("**Selected class**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Pixels", f"{n_sel:,}")
        m2.metric("Area (km²)", f"{area_m2 / 1e6:,.2f}")
        m3.metric("% of scene", f"{pct:.1f}%")
        st.caption(f"Area = pixels × ({PIXEL_M:g} m)² = {area_m2/1e6:,.2f} km²  ·  "
                   f"threshold {thr:.1f}° spectral angle")
else:
    with col_img:
        st.image(rgb, caption="RGB preview — click to search")
