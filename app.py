"""
================================================================================
  Interface Streamlit — Lightroom Preset Generator  v2.0
================================================================================
  Lancement : streamlit run app.py
  Nouveautés v2 : Courbe des tons · HSL · Netteté · Réduction du bruit
================================================================================
"""

import io
import os
import tempfile
import streamlit as st
from PIL import Image

from lightroom_preset_generator import (
    run_full_analysis,
    generate_xmp_file,
    format_tone_curve_xmp,
    DEMO_PARAMS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lightroom Preset Generator",
    page_icon="📷",
    layout="wide",
)

# CSS minimal pour un look plus pro
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-label { font-size: 0.75rem !important; }
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin-bottom: 0.5rem;
    }
    .tag {
        display: inline-block;
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-family: monospace;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  EN-TÊTE
# ─────────────────────────────────────────────────────────────────────────────

st.title("📷 Lightroom Preset Generator")
st.caption(
    "Upload une photo d'inspiration · L'IA analyse ses caractéristiques visuelles · "
    "Génère un preset `.xmp` prêt à importer dans Lightroom Classic."
)

col_badges, _ = st.columns([2, 3])
with col_badges:
    st.markdown(
        '<span class="tag">✦ Tonalité</span>'
        '<span class="tag">✦ Courbe des tons</span>'
        '<span class="tag">✦ HSL</span>'
        '<span class="tag">✦ Color Grading</span>'
        '<span class="tag">✦ Netteté</span>'
        '<span class="tag">✦ Bruit</span>',
        unsafe_allow_html=True
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
#  UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<p class="section-title">1 — Importer une image</p>', unsafe_allow_html=True)

    preset_name = st.text_input(
        "Nom du preset",
        value="MonPreset",
        placeholder="Ex : Golden Hour, Cinematic Blue…",
    )

    uploaded_file = st.file_uploader(
        "Choisir une photo (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Traitement 100 % local — aucune image envoyée sur un serveur.",
    )

    use_demo = st.checkbox("Utiliser les valeurs de démonstration (sans image)", value=False)

with col_preview:
    st.markdown('<p class="section-title">Aperçu</p>', unsafe_allow_html=True)
    if uploaded_file:
        st.image(Image.open(uploaded_file), use_container_width=True, caption=uploaded_file.name)
    else:
        st.info("Aucune image importée pour l'instant.")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="section-title">2 — Analyser et générer</p>', unsafe_allow_html=True)

if st.button("🔍 Lancer l'analyse", type="primary", use_container_width=True):

    if not uploaded_file and not use_demo:
        st.warning("⚠️ Importe une image ou active le mode démonstration.")
        st.stop()
    if not preset_name.strip():
        st.warning("⚠️ Donne un nom à ton preset.")
        st.stop()

    with st.spinner("Analyse en cours…"):
        if use_demo:
            params = DEMO_PARAMS
            st.info("Mode démonstration — valeurs d'exemple.")
        else:
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                params = run_full_analysis(tmp_path)
            finally:
                os.unlink(tmp_path)

    st.success("✅ Analyse terminée !")
    st.divider()


    # ── Tonalité de base ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Tonalité de base</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Exposition",      params["Exposure2012"])
    c2.metric("Contraste",       params["Contrast2012"])
    c3.metric("Hautes lumières", params["Highlights2012"])
    c4.metric("Ombres",          params["Shadows2012"])
    c5.metric("Blancs",          params["Whites2012"])
    c6.metric("Noirs",           params["Blacks2012"])

    st.markdown("")

    col_wb, col_pres = st.columns(2, gap="large")
    with col_wb:
        st.markdown('<p class="section-title">Balance des blancs</p>', unsafe_allow_html=True)
        w1, w2 = st.columns(2)
        w1.metric("Température", params["Temperature"])
        w2.metric("Teinte",      params["Tint"])
    with col_pres:
        st.markdown('<p class="section-title">Présence</p>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        p1.metric("Vibrance",   params["Vibrance"])
        p2.metric("Saturation", params["Saturation"])

    st.divider()


    # ── Courbe des tons ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Courbe des tons (Point Curve)</p>', unsafe_allow_html=True)

    tc = params["ToneCurve"]
    zone_labels = {
        "blacks":     "⬛ Noirs",
        "shadows":    "🌑 Ombres",
        "midtones":   "🌗 Tons moyens",
        "highlights": "🌤 Hautes lumières",
        "whites":     "⬜ Blancs",
    }
    tc_cols = st.columns(5)
    for col, (zone, label) in zip(tc_cols, zone_labels.items()):
        inp, out = tc[zone]
        delta = out - inp
        col.metric(label, f"{out}", delta=f"{delta:+d}" if delta != 0 else "neutre")

    st.caption("Valeurs : Input (fixe) → Output (ajusté). Delta positif = zone relevée.")
    st.divider()


    # ── HSL ──────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">HSL — Corrections par couleur</p>', unsafe_allow_html=True)

    hsl = params["HSL"]
    color_emojis = {
        "Red": "🔴", "Orange": "🟠", "Yellow": "🟡", "Green": "🟢",
        "Aqua": "🩵", "Blue": "🔵", "Purple": "🟣", "Magenta": "🩷",
    }

    hsl_cols = st.columns(4)
    for i, (color, vals) in enumerate(hsl.items()):
        with hsl_cols[i % 4]:
            emoji = color_emojis.get(color, "")
            st.markdown(f"**{emoji} {color}**")
            st.markdown(
                f"Sat `{vals['Sat']:+d}` &nbsp; Lum `{vals['Lum']:+d}`",
                unsafe_allow_html=True
            )
    st.divider()


    # ── Color Grading ────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Color Grading</p>', unsafe_allow_html=True)
    cg = params["ColorGrading"]
    cg1, cg2, cg3 = st.columns(3)
    for col, (zone_key, label, emoji) in zip(
        [cg1, cg2, cg3],
        [("Shadow","Ombres","🌑"), ("Midtone","Tons moyens","🌗"), ("Highlight","Hautes lumières","☀️")]
    ):
        with col:
            st.markdown(f"**{emoji} {label}**")
            st.write(f"Teinte : `{int(cg[zone_key]['Hue'])}°`")
            st.write(f"Saturation : `{cg[zone_key]['Sat']}`")
            st.write(f"Luminance : `{cg[zone_key]['Lum']}`")
    st.divider()


    # ── Netteté & Bruit ──────────────────────────────────────────────────────
    col_sharp, col_noise = st.columns(2, gap="large")

    with col_sharp:
        st.markdown('<p class="section-title">Netteté</p>', unsafe_allow_html=True)
        sh = params["Sharpness"]
        s1, s2 = st.columns(2)
        s1.metric("Force",  sh["Sharpness"])
        s2.metric("Rayon",  sh["SharpenRadius"])
        s3, s4 = st.columns(2)
        s3.metric("Détail", sh["SharpenDetail"])
        s4.metric("Masque", sh["SharpenEdgeMasking"])

    with col_noise:
        st.markdown('<p class="section-title">Réduction du bruit</p>', unsafe_allow_html=True)
        nr = params["NoiseReduction"]
        n1, n2 = st.columns(2)
        n1.metric("Luminance",        nr["LuminanceSmoothing"])
        n2.metric("Détail luminance", nr["LuminanceNoiseReductionDetail"])
        n3, n4 = st.columns(2)
        n3.metric("Couleur",          nr["ColorNoiseReduction"])
        n4.metric("Détail couleur",   nr["ColorNoiseReductionDetail"])

    st.divider()


    # ── Téléchargement XMP ───────────────────────────────────────────────────
    st.markdown('<p class="section-title">4 — Télécharger le preset</p>', unsafe_allow_html=True)

    xmp_filename = preset_name.strip().replace(" ", "_") + ".xmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xmp", mode="w") as tmp_xmp:
        tmp_xmp_path = tmp_xmp.name

    generate_xmp_file(params, preset_name.strip(), tmp_xmp_path)
    with open(tmp_xmp_path, "r", encoding="utf-8") as f:
        xmp_bytes = f.read().encode("utf-8")
    os.unlink(tmp_xmp_path)

    st.download_button(
        label=f"⬇️  Télécharger  {xmp_filename}",
        data=xmp_bytes,
        file_name=xmp_filename,
        mime="application/octet-stream",
        use_container_width=True,
        type="primary",
    )
    st.caption(
        "**Importer dans Lightroom Classic** : "
        "Panneau Développement → Presets → **+** → Importer les préréglages → sélectionne le `.xmp`"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PIED DE PAGE
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption("v2.0 · Lightroom Preset Generator · Traitement 100% local · Aucune donnée envoyée.")
